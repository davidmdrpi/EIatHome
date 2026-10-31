"""Verified signal sequences, observation splits, and a training-only tokenizer."""

import hashlib
import json
from pathlib import Path

import numpy as np

CHANNELS = 64
SPLITS = ("train", "validation", "test")
# Real families, and the synthetic ones (tracks/llm/synthetic.py: the Rosetta control and an RFI
# sweep), in the same units.
FAMILIES = ("voyager", "offsource", "rosetta", "sweep")
SESSION_GAP_DAYS = 0.5  # as in web/js/splits.js and tools/extract_slices.py
SCAN_GAP_DAYS = 1e-8


def digest(value):
    return hashlib.sha256(value).hexdigest()


def load_slices(manifest_path, slice_id=None):
    """Training validates all observations; sampling verifies only the selected payload.

    Sampling needs no session/family metadata because it performs no train/test split.
    """
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    version = manifest.get("version")
    if manifest.get("format") != "eiathome-slices" or version not in (1, 2):
        raise ValueError("Expected an eiathome-slices version 1 or 2 manifest")
    entries = manifest["slices"]
    if version == 2:  # version 2 adds where the telescope pointed; nothing here uses it
        for entry in entries:
            sky = entry.get("source", {}).get("sky")
            ra, dec = (sky.get("ra_hours"), sky.get("dec_deg")) if isinstance(sky, dict) else (None, None)
            if not (isinstance(ra, (int, float)) and 0 <= ra < 24 and isinstance(dec, (int, float)) and -90 <= dec <= 90):
                raise ValueError(f"A version 2 slice needs source.sky with ra_hours and dec_deg: {entry.get('id')}")
    if slice_id is not None:
        entries = [entry for entry in entries if entry.get("id") == slice_id]
        if len(entries) != 1:
            raise ValueError("--slice-id must match exactly one manifest entry")
    records = []
    for entry in entries:
        file = (path.parent / entry["file"]).resolve()
        if not file.is_relative_to(path.parent.resolve()):
            raise ValueError("Slice paths must stay inside the manifest directory")
        payload = file.read_bytes()
        sha = digest(payload)
        if sha != entry.get("sha256"):
            raise ValueError(f"SHA-256 mismatch: {entry['id']}")
        shape = entry["shape"]
        if (entry.get("dtype") != "float32-le" or len(shape) != 2
                or any(type(n) is not int for n in shape)
                or shape[0] < 2 or shape[1] < CHANNELS
                or len(payload) != shape[0] * shape[1] * 4):
            raise ValueError(f"Invalid slice shape/dtype: {entry['id']}")
        raw = np.frombuffer(payload, dtype="<f4").reshape(shape)
        if not np.isfinite(raw).all():
            raise ValueError("Non-finite signal samples")
        start = entry.get("source", {}).get("tstart_mjd")
        valid_start = isinstance(start, (int, float)) and np.isfinite(start)
        if slice_id is None and not valid_start:
            raise ValueError("Every slice needs a finite source.tstart_mjd for splitting")
        if slice_id is None and entry.get("family") not in FAMILIES:
            raise ValueError("Unknown signal family")
        first = (shape[1] - CHANNELS) // 2
        frames = raw[:, first:first + CHANNELS].copy()
        records.append({"entry": entry, "sha256": sha, "start": float(start) if valid_start else None,
                        "frames": frames, "frame_hash": digest(frames.tobytes())})
    if not records:
        raise ValueError("The manifest contains no slices")
    families_by_hash = {}
    for record in records:
        previous = families_by_hash.setdefault(record["frame_hash"], record["entry"].get("family"))
        if previous != record["entry"].get("family"):
            raise ValueError("Identical signal windows cannot be both signal and off-source controls")
    return sorted(records, key=lambda r: (r["start"], r["sha256"], r["entry"]["id"]))


def groups(records, gap_days):
    """O(n log n) time/hash components, including copies with inconsistent timestamps."""
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_sha, by_frame = {}, {}
    previous = None
    for i in sorted(range(len(records)), key=lambda i: records[i]["start"]):
        record = records[i]
        if previous is not None and record["start"] - records[previous]["start"] <= gap_days:
            parent[root(i)] = root(previous)
        for mapping, key in ((by_sha, "sha256"), (by_frame, "frame_hash")):
            other = mapping.setdefault(record[key], i)
            parent[root(i)] = root(other)
        previous = i
    out = {}
    for i, record in enumerate(records):
        out.setdefault(root(i), []).append(record)
    return sorted(out.values(), key=lambda g: (min(r["start"] for r in g), min(r["sha256"] for r in g)))


def dataset_digest(records):
    """The manifest's slices by content and entry: the dataset_sha256 a run records."""
    identity = [{"sha256": r["sha256"], "entry": r["entry"]} for r in records]
    return digest(json.dumps(identity, sort_keys=True).encode())


def control_family(records, family):
    """The family a run on `family` is scored on as a control: another family in the manifest,
    off-source first (real noise, nothing in the beam). None if there isn't one."""
    others = sorted({r["entry"]["family"] for r in records} - {family})
    return "offsource" if "offsource" in others else (others[0] if others else None)


def unique_records(records):
    """Copies don't provide extra training or evaluation evidence, including controls."""
    unique = {}
    for record in records:
        unique.setdefault(record["frame_hash"], record)
    return list(unique.values())


def split_records(records, family="voyager", allow_within_session=False):
    # Include both families when establishing sessions, so a bridging scan isn't ignored.
    units = [[r for r in g if r["entry"]["family"] == family] for g in groups(records, SESSION_GAP_DAYS)]
    units = [g for g in units if g]
    independent = len(units) >= 3
    if not independent:
        if not allow_within_session:
            raise ValueError("Need at least 3 independent sessions for train/validation/test. "
                             "For a labelled smoke test only, pass --allow-within-session.")
        units = [[r for r in g if r["entry"]["family"] == family] for g in groups(records, SCAN_GAP_DAYS)]
        units = [g for g in units if g]
        if len(units) < 3:
            raise ValueError("Need at least 3 distinct scans even for a within-session smoke test")
    # Chronological 60/20/20, with at least one whole group in every split.
    n_eval = max(1, round(len(units) * 0.2))
    partitions = (units[:-2 * n_eval], units[-2 * n_eval:-n_eval], units[-n_eval:])
    result = {}
    for name, partition in zip(SPLITS, partitions):
        result[name] = unique_records(r for group in partition for r in group)
    return result, independent


def split_sizes(records, partitions, independent):
    units = groups(records, SESSION_GAP_DAYS if independent else SCAN_GAP_DAYS)
    unit_hashes = [{r["frame_hash"] for r in group} for group in units]
    sizes = {}
    for name, part in partitions.items():
        part_hashes = {r["frame_hash"] for r in part}
        sizes[name] = {"groups": sum(bool(hashes & part_hashes) for hashes in unit_hashes),
                       "slices": len(part), "frames": sum(len(r["frames"]) for r in part),
                       "targets": sum(len(r["frames"]) - 1 for r in part)}
    return sizes


def signed_log(frames):
    return np.sign(frames) * np.log1p(np.abs(frames))


class Tokenizer:
    """One categorical token per 64-channel frame; all fitted state comes from train."""

    def __init__(self, mean, scale, centroids):
        self.mean = float(mean)
        self.scale = float(scale)
        self.centroids = np.asarray(centroids, dtype=np.float32)

    @classmethod
    def fit(cls, sequences, vocab_size, seed):
        x = signed_log(np.concatenate(sequences))
        if not 2 <= vocab_size <= len(x):
            raise ValueError("Vocabulary must be between 2 and the number of training frames")
        mean, scale = float(x.mean()), max(float(x.std()), 1e-6)
        x = (x - mean) / scale
        rng = np.random.default_rng(seed)
        centres = x[rng.choice(len(x), vocab_size, replace=False)].copy()
        for _ in range(40):
            labels = cls(mean, scale, centres)._nearest(x)
            updated = np.stack([x[labels == i].mean(axis=0) if np.any(labels == i) else centres[i]
                                for i in range(vocab_size)])
            if np.allclose(updated, centres, atol=1e-6):
                break
            centres = updated
        return cls(mean, scale, centres)

    def _nearest(self, x):
        # O(batch * vocabulary) temporary storage, not batch * vocabulary * channels.
        # Float64 reduces cancellation near a centroid while keeping the matrix bounded.
        centres = self.centroids.astype(np.float64)
        norms = (centres ** 2).sum(axis=1)
        labels = []
        for start in range(0, len(x), 256):
            batch = np.asarray(x[start:start + 256], dtype=np.float64)
            distances = (batch ** 2).sum(axis=1, keepdims=True) - 2 * batch @ centres.T + norms
            labels.append(distances.argmin(axis=1))
        return np.concatenate(labels) if labels else np.empty(0, dtype=np.int64)

    def encode(self, frames):
        return self._nearest((signed_log(frames) - self.mean) / self.scale).astype(np.int64)

    def decode(self, tokens):
        x = self.centroids[np.asarray(tokens, dtype=np.int64)] * self.scale + self.mean
        return np.sign(x) * np.expm1(np.abs(x))

    def state(self):
        return {"mean": self.mean, "scale": self.scale, "centroids": self.centroids.tolist()}


class PeakTokenizer:
    """One token per frame from its brightest channel: token 0 when no channel reaches
    `threshold` (noise sigma), else 1 + the channel's bin of `width` channels. Fitted on
    nothing, so it splits no symbol by its noise: a narrowband line's token is its position."""

    def __init__(self, threshold, width, channels=CHANNELS, kind="peak"):
        if kind != "peak" or not threshold > 0 or width < 1 or channels % width:
            raise ValueError("A peak tokenizer takes a positive threshold and a bin width that divides the channels")
        self.threshold, self.width, self.channels = float(threshold), int(width), int(channels)

    @property
    def vocab_size(self):
        return 1 + self.channels // self.width

    def encode(self, frames):
        frames = np.asarray(frames)
        return np.where(frames.max(axis=1) >= self.threshold, 1 + frames.argmax(axis=1) // self.width, 0).astype(np.int64)

    def decode(self, tokens):
        """A line twice the threshold at each token's bin centre; no line for token 0."""
        tokens = np.asarray(tokens, dtype=np.int64)
        frames = np.zeros((len(tokens), self.channels), dtype=np.float32)
        lit = tokens > 0
        frames[np.flatnonzero(lit), (tokens[lit] - 1) * self.width + self.width // 2] = 2 * self.threshold
        return frames

    def state(self):
        return {"kind": "peak", "threshold": self.threshold, "width": self.width, "channels": self.channels}


def tokenizer_from_state(state):
    """The tokenizer a state describes: a peak tokenizer, or the k-means one."""
    return PeakTokenizer(**state) if state.get("kind") == "peak" else Tokenizer(**state)


def markov_probabilities(sequences, k):
    counts = np.ones((k, k), dtype=np.float64)  # fixed add-one smoothing; no test tuning
    unigram = np.ones(k, dtype=np.float64)
    for seq in sequences:
        np.add.at(unigram, seq, 1)
        np.add.at(counts, (seq[:-1], seq[1:]), 1)
    return unigram / unigram.sum(), counts / counts.sum(axis=1, keepdims=True)


def surrogate(sequences, kind, seed, probabilities):
    rng = np.random.default_rng(seed)
    unigram, transitions = probabilities
    if kind == "markov":
        first_cdf = np.cumsum(unigram)
        transition_cdf = np.cumsum(transitions, axis=1)
        first_cdf[-1] = 1.0
        transition_cdf[:, -1] = 1.0  # eliminate cumulative-rounding overflow at the last token
    out = []
    for seq in sequences:
        if kind == "real":
            out.append(seq.copy())
        elif kind == "shuffled":
            out.append(rng.permutation(seq))
        elif kind == "markov":
            generated = [int(np.searchsorted(first_cdf, rng.random(), side="right"))]
            for _ in range(len(seq) - 1):
                generated.append(int(np.searchsorted(transition_cdf[generated[-1]], rng.random(), side="right")))
            out.append(np.array(generated, dtype=np.int64))
        else:
            raise ValueError(kind)
    return out
