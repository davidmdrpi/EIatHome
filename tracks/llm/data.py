"""Verified signal sequences, observation splits, and a training-only tokenizer."""

import hashlib
import json
from pathlib import Path

import numpy as np

CHANNELS = 64
SPLITS = ("train", "validation", "test")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def load_slices(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "eiathome-slices" or manifest.get("version") != 1:
        raise ValueError("Expected an eiathome-slices version 1 manifest")
    records = []
    for entry in manifest["slices"]:
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
        if not isinstance(start, (int, float)) or not np.isfinite(start):
            raise ValueError("Every slice needs a finite source.tstart_mjd for splitting")
        if entry.get("family") not in ("voyager", "offsource"):
            raise ValueError("Unknown real-data family")
        first = (shape[1] - CHANNELS) // 2
        frames = raw[:, first:first + CHANNELS].copy()
        records.append({"entry": entry, "sha256": sha, "start": float(start),
                        "frames": frames, "frame_hash": digest(frames.tobytes())})
    if not records:
        raise ValueError("The manifest contains no slices")
    families_by_hash = {}
    for record in records:
        previous = families_by_hash.setdefault(record["frame_hash"], record["entry"]["family"])
        if previous != record["entry"]["family"]:
            raise ValueError("Identical signal windows cannot be both signal and off-source controls")
    return sorted(records, key=lambda r: (r["start"], r["sha256"], r["entry"]["id"]))


def groups(records, gap_days):
    """Transitive time/hash components, including copies with inconsistent timestamps."""
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(records):
        for j in range(i):
            b = records[j]
            if (abs(a["start"] - b["start"]) <= gap_days
                    or a["sha256"] == b["sha256"] or a["frame_hash"] == b["frame_hash"]):
                parent[root(i)] = root(j)
    out = {}
    for i, record in enumerate(records):
        out.setdefault(root(i), []).append(record)
    return sorted(out.values(), key=lambda g: (min(r["start"] for r in g), min(r["sha256"] for r in g)))


def split_records(records, family="voyager", allow_within_session=False):
    # Include both families when establishing sessions, so a bridging scan isn't ignored.
    units = [[r for r in g if r["entry"]["family"] == family] for g in groups(records, 0.5)]
    units = [g for g in units if g]
    independent = len(units) >= 3
    if not independent:
        if not allow_within_session:
            raise ValueError("Need at least 3 independent sessions for train/validation/test. "
                             "For a labelled smoke test only, pass --allow-within-session.")
        units = [[r for r in g if r["entry"]["family"] == family] for g in groups(records, 1e-8)]
        units = [g for g in units if g]
        if len(units) < 3:
            raise ValueError("Need at least 3 distinct scans even for a within-session smoke test")
    # Chronological 60/20/20, with at least one whole group in every split.
    n_eval = max(1, len(units) // 5)
    partitions = (units[:-2 * n_eval], units[-2 * n_eval:-n_eval], units[-n_eval:])
    result = {}
    for name, partition in zip(SPLITS, partitions):
        # Copies don't provide extra training or evaluation evidence.
        unique = {}
        for group in partition:
            for r in group:
                unique.setdefault(r["frame_hash"], r)
        result[name] = list(unique.values())
    return result, independent


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
        # Bound the temporary distance matrix for larger slice collections.
        return np.concatenate([((batch[:, None] - self.centroids[None]) ** 2).sum(-1).argmin(-1)
                               for batch in np.array_split(x, max(1, (len(x) + 255) // 256))])

    def encode(self, frames):
        return self._nearest((signed_log(frames) - self.mean) / self.scale).astype(np.int64)

    def decode(self, tokens):
        x = self.centroids[np.asarray(tokens, dtype=np.int64)] * self.scale + self.mean
        return np.sign(x) * np.expm1(np.abs(x))

    def state(self):
        return {"mean": self.mean, "scale": self.scale, "centroids": self.centroids.tolist()}


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
    out = []
    for seq in sequences:
        if kind == "real":
            out.append(seq.copy())
        elif kind == "shuffled":
            out.append(rng.permutation(seq))
        elif kind == "markov":
            generated = [rng.choice(len(unigram), p=unigram)]
            for _ in range(len(seq) - 1):
                generated.append(rng.choice(len(unigram), p=transitions[generated[-1]]))
            out.append(np.array(generated, dtype=np.int64))
        else:
            raise ValueError(kind)
    return out
