"""The volunteer checkpoint pilot (signal-lm-pilot/v0): volunteers' own training runs, sent as
versioned delta artifacts, independently evaluated, and merged into one model.

Every earlier merge in this repository merged workers one person trained. Here each worker is a
volunteer's: they train on a shard of the session benchmark's training sessions (BENCHMARK.md),
from the pilot's current model, and send the change as a delta artifact in a GitHub issue. A
coordinator (CI) checks and stores it; once two or more volunteers' contributions are in, a
maintainer merges them, and the next round starts from the merge.

**The setting.** The session benchmark's: anchor v4, the sessions corpus and its pinned split, five
session shards. Round 1 is the benchmark's published seed-42 base (shard 1). Rounds after it are
open to shards 2 to 5. Training uses the benchmark's settings (up to 1,000 epochs, patience 50,
lr 3e-4, batch 32, one CPU thread).

**The artifact** (`eiathome-signal-lm-delta`, version 2) is one compact JSON document:

- its identities: the pilot; the anchor (its id, codebook, record and θ₀ digests); the tokenizer's
  digest; the dataset (the corpus manifest's digest, the split's pin, the family and evaluation);
  the round and the digest of the weights it starts from; the shard (index, count, unit, a digest
  of its slices, its training targets);
- how it was trained: seed, settings, best epoch, seconds, environment;
- the delta: int8 values with one float32 scale per tensor, in the anchor's parameter order, and
  the SHA-256 of the float32 delta they rebuild. (Quantizing so moves validation by under 1e-4
  bits, and the whole artifact fits in an issue: about 40 KB.)

Its id is the SHA-256 of its text. The issue carries it with a card: a short summary (round,
shard, seed, the artifact's id and size, the volunteer's own validation score) that the
coordinator's reply turns into a receipt.

**Independent evaluation.** The coordinator trusts nothing the issue says. It refuses an artifact
that doesn't decode to its recorded digest, or whose identities aren't the pilot's next round. It
then rebuilds the worker (the current model plus the delta) as a checkpoint and runs merge.py's
own round on it alone. That checks the round, the start, the shard's slices and training targets,
and the bound on the delta, and gates it on validation against the current model (within 0.02
bits). Last, it retrains the claimed shard with the claimed seed itself. The contribution is
**replicated** when that run lands within 0.02 bits of it on validation and their deltas' cosine
is at least 0.9. Only validated and replicated contributions are merged. (Retraining every
contribution is affordable only because the model is tiny; at scale, it would be spot checks.)

**Influence.** Each volunteer counts once per shard per round; a second contribution, an identical
artifact, or the same shard and seed again is a duplicate. In a merge, merge.py's shard weights
give each shard its share of the training targets, split equally among the contributions on it,
so ten volunteers on one shard weigh what one would. A merge needs contributions from at least two
volunteers, and the merged model must pass the same gate against the current one.

**Storage, rollback and replay.** Accepted artifacts are stored by id under `pilot/v0/submissions`,
merged models under `pilot/v0/rounds` (raw float32, as the session benchmark publishes them), and
`pilot/v0/ledger.json` records every event in order: submissions (with the coordinator's scores),
exclusions, rounds (merge.py's whole record) and rollbacks. The current model is the latest round
not rolled back. A rollback returns to an earlier round and keeps everything: the contributions
merged after it are eligible again unless excluded. `verify` replays the ledger: every artifact's
digest and identities, every evaluation, and every merge from its stored artifacts, bit for bit.

    python -m tracks.llm.pilot status                       # the current round, and each shard's coverage
    python -m tracks.llm.pilot contribute --output my-run   # train a shard; writes the artifact and the issue to open
    python -m tracks.llm.pilot merge --by <login>           # maintainers: merge the round (needs 2+ volunteers)
    python -m tracks.llm.pilot rollback --to-round 2 --reason "..." --by <login>
    python -m tracks.llm.pilot exclude <artifact id> --reason "..." --by <login>
    python -m tracks.llm.pilot verify                       # replay the ledger
    python -m tracks.llm.pilot export [--check]             # the Live Lab's copy of the status (web/data/pilot_status.json)

Every command that changes the ledger (ingest, merge, rollback, exclude) also rewrites that copy.
"""

import argparse
import base64
from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import tempfile
from typing import NamedTuple

import numpy as np
import torch

from . import session_bench as sessions
from .anchor import ROOT, canonical_layout, codebook_digest, difference, load_anchor, weights_digest
from .data import digest
from .merge import check_rule, held_out, load_checkpoint, merge_round, model_from, shard_records
from .model import evaluate
from .runtime import training_runtime

PILOT = "signal-lm-pilot/v0"
FOLDER = Path(__file__).resolve().parent / "pilot" / "v0"
REPOSITORY = "davidmdrpi/EIatHome"
ISSUE_PREFIX = "Checkpoint contribution:"
ISSUE_LIMIT = 65_536  # characters in a GitHub issue body
START = sessions.FOLDER / "checkpoints" / "seed42" / "base.bin"  # round 1: the session benchmark's base
START_SHA256 = "ffbaa49dd11479e9f6b8baac10aa349bff86334b54dcb1f6c1e88564931e11da"
OPEN_SHARDS = sessions.WORKER_SHARDS
TRAINING = {"epochs": sessions.EPOCHS, "patience": sessions.PATIENCE, "lr": 3e-4, "batch_size": 32, "device": "cpu", "threads": 1}
RULE = check_rule(None)  # merge.py's plain average and its gate
MIN_VOLUNTEERS = 2
REPLICATION = {"validation_bits": 0.02, "cosine": 0.9}
SEEDS = range(0, 2**31)  # what NumPy's and PyTorch's generators all take (train.py seeds both)
SCORE_TOLERANCE = 1e-6  # bits, between environments (session_bench.py's)
ARTIFACT_FORMAT, ARTIFACT_VERSION = "eiathome-signal-lm-delta", 2
CARD_FORMAT, CARD_VERSION = "eiathome-checkpoint-card", 1
LEDGER_FORMAT, LEDGER_VERSION = "eiathome-checkpoint-ledger", 1
BLOCK = re.compile(r"```(eiathome-checkpoint-card|eiathome-delta)[ \t]*\r?\n(.*?)\r?\n```", re.S)


# ---------- the setting ----------

def posix(path):
    """A path relative to the repository, with forward slashes on every system."""
    return Path(os.path.relpath(Path(path).resolve(), ROOT)).as_posix()


class Pilot:
    """The pilot's fixed setting: the anchor, the corpus and its pinned split, and each shard's
    slices; and where its ledger, artifacts and rounds are kept."""

    def __init__(self, folder=FOLDER, manifest=None):
        self.folder = Path(folder)
        self.anchor = load_anchor(sessions.ANCHOR)
        self.manifest = sessions.MANIFEST if manifest is None else Path(manifest)
        split = sessions.check_split(self.manifest)
        self.split_sha256 = digest(sessions.split_text(split).encode())
        self.dataset_sha256 = split["dataset_sha256"]
        _, self.partitions = held_out(self.anchor, sessions.FAMILY, sessions.EVALUATION, self.manifest, self.dataset_sha256)
        self.shards = {k: shard_records(self.partitions["train"], k, sessions.SHARDS, sessions.UNIT) for k in range(1, sessions.SHARDS + 1)}
        self.validation = [self.anchor.tokenizer.encode(r["frames"]) for r in self.partitions["validation"]]

    def identities(self):
        """What every artifact must say it was trained on."""
        return {"pilot": PILOT, "anchor": self.anchor.identity(),
                "tokenizer": {"kind": self.anchor.tokenizer.state().get("kind"), "sha256": codebook_digest(self.anchor.tokenizer.state())},
                "dataset": {"manifest": posix(self.manifest), "sha256": self.dataset_sha256,
                            "split_sha256": self.split_sha256, "family": sessions.FAMILY, "evaluation": sessions.EVALUATION}}

    def shard(self, index):
        records = self.shards[index]
        return {"index": index, "count": sessions.SHARDS, "unit": sessions.UNIT,
                "slices_sha256": digest(json.dumps(sorted(r["sha256"] for r in records)).encode()),
                "training_targets": sum(len(r["frames"]) - 1 for r in records)}

    def bits(self, state):
        with training_runtime(1, "cpu"):
            return evaluate(model_from(self.anchor, state), self.validation)["bits_per_token"]

    def held(self):
        return {name: [r["entry"] for r in self.partitions[name]] for name in ("validation", "test")}


# ---------- artifacts ----------

def layout_of(anchor):
    return canonical_layout(anchor.spec["model"])[0]


def rebuild(layout, q, scales):
    """The float64 delta int8 values and per-tensor scales stand for, by tensor name."""
    out, offset = OrderedDict(), 0
    for t, scale in zip(layout, scales):
        out[t["name"]] = torch.from_numpy(q[offset:offset + t["count"]].astype(np.float64) * scale).reshape(t["shape"])
        offset += t["count"]
    return out


def delta_digest(tau):
    return digest(b"".join(np.ascontiguousarray(v.numpy(), dtype="<f4").tobytes() for v in tau.values()))


def encode_artifact(pilot, head, shard, training, tau):
    """A worker's delta from the head as an artifact: its text (compact JSON) and its id."""
    layout = layout_of(pilot.anchor)
    scales, chunks = [], []
    for t in layout:
        flat = tau[t["name"]].double().flatten().numpy()
        scale = float(np.float32(np.abs(flat).max() / 127)) if flat.size else 0.0
        chunks.append(np.zeros(flat.size, np.int8) if scale == 0 else np.clip(np.rint(flat / scale), -127, 127).astype(np.int8))
        scales.append(scale)
    q = np.concatenate(chunks)
    artifact = {
        "format": ARTIFACT_FORMAT, "version": ARTIFACT_VERSION, **pilot.identities(),
        "round": {"number": head.round + 1, "start_sha256": head.weights_sha256},
        "shard": pilot.shard(shard), "training": training,
        "encoding": {"values": "int8", "scales": "float32, one per tensor", "order": "the anchor's parameters"},
        "tensors": [[t["name"], list(t["shape"])] for t in layout],
        "scales": scales, "values": base64.b64encode(q.tobytes()).decode(),
        "delta_sha256": delta_digest(rebuild(layout, q, scales)),
    }
    text = json.dumps(artifact, separators=(",", ":"), sort_keys=True)
    return text, digest(text.encode())


def decode_artifact(anchor, text):
    """An artifact's fields and the float64 delta it rebuilds. Refuses one that is malformed or
    doesn't rebuild the delta its digest names."""
    try:
        artifact = json.loads(text)
    except ValueError as err:
        raise ValueError("the artifact isn't JSON") from err
    if not isinstance(artifact, dict) or artifact.get("format") != ARTIFACT_FORMAT or artifact.get("version") != ARTIFACT_VERSION:
        raise ValueError(f"it isn't a version-{ARTIFACT_VERSION} {ARTIFACT_FORMAT} artifact")
    layout = layout_of(anchor)
    if artifact.get("tensors") != [[t["name"], list(t["shape"])] for t in layout]:
        raise ValueError("its tensors aren't anchor v4's parameters")
    scales = artifact.get("scales")
    if (not isinstance(scales, list) or len(scales) != len(layout)
            or not all(isinstance(s, float) and np.isfinite(s) and s >= 0 and float(np.float32(s)) == s for s in scales)):
        raise ValueError("its scales aren't one finite, non-negative float32 per tensor")
    try:
        q = np.frombuffer(base64.b64decode(artifact.get("values", ""), validate=True), dtype=np.int8)
    except (ValueError, TypeError) as err:
        raise ValueError("its values aren't base64") from err
    if q.size != sum(t["count"] for t in layout) or (q == -128).any():
        raise ValueError("its values aren't one int8 in [-127, 127] per parameter")
    tau = rebuild(layout, q, scales)
    if delta_digest(tau) != artifact.get("delta_sha256"):
        raise ValueError("its values don't rebuild the delta its digest names")
    return artifact, tau


def valid_seed(seed):
    return type(seed) is int and seed in SEEDS


def identity_problems(pilot, artifact, head):
    """How an artifact's identities differ from what the pilot's next round needs (empty: none)."""
    expected = pilot.identities()
    problems = [f"its {name} isn't the pilot's" for name in ("pilot", "anchor", "tokenizer", "dataset") if artifact.get(name) != expected[name]]
    if artifact.get("round") != {"number": head.round + 1, "start_sha256": head.weights_sha256}:
        problems.append(f"it isn't for the pilot's next round, round {head.round + 1} from the current model: "
                        "pull the repository and run `contribute` again")
    shard = artifact.get("shard")
    index = shard.get("index") if isinstance(shard, dict) else None
    if type(index) is not int or index not in OPEN_SHARDS:
        problems.append(f"its shard isn't one of the open shards ({', '.join(map(str, OPEN_SHARDS))})")
    elif shard != pilot.shard(index):
        problems.append(f"its shard's slices or training targets aren't shard {index}'s")
    training = artifact.get("training")
    if not isinstance(training, dict) or any(training.get(k) != v for k, v in TRAINING.items()):
        problems.append("it wasn't trained with the pilot's settings")
    elif not valid_seed(training.get("seed")):
        problems.append(f"its seed isn't an integer from 0 to {SEEDS.stop - 1}")
    return problems


# ---------- checkpoints ----------

class Head(NamedTuple):
    round: int
    weights_sha256: str
    checkpoint: str  # its weights: round 1's relative to the repository, later rounds' to the pilot's folder

    def path(self, pilot):
        return ROOT / self.checkpoint if self.round == 1 else pilot.folder / self.checkpoint


def weights_header(pilot, state, merge_info):
    data = b"".join(np.ascontiguousarray(v.detach().cpu().numpy(), dtype="<f4").tobytes() for v in state.values())
    return {"format": "eiathome-signal-lm-weights", "version": 1, "pilot": PILOT, "anchor": pilot.anchor.identity(),
            "config": pilot.anchor.spec["model"], "dtype": "float32-le", "dataset_sha256": pilot.dataset_sha256,
            "evaluation": sessions.EVALUATION, "merge": merge_info, "tensors": canonical_layout(pilot.anchor.spec["model"])[0],
            "sha256": digest(data)}, data


def head_checkpoint(pilot, head):
    """The head as a merged checkpoint merge.py and train.py take as a start."""
    path = head.path(pilot)
    if head.round == 1:
        checkpoint = sessions.load_published(pilot.anchor, path, pilot.manifest)
    else:
        header = json.loads(path.with_suffix(".json").read_text())
        data = path.with_suffix(".bin").read_bytes()
        if (header.get("pilot") != PILOT or header.get("anchor") != pilot.anchor.identity() or digest(data) != header.get("sha256")
                or header.get("tensors") != canonical_layout(pilot.anchor.spec["model"])[0]):
            raise ValueError(f"{head.checkpoint}: not this pilot's weights, or they don't match their header")
        values = np.frombuffer(data, dtype="<f4")
        state = OrderedDict((t["name"], torch.from_numpy(values[t["offset"]:t["offset"] + t["count"]].copy()).reshape(t["shape"]))
                            for t in header["tensors"])
        checkpoint = {"format": "eiathome-signal-lm", "version": 1, "variant": "real", "config": pilot.anchor.spec["model"], "model": state,
                      "tokenizer": pilot.anchor.tokenizer.state(), "dataset_sha256": pilot.dataset_sha256, "evaluation": sessions.EVALUATION,
                      "anchor": pilot.anchor.identity(), "splits": {"train": None, **pilot.held()}, "outer_momentum": None,
                      "merge": header["merge"]}
    if weights_digest(checkpoint["model"]) != head.weights_sha256 or checkpoint["merge"].get("weights_sha256") != head.weights_sha256:
        raise ValueError(f"{head.checkpoint}: its weights aren't the round's")
    return checkpoint


def worker_checkpoint(pilot, head_state, artifact, tau):
    """A contribution as the worker checkpoint merge.py checks: the head plus its delta, with the
    training slices and round record its artifact claims."""
    shard, training = artifact["shard"], artifact["training"]
    return {"format": "eiathome-signal-lm", "version": 1, "variant": "real", "config": pilot.anchor.spec["model"],
            "model": OrderedDict((k, (v.double() + tau[k]).float()) for k, v in head_state.items()),
            "tokenizer": pilot.anchor.tokenizer.state(), "dataset_sha256": pilot.dataset_sha256, "evaluation": sessions.EVALUATION,
            "anchor": pilot.anchor.identity(), "best_epoch": training.get("best_epoch"),
            "splits": {"train": [r["entry"] for r in pilot.shards[shard["index"]]], **pilot.held()},
            "merge": {"round": artifact["round"]["number"], "start_sha256": artifact["round"]["start_sha256"],
                      "shard": [shard["index"], shard["count"]], "shard_unit": shard["unit"], "seed": training["seed"],
                      "training_targets": shard["training_targets"]}}


def saved(checkpoint, folder, name):
    path = Path(folder) / f"{name}.pt"
    torch.save(checkpoint, path)
    return path


# ---------- the ledger ----------

def new_ledger():
    return {"format": LEDGER_FORMAT, "version": LEDGER_VERSION, "pilot": PILOT,
            "start": {"round": 1, "weights_sha256": START_SHA256, "checkpoint": posix(START)}, "events": []}


def load_ledger(folder=FOLDER):
    path = Path(folder) / "ledger.json"
    ledger = json.loads(path.read_text()) if path.exists() else new_ledger()
    if (ledger.get("format"), ledger.get("version"), ledger.get("pilot")) != (LEDGER_FORMAT, LEDGER_VERSION, PILOT):
        raise ValueError(f"{path}: not this pilot's ledger")
    if ledger["start"] != new_ledger()["start"]:
        raise ValueError(f"{path}: it doesn't start from the session benchmark's base")
    return ledger


def save_ledger(ledger, folder=FOLDER):
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / "ledger.json").write_text(json.dumps(ledger, indent=1, ensure_ascii=False) + "\n")


class State(NamedTuple):
    heads: list  # the live chain of rounds, round 1 first; the last is the current model
    submissions: dict  # artifact id -> its submission event, in ledger order
    excluded: dict  # artifact id -> the reason
    merged: dict  # artifact id -> the live round that merged it

    @property
    def head(self):
        return self.heads[-1]

    def eligible(self):
        """The contributions the next merge takes: validated and replicated (by the rule, on the
        recorded numbers), for the next round from the current model, neither excluded nor already
        in a live round."""
        return [s for i, s in self.submissions.items()
                if s["round"] == self.head.round + 1 and s["start_sha256"] == self.head.weights_sha256
                and s["status"] == "validated" and replication_problem(s) is None and replicates(s["evaluation"], s["replication"])
                and i not in self.excluded and i not in self.merged]


def state_of(ledger):
    """What the ledger's events add up to. Refuses events that don't follow from the ones before."""
    start = ledger["start"]
    heads = [Head(1, start["weights_sha256"], start["checkpoint"])]
    submissions, excluded, merged = OrderedDict(), {}, {}
    for n, event in enumerate(ledger["events"], 1):
        kind = event.get("event")
        if kind == "submission":
            if event["id"] in submissions:
                raise ValueError(f"event {n}: artifact {event['id'][:12]} is recorded twice")
            submissions[event["id"]] = event
        elif kind == "exclusion":
            if event["submission"] not in submissions or event["submission"] in excluded:
                raise ValueError(f"event {n}: excludes an unknown or already excluded contribution")
            excluded[event["submission"]] = event["reason"]
        elif kind == "round":
            if event["round"] != heads[-1].round + 1 or event["start_sha256"] != heads[-1].weights_sha256:
                raise ValueError(f"event {n}: a round that doesn't follow the current model")
            if event["decision"] == "accepted":
                heads.append(Head(event["round"], event["weights_sha256"], event["checkpoint"]))
                for i in event["merged"]:
                    merged[i] = event["round"]
        elif kind == "rollback":
            if not any(h.round == event["to_round"] for h in heads[:-1]):
                raise ValueError(f"event {n}: rolls back to a round that isn't an earlier live one")
            while heads[-1].round != event["to_round"]:
                gone = heads.pop().round
                merged = {i: r for i, r in merged.items() if r != gone}
            if heads[-1].weights_sha256 != event["weights_sha256"]:
                raise ValueError(f"event {n}: the rollback's weights aren't round {event['to_round']}'s")
        else:
            raise ValueError(f"event {n}: unknown event")
    return State(heads, submissions, excluded, merged)


# ---------- volunteers ----------

def environment():
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__), "platform": platform.platform()}


def coverage(state):
    """Contributions waiting for the next merge, by open shard: how many, and from how many volunteers."""
    eligible = state.eligible()
    return {k: {"contributions": sum(s["shard"] == k for s in eligible), "volunteers": len({s["volunteer"] for s in eligible if s["shard"] == k})}
            for k in OPEN_SHARDS}


def train_worker(pilot, head_pt, shard, seed, folder):
    """One run of train.py on a shard from the head, with the pilot's settings."""
    return sessions.train(Path(folder) / "worker", seed, TRAINING["epochs"], pilot.manifest, shard=shard, start=head_pt,
                          patience=TRAINING["patience"])


def card_of(artifact_id, text, artifact, validation_bits):
    shard, training = artifact["shard"], artifact["training"]
    return {"format": CARD_FORMAT, "version": CARD_VERSION, "pilot": PILOT, "round": artifact["round"]["number"],
            "shard": shard["index"], "seed": training["seed"], "artifact_sha256": artifact_id, "artifact_bytes": len(text.encode()),
            "training_targets": shard["training_targets"], "best_epoch": training["best_epoch"], "validation_bits": round(validation_bits, 6)}


def issue_text(card, text):
    title = f"{ISSUE_PREFIX} round {card['round']}, shard {card['shard']}"
    body = (f"A checkpoint contribution to the volunteer pilot (`{PILOT}`, tracks/llm/PILOT.md): round {card['round']}, shard "
            f"{card['shard']}, seed {card['seed']}. It reaches {card['validation_bits']:.4f} bits per token on validation by the "
            "volunteer's own run; the coordinator scores it again and replies with a receipt.\n\n"
            f"```eiathome-checkpoint-card\n{json.dumps(card, indent=1)}\n```\n\n```eiathome-delta\n{text}\n```\n")
    return title, body


def contribute(pilot, output, shard=None, seed=None, ledger=None):
    """Trains a shard from the current model and writes what to submit: the artifact, its card,
    and the issue's title and body."""
    state = state_of(load_ledger(pilot.folder) if ledger is None else ledger)
    head = state.head
    if shard is None:  # the open shard with the fewest contributions waiting, the lowest on a tie
        cover = coverage(state)
        shard = min(OPEN_SHARDS, key=lambda k: (cover[k]["contributions"], k))
    if shard not in OPEN_SHARDS:
        raise ValueError(f"Shard {shard} isn't open: contribute to one of {', '.join(map(str, OPEN_SHARDS))}")
    seed = random.SystemRandom().randrange(SEEDS.start, SEEDS.stop) if seed is None else seed
    if not valid_seed(seed):
        raise ValueError(f"The seed must be an integer from 0 to {SEEDS.stop - 1}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = head_checkpoint(pilot, head)
    with tempfile.TemporaryDirectory() as tmp:
        run = train_worker(pilot, saved(start, tmp, "head"), shard, seed, tmp)
        worker = load_checkpoint(run["checkpoint"], pilot.anchor)
    tau = difference(worker["model"], start["model"])
    training = {"seed": seed, **TRAINING, "best_epoch": run["best_epoch"], "epochs_run": run["epochs_run"],
                "seconds": round(run["seconds"], 3), "environment": environment()}
    text, artifact_id = encode_artifact(pilot, head, shard, training, tau)
    artifact, rebuilt = decode_artifact(pilot.anchor, text)
    bits = pilot.bits(OrderedDict((k, (v.double() + rebuilt[k]).float()) for k, v in start["model"].items()))
    card = card_of(artifact_id, text, artifact, bits)
    title, body = issue_text(card, text)
    if len(body) > ISSUE_LIMIT:
        raise ValueError(f"The issue body takes {len(body):,} characters, over GitHub's {ISSUE_LIMIT:,}")
    (output / "artifact.json").write_text(text)
    (output / "card.json").write_text(json.dumps(card, indent=1) + "\n")
    (output / "issue.md").write_text(body)
    (output / "title.txt").write_text(title + "\n")
    return {"card": card, "title": title, "issue": output / "issue.md", "artifact_id": artifact_id}


# ---------- the coordinator ----------

def parse_issue(body):
    """The card and the artifact's text in an issue body."""
    blocks = BLOCK.findall(body or "")
    kinds = [kind for kind, _ in blocks]
    if sorted(kinds) != ["eiathome-checkpoint-card", "eiathome-delta"]:
        raise ValueError("the issue needs exactly one `eiathome-checkpoint-card` block and one `eiathome-delta` block")
    found = dict(blocks)
    try:
        card = json.loads(found["eiathome-checkpoint-card"])
    except ValueError as err:
        raise ValueError("its card isn't JSON") from err
    if not isinstance(card, dict) or (card.get("format"), card.get("version"), card.get("pilot")) != (CARD_FORMAT, CARD_VERSION, PILOT):
        raise ValueError(f"its card isn't a version-{CARD_VERSION} {CARD_FORMAT} for {PILOT}")
    return card, found["eiathome-delta"].strip()


def replicate(pilot, head_pt, start_state, artifact, tau, folder):
    """The coordinator's own run of the claimed shard and seed, compared with the contribution."""
    run = train_worker(pilot, head_pt, artifact["shard"]["index"], artifact["training"]["seed"], folder)
    again = difference(load_checkpoint(run["checkpoint"], pilot.anchor)["model"], start_state)
    a = torch.cat([v.flatten() for v in tau.values()])
    b = torch.cat([v.flatten() for v in again.values()])
    cosine = float(a @ b / (a.norm() * b.norm())) if a.norm() > 0 and b.norm() > 0 else 0.0
    bits = pilot.bits(OrderedDict((k, (v.double() + again[k]).float()) for k, v in start_state.items()))
    return {"validation_bits": bits, "cosine": round(cosine, 6), "best_epoch": run["best_epoch"]}


def evaluate_contribution(pilot, head, artifact, tau, replicate_run=True):
    """The coordinator's evaluation: merge.py's checks and gate on the contribution alone, and its
    own run of the claimed shard and seed."""
    start = head_checkpoint(pilot, head)
    with tempfile.TemporaryDirectory() as tmp:
        head_pt = saved(start, tmp, "head")
        worker_pt = saved(worker_checkpoint(pilot, start["model"], artifact, tau), tmp, "worker")
        record, _ = merge_round(pilot.anchor, [worker_pt], start=head_pt, rule=RULE, manifest=pilot.manifest)
        entry = record["contributors"][0]
        evaluation = {"status": "validated" if entry["status"] == "accepted" else "rejected", "reason": entry["reason"],
                      "validation_bits": entry["validation_bits"], "start_validation_bits": record["start"]["validation_bits"],
                      "max_abs_delta": max(float(v.abs().max()) for v in tau.values())}
        replication = replicate(pilot, head_pt, start["model"], artifact, tau, tmp) if replicate_run and entry["status"] == "accepted" else None
    if replication is not None:
        replication["replicated"] = replicates(evaluation, replication)
    return evaluation, replication


def replicates(evaluation, replication):
    """The replication rule, on recorded numbers: the coordinator's run within REPLICATION's bits of
    the contribution on validation, and their deltas' cosine at least REPLICATION's."""
    return (abs(replication["validation_bits"] - evaluation["validation_bits"]) <= REPLICATION["validation_bits"]
            and replication["cosine"] >= REPLICATION["cosine"])


def replication_problem(event):
    """Why a submission's recorded replication doesn't hold together (None if it does): its numbers
    must be finite and in range, and its decision the rule's on them."""
    finite = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    evaluation, replication = event.get("evaluation"), event.get("replication")
    if not isinstance(evaluation, dict) or not finite(evaluation.get("validation_bits")):
        return "its evaluation has no finite validation score"
    if (not isinstance(replication, dict) or not finite(replication.get("validation_bits")) or not finite(replication.get("cosine"))
            or not -1 <= replication["cosine"] <= 1 or type(replication.get("best_epoch")) is not int
            or type(replication.get("replicated")) is not bool):
        return "its replication record is malformed"
    if replication["replicated"] != replicates(evaluation, replication):
        return f"its replication is recorded as {replication['replicated']}, but the rule makes it {not replication['replicated']}"
    return None


def receipt(event, state):
    cover = coverage(state)
    shards = ", ".join(f"shard {k}: {c['contributions']} from {c['volunteers']} volunteer(s)" for k, c in cover.items())
    volunteers = len({s["volunteer"] for s in state.eligible()})
    replication = event["replication"]
    replicated = ("**replicated**" if replication["replicated"] else
                  f"**not replicated** (that needs a run within {REPLICATION['validation_bits']} bits and a cosine of at least "
                  f"{REPLICATION['cosine']}), so it won't be merged")
    return (f"Thanks for contributing a checkpoint! Here's your receipt.\n\n"
            f"- **Stored:** artifact `{event['id'][:16]}…` ({event['bytes']:,} bytes) at "
            f"[`tracks/llm/pilot/v0/{event['artifact']}`](https://github.com/{REPOSITORY}/blob/main/tracks/llm/pilot/v0/{event['artifact']}).\n"
            f"- **Independently evaluated:** round {event['round']}, shard {event['shard']}, seed {event['seed']}. On validation it "
            f"scores {event['evaluation']['validation_bits']:.4f} bits per token, against the current model's "
            f"{event['evaluation']['start_validation_bits']:.4f}: validated.\n"
            f"- **Replication:** the coordinator retrained shard {event['shard']} with seed {event['seed']}: "
            f"{replication['validation_bits']:.4f} bits, cosine {replication['cosine']:.3f} with yours. {replicated}.\n\n"
            f"Round {event['round']} now has contributions waiting from {volunteers} volunteer(s) ({shards}). A maintainer merges "
            f"the round once at least {MIN_VOLUNTEERS} volunteers' contributions are in; tracks/llm/PILOT.md explains the rest.")


def ingest(pilot, issue, ledger):
    """One issue, checked against the ledger as it is now. Returns (status, comment, artifact text
    or None): accepted (record the event and store the artifact), recorded (this issue was recorded
    before), duplicate, or rejected. Nothing from the issue is echoed back but numbers it checked."""
    state = state_of(ledger)
    login, number = str(issue.get("user", {}).get("login", "")), issue.get("number")
    earlier = next((s for s in state.submissions.values() if s["issue"] == number), None)
    if earlier is not None:
        return "recorded", receipt(earlier, state), None
    how = "\n\nRun `python -m tracks.llm.pilot contribute` from an up-to-date clone (tracks/llm/PILOT.md) and open a new issue with what it writes."
    try:
        card, text = parse_issue(issue.get("body"))
        artifact_id = digest(text.encode())
        if card.get("artifact_sha256") != artifact_id or card.get("artifact_bytes") != len(text.encode()):
            raise ValueError("the card's artifact digest or size isn't the artifact's")
        artifact, tau = decode_artifact(pilot.anchor, text)
    except ValueError as err:
        return "rejected", f"This contribution couldn't be read: {err}.{how}", None
    problems = identity_problems(pilot, artifact, state.head)
    if not problems and (card.get("round"), card.get("shard"), card.get("seed")) != (
            artifact["round"]["number"], artifact["shard"]["index"], artifact["training"]["seed"]):
        problems.append("the card's round, shard or seed isn't the artifact's")
    if problems:
        return "rejected", "This contribution couldn't be accepted:\n\n" + "\n".join(f"- {p}" for p in problems) + how, None
    shard, seed = artifact["shard"]["index"], artifact["training"]["seed"]
    same = lambda s: s["round"] == state.head.round + 1 and s["start_sha256"] == state.head.weights_sha256 and s["shard"] == shard
    if artifact_id in state.submissions:
        return "duplicate", f"This artifact was already recorded from #{state.submissions[artifact_id]['issue']}, so it wasn't counted again.", None
    mine = next((s for s in state.submissions.values() if same(s) and s["volunteer"] == login), None)
    if mine is not None:
        return "duplicate", (f"You already contributed to shard {shard} of round {mine['round']} in #{mine['issue']}. Each volunteer counts "
                             f"once per shard per round, so this one wasn't added: try another shard."), None
    twin = next((s for s in state.submissions.values() if same(s) and s["seed"] == seed), None)
    if twin is not None:
        return "duplicate", (f"Shard {shard} was already trained with seed {seed} for this round (#{twin['issue']}): the same run "
                             "counts once. `contribute` picks a random seed if you leave it out."), None
    evaluation, replication = evaluate_contribution(pilot, state.head, artifact, tau)
    if evaluation["status"] != "validated":
        return "rejected", f"The coordinator's evaluation didn't accept this contribution: {evaluation['reason']}.{how}", None
    event = {"event": "submission", "id": artifact_id, "issue": number, "volunteer": login, "submitted_at": issue.get("created_at"),
             "round": artifact["round"]["number"], "start_sha256": artifact["round"]["start_sha256"], "shard": shard, "seed": seed,
             "artifact": f"submissions/{artifact_id}.json", "bytes": len(text.encode()), "card_sha256": digest(json.dumps(card, sort_keys=True).encode()),
             "volunteer_validation_bits": claimed if isinstance(claimed := card.get("validation_bits"), float) and math.isfinite(claimed) else None,
             "status": "validated", "evaluation": evaluation, "replication": replication}
    ledger["events"].append(event)
    return "accepted", receipt(event, state_of(ledger)), text


def store(folder, ledger, artifact_id, text):
    (Path(folder) / "submissions").mkdir(parents=True, exist_ok=True)
    (Path(folder) / "submissions" / f"{artifact_id}.json").write_text(text)
    save_ledger(ledger, folder)


# ---------- maintainers ----------

def stored_artifact(pilot, event):
    text = (pilot.folder / event["artifact"]).read_text()
    if digest(text.encode()) != event["id"]:
        raise ValueError(f"{event['artifact']}: its text isn't the artifact its id names")
    return decode_artifact(pilot.anchor, text)


def run_merge(pilot, head, events):
    """merge.py's round on stored contributions, from the head. Returns (record, merged state or
    None), the record naming contributions by artifact id."""
    start = head_checkpoint(pilot, head)
    with tempfile.TemporaryDirectory() as tmp:
        head_pt = saved(start, tmp, "head")
        paths = {}
        for event in events:
            artifact, tau = stored_artifact(pilot, event)
            paths[str(saved(worker_checkpoint(pilot, start["model"], artifact, tau), tmp, event["id"]))] = event["id"]
        record, merged = merge_round(pilot.anchor, list(paths), start=head_pt, rule=RULE, manifest=pilot.manifest)
    for entry in record["contributors"]:
        entry["submission"] = paths[entry.pop("file")]
        entry.pop("file_sha256")
    if "best_worker" in record:
        record["best_worker"] = {"submission": paths[record["best_worker"]["file"]], "validation_bits": record["best_worker"]["validation_bits"]}
    record["start"]["checkpoint"] = head.checkpoint
    control = record.get("control")
    if control:  # the control's slices, by digest rather than by their whole manifest entries
        ids = [entry["id"] for entry in control["selection"]]
        control["selection"] = {"slices": len(ids), "ids_sha256": digest(json.dumps(ids).encode())}
    return record, None if merged is None else merged.state


def volunteer_weights(record, events):
    """Each volunteer's total weight in a merge: the shard cap bounds each shard, not each volunteer,
    so one volunteer on several shards can hold most of it. Recorded so that shows."""
    who = {e["id"]: e["volunteer"] for e in events}
    out = {}
    for entry in record["contributors"]:
        out[who[entry["submission"]]] = out.get(who[entry["submission"]], 0.0) + entry["weight"]
    return dict(sorted(out.items()))


def merge(pilot, ledger, by):
    """Merges the current round's eligible contributions. Adds the round to the ledger (and, if
    accepted, writes its weights); returns the event."""
    state = state_of(ledger)
    eligible = state.eligible()
    volunteers = sorted({s["volunteer"] for s in eligible})
    if len(volunteers) < MIN_VOLUNTEERS:
        raise ValueError(f"Round {state.head.round + 1} has contributions from {len(volunteers)} volunteer(s); a merge needs {MIN_VOLUNTEERS}")
    record, merged = run_merge(pilot, state.head, eligible)
    accepted = record["merged"]["decision"] == "accepted"
    event = {"event": "round", "round": record["round"], "start_sha256": state.head.weights_sha256, "by": by,
             "submissions": [s["id"] for s in eligible], "volunteers": volunteers,
             "volunteer_weights": volunteer_weights(record, eligible), "decision": record["merged"]["decision"],
             "merged": [e["submission"] for e in record["contributors"] if e["weight"] > 0] if accepted else [],
             "weights_sha256": record["merged"].get("weights_sha256") if accepted else None,
             "checkpoint": None, "record": record}
    if accepted:
        event["checkpoint"] = f"rounds/{event['weights_sha256']}.bin"
        path = pilot.folder / event["checkpoint"]
        info = {"merged": True, "round": record["round"], "start_sha256": state.head.weights_sha256,
                "weights_sha256": event["weights_sha256"], "outer": record["rule"]["outer"], "momentum_sha256": None}
        header, data = weights_header(pilot, merged, info)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.with_suffix(".json").write_text(json.dumps(header, indent=1) + "\n")
    ledger["events"].append(event)
    return event


def rollback(ledger, to_round, reason, by):
    state = state_of(ledger)
    target = next((h for h in state.heads[:-1] if h.round == to_round), None)
    if target is None:
        raise ValueError(f"Round {to_round} isn't an earlier round of the current chain ({', '.join(str(h.round) for h in state.heads)})")
    event = {"event": "rollback", "to_round": to_round, "weights_sha256": target.weights_sha256, "reason": reason, "by": by,
             "rolled_back": [h.round for h in state.heads if h.round > to_round]}
    ledger["events"].append(event)
    state_of(ledger)
    return event


def exclude(ledger, submission, reason, by):
    state = state_of(ledger)
    matches = [i for i in state.submissions if i.startswith(submission)] if len(submission) >= 8 else []
    if len(matches) != 1:
        raise ValueError(f"{submission!r} names {len(matches)} recorded contributions; give at least 8 characters of one artifact id")
    if matches[0] in state.excluded:
        raise ValueError("That contribution is already excluded")
    event = {"event": "exclusion", "submission": matches[0], "reason": reason, "by": by}
    ledger["events"].append(event)
    note = state.merged.get(matches[0])
    return event, None if note is None else f"It stays in round {note} until that round is rolled back."


# ---------- replay ----------

def close(a, b, tolerance=SCORE_TOLERANCE):
    return a is None and b is None or a is not None and b is not None and abs(a - b) <= tolerance


def verify(pilot, ledger=None, replicate_runs=False):
    """Replays the ledger. Each contribution: stored under its id, decoding to its recorded delta,
    for the round and start that were current when it came, with the pilot's identities, and
    evaluating as recorded (with replicate_runs, retrained as well). Each round: from contributions
    eligible then, from at least two volunteers, merged again from the stored artifacts to the same
    record and weights. No file is stored that the ledger doesn't name."""
    ledger = load_ledger(pilot.folder) if ledger is None else ledger
    state_of(ledger)  # the events are consistent
    partial = {**ledger, "events": []}
    named = set()
    for n, event in enumerate(ledger["events"], 1):
        before = state_of(partial)
        where = f"event {n}"
        if event["event"] == "submission":
            artifact, tau = stored_artifact(pilot, event)
            named.add(event["artifact"])
            problems = identity_problems(pilot, artifact, before.head)
            if problems:
                raise ValueError(f"{where}: {problems[0]}")
            if (event["round"], event["start_sha256"], event["shard"], event["seed"]) != (
                    artifact["round"]["number"], artifact["round"]["start_sha256"], artifact["shard"]["index"], artifact["training"]["seed"]):
                raise ValueError(f"{where}: the event's round, start, shard or seed isn't its artifact's")
            evaluation, replication = evaluate_contribution(pilot, before.head, artifact, tau, replicate_run=replicate_runs)
            recorded = event["evaluation"]
            if evaluation["status"] != event["status"] or not all(close(evaluation[k], recorded[k]) for k in ("validation_bits", "start_validation_bits")):
                raise ValueError(f"{where}: the contribution evaluates as {evaluation['status']} at {evaluation['validation_bits']!r} bits; "
                                 f"the ledger says {event['status']} at {recorded['validation_bits']!r}")
            problem = replication_problem(event)
            if problem:
                raise ValueError(f"{where}: {problem}")
            if replication is not None and replication["replicated"] != event["replication"]["replicated"]:
                raise ValueError(f"{where}: retrained here, the contribution's replication comes out {replication['replicated']}")
        elif event["event"] == "round":
            eligible = before.eligible()
            if event["submissions"] != [s["id"] for s in eligible]:
                raise ValueError(f"{where}: the round didn't merge exactly the contributions eligible then")
            if len({s["volunteer"] for s in eligible}) < MIN_VOLUNTEERS:
                raise ValueError(f"{where}: the round merged fewer than {MIN_VOLUNTEERS} volunteers' contributions")
            record, merged = run_merge(pilot, before.head, eligible)
            check_record(record, event["record"], where)
            check_record(volunteer_weights(record, eligible), event["volunteer_weights"], f"{where} (volunteer weights)")
            if event["decision"] != record["merged"]["decision"]:
                raise ValueError(f"{where}: merged again, the round is {record['merged']['decision']}")
            if record["merged"]["decision"] == "accepted":
                if event["weights_sha256"] != weights_digest(merged):
                    raise ValueError(f"{where}: merged again, the round makes other weights")
                head = Head(event["round"], event["weights_sha256"], event["checkpoint"])
                head_checkpoint(pilot, head)  # the stored weights are these
                named.update({event["checkpoint"], str(Path(event["checkpoint"]).with_suffix(".json"))})
        partial["events"].append(event)
    stored = {str(p.relative_to(pilot.folder)) for sub in ("submissions", "rounds") if (pilot.folder / sub).is_dir()
              for p in (pilot.folder / sub).iterdir()}
    if stored != named:
        raise ValueError(f"{pilot.folder}: stores files the ledger doesn't name, or misses some it does: {sorted(stored ^ named)}")
    return len(ledger["events"])


def check_record(again, recorded, where):
    """A replayed merge record against the stored one: everything equal, scores to the tolerance."""
    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
            return all(walk(a[k], b[k], f"{path}.{k}") for k in a)
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            return all(walk(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b)))
        if isinstance(a, float) and isinstance(b, float):
            if abs(a - b) <= SCORE_TOLERANCE:
                return True
        elif a == b or (path.endswith(".reason") and isinstance(a, str) and isinstance(b, str)):
            return True
        raise ValueError(f"{where}: merged again, the record's {path.lstrip('.')} is {a!r}, not {b!r}")
    walk(json.loads(json.dumps(again)), recorded, "")


# ---------- the published status ----------

STATUS_FORMAT, STATUS_VERSION = "eiathome-pilot-status", 1
STATUS_PATH = ROOT / "web" / "data" / "pilot_status.json"  # what the Live Lab's Contributions tab reads


def ledger_text(ledger):
    """The ledger as save_ledger writes it: its digest names the ledger a status was made from."""
    return json.dumps(ledger, indent=1, ensure_ascii=False) + "\n"


def export(pilot, ledger=None):
    """The ledger as the Live Lab shows it (web/data/pilot_status.json): each contribution's stage
    and the numbers each check decided it on, each round's merge with its gate and shard weights,
    rollbacks and exclusions, and the open round's coverage. Made from the ledger alone, nothing
    rescored, so `export --check` can tell whether the published copy is the ledger's."""
    ledger = load_ledger(pilot.folder) if ledger is None else ledger
    state = state_of(ledger)
    eligible = {s["id"] for s in state.eligible()}
    volunteer = {i: s["volunteer"] for i, s in state.submissions.items()}

    # Rounds in ledger order, each marked live or with the rollback that undid it.
    rounds, live, rollbacks, exclusions = [], [], [], []
    for n, event in enumerate(ledger["events"], 1):
        if event["event"] == "round":
            record = event["record"]
            entry = {
                "event": n, "round": event["round"], "by": event["by"], "decision": event["decision"],
                "reason": record["merged"].get("reason"), "volunteers": event["volunteers"],
                "volunteer_weights": event["volunteer_weights"], "weights_sha256": event["weights_sha256"],
                "checkpoint": None if event["checkpoint"] is None else f"tracks/llm/pilot/v0/{event['checkpoint']}",
                "gate_bits": record["rule"]["tolerance_bits"],
                "start_validation_bits": record["start"]["validation_bits"],
                "merged_validation_bits": record["merged"].get("validation_bits"),
                "best_worker": record.get("best_worker"),
                "contributors": [{"submission": c["submission"], "volunteer": volunteer.get(c["submission"]), "shard": c["shard"][0],
                                  "seed": c["seed"], "training_targets": c["training_targets"], "status": c["status"],
                                  "reason": c["reason"], "validation_bits": c["validation_bits"], "weight": c["weight"]}
                                 for c in record["contributors"]],
                "live": False, "rolled_back": None,
            }
            rounds.append(entry)
            if event["decision"] == "accepted":
                entry["live"] = True
                live.append(entry)
        elif event["event"] == "rollback":
            while live and live[-1]["round"] > event["to_round"]:
                gone = live.pop()
                gone.update(live=False, rolled_back={"event": n, "to_round": event["to_round"], "reason": event["reason"], "by": event["by"]})
            rollbacks.append({"event": n, **{k: event[k] for k in ("to_round", "reason", "by", "rolled_back")}})
        elif event["event"] == "exclusion":
            exclusions.append({"event": n, **{k: event[k] for k in ("submission", "reason", "by")}})

    excluded = {x["submission"]: x for x in exclusions}

    def stage(i, s):
        """Where a recorded contribution stands, and why. A contribution excluded after a merge took
        it stays merged: it's in the current model until that round is rolled back, and its
        `exclusion` says it's out of later merges."""
        if i in state.merged:
            later = f"; excluded from later merges, it stays in the current model until round {state.merged[i]} is rolled back" if i in state.excluded else ""
            return "merged", f"in round {state.merged[i]}, the current chain{later}"
        if i in state.excluded:
            return "excluded", state.excluded[i]
        problem = replication_problem(s)
        if problem is not None:
            return "not replicated", problem
        if not replicates(s["evaluation"], s["replication"]):
            return "not replicated", "the coordinator's run of its shard and seed didn't land within the rule"
        if i in eligible:
            return "eligible", f"waiting for round {s['round']}'s merge"
        took = [r for r in rounds if any(c["submission"] == i for c in r["contributors"])]
        if took and took[-1]["rolled_back"]:
            r = took[-1]
            return "rolled back", f"round {r['round']} was rolled back to round {r['rolled_back']['to_round']}: {r['rolled_back']['reason']}"
        if took:
            mine = next(c for c in took[-1]["contributors"] if c["submission"] == i)
            return "not merged", mine["reason"] or f"round {took[-1]['round']} gave it no weight"
        return "superseded", f"round {s['round']}'s start is no longer the current model"

    submissions = []
    for i, s in state.submissions.items():
        where, why = stage(i, s)
        evaluation, replication = s["evaluation"], s["replication"]
        submissions.append({
            "id": i, "issue": s["issue"], "volunteer": s["volunteer"], "submitted_at": s["submitted_at"],
            "round": s["round"], "start_sha256": s["start_sha256"], "shard": s["shard"], "seed": s["seed"],
            "bytes": s["bytes"], "artifact": f"tracks/llm/pilot/v0/{s['artifact']}",
            "volunteer_validation_bits": s["volunteer_validation_bits"],
            "evaluation": {k: evaluation.get(k) for k in ("status", "reason", "validation_bits", "start_validation_bits", "max_abs_delta")},
            "replication": None if replication is None else {k: replication.get(k) for k in ("validation_bits", "cosine", "best_epoch", "replicated")},
            "stage": where, "why": why,
            "exclusion": None if i not in excluded else {k: excluded[i][k] for k in ("event", "reason", "by")},
            "rounds": [{"event": r["event"], "round": r["round"], "decision": r["decision"], "live": r["live"],
                        **{k: c[k] for k in ("status", "reason", "weight")}}
                       for r in rounds for c in r["contributors"] if c["submission"] == i],
        })

    waiting = state.eligible()
    return {
        "format": STATUS_FORMAT, "version": STATUS_VERSION, "pilot": PILOT,
        "ledger": {"path": "tracks/llm/pilot/v0/ledger.json", "sha256": digest(ledger_text(ledger).encode()), "events": len(ledger["events"])},
        "rules": {
            "gate_bits": RULE["tolerance_bits"], "max_abs_delta": RULE["max_abs_delta"], "merge": RULE["merge"],
            "replication": REPLICATION, "min_volunteers": MIN_VOLUNTEERS, "seeds": [SEEDS.start, SEEDS.stop - 1],
            "influence": "each shard weighs its share of the merged shards' training targets, split equally among its contributions",
        },
        "current": {"round": state.head.round, "weights_sha256": state.head.weights_sha256, "checkpoint": posix(state.head.path(pilot))},
        "open": {
            "round": state.head.round + 1,
            "volunteers": len({s["volunteer"] for s in waiting}),
            "shards": [{"index": k, "training_targets": pilot.shard(k)["training_targets"], **c} for k, c in coverage(state).items()],
        },
        "submissions": submissions,
        "rounds": rounds,
        "rollbacks": rollbacks,
        "exclusions": exclusions,
    }


def status_text(pilot, ledger=None):
    return json.dumps(export(pilot, ledger), indent=1, ensure_ascii=False) + "\n"


def write_status(pilot, ledger=None, path=STATUS_PATH):
    Path(path).write_text(status_text(pilot, ledger))


def status_problem(pilot, ledger=None, path=STATUS_PATH):
    """Why the published status isn't the ledger's (None if it is)."""
    path = Path(path)
    if not path.exists():
        return f"{posix(path)} is missing: run python -m tracks.llm.pilot export"
    if path.read_text() != status_text(pilot, ledger):
        return f"{posix(path)} isn't the ledger's status: run python -m tracks.llm.pilot export"
    return None


# ---------- the command line ----------

def status(pilot, ledger=None):
    state = state_of(load_ledger(pilot.folder) if ledger is None else ledger)
    where = os.path.relpath(state.head.path(pilot), ROOT)
    lines = [f"{PILOT}: the current model is round {state.head.round} ({state.head.weights_sha256[:12]}, {where}).",
             f"Round {state.head.round + 1} is open to shards {', '.join(map(str, OPEN_SHARDS))}."]
    for k, c in coverage(state).items():
        lines.append(f"  shard {k}: {c['contributions']} contribution(s) waiting, from {c['volunteers']} volunteer(s)")
    volunteers = len({s["volunteer"] for s in state.eligible()})
    lines.append(f"{volunteers} volunteer(s) so far; a merge needs {MIN_VOLUNTEERS}.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="the current round and each shard's coverage")
    c = commands.add_parser("contribute", help="train a shard and write the artifact and issue to submit")
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--shard", type=int, choices=OPEN_SHARDS, default=None, help="default: the open shard with the fewest contributions")
    c.add_argument("--seed", type=int, default=None, help="default: random")
    i = commands.add_parser("ingest", help="(CI) check one issue's contribution: tools/ingest_issues.mjs --kind checkpoint runs it")
    i.add_argument("event", type=Path)
    i.add_argument("comment", type=Path)
    m = commands.add_parser("merge", help="merge the open round's contributions")
    m.add_argument("--by", required=True)
    r = commands.add_parser("rollback", help="return to an earlier round")
    r.add_argument("--to-round", type=int, required=True)
    r.add_argument("--reason", required=True)
    r.add_argument("--by", required=True)
    x = commands.add_parser("exclude", help="keep a contribution out of later merges")
    x.add_argument("submission", help="its artifact id (at least 8 characters)")
    x.add_argument("--reason", required=True)
    x.add_argument("--by", required=True)
    v = commands.add_parser("verify", help="replay the ledger")
    v.add_argument("--replicate", action="store_true", help="retrain every contribution too")
    e = commands.add_parser("export", help="write the Live Lab's copy of the pilot's status (web/data/pilot_status.json)")
    e.add_argument("--check", action="store_true", help="only check that the published copy is the ledger's")
    args = parser.parse_args(argv)

    if args.command == "ingest":
        # The issue is untrusted: it's only parsed as JSON and checked. Status goes to stdout's last line.
        result = "error"
        try:
            pilot = Pilot()
            issue = json.loads(args.event.read_text())["issue"]
            ledger = load_ledger(pilot.folder)
            result, comment, text = ingest(pilot, issue, ledger)
            if result == "accepted":
                store(pilot.folder, ledger, ledger["events"][-1]["id"], text)
                write_status(pilot, ledger)
        except Exception as err:  # noqa: BLE001 - reported to the driver, which retries
            comment = f"Something went wrong while checking this contribution ({type(err).__name__}). It will be retried."
            print(f"{type(err).__name__}: {err}", file=sys.stderr)
        args.comment.write_text(comment.strip() + "\n")
        print(json.dumps({"status": result}))
        return
    pilot = Pilot()
    if args.command == "status":
        print(status(pilot))
    elif args.command == "contribute":
        out = contribute(pilot, args.output, args.shard, args.seed)
        print(f"Trained shard {out['card']['shard']} (seed {out['card']['seed']}): {out['card']['validation_bits']:.4f} bits on validation.\n"
              f"Open an issue at https://github.com/{REPOSITORY}/issues/new titled\n\n  {out['title']}\n\n"
              f"with the contents of {out['issue']} as its body, or run\n\n"
              f"  gh issue create --repo {REPOSITORY} --title \"{out['title']}\" --body-file {out['issue']}")
    elif args.command in ("merge", "rollback", "exclude"):
        ledger = load_ledger(pilot.folder)
        if args.command == "merge":
            event = merge(pilot, ledger, args.by)
            print(f"Round {event['round']}: {event['decision']} ({event['record']['merged'].get('reason') or 'passed the gate'}); "
                  f"{len(event['merged'])} of {len(event['submissions'])} contributions merged.")
        elif args.command == "rollback":
            event = rollback(ledger, args.to_round, args.reason, args.by)
            print(f"Rolled back to round {event['to_round']}; rounds {event['rolled_back']} are kept in the ledger.")
        else:
            event, note = exclude(ledger, args.submission, args.reason, args.by)
            print(f"Excluded {event['submission'][:16]}… from later merges." + (f" {note}" if note else ""))
        save_ledger(ledger, pilot.folder)
        write_status(pilot, ledger)
    elif args.command == "export":
        if args.check:
            problem = status_problem(pilot)
            if problem:
                sys.exit(problem)
            print(f"{posix(STATUS_PATH)} is the ledger's status")
        else:
            write_status(pilot)
            print(f"Wrote {posix(STATUS_PATH)}")
    else:
        count = verify(pilot, replicate_runs=args.replicate)
        print(f"{pilot.folder}: {count} events replayed")


if __name__ == "__main__":
    main()
