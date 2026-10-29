"""Merging volunteer runs within a family (issue #20, milestone 2), starting with the simplest
merge that can be measured: rounds of workers trained from a shared start, each checked and
gated on validation, then averaged by training samples.

A round starts from an incumbent: θ₀ for round 1, then the previous round's merged checkpoint.
Each worker trains from it on a disjoint shard of the training slices and contributes
τᵢ = θᵢ − θ_start:

    python -m tracks.llm.train --anchor v1 --manifest tracks/llm/data/slices/index.json \\
      --allow-within-session --variants real --shard 1/3 --seed 42 --output runs/r1-s1
    python -m tracks.llm.merge round --output runs/round1 runs/r1-s*/real.pt
    python -m tracks.llm.train ... --start runs/round1/merged.pt --shard 1/3 --output runs/r2-s1
    python -m tracks.llm.merge round --start runs/round1/merged.pt --output runs/round2 runs/r2-s*/real.pt

    python -m tracks.llm.merge replay runs/round1/round.json   # recompute the round from its record
    python -m tracks.llm.merge score runs/round1/merged.pt     # test and control scores, for reporting only

A round never reads the test split: workers and the merged model are selected on validation
only. Its record names every contributor (by file hash), what became of it and why, the
weighting rule, and the merged weights' digest, so anyone holding the same files can replay it.
"""

import argparse
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .anchor import ANCHORS, ROOT, codebook_digest, difference, load_anchor, weights_digest
from .data import SPLITS, load_slices, split_records, unique_records
from .model import SignalLM, evaluate
from .runtime import atomic_output, require_new_output, training_runtime

TOLERANCE_BITS = 0.02  # how much worse than the incumbent on validation a worker or merge may be
MAX_ABS_DELTA = 10.0  # a sanity bound on any one parameter's change: θ₀'s weights are of order 1


def parse_shard(text):
    """'2/3' -> (2, 3): the second of three shards."""
    try:
        index, count = (int(part) for part in str(text).split("/"))
    except ValueError:
        raise ValueError(f"--shard takes i/n, like 2/3; got {text!r}") from None
    if not 1 <= index <= count:
        raise ValueError(f"--shard {text}: need 1 ≤ i ≤ n")
    return index, count


def shard_records(records, index, count):
    """Shard `index` of `count` (1-based) of the training slices: every count-th slice in a fixed
    order (by content), so shards are disjoint, cover the split together, and don't depend on the
    manifest's order."""
    ordered = sorted(records, key=lambda r: r["sha256"])
    if len(ordered) < count:
        raise ValueError(f"{count} shards need at least {count} training slices; this split has {len(ordered)}")
    return ordered[index - 1::count]


def anchor_named(identity):
    """The pinned anchor a record or checkpoint names, refusing one this code doesn't pin."""
    for name, spec in ANCHORS.items():
        if spec["id"] == (identity or {}).get("anchor"):
            anchor = load_anchor(name)
            if anchor.identity() == identity:
                return anchor
    raise ValueError("It names an anchor this code doesn't pin")


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_checkpoint(path, anchor):
    """A checkpoint trained or merged from this anchor, with its tokenizer and model settings."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "eiathome-signal-lm" or checkpoint.get("version") != 1:
        raise ValueError(f"{path}: not a supported signal-LM checkpoint")
    if checkpoint.get("anchor") != anchor.identity():
        raise ValueError(f"{path}: not trained from anchor {anchor.name}")
    if codebook_digest(checkpoint["tokenizer"]) != anchor.spec["codebook_sha256"] or checkpoint.get("config") != anchor.spec["model"]:
        raise ValueError(f"{path}: its tokenizer or model settings aren't anchor {anchor.name}'s")
    if not isinstance(checkpoint.get("merge"), dict):
        raise ValueError(f"{path}: has no round record (train it with --anchor, after this change)")
    return checkpoint


def load_round(anchor, path):
    """A merged round's checkpoint, to start the next round from: its weights, its round number
    and their digest. Refuses one whose weights don't match the digest its round recorded."""
    checkpoint = load_checkpoint(path, anchor)
    merge = checkpoint["merge"]
    if not merge.get("merged") or not isinstance(merge.get("round"), int) or merge["round"] < 1:
        raise ValueError(f"{path}: not a merged round's checkpoint")
    state = checkpoint["model"]
    difference(state, anchor.theta0)  # the anchor's parameters, in order
    digest = weights_digest(state)
    if digest != merge.get("weights_sha256"):
        raise ValueError(f"{path}: its weights don't match the digest its round recorded")
    return state, merge["round"], digest


def held_out(anchor, family, evaluation, manifest=None):
    """The family's split as train.py made it, and the validation tokens the gate scores on."""
    records = load_slices(manifest or ROOT / anchor.spec["manifest"])
    partitions, independent = split_records(records, family, allow_within_session=evaluation != "independent_sessions")
    if ("independent_sessions" if independent else "within_session_scan_holdout") != evaluation:
        raise ValueError("The manifest no longer splits as the workers' runs did")
    return records, partitions


def worker_problem(checkpoint, start_round, start_sha, max_abs_delta, tau):
    """Why a worker's delta can't join the round, or None."""
    merge = checkpoint["merge"]
    if merge.get("merged"):
        return "it's a merged checkpoint, not a worker's"
    if checkpoint.get("variant") != "real":
        return f"it's the {checkpoint.get('variant')} surrogate, not the real stream"
    if merge.get("round") != start_round + 1 or merge.get("start_sha256") != start_sha:
        return f"it was trained for round {merge.get('round')} from another start"
    if not all(torch.isfinite(v).all() for v in tau.values()):
        return "its delta isn't finite"
    largest = max(v.abs().max().item() for v in tau.values())
    if largest > max_abs_delta:
        return f"its delta reaches {largest:.3g}, beyond the bound of {max_abs_delta:g}"
    if not isinstance(merge.get("training_targets"), int) or merge["training_targets"] < 1:
        return "it records no training targets to weight it by"
    return None


def model_from(anchor, state):
    model = SignalLM(**anchor.spec["model"])
    model.load_state_dict(OrderedDict((k, v.float()) for k, v in state.items()))
    return model


def add(base, delta):
    """base + delta, as float32 weights."""
    return OrderedDict((k, (v.double() + delta[k]).float()) for k, v in base.items())


def weighted_average(deltas, weights):
    """Σ wᵢ τᵢ with Σ wᵢ = 1, in float64."""
    total = float(sum(weights))
    return OrderedDict((name, sum(w / total * d[name] for d, w in zip(deltas, weights))) for name in deltas[0])


def merge_round(anchor, workers, start=None, tolerance=TOLERANCE_BITS, max_abs_delta=MAX_ABS_DELTA, manifest=None):
    """One round: check each worker, gate it on validation against the incumbent, average the
    accepted deltas by training targets, and gate the result. workers: checkpoint paths.
    Returns (record, merged state or None)."""
    if start is None:
        start_state, start_round, start_sha, start_name = anchor.theta0, 0, anchor.spec["theta0_sha256"], "theta0"
    else:
        start_state, start_round, start_sha = load_round(anchor, start)
        start_name = str(start)
    loaded = [(str(path), load_checkpoint(path, anchor)) for path in workers]
    if not loaded:
        raise ValueError("A round needs at least one worker")
    families = {c["splits"]["train"][0]["family"] for _, c in loaded}
    evaluations = {c["evaluation"] for _, c in loaded}
    if len(families) != 1 or len(evaluations) != 1 or len({c["dataset_sha256"] for _, c in loaded}) != 1:
        raise ValueError("A round merges workers of one family, on one dataset and split")
    family, evaluation = families.pop(), evaluations.pop()
    records, partitions = held_out(anchor, family, evaluation, manifest)
    for path, c in loaded:
        if any(c["splits"][name] != [r["entry"] for r in partitions[name]] for name in ("validation", "test")):
            raise ValueError(f"{path}: its validation or test split isn't this round's")
    validation = [anchor.tokenizer.encode(r["frames"]) for r in partitions["validation"]]
    score = lambda state: evaluate(model_from(anchor, state), validation)["bits_per_token"]
    incumbent = score(start_state)

    contributors, accepted, seen = [], [], set()
    for path, c in loaded:
        merge = c["merge"]
        entry = {"file": path, "file_sha256": file_digest(path), "shard": merge.get("shard"), "seed": merge.get("seed"),
                 "training_targets": merge.get("training_targets"), "validation_bits": None, "weight": 0.0}
        key = (tuple(merge.get("shard") or ()), merge.get("seed"))
        tau = difference(c["model"], start_state)
        problem = worker_problem(c, start_round, start_sha, max_abs_delta, tau)
        if problem is None and key in seen:
            problem = "duplicate: the same shard and seed as an earlier contributor, so counted once"
        if problem is None:
            seen.add(key)
            entry["validation_bits"] = score(add(start_state, tau))
            if entry["validation_bits"] > incumbent + tolerance:
                problem = f"it scores {entry['validation_bits']:.4f} bits on validation, worse than the incumbent's {incumbent:.4f} by more than {tolerance:g}"
        entry.update(status="rejected" if problem else "accepted", reason=problem)
        contributors.append(entry)
        if not problem:
            accepted.append((entry, tau))

    record = {
        "format": "eiathome-signal-lm-merge-round", "version": 1, "anchor": anchor.identity(),
        "round": start_round + 1, "family": family, "evaluation": evaluation, "dataset_sha256": loaded[0][1]["dataset_sha256"],
        "start": {"checkpoint": start_name, "round": start_round, "weights_sha256": start_sha, "validation_bits": incumbent},
        "rule": {"merge": "average weighted by training targets", "gate_split": "validation", "tolerance_bits": tolerance,
                 "max_abs_delta": max_abs_delta, "duplicates": "same shard and seed counted once"},
        "contributors": contributors,
    }
    if not accepted:
        record["merged"] = {"decision": "rejected", "reason": "no worker passed the checks and the gate"}
        return record, None
    total = sum(e["training_targets"] for e, _ in accepted)
    for e, _ in accepted:
        e["weight"] = e["training_targets"] / total
    merged = add(start_state, weighted_average([t for _, t in accepted], [e["weight"] for e, _ in accepted]))
    merged_bits = score(merged)
    best = min((e for e, _ in accepted), key=lambda e: e["validation_bits"])
    record["best_worker"] = {"file": best["file"], "validation_bits": best["validation_bits"]}
    worse = merged_bits > incumbent + tolerance
    record["merged"] = {
        "decision": "rejected" if worse else "accepted",
        "reason": f"the merged model scores {merged_bits:.4f} bits on validation, worse than the incumbent's {incumbent:.4f}" if worse else None,
        "validation_bits": merged_bits, "weights_sha256": weights_digest(merged), "contributors": len(accepted),
    }
    return record, None if worse else merged


def merged_checkpoint(anchor, record, state, worker):
    """The merged model as a checkpoint: loadable by sample.py, and a start for the next round."""
    return {
        "format": "eiathome-signal-lm", "version": 1, "variant": "real", "config": anchor.spec["model"],
        "model": OrderedDict((k, v.clone()) for k, v in state.items()), "tokenizer": anchor.tokenizer.state(),
        "dataset_sha256": record["dataset_sha256"], "splits": {**worker["splits"], "train": None},
        "evaluation": record["evaluation"], "anchor": anchor.identity(),
        "merge": {"merged": True, "round": record["round"], "start_sha256": record["start"]["weights_sha256"],
                  "weights_sha256": record["merged"]["weights_sha256"]},
    }


def run_round(args):
    require_new_output(args.output)
    anchor = load_anchor(args.anchor)
    with training_runtime(1, "cpu"), atomic_output(args.output) as output:
        record, state = merge_round(anchor, args.workers, args.start, args.tolerance, args.max_abs_delta)
        (output / "round.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        if state is not None:
            worker = load_checkpoint(args.workers[0], anchor)
            torch.save(merged_checkpoint(anchor, record, state, worker), output / "merged.pt")
    print(summary(record))
    return record


def summary(record):
    lines = [f"round {record['round']}: incumbent {record['start']['validation_bits']:.4f} bits on validation"]
    for c in record["contributors"]:
        bits = "" if c["validation_bits"] is None else f" {c['validation_bits']:.4f} bits,"
        lines.append(f"  {c['file']}: shard {c['shard']}, seed {c['seed']},{bits} {c['status']}" + (f" ({c['reason']})" if c["reason"] else f", weight {c['weight']:.3f}"))
    m = record["merged"]
    lines.append(f"  merged: {m['decision']}" + (f", {m['validation_bits']:.4f} bits on validation" if "validation_bits" in m else "")
                 + (f" ({m['reason']})" if m.get("reason") else ""))
    return "\n".join(lines)


def replay(path, manifest=None):
    """Recomputes a round from its record and the same files, and checks it comes out the same."""
    record = json.loads(Path(path).read_text())
    if record.get("format") != "eiathome-signal-lm-merge-round":
        raise ValueError(f"{path}: not a merge round record")
    anchor = anchor_named(record.get("anchor"))
    for c in record["contributors"]:
        if file_digest(c["file"]) != c["file_sha256"]:
            raise ValueError(f"{c['file']} isn't the file the round used")
    start = None if record["start"]["checkpoint"] == "theta0" else record["start"]["checkpoint"]
    rule = record["rule"]
    with training_runtime(1, "cpu"):
        again, _ = merge_round(anchor, [c["file"] for c in record["contributors"]], start, rule["tolerance_bits"],
                               rule["max_abs_delta"], manifest)
    if again != record:
        raise ValueError(f"{path}: replaying the round gave a different result")
    return record


def score(path, manifest=None):
    """Test bits for a checkpoint from an anchor, and its family's control: for reporting, never
    for choosing a worker or a merge."""
    anchor = anchor_named(torch.load(path, map_location="cpu", weights_only=True).get("anchor"))
    checkpoint = load_checkpoint(path, anchor)
    family = next(s for name in ("validation", "test") for s in checkpoint["splits"][name])["family"]
    records, partitions = held_out(anchor, family, checkpoint["evaluation"], manifest)
    model = model_from(anchor, checkpoint["model"])
    controls = unique_records(r for r in records if r["entry"]["family"] != family)
    encode = lambda rs: [anchor.tokenizer.encode(r["frames"]) for r in rs]
    out = {name: evaluate(model, encode(partitions[name]))["bits_per_token"] for name in ("validation", "test")}
    if controls:
        out["control"] = {"family": controls[0]["entry"]["family"], "bits_per_token": evaluate(model, encode(controls))["bits_per_token"]}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("round", help="merge workers' checkpoints into the next round")
    r.add_argument("workers", nargs="+", type=Path, help="workers' real.pt checkpoints")
    r.add_argument("--output", type=Path, required=True, help="a new folder for round.json and merged.pt")
    r.add_argument("--start", type=Path, default=None, help="the previous round's merged.pt (default: θ₀, round 1)")
    r.add_argument("--anchor", default="v1", choices=list(ANCHORS))
    r.add_argument("--tolerance", type=float, default=TOLERANCE_BITS, help="bits/token a worker or merge may lose on validation")
    r.add_argument("--max-abs-delta", type=float, default=MAX_ABS_DELTA)
    sub.add_parser("replay", help="recompute a round from its record").add_argument("record", type=Path)
    sub.add_parser("score", help="test and control bits, for reporting only").add_argument("checkpoint", type=Path)
    args = p.parse_args(argv)
    try:
        if args.command == "round":
            if not math.isfinite(args.tolerance) or args.tolerance < 0 or not args.max_abs_delta > 0:
                raise ValueError("The tolerance must be ≥ 0 and the bound positive")
            run_round(args)
        elif args.command == "replay":
            record = replay(args.record)
            print(f"round {record['round']} replays exactly: merged weights {record['merged'].get('weights_sha256', '(none)')}")
        else:
            with training_runtime(1, "cpu"):
                print(json.dumps(score(args.checkpoint), indent=2))
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
