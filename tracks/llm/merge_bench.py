"""The milestone 2 baseline benchmark (issue #20): does merging shard workers beat the workers,
and how does it compare with one worker trained on all their data?

Fixed before any run: anchor v1 on the 49-frame slices, Voyager, the real stream only, training
windows at every start (stride 1), 50 epochs with the defaults otherwise, seeds 42, 43 and 44,
three shards (one training slice each). Per seed:

- shard workers: each trains 50 epochs from θ₀ on its shard;
- the best worker: the shard worker with the lowest validation score;
- one-shot merge: one round of the three, averaged by training targets;
- one worker on all three slices for 50 epochs: the same data and epochs as the shards together;
- ten rounds of averaging: each round, the three shards train 5 epochs from the last merge.

Added after the first run, as a control for optimizer steps: one worker on all three slices for
as many steps as one shard worker takes (a shard's 48 training windows make 2 batches an epoch,
all three slices' 144 make 5, so 20 epochs instead of 50).

Workers and merges are chosen on validation only. Test and the off-source control are scored
afterwards, for reporting.

    python -m tracks.llm.merge_bench --output runs/merge-bench
"""

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

from .anchor import ROOT, load_anchor
from .data import unique_records
from .merge import held_out, run_round, score
from .model import evaluate
from .runtime import require_new_output, training_runtime
from .train import parser, run

MANIFEST = ROOT / "tracks/llm/data/slices/index.json"
SHARDS = 3
BATCH = 32  # train.py's default batch size


def batches_per_epoch(slices):
    """Training batches an epoch, at stride 1: a window starts at every frame but the last."""
    return math.ceil(sum(len(r["frames"]) - 1 for r in slices) / BATCH)


def step_matched_epochs(epochs):
    """Epochs on all the training slices that take as many optimizer steps as `epochs` on one shard."""
    from .merge import shard_records
    anchor = load_anchor("v1")
    _, partitions = held_out(anchor, "voyager", "within_session_scan_holdout")
    shard = shard_records(partitions["train"], 1, SHARDS)
    return round(epochs * batches_per_epoch(shard) / batches_per_epoch(partitions["train"]))


def train(output, seed, epochs, shard=None, start=None):
    argv = ["--anchor", "v1", "--manifest", str(MANIFEST), "--allow-within-session", "--variants", "real",
            "--train-stride", "1", "--epochs", str(epochs), "--seed", str(seed), "--output", str(output)]
    if shard:
        argv += ["--shard", f"{shard}/{SHARDS}"]
    if start:
        argv += ["--start", str(start)]
    with contextlib.redirect_stdout(io.StringIO()):
        run(parser().parse_args(argv))
    return output / "real.pt"


def merge(output, workers, start=None):
    with contextlib.redirect_stdout(io.StringIO()):
        record = run_round(SimpleNamespace(output=output, anchor="v1", workers=workers, start=start,
                                           tolerance=0.02, max_abs_delta=10.0))
    return record, (output / "merged.pt") if record["merged"]["decision"] == "accepted" else None


def theta0_scores():
    anchor = load_anchor("v1")
    records, partitions = held_out(anchor, "voyager", "within_session_scan_holdout")
    encode = lambda rs: [anchor.tokenizer.encode(r["frames"]) for r in rs]
    model = anchor.model()
    controls = unique_records(r for r in records if r["entry"]["family"] != "voyager")
    return {"validation": evaluate(model, encode(partitions["validation"]))["bits_per_token"],
            "test": evaluate(model, encode(partitions["test"]))["bits_per_token"],
            "control": {"family": "offsource", "bits_per_token": evaluate(model, encode(controls))["bits_per_token"]}}


def one_seed(folder, seed, epochs, rounds):
    out = {}
    shards = [train(folder / f"shard{k}", seed, epochs, shard=k) for k in range(1, SHARDS + 1)]
    out["shard_workers"] = [score(p) for p in shards]
    best = min(range(SHARDS), key=lambda k: out["shard_workers"][k]["validation"])
    out["best_worker"] = {**out["shard_workers"][best], "shard": best + 1}
    record, merged = merge(folder / "one-shot", shards)
    out["one_shot"] = {"decision": record["merged"]["decision"], **(score(merged) if merged else {})}
    out["full_data"] = score(train(folder / "full", seed, epochs))
    out["full_data_step_matched"] = score(train(folder / "full-steps", seed, step_matched_epochs(epochs)))

    incumbent, history = None, []
    for r in range(1, rounds + 1):
        workers = [train(folder / f"r{r}-shard{k}", seed + r, epochs // rounds, shard=k, start=incumbent)
                   for k in range(1, SHARDS + 1)]
        record, merged = merge(folder / f"round{r}", workers, start=incumbent)
        history.append({"round": r, "decision": record["merged"]["decision"],
                        "validation_bits": record["merged"].get("validation_bits")})
        incumbent = merged or incumbent
    out["rounds"] = {"history": history, **(score(incumbent) if incumbent else {})}
    return out


def summarize(results):
    def stats(values):
        return {"mean": statistics.fmean(values), "min": min(values), "max": max(values)} if values else None

    rows = {}
    seeds = list(results["seeds"].values())
    for name, pick in [
        ("shard workers", lambda s: s["shard_workers"]),
        ("best worker (by validation)", lambda s: [s["best_worker"]]),
        ("one-shot merge", lambda s: [s["one_shot"]] if "test" in s["one_shot"] else []),
        ("one worker on all the data", lambda s: [s["full_data"]]),
        ("one worker on all the data, a shard worker's steps", lambda s: [s["full_data_step_matched"]]),
        (f"{results['rounds']} rounds of averaging", lambda s: [s["rounds"]] if "test" in s["rounds"] else []),
    ]:
        runs = [r for s in seeds for r in pick(s)]
        rows[name] = {key: stats([r[key] for r in runs]) for key in ("validation", "test")}
        rows[name]["control"] = stats([r["control"]["bits_per_token"] for r in runs])
        rows[name]["runs"] = len(runs)
    return rows


def table(results):
    fmt = lambda s: "–" if s is None else f"{s['mean']:.2f} ({s['min']:.2f}–{s['max']:.2f})"
    t0 = results["theta0"]
    lines = ["| | Runs | Validation | Test | Off-source control |", "|---|---:|---:|---:|---:|",
             f"| θ₀, untrained | 1 | {t0['validation']:.2f} | {t0['test']:.2f} | {t0['control']['bits_per_token']:.2f} |"]
    for name, row in results["summary"].items():
        lines.append(f"| {name} | {row['runs']} | {fmt(row['validation'])} | {fmt(row['test'])} | {fmt(row['control'])} |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--rounds", type=int, default=10)
    args = p.parse_args(argv)
    if args.epochs % args.rounds:
        p.error("--epochs must be a multiple of --rounds, so the rounds train as many epochs as the one-shot workers")
    require_new_output(args.output)
    args.output.mkdir(parents=True)
    results = {"protocol": __doc__.split("\n\n")[1], "epochs": args.epochs, "rounds": args.rounds,
               "step_matched_epochs": step_matched_epochs(args.epochs), "seeds": {}}
    with training_runtime(1, "cpu"):
        results["theta0"] = theta0_scores()
    for seed in args.seeds:
        results["seeds"][str(seed)] = one_seed(args.output / f"seed{seed}", seed, args.epochs, args.rounds)
        print(f"seed {seed} done", flush=True)
    results["summary"] = summarize(results)
    (args.output / "report.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(table(results))


if __name__ == "__main__":
    main()
