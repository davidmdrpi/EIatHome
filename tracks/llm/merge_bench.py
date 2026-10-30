"""The milestone 2 benchmark (issue #20): do merged shard workers beat the workers, and how do
they compare with one worker trained on all their data?

Fixed before any run: anchor v1 on the 49-frame slices, Voyager, the real stream only, training
windows at every start (stride 1), 50 epochs with the defaults otherwise, seeds 42, 43 and 44,
three shards (one training slice each). Per seed:

- shard workers: each trains 50 epochs from θ₀ on its shard;
- the best worker: the shard worker with the lowest validation score;
- one-shot merges of those three workers, one round from θ₀, by each rule below;
- one worker on all three slices for 50 epochs: the same data and epochs as the shards together;
- ten rounds: each round, the three shards train 5 epochs from the last merge, by each chain's
  rule below.

Added after the first run, as a control for optimizer steps: one worker on all three slices for
as many steps as one shard worker takes (a shard's 48 training windows make 2 batches an epoch,
all three slices' 144 make 5, so 20 epochs instead of 50).

The rules (milestone 2's second part, fixed before its first run): DARE drops p = 0.9 at random
(the issue's card estimate) and sends int8 values; TIES keeps the top 20% by magnitude (the TIES
paper's default); DiLoCo's outer step uses lr 0.7 and Nesterov momentum 0.9 (the DiLoCo paper's).
Every gate uses a tolerance of 0.02 bits on validation. (Until the review of the first run, the
gate also vetoed a worker that scored better than θ₀ on the off-source selection control; it's
now recorded as a diagnostic.)

Workers and merges are chosen on validation only. Test and the reporting control, the
off-source test scan, are scored afterwards, for reporting.

The selection control's diagnostic, how much a model gains from the real order of the off-source
validation scan over shuffled orders, is recorded for every model, and for a positive control
added with it: one worker per seed trained 50 epochs on the off-source family itself.

    python -m tracks.llm.merge_bench --output runs/merge-bench
"""

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import statistics

from .anchor import ROOT, load_anchor
from .merge import controls, held_out, load_checkpoint, model_from, order_gain, run_round, score, shard_records
from .model import evaluate
from .runtime import require_new_output, training_runtime
from .train import parser, run

MANIFEST = ROOT / "tracks/llm/data/slices/index.json"
SHARDS = 3
BATCH = 32  # train.py's default batch size
DARE = {"method": "random", "p": 0.9}
DILOCO = {"lr": 0.7, "momentum": 0.9}
ONE_SHOT = {  # name: rule
    "average (FedAvg)": {},
    "average of compressed deltas (DARE)": {"dare": DARE},
    "average, greedy soup": {"soup": "greedy"},
    "TIES": {"merge": "ties", "ties_keep": 0.2},
    "DARE-TIES": {"merge": "ties", "dare": DARE},
    "DARE-TIES, greedy soup": {"merge": "ties", "dare": DARE, "soup": "greedy"},
}
CHAINS = {
    "average (FedAvg)": {},
    "average + DiLoCo": {"outer": DILOCO},
    "DARE-TIES, greedy soup + DiLoCo": {"merge": "ties", "dare": DARE, "soup": "greedy", "outer": DILOCO},
}


def batches_per_epoch(slices):
    """Training batches an epoch, at stride 1: a window starts at every frame but the last."""
    return math.ceil(sum(len(r["frames"]) - 1 for r in slices) / BATCH)


def step_matched_epochs(epochs):
    """Epochs on all the training slices that take as many optimizer steps as `epochs` on one shard."""
    anchor = load_anchor("v1")
    _, partitions = held_out(anchor, "voyager", "within_session_scan_holdout")
    shard = shard_records(partitions["train"], 1, SHARDS)
    return round(epochs * batches_per_epoch(shard) / batches_per_epoch(partitions["train"]))


def train(output, seed, epochs, shard=None, start=None, family="voyager"):
    argv = ["--anchor", "v1", "--manifest", str(MANIFEST), "--allow-within-session", "--variants", "real", "--family", family,
            "--train-stride", "1", "--epochs", str(epochs), "--seed", str(seed), "--output", str(output)]
    if shard:
        argv += ["--shard", f"{shard}/{SHARDS}"]
    if start:
        argv += ["--start", str(start)]
    with contextlib.redirect_stdout(io.StringIO()):
        run(parser().parse_args(argv))
    return output / "real.pt"


def merge(output, workers, rule, start=None):
    record = run_round(output, workers, start, rule)
    return record, (output / "merged.pt") if record["merged"]["decision"] == "accepted" else None


def outcome(record):
    """What a round did with its workers: the shards it merged, set aside and rejected, and the
    selection control's diagnostics for every model it scored."""
    by = lambda status: [c["shard"][0] for c in record["contributors"] if c["status"] == status]
    payloads = [c["payload"]["bytes"] for c in record["contributors"] if "payload" in c]
    scored = [record["start"], *record["contributors"], record["merged"]]
    return {"decision": record["merged"]["decision"], "validation_bits": record["merged"].get("validation_bits"),
            "merged": by("accepted"), "set_aside": by("set aside"), "rejected": by("rejected"),
            "selection_order": [m["control"] for m in scored if m.get("control")], **({"payload_bytes": payloads} if payloads else {})}


def selection_order(path):
    """The selection control's diagnostic (the off-source validation scan) for a checkpoint."""
    anchor = load_anchor("v1")
    records, _ = held_out(anchor, "voyager", "within_session_scan_holdout")
    tokens = [anchor.tokenizer.encode(r["frames"]) for r in controls(records, "voyager", "within_session_scan_holdout")["selection"]]
    return order_gain(model_from(anchor, load_checkpoint(path, anchor)["model"]), tokens)


def theta0_scores():
    anchor = load_anchor("v1")
    records, partitions = held_out(anchor, "voyager", "within_session_scan_holdout")
    encode = lambda rs: [anchor.tokenizer.encode(r["frames"]) for r in rs]
    model = anchor.model()
    control = controls(records, "voyager", "within_session_scan_holdout")
    return {"validation": evaluate(model, encode(partitions["validation"]))["bits_per_token"],
            "test": evaluate(model, encode(partitions["test"]))["bits_per_token"],
            "control": {"family": control["family"], "bits_per_token": evaluate(model, encode(control["reporting"]))["bits_per_token"]},
            "selection_order": order_gain(model, encode(control["selection"]))}


def one_seed(folder, seed, epochs, rounds):
    out = {}
    shards = [train(folder / f"shard{k}", seed, epochs, shard=k) for k in range(1, SHARDS + 1)]
    out["shard_workers"] = [score(p) for p in shards]
    best = min(range(SHARDS), key=lambda k: out["shard_workers"][k]["validation"])
    out["best_worker"] = {**out["shard_workers"][best], "shard": best + 1}
    out["one_shot"] = {}
    for i, (name, rule) in enumerate(ONE_SHOT.items()):
        record, merged = merge(folder / f"one-shot{i}", shards, rule)
        out["one_shot"][name] = {**outcome(record), **(score(merged) if merged else {})}
    full = train(folder / "full", seed, epochs)
    out["full_data"] = {**score(full), "selection_order": selection_order(full)}
    out["full_data_step_matched"] = score(train(folder / "full-steps", seed, step_matched_epochs(epochs)))
    # The positive control for the selection diagnostic: a worker trained on the off-source family.
    out["offsource_worker"] = {"selection_order": selection_order(train(folder / "offsource", seed, epochs, family="offsource"))}

    out["rounds"] = {}
    for i, (name, rule) in enumerate(CHAINS.items()):
        incumbent, history = None, []
        for r in range(1, rounds + 1):
            workers = [train(folder / f"chain{i}-r{r}-shard{k}", seed + r, epochs // rounds, shard=k, start=incumbent)
                       for k in range(1, SHARDS + 1)]
            record, merged = merge(folder / f"chain{i}-round{r}", workers, rule, start=incumbent)
            history.append({"round": r, **outcome(record)})
            incumbent = merged or incumbent
        out["rounds"][name] = {"history": history, **(score(incumbent) if incumbent else {})}
    return out


def summarize(results):
    def stats(values):
        return {"mean": statistics.fmean(values), "min": min(values), "max": max(values)} if values else None

    picks = [("shard workers", lambda s: s["shard_workers"]), ("best worker (by validation)", lambda s: [s["best_worker"]])]
    picks += [(f"one-shot: {name}", lambda s, name=name: [s["one_shot"][name]]) for name in ONE_SHOT]
    picks += [("one worker on all the data", lambda s: [s["full_data"]]),
              ("one worker on all the data, a shard worker's steps", lambda s: [s["full_data_step_matched"]])]
    picks += [(f"{results['rounds']} rounds: {name}", lambda s, name=name: [s["rounds"][name]]) for name in CHAINS]
    rows = {}
    for name, pick in picks:
        runs = [r for s in results["seeds"].values() for r in pick(s) if "test" in r]
        rows[name] = {key: stats([r[key] for r in runs]) for key in ("validation", "test")}
        rows[name]["control"] = stats([r["control"]["bits_per_token"] for r in runs])
        rows[name]["runs"] = len(runs)
    return rows


def selection_summary(results):
    """The selection control's diagnostics: over every model a round scored (incumbents, workers
    and merges), the full-data workers, and the off-source positive control."""
    seeds = list(results["seeds"].values())
    rounds = [d for s in seeds for o in [*s["one_shot"].values(), *(h for c in s["rounds"].values() for h in c["history"])]
              for d in o["selection_order"]]
    span = lambda ds, key: [min(d[key] for d in ds), max(d[key] for d in ds)]
    groups = {"round models": rounds, "full-data workers": [s["full_data"]["selection_order"] for s in seeds],
              "off-source workers": [s["offsource_worker"]["selection_order"] for s in seeds]}
    return {"theta0": results["theta0"]["selection_order"],
            **{name: {"n": len(ds), "bits": span(ds, "bits"), "order_gain_bits": span(ds, "order_gain_bits"),
                      "order_p": span(ds, "order_p")} for name, ds in groups.items()}}


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
    results = {"protocol": __doc__.split("\n\n", 1)[1].rsplit("\n\n", 1)[0], "epochs": args.epochs, "rounds": args.rounds,
               "step_matched_epochs": step_matched_epochs(args.epochs), "one_shot_rules": ONE_SHOT, "chain_rules": CHAINS,
               "seeds": {}}
    with training_runtime(1, "cpu"):
        results["theta0"] = theta0_scores()
    for seed in args.seeds:
        results["seeds"][str(seed)] = one_seed(args.output / f"seed{seed}", seed, args.epochs, args.rounds)
        print(f"seed {seed} done", flush=True)
    results["summary"] = summarize(results)
    results["selection_control"] = selection_summary(results)
    (args.output / "report.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(table(results))
    print(json.dumps(results["selection_control"], indent=1))


if __name__ == "__main__":
    main()
