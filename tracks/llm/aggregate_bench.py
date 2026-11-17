"""Aggregating rounds of volunteers' training (signal-lm-aggregation-bench/v0): what a chain of
merge rounds should do when its gate refuses a merge, and which round it should hand on.

The session benchmark (session_bench.py, BENCHMARK.md) runs two ten-round chains from each seed's
published base: the plain average, and the average with DiLoCo's outer step (lr 0.7, Nesterov
momentum 0.9). Reading the rounds it published shows two things its summary rows don't:

- **A refused merge deadlocks DiLoCo.** merge.py gates every merge against the incumbent, and a
  refused round leaves the incumbent as it was, its outer momentum included. The momentum is what
  made the step overshoot, so the next round's step overshoots too. On sessions-v1 every seed's
  DiLoCo chain was accepted once and then refused nine times: its row there scores one outer step,
  not ten rounds.
- **A chain reports its last round, not its best.** The gate accepts a merge up to 0.02 bits
  worse than the incumbent on validation, so a chain can drift, and the workers it merges each
  keep their best epoch while the chain itself keeps whatever came last.

This benchmark reruns those chains from the same published bases, with the same training and
merge.py's unchanged rules and gate, and adds what answers each:

- `restart`: when the gate refuses a merge that carried outer momentum, the momentum is cleared
  and the same workers are merged again (no retraining). It's the adaptive restart of accelerated
  gradient methods (O'Donoghue and Candès, 2015): drop the momentum when the objective rises. If
  the merge without momentum is refused too, the incumbent continues with its momentum cleared.
- `soup`: merge.py's greedy soup in every round, so a round keeps only the workers whose deltas
  improve the merge on validation. The session benchmark only tries it one-shot.
- the **best round**: every chain is scored twice, at its last round and at the accepted round
  with the lowest validation bits (the base counts; ties go to the earlier). It costs nothing
  more: the gate already scored every round on validation.

Arms, per seed, from the published base, ten rounds each (round r's workers train up to 100
epochs with seed + r, as the session benchmark's chains do):

- `average` and `average + DiLoCo`: the session benchmark's two chains again, as references.
  They must come out with the weights it published scores for, or the run stops.
- `average + DiLoCo, restarting`
- `greedy soup`
- `greedy soup + DiLoCo, restarting`

Everything is chosen on validation. Test, the worst test session, the blind-window control and the
order gain are session_bench.py's measures, scored after. The arms were chosen on v0, whose test
scores have been seen, and then run once on v1 (AGGREGATION_RESULTS.md says what had been seen of
v1 by then). This module changes none of the code the WSP protocol pins: merge.py, the scoring and
the published cohorts are used as they are.

    python -m tracks.llm.aggregate_bench run --output runs/aggregation-v0
    python -m tracks.llm.aggregate_bench run --benchmark v1 --output runs/aggregation-v1 --publish
    python -m tracks.llm.aggregate_bench verify [--benchmark v1]   # the published report against the session benchmark's
"""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from . import session_bench as sessions
from .anchor import load_anchor, weights_digest
from .data import digest
from .merge import RULE, check_rule, load_checkpoint
from .runtime import require_new_output

BENCHMARK = "signal-lm-aggregation-bench/v0"
FOLDERS = {"v0": Path(__file__).resolve().parent / "benchmarks" / "aggregation-v0",
           "v1": Path(__file__).resolve().parent / "benchmarks" / "aggregation-v1"}
DILOCO = sessions.DILOCO
ARMS = {  # name: merge.py's rule for every round, and whether a refused momentum merge restarts
    "average": {"rule": {}, "restart": False},
    "average + DiLoCo": {"rule": {"outer": DILOCO}, "restart": False},
    "average + DiLoCo, restarting": {"rule": {"outer": DILOCO}, "restart": True},
    "greedy soup": {"rule": {"soup": "greedy"}, "restart": False},
    "greedy soup + DiLoCo, restarting": {"rule": {"soup": "greedy", "outer": DILOCO}, "restart": True},
}
REFERENCES = {"average": "average (FedAvg)", "average + DiLoCo": "average + DiLoCo"}  # arm: the session benchmark's chain
SCORES = ("validation", "test", "test_worst_session", "control", "test_order_gain_bits")
SCORE_TOLERANCE = 1e-6  # bits, between environments (session_bench.py's)


def without_momentum(path, target):
    """A merged checkpoint again with its outer momentum cleared: the same weights and chain, so
    workers trained from the original continue from it, and its next outer step starts afresh."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["outer_momentum"] = None
    checkpoint["merge"] = {**checkpoint["merge"], "momentum_sha256": None}
    torch.save(checkpoint, target)
    return target


def carries_momentum(path):
    return torch.load(path, map_location="cpu", weights_only=True).get("outer_momentum") is not None


def best_round(history):
    """The accepted round with the lowest validation bits; the start (the first entry) counts, and
    ties go to the earlier. history: the start, then each round, as {"accepted", "validation_bits"}."""
    accepted = [h for h in history if h["accepted"]]
    return min(accepted, key=lambda h: h["validation_bits"])


def chain(folder, start, seed, rule, restart, rounds, epochs, manifest, ctx, start_validation):
    """A chain of merge rounds from `start` (a merged checkpoint), as session_bench.py runs its
    chains: each round's workers train from the incumbent with seed + r and are merged by `rule`;
    a refused round leaves the incumbent. With `restart`, a refused merge whose incumbent carried
    outer momentum is merged again from the incumbent without it, and if that's refused too the
    incumbent continues without its momentum. Returns the arm's record: its rounds, costs, and its
    scores at the last round and at the best round by validation."""
    anchor, delta = ctx.anchor, sessions.bytes_per_delta(ctx.anchor)
    incumbent, seconds, contributions = start, 0.0, 0
    history = [{"round": None, "accepted": True, "validation_bits": start_validation, "checkpoint": start}]
    restarts = 0
    for r in range(1, rounds + 1):
        workers = [sessions.train(folder / f"r{r}-worker{k}", seed + r, epochs // rounds, manifest, shard=k, start=incumbent, base=folder)
                   for k in sessions.WORKER_SHARDS]
        record, merged, merge_seconds = sessions.merge(folder / f"round{r}", [w["checkpoint"] for w in workers], rule, incumbent,
                                                       manifest, anchor)
        entry = sessions.outcome(record, folder)
        if merged is None and restart and carries_momentum(incumbent):
            # The momentum made the step the gate refused: drop it and merge the same workers again.
            incumbent = without_momentum(incumbent, folder / f"round{r}-restart-start.pt")
            record, merged, again_seconds = sessions.merge(folder / f"round{r}-restart", [w["checkpoint"] for w in workers], rule,
                                                           incumbent, manifest, anchor)
            merge_seconds += again_seconds
            restarts += 1
            entry = {**sessions.outcome(record, folder), "restarted_after": {"validation_bits": entry["validation_bits"],
                                                                             "reason": entry["reason"]}}
        seconds += sum(w["seconds"] for w in workers) + merge_seconds
        contributions += len(workers)
        incumbent = merged or incumbent
        history.append({"round": r, "accepted": merged is not None, "validation_bits": entry["validation_bits"],
                        "checkpoint": merged, "record": entry})
    best = best_round(history)
    last = sessions.assess(ctx, incumbent)
    return {"rule": {**RULE, **rule}, "restart": restart, "rounds": [h["record"] for h in history[1:]],
            "accepted_rounds": sum(h["accepted"] for h in history[1:]), "restarts": restarts,
            "training_seconds": seconds, "contributions": contributions, "bytes_sent": contributions * delta,
            "last": last,
            "best_round": {"round": best["round"], **(last if best["checkpoint"] == incumbent else sessions.assess(ctx, best["checkpoint"]))}}


def one_seed(folder, bench, seed, arms, rounds, epochs, ctx, session_report, base):
    """Every arm's chain from one seed's published base (a .pt, as `materialize` writes it)."""
    recorded = session_report["seeds"][str(seed)]
    base_scores = sessions.assess(ctx, base)
    if weights_digest(load_checkpoint(base, ctx.anchor)["model"]) != recorded["base"]["weights_sha256"]:
        raise ValueError(f"seed {seed}: the published base isn't the one the session benchmark scored")
    out = {"base": {k: base_scores[k] for k in (*SCORES, "weights_sha256")}, "arms": {}}
    for i, name in enumerate(arms):
        arm = ARMS[name]
        check_rule(arm["rule"])
        out["arms"][name] = chain(folder / f"arm{i}", base, seed, arm["rule"], arm["restart"], rounds, epochs, bench.manifest, ctx,
                                  base_scores["validation"])
    return out


def reference_problem(seed_report, session_seed, same_settings):
    """Why a seed's reference chains aren't the session benchmark's (None if they are, or if the
    run isn't on its rounds and epochs): the same decisions round by round, and the same final
    weights or, in an environment that trains to other weights, the same scores."""
    if not same_settings:
        return None
    for name, theirs in REFERENCES.items():
        if name not in seed_report["arms"]:
            continue
        ours, published = seed_report["arms"][name], session_seed["chains"][theirs]
        if [r["decision"] for r in ours["rounds"]] != [h["decision"] for h in published["history"]]:
            return f"the {name} chain's rounds weren't accepted and refused as the session benchmark's were"
        for key in ("validation", "test"):
            if ours["last"]["weights_sha256"] != published["weights_sha256"] and abs(ours["last"][key] - published[key]) > SCORE_TOLERANCE:
                return f"the {name} chain scores {ours['last'][key]:.6f} {key}; the session benchmark published {published[key]:.6f}"
    return None


def summarize(seeds):
    """Each arm's mean, lowest and highest score over seeds, at its last round and its best round;
    how many of its rounds were accepted; and its test sessions against the plain average's last
    round (better / worse, over sessions × seeds)."""
    def stats(values):
        return {"mean": statistics.fmean(values), "min": min(values), "max": max(values)}

    names = list(next(iter(seeds.values()))["arms"])
    rows = {}
    for name in names:
        runs = [s["arms"][name] for s in seeds.values()]
        row = {"accepted_rounds": [r["accepted_rounds"] for r in runs], "restarts": [r["restarts"] for r in runs],
               "best_rounds": [r["best_round"]["round"] for r in runs]}
        for pick in ("last", "best_round"):
            row[pick] = {key: stats([r[pick][key] for r in runs]) for key in SCORES}
            if "average" in names:
                pairs = [(a, b) for s in seeds.values()
                         for a, b in zip(s["arms"][name][pick]["test_sessions"], s["arms"]["average"]["last"]["test_sessions"])]
                row[pick]["sessions_vs_average"] = {"better": sum(a < b for a, b in pairs), "worse": sum(a > b for a, b in pairs)}
        rows[name] = row
    return rows


def table(report):
    fmt = lambda s: f"{s['mean']:.3f}"
    lines = ["| Arm | Rounds accepted (per seed) | Validation | Test | Worst test session | Best round: validation | test | rounds picked |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in report["summary"].items():
        last, best = row["last"], row["best_round"]
        lines.append(f"| {name} | {', '.join(map(str, row['accepted_rounds']))} | {fmt(last['validation'])} | {fmt(last['test'])} | "
                     f"{fmt(last['test_worst_session'])} | {fmt(best['validation'])} | {fmt(best['test'])} | "
                     f"{', '.join('base' if r is None else str(r) for r in row['best_rounds'])} |")
    return "\n".join(lines)


def session_report_of(bench):
    return json.loads((bench.folder / "report.json").read_text())


def run(args):
    bench = sessions.benchmark(args.benchmark)
    arms = list(dict.fromkeys(args.arms))
    if args.publish and (tuple(args.seeds) != sessions.SEEDS or args.epochs != sessions.EPOCHS or args.rounds != sessions.ROUNDS
                         or arms != list(ARMS)):
        raise ValueError("--publish writes the benchmark's report: every arm, on its seeds, epochs and rounds only")
    if args.epochs % args.rounds:
        raise ValueError("--epochs must be a multiple of --rounds")
    split = sessions.check_split(bench=bench)
    session_report = session_report_of(bench)
    require_new_output(args.output)
    args.output.mkdir(parents=True)
    anchor = load_anchor(sessions.ANCHOR)
    ctx = sessions.Context(anchor, bench.manifest)
    flag = "" if bench.name == "v0" else f" --benchmark {bench.name}"
    report = {"format": "eiathome-signal-lm-aggregation-report", "version": 1, "benchmark": BENCHMARK, "sessions_benchmark": bench.id,
              "split_sha256": digest(sessions.split_text(split).encode()), "dataset_sha256": ctx.dataset_sha256,
              "anchor": anchor.identity(), "code_sha256": sessions.code_sha256(),
              "aggregate_code_sha256": digest(Path(__file__).read_bytes()), "environment": sessions.environment(),
              "settings": {"seeds": list(args.seeds), "epochs": args.epochs, "patience": sessions.PATIENCE, "rounds": args.rounds,
                           "arms": {name: ARMS[name] for name in arms}, "tolerance_bits": RULE["tolerance_bits"]},
              "commands": {"run": f"python -m tracks.llm.aggregate_bench run --output {args.output}{flag}",
                           "verify": f"python -m tracks.llm.aggregate_bench verify{flag}"},
              "seeds": {}}
    published = session_report["settings"]
    same_settings = (args.rounds, args.epochs) == (published["rounds"], published["epochs"])
    cohort = sessions.materialize(args.output / "cohort", bench=bench, seeds=tuple(published["seeds"]))
    for seed in args.seeds:
        started = time.perf_counter()
        seed_report = one_seed(args.output / f"seed{seed}", bench, seed, arms, args.rounds, args.epochs, ctx, session_report,
                               cohort[(seed, "base")])
        seed_report["reference_problem"] = reference_problem(seed_report, session_report["seeds"][str(seed)], same_settings)
        report["seeds"][str(seed)] = seed_report
        print(f"seed {seed} done in {time.perf_counter() - started:.0f} s", flush=True)
    report["summary"] = summarize(report["seeds"])
    (args.output / "report.json").write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print(table(report))
    problems = {seed: s["reference_problem"] for seed, s in report["seeds"].items() if s["reference_problem"]}
    if problems:
        print("\n".join(f"seed {seed}: {problem}" for seed, problem in problems.items()))
    if args.publish:
        if problems:
            raise ValueError("Not published: the reference chains didn't reproduce the session benchmark's in this environment")
        folder = FOLDERS[bench.name]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "report.json").write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    return report


def verify(bench=None, folder=None):
    """The published report, checked without retraining: it's on this benchmark's pinned split and
    the session benchmark's published bases; its reference chains are the session benchmark's
    chains (decisions and final weights); every restart followed a refused momentum merge; each
    best round is the accepted round with the lowest validation bits; and the summary is what its
    seeds give."""
    bench = sessions.benchmark(bench)
    folder = FOLDERS[bench.name] if folder is None else folder
    report = json.loads((folder / "report.json").read_text())
    if report.get("format") != "eiathome-signal-lm-aggregation-report" or report.get("sessions_benchmark") != bench.id:
        raise ValueError(f"{folder}: not this benchmark's aggregation report")
    if report["split_sha256"] != bench.split_sha256:
        raise ValueError(f"{folder}: the report is on another split")
    settings = report["settings"]
    if (settings["seeds"], settings["epochs"], settings["rounds"]) != (list(sessions.SEEDS), sessions.EPOCHS, sessions.ROUNDS) \
            or settings["arms"] != ARMS:
        raise ValueError(f"{folder}: the report isn't on the benchmark's seeds, epochs, rounds and arms")
    session_report = session_report_of(bench)
    for seed, seed_report in report["seeds"].items():
        where = f"seed {seed}"
        recorded = session_report["seeds"][seed]
        if seed_report["base"]["weights_sha256"] != recorded["base"]["weights_sha256"]:
            raise ValueError(f"{where}: the chains didn't start from the published base")
        for name, theirs in REFERENCES.items():
            ours, published = seed_report["arms"][name], recorded["chains"][theirs]
            if ours["last"]["weights_sha256"] != published["weights_sha256"] \
                    or [r["decision"] for r in ours["rounds"]] != [h["decision"] for h in published["history"]]:
                raise ValueError(f"{where}: the {name} chain isn't the session benchmark's")
        for name, arm in seed_report["arms"].items():
            rounds = arm["rounds"]
            if len(rounds) != settings["rounds"] or arm["accepted_rounds"] != sum(r["decision"] == "accepted" for r in rounds):
                raise ValueError(f"{where}: {name} doesn't record its {settings['rounds']} rounds")
            restarted = [r for r in rounds if "restarted_after" in r]
            if len(restarted) != arm["restarts"] or (restarted and not ARMS[name]["restart"]):
                raise ValueError(f"{where}: {name} records restarts it didn't make or may not make")
            history = [{"round": None, "accepted": True, "validation_bits": seed_report["base"]["validation"]}]
            history += [{"round": i, "accepted": r["decision"] == "accepted", "validation_bits": r["validation_bits"]}
                        for i, r in enumerate(rounds, 1)]
            best = best_round(history)
            if arm["best_round"]["round"] != best["round"] or abs(arm["best_round"]["validation"] - best["validation_bits"]) > SCORE_TOLERANCE:
                raise ValueError(f"{where}: {name}'s best round isn't its accepted round with the lowest validation bits")
            final = [h for h in history if h["accepted"]][-1]
            if abs(arm["last"]["validation"] - final["validation_bits"]) > SCORE_TOLERANCE:
                raise ValueError(f"{where}: {name}'s last score isn't its last accepted round's")
    if json.dumps(summarize(report["seeds"]), sort_keys=True) != json.dumps(report["summary"], sort_keys=True):
        raise ValueError(f"{folder}: the summary isn't what the seeds give")
    return sorted(report["seeds"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run the chains from the published bases")
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    r.add_argument("--seeds", type=int, nargs="+", default=list(sessions.SEEDS))
    r.add_argument("--epochs", type=int, default=sessions.EPOCHS)
    r.add_argument("--rounds", type=int, default=sessions.ROUNDS)
    r.add_argument("--publish", action="store_true", help="write the published report (the benchmark's settings only)")
    v = sub.add_parser("verify", help="check the published report against the session benchmark's")
    for command in (r, v):
        command.add_argument("--benchmark", choices=("v0", "v1"), default="v0", help="the session benchmark version (default v0)")
    args = p.parse_args(argv)
    try:
        if args.command == "run":
            run(args)
        else:
            seeds = verify(args.benchmark)
            print(f"verified: the aggregation report on {sessions.benchmark(args.benchmark).id}, seeds {', '.join(seeds)}")
    except (ValueError, OSError, RuntimeError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
