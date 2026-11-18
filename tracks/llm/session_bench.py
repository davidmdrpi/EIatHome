"""The session benchmark (signal-lm-sessions-bench/v0): do volunteers' workers, each trained on
observing sessions no other worker saw, merge into a model that predicts new sessions better?

Every earlier merge benchmark (merge_bench.py, moe_bench.py, route_bench.py) trains and tests on
one 2020 Voyager session; they stay as smoke tests. This one is frozen on the sessions corpus
(tracks/llm/data/sessions, made by sessions_corpus.py): on-target scans of nearby stars from many
Green Bank observing sessions, 16-frame windows of the fine-resolution product. BENCHMARK.md has
the results; split.json and checkpoints/ are what it publishes.

Fixed before any run:

- Data: the corpus's on-target hits (family stars), split by whole observing session,
  chronologically, 60/20/20 (data.split_records): every validation and test session was observed
  after every training session. The blind windows of the same sessions (family blind-window)
  are the control: its validation part is the selection diagnostic, its test part the reporting
  control. split.json lists every session and slice by content hash; its SHA-256 is pinned below.
- Anchor v4: the peak tokenizer (fitted on nothing) and θ₀, so every run's change is in the
  same coordinates and no session shaped the tokens.
- The cohort: the training sessions in 5 session shards (merge.shard_records, by session).
  Shard 1 trains the base, from θ₀, recorded as round 1 of a chain. Shards 2 to 5 are four
  volunteers: each worker trains from the base on its own sessions. No two of them, and no
  worker and the base, share a session.
- Training: to convergence. Up to 1,000 epochs, stopping once validation hasn't improved for 50,
  and keeping the best epoch (train.py, lr 3e-4, batch 32), the real stream only, seeds 42, 43
  and 44. A worker's seed is its seed's; in the chains, round r's workers train up to
  epochs / rounds = 100 epochs with seed + r and the same patience. (The first run used 50
  epochs and train.py's patience of 8: every run's best epoch was its last, so it compared
  models still training. BENCHMARK.md records that run.)
- Arms, per seed: the base; each worker; the best worker by validation; one-shot merges of the
  four workers from the base (the plain average, a greedy soup of it, and TIES keeping 20%, the
  TIES paper's default); ten-round chains from the base (the plain average, and the average with
  DiLoCo's outer step, lr 0.7 and Nesterov momentum 0.9, the DiLoCo paper's); and one model on all
  the training sessions, from θ₀ and from the base, trained the same way.
- Every round gates its workers and merge against the incumbent on validation (merge.py, a
  tolerance of 0.02 bits). Nothing is chosen on test.
- Reported, on the test sessions only after everything is chosen: bits per token (the mean over
  test targets), the worst test session's, the control's, and the order gain (how many fewer
  bits the model needs for the test tokens in their real order than shuffled within each slice,
  merge.order_gain). Also: each arm's training seconds, contributions and bytes sent (a float32
  delta per contribution), and evaluation seconds.

The base and the four workers of each seed are published as raw float32 weights with a header
(checkpoints/seed*/), for merge methods to start from (the next milestone's weight-space
programming experiment). `materialize` rebuilds them as checkpoints merge.py and sample.py load,
with their splits taken from the manifest, and `verify` checks them, the split and the one-shot
merges against report.json.

    python -m tracks.llm.session_bench split --check     # recompute split.json and check its pin
    python -m tracks.llm.session_bench run --output runs/sessions-bench
    python -m tracks.llm.session_bench run --output runs/sessions-bench --publish   # rewrite the published files
    python -m tracks.llm.session_bench verify            # the published checkpoints, split and merges
    python -m tracks.llm.session_bench materialize --output runs/sessions-v0-checkpoints
"""

import argparse
from collections import OrderedDict
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time
from typing import NamedTuple

import numpy as np
import torch

from .anchor import ROOT, canonical_layout, difference, load_anchor, weights_digest
from .data import dataset_digest, digest, load_slices, session_groups, split_records
from .merge import RULE, controls, held_out, lineage, load_checkpoint, merge_round, merged_checkpoint, model_from, order_gain, \
    shard_records
from .model import evaluate
from .runtime import atomic_output, require_new_output, training_runtime
from .train import baseline_scores, parser as train_parser, run as train_run

BENCHMARK = "signal-lm-sessions-bench/v0"
ANCHOR = "v4"
MANIFEST = ROOT / "tracks/llm/data/sessions/index.json"
FOLDER = Path(__file__).resolve().parent / "benchmarks" / "sessions-v0"
FAMILY = "stars"
EVALUATION = "independent_sessions"
SHARDS = 5
UNIT = "session"
BASE_SHARD = 1
WORKER_SHARDS = (2, 3, 4, 5)
SEEDS = (42, 43, 44)
EPOCHS = 1000  # a cap: every run stops once validation hasn't improved for PATIENCE epochs
PATIENCE = 50
ROUNDS = 10
DILOCO = {"lr": 0.7, "momentum": 0.9}
ONE_SHOT = {  # name: rule (merge.py's)
    "average (FedAvg)": {},
    "average, greedy soup": {"soup": "greedy"},
    "TIES, keep 20%": {"merge": "ties", "ties_keep": 0.2},
}
CHAINS = {
    "average (FedAvg)": {},
    "average + DiLoCo": {"outer": DILOCO},
}
# split.json's SHA-256: the frozen split. A different split is a new benchmark version.
SPLIT_SHA256 = "4023ba4f1795914b34aced6b093da765a2680554d8bb74f62dcd4fb32decc418"
# v1: the same protocol on a second draw of sessions (sessions_corpus.py --exclude the first's),
# for scoring merge methods frozen on v0 (wsp_bench.py) on sessions nobody has looked at. It was
# cut only once the methods it scores were frozen (WSP_RESULTS.md): drawn after wsp_bench.py's
# PROTOCOL_SHA256 was pinned on main, and split and pinned here before anything trained on it.
BENCHMARK_V1 = "signal-lm-sessions-bench/v1"
MANIFEST_V1 = ROOT / "tracks/llm/data/sessions-v1/index.json"
FOLDER_V1 = Path(__file__).resolve().parent / "benchmarks" / "sessions-v1"
SPLIT_SHA256_V1 = "df8ceba04c06ba04843d043e6ed699f168587dd57a9d9239bb7af86f4ccd38c5"


class Benchmark(NamedTuple):
    name: str
    id: str
    manifest: Path
    folder: Path
    split_sha256: str | None


def benchmark(bench=None):
    """A benchmark version by name ("v0", the default, or "v1"), read when called, so the module's
    settings are the ones in force."""
    if isinstance(bench, Benchmark):
        return bench
    if bench in (None, "v0"):
        return Benchmark("v0", BENCHMARK, MANIFEST, FOLDER, SPLIT_SHA256)
    if bench == "v1":
        return Benchmark("v1", BENCHMARK_V1, MANIFEST_V1, FOLDER_V1, SPLIT_SHA256_V1)
    raise ValueError(f"Unknown session benchmark {bench!r}: v0 or v1")


CODE_FILES = ("data.py", "model.py", "train.py", "runtime.py", "anchor.py", "merge.py", "session_bench.py")


# ---------- the split ----------

def sessions_of(records):
    """Observing sessions as split.json lists them: start, targets and slices, by content."""
    return [{"start_mjd": round(min(r["start"] for r in g), 6),
             "targets": sorted({r["entry"].get("source", {}).get("sky", {}).get("target") or r["entry"]["family"] for r in g}),
             "slices": [{"id": r["entry"]["id"], "sha256": r["sha256"]} for r in sorted(g, key=lambda r: r["sha256"])]}
            for g in session_groups(records)]


def build_split(manifest=None, bench=None):
    """split.json's content, from the manifest: the family's sessions in each split, the control's,
    and the cohort's session shards."""
    bench = benchmark(bench)
    manifest = bench.manifest if manifest is None else manifest
    records = load_slices(manifest)
    partitions, independent = split_records(records, FAMILY)
    if not independent:
        raise ValueError("The session benchmark needs at least 3 independent sessions")
    control = controls(records, FAMILY, EVALUATION)
    if control is None:
        raise ValueError(f"The manifest has no control family for {FAMILY}")
    targets = lambda rs: sum(len(r["frames"]) - 1 for r in rs)
    split = {
        "format": "eiathome-signal-lm-benchmark-split", "version": 1, "benchmark": bench.id,
        "manifest": os.path.relpath(Path(manifest).resolve(), ROOT), "dataset_sha256": dataset_digest(records),
        "anchor": ANCHOR, "family": FAMILY, "evaluation": EVALUATION, "split_unit": "session",
        "splits": {name: {"sessions": sessions_of(rs), "slices": len(rs), "targets": targets(rs)} for name, rs in partitions.items()},
        "control": {"family": control["family"],
                    "selection": {"sessions": sessions_of(control["selection"]), "targets": targets(control["selection"])},
                    "reporting": {"sessions": sessions_of(control["reporting"]), "targets": targets(control["reporting"])}},
        "cohort": {"shards": SHARDS, "unit": UNIT, "base": BASE_SHARD, "workers": list(WORKER_SHARDS),
                   "shard_sessions": {str(i): [s["start_mjd"] for s in sessions_of(shard_records(partitions["train"], i, SHARDS, UNIT))]
                                      for i in range(1, SHARDS + 1)}},
    }
    return split


def split_text(split):
    return json.dumps(split, indent=1, sort_keys=True) + "\n"


def check_split(manifest=None, folder=None, bench=None):
    """The committed split.json: pinned, and what the manifest splits into today."""
    bench = benchmark(bench)
    if bench.split_sha256 is None:
        raise ValueError(f"{bench.id} hasn't been cut: its sessions wait until the methods it scores are frozen")
    folder = bench.folder if folder is None else folder
    text = (folder / "split.json").read_text()
    if digest(text.encode()) != bench.split_sha256:
        raise ValueError("split.json doesn't match its pinned SHA-256: a different split is a new benchmark version")
    if split_text(build_split(manifest, bench)) != text:
        raise ValueError("The manifest no longer splits as split.json records")
    return json.loads(text)


# ---------- runs ----------

def environment():
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
            "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "threads": 1, "device": "cpu", "deterministic_algorithms": True}


def code_sha256():
    here = Path(__file__).resolve().parent
    return digest(b"".join((here / name).read_bytes() for name in CODE_FILES))


def relative(path, base):
    return os.path.relpath(path, base)


def train(output, seed, epochs, manifest, shard=None, start=None, base=None, patience=PATIENCE):
    """One train.py run (the real stream only); its checkpoint and what it cost."""
    argv = ["--anchor", ANCHOR, "--manifest", str(manifest), "--family", FAMILY, "--variants", "real",
            "--epochs", str(epochs), "--patience", str(patience), "--seed", str(seed), "--output", str(output)]
    if shard:
        argv += ["--shard", f"{shard}/{SHARDS}", "--shard-unit", UNIT]
    if start:
        argv += ["--start", str(start)]
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        report = train_run(train_parser().parse_args(argv))
    real = report["variants"]["real"]
    shown = [relative(a, base) if base and i and argv[i - 1] in ("--output", "--start") else a for i, a in enumerate(argv)]
    shown = [os.path.relpath(Path(a).resolve(), ROOT) if i and argv[i - 1] == "--manifest" else a for i, a in enumerate(shown)]
    return {"checkpoint": output / "real.pt", "command": "python -m tracks.llm.train " + " ".join(shown),
            "seconds": time.perf_counter() - started, "best_epoch": real["best_epoch"], "epochs_run": len(real["history"]),
            "training_targets": report["merge"]["training_targets"]}


def merge(output, workers, rule, start, manifest, anchor):
    """A merge.py round into a new folder (round.json, and merged.pt if it passed), on `manifest`."""
    require_new_output(output)
    started = time.perf_counter()
    with training_runtime(1, "cpu"), atomic_output(output) as folder:
        record, merged = merge_round(anchor, [str(w) for w in workers], None if start is None else str(start), rule, manifest)
        (folder / "round.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        if merged is not None:
            chain = lineage(load_checkpoint(workers[0], anchor))
            torch.save(merged_checkpoint(anchor, record, merged, chain), folder / "merged.pt")
    return record, (output / "merged.pt" if merged is not None else None), time.perf_counter() - started


class Context:
    """The held-out tokens every arm is scored on, encoded once."""

    def __init__(self, anchor, manifest):
        self.anchor = anchor
        records = load_slices(manifest)
        self.dataset_sha256 = dataset_digest(records)
        records, partitions = held_out(anchor, FAMILY, EVALUATION, manifest, self.dataset_sha256)
        encode = lambda rs: [anchor.tokenizer.encode(r["frames"]) for r in rs]
        self.train = encode(partitions["train"])
        self.validation = encode(partitions["validation"])
        self.test = encode(partitions["test"])
        self.test_sessions = [encode(g) for g in session_groups(partitions["test"])]
        # Per validation session too, and the records, for merge methods that look (wsp_bench.py).
        self.validation_sessions = [encode(g) for g in session_groups(partitions["validation"])]
        self.partitions = partitions
        control = controls(records, FAMILY, EVALUATION)
        self.control = encode(control["reporting"])


def assess(ctx, path):
    """An arm's scores: validation (what chose it) and, for reporting only, test."""
    checkpoint = load_checkpoint(path, ctx.anchor)
    model = model_from(ctx.anchor, checkpoint["model"])
    started = time.perf_counter()
    with training_runtime(1, "cpu"):
        sessions = [evaluate(model, s)["bits_per_token"] for s in ctx.test_sessions]
        order = order_gain(model, ctx.test)
        out = {"validation": evaluate(model, ctx.validation)["bits_per_token"], "test": evaluate(model, ctx.test)["bits_per_token"],
               "test_worst_session": max(sessions), "test_sessions": sessions,
               "control": evaluate(model, ctx.control)["bits_per_token"],
               "test_order_gain_bits": order["order_gain_bits"], "test_order_p": order["order_p"]}
    out.update(weights_sha256=weights_digest(checkpoint["model"]), eval_seconds=time.perf_counter() - started)
    return out


def baselines(ctx):
    """Token statistics fitted on every training session (train.py's baselines, add-one smoothed):
    the unigram and the first-order Markov chain, scored on the same held-out sets as every arm."""
    k = ctx.anchor.spec["model"]["vocab_size"]
    out = {}
    for name, key in (("unigram", "unigram_bits"), ("Markov-1", "markov1_bits")):
        score = lambda tokens: baseline_scores(ctx.train, tokens, k)[key]
        sessions = [score(s) for s in ctx.test_sessions]
        out[name] = {"validation": score(ctx.validation), "test": score(ctx.test), "test_worst_session": max(sessions),
                     "test_sessions": sessions, "control": score(ctx.control)}
    return out


def bytes_per_delta(anchor):
    return canonical_layout(anchor.spec["model"])[1] * 4  # a float32 delta


def outcome(record, base):
    """What a round did with its contributors, with paths relative to the run folder."""
    return {"round": record["round"], "decision": record["merged"]["decision"], "reason": record["merged"].get("reason"),
            "validation_bits": record["merged"].get("validation_bits"), "weights_sha256": record["merged"].get("weights_sha256"),
            "contributors": [{"file": relative(c["file"], base), "shard": c["shard"], "status": c["status"], "reason": c["reason"],
                              "weight": c["weight"], "validation_bits": c["validation_bits"]} for c in record["contributors"]]}


def one_seed(folder, seed, epochs, rounds, manifest, ctx):
    anchor = ctx.anchor
    delta = bytes_per_delta(anchor)
    out = {}
    # The base: shard 1 from θ₀, merged alone as round 1, so the workers' round can start from it.
    base_run = train(folder / "base-worker", seed, epochs, manifest, shard=BASE_SHARD, base=folder)
    base_record, base, merge_seconds = merge(folder / "base", [base_run["checkpoint"]], {}, None, manifest, anchor)
    if base is None:
        raise RuntimeError(f"seed {seed}: the base round was rejected: {base_record['merged'].get('reason')}")
    out["base"] = {**assess(ctx, base), "shard": BASE_SHARD, "training": {k: v for k, v in base_run.items() if k != "checkpoint"},
                   "round": outcome(base_record, folder), "merge_seconds": merge_seconds}

    runs = {k: train(folder / f"worker{k}", seed, epochs, manifest, shard=k, start=base, base=folder) for k in WORKER_SHARDS}
    out["workers"] = [{**assess(ctx, run["checkpoint"]), "shard": k, "training": {kk: v for kk, v in run.items() if kk != "checkpoint"}}
                      for k, run in runs.items()]
    best = min(out["workers"], key=lambda w: w["validation"])
    out["best_worker"] = {**best}
    worker_seconds = sum(w["training"]["seconds"] for w in out["workers"])

    out["one_shot"] = {}
    for i, (name, rule) in enumerate(ONE_SHOT.items()):
        record, merged, seconds = merge(folder / f"one-shot{i}", [r["checkpoint"] for r in runs.values()], rule, base, manifest, anchor)
        out["one_shot"][name] = {"rule": record["rule"], "round": outcome(record, folder), "merge_seconds": seconds,
                                 "training_seconds": worker_seconds, "contributions": len(runs), "bytes_sent": len(runs) * delta,
                                 **(assess(ctx, merged) if merged else {})}

    out["chains"] = {}
    for i, (name, rule) in enumerate(CHAINS.items()):
        incumbent, history, seconds, contributions = base, [], 0.0, 0
        for r in range(1, rounds + 1):
            workers = [train(folder / f"chain{i}-r{r}-worker{k}", seed + r, epochs // rounds, manifest, shard=k, start=incumbent, base=folder)
                       for k in WORKER_SHARDS]
            record, merged, merge_seconds = merge(folder / f"chain{i}-round{r}", [w["checkpoint"] for w in workers], rule, incumbent,
                                                  manifest, anchor)
            seconds += sum(w["seconds"] for w in workers) + merge_seconds
            contributions += len(workers)
            history.append(outcome(record, folder))
            incumbent = merged or incumbent
        out["chains"][name] = {"rule": {**RULE, **rule}, "history": history, "training_seconds": seconds,
                               "contributions": contributions, "bytes_sent": contributions * delta, **assess(ctx, incumbent)}

    out["full_data"] = {}
    for name, start in (("from θ₀", None), ("from the base", base)):
        run = train(folder / ("full-theta0" if start is None else "full-base"), seed, epochs, manifest, start=start, base=folder)
        out["full_data"][name] = {**assess(ctx, run["checkpoint"]), "training": {k: v for k, v in run.items() if k != "checkpoint"}}
    return out


def cohort_paths(output, seed):
    """A run's base and workers for one seed: what --publish publishes."""
    folder = Path(output) / f"seed{seed}"
    return folder / "base" / "merged.pt", {k: folder / f"worker{k}" / "real.pt" for k in WORKER_SHARDS}


ARMS = [("base (shard 1 only)", lambda s: [s["base"]]),
        ("workers (one volunteer each)", lambda s: s["workers"]),
        ("best worker (by validation)", lambda s: [s["best_worker"]])]
ARMS += [(f"one-shot: {name}", lambda s, name=name: [s["one_shot"][name]]) for name in ONE_SHOT]
ARMS += [(f"rounds: {name}", lambda s, name=name: [s["chains"][name]]) for name in CHAINS]
ARMS += [(f"all training sessions, {name}", lambda s, name=name: [s["full_data"][name]]) for name in ("from θ₀", "from the base")]


def summarize(seeds, arms=ARMS):
    def stats(values):
        return {"mean": statistics.fmean(values), "min": min(values), "max": max(values)} if values else None

    rows = {}
    for name, pick in arms:
        runs = [r for s in seeds.values() for r in pick(s) if "test" in r]
        rows[name] = {key: stats([r[key] for r in runs]) for key in
                      ("validation", "test", "test_worst_session", "control", "test_order_gain_bits")}
        rows[name]["runs"] = len(runs)
    return rows


def table(report):
    fmt = lambda s: "–" if s is None else f"{s['mean']:.3f} ({s['min']:.3f}–{s['max']:.3f})"
    lines = ["| Arm | Runs | Validation | Test | Worst test session | Control (blind windows) | Order gain |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    t0 = report["theta0"]
    lines.append(f"| θ₀, untrained | 1 | {t0['validation']:.3f} | {t0['test']:.3f} | {t0['test_worst_session']:.3f} | "
                 f"{t0['control']:.3f} | {t0['test_order_gain_bits']:.3f} |")
    for name, b in report["baselines"].items():
        lines.append(f"| {name}, fitted on the training sessions | 1 | {b['validation']:.3f} | {b['test']:.3f} | "
                     f"{b['test_worst_session']:.3f} | {b['control']:.3f} | – |")
    for name, row in report["summary"].items():
        lines.append(f"| {name} | {row['runs']} | {fmt(row['validation'])} | {fmt(row['test'])} | {fmt(row['test_worst_session'])} | "
                     f"{fmt(row['control'])} | {fmt(row['test_order_gain_bits'])} |")
    return "\n".join(lines)


def theta0_scores(ctx):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "theta0.pt"
        anchor = ctx.anchor
        torch.save({"format": "eiathome-signal-lm", "version": 1, "variant": "real", "config": anchor.spec["model"],
                    "model": OrderedDict((k, v.clone()) for k, v in anchor.theta0.items()), "tokenizer": anchor.tokenizer.state(),
                    "anchor": anchor.identity(), "merge": {"round": 0}}, path)
        return assess(ctx, path)


def run(args):
    bench = benchmark(getattr(args, "benchmark", None))
    manifest = Path(args.manifest or bench.manifest)
    split = build_split(manifest, bench)
    if args.publish:
        if manifest.resolve() != bench.manifest.resolve() or tuple(args.seeds) != SEEDS or args.epochs != EPOCHS or args.rounds != ROUNDS:
            raise ValueError("--publish writes the frozen benchmark: its manifest, seeds, epochs and rounds only")
        check_split(manifest, bench=bench)
    if args.epochs % args.rounds:
        raise ValueError("--epochs must be a multiple of --rounds, so the chains train as many epochs as the one-shot workers")
    require_new_output(args.output)
    args.output.mkdir(parents=True)
    anchor = load_anchor(ANCHOR)
    ctx = Context(anchor, manifest)
    flag = "" if bench.name == "v0" else f" --benchmark {bench.name}"
    report = {"format": "eiathome-signal-lm-benchmark-report", "version": 1, "benchmark": bench.id,
              "protocol": __doc__.split("\n\n", 1)[1].split("\n\nThe base and the four", 1)[0],
              "split_sha256": digest(split_text(split).encode()), "dataset_sha256": ctx.dataset_sha256,
              "anchor": anchor.identity(), "code_sha256": code_sha256(), "environment": environment(),
              "settings": {"seeds": list(args.seeds), "epochs": args.epochs, "patience": PATIENCE, "rounds": args.rounds,
                           "shards": SHARDS, "unit": UNIT,
                           "one_shot_rules": ONE_SHOT, "chain_rules": CHAINS, "tolerance_bits": RULE["tolerance_bits"]},
              "commands": {"split": f"python -m tracks.llm.session_bench split --check{flag}",
                           "run": f"python -m tracks.llm.session_bench run --output {args.output}{flag}",
                           "verify": f"python -m tracks.llm.session_bench verify{flag}"},
              "theta0": theta0_scores(ctx), "baselines": baselines(ctx), "seeds": {}}
    for seed in args.seeds:
        report["seeds"][str(seed)] = one_seed(args.output / f"seed{seed}", seed, args.epochs, args.rounds, manifest, ctx)
        print(f"seed {seed} done", flush=True)
    report["summary"] = summarize(report["seeds"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(table(report))
    if args.publish:
        publish(report, {seed: cohort_paths(args.output, seed) for seed in args.seeds}, anchor, bench=bench)
    return report


# ---------- published checkpoints ----------

def export(anchor, path, header, bench=None):
    """A checkpoint's weights as θ₀ is stored (raw float32 in parameter order) and its header."""
    checkpoint = load_checkpoint(path, anchor)
    state = checkpoint["model"]
    difference(state, anchor.theta0)  # the anchor's parameters, in its order
    data = b"".join(np.ascontiguousarray(v.detach().cpu().numpy(), dtype="<f4").tobytes() for v in state.values())
    merge_info = {k: v for k, v in checkpoint["merge"].items()}
    return {"format": "eiathome-signal-lm-weights", "version": 1, "benchmark": benchmark(bench).id, **header,
            "anchor": anchor.identity(), "config": anchor.spec["model"], "dtype": "float32-le",
            "dataset_sha256": checkpoint["dataset_sha256"], "evaluation": checkpoint["evaluation"],
            "best_epoch": checkpoint.get("best_epoch"), "merge": merge_info,
            "tensors": canonical_layout(anchor.spec["model"])[0], "sha256": digest(data)}, data


def publish(report, published, anchor, folder=None, bench=None):
    """Writes split.json's companions: the report and each seed's base and workers."""
    bench = benchmark(bench)
    folder = bench.folder if folder is None else folder
    (folder / "checkpoints").mkdir(parents=True, exist_ok=True)
    for seed, (base, workers) in published.items():
        target = folder / "checkpoints" / f"seed{seed}"
        target.mkdir(parents=True, exist_ok=True)
        entries = [("base", base, {"role": "base", "seed": seed, "shard": [BASE_SHARD, SHARDS], "shard_unit": UNIT})]
        entries += [(f"worker{k}", path, {"role": "worker", "seed": seed, "shard": [k, SHARDS], "shard_unit": UNIT})
                    for k, path in workers.items()]
        for name, path, header in entries:
            head, data = export(anchor, path, {**header, "split_sha256": report["split_sha256"]}, bench)
            (target / f"{name}.bin").write_bytes(data)
            (target / f"{name}.json").write_text(json.dumps(head, indent=1) + "\n")
    (folder / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def load_published(anchor, path, manifest=None, bench=None):
    """A published base or worker, as a checkpoint merge.py loads: the weights from the .bin,
    checked against its header, and its splits from the manifest."""
    bench = benchmark(bench)
    manifest = bench.manifest if manifest is None else manifest
    path = Path(path)
    header = json.loads(path.with_suffix(".json").read_text())
    data = path.with_suffix(".bin").read_bytes()
    if header.get("format") != "eiathome-signal-lm-weights" or header.get("benchmark") != bench.id:
        raise ValueError(f"{path}: not this benchmark's published weights")
    if header.get("anchor") != anchor.identity() or header.get("tensors") != canonical_layout(anchor.spec["model"])[0]:
        raise ValueError(f"{path}: not anchor {anchor.name}'s parameters")
    if digest(data) != header["sha256"]:
        raise ValueError(f"{path}: its weights don't match the SHA-256 in its header")
    values = np.frombuffer(data, dtype="<f4")
    state = OrderedDict((t["name"], torch.from_numpy(values[t["offset"]:t["offset"] + t["count"]].copy()).reshape(t["shape"]))
                        for t in header["tensors"])
    if header["role"] == "base" and header["merge"].get("weights_sha256") != header["sha256"]:
        raise ValueError(f"{path}: its weights don't match the digest its round recorded")
    records, partitions = held_out(anchor, FAMILY, header["evaluation"], manifest, header["dataset_sha256"])
    held = {name: [r["entry"] for r in partitions[name]] for name in ("validation", "test")}
    common = {"format": "eiathome-signal-lm", "version": 1, "variant": "real", "config": anchor.spec["model"], "model": state,
              "tokenizer": anchor.tokenizer.state(), "dataset_sha256": header["dataset_sha256"], "evaluation": header["evaluation"],
              "anchor": anchor.identity(), "merge": header["merge"]}
    if header["role"] == "base":
        return {**common, "splits": {"train": None, **held}, "outer_momentum": None}
    shard = header["merge"]["shard"]
    train = shard_records(partitions["train"], shard[0], shard[1], header["merge"].get("shard_unit", "slice"))
    return {**common, "splits": {"train": [r["entry"] for r in train], **held}, "best_epoch": header["best_epoch"]}


PUBLISHED = ("base", *(f"worker{k}" for k in WORKER_SHARDS))


def published_files(folder=None, seeds=SEEDS, bench=None):
    """Each seed's published base and workers 2 to 5: all of them, and nothing else. Refuses a
    missing or extra seed, and a seed missing a checkpoint or its header."""
    root = (benchmark(bench).folder if folder is None else folder) / "checkpoints"
    found = sorted(p.name for p in root.iterdir()) if root.is_dir() else []
    expected = sorted(f"seed{s}" for s in seeds)
    if found != expected:
        raise ValueError(f"{root} holds {', '.join(found) or 'nothing'}; the benchmark publishes {', '.join(expected)}")
    files = {}
    for seed in sorted(seeds):
        folder_ = root / f"seed{seed}"
        names = sorted(p.name for p in folder_.iterdir())
        wanted = sorted(f"{name}.{suffix}" for name in PUBLISHED for suffix in ("bin", "json"))
        if names != wanted:
            missing, extra = sorted(set(wanted) - set(names)), sorted(set(names) - set(wanted))
            raise ValueError(f"{folder_}: " + "; ".join(([f"missing {', '.join(missing)}"] if missing else [])
                                                       + ([f"unexpected {', '.join(extra)}"] if extra else [])))
        files[seed] = {name: folder_ / f"{name}.bin" for name in PUBLISHED}
    return files


def materialize(output, manifest=None, folder=None, seeds=SEEDS, bench=None):
    """Every published base and worker as a .pt checkpoint, under output/seed*/."""
    bench = benchmark(bench)
    manifest = bench.manifest if manifest is None else manifest
    anchor = load_anchor(ANCHOR)
    require_new_output(output)
    paths = {}
    for seed, files in published_files(folder, seeds, bench).items():
        (output / f"seed{seed}").mkdir(parents=True)
        for name, path in files.items():
            target = output / f"seed{seed}" / f"{name}.pt"
            torch.save(load_published(anchor, path, manifest, bench), target)
            paths[(seed, name)] = target
    return paths


def verify(manifest=None, folder=None, seeds=SEEDS, bench=None):
    """The frozen benchmark, checked: the split's pin; every seed's base and workers 2 to 5, all
    present (hashes, starts, disjoint session shards, none trained on a held-out session); their
    one-shot merges and scores, and the baselines, recomputed and compared with report.json."""
    bench = benchmark(bench)
    manifest = bench.manifest if manifest is None else manifest
    folder = bench.folder if folder is None else folder
    split = check_split(manifest, folder, bench)
    report = json.loads((folder / "report.json").read_text())
    if report["split_sha256"] != bench.split_sha256:
        raise ValueError("report.json was made on another split")
    if sorted(report["seeds"]) != sorted(map(str, seeds)) or report["settings"]["seeds"] != list(seeds):
        raise ValueError(f"report.json covers seeds {', '.join(sorted(report['seeds']))}; the benchmark's are {', '.join(map(str, seeds))}")
    anchor = load_anchor(ANCHOR)
    ctx = Context(anchor, manifest)
    for name, again in baselines(ctx).items():
        for key in ("validation", "test", "test_worst_session", "control"):
            if abs(again[key] - report["baselines"][name][key]) > 1e-9:
                raise ValueError(f"The {name} baseline scores {again[key]} {key}, not {report['baselines'][name][key]}")
    held_out_ids = {s["sha256"] for name in ("validation", "test") for sess in split["splits"][name]["sessions"] for s in sess["slices"]}
    checked = []
    with tempfile.TemporaryDirectory() as tmp:
        paths = materialize(Path(tmp) / "checkpoints", manifest, folder, seeds, bench)
        for seed in seeds:
            recorded_seed = report["seeds"][str(seed)]
            if sorted(w["shard"] for w in recorded_seed["workers"]) != list(WORKER_SHARDS):
                raise ValueError(f"seed {seed}: report.json doesn't score workers {', '.join(map(str, WORKER_SHARDS))}")
            base = paths[(seed, "base")]
            base_cp = load_checkpoint(base, anchor)
            if base_cp["merge"].get("round") != 1 or base_cp["merge"].get("start_sha256") != anchor.spec["theta0_sha256"]:
                raise ValueError(f"seed {seed}: the base isn't round 1 from θ₀")
            if weights_digest(base_cp["model"]) != recorded_seed["base"]["weights_sha256"]:
                raise ValueError(f"seed {seed}: the published base isn't the one report.json scored")
            workers, seen = [], set()
            for w in recorded_seed["workers"]:
                path = paths[(seed, f"worker{w['shard']}")]
                cp = load_checkpoint(path, anchor)
                if cp["merge"].get("start_sha256") != recorded_seed["base"]["weights_sha256"] or cp["merge"].get("round") != 2:
                    raise ValueError(f"seed {seed}: worker {w['shard']} didn't start from the base")
                if weights_digest(cp["model"]) != w["weights_sha256"]:
                    raise ValueError(f"seed {seed}: worker {w['shard']} isn't the one report.json scored")
                hashes = {e["sha256"] for e in cp["splits"]["train"]}
                if hashes & seen or hashes & held_out_ids:
                    raise ValueError(f"seed {seed}: worker {w['shard']} shares a slice with another worker or a held-out session")
                seen |= hashes
                workers.append(path)
            for i, (name, rule) in enumerate(ONE_SHOT.items()):
                with training_runtime(1, "cpu"):
                    record, merged = merge_round(anchor, [str(w) for w in workers], str(base), rule, manifest)
                recorded = recorded_seed["one_shot"][name]
                if record["merged"].get("weights_sha256") != recorded["round"]["weights_sha256"]:
                    raise ValueError(f"seed {seed}: replaying {name} gave different weights")
                if merged is not None:
                    target = Path(tmp) / f"seed{seed}-merge{i}.pt"
                    torch.save(merged_checkpoint(anchor, record, merged, lineage(load_checkpoint(workers[0], anchor))), target)
                    again = assess(ctx, target)
                    for key in ("validation", "test", "test_worst_session", "control"):
                        if abs(again[key] - recorded[key]) > 1e-6:
                            raise ValueError(f"seed {seed}: {name} scores {again[key]} {key}, not {recorded[key]}")
            scored = [("base", base, recorded_seed["base"])] + [(f"worker {w['shard']}", path, w) for w, path in zip(recorded_seed["workers"], workers)]
            for name, path, recorded in scored:
                again = assess(ctx, path)
                for key in ("validation", "test", "control"):
                    if abs(again[key] - recorded[key]) > 1e-6:
                        raise ValueError(f"seed {seed}: {name} scores {again[key]} {key}, not {recorded[key]}")
            checked.append(seed)
    return checked


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("split", help="build split.json")
    s.add_argument("--manifest", type=Path, default=None, help="default: the benchmark's corpus")
    s.add_argument("--write", action="store_true", help="write split.json (a new benchmark version: re-pin SPLIT_SHA256)")
    s.add_argument("--check", action="store_true", help="check split.json against its pin and the manifest")
    r = sub.add_parser("run", help="run the benchmark")
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--manifest", type=Path, default=None, help="default: the benchmark's corpus")
    r.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    r.add_argument("--epochs", type=int, default=EPOCHS)
    r.add_argument("--rounds", type=int, default=ROUNDS)
    r.add_argument("--publish", action="store_true", help="rewrite the published report and checkpoints (the frozen settings only)")
    v = sub.add_parser("verify", help="check the published split, checkpoints and one-shot merges")
    v.add_argument("--manifest", type=Path, default=None, help="default: the benchmark's corpus")
    m = sub.add_parser("materialize", help="write the published checkpoints as .pt files")
    m.add_argument("--output", type=Path, required=True)
    for command in (s, r, v, m):
        command.add_argument("--benchmark", choices=("v0", "v1"), default="v0", help="the benchmark version (default v0)")
    args = p.parse_args(argv)
    try:
        bench = benchmark(args.benchmark)
        if args.command == "split":
            split = build_split(args.manifest, bench)
            if args.write:
                bench.folder.mkdir(parents=True, exist_ok=True)
                (bench.folder / "split.json").write_text(split_text(split))
                print(f"wrote split.json: {digest(split_text(split).encode())}")
            if args.check:
                check_split(args.manifest, bench=bench)
                print(f"split.json matches its pin ({bench.split_sha256[:16]}) and the manifest")
            for name, part in split["splits"].items():
                print(f"{name}: {len(part['sessions'])} sessions, {part['slices']} slices, {part['targets']} targets")
        elif args.command == "run":
            run(args)
        elif args.command == "verify":
            seeds = verify(args.manifest, bench=bench)
            print(f"verified: split.json, and the published base, workers and one-shot merges of seeds {', '.join(map(str, seeds))}")
        else:
            paths = materialize(args.output, bench=bench)
            print(f"wrote {len(paths)} checkpoints under {args.output}")
    except (ValueError, OSError, RuntimeError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
