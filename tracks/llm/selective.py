"""Selective specialist execution: milestone 5's committed routers, run the way a deployment would
run them, and measured.

Milestone 5 (route_bench.py, MERGE_RESULTS.md) found a router that keeps milestone 3b's three
family specialists whole and picks one per sequence from the tokens already seen. Its cheaper
variant commits to one specialist after the first 8 tokens. But the benchmark itself runs every
specialist on every target and routes over their cached scores, so the cost it reports for
committing (35,728 parameters per target, against 82,851 for the whole bank) is a projection.

This module runs the router selectively: every specialist on each sequence's first n − 1 targets
(the evidence the router needs), then only the specialist that evidence picks on the rest. It
checks that this reproduces the benchmark's committed router, target for target, and measures what
the bank, the committed routers and a single specialist cost:

- **compute actually run:** model forward calls and token positions, so the parameter-positions
  per target are counted, not projected (the committed specialist reruns the prefix: the model has
  no key-value cache, so its first window starts from the sequence's start);
- **latency:** seconds per sequence one at a time (batch 1, as a stream would arrive), and
  sequences per second batched: every family's sequences together, as a router that doesn't
  know the family would get them, and each family's apart. Within a batch, the windows run as one
  call per (specialist, window length). One CPU thread, the median of repeated runs;
- **memory:** the exact bytes of weights each arm must keep resident (every arm that may see any
  family keeps all three specialists; only a control serving one family keeps one), and, in a
  fresh process running the arm's timed workload, what loading its specialists and then running
  it add to the peak resident memory (/proc/self/status VmHWM). Each probe must score what the
  timed workload scores;
- **energy:** RAPL package energy (/sys/class/powercap) where the machine exposes it; otherwise
  it isn't measured, and only CPU seconds are reported. Nothing is estimated.

The bank is route_bench.py's: anchor v3, Voyager, Rosetta and the RFI sweep, a warm start and one
specialist per family, trained by its own code (`train_bank`), or read from a route_bench run.

    python -m tracks.llm.selective --output runs/selective                        # trains seeds 42-44 (minutes each), then measures
    python -m tracks.llm.selective --output runs/selective --banks runs/route-bench  # reuse route_bench's trained banks
"""

import argparse
import gc
import glob
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .anchor import load_anchor
from .data import load_slices
from .merge import load_checkpoint, model_from
from .moe import load_specialists, warmup
from .moe_ablation import ANCHOR, FAMILIES, MANIFEST, TRAINING, WARM_EPOCHS, specialist
from .route_bench import LOG2, ROUTERS, frequencies, target_bits
from .runtime import require_new_output, training_runtime

COMMIT_AFTER = (8, 4)
SEEDS = (42, 43, 44)
REPEATS = 25
ARMS = ("bank", "committed after 8", "committed after 4", "one specialist")
# A memory control, not a deployment arm: a worker that only ever serves Voyager's sequences, so it
# loads Voyager's specialist alone. Every arm above may see any family, so keeps all three loaded.
CONTROL = "one family's specialist alone (memory control)"
MEMORY_ARMS = (*ARMS, CONTROL)
FOLDER = Path(__file__).resolve().parent / "benchmarks" / "selective-v0"


# ---------- scoring in windows, counted ----------

class Work:
    """What a model actually ran: forward calls and token positions."""

    def __init__(self):
        self.calls = self.positions = 0


@torch.no_grad()
def window_bits(model, sequences, batched=True, work=None):
    """Each next-token target's bits, per sequence, as target_bits (and evaluate) score them:
    windows of the context from the sequence's start. Batched, every window of one length runs as
    one batch; otherwise one window at a time."""
    model.eval()
    context = model.config["context"]
    windows = []
    for i, s in enumerate(sequences):
        for start in range(0, len(s) - 1, context):
            y = s[start + 1:start + context + 1]
            windows.append((i, start, s[start:start + len(y)], y))
    out = [np.empty(len(s) - 1) for s in sequences]
    if batched:
        groups = {}
        for w in windows:
            groups.setdefault(len(w[3]), []).append(w)
        batches = list(groups.values())
    else:
        batches = [[w] for w in windows]
    for batch in batches:
        x = torch.from_numpy(np.asarray([w[2] for w in batch], dtype=np.int64))
        y = np.asarray([w[3] for w in batch])
        logp = F.log_softmax(model(x).double(), dim=-1)
        bits = -torch.gather(logp, 2, torch.from_numpy(y).long()[..., None])[..., 0].numpy() / LOG2
        for (i, start, _, _), b in zip(batch, bits):
            out[i][start:start + len(b)] = b
        if work is not None:
            work.calls += 1
            work.positions += x.numel()
    return out


# ---------- the arms ----------

def committed(models, logfreq, sequences, n, batched=True, works=None):
    """The specialists' likelihood router, committed after n tokens, run selectively. Every
    specialist predicts each sequence's first n − 1 targets; each of those goes to the specialist
    with the best evidence before it (route_bench's hard router). The evidence after them picks
    the one specialist that alone predicts the rest. Returns per sequence (bits per target,
    specialist per target)."""
    works = works or [None] * len(models)
    everyone = [min(n - 1, len(s) - 1) for s in sequences]
    prefixes = [s[:e + 1] for s, e in zip(sequences, everyone)]
    ran = [i for i, e in enumerate(everyone) if e > 0]
    pre = [window_bits(m, [prefixes[i] for i in ran], batched, w) for m, w in zip(models, works)] if ran else []
    prefix_bits = {i: np.stack([p[j] for p in pre]) for j, i in enumerate(ran)}
    decided, rest = [], {}
    for i, s in enumerate(sequences):
        first = logfreq[:, s[0]]
        bits = prefix_bits.get(i, np.zeros((len(models), 0)))
        before = first[:, None] + np.concatenate([np.zeros((len(models), 1)), -np.cumsum(bits, axis=1)], axis=1)
        choice = before[:, :-1].argmax(axis=0)  # each evidence target to the best specialist before it
        commit = int(before[:, -1].argmax())  # the evidence after them all
        decided.append((bits[choice, np.arange(bits.shape[1])], choice, commit))
        if len(s) - 1 > everyone[i]:
            rest.setdefault(commit, []).append(i)
    # Windows are scored independently, each from its own start, so the committed specialist
    # resumes at the window holding its first target after the evidence. Windows the evidence
    # completed aren't run again; only the one it ends inside is (there's no key-value cache).
    context = next(m for m in models if m is not None).config["context"]
    tails = {}
    for k, members in rest.items():  # each committed specialist runs once, on its own sequences
        starts = [everyone[i] // context * context for i in members]
        suffixes = [sequences[i][start:] for i, start in zip(members, starts)]
        for i, start, b in zip(members, starts, window_bits(models[k], suffixes, batched, works[k])):
            tails[i] = b[everyone[i] - start:]
    out = []
    for i, s in enumerate(sequences):
        bits, choice, commit = decided[i]
        tail = tails.get(i, np.empty(0))
        out.append((np.concatenate([bits, tail]), np.concatenate([choice, np.full(len(tail), commit)]).astype(int)))
    return out


def bank(models, logfreq, sequences, batched=True, works=None, router="posterior mixture"):
    """Every specialist on every target, then a router over their scores (route_bench's way)."""
    works = works or [None] * len(models)
    per_model = [window_bits(m, sequences, batched, w) for m, w in zip(models, works)]
    return [ROUTERS[router](np.asarray(s), np.stack([p[i] for p in per_model]), logfreq) for i, s in enumerate(sequences)]


def one(model, sequences, batched=True, work=None):
    """A single specialist on every target (dispatch by label: the cheapest a bank can be)."""
    return window_bits(model, sequences, batched, work)


def resident(arm):
    """The specialists an arm keeps loaded. Dispatch by label keeps all three too: any family may
    arrive next. Only the control, which serves one family, loads one."""
    return [0] if arm == CONTROL else list(range(len(FAMILIES)))


def workload(arm, models, logfreq, sequences):
    """What an arm runs over each family's sequences (one batch per family), the same for the
    timings and the memory probes: {family: per-sequence results}. Dispatch by label sends each
    family to its own specialist; the control runs Voyager's sequences on Voyager's specialist."""
    if arm == CONTROL:
        return {FAMILIES[0]: one(models[0], sequences[FAMILIES[0]])}
    return {f: run_arm(arm, models, logfreq, sequences[f], k) for k, f in enumerate(FAMILIES)}


def workload_bits(results):
    """Each family's mean bits per target from a workload's results (routers return (bits, routing))."""
    bits = lambda r: r[0] if isinstance(r, tuple) else r
    return {f: float(np.concatenate([bits(r) for r in rs]).mean()) for f, rs in results.items()}


def run_arm(arm, models, logfreq, sequences, own, batched=True, works=None):
    if arm == "bank":
        return bank(models, logfreq, sequences, batched, works)
    if arm.startswith("committed after"):
        return committed(models, logfreq, sequences, int(arm.split()[-1]), batched, works)
    return one(models[own], sequences, batched, None if works is None else works[own])


# ---------- the bank ----------

def train_bank(folder, seed):
    """route_bench.py's bank for one seed, by its own code: θ_warm, then one specialist per family."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    anchor = load_anchor(ANCHOR)
    with training_runtime(1, "cpu"):
        torch.save(warmup(anchor, load_slices(MANIFEST), FAMILIES, WARM_EPOCHS, seed), folder / "warm.pt")
    for family in FAMILIES:
        specialist(folder / f"warm-{family}", family, seed, TRAINING["epochs"], folder / "warm.pt")
    return folder


def load_bank(folder):
    """A trained bank (route_bench's layout: warm.pt and warm-<family>/real.pt), verified as
    route_bench verifies it: its models, the family frequencies, and each family's test and
    validation sequences."""
    anchor = load_anchor(ANCHOR)
    folder = Path(folder)
    states, _, _, partitions = load_specialists(anchor, [folder / f"warm-{f}" / "real.pt" for f in FAMILIES], MANIFEST, folder / "warm.pt")
    models = [model_from(anchor, s) for s in states]
    sequences = {split: {f: [np.asarray(anchor.tokenizer.encode(r["frames"])) for r in partitions[f][split]] for f in FAMILIES}
                 for split in ("validation", "test")}
    return anchor, models, frequencies(anchor, partitions), sequences


# ---------- measuring ----------

def rapl_domains(root="/sys/class/powercap"):
    """RAPL package energy counters (energy_uj and its wrap-around range), where the kernel
    exposes them and they can be read."""
    domains = []
    for path in sorted(glob.glob(os.path.join(root, "intel-rapl:*"))):
        try:
            name = Path(path, "name").read_text().strip()
            if name.startswith("package"):
                domains.append((Path(path, "energy_uj"), int(Path(path, "max_energy_range_uj").read_text())))
        except OSError:
            continue
    try:
        [int(p.read_text()) for p, _ in domains]
    except OSError:  # present but unreadable without privileges
        return []
    return domains


def energy_reading(domains):
    return [int(p.read_text()) for p, _ in domains]


def energy_joules(domains, before, after):
    return sum(((b - a) % wrap) for (_, wrap), a, b in zip(domains, before, after)) / 1e6


def timed(fns, repeats=REPEATS, domains=None):
    """Median wall and CPU seconds of each fn() over repeats, and the median package energy where
    RAPL is readable. The functions take turns within each repeat (in an order that rotates), so a
    machine that speeds up or slows down during the run affects them all alike. Each runs once
    first, untimed."""
    names = list(fns)
    for name in names:
        fns[name]()
    samples = {name: {"wall": [], "cpu": [], "energy": []} for name in names}
    for r in range(repeats):
        for name in names[r % len(names):] + names[:r % len(names)]:
            e0 = energy_reading(domains) if domains else None
            c0, w0 = time.process_time(), time.perf_counter()
            fns[name]()
            samples[name]["wall"].append(time.perf_counter() - w0)
            samples[name]["cpu"].append(time.process_time() - c0)
            if domains:
                samples[name]["energy"].append(energy_joules(domains, e0, energy_reading(domains)))
    return {name: {"wall_s": statistics.median(s["wall"]), "wall_s_quartiles": [float(q) for q in np.quantile(s["wall"], [0.25, 0.75])],
                   "cpu_s": statistics.median(s["cpu"]), "energy_j": statistics.median(s["energy"]) if s["energy"] else None}
            for name, s in samples.items()}


def peak_memory_kib():
    """This process's peak and current resident memory, in KiB (what /proc calls kB)."""
    status = Path("/proc/self/status").read_text()
    return {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in status.splitlines()) if k in ("VmHWM", "VmRSS")}


def probe_memory(arm, load, logfreq, sequences, warm):
    """One arm's workload, measured: the peak resident memory (VmHWM, KiB) before any of its
    specialists is loaded, after loading them, and after the workload ran (so what loading and
    running each add), and what it scored, so a probe can be checked against the timed workload. The runtime is warmed up first on `warm` (θ₀), so what's left is the arm's
    own weights and activations. load(k): specialist k."""
    for f in FAMILIES:
        window_bits(warm, sequences[f])
    gc.collect()
    # The warm-up model stays alive until after the measurement: freed, its memory would be
    # reused by the arm's own weights without raising the peak, by an amount that depends on the
    # order of allocations.
    before = peak_memory_kib()
    loaded = {k: load(k) for k in resident(arm)}
    models = [loaded.get(k) for k in range(len(FAMILIES))]
    after_load = peak_memory_kib()
    results = workload(arm, models, logfreq, sequences)
    after = peak_memory_kib()
    del warm
    return {"resident": [FAMILIES[k] for k in resident(arm)], "before_kib": before, "after_load_kib": after_load, "after_kib": after,
            "load_increase_kib": after_load["VmHWM"] - before["VmHWM"], "run_increase_kib": after["VmHWM"] - after_load["VmHWM"],
            "peak_increase_kib": after["VmHWM"] - before["VmHWM"], "bits": workload_bits(results)}


def memory_probe(folder, arm, data):
    """probe_memory in this (fresh) process, on a trained bank's checkpoints."""
    anchor = load_anchor(ANCHOR)
    folder = Path(folder)
    data = json.loads(Path(data).read_text())
    sequences = {f: [np.asarray(s) for s in seqs] for f, seqs in data["sequences"].items()}
    load = lambda k: model_from(anchor, load_checkpoint(folder / f"warm-{FAMILIES[k]}" / "real.pt", anchor)["model"])
    with training_runtime(1, "cpu"):
        return probe_memory(arm, load, np.asarray(data["logfreq"]), sequences, model_from(anchor, anchor.theta0))


def exactness(models, logfreq, sequences):
    """The selective committed routers against route_bench's, computed from every specialist's
    cached scores: the largest difference in any target's bits, and how many targets go to
    another specialist."""
    out = {}
    for n in COMMIT_AFTER:
        name = f"specialists' likelihood, committed after {n} tokens"
        diff, moved, targets, bits = 0.0, 0, 0, {}
        for f, seqs in sequences.items():
            per_model = [target_bits(m, seqs) for m in models]
            total, count = 0.0, 0
            for i, s in enumerate(seqs):
                ref_bits, ref_weights = ROUTERS[name](s, np.stack([p[i] for p in per_model]), logfreq)
                sel_bits, sel_choice = committed(models, logfreq, [s], n)[0]
                diff = max(diff, float(np.abs(sel_bits - ref_bits).max()))
                moved += int((ref_weights.argmax(axis=0) != sel_choice).sum())
                targets += len(sel_bits)
                total += sel_bits.sum()
                count += len(sel_bits)
            bits[f] = total / count
        out[f"committed after {n}"] = {"max_abs_bits_difference": diff, "targets_routed_differently": moved, "targets": targets,
                                       "test_bits": {**bits, "mean": statistics.fmean(bits.values())}}
    return out


def measure_seed(folder, repeats=REPEATS, memory=True):
    anchor, models, logfreq, sequences = load_bank(folder)
    single = sum(p.numel() for p in models[0].parameters())
    domains = rapl_domains()
    out = {"exactness": None, "arms": {}}
    with training_runtime(1, "cpu"):
        out["exactness"] = exactness(models, logfreq, sequences["test"])
        # Timed on every held-out sequence (validation and test), for more of them than test's 12.
        test = {f: sequences["validation"][f] + sequences["test"][f] for f in FAMILIES}
        count = sum(len(test[f]) for f in FAMILIES)
        targets = sum(len(s) - 1 for f in FAMILIES for s in test[f])
        mixed = [s for f in FAMILIES for s in test[f]]
        # Three ways to run an arm over the held-out sequences:
        # - stream: one sequence at a time;
        # - mixed: every family's sequences in one batch, as a router that doesn't know the
        #   family would get them (dispatch by label still sends each family to its specialist);
        # - per family: each family's sequences in a batch of their own.
        # Within a batch, windows run grouped by (specialist, window length): one call per group.
        def plan(arm, mode):
            if mode == "stream":
                return [(arm, [s], k, False) for k, f in enumerate(FAMILIES) for s in test[f]]
            if mode == "mixed" and arm != "one specialist":
                return [(arm, mixed, 0, True)]
            return [(arm, test[f], k, True) for k, f in enumerate(FAMILIES)]
        runner = lambda arm, mode: lambda: [run_arm(a, models, logfreq, seqs, k, batched=b) for a, seqs, k, b in plan(arm, mode)]
        timings = {mode: timed({arm: runner(arm, mode) for arm in ARMS}, repeats, domains) for mode in ("stream", "mixed", "per family")}
        for arm in ARMS:
            counted = {}
            for mode in ("stream", "mixed", "per family"):
                works = [Work() for _ in models]
                for a, seqs, k, b in plan(arm, mode):
                    run_arm(a, models, logfreq, seqs, k, batched=b, works=works)
                counted[mode] = (sum(w.calls for w in works), sum(w.positions for w in works))
            calls, positions = counted["stream"]
            loaded = len(resident(arm)) * single
            s, b, bf = timings["stream"][arm], timings["mixed"][arm], timings["per family"][arm]
            committed_to = None
            if arm.startswith("committed after"):  # where each family's sequences go, which sets the batched calls
                committed_to = {}
                for f in FAMILIES:
                    picks = [int(c[-1]) for _, c in committed(models, logfreq, test[f], int(arm.split()[-1]))]
                    committed_to[f] = {FAMILIES[k]: picks.count(k) for k in sorted(set(picks))}
            out["arms"][arm] = {
                "sequences": count, "targets": targets,
                "forward_calls_per_sequence": calls / count, "positions_per_target": positions / targets,
                "parameter_positions_per_target": positions * single / targets,
                "per_sequence": {**s, "ms_per_sequence": 1000 * s["wall_s"] / count,
                                 "energy_mj_per_sequence": None if s["energy_j"] is None else 1000 * s["energy_j"] / count},
                "batched": {**b, "sequences_per_s": count / b["wall_s"], "forward_calls": counted["mixed"][0]},
                "batched_per_family": {**bf, "sequences_per_s": count / bf["wall_s"], "forward_calls": counted["per family"][0]},
                "committed_to": committed_to,
                "weights_resident_bytes": loaded * 4, "weights_resident_parameters": loaded,
            }
    if memory:
        # Each probe runs, in a fresh process, the same workload as the per-family timings, on the
        # same held-out sequences, and must score what it scores here.
        data = Path(folder) / "selective-probe.json"
        data.write_text(json.dumps({"logfreq": logfreq.tolist(), "sequences": {f: [s.tolist() for s in seqs] for f, seqs in test.items()}}))
        out["memory"] = {}
        with training_runtime(1, "cpu"):
            for arm in MEMORY_ARMS:
                probe = subprocess.run([sys.executable, "-m", "tracks.llm.selective", "memory", str(folder), arm, str(data)],
                                       capture_output=True, text=True, check=True, cwd=Path(__file__).resolve().parents[2])
                measured = json.loads(probe.stdout.strip().splitlines()[-1])
                here = workload_bits(workload(arm, models, logfreq, test))
                if measured["bits"].keys() != here.keys() or any(abs(measured["bits"][f] - here[f]) > 1e-9 for f in here):
                    raise RuntimeError(f"{arm}: the memory probe ran another workload than the timings")
                measured["weights_resident_bytes"] = len(resident(arm)) * single * 4
                out["memory"][arm] = measured
                if arm in out["arms"]:
                    out["arms"][arm]["memory"] = measured
        data.unlink()
    out["energy"] = "RAPL package counters" if domains else "not measured: this machine exposes no readable RAPL counters"
    return out


def environment():
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor()) if Path("/proc/cpuinfo").exists() else platform.processor()
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__), "platform": platform.platform(),
            "cpu": cpu, "cpu_count": os.cpu_count(), "threads": 1}


def summarize(seeds):
    out = {}
    for arm in ARMS:
        rows = [s["arms"][arm] for s in seeds.values()]
        mean = lambda f: statistics.fmean(f(r) for r in rows)
        out[arm] = {"parameter_positions_per_target": mean(lambda r: r["parameter_positions_per_target"]),
                    "forward_calls_per_sequence": mean(lambda r: r["forward_calls_per_sequence"]),
                    "forward_calls_batched": mean(lambda r: r["batched"]["forward_calls"]),
                    "forward_calls_batched_per_family": mean(lambda r: r["batched_per_family"]["forward_calls"]),
                    "sequences_per_s_batched_per_family": mean(lambda r: r["batched_per_family"]["sequences_per_s"]),
                    "ms_per_sequence": mean(lambda r: r["per_sequence"]["ms_per_sequence"]),
                    "sequences_per_s_batched": mean(lambda r: r["batched"]["sequences_per_s"]),
                    "cpu_s_per_sequence_stream": mean(lambda r: r["per_sequence"]["cpu_s"] / r["sequences"]),
                    "weights_resident_bytes": rows[0]["weights_resident_bytes"],  # exact: parameters × 4 bytes
                    "peak_rss_increase_kib": [r["memory"]["peak_increase_kib"] for r in rows if "memory" in r],  # measured
                    "peak_rss_load_kib": [r["memory"]["load_increase_kib"] for r in rows if "memory" in r],
                    "peak_rss_run_kib": [r["memory"]["run_increase_kib"] for r in rows if "memory" in r]}
    probes = [s["memory"][CONTROL] for s in seeds.values() if CONTROL in s.get("memory", {})]
    control = {"weights_resident_bytes": probes[0]["weights_resident_bytes"] if probes else None,
               "peak_rss_increase_kib": [m["peak_increase_kib"] for m in probes],
               "peak_rss_load_kib": [m["load_increase_kib"] for m in probes], "peak_rss_run_kib": [m["run_increase_kib"] for m in probes]}
    bank = out["bank"]
    for arm, row in out.items():
        row["latency_vs_bank"] = row["ms_per_sequence"] / bank["ms_per_sequence"]
        row["throughput_vs_bank"] = row["sequences_per_s_batched"] / bank["sequences_per_s_batched"]
        row["throughput_per_family_vs_bank"] = row["sequences_per_s_batched_per_family"] / bank["sequences_per_s_batched_per_family"]
        row["compute_vs_bank"] = row["parameter_positions_per_target"] / bank["parameter_positions_per_target"]
    out[CONTROL] = control
    return out


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["memory"]:
        folder, arm, data = argv[1], argv[2], argv[3]
        print(json.dumps(memory_probe(folder, arm, data)))
        return
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--banks", type=Path, default=None, help="a route_bench (or earlier) output with seed<N>/ banks to reuse")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--repeats", type=int, default=REPEATS)
    p.add_argument("--publish", action="store_true", help=f"also write the report to {FOLDER}")
    args = p.parse_args(argv)
    require_new_output(args.output)
    args.output.mkdir(parents=True)
    report = {"format": "eiathome-selective-execution", "version": 1, "environment": environment(),
              "settings": {"anchor": ANCHOR, "families": list(FAMILIES), "commit_after": list(COMMIT_AFTER), "repeats": args.repeats,
                           "seeds": args.seeds, "banks": None if args.banks is None else str(args.banks)},
              "seeds": {}}
    for seed in args.seeds:
        folder = args.banks / f"seed{seed}" if args.banks else train_bank(args.output / f"seed{seed}", seed)
        report["seeds"][str(seed)] = measure_seed(folder, args.repeats)
        print(f"seed {seed}: measured", flush=True)
    report["summary"] = summarize(report["seeds"])
    text = json.dumps(report, indent=1) + "\n"
    (args.output / "report.json").write_text(text)
    if args.publish:
        FOLDER.mkdir(parents=True, exist_ok=True)
        (FOLDER / "report.json").write_text(text)
    for arm, row in report["summary"].items():
        if arm == CONTROL:
            print(f"{arm}: {row['weights_resident_bytes']:,} bytes of weights, peak +{row['peak_rss_increase_kib']} KiB")
            continue
        print(f"{arm}: {row['parameter_positions_per_target']:,.0f} parameter-positions per target ({row['compute_vs_bank']:.2f} of the bank's), "
              f"{row['ms_per_sequence']:.3f} ms per sequence ({row['latency_vs_bank']:.2f}), "
              f"{row['sequences_per_s_batched']:,.0f} sequences/s batched ({row['throughput_vs_bank']:.2f})")


if __name__ == "__main__":
    main()
