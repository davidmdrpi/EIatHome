"""Blinded signal-recovery test: python -m tracks.recovery.run --output runs/recovery

Thresholds come from the signal-free windows of the calibration scan, at each false-alarm
rate. Signals are then dealt blind into the test scan's windows (with signal-free windows
among them), the detectors score the blind windows, and only then are the scores joined to
the key. Reported per detector and false-alarm rate: the false alarms measured on the test
scan, recovery for each signal class and strength, how often each decoy is flagged, and the
paired outcome of every two detectors on every injection.

All windows come from one observing session and one receiver; every result says so.
"""

import argparse
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import platform
import subprocess

import numpy as np

from . import background
from .detectors import DETECTORS
from .inject import DECOYS, KINDS, MARGIN, MAX_DRIFT, SIGNALS, STRENGTHS, deal, design

ROOT = Path(__file__).resolve().parents[2]
FALSE_ALARM_RATES = (0.01, 0.001)
PER_CELL = 72
LIMITATION = ("One observing session and one receiver: the calibration and test scans are off-source scans "
              "11 minutes apart. Intervals are per window; scan-to-scan variation can't be estimated yet.")


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes in n trials."""
    if not n:
        return [0.0, 1.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def threshold(scores, rate):
    """The smallest threshold that at most `rate` of these signal-free scores exceed."""
    return float(np.quantile(scores, 1 - rate, method="higher"))


def rate(flags):
    k, n = int(np.sum(flags)), len(flags)
    return {"flagged": k, "windows": n, "fraction": k / n if n else None, "interval": wilson(k, n)}


def code_identity():
    """What produced the scores: a fingerprint of this package's source and, in a git checkout,
    the revision and whether this package had uncommitted changes."""
    package = Path(__file__).parent
    digest = hashlib.sha256(b"".join(f.read_bytes() for f in sorted(package.glob("*.py")))).hexdigest()
    git = None
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=package, capture_output=True, text=True, check=True).stdout
        git = {"revision": run("rev-parse", "HEAD").strip(), "uncommitted_changes": bool(run("status", "--porcelain", "--", ".").strip())}
    except (OSError, subprocess.CalledProcessError):
        pass  # not a git checkout: the fingerprint still identifies the code
    return {"sha256": digest, "git": git}


def describe(detect):
    return {"function": f"{detect.__module__}.{detect.__qualname__}", **getattr(detect, "config", {})}


def evaluate(calibration, test, seed, per_cell=PER_CELL, detectors=DETECTORS, rates=FALSE_ALARM_RATES):
    """The full blinded comparison on these background windows. Returns the report."""
    cells = design(per_cell, len(test))
    blind, key = deal(test, cells, seed)
    ids = [i for i, _ in blind]
    windows = np.stack([w for _, w in blind])
    del blind  # detectors get the windows only; the key is used after scoring
    scores = {name: np.asarray(detect(windows)) for name, detect in detectors.items()}
    reference = {name: np.asarray(detect(calibration)) for name, detect in detectors.items()}

    specs = [key[i] for i in ids]
    kinds = np.array([s["kind"] for s in specs])
    strengths = np.array([s["strength"] for s in specs])
    report = {"thresholds": {}, "false_alarms": {}, "recovery": {}, "paired": {}}
    for target in rates:
        label = f"{target:g}"
        flags = {}
        for name in detectors:
            cut = threshold(reference[name], target)
            flags[name] = scores[name] > cut
            report["thresholds"].setdefault(name, {})[label] = cut
            report["false_alarms"].setdefault(name, {})[label] = rate(flags[name][kinds == "none"])
            report["recovery"].setdefault(name, {})[label] = {
                kind: {f"{s:g}": rate(flags[name][(kinds == kind) & (strengths == s)]) for s in STRENGTHS}
                for kind in KINDS}
        report["paired"][label] = {}
        for first, second in combinations(detectors, 2):  # each injection's outcome under both
            a, b = flags[first], flags[second]
            outcomes = {"both": a & b, f"{first} only": a & ~b, f"{second} only": ~a & b, "neither": ~a & ~b}
            report["paired"][label][f"{first} vs {second}"] = {
                kind: {f"{s:g}": {o: int(np.sum(m[(kinds == kind) & (strengths == s)])) for o, m in outcomes.items()}
                       for s in STRENGTHS}
                for kind in KINDS}
    report["windows"] = {"calibration": int(len(calibration)), "test": int(len(test)),
                         "test_signal_free": int(np.sum(kinds == "none")), "per_cell": per_cell}
    report["design"] = {"kinds": list(KINDS), "strengths": list(STRENGTHS), "per_cell": per_cell,
                        "max_drift": MAX_DRIFT, "margin": MARGIN, "window": list(test.shape[1:])}
    report["detectors"] = {name: describe(detect) for name, detect in detectors.items()}
    report["code"] = code_identity()
    report["key_sha256"] = hashlib.sha256(json.dumps([key[i] for i in ids], sort_keys=True).encode()).hexdigest()
    return report, {"ids": ids, "key": key, "scores": {n: s.tolist() for n, s in scores.items()}}


def table(report):
    """Recovery (signals) and flag rates (decoys) at each false-alarm rate, as Markdown."""
    names = list(report["recovery"])
    lines = []
    for label in report["recovery"][names[0]]:
        lines.append(f"\n### False-alarm rate {float(label):.1%} per window (thresholds from the calibration scan)\n")
        fa = ", ".join(f"{n} {report['false_alarms'][n][label]['flagged']}/{report['false_alarms'][n][label]['windows']}"
                       for n in names)
        lines.append(f"Measured on the test scan's signal-free windows: {fa}.\n")
        lines.append("| Class | Detector | " + " | ".join(f"SNR {s:g}" for s in STRENGTHS) + " |")
        lines.append("|---|---|" + "---:|" * len(STRENGTHS))
        for kind in SIGNALS + DECOYS:
            for n in names:
                cells = report["recovery"][n][label][kind]
                lines.append(f"| {kind}{' (decoy)' if kind in DECOYS else ''} | {n} | "
                             + " | ".join(f"{cells[f'{s:g}']['fraction']:.0%}" for s in STRENGTHS) + " |")
    return "\n".join(lines) + "\n"


def detectors_for(cache, learned, train_seed=0):
    """The baselines, plus (with learned) the learned detectors, trained on the train scan alone
    before anything is dealt. Returns (detectors, trained state or None)."""
    if not learned:
        return dict(DETECTORS), None
    from .learned import learned_detectors  # needs torch
    trained, state = learned_detectors(background.load(cache, "train"), seed=train_seed)
    return {**DETECTORS, **trained}, state


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, default=ROOT / "data/recovery", help="background windows (see background.py)")
    p.add_argument("--seed", type=int, default=0, help="the injection draw")
    p.add_argument("--per-cell", type=int, default=PER_CELL, help="injections per class and strength")
    p.add_argument("--output", type=Path, required=True, help="a new folder for report.json, key.json and report.md")
    p.add_argument("--learned", action="store_true", help="also train and score the learned detectors (needs torch)")
    p.add_argument("--train-seed", type=int, default=0, help="the learned detectors' training draw")
    args = p.parse_args(argv)
    if args.output.exists():
        p.error(f"{args.output} already exists")
    detectors, trained = detectors_for(args.cache, args.learned, args.train_seed)
    calibration = background.load(args.cache, "calibration")
    test = background.load(args.cache, "test")
    report, blinded = evaluate(calibration, test, args.seed, args.per_cell, detectors=detectors)
    manifest = json.loads(background.MANIFEST.read_text())
    report = {
        "format": "eiathome-recovery", "version": 1, "seed": args.seed,
        "limitation": LIMITATION,
        "backgrounds": {role: {k: manifest["scans"][role][k] for k in ("url", "tstart_mjd", "sha256", "shape")}
                        for role in (("train",) if trained else ()) + ("calibration", "test")},
        "strength": "optimal matched-filter SNR: the injected signal's norm, in noise sigma",
        "false_alarm_rates": list(FALSE_ALARM_RATES),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     **({"torch": str(__import__("torch").__version__)} if trained else {})},
        **report,
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    (args.output / "key.json").write_text(json.dumps(blinded, indent=1) + "\n")
    (args.output / "report.md").write_text(f"{LIMITATION}\n{table(report)}")
    if trained:  # the trained models, to rescore or inspect
        __import__("torch").save(trained, args.output / "learned.pt")
    print(LIMITATION)
    print(table(report))


if __name__ == "__main__":
    main()
