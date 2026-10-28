"""The Live Lab's recovery work units: their published windows, and a Python port to check them.

A browser can't download the test scan's 4,096 windows (51 MB), so the units use every 4th of
them, quantized to steps of 1/16 noise sigma: 1,024 windows in 3.2 MB, published in
web/data/recovery/. The rounding moves a value by at most 1/32 sigma, and no value in the test or
calibration scan is large enough to clip. The thresholds are set on all 4,096 calibration windows,
quantized the same way, and the spec (web/js/recovery.js) pins them with the file's SHA-256.

A unit is run.py's blinded comparison on those windows, with 18 injections per class and strength,
drawn with mulberry32 (web/js/rng.js) so that JavaScript and Python draw the same signals.
run_unit() ports the unit using inject.py's own draw() and render() and detectors.py's detectors;
tests/test_recovery_browser.py runs both and checks that they agree.

    python -m tracks.recovery.browser --cache data/recovery           # check the published files
    python -m tracks.recovery.browser --cache data/recovery --write   # (re)write them
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import background
from .detectors import DETECTORS
from .inject import KINDS, STRENGTHS, design, draw, render
from .run import FALSE_ALARM_RATES, describe, threshold

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = ROOT / "web" / "data" / "recovery"
DATA_FILE = "test.i8"
STRIDE = 4  # every 4th test window, so the published ones span the whole coarse channel
STEP = 1 / 16  # noise sigma per quantization step: int8 holds up to 127 / 16 = 7.94 sigma
PER_CELL = 18


def quantize(windows):
    q = np.round(np.asarray(windows, dtype=np.float32) / STEP)
    if np.abs(q).max() > 127:
        raise ValueError(f"A value beyond {127 * STEP} noise sigma would clip")
    return q.astype(np.int8)


def dequantize(q):
    return q.astype(np.float32) * np.float32(STEP)


def build(cache):
    """The published file's bytes and its index, from the verified background cache."""
    manifest = json.loads(background.MANIFEST.read_text())
    test = quantize(background.load(cache, "test")[::STRIDE])
    calibration = dequantize(quantize(background.load(cache, "calibration")))
    scan = lambda role: {k: manifest["scans"][role][k] for k in ("url", "tstart_mjd", "sha256", "shape")}
    index = {
        "format": "eiathome-recovery-windows", "version": 1,
        "file": DATA_FILE, "dtype": "int8", "step": STEP, "shape": list(test.shape),
        "sha256": hashlib.sha256(test.tobytes()).hexdigest(),
        "windows": f"Every {STRIDE}th window of the test scan, in noise sigma: value = byte x {STEP}.",
        "test": {**scan("test"), "stride": STRIDE},
        "calibration": {**scan("calibration"), "quantized": "as the published windows"},
        "thresholds": {name: {f"{rate:g}": threshold(detect(calibration), rate) for rate in FALSE_ALARM_RATES}
                       for name, detect in DETECTORS.items()},
        "detectors": {name: describe(detect) for name, detect in DETECTORS.items()},
    }
    return test.tobytes(), index


def published(folder=PUBLISHED):
    """The published windows (float32, noise sigma) and their index. Raises if the file doesn't
    match the index's SHA-256."""
    index = json.loads((Path(folder) / "index.json").read_text())
    raw = (Path(folder) / index["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != index["sha256"]:
        raise ValueError(f"{index['file']} doesn't match its SHA-256 in index.json")
    return dequantize(np.frombuffer(raw, dtype=np.int8).reshape(index["shape"])), index


class Mulberry32:
    """web/js/rng.js's makeRng, plus the two numpy Generator methods inject.draw() calls."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next(self):
        mul = lambda a, b: (a * b) & 0xFFFFFFFF  # Math.imul's low 32 bits
        self.state = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        t = mul(self.state ^ (self.state >> 15), self.state | 1)
        t ^= (t + mul(t ^ (t >> 7), t | 61)) & 0xFFFFFFFF
        return (t ^ (t >> 14)) / 4294967296

    def uniform(self, low=0.0, high=1.0):
        return low + (high - low) * self.next()

    def integers(self, low, high=None, size=None):
        """[low, high), or [0, low) without high, as rng.js's int()."""
        low, high = (0, low) if high is None else (low, high)
        one = lambda: low + math.floor(self.next() * (high - low))
        return one() if size is None else np.array([one() for _ in range(size)])

    def permutation(self, n):
        p = list(range(n))
        for i in range(n - 1, 0, -1):
            j = math.floor(self.next() * (i + 1))
            p[i], p[j] = p[j], p[i]
        return p


def derive_seed(seed, label):
    """web/js/rng.js's deriveSeed."""
    h = (2166136261 ^ seed) & 0xFFFFFFFF
    for ch in label:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h


def run_unit(unit, windows, thresholds, per_cell=PER_CELL):
    """Recovery unit `unit` on these background windows, as web/js/recovery.js runs it.
    Returns (results, key, scores): the card's results, each dealt window's spec, and the
    detectors' scores in dealing order."""
    n, frames, channels = windows.shape
    cells = design(per_cell, n)
    rng = Mulberry32(derive_seed(unit, "recovery"))
    assignment = rng.permutation(n)  # which design cell each background window gets
    dealt = np.empty_like(windows)
    key = []
    for position, i in enumerate(rng.permutation(n)):  # dealing order
        kind, strength = cells[assignment[i]]
        spec = draw(rng, kind, strength, frames, channels)
        dealt[position] = windows[i] + render(spec, frames, channels)
        key.append({**spec, "background": i})
    scores = {name: np.asarray(detect(dealt)) for name, detect in DETECTORS.items()}
    return tally(key, scores, thresholds), key, scores


def tally(key, scores, thresholds):
    """Counts per detector and false-alarm rate: false alarms among the signal-free windows,
    windows flagged per class and strength, and per class and strength those flagged by both."""
    kinds = np.array([s["kind"] for s in key])
    strengths = np.array([s["strength"] for s in key])
    results = {"signal_free": int(np.sum(kinds == "none")), "false_alarms": {}, "recovery": {}, "both": {}}
    for rate in thresholds[next(iter(thresholds))]:
        flags = {name: scores[name] > thresholds[name][rate] for name in scores}
        count = lambda f, kind, s=None: int(np.sum(f & (kinds == kind) & (True if s is None else strengths == s)))
        for name, f in flags.items():
            results["false_alarms"].setdefault(name, {})[rate] = count(f, "none")
            results["recovery"].setdefault(name, {})[rate] = {kind: [count(f, kind, s) for s in STRENGTHS] for kind in KINDS}
        both = np.logical_and.reduce(list(flags.values()))
        results["both"][rate] = {kind: [count(both, kind, s) for s in STRENGTHS] for kind in KINDS}
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, default=ROOT / "data/recovery", help="background windows (see background.py)")
    p.add_argument("--write", action="store_true", help="write the published files instead of checking them")
    args = p.parse_args(argv)
    data, index = build(args.cache)
    text = json.dumps(index, indent=1) + "\n"
    if args.write:
        PUBLISHED.mkdir(parents=True, exist_ok=True)
        (PUBLISHED / DATA_FILE).write_bytes(data)
        (PUBLISHED / "index.json").write_text(text)
        print(f"Wrote {len(data):,} bytes, sha256 {index['sha256'][:16]}; thresholds {index['thresholds']}")
    elif (PUBLISHED / DATA_FILE).read_bytes() != data or (PUBLISHED / "index.json").read_text() != text:
        raise SystemExit("The published files don't match a rebuild from the cache")
    else:
        print("The published files match a rebuild from the cache")


if __name__ == "__main__":
    main()
