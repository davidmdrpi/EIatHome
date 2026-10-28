"""Milestone 4: the recovery benchmark across two observing sessions.

The same coarse channel (8418.46 to 8421.39 MHz), in the same product, from two sessions on the
Green Bank Telescope four years apart. The product is Breakthrough Listen's
high-frequency-resolution files: 16 frames of 18.25 s, 2.79 Hz channels.

- **2020** (MJD 59046): off-source scans of the Voyager 1 cadence. Thresholds come from scan
  0014 and signals go into scan 0016, as in milestone 1.
- **2016** (MJD 57650): one on-source scan of Voyager 1, the file turboSETI's tutorial uses.
  Windows within 180 kHz of Voyager's carrier are left out: that covers the sidebands of its
  22.5 kHz telemetry subcarrier and their harmonics. The other windows alternate between
  calibration and test.

Thresholds set on each session's calibration windows are applied to each session's test windows,
with the same injections. The diagonal is the benchmark within a session; the rest shows whether
a threshold, and recovery at that threshold, carry over to a session it wasn't set on.

    python -m tracks.recovery.sessions build --cache data/recovery            # fetch, cut, check against sessions.json
    python -m tracks.recovery.sessions build --cache data/recovery --record   # (re)write sessions.json
    python -m tracks.recovery.sessions run --cache data/recovery --output runs/sessions

The 2016 file (50 MB) comes from this repository's v0.1 release, checked against its SHA-256:
the original is served over plain HTTP only. The 2020 files are read with HTTPS range requests
(about 50 MB each).
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import urllib.request

import numpy as np

from . import background, run
from .inject import DECOYS, KINDS, SIGNALS, STRENGTHS

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("sessions.json")
BASE_2020 = "https://bldata.berkeley.edu/voyager_2020/single_coarse_channel"
SCANS_2020 = {
    "calibration": f"{BASE_2020}/single_coarse_guppi_59046_80989_DIAG_VOYAGER-1_0014.rawspec.0000.h5",
    "test": f"{BASE_2020}/single_coarse_guppi_59046_81628_DIAG_VOYAGER-1_0016.rawspec.0000.h5",
}
VOYAGER_2016 = {
    "url": "https://github.com/davidmdrpi/EIatHome/releases/download/v0.1/Voyager1.single_coarse.fine_res.h5",
    "original": "http://blpd0.ssl.berkeley.edu/Voyager_data/Voyager1.single_coarse.fine_res.h5",
    "sha256": "c9a9a54f4140e3754ffb2455fae4eeb2eb70c8207123116ee953e4fce15c36ac",
}
SESSIONS = ("2020", "2016")
ROLES = tuple(f"{session}-{part}" for session in SESSIONS for part in ("calibration", "test"))
WINDOWS = 4096  # per role
GUARD_KHZ = 180  # either side of Voyager's carrier: past the 7th harmonic of its 22.5 kHz subcarrier
SEEDS = {"2020-calibration": 1, "2020-test": 2, "2016": 3}
LIMITATION = ("Two sessions of one receiver and one coarse channel. The 2020 calibration and test windows come from "
              "off-source scans 11 minutes apart; the 2016 ones alternate along one on-source scan, away from "
              "Voyager's lines. Intervals are per window.")


def fetch_2016(cache):
    """The 2016 file in the cache, downloaded once. Raises if its SHA-256 isn't the expected one."""
    path = Path(cache) / Path(VOYAGER_2016["url"]).name
    if not path.exists():
        part = path.with_suffix(".part")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(VOYAGER_2016["url"], timeout=120) as response, open(part, "wb") as out:
                    while block := response.read(1 << 20):
                        out.write(block)
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        part.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != VOYAGER_2016["sha256"]:
        raise ValueError(f"{path} has SHA-256 {digest}, not the expected {VOYAGER_2016['sha256']}: delete it and rebuild")
    return path


def flattened(source):
    """A file's spectrogram in noise sigma, by the published slices' pipeline, and its header."""
    x = background.extractor()
    data, header = x.load(source)
    return x.flatten_frames(x.mask_dc_spike(data)), header


def carrier(z):
    """The channel where the strongest straight track starts: Voyager's carrier, on-source."""
    x = background.extractor()
    snr, _ = x.dedoppler(z, x.drift_trials(z.shape[0]))
    return int(np.argmax(snr))


def describe(header, windows, seed, first):
    return {
        "source_name": str(header.get("source_name", "")),
        "tstart_mjd": float(header["tstart"]),
        "tsamp_s": float(header["tsamp"]),
        "foff_hz": float(header["foff"]) * 1e6,
        "fch1_mhz": float(header["fch1"]),
        "shape": list(windows.shape),
        "seed": seed,
        "first_channels": first.tolist(),
        "sha256": background.digest(windows),
    }


def build(cache, sessions=SESSIONS):
    """{role: (windows, manifest entry)} for the given sessions."""
    out = {}
    if "2020" in sessions:
        for part, url in SCANS_2020.items():
            role = f"2020-{part}"
            z, header = flattened(url)
            first, windows = background.cut(z, WINDOWS, SEEDS[role])
            out[role] = windows, {"url": url, **describe(header, windows, SEEDS[role], first)}
    if "2016" in sessions:
        path = fetch_2016(cache)
        z, header = flattened(path)
        centre = carrier(z)
        guard = int(round(GUARD_KHZ * 1e3 / abs(float(header["foff"]) * 1e6)))
        first, windows = background.cut(z, 2 * WINDOWS, SEEDS["2016"], exclude=[(centre - guard, centre + guard + 1)])
        source = {"url": VOYAGER_2016["url"], "original_url": VOYAGER_2016["original"], "file_sha256": VOYAGER_2016["sha256"],
                  "carrier_channel": centre, "carrier_mhz": float(header["fch1"]) + float(header["foff"]) * centre,
                  "guard_channels": guard}
        for part, half in (("calibration", slice(0, None, 2)), ("test", slice(1, None, 2))):  # alternating along the band
            out[f"2016-{part}"] = windows[half], {**source, "split": f"every other window of {len(windows):,}, in channel order, from the {'first' if half.start == 0 else 'second'}",
                                                  **describe(header, windows[half], SEEDS["2016"], first[half])}
    return out


def load(cache, role):
    """The verified windows for a role (see background.load)."""
    try:
        return background.load(cache, role, manifest=MANIFEST)
    except FileNotFoundError:
        raise FileNotFoundError(f"{background.cache_path(cache, role)} is missing: run python -m tracks.recovery.sessions build --cache {cache}") from None


def evaluate_sessions(windows, seed=0, per_cell=run.PER_CELL, detectors=None):
    """Thresholds from each session's calibration windows, applied to each session's test windows
    with the same injections: the dealer's draws don't depend on the windows, so each injection
    has a twin in the other session. windows: {role: array}.

    Returns (pairs, across): pairs maps "thresholds S1, test S2" to run.evaluate's report. across
    compares the twins (see across_sessions) at each session's own thresholds, which is the whole
    difference between the sessions, and at the first session's thresholds for both, which leaves
    only the difference between their windows."""
    detectors = detectors or dict(run.DETECTORS)
    pairs, outcomes = {}, {}
    for cal in SESSIONS:
        for test in SESSIONS:
            report, blinded = run.evaluate(windows[f"{cal}-calibration"], windows[f"{test}-test"], seed, per_cell, detectors)
            pairs[f"thresholds {cal}, test {test}"] = report
            specs = [blinded["key"][i] for i in blinded["ids"]]
            flags = {name: {rate: np.asarray(blinded["scores"][name]) > cut for rate, cut in report["thresholds"][name].items()}
                     for name in detectors}
            outcomes[cal, test] = specs, flags
    first, second = SESSIONS
    return pairs, {
        "own thresholds": across_sessions({first: outcomes[first, first], second: outcomes[second, second]}),
        f"{first} thresholds": across_sessions({first: outcomes[first, first], second: outcomes[first, second]}),
    }


def mcnemar(only_first, only_second):
    """Exact two-sided McNemar p-value: how likely so uneven a split of the discordant twins is
    if recovery doesn't depend on the session."""
    n = only_first + only_second
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(only_first, only_second) + 1)) / 2 ** n) if n else 1.0


def across_sessions(outcomes):
    """Each injection against its twin in the other session, both at their own session's
    thresholds: per detector and rate, how many were flagged in both sessions, in one only, or in
    neither, per class and strength, over each class, and over all signals and all decoys, with
    McNemar p-values for the totals. outcomes: {session: (specs in dealing order, flags)}."""
    (first, (specs, a)), (second, (twins, b)) = outcomes.items()
    if specs != twins:
        raise ValueError("The sessions weren't dealt the same injections")
    kinds = np.array([s["kind"] for s in specs])
    strengths = np.array([s["strength"] for s in specs])

    def split(x, y, mask, test=False):
        out = {"both": int(np.sum(x & y & mask)), f"{first} only": int(np.sum(x & ~y & mask)),
               f"{second} only": int(np.sum(~x & y & mask)), "neither": int(np.sum(~x & ~y & mask))}
        return {**out, "p": mcnemar(out[f"{first} only"], out[f"{second} only"])} if test else out

    result = {}
    for name in a:
        for rate in a[name]:
            x, y = a[name][rate], b[name][rate]
            result.setdefault(name, {})[rate] = {
                "signals": split(x, y, np.isin(kinds, SIGNALS), test=True),
                "decoys": split(x, y, np.isin(kinds, DECOYS), test=True),
                "classes": {kind: split(x, y, kinds == kind, test=True) for kind in KINDS},
                "cells": {kind: {f"{s:g}": split(x, y, (kinds == kind) & (strengths == s)) for s in STRENGTHS} for kind in KINDS},
            }
    return result


def table(pairs, across):
    """False alarms for every pairing of thresholds and test windows, the twins compared across
    sessions, then each pairing's recovery, as Markdown."""
    first = next(iter(pairs.values()))
    names, rates = list(first["false_alarms"]), list(first["false_alarms"][next(iter(first["false_alarms"]))])
    lines = ["\n### False alarms on the test windows left empty\n",
             "| Detector | Rate | Thresholds from | " + " | ".join(f"Test {s}" for s in SESSIONS) + " |",
             "|---|---|---|" + "---:|" * len(SESSIONS)]
    for name in names:
        for rate in rates:
            for cal in SESSIONS:
                cells = []
                for test in SESSIONS:
                    fa = pairs[f"thresholds {cal}, test {test}"]["false_alarms"][name][rate]
                    cells.append(f"{fa['flagged']} of {fa['windows']:,} ({fa['fraction']:.1%})")
                lines.append(f"| {name} | {float(rate):.1%} | {cal} | " + " | ".join(cells) + " |")
    lines.append("\n### Thresholds\n")
    lines.append("| Detector | Rate | " + " | ".join(SESSIONS) + " |")
    lines.append("|---|---|" + "---:|" * len(SESSIONS))
    for name in names:
        for rate in rates:
            lines.append(f"| {name} | {float(rate):.1%} | " + " | ".join(f"{pairs[f'thresholds {s}, test {s}']['thresholds'][name][rate]:.3f}" for s in SESSIONS) + " |")
    for variant, compared in across.items():
        lines.append(f"\n### The same injections in both sessions, at {'each session' + chr(39) + 's own' if variant == 'own thresholds' else 'the ' + variant.split()[0]} thresholds\n")
        lines.append(f"| Detector | Rate | Injections | Flagged in {SESSIONS[0]} | Flagged in {SESSIONS[1]} | {SESSIONS[0]} only | {SESSIONS[1]} only | McNemar p |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for name in names:
            for rate in rates:
                groups = compared[name][rate]
                rows = [("all signals", groups["signals"]), ("all decoys", groups["decoys"])]
                if variant == "own thresholds":
                    rows += list(groups["classes"].items())
                for label, c in rows:
                    n = sum(v for k, v in c.items() if k != "p")
                    found = [c["both"] + c[f"{s} only"] for s in SESSIONS]
                    lines.append(f"| {name} | {float(rate):.1%} | {label} ({n:,}) | {found[0]:,} | {found[1]:,} | "
                                 f"{c[f'{SESSIONS[0]} only']:,} | {c[f'{SESSIONS[1]} only']:,} | {c['p']:.2g} |")
    for key, report in pairs.items():
        lines.append(f"\n## Recovery: {key}\n{run.table(report)}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["build", "run"])
    p.add_argument("--cache", type=Path, default=ROOT / "data/recovery", help="where windows (and the 2016 file) are kept")
    p.add_argument("--sessions", nargs="+", choices=SESSIONS, default=list(SESSIONS), help="build: which sessions")
    p.add_argument("--record", action="store_true", help="build: write sessions.json instead of checking against it")
    p.add_argument("--seed", type=int, default=0, help="run: the injection draw")
    p.add_argument("--per-cell", type=int, default=run.PER_CELL, help="run: injections per class and strength")
    p.add_argument("--output", type=Path, help="run: a new folder for report.json and report.md")
    args = p.parse_args(argv)

    if args.command == "build":
        args.cache.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {
            "format": "eiathome-recovery-sessions", "version": 1,
            "pipeline": "tools/extract_slices.py: DC spike masked, bandpass flattened (noise sigma)",
            "product": "high frequency resolution: 16 frames of 18.25 s, 2.79 Hz channels, one coarse channel",
            "window": {"frames": 16, "channels": background.WIDTH, "edge_fraction": background.EDGE_FRACTION,
                       "dc_guard": background.DC_GUARD, "voyager_guard_khz": GUARD_KHZ},
            "scans": {},
        }
        for role, (windows, entry) in build(args.cache, args.sessions).items():
            np.save(background.cache_path(args.cache, role), windows, allow_pickle=False)
            if args.record:
                manifest["scans"][role] = entry
                print(f"{role}: recorded {entry['shape']} windows, sha256 {entry['sha256'][:16]}")
            elif manifest["scans"].get(role) != entry:
                raise SystemExit(f"{role}: the rebuilt windows don't match {MANIFEST.name}")
            else:
                print(f"{role}: {entry['shape']} windows match the manifest")
        if args.record:
            MANIFEST.write_text(background.manifest_text(manifest))
        return

    if args.output is None:
        p.error("run needs --output")
    if args.output.exists():
        p.error(f"{args.output} already exists")
    pairs, across = evaluate_sessions({role: load(args.cache, role) for role in ROLES}, args.seed, args.per_cell)
    manifest = json.loads(MANIFEST.read_text())
    report = {
        "format": "eiathome-recovery-sessions-report", "version": 1, "seed": args.seed, "limitation": LIMITATION,
        "sessions": {role: {k: v for k, v in manifest["scans"][role].items() if k != "first_channels"} for role in ROLES},
        "strength": "optimal matched-filter SNR: the injected signal's norm, in noise sigma",
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "pairs": pairs,
        "across_sessions": across,
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    text = f"{LIMITATION}\n{table(pairs, across)}"
    (args.output / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
