"""Signal-free background windows from real off-source scans, for injection tests.

Each window is 49 frames x 64 channels of bandpass-flattened spectrogram (noise-sigma units),
cut from coarse channel 6 of an off-source scan of the July 2020 Voyager cadence (the 6-second
product), by the same pipeline as the published slices. Windows don't overlap and skip the
coarse channel's edges and its DC region. Real interference stays in: false alarms on it count.

The windows aren't committed. backgrounds.json records where each comes from and the SHA-256 of
the result, so a rebuild can be checked:

    python -m tracks.recovery.background --cache data/recovery            # build and verify
    python -m tracks.recovery.background --cache data/recovery --record   # (re)write the manifest

Building reads about 170 MB per scan over HTTPS (tools/extract_slices.py, range requests).
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("backgrounds.json")
BASE_URL = "https://bldata.berkeley.edu/voyager_2020/6_sec_integration"
# One scan per role: model training (later milestones), threshold calibration, and the test.
SCANS = {
    "train": f"{BASE_URL}/blc23_guppi_59046_80354_DIAG_VOYAGER-1_0012.rawspec.0000.h5",
    "calibration": f"{BASE_URL}/blc23_guppi_59046_80989_DIAG_VOYAGER-1_0014.rawspec.0000.h5",
    "test": f"{BASE_URL}/blc23_guppi_59046_81628_DIAG_VOYAGER-1_0016.rawspec.0000.h5",
}
ROLES = tuple(SCANS)
COARSE_AT_MHZ = 8419.54  # Voyager's coarse channel; the off-source scans point 1 degree away
WIDTH = 64
WINDOWS = 4096
EDGE_FRACTION = 0.05  # each side: the coarse channel's bandpass rolls off there
DC_GUARD = 256  # channels either side of the coarse channel's centre (the DC spike)


def extractor():
    spec = importlib.util.spec_from_file_location("extract_slices", ROOT / "tools/extract_slices.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def slots(n_chan, width=WIDTH):
    """First channels of the non-overlapping windows that avoid the edges and the DC region."""
    edge = int(n_chan * EDGE_FRACTION)
    mid = n_chan // 2
    starts = np.arange(edge, n_chan - edge - width + 1, width)
    return starts[(starts + width <= mid - DC_GUARD) | (starts >= mid + DC_GUARD)]


def cut(z, count, seed, width=WIDTH):
    """count windows at seeded, sorted, non-overlapping positions of a flattened spectrogram."""
    available = slots(z.shape[1], width)
    if count > len(available):
        raise ValueError(f"Only {len(available)} windows fit; asked for {count}")
    first = np.sort(np.random.default_rng(seed).choice(available, count, replace=False))
    windows = np.stack([z[:, f:f + width] for f in first]).astype("<f4")
    return first, windows


def digest(windows):
    return hashlib.sha256(np.ascontiguousarray(windows, dtype="<f4").tobytes()).hexdigest()


def build(role, count=WINDOWS, seed=None):
    """Windows and their provenance for one role, from its scan."""
    x = extractor()
    seed = ROLES.index(role) if seed is None else seed
    data, header = x.load(SCANS[role], COARSE_AT_MHZ)
    z = x.flatten_frames(x.mask_dc_spike(data))
    first, windows = cut(z, count, seed)
    entry = {
        "url": SCANS[role],
        "source_name": str(header.get("source_name", "")),
        "tstart_mjd": float(header["tstart"]),
        "tsamp_s": float(header["tsamp"]),
        "foff_hz": float(header["foff"]) * 1e6,
        "coarse_channel": int(header["coarse_channel"]),
        "coarse_fch1_mhz": float(header["fch1"]),
        "shape": list(windows.shape),
        "seed": seed,
        "first_channels": first.tolist(),
        "sha256": digest(windows),
    }
    return windows, entry


def cache_path(cache, role):
    return Path(cache) / f"background-{role}.npy"


def load(cache, role, manifest=MANIFEST):
    """The verified windows for a role. Raises if they're missing or don't match the manifest."""
    path = cache_path(cache, role)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run python -m tracks.recovery.background --cache {cache}")
    windows = np.load(path, allow_pickle=False)
    expected = json.loads(Path(manifest).read_text())["scans"][role]
    if list(windows.shape) != expected["shape"] or digest(windows) != expected["sha256"]:
        raise ValueError(f"{path} doesn't match {Path(manifest).name}: rebuild it")
    return windows


def manifest_text(manifest):
    """The manifest as written: indented, with each list of window positions on one line."""
    text = json.dumps(manifest, indent=1)
    return re.sub(r"\[\n\s+(-?\d[-\d,\s]*?)\n\s*\]", lambda m: "[" + re.sub(r",\s+", ",", m.group(1)) + "]", text) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, default=ROOT / "data/recovery", help="where the windows are written")
    p.add_argument("--roles", nargs="+", choices=ROLES, default=list(ROLES))
    p.add_argument("--record", action="store_true", help="write the manifest instead of checking against it")
    args = p.parse_args(argv)
    args.cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {
        "format": "eiathome-recovery-backgrounds", "version": 1,
        "pipeline": "tools/extract_slices.py: DC spike masked, bandpass flattened (noise sigma)",
        "window": {"frames": None, "channels": WIDTH, "edge_fraction": EDGE_FRACTION, "dc_guard": DC_GUARD},
        "scans": {},
    }
    for role in args.roles:
        windows, entry = build(role)
        np.save(cache_path(args.cache, role), windows, allow_pickle=False)
        manifest["window"]["frames"] = entry["shape"][1]
        if args.record:
            manifest["scans"][role] = entry
            print(f"{role}: recorded {entry['shape']} windows, sha256 {entry['sha256'][:16]}")
        elif manifest["scans"].get(role) != entry:
            raise SystemExit(f"{role}: the rebuilt windows don't match {MANIFEST.name}")
        else:
            print(f"{role}: {entry['shape']} windows match the manifest")
    if args.record:
        MANIFEST.write_text(manifest_text(manifest))


if __name__ == "__main__":
    main()
