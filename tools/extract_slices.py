#!/usr/bin/env python3
"""Cut Live Lab micro-slices out of a Breakthrough Listen filterbank HDF5 file.

A micro-slice is a small window of bandpass-flattened spectrogram (noise-sigma units) around
one detected narrowband signal: all time steps x 256 channels, about 16 KB as float32. The
web app reads them through web/data/slices/index.json.

Example (the Voyager 1 file from the Hello, Voyager notebook):

    python tools/extract_slices.py data/Voyager1.single_coarse.fine_res.h5

This writes web/data/slices/index.json plus one .bin file per signal. Pass several files to
extract them together, and --append to add to the slices already published. The detection steps
match tracks/rnn/hello_voyager.ipynb: mask the DC spike, flatten the bandpass, run a small
de-Doppler search, and measure each signal's drift by tracking it frame by frame.

Off-source scans of an on/off cadence (--off) give the control: for each signal of the nearest
on-source scan, the window where that signal would have drifted to by the off-source scan. Same
receiver, frequencies and processing, but nothing in the beam.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the bitshuffle filter used in Breakthrough Listen files)
import numpy as np

VERSION = "0.3.0"
SESSION_GAP_DAYS = 0.5  # as in web/js/splits.js
DRIFTS = np.arange(-4, 4.01, 0.25)  # de-Doppler search, channels per time step


def load(path):
    with h5py.File(path, "r") as f:
        dset = f["data"]  # (time, feed, frequency)
        header = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in dset.attrs.items()}
        data = dset[:, 0, :].astype(np.float32)
    return data, header


def mask_dc_spike(data, half_width=1):
    centre = data.shape[1] // 2
    lo, hi = centre - half_width, centre + half_width + 1
    fill = np.median(np.concatenate([data[:, lo - 8:lo], data[:, hi:hi + 8]], axis=1), axis=1)
    data[:, lo:hi] = fill[:, None]
    return data


def flatten_frames(data, block=1024):
    """Divide each frame by its bandpass (block medians) and convert to noise-sigma units."""
    n_block = data.shape[1] // block
    centres = (np.arange(n_block) + 0.5) * block
    channels = np.arange(data.shape[1])
    z = np.empty_like(data)
    for t, frame in enumerate(data):
        medians = np.median(frame[: n_block * block].reshape(n_block, block), axis=1)
        ratio = frame / np.interp(channels, centres, medians) - 1
        z[t] = ratio / (1.4826 * np.median(np.abs(ratio)))
    return z


def dedoppler(z, drifts):
    steps = np.arange(z.shape[0])
    best_snr = np.full(z.shape[1], -np.inf, dtype=np.float32)
    best_drift = np.zeros(z.shape[1], dtype=np.float32)
    for drift in drifts:
        shifts = np.round(drift * steps).astype(int)
        total = sum(np.roll(z[t], -s) for t, s in enumerate(shifts))
        snr = total / np.sqrt(z.shape[0])
        better = snr > best_snr
        best_snr[better] = snr[better]
        best_drift[better] = drift
    return best_snr, best_drift


def find_peaks(snr, n_max, threshold, exclusion):
    snr = snr.copy()
    peaks = []
    while len(peaks) < n_max:
        idx = int(np.argmax(snr))
        if snr[idx] < threshold:
            break
        peaks.append(idx)
        snr[max(0, idx - exclusion): idx + exclusion] = -np.inf
    return peaks


def track(z, start, drift, half=6):
    positions = []
    for t in range(z.shape[0]):
        guess = int(round(start + drift * t))
        segment = np.clip(z[t, guess - half: guess + half + 1], 0, None)
        if segment.sum() > 0:
            positions.append(guess - half + (segment * np.arange(segment.size)).sum() / segment.sum())
        else:
            positions.append(guess)
    return np.array(positions)


def scan_label(header):
    """MJD day and seconds after midnight, e.g. 59046-80036: unique per scan, unlike file names."""
    tstart = float(header["tstart"])
    day = int(tstart)
    return f"{day}-{round((tstart - day) * 86400):05d}"


def prepare(path):
    """A file's bandpass-flattened spectrogram, its header, and its de-Doppler search."""
    data, header = load(path)
    z = flatten_frames(mask_dc_spike(data))
    snr, best_drift = dedoppler(z, DRIFTS)
    return z, header, snr, best_drift


def make_entry(path, args, header, window, slice_id, family, group, source):
    source = {
        "file": path.name,
        "source_name": str(header.get("source_name", "")),
        "tstart_mjd": float(header.get("tstart", np.nan)),
        "fch1_mhz": float(header["fch1"]),
        "foff_hz": float(header["foff"]) * 1e6,
        "tsamp_s": float(header["tsamp"]),
        **source,
    }
    if args.source_base_url:
        source["url"] = args.source_base_url.rstrip("/") + "/" + path.name
    payload = np.ascontiguousarray(window, dtype="<f4").tobytes()
    return payload, {
        "id": slice_id,
        "family": family,
        "group": group,
        "file": f"{slice_id}.bin",
        "dtype": "float32-le",
        "shape": list(window.shape),
        "units": "noise sigma, bandpass-flattened",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source": source,
    }


def extract(path, args):
    """Slices for every signal found in one file."""
    z, header, snr, best_drift = prepare(path)
    n_t, n_chan = z.shape
    fch1, foff, tsamp = float(header["fch1"]), float(header["foff"]), float(header["tsamp"])
    foff_hz = foff * 1e6
    mid, edge = n_chan // 2, n_chan // 20

    snr[:edge] = -np.inf
    snr[-edge:] = -np.inf
    snr[mid - args.width: mid + args.width] = -np.inf
    peaks = find_peaks(snr, args.max_signals, args.threshold, exclusion=args.width)
    if not peaks:
        # Normal for the off-source scans of an on/off cadence: the transmitter isn't in the beam.
        print(f"  skipped {path.name}: no signal above SNR {args.threshold:g} (an off-source scan?)")
        return []

    label = scan_label(header)
    brightest = max(peaks, key=lambda p: snr[p])
    slices = []
    sidebands = 0
    for p in sorted(peaks, key=lambda p: fch1 + foff * p):
        positions = track(z, p, best_drift[p])
        slope, intercept = np.polyfit(np.arange(n_t), positions, 1)
        centre = intercept + slope * n_t / 2
        first = int(round(centre)) - args.width // 2
        window = z[:, first:first + args.width]
        if p == brightest:
            group, name = "carrier", "carrier"
        else:
            sidebands += 1
            group, name = "sideband", f"sideband-{sidebands}"
        slices.append(make_entry(path, args, header, window, f"{args.prefix}-{label}-{name}", "voyager", group, {
            "first_channel": first,
            "centre_freq_mhz": fch1 + foff * centre,
            "drift_channels_per_step": float(slope),
            "drift_hz_per_s": float(slope * foff_hz / tsamp),
            "snr": float(snr[p]),
        }))
    return slices


def extract_off(path, args, references):
    """Control windows from an off-source scan: for each signal of the nearest on-source scan in
    the same session, the window where that signal would have drifted to by this scan."""
    z, header, snr, _ = prepare(path)
    n_chan = z.shape[1]
    tstart = float(header["tstart"])
    fch1, foff = float(header["fch1"]), float(header["foff"])
    nearby = [e for e in references if abs(e["source"]["tstart_mjd"] - tstart) <= SESSION_GAP_DAYS]
    if not nearby:
        raise SystemExit(f"{path.name}: no on-source scan within 12 hours to take signal positions from.")
    # The nearest on-source scan; the earlier one if two are equally near.
    ref_start = min({e["source"]["tstart_mjd"] for e in nearby}, key=lambda t: (abs(t - tstart), t))

    label = scan_label(header)
    slices = []
    for ref in sorted((e for e in nearby if e["source"]["tstart_mjd"] == ref_start), key=lambda e: e["id"]):
        src = ref["source"]
        name = ref["id"].split(f"-{scan_label({'tstart': ref_start})}-", 1)[1]
        # Both centres are at the scan's middle time step, so the scans' start times give the gap.
        freq = src["centre_freq_mhz"] + src["drift_hz_per_s"] * (tstart - ref_start) * 86400 / 1e6
        first = int(round((freq - fch1) / foff)) - args.width // 2
        if first < 0 or first + args.width > n_chan:
            raise SystemExit(f"{path.name}: the window for {ref['id']} falls outside the file.")
        peak = float(snr[first:first + args.width].max())
        if peak >= args.threshold:
            print(f"  warning: {path.name} has a signal of SNR {peak:.1f} where {ref['id']} would be (a sidelobe or RFI?)")
        slices.append(make_entry(path, args, header, z[:, first:first + args.width], f"{args.prefix}-{label}-off-{name}", "offsource", f"{ref['group']} window", {
            "first_channel": first,
            "centre_freq_mhz": freq,
            "window_of": ref["id"],
            # The strongest de-Doppler peak in the window: noise level when nothing is in the beam.
            "snr": peak,
        }))
    return slices


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5", type=Path, nargs="*", help="Breakthrough Listen filterbank HDF5 file(s), on-source")
    parser.add_argument("--off", type=Path, nargs="+", default=[], help="off-source scans of the same cadence, for the control windows")
    parser.add_argument("--out", type=Path, default=Path("web/data/slices"), help="output folder")
    parser.add_argument("--append", action="store_true", help="add to the manifest already in --out instead of replacing it")
    parser.add_argument("--source-base-url", help="where the input files can be downloaded (recorded for each slice)")
    parser.add_argument("--width", type=int, default=256, help="channels per slice")
    parser.add_argument("--max-signals", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=15, help="minimum de-Doppler SNR")
    parser.add_argument("--prefix", default="voyager1", help="prefix for slice ids and file names")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "index.json"
    entries = []
    if args.append and manifest_path.exists():
        entries = json.loads(manifest_path.read_text())["slices"]
    known_hashes = {e["sha256"] for e in entries}
    known_ids = {e["id"] for e in entries}

    added = []

    def add(payload, entry):
        if entry["sha256"] in known_hashes:
            print(f"  skipped {entry['id']}: identical data is already in the manifest")
            return
        if entry["id"] in known_ids:
            raise SystemExit(f"{entry['id']} is already in the manifest with different data.")
        (args.out / entry["file"]).write_bytes(payload)
        entries.append(entry)
        added.append(entry)
        known_hashes.add(entry["sha256"])
        known_ids.add(entry["id"])

    for path in args.h5:
        for payload, entry in extract(path, args):
            add(payload, entry)
    # On-source signals from this run and, with --append, from the manifest.
    references = [e for e in entries if e["family"] == "voyager" and e["source"].get("tstart_mjd") is not None]
    for path in args.off:
        for payload, entry in extract_off(path, args, references):
            add(payload, entry)


    manifest = {
        "format": "eiathome-slices",
        "version": 1,
        "extracted_by": f"tools/extract_slices.py {VERSION}",
        "slices": sorted(entries, key=lambda e: e["id"]),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Added {len(added)} slices to {args.out}/ ({len(entries)} in total)")
    for e in added:
        s = e["source"]
        drift = f"drift {s['drift_hz_per_s']:+.3f} Hz/s" if "drift_hz_per_s" in s else f"window of {s['window_of']}"
        print(f"  {e['id']:<38} {s['centre_freq_mhz']:.6f} MHz  SNR {s['snr']:6.1f}  {drift}")


if __name__ == "__main__":
    main()
