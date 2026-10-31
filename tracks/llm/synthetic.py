"""Synthetic signal families for the signal language model (issue #20, milestone 3): the Rosetta
control, English text sent as 16-tone FSK, ported from the Live Lab's generator
(web/js/signals.js) into the real slices' units (noise sigma, bandpass-flattened).

Each frame carries one 4-bit symbol: the text's UTF-8 bytes, high nibble then low nibble, no
compression or scrambling. Symbol s is a narrow line at channel 8 + 3s of 64, at one flux per
slice drawn log-uniformly from 30 to 300, on unit Gaussian noise, as in the browser.

The family mirrors the real ones: three scans of three 49-frame slices, consecutive stretches of
the text, so training, validation and test hold different text. The slices are generated once
and published with their SHA-256s in a manifest that also lists the real slices, like any other
data: every run tokenizes the same bytes, wherever it runs. Regenerating them matches to float32
rounding (a maths library can move a sample by one rounding), so the check compares values.

    python -m tracks.llm.synthetic build           # regenerate and compare with the committed slices
    python -m tracks.llm.synthetic build --write   # (re)write them and the manifest

Milestone 3b (`--set moe-v2`) adds a second, larger set for testing whether better-structured
experts change the mixture's result: 18 Rosetta slices (six per scan) and the Live Lab's RFI
sweep, a line sweeping fast across the window and wrapping (nine slices), listed with the real
slices in index-moe-v2.json. The first set and its manifest are unchanged.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SLICES = Path(__file__).resolve().parent / "data/slices"
REAL_MANIFEST = SLICES / "index.json"
MANIFEST = SLICES / "index-rosetta.json"  # the real slices and the Rosetta family
MOE_MANIFEST = SLICES / "index-moe-v2.json"  # milestone 3b: the real slices, longer Rosetta, an RFI sweep

# The corpus is the Live Lab's, pinned the same way (ROSETTA_CORPUS in web/js/workunits.js).
CORPUS_ID = "en-gettysburg-bliss+us-preamble/v1"
CORPUS_SHA256 = "e2369a7f60ca396a402206a1f9b8d5d8bf2c375069debcc8527422552cc8e854"
CORPUS = """Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.

Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.

But, in a larger sense, we can not dedicate -- we can not consecrate -- we can not hallow -- this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us -- that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion -- that we here highly resolve that these dead shall not have died in vain -- that this nation, under God, shall have a new birth of freedom -- and that government of the people, by the people, for the people, shall not perish from the earth.

We the People of the United States, in Order to form a more perfect Union, establish Justice, insure domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity, do ordain and establish this Constitution for the United States of America."""

FRAMES, CHANNELS = 49, 64  # the real slices' length, and the window tracks.llm reads
SCANS, PER_SCAN = 3, 3  # like the real families: three scans of three slices
FLUX = (30.0, 300.0)  # a slice's line power per frame, log-uniform, as in the browser
LINE_WIDTH = 0.6  # channels (Gaussian sigma)
SEED = 0
# A clearly synthetic epoch: the scans are 0.01 days apart, one session, far from any real one.
SCAN_START_MJD = (0.0, 0.01, 0.02)
TOLERANCE = 1e-5  # how far a regenerated sample may be from the committed one
# Milestone 3b. The RFI sweep, as in the browser: rate 8 to 20 channels per step either way,
# power 20 to 1000 log-uniform, smeared over 16 sub-steps per frame. Another synthetic session.
MOE_PER_SCAN = {"rosetta": 6, "sweep": 3}
MOE_SEEDS = {"rosetta": 1, "sweep": 2}
SWEEP_RATE, SWEEP_FLUX, SUBSTEPS = (8.0, 20.0), (20.0, 1000.0), 16
SWEEP_START_MJD = (1.0, 1.01, 1.02)


def text_symbols(text=CORPUS):
    """UTF-8 bytes -> 4-bit symbols, high nibble then low (textToSymbols in web/js/signals.js)."""
    data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    return np.stack([data >> 4, data & 15], axis=1).reshape(-1).astype(np.int64)


def fsk_channel(symbol):
    return 8 + 3 * symbol  # tones 3 channels apart, channels 8..53


def fsk_frames(symbols, flux, noise):
    """Noise plus one line per frame at its symbol's channel, with total power `flux`."""
    x = np.array(noise, dtype=np.float64)
    channels = np.arange(CHANNELS)
    for t, symbol in enumerate(symbols):
        centre = fsk_channel(int(symbol))
        lo, hi = max(0, math.floor(centre - 4 * LINE_WIDTH)), min(CHANNELS - 1, math.ceil(centre + 4 * LINE_WIDTH))
        near = channels[lo:hi + 1]
        x[t, lo:hi + 1] += flux * np.exp(-0.5 * ((near - centre) / LINE_WIDTH) ** 2) / (LINE_WIDTH * math.sqrt(2 * math.pi))
    return x.astype(np.float32)


def log_uniform(rng, lo, hi):
    return math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * rng.random())


def sweep_frames(rate, flux, start, noise):
    """Noise plus a line at channel (start + rate·u) mod 64 at time u, smeared over its motion
    within each frame (addLine in web/js/signals.js, with 16 sub-steps)."""
    x = np.array(noise, dtype=np.float64)
    channels = np.arange(CHANNELS)
    norm = SUBSTEPS * LINE_WIDTH * math.sqrt(2 * math.pi)
    for t in range(len(x)):
        for sub in range(SUBSTEPS):
            centre = (start + rate * (t + sub / SUBSTEPS)) % CHANNELS
            lo, hi = max(0, math.floor(centre - 4 * LINE_WIDTH)), min(CHANNELS - 1, math.ceil(centre + 4 * LINE_WIDTH))
            near = channels[lo:hi + 1]
            x[t, lo:hi + 1] += flux * np.exp(-0.5 * ((near - centre) / LINE_WIDTH) ** 2) / norm
    return x.astype(np.float32)


def generate():
    """The Rosetta slices: [(manifest entry, float32 frames)], in text order."""
    return rosetta_slices("rosetta-fsk", PER_SCAN, SEED, SCAN_START_MJD)


def generate_moe():
    """Milestone 3b's synthetic slices: longer Rosetta, then the RFI sweep."""
    return (rosetta_slices("rosetta2-fsk", MOE_PER_SCAN["rosetta"], MOE_SEEDS["rosetta"], SCAN_START_MJD)
            + sweep_slices(MOE_PER_SCAN["sweep"], MOE_SEEDS["sweep"], SWEEP_START_MJD))


def rosetta_slices(prefix, per_scan, seed, scan_start_mjd):
    """Three scans of `per_scan` Rosetta slices, consecutive stretches of the text."""
    if hashlib.sha256(CORPUS.encode()).hexdigest() != CORPUS_SHA256:
        raise ValueError("The Rosetta corpus doesn't match its pinned SHA-256")
    symbols = text_symbols()
    rng = np.random.default_rng(seed)
    out = []
    for k in range(SCANS * per_scan):
        window = symbols[k * FRAMES:(k + 1) * FRAMES]
        flux = log_uniform(rng, *FLUX)
        frames = fsk_frames(window, flux, rng.standard_normal((FRAMES, CHANNELS)))
        scan, part = divmod(k, per_scan)
        name = f"{prefix}-scan{scan + 1}-part{part + 1}"
        entry = {
            "id": name, "family": "rosetta", "group": "fsk", "file": f"{name}.bin", "dtype": "float32-le",
            "shape": [FRAMES, CHANNELS], "units": "noise sigma",
            "sha256": hashlib.sha256(frames.astype("<f4").tobytes()).hexdigest(),
            "source": {
                "product": "synthetic", "generator": "tracks/llm/synthetic.py",
                "modulation": "16-tone FSK, one 4-bit symbol per frame at channel 8 + 3 x symbol",
                "corpus": CORPUS_ID, "corpus_sha256": CORPUS_SHA256, "symbols": [k * FRAMES, (k + 1) * FRAMES],
                "flux": flux, "line_width_channels": LINE_WIDTH, "seed": seed, "slice": k,
                "tstart_mjd": scan_start_mjd[scan],
            },
        }
        out.append((entry, frames))
    return out


def sweep_slices(per_scan, seed, scan_start_mjd):
    """Three scans of `per_scan` RFI sweeps, each with its own rate, direction, power and start."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(SCANS * per_scan):
        rate = (SWEEP_RATE[0] + (SWEEP_RATE[1] - SWEEP_RATE[0]) * rng.random()) * (-1 if rng.random() < 0.5 else 1)
        flux = log_uniform(rng, *SWEEP_FLUX)
        start = CHANNELS * rng.random()
        frames = sweep_frames(rate, flux, start, rng.standard_normal((FRAMES, CHANNELS)))
        scan, part = divmod(k, per_scan)
        name = f"rfi-sweep-scan{scan + 1}-part{part + 1}"
        out.append(({
            "id": name, "family": "sweep", "group": "rfi", "file": f"{name}.bin", "dtype": "float32-le",
            "shape": [FRAMES, CHANNELS], "units": "noise sigma",
            "sha256": hashlib.sha256(frames.astype("<f4").tobytes()).hexdigest(),
            "source": {
                "product": "synthetic", "generator": "tracks/llm/synthetic.py",
                "modulation": "RFI sweep: a line moving linearly across the window and wrapping, smeared within each frame",
                "rate_channels_per_step": rate, "flux": flux, "start_channel": start,
                "line_width_channels": LINE_WIDTH, "seed": seed, "slice": k, "tstart_mjd": scan_start_mjd[scan],
            },
        }, frames))
    return out


def symbols_of(entry):
    """The symbols a Rosetta slice carries, one per frame."""
    start, end = entry["source"]["symbols"]
    return text_symbols()[start:end]


def manifest(entries):
    real = json.loads(REAL_MANIFEST.read_text())
    return {**{k: v for k, v in real.items() if k != "slices"}, "synthetic_by": "tracks/llm/synthetic.py",
            "slices": real["slices"] + entries}


SETS = {"m3": (generate, MANIFEST), "moe-v2": (generate_moe, MOE_MANIFEST)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["build"])
    p.add_argument("--set", choices=list(SETS), default="m3", help="m3: the Rosetta set; moe-v2: milestone 3b's")
    p.add_argument("--write", action="store_true", help="write the slices and the manifest instead of checking them")
    args = p.parse_args(argv)
    make, path = SETS[args.set]
    generated = make()
    if args.write:
        for entry, frames in generated:
            (SLICES / entry["file"]).write_bytes(frames.astype("<f4").tobytes())
        path.write_text(json.dumps(manifest([e for e, _ in generated]), indent=1) + "\n")
        print(f"wrote {len(generated)} synthetic slices and {path.name}")
        return
    committed = {e["id"]: e for e in json.loads(path.read_text())["slices"] if e["source"].get("product") == "synthetic"}
    worst, exact = 0.0, True
    for entry, frames in generated:
        stored = np.frombuffer((SLICES / committed[entry["id"]]["file"]).read_bytes(), dtype="<f4").reshape(frames.shape)
        worst = max(worst, float(np.abs(stored - frames).max()))
        exact &= entry == committed[entry["id"]]
    if worst > TOLERANCE:
        p.error(f"The regenerated synthetic slices differ from the committed ones by up to {worst:.3g}")
    print(f"{len(generated)} synthetic slices regenerate {'exactly' if exact else f'to within {worst:.2g}'}")


if __name__ == "__main__":
    main()
