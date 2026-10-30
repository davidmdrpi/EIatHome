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


def generate():
    """The Rosetta slices: [(manifest entry, float32 frames)], in text order."""
    if hashlib.sha256(CORPUS.encode()).hexdigest() != CORPUS_SHA256:
        raise ValueError("The Rosetta corpus doesn't match its pinned SHA-256")
    symbols = text_symbols()
    rng = np.random.default_rng(SEED)
    out = []
    for k in range(SCANS * PER_SCAN):
        window = symbols[k * FRAMES:(k + 1) * FRAMES]
        flux = math.exp(math.log(FLUX[0]) + (math.log(FLUX[1]) - math.log(FLUX[0])) * rng.random())
        frames = fsk_frames(window, flux, rng.standard_normal((FRAMES, CHANNELS)))
        scan, part = divmod(k, PER_SCAN)
        name = f"rosetta-fsk-scan{scan + 1}-part{part + 1}"
        entry = {
            "id": name, "family": "rosetta", "group": "fsk", "file": f"{name}.bin", "dtype": "float32-le",
            "shape": [FRAMES, CHANNELS], "units": "noise sigma",
            "sha256": hashlib.sha256(frames.astype("<f4").tobytes()).hexdigest(),
            "source": {
                "product": "synthetic", "generator": "tracks/llm/synthetic.py",
                "modulation": "16-tone FSK, one 4-bit symbol per frame at channel 8 + 3 x symbol",
                "corpus": CORPUS_ID, "corpus_sha256": CORPUS_SHA256, "symbols": [k * FRAMES, (k + 1) * FRAMES],
                "flux": flux, "line_width_channels": LINE_WIDTH, "seed": SEED, "slice": k,
                "tstart_mjd": SCAN_START_MJD[scan],
            },
        }
        out.append((entry, frames))
    return out


def symbols_of(entry):
    """The symbols a Rosetta slice carries, one per frame."""
    start, end = entry["source"]["symbols"]
    return text_symbols()[start:end]


def manifest(entries):
    real = json.loads(REAL_MANIFEST.read_text())
    return {**{k: v for k, v in real.items() if k != "slices"}, "synthetic_by": "tracks/llm/synthetic.py",
            "slices": real["slices"] + entries}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["build"])
    p.add_argument("--write", action="store_true", help="write the slices and the manifest instead of checking them")
    args = p.parse_args(argv)
    generated = generate()
    if args.write:
        for entry, frames in generated:
            (SLICES / entry["file"]).write_bytes(frames.astype("<f4").tobytes())
        MANIFEST.write_text(json.dumps(manifest([e for e, _ in generated]), indent=1) + "\n")
        print(f"wrote {len(generated)} Rosetta slices and {MANIFEST.name}")
        return
    committed = {e["id"]: e for e in json.loads(MANIFEST.read_text())["slices"] if e["family"] == "rosetta"}
    worst, exact = 0.0, True
    for entry, frames in generated:
        stored = np.frombuffer((SLICES / committed[entry["id"]]["file"]).read_bytes(), dtype="<f4").reshape(frames.shape)
        worst = max(worst, float(np.abs(stored - frames).max()))
        exact &= entry == committed[entry["id"]]
    if worst > TOLERANCE:
        p.error(f"The regenerated Rosetta slices differ from the committed ones by up to {worst:.3g}")
    print(f"{len(generated)} Rosetta slices regenerate {'exactly' if exact else f'to within {worst:.2g}'}")


if __name__ == "__main__":
    main()
