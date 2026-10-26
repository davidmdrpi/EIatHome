"""Synthetic signals for injection into background windows, and the blinded dealer.

A signal is a spec (a JSON-serializable dict) drawn from a seeded generator, rendered to a
frames x channels array in noise-sigma units and added to a background window. Its strength is
the optimal matched-filter SNR, the signal's norm: a detector that knew the exact signal would
score it at that SNR in unit Gaussian noise, whatever its class.

The dealer hands detectors only arrays with opaque ids, in shuffled order. The specs stay in a
separate key, joined to the detectors' scores after detection.
"""

import numpy as np

SIGNALS = ("tone", "intermittent", "wideband", "sidebands", "fsk")
DECOYS = ("stationary", "pulse")  # interference-like: zero drift, or one broadband frame
KINDS = SIGNALS + DECOYS
STRENGTHS = (5, 7, 10, 14, 20, 28)
# Channels per step; Voyager drifts about 0.9 in the 6-second product. Wider signals get less,
# so that they stay inside the 64-channel window.
MAX_DRIFT = 1.2
MARGIN = 2  # channels kept free at each edge of the window


def profile(center, width, channels):
    """One frame's power across channels for a line at a fractional channel, summing to 1.
    Narrow lines (width <= 1) split between the two nearest channels; wider ones are Gaussian
    with that full width at half maximum, cut off beyond 1.5 widths from the centre."""
    out = np.zeros(channels)
    if width <= 1:
        low = int(np.floor(center))
        frac = center - low
        out[low] += 1 - frac
        if frac:
            out[low + 1] += frac
        return out
    distance = np.arange(channels) - center
    out = np.exp(-0.5 * (distance / (width / 2.3548)) ** 2)
    out[np.abs(distance) > 1.5 * width] = 0
    return out / out.sum()


def track_start(rng, drift, frames, channels, extent=0.0):
    """A start channel whose track, plus extent either side, stays inside the window."""
    travel = drift * (frames - 1)
    low = MARGIN + extent - min(0.0, travel)
    high = channels - 1 - MARGIN - extent - max(0.0, travel)
    if high < low:
        raise ValueError("The track doesn't fit in the window")
    return float(rng.uniform(low, high))


def draw(rng, kind, strength, frames, channels):
    """A random spec of this kind and strength."""
    spec = {"kind": kind, "strength": float(strength)}
    if kind == "none":
        return spec
    if kind == "pulse":
        spec["frame"] = int(rng.integers(frames))
        return spec
    if kind == "stationary":
        spec.update(drift=0.0, start=track_start(rng, 0.0, frames, channels), width=1.0)
        return spec
    # The shape first, then a drift that keeps the whole signal inside the window.
    spec["width"] = 1.0
    extent = 0.0
    if kind == "intermittent":
        period = int(rng.integers(4, 13))
        spec.update(period=period, on=int(max(1, round(period * rng.uniform(0.25, 0.5)))), phase=int(rng.integers(period)))
    elif kind == "wideband":
        spec["width"] = float(rng.uniform(3, 8))
        extent = 1.5 * spec["width"]
    elif kind == "sidebands":
        spec.update(offset=float(rng.uniform(6, 14)), ratio=float(rng.uniform(0.2, 0.6)))
        extent = spec["offset"] + 1
    elif kind == "fsk":
        symbol = int(rng.integers(1, 5))
        spec.update(shift=float(rng.uniform(2, 6)), symbol=symbol,
                    symbols=rng.integers(2, size=-(-frames // symbol)).tolist())
        extent = spec["shift"] + 1
    elif kind != "tone":
        raise ValueError(f"Unknown kind: {kind}")
    limit = min(MAX_DRIFT, (channels - 1 - 2 * (MARGIN + extent)) / (frames - 1))
    spec["drift"] = float(rng.uniform(-limit, limit))
    spec["start"] = track_start(rng, spec["drift"], frames, channels, extent)
    return spec


def render(spec, frames, channels):
    """The signal described by spec, scaled to its strength (zero for "none")."""
    kind = spec["kind"]
    s = np.zeros((frames, channels))
    if kind == "none":
        return s
    if kind == "pulse":
        s[spec["frame"]] = 1.0
    else:
        for t in range(frames):
            centre = spec["start"] + spec["drift"] * t
            if kind == "intermittent" and (t + spec["phase"]) % spec["period"] >= spec["on"]:
                continue
            if kind == "fsk":
                centre += spec["shift"] * spec["symbols"][t // spec["symbol"]]
            s[t] = profile(centre, spec["width"], channels)
            if kind == "sidebands":
                for side in (-1, 1):
                    s[t] += spec["ratio"] * profile(centre + side * spec["offset"], 1.0, channels)
    return s * (spec["strength"] / np.linalg.norm(s))


def design(per_cell, total, kinds=KINDS, strengths=STRENGTHS):
    """(kind, strength) for each of total windows: per_cell of every kind and strength, and
    signal-free windows for the rest."""
    cells = [(k, s) for k in kinds for s in strengths for _ in range(per_cell)]
    if len(cells) > total:
        raise ValueError(f"{len(cells)} injections don't fit in {total} windows")
    return cells + [("none", 0.0)] * (total - len(cells))


def deal(backgrounds, cells, seed):
    """Inject one signal (or none) per background window, in random order.

    Returns (blind, key): blind is a list of (id, window) in shuffled order, the only thing
    detectors see; key maps each id to its spec and the index of its background window."""
    rng = np.random.default_rng(seed)
    n, frames, channels = backgrounds.shape
    if len(cells) != n:
        raise ValueError("Need one design cell per background window")
    assignment = rng.permutation(n)  # which design cell each window gets
    ids = [f"w{v:016x}" for v in rng.choice(2 ** 62, n, replace=False)]
    blind, key = [], {}
    for i in rng.permutation(n):  # dealing order is independent of window and kind
        kind, strength = cells[assignment[i]]
        spec = draw(rng, kind, strength, frames, channels)
        window = (backgrounds[i] + render(spec, frames, channels)).astype(np.float32)
        blind.append((ids[i], window))
        key[ids[i]] = {**spec, "background": int(i)}
    return blind, key
