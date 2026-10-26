"""Detectors score a window (frames x channels, noise sigma) without seeing any label: higher
means more likely to hold a signal. These are the fixed baselines that learned detectors
must beat at the same false-alarm rate."""

import numpy as np


def drift_trials(frames, max_drift=4):
    """As tools/extract_slices.py: spaced 4 / frames channels per step, so a line between two
    trials ends at most 2 channels off over the window."""
    return np.arange(-max_drift, max_drift + 1e-9, 4 / frames)


def dedoppler(windows, chunk=256):
    """The conventional narrowband search: the strongest straight track, summed over frames
    and divided by sqrt(frames), over every drift trial and start channel. Tracks never wrap
    around the window's edge. windows: (n, frames, channels); returns n scores."""
    if len(windows) > chunk:  # bounded memory: each trial gathers n x starts x frames values
        return np.concatenate([dedoppler(windows[i:i + chunk]) for i in range(0, len(windows), chunk)])
    windows = np.asarray(windows, dtype=np.float64)
    n, frames, channels = windows.shape
    t = np.arange(frames)
    best = np.full(n, -np.inf)
    for drift in drift_trials(frames):
        offsets = np.round(drift * t).astype(int)
        starts = np.arange(-offsets.min(), channels - offsets.max())
        if not len(starts):
            continue  # the track leaves the window at any start
        tracks = windows[:, t[None, :], starts[:, None] + offsets[None, :]]  # n x starts x frames
        best = np.maximum(best, tracks.sum(axis=2).max(axis=1))
    return best / np.sqrt(frames)


def energy(windows):
    """A radiometer: the window's total power above what unit Gaussian noise would give, in
    units of that noise's standard deviation."""
    windows = np.asarray(windows, dtype=np.float64)
    count = windows.shape[1] * windows.shape[2]
    return ((windows ** 2).sum(axis=(1, 2)) - count) / np.sqrt(2 * count)


DETECTORS = {"dedoppler": dedoppler, "energy": energy}
