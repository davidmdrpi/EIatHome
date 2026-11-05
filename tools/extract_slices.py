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

Each slice records where the telescope pointed (manifest version 2): right ascension and
declination from the file header, the scan's start, the constellation (IAU boundaries in
web/data/sky/sky.json), and a distance label: Voyager 1's distance from the Sun on that date,
or, for an off-source scan, how far and in which direction it pointed from the target.

Inputs can also be https:// URLs, read with HTTP range requests. With --coarse-channel-at, only
the coarse channel containing that frequency is read and processed, as if it were a
single-coarse-channel file. Together they cut slices from a whole-node file (such as the
10.5 GB 6-second product) while downloading only the chunks of one coarse channel. Inputs can be
filterbank HDF5 or SIGPROC filterbank (.fil) files.

--pool cuts a whole pool of slices from a JSON file (tools/slice_pool.json): for each target, the
Breakthrough Listen Open Data Archive's query-files API lists its observations, and from a few of
them the tool cuts on-target hits (the strongest narrowband signals the de-Doppler search finds
in one coarse channel; family "stars") and blind windows (at seeded random frequencies, placed
before looking at the data; family "blind-window"). A target can also list file URLs directly.
Re-running the pool replaces the slices it cut before and leaves every other slice alone. The
archive serves its API and most files over plain HTTP only, so --allow-http is needed to reach
them; the Cut the slice pool workflow (.github/workflows/slice-pool.yml) runs it in GitHub Actions.
"""

import argparse
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import http.client
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import time
import urllib.error
from urllib.parse import urlencode, urljoin, urlparse
import urllib.request

import h5py
import hdf5plugin  # noqa: F401  (registers the bitshuffle filter used in Breakthrough Listen files)
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from skycoords import constellation_at, separation  # noqa: E402

VERSION = "0.6.0"
SESSION_GAP_DAYS = 0.5  # as in web/js/splits.js
COARSE_WIDTH = 1 << 20  # fine channels per coarse channel when nothing says otherwise
# Coarse channel bandwidths (MHz) by sigproc telescope_id: Green Bank's 187.5 MHz nodes hold 64,
# and Parkes's are 1 MHz. A file's fine channels per coarse channel are its coarse bandwidth over
# its channel width: 2^20 at Green Bank's usual 2.79 Hz, 1,033,216 at the 2.84 Hz of 2016's
# files, 500,000 at Parkes's 2 Hz.
COARSE_MHZ = {6: 2.9296875, 4: 1.0}
# Where coarse channels start, in coarse channels before the fch1-aligned blocks: Parkes's are
# centred on whole MHz, so on the blocks' edges (its DC spikes fall at whole MHz).
COARSE_PHASE = {4: 0.5}
# The alignments a pool run tries, in order, until a coarse channel has its DC spike in the middle:
# Parkes files differ (whole MHz in 2021, half MHz in others).
COARSE_PHASES = {4: (0.5, 0.0)}
TIME_STEPS = 16  # time steps per Live Lab slice (web/js/signals.js's T)
# Archive hosts with valid TLS certificates: their http:// URLs are read over https instead.
HTTPS_HOSTS = {"bldata.berkeley.edu"}
# Every failure to retrieve something, after retries: a pool run keeps the earlier slices of any
# target it couldn't fully retrieve (run_pool). A file the server says isn't there (404, 410) is
# no such failure: it's skipped like any other unusable file.
RETRIEVAL_FAILURES = []
GONE = (404, 410)


class SkipFile(Exception):
    """A file the tool can't cut slices from (too short, outside its band, an unsupported format)."""


class NoDCSpike(SkipFile):
    """A coarse channel without its DC spike in the middle: not the channelization assumed."""

    def __init__(self, message, height):
        super().__init__(message)
        self.height = height


class Unsuitable(SkipFile):
    """A file its header rules out (too few time steps, too coarse, the wrong band) before any
    of its data is read."""


class RangeFile(io.RawIOBase):
    """A remote file for h5py, fetched with HTTP range requests: only the bytes h5py reads. A file
    of at most WHOLE bytes is fetched in one request instead, which servers that rate-limit many
    small requests (GitHub's release downloads) prefer."""

    BLOCK = 1 << 16  # small (metadata) reads are fetched and cached in blocks of this size
    WHOLE = 64 << 20

    def __init__(self, url, cached_blocks=1024):
        self.url, self.pos, self.fetched = url, 0, 0
        # The size from a one-byte range request rather than HEAD: pre-signed download URLs (a
        # GitHub release asset's, say) accept only the method they were signed for.
        with self._request({"Range": "bytes=0-0"}) as response:
            if response.status != 206:
                raise OSError(f"{url} doesn't support range requests (HTTP {response.status})")
            total = response.headers.get("Content-Range", "").rpartition("/")[2]
        if total.isdigit():
            self.size = int(total)
        else:
            with self._request(method="HEAD") as response:
                self.size = int(response.headers["Content-Length"])
        self.whole = self._get(0, self.size) if self.size <= self.WHOLE else None
        self.blocks = OrderedDict()
        self.cached_blocks = cached_blocks
        self.prefetched = {}  # (offset, length) -> bytes, for whole chunks read once

    def _request(self, headers=None, method="GET"):
        request = urllib.request.Request(self.url, headers=headers or {}, method=method)
        for attempt in range(4):
            try:
                return urllib.request.urlopen(request, timeout=120)
            except urllib.error.HTTPError as err:
                if err.code in GONE:
                    raise
                if attempt == 3:
                    RETRIEVAL_FAILURES.append(f"{self.url}: {err}")
                    raise
            except OSError as err:
                if attempt == 3:
                    RETRIEVAL_FAILURES.append(f"{self.url}: {err}")
                    raise
            time.sleep(2 ** attempt)

    def _get(self, offset, length):
        """Bytes offset to offset + length, retried like the request itself: a body cut short or
        timing out after the headers arrived is a failure to retrieve as much as a refused request."""
        for attempt in range(4):
            with self._request({"Range": f"bytes={offset}-{offset + length - 1}"}) as response:
                # A server that ignores Range would send the whole file: stop instead.
                if response.status != 206:
                    raise OSError(f"{self.url} doesn't support range requests (HTTP {response.status})")
                try:
                    data = response.read()
                    problem = None if len(data) == length else f"short read at byte {offset}"
                except (OSError, http.client.HTTPException) as err:  # a timeout, a reset, an incomplete read
                    problem = f"{type(err).__name__} reading from byte {offset}: {err}"
            if problem is None:
                self.fetched += length
                return data
            if attempt == 3:
                RETRIEVAL_FAILURES.append(f"{self.url}: {problem}")
                raise OSError(f"{self.url}: {problem}")
            time.sleep(2 ** attempt)

    def prefetch(self, ranges, workers=8):
        """Fetch these (offset, length) ranges in parallel, for the reads h5py is about to make."""
        if self.whole is not None:
            return
        ranges = [r for r in dict.fromkeys(ranges) if r not in self.prefetched]  # each once
        with ThreadPoolExecutor(workers) as pool:
            for r, data in zip(ranges, pool.map(lambda r: self._get(*r), ranges)):
                self.prefetched[r] = data

    def _block(self, i):
        if i in self.blocks:
            self.blocks.move_to_end(i)
        else:
            self.blocks[i] = self._get(i * self.BLOCK, min(self.BLOCK, self.size - i * self.BLOCK))
            if len(self.blocks) > self.cached_blocks:
                self.blocks.popitem(last=False)
        return self.blocks[i]

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if self.whole is not None:
            b[:n] = self.whole[self.pos:self.pos + n]
            self.pos += n
            return n
        data = self.prefetched.pop((self.pos, n), None)
        if data is None and n > self.BLOCK:
            data = self._get(self.pos, n)
        if data is not None:
            b[:n] = data
            self.pos += n
            return n
        done = 0
        while done < n:
            i, offset = divmod(self.pos, self.BLOCK)
            part = self._block(i)[offset:offset + n - done]
            b[done:done + len(part)] = part
            done += len(part)
            self.pos += len(part)
        return n


_remotes = OrderedDict()


def remote_file(url):
    """An open RangeFile for url, shared by consecutive reads of one file (its header, then data)."""
    if url not in _remotes:
        _remotes[url] = RangeFile(url)
        while len(_remotes) > 2:
            _remotes.popitem(last=False)
    _remotes[url].seek(0)
    return _remotes[url]


def is_url(path):
    return isinstance(path, str) and path.startswith(("https://", "http://"))


def prefer_https(url):
    """An archive URL, over TLS when its host serves a valid certificate."""
    parts = urlparse(url)
    return url.replace("http://", "https://", 1) if parts.scheme == "http" and parts.hostname in HTTPS_HOSTS else url


def name_of(path):
    return PurePosixPath(urlparse(path).path).name if is_url(path) else path.name


def input_path(text):
    if text.startswith("http://"):
        raise argparse.ArgumentTypeError("only https:// URLs can be read remotely; download http:// files first")
    return text if is_url(text) else Path(text)


def channels_per_coarse(header):
    """Fine channels per coarse channel: the header's nfpc when it has one, else the telescope's
    coarse bandwidth over the channel width (when that's a whole number), else COARSE_WIDTH."""
    if "nfpc" in header:
        return int(header["nfpc"])
    bandwidth = COARSE_MHZ.get(int(header.get("telescope_id", -1)))
    if bandwidth:
        n = bandwidth / abs(float(header["foff"]))
        if abs(n - round(n)) < 0.01:
            return int(round(n))
    return COARSE_WIDTH


def coarse_shift(header, width):
    """Fine channels by which coarse channels start before the fch1-aligned blocks: the header's
    coarse_phase (a pool run's choice), else the telescope's (COARSE_PHASE), in coarse channels."""
    if "nfpc" in header:
        return 0
    phase = header.get("coarse_phase", COARSE_PHASE.get(int(header.get("telescope_id", -1)), 0))
    return int(round(phase * width))


def coarse_range(header, n_chans, coarse_at, name):
    """The fine channels of the coarse channel containing coarse_at (MHz), updating the header's
    fch1 and nchans to describe it."""
    fch1, foff = float(header["fch1"]), float(header["foff"])
    width = channels_per_coarse(header)
    shift = coarse_shift(header, width)
    coarse = (int(np.floor((coarse_at - fch1) / foff)) + shift) // width
    lo, hi = coarse * width - shift, (coarse + 1) * width - shift
    if lo < 0 or hi > n_chans:
        raise SystemExit(f"{name}: {coarse_at} MHz is outside the file.")
    header.update(fch1=fch1 + foff * lo, nchans=width, coarse_channel=coarse)
    return lo, hi


def time_rows(n_t, steps, name):
    """How many time steps to read: all, or the first `steps` of a longer scan."""
    if steps is None:
        return n_t
    if n_t < steps:
        raise Unsuitable(f"{name} has {n_t} time steps; a slice needs {steps}")
    return steps


FIL_KEYS = {
    "telescope_id": "i", "machine_id": "i", "data_type": "i", "barycentric": "i", "pulsarcentric": "i",
    "nbits": "i", "nsamples": "i", "nchans": "i", "nifs": "i", "nbeams": "i", "ibeam": "i",
    "rawdatafile": "s", "source_name": "s", "az_start": "d", "za_start": "d", "src_raj": "d", "src_dej": "d",
    "tstart": "d", "tsamp": "d", "fch1": "d", "foff": "d", "refdm": "d", "period": "d",
}


def read_fil_header(handle):
    """A SIGPROC filterbank header: its keywords, and where the data starts."""
    def string():
        (n,) = struct.unpack("<i", handle.read(4))
        if not 0 < n < 256:
            raise SkipFile("not a SIGPROC filterbank file")
        return handle.read(n).decode("ascii", "replace")

    if string() != "HEADER_START":
        raise SkipFile("not a SIGPROC filterbank file")
    header = {}
    while (key := string()) != "HEADER_END":
        kind = FIL_KEYS.get(key)
        if kind is None:
            raise SkipFile(f"unknown filterbank header keyword {key!r}")
        header[key] = string() if kind == "s" else struct.unpack(f"<{kind}", handle.read(struct.calcsize(kind)))[0]
    return header, handle.tell()


def apply_hints(header, hints):
    """What the archive knows that the header doesn't: a telescope the header names by an id
    this tool doesn't know (some Parkes filterbank files). Also a pool run's coarse_phase."""
    if hints and "telescope_id" in hints and int(header.get("telescope_id", -1)) not in COARSE_MHZ:
        header["telescope_id"] = hints["telescope_id"]
    if hints and "coarse_phase" in hints:
        header["coarse_phase"] = hints["coarse_phase"]
    return header


def load_fil(handle, size, remote, coarse_at, steps, name, hints=None):
    header, offset = read_fil_header(handle)
    apply_hints(header, hints)
    if int(header.get("nbits", 32)) != 32 or int(header.get("nifs", 1)) != 1:
        raise SkipFile(f"{name}: only 32-bit, single-polarization filterbank files are supported")
    n_chans = int(header["nchans"])
    n_t = (size - offset) // (n_chans * 4)
    header["time_steps"] = n_t
    lo, hi = (0, n_chans) if coarse_at is None else coarse_range(header, n_chans, coarse_at, name)
    rows = time_rows(n_t, steps, name)
    ranges = [(offset + (t * n_chans + lo) * 4, (hi - lo) * 4) for t in range(rows)]
    if remote:
        remote.prefetch(ranges)
    data = np.empty((rows, hi - lo), dtype=np.float32)
    for t, (start, length) in enumerate(ranges):
        handle.seek(start)
        data[t] = np.frombuffer(handle.read(length), dtype="<f4")
    return data, header


def load_h5(source, remote, coarse_at, steps, name, hints=None):
    with h5py.File(source, "r") as f:
        dset = f["data"]  # (time, feed, frequency)
        header = apply_hints({k: (v.decode() if isinstance(v, bytes) else v) for k, v in dset.attrs.items()}, hints)
        header["time_steps"] = dset.shape[0]
        lo, hi = (0, dset.shape[2]) if coarse_at is None else coarse_range(header, dset.shape[2], coarse_at, name)
        rows = time_rows(dset.shape[0], steps, name)
        if remote and dset.chunks:
            # Every chunk of the selected channels and time steps, fetched in parallel before h5py
            # reads them. Chunk origins only: a chunk can span several time steps (and feeds).
            t_step, f_step = dset.chunks[0], dset.chunks[2]
            chunks = (dset.id.get_chunk_info_by_coord((t, 0, c)) for t in range(0, rows, t_step)
                      for c in range(lo - lo % f_step, hi, f_step))
            remote.prefetch([(c.byte_offset, c.size) for c in chunks])
        data = dset[:rows, 0, lo:hi].astype(np.float32)
    return data, header


def load(path, coarse_at=None, steps=None, hints=None):
    """The spectrogram (time x frequency) and header. With coarse_at (MHz), only the coarse
    channel containing that frequency, with the header's fch1 and nchans describing it. With
    steps, only the first that many time steps (SkipFile if the scan has fewer)."""
    name = name_of(path)
    remote = remote_file(path) if is_url(path) else None
    handle = remote or open(path, "rb")
    fetched = remote.fetched if remote else 0
    try:
        magic = handle.read(8)
        handle.seek(0)
        if magic == b"\x89HDF\r\n\x1a\n":
            data, header = load_h5(remote or path, remote, coarse_at, steps, name, hints)
        else:
            size = remote.size if remote else Path(path).stat().st_size
            data, header = load_fil(handle, size, remote, coarse_at, steps, name, hints)
    finally:
        if not remote:
            handle.close()
    if remote and remote.whole is not None:
        print(f"  {name}: {remote.size / 1e6:,.0f} MB, fetched whole")
    elif remote:
        print(f"  {name}: read {(remote.fetched - fetched) / 1e6:,.0f} MB of {remote.size / 1e6:,.0f} MB")
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


def drift_trials(n_t, max_drift=4):
    """De-Doppler drift rates (channels per time step), spaced 4 / n_t apart: whatever the number
    of steps, a drift between two trials ends at most 2 channels off over the scan. For 16 steps
    that's the 0.25 grid the published 16-frame slices were cut with."""
    return np.arange(-max_drift, max_drift + 1e-9, 4 / n_t)


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


TELESCOPES = {6: "Green Bank Telescope", 4: "Parkes"}  # sigproc telescope_id
TARGET_NAMES = {"voyager1": "Voyager 1"}
MJD_EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)


def voyager1_distance(mjd):
    """Voyager 1's distance from the Sun on a date, from NASA's figures: it crossed the heliopause
    on 2012-08-25 at about 121.6 AU, and recedes at about 3.58 AU a year (17 km/s)."""
    years = (MJD_EPOCH + timedelta(days=mjd) - datetime(2012, 8, 25, tzinfo=timezone.utc)).days / 365.25
    return f"≈{121.6 + 3.58 * years:.0f} AU from the Sun"


DISTANCES = {"voyager1": voyager1_distance}


def target_key(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def unpack_sexagesimal(value):
    """SIGPROC's packed angles: hhmmss.s (or ddmmss.s) as one number, e.g. 142942.95 is 14h29m42.95s."""
    sign, value = (-1 if value < 0 else 1), abs(value)
    return sign * (value // 10000 + (value // 100 % 100) / 60 + value % 100 / 3600)


def pointing(header):
    """Where the telescope pointed, from the header: (RA in hours, Dec in degrees), or None.
    Filterbank files pack them as hhmmss.s and ddmmss.s; Breakthrough Listen's HDF5 files usually
    store plain hours and degrees. A value out of range for the plain form is a packed one."""
    if "src_raj" not in header or "src_dej" not in header:
        return None
    ra, dec = float(header["src_raj"]), float(header["src_dej"])
    return (unpack_sexagesimal(ra) if abs(ra) >= 24 else ra), (unpack_sexagesimal(dec) if abs(dec) > 90 else dec)


def sky_of(header, reference=None):
    """A slice's position on the sky. reference: the source block of the on-source slice an
    off-source window mirrors, whose pointing gives the offset."""
    where = pointing(header)
    if where is None:
        return None
    ra, dec = where
    mjd = float(header["tstart"])
    sky = {
        "ra_hours": round(ra, 6),
        "dec_deg": round(dec, 6),
        "frame": "J2000, the telescope's pointing from the file header",
        "epoch": (MJD_EPOCH + timedelta(days=mjd)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "constellation": constellation_at(ra * 15, dec),
    }
    telescope = TELESCOPES.get(int(header.get("telescope_id", -1)))
    if telescope:
        sky["telescope"] = telescope
    if reference and reference.get("sky"):
        on = reference["sky"]
        off = separation(ra * 15, dec, on["ra_hours"] * 15, on["dec_deg"])
        east = ((ra - on["ra_hours"] + 12) % 24 - 12) * 15 * math.cos(math.radians(on["dec_deg"]))
        north = dec - on["dec_deg"]
        direction = ("north" if north > 0 else "south") if abs(north) >= abs(east) else ("east" if east > 0 else "west")
        name = TARGET_NAMES.get(target_key(reference["source_name"]), reference["source_name"])
        sky["distance_label"] = f"Empty sky, {off:.1f}° {direction} of {name}"
    elif target_key(header.get("source_name", "")) in DISTANCES:
        sky["distance_label"] = DISTANCES[target_key(header["source_name"])](mjd)
    return sky


def dc_spike(data):
    """How far the middle channel of a coarse channel stands above its neighbours, averaged over
    time: the DC spike's height. Well above 1 when the coarse channel is where the tool thinks."""
    spectrum = data.mean(axis=0)
    mid = spectrum.size // 2
    near = np.concatenate([spectrum[mid - 64:mid - 2], spectrum[mid + 3:mid + 65]])
    return float(spectrum[mid] / np.median(near))


# How far a coarse channel's DC spike must stand above its neighbours: clearly (DC_CLEAR), or,
# where no alignment shows one that clearly, at least DC_RESIDUAL, the bump a pipeline leaves when
# it suppresses the spike (some Parkes files). Noise moves the height by about 0.03.
DC_CLEAR, DC_RESIDUAL = 1.5, 1.1


def prepare(path, coarse_at=None, steps=None, check_dc=False, hints=None, dc_min=DC_CLEAR):
    """A file's bandpass-flattened spectrogram, its header, and its de-Doppler search. With
    check_dc, a coarse channel whose middle doesn't stand dc_min above its neighbours is refused:
    the file's channelization isn't the one assumed, so a DC spike elsewhere would pass for a
    signal."""
    data, header = load(path, coarse_at, steps, hints)
    if check_dc and (height := dc_spike(data)) < dc_min:
        raise NoDCSpike(f"no DC spike in the middle of coarse channel {header.get('coarse_channel')} "
                        f"(height {height:.2f}): its coarse channels aren't {header['nchans']:,} fine channels, aligned as assumed", height)
    z = flatten_frames(mask_dc_spike(data))
    if check_dc and not np.abs(z).max() < 1e5:
        # Quantized data (2-bit recordings, say) can leave the noise scale near zero.
        raise SkipFile(f"coarse channel {header.get('coarse_channel')} doesn't normalize to noise-sigma units (peak {np.abs(z).max():.3g} sigma)")
    snr, best_drift = dedoppler(z, drift_trials(z.shape[0]))
    return z, header, snr, best_drift


def make_entry(path, args, header, window, slice_id, family, group, source):
    source = {
        "file": name_of(path),
        "source_name": str(header.get("source_name", "")),
        "tstart_mjd": float(header.get("tstart", np.nan)),
        "fch1_mhz": float(header["fch1"]),
        "foff_hz": float(header["foff"]) * 1e6,
        "tsamp_s": float(header["tsamp"]),
        **source,
    }
    if "coarse_channel" in header:
        source["coarse_channel"] = int(header["coarse_channel"])
    if header.get("time_steps", window.shape[0]) > window.shape[0]:
        source["time_steps"] = f"the first {window.shape[0]} of {int(header['time_steps'])}"
    sky = sky_of(header, source.pop("reference", None))
    if sky:
        source["sky"] = sky
    if args.product:
        source["product"] = args.product
    if is_url(path):
        source["url"] = path
    elif args.source_base_url:
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


def searchable(snr, width):
    """The de-Doppler SNRs a hit may come from: not the band edges (the bandpass rolls off), nor
    the coarse channel's DC spike in the middle."""
    snr = snr.copy()
    n_chan = snr.size
    mid, edge = n_chan // 2, n_chan // 20
    snr[:edge] = -np.inf
    snr[-edge:] = -np.inf
    snr[mid - width: mid + width] = -np.inf
    return snr


def hit_window(z, header, snr, best_drift, p, width):
    """The window centred on the signal found at channel p, tracked frame by frame, and what it
    records: where it is, its drift and its SNR."""
    n_t = z.shape[0]
    fch1, foff, tsamp = float(header["fch1"]), float(header["foff"]), float(header["tsamp"])
    positions = track(z, p, best_drift[p])
    slope, intercept = np.polyfit(np.arange(n_t), positions, 1)
    centre = intercept + slope * n_t / 2
    first = int(round(centre)) - width // 2
    return z[:, first:first + width], {
        "first_channel": first,
        "centre_freq_mhz": fch1 + foff * centre,
        "drift_channels_per_step": float(slope),
        "drift_hz_per_s": float(slope * foff * 1e6 / tsamp),
        "snr": float(snr[p]),
    }


def extract(path, args):
    """Slices for every signal found in one file."""
    z, header, snr, best_drift = prepare(path, args.coarse_channel_at)
    fch1, foff = float(header["fch1"]), float(header["foff"])
    peaks = find_peaks(searchable(snr, args.width), args.max_signals, args.threshold, exclusion=args.width)
    if not peaks:
        # Normal for the off-source scans of an on/off cadence: the transmitter isn't in the beam.
        print(f"  skipped {name_of(path)}: no signal above SNR {args.threshold:g} (an off-source scan?)")
        return []

    label = scan_label(header)
    brightest = max(peaks, key=lambda p: snr[p])
    slices = []
    sidebands = 0
    for p in sorted(peaks, key=lambda p: fch1 + foff * p):
        window, source = hit_window(z, header, snr, best_drift, p, args.width)
        if p == brightest:
            group, name = "carrier", "carrier"
        else:
            sidebands += 1
            group, name = "sideband", f"sideband-{sidebands}"
        slices.append(make_entry(path, args, header, window, f"{args.prefix}-{label}-{name}", "voyager", group, source))
    return slices


def extract_off(path, args, references):
    """Control windows from an off-source scan: for each signal of the nearest on-source scan in
    the same session, the window where that signal would have drifted to by this scan."""
    z, header, snr, _ = prepare(path, args.coarse_channel_at)
    n_chan = z.shape[1]
    tstart = float(header["tstart"])
    fch1, foff = float(header["fch1"]), float(header["foff"])
    nearby = [e for e in references if abs(e["source"]["tstart_mjd"] - tstart) <= SESSION_GAP_DAYS]
    if not nearby:
        raise SystemExit(f"{name_of(path)}: no on-source scan within 12 hours to take signal positions from.")
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
            raise SystemExit(f"{name_of(path)}: the window for {ref['id']} falls outside the file.")
        peak = float(snr[first:first + args.width].max())
        if peak >= args.threshold:
            print(f"  warning: {name_of(path)} has a signal of SNR {peak:.1f} where {ref['id']} would be (a sidelobe or RFI?)")
        slices.append(make_entry(path, args, header, z[:, first:first + args.width], f"{args.prefix}-{label}-off-{name}", "offsource", f"{ref['group']} window", {
            "first_channel": first,
            "centre_freq_mhz": freq,
            "window_of": ref["id"],
            "reference": src,
            # The strongest de-Doppler peak in the window: noise level when nothing is in the beam.
            "snr": peak,
        }))
    return slices


# ---------- the slice pool (--pool) ----------

DEFAULT_LIMITS = {"max_slices": 100, "max_bytes": 1_500_000, "max_targets": 8}
ARCHIVE_TELESCOPES = {"GBT": 6, "Parkes": 4}  # the archive's telescope names, as sigproc ids
MAX_CANDIDATES = 40  # archive files examined per target, at most


def draw(seed, *parts):
    """A number in [0, 1) fixed by the pool's seed and the parts: the same on every run."""
    digest = hashlib.sha256(":".join(str(p) for p in (seed, *parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def fetch_json(url, allow_http):
    """GET a JSON document, retrying network errors."""
    url = prefer_https(url)
    if url.startswith("http://") and not allow_http:
        raise SystemExit(f"{url} is plain HTTP: pass --allow-http to read it")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return json.loads(response.read())
        except (OSError, ValueError, http.client.HTTPException) as err:  # ValueError: not JSON (an error page); HTTPException: cut short
            if attempt == 3:
                RETRIEVAL_FAILURES.append(f"{url}: {err}")
                raise
            time.sleep(2 ** attempt)


def query_files(api, params, allow_http):
    """The archive's files matching params (api/query-files), as it lists them."""
    reply = fetch_json(f"{api.rstrip('/')}/query-files?{urlencode(params)}", allow_http)
    if reply.get("result") != "success":
        raise RuntimeError(f"query-files {params}: {reply.get('message', 'no result')}")
    return reply.get("data") or []


# Breakthrough Listen's lower-resolution products (0001: high time resolution, 0002: mid), named
# in the file name; fine-resolution files are 0000, "fine", or (some Parkes files) unmarked. The
# header's channel width settles it once the file is opened (FINE_HZ).
COARSER_PRODUCT = re.compile(r"[._]000[12][._]|_(mid|time)\.", re.IGNORECASE)
FINE_HZ = 10.0


def fine_resolution(entry, product=None):
    """Whether an archive entry may be a high-frequency-resolution filterbank product (HDF5 or
    .fil): not a lower-resolution product, nor a 2-bit recording, whose quantized samples don't
    normalize. product: a marker the file name must contain."""
    url = str(entry.get("url", ""))
    name = PurePosixPath(urlparse(url).path).name
    kind = str(entry.get("file_type", "")).lower()
    filterbank = "hdf5" in kind or "filterbank" in kind or name.endswith((".h5", ".fil"))
    return (filterbank and not COARSER_PRODUCT.search(name) and "2bit" not in name.lower()
            and (product is None or product in name))


OFF_SOURCE = re.compile(r"(^|[^A-Z])OFF([^A-Z]|$)", re.IGNORECASE)


def choose_files(entries, target, seed, wanted=None):
    """Archive entries to read, in order: on-target fine-resolution products, nearest the target's
    frequency if it has one, then in a seeded random order, one per observing session first.
    wanted: how many (all usable ones, in that order, if None)."""
    freq = target.get("freq_mhz")
    usable = [e for e in entries if fine_resolution(e, target.get("product"))
              and not OFF_SOURCE.search(str(e.get("target", "")))]
    if freq is not None:
        window = float(target.get("freq_window_mhz", 2000))
        usable = [e for e in usable if e.get("center_freq") is None or abs(float(e["center_freq"]) - freq) <= window]
    def order(e):
        distance = abs(float(e["center_freq"]) - freq) if freq is not None and e.get("center_freq") is not None else 0
        return (round(distance), draw(seed, target["key"], e.get("url")))
    usable.sort(key=order)
    chosen = []
    wanted = len(usable) if wanted is None else wanted
    for distinct in (True, False):  # distinct sessions first, then anything left
        for e in usable:
            if len(chosen) == wanted:
                break
            mjd = float(e.get("mjd") or 0)
            if e in chosen or (distinct and any(abs(mjd - float(c.get("mjd") or 0)) <= SESSION_GAP_DAYS for c in chosen)):
                continue
            chosen.append(e)
    return chosen


def peek_header(path):
    """A file's header and channel count, reading only its metadata."""
    remote = remote_file(path) if is_url(path) else None
    handle = remote or open(path, "rb")
    try:
        if handle.read(8) == b"\x89HDF\r\n\x1a\n":
            with h5py.File(remote or path, "r") as f:
                dset = f["data"]
                header = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in dset.attrs.items()}
                return header, dset.shape[2]
        handle.seek(0)
        header, _ = read_fil_header(handle)
        return header, int(header["nchans"])
    finally:
        if not remote:
            handle.close()


def coarse_frequencies(header, n_chans, target, path, seed):
    """The coarse channels to search, as the frequency at the middle of each: the target's
    frequency, or channels drawn at random away from the band edges (up to target["tries"])."""
    if target.get("freq_mhz") is not None:
        return [float(target["freq_mhz"])]
    width = channels_per_coarse(header)
    shift = coarse_shift(header, width)
    n_coarse = (n_chans + shift) // width
    if n_coarse < 1:
        raise SkipFile(f"{name_of(path)} has fewer channels than one coarse channel")
    lo, hi = (n_coarse // 10, n_coarse - n_coarse // 10) if n_coarse >= 10 else (int(shift > 0), n_coarse)
    picks = []
    for i in range(int(target.get("tries", 2))):
        c = lo + int(draw(seed, target["key"], name_of(path), "coarse", i) * (hi - lo))
        if c not in picks:
            picks.append(c)
    fch1, foff = float(header["fch1"]), float(header["foff"])
    return [fch1 + foff * (c * width - shift + width // 2) for c in picks]


def blind_starts(n_chan, width, count, seed, key, path, coarse):
    """Where blind windows start: seeded draws fixed before looking at the data, away from the
    band edges and the DC spike."""
    edge, mid = n_chan // 20, n_chan // 2
    allowed = [(edge, mid - width - width // 2), (mid + width // 2, n_chan - edge - width)]
    spans = [(a, b) for a, b in allowed if b > a]
    total = sum(b - a for a, b in spans)
    starts = []
    for i in range(count):
        u = int(draw(seed, key, name_of(path), coarse, "blind", i) * total)
        for a, b in spans:
            if u < b - a:
                starts.append(a + u)
                break
            u -= b - a
    return starts


def archive_record(entry):
    """What the archive said about the file a slice came from."""
    keys = ("target", "telescope", "utc", "mjd", "ra", "decl", "center_freq", "file_type", "size", "md5sum", "url")
    return {k: entry[k] for k in keys if entry.get(k) is not None}


def pool_slices(path, target, pool, args, archive=None):
    """Hits and blind windows from one file of a pool target."""
    hints = {"telescope_id": ARCHIVE_TELESCOPES[archive["telescope"]]} if archive and archive.get("telescope") in ARCHIVE_TELESCOPES else None
    header, n_chans = peek_header(path)
    apply_hints(header, hints)
    if abs(float(header["foff"])) * 1e6 > FINE_HZ:
        raise Unsuitable(f"{name_of(path)} has {abs(float(header['foff'])) * 1e6:.3g} Hz channels: not a fine-resolution product")
    if target.get("freq_mhz") is not None:
        edges = sorted([float(header["fch1"]), float(header["fch1"]) + float(header["foff"]) * n_chans])
        if not edges[0] <= float(target["freq_mhz"]) <= edges[1]:
            raise Unsuitable(f"{name_of(path)} covers {edges[0]:.1f}-{edges[1]:.1f} MHz, not {target['freq_mhz']} MHz")
    # The coarse channel alignments to try, until one has its DC spike in the middle.
    phases = COARSE_PHASES.get(int(header.get("telescope_id", -1)), (None,))
    wanted_hits, wanted_blind = int(target.get("hits", 3)), int(target.get("blind", 2))
    threshold = float(target.get("threshold", pool.get("threshold", 10)))
    def attempt(phase, dc_min):
        nonlocal hints
        if phase is not None:
            hints = {**(hints or {}), "coarse_phase": phase}
            header["coarse_phase"] = phase
        cut = cut_file(path, target, pool, args, archive, header, n_chans, hints, wanted_hits, wanted_blind, threshold, dc_min)
        return located_slices(cut, target, archive)

    heights = {}
    for phase in phases:
        try:
            return attempt(phase, DC_CLEAR)
        except NoDCSpike as err:
            heights[phase] = err.height
            if phase != phases[-1]:
                print(f"  {name_of(path)}: {err}; trying coarse channels offset by {phases[phases.index(phase) + 1]} of a channel")
    # No alignment shows a clear spike. A residual bump where one alignment puts it means the
    # pipeline suppressed the spike there; a misplaced spike would show nothing at all.
    best = max(heights, key=heights.get)
    if heights[best] < DC_RESIDUAL:
        raise NoDCSpike(f"no alignment puts a DC spike in the middle of a coarse channel (heights {heights})", heights[best])
    print(f"  {name_of(path)}: a suppressed DC spike (height {heights[best]:.2f}) where coarse channels"
          f"{'' if best is None else f' offset by {best} of a channel'} put it; taking that alignment")
    return attempt(best, DC_RESIDUAL)


def dc_like(window):
    """Whether a hit's window holds what a DC spike would: all its power in one channel in the
    middle, the same at every step (no drift)."""
    middle = window.shape[1] // 2
    columns = window.mean(axis=0)
    centre = int(np.argmax(columns[middle - 2:middle + 3])) + middle - 2
    return columns[centre] > 4 * max(columns[centre - 1], columns[centre + 1], 1.0)


def cut_file(path, target, pool, args, archive, header, n_chans, hints, wanted_hits, wanted_blind, threshold, dc_min=DC_CLEAR):
    """The hits and blind windows of one file, with one coarse channel alignment. When the
    alignment rests on a suppressed DC spike (dc_min below DC_CLEAR), a hit that looks like one is
    left out."""
    seed, key, width = pool["seed"], target["key"], args.width
    freqs = coarse_frequencies(header, n_chans, target, path, seed)
    out = []
    for i, freq in enumerate(freqs):
        z, header, snr, best_drift = prepare(path, freq, TIME_STEPS, check_dc=True, hints=hints, dc_min=dc_min)
        label = scan_label(header)
        coarse = header.get("coarse_channel", 0)
        extra = {"pool": {"id": pool["id"], "key": key}}
        if archive:
            extra["archive"] = archive_record(archive)
        if i == 0:  # windows at named frequencies, and blind windows, from the first coarse channel read
            fch1, foff = float(header["fch1"]), float(header["foff"])
            for named in target.get("windows", []):
                first = int(round((float(named["freq_mhz"]) - fch1) / foff)) - width // 2
                if not 0 <= first <= z.shape[1] - width:
                    print(f"  {name_of(path)}: {named['name']}'s frequency isn't in coarse channel {coarse}")
                    continue
                out.append(make_entry(path, args, header, z[:, first:first + width], f"{key}-{label}-{named['name']}", "stars", f"{named['name']} window", {
                    "first_channel": first,
                    "centre_freq_mhz": fch1 + foff * (first + width / 2),
                    "window_type": named["label"],
                    "window_of": named["about"],
                    "snr": float(snr[first:first + width].max()),
                    **extra,
                }))
            for n, first in enumerate(blind_starts(z.shape[1], width, wanted_blind, seed, key, path, coarse), 1):
                window = z[:, first:first + width]
                out.append(make_entry(path, args, header, window, f"{key}-{label}-blind-{n}", "blind-window", "blind window", {
                    "first_channel": first,
                    "centre_freq_mhz": float(header["fch1"]) + float(header["foff"]) * (first + width / 2),
                    # The strongest de-Doppler peak in the window: noise level when nothing's there.
                    "snr": float(snr[first:first + width].max()),
                    **extra,
                }))
        if wanted_hits:
            peaks = find_peaks(searchable(snr, width), wanted_hits, threshold, exclusion=width)
            print(f"  {name_of(path)} coarse channel {coarse} ({freq:.3f} MHz): {len(peaks)} hits above SNR {threshold:g}")
            n = 0
            for p in sorted(peaks, key=lambda p: -snr[p]):
                window, source = hit_window(z, header, snr, best_drift, p, width)
                if dc_min < DC_CLEAR and abs(source["drift_channels_per_step"]) < 0.05 and dc_like(window):
                    print(f"  left out a hit at {source['centre_freq_mhz']:.6f} MHz: one channel wide and not drifting, like a DC spike")
                    continue
                n += 1
                out.append(make_entry(path, args, header, window, f"{key}-{label}-hit-{n}", "stars", "hit", {**source, **extra}))
            if peaks:
                break
    return out, header


def located_slices(cut, target, archive):
    """The slices that say where the telescope pointed, with the target's name and telescope."""
    out, header = cut
    located = []
    for payload, entry in out:
        sky = entry["source"].get("sky")
        if sky is None and archive and archive.get("ra") is not None and archive.get("decl") is not None:
            # No pointing in the header: the archive's record of it (RA in degrees).
            ra, dec = float(archive["ra"]) / 15, float(archive["decl"])
            sky = entry["source"]["sky"] = {
                "ra_hours": round(ra, 6), "dec_deg": round(dec, 6), "frame": "J2000, from the archive's record",
                "epoch": (MJD_EPOCH + timedelta(days=float(header["tstart"]))).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "constellation": constellation_at(ra * 15, dec),
            }
        if sky is None:
            print(f"  skipped {entry['id']}: neither the file nor the archive says where the telescope pointed")
            continue
        if archive and archive.get("ra") is not None and archive.get("decl") is not None:
            off = separation(sky["ra_hours"] * 15, sky["dec_deg"], float(archive["ra"]), float(archive["decl"]))
            if off > 0.5:
                print(f"  warning: {entry['id']} points {off:.2f}° from the archive's position for it")
        sky["target"] = target["name"]
        if target.get("distance_label"):
            sky["distance_label"] = target["distance_label"]
        if archive and archive.get("telescope"):
            # The archive's record, over the header: some Parkes files carry Green Bank's id.
            sky["telescope"] = TELESCOPES.get(ARCHIVE_TELESCOPES.get(archive["telescope"]), str(archive["telescope"]))
        located.append((payload, entry))
    return located


def run_pool(args):
    """Cut the slice pool: every target's hits and blind windows, replacing the pool's earlier ones.

    Each target's refresh is staged. If anything couldn't be retrieved while refreshing a target
    (its archive query, or one of its files, after retries), the target keeps its earlier slices
    as they were, and once the rest of the pool is written the run exits with an error, so the
    workflow commits nothing and no slice is lost to a passing network failure."""
    pool = json.loads(args.pool.read_text())
    limits = {**DEFAULT_LIMITS, **pool.get("limits", {})}
    targets = [t for t in pool["targets"] if not args.only or t["key"] in args.only]
    manifest_path = args.out / "index.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"slices": []}
    # The pool's earlier slices: all of them (a target dropped from the pool loses its slices), or
    # with --only, those of the targets named.
    def mine(e):
        record = e["source"].get("pool", {})
        return record.get("id") == pool["id"] and (not args.only or record.get("key") in args.only)
    kept = [e for e in manifest["slices"] if not mine(e)]
    previous = [e for e in manifest["slices"] if mine(e)]
    others = sum(1 for e in kept if e["source"].get("pool", {}).get("id") == pool["id"])
    known_hashes = {e["sha256"] for e in kept}
    known_ids = {e["id"] for e in kept}

    final = []  # (payload, entry); payload None for an earlier slice kept as it is
    report, failed = [], []
    budget_slices = limits["max_slices"] - others
    budget_bytes = limits["max_bytes"] - others * TIME_STEPS * args.width * 4
    slice_bytes = TIME_STEPS * args.width * 4
    fits = lambda n: n <= budget_slices and n * slice_bytes <= budget_bytes  # noqa: E731
    full = False
    stars = 0  # archive targets that gave slices
    for target in targets:
        if full:
            report.append((target["key"], 0, "not read: the pool was full"))
            continue
        print(f"{target['name']} ({target['key']}):")
        queries = target.get("queries", [])
        if queries and stars == limits["max_targets"]:
            report.append((target["key"], 0, f"not read: {stars} archive targets already have slices"))
            continue
        used_files = 0
        wanted_files = int(target.get("files", 2))
        failures_before = len(RETRIEVAL_FAILURES)
        fresh, fresh_hashes, fresh_ids = [], set(), set()
        retrieval_failed = lambda: len(RETRIEVAL_FAILURES) > failures_before  # noqa: E731

        def read(sources):
            """Slices from these (path, archive record) sources, until the target has its files:
            at most three tries per file wanted (files ruled out by their headers alone don't
            count) among the first MAX_CANDIDATES. Stops at a retrieval failure."""
            nonlocal used_files, full
            tried = examined = 0
            for path, archive in sources:
                if full or retrieval_failed() or (archive is not None and (used_files == wanted_files or tried == 3 * wanted_files or examined == MAX_CANDIDATES)):
                    return
                examined += archive is not None
                if is_url(path) and path.startswith("http://") and not args.allow_http:
                    raise SystemExit(f"{path} is plain HTTP: pass --allow-http to read it")
                try:
                    slices = pool_slices(path, target, pool, args, archive)
                except Unsuitable as err:
                    print(f"  passed over {name_of(path)}: {err}")
                    continue
                except (Exception, SystemExit) as err:  # noqa: BLE001 (a remote file we can't use: report it and go on)
                    print(f"  skipped {name_of(path)}: {type(err).__name__}: {err}")
                    tried += archive is not None
                    continue
                tried += archive is not None
                used_files += archive is not None and bool(slices)
                for payload, entry in slices:
                    if entry["sha256"] in known_hashes | fresh_hashes or entry["id"] in known_ids | fresh_ids:
                        print(f"  skipped {entry['id']}: already in the manifest")
                        continue
                    if not fits(len(final) + len(fresh) + 1):
                        print(f"  stopped at {len(final) + len(fresh)} slices: the pool's limits are {limits}")
                        full = True
                        return
                    fresh.append((payload, entry))
                    fresh_hashes.add(entry["sha256"])
                    fresh_ids.add(entry["id"])

        read([(input_path(u) if not args.allow_http else prefer_https(u), None) for u in target.get("urls", [])])
        for query in queries:  # in order, while the target still wants files
            if full or used_files == wanted_files or retrieval_failed():
                break
            try:
                entries = query_files(pool["api"], query, args.allow_http)
            except Exception as err:  # noqa: BLE001 (the archive is remote: report it and go on)
                print(f"  couldn't query the archive with {query}: {err}")
                continue
            candidates = choose_files(entries, target, pool["seed"])
            print(f"  {query}: the archive lists {len(entries)} files; {len(candidates)} usable")
            if not candidates and entries:
                kinds = Counter(f"{e.get('file_type')} {PurePosixPath(urlparse(str(e.get('url', ''))).path).suffix}" for e in entries)
                print(f"    none usable: {dict(kinds.most_common(4))}, e.g. {[PurePosixPath(urlparse(str(e.get('url', ''))).path).name for e in entries[:3]]}")
            # The archive's URLs are absolute; resolve any that aren't against the API's site.
            read([(prefer_https(urljoin(pool["api"], e["url"])), e) for e in candidates])

        if retrieval_failed():
            # Not a refresh: keep what the last complete one cut for this target.
            earlier = [e for e in previous if e["source"]["pool"].get("key") == target["key"]]
            print(f"  kept its {len(earlier)} earlier slices: couldn't retrieve {RETRIEVAL_FAILURES[-1]}")
            failed.append(target["key"])
            chosen = [(None, e) for e in earlier]
        else:
            chosen = fresh
        for payload, entry in chosen:
            if not fits(len(final) + 1):
                print(f"  stopped at {len(final)} slices: the pool's limits are {limits}")
                full = True
                break
            final.append((payload, entry))
            known_hashes.add(entry["sha256"])
            known_ids.add(entry["id"])
        cut = sum(1 for _, e in final if e["source"]["pool"].get("key") == target["key"])
        stars += bool(queries and cut)
        report.append((target["key"], cut, "kept its earlier slices: retrieval failed" if target["key"] in failed else "" if cut else "no slices"))

    entries = [e for _, e in final]
    in_use = {e["file"] for e in kept + entries}
    for e in previous:
        if e["file"] not in in_use:
            (args.out / e["file"]).unlink(missing_ok=True)
    for payload, entry in final:
        if payload is not None:
            (args.out / entry["file"]).write_bytes(payload)
    write_manifest(manifest_path, kept + entries)
    new = sum(1 for payload, _ in final if payload is not None)
    print(f"Pool {pool['id']}: {len(entries)} slices from {sum(1 for _, n, _ in report if n)} targets "
          f"({new} cut now, replacing {len(previous)}); {len(kept) + len(entries)} in the manifest")
    for key, n, note in report:
        print(f"  {key:<20} {n:3d} slices {note}")
    if failed:
        raise SystemExit(f"Couldn't retrieve everything for {', '.join(failed)}, so they kept their earlier slices; "
                         "the rest of the pool was refreshed. Run it again to refresh them.")
    if not entries:
        raise SystemExit("The pool produced no slices.")


def write_manifest(path, entries):
    manifest = {
        "format": "eiathome-slices",
        "version": 2,
        "extracted_by": f"tools/extract_slices.py {VERSION}",
        "slices": sorted(entries, key=lambda e: e["id"]),
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5", type=input_path, nargs="*", help="Breakthrough Listen filterbank HDF5 file(s) or https:// URL(s), on-source")
    parser.add_argument("--off", type=input_path, nargs="+", default=[], help="off-source scans of the same cadence, for the control windows")
    parser.add_argument("--out", type=Path, default=Path("web/data/slices"), help="output folder")
    parser.add_argument("--append", action="store_true", help="add to the manifest already in --out instead of replacing it")
    parser.add_argument("--source-base-url", help="where the input files can be downloaded (recorded for each slice)")
    parser.add_argument("--width", type=int, default=256, help="channels per slice")
    parser.add_argument("--max-signals", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=15, help="minimum de-Doppler SNR")
    parser.add_argument("--prefix", default="voyager1", help="prefix for slice ids and file names")
    parser.add_argument("--coarse-channel-at", type=float, metavar="MHZ",
                        help="read only the coarse channel (2^20 fine channels) containing this frequency")
    parser.add_argument("--product", help="name of the data product, recorded for each slice (e.g. 6_sec_integration)")
    parser.add_argument("--pool", type=Path, help="cut the slice pool this JSON file defines (tools/slice_pool.json)")
    parser.add_argument("--only", nargs="+", metavar="KEY", help="with --pool: only these targets (the others' slices stay)")
    parser.add_argument("--allow-http", action="store_true",
                        help="with --pool: read plain-HTTP URLs (the archive's API and files), which have no TLS")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.pool:
        if args.h5 or args.off:
            raise SystemExit("--pool takes its files from the pool, not the command line.")
        run_pool(args)
        return
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

    write_manifest(manifest_path, entries)

    print(f"Added {len(added)} slices to {args.out}/ ({len(entries)} in total)")
    for e in added:
        s = e["source"]
        drift = f"drift {s['drift_hz_per_s']:+.3f} Hz/s" if "drift_hz_per_s" in s else f"window of {s['window_of']}"
        print(f"  {e['id']:<38} {s['centre_freq_mhz']:.6f} MHz  SNR {s['snr']:6.1f}  {drift}")


if __name__ == "__main__":
    main()
