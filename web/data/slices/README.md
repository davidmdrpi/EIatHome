# Micro-slices

Real-data inputs for the Live Lab: small windows of spectrogram around detected signals, so
the browser downloads kilobytes instead of a multi-gigabyte HDF5 file.

## What's published

Twenty-one slices from two Breakthrough Listen observing sessions of Voyager 1 with the Green
Bank Telescope, four years apart (manifest version 2). Every entry's `source.url` points to the
file it was cut from, and `source.sky` records where the telescope pointed.

- **2020-07-16 (MJD 59046):** an on/off cadence of six scans, from
  [bldata.berkeley.edu](https://bldata.berkeley.edu/voyager_2020/single_coarse_channel/).
  - **Voyager 1** (family `voyager`, 9 slices): from the three on-source scans, the carrier
    near 8419.54 MHz and the two telemetry sidebands about 22.6 kHz either side, drifting at
    −0.36 to −0.43 Hz/s.
  - **Off-source** (family `offsource`, 9 slices): from the three off-source scans, pointed 1°
    north, east and south of Voyager 1. Each is cut where one of the previous on-source scan's
    signals would have drifted to by then (`source.window_of` names it), so it has the same
    receiver, frequencies and processing, but nothing in the beam. The strongest de-Doppler peak
    in any of these windows is SNR 4.7, which is noise.
- **2016-09-19 (MJD 57650):** one on-source scan, the file turboSETI's tutorial uses
  (`Voyager1.single_coarse.fine_res.h5`). Breakthrough Listen serves it over plain HTTP only
  (`http://blpd0.ssl.berkeley.edu/Voyager_data/`), so the manifest links to its copy in this
  repository's [v0.1 release](https://github.com/davidmdrpi/EIatHome/releases/tag/v0.1).
  - **Voyager 1** (3 slices): the carrier near 8419.30 MHz and its two sidebands.

With two Voyager sessions, the Live Lab trains on one and evaluates on the other, so the Voyager
score is held out. Off-source data comes from one session only, so its score is still labelled
as reconstruction when the Live Lab trains on it.

The same 2020 scans are also cut from the 6-second product, with 49 time steps instead of 16, for
the Python signal language model only: see
[tracks/llm/data/slices](../../../tracks/llm/data/slices/README.md). The Live Lab's models and work
units use these 16-frame slices.

## Rebuilding them

From the repository root, with the 2016 file downloaded to `data/voyager2016/`, one command
cuts every slice. The 2020 files are read with HTTPS range requests (about 50 MB each):

```bash
pip install -r requirements.txt
base=https://bldata.berkeley.edu/voyager_2020/single_coarse_channel
python tools/extract_slices.py data/voyager2016/Voyager1.single_coarse.fine_res.h5 \
  "$base/single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5" \
  "$base/single_coarse_guppi_59046_80672_DIAG_VOYAGER-1_0013.rawspec.0000.h5" \
  "$base/single_coarse_guppi_59046_81310_DIAG_VOYAGER-1_0015.rawspec.0000.h5" \
  --off "$base/single_coarse_guppi_59046_80354_DIAG_VOYAGER-1_0012.rawspec.0000.h5" \
        "$base/single_coarse_guppi_59046_80989_DIAG_VOYAGER-1_0014.rawspec.0000.h5" \
        "$base/single_coarse_guppi_59046_81628_DIAG_VOYAGER-1_0016.rawspec.0000.h5" \
  --source-base-url https://github.com/davidmdrpi/EIatHome/releases/download/v0.1
```

That rewrites `index.json` and one `.bin` file per signal (16 KB each). The 2020 slices come out
byte for byte as first published (with tools/extract_slices.py 0.3.0). Off-source scans
(`--off`) take their window positions from the nearest on-source scan in the same session, and
the tool warns if a window contains a signal above the detection threshold, which would mean a
sidelobe or interference rather than a clean control. Slice ids include the scan's start day and
second, so slices from different scans never collide.

To add another session without regenerating the existing slices, pass its files with
`--append`. That skips slices whose bytes are already in the manifest and stops if a new slice
would reuse an existing id. Commit the new `.bin` files and `index.json` together. Breakthrough
Listen data is public: keep the `source` metadata so every slice credits the observation it came
from.

Publishing new slices changes the dataset id, so result cards made with the old slices no
longer validate for submission. Cards already collected stay in the community results.

## Format

- `index.json`: `{"format": "eiathome-slices", "version": 2, "slices": [...]}`. Version 2
  adds `source.sky`; the Live Lab reads version 1 manifests too.
- Each slice entry has an `id`, a `family` (`voyager` or `offsource`), a `group` (`carrier`
  or `sideband`; `carrier window` or `sideband window` for off-source), its `file`, `dtype`
  (`float32-le`), `shape` (`[time steps, channels]`, currently `[16, 256]`), a `sha256` of the
  file, and `source` metadata: the original file and its `url`, channel width (`foff_hz`), time
  step (`tsamp_s`), first channel, centre frequency and SNR. Voyager slices also have the
  measured drift; off-source slices have `window_of`, the Voyager slice whose position they
  follow, and their SNR is the strongest de-Doppler peak in the window.
- `source.sky` (version 2) says where the telescope pointed, for the Live Lab's sky map:
  - `ra_hours`, `dec_deg`: the pointing in the file header (J2000);
  - `epoch`: the scan's start (UTC);
  - `constellation`: from the IAU boundaries in `web/data/sky/sky.json`;
  - `distance_label`: Voyager 1's distance from the Sun on that date (it recedes about 3.6 AU
    a year), or, for an off-source scan, how far and in which direction it pointed from Voyager;
  - `telescope`.
- Each `.bin` file is raw little-endian float32, row-major (time step by time step), in
  noise-sigma units after bandpass flattening.
- The Live Lab computes each file's SHA-256 when it loads it and rejects the slice if the
  manifest's `sha256` doesn't match. Result cards record the computed digest.
- Keep `source.tstart_mjd` (the scan's start time). When the Live Lab trains on a real family,
  it evaluates on that family's other observing sessions: scans that started within 12 hours of each other
  count as one session, and identical files always count as one, whatever they're named. With
  one session, or a slice without a start time, it labels the score as reconstruction rather
  than a held-out result.
- Manifest order doesn't matter: slices are sorted by content hash before any sampling.

Anyone can also drop a manifest plus its `.bin` files into the Live Lab's Data tab to try
slices locally without publishing them.
