# Micro-slices

Real-data inputs for the Live Lab: small windows of spectrogram around detected signals, so
the browser downloads kilobytes instead of a multi-gigabyte HDF5 file.

## What's published

Two sets of slices, in one manifest (version 2). Every entry's `source.url` points to the file it
was cut from, and `source.sky` records where the telescope pointed.

- **Voyager 1 and its off-source scans** (families `voyager` and `offsource`, 21 slices): the
  known technosignature control, from two Green Bank observing sessions four years apart.
  These are the only slices `track-a/v0` work units read, so adding the others changed nothing
  in their inputs or their dataset id.
- **The slice pool** (families `stars` and `blind-window`, below): the random sky sample's
  draws, cut by `tools/extract_slices.py --pool` from `tools/slice_pool.json`.

### Voyager 1 and off-source

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

### The slice pool

The Live Lab's **Random sky sample & analyze** button draws from every published slice. The pool
adds two families to Voyager's:

- **Nearby stars** (family `stars`, group `hit`): on-target hits. For each target in
  `tools/slice_pool.json`, the [Breakthrough Listen Open Data Archive](http://seti.berkeley.edu/opendata)'s
  `api/query-files` lists its observations, and the tool picks a few fine-resolution files
  (`.0000.h5` or `.fil`) in a seeded random order, from different sessions where it can. In
  one coarse channel of each (the one containing the target's `freq_mhz`, else up to `tries`
  drawn at random away from the band edges), it runs the de-Doppler search and cuts a window
  around each of the strongest hits above SNR 10, like Voyager's. Only the first 16 time steps of
  a longer scan are read. Almost every hit like this is human interference.
  - **Proxima Centauri** is blc1's window: Parkes (Murriyang) scans from 2019-04-29, the day
    blc1 was recorded, in the coarse channel containing 982.002 MHz, where it drifted. blc1 was
    later shown to be interference.
  - The other targets are nearby stars seen from Green Bank: TRAPPIST-1, Tau Ceti, Barnard's
    Star (GJ 699), Lalande 21185, Wolf 359, Epsilon Eridani, Ross 128 and 61 Cygni, at most 8
    targets from the archive in all. Each target lists the archive queries to try in order (its
    name, its HIP number, then a cone of 0.1° around its position); the first that finds files wins.
- **Blind windows** (family `blind-window`, group `blind window`): windows at random frequencies
  in the same on-target scans, two per file, in the first coarse channel read. A seeded draw
  places them before anything is read, away from the band edges and the DC spike in the middle,
  so no detection chooses them. The Voyager target adds two from each of the four scans its
  slices come from. Blind means nobody chose where they fall, not that nothing is there: one of
  the 2020 windows lands on Voyager's carrier (SNR 615).

The pool's seed fixes every random choice, so re-running it cuts the same slices. Its limits
keep it small: at most 100 slices and 1.5 MB.

Every pool slice records, as well as the usual `source` fields:

- `source.sky.target` (the target's name, as on the sky map) along with the pointing,
  `telescope`, `epoch` and `constellation`; the scan's start is `source.tstart_mjd` and the
  window's centre frequency is `source.centre_freq_mhz`;
- `source.pool`: the pool's id and the target's key;
- `source.archive`: what the archive's API said about the file: `target`, `telescope`, `utc`,
  `mjd`, `ra` and `decl` (degrees), `center_freq`, `file_type`, `size`, `md5sum` and `url`. The
  archive serves most files over plain HTTP, so the slices were read without TLS; with the URL
  and the MD5, anyone can download the file and check a slice against it;
- `source.time_steps`, for a scan longer than 16 steps: which steps the slice holds.

#### Cutting the pool

The archive's API (`http://seti.berkeley.edu/opendata/api`) and most of its files are served
over plain HTTP only. The **Cut the slice pool** workflow (`.github/workflows/slice-pool.yml`)
runs the extraction in GitHub Actions whenever `tools/slice_pool.json` or the extractor changes
on a branch, checks every slice with the tests, and commits the slices to that branch. Run by
hand on `main`, it pushes a new branch for a pull request instead. To run it yourself:

```bash
pip install -r requirements.txt
python tools/extract_slices.py --pool tools/slice_pool.json --allow-http
```

`--only voyager1` cuts one target (over HTTPS, so without `--allow-http`); the other targets'
slices stay. A full run replaces every slice the pool cut before, so a target dropped from the
pool loses its slices. Commit the `.bin` files and `index.json` together.

### Held-out scores

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

Publishing new Voyager or off-source slices changes the dataset id that `track-a/v0` work units
record, so result cards made with the old slices no longer validate for submission. Cards
already collected stay in the community results. Slices of the other families never enter a
work unit, so they don't change it.

## Format

- `index.json`: `{"format": "eiathome-slices", "version": 2, "slices": [...]}`. Version 2
  adds `source.sky`; the Live Lab reads version 1 manifests too.
- Each slice entry has an `id`, a `family` (`voyager`, `offsource`, `stars` or
  `blind-window`), a `group` (`carrier` or `sideband`; `carrier window` or `sideband window`
  for off-source; `hit` or `blind window` for the pool), its `file`, `dtype`
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
