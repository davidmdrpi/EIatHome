# Micro-slices

Real-data inputs for the Live Lab: small windows of spectrogram around detected signals, so
the browser downloads kilobytes instead of a multi-gigabyte HDF5 file.

## What's published

Nine slices from one Breakthrough Listen observing session: Voyager 1 with the Green Bank
Telescope on 2020-07-16 (MJD 59046), from the three on-source scans of an on/off cadence. Each
scan gives the carrier near 8419.54 MHz and the two telemetry sidebands about 22.6 kHz either
side, drifting at −0.36 to −0.43 Hz/s. The three off-source scans contain no signal and are
skipped. Every entry's `source.url` points to the file it was cut from on
[bldata.berkeley.edu](https://bldata.berkeley.edu/voyager_2020/single_coarse_channel/).

The three scans started within 21 minutes of each other, so they count as one session, and the
Live Lab labels Voyager scores as reconstruction rather than held out. A second session (scans
more than 12 hours from these) fixes that.

## Adding slices

Download the observation files, then run from the repository root, for example:

```bash
pip install -r requirements.txt
base=https://bldata.berkeley.edu/voyager_2020/single_coarse_channel
curl -O "$base/single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5"   # and the others
python tools/extract_slices.py single_coarse_guppi_*.h5 --source-base-url "$base"
```

That writes `index.json` plus one `.bin` file per signal (16 KB each). It accepts several files
at once and skips those with no signal (off-source scans). Slice ids include the scan's start
day and second, so slices from different scans never collide.

To add another session without regenerating the existing slices, use `--append`. For the 2016
observation from turboSETI's example (MJD 57650, served over plain HTTP only from
`http://blpd0.ssl.berkeley.edu/Voyager_data/Voyager1.single_coarse.fine_res.h5`):

```bash
python tools/extract_slices.py Voyager1.single_coarse.fine_res.h5 --append \
  --source-base-url http://blpd0.ssl.berkeley.edu/Voyager_data
```

`--append` skips slices whose bytes are already in the manifest and stops if a new slice would
reuse an existing id. Commit the new `.bin` files and `index.json` together. Breakthrough Listen
data is public: keep the `source` metadata so every slice credits the observation it came from.

Publishing new slices changes the dataset id, so result cards made with the old slices no
longer validate for submission. Cards already collected stay in the community results.

## Format

- `index.json`: `{"format": "eiathome-slices", "version": 1, "slices": [...]}`.
- Each slice entry has an `id`, a `family` (`voyager`), a `group` (`carrier` or `sideband`),
  its `file`, `dtype` (`float32-le`), `shape` (`[time steps, channels]`, currently `[16, 256]`),
  a `sha256` of the file, and `source` metadata: the original file and its `url`, channel width
  (`foff_hz`), time step (`tsamp_s`), first channel, centre frequency, measured drift and SNR.
- Each `.bin` file is raw little-endian float32, row-major (time step by time step), in
  noise-sigma units after bandpass flattening.
- The Live Lab computes each file's SHA-256 when it loads it and rejects the slice if the
  manifest's `sha256` doesn't match. Result cards record the computed digest.
- Keep `source.tstart_mjd` (the scan's start time). When the Live Lab trains on Voyager, it
  evaluates on other observing sessions: scans that started within 12 hours of each other
  count as one session, and identical files always count as one, whatever they're named. With
  one session, or a slice without a start time, it labels the score as reconstruction rather
  than a held-out result.
- Manifest order doesn't matter: slices are sorted by content hash before any sampling.

Anyone can also drop a manifest plus its `.bin` files into the Live Lab's Data tab to try
slices locally without publishing them.
