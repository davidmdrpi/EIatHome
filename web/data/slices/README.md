# Micro-slices

Real-data inputs for the Live Lab: small windows of spectrogram around detected signals, so
the browser downloads kilobytes instead of a multi-gigabyte HDF5 file.

## What's published

Eighteen slices from one Breakthrough Listen observing session: Voyager 1 with the Green Bank
Telescope on 2020-07-16 (MJD 59046), an on/off cadence of six scans. Every entry's `source.url`
points to the file it was cut from on
[bldata.berkeley.edu](https://bldata.berkeley.edu/voyager_2020/single_coarse_channel/).

- **Voyager 1** (family `voyager`, 9 slices): from the three on-source scans, the carrier near
  8419.54 MHz and the two telemetry sidebands about 22.6 kHz either side, drifting at −0.36 to
  −0.43 Hz/s.
- **Off-source** (family `offsource`, 9 slices): from the three off-source scans, with the
  telescope pointed 1° away. Each is cut where one of the previous on-source scan's signals would
  have drifted to by then (`source.window_of` names it), so it has the same receiver, frequencies
  and processing, but nothing in the beam. The strongest de-Doppler peak in any of these windows
  is SNR 4.7, which is noise.

The scans started within 30 minutes of each other, so they count as one session, and the Live Lab
labels a score on the family it trained on as reconstruction rather than held out. A second
session (scans more than 12 hours from these) fixes that.

## Adding slices

Download the observation files, then run from the repository root, for example:

```bash
pip install -r requirements.txt
base=https://bldata.berkeley.edu/voyager_2020/single_coarse_channel
curl -O "$base/single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5"   # and the others
python tools/extract_slices.py single_coarse_guppi_*.h5 --source-base-url "$base"
```

That writes `index.json` plus one `.bin` file per signal (16 KB each). It accepts several files
at once and skips those with no signal. Slice ids include the scan's start day and second, so
slices from different scans never collide.

Pass the off-source scans of the cadence with `--off` to cut the control windows. They take
their positions from the nearest on-source scan in the same session, from this run or (with
`--append`) from the manifest:

```bash
python tools/extract_slices.py --append --source-base-url "$base" \
  --off single_coarse_guppi_59046_{80354,80989,81628}_*.h5
```

The tool warns if a window contains a signal above the detection threshold, which would mean
a sidelobe or interference rather than a clean control.

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
- Each slice entry has an `id`, a `family` (`voyager` or `offsource`), a `group` (`carrier`
  or `sideband`; `carrier window` or `sideband window` for off-source), its `file`, `dtype`
  (`float32-le`), `shape` (`[time steps, channels]`, currently `[16, 256]`), a `sha256` of the
  file, and `source` metadata: the original file and its `url`, channel width (`foff_hz`), time
  step (`tsamp_s`), first channel, centre frequency and SNR. Voyager slices also have the
  measured drift; off-source slices have `window_of`, the Voyager slice whose position they
  follow, and their SNR is the strongest de-Doppler peak in the window.
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
