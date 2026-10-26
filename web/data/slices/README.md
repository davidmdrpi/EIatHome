# Micro-slices

Real-data inputs for the Live Lab: small windows of spectrogram around detected signals, so
the browser downloads kilobytes instead of a multi-gigabyte HDF5 file.

This folder starts empty. To add the Voyager 1 slices, download the observation (see
`tracks/rnn/hello_voyager.ipynb`) and run from the repository root:

```bash
pip install -r requirements.txt
python tools/extract_slices.py data/Voyager1.single_coarse.fine_res.h5
```

That writes `index.json` plus one `.bin` file per signal (about 16 KB each). Commit them and
the Voyager button in the Live Lab is enabled for everyone. Breakthrough Listen data is public:
keep the `source` metadata in `index.json` so every slice credits the observation it came from.

## Format

- `index.json`: `{"format": "eiathome-slices", "version": 1, "slices": [...]}`.
- Each slice entry has an `id`, a `family` (`voyager`), a `group` (`carrier` or `sideband`),
  its `file`, `dtype` (`float32-le`), `shape` (`[time steps, channels]`, currently `[16, 256]`),
  a `sha256` of the file, and `source` metadata: the original file, channel width (`foff_hz`),
  time step (`tsamp_s`), first channel, centre frequency, measured drift and SNR.
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
