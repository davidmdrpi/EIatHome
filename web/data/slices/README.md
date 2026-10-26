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

Anyone can also drop a manifest plus its `.bin` files into the Live Lab's Data tab to try
slices locally without publishing them.
