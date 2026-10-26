# 49-frame slices for the signal language model

The same Voyager 1 cadence as the Live Lab's slices ([web/data/slices](../../../../web/data/slices/README.md)),
cut from Breakthrough Listen's **6-second product** instead of the high-frequency-resolution one:
49 time steps of 6.09 s per 5-minute scan instead of 16 of 18.25 s. The channels are the same
(2.79 Hz), so Voyager's drift is still resolved, at about 0.9 channels per step. These are for
`tracks/llm`, which can train on longer windows and score windows that start mid-scan
([issue #7](https://github.com/davidmdrpi/EIatHome/issues/7)).

They're published separately because the Live Lab's models and work units are built around
16-frame slices, and its dataset id, work units and community results stay unchanged this way.
The 16-frame slices cover the same scans at the same channel width.

## What's published

Eighteen `[49, 256]` slices from one observing session (2020-07-16, MJD 59046), cut by the same
pipeline as the Live Lab's: the DC spike masked, the bandpass flattened, a de-Doppler search,
and the drift measured frame by frame. The search's drift trials are spaced 4 / 49 channels
per step, so a line between two trials ends at most 2 channels off over the scan, as with the
16-frame slices' 0.25 spacing.

- **Voyager 1** (family `voyager`, 9 slices): the carrier near 8419.54 MHz and the two telemetry
  sidebands either side, in the three on-source scans. Seven match their 16-frame slices to
  within 3 Hz. Each sideband has two lines about 320 Hz apart, and the extractor follows the
  brightest: for the lower sideband of scans 0011 and 0013 (`sideband-1`), that's the line
  320 Hz nearer the carrier (22.34 kHz from it rather than 22.66 kHz). Both lines are
  Voyager's, and each 256-channel window contains both; the 64-channel window
  `tracks.llm` uses holds the one followed.
- **Off-source** (family `offsource`, 9 slices): the three off-source scans, cut where the
  previous on-source scan's signals would have drifted to (`source.window_of`). The strongest
  de-Doppler peak in any of them is SNR 5.1, which is noise.

Each entry's `source` records `product: "6_sec_integration"`, `tsamp_s`, the `coarse_channel`
it was cut from (6 of the node's 64; the high-resolution files are that coarse channel alone),
and the `url` of the file.

All scans share one session, so `tracks.llm.train` needs `--allow-within-session`, and the score
on the held-out scan is a within-session result.

## How they were made

The 6-second files are 10.5 GB each, a whole compute node of 64 coarse channels. Each coarse
channel at each time step is one compressed HDF5 chunk, so `tools/extract_slices.py` reads them
over HTTPS with range requests and fetches only coarse channel 6: about 220 MB per scan.

```bash
base=https://bldata.berkeley.edu/voyager_2020/6_sec_integration
f() { echo "$base/blc23_guppi_59046_$1_DIAG_VOYAGER-1_$2.rawspec.0000.h5"; }
common="--coarse-channel-at 8419.54 --product 6_sec_integration --prefix voyager1-6s --out tracks/llm/data/slices"
python tools/extract_slices.py $(f 80036 0011) $(f 80672 0013) $(f 81310 0015) $common
python tools/extract_slices.py --append $common --off $(f 80354 0012) $(f 80989 0014) $(f 81628 0016)
```

Each scan takes a couple of minutes. Local copies of the files work too: pass their paths
instead of the URLs. The format is the Live Lab's (`index.json` plus raw little-endian float32
`.bin` files in noise-sigma units), with `shape` `[49, 256]`.
