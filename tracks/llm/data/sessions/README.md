# The sessions corpus: independent observing sessions

Every merge benchmark before this corpus trained and tested on one 2020 Voyager session
([../slices](../slices/README.md)), so none could show that a model generalizes to observations
it never saw. This corpus holds one on-target scan from each of 57 Green Bank observing sessions,
from February 2016 to October 2024. The session benchmark
([BENCHMARK.md](../../BENCHMARK.md), [session_bench.py](../../session_bench.py)) trains,
validates and tests on different sessions of it.

## What's published

259 slices of `[16, 64]` float32 in noise-sigma units (1.5 MB): 16 time steps of 18.25 s by 64
channels of 2.79 Hz, from the fine-resolution product (`source.product: "fine_resolution"`).
Each session gives one scan of a nearby star (a Hipparcos or Gliese target):

- **On-target hits** (family `stars`, 145 slices): up to 6 of the strongest narrowband signals a
  de-Doppler search finds above SNR 10 in one coarse channel of the scan. As in the Live Lab's
  pool, hits like these are almost always human interference. Their SNRs run from 10 to over
  10⁵ (median 76).
- **Blind windows** (family `blind-window`, 114 slices): 2 windows per scan at seeded random
  frequencies, placed before any data is read.

Hits come from 37 of the sessions; the other 20 found nothing above SNR 10 and give blind windows
only. The coarse channels span L band (93 slices), S (55), C (21) and X (90). Each entry records
the file's URL, the session's pointing (`source.sky`), the scan's start, the window's frequency,
its drift and SNR, as in [web/data/slices](../../../../web/data/slices/README.md).

## How the sessions were drawn

[`sessions_corpus.py`](../../sessions_corpus.py) chose them before anything was read
(`pool.json` records the draw):

1. The pipeline index, https://bldata.berkeley.edu/pipeline/, lists 1,285 sessions and their
   targets. The candidates are the 1,224 sessions with a target named `HIP…` or `GJ…`.
2. A shuffle seeded 20261107 orders the candidates.
3. In that order, a session is taken if its directory (or one of its first four subdirectories)
   lists a fine-resolution scan (`.0000.h5`) of one of those targets; the first such file by
   name is the one cut. 60 sessions were taken; 7 candidates on the way listed no such scan.
4. Three of the 60 scans were too short for a slice (14, 9 and 3 time steps; a slice needs 16),
   leaving 57 sessions.

## How the slices were cut

`tools/extract_slices.py --pool`, as for the Live Lab's pool, but 64 channels wide (the window
the language model reads) and with the product named:

```bash
python -m tracks.llm.sessions_corpus draw --output tracks/llm/data/sessions/pool.json
python tools/extract_slices.py --pool tracks/llm/data/sessions/pool.json \
  --out tracks/llm/data/sessions --width 64 --product fine_resolution
```

Everything is read over HTTPS with range requests: about 100 MB of each multi-GB file, one coarse
channel at a time. The published slices were cut in six parallel runs of ten sessions each
(`--only`), and their manifests combined; the extractor sorts a manifest's entries by id, so the
result is what a single run writes.

## Use

Anchor v4 ([anchors/README.md](../../anchors/README.md)) is defined on this corpus. Sessions are
split as `train.py` splits any family with three or more of them: whole sessions,
chronologically, 60/20/20. For the stars, that's 23 training sessions (February 2016 to March 2020), 7
validation (September 2020 to December 2021) and 7 test (April 2022 to February 2024).
