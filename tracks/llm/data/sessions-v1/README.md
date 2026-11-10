# The sessions corpus, v1: a fresh draw

The second draw of the sessions corpus ([v0](../sessions/README.md)), for
`signal-lm-sessions-bench/v1` ([BENCHMARK.md](../../BENCHMARK.md#v1-the-fresh-cohort)). It
scores merge methods that were frozen on v0 (the WSP protocol,
[WSP_RESULTS.md](../../WSP_RESULTS.md)) on observing sessions nobody had looked at. No session
of v0's pool is in it.

**The order of events.** wsp_bench.py's protocol was pinned on `main` (`PROTOCOL_SHA256`
`868d40e6…`, at f2bf314) before these sessions were drawn. The draw (`pool.json`) was committed
before anything was cut, and the split (`../../benchmarks/sessions-v1/split.json`, pinned in
`session_bench.py` as `SPLIT_SHA256_V1`) before anything trained on it.

## What's published

534 slices of `[16, 64]` float32 in noise-sigma units (2.2 MB), cut exactly as v0's: 16 time
steps of about 18 s by 64 channels of 2.79 Hz, from the fine-resolution product. Each session
gives one scan of a nearby star (a Hipparcos or Gliese target):

- **On-target hits** (family `stars`, 310 slices): up to 6 of the strongest narrowband signals a
  de-Doppler search finds above SNR 10 in one coarse channel of the scan. As in v0, they're
  almost all human interference. SNRs run from 10 to 3 × 10⁵ (median 40).
- **Blind windows** (family `blind-window`, 224 slices): 2 per scan at seeded random frequencies,
  placed before any data is read.

The scans run from January 2016 to April 2025. Hits come from 74 of the 112 sessions that gave
slices; the other 38 found nothing above SNR 10 and give blind windows only.

**Frequencies.** 486 slices are at 1 to 11.6 GHz. 32 come from four sessions at about 12.3 GHz
(2023 and 2024), as 21 of v0's do. Two sessions are at frequencies v0 has none of: about 700 MHz
(2017) and about 22.6 GHz (2024), with 8 slices each. The split is chronological and the higher
frequencies are recent, so five of the 15 test sessions (the 12.3 and 22.6 GHz ones) are at
frequencies no training session covers. v0 had the same pattern: three of its seven test sessions
were its 12.3 GHz ones. The benchmark also reports the test score of the other ten
([BENCHMARK.md](../../BENCHMARK.md#v1-the-fresh-cohort)).

## How the sessions were drawn

[`sessions_corpus.py`](../../sessions_corpus.py), before anything was read (`pool.json` records
the draw):

```bash
python -m tracks.llm.sessions_corpus draw --output tracks/llm/data/sessions-v1/pool.json \
  --sessions 120 --seed 20261108 --id llm-sessions/v2 --exclude tracks/llm/data/sessions/pool.json
```

1. The pipeline index, https://bldata.berkeley.edu/pipeline/, lists 1,224 sessions with a target
   named `HIP…` or `GJ…`. The 60 that v0's pool drew are excluded, leaving 1,164 candidates.
2. A shuffle seeded 20261108 orders them.
3. In that order, a session is taken if its directory (or one of its first four subdirectories)
   lists a fine-resolution scan (`.0000.h5`) of one of those targets; the first such file by name
   is the one cut. 120 sessions were taken, twice v0's 60: more independent sessions narrow the
   uncertainty, where more seeds wouldn't. Eight candidates on the way listed no such scan.
4. Eight of the 120 gave no slice. Seven scans are too short (1 to 15 time steps; a slice needs
   16). In the eighth (2024), the coarse channel doesn't normalize to noise-sigma units (a peak of
   6 × 10⁵ sigma), and the extractor's check skips such a file. That leaves 112 sessions.

## How the slices were cut

As v0's: `tools/extract_slices.py --pool`, 64 channels wide, with the product named. Everything
is read over HTTPS with range requests, about 100 MB of each multi-GB file.

```bash
python tools/extract_slices.py --pool tracks/llm/data/sessions-v1/pool.json \
  --out tracks/llm/data/sessions-v1 --width 64 --product fine_resolution
```

They were cut in twelve parallel runs of ten sessions each (`--only`, each into its own folder).
The manifests were then combined, every file checked against its manifest hash, and the entries
sorted by id, so `index.json` is what a single run writes.

## Use

```bash
python -m tracks.llm.session_bench split --check --benchmark v1   # the pinned split
python -m tracks.llm.session_bench verify --benchmark v1          # the published cohort and merges
```

Anchor v4 is used unchanged: its peak tokenizer fits nothing and θ₀ is deterministic, so no session
of either draw shaped them.

| Split | Sessions | Slices | Targets | Observed |
|---|---:|---:|---:|---|
| Train | 44 | 186 | 2,790 | February 2016 to July 2020 |
| Validation | 15 | 59 | 885 | August 2020 to April 2022 |
| Test | 15 | 65 | 975 | May 2022 to March 2025 |

v0's is 23, 7 and 7 sessions. Every session v1 is scored on (validation, test, and the blind
windows' sessions) is more than half a day from every v0 session, so it's another observing
session by the corpus's own rule. One v1 training session (2016-03-29, HIP 57274) started 10
hours before a v0 training session. v1 trains its own cohort from θ₀, so that leaks nothing.
