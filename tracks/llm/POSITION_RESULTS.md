# Position encodings and offset windows on 49-frame slices

[Issue #7](https://github.com/davidmdrpi/EIatHome/issues/7) asked whether a model trained on
windows that start at a scan's first frame handles windows that start mid-scan, and whether
rotary positions (RoPE) and offset training windows fix that. These runs compare learned
absolute positions and RoPE, each trained on non-overlapping windows (stride 15, the default)
and on windows at every start (stride 1), on the [49-frame slices](data/slices/README.md).

Like the [smoke results](SMOKE_RESULTS.md), this is one observing session: training, validation
and test are three different scans of the same 2020 cadence (`--allow-within-session`). It is
not evidence of generalization across sessions, or of language.

## Setup

The configuration was fixed before any run: the defaults (16 tokens, context 15, width 128,
4 layers, 4 heads, dropout 0.1, 50 maximum epochs, patience 8, learning rate 0.0003,
799,360 parameters), Voyager family, `--position absolute|rope`, `--train-stride 15|1`,
seeds 42, 43 and 44. Twelve runs, each training the real, shuffled and Markov-1 streams:

```bash
for seed in 42 43 44; do for position in absolute rope; do for stride in 15 1; do
  python -m tracks.llm.train --manifest tracks/llm/data/slices/index.json --allow-within-session \
    --position $position --train-stride $stride --seed $seed --output runs/p-$position-s$stride-$seed
done; done; done
```

Each split is one scan: 3 slices, 147 frames, 144 next-token targets. Scores, in bits per token
(uniform is 4):

- **Test**: the standard score, each target once in non-overlapping windows from frame 0.
- **Scan start**: the full 16-frame window at frame 0 of each test slice (3 windows).
- **Mid-scan**: the full windows starting at frames 1 to 33 (99 windows), pooled.

Every window has the same length and the same contexts within it, so scan-start and mid-scan
scores differ only in where the window starts (see `test_window_starts` in the
[README](README.md#what-the-pipeline-learns)).

## Results

Mean over the three seeds, with the range in brackets. Best epoch is per seed.

| Positions | Training stride | Stream | Test | Scan start | Mid-scan | Best epoch |
|---|---:|---|---:|---:|---:|---:|
| Absolute | 15 | Voyager (real) | 1.68 (1.56–1.83) | 1.95 | 1.58 (1.44–1.76) | 22 / 29 / 23 |
| Absolute | 15 | shuffled | 3.18 (2.94–3.36) | 3.21 | 3.14 (2.73–3.34) | 17 / 10 / 11 |
| Absolute | 15 | Markov-1 | 4.00 (3.82–4.28) | 4.02 | 3.98 (3.66–4.31) | 8 / 8 / 12 |
| Absolute | 15 | off-source control | 3.41 (2.71–4.03) | 3.14 | 3.47 (2.83–4.03) | – |
| Absolute | 1 | Voyager (real) | 1.61 (1.39–1.79) | 2.24 | 1.28 (1.02–1.48) | 12 / 9 / 9 |
| Absolute | 1 | shuffled | 3.07 (2.78–3.25) | 3.11 | 3.02 (2.63–3.26) | 4 / 2 / 3 |
| Absolute | 1 | Markov-1 | 3.95 (3.71–4.25) | 3.96 | 3.91 (3.59–4.25) | 2 / 3 / 4 |
| Absolute | 1 | off-source control | 4.08 (3.14–4.96) | 3.92 | 4.14 (3.27–4.93) | – |
| RoPE | 15 | Voyager (real) | 1.54 (1.46–1.66) | 1.80 | 1.39 (1.32–1.53) | 13 / 24 / 19 |
| RoPE | 15 | shuffled | 3.06 (2.76–3.24) | 3.09 | 3.04 (2.66–3.28) | 11 / 5 / 8 |
| RoPE | 15 | Markov-1 | 3.89 (3.63–4.14) | 3.81 | 3.86 (3.57–4.16) | 6 / 9 / 4 |
| RoPE | 15 | off-source control | 3.48 (2.86–4.02) | 3.31 | 3.53 (2.98–3.99) | – |
| RoPE | 1 | Voyager (real) | 1.57 (1.37–1.70) | 2.11 | 1.30 (1.05–1.51) | 8 / 5 / 7 |
| RoPE | 1 | shuffled | 3.00 (2.72–3.21) | 3.04 | 2.97 (2.66–3.19) | 2 / 2 / 2 |
| RoPE | 1 | Markov-1 | 3.92 (3.66–4.16) | 3.87 | 3.89 (3.61–4.17) | 1 / 1 / 1 |
| RoPE | 1 | off-source control | 3.92 (3.10–4.57) | 3.76 | 3.95 (3.22–4.51) | – |

Baselines fitted on each stream's training scan, on its test scan: real unigram 3.63 and
Markov-1 2.24; shuffled 3.63 and 3.24; Markov-1 stream 3.99 and 3.72; off-source 3.90 and 3.56.

Mid-scan differences between configurations, paired by seed (same tokenizer, data and
initialization seed), for seeds 42, 43 and 44:

| Comparison | Mid-scan difference (bits/token) |
|---|---|
| Absolute stride 15 − RoPE stride 15 | +0.12, +0.22, +0.23 |
| Absolute stride 15 − absolute stride 1 | +0.42, +0.21, +0.28 |
| Absolute stride 15 − RoPE stride 1 | +0.39, +0.22, +0.25 |
| RoPE stride 15 − RoPE stride 1 | +0.27, −0.01, +0.02 |
| Absolute stride 1 − RoPE stride 1 | −0.03, +0.01, −0.03 |

## What this shows

- **Absolute positions with non-overlapping windows handle mid-scan windows worst.** On the
  same 99 mid-scan windows it is behind every other configuration in all three seeds: by
  0.12–0.23 bits against RoPE with the same windows, and 0.21–0.42 against either model trained
  on every start.
- **Training on every start removes the difference.** With stride 1, absolute and RoPE models
  score within 0.03 bits of each other. RoPE alone, with non-overlapping windows, closes
  between a third and all of the gap, depending on the seed.
- **The models learn order in the real stream.** Real Voyager tokens score 1.3 to 1.7 bits,
  below the training-fitted Markov-1 baseline (2.24). Shuffled within each slice, the same
  tokens score 3.0 to 3.2: below the unigram baseline (3.63), because a window's tokens show
  which slice it comes from, but far above the real order. The Markov-1 stream scores 3.9 to
  4.0, near its unigram baseline (3.99): from 147 training frames the models don't pick up its
  weak first-order structure (its Markov-1 baseline is 3.72).
- **Off-source noise scores far worse than Voyager.** Through Voyager's tokenizer, the models
  trained on Voyager score the off-source control at 3.4 to 4.1 bits on average, around the
  Voyager-fitted baselines on that control (unigram 3.90, Markov-1 3.56) and well above their
  1.3 to 1.7 on Voyager.
- **The scan-start window is harder for every model, so "mid-scan minus scan start" within
  one model isn't the test issue #7 predicted.** On the real stream, every configuration scores
  its 3 scan-start windows worse than its mid-scan windows. That includes RoPE with stride 1,
  which has no preferred alignment. The shuffled and Markov-1 streams show no such difference.
  The cause is the data: in the test scan's first frames, the carrier is near the edge of the
  64-channel window and fainter than later, and both sidebands are at noise level in frame 0.
  The bits-per-window profile over starts has the same shape for all four configurations.
  Stride-1 models score the scan start worse than stride-15 models (2.1–2.2 against 1.8–2.0),
  plausibly because only 1 of 48 training windows per slice starts at frame 0, against 1 of 4.

By the issue's criterion, a RoPE model trained on offset windows scores no worse mid-scan than
at the scan start (1.30 against 2.11). That comparison is dominated by the harder scan-start
content, though. The absolute-position gap shows up between models on the same windows, not
within one model.

## Limitations

- One observing session and one test scan (three slices). Three seeds. The scan-start score
  rests on 3 windows (45 targets).
- Mid-scan windows overlap, so a frame is scored in up to 15 of them. The pooled mid-scan
  score is an average over windows, not a likelihood of the scan.
- Continuations past the context from RoPE models (41 tokens after an 8-token prompt in these
  reports) are qualitative. They repeat what the model learned about 16-frame windows and
  aren't evidence of longer structure.

## Environment and fingerprints

- Python 3.13.17; NumPy 2.5.4; PyTorch 2.14.1 (CPU, one thread).
- Dataset fingerprint: `970bc3fe3c9727d113d35418eeb16d8db6dad64172d202173a8a41cf0885d667`.
- Training code fingerprint: `c3afa2fb48b541d7e0a15368fd3eca1bdae91e347982809549fa25ef919b5c63`.
- With the defaults, the new code reproduces [SMOKE_RESULTS.md](SMOKE_RESULTS.md) exactly on the
  16-frame slices (the same loss histories, test scores, control scores and sample).
