# Merging volunteer runs

[Issue #20](https://github.com/davidmdrpi/EIatHome/issues/20) merges small `SignalLM` runs
from many volunteers into stronger models. Milestone 1 gave every run a shared anchor (a pinned
codebook and starting weights θ₀; see [anchors/](anchors/README.md)), so their changes
τ = θ − θ₀ line up. This file records what merging does on the published data, milestone by
milestone.

Everything here is one 2020 observing session: training, validation and test are three scans
of the same cadence (`--allow-within-session`). It isn't evidence of generalization across
sessions, or of language.

## Milestone 2, first part: the plain baseline

The review of #21 asked for the simplest merge that can be measured before anything cleverer:
does averaging workers trained on different data beat those workers, and how does it compare
with one worker trained on all their data?

### Setup

Fixed before the first run ([`merge_bench.py`](merge_bench.py)):

- **Data:** anchor v1 on the 49-frame slices, Voyager. Each split is one scan, three slices:
  the carrier and its two telemetry sidebands. Validation and test each have 144 next-token
  targets.
- **Training:** the real stream only, training windows at every start (`--train-stride 1`),
  50 epochs, the defaults otherwise (learning rate 0.0003, batch 32, patience 8, dropout 0.1).
  Seeds 42, 43 and 44.
- **Shards:** three, one training slice each, in a fixed order by content: sideband 1, the
  carrier, sideband 2. That makes the shards non-IID by construction: each worker sees one
  signal.
- **Merging** ([`merge.py`](merge.py)): an average weighted by training targets (48 per shard,
  so equal weights), counted by the round from each worker's shard rather than taken from its
  record. Each worker, and then the merge, has to score within 0.02 bits per token
  of the incumbent on validation, and every change must be finite and under 10 in size. A round
  never reads the test split; test and the off-source control are scored afterwards, for
  reporting.
- **Compared, per seed:**
  - the three shard workers, each trained 50 epochs from θ₀;
  - the best of them by validation;
  - one round merging the three;
  - one worker on all three slices for 50 epochs: the same data and epochs as the shards
    together;
  - ten rounds of averaging: each round, the three shards train 5 epochs from the last merge.
- **Added after the first run**, as a control for optimizer steps: one worker on all three
  slices for 20 epochs. A shard's 48 training windows make 2 batches an epoch and all three
  slices' 144 make 5, so 20 epochs take as many steps as one shard worker's 50.

```bash
python -m tracks.llm.merge_bench --output runs/merge-bench   # about a minute on a CPU
```

### Results

Bits per token, mean over the three seeds (range in brackets). Uniform over 16 tokens is 4.

| | Runs | Validation | Test | Off-source control |
|---|---:|---:|---:|---:|
| θ₀, untrained | 1 | 4.00 | 3.78 | 4.28 |
| shard workers | 9 | 3.53 (3.26–4.00) | 3.20 (2.90–3.71) | 4.44 (4.29–4.62) |
| best worker (by validation) | 3 | 3.26 (3.26–3.26) | 2.90 (2.90–2.91) | 4.43 (4.41–4.44) |
| one-shot merge | 3 | 3.22 (3.22–3.23) | 2.87 (2.86–2.88) | 4.39 (4.39–4.40) |
| one worker on all the data | 3 | 2.77 (2.77–2.77) | 1.43 (1.42–1.45) | 4.89 (4.86–4.92) |
| one worker on all the data, a shard worker's steps | 3 | 2.84 (2.83–2.84) | 1.60 (1.59–1.60) | 4.70 (4.69–4.71) |
| 10 rounds of averaging | 3 | 3.00 (3.00–3.00) | 2.42 (2.42–2.42) | 4.44 (4.44–4.44) |

By shard (seed 42), test: sideband 1 2.91, the carrier 2.99, sideband 2 3.71. The sideband 2
worker barely leaves θ₀ on validation (4.00), so early stopping keeps it close to its start,
and its equal weight dilutes every merge.

Validation after each of the ten rounds (seed 42): 3.85, 3.70, 3.57, 3.45, 3.34, 3.24, 3.17,
3.10, 3.05, 3.00. It was still falling at the end.

Every worker and every merge passed the gate. The tests check that the gate rejects poisoned,
non-finite, out-of-bound and surrogate deltas.

### What this shows

- **One round of averaging is no better than the best worker, as far as this test set can
  tell.** It scores 2.87 against 2.90, a difference this test set can't resolve (144
  targets, within one session).
- **Repeated rounds help.** Ten rounds of five epochs reach 2.42, clearly better than any
  worker, and were still improving.
- **One worker on all the data is far better than any merge: 1.43.** Given only as many
  optimizer steps as a shard worker, it still reaches 1.60. So the gap isn't mainly steps. One
  model that sees the carrier and both sidebands in every batch learns much more than three
  models that each see one signal, averaged afterwards. This experiment shows the gap; it
  doesn't show its cause. Interference between the shards' task vectors, which TIES and DARE
  are designed to reduce, is one hypothesis; dilution by a weak shard and too few rounds are
  others. The next part tests it; it doesn't assume it.
- **The seed hardly matters; the shard does.** Every row's range across seeds is within 0.02
  bits. Which slice a worker trains on changes its score by up to 0.8 bits.
- **The off-source control:** the merged models stay near θ₀ on off-source noise (4.39–4.44
  against 4.28). The full-data workers score worse there (4.70–4.89): they've specialized
  further on Voyager's structure, and spread less probability over noise. The control isn't
  used to choose anything.

### Limitations

- **One session, three scans.** Validation and test each have 144 targets, so differences of a
  few hundredths of a bit are noise.
- **Three seeds,** and shards of one slice each: non-IID by construction. Shards that each
  mixed all three signals might merge better, and volunteers' data wouldn't be split this
  way.
- **Not converged:** the rounds were still improving, and so were the M1 workers at 50 epochs.

Environment: Python 3.13.17, NumPy 2.5.4, PyTorch 2.14.1, CPU. The benchmark is deterministic:
rerunning it reproduces every number.

## Next

- **The rest of milestone 2:** DARE compression, TIES sign election and DiLoCo's outer
  momentum, compared with this baseline on the same shards.
- **Milestones 3 to 5:** the mixture-of-experts upcycling, the browser, and the full benchmark.
