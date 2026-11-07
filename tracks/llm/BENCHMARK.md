# The session benchmark: merging volunteers' work on new observations

`signal-lm-sessions-bench/v0` ([session_bench.py](session_bench.py)) asks the question the merge
benchmarks in [MERGE_RESULTS.md](MERGE_RESULTS.md) couldn't. Do workers, each trained on
observing sessions no other worker saw, merge into a model that predicts **new sessions** better?

Every earlier merge result trains, selects and tests on scans of one 2020 Voyager session, so
none of them is evidence of generalization. Those benchmarks stay as they are, as smoke tests.
This one is frozen on 57 Green Bank sessions from 2016 to 2024, and it publishes the checkpoints
the next milestone's merge methods start from (weight-space programming among them).

## Summary

Bits per token on the 7 test sessions (lower is better): the mean over seeds 42, 43 and 44.

| Arm | Test | Worst test session | Validation |
|---|---:|---:|---:|
| θ₀, untrained | 5.33 | 5.70 | 5.39 |
| token frequencies (unigram), fitted on the training sessions | 2.090 | 3.92 | 2.422 |
| first-order Markov chain, fitted on the training sessions | 1.488 | 3.06 | 1.612 |
| base (shard 1, 5 sessions) | 1.665 | 3.46 | 1.745 |
| one worker (a volunteer's 4–5 new sessions, from the base): mean of 4 | 1.558 | 3.17 | 1.687 |
| best worker, chosen on validation | **1.344** | **2.42** | 1.558 |
| one-shot merges of the 4 workers: plain average | 1.500 | 3.06 | 1.648 |
| — greedy soup (keeps only the best worker) | 1.344 | 2.42 | 1.558 |
| — TIES, keeping 20% | 1.490 | 3.04 | 1.607 |
| 10 rounds of the 4 workers: plain average | 1.358 | 2.58 | 1.558 |
| — average + DiLoCo | 1.385 | 2.92 | 1.550 |
| one model on all 23 training sessions, from θ₀ | 1.338 | 2.51 | 1.473 |
| — from the base | **1.329** | 2.53 | 1.510 |

**What it shows:**

- **The volunteers' data differ in value.** Shard 5's worker reaches 1.34 on test; the other
  three reach 1.57 to 1.66. Worker 3's best epoch was its first: its sessions didn't help on
  validation at all.
- **No one-shot merge beats the best worker.** The plain average and TIES dilute shard 5's
  contribution with the other three (each is weighted by its training targets). They end up
  about level with a first-order Markov chain fitted on the training sessions (1.50 and 1.49
  against 1.488). The greedy soup tries the workers best first and keeps only shard 5.
- **Ten rounds of plain averaging nearly match the best worker and the full-data model.** At
  1.358, they're within 0.03 bits of one model trained on every training session.
- **DiLoCo's outer momentum doesn't help here.** It's level on validation (1.550 against 1.558)
  but worse on test (1.385 against 1.358), and much worse on the worst session (2.92 against
  2.58). That's unlike the within-session benchmark, where it was the largest gain (MERGE_RESULTS.md).
- **The worst session tells a different story from the mean.** On HIP 90461's session
  (2023-06), every merge does worse than the best worker. On HIP 8962's (2022-04), the best
  worker does worse than the base (1.86 against 1.51), while the full-data models do best
  (1.30 to 1.37).

So the gap is not "merging can't work". The plain merges here can't tell a volunteer's useful
sessions from unhelpful ones, and the shard that helps on average isn't the one that helps on
every new session. That leaves room for selective, layer-wise or routed combination, which the
next benchmark tests against these baselines.

**What v0 is for now.** Its test scores have been seen, and its training budget was revised after
the first run (below). So v0 is a public development and replay benchmark: methods can be
built and compared on it, and its checkpoints reused. A method's claim to generalize needs
sessions nobody has looked at yet. For the next milestone, its editor and selection procedure
should be frozen first, then scored once on a fresh cohort of sessions (a new draw from the
pipeline index, a new split and a new benchmark version).

## The data and the split

The sessions corpus ([data/sessions](data/sessions/README.md), [sessions_corpus.py](sessions_corpus.py))
holds one on-target scan of a nearby star from each of 57 Green Bank sessions. They were drawn at
random from bldata.berkeley.edu's pipeline index before any data was read. Each scan gives up to
6 hits (family `stars`, mostly human interference) and 2 blind windows (family `blind-window`),
16 steps by 64 channels at 2.79 Hz. Anchor v4's peak tokenizer, which fits nothing, turns each
frame into one of 33 tokens.

The hits split by whole session, chronologically (`train.py`'s rule for three or more sessions):

| Split | Sessions | Slices | Targets | Observed |
|---|---:|---:|---:|---|
| Train | 23 | 84 | 1,260 | 2016-02 to 2020-03 |
| Validation | 7 | 37 | 555 | 2020-09 to 2021-12 |
| Test | 7 | 24 | 360 | 2022-04 to 2024-02 |

Every validation and test session was observed after every training session. The blind windows
of the corpus are the control family, split the same way: 11 sessions select, 11 report.
[split.json](benchmarks/sessions-v0/split.json) lists every session and slice by content hash,
and `session_bench.py` pins its SHA-256 (`4023ba4f…`). A different split is a new benchmark
version.

## The protocol

Fixed before the run reported above (the module's docstring has it in full):

- **The cohort.** The 23 training sessions go into 5 session shards (`--shard-unit session`: 5,
  5, 5, 4 and 4 sessions). Sessions are the manifest's, found among every family's slices, as
  `split_records` finds them: a blind window that links two hit scans keeps them in one shard. Shard 1 trains the base from θ₀, recorded as round 1 of a merge
  chain. Shards 2 to 5 are four volunteers, each training from the base on its own sessions. No
  two contributors share a session, and none trains on a held-out one.
- **Training.** Up to 1,000 epochs, stopping once validation hasn't improved for 50 and keeping
  the best epoch (lr 3e-4, batch 32), on the real token stream. In the chains, each round's
  workers train up to 100 epochs, with the round's seed.
- **Merging.** `merge.py`'s rules and gate: every worker and merge must be within 0.02 bits of
  the incumbent on validation. The one-shot merges start from the base; the chains continue its
  chain for 10 rounds. DiLoCo uses lr 0.7 and Nesterov momentum 0.9; TIES keeps 20%.
- **Baselines.** A unigram and a first-order Markov chain over the tokens, fitted on every
  training session with add-one smoothing (`train.py`'s), scored on the same sets.
- **Selection and reporting.** Everything is chosen on validation. Test is scored once
  everything is chosen: the mean over test targets, the worst test session, the blind-window
  control, and the order gain (how many fewer bits the model needs for the test tokens in their
  real order than shuffled within each slice; on average, every trained arm gains 0.26 to 0.35 bits).

**A change after the first run, recorded here.** The first run used 50 epochs and `train.py`'s
patience of 8. Every run's best epoch was its last, and both chains' validation was still
falling at round 10. That run measured how fast each arm trains, not what it learned. On
validation alone, the base kept improving until about epoch 190, and patience 8 stopped the
full-data model at epoch 109 where patience 50 found a better one at 207. So the budget became
1,000 epochs with patience 50. That run's test scores had been printed before the change:

| Arm (first run, 50 epochs) | Test |
|---|---:|
| base | 2.667 |
| best worker | 1.802 |
| one-shot plain average | 1.867 |
| 10 rounds, average + DiLoCo | 1.519 |
| all training sessions, from θ₀ / from the base | 1.538 / 1.456 |

Undertrained, DiLoCo looked like the winner, because its momentum trains faster. Trained to
convergence, it isn't.

## Costs

Per seed, on one CPU thread (`report.json` records each run):

- **Training.** The base takes 3.5 s; the four workers 5 s together; each chain about 36 s over
  its 10 rounds; one full-data model about 9 s.
- **Communication.** A contribution is a float32 delta of 27,584 parameters, 110 KB. A one-shot
  merge receives 4 (441 KB); a 10-round chain 40 (4.4 MB).
- **Merging and inference.** A one-shot merge, with its gate, takes 0.5 s. Every arm is the same
  model, so they all cost the same at inference.

## Seeds are not replicates

Every run starts from the same θ₀ or base, and a worker's 225 training targets fit in one batch.
A seed changes only dropout and the order of a few batches, so the three seeds agree to within
0.05 bits. They show that the result reproduces, not that it would hold on other sessions. The
uncertainty that matters is the 7 test sessions: the per-session scores in `report.json` vary by
more than a factor of 10, from 0.08 to 3.5 bits. More seeds can't substitute for more sessions;
a larger corpus can (the draw takes `--sessions`).

## What's published

[benchmarks/sessions-v0](benchmarks/sessions-v0):

- `split.json`, pinned: every session and slice of each split, the control's, and the cohort's
  shards.
- `report.json`: the whole run, not just the summary. It has every arm's scores (per test
  session too), every round's contributors and decisions, every training command, its seconds
  and best epoch, and the weights digests. It also records the environment (Python 3.13.17,
  NumPy 2.5.3, PyTorch 2.15.0+cpu, Linux x86_64, 1 thread, deterministic algorithms) and the
  digests of the code, the corpus and the split.
- `checkpoints/seed{42,43,44}/`: the base and the four workers of each seed. Each is raw
  little-endian float32 weights in θ₀'s layout (110 KB), with a JSON header giving:
  - its role, seed and shard;
  - its round record: start digest, shard unit, training targets;
  - the anchor, corpus and split it belongs to;
  - its SHA-256.

```bash
python -m tracks.llm.session_bench verify                 # the split's pin; every seed's base and workers 2-5, all present,
                                                          # each one's hash, start and shard; the one-shot merges and the
                                                          # baselines recomputed against report.json
python -m tracks.llm.session_bench materialize --output runs/sessions-v0-checkpoints   # as .pt files merge.py and sample.py load
python -m tracks.llm.merge round --start runs/sessions-v0-checkpoints/seed42/base.pt \
  --anchor v4 --output runs/my-merge runs/sessions-v0-checkpoints/seed42/worker*.pt   # any rule, from the published cohort
python -m tracks.llm.session_bench run --output runs/sessions-bench                  # the whole benchmark again (about 5 minutes)
```

`materialize` rebuilds each checkpoint's splits from the corpus, so `merge.py` refuses a
published worker on another corpus or split as it would any other. The tests in CI run `verify`.
They also run the whole benchmark on a small synthetic corpus of 12 sessions, which checks:

- that no two contributors share a session, and none trains on a held-out one;
- the arms and their costs;
- that publishing round-trips;
- that a changed byte is refused.

**Replay.** Training is deterministic on one CPU thread with PyTorch's deterministic
algorithms. In the recorded environment, retraining the seed-42 base and worker 3, and the seed-43
worker 5, from their recorded commands reproduces their published weights bit for bit. Another
PyTorch version or CPU may not; the merges and scores `verify` replays are plain arithmetic on the
published weights.

## Limits

- **Small.** 1,260 training targets, 7 test sessions, 27,584 parameters. The benchmark is for
  comparing merge methods on equal footing, not for a model of the radio sky.
- **What the tokens are.** The hits are the strongest narrowband lines in a coarse channel,
  nearly all human interference. Bits per token measure how well a model predicts their
  trajectories in new sessions, not whether it would find a signal from elsewhere. The recovery
  benchmark ([tracks/recovery](../recovery/RESULTS.md)) is where detection is measured: no learned
  detector there beats de-Doppler yet.
- **One telescope.** Green Bank only, 2016 to 2024, four bands.
