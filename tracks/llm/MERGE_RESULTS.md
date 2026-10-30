# Merging volunteer runs

[Issue #20](https://github.com/davidmdrpi/EIatHome/issues/20) merges small `SignalLM` runs
from many volunteers into stronger models. Milestone 1 gave every run a shared anchor (a pinned
codebook and starting weights θ₀; see [anchors/](anchors/README.md)), so their changes
τ = θ − θ₀ line up. This file records what merging does on the published data, milestone by
milestone.

Everything here is one 2020 observing session: training, validation and test are three scans
of the same cadence (`--allow-within-session`). It isn't evidence of generalization across
sessions, or of language.

## Milestone 2: merging within a family

Milestone 2 has two parts:

- **First part:** the plainest merge that can be measured. The review of #21 asked for it
  before anything cleverer: does averaging workers trained on different data beat those
  workers, and how does it compare with one worker trained on all their data?
- **Second part:** the methods issue #20 names, on the same workers:
  - DARE compression into a result card's budget;
  - TIES merging;
  - a greedy-soup gate;
  - DiLoCo's outer momentum across rounds.

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
- **Controls:** the off-source family is split by scan like Voyager.
  - Its validation scan is the **selection control**, which the gate may read.
  - Its test scan is the **reporting control**: the table's last column, scored afterwards,
    like the test split.
  - The codebook was fitted on its training scan, which no model trains on.
- **The gate** ([`merge.py`](merge.py)):
  - every change must be finite and under 10 in size;
  - each worker, and then the merge, has to score within 0.02 bits per token of the incumbent
    on validation;
  - neither may score more than 0.02 bits better than the untrained θ₀ on the selection
    control.

  A round never reads the test split or the reporting control.
- **Compared, per seed:**
  - the three shard workers, each trained 50 epochs from θ₀;
  - the best of them by validation;
  - one-shot merges of those three workers, one round from θ₀, by each rule below;
  - one worker on all three slices for 50 epochs: the same data and epochs as the shards
    together;
  - ten rounds: each round, the three shards train 5 epochs from the last merge, by each
    chain's rule below.
- **Added after the first run of the first part**, as a control for optimizer steps: one
  worker on all three slices for 20 epochs. A shard's 48 training windows make 2 batches an
  epoch and all three slices' 144 make 5, so 20 epochs take as many steps as one shard
  worker's 50.

The rules:

- **Average (FedAvg):** the first part's baseline. An average weighted by training targets,
  counted by the round from each worker's shard: 48 per shard, so equal weights.
- **The second part's settings**, fixed before its first run and taken from the papers, not
  tuned:
  - **DARE:** a random drop of p = 0.9, the issue's estimate for a card, with survivors scaled
    by 10. Each worker's delta is sent as int8 values with gap-coded indices; the round merges
    and gates what the payload rebuilds.
  - **TIES:** each delta trimmed to its top 20% by magnitude, the TIES paper's default. The
    signs are then elected, and only the entries that agree are averaged. DARE-TIES uses DARE's
    drop in place of the trim.
  - **Greedy soup:** workers are tried best first by validation, and each is kept only if the
    merge scores lower on validation with it.
  - **DiLoCo:** an outer Nesterov step with lr 0.7 and momentum 0.9, the DiLoCo paper's
    settings, carried across the ten rounds.
- **The ten-round chains:** the plain average; the average with DiLoCo; and DARE-TIES with the
  greedy soup and DiLoCo, the full method the issue describes.

```bash
python -m tracks.llm.merge_bench --output runs/merge-bench   # about 3 minutes on a CPU
```

### Results

Bits per token, mean over the three seeds (range in brackets). Uniform over 16 tokens is 4.

| | Runs | Validation | Test | Off-source control |
|---|---:|---:|---:|---:|
| θ₀, untrained | 1 | 4.00 | 3.78 | 4.24 |
| shard workers | 9 | 3.53 (3.26–4.00) | 3.20 (2.90–3.71) | 4.39 (4.24–4.57) |
| best worker (by validation) | 3 | 3.26 (3.26–3.26) | 2.90 (2.90–2.91) | 4.38 (4.36–4.40) |
| one-shot: average (FedAvg) | 3 | 3.22 (3.22–3.23) | 2.87 (2.86–2.88) | 4.34 (4.34–4.35) |
| one-shot: average of compressed deltas (DARE) | 3 | 3.27 (3.25–3.29) | 2.98 (2.94–3.04) | 4.43 (4.42–4.45) |
| one-shot: average, greedy soup | 3 | 3.04 (3.03–3.04) | 2.65 (2.64–2.65) | 4.43 (4.42–4.44) |
| one-shot: TIES | 3 | 3.36 (3.35–3.37) | 3.11 (3.09–3.13) | 4.30 (4.29–4.31) |
| one-shot: DARE-TIES | 3 | 3.22 (3.02–3.43) | 2.93 (2.80–3.09) | 5.09 (5.00–5.17) |
| one-shot: DARE-TIES, greedy soup | 3 | 3.19 (3.02–3.35) | 3.00 (2.80–3.29) | 5.02 (4.88–5.17) |
| one worker on all the data | 3 | 2.77 (2.77–2.77) | 1.43 (1.42–1.45) | 4.85 (4.82–4.88) |
| one worker on all the data, a shard worker's steps | 3 | 2.84 (2.83–2.84) | 1.60 (1.59–1.60) | 4.65 (4.65–4.66) |
| 10 rounds: average (FedAvg) | 3 | 3.00 (3.00–3.00) | 2.42 (2.42–2.42) | 4.40 (4.39–4.40) |
| 10 rounds: average + DiLoCo | 3 | 2.85 (2.84–2.85) | 1.83 (1.83–1.83) | 5.00 (4.99–5.02) |
| 10 rounds: DARE-TIES, greedy soup + DiLoCo | 3 | 2.83 (2.79–2.86) | 2.15 (2.09–2.24) | 5.03 (4.83–5.14) |

The validation scores of the greedy-soup rows are optimistic: the soup chose its members on
those same 144 targets. Test is the fair comparison.

**The shards.** Test scores by shard (seed 42): sideband 1 2.91, the carrier 2.99,
sideband 2 3.71.

- The sideband 2 worker's best validation epoch is the first, in every seed. Early stopping
  keeps it next to θ₀: its delta's norm is 0.09, against 2.3–2.7 for the other two.
- The two strong deltas are nearly orthogonal: their cosine similarity is 0.14, and their
  signs agree on 56% of parameters (50% would be chance). The seeds give the same numbers to
  two decimals.

**The greedy soup** set sideband 2 aside in every seed's one-shot average, and merged
sideband 1 with the carrier. Under DARE-TIES, it kept all three workers in seeds 43 and 44.
In seed 42 it kept only the carrier: better on validation (3.35 against 3.43), worse on test
(3.29 against 3.09).

**DARE payloads:** all 117 payloads (one-shot and chains) took 7,641 to 8,370 bytes, mean
7,976, about half the 16,384-byte card budget. Without dropping, a delta takes about 71 KB.

**The ten rounds**, the merge proposed each round, on validation (seed 42):

- **average:** 3.85, 3.70, 3.57, 3.45, 3.34, 3.24, 3.17, 3.10, 3.05, 3.00. All accepted, still
  falling.
- **average + DiLoCo:** 3.80, 3.55, 3.30, 3.11, 2.99, 2.92, 2.88, 2.85, 2.84, 2.85. All
  accepted; flat by round 9.
- **DARE-TIES, greedy soup + DiLoCo:** 3.50, 3.06, 2.92, 2.86, 2.86, then about 2.92 in each of
  rounds 6 to 10, all rejected by the gate.
  - The chain stalled in every seed. The gate rejected rounds 6–10 in seed 42, 5–9 in seed 43
    and 7–10 in seed 44.
  - By then the soup's merge barely improves on the incumbent: 2.864 against 2.865 in seed
    42's round 6. The momentum step overshoots it by about 0.05 bits.
  - A rejected round keeps the incumbent and its momentum, so the next round repeats the same
    overshoot.

**The control check:** one worker was rejected for finding structure in the off-source
control. It was the carrier worker in seed 44's round 2 of the DARE-TIES chain: it predicted
the noise 0.035 bits better than θ₀.

- **Changed after the second part's first run:** the check compared each worker with the
  incumbent, not with θ₀.
- **What the first run showed:** that version rejected 8 workers, all in the DARE-TIES
  chain. Each scored more than 0.02 bits better on the noise than the incumbent it started
  from.
  - Training on Voyager makes noise less predictable, not more. A worker starting from an
    over-specialized incumbent (5.1 bits on the noise, worse than uniform) could
    legitimately move back.
  - Every other number in the table came out the same in both runs; that chain's test score
    moved from 2.17 to 2.15.

Every other worker and merge passed the checks. The tests check that the gate rejects the
following:

- poisoned, non-finite, out-of-bound and surrogate deltas;
- relabelled and inflated shards;
- deltas over the card budget;
- an off-source delta grafted onto a Voyager worker.

### What this shows

- **The greedy soup is the clearest one-shot gain:** 2.65 on test, against 2.87 for the plain
  average and 2.90 for the best worker.
  - It keeps a worker that barely moved, sideband 2's, from diluting the other two. That
    happened in every seed.
- **DiLoCo's outer momentum is the largest gain over rounds:** 1.83 on test after ten rounds,
  against 2.42 for plain averaging.
  - One worker on all the data, given as many optimizer steps as each shard worker, reaches
    1.60. So the gap to it shrinks from 0.82 bits to 0.23.
  - Averaging rounds were still improving at round 10; the DiLoCo chain had flattened.
- **TIES didn't help here.** Trimmed to 20% with sign election, it scores 3.11, worse than
  the plain average.
  - The shard deltas are nearly orthogonal, and disagree in sign on 44% of parameters. Sign
    election and trimming discard much of each delta.
  - So this run gives no support to sign interference as the main cause of the gap to the
    full-data worker.
  - What helped matches two other explanations better. The soup removes a weak worker's
    dilution, and DiLoCo takes larger steps than an average of near-orthogonal deltas, each
    shrunk to a third.
  - That is one dataset with three shards, not a general finding.
- **Compression costs a little:** averaging the 8 KB DARE payloads instead of the full deltas
  scores 2.98 on test against 2.87 (validation 3.27 against 3.22).
- **DARE-TIES is mixed.** It scores 2.93 on test, between 2.80 and 3.09 depending on the
  seed. It moves much further on the off-source control (5.09).
  - Survivors scaled by 10 and a mean taken only over the nonzero entries make it behave more
    like a sum of the deltas than an average.
- **The full method was worse than its simplest part.** DARE-TIES with the soup and DiLoCo
  scores 2.15 on test, against 1.83 for averaging with DiLoCo. Its validation score is
  slightly better (2.83 against 2.85).
  - Choosing soups on 144 validation targets every round seems to fit validation rather than
    the signal.
  - It also stalls: once the soup stops improving, every momentum step overshoots.
  - Resetting the momentum when the gate rejects a round (an adaptive restart) is an obvious
    remedy. It's untested here.
- **The off-source control:**
  - The averages and TIES stay near θ₀ on the noise (4.30–4.43 against 4.24).
  - The DiLoCo chains and the full-data workers score 4.65–5.03, and DARE-TIES 5.02–5.09.
    They've specialized further on Voyager's structure and put less probability on noise.
  - Only the selection control is used in choosing; the reporting control isn't.
- **The seed hardly matters; the shard does.** Test ranges across seeds are within 0.05
  bits, except for the DARE rows, whose random drop depends on the weights. Which slice a
  worker trains on changes its test score by up to 0.8 bits.

### Limitations

- **One session, three scans.** Validation and test each have 144 targets, so differences of a
  few hundredths of a bit are noise.
- **Three seeds,** and shards of one slice each: non-IID by construction. Volunteers' data
  wouldn't be split this way.
- **Untuned settings.** DARE's p, TIES's 20% and DiLoCo's lr and momentum come from papers on
  far larger models. They were chosen once, not swept, so a method that lost here might do
  better with other settings. A sweep would choose on validation only.
- **Not converged:** the averaging rounds were still improving, and so were the 50-epoch
  workers.
- **The gate isn't a defense against adversaries.** The validation data and the control are
  public, so a delta crafted to pass them could pass.

Environment: Python 3.13.17, NumPy 2.5.4, PyTorch 2.14.1, CPU. The benchmark is deterministic:
rerunning it reproduces every number.

## Next

- **Milestone 3:** the mixture-of-experts upcycling across families.
- **Milestone 4:** the browser's anchor and summary cards.
- **Milestone 5:** the full benchmark, adding the mixture of experts.
- **Worth trying before then:**
  - the average with the greedy soup and DiLoCo, the two parts that helped, chosen on
    validation;
  - a momentum restart when the gate rejects a round.
