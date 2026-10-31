# Merging volunteer runs

[Issue #20](https://github.com/davidmdrpi/EIatHome/issues/20) merges small `SignalLM` runs
from many volunteers into stronger models. Milestone 1 gave every run a shared anchor (a pinned
codebook and starting weights θ₀; see [anchors/](anchors/README.md)), so their changes
τ = θ − θ₀ line up. This file records what merging does on the published data, milestone by
milestone: within one family (milestone 2), then across families (milestone 3).

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
  - Its validation scan is the **selection control**. A round records it for every model it
    scores, as a diagnostic: the bits, and how much the model gains from the real order of
    the scan's tokens over 19 shuffled orders, with a permutation p-value. Nothing is chosen
    by it.
  - Its test scan is the **reporting control**: the table's last column, scored afterwards,
    like the test split.
  - The codebook was fitted on its training scan, which no Voyager model trains on.
- **The gate** ([`merge.py`](merge.py)):
  - every change must be finite and under 10 in size, and so must a DARE delta as rebuilt,
    after its rescaling;
  - each worker, and then the merge, has to score within 0.02 bits per token of the incumbent
    on validation.

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
python -m tracks.llm.merge_bench --output runs/merge-bench   # about 5 minutes on a CPU
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
| 10 rounds: DARE-TIES, greedy soup + DiLoCo | 3 | 2.84 (2.83–2.86) | 2.11 (1.99–2.24) | 5.00 (4.83–5.14) |

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

**DARE payloads:** all 117 payloads (one-shot and chains) took 7,641 to 8,233 bytes, mean
7,970, about half the 16,384-byte card budget. None was rejected, by size or by the bound on
the rebuilt delta. Without dropping, a delta takes about 71 KB.

**The ten rounds**, the merge proposed each round, on validation (seed 42):

- **average:** 3.85, 3.70, 3.57, 3.45, 3.34, 3.24, 3.17, 3.10, 3.05, 3.00. All accepted, still
  falling.
- **average + DiLoCo:** 3.80, 3.55, 3.30, 3.11, 2.99, 2.92, 2.88, 2.85, 2.84, 2.85. All
  accepted; flat by round 9.
- **DARE-TIES, greedy soup + DiLoCo:** 3.50, 3.06, 2.92, 2.86, 2.86, then about 2.92 in each of
  rounds 6 to 10, all rejected by the gate.
  - The chain stalled in every seed. The gate rejected rounds 6–10 in seeds 42 and 44, and
    5–9 in seed 43.
  - By then the soup's merge barely improves on the incumbent: 2.864 against 2.865 in seed
    42's round 6. The momentum step overshoots it by about 0.05 bits.
  - A rejected round keeps the incumbent and its momentum, so the next round repeats the same
    overshoot.

**The selection control, a diagnostic.** For every model a round scored (540 incumbents,
workers and merges), and for a positive control: one worker per seed trained 50 epochs on the
off-source family itself. Bits on the off-source validation scan, and the gain from its real
order over 19 shuffles (p = 0.05 is the floor: the real order beat all 19):

| | Bits | Gain from order | p |
|---|---:|---:|---:|
| θ₀, untrained | 4.30 | 0.06 | 0.15 |
| every model in a round | 4.26–5.34 | 0.05–0.16 | 0.05–0.30 |
| one worker on all the data | 4.82–4.88 | 0.09 | 0.05 |
| a worker trained on the off-source family | 2.50 | −0.003 to 0.004 | 0.45–0.50 |

- **The likelihood measures frequencies, not structure.** The off-source worker predicts the
  scan far better than any Voyager model (2.50 bits against 4.30 for θ₀) but gains nothing
  from its order: it learned which tokens are common.
- **The order gain doesn't single out off-source training either.** Voyager models gain a
  little from the real order (up to 0.16 bits), and the most specialized beat every shuffle.
  The scan may have some order that Voyager's sequence priors pick up; the off-source worker,
  trained on it, doesn't use it.
- **So neither number is a veto.** On this data, neither separates a model trained on the
  noise from a Voyager model.
- **History:** until the review of the second part, the gate vetoed on the likelihood.
  - First, against the incumbent: that rejected 8 Voyager workers in the DARE-TIES chain, whose
    incumbent had specialized to 5.1 bits on the noise, so workers from it legitimately moved
    back.
  - Then, against θ₀: that rejected one, the carrier worker in seed 44's round 2, for scoring
    0.035 bits better than θ₀. The review pointed out that θ₀ is a random predictor, worse
    than uniform (4 bits) on this scan. Beating it shows better calibration or shared token
    frequencies, not that the worker found structure. The table above bears that out.
  - With no veto, that worker is accepted. The DARE-TIES chain's test score moves from 2.15
    to 2.11 (seed 44: 2.13 to 1.99); every other number is unchanged.

No worker in this run was rejected, by any check. The tests check that the gate rejects:

- poisoned, non-finite, out-of-bound and surrogate deltas;
- relabelled and inflated shards;
- deltas over the card budget, or whose rebuilt delta exceeds the bound. A delta of 9 is
  within it; after DARE's rescaling it rebuilds near 90.

They also check that a context-free predictor, calibrated to the off-source scan's token
frequencies, gains nothing from its order and isn't rejected.

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
  scores 2.11 on test (1.99–2.24), against 1.83 for averaging with DiLoCo. Its validation
  score is about the same (2.84 against 2.85).
  - Choosing soups on 144 validation targets every round seems to fit validation rather than
    the signal.
  - It also stalls: once the soup stops improving, every momentum step overshoots.
  - Resetting the momentum when the gate rejects a round (an adaptive restart) is an obvious
    remedy. It's untested here.
- **The off-source control:**
  - The averages and TIES stay near θ₀ on the noise (4.30–4.43 against 4.24).
  - The DiLoCo chains and the full-data workers score 4.65–5.00, and DARE-TIES 5.02–5.09.
    They've specialized further on Voyager's structure and put less probability on noise.
  - Nothing is chosen by either control.
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
- **No structure test.** The issue asks the gate to reject deltas that find structure in the
  off-source control. On these scans no measure tried here separates training on the noise
  from training on Voyager, so there's no veto. Setting a justified threshold would need a
  control with known temporal structure, such as injected or real RFI.
- **The gate isn't a defense against adversaries.** The validation data is public, so a delta
  crafted to pass the gate could pass it.

Environment: Python 3.13.17, NumPy 2.5.4, PyTorch 2.14.1, CPU. The benchmark is deterministic:
rerunning it reproduces every number.

## Milestone 3: upcycling family specialists into a mixture of experts

Milestone 3 combines specialists for different families, Branch-Train-MiX style
([`moe.py`](moe.py)):

- each specialist branches from one θ₀ and trains on one family;
- their feed-forward layers become the experts of one model;
- the rest (embeddings, attention, layer norms, head) is merged;
- a router learns which experts each token uses.

The question: does that beat one dense model trained on all the families?

### What it needed first

- **A third family.** The Python track had only Voyager and off-source. [`synthetic.py`](synthetic.py)
  ports the Live Lab's Rosetta control: English text sent as 16-tone FSK, one 4-bit symbol per
  frame, in the real slices' units (noise sigma).
  - It uses the same pinned corpus as the browser (`en-gettysburg-bliss+us-preamble/v1`).
  - Like the real families, it's three scans of three 49-frame slices, so training,
    validation and test hold different stretches of the text.
  - The slices are generated once and published with their SHA-256s, in
    `data/slices/index-rosetta.json`: v1's manifest plus these nine. v1 and milestone 2 are
    unchanged.
- **A codebook that covers it.** v1's codebook, fitted on real frames only, recovers a
  Rosetta frame's symbol from its token for only 71% of frames.
  - **Anchor v2** fits 32 tokens on the training scans of all three families. That's the
    smallest of 16, 24, 32 and 48 for which the token gives the symbol for at least 95% of
    Rosetta training frames (96%).
  - It also keeps both real families at least as finely tokenized as v1: Voyager's training
    frames use 24 tokens (12 under v1), off-source's 6 (6).
  - That second condition was added after seeing that at 16 tokens every off-source frame
    gets the same one.
- **A ceiling on Rosetta.** k-means spends spare tokens splitting frequent symbols by their
  noise. Symbol 6, the high nibble of most lowercase letters, gets up to five tokens.
  - Given its symbol, a Rosetta token still carries 1.0 bit of noise, out of 3.1 bits of
    symbol entropy, and no model can predict that part.

### Setup

Fixed before the first run ([`moe_bench.py`](moe_bench.py)):

- **Data:** anchor v2; Voyager, off-source and Rosetta, each split into three scans within
  one session.
- **Training:** the real stream, stride 1, 50 epochs, the defaults otherwise; seeds 42, 43
  and 44.
- **Compared, per seed:**
  - **Specialists:** three, one per family, each trained from θ₀ on its family's training
    scan.
  - **One dense model on all three families:** trained from θ₀ on all three training scans,
    chosen on all three validation scans. The same data and epochs as the specialists
    together, with at most 700 optimizer steps, against at most 3 × 250 for the specialists.
  - **Specialists merged into one dense model:** averaged, or by TIES (top 20%).
  - **The mixture** (61,120 parameters, 44,416 used per token): the three specialists'
    feed-forward layers as experts, with a top-2 router and the rest averaged. Each expert
    runs only on the tokens routed to it. Scored four ways:
    - with the router untrained;
    - with each family sent to its own expert, an oracle that uses the family label;
    - with the router finetuned on the three training scans, experts and backbone frozen;
    - with the router and backbone finetuned, experts frozen.
  - **The same mixture with a TIES-merged backbone,** router and backbone finetuned.
- **Finetuning:** the training settings, plus a load-balancing weight of 0.01, chosen on the
  three validation scans.

```bash
python -m tracks.llm.moe_bench --output runs/moe-bench   # about 2 minutes on a CPU
```

### Results

Test bits per token, mean over the three seeds (range when it's 0.01 or more). Uniform over
32 tokens is 5.

| | Voyager | Off-source | Rosetta | Mean | Validation mean |
|---|---:|---:|---:|---:|---:|
| θ₀, untrained | 5.11 | 4.79 | 5.44 | 5.11 | 5.15 |
| Voyager specialist | **2.04** (2.03–2.04) | 5.44 (5.41–5.46) | 5.83 (5.80–5.84) | 4.43 (4.42–4.44) | 4.80 (4.78–4.81) |
| off-source specialist | 6.21 (6.18–6.26) | 2.09 (2.08–2.09) | 6.71 (6.67–6.73) | 5.00 (4.98–5.03) | 4.90 (4.88–4.92) |
| Rosetta specialist | 6.40 (6.39–6.41) | 5.12 (5.11–5.13) | **4.22** (4.21–4.23) | 5.25 (5.24–5.25) | 4.99 (4.98–4.99) |
| **one dense model on all three families** | 2.13 (2.12–2.14) | 2.06 (2.05–2.07) | 4.46 (4.42–4.50) | **2.88** (2.87–2.89) | **3.16** |
| specialists averaged (dense) | 3.92 (3.91–3.92) | 3.44 (3.43–3.44) | 5.04 (5.03–5.04) | 4.13 | 4.18 (4.17–4.18) |
| specialists merged by TIES (dense) | 4.21 (4.19–4.24) | 3.67 (3.49–3.77) | 5.32 (5.30–5.34) | 4.40 (4.35–4.42) | 4.40 (4.35–4.43) |
| mixture, untrained router | 3.90 | 3.64 (3.64–3.65) | 4.98 (4.96–4.99) | 4.17 | 4.22 (4.22–4.23) |
| mixture, each family to its own expert (oracle) | 2.62 (2.61–2.63) | 2.83 (2.82–2.83) | 4.43 (4.42–4.44) | 3.29 (3.29–3.30) | 3.47 (3.46–3.48) |
| mixture, router finetuned | 3.29 (3.27–3.30) | 2.88 (2.87–2.89) | 4.74 (4.73–4.75) | 3.63 (3.63–3.64) | 3.82 (3.82–3.83) |
| mixture, router and backbone finetuned | 2.37 (2.32–2.39) | **2.05** (2.05–2.07) | 4.46 (4.43–4.50) | 2.96 (2.95–2.97) | 3.26 (3.25–3.26) |
| mixture (TIES backbone), router and backbone finetuned | 2.38 (2.34–2.40) | **2.05** (2.05–2.06) | 4.44 (4.41–4.48) | 2.96 (2.95–2.96) | 3.31 (3.30–3.32) |

Each family's simple baselines on test, fitted on its training scan:

| | Unigram | Markov-1 |
|---|---:|---:|
| Voyager | 3.98 | 3.02 |
| Off-source | 2.10 | 2.60 |
| Rosetta | 4.28 | 4.40 |

**Routing** (seed 42): the share of each family's test tokens whose first choice is its own
expert, in blocks 1 and 2.

| | Voyager | Off-source | Rosetta |
|---|---:|---:|---:|
| router finetuned | 0.72, 0.65 | 1.00, 0.99 | 0.71, 0.71 |
| router and backbone finetuned | 0.49, 0.44 | 0.99, 0.99 | 0.83, 0.75 |

In the second case, Voyager's tokens go about equally to the Voyager and Rosetta experts.

Over the three seeds, averaged over the two blocks:

- off-source goes to its own expert 0.99–1.00 of the time in the mixtures with an averaged
  backbone, and 0.83–0.98 with TIES;
- Voyager and Rosetta go to theirs 0.47–0.80 of the time.

**Training lengths:**

- the router alone was still improving at its 50-epoch limit, in every seed;
- with the backbone, the best epochs were 21–26;
- the dense model's were 28–31.

### What this shows

- **The mixture doesn't beat one dense model trained on all the families.** Its best
  version, with the router and backbone finetuned, scores 2.96 bits per token on average,
  against 2.88. On validation it's 3.26 against 3.16.
  - It is 2.2 times the parameters, 1.6 times those used per token.
  - Per family, it matches the dense model on off-source (2.05 against 2.06) and Rosetta
    (4.46 against 4.46), and is worse on Voyager (2.37 against 2.13).
- **The merged backbone is what costs most.**
  - Even with every token sent to its family's own expert, the mixture scores 3.29, against
    2.78 for the three specialists on their own families. The averaged attention and
    embeddings serve no family well.
  - A router finetuned alone can't fix that, though it learns the routing: off-source tokens
    go to their expert almost every time, Voyager's and Rosetta's about 70%. It scores 3.63.
  - Finetuning the backbone too recovers most of the loss. TIES merging the backbone instead
    of averaging makes no difference (2.96).
- **Specialists stay best on their own family.** Voyager's scores 2.04 there, Rosetta's
  4.22. Each is useless on the others: 5.1 to 6.7 bits, worse than θ₀. The families are very
  different, which is the case a mixture is meant for. Here, though, a single dense model of
  the same size as one specialist fits all three nearly as well.
- **Merging across families as one dense model fails:** 4.13 averaged, 4.40 by TIES, little
  better than θ₀. That's why the feed-forward layers are kept apart.
- **Rosetta's language isn't learned yet.** Its specialist scores 4.22 against a unigram
  model's 4.28: from 144 training targets, it learned little beyond token frequencies.
  - Part of the reason is the ceiling above: about one bit of each token is noise.
  - The positive control for language-like structure isn't positive at this size. Every
    model trained on Rosetta scores between 4.22 and 4.46, apart from the router-only mixture
    (4.74).
- **Off-source is pure token frequencies:** every model that has seen it scores about its
  unigram baseline (2.05–2.10).

### Limitations

- **Tiny data, one session:** 144 test targets per family, three seeds. Differences of a few
  hundredths of a bit are noise.
- **Two of the three families have little sequential structure at this size:** off-source is
  noise, and Rosetta's text is hidden behind tokens that are a third noise. A mixture's case
  rests on experts that each know something the others don't. Here only Voyager's expert
  clearly does.
- **One configuration, untuned:**
  - **Routing:** top-2 over three experts. With three experts that leaves only one out per
    token.
  - **Finetuning:** the router alone was stopped by the epoch limit, not by validation.
  - **Expert finetuning:** none; the experts stay frozen throughout.
  - **Router initialization:** a single router seed.
- **The dense comparison has fewer parameters.** A dense model with as many parameters as the
  mixture isn't measured.

Environment: Python 3.13.17, NumPy 2.5.4, PyTorch 2.14.1, CPU. The benchmark is deterministic:
rerunning it reproduces every number.

## Next

- **Milestone 4:** the browser's anchor and summary cards.
- **Milestone 5:** the full benchmark, adding the mixture of experts.
- **Worth trying:**
  - the average with the greedy soup and DiLoCo, the two parts of milestone 2 that helped,
    chosen on validation;
  - a momentum restart when the gate rejects a round;
  - a control with known temporal structure, to calibrate a test for found structure;
  - for the mixture: more Rosetta text, or a tokenizer that doesn't split a symbol by its
    noise, so a second expert has sequential structure to bring.
