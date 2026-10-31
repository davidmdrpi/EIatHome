# Merging volunteer runs

[Issue #20](https://github.com/davidmdrpi/EIatHome/issues/20) merges small `SignalLM` runs
from many volunteers into stronger models. Milestone 1 gave every run a shared anchor (a pinned
codebook and starting weights θ₀; see [anchors/](anchors/README.md)), so their changes
τ = θ − θ₀ line up. This file records what merging does on the published data, milestone by
milestone:

- within one family (milestone 2);
- across families (milestones 3 and 3b);
- in the browser (milestone 4);
- the open findings of all three, settled (milestone 5).

Milestones 2 to 3b, and milestone 5's Python results, use one 2020 observing session: training,
validation and test are three scans of the same cadence (`--allow-within-session`). The browser
uses its synthetic Rosetta text. None of it is evidence of generalization across sessions, or
of language.

## Summary

What issue #20 found, from milestone 1 to milestone 5. Test bits per token (lower is better),
the mean over seeds or browser units. Each milestone's section has the full tables, ranges and
caveats.

| Question | Milestones | Best result | Against | Verdict |
|---|---|---|---|---|
| Can runs share a starting point? | 1, 4 | pinned codebooks and θ₀ rebuild bit for bit: anchors v1–v3 in Python, `token-anchor/v1` in the browser | | the prerequisite holds |
| Merge shard workers once (Voyager, 3 shards) | 2 | greedy soup 2.65 | plain average 2.87; best worker 2.90; one worker on all the data 1.43 | beats every worker, far from training on all the data |
| Merge over ten rounds | 2, 5 | plain average + DiLoCo 1.83 | plain average 2.42; one worker on all the data with a shard worker's steps, 1.60 | DiLoCo's momentum is the largest gain |
| DARE-TIES in the ten-round chain | 2, 5 | 2.11 with DARE's 10× kept; 2.51 unbiased | the same chain without DARE or TIES, 2.34; without the soup too, 1.83 | the per-round soup causes the gap, not the rescale |
| Compress a delta for a card (DARE, p = 0.9) | 2, 4, 5 | about 8 KB (Python model), 1.4 KB (browser model) | averaged, it costs 0.11 bits one-shot | fits a card; costly inside TIES or from a random start |
| Mix families token by token (Branch-Train-MiX) | 3, 3b | 3.76, the best mixture | dense rivals 3.56–3.73 | loses |
| Keep family specialists whole, routed by sequence | 3b, 5 | 3.39 without the label (posterior mixture, chosen on validation); 3.43 committing after 8 tokens | 3.37 with the label; largest dense rival 3.56 | **wins, without the label** |
| Browser: 4 seeds on all the text | 4, 5 | plain average 2.31 | the kept seed 2.31 | nothing to merge |
| Browser: 4 disjoint shards from θ₀ | 5 | plain average 2.48 | the kept shard worker 2.49 (ahead in 6 of 10 units) | level |
| Browser: 4 shards from a 3-epoch warm start | 5 | TIES 2.43; plain average 2.46 | the kept worker 2.49 (each ahead in 9 of 10 units); one worker on all the text 2.31 | **beats the workers**, not the full-data worker |

**What holds up:**

- **Within a family, DiLoCo over rounds.** The plain average with Nesterov outer momentum
  closes most of the gap to a worker trained on all the data. A greedy soup helps once, to drop
  a weak worker.
- **Across families, keeping the specialists whole and routing whole sequences.** No family
  label is needed, and it beats parameter-matched dense models trained on twice the budget.
- **In the browser, merging shards from a shared warm start.** From a trained start, TIES and
  the plain average beat the workers.

**What doesn't:**

- **Token-level mixtures of experts** with merged backbones.
- **TIES or DARE from a random start**, where a task vector is the whole model.
- **DARE's rescale inside TIES's average over survivors.**
- **A greedy soup in every round.**
- **No merge in any setting reaches one model trained on all the data.**

Nothing here shows English learned beyond byte statistics. On Rosetta, a byte-statistics
control that knows the UTF-8 framing, scored over the 16 symbols, does as well as or better
than every model.

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

### Results (baseline M3a)

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

## Milestone 3b: experts with something to bring, a warm start, and fair dense rivals

Milestone 3's mixture lost to one dense model on all the families. Two bottlenecks were
diagnosed:

- **Only Voyager's expert knew anything.** Rosetta had 73 characters of training text, and
  tokens that were about a third noise. Its specialist barely beat a unigram model.
- **The merged backbone cost a lot even under oracle routing,** since the specialists had
  branched from random weights.

Milestone 3b fixes both and asks again, against dense models given at least the costliest
mixture's training budget, and parameter-matched ones. Baseline M3a, above, is unchanged.

### What changed

- **A tokenizer under which Rosetta's symbols are exact.**
  - **Exploration first, on training frames only.** k-means kept splitting symbols by their
    noise: H(token | symbol) stayed at 0.8–1.3 bits at fluxes from 30 to 3000, with 24 to 48
    centroids, with k-means++ and 8 restarts, and with a fixed flux. Soft-thresholding the
    noise got it down only to 0.2–0.4 bits.
  - **Anchor v3 uses a peak tokenizer instead.** A frame's token is its brightest channel's
    2-channel bin, or "no line" below 6σ: 33 tokens, nothing fitted. It was the only setting
    tried.
  - **The result:** H(token | symbol) = 0 on every Rosetta frame, training, validation and
    test.
- **Longer Rosetta:** 18 slices, six per scan (`index-moe-v2.json`), so each split holds about
  147 characters.
  - **Against frequency baselines:** the specialist beats a unigram model on test by 0.80–0.86
    bits (2.45–2.51 against 3.31), and Markov-1 (3.18) too.
  - **Against byte statistics:** that's better prediction of the encoded text, not evidence of
    English. UTF-8 alternates high and low nibbles, and byte statistics that know that framing
    (added after review; below) score 2.54 over the 33 tokens, 2.30 over the 16 symbols. That's
    as well as the specialist or better.
- **A third structured family instead of off-source.** Under the peak tokenizer, off-source
  noise is "no line" in every frame, nothing to be expert in. The Live Lab's RFI sweep, a line
  sweeping 8 to 20 channels per step and wrapping, takes its place. It was chosen before any
  run.
  - **It turned out weak:** a run that also printed test scores showed its specialist barely
    beats unigram (4.54–4.59 against 4.72). Three training slices, each with its own rate,
    aren't enough to learn the sweep.
  - **It was kept as chosen.** So two of the three experts, Voyager and Rosetta, have
    sequential structure.
- **Warm-start branching.** θ_warm is θ₀ trained 10 epochs on the three training scans
  together. Specialists branch from it (`train.py --warm`), and the mixture's backbone is
  θ_warm plus their averaged deltas.
  - The warm start records the slices it trained on, and is refused unless they are its
    families' training splits.
  - A mixture takes only specialists of one start.
- **Dense rivals:**
  - the anchor's size, 27,584 parameters, trained as milestone 3 did;
  - the anchor's size;
  - the mixture's active parameters: width 40, feed-forward 174, 44,428 parameters;
  - the mixture's total: width 48, feed-forward 199, 61,166 parameters.

  The last three train for at least as many training targets as the costliest mixture
  pipeline of the seed (745k–760k), with no early stopping; the best epoch on validation is
  kept.

### Setup

Fixed before the first run ([`moe_ablation.py`](moe_ablation.py)):

- **Data:** anchor v3 on `index-moe-v2.json`. Voyager, Rosetta and the RFI sweep are each split
  into three scans within a session.
- **Training:** the real stream, stride 1, the defaults otherwise; seeds 42–44.
- **Specialists:** 50 epochs, patience 8, from θ₀ (cold) or θ_warm (warm).
- **The mixture's finetuning:** 50 epochs, patience 8, load-balancing weight 0.01, chosen on
  the three validation scans.
- **Cost:** every training target a pipeline saw, warm-up, specialists and finetuning
  included.
- **Added after review, as a control for Rosetta: byte statistics with known framing.**
  - A high-nibble distribution, and each low nibble's distribution given its high nibble, fitted
    on the training scan with add-one smoothing.
  - On test, each frame's phase (high or low nibble) comes from its slice's symbol offset: an
    alignment no model is given.
  - No context beyond the byte.

```bash
python -m tracks.llm.moe_ablation --output runs/moe-ablation   # about 8 minutes on a CPU
```

### Results (M3b ablation)

Test bits per token, mean over the three seeds (range when it's 0.01 or more). Uniform over 33
tokens is 5.04.

| | Voyager | Rosetta | RFI sweep | Mean | Validation mean | Parameters (active) | Training targets |
|---|---:|---:|---:|---:|---:|---:|---:|
| θ₀, untrained | 5.36 | 5.26 | 5.62 | 5.42 | 5.43 | 27,584 | |
| specialist bank, dispatched by family label, cold | **3.13** (3.13–3.14) | 2.51 (2.50–2.52) | 4.59 (4.58–4.59) | 3.41 (3.40–3.42) | 3.24 | 82,752 (27,584) | 364k |
| specialist bank, dispatched by family label, warm | **3.12** (3.12–3.13) | **2.45** (2.44–2.45) | 4.54 (4.53–4.57) | **3.37** (3.36–3.38) | **3.22** (3.22–3.23) | 82,752 (27,584) | 383k |
| mixture, cold, oracle routing | 3.58 (3.57–3.59) | 2.84 (2.82–2.86) | 5.07 (5.04–5.08) | 3.83 | 3.76 (3.76–3.77) | 61,184 (44,480) | 364k |
| mixture, cold, router finetuned | 4.58 (4.58–4.59) | 3.27 (3.24–3.30) | 5.03 (5.02–5.04) | 4.30 (4.29–4.31) | 4.24 (4.23–4.25) | 61,184 (44,480) | 733k |
| mixture, cold, router and backbone finetuned | 3.74 (3.73–3.76) | 2.60 (2.58–2.62) | 4.94 (4.92–4.95) | 3.76 (3.75–3.77) | 3.67 (3.66–3.68) | 61,184 (44,480) | 647k |
| mixture, warm, oracle routing | 3.37 | 2.56 | 5.11 (5.09–5.13) | 3.68 (3.68–3.69) | 3.55 (3.54–3.55) | 61,184 (44,480) | 383k |
| mixture, warm, router finetuned | 4.13 (4.10–4.16) | 2.80 (2.79–2.80) | 5.07 (5.06–5.10) | 4.00 (3.99–4.02) | 3.89 (3.88–3.90) | 61,184 (44,480) | 752k |
| mixture, warm, router and backbone finetuned | 3.78 (3.77–3.78) | 2.57 (2.56–2.58) | 5.00 (4.98–5.01) | 3.78 (3.78–3.79) | 3.62 (3.61–3.63) | 61,184 (44,480) | 604k |
| dense, anchor size, milestone 3's training | 3.72 (3.70–3.75) | 2.51 (2.50–2.53) | 4.94 (4.93–4.95) | 3.73 (3.71–3.74) | 3.55 (3.54–3.56) | 27,584 | 317k |
| dense, anchor size, the mixtures' budget | 3.72 (3.70–3.75) | 2.51 (2.50–2.53) | 4.94 (4.93–4.95) | 3.73 (3.71–3.74) | 3.55 (3.54–3.56) | 27,584 | 753k |
| dense, the mixture's active size, its budget | 3.76 | 2.49 (2.48–2.51) | 4.76 (4.69–4.83) | 3.67 (3.65–3.69) | 3.53 (3.50–3.56) | 44,428 | 753k |
| **dense, the mixture's total size, its budget** | 3.61 (3.60–3.63) | 2.58 (2.56–2.60) | **4.50** (4.49–4.51) | **3.56** (3.55–3.58) | **3.38** (3.37–3.38) | 61,166 | 753k |

Each family's simple baselines on test, fitted on its training scan:

| | Unigram | Markov-1 | Byte statistics with known framing (33 tokens / 16 symbols) |
|---|---:|---:|---:|
| Voyager | 4.84 | 3.85 | |
| Rosetta | 3.31 | 3.18 | 2.54 / 2.30 |
| RFI sweep | 4.72 | 4.87 | |

**The specialist bank** is the three specialists kept whole, each family's data sent to its own
by the family label. All three are stored (82,752 parameters), and one runs per token (27,584).
Its cost is the specialists' training, plus the warm-up when warm.

**Routing:** the share of each family's test tokens whose first choice is its own expert,
averaged over the two blocks, over the three seeds.

| | Voyager | Rosetta | RFI sweep |
|---|---:|---:|---:|
| cold, router finetuned | 0.38–0.41 | 0.68–0.71 | 0.43–0.44 |
| cold, router and backbone finetuned | 0.49–0.50 | 0.64–0.67 | 0.52–0.53 |
| warm, router finetuned | 0.31–0.32 | 0.72–0.73 | 0.50–0.51 |
| warm, router and backbone finetuned | 0.40–0.41 | 0.61–0.62 | 0.52–0.55 |

**Training lengths:**

- the dense models' best epochs were 18–37, though the budget allowed 101–103;
- the mixtures' router alone was still improving at 50 epochs in five of six runs, and with
  the backbone it stopped at 21–33;
- the Voyager specialists' best epochs were 49–50 of 50, and the sweep's 43–50.

### What this shows

- **The result is still negative.** Given clean, structured experts and a warm start, against
  dense models trained for at least the costliest mixture's budget, the best mixture scores
  3.76 bits per token on average. That's against:
  - 3.73 for a dense model the size of one specialist;
  - 3.67 for one with as many parameters as the mixture uses per token;
  - 3.56 for one with as many as it has in total.
- **The positive result: a specialist bank with known family labels.** Kept whole, with each
  family's data sent to its own specialist, the bank averages 3.37. That beats every dense
  model, by 0.19 bits against the largest (3.56), and on validation too (3.22 against 3.38).
  - The gain is mostly on Voyager (3.12 against 3.61) and Rosetta (2.45 against 2.49–2.58).
    Cold specialists show it as well (3.41).
  - **Cost:** it stores 82,752 parameters, more than the largest dense model's 61,166, and
    runs 27,584 per token. Its training (364k–383k targets) is half the dense models' budget.
  - **What it needs:** the family label. That makes it a candidate for family-labelled work
    units, not a routing or weight-merging result.
  - The mixture doesn't capture this gain.
- **The warm start shrinks the merged backbone's cost, but doesn't change the outcome.** Under
  oracle routing it cuts what merging costs:
  - on Voyager, from 0.45 bits to 0.25;
  - on Rosetta, from 0.33 to 0.11;
  - on the sweep, not at all (0.48 to 0.57).

  Overall, the oracle improves from 3.83 to 3.68. Once the backbone is finetuned, cold and
  warm end level (3.76 against 3.78).
- **The router is the other loss.** Even warm, sending tokens by the router instead of by
  family costs 0.32 bits (4.00 against 3.68), and finetuning the backbone too recovers only
  part of it.
  - A token's first-choice expert is its family's for only 31–73% of tokens.
  - The peak tokenizer's positions overlap between families. A router reading one token's
    hidden state, two layers deep, can't tell a Voyager position from the same position in a
    sweep.
- **The training budget isn't what limits the dense models.** They overfit after 18–37
  epochs, so the extra budget doesn't change their scores; the anchor-sized one is identical
  under either training. Their size helps: 3.73 at 27,584 parameters, 3.56 at 61,166.
- **Rosetta's encoded text is better predicted; its language isn't shown to be learned.** The
  specialist scores 2.45–2.51, against 3.31 for unigram and 3.18 for Markov-1. Byte statistics
  that know the UTF-8 framing do as well or better: 2.54 over the 33 tokens, 2.30 over the 16
  symbols.
  - No model here clearly beats them, so nothing shows English learned beyond byte
    statistics.
  - That claim needs an improvement beyond this control.

### Limitations

- **Tiny data, one session, three seeds.** The tests have 144 targets each for Voyager and the
  sweep, and 288 for Rosetta.
- **The sweep's expert has little to bring.** Three training slices aren't enough to learn it.
  So the case still rests on two experts.
- **Routing per token is hard here.** With longer context, or a router over a window instead of
  one token, routing might work better. Neither is tried.
- **One configuration, untuned:**
  - top-2 of three experts;
  - finetuning settings from milestone 3, under which the router alone hadn't converged;
  - the warm-up's length, fixed at 10 epochs;
  - the experts, never finetuned.
- **Selection weights Rosetta twice.** Every model chosen on the concatenated validation scans
  weighs Rosetta's 288 targets twice as much as each other family's 144, while the table
  reports an equal-family mean. A family-balanced selection objective is untested; it would
  have to be chosen on validation.
- **The peak tokenizer discards amplitude.** It suits narrowband lines, and makes off-source
  noise a single token. v3's results aren't comparable with v2's, so M3a and M3b are separate
  tables.

## Milestone 4: the browser's anchor and merge cards

The Live Lab trains its own models in TF.js: Track B's token transformer, on 16-frame snippets
normalized with asinh. Milestone 4 gives it an anchor of its own, and a work unit that trains
several runs from it in the browser and merges them. The unit's result card carries a summary
of the merge, not weights.

### What it adds

- **The anchor, `token-anchor/v1`** ([`web/js/anchor.js`](../../web/js/anchor.js)):
  - **Tokens:** anchor v3's peak rule in the browser's units. A frame is token 0 when no channel
    reaches 6 noise sigma (asinh(6) after the Live Lab's normalization); otherwise it's 1 plus
    the 2-channel bin of its brightest channel. That gives 33 tokens, with nothing fitted.
    - Each Rosetta symbol always becomes the same token. The tests check every symbol of the
      corpus under three noise draws.
  - **θ₀:** the token transformer at that vocabulary: width 22 and 6,325 parameters. Its
    weights are uniform draws from the Live Lab's seeded stream, in a fixed order, rounded to
    float32:
    - Glorot-uniform kernels;
    - small uniform positions;
    - zero biases and offsets;
    - unit scales.

    Its SHA-256 is pinned, and Node and Chromium both rebuild it to that hash. A run refuses an
    anchor that doesn't rebuild to it, and a model whose weights aren't laid out as the anchor's.
  - **In the Live Lab:** Track B can tokenize with the pinned rule. The token transformer then
    trains from θ₀ instead of from a random start.
- **The work unit, `trackb-merge/v0`** ([`web/js/merge.js`](../../web/js/merge.js), in the
  Work units tab):
  - **Data:** the frozen Rosetta corpus as FSK, split in text order into 157 training, 23
    validation and 45 test snippets (2,511, 367 and 719 targets).
  - **Workers:** 4 seeds. Each trains the transformer from θ₀ for up to 20 epochs and keeps its
    best validation epoch.
    - The seeds change only the order of examples: every worker sees the same tokens.
  - **Merges,** with milestone 2's arithmetic and settings:
    1. the plain average;
    2. DARE, then the average. DARE drops 90% of the entries at random and rescales the rest by
       10. The survivors travel in int8, with one scale per tensor.
    3. TIES: keep the largest 20%, elect signs, average the agreeing entries;
    4. DARE, then TIES.
  - **Card (version 1):**
    - test and validation bits for θ₀, a unigram baseline, each seed and each merge;
    - the share of entries DARE keeps, and each merge's nonzero share;
    - sign agreement across the seeds, TIES's sign conflicts, and the mean pairwise cosine;
    - each seed's best epoch and DARE payload size.

    It holds no weights or deltas. A card is about 1.3 KB of compact JSON, and its issue link
    about 2,100 characters, well under the 7,500 the Live Lab allows.
  - **Checks:** ingestion validates a card's form: its config, anchor, corpus, target counts,
    the seed it keeps, and the ranges of its values.
    - Its inputs must be the pinned corpus and anchor only. A slice dataset would put the card
      in another replication group, and let one volunteer count a unit twice.
    - What follows from the spec alone is recomputed and must match: each DARE payload's size
      and the share DARE keeps (the masks depend only on the seeds), and the unigram baseline.
    - The rest comes from training, so it's replicated rather than recomputed. On one backend
      a unit gives an identical card, which the tests check.
    - Runs from two volunteers count as replicated when every metric agrees within 0.02.

### Results

Units 0–9, on WebAssembly in headless Chromium, at about 19 s per unit. The table gives test
bits per token on the 719 held-out targets: the mean over units, with the range.

| Model | Test bits per token |
|---|---|
| θ₀, untrained | 5.71 |
| Unigram (training token counts) | 3.33 |
| One seed, mean of the four | 2.31 (2.30–2.32) |
| The seed kept without merging (best on validation) | 2.31 (2.28–2.32) |
| The best seed on test (an oracle) | 2.30 (2.28–2.31) |
| **Plain average** | **2.31** (2.30–2.32) |
| DARE, then the average | 4.26 (3.46–6.61) |
| TIES | 3.59 (3.45–3.69) |
| DARE, then TIES | 47.1 (28.3–75.4) |

The cards' summaries of the deltas:

| Summary | Mean (range) |
|---|---|
| Parameters all four seeds change, changed with the same sign | 0.86 (0.83–0.87) |
| Mean pairwise cosine of the seeds' τ | 0.95 (0.94–0.96) |
| TIES entries against the elected sign | 0.000 (0.000–0.001) |
| Share of entries DARE keeps | 0.100 (0.098–0.102) |
| Nonzero share: average, DARE then average, TIES, DARE then TIES | 0.94, 0.32, 0.27, 0.32 |
| DARE payload per seed | 1,276–1,432 bytes |
| Seeds' best epochs | 10–20 (mean 16.6) |

### What this shows

- **Averaging seeds from a shared start keeps what they learned, and adds almost nothing.**
  - The plain average beats the mean seed in all 10 units, by 0.001–0.012 bits.
  - Against the seed a volunteer would keep, it's ahead in 6 of 10, by 0.003 bits on average
    (range −0.023 to +0.023). That's within the noise of 719 targets.
  - It beats the best seed on test in 1 of 10.
- **DARE and TIES fail from a random start.** These models train from θ₀, not a pretrained
  model, so a task vector isn't a small adjustment with redundant entries: it's everything the
  model learned.
  - Keeping the largest 20% (TIES) costs 1.3 bits.
  - Dropping 90% (DARE) costs 2 bits, even though averaging keeps it unbiased.
  - The only parameters no seed changes (6%) are the input rows of the 17 tokens Rosetta never
    uses.
- **DARE then TIES blows up, by construction.** DARE's rescale keeps a sum right on average.
  TIES averages only the entries that survive, though, so every surviving entry stays 10 times
  too large. A unit test checks this: identical seeds come out scaled by 10 wherever DARE kept
  an entry.
  - Milestone 2 uses the same arithmetic. Milestone 5 tested whether this rescale held its
    DARE-TIES chain back: it didn't (see milestone 5).
- **TIES has no conflicts to resolve here.**
  - The seeds change 86% of the weights they all change in the same direction, and their task
    vectors point the same way (cosine 0.95).
  - Among each seed's largest 20%, almost none conflict.
  - TIES's sign election is for deltas that interfere, and these don't, so its whole cost is
    the trimming.
- **This doesn't show language.** Milestone 3b's byte-statistics control, which knows the
  UTF-8 framing, scores 2.28 bits per token over the 16 symbols on the same 719 test targets.
  That's add-one statistics fitted on the training tokens, and every unit has the same tokens,
  so it's the same for all of them.
  - It's slightly better than any model here (2.31), though it has framing information the
    transformer lacks.
  - The merge comparison stands, since every row is scored on the same text. Whether the
    transformer learns anything beyond byte statistics isn't shown.
- **The cards fit.** A card carries the summary with plenty of room in an issue link. A DARE
  delta of this model would take about 1.4 KB, so deltas could travel too, but merging them
  doesn't help here.

### Limitations

- **Synthetic text, one corpus.** Rosetta is the browser's positive control: 1,804 bytes of
  English sent as FSK. No real data is used.
- **Seeds, not shards.** Every worker sees the same tokens, and only their order differs. That's
  the hardest case for a merge to add anything. Shards, as milestone 2 used in Python, are
  untested in the browser.
- **One backend.** Cards from other backends will differ in the last digits. Whether merge
  units replicate across backends within 0.02 is for the community results to show. DARE then
  TIES, with its blown-up scores, is unlikely to.
- **One configuration,** milestone 2's: p = 0.9, TIES keeping 20%, and no scale for the merged
  delta chosen on validation.

## Milestone 5: a sequence router, DARE inside TIES, and shards in the browser

Milestone 5 takes up the three open findings of milestones 2 to 4:

- the specialist bank won in 3b only with the family label;
- DARE-TIES fell behind the plain average with DiLoCo in milestone 2;
- milestone 4's browser seeds had nothing to merge.

### 1. Routing whole sequences to the specialist bank

**Setup** (fixed before the first run, [`route_bench.py`](route_bench.py)):

- **The bank:** milestone 3b's warm specialists, retrained exactly as there, and unchanged.
  - Same seeds (42–44), warm start and 50 epochs.
  - They reproduce milestone 3b's bank with the label to two decimals (3.12, 2.45, 4.54).
- **What a router may use:** it picks a specialist for each sequence (a slice of 48 targets)
  from the tokens before each target only. So no target ever helps choose the model that
  predicts it.
- **What each router scores a specialist by** before each target, with an equal prior. The
  first token, which no specialist predicts, counts by its add-one frequency in that
  specialist's family's training scan. Then, by router:
  - **first token:** that first token alone, for the whole sequence;
  - **token frequencies:** every token so far, each by that frequency (naive Bayes; the
    3 × 33 frequencies are the router's only fitted numbers);
  - **specialists' likelihood:** each specialist's own predictions of the tokens so far,
    followed throughout, or committed after the first 4 or 8 tokens;
  - **posterior mixture:** all three specialists' predictions, weighted by that likelihood
    (Bayes's rule).
- **Selection:** one router per seed, chosen on validation by the mean over the three
  families, each counting once. What the concatenated validation (every target once, milestone
  3b's objective) would choose is recorded beside it. Test is scored afterwards, for every
  router.
- **The check from 3b's review:** milestone 3b's dense rivals, retrained exactly as there.
  - Same starts and seeds, no early stopping, 104 epochs (more than any seed's budget of
    101–103).
  - Their best epoch is chosen twice: on the concatenated validation scans, and on the
    equal-family mean (the Voyager and sweep validation scans counted twice).
  - Chosen on the concatenated scans, they reproduce milestone 3b's scores.

**Results.** Test bits per token, mean over the three seeds (range in brackets). The last two
columns are the share of each family's targets the router sends to that family's own specialist
(for the mixture, the mean weight on it), and the parameters run per target, on average over a
sequence.

| | Voyager | Rosetta | RFI sweep | Mean | Validation mean | Own specialist (V, R, S) | Run per target |
|---|---:|---:|---:|---:|---:|---:|---:|
| bank, dispatched by family label (3b) | 3.12 | 2.45 | 4.54 (4.53–4.57) | **3.37** (3.36–3.38) | 3.22 | 1, 1, 1 | 27,584 |
| first token | 5.75 (5.73–5.76) | 2.45 | 4.54 (4.53–4.57) | 4.25 | 4.40 (4.38–4.45) | 0.00, 1.00, 1.00 | 27,683 |
| token frequencies | 4.23 | 2.45 | 4.57 (4.56–4.60) | 3.75 | 3.32 | 0.55, 1.00, 0.97 | 27,683 |
| specialists' likelihood | 3.16 | 2.45 | 4.63 | 3.41 | 3.29 | 0.96, 1.00, 0.85 | 82,851 |
| specialists' likelihood, committed after 4 tokens | 3.16 | 2.45 | 4.84 (4.82–4.87) | 3.48 | 3.91 | 0.96, 1.00, 0.68 | 31,131 |
| specialists' likelihood, committed after 8 tokens | 3.16 | 2.45 | 4.70 (4.69–4.72) | 3.43 | 3.39 | 0.96, 1.00, 0.70 | 35,728 |
| **posterior mixture (chosen on validation, every seed)** | 3.16 | 2.46 (2.45–2.47) | 4.56 (4.54–4.58) | **3.39** (3.39–3.40) | 3.26 | 0.95, 0.99, 0.85 | 82,851 |
| dense, anchor size (3b) | 3.72 (3.70–3.75) | 2.51 (2.50–2.53) | 4.94 (4.93–4.95) | 3.73 | 3.55 | | 27,584 |
| dense, the mixture's active size (3b) | 3.76 | 2.49 (2.48–2.51) | 4.76 (4.69–4.83) | 3.67 | 3.53 | | 44,428 |
| dense, the mixture's total size (3b) | 3.61 (3.60–3.63) | 2.58 (2.56–2.60) | 4.50 (4.49–4.51) | 3.56 | 3.38 | | 61,166 |
| token-level mixture, 3b's best (cold, router + backbone finetuned) | 3.74 | 2.60 | 4.94 | 3.76 | | | 44,480 |

**Parameters and budget:**

- **Stored:** 82,851, the three specialists (82,752) and the router's 99 frequencies.
- **Run per target:**
  - one specialist plus the router, for the frequency routers;
  - all three, for the mixture and the followed likelihood;
  - the committed routers in between: all three until they commit, one after.
- **Training:** the bank's budget is 376k–391k targets (the warm-up's 74k included). The router
  adds no training, only a count of the training scans' tokens. The dense rivals trained on
  768k targets, twice as many.

**What sending a family's sequences to the wrong specialist costs.** Test bits per token over
own specialist, mean over seeds:

| Sequences from | to Voyager's | to Rosetta's | to the sweep's |
|---|---:|---:|---:|
| Voyager | (3.12) | +2.98 | +2.63 |
| Rosetta | +2.44 | (2.45) | +2.66 |
| RFI sweep | +1.02 | +2.52 | (4.54) |

**The equal-family check.**

- **Routers:** the equal-family mean and the concatenated validation choose the same router,
  the posterior mixture, in every seed.
- **Dense rivals:** re-choosing their epoch by the equal-family mean leaves the largest exactly
  as it was (epoch 18, 3.56). It moves the other two by 0.01 bits, both slightly worse (3.68,
  3.74).
- So milestone 3b's double weight on Rosetta changed no conclusion.

**What this shows:**

- **The specialist bank keeps its gain without the label.** The posterior mixture, chosen on
  validation, scores 3.39: 0.02 bits behind dispatch by label, and 0.17 ahead of the largest
  dense rival (3.56). It's ahead of every dense model on every seed, and on validation (3.26
  against 3.38).
- **Routing costs little because a wrong route costs so much.** A family's sequences cost 1–3
  bits more on another family's specialist. That makes them easy to tell apart by likelihood
  within a few tokens:
  - the mixture puts 95% of its weight on Voyager's own specialist for Voyager targets, and
    99% for Rosetta's;
  - the sweep, whose own specialist is weakest (milestone 3b), gets 85%.
- **It's sequence-level, not token-level, routing that works here.** The token-level mixture
  in milestone 3b, which merged the backbones, lost to the dense rivals (3.76 at best). This router
  keeps every specialist whole, and loses almost nothing to the label.
- **What the best router costs.** The posterior mixture runs all three specialists for every
  target: 82,851 parameters per target, more than the largest dense rival's 61,166, and 3 times
  the dense model of the anchor's size. Cheaper choices:
  - committing to one specialist after 8 tokens runs 35,728 per target on average and scores
    3.43, still 0.13 ahead of the largest dense rival;
  - the frequency routers run one specialist, but misroute Voyager: its first frames look like
    another family's. A sequence's first token alone sends every Voyager sequence elsewhere.
- **The family label was a crutch for the bank, not a requirement.** What remains open is
  whether the bank's advantage holds with more families, or with sequences that mix families.

### 2. DARE inside TIES: what the 10× rescale cost milestone 2

DARE rescales its survivors by 1/(1 − p), 10 at p = 0.9, so that a sum over every contributor
keeps its expected value. TIES then averages only the agreeing nonzero entries. So a surviving
entry keeps the factor, and with three agreeing workers the merge is about 10 × (1 − 0.9³) = 2.7
times the mean delta in expectation.

Milestone 5 adds TIES's mean over every contributor's weight (`merge.py --ties-mean all`). It
counts a dropped or disagreeing entry as zero, and is unbiased after DARE. Averaging unrescaled
survivors, the other fix, would shrink each entry by 1 − p³ and stay biased.

Rerunning milestone 2's benchmark ([`merge_bench.py`](merge_bench.py)) reproduces every earlier
row exactly. The added rows:

| | Validation | Test | Off-source control |
|---|---:|---:|---:|
| one-shot: DARE-TIES (TIES's own mean, milestone 2) | 3.22 (3.02–3.43) | 2.93 (2.80–3.09) | 5.09 (5.00–5.17) |
| one-shot: DARE-TIES, mean over all | 3.28 (3.26–3.29) | 2.98 (2.94–3.05) | 4.43 (4.42–4.46) |
| one-shot: DARE-TIES, mean over all, greedy soup | 3.12 (3.09–3.16) | 2.83 (2.75–2.96) | 4.57 (4.54–4.60) |
| 10 rounds: average + DiLoCo (milestone 2) | 2.85 (2.84–2.85) | **1.83** | 5.00 (4.99–5.02) |
| 10 rounds: DARE-TIES, greedy soup + DiLoCo (milestone 2) | 2.84 (2.83–2.86) | 2.11 (1.99–2.24) | 5.00 (4.83–5.14) |
| 10 rounds: DARE-TIES, mean over all, greedy soup + DiLoCo | 2.93 (2.84–3.00) | 2.51 (2.32–2.65) | 5.02 (4.92–5.13) |
| 10 rounds: average, greedy soup + DiLoCo | 2.85 (2.83–2.86) | 2.34 (2.34–2.35) | 5.04 (4.99–5.13) |

**What this shows:**

- **The rescale didn't cause the gap: removing it makes the chain worse.** Unbiased DARE-TIES
  with the same soup and DiLoCo scores 2.51, against 2.11 with the 10× factor kept.
  - The factor had been working as a larger outer step: about 2.7 times the averaged update on
    the entries the workers agree on.
  - One-shot, it's the same: 2.98 unbiased (exactly the DARE average's score) against 2.93.
- **The greedy soup causes most of the gap.** Step by step from the plain average with DiLoCo
  (1.83):
  - adding the greedy soup: 2.34 (+0.51);
  - then unbiased DARE-TIES, compression and sign election: 2.51 (+0.17);
  - then keeping DARE's 10× factor instead: 2.11 (−0.40).
  - **The soup merges too few shards.** Its rounds kept 1.3 of the 3 shards on average, against
    2.0 for the original DARE-TIES chain and 3 without a soup, so most of the data is left out
    of most rounds.
  - **Its validation doesn't show this.** It's as good as the plain chain's (2.85 against
    2.85): the soup fits its 144 validation targets.
- **So the chain's best merge is the simplest:** the plain average with DiLoCo (1.83).
  - A soup helps one shot (2.65, milestone 2), where it drops one weak worker once.
  - Applied every round, it starves the chain of data.
  - Unbiased DARE is still the right way to compress a delta for a card. The one-shot cost of
    compression is 0.11 bits (2.98 against 2.87).

### 3. Shards and a warm start in the browser (`trackb-merge/v1`)

Milestone 4's seeds all saw the same text, so there was nothing for a merge to add. The browser's
new spec, `trackb-merge/v1` ([`web/js/merge.js`](../../web/js/merge.js)), runs three sets of 4
workers per unit on the same Rosetta text and anchor:

- **seeds:** milestone 4's unit exactly: the same seeds, data and settings, and the same
  numbers;
- **shards:** each worker trains from θ₀ on its own quarter of the training text, in order
  (623 to 639 targets each);
- **warm shards:** the same quarters, from a shared warm start: θ₀ trained 3 epochs on all the
  training text, keeping its best validation epoch.

Each set is merged five ways: milestone 4's four, and DARE then TIES with the mean over all
workers. The kept worker is each set's best on validation. The seeds set's kept worker is the
one trained on all the text.

The card is about 2.9 KB, with a 4,650-character issue link, under the 7,500 limit. As in
milestone 4, ingestion recomputes what follows from the spec: here per set, each DARE payload's
size, the share DARE keeps, and the unigram baseline.

**Results:** units 0–9, on WebAssembly in headless Chromium, about 53 s per unit. Test bits per
token on the 719 held-out targets, mean over units (range). The last column counts the units in
which the merge beats the set's kept worker.

| | Seeds, all the text | Shards, from θ₀ | Shards, from the warm start | Beats the kept worker (seeds, shards, warm shards) |
|---|---:|---:|---:|---:|
| kept worker (best on validation) | 2.31 (2.28–2.32) | 2.49 (2.42–2.57) | 2.49 (2.45–2.52) | |
| best worker on test (an oracle) | 2.30 (2.28–2.31) | 2.48 (2.42–2.51) | 2.46 (2.43–2.51) | |
| **plain average** | 2.31 (2.30–2.31) | 2.48 (2.45–2.53) | **2.46** (2.44–2.47) | 6, 6, **9** of 10 |
| DARE, then the average | 4.26 (3.46–6.61) | 3.56 (2.93–4.07) | 2.74 (2.54–2.95) | 0, 0, 0 |
| **TIES** | 3.59 (3.45–3.69) | 2.80 (2.68–2.89) | **2.43** (2.36–2.47) | 0, 0, **9** |
| DARE, then TIES (TIES's own mean) | 47.1 (28.3–75.4) | 50.5 (24.9–87.6) | 11.7 (7.8–16.1) | 0, 0, 0 |
| DARE, then TIES (mean over all workers) | 4.26 (3.46–6.62) | 3.57 (2.96–4.10) | 2.74 (2.54–2.96) | 0, 0, 0 |
| sign agreement across workers | 0.86 | 0.61 | 0.51 | |
| mean pairwise cosine of τ | 0.95 | 0.71 | 0.63 | |

For scale: θ₀ scores 5.71, the unigram baseline 3.33, and the warm start alone 2.70.

**What this shows:**

- **Merging beats the workers once they see different data and share a trained start.** From
  the warm start, TIES (2.43) beats the kept worker in 9 of 10 units, by 0.06 bits on average,
  and the best worker on test in 7. The plain average (2.46) beats the kept worker in 9.
- **From θ₀, sharding alone isn't enough.** The plain average is level with the kept shard
  worker (2.48 against 2.49, ahead in 6 of 10), and TIES and DARE still cost 0.3–1 bits.
- **The warm start is what turns TIES around:** 3.59 for the seeds and 2.80 for the shards
  from θ₀, but 2.43 from the warm start.
  - From a trained start, each change is a smaller adjustment on top of shared structure.
    Trimming its smallest 80% loses little, and sign election has something to do: the warm
    shards agree in sign on only 51% of the weights they all change.
  - That matches what TIES was designed for: fine-tuning deltas from a pretrained base.
- **No merge reaches one worker on all the text** (2.31) in any unit. Within the same total
  data and epochs, training together still beats training apart and merging, as in milestone 2.
- **DARE's factor inside TIES blows up in every set** (12–51 bits). The mean over all workers
  removes it, and then matches DARE then the average to two decimals, since sign election
  drops almost nothing. Compression itself still costs 0.3 bits from the warm start (2.74
  against 2.46), and far more from θ₀.

### Limitations of milestone 5

- **The router's families are easy:** three families whose sequences cost 1–3 bits more on the
  wrong specialist. Families closer together, or sequences that mix them, would be harder.
- **The best router isn't cheap:** the posterior mixture runs all three specialists per token.
- **Still one session, three seeds,** for the Python results. The data are tiny, as in every
  milestone.
- **The browser's sets use one corpus,** in order: each shard is a different stretch of the
  text, so shards also differ in what they're about. The warm start's length (3 epochs) was
  fixed before any run and not tuned. One backend was measured.
- **The chains' soup** is milestone 2's. A soup that must keep at least two shards a round is
  untested.

## Next

Milestone 5 closes issue #20. What it leaves open:

- **More families, and mixed sequences,** for the sequence router: three families this
  different are easy to tell apart within a few tokens.
- **A cheaper router:** commit after a few tokens, and send ambiguous sequences on to the
  mixture.
- **For the chains:**
  - a soup that keeps at least two shards a round;
  - a momentum restart when the gate rejects a round;
  - a larger outer step for the plain average, the role DARE's factor played by accident.
- **A control with known temporal structure,** to calibrate a test for found structure.
- **For Rosetta:** a language test that beats the byte-statistics control, such as more text,
  or a comparison with whole characters shuffled while keeping the framing.
- **In the browser:**
  - DARE at a lower drop rate;
  - a merge scale chosen on validation;
  - shards of other families once more real data is published;
  - whether units replicate across compute backends.
