# EI@Home experiment protocol

**Status: draft v0.** This is a living document. Propose changes with a pull request.

The [README](../README.md) gives the overview. This document specifies the data, models, metrics
and controls for the two experiment tracks, and the checks a result has to pass before we call it
interesting.

## Contents

1. [Research questions](#1-research-questions)
2. [Data](#2-data)
3. [Track A: continuous sequence modeling](#3-track-a-continuous-sequence-modeling)
4. [Track B: signal-to-language tokenization](#4-track-b-signal-to-language-tokenization)
5. [Controls](#5-controls)
6. [Pitfalls checklist](#6-pitfalls-checklist)
7. [What counts as interesting](#7-what-counts-as-interesting)
8. [Reporting template](#8-reporting-template)
9. [Work-unit specification](#9-work-unit-specification)

---

## 1. Research questions

- **Q1 (Track A):** When recurrent models learn to predict candidate signals, do their learned
  dynamics separate candidates from noise, simple synthetic signals and known human
  transmissions?
- **Q2 (Track B):** When candidates are turned into token streams, do the streams show
  language-like statistics beyond what null models and tokenizer artifacts explain?
- **Q3 (both):** Which conclusions survive changes of random seed, model family, tokenizer and
  data resolution?

## 2. Data

### 2.1 Resolution matters

Breakthrough Listen's Green Bank data comes in several products made from the same raw voltages.
Each trades frequency resolution for time resolution. The values below are approximate, from the
[1,327-star survey paper](https://arxiv.org/abs/1906.07750) and the
[data formats paper](https://arxiv.org/abs/1906.07391). Always read resolutions from file headers
rather than hard-coding them.

| Product | Channel width | Time step | Steps in a 5-min scan | Best for |
|---|---|---|---|---|
| High spectral resolution | ~2.8 Hz | ~18 s | ~16 | Detecting narrowband hits and measuring drift; Track B on time-frequency patches |
| Mid resolution | ~2.9 kHz | ~1 s | ~280 | Track A on drift and pulsing, at the cost of diluting very narrow signals |
| High time resolution | ~366 kHz | ~0.35 ms | ~860,000 | Fast pulses and broadband transients |
| Raw voltages (selected observations) | complex samples | native | n/a | Phase and modulation; I/Q inputs for both tracks |

Two consequences:

- **Power spectrograms discard phase.** Questions about phase or digital modulation need raw
  voltages or synthetic I/Q (from setigen's voltage tools or TorchSig).
- **Better time resolution means coarser frequency resolution.** For selected observations, raw
  voltages can be re-channelized with [rawspec](https://github.com/UCBerkeleySETI/rawspec) to
  pick a compromise suited to the experiment.

### 2.2 Snippet specification

A **snippet** is the unit of data for both tracks.

- **Window:** centered on a hit's start frequency. It spans the hit's full drift over the scan plus
  a margin, with a fixed number of channels per product.
- **Two versions:** *raw* (drifting, as observed) and *de-drifted* (shifted so that the hit's
  frequency is constant).
- **Cadence:** when a hit comes from an on/off observing cadence, keep the on-source and
  off-source scans together and label each frame with its pointing.
- **Normalization:** per-snippet median and robust-scale normalization, with the parameters
  stored so the operation can be reversed.
- **Instrumental artifacts:** flag known features such as the spike at the center of each coarse
  channel and the roll-off at coarse-channel edges.
- **Metadata:** telescope, receiver, backend, product, timestamp (MJD), target, coordinates,
  center frequency, channel width, time step, hit SNR, drift rate, candidate tier and a hash of
  the source file.
- **Storage:** WebDataset or Zarr shards with a versioned manifest. No raw data goes in git.

### 2.3 Candidate tiers

| Tier | Definition | Role |
|---|---|---|
| Hits | Any narrowband detection above threshold (turboSETI, hyperseti, seticore) | Bulk training |
| Events | Hits present in on-source scans and absent from off-source scans | Main training and analysis |
| Signals of interest | Named, closely studied candidates (blc1, the Ma et al. 2023 signals, SETI@home reobservation candidates) | Held out entirely; analysis only |

### 2.4 Splits

- Split by **observing session and target**, never by individual snippet. Snippets from one
  session share an interference environment.
- The same interference source (for example a satellite) recurs across sessions. For a harder test,
  also hold out entire **frequency bands**.
- Signals of interest are never used for training or tuning.

## 3. Track A: continuous sequence modeling

### 3.1 Inputs

- **A1, frames over time:** a sequence of `T` spectrogram frames, each a vector of `C` channel
  powers. This is the default.
- **A2, frequency stepping:** for high-spectral-resolution products with few time steps, step
  across channels instead of time. This is optional and exploratory.
- **A3, I/Q:** complex baseband samples as two real channels, from raw voltages or synthetic
  sources.

### 3.2 Models

- GRU, LSTM, minGRU, minLSTM, Mamba (via mamba.py, so any hardware works), RWKV, and a
  **causal tiny transformer** baseline.
- **Matched parameter budgets** across families (for example about 10k, 100k and 1M parameters),
  so that differences between families reflect architecture rather than size.

### 3.3 Objective

- Next-step prediction: mean squared error or Gaussian negative log-likelihood on normalized
  power. For I/Q, predict the next complex sample or block of samples.
- Optional: multi-step rollouts, to test whether models capture dynamics rather than just
  smoothing.

### 3.4 Readouts

- **Prediction gain.** Error relative to two simple baselines: *persistence* (predict that the next
  frame equals the current one) and a *linear autoregressive* model. If a recurrent model can't
  beat a linear autoregressive model, there is no nonlinear structure to discuss.
- **Hidden-state geometry.** Summarize each snippet by its mean or final hidden state. Project
  with PCA then UMAP, cluster with HDBSCAN, and report each cluster's makeup by control type and
  candidate tier.
- **Linear probes.** On controls with known parameters (drift rate, pulse period, modulation class,
  SNR), test whether those parameters can be decoded linearly from hidden states. If they can, the
  representation encodes physics. Then apply the same probes to candidates.
- **Optional dynamics analysis.** Fixed points and local linearization of the trained recurrent
  model.

### 3.5 Ablations

- Shuffle frames in time, which destroys temporal structure while keeping per-frame content.
- Compare raw and de-drifted snippets.
- Vary resolution and model size.

## 4. Track B: signal-to-language tokenization

### 4.1 Tokenizers

- **Learned:** VQ-VAE (codebook size `K`), FSQ (levels per dimension), and residual VQ in the
  style of neural audio codecs (use the first level, or flatten levels in a fixed order).
- **Simple baselines:** k-means on frames, and uniform scalar quantization of amplitude and phase.
  They are interpretable and show how much the learned tokenizers add.
- **Sweeps:** slice length and stride, and vocabulary size (for example `K` = 16, 64, 256, 1024).
- **Report codebook health:** code-usage perplexity and the number of dead codes.
- **Training data:** train tokenizers on a mix of candidates and controls, and also on each group
  separately. A tokenizer trained only on candidates can overfit to them.

### 4.2 Linguistic metrics

| ID | Metric | Notes |
|---|---|---|
| M1 | Zipf rank-frequency | Fit with `powerlaw`. Compare power-law, lognormal and exponential fits by likelihood ratio, and report exponents with confidence intervals |
| M2 | Block entropy `H_n` and conditional entropy `h_n = H_n − H_(n−1)` | Drops in `h_n` with `n` indicate rules and redundancy. Plug-in estimates collapse when contexts outnumber tokens (`K^n` large), even with bias correction, so the Live Lab reports the held-out cross-entropy of a smoothed n-gram model instead: fitted on the training split, its smoothing chosen on the validation split, and scored on the test split (see 4.4) |
| M3 | Mutual information `I(d)` between tokens `d` apart | Compare exponential and power-law fits. Power-law decay suggests hierarchical structure ([Lin & Tegmark](https://arxiv.org/abs/1606.06737)) |
| M4 | Compressibility | Compressed bits per token (zstd, xz) compared with surrogates |
| M5 | Repeated motifs | Repeated n-grams beyond what a Markov-`k` surrogate predicts; longest repeated substring |
| M6 | Language-model perplexity | Tiny causal transformer (nanoGPT) and an RNN language model. Held-out bits per token compared with unigram entropy and with surrogate streams. The Live Lab trains a small GRU token model with early stopping |
| M7 | "Speaks back" | Sample from the trained language model and check whether M1–M5 of the samples match the real streams, especially long-range structure |

### 4.3 Surrogates and null models

| ID | Surrogate | What it preserves | What it destroys |
|---|---|---|---|
| S0 | Same tokenizer applied to pure noise and to off-source scans | The tokenizer's own biases | Any signal |
| S1 | Token shuffle | Token frequencies, so the Zipf plot is identical | All ordering |
| S2 | Markov-`k` surrogate | n-gram statistics up to order `k` | Longer-range structure |
| S3 | Time reversal | Token frequencies and mutual information at every distance | Causal direction |
| S4 | Phase-randomized (IAAFT) surrogate, applied before tokenization | Power spectrum and amplitude distribution | Nonlinear structure |

S1 makes a key point: a shuffled stream has exactly the same Zipf plot as the original, so Zipf
alone says nothing about order or grammar.

### 4.4 Held-out evaluation in the Live Lab

Track B splits each family's frames into training, validation and test parts **before fitting
anything**, as the Python signal-language-model track does:

- **Synthetic signals:** independent snippets, drawn separately for each split (70/10/20).
- **Rosetta:** the text in order: the first 70% for training, the next 10% for validation, the
  last 20% for testing, so the test split is text the tokenizer never saw.
- **Real data:** one fixed central window per slice (no shifted copies), holding out whole
  scans, or whole sessions once a family has three. Copies stay in one split and count once.
  Within one session this is a within-session holdout, and the Live Lab labels it so.

The k-means tokenizer is fitted on training frames only. S1 shuffles the tokens within each
segment (snippet, slice or split), so it isolates order: it keeps each segment's tokens and can
still dip a little when segments differ in content. S2 is a Markov-1 chain with add-one smoothing
fitted to training transitions only. No statistic, n-gram context or token-model window crosses
a segment boundary, and every target after a segment's first token is scored once.

For real data, a **control** row encodes the other real family with this family's tokenizer and
scores it with this family's n-gram and token models. On the 2020 Voyager cadence, context
predicts Voyager's held-out tokens (3.67 → 1.84 bits) but not off-source tokens (2.57 → 2.69),
and Voyager's model gets worse with context on off-source frames. Before this protocol, shifted
copies of the same few windows on both sides of the split made off-source noise look structured
(3.90 → 3.22 bits). Each real-data test split has only 45 scored tokens, so small differences are
noise.

## 5. Controls

| ID | Control | Ground truth | Purpose |
|---|---|---|---|
| C0 | Thermal noise (synthetic) | No structure | Calibrates every metric |
| C1 | Off-source scans (in the Live Lab: the off-source family, cut where Voyager’s signals would be) | Real sky and instrument without the hit | Realistic null |
| C2 | setigen continuous tones with drift | Drift rate, SNR | Simplest predictable structure |
| C3 | setigen pulsed signals | Pulse period, duty cycle | Periodic structure |
| C4 | Chirps and frequency sweeps (radar-like) | Sweep rate | Deterministic but meaningless structure |
| C5 | Voyager 1 (turboSETI example observation) | Human-made carrier with telemetry sidebands | Real transmitter, real instrument |
| C6 | Digital modulations with **random** payload bits (TorchSig, GNU Radio) | Modulation class | Information-carrying signals with no linguistic content |
| C7 | **Rosetta control:** the same modulations carrying **uncompressed natural-language text** | Modulation class and the text itself | End-to-end positive control for Track B |

**Why the Rosetta control matters:** if the pipeline cannot find language-like statistics in C7
(and tell C7 apart from C6), a negative result on candidates means nothing. Real human
transmissions usually scramble, compress or encrypt their payloads, which whitens them. That's why
real interference that *does* carry human language can still look random, and why C7 uses
uncompressed, unscrambled text.

## 6. Pitfalls checklist

- [ ] **Zipf from random processes or tokenizer artifacts.** Always compare with S0 and S1.
- [ ] **Undersampled entropy estimates.** Don't report `h_n` for orders where counts are too small.
- [ ] **Leakage.** Splits are by session, target and, where possible, frequency band.
- [ ] **Resolution.** State the product and resolution. Power spectrograms carry no phase.
- [ ] **Trivial drift structure.** A drifting tone is perfectly predictable. Compare raw and
  de-drifted snippets, and compare with C2.
- [ ] **Normalization artifacts.** Run the same preprocessing on C0 and check that no structure
  appears.
- [ ] **Instrumental artifacts.** Coarse-channel center spikes and edge roll-off are flagged or
  masked.
- [ ] **Whitened payloads.** Compressed, scrambled or encrypted data looks like noise. State that
  limitation in every report.
- [ ] **Multiple comparisons.** Many candidates times many metrics times many configurations means
  some results will look unusual by chance. Choose metrics in advance, correct for multiple tests,
  and confirm on held-out data.
- [ ] **Confirmation bias.** Use the criteria in section 7, written before results are in.

## 7. What counts as interesting

A candidate group is flagged for follow-up only if **all** of the following hold:

1. It beats every applicable surrogate (S0–S4) on a metric chosen in advance, after correcting for
   multiple comparisons.
2. It is not explained by any control class (C0–C6), judged by both cluster membership and a
   classifier trained on the controls.
3. It replicates across at least 3 seeds, at least 2 model families or tokenizers, and on held-out
   data.
4. It survives interference checks: it doesn't appear in off-source scans or at unrelated targets,
   and its frequency doesn't match known transmitter bands.

Even then, the expected explanation is human interference. A flagged result is written up and
shared privately with SETI researchers before anything is said publicly.

## 8. Reporting template

Each experiment report (a Markdown file under `reports/` or a GitHub issue) includes:

- **Question:** which of Q1–Q3 it addresses, and the pre-chosen metric.
- **Data:** snippet set version and manifest hash, product and resolution, candidate tiers.
- **Controls and surrogates:** which ones were run.
- **Configuration:** model or tokenizer settings, parameter count, seeds.
- **Compute:** hardware and run time.
- **Results:** a table comparing candidates with every control and surrogate, with uncertainties.
- **Negative results and limitations:** including the whitened-payload limitation.
- **Reproduce:** the exact command or notebook.

## 9. Work-unit specification

### 9.1 Browser work units (implemented)

The [Live Lab](../web/) runs work units in a Web Worker. Each spec is versioned, and a unit is
fully determined by its spec and unit number.

**`track-a/v0`** (`web/js/workunits.js`) replicates the Hello, Voyager notebook:

- seed = unit number; hidden size cycles through 16, 32 and 64 with the unit number;
- 600 synthetic drifting tones plus 600 noise snippets for training (16 x 64, normalized as in
  section 2.2), 12 epochs, Adam (learning rate 0.003), batch 64;
- 48 held-out snippets per family, including the real-data families (Voyager, and off-source
  scans of the same cadence) if their micro-slices are loaded. The model never trains on real
  data in this spec, so every real snippet is unseen;
- the Rosetta corpus is part of the spec (`en-gettysburg-bliss+us-preamble/v1`, pinned by
  SHA-256). Work units ignore any custom text set in the Live Lab, and refuse to run if the
  built-in text no longer matches the pinned hash;
- reported: prediction gain against mean, persistence and linear baselines; GRU gain on
  frame-shuffled snippets; and a drift probe (R² and error on synthetic tones, plus the
  readout on Voyager). The probe is also fitted to the same GRU **before training** (same
  architecture and seed), which shows how much drift is decodable from random features. The
  trained-minus-untrained difference is what training added.

All inputs are captured when a unit starts, so loading new slices or text mid-run can't
change them.

**Result card (version 3):** a JSON object with `format: "eiathome-result"`, the spec, unit,
full configuration, app and TensorFlow.js versions, compute backend, parameter count, run time,
metrics, and an `inputs` block:

- `rosetta_corpus`: the corpus id and the SHA-256 of the text actually used;
- `slices`: `null`, or a `dataset_id` plus each slice's id, family, SHA-256 and scan start
  time. Slice hashes are computed from the loaded bytes, and a slice whose manifest digest
  doesn't match is rejected. The `dataset_id` hashes the slice hashes together with the
  metadata that changes results (family, start time, shape, channel width, time step, drift
  labels). Version 2 called this block `voyager`; version 3 renamed it when off-source slices
  joined the dataset.

Runs of a unit are compared for replication only when they used the same `dataset_id`.

Every run puts the loaded slices in one canonical order (by content hash, then id) before
sampling, and the `dataset_id` uses the same order. Reordering a manifest, or renaming its
files, therefore changes neither the inputs nor the results.

The card contains nothing about the volunteer or their machine beyond the backend name.
Volunteers download it or submit it as a pre-filled GitHub issue.

**Training on real data in the Live Lab:** when the Live Lab trains on a real family (Voyager
or off-source), it evaluates that family on observing sessions it didn't train on (about 30% of
the family's sessions). Slices of the other real family are unseen, so they're all evaluated. Sessions come from
the data, never from file names:

- slices whose scans started within 12 hours of each other, directly or through a chain of
  scans, form one session;
- slices with identical bytes always share a session, even if their metadata disagrees;
- if any slice has no start time, independence can't be established.

When there's only one session, or a start time is missing, nothing independent can be held out.
The Voyager row is then labelled "not held out", because the score measures reconstruction of
the training data, not generalization. The split trusts the manifest's start times, so it can't
detect a copy whose bytes and start time were both deliberately altered.

**Determinism:** in our tests, the same unit on the same backend (WebAssembly) gave identical
metrics across repeat runs and fresh tabs. Different backends (WebGPU, WebGL, WebAssembly)
use different floating-point paths, so compare across backends within a tolerance.

**Trust:** a submitted card is a claim, not a verified result. Replication by an independent
volunteer, on the same spec and unit, is what confirms it.

**Collection:** "Submit as a GitHub issue" opens a pre-filled issue. A GitHub Action
(`.github/workflows/ingest-results.yml`) then checks the card against its spec:

- the configuration must equal the spec's configuration for that unit;
- the Rosetta corpus must be the pinned one;
- any Voyager data must be the slices the site publishes;
- every metric must be a number.

The Action rejects duplicates: the same issue, an identical card, or the same volunteer
re-submitting the same unit. It appends a valid card to `web/data/community_results.json` and
redeploys the Live Lab. The Work units tab shows each unit's status. A unit is **replicated**
when runs from at least two volunteers agree within 0.02 on every metric, and marked **runs
disagree** when they don't. Each volunteer's first run of a unit is the one compared.

### 9.2 BOINC work units (later, if needed)

For heavier jobs such as raw voltages or long sweeps:

- **Input:** snippet shard IDs, a configuration file and a seed.
- **Output:** a metrics JSON file, hidden-state summaries (PCA-reduced) and, optionally, model
  weights. Outputs should be small.
- **Determinism:** fixed seeds and pinned library versions. Floating-point results differ across
  hardware, so validation compares metrics within a tolerance rather than bit for bit.
- **Validation:** BOINC replication on at least two hosts, with results compared within that
  tolerance.
- **Budget:** each unit should finish in under an hour on a typical laptop CPU.
