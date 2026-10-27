# First recovery measurement: the fixed baselines

Milestone 1 of [issue #13](https://github.com/davidmdrpi/EIatHome/issues/13). Synthetic signals
dealt blind into real off-source data and scored by the two fixed baselines: the de-Doppler
search and an energy detector (radiometer). Learned detectors (milestone 2) will be scored
against these numbers, at the same false-alarm rates, on the same windows.

**One observing session and one receiver.** The calibration and test scans are off-source
scans 11 minutes apart. Intervals are per window, and scan-to-scan variation can't be estimated
until a second session is added.

## Setup

The design was fixed before the run.

- **Backgrounds:** 4,096 windows of 49 frames × 64 channels from each off-source scan, cut
  from coarse channel 6 of the 6-second product, avoiding its edges and DC region
  ([backgrounds.json](backgrounds.json) has their positions and hashes). Thresholds come from
  scan 0014 (calibration); signals are dealt into scan 0016 (test).
- **Injections:** seed 0. 72 per class and strength for five signal classes and two decoys,
  at strengths 5, 7, 10, 14, 20 and 28, plus 1,072 signal-free windows. Strength is the
  optimal matched-filter SNR (the signal's norm in noise sigma), so the same strength means
  the same detectability for every class.
- **Blinding:** detectors see only shuffled windows with opaque ids. The key is joined after
  scoring.
- **False-alarm rates:** 1% and 0.1% per window.

```bash
python -m tracks.recovery.background --cache data/recovery   # rebuild and verify the windows
python -m tracks.recovery.run --cache data/recovery --seed 0 --output runs/recovery-m1
```

## Results

The thresholds set on the calibration scan hold on the test scan's 1,072 signal-free windows:

| Detector | Threshold at 1% | False alarms | Threshold at 0.1% | False alarms |
|---|---:|---:|---:|---:|
| de-Doppler | 4.93 | 13 (1.2%, 0.7–2.1%) | 5.46 | 0 (0–0.4%) |
| energy | 3.64 | 9 (0.8%, 0.4–1.6%) | 4.77 | 2 (0.2%, 0.05–0.7%) |

Recovery at a 1% false-alarm rate (decoys: how often they're flagged). Each cell is 72
injections; at 50% its 95% interval is about ±11 points.

| Class | Detector | SNR 5 | SNR 7 | SNR 10 | SNR 14 | SNR 20 | SNR 28 |
|---|---|---:|---:|---:|---:|---:|---:|
| tone | de-Doppler | 24% | 71% | 100% | 100% | 100% | 100% |
| tone | energy | 6% | 3% | 14% | 46% | 97% | 100% |
| intermittent | de-Doppler | 4% | 12% | 60% | 90% | 100% | 100% |
| intermittent | energy | 1% | 1% | 17% | 50% | 100% | 100% |
| wideband | de-Doppler | 15% | 69% | 97% | 100% | 100% | 100% |
| wideband | energy | 6% | 3% | 29% | 60% | 100% | 100% |
| sidebands | de-Doppler | 17% | 46% | 96% | 100% | 100% | 100% |
| sidebands | energy | 4% | 7% | 17% | 49% | 100% | 100% |
| fsk | de-Doppler | 1% | 25% | 56% | 99% | 100% | 100% |
| fsk | energy | 0% | 6% | 14% | 43% | 100% | 100% |
| stationary (decoy) | de-Doppler | 57% | 94% | 100% | 100% | 100% | 100% |
| stationary (decoy) | energy | 3% | 4% | 24% | 56% | 99% | 100% |
| pulse (decoy) | de-Doppler | 0% | 1% | 1% | 1% | 6% | 6% |
| pulse (decoy) | energy | 3% | 4% | 14% | 50% | 99% | 100% |

At a 0.1% false-alarm rate:

| Class | Detector | SNR 5 | SNR 7 | SNR 10 | SNR 14 | SNR 20 | SNR 28 |
|---|---|---:|---:|---:|---:|---:|---:|
| tone | de-Doppler | 15% | 61% | 99% | 100% | 100% | 100% |
| tone | energy | 0% | 0% | 1% | 10% | 85% | 100% |
| intermittent | de-Doppler | 1% | 0% | 42% | 79% | 99% | 100% |
| intermittent | energy | 0% | 0% | 1% | 12% | 88% | 100% |
| wideband | de-Doppler | 3% | 33% | 89% | 100% | 100% | 100% |
| wideband | energy | 0% | 0% | 3% | 22% | 90% | 100% |
| sidebands | de-Doppler | 7% | 29% | 90% | 100% | 100% | 100% |
| sidebands | energy | 0% | 0% | 4% | 11% | 93% | 100% |
| fsk | de-Doppler | 0% | 4% | 36% | 88% | 100% | 100% |
| fsk | energy | 0% | 0% | 1% | 19% | 90% | 100% |
| stationary (decoy) | de-Doppler | 31% | 89% | 99% | 100% | 100% | 100% |
| stationary (decoy) | energy | 0% | 1% | 4% | 22% | 92% | 100% |
| pulse (decoy) | de-Doppler | 0% | 0% | 0% | 0% | 0% | 1% |
| pulse (decoy) | energy | 1% | 0% | 1% | 17% | 85% | 100% |

Paired outcomes over all 2,160 signal injections (the decoys excluded; `dedoppler vs energy` in
the report):

| False-alarm rate | Both | De-Doppler only | Energy only | Neither |
|---|---:|---:|---:|---:|
| 1% | 958 | 613 | 29 | 560 |
| 0.1% | 739 | 683 | 4 | 734 |

## What this shows

- **The de-Doppler search is the bar to beat for drifting narrowband signals.** At a 1% false-alarm
  rate it finds 90% or more of continuous tones, wideband signals and carriers with sidebands
  from SNR 10. Intermittent and FSK signals, whose power doesn't follow one straight track,
  need SNR 14. The energy detector needs SNR 20 for every class.
- **The two baselines are complementary only for broadband pulses.** The energy detector flags
  pulses from SNR 20, and the de-Doppler search almost never does. Over all signal injections,
  the energy detector alone catches 29 that the de-Doppler search misses, against 613 the
  other way round.
- **A zero-drift line is the easiest thing for the de-Doppler search to flag** (94% at SNR 7):
  it sits exactly on a drift trial. Conventional SETI pipelines discard zero drift as local
  interference. These results report decoys rather than filter them, so a learned detector
  can be judged on whether it separates them.
- **Thresholds must come from real off-source data.** The backgrounds aren't quite unit
  Gaussian noise. Bandpass flattening divides by the median, and the power distribution is
  slightly right-skewed, so the noise has a mean of 0.06 sigma and a standard deviation of 1.01.
  Summed over 49 frames, that raises every score. Thresholds from ideal Gaussian noise would
  give 99 and 97 false alarms in 1,072 windows (about 9%) instead of 1%, and 26 and 41 (2–4%)
  instead of 0.1%. Neighbouring channels and frames are uncorrelated, and channel offsets
  that persist over time are at white-noise level.
- **This coarse channel is quiet.** No window in either scan has a de-Doppler score above 10.
  These false-alarm rates say little about bands with strong interference.

## Limitations

- One session, one receiver and one coarse channel. Only signal-free windows from the
  calibration scan set the thresholds. At 0.1% that's about 4 exceedances, so the realized
  rate can be well off target (the test-scan measurement and its interval show how far).
- The injections are idealized: added in flattened noise-sigma units, with straight drifts,
  no scintillation, and Gaussian or two-channel line profiles. Their morphology is simpler than
  real interference.
- "Recovered" means the window was flagged. Whether a detector also locates the signal (its
  frequency and drift) isn't scored yet.

Environment: Python 3.13.17, NumPy 2.5.4. Background hashes are in
[backgrounds.json](backgrounds.json). Key fingerprint: `3edd308a156144bc` (the first 16 hex
digits of `key_sha256` in the report). Code fingerprint: `1f1660b93691a94f` (`code.sha256` in the
report). Rerunning with that code reproduces the first run's tables exactly.

# Milestone 2: learned detectors

Two of the project's models join the benchmark, scored on the same injections (key
fingerprint `3edd308a156144bc`) at thresholds set on the same calibration scan. Each model gives
two statistics:

- **surprise:** how badly the model predicts the window;
- **gain:** how much better it predicts than a model of no structure.

A carrier can be very predictable and a glitch very surprising, so the two measure different
things.

## Setup

The configurations were fixed before any training, from each model's existing settings, and
nothing was tuned afterwards.

- **Next-frame GRU** (Track A's model, `learned.py`): 28,928 parameters, as `web/js/models.js`
  counts.
  - Architecture: TF.js's GRU cell (one bias per gate, hard-sigmoid gates, the reset applied
    before the recurrent product) and a dense head, with hidden size 64 (the top of Track A's
    grid).
  - Initialization: the engine's (Glorot uniform, orthogonal recurrent kernel, zero biases).
  - Preprocessing: Track A's per-window median and MAD, then asinh.
  - Training: Track A's Adam (3e-3, epsilon 1e-7) on whole batches of 64, mean squared error
    predicting each frame from the frames before it.
  - Fidelity: the port is checked against TF.js itself. A NumPy transcription of TF.js's
    equations produced a reference (`tests/track_a_gru.json`); both the port and
    `tf.layers.gru` in the browser reproduce it to 1e-5.
  - Statistics: **surprise** is the mean squared residual; **gain** is the residual of
    predicting empty frames minus the model's.
- **Token model** (`tracks/llm`): 803,584 parameters.
  - Architecture: 16 k-means tokens of whole frames, and `SignalLM` with the track's defaults,
    its context spanning the whole window.
  - Statistics: **surprise** is bits per token; **gain** is bits per token under the training
    token frequencies minus under the model.
- **Training data:** the train scan only (0012). Of its windows, 3,072 are for fitting and 1,024
  for early stopping (patience 5, at most 30 epochs). Like Track A's work units, half of each
  set carries drifting tones at the design strengths, injected with train-only seeds.
  - The models only predict; no labels are used.
  - **Tones are a class seen in training. The other four signal classes and both decoys aren't.**

**Training outcome.**

- **GRU:** epoch 29 had the lowest development loss, 0.6414. It was still improving slowly at
  the 30-epoch limit fixed in advance.
- **Token model:** early stopping chose epoch 3, at 4.006 bits per token. That's chance level
  for 16 tokens: it learned nothing beyond their frequencies.

**Development check.** Before any test-scan scoring, a check on the train scan's development
windows measured how well each statistic tells tone windows from empty ones (AUC). The GRU
statistics here come from the first run's PyTorch cell (see below). Nothing was changed because
of the check.

| de-Doppler | energy | GRU gain | GRU surprise | token gain | token surprise |
|---:|---:|---:|---:|---:|---:|
| 0.975 | 0.849 | 0.827 | 0.625 | 0.546 | 0.466 |

```bash
python -m tracks.recovery.run --cache data/recovery --seed 0 --learned --output runs/recovery-m2
```

## Results

False alarms measured on the test scan's 1,072 signal-free windows:

| Rate | GRU surprise | GRU gain | Token surprise | Token gain |
|---|---:|---:|---:|---:|
| 1% | 13 | 13 | 10 | 11 |
| 0.1% | 2 | 0 | 0 | 0 |

The lowest strength at which each detector recovers 90% at a 1% false-alarm rate (decoys:
flagged 90%). "Over 28" means never within the design.

| Class | de-Doppler | energy | GRU gain | GRU surprise | token gain | token surprise |
|---|---:|---:|---:|---:|---:|---:|
| tone (seen in training) | 10 | 20 | 20 | over 28 | over 28 | over 28 |
| intermittent | 14 | 20 | over 28 | over 28 | over 28 | over 28 |
| wideband | 10 | 20 | 14 | over 28 | over 28 | over 28 |
| sidebands | 10 | 20 | 20 | over 28 | over 28 | over 28 |
| fsk | 14 | 20 | over 28 | over 28 | over 28 | over 28 |
| stationary (decoy) | 7 | 20 | 28 | over 28 | over 28 | over 28 |
| pulse (decoy) | over 28 | 20 | over 28 | over 28 | over 28 | over 28 |

Recovery of the learned detectors at a 1% false-alarm rate (the baselines are in the first
table above):

| Class | Detector | SNR 5 | SNR 7 | SNR 10 | SNR 14 | SNR 20 | SNR 28 |
|---|---|---:|---:|---:|---:|---:|---:|
| tone | GRU gain | 3% | 4% | 18% | 47% | 90% | 99% |
| tone | GRU surprise | 0% | 3% | 1% | 1% | 3% | 11% |
| intermittent | GRU gain | 1% | 0% | 3% | 1% | 14% | 25% |
| wideband | GRU gain | 6% | 26% | 82% | 99% | 100% | 100% |
| sidebands | GRU gain | 6% | 3% | 11% | 46% | 94% | 100% |
| fsk | GRU gain | 1% | 1% | 8% | 15% | 61% | 82% |
| stationary (decoy) | GRU gain | 3% | 4% | 17% | 44% | 85% | 94% |
| pulse (decoy) | GRU gain | 0% | 1% | 1% | 3% | 3% | 6% |
| pulse (decoy) | GRU surprise | 1% | 0% | 1% | 0% | 17% | 67% |
| any class | token surprise or gain | 0–4% | 0–3% | 0–4% | 0–6% | 0–6% | 0–22% |

Paired outcomes over all 2,160 signal injections at a 1% false-alarm rate:

| Pair | Both | Baseline only | Learned only | Neither |
|---|---:|---:|---:|---:|
| de-Doppler vs GRU gain | 810 | 761 | 16 | 573 |
| energy vs GRU gain | 653 | 334 | 173 | 1,000 |
| de-Doppler vs GRU surprise | 99 | 1,472 | 10 | 579 |
| de-Doppler vs token gain | 47 | 1,524 | 7 | 582 |

## What this shows

- **No learned detector beats the de-Doppler search.** Over all signal injections at a 1%
  false-alarm rate, it alone finds 761 that GRU gain misses, against 16 the other way round.
  It's ahead or level on every class.
- **GRU gain detects predictable structure, and partly complements the energy detector.**
  - It finds wideband signals from SNR 14, and carriers with sidebands from SNR 20.
  - It flags zero-drift lines, which are predictable. It rarely flags broadband pulses, which
    aren't (6% at SNR 28).
  - Against the energy detector, it catches 173 injections the energy detector misses, which
    in turn catches 334 that GRU gain misses.
- **Even on tones, the class it trained on, the GRU is far behind.** It recovers 47% at SNR 14,
  against the de-Doppler search's 100%. A line in one of 64 channels is a small part of a
  mean-squared-error loss dominated by unpredictable noise: 29 epochs moved the development loss
  only to 0.6414. Intermittent and FSK signals, which break the frame-to-frame continuity it
  learned, are the hardest for it (25% and 82% at SNR 28).
- **GRU surprise loses the energy detector's information.** Track A's preprocessing divides
  each window by its own spread, which removes the excess power a radiometer relies on. It only
  responds to the strongest signals (pulses: 67% at SNR 28).
- **The token model detects almost nothing.** Both statistics stay within a few percent of the
  false-alarm rate for every class and strength. The exceptions are token gain on wideband
  signals at SNR 28 (22%) and on zero-drift lines (10%). Its 16 k-means tokens of whole
  64-channel frames don't preserve signals this weak, and the model learned nothing beyond token
  frequencies.

As the project builds them now, these models measure predictability, not detection. Directions
this points to:

- **Supervised detectors**, trained on injected classes, with each class held out in turn to
  test morphologies they haven't seen.
- **Tokenizers that keep narrowband structure**, for example per channel or drift-aware.
- **Preprocessing that keeps a window's power.**

## First run: PyTorch's default GRU

The first run of this milestone (commit `b0ed65c`, code fingerprint `4ccee4de24def65c`) used
PyTorch's `nn.GRU` in place of Track A's cell. That cell has 29,120 parameters, separate input
and recurrent biases, logistic gates and the reset applied after the recurrent product. It was
trained otherwise identically, apart from PyTorch's Adam epsilon and a partial last batch.

Review found that it wasn't Track A's model, so the port above replaced it. Under the benchmark
protocol, a method changed after its test-scan score is a new method, and the earlier score
stays on record:

- **Training:** early stopping chose epoch 4 (development loss 0.6416).
- **90% recovery at 1% false alarms (gain):** tones from SNR 28, wideband from SNR 10,
  sidebands from SNR 20, FSK from SNR 28, intermittent never.
- **Paired with the de-Doppler search:** 910 both, 661 de-Doppler only, 30 GRU gain only, 559
  neither.
- **Paired with the energy detector:** 225 GRU gain only, 272 energy only.

Its gain was a little stronger than the port's on wideband, sideband and FSK signals, and its
surprise slightly stronger on tones at SNR 28 (36% against 11%). The conclusions are the same.
The token model and the baselines are bit-for-bit identical in both runs.

## Limitations

- These results inherit everything in the first measurement's limitations: one session, one
  receiver and one coarse channel, idealized injections, and window-level detection only.
- **One configuration and one training seed per model.** Other configurations might do better,
  and the GRU was still improving slowly at its epoch limit. This measures the models as the
  project builds them, not their best possible versions.
- The port is checked against TF.js's layer, not against a model trained in the browser. The
  training data, epochs and frame counts are this benchmark's, not those of Track A's work
  units.
- The development check was seen before the test runs. Nothing was changed because of it.

Environment: Python 3.13.17, NumPy 2.5.4, PyTorch 2.14.1 (CPU, one thread for training).

Fingerprints (the first 16 hex digits of the hashes in the report):

- code: `c2819ce57b2f0df9`, over `tracks/recovery` and the `tracks/llm` modules it runs;
- training windows: `728ad428140c6c6c`;
- GRU state: `9cc159f1505ed0e8`;
- token model state: `c9c25f4c59453105`.
