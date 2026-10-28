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

# Milestone 3: recovery units in the browser

The Live Lab's **Recovery** tab runs the blinded comparison as a work unit, `recovery/v0`
([web/js/recovery.js](../../web/js/recovery.js)). Every unit number draws different signals
into the same real windows, so units add up. A submitted card is recomputed before it's
recorded.

## The spec

- **Windows:** every 4th of the test scan's 4,096 windows. These 1,024 windows span the whole
  coarse channel, in 3.2 MB (`web/data/recovery/test.i8`).
  - They're quantized to steps of 1/16 noise sigma, which moves each value by at most 1/32
    sigma.
  - Nothing clips: the largest value in either scan, 7.92 sigma, is under the 7.94 sigma that
    8 bits hold.
- **Thresholds:** set on all 4,096 calibration windows, quantized the same way. The spec pins
  them with the file's SHA-256, so changing either needs a new spec.

  | Detector | 1% | 0.1% |
  |---|---:|---:|
  | de-Doppler | 4.9375 | 5.4732 |
  | energy | 3.6532 | 4.7873 |

  Milestone 1's full-precision thresholds were 4.93 and 5.46, and 3.64 and 4.77.
- **Design:** each unit has 18 injections per class and strength (756 in all) and 268 windows
  left empty. Its seed is derived from the unit number, drawn with mulberry32
  (`web/js/rng.js`).
- **Detectors:** the same de-Doppler search and energy detector, in JavaScript.
- **Card:** about 2 KB. It holds the counts flagged per detector, false-alarm rate, class and
  strength, and how many were flagged by both detectors.

```bash
python -m tracks.recovery.browser --cache data/recovery   # rebuild the published files and compare
node tools/recovery_unit.mjs 0 999                         # units 0 to 999, added up (about 6 minutes)
```

## Checks

- **The two implementations agree.** `tracks/recovery/browser.py` ports the unit to Python,
  using `inject.py`'s own `draw()` and `render()` and `detectors.py`'s detectors. Its tests run
  units 0 and 41 in Node and in Python and compare:
  - the injected signals are identical;
  - the de-Doppler scores are bit-identical, and the energy scores agree to 1e-13 (the sums run
    in a different order);
  - every count is identical.
- **Ingestion recomputes.** The GitHub Action reruns each submitted unit on the published
  windows and records the card only if its counts match. A unit takes under a second.
- **Units reproduce.** A unit run twice gives the same counts. The page and Node give the same
  counts for the same unit.

## Reference: 1,000 units

Units 0 to 999, added up: 18,000 injections per class and strength. At 50% recovery, the
injections' 95% interval is about ±0.7 points. In brackets, milestone 1's single draw: 72
injections on all 4,096 windows at full precision, seed 0.

Recovery at a 1% false-alarm rate (decoys: how often they're flagged):

| Class | Detector | SNR 5 | SNR 7 | SNR 10 | SNR 14 | SNR 20 | SNR 28 |
|---|---|---:|---:|---:|---:|---:|---:|
| tone | de-Doppler | 25% (24%) | 74% (71%) | 99% (100%) | 100% (100%) | 100% (100%) | 100% (100%) |
| tone | energy | 2% (6%) | 3% (3%) | 12% (14%) | 48% (46%) | 99% (97%) | 100% (100%) |
| intermittent | de-Doppler | 3% (4%) | 14% (12%) | 53% (60%) | 91% (90%) | 100% (100%) | 100% (100%) |
| intermittent | energy | 2% (1%) | 3% (1%) | 11% (17%) | 46% (50%) | 98% (100%) | 100% (100%) |
| wideband | de-Doppler | 17% (15%) | 56% (69%) | 98% (97%) | 100% (100%) | 100% (100%) | 100% (100%) |
| wideband | energy | 2% (6%) | 5% (3%) | 16% (29%) | 57% (60%) | 99% (100%) | 100% (100%) |
| sidebands | de-Doppler | 15% (17%) | 54% (46%) | 95% (96%) | 100% (100%) | 100% (100%) | 100% (100%) |
| sidebands | energy | 2% (4%) | 4% (7%) | 13% (17%) | 52% (49%) | 99% (100%) | 100% (100%) |
| fsk | de-Doppler | 4% (1%) | 17% (25%) | 60% (56%) | 97% (99%) | 100% (100%) | 100% (100%) |
| fsk | energy | 2% (0%) | 3% (6%) | 12% (14%) | 48% (43%) | 99% (100%) | 100% (100%) |
| stationary (decoy) | de-Doppler | 55% (57%) | 97% (94%) | 100% (100%) | 100% (100%) | 100% (100%) | 100% (100%) |
| stationary (decoy) | energy | 2% (3%) | 4% (4%) | 12% (24%) | 48% (56%) | 99% (99%) | 100% (100%) |
| pulse (decoy) | de-Doppler | 2% (0%) | 1% (1%) | 2% (1%) | 2% (1%) | 4% (6%) | 6% (6%) |
| pulse (decoy) | energy | 2% (3%) | 3% (4%) | 12% (14%) | 47% (50%) | 99% (99%) | 100% (100%) |

At 0.1%:

- **de-Doppler search:** 90% of tones from SNR 10 (97%). Wideband signals and carriers with
  sidebands need SNR 14 (87% and 86% at SNR 10), and intermittent and FSK signals SNR 20 (81%
  and 88% at SNR 14).
- **Energy detector:** 90% of wideband signals and carriers with sidebands from SNR 20 (94% and
  91%). The other classes need SNR 28 (88–90% at SNR 20).

**False alarms.** With nothing injected, the de-Doppler search flags 9 of the 1,024 windows at
the 1% threshold and 1 at 0.1%. The energy detector flags 7 and 0. Every unit's empty windows
are drawn from these windows, so combined units measure the same false alarms over and over:
2,343 and 1,846 of 268,000 at 1% (0.87% and 0.69%), and 259 and 0 at 0.1%.

**Paired outcomes** over all 540,000 signal injections at 1%:

| | Both | De-Doppler only | Energy only | Neither |
|---|---:|---:|---:|---:|
| 1,000 units | 43.7% | 28.8% | 1.0% | 26.6% |
| Milestone 1 | 44.4% | 28.4% | 1.3% | 25.9% |

## What this shows

- **The browser benchmark reproduces milestone 1.** Of the 168 cells at both rates, 160 lie in
  milestone 1's 95% intervals. Chance alone would put about 8 outside, and 8 are outside.
  At 1%, each signal class reaches 90% at the same strength as before:
  - **de-Doppler at 1%:** SNR 10 for tones, wideband signals and carriers with sidebands; SNR
    14 for intermittent and FSK signals;
  - **energy detector:** SNR 20 for every class.
- **Combining units removes the variation between injection draws.**
  - Milestone 1's single draw: the energy detector found 18% of all injections at SNR 10,
    decoys included.
  - Five other seeds on the same full scan give 12–15% (13.5% on average), and the 1,000 units
    give 12.8%.
  - Milestone 1's outlying cells (energy on wideband signals at SNR 10: 29%) were that
    variation, not a property of the class.
  - The small remaining gap fits the window subset: every 4th window has slightly lower energy
    scores than the full scan.
- **The energy detector sees only power, so it barely distinguishes classes.** At SNR 10 it
  finds 11–13% of every class except wideband signals (16%). The likely reason: the backgrounds'
  mean of 0.06 sigma (milestone 1) adds to a signal spread over more channels. The de-Doppler
  search spreads widely across classes, from 53% (intermittent) to 99% (tones) at SNR 10.

## Limitations

- **Every unit uses the same 1,024 windows from one scan.** Combining units narrows the
  intervals over injection draws only.
  - False alarms are fixed by those windows.
  - Recovery intervals from many units leave out window-to-window, scan-to-scan and
    receiver-to-receiver variation. Milestone 4's second session is where that gets measured.
- **The thresholds aren't recomputed in the browser.** They come from the calibration scan,
  which isn't published. The cross-check shows the JavaScript detectors equal the Python ones,
  and `python -m tracks.recovery.browser` rebuilds the thresholds from the background cache.
- **Recomputation confirms a card's counts, not the volunteer's computation.** For these
  baselines a unit costs under a second, so the Action can rerun it. Units with detectors
  trained in the browser would be too costly to recompute and will need replication instead.
- Everything in the earlier limitations still applies: one session, one receiver and one coarse
  channel, idealized injections, and window-level detection only.

# Milestone 4: a second session, and evaluation across sessions

Everything above comes from one observing session (2020-07-16). This milestone adds a second,
four years earlier: Voyager 1 on 2016-09-19 (MJD 57650), the file turboSETI's tutorial uses
(`Voyager1.single_coarse.fine_res.h5`). Both sessions come from the same telescope (Green Bank),
the same receiver and the same coarse channel (8418.46 to 8421.39 MHz). The question: does a
threshold set on one session hold on another, and do detectors recover the same signals there?
Code: [sessions.py](sessions.py).

## Setup

- **Product:** the 2016 file exists only as a high-frequency-resolution product (16 frames of
  18.25 s, 2.79 Hz channels), so both sessions use that product.
  - Windows are 16 frames × 64 channels, not the 49 × 64 of milestones 1 to 3.
  - These numbers compare with each other, not with the earlier milestones'.
- **2020** (MJD 59046): the high-resolution files of off-source scans 0014 (calibration) and
  0016 (test), 4,096 windows each.
- **2016** (MJD 57650): one on-source scan, downloaded from this repository's v0.1 release and
  checked against its SHA-256. Voyager's signals are in the band:
  - **carrier:** at 8419.29703 MHz, the strongest de-Doppler track (SNR 986);
  - **telemetry sidebands:** at ±22.5 kHz (SNR 118 and 121);
  - **third harmonics:** at ±67.5 kHz (SNR 17 and 15);
  - **fifth and seventh harmonics:** at noise level (SNR about 5, as in control regions far from
    the carrier).

  Windows within 180 kHz of the carrier are left out. Of the 8,192 cut, alternate windows go to
  calibration and test.
- **Noise:** away from Voyager, the 2016 data has a mean of 0.037 sigma and a standard
  deviation of 1.004, its largest value 6.5. The 2020 off-source scans have 0.036 and 1.004, the
  largest 6.0. Both are quiet.
- **Injections:** seed 0, 72 per class and strength, the design of milestone 1.
  - The dealer's draws don't depend on the windows, so both test sets get the same injections.
    Every injection has a **twin** in the other session: the same signal, in the window at the
    same position of the other session's test set.
  - The key fingerprint `4d7a2ba7d178a86f` is the same in all four pairings below.
- **Thresholds:** set on each session's calibration windows at 1% and 0.1%, and applied to both
  sessions' test windows.
- **Comparison across sessions:** each injection against its twin, done twice:
  - **at each session's own thresholds:** the whole difference between the sessions;
  - **at 2020's thresholds for both:** only the difference between their windows.

  Each comparison gives an exact McNemar test and the paired difference in recovery (2016 minus
  2020), with a 95% interval (Newcombe's hybrid score interval for paired proportions). No
  equivalence margin was set in advance. So these comparisons can find a difference, or bound
  its size; they can't show that the sessions are equivalent.

```bash
python -m tracks.recovery.sessions build --cache data/recovery   # rebuild and verify the windows (about 3 minutes)
python -m tracks.recovery.sessions run --cache data/recovery --seed 0 --output runs/sessions
```

## Results

The thresholds:

| Detector | Rate | 2020 | 2016 |
|---|---|---:|---:|
| de-Doppler | 1% | 4.594 | 4.590 |
| de-Doppler | 0.1% | 5.019 | 5.031 |
| energy | 1% | 2.846 | 2.858 |
| energy | 0.1% | 3.585 | 3.773 |

False alarms on each test set's 1,072 empty windows, with thresholds from either session:

| Detector | Rate | Thresholds from | Test 2020 | Test 2016 |
|---|---|---|---:|---:|
| de-Doppler | 1% | 2020 | 11 (1.0%) | 6 (0.6%) |
| de-Doppler | 1% | 2016 | 11 (1.0%) | 6 (0.6%) |
| de-Doppler | 0.1% | 2020 | 3 (0.3%) | 1 (0.1%) |
| de-Doppler | 0.1% | 2016 | 2 (0.2%) | 1 (0.1%) |
| energy | 1% | 2020 | 6 (0.6%) | 11 (1.0%) |
| energy | 1% | 2016 | 4 (0.4%) | 10 (0.9%) |
| energy | 0.1% | 2020 | 1 (0.1%) | 0 (0%) |
| energy | 0.1% | 2016 | 0 (0%) | 0 (0%) |

**Where recovery reaches 90%:** at 1%, every class reaches it at the same strength in both
sessions, for both detectors:

- **de-Doppler search:** SNR 10 for tones, wideband signals and carriers with sidebands; SNR 14
  for intermittent and FSK signals; SNR 7 for zero-drift lines; never for pulses.
- **Energy detector:** SNR 14 for every class.

At 0.1%, only wideband signals and carriers with sidebands differ: they cross 90% at SNR 10 in
one session and at SNR 14 in the other, from cells of 86% to 93% at SNR 10.

**The same injections in both sessions:**

| Detector | Rate | Thresholds | Injections | Flagged in 2020 | Flagged in 2016 | McNemar p | 2016 minus 2020, points (95% interval) |
|---|---|---|---|---:|---:|---:|---:|
| de-Doppler | 1% | own | 2,160 signals | 1,629 | 1,634 | 0.80 | +0.2 (−1.2 to +1.7) |
| de-Doppler | 1% | own | 864 decoys | 428 | 441 | 0.26 | +1.5 (−0.9 to +3.9) |
| de-Doppler | 0.1% | own | 2,160 signals | 1,499 | 1,494 | 0.79 | −0.2 (−1.6 to +1.1) |
| energy | 1% | own | 2,160 signals | 1,232 | 1,254 | 0.22 | +1.0 (−0.5 to +2.6) |
| energy | 0.1% | own | 2,160 signals | 1,082 | 1,058 | 0.13 | −1.1 (−2.5 to +0.3) |
| energy | 0.1% | own | 864 decoys | 450 | 424 | 0.015 | −3.0 (−5.3 to −0.7) |
| energy | 0.1% | 2020's | 2,160 signals | 1,082 | 1,100 | 0.28 | +0.8 (−0.6 to +2.3) |
| energy | 0.1% | 2020's | 864 decoys | 450 | 431 | 0.081 | −2.2 (−4.5 to +0.1) |

The report has the rest: each class, the other rows at 2020's thresholds, and every pairing's
recovery table.

## What this shows

- **In this coarse channel, overall recovery shows no significant difference between sessions
  four years apart, and the paired intervals keep it within a few points.**
  - The thresholds are close: 4.594 and 4.590 for the de-Doppler search at 1%.
  - Each session's thresholds hold their false-alarm rate on the other session's windows.
  - Over all 2,160 signals at 1%, the de-Doppler search finds 1,629 of the same injections in
    2020 and 1,634 in 2016 (p = 0.80): a paired difference of +0.2 percentage points, with a 95%
    interval of −1.2 to +1.7. The energy detector finds 1,232 and 1,254 (p = 0.22): +1.0 points
    (−0.5 to +2.6).
  - These intervals bound the overall difference at a few points. Without an equivalence
    margin set in advance, they don't show that the sessions are equivalent.
- **One comparison does differ: decoys at the energy detector's 0.1% threshold.**
  - At its own thresholds, 2016 flags 26 fewer of the 864 decoys (424 against 450): −3.0 points
    (−5.3 to −0.7), p = 0.015. Pulses alone give p = 0.044.
  - The 0.1% thresholds rest on about 4 of 4,096 calibration windows, and 2016's energy
    threshold came out higher (3.77 against 3.59).
  - With 2020's threshold on both sessions, the gap shrinks to 19 decoys (431 against 450):
    −2.2 points (−4.5 to +0.1), p = 0.081. The shared threshold removes 7 of the 26.
  - So the difference is reduced, and what remains isn't significant in this comparison. It
    isn't shown to be absent: the interval allows a difference between the sessions' windows of
    up to 4.5 points. Threshold sampling noise is one contributor, not shown to be the only one.
  - Of the report's 36 comparisons at own thresholds (per class, and over all signals and all
    decoys), two fall below p = 0.05, about what chance alone would give across 36 tests.
- **Milestone 3 asked how far results from one session's windows would carry.** Here, the paired
  intervals for overall recovery are a few points wide, much narrower than a single cell's 95%
  interval (about ±7 to ±11 points with 72 injections). We found no sign, at the resolution of
  the recovery curves, that recovery measured on one session's windows misleads about the other.
  That's for this channel and these two sessions; it isn't a demonstration of equivalence.
- **Shorter windows favour the energy detector.** With 16 frames (1,024 values per window), it
  reaches 90% at SNR 14, against SNR 20 with 49 frames (3,136 values). A signal of a given
  strength stands out further from the total noise power of fewer values. The de-Doppler search
  reaches 90% at the same strengths as with 49 frames.

## Limitations

- **A quiet channel.** Both sessions' windows are close to the noise model, with no strong
  interference. Bands crowded with interference may not carry over this well.
- **One telescope, one receiver, one coarse channel.**
  - The 2016 windows come from an on-source scan, away from Voyager's lines. The 2020 windows
    come from off-source scans.
  - Within 2016, the calibration and test windows alternate along one scan. 2020's come from two
    scans 11 minutes apart.
- **The fixed baselines only.** The learned detectors of milestone 2 were trained on 49-frame
  windows. Comparing them across sessions needs retraining on 16-frame windows.
- **The browser unit (milestone 3) still uses 2020's windows only.**

Environment: Python 3.13.17, NumPy 2.5.3. The windows' hashes are in
[sessions.json](sessions.json), and the 2016 file's in [sessions.py](sessions.py) (SHA-256
`c9a9a54f…`). Code fingerprint: `fbd2b42166aa3d06`. A rebuild from the original files
reproduces every window exactly, and a rerun reproduces every table.
