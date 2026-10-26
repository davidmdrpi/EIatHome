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
