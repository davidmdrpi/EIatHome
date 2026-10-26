# Blinded signal recovery

Can a method recover a weak signal without raising false alarms? This track measures it.
Synthetic signals of known class and strength are injected into real off-source data, dealt
blind to detectors, and scored at a fixed false-alarm rate against fixed baselines. The plan
and milestones are in [issue #13](https://github.com/davidmdrpi/EIatHome/issues/13); the first
measurement is in [RESULTS.md](RESULTS.md).

## Run it

From the repository root, with `numpy` (and `h5py` and `hdf5plugin` to build the backgrounds):

```bash
python -m tracks.recovery.background --cache data/recovery    # ~510 MB of range reads, once
python -m tracks.recovery.run --cache data/recovery --seed 0 --output runs/recovery
```

The first command reads coarse channel 6 of the three off-source 6-second scans over HTTPS. It
writes 4,096 windows per scan to `data/recovery/` (git-ignored) and checks them against
[backgrounds.json](backgrounds.json), which records each window's position and each scan's
SHA-256. The second command writes three files:

- `report.json`: thresholds, false alarms, recovery and paired outcomes, together with the
  design, each detector's configuration, a fingerprint of this package's code and its git
  revision;
- `key.json`: every injection and every score;
- `report.md`.

## How it works

1. **Backgrounds** ([background.py](background.py)): 49-frame × 64-channel windows of
   bandpass-flattened spectrogram, at seeded positions that don't overlap and that skip the
   coarse channel's edges and DC region. Real interference stays in. Each scan has a role:
   - **train** (0012), for learned detectors in later milestones;
   - **calibration** (0014), whose windows set the thresholds;
   - **test** (0016), which receives the injections.
2. **Injections** ([inject.py](inject.py)): the classes are
   - **signals:** a drifting tone, an intermittent tone, a wideband signal, a carrier with two
     sidebands, and frequency-shift keying;
   - **decoys:** a zero-drift line and a one-frame broadband pulse;
   - **signal-free windows.**

   Each spec is drawn from a seeded generator, and the whole signal always fits inside the
   window. A signal's strength is its norm in noise sigma: the SNR of a matched filter that
   knows the exact signal.
3. **Blinding:** `deal` gives detectors only the windows, shuffled, with opaque ids. The specs
   stay in a separate key, joined after scoring. Because the code is public, nothing hides the
   key from a person. What the dealer guarantees is that detectors never see it, and that the
   design is fixed before results come in.
4. **Detectors** ([detectors.py](detectors.py)) map windows to scores, higher meaning more
   likely a signal. The fixed baselines are:
   - **de-Doppler:** the strongest straight track over every drift trial and start channel,
     with no wrap-around (as in `tools/extract_slices.py`);
   - **energy:** total power above the noise expectation.
5. **Scoring** ([run.py](run.py)): thresholds at 1% and 0.1% false alarms per window, set on
   the calibration scan. On the test scan the report gives:
   - false alarms on its signal-free windows, with 95% Wilson intervals;
   - recovery per class and strength;
   - flag rates for the decoys;
   - each injection's paired outcome under the two detectors.

A learned detector is any function from a stack of windows to scores. Add it to `DETECTORS`
and it gets the same treatment as the baselines:

- thresholds from the calibration scan;
- the same injections;
- paired outcomes against every other detector.

Give it a `config` dict naming the model, checkpoint hash and training data. Reports record it
next to the code fingerprint, so scores from different revisions or volunteers aren't pooled
by mistake.

## Benchmark protocol

The test scan's injections are a benchmark. They measure a finished method; they're not for
tuning one.

- **Development:** develop and select learned detectors on the train scan (0012) only, with
  their own injection seeds. Split that scan into a fitting part and a development part if
  model selection needs one.
- **Evaluation:** score each selected configuration once on the test scan, and report it with
  its configuration and code fingerprint.
- **Changes after scoring:** a method changed after seeing test-scan results is a new method.
  Its test-scan score goes next to the earlier one, not in place of it.

A second observing session (milestone 4) will give a test that no development has seen at all.

## Tests

```bash
python -m unittest discover -s tests -p 'test_recovery.py' -v
```

They use synthetic Gaussian backgrounds, so they need only `numpy`. They cover:

- **Injections:** every class's norm equals its strength, signals stay inside the window,
  tracks follow their drift, each class has its shape, and draws are seeded.
- **Dealer:** detectors get only opaque ids and arrays, and the key rebuilds every window. The
  dealing order and the background each window gets are both independent of class.
- **Detectors:** the de-Doppler search equals a brute-force search and never wraps, it finds a
  tone at its strength, and the energy score is standardized.
- **Thresholds:** they hold their false-alarm rate on fresh noise, and come from the
  calibration scan only.
- **More detectors:** adding a third detector keeps every pairing, and leaves the existing
  pair's outcomes unchanged.
- **Provenance:** reports record the code fingerprint, the git revision and each detector's
  configuration.
- **End to end:** an evaluation reproduces the baselines' known behaviour, and the background
  manifest check rejects altered windows.
