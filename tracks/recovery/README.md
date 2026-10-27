# Blinded signal recovery

Can a method recover a weak signal without raising false alarms? This track measures it.
Synthetic signals of known class and strength are injected into real off-source data, dealt
blind to detectors, and scored at a fixed false-alarm rate against fixed baselines. The plan
and milestones are in [issue #13](https://github.com/davidmdrpi/EIatHome/issues/13). The
measurements are in [RESULTS.md](RESULTS.md): the baselines, then the project's learned
models, then the browser work unit. The Live Lab's **Recovery** tab runs the comparison in a
browser (see [In the browser](#in-the-browser)).

## Run it

From the repository root, with `numpy` (and `h5py` and `hdf5plugin` to build the backgrounds):

```bash
python -m tracks.recovery.background --cache data/recovery    # ~510 MB of range reads, once
python -m tracks.recovery.run --cache data/recovery --seed 0 --output runs/recovery
python -m tracks.recovery.run --cache data/recovery --seed 0 --learned --output runs/recovery-learned
```

`--learned` also trains and scores the learned detectors, on the train scan only. It needs
PyTorch, installed as for [tracks/llm](../llm/README.md), and takes about 4 minutes on one CPU
thread. The trained models are saved as `learned.pt`.

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

## Learned detectors

[learned.py](learned.py) trains two of the project's models on the train scan:

- Track A's next-frame GRU, ported from TF.js: its GRU cell, initializers and optimizer settings.
  `tests/track_a_gru.json` holds a reference computed from TF.js's equations. The port
  reproduces it, and so does `tf.layers.gru` itself in the browser tests.
- the `tracks/llm` token model.

Each gives two detectors:

- **surprise:** how badly the model predicts the window;
- **gain:** how much better it predicts than a model of no structure.

Training follows Track A's work units: signal-free windows plus synthetic drifting tones,
injected with train-only seeds. No labels are used. Each detector's `config` records:

- the model and its hyperparameters;
- a hash of its training windows;
- its best epoch and a hash of its trained state.

Any function from a stack of windows to scores can be a detector. Add it to `DETECTORS` and it
gets the same treatment as the baselines:

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

## In the browser

The Live Lab's **Recovery** tab runs this comparison as a work unit, `recovery/v0`
([web/js/recovery.js](../../web/js/recovery.js)).

- **Windows:** every 4th of the test scan's windows, 1,024 in all, quantized to steps of 1/16
  noise sigma: 3.2 MB in `web/data/recovery/`. [browser.py](browser.py) writes them, together
  with thresholds set on the whole calibration scan, quantized the same way. The spec pins the
  file's SHA-256 and the thresholds.
- **A unit:** 18 injections per class and strength, and 268 windows left empty, drawn from the
  unit number. Every unit draws different signals into the same windows, so their counts add
  up.
- **Checks:** `browser.py` ports the unit to Python, using this package's own injections and
  detectors. Its tests check that both draw the same signals and give the same scores and
  counts. Ingestion reruns each submitted unit and records only cards whose counts match.

```bash
python -m tracks.recovery.browser --cache data/recovery          # check the published files against a rebuild
python -m tracks.recovery.browser --cache data/recovery --write  # rewrite them (a new spec must pin them)
node tools/recovery_unit.mjs 0 999                               # run units 0 to 999 in Node, added up
```

## Tests

```bash
python -m unittest discover -s tests -p 'test_recovery.py' -v
```

They use synthetic Gaussian backgrounds, so they need only `numpy`. The browser unit's tests
need Node as well, to run the JavaScript unit they compare against:

```bash
python -m unittest discover -s tests -p 'test_recovery_browser.py' -v
```

The learned detectors' tests need PyTorch too:

```bash
python -m unittest discover -s tests -p 'test_recovery_learned.py' -v
```

Those tests cover:

- the fit and development windows split the train scan, with tones in half of each;
- the GRU reproduces the TF.js reference, has Track A's parameter count (28,928 at hidden 64)
  and its initialization;
- the preprocessing is Track A's, checked against a line-by-line port of the JavaScript;
- training is seeded and keeps the best development epoch's weights;
- token scores match `tracks/llm`'s own evaluation;
- every config records the model and its training;
- learned detectors are trained on the train scan alone and compared with every baseline.

The numpy-only tests cover:

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
  configuration. The fingerprint and the uncommitted-changes check cover the `tracks/llm`
  modules the learned detectors run, and changing any of them changes the identity.
- **End to end:** an evaluation reproduces the baselines' known behaviour, and the background
  manifest check rejects altered windows.
- **Browser unit** (`test_recovery_browser.py`, which also needs Node): the Python port and
  the JavaScript unit deal identical signals, and give the same scores and counts. The
  published windows match their index and refuse a changed byte, and quantizing refuses to
  clip. With the background cache, a rebuild reproduces the published files exactly.
