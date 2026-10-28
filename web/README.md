# EI@Home Live Lab

A static, zero-install web app: pick a signal, train a small GRU on it in the browser, look at
its predictions and hidden states, measure how language-like its radio tokens are, and run
reproducible work units. The Recovery tab measures which weak signals two standard detectors
find in real data, at a fixed false-alarm rate. Everything computes locally; there is no server.

## Run it locally

Any static file server works. From this folder:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>. Opening `index.html` straight from disk won't work, because
browsers don't run module workers from `file://` URLs.

## How it's built

| File | What it does |
|---|---|
| `index.html`, `css/app.css` | Page structure and theme tokens (light and dark have separate, validated chart colours) |
| `js/main.js` | UI: tabs, signal switcher, rendering, work units, slice loading |
| `js/engine.worker.js` | Track A engine (module worker): builds snippets, trains the GRU, LSTM or causal-attention model with TensorFlow.js, computes baselines, gains, the hidden-state PCA and the drift probe. Also trains Track B's token model (GRU or transformer) |
| `js/layers.js` | Model layers shared by Track A and Track B (causal attention, learned positions, layer normalization) and the token-model builder, importable outside the engine so tests can check them |
| `js/models.js` | Model families and their parameter counts, so architectures can be compared at about the same size (Track A models, and Track B's token models) |
| `js/trackb-splits.js` | Track B's training, validation and test frames for each kind of family, split before anything is fitted |
| `js/trackb.worker.js`, `js/trackb.js` | Track B: splits each family into training, validation and test segments, fits the k-means tokenizer and the shuffled and Markov-1 surrogates on the training split only, and measures held-out cross-entropy, conditional entropy, mutual information, Zipf and compression without crossing a segment boundary |
| `js/signals.js` | Signal families (two real-data ones, Voyager 1 and off-source, and the synthetic controls) and the Rosetta text; snippet normalization |
| `js/slices.js` | Loads real-data micro-slices (`data/slices/`) or user-dropped files, verifies their SHA-256 and computes the dataset identity |
| `js/splits.js` | Groups a real family's slices into observing sessions and keeps training and evaluation on different sessions |
| `js/workunits.js` | Versioned work-unit specs |
| `js/recovery.js`, `js/recovery.worker.js` | The `recovery/v0` work unit (`tracks/recovery`, issue #13): injects synthetic signals into the real off-source windows in `data/recovery/`, deals them blind to the de-Doppler and energy detectors, and counts what each finds at thresholds pinned in the spec. Also checks recovery cards and adds units up |
| `js/rng.js` | Seeded random numbers, so everything is reproducible |
| `js/plots.js` | Small canvas charts with hover tooltips |
| `vendor/tfjs/` | TensorFlow.js 4.22.0, vendored unmodified (see its README) |

Training runs in a Web Worker so the page never freezes. "Auto" starts with WebAssembly:
for models this small it's usually faster than a GPU, because GPU start-up and per-step
overhead outweigh the speed-up. WebGPU and WebGL are one click away, and the status line
shows milliseconds per epoch so you can compare on your own machine. Multithreaded
WebAssembly needs cross-origin isolation headers, which GitHub Pages doesn't send, so it
runs single-threaded there.

## Tests

From the repository root:

```bash
npm ci
npx playwright install chromium
npm test
```

`tests/unit.test.mjs` checks the data layer in Node: slice hash verification, dataset identity,
that each real family samples only its own slices, session splits (including renamed copies and
missing start times), order independence of the sampled inputs, and the pinned work-unit corpus.
For Track B it checks that the tokenizer, surrogates and n-gram model never see the test split
(including a Rosetta test text change end to end), that the n-gram fits the same data as the token
model, that no statistic or compressed block crosses a segment boundary, and that real data holds
out whole scans or sessions.
`tests/recovery.test.mjs` checks the recovery unit: the spec pins the published windows and
thresholds, every class is injected at its strength inside the window, the dealer's key
rebuilds every window, the de-Doppler search equals a brute-force search, units reproduce,
and tampered cards are caught by their form or by recomputation.
`tests/community.test.mjs` checks card validation, replication summaries and the ingestion
script, and `tests/ingest_issues.test.mjs` runs the ingestion driver against a local git
remote and a fake `gh`. `tests/browser.test.mjs` serves this folder and drives the Live Lab in
headless Chromium. It checks that:

- slice metadata is always shown as text;
- the published slices load, and a run on one session is labelled as reconstruction;
- reordering a manifest and renaming its files gives identical work-unit results;
- work units give identical metrics whatever custom text is set;
- Track B finds held-out structure in Voyager data but not in off-source noise, and scores the
  other real family as a control;
- both token models (GRU and transformer) find the structure in Rosetta text but not in shuffled
  tokens, at the parameter count the size matching promises, and the transformer's continuation
  stays within its trained positions;
- attention is causal (later tokens or frames never change earlier outputs, and a prefix pass
  matches the full pass), and the custom layer norm matches TF.js's in outputs and gradients;
- token-model training can be stopped, and the controls come back;
- a recovery unit run on the page gives a card that ingestion's checks accept, recomputation
  included, and community recovery units are combined once per unit, apart from the replicated
  units.

CI runs both on every pull request.

Build table cells with `textContent` (see `makeRow` in `js/main.js`). Never put text from a
manifest, a file or browser storage into `innerHTML`.

## Common changes

- **Add a signal family:** add an entry to `FAMILIES` and a generator to `GENERATORS` in
  `js/signals.js`. It appears in the switcher, the gains table, the hidden-state map and
  Track B automatically.
- **Add a Track B metric:** write it in `js/trackb.js` so that it takes an array of token
  segments and never looks across their boundaries, compute it for every stream in
  `analyzeSplits()`, and plot it in `renderTrackB()` in `js/main.js`. Anything fitted (like a
  codebook or a surrogate) must use the training split only.
- **Change the recovery windows or thresholds:** they're part of `recovery/v0`. Write them with
  `python -m tracks.recovery.browser --write`, then pin them in a new spec in `js/recovery.js`.
- **Add a work-unit spec:** add a new versioned config in `js/workunits.js`. Never change an
  existing spec, because published result cards refer to it. Make a `v1` instead.
- **Publish real data:** see `data/slices/README.md`. A new real family also needs an entry
  with `real: true` in `FAMILIES` and a generator that calls `sliceWindow()`.

## Deploying

`.github/workflows/pages.yml` deploys this folder to GitHub Pages on every push to `main`
that touches `web/`. Enable it once in the repository's **Settings → Pages → Build and
deployment → Source: GitHub Actions**.

## Community results

`.github/workflows/ingest-results.yml` runs when an issue titled `Work unit result: …` is
opened, every six hours, and on demand. Each run calls `tools/ingest_issues.mjs`, which runs
`tools/ingest_result.mjs` to validate the card with `js/community.js` and append it to
`data/community_results.json`. It then commits that file to `main`, starts the Pages deploy
(pushes made with the workflow token don't trigger other workflows), replies on the issue, and
labels it:

| Label | Meaning |
|---|---|
| `work-unit` | Recorded and a Live Lab deploy requested (or a duplicate); the issue is closed |
| `work-unit-rejected` | The card doesn't match its spec; the issue stays open |
| `work-unit-pending` | The card couldn't be saved yet; the next run retries it |

Things to know as a maintainer:

- Runs share one concurrency group so their commits can't compete, with `queue: max` so
  waiting runs aren't cancelled. The queue holds 100 runs. Anything a cancelled or failed run
  missed is recorded by the next scheduled run, which processes every open result issue
  without a `work-unit` or `work-unit-rejected` label. Run the workflow by hand to catch up
  sooner.
- A card is announced, and its issue labelled and closed, only after it's on `main` and a
  Pages deploy was requested. If a run fails after the push (the deploy request or the reply),
  the issue stays open and unlabelled, and the next run finds the card already stored,
  requests the deploy again and settles the issue.
- The issue text is untrusted. The scripts only parse it as JSON and validate it. Nothing
  from the issue reaches a shell command, and replies never repeat strings from the card.
- The workflow needs to push to `main`. If you add branch protection, allow GitHub Actions
  to push. Until then, valid cards wait with `work-unit-pending` and the run fails, so you
  get GitHub's failed-workflow notice.
- Anyone can open an issue, so the stored results are claims. The page only calls a unit
  replicated when different volunteers agree, comparing runs only on the same slice dataset.
- Recovery cards (`recovery/v0`) are different: `tools/ingest_result.mjs` reruns the unit on
  the published windows (`data/recovery/`) and records the card only if its counts match. The
  Recovery tab adds up the recorded units, counting each unit number once.
