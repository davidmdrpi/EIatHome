# EI@Home Live Lab

A static, zero-install web app: pick a signal, train a small GRU on it in the browser, look at
its predictions and hidden states, measure how language-like its radio tokens are, and run
reproducible work units. Everything computes locally; there is no server.

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
| `js/engine.worker.js` | Track A engine (module worker): builds snippets, trains the GRU with TensorFlow.js, computes baselines, gains, the hidden-state PCA and the drift probe |
| `js/trackb.worker.js`, `js/trackb.js` | Track B: k-means tokenizer, conditional entropy, mutual information, Zipf, compression, and the shuffled and Markov-1 surrogates |
| `js/signals.js` | Signal families (two real-data ones, Voyager 1 and off-source, and the synthetic controls) and the Rosetta text; snippet normalization |
| `js/slices.js` | Loads real-data micro-slices (`data/slices/`) or user-dropped files, verifies their SHA-256 and computes the dataset identity |
| `js/splits.js` | Groups a real family's slices into observing sessions and keeps training and evaluation on different sessions |
| `js/workunits.js` | Versioned work-unit specs |
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
`tests/community.test.mjs` checks card validation, replication summaries and the ingestion
script, and `tests/ingest_issues.test.mjs` runs the ingestion driver against a local git
remote and a fake `gh`. `tests/browser.test.mjs` serves this folder and drives the Live Lab in
headless Chromium. It checks that:

- slice metadata is always shown as text;
- the published slices load, and a run on one session is labelled as reconstruction;
- reordering a manifest and renaming its files gives identical work-unit results;
- work units give identical metrics whatever custom text is set;
- token-model training can be stopped, and the controls come back.

CI runs both on every pull request.

Build table cells with `textContent` (see `makeRow` in `js/main.js`). Never put text from a
manifest, a file or browser storage into `innerHTML`.

## Common changes

- **Add a signal family:** add an entry to `FAMILIES` and a generator to `GENERATORS` in
  `js/signals.js`. It appears in the switcher, the gains table, the hidden-state map and
  Track B automatically.
- **Add a Track B metric:** write it in `js/trackb.js`, compute it for every stream in
  `js/trackb.worker.js`, and plot it in `renderTrackB()` in `js/main.js`.
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
| `work-unit` | Recorded (or a duplicate); the issue is closed |
| `work-unit-rejected` | The card doesn't match its spec; the issue stays open |
| `work-unit-pending` | The card couldn't be saved yet; the next run retries it |

Things to know as a maintainer:

- Runs share one concurrency group so their commits can't compete, with `queue: max` so
  waiting runs aren't cancelled. The queue holds 100 runs. Anything a cancelled or failed run
  missed is recorded by the next scheduled run, which processes every open result issue
  without a `work-unit` or `work-unit-rejected` label. Run the workflow by hand to catch up
  sooner.
- The issue text is untrusted. The scripts only parse it as JSON and validate it. Nothing
  from the issue reaches a shell command, and replies never repeat strings from the card.
- The workflow needs to push to `main`. If you add branch protection, allow GitHub Actions
  to push. Until then, valid cards wait with `work-unit-pending` and the run fails, so you
  get GitHub's failed-workflow notice.
- Anyone can open an issue, so the stored results are claims. The page only calls a unit
  replicated when different volunteers agree, comparing runs only on the same slice dataset.
