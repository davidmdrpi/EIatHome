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
| `js/signals.js` | Signal families and the Rosetta text; snippet normalization |
| `js/slices.js` | Loads real-data micro-slices (`data/slices/`) or user-dropped files |
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

## Common changes

- **Add a signal family:** add an entry to `FAMILIES` and a generator to `GENERATORS` in
  `js/signals.js`. It appears in the switcher, the gains table, the hidden-state map and
  Track B automatically.
- **Add a Track B metric:** write it in `js/trackb.js`, compute it for every stream in
  `js/trackb.worker.js`, and plot it in `renderTrackB()` in `js/main.js`.
- **Add a work-unit spec:** add a new versioned config in `js/workunits.js`. Never change an
  existing spec, because published result cards refer to it. Make a `v1` instead.
- **Publish real data:** see `data/slices/README.md`.

## Deploying

`.github/workflows/pages.yml` deploys this folder to GitHub Pages on every push to `main`
that touches `web/`. Enable it once in the repository's **Settings → Pages → Build and
deployment → Source: GitHub Actions**.
