# EI@Home Live Lab

A static, zero-install web app: pick a signal (or press **Random sky sample & analyze** for a
random slice of real data, shown on the sky map where the telescope pointed), train a small GRU
on it in the browser, look at its predictions and hidden states, measure how language-like its radio tokens are, and run
reproducible work units. The Recovery tab measures which weak signals two standard detectors
find in real data, at a fixed false-alarm rate. Above both, the Observatory viewport shows each
signal's place on the sky and zooms in on it while the computation runs. The Contributions tab
shows where every contribution stands, stage by stage, with what decided each stage. Everything
computes locally; there is no server.

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
| `js/anchor.js` | The merge anchor `token-anchor/v1` (issue #20): a pinned peak tokenizer (33 tokens, nothing fitted) and the token transformer's starting weights θ₀, rebuilt from seeded uniform draws and checked against a pinned SHA-256 |
| `js/merge.js` | The merge work units: `trackb-merge/v1` (seeds on all the text, shards from θ₀ and shards from a warm start, each merged five ways) and the earlier `trackb-merge/v0` (the seeds alone). Task vectors, DARE (with the int8 payload's size), TIES (its own mean, or the mean over all workers), sign agreement and the other delta summaries, and each spec's card and its checks |
| `js/trackb-splits.js` | Track B's training, validation and test frames for each kind of family, split before anything is fitted |
| `js/trackb.worker.js`, `js/trackb.js` | Track B: splits each family into training, validation and test segments, fits the k-means tokenizer and the shuffled and Markov-1 surrogates on the training split only, and measures held-out cross-entropy, conditional entropy, mutual information, Zipf and compression without crossing a segment boundary |
| `js/signals.js` | Signal families (four real-data ones: Voyager 1, off-source, SETI targets and blind windows; a reconstruction of the 1977 Wow! signal from its printout readings, 6EQUJ5; and the synthetic controls) and the Rosetta text; snippet normalization. A real family says how its windows were cut (on-target hit, off-source or random blind window) and which real family is its control |
| `js/sample.js` | The random sky sample: picks a real slice that isn't the one already shown, half the time a signal (Voyager 1 or a SETI target's hit) and half the time a control (blind window or off-source, a quarter each), names its sky-map target, and writes its card (source, telescope, MJD, centre frequency, drift and SNR, and how the window was cut) and captions ("Tau Ceti · 3285.618 MHz · Parkes"). `main.js` then selects its family, flies the map to it, shows the slice's spectrogram in the map's card and in Section 1, and runs Track A (with the sample in the waterfall), Track B's tokens and, once Track A is done, the token model. Both tracks score the sample itself, in a highlighted "This sample" row under its family's averages and a callout: Track A its prediction gains and residual error, Track B the cross-entropy of its tokens under the n-gram model and the token model |
| `js/slices.js` | Loads real-data micro-slices (`data/slices/`) or user-dropped files, verifies their SHA-256 and computes the dataset identity |
| `js/splits.js` | Groups a real family's slices into observing sessions and keeps training and evaluation on different sessions |
| `js/workunits.js` | Versioned work-unit specs (`track-a/v0`; the recovery and merge specs live in `js/recovery.js` and `js/merge.js`). A `track-a/v0` unit reads only the slices of its spec's families (`specSlices`), so slices of later families leave its inputs and dataset id alone |
| `js/contributions.js` | What became of the result cards computed in this browser (computed here, saved to a file, opened on GitHub, accepted into the community results), and the rule for keeping them: a card is dropped to make room only if it's accepted or saved |
| `js/lifecycle.js` | The Contributions tab's stages and criteria, with no DOM: each browser card (computed, submitted, validated, replicated or recomputed, counted) against the community results, each checkpoint-pilot contribution (submitted, validated, replicated, merged) and each merge from `data/pilot_status.json`, and what would help next |
| `js/format.js` | Formatting shared by the panels: numbers that never read `-0.00`, and a family's possessive (`SETI targets’`, `Voyager 1’s`) |
| `js/skymap.js` | The Observatory viewport: a sky map on a canvas (stereographic projection, so stars stay round at any zoom). Slews along great circles to each signal's pointing, zooms from 60° to the 2° around the telescope beam with a computation's progress (a random sample instead gets a flight: out to a 35° view of its part of the sky with the constellations showing, then into the beam, which stays framed while the tracks run; the reticle names the target and frequency, "Tau Ceti · 3.29 GHz"), and draws its live effects: wavefronts for Track A and B, and the 1,024-window grid for recovery units. Observatory mode fills the window and can keep running work units. A running job stays on the target whose data it uses, and the HUD names that data. The coordinate math has no DOM dependencies, so tests run it in Node |
| `data/sky/` | The sky it draws (`tools/build_sky.py`): the 1,500 brightest Hipparcos stars with B−V colours, every Hipparcos star near the observed targets, constellation lines, names and IAU boundaries, and reference targets (among them the Wow! signal's two possible fields), from d3-celestial (BSD-3-Clause, notice in `LICENSE.d3-celestial.txt`) |
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
`tests/sample.test.mjs` checks the random sky sample: it never repeats the slice shown, can
reach every published family, draws signals and controls half the time each (blind windows and
off-source a quarter each), prefers slices with a pointing, and writes the card from the
manifest and Track A's callout, which says "held out in the test split" only when the family has a holdout (Track A and Track B name splits the same way); and that the slice pool stays within its limits, with every slice located and credited.
`tests/skymap.test.mjs` checks the viewport's math and data: coordinates, galactic conversion,
the projection and its inverse, slews and flights, the reticle's label, sidereal time, constellations from the IAU boundaries,
and that the manifest records where every slice pointed.
`tests/recovery.test.mjs` checks the recovery unit: the spec pins the published windows and
thresholds, every class is injected at its strength inside the window, the dealer's key
rebuilds every window, the de-Doppler search equals a brute-force search, units reproduce,
and tampered cards are caught by their form or by recomputation.
`tests/merge.test.mjs` checks the merge anchor and unit: θ₀ rebuilds to its pin in the
anchor's layout, the pinned tokenizer reads every Rosetta symbol whatever the noise, DARE keeps
the expected delta and its payload rebuilds within half a quantization step, TIES elects signs
and averages only agreeing entries on hand-checked vectors, and merge cards fit an issue link
and are rejected when they don't match their spec: an input besides the pinned corpus and
anchor, or DARE payload sizes, a kept share or a unigram baseline that differ from what the
unit's seeds and corpus give. For v1 it also checks that the seeds set is v0's unit, that the
shards are the training text in order without overlap, that TIES's mean over all workers keeps
the expected merge where the workers agree in sign and the trim removes no survivor (and not
when they disagree, or when trimming after the drop removes survivors), and that a
malformed value anywhere in a card is rejected, never thrown on.
`tests/community.test.mjs` checks card validation, replication summaries and the ingestion
script, and `tests/ingest_issues.test.mjs` runs the ingestion driver against a local git
remote and a fake `gh`. `tests/lifecycle.test.mjs` checks the Contributions tab's stages: a card
from computed to submitted as the browser's marks say, recorded cards single, replicated or
disagreeing (naming the metrics apart) as their unit's runs are, recovery cards recomputed and
combined, the order cards are listed in, each pilot contribution's gate, replication and merge
with their numbers (from `tests/fixtures/pilot_status.json`, whose shape a Python test checks
against the export), and what would help next. `tests/format.test.mjs` checks that a value that
rounds to zero never reads `-0.00`, plural possessives, and the split names Track A and Track B
share. `tests/browser.test.mjs` serves this folder and drives the Live Lab in
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
- the merge anchor's θ₀ rebuilds to its pin in Chromium and fits the token transformer exactly;
  a merge unit trains its three sets of workers (seeds, shards, shards from a warm start),
  merges each, gives the same card when run again, keeps it in the browser, and its card passes
  ingestion's checks; and with pinned tokens Track B's transformer trains from θ₀ and finds
  Rosetta's structure;
- a recovery unit run on the page gives a card that ingestion's checks accept, recomputation
  included, and community recovery units are combined once per unit, apart from the replicated
  units;
- the viewport slews to a signal's pointing, and clicking a beacon selects its signal;
- one click on **Random sky sample & analyze** picks a published slice, selects its family,
  flies the map through a 35° view of its part of the sky into the beam at its pointing (the
  reticle naming the target and frequency), shows the slice's spectrogram and labels in the
  target's card and in Section 1, captions the waterfall with its target, frequency and
  telescope, starts Track A at once and Track B's tokens alongside, trains the token model when
  Track A is done, scores the sample itself in both tracks next to its family's averages, and
  the next click picks another slice, with scores of its own;
- an off-source sample (one session, so nothing held out) is compared with what the model
  trained on, and says so, and the last run's waterfall keeps its sample's name after the
  loaded files are replaced;
- the token model a sample starts waits for Track B's analysis of that sample, even when it
  arrives after Track A, rather than taking the previous sample's;
- a recovery unit zooms the viewport to the telescope beam and colours each window by its outcome;
- observatory mode fills the window, keeps running units, and stops when it's closed; auto-run
  ticked while the recovery windows load starts once they're ready, and clicking another beacon
  mid-unit moves the view, not the unit;
- the review lists the cards computed here by status (accepted ones are matched against the
  community results), submits a card and marks it, reaches the 51st card with **Show more**,
  and opens when observatory mode is left with cards to submit; a browser full of cards that
  exist nowhere else pauses auto-run instead of dropping one, an unconfirmed download frees
  nothing, and a confirmed one lets the oldest saved cards make room; the claude.ai preview
  offers no download;
- Track B describes a real family's sessions as loaded (Voyager 1's four scans span 2016 and
  2020, two sessions, so its held-out scans share a session with training), and explains the
  control row from the run's own numbers.
- the control note writes a plural family's possessive with the apostrophe alone ("SETI
  targets’", "Blind windows’"), and no gain reads `-0.00`;
- the Contributions tab lists the next steps (missing shard coverage first), each pilot
  contribution's stages with its gate, replication and weight (unreplicated first), each merge
  with its shard weights and a rollback, the recorded units in replication order with what
  decides each, and this browser's cards by what they need, treating maintainers' text as text;
  with the published status it shows the open round and its four uncovered shards, and a status
  it can't read is reported, not shown.

CI runs both on every pull request.

Build table cells with `textContent` (see `makeRow` in `js/main.js`). Never put text from a
manifest, a file or browser storage into `innerHTML`.

## Common changes

- **Add a signal family:** add an entry to `FAMILIES` and a generator to `GENERATORS` in
  `js/signals.js`. It appears in the switcher, the gains table, the hidden-state map and
  Track B automatically. It doesn't join `track-a/v0` work units: they keep the families of
  their spec (`SPEC_FAMILIES` in `js/workunits.js`), so their cards stay comparable. To place a
  family on the sky map, give a reference target in `tools/build_sky.py` its `family`, as the
  Wow! signal's two fields have.
- **Add a Track B metric:** write it in `js/trackb.js` so that it takes an array of token
  segments and never looks across their boundaries, compute it for every stream in
  `analyzeSplits()`, and plot it in `renderTrackB()` in `js/main.js`. Anything fitted (like a
  codebook or a surrogate) must use the training split only.
- **Rebuild the sky data:** `python tools/build_sky.py` (it downloads d3-celestial from npm and
  checks it against its published hash); `--check` compares a rebuild with the committed files.
- **Change the recovery windows or thresholds:** they're part of `recovery/v0`. Write them with
  `python -m tracks.recovery.browser --write`, then pin them in a new spec in `js/recovery.js`.
- **Add a work-unit spec:** add a new versioned config in `js/workunits.js`. Never change an
  existing spec, because published result cards refer to it. Make a `v1` instead.
- **Publish real data:** see `data/slices/README.md`. More SETI targets or blind windows go
  in `tools/slice_pool.json`; CI cuts them. A new real family also needs an entry with
  `real: true`, its `window` and its `control` in `FAMILIES`, a generator that calls
  `sliceWindow()`, and its name in `FAMILIES` in `tracks/llm/data.py`.

## Your results in this browser

Units computed here are kept in browser storage (up to 200 Track A work units, 300 recovery
units and 200 merge units) until you delete them. The viewport's **Review and submit** button,
also offered when you leave observatory mode, lists them with what became of each (for merge
units, the button is in their panel):

| Status | Meaning |
|---|---|
| Computed here | Only this browser has the card |
| Saved to a file | You downloaded the cards from the review and confirmed you have the file |
| Opened on GitHub | You opened its submission issue: a draft until you post it, after which it's checked and recorded |
| Accepted | Its exact card is in `data/community_results.json` |

The review lists the cards not yet accepted (or all of them), 50 at a time with **Show more**, so
every kept card can be submitted. **Check for accepted cards** reloads the community results;
the page also does this on its own when you come back to the tab with cards opened on GitHub.

To make room for a new card, the oldest cards that are accepted, or saved and confirmed, are
dropped, whatever their status shows (a saved card stays safe to drop after its issue is
opened). A download only counts once you confirm you have the file, because the page can't see
whether the browser saved it; in the claude.ai preview, which blocks downloads, the review
offers none. A card that exists nowhere else is never dropped: when every kept card is like
that, auto-run pauses and the review opens, so you can submit or save them. The HUD counts units
computed this session, cards awaiting submission and cards accepted. Nothing about this leaves
the browser.

## Contributions

The Contributions tab shows every contribution's stages, each **done**, **now** (where it
stands: the next step is someone's), **later**, **failed** or **not applicable**, with the
criterion that decides it and the numbers it was decided on. It reads only the records:

- **Result cards in this browser** (`track-a/v0`, `trackb-merge/v0` and `v1`, `recovery/v0`):
  computed here; submitted (an opened issue is a draft until posted); validated, once
  `data/community_results.json` holds the exact card, against the checks
  `tools/ingest_result.mjs` applies to its kind; replicated, when a different volunteer's run of
  the unit on the same data agrees within 0.02 on every metric (the tab lists the metrics that
  don't), or for a recovery unit recomputed on the published windows; and counted. The cards
  carry measurements, not weights, so nothing merges them into a model.
- **Recorded result cards**, by unit: not replicated yet, disagreeing, replicated, in that order.
- **The checkpoint pilot** (`tracks/llm/PILOT.md`), from `data/pilot_status.json`: each
  contribution's held-out validation gate (its score, the current model's, and the margin),
  its replication (the coordinator's run: the bits apart and the delta cosine, against 0.02 and
  0.9), and its merge (its weight: its shard's share of the round's training targets, split
  among the shard's contributions), or why it wasn't merged (not replicated, excluded, rolled
  back, superseded). A contribution excluded after a live round merged it shows both: merged,
  and excluded (by whom and why) with its weight in the current model until that round is
  rolled back. Each merge shows its gate, its shard weights, each volunteer's total, and any
  contribution excluded since.
  The open round's shards are listed least covered first.

**What would help next** puts the pilot's shards without a contribution first, then the
volunteers a merge still needs, units with one volunteer's run, units whose runs disagree, and
cards in this browser not yet submitted. Submissions refused before they were recorded aren't in
either record; the tab links to the issues labelled `work-unit-rejected` and
`checkpoint-rejected`, whose replies say why.

`data/pilot_status.json` is made from the pilot's ledger by `python -m tracks.llm.pilot export`,
which every command that changes the ledger also runs; CI checks it matches the ledger
(`export --check`).

## Deploying

`.github/workflows/pages.yml` deploys this folder to GitHub Pages on every push to `main`
that touches `web/`. Enable it once in the repository's **Settings → Pages → Build and
deployment → Source: GitHub Actions**.

## Community results

Checkpoint contributions to the pilot go through the same driver with `--kind checkpoint`
(`.github/workflows/ingest-checkpoints.yml`): `python -m tracks.llm.pilot ingest` checks each,
and an accepted one is committed with the rewritten `data/pilot_status.json` and announced after
a Pages deploy is requested, as result cards are.

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
