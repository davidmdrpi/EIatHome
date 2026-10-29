import { isRecoveryCard, issueLink, submissionText, summarize, validateCard } from './community.js';
import { STATUSES, acceptedCards, cardKey, cardStatus, keepCard, pruneMarks, safeToDrop, tally } from './contributions.js';
import { architectureLabel, tokenArchitectureLabel } from './models.js';
import { fmt, heatmap, lineChart, scatterPanel } from './plots.js';
import { DECOYS, KINDS, RECOVERY_DATA, RECOVERY_SPEC, SIGNALS, STRENGTHS, combine, wilson } from './recovery.js';
import { deriveSeed, makeRng } from './rng.js';
import { forWorkers, loadPublishedSlices, loadSliceFiles } from './slices.js';
import { SCAN_GAP_DAYS, groups, mjdDate, sessions } from './splits.js';
import { C, FAMILIES, ROSETTA_TEXT, T, driftToHzPerS, familyById, makeSnippet, normalise, shuffleFrames } from './signals.js';
import { GREEN_BANK, OUTCOMES, SkyMap, constellationAt, horizonToEquatorial, loadSky, siderealHours } from './skymap.js';
import { MAX_UNIT, SPEC, workUnitConfig } from './workunits.js';

const REPO = 'https://github.com/davidmdrpi/EIatHome';
const BACKEND_NAMES = { webgpu: 'WebGPU', webgl: 'WebGL', wasm: 'WebAssembly', cpu: 'plain JS' };
const $ = (id) => document.getElementById(id);

// In the private claude.ai preview the viewer owns the theme and file downloads are blocked.
const PREVIEW = document.querySelector('meta[name="eiathome-host"]')?.content === 'claude-artifact';

// Tables are built with textContent, so text from slice manifests or browser storage is
// always shown literally and never parsed as HTML.
function makeRow(cells, tag = 'td') {
  const tr = document.createElement('tr');
  for (const value of cells) {
    const cell = document.createElement(tag);
    if (tag === 'th') cell.scope = 'col';
    cell.textContent = String(value ?? '');
    tr.append(cell);
  }
  return tr;
}

function fillTable(table, header, rows) {
  const head = document.createElement('thead');
  head.append(makeRow(header, 'th'));
  const body = document.createElement('tbody');
  body.append(...rows.map((cells) => makeRow(cells)));
  table.replaceChildren(head, body);
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

// Browser storage is a convenience only: everything works without it.
const store = {
  get(key, fallback) {
    try {
      const value = localStorage.getItem(`eiathome.${key}`);
      return value === null ? fallback : JSON.parse(value);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(`eiathome.${key}`, JSON.stringify(value));
    } catch {
      // storage unavailable: ignore
    }
  },
};

const state = {
  family: 'tone',
  slices: [],
  sliceSource: null, // 'site' | 'files' | null
  running: null, // 'lab' | 'unit' | null
  losses: [],
  lastEpoch: null,
  lastProbe: null,
  trainedFamily: null,
  trackb: null,
  card: null,
  issue: null, // the card's GitHub issue link: { url, prefilled }
  stopLoop: false,
  text: store.get('rosettaText', ''),
  results: store.get('results', []),
  theme: store.get('theme', 'auto'),
};

// ---------- workers ----------

let engine;
let tokens;
let analyzeId = 0;

function startEngine() {
  engine?.terminate();
  setChip('Starting…');
  engine = new Worker(new URL('./engine.worker.js', import.meta.url), { type: 'module' });
  engine.onmessage = ({ data }) => onEngineMessage(data);
  engine.onerror = (event) => {
    setChip('Engine failed to start');
    labStatus(`The training engine couldn't start (${event.message || 'unknown error'}).`, true);
  };
  engine.postMessage({ type: 'init', backend: $('backend').value });
  if (state.slices.length) engine.postMessage({ type: 'slices', slices: forWorkers(state.slices) });
  if (state.text) engine.postMessage({ type: 'text', text: state.text });
}

function startTokens() {
  tokens = new Worker(new URL('./trackb.worker.js', import.meta.url), { type: 'module' });
  tokens.onmessage = ({ data }) => onTrackB(data);
}

function broadcast(msg) {
  engine.postMessage(msg);
  tokens.postMessage(msg);
}

function setChip(text) {
  $('backend-status').textContent = text;
}

function labStatus(text, error = false) {
  $('lab-status').textContent = text;
  $('lab-status').classList.toggle('error', error);
}

function unitStatus(text, error = false) {
  $('wu-status').textContent = text;
  $('wu-status').classList.toggle('error', error);
}

function onEngineMessage(msg) {
  switch (msg.type) {
    case 'ready':
      setChip(`Computing on ${BACKEND_NAMES[msg.backend] ?? msg.backend} · TF.js ${msg.version}`);
      break;
    case 'epoch':
      state.losses.push([msg.epoch, msg.loss]);
      state.lastEpoch = msg;
      labStatus(`${architectureLabel(msg.arch)}, ${msg.parameters.toLocaleString()} parameters · epoch ${msg.epoch} of ${msg.epochs} · ${Math.round(msg.ms)} ms per epoch on ${BACKEND_NAMES[msg.backend] ?? msg.backend}`);
      viewports.lab?.telemetry(`Track A · ${architectureLabel(msg.arch)} · epoch ${msg.epoch} of ${msg.epochs} · loss ${fmt(msg.loss, 4)} · ${Math.round(msg.ms)} ms per epoch`, msg.epoch / msg.epochs);
      renderLab();
      break;
    case 'trained':
      state.lastProbe = msg.probe;
      state.lastUntrainedProbe = msg.untrainedProbe;
      setRunning(null);
      if (msg.stopped) labStatus('Stopped.');
      else if (msg.reconstruction)
        labStatus(
          `Done: ${architectureLabel(msg.arch)}, ${msg.parameters.toLocaleString()} parameters. ${msg.reconstruction} So the ${familyById[state.trainedFamily].label} score measures how well the model reconstructs data it trained on, not how well it generalizes.`,
        );
      else labStatus(`Done: ${architectureLabel(msg.arch)}, ${msg.parameters.toLocaleString()} parameters. Compare it with the baselines, then try another signal or model.`);
      viewports.lab?.finishJob(msg.stopped ? 'Track A · stopped' : `Track A · done: ${architectureLabel(msg.arch)}, ${msg.parameters.toLocaleString()} parameters. See its prediction gains below.`);
      renderProbe();
      break;
    case 'wu-progress':
      $('wu-progress').value = msg.epoch / msg.epochs;
      unitStatus(`Unit ${msg.unit}: epoch ${msg.epoch} of ${msg.epochs}, training loss ${fmt(msg.loss, 3)}`);
      viewports.lab?.telemetry(`Work unit ${msg.unit} · ${SPEC} · epoch ${msg.epoch} of ${msg.epochs} · loss ${fmt(msg.loss, 3)}`, msg.epoch / msg.epochs);
      break;
    case 'wu-result':
      onWorkUnitResult(msg.card);
      break;
    case 'wu-stopped':
      setRunning(null);
      unitStatus(`Unit ${msg.unit} stopped.`);
      viewports.lab?.finishJob(`Work unit ${msg.unit} · stopped`);
      break;
    case 'token-lm-progress':
      $('tb-lm-status').textContent = `Training on the ${STREAM_NAMES[msg.stream]} stream (${tokenArchitectureLabel(msg.arch)}): epoch ${msg.epoch} of ${msg.epochs}…`;
      viewports.lab?.telemetry(`Track B · token ${tokenArchitectureLabel(msg.arch)} on the ${STREAM_NAMES[msg.stream]} stream · epoch ${msg.epoch} of ${msg.epochs}`, msg.epoch / msg.epochs);
      break;
    case 'token-lm-result':
      setRunning(null);
      state.tokenLm = msg.stopped ? null : { ...msg, analysis: state.tokenLmFor };
      $('tb-lm-status').textContent = msg.stopped
        ? 'Stopped.'
        : `Done: ${tokenArchitectureLabel(msg.arch)}, ${msg.parameters.toLocaleString()} parameters. Lower is better; compare the real stream with its surrogates.`;
      viewports.lab?.finishJob(msg.stopped ? 'Track B · token model stopped' : `Track B · token ${tokenArchitectureLabel(msg.arch)} trained. Its held-out scores are in the Track B panel.`);
      renderTokenModel();
      break;
    case 'error':
      viewports.lab?.finishJob(`Stopped: ${msg.message}`);
      if (state.running === 'unit') unitStatus(msg.message, true);
      else if (state.running === 'tokens') {
        $('tb-lm-status').textContent = msg.message;
        $('tb-lm-status').classList.add('error');
      } else labStatus(msg.message, true);
      setRunning(null);
      break;
  }
}

function setRunning(mode) {
  state.running = mode;
  const busy = mode !== null;
  $('train').disabled = busy;
  $('wu-run').disabled = busy;
  $('tb-lm').disabled = busy;
  $('stop').disabled = mode !== 'lab';
  $('wu-stop').disabled = mode !== 'unit';
  $('tb-lm-stop').disabled = mode !== 'tokens';
  $('backend').disabled = busy;
}

// ---------- Live Lab ----------

// Real families need loaded slices of their own.
const familyAvailable = (id) => !familyById[id].real || state.slices.some((s) => s.family === id);

function renderFamilies() {
  const box = $('families');
  box.replaceChildren(
    ...FAMILIES.map((f) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('role', 'radio');
      b.textContent = f.label;
      b.dataset.family = f.id;
      b.setAttribute('aria-checked', String(f.id === state.family));
      if (!familyAvailable(f.id)) {
        b.disabled = true;
        b.title = `No ${f.label} slices loaded yet. See the Data tab.`;
      }
      b.onclick = () => selectFamily(f.id);
      return b;
    }),
  );
  $('family-about').textContent = familyById[state.family].about;
}

// fromMap: the viewport already selected the clicked pointing, so don't slew elsewhere.
function selectFamily(id, { fromMap = false } = {}) {
  state.family = id;
  renderFamilies();
  if (!fromMap) viewports.lab?.select(targetForFamily(id));
  if (!state.running) renderPreview();
  analyzeTokens();
}

// Before training: show a sample of the chosen family.
function renderPreview() {
  const seed = Number($('seed').value) || 0;
  const rng = makeRng(deriveSeed(seed, `preview-${state.family}`));
  let x;
  try {
    x = normalise(makeSnippet(state.family, rng, { slices: forWorkers(state.slices), region: 'eval', text: state.text || undefined }).raw);
  } catch {
    return;
  }
  if ($('shuffle').checked) x = shuffleFrames(x, rng.permutation(T));
  const frames = x.slice(2 * C);
  heatmap($('wf-actual'), frames, T - 2, C, { range: [-1, Math.max(3, ...frames)], label: 'power (asinh)' });
  for (const id of ['wf-pred', 'wf-resid']) {
    const canvas = $(id);
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.clientWidth;
    canvas.height = 190;
    canvas.style.height = '190px';
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--ink-muted');
    ctx.font = '12px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Train to see predictions', canvas.width / 2, 95);
    canvas._hover = null;
  }
}

function renderLab() {
  const e = state.lastEpoch;
  if (!e) return;
  const { actual, predicted, residual, rows } = e.example;
  const top = Math.max(3, ...actual);
  heatmap($('wf-actual'), actual, rows, C, { range: [-1, top], label: 'power (asinh)' });
  heatmap($('wf-pred'), predicted, rows, C, { range: [-1, top], label: 'predicted' });
  const r = Math.min(3, Math.max(0.5, ...residual.map(Math.abs)));
  heatmap($('wf-resid'), residual, rows, C, { kind: 'diverging', range: [-r, r], label: 'residual' });

  lineChart($('loss'), [{ name: 'training loss', color: '--series-1', points: state.losses }], {
    xLabel: 'epoch',
    yLabel: 'training loss (mean squared error)',
    markers: state.losses.length < 25,
    yFormat: (v) => fmt(v, 3),
  });

  const body = $('gains').tBodies[0];
  body.replaceChildren(
    ...FAMILIES.filter((f) => e.gains[f.id]).map((f) => {
      const g = e.gains[f.id];
      let name = f.label;
      if (f.id === state.trainedFamily) name += e.reconstruction ? ' (trained on; not held out)' : ' (trained on)';
      const tr = makeRow([name, fmt(g.persistence), fmt(g.linear), fmt(g.gru), fmt(e.shuffledGains[f.id])]);
      if (f.id === state.trainedFamily) tr.className = 'current';
      return tr;
    }),
  );
  $('gains-note').hidden = !e.reconstruction;
  $('gains-model').textContent = architectureLabel(e.arch);
  renderMap();
}

function renderMap() {
  const e = state.lastEpoch;
  if (!e) return;
  const { points, families } = e.map;
  const xs = [];
  const ys = [];
  for (let i = 0; i < families.length; i++) {
    xs.push(points[2 * i]);
    ys.push(points[2 * i + 1]);
  }
  const span = (v) => {
    const lo = Math.min(...v);
    const hi = Math.max(...v);
    const m = (hi - lo) * 0.06 || 1;
    return [lo - m, hi + m];
  };
  const bounds = [...span(xs), ...span(ys)];
  const box = $('map');
  const present = FAMILIES.filter((f) => families.includes(f.id));
  if (box.dataset.families !== present.map((f) => f.id).join()) {
    box.dataset.families = present.map((f) => f.id).join();
    box.replaceChildren(
      ...present.map((f) => {
        const fig = document.createElement('figure');
        fig.innerHTML = `<figcaption>${f.label}</figcaption><canvas data-height="110" aria-label="Hidden states: ${f.label} highlighted"></canvas>`;
        return fig;
      }),
    );
  }
  [...box.querySelectorAll('canvas')].forEach((canvas, j) => {
    const f = present[j];
    scatterPanel(canvas, points, families, f.id, bounds, (i) => `${f.label}<br>PC1 ${fmt(points[2 * i])}, PC2 ${fmt(points[2 * i + 1])}`);
  });
}

function renderProbe() {
  const box = $('probe');
  const p = state.lastProbe;
  const u = state.lastUntrainedProbe;
  if (!p) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  const parts = [
    element('h3', 'Drift probe'),
    element(
      'p',
      `A linear model reads drift rate from the model’s hidden state, trained on synthetic tones only. On held-out tones: R² = ${fmt(p.r2)}, mean error ${fmt(p.maeHzPerS, 3)} Hz/s.`,
      'small',
    ),
  ];
  if (u) {
    parts.push(
      element('p', `Control: the same model before training (same architecture and seed) gives R² = ${fmt(u.r2)}, mean error ${fmt(u.maeHzPerS, 3)} Hz/s. The difference is what training added.`, 'small'),
    );
  }
  if (p.voyager) {
    const table = document.createElement('table');
    fillTable(
      table,
      ['Voyager', 'Probe readout (Hz/s)', 'Untrained GRU (Hz/s)', 'Measured (Hz/s)'],
      p.voyager.map((v, i) => [v.group, `${fmt(v.readout, 3)} ± ${fmt(v.sd, 3)}`, fmt(u?.voyager?.[i]?.readout, 3), fmt(v.measured, 3)]),
    );
    const wrap = element('div', undefined, 'table-wrap');
    wrap.append(table);
    parts.push(wrap);
  }
  box.replaceChildren(...parts);
}

function startTraining() {
  const config = {
    target: state.family,
    arch: $('arch').value,
    hidden: Number($('hidden').value),
    epochs: Math.min(60, Math.max(1, Number($('epochs').value) || 15)),
    nTrain: Number($('ntrain').value),
    batch: 64,
    lr: 3e-3,
    seed: Number($('seed').value) || 0,
    mixNoise: $('mix-noise').checked,
    shuffle: $('shuffle').checked,
  };
  state.losses = [];
  state.lastEpoch = null;
  state.lastProbe = null;
  state.lastUntrainedProbe = null;
  state.trainedFamily = config.target;
  renderProbe();
  setRunning('lab');
  labStatus('Building snippets and baselines…');
  viewports.lab?.startJob('train', targetForFamily(config.target), `Track A · ${architectureLabel(config.arch)} learning ${familyById[config.target].label}: building snippets…`, dataSource(config.target));
  engine.postMessage({ type: 'train', config });
}

// ---------- Track B ----------

function analyzeTokens() {
  if (!familyAvailable(state.family)) return;
  const id = ++analyzeId;
  $('tb-status').textContent = 'Tokenizing…';
  if (!state.running) viewports.lab?.startJob('analysis', targetForFamily(state.family), `Track B · tokenizing ${familyById[state.family].label} into radio tokens…`, dataSource(state.family));
  $('tb-status').classList.remove('error');
  tokens.postMessage({
    type: 'analyze',
    id,
    family: state.family,
    k: Number($('k').value),
    nFrames: Number($('nframes').value),
    seed: Number($('seed').value) || 0,
    text: state.text || undefined,
  });
}

function onTrackB(msg) {
  if (msg.id !== analyzeId) return; // a newer request superseded this one
  if (msg.type === 'error') {
    $('tb-status').textContent = msg.message;
    $('tb-status').classList.add('error');
    // Never leave another family's analysis behind for the token model to train on.
    state.trackb = null;
    state.tokenLm = null;
    for (const id of ['tb-tiles', 'tb-entropy-table', 'tb-plugin-table']) $(id).replaceChildren();
    $('tb-notes').textContent = '';
    renderTokenModel();
    return;
  }
  state.trackb = msg;
  if (!state.running) viewports.lab?.finishJob(`Track B · ${msg.nTokens.toLocaleString()} tokens of ${familyById[msg.family].label}, ${msg.used} of ${msg.k} token types used`);
  if (state.tokenLm?.analysis !== msg) {
    state.tokenLm = null; // the token model belongs to the previous token stream
    if (state.running !== 'tokens') $('tb-lm-status').textContent = '';
  }
  $('tb-status').textContent =
    `${familyById[msg.family].label}: ${msg.nTokens.toLocaleString()} tokens, ${msg.used} of ${msg.k} token types used. ` +
    `Held out: ${msg.evaluation.description}; ${msg.sizes.test.targets.toLocaleString()} test tokens scored. ` +
    `First training tokens: ${msg.preview.map((t) => t.toString(36)).join(' ')}`;
  renderTrackB();
}

const STREAMS = [
  { id: 'real', name: 'real', color: '--series-1' },
  { id: 'shuffled', name: 'shuffled', color: '--series-2' },
  { id: 'markov', name: 'Markov-1', color: '--series-3' },
];

function legend(beforeCanvas, series) {
  let el = beforeCanvas.previousElementSibling;
  if (!el?.classList.contains('legend')) {
    el = document.createElement('div');
    el.className = 'legend';
    beforeCanvas.before(el);
  }
  el.innerHTML = series.map((s) => `<span><span class="key" style="background:var(${s.color})"></span>${s.name}</span>`).join('');
}

function renderTrackB() {
  const r = state.trackb;
  if (!r) return;
  $('tb-tiles').innerHTML = STREAMS.map(
    (s) => `<div class="tile"><div class="label">Compressed size, ${s.name}</div><div class="value">${fmt(r.compression[s.id])} bits/token</div></div>`,
  ).join('');

  const entropySeries = STREAMS.map((s) => ({
    name: s.name,
    color: s.color,
    points: r.crossEntropy[s.id].map((e) => [e.context, e.bits]),
  }));
  legend($('tb-entropy'), entropySeries);
  lineChart($('tb-entropy'), entropySeries, { xLabel: 'tokens of context', yLabel: 'bits per token (held out)', markers: true, yMin: 0 });
  const rows = STREAMS.map((s) => [s.name, ...r.crossEntropy[s.id].map((e) => fmt(e.bits)), r.sizes.test.targets.toLocaleString()]);
  if (r.control) rows.push([controlName(r), ...r.control.crossEntropy.map((e) => fmt(e.bits)), r.control.targets.toLocaleString()]);
  fillTable($('tb-entropy-table'), ['Context', ...r.crossEntropy.real.map((e) => e.context), 'Tokens scored'], rows);
  $('tb-control-note').textContent = r.control?.profile ? controlNote(r) : '';
  $('tb-control-note').hidden = !r.control?.profile;
  fillTable(
    $('tb-plugin-table'),
    ['Context', ...r.entropy.real.map((e) => e.n - 1)],
    STREAMS.map((s) => [s.name, ...r.entropy[s.id].map((e) => `${fmt(e.h)}${e.reliable ? '' : '*'}`)]),
  );

  const two = STREAMS.slice(0, 2);
  const miSeries = two.map((s) => ({ name: s.name, color: s.color, points: r.mi.lags.map((d, i) => [d, r.mi[s.id][i]]) }));
  legend($('tb-mi'), miSeries);
  lineChart($('tb-mi'), miSeries, { xLabel: 'distance d', yLabel: 'bits', yMin: 0, yFormat: (v) => fmt(v, 2) });

  for (const [canvasId, key] of [['tb-zipf1', 'unigram'], ['tb-zipf2', 'bigram']]) {
    const series = two.map((s) => ({ name: s.name, color: s.color, points: r.zipf[key][s.id].map((f, i) => [i + 1, f]) }));
    legend($(canvasId), series);
    // Draw "real" last so it stays visible where the curves coincide.
    lineChart($(canvasId), [...series].reverse(), { xLabel: 'rank', yLabel: 'frequency', logX: true, logY: true, yFormat: (v) => v.toPrecision(2) });
  }
  $('tb-notes').textContent = r.notes.join(' ');
}

const STREAM_NAMES = Object.fromEntries(STREAMS.map((s) => [s.id, s.name]));

// The other real family's frames, tokenized and scored with this family's models.
const controlName = (r) => `${familyById[r.control.family].label} (control)`;

// Why the control row sits where it does: how its frames fall on this family's codebook, and
// what context adds for it and for this family.
function controlNote(r) {
  const p = r.control.profile;
  const name = familyById[r.control.family].label;
  const own = familyById[r.family].label;
  const control = r.control.crossEntropy.map((e) => e.bits);
  const real = r.crossEntropy.real.map((e) => e.bits);
  const inTraining = p.trainCommonest ? `${own}’s commonest training token (${pct(p.trainShare)})` : `${pct(p.trainShare)} of ${own}’s training tokens`;
  const parts = [
    p.share >= 0.5
      ? `${pct(p.share)} of ${name}’s ${p.frames} frames land on one token type${p.flattest ? ', the codebook’s flattest frame (the one plain noise is nearest)' : ''}. It is also ${inTraining}.`
      : `${name}’s ${p.frames} frames use ${p.typesUsed} of the ${r.k} token types. The commonest takes ${pct(p.share)} of them and is ${inTraining}.`,
  ];
  if (control[0] < real[0]) {
    parts.push(`Frames concentrated on few token types are cheap to predict, so ${name} costs less than ${own} with no context (${fmt(control[0])} against ${fmt(real[0])} bits), with no structure behind it.`);
  }
  const gain = (bits) => bits[0] - Math.min(...bits.slice(1));
  const says = (bits) => {
    const best = Math.min(...bits.slice(1));
    return gain(bits) > 0.005 ? `improves by ${fmt(gain(bits))} bits at best (${fmt(bits[0])} → ${fmt(best)})` : `doesn’t improve (${fmt(bits[0])} bits with none, ${fmt(best)} at best)`;
  };
  const drift = r.family === 'voyager' && gain(real) > 0.3 ? ', as its carrier and sidebands drift steadily across the window from frame to frame' : '';
  parts.push(`Context is the test. With up to ${real.length - 1} tokens of it, ${name} ${says(control)}, ${gain(real) > 0.005 ? 'while' : 'and'} ${own} ${says(real)}${drift}.`);
  return parts.join(' ');
}

function startTokenModel() {
  if (state.running) return;
  if (!state.trackb) {
    $('tb-lm-status').textContent = 'Analyze tokens first.';
    return;
  }
  const r = state.trackb;
  state.tokenLmFor = r;
  state.tokenLm = null;
  renderTokenModel();
  $('tb-lm-status').classList.remove('error');
  $('tb-lm-status').textContent = 'Starting…';
  setRunning('tokens');
  viewports.lab?.startJob('tokens', targetForFamily(r.family), `Track B · training a token ${tokenArchitectureLabel($('tb-lm-arch').value)}…`, dataSource(r.family));
  engine.postMessage({
    type: 'token-lm',
    arch: $('tb-lm-arch').value,
    k: r.k,
    splits: r.streams,
    control: r.control?.segments ?? null,
    centroids: r.centroids,
    seed: Number($('seed').value) || 0,
  });
}

function renderTokenModel() {
  const m = state.tokenLm;
  const r = state.trackb;
  const show = Boolean(m && r && m.analysis === r);
  $('tb-lm-sample').hidden = !show || !m.sample;
  if (!show) {
    $('tb-lm-table').replaceChildren();
    return;
  }
  const tableRows = STREAMS.filter((s) => m.results[s.id]).map((s) => [
    s.name,
    fmt(m.results[s.id].bits),
    fmt(r.crossEntropy[s.id][3].bits),
    fmt(r.crossEntropy[s.id][0].bits),
    m.results[s.id].targets.toLocaleString(),
    `${m.results[s.id].bestEpoch} of ${m.epochs}`,
  ]);
  if (m.control && r.control) {
    tableRows.push([controlName(r), fmt(m.control.bits), fmt(r.control.crossEntropy[3].bits), fmt(r.control.crossEntropy[0].bits), m.control.targets.toLocaleString(), 'real model']);
  }
  fillTable($('tb-lm-table'), ['Stream (held out, bits per token)', `Token ${tokenArchitectureLabel(m.arch)}`, 'n-gram, 3 tokens of context', 'No context', 'Tokens scored', 'Best epoch'], tableRows);
  if (m.sample) {
    const { real, generated, rows } = m.sample;
    $('tb-lm-real-caption').textContent = `The real next ${rows} tokens`;
    $('tb-lm-generated-caption').textContent = `The model’s ${rows}-token continuation (sampled)`;
    const top = Math.max(3, ...real, ...generated);
    const options = { range: [-1, top], label: 'power (asinh)', rowLabel: 'token', firstRow: 1 };
    heatmap($('tb-lm-real'), real, rows, C, options);
    heatmap($('tb-lm-generated'), generated, rows, C, options);
  }
}

// ---------- work units ----------

function runUnit(unit) {
  state.stopLoop = false;
  $('wu-number').value = unit;
  $('wu-progress').value = 0;
  setRunning('unit');
  unitStatus(`Unit ${unit}: building snippets…`);
  viewports.lab?.startJob('unit', targetForFamily('voyager'), `Work unit ${unit} · ${SPEC}: trains on synthetic tones, tests on every signal…`, unitDataSource());
  engine.postMessage({ type: 'workunit', unit });
}

const randomUnit = () => Math.floor(Math.random() * (MAX_UNIT + 1));

function onWorkUnitResult(card) {
  setRunning(null);
  state.card = card;
  $('wu-progress').value = 1;
  unitStatus(`Unit ${card.unit} finished in ${card.seconds} s on ${BACKEND_NAMES[card.backend] ?? card.backend}.`);
  $('wu-card-panel').hidden = false;
  $('wu-card').textContent = JSON.stringify(card, null, 2);
  state.issue = issueLink(card, REPO);
  $('wu-submit').href = state.issue.url;
  $('wu-submit-text').hidden = true;
  const full = keep('lab', card);
  renderHistory();
  session.units += 1;
  viewports.lab?.finishJob(`Work unit ${card.unit} · done in ${card.seconds} s · GRU gain on tones ${fmt(card.metrics?.prediction_gain?.tone?.gru)}`);
  if (full) pauseForReview('lab');
  refreshContributions();
  // The loop is checked again when the timer fires: it may have been turned off meanwhile.
  if ($('wu-loop').checked && !state.stopLoop) setTimeout(() => $('wu-loop').checked && !state.stopLoop && !state.running && runUnit(randomUnit()), 1000);
}

function renderHistory() {
  $('wu-history').tBodies[0].replaceChildren(
    ...state.results.map((card) => {
      const g = card.metrics?.prediction_gain ?? {};
      return makeRow([
        card.unit,
        card.config?.hidden,
        fmt(g.tone?.gru),
        fmt(g.noise?.gru),
        fmt(card.metrics?.drift_probe?.r2_synthetic),
        BACKEND_NAMES[card.backend] ?? card.backend,
        card.seconds,
      ]);
    }),
  );
}

function download(name, data) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  const a = Object.assign(document.createElement('a'), { href: url, download: name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// The submit link opens the issue in a new tab. The card also goes on the clipboard: a card too
// long for a link is pasted from there, and so is any card GitHub drops while the volunteer signs
// in. Compact, like the link: pretty-printed, a card ingestion accepts can exceed its size limit.
function onSubmit(current, boxId, say) {
  return () => {
    const { card, issue } = current();
    const text = submissionText(card);
    const copied = navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject(new Error('no clipboard'));
    copied.then(
      () => say(issue.prefilled
        ? 'Opened a GitHub issue with the card filled in. The card is also on your clipboard, in case GitHub asks you to sign in first.'
        : 'This card is too long to fill in automatically, so it’s on your clipboard: paste it into the empty json block of the GitHub issue.'),
      () => {
        if (issue.prefilled) return say('Opened a GitHub issue with the card filled in.');
        const box = $(boxId);
        box.value = text;
        box.hidden = false;
        box.select();
        say('This card is too long to fill in automatically, and your browser blocked copying, so the text to paste is selected below: press Ctrl+C or ⌘C and paste it into the empty json block of the GitHub issue.', true);
      },
    );
  };
}

function onCopy(current, preId, say) {
  return () => {
    navigator.clipboard.writeText(JSON.stringify(current(), null, 2)).then(
      () => say('Result card copied.'),
      () => {
        getSelection().selectAllChildren($(preId));
        say('Your browser blocked copying, so the card is selected: press Ctrl+C or ⌘C.');
      },
    );
  };
}

// Two-step confirmation on the page itself (some hosts block confirm() dialogs).
function onConfirmedClick(button, action) {
  let timer;
  const label = button.textContent;
  return () => {
    if (!button.dataset.armed) {
      button.dataset.armed = 'yes';
      button.textContent = 'Click again to delete all';
      timer = setTimeout(() => {
        delete button.dataset.armed;
        button.textContent = label;
      }, 4000);
      return;
    }
    clearTimeout(timer);
    delete button.dataset.armed;
    button.textContent = label;
    action();
  };
}

// ---------- observatory viewport ----------

const viewports = { lab: null, recovery: null };
const session = { units: 0, recovery: 0 }; // units computed since the page opened
let skyData = null;

const plural = (n, word) => `${n.toLocaleString()} ${word}${n === 1 ? '' : 's'}`;
// What the volunteer's compute became: units computed this session, and of the cards kept in
// this browser, how many await submission and how many the community results accepted.
const sessionStats = (kind) => {
  const t = tally(keptCards(kind), contributions.marks, contributions.accepted);
  const computed = kind === 'recovery' ? plural(session.recovery, 'recovery unit') : plural(session.units, 'Track A work unit');
  return `This session: ${computed} computed · In this browser: ${t.awaiting.toLocaleString()} awaiting submission, ${t.accepted.toLocaleString()} accepted into community results`;
};

// The compass direction of position b seen from a, for short labels ("Off N").
function compass(a, b) {
  const east = ((((b.ra_hours - a.ra_hours + 12) % 24) + 24) % 24 - 12) * 15 * Math.cos((a.dec_deg * Math.PI) / 180);
  const north = b.dec_deg - a.dec_deg;
  return Math.abs(north) >= Math.abs(east) ? (north > 0 ? 'N' : 'S') : east > 0 ? 'E' : 'W';
}

const skyFields = ({ ra_hours, dec_deg, constellation, distance_label, epoch, telescope }) => ({ ra_hours, dec_deg, constellation, distance_label, epoch, telescope });

// Where the telescope pointed for the loaded slices (manifest version 2): one target per
// family, position and date.
function pointings() {
  const seen = new Map();
  for (const s of state.slices) {
    const sky = s.source?.sky;
    if (!sky || !Number.isFinite(sky.ra_hours) || !Number.isFinite(sky.dec_deg) || !familyById[s.family]) continue;
    const date = String(sky.epoch ?? '').slice(0, 10);
    const id = `${s.family}@${sky.ra_hours.toFixed(4)},${sky.dec_deg.toFixed(3)}@${date}`;
    if (!seen.has(id)) seen.set(id, { id, family: s.family, kind: s.family === 'offsource' ? 'off' : 'signal', date, ...skyFields(sky) });
  }
  const list = [...seen.values()];
  for (const t of list) {
    const on = list.find((o) => o.kind === 'signal' && o.date === t.date);
    t.label = t.kind === 'off' ? `Off-source · ${t.date}` : `${familyById[t.family].label} · ${t.date}`;
    t.short = t.kind === 'off' ? (on ? `Off ${compass(on, t)}` : 'Off-source') : `${familyById[t.family].label} ${t.date.slice(0, 4)}`;
  }
  return list;
}

// Synthetic signals don't come from the sky: their marker sits low over the Green Bank horizon,
// where a local transmitter would be, and moves with the time of day.
function localTarget() {
  const [ra, dec] = horizonToEquatorial(12, 160, GREEN_BANK.lat, siderealHours(new Date(), GREEN_BANK.lon));
  return {
    id: 'local',
    kind: 'local',
    label: 'Synthetic signals',
    short: 'Local (synthetic)',
    ra_hours: ra / 15,
    dec_deg: dec,
    constellation: skyData ? constellationAt(ra, dec, skyData.boundaries) : null,
    distance_label: 'Generated in this browser, not received',
    note: 'Marked low over the Green Bank horizon, where a local transmitter would be',
  };
}

const recoveryTarget = () => ({ id: 'recovery', kind: 'off', label: 'Recovery windows · 2020-07-16', short: 'Recovery windows', ...RECOVERY_DATA.sky });
const near = (a, b) => Math.abs(a.ra_hours - b.ra_hours) < 1e-3 && Math.abs(a.dec_deg - b.dec_deg) < 1e-2;

function refreshTargets() {
  const references = (skyData?.references ?? []).map((r) => ({ ...r, kind: 'reference', short: r.label }));
  const all = pointings();
  viewports.lab?.setTargets([...all, localTarget(), ...references]);
  viewports.recovery?.setTargets([...all.filter((t) => !near(t, RECOVERY_DATA.sky)), recoveryTarget(), ...references]);
}

// What a real family's loaded slices span, for the map's "Data from" line: every observing
// session, not just the one its beacon marks. null when one session (the beacon's label says it).
function dataSource(id) {
  const dated = state.slices.filter((s) => s.family === id && Number.isFinite(s.startMjd));
  const byDate = sessions(dated).map((g) => mjdDate(Math.min(...g.map((s) => s.startMjd))));
  if (byDate.length < 2) return null;
  const scans = groups(dated, SCAN_GAP_DAYS).length;
  return `Data from ${familyById[id].label} · ${byDate.slice(0, -1).join(', ')} and ${byDate.at(-1)}: ${scans} scans in ${byDate.length} observing sessions`;
}

// A work unit trains on synthetic tones and tests on every family, the loaded real ones too.
function unitDataSource() {
  const dates = [...new Set(state.slices.filter((s) => Number.isFinite(s.startMjd)).map((s) => mjdDate(s.startMjd)))].sort();
  const real = dates.length ? `, with real data from ${dates.slice(0, -1).join(', ')}${dates.length > 1 ? ' and ' : ''}${dates.at(-1)}` : '';
  return `Trains on synthetic tones; tests on every family${real}`;
}

// The target a family's computations point at: its latest pointing (for off-source, the one the
// recovery windows come from, if loaded), or the local marker for synthetic signals.
function targetForFamily(id) {
  if (!familyById[id]?.real) return 'local';
  const own = pointings().filter((t) => t.family === id).sort((a, b) => b.date.localeCompare(a.date) || Number(near(b, RECOVERY_DATA.sky)) - Number(near(a, RECOVERY_DATA.sky)));
  return own[0]?.id ?? null;
}

function onSkySelect(t) {
  if (state.running) return;
  if (t.family && t.family !== state.family && familyAvailable(t.family)) selectFamily(t.family, { fromMap: true });
  else if (t.kind === 'local' && familyById[state.family].real) selectFamily('tone', { fromMap: true });
}

function startViewports() {
  viewports.lab = new SkyMap($('lab-viewport'), {
    onSelect: onSkySelect,
    autoRunLabel: 'Auto-run Track A work units',
    onAutoRun: (on) => {
      $('wu-loop').checked = on;
      if (on && !state.running) runUnit(randomUnit());
    },
    onReview: () => openReview('lab'),
    onObservatory: (on) => !on && offerReview('lab'),
  });
  viewports.recovery = new SkyMap($('rc-viewport'), {
    autoRunLabel: 'Auto-run recovery units',
    onAutoRun: (on) => {
      $('rc-loop').checked = on;
      rc.autoStart = on && !rc.ready; // started once the windows are ready, if still ticked then
      if (on && rc.ready && !rc.running) runRecovery(randomUnit());
      if (!on) {
        clearTimeout(rc.timer);
        setRecoveryRunning(rc.running);
      }
    },
    onReview: () => openReview('recovery'),
    onObservatory: (on) => !on && offerReview('recovery'),
  });
  refreshContributions();
  loadSky()
    .then((sky) => {
      skyData = sky;
      refreshTargets();
      viewports.lab.select(targetForFamily(state.family), 60);
      viewports.recovery.select('recovery', 60);
    })
    .catch(() => {}); // the viewport reports it
}

// ---------- recovery units ----------

const KIND_LABELS = {
  tone: 'Drifting tone',
  intermittent: 'Intermittent tone',
  wideband: 'Wideband signal',
  sidebands: 'Carrier with sidebands',
  fsk: 'Frequency-shift keying',
  stationary: 'Zero-drift line (decoy)',
  pulse: 'Broadband pulse (decoy)',
};
const RECOVERY_SERIES = [
  { id: 'dedoppler', name: 'de-Doppler search', color: '--series-1' },
  { id: 'energy', name: 'Energy detector', color: '--series-2' },
];
const pct = (v, digits = 0) => `${(100 * v).toFixed(digits)}%`;
const rateLabel = (rate) => pct(Number(rate), rate === '0.001' ? 1 : 0);

const rc = {
  worker: null,
  ready: false,
  windows: 0,
  empty: null, // flags per detector and rate on the windows with nothing injected
  running: false,
  stopLoop: false,
  autoStart: false, // observatory auto-run asked for before the windows were ready
  timer: null,
  card: null,
  issue: null,
  results: [].concat(store.get('recoveryResults', [])).filter((card) => isRecoveryCard(card) && !validateCard(card).length),
  community: null, // the community's recovery results, once loaded
  communityFailed: false,
};

function recoveryStatus(text, error = false) {
  $('rc-status').textContent = text;
  $('rc-status').classList.toggle('error', error);
}

function setRecoveryRunning(running, looping = false) {
  rc.running = running;
  $('rc-run').disabled = running || !rc.ready;
  $('rc-random').disabled = running;
  $('rc-stop').disabled = !running && !looping;
}

function startRecovery() {
  if (rc.worker) return;
  recoveryStatus('Loading the windows (3.2 MB) and checking their SHA-256…');
  rc.worker = new Worker(new URL('./recovery.worker.js', import.meta.url), { type: 'module' });
  rc.worker.onmessage = ({ data }) => onRecoveryMessage(data);
  rc.worker.onerror = (event) => recoveryStatus(`The recovery worker couldn't start (${event.message || 'unknown error'}).`, true);
}

function onRecoveryMessage(msg) {
  if (msg.type === 'error') {
    recoveryStatus(`Couldn't load the windows: ${msg.message}.`, true);
  } else if (msg.type === 'ready') {
    Object.assign(rc, { ready: true, windows: msg.windows, empty: msg.empty });
    setRecoveryRunning(rc.running);
    recoveryStatus(`Ready: ${msg.windows.toLocaleString()} windows loaded, matching the spec’s SHA-256.`);
    renderCombined();
    if (rc.autoStart && $('rc-loop').checked && !rc.running) runRecovery(randomUnit());
    rc.autoStart = false;
  } else if (msg.type === 'progress') {
    viewports.recovery?.telemetry(`Recovery unit ${msg.unit} · de-Doppler search · ${Math.round(msg.fraction * 1024).toLocaleString()} of 1,024 windows scored`, msg.fraction);
    viewports.recovery?.scanned(msg.cells);
  } else if (msg.type === 'result') {
    onRecoveryResult(msg.card, msg.outcomes);
  }
}

function runRecovery(unit) {
  clearTimeout(rc.timer);
  rc.stopLoop = false;
  $('rc-number').value = unit;
  setRecoveryRunning(true);
  recoveryStatus(`Unit ${unit}: dealing the windows and scoring them…`);
  viewports.recovery?.startJob('recovery', 'recovery', `Recovery unit ${unit} · dealing 756 hidden signals into 1,024 windows…`);
  rc.worker.postMessage({ type: 'unit', unit });
}

function onRecoveryResult(card, outcomes = []) {
  rc.card = card;
  rc.issue = issueLink(card, REPO);
  $('rc-submit').href = rc.issue.url;
  $('rc-submit-text').hidden = true;
  const full = keep('recovery', card);
  recoveryStatus(`Unit ${card.unit} finished in ${card.seconds} s.`);
  $('rc-unit-panel').hidden = false;
  $('rc-card').textContent = JSON.stringify(card, null, 2);
  renderRecoveryUnit();
  renderCombined();
  session.recovery += 1;
  const totals = recoveryTotals(card.results, card.config.per_cell, '0.01');
  viewports.recovery?.outcomes(outcomes, OUTCOMES.map((outcome, i) => ({ outcome, count: outcomes.filter((c) => c === i).length })), 'Each cell is one frequency window of this one pointing, in order along the band (the layout is illustrative, not positions on the sky). De-Doppler at 1% false alarms:');
  viewports.recovery?.finishJob(`Recovery unit ${card.unit} · de-Doppler found ${totals.found.dedoppler} of ${totals.signals} hidden signals, the energy detector ${totals.found.energy} · ${card.seconds} s`);
  if (full) pauseForReview('recovery');
  refreshContributions();
  const again = $('rc-loop').checked && !rc.stopLoop;
  setRecoveryRunning(false, again);
  if (again) rc.timer = setTimeout(() => $('rc-loop').checked && !rc.stopLoop && !rc.running && runRecovery(randomUnit()), 1000);
}

// Signals found and decoys flagged in total, per detector, and the signals' paired outcomes.
function recoveryTotals(results, perCell, rate) {
  const sum = (kinds, table) => kinds.reduce((total, kind) => total + table[kind].reduce((a, b) => a + b, 0), 0);
  const [a, b] = RECOVERY_SERIES.map((s) => s.id);
  const found = Object.fromEntries(RECOVERY_SERIES.map((s) => [s.id, sum(SIGNALS, results.recovery[s.id][rate])]));
  const both = sum(SIGNALS, results.both[rate]);
  return {
    signals: SIGNALS.length * STRENGTHS.length * perCell,
    decoys: DECOYS.length * STRENGTHS.length * perCell,
    found,
    flagged: Object.fromEntries(RECOVERY_SERIES.map((s) => [s.id, sum(DECOYS, results.recovery[s.id][rate])])),
    both,
    onlyFirst: found[a] - both,
    onlySecond: found[b] - both,
  };
}

// What the volunteer's computation tested, and what it found.
function recoverySummary(card, rate) {
  const { results } = card;
  const t = recoveryTotals(results, card.config.per_cell, rate);
  const fa = results.false_alarms;
  return `Unit ${card.unit} tested whether two standard detectors find weak signals in real data without extra false alarms, at thresholds that flag ${rateLabel(rate)} of empty windows on a separate scan. Of the ${t.signals} signals hidden, the de-Doppler search found ${t.found.dedoppler} (${pct(t.found.dedoppler / t.signals)}) and the energy detector ${t.found.energy} (${pct(t.found.energy / t.signals)}): ${t.both} by both, ${t.onlyFirst} by de-Doppler alone and ${t.onlySecond} by energy alone. On the ${results.signal_free} windows left empty, the de-Doppler search raised ${fa.dedoppler[rate]} false alarms and the energy detector ${fa.energy[rate]}.`;
}

function recoveryCharts(box, results, perCell, rate) {
  if (!box.childElementCount) {
    box.replaceChildren(
      ...KINDS.map((kind) => {
        const fig = document.createElement('figure');
        const canvas = document.createElement('canvas');
        canvas.dataset.height = '150';
        canvas.setAttribute('aria-label', `${KIND_LABELS[kind]}: ${DECOYS.includes(kind) ? 'flagged' : 'found'} by strength`);
        fig.append(element('figcaption', KIND_LABELS[kind]), canvas);
        return fig;
      }),
    );
    legend(box, RECOVERY_SERIES);
  }
  [...box.querySelectorAll('canvas')].forEach((canvas, j) => {
    const kind = KINDS[j];
    const series = RECOVERY_SERIES.map((s) => {
      const counts = results.recovery[s.id][rate][kind];
      const intervals = counts.map((k) => wilson(k, perCell));
      return {
        name: s.name,
        color: s.color,
        points: STRENGTHS.map((x, i) => [x, counts[i] / perCell]),
        band: STRENGTHS.map((x, i) => [x, ...intervals[i]]),
        describe: (i) => `${counts[i]} of ${perCell} (95%: ${pct(intervals[i][0])}–${pct(intervals[i][1])})`,
      };
    });
    lineChart(canvas, series, { xLabel: 'SNR', yLabel: DECOYS.includes(kind) ? 'flagged' : 'found', yMin: 0, yMax: 1, logX: true, xTicks: STRENGTHS, markers: true, xFormat: String, yFormat: (v) => pct(v) });
  });
}

function recoveryTable(table, results, perCell, signalFree, rate) {
  const t = recoveryTotals(results, perCell, rate);
  const cell = (k, n) => {
    const [low, high] = wilson(k, n);
    return `${k.toLocaleString()} of ${n.toLocaleString()}: ${pct(k / n, 1)} (95%: ${pct(low, 1)}–${pct(high, 1)})`;
  };
  fillTable(table, ['', ...RECOVERY_SERIES.map((s) => s.name)], [
    ['False alarms: empty windows flagged', ...RECOVERY_SERIES.map((s) => cell(results.false_alarms[s.id][rate], signalFree))],
    ['Signals found', ...RECOVERY_SERIES.map((s) => cell(t.found[s.id], t.signals))],
    ['Signals found by this detector only', cell(t.onlyFirst, t.signals), cell(t.onlySecond, t.signals)],
    ['Decoys flagged', ...RECOVERY_SERIES.map((s) => cell(t.flagged[s.id], t.decoys))],
  ]);
}

function renderRecoveryUnit() {
  if (!rc.card || $('rc-unit-panel').hidden) return;
  const rate = $('rc-rate').value;
  const { results, config } = rc.card;
  $('rc-unit-title').textContent = `Unit ${rc.card.unit}`;
  $('rc-summary').textContent = recoverySummary(rc.card, rate);
  recoveryCharts($('rc-unit-charts'), results, config.per_cell, rate);
  recoveryTable($('rc-unit-table'), results, config.per_cell, results.signal_free, rate);
}

function renderCombined() {
  const rate = $('rc-rate').value;
  const fromBrowser = $('rc-source').value === 'browser';
  const status = $('rc-combined-status');
  const clear = () => {
    const box = $('rc-combined-charts');
    if (box.previousElementSibling?.classList.contains('legend')) box.previousElementSibling.remove();
    box.replaceChildren();
    $('rc-combined-table').replaceChildren();
  };
  const e = rc.empty;
  $('rc-empty-note').textContent = e
    ? `With nothing injected, the de-Doppler search flags ${e.dedoppler[rate]} of the ${rc.windows.toLocaleString()} windows at this threshold and the energy detector ${e.energy[rate]}. Every unit’s empty windows are drawn from these, so however many units are combined, their false alarms approach ${pct(e.dedoppler[rate] / rc.windows, 1)} and ${pct(e.energy[rate] / rc.windows, 1)}.`
    : '';
  if (!fromBrowser && !rc.community) {
    clear();
    status.textContent = rc.communityFailed ? "Couldn't load community results." : 'Loading community results…';
    return;
  }
  const runs = fromBrowser ? rc.results.map((card) => ({ card })) : rc.community;
  const { units, perCell, signalFree, results } = combine(runs.map((r) => r.card));
  if (!units) {
    clear();
    status.textContent = fromBrowser ? 'No recovery units run in this browser yet.' : 'No recovery units submitted yet. Run one and submit its card to be the first.';
    return;
  }
  const volunteers = new Set(runs.map((r) => r.submitted_by)).size;
  const count = (n, noun) => `${n.toLocaleString()} ${noun}${n === 1 ? '' : 's'}`;
  status.textContent = fromBrowser
    ? `${count(units, 'unit')} run in this browser: ${perCell.toLocaleString()} injections of each class at each strength.`
    : `${count(units, 'unit')} from ${count(volunteers, 'volunteer')}, each recomputed when submitted: ${perCell.toLocaleString()} injections of each class at each strength.`;
  if ($('tab-recovery').hidden) return;
  recoveryCharts($('rc-combined-charts'), results, perCell, rate);
  recoveryTable($('rc-combined-table'), results, perCell, signalFree, rate);
}

// ---------- your contributions ----------

// What happened to the cards computed here (see contributions.js): marks record when a card was
// saved to a file or opened as a GitHub issue; accepted holds the community results' cards.
const contributions = {
  marks: store.get('contributionMarks', {}),
  accepted: new Set(),
  kind: 'lab', // whose cards the review shows: 'lab' (Track A work units) or 'recovery'
  view: 'pending', // 'pending' (not yet accepted) or 'all'
  shown: 0, // rows listed before "Show more"
  pendingSave: null, // cards downloaded but not yet confirmed saved: { kind, cards }
  offered: { lab: 0, recovery: 0 }, // units computed when the review was last offered
  checked: 0, // when the community results were last loaded
};
const KEEP = { lab: 200, recovery: 300 };
const KIND_NAMES = { lab: 'Track A work units', recovery: 'recovery units' };
const REVIEW_SHOWN = 50;
const keptCards = (kind) => (kind === 'recovery' ? rc.results : state.results);
const statusOf = (card) => cardStatus(card, contributions.marks, contributions.accepted);

// Keeps a new card without dropping one that exists nowhere else. Returns true when the kept
// cards are over their limit because none could be dropped.
function keep(kind, card) {
  const replaces = kind === 'recovery' ? (c) => c.unit === card.unit : undefined; // one card per recovery unit
  const { cards, full } = keepCard(keptCards(kind), card, KEEP[kind], (c) => safeToDrop(c, contributions.marks, contributions.accepted), replaces);
  if (kind === 'recovery') rc.results = cards;
  else state.results = cards;
  store.set(kind === 'recovery' ? 'recoveryResults' : 'results', cards);
  saveMarks();
  return full;
}

function saveMarks() {
  contributions.marks = pruneMarks(contributions.marks, [state.results, rc.results]);
  store.set('contributionMarks', contributions.marks);
}

function markCards(cards, what) {
  const now = new Date().toISOString();
  for (const card of cards) {
    if (card) contributions.marks[cardKey(card)] = { ...contributions.marks[cardKey(card)], [what]: now };
  }
  saveMarks();
  refreshContributions();
}

// The review buttons, the HUD's contribution counts and, if it's open, the review.
function refreshContributions() {
  for (const kind of ['lab', 'recovery']) {
    const t = tally(keptCards(kind), contributions.marks, contributions.accepted);
    viewports[kind]?.setReview(t.total ? (t.awaiting ? `Review and submit (${t.awaiting.toLocaleString()})` : 'Review results') : null, t.awaiting > 0);
    viewports[kind]?.setStats(sessionStats(kind));
  }
  if ($('review').open) renderReview();
}

// Every kept card exists nowhere else: stop auto-run rather than drop one, and show the review.
function pauseForReview(kind) {
  const box = kind === 'recovery' ? $('rc-loop') : $('wu-loop');
  const looping = box.checked;
  box.checked = false;
  viewports[kind]?.setAutoRun(false);
  const say = kind === 'recovery' ? recoveryStatus : unitStatus;
  say(`This browser keeps up to ${KEEP[kind]} ${KIND_NAMES[kind]}, and none of the kept cards is accepted into community results or saved to a file yet${looping ? ', so auto-run has paused' : ''}. Submit or download some to make room: no card is dropped until it’s accepted or saved.`);
  openReview(kind);
}

function openReview(kind) {
  contributions.kind = kind;
  contributions.offered[kind] = kind === 'recovery' ? session.recovery : session.units;
  const { total, accepted } = tally(keptCards(kind), contributions.marks, contributions.accepted);
  contributions.view = total > accepted ? 'pending' : 'all';
  contributions.shown = REVIEW_SHOWN;
  $('review-status').textContent = '';
  $('review-submit-text').hidden = true;
  $('review-confirm').hidden = contributions.pendingSave?.kind !== kind;
  renderReview();
  if (!$('review').open) $('review').showModal();
}

// Leaving observatory mode ends a viewing session: offer the review if units computed since it
// was last offered left cards to submit. Deferred, so the Escape that closed the observatory
// doesn't also close the review.
function offerReview(kind) {
  const computed = kind === 'recovery' ? session.recovery : session.units;
  const { awaiting } = tally(keptCards(kind), contributions.marks, contributions.accepted);
  if (computed > contributions.offered[kind] && awaiting) setTimeout(() => openReview(kind));
}

function reviewStatus(text, error = false) {
  $('review-status').textContent = text;
  $('review-status').classList.toggle('error', error);
}

function renderReview() {
  const kind = contributions.kind;
  const cards = keptCards(kind);
  const t = tally(cards, contributions.marks, contributions.accepted);
  const thisSession = kind === 'recovery' ? session.recovery : session.units;
  $('review-title').textContent = `Your ${KIND_NAMES[kind]} in this browser`;
  $('review-counts').textContent = cards.length
    ? `${plural(t.total, 'result')} kept here, ${thisSession.toLocaleString()} computed this session: ${t.awaiting.toLocaleString()} awaiting submission, ${t.opened.toLocaleString()} opened on GitHub, ${t.accepted.toLocaleString()} accepted into community results.`
    : 'No results in this browser yet.';
  // Opening an issue is a draft: say so wherever cards are still only opened.
  $('review-opened').hidden = !t.opened;
  $('review-opened').textContent = t.opened
    ? `${plural(t.opened, 'card')} opened on GitHub ${t.opened === 1 ? 'isn’t' : 'aren’t'} submitted yet: post each issue to submit it. Its status changes to Accepted once the automatic check records it; use “Check for accepted cards” to look again.`
    : '';
  $('review-outcome').textContent = cards.length ? (kind === 'recovery' ? recoveryOutcome(cards[0]) : workUnitOutcome(cards[0])) : 'Run a unit to see what it measures.';
  const pending = cards.filter((c) => statusOf(c) !== 'accepted');
  const listed = contributions.view === 'pending' ? pending : cards;
  $('review-view-pending').textContent = `Not yet accepted (${pending.length.toLocaleString()})`;
  $('review-view-all').textContent = `All (${cards.length.toLocaleString()})`;
  $('review-view-pending').setAttribute('aria-pressed', String(contributions.view === 'pending'));
  $('review-view-all').setAttribute('aria-pressed', String(contributions.view === 'all'));
  fillTable($('review-table'), ['Unit', 'Started', kind === 'recovery' ? 'Signals found at 1% false alarms' : 'GRU gain: drifting tones', 'Status', ''], []);
  $('review-table').tBodies[0].append(...listed.slice(0, contributions.shown).map((card) => reviewRow(kind, card)));
  const shown = Math.min(listed.length, contributions.shown);
  $('review-more').textContent = listed.length ? `Showing ${shown.toLocaleString()} of ${listed.length.toLocaleString()}, newest first.` : 'Nothing to list here.';
  $('review-show-more').hidden = shown >= listed.length;
  $('review-show-more').textContent = `Show ${Math.min(REVIEW_SHOWN, listed.length - shown).toLocaleString()} more`;
  $('review-keep').textContent = `This browser keeps up to ${KEEP[kind]} ${KIND_NAMES[kind]} (${cards.length.toLocaleString()} now). To make room it drops only cards that are accepted, or that you downloaded and confirmed saving; if every kept card is neither, auto-run pauses until you submit or save some.`;
}

function reviewRow(kind, card) {
  const status = statusOf(card);
  const started = new Date(card.started);
  const tr = makeRow([card.unit, Number.isNaN(started.getTime()) ? '' : started.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }), cardOutcome(kind, card)]);
  const chip = document.createElement('td');
  chip.append(element('span', STATUSES[status], `chip ${status}`));
  const action = document.createElement('td');
  if (status !== 'accepted') {
    const issue = issueLink(card, REPO);
    const a = link(status === 'opened' ? 'Open again' : 'Submit', issue.url);
    a.className = 'button';
    a.onclick = () => {
      onSubmit(() => ({ card, issue }), 'review-submit-text', reviewStatus)();
      setTimeout(() => markCards([card], 'opened')); // after the link opens: marking redraws the table
    };
    action.append(a);
  }
  tr.append(chip, action);
  return tr;
}

function cardOutcome(kind, card) {
  if (kind === 'recovery') {
    const t = recoveryTotals(card.results, card.config.per_cell, '0.01');
    return `de-Doppler ${t.found.dedoppler} of ${t.signals} · energy ${t.found.energy}`;
  }
  const g = card.metrics?.prediction_gain?.tone ?? {};
  return `${fmt(g.gru)} (linear ${fmt(g.linear)})`;
}

// What a recovery unit measured, in plain words, with the two detectors side by side.
function recoveryOutcome(card) {
  const t = recoveryTotals(card.results, card.config.per_cell, '0.01');
  const fa = card.results.false_alarms;
  return `Unit ${card.unit} measured how well two standard detectors recover synthetic signals hidden in archived telescope data: ${(t.signals + t.decoys).toLocaleString()} signals and interference-like decoys, injected into real windows of a Green Bank off-source scan (2020-07-16). ` +
    `At thresholds that flag 1% of empty windows, the de-Doppler search (the usual search for drifting signals) found ${t.found.dedoppler} of the ${t.signals} signals (${pct(t.found.dedoppler / t.signals)}), and the energy detector (a simpler baseline that measures excess power) found ${t.found.energy} (${pct(t.found.energy / t.signals)}). ` +
    `The energy detector caught ${t.onlySecond} that the de-Doppler search missed, against ${t.onlyFirst} the other way. On the ${card.results.signal_free} windows left empty they raised ${fa.dedoppler['0.01']} and ${fa.energy['0.01']} false alarms.`;
}

// What a Track A work unit measured, against its baselines.
function workUnitOutcome(card) {
  const g = card.metrics?.prediction_gain ?? {};
  const line = (id) => g[id] && `${familyById[id].label} ${fmt(g[id].gru)} (linear ${fmt(g[id].linear)}, persistence ${fmt(g[id].persistence)})`;
  const real = ['voyager', 'offsource'].map(line).filter(Boolean);
  return `Unit ${card.unit} trained a small GRU (hidden size ${card.config?.hidden}) on noise and synthetic drifting tones, then measured how well it predicts each signal’s next frame against two simple baselines: a linear predictor, and persistence (the next frame repeats this one). ` +
    `A prediction gain of 0 is no better than predicting the average frame; 1 is perfect. Drifting tones: ${line('tone') ?? 'not measured'}.${real.length ? ` Real data it never trained on: ${real.join('; ')}.` : ''}`;
}

// ---------- community results ----------

const STATUS_LABELS = { single: 'One volunteer so far', replicated: 'Replicated', disagree: 'Runs disagree' };
const GITHUB_LOGIN = /^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$/;

function link(text, href) {
  const a = element('a', text);
  a.href = href;
  a.target = '_blank';
  a.rel = 'noopener';
  return a;
}

function volunteerCell(login) {
  const td = document.createElement('td');
  td.append(GITHUB_LOGIN.test(login) ? link(login, `https://github.com/${login}`) : document.createTextNode(login));
  return td;
}

async function loadCommunity() {
  const status = $('community-status');
  let store;
  try {
    const response = await fetch('data/community_results.json', { cache: 'no-cache' });
    if (!response.ok) throw new Error(String(response.status));
    store = await response.json();
  } catch {
    status.textContent = "Couldn't load community results.";
    rc.communityFailed = true;
    renderCombined();
    return;
  }
  const all = Array.isArray(store.results) ? store.results : [];
  contributions.accepted = acceptedCards(all);
  contributions.checked = Date.now();
  refreshContributions();
  rc.community = all.filter((r) => isRecoveryCard(r.card) && !validateCard(r.card).length);
  renderCombined();
  const results = all.filter((r) => !isRecoveryCard(r.card));
  if (!results.length) {
    status.textContent = 'No community results yet. Run a unit and submit its card to be the first.';
    return;
  }
  const { units, leaderboard } = summarize(results);
  const replicated = units.filter((u) => u.status === 'replicated').length;
  status.textContent = `${results.length} result cards for ${units.length} units from ${leaderboard.length} volunteers. ${replicated} of ${units.length} units replicated.`;

  fillTable($('community-units'), ['Unit', 'Hidden', 'Data', 'Runs', 'Volunteers', 'Backends', 'Status', 'Issues'], []);
  $('community-units').tBodies[0].append(
    ...units.map((u) => {
      // Runs on different slice datasets are separate groups: name the dataset by its id's first characters.
      const data = u.dataset_id ? `slices ${u.dataset_id.slice(0, 8)}` : 'synthetic only';
      const tr = makeRow([u.unit, u.hidden, data, u.runs, u.volunteers.length, u.backends.map((b) => BACKEND_NAMES[b] ?? b).join(', '), STATUS_LABELS[u.status]]);
      if (u.dataset_id) tr.cells[2].title = `Dataset ${u.dataset_id}`;
      const issues = document.createElement('td');
      u.issues.filter(Number.isInteger).forEach((n, i) => {
        if (i) issues.append(', ');
        issues.append(link(`#${n}`, `${REPO}/issues/${n}`));
      });
      tr.append(issues);
      return tr;
    }),
  );
  fillTable($('community-volunteers'), ['Volunteer', 'Units run', 'Units replicated'], []);
  $('community-volunteers').tBodies[0].append(
    ...leaderboard.map((p) => {
      const tr = document.createElement('tr');
      tr.append(volunteerCell(p.volunteer), ...makeRow([p.units, p.replicated]).children);
      return tr;
    }),
  );
  $('community-tables').hidden = false;
}

// ---------- data ----------

function setSlices(slices, source) {
  // The published slices load in the background; never let them replace files the volunteer dropped.
  if (source === 'site' && state.sliceSource === 'files') return;
  state.slices = slices;
  state.sliceSource = source;
  broadcast({ type: 'slices', slices: forWorkers(slices) });
  refreshTargets();
  if (familyAvailable(state.family)) renderFamilies();
  else selectFamily('tone');
  const status = $('slice-status');
  if (!slices.length) {
    status.textContent = 'No real-data slices are published with this site yet. Maintainers create them with tools/extract_slices.py; you can also drop slice files below.';
  } else {
    const real = FAMILIES.filter((f) => f.real && familyAvailable(f.id));
    const counts = real.map((f) => `${slices.filter((s) => s.family === f.id).length} ${f.label}`).join(', ');
    const files = new Set(slices.map((s) => s.source?.file ?? 'unknown file'));
    const origin = files.size === 1 ? `source: ${[...files][0]}` : `from ${files.size} scans`;
    const buttons = real.map((f) => f.label).join(' and ');
    status.textContent = `${slices.length} slices loaded ${source === 'site' ? 'from this site' : 'from your files'} (${counts}; ${origin}). The ${buttons} ${real.length > 1 ? 'buttons are' : 'button is'} now enabled.`;
  }
  if (slices.length) {
    fillTable(
      $('slice-table'),
      ['Slice', 'Family', 'Signal or window', 'Scan start (MJD)', 'Size', 'Drift (Hz/s)', 'Peak SNR', 'SHA-256 (verified)'],
      slices.map((s) => [
        s.id,
        familyById[s.family].label,
        s.group,
        Number.isFinite(s.startMjd) ? s.startMjd.toFixed(5) : 'unknown',
        `${T} × ${s.width}`,
        fmt(driftToHzPerS(s.driftChannelsPerStep, s.foffHz, s.tsamp), 3),
        fmt(Number(s.source?.snr), 1),
        `${s.sha256.slice(0, 12)}…`,
      ]),
    );
  } else {
    $('slice-table').replaceChildren();
  }
}

async function handleSliceFiles(files) {
  try {
    setSlices(await loadSliceFiles(files), 'files');
  } catch (err) {
    $('slice-status').textContent = `Couldn't load those files: ${err.message}`;
  }
}

// ---------- theme, tabs, resize ----------

function applyTheme() {
  if (PREVIEW) return;
  if (state.theme === 'auto') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = state.theme;
  $('theme').textContent = `Theme: ${state.theme}`;
}

function redraw() {
  if (state.lastEpoch) renderLab();
  else renderPreview();
  renderTrackB();
  renderTokenModel();
  if (!$('tab-recovery').hidden) {
    renderRecoveryUnit();
    renderCombined();
  }
}

function showTab(name) {
  for (const b of document.querySelectorAll('.tabs button')) b.setAttribute('aria-selected', String(b.dataset.tab === name));
  for (const s of document.querySelectorAll('.tab')) s.hidden = s.id !== `tab-${name}`;
  if (name === 'lab') redraw();
  if (name === 'recovery') {
    startRecovery();
    renderRecoveryUnit();
    renderCombined();
  }
}

// ---------- wiring ----------

function init() {
  if (PREVIEW) {
    for (const id of ['theme', 'wu-download', 'wu-download-all', 'rc-download', 'rc-download-all', 'review-download']) $(id).hidden = true;
  }
  applyTheme();
  $('backend').value = store.get('backend', 'auto');
  $('wu-spec').textContent = SPEC;
  $('wu-number').max = MAX_UNIT;
  $('rc-spec').textContent = RECOVERY_SPEC;
  $('rc-number').max = MAX_UNIT;
  $('rosetta-text').value = state.text || ROSETTA_TEXT;

  startTokens();
  startEngine();
  startViewports();
  renderFamilies();
  renderHistory();

  for (const b of document.querySelectorAll('.tabs button')) b.onclick = () => (location.hash = b.dataset.tab);
  window.addEventListener('hashchange', () => showTab(location.hash.slice(1) || 'lab'));
  showTab(location.hash.slice(1) || 'lab');

  $('backend').onchange = () => {
    store.set('backend', $('backend').value);
    startEngine();
  };
  $('theme').onclick = () => {
    state.theme = { auto: 'light', light: 'dark', dark: 'auto' }[state.theme];
    store.set('theme', state.theme);
    applyTheme();
    redraw();
  };
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);

  $('train').onclick = startTraining;
  $('stop').onclick = () => engine.postMessage({ type: 'stop' });
  $('shuffle').onchange = () => !state.running && renderPreview();
  $('seed').onchange = () => {
    if (!state.running) renderPreview();
    analyzeTokens();
  };
  $('analyze').onclick = analyzeTokens;
  $('tb-lm').onclick = startTokenModel;
  $('tb-lm-stop').onclick = () => {
    $('tb-lm-status').textContent = 'Stopping…';
    engine.postMessage({ type: 'stop' });
  };
  $('k').onchange = analyzeTokens;
  $('nframes').onchange = analyzeTokens;

  $('wu-random').onclick = () => ($('wu-number').value = randomUnit());
  $('wu-run').onclick = () => runUnit(Math.min(MAX_UNIT, Math.max(0, Math.floor(Number($('wu-number').value) || 0))));
  $('wu-stop').onclick = () => {
    state.stopLoop = true;
    engine.postMessage({ type: 'stop' });
  };
  $('wu-download').onclick = () => state.card && download(`eiathome-${state.card.spec.replace('/', '-')}-unit${state.card.unit}.json`, state.card);
  const submitUnit = onSubmit(() => ({ card: state.card, issue: state.issue }), 'wu-submit-text', unitStatus);
  $('wu-submit').onclick = () => {
    submitUnit();
    markCards([state.card], 'opened');
  };
  $('wu-copy').onclick = onCopy(() => state.card, 'wu-card', unitStatus);
  $('wu-download-all').onclick = () => download('eiathome-results.json', state.results);
  $('wu-clear').onclick = onConfirmedClick($('wu-clear'), () => {
    state.results = [];
    store.set('results', []);
    renderHistory();
    saveMarks();
    refreshContributions();
    unitStatus('Deleted the results stored in this browser.');
  });

  $('rc-random').onclick = () => ($('rc-number').value = randomUnit());
  $('rc-run').onclick = () => runRecovery(Math.min(MAX_UNIT, Math.max(0, Math.floor(Number($('rc-number').value) || 0))));
  $('rc-stop').onclick = () => {
    rc.stopLoop = true;
    clearTimeout(rc.timer);
    setRecoveryRunning(rc.running);
    recoveryStatus(rc.running ? 'Stopping after this unit…' : 'Stopped.');
  };
  $('rc-rate').onchange = () => {
    renderRecoveryUnit();
    renderCombined();
  };
  $('rc-source').onchange = renderCombined;
  $('rc-download').onclick = () => rc.card && download(`eiathome-recovery-unit${rc.card.unit}.json`, rc.card);
  const submitRecovery = onSubmit(() => ({ card: rc.card, issue: rc.issue }), 'rc-submit-text', recoveryStatus);
  $('rc-submit').onclick = () => {
    submitRecovery();
    markCards([rc.card], 'opened');
  };
  $('rc-copy').onclick = onCopy(() => rc.card, 'rc-card', recoveryStatus);
  $('rc-download-all').onclick = () => download('eiathome-recovery-results.json', rc.results);
  $('rc-clear').onclick = onConfirmedClick($('rc-clear'), () => {
    rc.results = [];
    store.set('recoveryResults', []);
    renderCombined();
    saveMarks();
    refreshContributions();
    recoveryStatus('Deleted the recovery cards stored in this browser.');
  });
  $('review-close').onclick = () => $('review').close();
  // A download can be blocked or cancelled without the page knowing, so the cards count as saved
  // (and can make room for new ones) only once the volunteer confirms having the file.
  $('review-download').onclick = () => {
    const kind = contributions.kind;
    const cards = [...keptCards(kind)];
    download(kind === 'recovery' ? 'eiathome-recovery-results.json' : 'eiathome-results.json', cards);
    contributions.pendingSave = { kind, cards };
    $('review-confirm').hidden = false;
    reviewStatus(`Asked your browser to download ${plural(cards.length, 'card')}. They count as saved only once you confirm you have the file.`);
  };
  $('review-saved').onclick = () => {
    const pending = contributions.pendingSave;
    contributions.pendingSave = null;
    $('review-confirm').hidden = true;
    if (!pending) return;
    markCards(pending.cards, 'saved');
    reviewStatus(`${plural(pending.cards.length, 'card')} marked as saved to your file. They can now make room for new results; they stay listed until they need to.`);
  };
  $('review-view-pending').onclick = () => {
    contributions.view = 'pending';
    contributions.shown = REVIEW_SHOWN;
    renderReview();
  };
  $('review-view-all').onclick = () => {
    contributions.view = 'all';
    contributions.shown = REVIEW_SHOWN;
    renderReview();
  };
  $('review-show-more').onclick = () => {
    contributions.shown += REVIEW_SHOWN;
    renderReview();
  };
  $('review-refresh').onclick = async () => {
    reviewStatus('Checking the community results…');
    const before = contributions.checked;
    await loadCommunity();
    if (contributions.checked === before) return reviewStatus('Couldn’t load the community results. Try again in a moment.', true);
    const { accepted } = tally(keptCards(contributions.kind), contributions.marks, contributions.accepted);
    reviewStatus(`Checked the community results: ${plural(accepted, 'card')} from this browser accepted. A posted card appears a few minutes after its issue is checked.`);
  };
  // Back from GitHub: look again for accepted cards, at most once a minute, if any were opened.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden || Date.now() - contributions.checked < 60000) return;
    if ([state.results, rc.results].some((cards) => cards.some((c) => statusOf(c) === 'opened'))) loadCommunity();
  });

  $('slice-files').onchange = (e) => handleSliceFiles(e.target.files);
  const drop = $('drop');
  drop.addEventListener('dragover', (e) => {
    e.preventDefault();
    drop.classList.add('over');
  });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('over');
    handleSliceFiles(e.dataTransfer.files);
  });

  $('rosetta-apply').onclick = () => {
    const text = $('rosetta-text').value;
    state.text = text === ROSETTA_TEXT ? '' : text;
    store.set('rosettaText', state.text);
    broadcast({ type: 'text', text: state.text });
    const symbols = new TextEncoder().encode(text).length * 2;
    $('rosetta-status').textContent = `Using ${state.text ? 'your' : 'the default'} text: ${symbols.toLocaleString()} symbols.${symbols < 2048 ? ' Short texts repeat in Track B, which adds artificial structure.' : ''}`;
    if (state.family === 'rosetta') analyzeTokens();
  };
  $('rosetta-reset').onclick = () => {
    $('rosetta-text').value = ROSETTA_TEXT;
    $('rosetta-apply').onclick();
  };

  let resizeTimer;
  new ResizeObserver(() => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(redraw, 150);
  }).observe(document.querySelector('main'));

  renderPreview();
  analyzeTokens();
  loadCommunity();
  loadPublishedSlices()
    .then((slices) => setSlices(slices, 'site'))
    .catch((err) => {
      $('slice-status').textContent = `Couldn't load the published slices: ${err.message}`;
    });
}

init();
