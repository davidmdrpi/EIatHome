import { summarize } from './community.js';
import { architectureLabel } from './models.js';
import { fmt, heatmap, lineChart, scatterPanel } from './plots.js';
import { deriveSeed, makeRng } from './rng.js';
import { forWorkers, loadPublishedSlices, loadSliceFiles } from './slices.js';
import { C, FAMILIES, ROSETTA_TEXT, T, driftToHzPerS, familyById, makeSnippet, normalise, shuffleFrames } from './signals.js';
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
      renderProbe();
      break;
    case 'wu-progress':
      $('wu-progress').value = msg.epoch / msg.epochs;
      unitStatus(`Unit ${msg.unit}: epoch ${msg.epoch} of ${msg.epochs}, training loss ${fmt(msg.loss, 3)}`);
      break;
    case 'wu-result':
      onWorkUnitResult(msg.card);
      break;
    case 'wu-stopped':
      setRunning(null);
      unitStatus(`Unit ${msg.unit} stopped.`);
      break;
    case 'token-lm-progress':
      $('tb-lm-status').textContent = `Training on the ${STREAM_NAMES[msg.stream]} stream: epoch ${msg.epoch} of ${msg.epochs}…`;
      break;
    case 'token-lm-result':
      setRunning(null);
      state.tokenLm = msg.stopped ? null : { ...msg, analysis: state.tokenLmFor };
      $('tb-lm-status').textContent = msg.stopped ? 'Stopped.' : 'Done. Lower is better; compare the real stream with its surrogates.';
      renderTokenModel();
      break;
    case 'error':
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

function selectFamily(id) {
  state.family = id;
  renderFamilies();
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
  engine.postMessage({ type: 'train', config });
}

// ---------- Track B ----------

function analyzeTokens() {
  if (!familyAvailable(state.family)) return;
  const id = ++analyzeId;
  $('tb-status').textContent = 'Tokenizing…';
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
    return;
  }
  state.trackb = msg;
  if (state.tokenLm?.analysis !== msg) {
    state.tokenLm = null; // the token model belongs to the previous token stream
    if (state.running !== 'tokens') $('tb-lm-status').textContent = '';
  }
  $('tb-status').textContent = `${familyById[msg.family].label}: ${msg.nTokens.toLocaleString()} tokens, ${msg.used} of ${msg.k} token types used. First tokens: ${msg.preview.map((t) => t.toString(36)).join(' ')}`;
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
  fillTable(
    $('tb-entropy-table'),
    ['Context', ...r.crossEntropy.real.map((e) => e.context)],
    STREAMS.map((s) => [s.name, ...r.crossEntropy[s.id].map((e) => fmt(e.bits))]),
  );
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
  engine.postMessage({ type: 'token-lm', k: r.k, streams: r.streams, centroids: r.centroids, seed: Number($('seed').value) || 0 });
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
  fillTable(
    $('tb-lm-table'),
    ['Stream (held out, bits per token)', 'Token GRU', 'n-gram, 3 tokens of context', 'No context', 'Best epoch'],
    STREAMS.filter((s) => m.results[s.id]).map((s) => [
      s.name,
      fmt(m.results[s.id].bits),
      fmt(r.crossEntropy[s.id][3].bits),
      fmt(r.crossEntropy[s.id][0].bits),
      `${m.results[s.id].bestEpoch} of ${m.epochs}`,
    ]),
  );
  if (m.sample) {
    const { real, generated, rows } = m.sample;
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
  $('wu-submit').href = issueUrl(card);
  state.results = [card, ...state.results].slice(0, 200);
  store.set('results', state.results);
  renderHistory();
  if ($('wu-loop').checked && !state.stopLoop) setTimeout(() => !state.running && runUnit(randomUnit()), 1000);
}

function issueUrl(card) {
  const title = `Work unit result: ${card.spec} #${card.unit}`;
  const body = `Result card from the EI@Home Live Lab.\n\n\`\`\`json\n${JSON.stringify(card, null, 2)}\n\`\`\`\n`;
  return `${REPO}/issues/new?${new URLSearchParams({ title, body, labels: 'work-unit' })}`;
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
    return;
  }
  const results = Array.isArray(store.results) ? store.results : [];
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
}

function showTab(name) {
  for (const b of document.querySelectorAll('.tabs button')) b.setAttribute('aria-selected', String(b.dataset.tab === name));
  for (const s of document.querySelectorAll('.tab')) s.hidden = s.id !== `tab-${name}`;
  if (name === 'lab') redraw();
}

// ---------- wiring ----------

function init() {
  if (PREVIEW) {
    for (const id of ['theme', 'wu-download', 'wu-download-all']) $(id).hidden = true;
  }
  applyTheme();
  $('backend').value = store.get('backend', 'auto');
  $('wu-spec').textContent = SPEC;
  $('wu-number').max = MAX_UNIT;
  $('rosetta-text').value = state.text || ROSETTA_TEXT;

  startTokens();
  startEngine();
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
  $('wu-copy').onclick = () => {
    navigator.clipboard.writeText(JSON.stringify(state.card, null, 2)).then(
      () => unitStatus('Result card copied.'),
      () => {
        getSelection().selectAllChildren($('wu-card'));
        unitStatus('Your browser blocked copying, so the card is selected: press Ctrl+C or ⌘C.');
      },
    );
  };
  $('wu-download-all').onclick = () => download('eiathome-results.json', state.results);
  // Two-step confirmation on the page itself (some hosts block confirm() dialogs).
  let clearTimer;
  $('wu-clear').onclick = () => {
    const button = $('wu-clear');
    if (!button.dataset.armed) {
      button.dataset.armed = 'yes';
      button.textContent = 'Click again to delete all';
      clearTimer = setTimeout(() => {
        delete button.dataset.armed;
        button.textContent = 'Clear';
      }, 4000);
      return;
    }
    clearTimeout(clearTimer);
    delete button.dataset.armed;
    button.textContent = 'Clear';
    state.results = [];
    store.set('results', []);
    renderHistory();
    unitStatus('Deleted the results stored in this browser.');
  };

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
