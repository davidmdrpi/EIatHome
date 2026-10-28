// End-to-end checks in headless Chromium: the Live Lab served as a static site.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import { extname, join, normalize } from 'node:path';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { MAX_ISSUE_URL, extractCard, submissionText, validateCard } from '../web/js/community.js';
import { matchedSize, matchedTokenWidth, parameterCount, tokenParameterCount } from '../web/js/models.js';
import { parseRecoveryWindows, recoveryCard, runRecoveryUnit, verifyRecoveryCard } from '../web/js/recovery.js';
import { RESULT_CARD_VERSION } from '../web/js/workunits.js';
import { describeDataset } from '../web/js/slices.js';
import { sessions } from '../web/js/splits.js';
import { lineSlice, makeCard, publishedSlices, writeSliceFolder } from './fixtures.mjs';

const { chromium } = createRequire(import.meta.url)('playwright');
const WEB = fileURLToPath(new URL('../web/', import.meta.url));
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm', '.bin': 'application/octet-stream' };
const ATTACK = '<img src=x onerror="window.__pwned=1">';

let server;
let browser;
let base;
const overrides = new Map(); // path -> body, to serve fixture data in place of the real file
const holds = new Map(); // path -> promise the server waits for before answering, to delay a file

before(async () => {
  server = createServer(async (req, res) => {
    const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^(\.\.[/\\])+/, '');
    await holds.get(path);
    if (overrides.has(path)) {
      res.writeHead(200, { 'content-type': TYPES[extname(path)] ?? 'application/json' });
      res.end(overrides.get(path));
      return;
    }
    const file = join(WEB, path.endsWith('/') ? `${path}index.html` : path);
    try {
      const body = await readFile(file);
      res.writeHead(200, { 'content-type': TYPES[extname(file)] ?? 'application/octet-stream' });
      res.end(body);
    } catch {
      res.writeHead(404);
      res.end();
    }
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${server.address().port}/`;
  browser = await chromium.launch(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {});
});

after(async () => {
  await browser?.close();
  server?.close();
});

async function openLab(init) {
  const page = await browser.newPage();
  if (init) await page.addInitScript(init);
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(base);
  await page.waitForFunction(() => document.getElementById('backend-status').textContent.startsWith('Computing'), null, { timeout: 60000 });
  return { page, errors };
}

async function loadSlices(page, slices) {
  await page.click('.tabs button[data-tab="data"]');
  await page.setInputFiles('#slice-files', writeSliceFolder(slices));
  await page.waitForFunction(() => /loaded/.test(document.getElementById('slice-status').textContent), null, { timeout: 10000 });
  await page.click('.tabs button[data-tab="lab"]');
}

async function train(page, family, epochs = '1') {
  await page.click(`#families button[data-family="${family}"]`);
  await page.fill('#epochs', epochs);
  await page.selectOption('#ntrain', '400');
  await page.click('#train');
  await page.waitForFunction(() => /^(Done|Stopped)/.test(document.getElementById('lab-status').textContent), null, { timeout: 120000 });
}

async function runUnit(page, unit) {
  await page.click('.tabs button[data-tab="units"]');
  await page.fill('#wu-number', String(unit));
  await page.click('#wu-run');
  await page.waitForFunction((u) => document.getElementById('wu-status').textContent.startsWith(`Unit ${u} finished`), unit, { timeout: 180000 });
  const card = JSON.parse(await page.textContent('#wu-card'));
  await page.evaluate(() => (document.getElementById('wu-status').textContent = ''));
  return card;
}

test('HTML in slice metadata is displayed literally, never run', async () => {
  const { page, errors } = await openLab();
  await loadSlices(page, [{ meta: { id: ATTACK, group: '<b>carrier</b>' }, bytes: lineSlice(2.4) }]);

  await page.click('.tabs button[data-tab="data"]');
  assert.equal(await page.textContent('#slice-table tbody td:first-child'), ATTACK);
  assert.equal(await page.locator('#slice-table img, #slice-table b').count(), 0);

  await page.click('.tabs button[data-tab="lab"]');
  await train(page, 'tone'); // trains on tones; the drift probe then reads out the Voyager groups
  assert.equal(await page.textContent('#probe tbody td:first-child'), '<b>carrier</b>');
  assert.equal(await page.locator('#probe img, #probe td b').count(), 0);
  assert.equal(await page.evaluate(() => window.__pwned), undefined);
  assert.deepEqual(errors, []);
  await page.close();
});

test('every Track A model trains, at the parameter count the size matching promises', async () => {
  const { page, errors } = await openLab();
  for (const arch of ['gru', 'lstm', 'attention']) {
    await page.selectOption('#arch', arch);
    await train(page, 'tone');
    const expected = parameterCount(arch, matchedSize(arch, 32)).toLocaleString('en-US');
    assert.match(await page.textContent('#lab-status'), new RegExp(`${expected} parameters`), arch);
    assert.ok(Number.isFinite(Number(await page.textContent('#gains tbody tr.current td:nth-child(4)'))), `${arch} reports a gain`);
  }
  assert.deepEqual(errors, []);
  await page.close();
});

test('both token models find the structure in Rosetta text but not in shuffled tokens', async () => {
  const { page, errors } = await openLab();
  await page.click('#families button[data-family="rosetta"]');
  await page.waitForFunction(() => document.getElementById('tb-status').textContent.startsWith('Rosetta'), null, { timeout: 60000 });
  const k = Number(await page.inputValue('#k'));
  const window = 32;
  const sizes = { gru: 32, transformer: matchedTokenWidth(k, window) };
  for (const [arch, label] of [['gru', 'GRU'], ['transformer', 'Transformer']]) {
    await page.selectOption('#tb-lm-arch', arch);
    await page.click('#tb-lm');
    await page.waitForFunction(() => /^Done/.test(document.getElementById('tb-lm-status').textContent) && !document.getElementById('tb-lm').disabled, null, { timeout: 240000 });
    const parameters = tokenParameterCount(arch, sizes[arch], k, window).toLocaleString('en-US');
    assert.match(await page.textContent('#tb-lm-status'), new RegExp(`^Done: ${label}, ${parameters} parameters`), 'TF.js counts what the size matching promises');
    const rows = await page.$$eval('#tb-lm-table tbody tr', (trs) => trs.map((tr) => [...tr.cells].map((td) => td.textContent)));
    const bits = Object.fromEntries(rows.map(([name, model, , noContext]) => [name, { model: Number(model), noContext: Number(noContext) }]));
    assert.ok(bits.real.model < bits.shuffled.model - 0.5, `${arch}: ${JSON.stringify(bits)}`);
    assert.ok(Math.abs(bits.shuffled.model - bits.shuffled.noContext) < 0.3, `${arch}: early stopping keeps shuffled near its no-context entropy`);
    for (const [name, b] of Object.entries(bits)) assert.ok(b.model > 0.5, `${arch} ${name}: a model that could see the next token would score near 0 bits`);
    assert.equal(await page.isVisible('#tb-lm-sample'), true);
    // The GRU continues for 32 tokens; the transformer stops when its input would exceed its 32 trained positions.
    const generated = arch === 'gru' ? 32 : window + 1 - 16;
    assert.equal(await page.textContent('#tb-lm-generated-caption'), `The model’s ${generated}-token continuation (sampled)`);
  }
  assert.deepEqual(errors, []);
  await page.close();
});

const waitForPublished = (page) =>
  page.waitForFunction(() => /loaded from this site/.test(document.getElementById('slice-status').textContent), null, { timeout: 30000 });

test('the published slices load at startup, and the gains table says whether each real family was held out', async () => {
  const published = await publishedSlices();
  const { page, errors } = await openLab();
  await waitForPublished(page);
  assert.equal(await page.locator('#slice-table tbody tr').count(), published.length);

  for (const family of ['voyager', 'offsource']) {
    const own = published.filter((s) => s.family === family);
    assert.ok(own.length, `${family} slices are published`);
    assert.equal(await page.isDisabled(`#families button[data-family="${family}"]`), false);
    await train(page, family);
    const label = await page.textContent('#gains tbody tr.current td:first-child');
    if (sessions(own).length < 2) assert.match(label, /not held out/);
    else assert.doesNotMatch(label, /not held out/);
    const rows = await page.$$eval('#gains tbody tr td:first-child', (tds) => tds.map((td) => td.textContent));
    assert.ok(rows.some((r) => r.startsWith('Voyager 1')) && rows.some((r) => r.startsWith('Off-source')), 'both real families are evaluated');
  }
  assert.deepEqual(errors, []);
  await page.close();
});

test('Track B finds held-out structure in Voyager data but not in off-source noise', async () => {
  const { page, errors } = await openLab();
  await waitForPublished(page);
  const entropyTable = async (label) => {
    await page.waitForFunction((l) => document.getElementById('tb-status').textContent.startsWith(l), label, { timeout: 120000 });
    const rows = await page.$$eval('#tb-entropy-table tbody tr', (trs) => trs.map((tr) => [...tr.cells].map((td) => td.textContent)));
    return Object.fromEntries(rows.map(([name, ...cells]) => [name, cells.slice(0, 4).map(Number)]));
  };

  await page.click('#families button[data-family="voyager"]');
  const voyager = await entropyTable('Voyager');
  assert.match(await page.textContent('#tb-status'), /whole scans .* 45 test tokens scored/);
  assert.ok(voyager.real[3] < voyager.real[0] - 1, `context predicts Voyager's next token: ${voyager.real}`);
  assert.ok(voyager.shuffled[3] > voyager.shuffled[0] - 0.25, `but not once its order is shuffled: ${voyager.shuffled}`);
  const control = voyager['Off-source (control)'];
  assert.ok(control[3] > control[0], `what Voyager's model learned doesn't transfer to off-source noise: ${control}`);

  await page.click('#tb-lm');
  await page.waitForFunction(() => /^Done/.test(document.getElementById('tb-lm-status').textContent), null, { timeout: 240000 });
  const lm = await page.$$eval('#tb-lm-table tbody tr', (trs) => trs.map((tr) => [...tr.cells].map((td) => td.textContent)));
  const controlRow = lm.find(([name]) => name === 'Off-source (control)');
  assert.ok(controlRow && Number.isFinite(Number(controlRow[1])), 'the token model is also scored on the control');
  assert.equal(controlRow[5], 'real model');

  await page.click('#families button[data-family="offsource"]');
  const offsource = await entropyTable('Off-source');
  assert.ok(offsource.real[3] > offsource.real[0] - 0.25, `no held-out structure in off-source noise: ${offsource.real}`);
  assert.deepEqual(errors, []);
  await page.close();
});

// Runs in a test-only page against the real layers (web/js/layers.js), on TF.js's CPU backend.
const LAYER_CHECKS = `<!doctype html><meta charset="utf-8"><script type="module">
import { LayerNorm, attentionStates, buildTokenModel } from '/js/layers.js';
const tf = globalThis.tf;
await tf.setBackend('cpu');
const maxDiff = (a, b) => tf.tidy(() => tf.max(tf.abs(tf.sub(a, b))).dataSync()[0]);
const out = {};

// Changing tokens from position 20 on must leave the logits at positions 0-19 exactly unchanged.
const k = 16;
const lm = buildTokenModel(k, 7, 'transformer');
const tokens = Array.from({ length: 32 }, (_, i) => (i * 7 + 3) % k);
const changed = tokens.map((t, i) => (i >= 20 ? (t + 5) % k : t));
const logits = (seq) => lm.apply(tf.oneHot(tf.tensor2d(seq, [1, seq.length], 'int32'), k).cast('float32'));
const [a, b] = [logits(tokens), logits(changed)];
out.tokenEarlier = maxDiff(a.slice([0, 0, 0], [1, 20, k]), b.slice([0, 0, 0], [1, 20, k]));
out.tokenLater = maxDiff(a.slice([0, 20, 0], [1, 12, k]), b.slice([0, 20, 0], [1, 12, k]));
// A shorter input gives the same logits as the prefix of a longer one.
out.tokenPrefix = maxDiff(logits(tokens.slice(0, 8)), a.slice([0, 0, 0], [1, 8, k]));

// Track A's attention block on continuous frames: the same causality check.
const input = tf.input({ shape: [null, 64] });
const block = tf.model({ inputs: input, outputs: attentionStates(input, 24, 3) });
const frames = tf.randomNormal([1, 16, 64], 0, 1, 'float32', 5);
const later = tf.concat([frames.slice([0, 0, 0], [1, 10, 64]), tf.randomNormal([1, 6, 64], 0, 1, 'float32', 6)], 1);
const [fa, fb] = [block.apply(frames), block.apply(later)];
out.frameEarlier = maxDiff(fa.slice([0, 0, 0], [1, 10, -1]), fb.slice([0, 0, 0], [1, 10, -1]));
out.frameLater = maxDiff(fa.slice([0, 10, 0], [1, 6, -1]), fb.slice([0, 10, 0], [1, 6, -1]));

// The custom layer norm against TF.js's own, with a non-default scale and offset.
const x = tf.randomNormal([4, 7, 22], 0, 3, 'float32', 1);
const ours = new LayerNorm({});
const theirs = tf.layers.layerNormalization();
ours.apply(x);
theirs.apply(x);
const weights = [tf.randomNormal([22], 1, 0.5, 'float32', 2), tf.randomNormal([22], 0, 0.5, 'float32', 3)];
ours.setWeights(weights);
theirs.setWeights(weights);
out.normOutput = maxDiff(ours.apply(x), theirs.apply(x));
const loss = (layer) => () => layer.apply(x).square().sum();
const inputGrad = (layer) => tf.grad((v) => layer.apply(v).square().sum())(x);
out.normInputGradient = maxDiff(inputGrad(ours), inputGrad(theirs));
const paramGrads = (layer) => tf.variableGrads(loss(layer), layer.trainableWeights.map((w) => w.val)).grads;
const [g1, g2] = [paramGrads(ours), paramGrads(theirs)];
out.normParameterGradient = Math.max(...ours.trainableWeights.map((w, i) => maxDiff(g1[w.val.name], g2[theirs.trainableWeights[i].val.name])));
out.epsilon = theirs.epsilon;
window.__layerChecks = out;
</script>`;

// Track A's next-frame model as the engine builds it (tf.layers.gru then a dense layer, as in
// buildModel), with the reference weights of tests/track_a_gru.json: TF.js must reproduce the
// reference outputs that the Python port (tracks/recovery/learned.py) is also checked against.
const GRU_CHECK = `<!doctype html><meta charset="utf-8"><script type="module">
import '/vendor/tfjs/tf.min.js';
const tf = globalThis.tf;
await tf.setBackend('cpu');
const f = await (await fetch('/__test/track_a_gru.json')).json();
const maxDiff = (a, b) => tf.tidy(() => tf.max(tf.abs(tf.sub(a, tf.tensor(b)))).dataSync()[0]);
const build = (channels, units) => {
  const input = tf.input({ shape: [null, channels] });
  const gru = tf.layers.gru({ units, returnSequences: true });
  const dense = tf.layers.dense({ units: channels });
  const states = gru.apply(input);
  return { gru, dense, model: tf.model({ inputs: input, outputs: [states, dense.apply(states)] }) };
};
const { gru, dense, model } = build(f.channels, f.units);
gru.setWeights([tf.tensor2d(f.kernel), tf.tensor2d(f.recurrent_kernel), tf.tensor1d(f.bias)]);
dense.setWeights([tf.tensor2d(f.dense_kernel), tf.tensor1d(f.dense_bias)]);
const [states, prediction] = model.predict(tf.tensor3d(f.input));
window.__gruCheck = {
  states: maxDiff(states, f.states),
  prediction: maxDiff(prediction, f.prediction),
  parameters: build(64, 64).model.countParams(),
};
</script>`;

test('Track A’s GRU in TF.js reproduces the reference that the Python port is checked against', async () => {
  overrides.set('/__test/gru.html', GRU_CHECK);
  overrides.set('/__test/track_a_gru.json', await readFile(new URL('./track_a_gru.json', import.meta.url), 'utf8'));
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  try {
    await page.goto(`${base}__test/gru.html`);
    await page.waitForFunction(() => window.__gruCheck, null, { timeout: 60000 });
    const c = await page.evaluate(() => window.__gruCheck);
    assert.ok(c.states < 1e-5, `GRU states: ${c.states}`);
    assert.ok(c.prediction < 1e-5, `next-frame predictions: ${c.prediction}`);
    assert.equal(c.parameters, parameterCount('gru', 64), 'one bias per gate, as models.js counts');
    assert.deepEqual(errors, []);
  } finally {
    overrides.delete('/__test/gru.html');
    overrides.delete('/__test/track_a_gru.json');
    await page.close();
  }
});

test('attention is causal, and the custom layer norm matches TF.js’s', async () => {
  overrides.set('/__test/layers.html', LAYER_CHECKS);
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  try {
    await page.goto(`${base}__test/layers.html`);
    await page.waitForFunction(() => window.__layerChecks, null, { timeout: 60000 });
    const c = await page.evaluate(() => window.__layerChecks);
    assert.equal(c.tokenEarlier, 0, 'later tokens never change earlier logits (token transformer)');
    assert.ok(c.tokenLater > 0, 'and the check would see a change');
    assert.ok(c.tokenPrefix < 1e-6, `a prefix pass matches the full pass: ${c.tokenPrefix}`);
    assert.equal(c.frameEarlier, 0, 'later frames never change earlier states (Track A attention)');
    assert.ok(c.frameLater > 0);
    assert.equal(c.epsilon, 1e-3, 'the custom layer norm uses TF.js’s epsilon');
    assert.ok(c.normOutput < 1e-5, `layer-norm outputs: ${c.normOutput}`);
    assert.ok(c.normInputGradient < 1e-4, `layer-norm input gradients: ${c.normInputGradient}`);
    assert.ok(c.normParameterGradient < 1e-4, `layer-norm scale and offset gradients: ${c.normParameterGradient}`);
    assert.deepEqual(errors, []);
  } finally {
    overrides.delete('/__test/layers.html');
    await page.close();
  }
});

test('token-model training can be stopped, and the controls come back', async () => {
  const { page, errors } = await openLab();
  await page.click('#families button[data-family="rosetta"]');
  await page.waitForFunction(() => document.getElementById('tb-status').textContent.startsWith('Rosetta'), null, { timeout: 60000 });
  await page.click('#tb-lm');
  await page.waitForFunction(() => /^Training on/.test(document.getElementById('tb-lm-status').textContent), null, { timeout: 60000 });
  assert.equal(await page.isDisabled('#tb-lm-stop'), false, 'Stop is available while the token model trains');
  assert.equal(await page.isDisabled('#train'), true);

  await page.click('#tb-lm-stop');
  await page.waitForFunction(() => document.getElementById('tb-lm-status').textContent === 'Stopped.', null, { timeout: 60000 });
  for (const id of ['#train', '#wu-run', '#tb-lm', '#backend']) assert.equal(await page.isDisabled(id), false, `${id} is enabled again`);
  for (const id of ['#tb-lm-stop', '#stop', '#wu-stop']) assert.equal(await page.isDisabled(id), true, `${id} is disabled again`);
  assert.equal(await page.isVisible('#tb-lm-sample'), false, 'a stopped run shows no results');
  assert.deepEqual(errors, []);
  await page.close();
});

test('training on the only Voyager observation is labelled as reconstruction', async () => {
  const { page, errors } = await openLab();
  await loadSlices(page, [
    { meta: { id: 'v-carrier', group: 'carrier' }, bytes: lineSlice(2.4, 1) },
    { meta: { id: 'v-sideband', group: 'sideband' }, bytes: lineSlice(2.5, 2) },
  ]);
  await train(page, 'voyager');
  assert.match(await page.textContent('#gains tbody tr.current td:first-child'), /not held out/);
  assert.equal(await page.isVisible('#gains-note'), true);
  assert.match(await page.textContent('#lab-status'), /reconstructs data it trained on/);

  // One scan can't be split for Track B, and the previous family's tokens must not be reused.
  await page.waitForFunction(() => /at least 3 scans/.test(document.getElementById('tb-status').textContent), null, { timeout: 30000 });
  await page.click('#tb-lm');
  assert.equal(await page.textContent('#tb-lm-status'), 'Analyze tokens first.');
  assert.deepEqual(errors, []);
  await page.close();
});

test('reordering the manifest (and renaming its files) changes neither inputs nor results', async () => {
  const slices = [
    { meta: { id: 'v-a', group: 'carrier', tstart: 57650.78 }, bytes: lineSlice(2.4, 1) },
    { meta: { id: 'v-b', group: 'sideband', tstart: 57652.9 }, bytes: lineSlice(2.6, 2) },
  ];
  const cards = [];
  for (const order of [slices, [...slices].reverse()]) {
    const { page, errors } = await openLab();
    await loadSlices(page, order); // files are named by position, so reversing also renames them
    cards.push(await runUnit(page, 0));
    assert.deepEqual(errors, []);
    await page.close();
  }
  assert.ok(cards[0].inputs.slices.dataset_id);
  assert.deepEqual(cards[1].inputs.slices, cards[0].inputs.slices);
  assert.deepEqual(cards[1].metrics, cards[0].metrics);
});

test('work units evaluate the frozen corpus, whatever custom text is set', async () => {
  const { page, errors } = await openLab();
  const before = await runUnit(page, 0);

  await page.click('.tabs button[data-tab="data"]');
  await page.fill('#rosetta-text', 'The quick brown fox jumps over the lazy dog. '.repeat(60));
  await page.click('#rosetta-apply');
  const afterCustomText = await runUnit(page, 0);

  assert.equal(before.version, RESULT_CARD_VERSION);
  assert.deepEqual(before.inputs.rosetta_corpus, afterCustomText.inputs.rosetta_corpus);
  assert.deepEqual(before.metrics, afterCustomText.metrics);
  assert.ok(Number.isFinite(before.metrics.drift_probe.untrained_r2_synthetic), 'reports the untrained-GRU control');
  assert.deepEqual(errors, []);
  await page.close();
});

test('a card from a real work unit passes the checks the ingestion workflow runs', async () => {
  const { page, errors } = await openLab();
  await waitForPublished(page);
  const card = await runUnit(page, 2);
  const published = await describeDataset(await publishedSlices());
  assert.equal(card.inputs.slices.dataset_id, published.datasetId);
  assert.ok(published.families.every((f) => card.families.includes(f)), 'the card covers every published family');
  assert.deepEqual(validateCard(card, { published }), []);

  // Submit: the link fits GitHub's URL limit and its issue body holds this card. Clicking it
  // opens the issue (stubbed here) and puts the card on the clipboard too.
  const href = await page.getAttribute('#wu-submit', 'href');
  assert.ok(href.length <= MAX_ISSUE_URL, `${href.length} characters`);
  assert.deepEqual(extractCard(new URL(href).searchParams.get('body')), card);
  await page.context().route('https://github.com/**', (route) => route.fulfill({ status: 200, contentType: 'text/html', body: 'stub' }));
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write'], { origin: base.slice(0, -1) });
  const [issue] = await Promise.all([page.waitForEvent('popup'), page.click('#wu-submit')]);
  assert.equal(issue.url(), href);
  await page.waitForFunction(() => /^Opened a GitHub issue/.test(document.getElementById('wu-status').textContent), null, { timeout: 10000 });
  assert.match(await page.textContent('#wu-status'), /also on your clipboard/);
  assert.equal(await page.evaluate(() => navigator.clipboard.readText()), submissionText(card));
  await issue.close();
  assert.deepEqual(errors, []);
  await page.close();
});

test('a card too long for a link is put on the clipboard, for an issue with a json block to paste into', async () => {
  const { page, errors } = await openLab();
  // Forty slices with long ids: the card lists every id, which makes it too long for a link.
  await loadSlices(page, Array.from({ length: 40 }, (_, i) => ({
    meta: { id: `voyager-session-${String(i).padStart(2, '0')}-${'carrier-window-'.repeat(7)}`, group: i % 3 ? 'sideband' : 'carrier', tstart: 57650.78 + Math.floor(i / 3) * 0.004 },
    bytes: lineSlice(2.4 + (i % 5) * 0.1, i + 1),
  })));
  const card = await runUnit(page, 0);
  assert.equal(card.inputs.slices.slices.length, 40);
  const href = await page.getAttribute('#wu-submit', 'href');
  assert.ok(href.length <= MAX_ISSUE_URL, `${href.length} characters`);
  const body = new URL(href).searchParams.get('body');
  assert.ok(body.includes('```json\n\n```') && !body.includes(card.inputs.slices.dataset_id), 'an empty json block, not the card');
  await page.context().route('https://github.com/**', (route) => route.fulfill({ status: 200, contentType: 'text/html', body: 'stub' }));
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write'], { origin: base.slice(0, -1) });
  const [issue] = await Promise.all([page.waitForEvent('popup'), page.click('#wu-submit')]);
  await page.waitForFunction(() => /too long to fill in automatically/.test(document.getElementById('wu-status').textContent), null, { timeout: 10000 });
  assert.match(await page.textContent('#wu-status'), /on your clipboard: paste it into the empty json block/);
  const clipboard = await page.evaluate(() => navigator.clipboard.readText());
  assert.equal(clipboard, submissionText(card), 'the compact submission text, not the pretty-printed card');
  assert.deepEqual(extractCard(body.replace('```json\n\n```', `\`\`\`json\n${clipboard}\n\`\`\``)), card, 'pasting the clipboard fills the issue with the card');
  await issue.close();
  assert.equal(await page.isVisible('#wu-submit-text'), false);

  // If the browser blocks copying, the same text is shown and selected, to copy by hand.
  await page.evaluate(() => (navigator.clipboard.writeText = () => Promise.reject(new Error('blocked'))));
  const [again] = await Promise.all([page.waitForEvent('popup'), page.click('#wu-submit')]);
  await page.waitForFunction(() => /blocked copying/.test(document.getElementById('wu-status').textContent), null, { timeout: 10000 });
  assert.equal(await page.isVisible('#wu-submit-text'), true);
  assert.equal(await page.inputValue('#wu-submit-text'), submissionText(card));
  assert.deepEqual(await page.$eval('#wu-submit-text', (box) => [box.selectionStart, box.selectionEnd, box.value.length]).then(([a, b, n]) => [a, b === n]), [0, true], 'all of it is selected');
  await again.close();
  assert.deepEqual(errors, []);
  await page.close();
});

test('community results are summarized, with volunteer names shown as text', async () => {
  const run = async (issue, by, unit, shift = 0) => ({ issue, submitted_by: by, submitted_at: '2026-10-26T01:00:00Z', card: await makeCard(unit, { shift }) });
  const results = [await run(21, 'alice', 4), await run(22, 'bob', 4, 0.01), await run(23, '<b>eve</b>', 5)];
  overrides.set('/data/community_results.json', JSON.stringify({ format: 'eiathome-community-results', version: 1, results }));
  try {
    const { page, errors } = await openLab();
    await page.click('.tabs button[data-tab="units"]');
    await page.waitForFunction(() => !document.getElementById('community-tables').hidden, null, { timeout: 10000 });
    assert.match(await page.textContent('#community-status'), /3 result cards for 2 units from 3 volunteers\. 1 of 2 units replicated/);
    const units = await page.$$eval('#community-units tbody tr', (trs) => trs.map((tr) => [...tr.cells].map((td) => td.textContent)));
    assert.deepEqual(units.map((u) => [u[0], u[6]]), [['4', 'Replicated'], ['5', 'One volunteer so far']]);
    const names = await page.$$eval('#community-volunteers tbody tr td:first-child', (tds) => tds.map((td) => td.textContent));
    assert.ok(names.includes('<b>eve</b>'));
    assert.equal(await page.locator('#community-volunteers b').count(), 0);
    assert.deepEqual(errors, []);
    await page.close();
  } finally {
    overrides.clear();
  }
});

async function recoveryWindows() {
  const bytes = await readFile(join(WEB, 'data/recovery/test.i8'));
  return parseRecoveryWindows(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

async function runRecovery(page, unit) {
  await page.click('.tabs button[data-tab="recovery"]');
  await page.waitForFunction(() => /^Ready/.test(document.getElementById('rc-status').textContent), null, { timeout: 30000 });
  await page.fill('#rc-number', String(unit));
  await page.click('#rc-run');
  await page.waitForFunction((u) => document.getElementById('rc-status').textContent.startsWith(`Unit ${u} finished`), unit, { timeout: 30000 });
  return JSON.parse(await page.textContent('#rc-card'));
}

test('a recovery unit runs in the browser, and its card passes the checks ingestion runs, recomputation included', async () => {
  const { page, errors } = await openLab();
  const card = await runRecovery(page, 12);
  assert.deepEqual(validateCard(card), []);
  assert.deepEqual(verifyRecoveryCard(card, await recoveryWindows()), [], 'Node recomputes the same counts');
  assert.match(await page.textContent('#rc-summary'), /^Unit 12 tested whether two standard detectors find weak signals in real data/);
  assert.equal(await page.locator('#rc-unit-charts canvas').count(), 7, 'one chart per class');
  const href = await page.getAttribute('#rc-submit', 'href');
  assert.ok(href.length <= MAX_ISSUE_URL, `${href.length} characters`);
  assert.deepEqual(extractCard(new URL(href).searchParams.get('body')), card);

  await page.selectOption('#rc-source', 'browser');
  assert.match(await page.textContent('#rc-combined-status'), /^1 unit run in this browser: 18 injections/);
  await page.selectOption('#rc-rate', '0.001');
  assert.match(await page.textContent('#rc-summary'), /flag 0\.1% of empty windows/);
  await page.selectOption('#rc-source', 'community');
  assert.match(await page.textContent('#rc-combined-status'), /No recovery units submitted yet/);
  assert.deepEqual(errors, []);
  await page.close();
});

test('community recovery units are combined once per unit, apart from the replicated units', async () => {
  const data = await recoveryWindows();
  const run = (issue, by, unit) => ({ issue, submitted_by: by, submitted_at: '2026-10-27T01:00:00Z', card: recoveryCard(unit, runRecoveryUnit(unit, data).results, { seconds: 0.6, started: '2026-10-27T00:00:00.000Z' }) });
  const results = [run(31, 'alice', 3), run(32, 'bob', 3), run(33, '<b>eve</b>', 8)];
  overrides.set('/data/community_results.json', JSON.stringify({ format: 'eiathome-community-results', version: 1, results }));
  try {
    const { page, errors } = await openLab();
    // Load the community results while the tab is hidden: the status line is then already set
    // when the tab opens, so wait for the table itself, drawn once the tab shows.
    await page.waitForFunction(() => document.getElementById('community-status').textContent !== '', null, { timeout: 30000 });
    await page.click('.tabs button[data-tab="recovery"]');
    await page.waitForFunction(() => !document.getElementById('tab-recovery').hidden && document.querySelectorAll('#rc-combined-table tbody tr').length === 4, null, { timeout: 30000 });
    assert.match(await page.textContent('#rc-combined-status'), /^2 units from 3 volunteers, each recomputed when submitted: 36 injections of each class at each strength/);
    const found = await page.$$eval('#rc-combined-table tbody tr', (trs) => trs.map((tr) => [...tr.cells].map((td) => td.textContent)));
    assert.match(found[0][1], / of 536: /, 'false alarms among both units’ empty windows');
    await page.waitForFunction(() => /flags \d+ of the 1,024 windows/.test(document.getElementById('rc-empty-note').textContent), null, { timeout: 30000 });
    await page.click('.tabs button[data-tab="units"]');
    assert.match(await page.textContent('#community-status'), /No community results yet/, 'recovery cards are not replicated units');
    assert.deepEqual(errors, []);
    await page.close();
  } finally {
    overrides.clear();
  }
});

const viewportState = (page, host) =>
  page.$eval(`#${host} canvas`, (c) => ({
    center: c.dataset.center.split(',').map(Number),
    fov: Number(c.dataset.fov),
    beacons: Object.fromEntries((c.dataset.beacons ?? '').split(';').filter(Boolean).map((b) => {
      const [id, x, y] = b.split(' ');
      return [id, [Number(x), Number(y)]];
    })),
  }));

test('the observatory viewport slews to a signal on the sky, and clicking a beacon selects its signal', async () => {
  const { page, errors } = await openLab();
  await waitForPublished(page);
  await page.click('#families button[data-family="voyager"]');
  // Voyager 1's 2020 pointing, from the manifest's sky block.
  await page.waitForFunction(() => {
    const [ra, dec] = document.querySelector('#lab-viewport canvas').dataset.center.split(',').map(Number);
    return Math.abs(ra - 258.1687) < 0.01 && Math.abs(dec - 12.4038) < 0.01;
  }, null, { timeout: 15000 });
  assert.match(await page.textContent('#lab-viewport .hud-target'), /Voyager 1 · 2020-07-16[\s\S]*Ophiuchus[\s\S]*≈150 AU from the Sun/);

  // Zoom in until the 1° nod separates the positions, then click the off-source one east of Voyager.
  const canvas = await page.$('#lab-viewport canvas');
  const box = await canvas.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.wheel(0, -1500);
  await page.waitForFunction(() => Number(document.querySelector('#lab-viewport canvas').dataset.fov) < 4, null, { timeout: 10000 });
  await page.waitForFunction(() => /offsource@17\.2795/.test(document.querySelector('#lab-viewport canvas').dataset.beacons), null, { timeout: 10000 });
  const { beacons } = await viewportState(page, 'lab-viewport');
  const [east] = Object.entries(beacons).filter(([id]) => id.startsWith('offsource@17.2795'));
  assert.ok(east, JSON.stringify(beacons));
  assert.ok(Object.keys(beacons).some((id) => id.startsWith('voyager@17.1678')), 'the 2016 pointing is in view too');
  await page.mouse.click(box.x + east[1][0], box.y + east[1][1]);
  await page.waitForFunction(() => document.querySelector('#families button[data-family="offsource"]').getAttribute('aria-checked') === 'true', null, { timeout: 10000 });
  assert.match(await page.textContent('#lab-viewport .hud-target'), /Off-source · 2020-07-16[\s\S]*Empty sky, 1\.0° east of Voyager 1/);
  assert.deepEqual(errors, []);
  await page.close();
});

test('a recovery unit zooms to the telescope beam and colours each of the 1,024 windows by its outcome', async () => {
  const { page, errors } = await openLab();
  await runRecovery(page, 21);
  await page.waitForFunction(() => Number(document.querySelector('#rc-viewport canvas').dataset.fov) < 2.1, null, { timeout: 15000 });
  const { center } = await viewportState(page, 'rc-viewport');
  assert.ok(Math.abs(center[0] - 258.1687) < 0.01 && Math.abs(center[1] - 11.4038) < 0.01, `centred on off-source scan 0016: ${center}`);
  const legend = await page.textContent('#rc-viewport .hud-legend');
  const counts = [...legend.matchAll(/(empty|found|missed|false alarm|decoy flagged|decoy passed) (\d+)/g)].map((m) => Number(m[2]));
  assert.equal(counts.length, 6);
  assert.equal(counts.reduce((a, b) => a + b), 1024);
  assert.match(await page.textContent('#rc-viewport .hud-telemetry'), /Recovery unit 21 · de-Doppler found \d+ of 540 hidden signals/);
  assert.deepEqual(errors, []);
  await page.close();
});

test('observatory mode fills the window and keeps running units until it is closed', async () => {
  const { page, errors } = await openLab();
  await page.click('.tabs button[data-tab="recovery"]');
  await page.waitForFunction(() => /^Ready/.test(document.getElementById('rc-status').textContent), null, { timeout: 30000 });
  await page.click('#rc-viewport .obs-toggle');
  const size = await page.$eval('#rc-viewport .viewport', (el) => [el.getBoundingClientRect().width, el.getBoundingClientRect().height, innerWidth, innerHeight]);
  assert.deepEqual(size.slice(0, 2), size.slice(2), 'fills the window');
  await page.check('#rc-viewport .auto-run input');
  await page.waitForFunction(() => /This session: ([2-9]|\d{2,}) recovery units/.test(document.querySelector('#rc-viewport .hud-stats').textContent), null, { timeout: 30000 });
  assert.equal(await page.isChecked('#rc-loop'), true);
  await page.keyboard.press('Escape');
  assert.equal(await page.$eval('#rc-viewport .viewport', (el) => el.classList.contains('observatory')), false);
  assert.equal(await page.isChecked('#rc-loop'), false, 'leaving observatory mode stops auto-run');
  const done = (await page.textContent('#rc-viewport .hud-stats')).match(/(\d+) recovery units/)[1];
  await page.waitForTimeout(2500);
  assert.equal((await page.textContent('#rc-viewport .hud-stats')).match(/(\d+) recovery units/)[1], done, 'no unit started after it stopped');
  assert.deepEqual(errors, []);
  await page.close();
});

// Holds the recovery worker's results until window.__release() delivers them, so a unit can be
// caught while it runs.
function holdRecoveryResults() {
  const held = [];
  window.__release = () => held.splice(0).forEach((deliver) => deliver());
  window.Worker = class extends window.Worker {
    constructor(url, options) {
      super(url, options);
      this.hold = String(url).includes('recovery.worker');
    }
    get onmessage() {
      return super.onmessage;
    }
    set onmessage(fn) {
      super.onmessage = (e) => (this.hold && e.data?.type === 'result' ? held.push(() => fn(e)) : fn(e));
    }
  };
}

test('clicking another beacon while a recovery unit runs moves the view, not the unit', async () => {
  const { page, errors } = await openLab(holdRecoveryResults);
  await page.click('.tabs button[data-tab="recovery"]');
  await page.waitForFunction(() => /^Ready/.test(document.getElementById('rc-status').textContent), null, { timeout: 30000 });
  await page.fill('#rc-number', '7');
  await page.click('#rc-run');
  // Every window scored and the result held back: the unit is still running.
  await page.waitForFunction(() => /1,024 of 1,024 windows scored/.test(document.querySelector('#rc-viewport .hud-telemetry').textContent), null, { timeout: 30000 });
  assert.equal(await page.$eval('#rc-viewport canvas', (c) => c.dataset.job), 'recovery');

  // Pull back until Barnard's Star is in view, and click it.
  const canvas = await page.$('#rc-viewport canvas');
  const box = await canvas.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.wheel(0, 2500);
  await page.waitForFunction(() => /(^|;)barnard /.test(document.querySelector('#rc-viewport canvas').dataset.beacons), null, { timeout: 10000 });
  const { beacons } = await viewportState(page, 'rc-viewport');
  await page.mouse.click(box.x + beacons.barnard[0], box.y + beacons.barnard[1]);
  await page.waitForFunction(() => document.querySelector('#rc-viewport .hud-title').textContent === "Barnard's Star", null, { timeout: 10000 });
  assert.match(await page.textContent('#rc-status'), /^Unit 7: /, 'still running');
  assert.equal(await page.$eval('#rc-viewport canvas', (c) => c.dataset.job), 'recovery', 'the unit stays drawn on its own windows');
  assert.match(await page.textContent('#rc-viewport .hud-telemetry'), /Data from Recovery windows · 2020-07-16/);

  await page.evaluate(() => window.__release());
  await page.waitForFunction(() => document.getElementById('rc-status').textContent.startsWith('Unit 7 finished'), null, { timeout: 30000 });
  assert.equal(await page.$eval('#rc-viewport canvas', (c) => c.dataset.job), 'recovery');
  assert.match(await page.textContent('#rc-viewport .hud-telemetry'), /Data from Recovery windows · 2020-07-16[\s\S]*Recovery unit 7 · de-Doppler found/);
  assert.equal(await page.textContent('#rc-viewport .hud-title'), "Barnard's Star", 'the view stays where it was taken');
  assert.deepEqual(errors, []);
  await page.close();
});

test('auto-run ticked in observatory mode while the windows are still loading starts once they are ready', async () => {
  let release;
  holds.set('/data/recovery/test.i8', new Promise((resolve) => { release = resolve; }));
  try {
    const { page, errors } = await openLab();
    await page.click('.tabs button[data-tab="recovery"]');
    await page.click('#rc-viewport .obs-toggle');
    await page.check('#rc-viewport .auto-run input');
    await page.waitForTimeout(500);
    assert.match(await page.textContent('#rc-status'), /^Loading the windows/, 'the windows are held back');
    assert.equal(await page.isChecked('#rc-loop'), true);
    release();
    await page.waitForFunction(() => /This session: [1-9]\d* recovery unit/.test(document.querySelector('#rc-viewport .hud-stats').textContent), null, { timeout: 30000 });
    await page.keyboard.press('Escape');
    assert.deepEqual(errors, []);
    await page.close();
  } finally {
    holds.delete('/data/recovery/test.i8');
    release?.();
  }
});
