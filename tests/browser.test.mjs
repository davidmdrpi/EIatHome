// End-to-end checks in headless Chromium: the Live Lab served as a static site.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import { extname, join, normalize } from 'node:path';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { lineSlice, writeSliceFolder } from './fixtures.mjs';

const { chromium } = createRequire(import.meta.url)('playwright');
const WEB = fileURLToPath(new URL('../web/', import.meta.url));
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm', '.bin': 'application/octet-stream' };
const ATTACK = '<img src=x onerror="window.__pwned=1">';

let server;
let browser;
let base;

before(async () => {
  server = createServer(async (req, res) => {
    const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^(\.\.[/\\])+/, '');
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

async function openLab() {
  const page = await browser.newPage();
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
  assert.deepEqual(errors, []);
  await page.close();
});

test('work units evaluate the frozen corpus, whatever custom text is set', async () => {
  const { page, errors } = await openLab();
  const before = await runUnit(page, 0);

  await page.click('.tabs button[data-tab="data"]');
  await page.fill('#rosetta-text', 'The quick brown fox jumps over the lazy dog. '.repeat(60));
  await page.click('#rosetta-apply');
  const afterCustomText = await runUnit(page, 0);

  assert.equal(before.version, 2);
  assert.deepEqual(before.inputs.rosetta_corpus, afterCustomText.inputs.rosetta_corpus);
  assert.deepEqual(before.metrics, afterCustomText.metrics);
  assert.ok(Number.isFinite(before.metrics.drift_probe.untrained_r2_synthetic), 'reports the untrained-GRU control');
  assert.deepEqual(errors, []);
  await page.close();
});
