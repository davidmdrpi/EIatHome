// web/js/recovery.js: the recovery work unit's injections, dealer, detectors and cards. The
// Python port (tracks/recovery/browser.py) is checked against it in tests/test_recovery_browser.py.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { extractCard, issueLink, validateCard } from '../web/js/community.js';
import {
  DETECTORS, KINDS, RECOVERY_DATA, STRENGTHS, combine, dedoppler, deal, design, draw, driftTrials, emptyWindowFlags, energy,
  parseRecoveryWindows, recoveryCard, recoveryConfig, render, runRecoveryUnit, verifyRecoveryCard,
} from '../web/js/recovery.js';
import { makeRng } from '../web/js/rng.js';

const F = 49;
const C = 64;
const MARGIN = recoveryConfig(0).margin;
const file = (name) => readFileSync(new URL(`../web/data/recovery/${name}`, import.meta.url));
const buffer = (bytes) => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
let loaded;
const published = () => (loaded ??= parseRecoveryWindows(buffer(file('test.i8'))));

function noise(n, seed) {
  const rng = makeRng(seed);
  return Float32Array.from({ length: n * F * C }, () => rng.gauss());
}

const card = (unit, results) => recoveryCard(unit, results, { seconds: 0.6, started: '2026-10-27T00:00:00.000Z' });

test('the spec pins the published windows and the thresholds set on the calibration scan', async () => {
  const index = JSON.parse(file('index.json'));
  assert.equal(RECOVERY_DATA.url, `data/recovery/${index.file}`);
  assert.equal(RECOVERY_DATA.sha256, index.sha256);
  assert.deepEqual(RECOVERY_DATA.shape, index.shape);
  assert.equal(RECOVERY_DATA.step, index.step);
  assert.deepEqual(RECOVERY_DATA.thresholds, index.thresholds);
  const data = await published();
  assert.deepEqual([data.n, data.frames, data.channels], index.shape);

  const altered = Uint8Array.from(file('test.i8'));
  altered[1000] ^= 1;
  await assert.rejects(parseRecoveryWindows(altered.buffer), /SHA-256/);
  await assert.rejects(parseRecoveryWindows(altered.buffer.slice(1)), /file size/);
});

test('every class is drawn at its strength, inside the window, from a seeded generator', () => {
  for (const kind of KINDS) {
    for (const strength of STRENGTHS) {
      for (let seed = 0; seed < 10; seed++) {
        const spec = draw(makeRng(seed), kind, strength, F, C);
        assert.deepEqual(draw(makeRng(seed), kind, strength, F, C), spec, 'seeded');
        const s = render(spec, F, C);
        assert.ok(Math.abs(Math.hypot(...s) - strength) < 1e-9, `${kind}: norm is the strength`);
        const frames = Array.from({ length: F }, (_, t) => [...s.subarray(t * C, (t + 1) * C)]);
        const lit = frames.map((row) => row.flatMap((v, c) => (v ? [c] : [])));
        if (kind === 'pulse') {
          assert.deepEqual(lit.map((cs) => cs.length).filter(Boolean), [C], 'one frame, every channel');
          continue;
        }
        assert.ok(lit.flat().every((c) => c >= MARGIN && c <= C - 1 - MARGIN), `${kind}: inside the window's margins`);
        if (kind === 'stationary') assert.ok(lit.every((cs) => cs.join() === lit[0].join()), 'a zero-drift line stays put');
        if (kind === 'intermittent') assert.ok(lit.some((cs) => !cs.length) && lit.some((cs) => cs.length), 'on and off');
        if (kind === 'sidebands' || kind === 'wideband') assert.ok(lit.every((cs) => cs.length >= 3), `${kind}: several channels per frame`);
      }
    }
  }
  assert.ok(render({ kind: 'none', strength: 0 }, F, C).every((v) => v === 0));
});

test('the dealer gives detectors only windows in shuffled order, and the key rebuilds every one', () => {
  const n = KINDS.length * STRENGTHS.length * 2 + 20;
  const backgrounds = noise(n, 1);
  const cells = design(2, n);
  const { windows, key } = deal(backgrounds, n, F, C, cells, 5);
  assert.deepEqual(key.map((k) => k.background).sort((a, b) => a - b), [...Array(n).keys()], 'each background once');
  for (let p = 0; p < n; p++) {
    const s = render(key[p], F, C);
    const rebuilt = Float32Array.from({ length: F * C }, (_, j) => backgrounds[key[p].background * F * C + j] + s[j]);
    assert.deepEqual(windows.subarray(p * F * C, (p + 1) * F * C), rebuilt);
  }
  const tally = (pick) => key.reduce((m, k) => m.set(pick(k), (m.get(pick(k)) ?? 0) + 1), new Map());
  assert.equal(tally((k) => k.kind).get('none'), 20);
  assert.ok([...tally((k) => `${k.kind} ${k.strength}`)].every(([cell, count]) => cell.startsWith('none') || count === 2));
  // The design lists the signals first; dealt, they're spread through the order and the backgrounds.
  assert.ok(key.slice(0, n - 20).some((k) => k.kind === 'none'), 'the dealing order is shuffled');
  assert.ok(key.some((k, p) => k.background !== p), 'so is the background each window gets');
  assert.deepEqual(deal(backgrounds, n, F, C, cells, 5).key, key);
  assert.notDeepEqual(deal(backgrounds, n, F, C, cells, 6).key, key);
});

test('the de-Doppler search equals a brute-force search over straight tracks, without wrapping', () => {
  const n = 4;
  const windows = noise(n, 2);
  const brute = Array.from({ length: n }, (_, w) => {
    let best = -Infinity;
    for (const drift of driftTrials(F)) {
      for (let s = 0; s < C; s++) {
        const channels = Array.from({ length: F }, (_, t) => s + Math.round(drift * t));
        if (channels.some((c) => c < 0 || c >= C)) continue;
        best = Math.max(best, channels.reduce((sum, c, t) => sum + windows[w * F * C + t * C + c], 0));
      }
    }
    return best / Math.sqrt(F);
  });
  dedoppler(windows, n, F, C).forEach((v, w) => assert.ok(Math.abs(v - brute[w]) < 1e-9));
  assert.equal(driftTrials(F).length, 99);
  // A zero-drift tone on whole channels, with no noise, scores its strength.
  const tone = Float32Array.from(render({ kind: 'stationary', strength: 10, drift: 0, start: 20, width: 1 }, F, C));
  assert.ok(Math.abs(dedoppler(tone, 1, F, C)[0] - 10) < 1e-5);
});

test('the energy score is standardized: zero mean and unit spread in Gaussian noise', () => {
  const n = 400;
  const scores = [...energy(noise(n, 3), n, F, C)];
  const mean = scores.reduce((a, b) => a + b) / n;
  const sd = Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1));
  assert.ok(Math.abs(mean) < 0.2 && Math.abs(sd - 1) < 0.15, `mean ${mean}, sd ${sd}`);
});

test('a unit is reproducible and different from other units, and its card passes the checks', async () => {
  const data = await published();
  const { results, key } = runRecoveryUnit(3, data);
  assert.deepEqual(runRecoveryUnit(3, data).results, results);
  assert.notDeepEqual(runRecoveryUnit(4, data).key, key);
  assert.equal(results.signal_free, 1024 - KINDS.length * STRENGTHS.length * 18);
  assert.equal(key.filter((k) => k.kind === 'tone' && k.strength === 10).length, 18);
  const valid = card(3, results);
  assert.deepEqual(validateCard(valid), []);
  assert.deepEqual(verifyRecoveryCard(valid, data), []);
  // De-Doppler, the matched filter for drifting lines, finds every strong tone.
  assert.deepEqual(results.recovery.dedoppler['0.01'].tone.slice(-2), [18, 18]);
});

test('cards that would not reproduce are rejected, by their form or by recomputation', async () => {
  const data = await published();
  const { results } = runRecoveryUnit(3, data);
  const tampered = (change) => {
    const c = structuredClone(card(3, results));
    change(c);
    return c;
  };
  assert.match(validateCard(tampered((c) => (c.config.per_cell = 20))).join(' '), /config/);
  assert.match(validateCard(tampered((c) => (c.config.thresholds.dedoppler['0.01'] = 4))).join(' '), /config/);
  assert.match(validateCard(tampered((c) => (c.inputs.windows_sha256 = '0'.repeat(64)))).join(' '), /background windows/);
  assert.match(validateCard(tampered((c) => (c.results.recovery.energy['0.01'].fsk[2] = 19))).join(' '), /recovery/);
  assert.match(validateCard(tampered((c) => (c.results.both['0.01'].tone[0] = 18))).join(' '), /consistent/);
  assert.match(validateCard(tampered((c) => (c.results.extra = 1))).join(' '), /signal_free, false_alarms/);
  // Possible counts, but not this unit's: only recomputing the unit catches it.
  const plausible = tampered((c) => (c.results.false_alarms.dedoppler['0.01'] += 1));
  assert.deepEqual(validateCard(plausible), []);
  assert.match(verifyRecoveryCard(plausible, data).join(' '), /recomputation/);
});

test('combining units adds their counts, once per unit number', async () => {
  const data = await published();
  const [a, b] = [card(3, runRecoveryUnit(3, data).results), card(8, runRecoveryUnit(8, data).results)];
  const combined = combine([a, b, structuredClone(a)]);
  assert.equal(combined.units, 2);
  assert.equal(combined.perCell, 36);
  assert.equal(combined.signalFree, 2 * a.results.signal_free);
  assert.equal(combined.results.false_alarms.energy['0.001'], a.results.false_alarms.energy['0.001'] + b.results.false_alarms.energy['0.001']);
  assert.deepEqual(combined.results.recovery.dedoppler['0.01'].fsk, a.results.recovery.dedoppler['0.01'].fsk.map((k, i) => k + b.results.recovery.dedoppler['0.01'].fsk[i]));
  assert.deepEqual(combine([]), { units: 0, perCell: 0, signalFree: 0, results: null });
});

test('every unit draws its empty windows from the published windows, whose flags are fixed', async () => {
  const data = await published();
  const empty = emptyWindowFlags(data);
  assert.deepEqual(Object.keys(empty), Object.keys(DETECTORS));
  for (const name of Object.keys(DETECTORS)) {
    assert.ok(empty[name]['0.001'] <= empty[name]['0.01'] && empty[name]['0.01'] <= 0.03 * data.n, `${name}: ${JSON.stringify(empty[name])}`);
  }
});

test('a recovery card fits in the submit link, and ingestion reads it back', async () => {
  const c = card(3, runRecoveryUnit(3, await published()).results);
  const { url, prefilled } = issueLink(c, 'https://github.com/davidmdrpi/EIatHome');
  assert.ok(prefilled);
  assert.deepEqual(extractCard(new URL(url).searchParams.get('body')), c);
  assert.equal(new URL(url).searchParams.get('title'), 'Work unit result: recovery/v0 #3');
});
