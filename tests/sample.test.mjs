// The Live Lab's random sky sample (web/js/sample.js) and the slice pool it draws from.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { describeSample, located, pickSample, pointingId, sampleLines } from '../web/js/sample.js';
import { REAL_FAMILIES, familyById } from '../web/js/signals.js';
import { publishedSlices } from './fixtures.mjs';

const pool = JSON.parse(readFileSync(new URL('../tools/slice_pool.json', import.meta.url), 'utf8'));

test('a random sample is a located real slice, never the one already shown', async () => {
  const slices = await publishedSlices();
  const seen = new Set();
  for (let i = 0; i < 200; i++) {
    const current = slices[i % slices.length];
    const sample = pickSample(slices, current.id, () => (i % 50) / 50);
    assert.notEqual(sample.id, current.id);
    assert.ok(located(sample), `${sample.id} says where the telescope pointed`);
    seen.add(sample.family);
  }
  assert.deepEqual([...seen].sort(), REAL_FAMILIES.filter((f) => slices.some((s) => s.family === f)).sort(), 'every published family can come up');
  // The extremes of the random draw stay in the pool.
  assert.ok(pickSample(slices, null, () => 0));
  assert.ok(pickSample(slices, null, () => 1 - 1e-12));
});

test('a sample prefers slices with a pointing, and gives up only when there is nothing else', () => {
  const sky = { ra_hours: 17.2, dec_deg: 12.4, epoch: '2020-07-16T22:13:56Z' };
  const a = { id: 'a', family: 'voyager', source: { sky } };
  const b = { id: 'b', family: 'voyager', source: {} };
  const synthetic = { id: 'c', family: 'tone', source: { sky } };
  assert.equal(pickSample([a, b, synthetic], null, () => 0.99).id, 'a', 'the one with a pointing');
  assert.equal(pickSample([a, b], 'a', () => 0.5).id, 'b', 'then any other real slice');
  assert.equal(pickSample([a], 'a'), null);
  assert.equal(pickSample([], null), null);
});

test('the marker card says where a sample came from and how its window was cut', async () => {
  const slices = await publishedSlices();
  const carrier = slices.find((s) => s.id === 'voyager1-59046-80036-carrier');
  const d = describeSample(carrier);
  assert.equal(d.window, 'on-target hit');
  assert.equal(d.sourceName, 'VOYAGER-1');
  assert.equal(d.telescope, 'Green Bank Telescope');
  assert.equal(d.mjd, carrier.source.tstart_mjd);
  assert.equal(d.date, '2020-07-16');
  assert.ok(Math.abs(d.freqMhz - 8419.54) < 0.01, `${d.freqMhz} MHz`);
  assert.ok(d.driftHzPerS < -0.3 && d.driftHzPerS > -0.5, `${d.driftHzPerS} Hz/s`);
  assert.ok(d.snr > 100);
  assert.equal([d.ra_hours, d.dec_deg].join(), [carrier.source.sky.ra_hours, carrier.source.sky.dec_deg].join());
  const [what, measured] = sampleLines(d);
  assert.equal(what, 'On-target hit · VOYAGER-1 · Green Bank Telescope');
  assert.match(measured, /^MJD 59046\.9263 · 8419\.5\d{5} MHz · drift −0\.\d{3} Hz\/s · SNR \d+\.\d$/);

  const off = describeSample(slices.find((s) => s.family === 'offsource'));
  assert.equal(off.window, 'off-source');
  assert.equal(off.driftHzPerS, null, 'nothing drifts in an off-source window');
  assert.match(sampleLines(off)[1], /no drift measured · peak SNR \d+\.\d in the window$/);

  for (const s of slices.filter((x) => x.family === 'blind-window' || x.family === 'stars')) {
    const card = describeSample(s);
    assert.equal(card.window, familyById[s.family].window);
    assert.ok(card.target, `${s.id} names its target`);
    assert.ok(card.telescope && card.mjd && card.freqMhz && card.snr !== null, `${s.id}: ${JSON.stringify(card)}`);
    assert.match(sampleLines(card)[0], s.family === 'stars' ? /^On-target hit · / : /^Random blind window · /);
  }
});

test('a sample belongs to the sky map target of its family, pointing and date', async () => {
  const slices = await publishedSlices();
  const carrier = slices.find((s) => s.id === 'voyager1-59046-80036-carrier');
  assert.equal(pointingId(carrier), 'voyager@17.2112,12.404@2020-07-16');
  assert.equal(pointingId({ family: 'voyager', source: {} }), null);
  const blind = slices.find((s) => s.family === 'blind-window' && s.source.tstart_mjd === carrier.source.tstart_mjd);
  assert.equal(pointingId(blind), 'blind-window@17.2112,12.404@2020-07-16', 'same place and date, its own family');
});

test('the slice pool stays small, located and credited', async () => {
  const slices = await publishedSlices();
  const pooled = slices.filter((s) => s.source.pool?.id === pool.id);
  assert.ok(pooled.length > 0, 'the pool has been cut');
  assert.ok(pooled.length <= pool.limits.max_slices, `${pooled.length} slices`);
  assert.ok(pooled.length * 16 * 256 * 4 <= pool.limits.max_bytes, 'under the size budget');
  const keys = new Set(pool.targets.map((t) => t.key));
  const archiveTargets = new Set(pooled.filter((s) => s.source.archive).map((s) => s.source.pool.key));
  assert.ok(archiveTargets.size <= pool.limits.max_targets);
  for (const s of pooled) {
    assert.ok(['stars', 'blind-window'].includes(s.family), `${s.id} is a hit or a blind window`);
    assert.ok(keys.has(s.source.pool.key), `${s.id} comes from a target in the pool`);
    assert.ok(located(s), `${s.id} says where the telescope pointed`);
    assert.equal(s.source.sky.target, pool.targets.find((t) => t.key === s.source.pool.key).name);
    assert.ok(Number.isFinite(s.startMjd) && Number.isFinite(Number(s.source.centre_freq_mhz)) && s.source.sky.telescope, s.id);
    if (s.source.archive) assert.match(s.source.archive.md5sum ?? '', /^[0-9a-f]{32}$/, `${s.id} records the archive's MD5 of its file`);
  }
});
