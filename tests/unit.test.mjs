import assert from 'node:assert/strict';
import { test } from 'node:test';

import { deriveSeed, makeRng } from '../web/js/rng.js';
import { T, makeSnippet, normalise, rosettaSymbols } from '../web/js/signals.js';
import { datasetId, parseSlice } from '../web/js/slices.js';
import { voyagerPools } from '../web/js/splits.js';
import { ROSETTA_CORPUS, workUnitConfig } from '../web/js/workunits.js';
import { entry, lineSlice, sha256, sliceBytes } from './fixtures.mjs';

test('slice loading verifies the SHA-256 of the bytes', async () => {
  const ones = sliceBytes(() => 1);
  const twos = sliceBytes(() => 2);
  const zeros = '0'.repeat(64);
  await assert.rejects(parseSlice(entry({ id: 'a', digest: zeros }), ones), /don't match the SHA-256/);
  await assert.rejects(parseSlice(entry({ id: 'b', digest: zeros }), twos), /don't match the SHA-256/);

  const loaded = await parseSlice(entry({ id: 'c', digest: sha256(ones) }), ones);
  assert.equal(loaded.sha256, sha256(ones));
  const noDigest = await parseSlice(entry({ id: 'd' }), twos);
  assert.equal(noDigest.sha256, sha256(twos), 'records the computed digest when the manifest has none');
});

test('slice loading rejects entries without string identifiers', async () => {
  await assert.rejects(parseSlice({ ...entry({ id: 'x' }), id: 42 }, sliceBytes(() => 0)), /missing its "id"/);
});

test('dataset identity covers the bytes and the metadata that changes results', async () => {
  const bytes = lineSlice(2.4);
  const a = await parseSlice(entry({ id: 'a', drift: 2.4 }), bytes);
  const sameAgain = await parseSlice(entry({ id: 'a', drift: 2.4 }), bytes);
  const otherDrift = await parseSlice(entry({ id: 'a', drift: 2.5 }), bytes);
  assert.equal(await datasetId([a]), await datasetId([sameAgain]));
  assert.notEqual(await datasetId([a]), await datasetId([otherDrift]));
});

const snippetKey = (x) => Array.from(x, (v) => v.toFixed(5)).join(',');

function voyagerKeys(pool, region, seedLabel, n) {
  const rng = makeRng(deriveSeed(0, seedLabel));
  const keys = [];
  for (let i = 0; i < n; i++) keys.push(snippetKey(normalise(makeSnippet('voyager', rng, { region, slices: pool }).raw)));
  return keys;
}

test('Voyager training and evaluation use different observations when there are several', async () => {
  const slices = await Promise.all(
    [
      ['a1', 'obs-a.h5', 1],
      ['a2', 'obs-a.h5', 2],
      ['b1', 'obs-b.h5', 3],
      ['c1', 'obs-c.h5', 4],
    ].map(([id, file, seed]) => parseSlice(entry({ id, observationFile: file }), lineSlice(2.4, seed))),
  );
  const pools = voyagerPools(slices);
  assert.equal(pools.heldOut, true);
  const trainObs = new Set(pools.train.map((s) => s.observation));
  for (const s of pools.eval) assert.ok(!trainObs.has(s.observation), 'no observation is in both splits');

  const train = new Set(voyagerKeys(pools.train, 'train', 'train', 400));
  const overlap = voyagerKeys(pools.eval, 'eval', 'eval-voyager', 48).filter((k) => train.has(k)).length;
  assert.equal(overlap, 0, 'no evaluation snippet is identical to a training snippet');
});

test('with one observation, Voyager evaluation is flagged as not held out', async () => {
  const one = await parseSlice(entry({ id: 'a' }), lineSlice(2.4));
  const pools = voyagerPools([one]);
  assert.equal(pools.heldOut, false);
});

test('Rosetta training and evaluation windows come from different parts of the text', () => {
  const length = rosettaSymbols().length;
  const split = Math.floor(length * 0.7);
  for (const [region, label] of [['train', 'train'], ['eval', 'eval-rosetta']]) {
    const rng = makeRng(deriveSeed(0, label));
    for (let i = 0; i < 200; i++) {
      const { start } = makeSnippet('rosetta', rng, { region }).label;
      if (region === 'train') assert.ok(start + T <= split);
      else assert.ok(start >= split);
    }
  }
});

test('the work-unit Rosetta corpus is pinned by its hash', () => {
  const digest = sha256(new TextEncoder().encode(ROSETTA_CORPUS.text).buffer);
  assert.equal(digest, ROSETTA_CORPUS.sha256, 'editing the built-in text changes track-a/v0: define a new spec instead');
  assert.equal(workUnitConfig(5).rosettaCorpus, ROSETTA_CORPUS.id);
});
