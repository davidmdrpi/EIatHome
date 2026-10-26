import assert from 'node:assert/strict';
import { test } from 'node:test';

import { matchedSize, parameterCount } from '../web/js/models.js';
import { deriveSeed, makeRng } from '../web/js/rng.js';
import { T, makeSnippet, normalise, rosettaSymbols } from '../web/js/signals.js';
import { canonicalOrder, datasetId, parseSlice } from '../web/js/slices.js';
import { voyagerPools } from '../web/js/splits.js';
import { conditionalEntropies, heldOutCrossEntropy, markovSurrogate, shuffleSurrogate } from '../web/js/trackb.js';
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

function voyagerSamples(pool, region, seedLabel, n) {
  const rng = makeRng(deriveSeed(0, seedLabel));
  const samples = [];
  for (let i = 0; i < n; i++) {
    const { raw, label } = makeSnippet('voyager', rng, { region, slices: pool });
    samples.push(`${label.slice}|${label.group}|${label.drift}|${snippetKey(normalise(raw))}`);
  }
  return samples;
}
const snippetsOf = (samples) => samples.map((s) => s.split('|').pop());

const load = (specs) => Promise.all(specs.map(([meta, bytes]) => parseSlice(entry(meta), bytes)));
const DAY_1 = 57650.78;
const DAY_3 = 57652.9;
const DAY_6 = 57655.4;

test('Voyager training and evaluation use different sessions when there are several', async () => {
  const slices = await load([
    [{ id: 'a1', tstart: DAY_1 }, lineSlice(2.4, 1)],
    [{ id: 'a2', tstart: DAY_1 }, lineSlice(2.4, 2)],
    [{ id: 'b1', tstart: DAY_3 }, lineSlice(2.4, 3)],
    [{ id: 'c1', tstart: DAY_6 }, lineSlice(2.4, 4)],
  ]);
  const pools = voyagerPools(slices);
  assert.equal(pools.heldOut, true);
  assert.equal(pools.sessions, 3);
  const trainStarts = new Set(pools.train.map((s) => s.startMjd));
  for (const s of pools.eval) assert.ok(!trainStarts.has(s.startMjd), 'no session is in both splits');

  const train = new Set(snippetsOf(voyagerSamples(pools.train, 'train', 'train', 400)));
  const overlap = snippetsOf(voyagerSamples(pools.eval, 'eval', 'eval-voyager', 48)).filter((k) => train.has(k)).length;
  assert.equal(overlap, 0, 'no evaluation snippet is identical to a training snippet');
});

test('a renamed copy of the same scan is never treated as held out', async () => {
  const bytes = lineSlice(2.4, 1);
  const slices = await load([
    [{ id: 'original', observationFile: 'original.h5' }, bytes],
    [{ id: 'copy', observationFile: 'renamed-copy.h5' }, bytes],
  ]);
  const pools = voyagerPools(slices);
  assert.equal(pools.heldOut, false);
  assert.match(pools.reason, /one observing session/);
});

test('identical bytes stay on one side of the split even when start times disagree', async () => {
  const copied = lineSlice(2.4, 1);
  const slices = await load([
    [{ id: 'scan', tstart: DAY_1 }, copied],
    [{ id: 'copy-with-edited-time', tstart: DAY_6 }, copied],
    [{ id: 'other', tstart: DAY_3 }, lineSlice(2.4, 2)],
  ]);
  const pools = voyagerPools(slices);
  assert.equal(pools.heldOut, true);
  const trainHashes = new Set(pools.train.map((s) => s.sha256));
  for (const s of pools.eval) assert.ok(!trainHashes.has(s.sha256), 'no slice content is in both splits');
});

test('scans from the same session stay together', async () => {
  const slices = await load([
    [{ id: 'on-1', tstart: DAY_1 }, lineSlice(2.4, 1)],
    [{ id: 'on-2', tstart: DAY_1 + 10 / 1440 }, lineSlice(2.4, 2)], // ten minutes later
  ]);
  assert.equal(voyagerPools(slices).heldOut, false);
});

test('a slice without a start time means nothing is held out', async () => {
  const slices = await load([
    [{ id: 'dated', tstart: DAY_1 }, lineSlice(2.4, 1)],
    [{ id: 'undated', tstart: null }, lineSlice(2.4, 2)],
  ]);
  const pools = voyagerPools(slices);
  assert.equal(pools.heldOut, false);
  assert.match(pools.reason, /no start time/);
});

test('manifest order changes neither the sampled inputs nor the dataset id', async () => {
  const [a, b] = await load([
    [{ id: 'a', tstart: DAY_1 }, lineSlice(2.4, 1)],
    [{ id: 'b', tstart: DAY_3 }, lineSlice(2.6, 2)],
  ]);
  const forward = voyagerSamples(canonicalOrder([a, b]), 'eval', 'eval-voyager', 48);
  const reversed = voyagerSamples(canonicalOrder([b, a]), 'eval', 'eval-voyager', 48);
  assert.deepEqual(reversed, forward, 'same slices, labels and windows at every index');
  assert.equal(await datasetId([b, a]), await datasetId([a, b]));
  assert.deepEqual(voyagerPools(canonicalOrder([b, a])), voyagerPools(canonicalOrder([a, b])));
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

test('held-out cross-entropy does not collapse for shuffled tokens', () => {
  const text = Uint16Array.from(rosettaSymbols().subarray(0, 2048));
  const shuffled = shuffleSurrogate(text, makeRng(1));
  const ce = heldOutCrossEntropy(shuffled, 16).map((r) => r.bits);
  for (const bits of ce.slice(1)) assert.ok(bits >= ce[0] - 0.05, `shuffled stays flat: ${ce.map((b) => b.toFixed(2))}`);
  const plugIn = conditionalEntropies(shuffled, 16).map((r) => r.h);
  assert.ok(plugIn[3] < plugIn[0] - 1, 'the plug-in estimate does collapse here, which is why it is no longer the headline');
});

test('held-out cross-entropy finds the structure in English text', () => {
  const text = Uint16Array.from(rosettaSymbols().subarray(0, 2048));
  const ce = heldOutCrossEntropy(text, 16).map((r) => r.bits);
  assert.ok(ce[1] < ce[0] - 0.5, `one token of context helps: ${ce.map((b) => b.toFixed(2))}`);
  const markov = heldOutCrossEntropy(markovSurrogate(text, 16, makeRng(2)), 16).map((r) => r.bits);
  assert.ok(ce[3] < markov[3] - 0.2, 'text has structure longer than pairs; a Markov-1 copy does not');
});

test('LSTM and attention are sized to within 3% of the GRU parameter count', () => {
  for (const hidden of [16, 32, 64, 128]) {
    const target = parameterCount('gru', hidden);
    for (const arch of ['lstm', 'attention']) {
      const ratio = parameterCount(arch, matchedSize(arch, hidden)) / target;
      assert.ok(Math.abs(ratio - 1) < 0.03, `${arch} at GRU size ${hidden}: ${ratio.toFixed(3)}`);
    }
  }
  assert.equal(parameterCount('gru', 32), 11424, 'matches what TensorFlow.js reports for the default GRU');
});

test('the work-unit Rosetta corpus is pinned by its hash', () => {
  const digest = sha256(new TextEncoder().encode(ROSETTA_CORPUS.text).buffer);
  assert.equal(digest, ROSETTA_CORPUS.sha256, 'editing the built-in text changes track-a/v0: define a new spec instead');
  assert.equal(workUnitConfig(5).rosettaCorpus, ROSETTA_CORPUS.id);
});
