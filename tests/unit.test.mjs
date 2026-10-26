import assert from 'node:assert/strict';
import { test } from 'node:test';

import { matchedSize, parameterCount } from '../web/js/models.js';
import { deriveSeed, makeRng } from '../web/js/rng.js';
import { T, makeSnippet, normalise, rosettaSymbols } from '../web/js/signals.js';
import { canonicalOrder, datasetId, parseSlice } from '../web/js/slices.js';
import { holdoutSplits, sessionPools } from '../web/js/splits.js';
import { analyzeSplits, conditionalEntropies, fitMarkov, heldOutCrossEntropy, mutualInformation, sampleMarkov, shuffleWithin, targetCount } from '../web/js/trackb.js';
import { ROSETTA_CORPUS, workUnitConfig } from '../web/js/workunits.js';
import { entry, lineSlice, publishedSlices, sha256, sliceBytes } from './fixtures.mjs';

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

test('the published slices verify and credit their observations', async () => {
  const slices = await publishedSlices(); // rejects if any file doesn't match its SHA-256
  const ids = new Set(slices.filter((s) => s.family === 'voyager').map((s) => s.id));
  assert.ok(ids.size > 0);
  for (const s of slices) {
    assert.match(s.source.url, /^https:\/\//, `${s.id} links to its observation`);
    assert.ok(Number.isFinite(s.startMjd), `${s.id} records when its scan started`);
    if (s.family === 'offsource') {
      assert.match(s.source.source_name, /OFF/, `${s.id} comes from an off-source scan`);
      assert.ok(ids.has(s.source.window_of), `${s.id} names the Voyager slice whose window it mirrors`);
    }
  }
});

test('slice loading rejects entries without string identifiers', async () => {
  await assert.rejects(parseSlice({ ...entry({ id: 'x' }), id: 42 }, sliceBytes(() => 0)), /missing its "id"/);
});

test('dataset identity covers the bytes and the metadata that changes results', async () => {
  const bytes = lineSlice(2.4);
  const a = await parseSlice(entry({ id: 'a', drift: 2.4 }), bytes);
  const sameAgain = await parseSlice(entry({ id: 'a', drift: 2.4 }), bytes);
  const otherDrift = await parseSlice(entry({ id: 'a', drift: 2.5 }), bytes);
  const otherFamily = await parseSlice(entry({ id: 'a', drift: 2.4, family: 'offsource' }), bytes);
  assert.equal(await datasetId([a]), await datasetId([sameAgain]));
  assert.notEqual(await datasetId([a]), await datasetId([otherDrift]));
  assert.notEqual(await datasetId([a]), await datasetId([otherFamily]));
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

test('each real family samples only its own slices', async () => {
  const pool = await load([
    [{ id: 'on', tstart: DAY_1 }, lineSlice(2.4, 1)],
    [{ id: 'off', family: 'offsource', group: 'carrier window', tstart: DAY_1 }, sliceBytes((i) => Math.sin(i))],
  ]);
  const rng = makeRng(1);
  for (let i = 0; i < 50; i++) {
    assert.equal(makeSnippet('voyager', rng, { slices: pool }).label.slice, 'on');
    assert.equal(makeSnippet('offsource', rng, { slices: pool }).label.slice, 'off');
  }
  assert.throws(() => makeSnippet('offsource', rng, { slices: pool.filter((s) => s.family === 'voyager') }), /No Off-source slices/);
  await assert.rejects(parseSlice(entry({ id: 'x', family: 'candidate' }), sliceBytes(() => 0)), /family must be one of/);
});

test('Voyager training and evaluation use different sessions when there are several', async () => {
  const slices = await load([
    [{ id: 'a1', tstart: DAY_1 }, lineSlice(2.4, 1)],
    [{ id: 'a2', tstart: DAY_1 }, lineSlice(2.4, 2)],
    [{ id: 'b1', tstart: DAY_3 }, lineSlice(2.4, 3)],
    [{ id: 'c1', tstart: DAY_6 }, lineSlice(2.4, 4)],
  ]);
  const pools = sessionPools(slices);
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
  const pools = sessionPools(slices);
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
  const pools = sessionPools(slices);
  assert.equal(pools.heldOut, true);
  const trainHashes = new Set(pools.train.map((s) => s.sha256));
  for (const s of pools.eval) assert.ok(!trainHashes.has(s.sha256), 'no slice content is in both splits');
});

test('scans from the same session stay together', async () => {
  const slices = await load([
    [{ id: 'on-1', tstart: DAY_1 }, lineSlice(2.4, 1)],
    [{ id: 'on-2', tstart: DAY_1 + 10 / 1440 }, lineSlice(2.4, 2)], // ten minutes later
  ]);
  assert.equal(sessionPools(slices).heldOut, false);
});

test('a slice without a start time means nothing is held out', async () => {
  const slices = await load([
    [{ id: 'dated', tstart: DAY_1 }, lineSlice(2.4, 1)],
    [{ id: 'undated', tstart: null }, lineSlice(2.4, 2)],
  ]);
  const pools = sessionPools(slices);
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
  assert.deepEqual(sessionPools(canonicalOrder([b, a])), sessionPools(canonicalOrder([a, b])));
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

// The Rosetta symbols as one continuous stream, split 70/10/20 in order.
const textSplits = (tokens) => ({ train: [tokens.subarray(0, 1434)], validation: [tokens.subarray(1434, 1638)], test: [tokens.subarray(1638)] });
const eachSplit = (splits, fn) => Object.fromEntries(Object.entries(splits).map(([name, segments]) => [name, fn(segments)]));

test('held-out cross-entropy does not collapse for shuffled tokens', () => {
  const rng = makeRng(1);
  const shuffled = eachSplit(textSplits(Uint16Array.from(rosettaSymbols().subarray(0, 2048))), (segments) => shuffleWithin(segments, rng));
  const ce = heldOutCrossEntropy(shuffled, 16).map((r) => r.bits);
  for (const bits of ce.slice(1)) assert.ok(bits >= ce[0] - 0.05, `shuffled stays flat: ${ce.map((b) => b.toFixed(2))}`);
  const plugIn = conditionalEntropies(Object.values(shuffled).flat(), 16).map((r) => r.h);
  assert.ok(plugIn[3] < plugIn[0] - 1, 'the plug-in estimate does collapse here, which is why it is no longer the headline');
});

test('held-out cross-entropy finds the structure in English text', () => {
  const text = textSplits(Uint16Array.from(rosettaSymbols().subarray(0, 2048)));
  const ce = heldOutCrossEntropy(text, 16).map((r) => r.bits);
  assert.ok(ce[1] < ce[0] - 0.5, `one token of context helps: ${ce.map((b) => b.toFixed(2))}`);
  const chain = fitMarkov(text.train, 16);
  const rng = makeRng(2);
  const markov = heldOutCrossEntropy(eachSplit(text, (segments) => sampleMarkov(chain, segments.map((t) => t.length), rng)), 16).map((r) => r.bits);
  assert.ok(ce[3] < markov[3] - 0.2, 'text has structure longer than pairs; a Markov-1 copy does not');
});

test('Track B fits its tokenizer and surrogates on the training split only', async () => {
  const frames = (seed, count) => {
    const rng = makeRng(seed);
    return Array.from({ length: count }, () => Float32Array.from({ length: 16 * 8 }, () => rng.gauss()));
  };
  const run = (test) => analyzeSplits({ train: frames(1, 20), validation: frames(2, 5), test }, { k: 6, dim: 8, rng: makeRng(9) });
  const a = await run(frames(3, 5));
  const b = await run(frames(4, 5).map((f) => f.map((v) => v * 3 + 1))); // different test data
  assert.deepEqual(b.centroids, a.centroids, 'test frames never reach the codebook');
  for (const stream of ['real', 'shuffled', 'markov']) {
    for (const split of ['train', 'validation']) assert.deepEqual(b.streams[stream][split], a.streams[stream][split], `${stream} ${split}`);
  }
  assert.notDeepEqual(b.streams.real.test, a.streams.real.test);
});

test('statistics never look across a segment boundary, and each target is scored once', () => {
  const pairs = Array.from({ length: 50 }, () => Uint16Array.of(0, 1));
  assert.equal(mutualInformation(pairs, 2, 1), 0, 'within each segment, 0 is always followed by 1');
  assert.ok(mutualInformation([Uint16Array.from(pairs.flatMap((p) => [...p]))], 2, 1) > 0.9, 'joined into one stream, 1 -> 0 pairs would appear');

  const segments = (n, length) => Array.from({ length: n }, (_, i) => Uint16Array.from({ length }, (_, j) => (i + j) % 5));
  const splits = { train: segments(12, 16), validation: segments(3, 16), test: [...segments(2, 16), Uint16Array.of(3), segments(1, 7)[0]] };
  const expected = 2 * 15 + 0 + 6;
  assert.equal(targetCount(splits.test), expected);
  for (const row of heldOutCrossEntropy(splits, 5)) assert.equal(row.targets, expected, `context ${row.context}`);
});

test('the shuffled surrogate keeps each segment’s tokens', () => {
  const segments = [Uint16Array.of(1, 2, 3, 4, 5, 6), Uint16Array.of(7, 7, 8)];
  const shuffled = shuffleWithin(segments, makeRng(3));
  shuffled.forEach((s, i) => assert.deepEqual([...s].sort(), [...segments[i]].sort()));
});

test('real-data token statistics hold out whole scans, or whole sessions once there are three', async () => {
  const scan = (id, tstart, seed, family = 'voyager') => [{ id, family, group: family === 'voyager' ? 'carrier' : 'carrier window', tstart }, lineSlice(2.4, seed)];
  const minutes = (m) => m / 1440;
  const oneSession = await load([
    scan('s1', DAY_1, 1), scan('s1-off', DAY_1 + minutes(5), 2, 'offsource'),
    scan('s2', DAY_1 + minutes(10), 3), scan('s3', DAY_1 + minutes(20), 4),
  ]);
  const byScan = holdoutSplits(oneSession, 'voyager');
  assert.equal(byScan.unit, 'scan');
  assert.equal(byScan.independent, false);
  assert.deepEqual(['train', 'validation', 'test'].map((n) => byScan[n].map((s) => s.id)), [['s1'], ['s2'], ['s3']]);

  const copy = await parseSlice(entry({ id: 's3-copy', tstart: DAY_1 + minutes(15) }), lineSlice(2.4, 4)); // same bytes as s3
  const withCopy = holdoutSplits([...oneSession, copy], 'voyager');
  const where = (id) => ['train', 'validation', 'test'].filter((n) => withCopy[n].some((s) => s.id === id || s.sha256 === oneSession[3].sha256));
  assert.deepEqual(where('s3'), ['test'], 'a copy stays in its original’s split and isn’t counted twice');
  assert.equal(withCopy.test.length, 1);

  const sessions = holdoutSplits(await load([scan('a', DAY_1, 1), scan('b', DAY_3, 2), scan('c', DAY_6, 3)]), 'voyager');
  assert.deepEqual([sessions.unit, sessions.independent], ['session', true]);

  assert.match(holdoutSplits(await load([scan('a', DAY_1, 1), scan('b', DAY_1 + minutes(10), 2)]), 'voyager').error, /at least 3 scans/);
  assert.match(holdoutSplits(await load([scan('a', null, 1), scan('b', DAY_3, 2), scan('c', DAY_6, 3)]), 'voyager').error, /no start time/);
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
