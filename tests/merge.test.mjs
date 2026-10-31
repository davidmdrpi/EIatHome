// The browser's merge anchor (web/js/anchor.js) and the trackb-merge/v0 unit's arithmetic and
// cards (web/js/merge.js): issue #20, milestone 4.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

import { ANCHOR, anchorLayout, anchorTheta0, buildTheta0, flatten, layoutBounds, layoutProblems, parametersSha256, peakFrames, peakTokens, unflatten } from '../web/js/anchor.js';
import { MAX_CARD_BYTES, MAX_ISSUE_URL, issueLink, summarize, validateCard } from '../web/js/community.js';
import {
  ARMS,
  MERGES,
  MERGES_V1,
  MERGE_TARGETS,
  MERGE_V1_TARGETS,
  addDelta,
  armConfig,
  compress,
  cosine,
  dare,
  expectedSummary,
  mergeCard,
  mergeCardV1,
  mergeConfig,
  mergeConfigV1,
  mergeDeltas,
  mergeTokens,
  shardTrain,
  signAgreement,
  taskVector,
  ties,
  trim,
  unigramBits,
  validateMergeCard,
  weightedSum,
} from '../web/js/merge.js';
import { matchedTokenWidth, tokenParameterCount } from '../web/js/models.js';
import { deriveSeed, makeRng } from '../web/js/rng.js';
import { C, T, normalise, rosettaSymbols } from '../web/js/signals.js';
import { SPLITS, targetCount } from '../web/js/trackb.js';
import { rosettaSplits } from '../web/js/trackb-splits.js';
import { ROSETTA_CORPUS } from '../web/js/workunits.js';

const close = (a, b, tolerance = 1e-6) => Math.abs(a - b) <= tolerance;

// ---------- the anchor ----------

test('θ₀ rebuilds to its pinned hash, in the token transformer’s layout and size', async () => {
  const layout = anchorLayout();
  assert.equal(ANCHOR.k, 33);
  assert.equal(ANCHOR.width, matchedTokenWidth(ANCHOR.k, ANCHOR.window));
  assert.equal(layoutBounds(layout).at(-1), tokenParameterCount('transformer', ANCHOR.width, ANCHOR.k, ANCHOR.window));
  const flat = flatten(await anchorTheta0());
  assert.equal(await parametersSha256(flat), ANCHOR.theta0_sha256);
  assert.equal(await parametersSha256(flatten(buildTheta0())), ANCHOR.theta0_sha256, 'and again, from scratch');
  assert.notEqual(await parametersSha256(flatten(buildTheta0(layout, ANCHOR.seed + 1))), ANCHOR.theta0_sha256, 'the pin covers the seed');
  assert.deepEqual(flatten(unflatten(flat)), flat);

  // Kernels are Glorot-uniform, positions small, biases and offsets zero, scales one.
  const tensors = buildTheta0();
  layout.forEach(({ name, shape, init }, i) => {
    const values = [...tensors[i]];
    if (init === 'glorot') {
      const limit = Math.sqrt(6 / (shape[0] + shape[1]));
      assert.ok(values.every((v) => Math.abs(v) <= limit) && values.some((v) => v !== 0), name);
    } else if (init === 'position') assert.ok(values.every((v) => Math.abs(v) <= 0.0347), name);
    else assert.ok(values.every((v) => v === (init === 'ones' ? 1 : 0)), name);
  });
});

test('a model whose weights aren’t the anchor’s layout is named, weight by weight', () => {
  const layout = anchorLayout();
  const weights = layout.map((t, i) => ({ name: `layer${i}/${t.name}`, shape: t.shape }));
  assert.deepEqual(layoutProblems(weights), []);
  assert.match(layoutProblems(weights.slice(1)).join(' '), /20 weight tensors; the anchor has 21/);
  const wider = weights.map((w, i) => (i === 0 ? { ...w, shape: [33, 24] } : w));
  assert.match(layoutProblems(wider).join(' '), /Weight 0 .*33,24.*expects kernel \[33,22\]/);
});

test('the pinned tokenizer reads each Rosetta symbol’s tone, whatever the noise, and plain noise as token 0', () => {
  const symbols = rosettaSymbols(ROSETTA_CORPUS.text);
  for (const seed of [1, 2, 3]) {
    const { frameSplits } = rosettaSplits(Math.floor(symbols.length / T), seed, ROSETTA_CORPUS.text);
    let offset = 0;
    for (const name of SPLITS) {
      for (const frames of frameSplits[name]) {
        const tokens = peakTokens(frames, frames.length / C);
        tokens.forEach((token, i) => assert.equal(token, 1 + Math.floor((8 + 3 * symbols[offset + i]) / 2), `seed ${seed}, symbol ${offset + i}`));
        offset += tokens.length;
      }
    }
  }
  const rng = makeRng(5);
  const noise = normalise(Float32Array.from({ length: T * C }, () => rng.gauss()));
  assert.deepEqual([...peakTokens(noise, T)], new Array(T).fill(0));
  // Drawn as a line in its bin; token 0 as nothing.
  const frames = peakFrames();
  assert.deepEqual([...peakTokens(frames, ANCHOR.k)], Array.from({ length: ANCHOR.k }, (_, i) => i));
});

test('a merge unit’s data is the whole frozen corpus in text order, the same tokens for every unit', () => {
  const a = mergeTokens(mergeConfig(0));
  const b = mergeTokens(mergeConfig(4321));
  assert.deepEqual(Object.fromEntries(SPLITS.map((name) => [name, targetCount(a[name])])), MERGE_TARGETS);
  for (const name of SPLITS) assert.deepEqual(a[name].map((s) => [...s]), b[name].map((s) => [...s]));
  const unigram = unigramBits(a.train, a.test);
  assert.ok(unigram > 3 && unigram < Math.log2(16), `16 symbols at their training frequencies: ${unigram}`);
});

// ---------- DARE, TIES and the summaries ----------

test('DARE keeps about 1 − p of the entries and rescales them so the expected delta is unchanged', () => {
  const flat = Float32Array.from({ length: 2000 }, (_, i) => Math.sin(i) * 0.01);
  const sums = new Float64Array(flat.length);
  const draws = 400;
  let kept = 0;
  for (let d = 0; d < draws; d++) {
    const { kept: indices, values } = dare(flat, 0.9, makeRng(deriveSeed(d, 'dare')));
    kept += indices.length;
    indices.forEach((index, i) => (sums[index] += values[i]));
    indices.forEach((index, i) => assert.ok(close(values[i], flat[index] / 0.1, 1e-9)));
  }
  assert.ok(close(kept / draws / flat.length, 0.1, 0.005), `kept ${kept / draws / flat.length}`);
  const bias = sums.reduce((s, v, i) => s + (v / draws - flat[i]), 0) / flat.length;
  assert.ok(Math.abs(bias) < 2e-4, `mean bias ${bias}`);
});

test('a dropped delta travels as int8 values, varint gaps and one scale per tensor, and rebuilds within half a step', () => {
  const layout = [{ shape: [4] }, { shape: [300] }];
  const kept = Int32Array.from([1, 3, 4, 200, 303]); // gaps 1, 2, 1, 196 (two varint bytes), 103
  const values = Float64Array.from([0.5, -1, 0.25, 0.011, -0.02]);
  const { delta, bytes } = compress({ kept, values }, layout);
  assert.equal(bytes, 5 + 6 + 4 * 2);
  assert.equal(delta.length, 304);
  const scales = [1 / 127, 0.25 / 127]; // indices 0-3 are the first tensor, 4-303 the second
  kept.forEach((index, i) => assert.ok(Math.abs(delta[index] - values[i]) <= scales[index < 4 ? 0 : 1] / 2 + 1e-9, `entry ${index}`));
  assert.deepEqual([delta[3], delta[4]], [-1, 0.25], 'each tensor’s largest entry is exact');
  assert.equal(delta.filter((v) => v !== 0).length, 5);
});

test('TIES trims, elects each sign from the weighted sum, and averages only the entries that agree', () => {
  assert.deepEqual([...trim(Float32Array.from([0.1, -3, 2, -2, 0.5]), 0.4)], [0, -3, 2, 0, 0], 'ties go to the lower index');
  // Parameter 0: +1, +3 against −2 → elected +, mean of 1 and 3. Parameter 1: −1, −1 against +1
  // → elected −. Parameter 2: only one delta changes it. Parameter 3: opposite and equal → no sign.
  const a = Float32Array.from([1, -1, 0, 2]);
  const b = Float32Array.from([3, -1, 0, -2]);
  const c = Float32Array.from([-2, 1, 0.5, 0]);
  const { delta, conflicts } = ties([a, b, c], [1 / 3, 1 / 3, 1 / 3], 1);
  assert.deepEqual([...delta], [2, -1, 0.5, 0]);
  assert.ok(close(conflicts, 4 / 9), `the 4 disagreeing of 9 nonzero entries: ${conflicts}`);
  // Weights count in the election and the mean.
  assert.deepEqual([...ties([a, c], [0.25, 0.75], 1).delta], [-2, 1, 0.5, 2]);
  // The mean over all workers: the agreeing entries' weighted sum over every weight.
  const all = ties([a, b, c], [1 / 3, 1 / 3, 1 / 3], 1, 'all').delta;
  [4 / 3, -2 / 3, 0.5 / 3, 0].forEach((v, i) => assert.ok(close(all[i], v), `parameter ${i}: ${all[i]}`));
});

test('sign agreement counts the parameters every delta changes, and cosine their alignment', () => {
  const a = Float32Array.from([1, 2, -1, 0, 3]);
  const b = Float32Array.from([2, -1, -3, 1, 1]);
  assert.equal(signAgreement([a, b]), 3 / 4);
  assert.ok(close(cosine(a, a), 1) && close(cosine(a, Float32Array.from(a, (v) => -v)), -1));
  assert.deepEqual([...weightedSum([a, b], [0.5, 0.5])], [1.5, 0.5, -2, 0.5, 2]);
});

test('merging a seed sweep: identical seeds merge to themselves, except where DARE and TIES drop their changes', () => {
  const layout = anchorLayout();
  const theta0 = flatten(buildTheta0());
  const rng = makeRng(9);
  const shared = Float32Array.from(theta0, () => rng.gauss() * 0.05);
  const thetas = [0, 1, 2, 3].map(() => addDelta(theta0, shared));
  const config = mergeConfig(0);
  const stats = mergeDeltas(theta0, thetas, config);
  const tau = taskVector(thetas[0], theta0);
  assert.deepEqual([...stats.deltas.average], [...weightedSum([tau, tau, tau, tau], [0.25, 0.25, 0.25, 0.25])]);
  assert.ok(stats.deltas.average.every((v, i) => close(v, tau[i], 1e-7)));
  assert.equal(stats.signs.agree, 1);
  assert.ok(close(stats.signs.cosine, 1));
  assert.equal(stats.signs.ties_conflicts, 0);
  assert.ok(close(stats.sparsity.ties, 0.2, 0.001), 'TIES keeps the same largest 20% of every seed');
  assert.ok(close(stats.sparsity.dare_kept, 0.1, 0.01));
  assert.ok(close(stats.sparsity.dare_average, 1 - 0.9 ** 4, 0.02), 'the union of 4 independent 10% draws');
  // TIES averages the survivors only, so DARE's 10× rescale stays on every surviving entry.
  const ratio = stats.deltas.dare_ties.reduce((s, v, i) => s + (v ? v / tau[i] : 0), 0) / stats.deltas.dare_ties.filter(Boolean).length;
  assert.ok(close(ratio, 10, 0.2), `DARE then TIES scales the kept changes by ${ratio}`);
  // Dividing by every worker's weight keeps the expected merge at the shared change.
  const fit = (delta) => delta.reduce((s, v, i) => s + v * tau[i], 0) / tau.reduce((s, v) => s + v * v, 0);
  assert.ok(close(fit(stats.deltas.dare_ties_all), 1, 0.1), `DARE then TIES over all workers: ${fit(stats.deltas.dare_ties_all)}`);
  assert.ok(close(fit(stats.deltas.dare_average), 1, 0.1));
  assert.ok(stats.payloadBytes.every((b) => b > 0.1 * theta0.length && b < 0.25 * theta0.length + 4 * layout.length), `${stats.payloadBytes}`);
  // The masks follow from the seeds alone, so a card's sizes and kept share can be recomputed.
  const expected = expectedSummary(config);
  assert.deepEqual(stats.payloadBytes, expected.payloadBytes);
  assert.equal(Number(stats.sparsity.dare_kept.toFixed(4)), expected.dareKept);
  const other = mergeDeltas(theta0, thetas.map((t) => addDelta(t, Float32Array.from(t, () => rng.gauss()))), config);
  assert.deepEqual(other.payloadBytes, expected.payloadBytes, 'whatever the weights');
  assert.deepEqual(expectedSummary(mergeConfig(7)).payloadBytes, [1318, 1380, 1332, 1396]);
});

test('v1’s seeds arm is v0’s unit, and its shards cut the training text into ordered quarters', () => {
  for (const unit of [0, 7, 4321]) {
    const v1 = mergeConfigV1(unit);
    assert.deepEqual(v1.arms.seeds.seeds, mergeConfig(unit).seeds);
    assert.deepEqual(expectedSummary(armConfig(v1, 'seeds')), expectedSummary(mergeConfig(unit)));
    assert.equal(new Set(ARMS.flatMap((a) => v1.arms[a].seeds)).size, 12, 'every worker has its own seed');
  }
  const tokens = mergeTokens(mergeConfigV1(0));
  const shards = shardTrain(tokens);
  assert.deepEqual(shards.map((s) => s.length - 1), MERGE_V1_TARGETS.shards);
  assert.deepEqual(shards.flatMap((s) => [...s]), [...tokens.train[0]], 'the shards are the training text, in order, without overlap');
});

// ---------- cards ----------

// A card as the engine makes it, with plausible synthetic scores.
function makeMergeCard(unit, { shift = 0, started = '2026-10-31T00:00:00.000Z' } = {}) {
  const expected = expectedSummary(mergeConfig(unit));
  const seeds = [2.31, 2.33, 2.3, 2.32].map((test, i) => ({ test: test + shift, validation: 1.98 + i / 100 + shift, bestEpoch: 15 + i }));
  const merge = (test) => ({ test: test + shift, validation: test - 0.3 + shift });
  const results = {
    targets: MERGE_TARGETS,
    theta0: 5.7149,
    unigram: expected.unigram,
    seeds,
    merges: { average: merge(2.3), dare_average: merge(3.8), ties: merge(3.6), dare_ties: merge(55.2) },
    stats: { payloadBytes: expected.payloadBytes, sparsity: { dare_kept: expected.dareKept, average: 0.94, dare_average: 0.31, ties: 0.26, dare_ties: 0.31 }, signs: { agree: 0.87, ties_conflicts: 0.0002, cosine: 0.96 } },
  };
  return mergeCard(unit, results, { backend: 'wasm', tfjs: '4.22.0', seconds: 19.6, started });
}

// A v1 card as the engine makes it, with plausible synthetic scores.
function makeMergeCardV1(unit, { shift = 0, started = '2026-10-31T00:00:00.000Z' } = {}) {
  const config = mergeConfigV1(unit);
  const arm = (name, base) => {
    const expected = expectedSummary(armConfig(config, name));
    const merges = { average: base - 0.04, dare_average: 3.9, ties: 2.8, dare_ties: 42, dare_ties_all: 3.88 };
    return {
      workers: [0, 1, 2, 3].map((i) => ({ test: base + i / 50 + shift, validation: base - 0.4 + i / 100 + shift, bestEpoch: 10 + i })),
      merges: Object.fromEntries(MERGES_V1.map((m) => [m, { test: merges[m] + shift, validation: merges[m] - 0.3 + shift }])),
      stats: { payloadBytes: expected.payloadBytes, sparsity: { dare_kept: expected.dareKept, average: 0.94, dare_average: 0.32, ties: 0.4, dare_ties: 0.32, dare_ties_all: 0.32 }, signs: { agree: 0.6, ties_conflicts: 0.01, cosine: 0.69 } },
    };
  };
  const results = {
    targets: MERGE_V1_TARGETS,
    theta0: 5.7149,
    unigram: expectedSummary(armConfig(config, 'seeds')).unigram,
    warm: { test: 2.7072, validation: 2.4244, bestEpoch: 3 },
    arms: { seeds: arm('seeds', 2.3), shards: arm('shards', 2.5), warm_shards: arm('warm_shards', 2.51) },
  };
  return mergeCardV1(unit, results, { backend: 'wasm', tfjs: '4.22.0', seconds: 53.2, started });
}

test('a v1 merge card validates, fits an issue link, and is rejected when it doesn’t match its spec', () => {
  const card = makeMergeCardV1(7);
  assert.deepEqual(validateCard(card), []);
  assert.deepEqual(Object.keys(card.metrics), ['theta0', 'unigram', 'warm', ...ARMS]);
  assert.deepEqual(ARMS.map((a) => card.arms[a].best_worker), [0, 0, 0], 'the lowest validation bits');
  const issue = issueLink(card, 'https://github.com/davidmdrpi/EIatHome');
  assert.ok(issue.prefilled && issue.url.length <= MAX_ISSUE_URL, `${issue.url.length} characters`);
  const broken = (edit) => {
    const c = makeMergeCardV1(7);
    edit(c);
    return validateCard(c).join(' ');
  };
  assert.match(broken((c) => (c.config.warm.epochs = 5)), /config/);
  assert.match(broken((c) => (c.config.arms.shards.seeds[0] += 1)), /config/);
  assert.match(broken((c) => (c.targets.shards = [623, 623, 623, 640])), /targets/);
  assert.match(broken((c) => (c.warm_best_epoch = 4)), /warm_best_epoch/);
  assert.match(broken((c) => (c.arms.shards.payload_bytes = [1, 1, 1, 1])), /arms\.shards\.payload_bytes. must be/);
  assert.match(broken((c) => (c.arms.warm_shards.payload_bytes = c.arms.shards.payload_bytes)), /arms\.warm_shards\.payload_bytes/);
  assert.match(broken((c) => (c.arms.seeds.best_worker = 3)), /arms\.seeds\.best_worker/);
  assert.match(broken((c) => (c.metrics.shards.sparsity.dare_kept = 0.1)), /metrics\.shards\.sparsity\.dare_kept. must be/);
  assert.match(broken((c) => (c.metrics.unigram = 3.2)), /unigram. must be/);
  assert.match(broken((c) => delete c.metrics.warm_shards.test.dare_ties_all), /metrics\.warm_shards/);
  assert.match(broken((c) => delete c.arms.warm_shards), /arms/);
  assert.match(broken((c) => (c.inputs.slices = { dataset_id: 'x', slices: [] })), /inputs/);
  assert.match(broken((c) => (c.version = 2)), /version/);
  assert.deepEqual(validateCard(makeMergeCardV1(7, { shift: 40 })), [], 'a merge that costs far more than log2(k) is still a result');
});

test('a merge card validates, carries no weights and fits an issue link', () => {
  const card = makeMergeCard(7);
  assert.deepEqual(validateCard(card), []);
  assert.deepEqual(Object.keys(card.metrics.test_bits), ['theta0', 'unigram', 'seeds', ...MERGES]);
  assert.equal(card.best_seed, 0, 'the lowest validation bits');
  const text = JSON.stringify(card);
  assert.ok(text.length < 2000 && new TextEncoder().encode(text).length < MAX_CARD_BYTES, `${text.length} characters`);
  const issue = issueLink(card, 'https://github.com/davidmdrpi/EIatHome');
  assert.ok(issue.prefilled && issue.url.length <= MAX_ISSUE_URL, `${issue.url.length} characters`);
  assert.match(decodeURIComponent(issue.url.replace(/\+/g, ' ')), /Work unit result: trackb-merge\/v0 #7/);
});

test('merge cards that don’t match their spec are rejected', () => {
  const broken = (edit) => {
    const card = makeMergeCard(7);
    edit(card);
    return validateMergeCard(card).join(' ');
  };
  assert.match(broken((c) => (c.config.dare.p = 0.5)), /config/);
  assert.match(broken((c) => (c.config.seeds[0] += 1)), /config/);
  assert.match(broken((c) => (c.inputs.anchor.theta0_sha256 = '0'.repeat(64))), /anchor/);
  assert.match(broken((c) => (c.inputs.rosetta_corpus.sha256 = '0'.repeat(64))), /Rosetta corpus/);
  assert.match(broken((c) => (c.parameters = 6000)), /parameters/);
  assert.match(broken((c) => (c.targets.test = 700)), /targets/);
  assert.match(broken((c) => (c.best_seed = 2)), /best_seed/);
  assert.match(broken((c) => (c.best_epochs = [0, 1, 2, 3])), /best_epochs/);
  assert.match(broken((c) => c.metrics.test_bits.seeds.pop()), /test_bits/);
  assert.match(broken((c) => (c.metrics.test_bits.ties = 'low')), /test_bits/);
  assert.match(broken((c) => (c.metrics.sparsity.dare_kept = 1.5)), /sparsity/);
  assert.match(broken((c) => delete c.metrics.signs.cosine), /signs/);
  assert.match(broken((c) => (c.version = 2)), /version/);
  assert.match(broken((c) => (c.unit = -1)), /unit/);
  // Only the pinned corpus: a slice dataset would change how the card is grouped.
  assert.match(broken((c) => (c.inputs.slices = { dataset_id: 'different', slices: [] })), /inputs/);
  assert.match(broken((c) => (c.inputs.slices = null)), /inputs/);
  // What follows from the spec alone is recomputed: DARE's payload sizes and kept share, and the
  // unigram baseline. Each card below is otherwise identical to a valid one.
  assert.match(broken((c) => (c.payload_bytes = [1, 1, 1, 1])), /payload_bytes. must be 1318, 1380, 1332, 1396/);
  assert.match(broken((c) => (c.payload_bytes[2] += 2)), /payload_bytes/);
  assert.match(broken((c) => (c.metrics.sparsity.dare_kept = 0.1)), /dare_kept. must be/);
  assert.match(broken((c) => (c.metrics.test_bits.unigram = 3.2)), /unigram. must be/);
  assert.deepEqual(validateMergeCard(makeMergeCard(7, { shift: 40 })), [], 'a merge that costs far more than log2(k) is still a result');
});

test('merge units are replicated per unit, apart from Track A’s units of the same number', async () => {
  const { makeCard } = await import('./fixtures.mjs');
  const run = (issue, by, card) => ({ issue, submitted_by: by, card });
  const { units } = summarize([
    run(1, 'alice', makeMergeCard(4)),
    run(2, 'bob', makeMergeCard(4, { shift: 0.01 })),
    run(3, 'carol', makeMergeCard(5)),
    run(4, 'dave', makeMergeCard(5, { shift: 0.1 })),
    run(5, 'erin', await makeCard(4)),
    run(6, 'frank', makeMergeCardV1(4)),
    run(7, 'grace', makeMergeCardV1(4, { shift: 0.01 })),
  ]);
  assert.deepEqual(units.map((u) => [u.spec, u.unit, u.status]), [['track-a/v0', 4, 'single'], ['trackb-merge/v0', 4, 'replicated'], ['trackb-merge/v0', 5, 'disagree'], ['trackb-merge/v1', 4, 'replicated']]);
});

test('the ingestion script records a valid merge card and rejects a tampered one', () => {
  const dir = mkdtempSync(join(tmpdir(), 'eiathome-merge-'));
  const ingest = (number, login, card) => {
    const event = join(dir, `event-${number}.json`);
    const comment = join(dir, 'comment.md');
    const output = join(dir, 'output.txt');
    writeFileSync(event, JSON.stringify({ issue: { number, user: { login }, created_at: '2026-10-31T01:00:00Z', body: `Result card.\n\n\`\`\`json\n${JSON.stringify(card)}\n\`\`\`\n` } }));
    writeFileSync(output, '');
    spawnSync(process.execPath, ['tools/ingest_result.mjs', event, comment, join(dir, 'results.json'), 'web/data/slices'], { env: { ...process.env, GITHUB_OUTPUT: output }, encoding: 'utf8' });
    return { status: readFileSync(output, 'utf8').trim(), comment: readFileSync(comment, 'utf8') };
  };
  const accepted = ingest(40, 'alice', makeMergeCard(9));
  assert.equal(accepted.status, 'status=accepted');
  const v1 = ingest(39, 'alice', makeMergeCardV1(9));
  assert.equal(v1.status, 'status=accepted', 'a v1 unit is another unit');
  assert.match(v1.comment, /Unit 9 \(`trackb-merge\/v1`, wasm\) is recorded/);
  assert.match(accepted.comment, /Unit 9 \(`trackb-merge\/v0`, wasm\) is recorded[\s\S]*1 run\(s\) from 1 volunteer/);
  const tampered = makeMergeCard(9, { shift: 0.01 });
  tampered.config.ties_keep = 0.5;
  const rejected = ingest(41, 'bob', tampered);
  assert.equal(rejected.status, 'status=rejected');
  assert.match(rejected.comment, /config/);

  // The same volunteer can't count a unit twice by adding a slice dataset to its card, or by
  // running it again.
  const withSlices = makeMergeCard(9, { started: '2026-10-31T02:00:00.000Z' });
  withSlices.inputs.slices = { dataset_id: 'different', slices: [] };
  const sliced = ingest(42, 'alice', withSlices);
  assert.equal(sliced.status, 'status=rejected');
  assert.match(sliced.comment, /inputs/);
  const again = ingest(43, 'alice', makeMergeCard(9, { started: '2026-10-31T03:00:00.000Z' }));
  assert.equal(again.status, 'status=duplicate');
  assert.match(again.comment, /already submitted unit 9/);

  // Two volunteers with identical metrics but impossible payload sizes don't replicate: the
  // second card is rejected.
  const sizes = makeMergeCard(9, { started: '2026-10-31T04:00:00.000Z' });
  sizes.payload_bytes = [1, 1, 1, 1];
  assert.equal(ingest(44, 'carol', sizes).status, 'status=rejected');
  const stored = JSON.parse(readFileSync(join(dir, 'results.json'), 'utf8')).results;
  assert.deepEqual(stored.map((r) => [r.issue, r.submitted_by]), [[39, 'alice'], [40, 'alice']]);
  assert.deepEqual(summarize(stored).units.map((u) => [u.spec, u.unit, u.runs, u.status]), [['trackb-merge/v0', 9, 1, 'single'], ['trackb-merge/v1', 9, 1, 'single']]);
});
