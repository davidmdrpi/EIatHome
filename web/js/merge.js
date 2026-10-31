// Merging token transformers trained from the browser anchor (anchor.js): the trackb-merge/v0
// work unit (issue #20, milestone 4).
//
// A unit trains the anchor's token transformer from θ₀ with several seeds on the same Rosetta
// text, then merges their task vectors τᵢ = θᵢ − θ₀ four ways: a naive weight average, DARE
// then the average, TIES, and DARE then TIES. It scores every model on held-out text. Its card
// carries a summary (the held-out cross-entropies, how sparse the deltas and the merges are, and
// how often the seeds' changes agree in sign) and no deltas, so it fits in an issue link.
//
// The arithmetic is tracks/llm/merge.py's (milestone 2), on flat Float32Array parameter vectors:
// - DARE drops each entry with probability p and rescales the survivors by 1/(1 − p), keeping
//   the expected delta. A dropped delta travels as its survivors in int8, with one float32 scale
//   per tensor and varint-coded gaps between their indices;
// - TIES trims each delta to its largest `keep` fraction by magnitude, elects each parameter's
//   sign as the sign of the weighted sum, and takes the weighted mean of only the entries that
//   agree with it.

import { ANCHOR, anchorLayout, layoutBounds, peakTokens } from './anchor.js';
import { tokenParameterCount } from './models.js';
import { deriveSeed, makeRng } from './rng.js';
import { C, T, rosettaSymbols } from './signals.js';
import { SPLITS, targetCount } from './trackb.js';
import { rosettaSplits } from './trackb-splits.js';
import { APP_VERSION, MAX_UNIT, ROSETTA_CORPUS } from './workunits.js';

export const MERGE_SPEC = 'trackb-merge/v0';
export const MERGE_CARD_VERSION = 1;
const WORKERS = 4;
const EPOCHS = 20;
const BACKENDS = ['wasm', 'webgpu', 'webgl', 'cpu'];

// Everything that determines a unit's results besides the anchor and the corpus. The settings
// are milestone 2's: DARE drops 90% at random, TIES keeps the largest 20%.
export function mergeConfig(unit) {
  return {
    spec: MERGE_SPEC,
    unit,
    anchor: ANCHOR.id,
    family: 'rosetta',
    data_seed: deriveSeed(unit, 'trackb-merge-data'),
    seeds: Array.from({ length: WORKERS }, (_, i) => deriveSeed(unit, `trackb-merge-worker-${i}`)),
    epochs: EPOCHS,
    dare: { method: 'random', p: 0.9 },
    ties_keep: 0.2,
  };
}

export const MERGE_SPEC_V1 = 'trackb-merge/v1';
export const isMergeCard = (card) => card?.spec === MERGE_SPEC || card?.spec === MERGE_SPEC_V1;

// A unit's data: the whole frozen corpus as one FSK transmission, split in text order into
// training, validation and test (trackb-splits.js), in the anchor's tokens. Every worker gets
// the same tokens: the peak tokenizer reads each symbol's tone whatever the noise.
export function mergeTokens(config) {
  const snippets = Math.floor(rosettaSymbols(ROSETTA_CORPUS.text).length / T);
  const { frameSplits, wrapped } = rosettaSplits(snippets, config.data_seed, ROSETTA_CORPUS.text);
  if (wrapped) throw new Error('The corpus is too short for its snippets.');
  return Object.fromEntries(SPLITS.map((name) => [name, frameSplits[name].map((f) => peakTokens(f, f.length / C))]));
}

// Bits per test target of the training tokens' frequencies (add-one): what a model gets from
// token counts alone, with no context.
export function unigramBits(train, test, k = ANCHOR.k) {
  const counts = new Float64Array(k).fill(1);
  for (const segment of train) for (const token of segment) counts[token]++;
  const total = counts.reduce((a, b) => a + b, 0);
  let bits = 0;
  for (const segment of test) for (let i = 1; i < segment.length; i++) bits -= Math.log2(counts[segment[i]] / total);
  return bits / targetCount(test);
}

// ---------- deltas ----------

// τ = θ − θ₀, in float32 like the weights.
export const taskVector = (theta, theta0) => theta.map((v, i) => v - theta0[i]);

// DARE with random drops: the indices of the entries kept and their rescaled values.
export function dare(flat, p, rng) {
  const kept = [];
  for (let i = 0; i < flat.length; i++) if (rng.next() >= p) kept.push(i);
  return { kept: Int32Array.from(kept), values: Float64Array.from(kept, (i) => flat[i] / (1 - p)) };
}

// Which tensor of the layout each flat index falls in.
function tensorOf(indices, bounds) {
  const out = new Int32Array(indices.length);
  let t = 0;
  indices.forEach((index, i) => {
    while (index >= bounds[t + 1]) t++;
    out[i] = t;
  });
  return out;
}

// The survivors as they'd travel: int8 values with one float32 scale per tensor (its largest
// magnitude over 127). Returns the delta those rebuild and the payload's size in bytes: the
// values, the varint-coded index gaps and the scales.
export function compress({ kept, values }, layout = anchorLayout()) {
  const bounds = layoutBounds(layout);
  const tensor = tensorOf(kept, bounds);
  const peak = new Float64Array(layout.length);
  values.forEach((v, i) => (peak[tensor[i]] = Math.max(peak[tensor[i]], Math.abs(v))));
  const scales = Float32Array.from(peak, (m) => m / 127);
  const delta = new Float32Array(bounds.at(-1));
  let gapBytes = 0;
  kept.forEach((index, i) => {
    const scale = scales[tensor[i]];
    const q = scale > 0 ? Math.max(-127, Math.min(127, Math.round(values[i] / scale))) : 0;
    delta[index] = q * scale;
    for (let gap = index - (i ? kept[i - 1] : 0); ; gap >>>= 7) {
      gapBytes++;
      if (gap < 0x80) break;
    }
  });
  return { delta, bytes: kept.length + gapBytes + 4 * layout.length };
}

// The largest `keep` fraction of a flat delta's entries by magnitude (ties to the lower index);
// the rest zero.
export function trim(flat, keep) {
  if (keep >= 1) return Float32Array.from(flat);
  const order = Array.from(flat.keys()).sort((a, b) => Math.abs(flat[b]) - Math.abs(flat[a]) || a - b);
  const out = new Float32Array(flat.length);
  for (const i of order.slice(0, Math.round(keep * flat.length))) out[i] = flat[i];
  return out;
}

// TIES-merging: trim each delta, elect each parameter's sign from the weighted sum, and take
// the weighted mean of the entries that agree. Also returns the share of trimmed entries that
// disagree with their parameter's elected sign: what the election throws away.
// mean 'all' divides the agreeing entries' weighted sum by every delta's weight instead, so a
// zero or disagreeing entry counts as zero (milestone 5). After DARE it removes the 1/(1 − p)
// that TIES's own mean, over the survivors, keeps on every entry. It's unbiased only where the
// deltas agree in sign: where they disagree, sign election leans to whichever side survives.
export function ties(flats, weights, keep, mean = 'agreeing') {
  const trimmed = flats.map((f) => trim(f, keep));
  const n = flats[0].length;
  const total = weights.reduce((a, b) => a + b, 0);
  const merged = new Float32Array(n);
  let entries = 0;
  let conflicts = 0;
  for (let p = 0; p < n; p++) {
    let sum = 0;
    trimmed.forEach((t, i) => (sum += weights[i] * t[p]));
    const elected = Math.sign(sum);
    let num = 0;
    let mass = 0;
    trimmed.forEach((t, i) => {
      if (!t[p]) return;
      entries++;
      if (Math.sign(t[p]) === elected) {
        num += weights[i] * t[p];
        mass += weights[i];
      } else conflicts++;
    });
    merged[p] = mean === 'all' ? num / total : mass > 0 ? num / mass : 0;
  }
  return { delta: merged, conflicts: entries ? conflicts / entries : 0 };
}

export function weightedSum(flats, weights) {
  const out = new Float32Array(flats[0].length);
  for (let p = 0; p < out.length; p++) {
    let sum = 0;
    flats.forEach((f, i) => (sum += weights[i] * f[p]));
    out[p] = sum;
  }
  return out;
}

export const addDelta = (theta0, delta) => theta0.map((v, i) => v + delta[i]);

export const nonzeroShare = (flat) => flat.reduce((n, v) => n + (v !== 0), 0) / flat.length;

// Of the parameters every delta changes, the share where all of them change it the same way.
export function signAgreement(flats) {
  let changed = 0;
  let agree = 0;
  for (let p = 0; p < flats[0].length; p++) {
    const signs = flats.map((f) => Math.sign(f[p]));
    if (signs.includes(0)) continue;
    changed++;
    if (signs.every((s) => s === signs[0])) agree++;
  }
  return changed ? agree / changed : 0;
}

export function cosine(a, b) {
  let ab = 0;
  let aa = 0;
  let bb = 0;
  for (let i = 0; i < a.length; i++) {
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  return aa && bb ? ab / Math.sqrt(aa * bb) : 0;
}

export function meanPairwiseCosine(flats) {
  const values = flats.flatMap((a, i) => flats.slice(i + 1).map((b) => cosine(a, b)));
  return values.reduce((s, v) => s + v, 0) / values.length;
}

// ---------- the unit's merges ----------

// Worker i's DARE draws: which entries it keeps depends on its seed alone, not its weights.
const dareRng = (config, i) => makeRng(deriveSeed(config.seeds[i], 'dare'));

// Every merge a unit scores, from the workers' trained weights: deltas and summary statistics.
export function mergeDeltas(theta0, thetas, config) {
  const taus = thetas.map((theta) => taskVector(theta, theta0));
  const weights = taus.map(() => 1 / taus.length);
  const dropped = taus.map((tau, i) => dare(tau, config.dare.p, dareRng(config, i)));
  const sent = dropped.map((d) => compress(d)); // each delta as it would travel, and rebuilt
  const tiesMerge = ties(taus, weights, config.ties_keep);
  const deltas = {
    average: weightedSum(taus, weights),
    dare_average: weightedSum(sent.map((d) => d.delta), weights),
    ties: tiesMerge.delta,
    dare_ties: ties(sent.map((d) => d.delta), weights, config.ties_keep).delta,
    dare_ties_all: ties(sent.map((d) => d.delta), weights, config.ties_keep, 'all').delta,
  };
  return {
    deltas,
    payloadBytes: sent.map((d) => d.bytes),
    // The share of each worker's entries DARE keeps, and of each merge's entries that are nonzero.
    sparsity: {
      dare_kept: dropped.reduce((s, d) => s + d.kept.length, 0) / dropped.length / theta0.length,
      ...Object.fromEntries(Object.entries(deltas).map(([name, delta]) => [name, nonzeroShare(delta)])),
    },
    signs: { agree: signAgreement(taus), ties_conflicts: tiesMerge.conflicts, cosine: meanPairwiseCosine(taus) },
  };
}

// What a unit's card must report whatever its training did. DARE's masks follow from the seeds,
// so the payload sizes (values, index gaps and scales) and the share kept are fixed; so is the
// unigram baseline, from the corpus's tokens (the same for every unit: see mergeTokens).
export function expectedSummary(config) {
  const size = anchorParameters();
  const zeros = new Float32Array(size);
  const kept = config.seeds.map((_, i) => dare(zeros, config.dare.p, dareRng(config, i)));
  return {
    payloadBytes: kept.map((d) => compress(d).bytes),
    dareKept: round(kept.reduce((s, d) => s + d.kept.length, 0) / kept.length / size),
    unigram: expectedUnigram(),
  };
}

let unigram = null;
const expectedUnigram = () => {
  if (unigram === null) {
    const tokens = mergeTokens(mergeConfig(0));
    unigram = round(unigramBits(tokens.train, tokens.test));
  }
  return unigram;
};

// The seed a volunteer would keep without merging: the best on validation (never on test).
export const bestSeed = (validationBits) => validationBits.indexOf(Math.min(...validationBits));

// ---------- the card ----------

const round = (v, digits = 4) => (Number.isFinite(v) ? Number(v.toFixed(digits)) : null);

export const MERGES = ['average', 'dare_average', 'ties', 'dare_ties'];

// results: { targets, theta0, unigram, seeds: [{ test, validation, bestEpoch }], merges:
// { [MERGES]: { test, validation } }, stats } (stats from mergeDeltas).
export function mergeCard(unit, results, { backend, tfjs, seconds, started }) {
  const merged = (key) => Object.fromEntries(MERGES.map((m) => [m, round(results.merges[m][key])]));
  const validation = results.seeds.map((s) => s.validation);
  return {
    format: 'eiathome-result',
    version: MERGE_CARD_VERSION,
    spec: MERGE_SPEC,
    unit,
    config: mergeConfig(unit),
    inputs: {
      rosetta_corpus: { id: ROSETTA_CORPUS.id, sha256: ROSETTA_CORPUS.sha256 },
      anchor: { id: ANCHOR.id, theta0_sha256: ANCHOR.theta0_sha256 },
    },
    backend,
    tfjs,
    app: APP_VERSION,
    parameters: anchorParameters(),
    seconds: round(seconds, 1),
    started,
    targets: { ...results.targets },
    best_epochs: results.seeds.map((s) => s.bestEpoch),
    best_seed: bestSeed(validation),
    payload_bytes: results.stats.payloadBytes,
    metrics: {
      test_bits: { theta0: round(results.theta0), unigram: round(results.unigram), seeds: results.seeds.map((s) => round(s.test)), ...merged('test') },
      validation_bits: { seeds: validation.map((v) => round(v)), ...merged('validation') },
      sparsity: Object.fromEntries(['dare_kept', ...MERGES].map((k) => [k, round(results.stats.sparsity[k])])),
      signs: Object.fromEntries(Object.entries(results.stats.signs).map(([k, v]) => [k, round(v)])),
    },
  };
}

export const anchorParameters = () => tokenParameterCount(ANCHOR.arch, ANCHOR.width, ANCHOR.k, ANCHOR.window);

// The targets a unit scores in each split: fixed by the corpus (mergeTokens), so a card's
// counts are checked against them.
export const MERGE_TARGETS = { train: 2511, validation: 367, test: 719 };

const isInt = (v, lo, hi) => Number.isInteger(v) && v >= lo && v <= hi;
const inRange = (v, lo, hi) => Number.isFinite(v) && v >= lo && v <= hi;

// Checks a merge card's form against its spec (v0 or v1). Returns a list of problems (empty
// means valid). The numbers that follow from the spec alone (DARE's payload sizes and kept
// share, the unigram baseline) are recomputed. The rest can't be without training, so they're
// checked for range and replicated by other volunteers instead.
export function validateMergeCard(card) {
  const errors = [];
  const fail = (message) => errors.push(message);
  const v1 = card.spec === MERGE_SPEC_V1;
  if (card.format !== 'eiathome-result') fail('`format` must be "eiathome-result".');
  if (card.version !== (v1 ? MERGE_V1_CARD_VERSION : MERGE_CARD_VERSION)) fail(`\`version\` must be ${v1 ? MERGE_V1_CARD_VERSION : MERGE_CARD_VERSION}. Re-run the unit in the current Live Lab.`);
  if (!isInt(card.unit, 0, MAX_UNIT)) return [...errors, `\`unit\` must be a whole number from 0 to ${MAX_UNIT}.`];
  const config = (v1 ? mergeConfigV1 : mergeConfig)(card.unit);
  if (JSON.stringify(card.config) !== JSON.stringify(config)) fail("`config` doesn't match the spec for this unit.");
  commonProblems(card, fail);
  if (v1) return [...errors, ...v1Problems(card, config)];
  if (JSON.stringify(card.targets) !== JSON.stringify(MERGE_TARGETS)) fail("`targets` doesn't match the corpus's splits.");
  const workers = config.seeds.length;
  const ints = (v, lo, hi) => Array.isArray(v) && v.length === workers && v.every((x) => isInt(x, lo, hi));
  if (!ints(card.best_epochs, 1, config.epochs)) fail(`\`best_epochs\` needs ${workers} epochs from 1 to ${config.epochs}.`);
  if (!isInt(card.best_seed, 0, workers - 1)) fail(`\`best_seed\` must be a worker from 0 to ${workers - 1}.`);
  const expected = expectedSummary(config);
  if (!ints(card.payload_bytes, 1, 4 * anchorParameters())) fail(`\`payload_bytes\` needs ${workers} positive sizes.`);
  else if (JSON.stringify(card.payload_bytes) !== JSON.stringify(expected.payloadBytes)) fail(`\`payload_bytes\` must be ${expected.payloadBytes.join(', ')}: the sizes this unit's DARE masks give.`);

  const m = card.metrics;
  if (!m || typeof m !== 'object') return [...errors, '`metrics` is missing.'];
  const merges = MERGES;
  const t = m.test_bits;
  if (!t || !isBits(t.theta0) || !isBits(t.unigram) || !Array.isArray(t.seeds) || t.seeds.length !== workers || !t.seeds.every(isBits) || !merges.every((k) => isBits(t[k]))) {
    fail(`\`metrics.test_bits\` needs theta0, unigram, ${workers} seeds and ${merges.join(', ')}, in bits.`);
  } else if (Math.abs(t.unigram - expected.unigram) > 1e-4) {
    fail(`\`metrics.test_bits.unigram\` must be ${expected.unigram}, the corpus's unigram baseline.`);
  }
  const v = m.validation_bits;
  if (!v || !Array.isArray(v.seeds) || v.seeds.length !== workers || !v.seeds.every(isBits) || !merges.every((k) => isBits(v[k]))) {
    fail(`\`metrics.validation_bits\` needs ${workers} seeds and ${merges.join(', ')}, in bits.`);
  } else if (isInt(card.best_seed, 0, workers - 1) && v.seeds[card.best_seed] !== Math.min(...v.seeds)) {
    fail('`best_seed` must be the seed with the lowest validation bits.');
  }
  const s = m.sparsity;
  if (!s || !['dare_kept', ...MERGES].every((k) => inRange(s[k], 0, 1))) fail(`\`metrics.sparsity\` needs dare_kept, ${MERGES.join(', ')}, from 0 to 1.`);
  else if (s.dare_kept !== expected.dareKept) fail(`\`metrics.sparsity.dare_kept\` must be ${expected.dareKept}, the share this unit's DARE masks keep.`);
  if (!signsOk(m.signs)) fail('`metrics.signs` needs agree and ties_conflicts from 0 to 1, and cosine from −1 to 1.');
  return errors;
}

const isBits = (v) => Number.isFinite(v) && v >= 0; // a bad merge can cost far more than log2(k)
const signsOk = (g) => g && inRange(g.agree, 0, 1) && inRange(g.ties_conflicts, 0, 1) && inRange(g.cosine, -1, 1);

// What every merge card checks the same way: its data, anchor, machine and timing.
function commonProblems(card, fail) {
  // The pinned corpus is the only data: anything else in `inputs`, such as a slice dataset,
  // would change how the card is grouped for replication and duplicate checks.
  const inputs = card.inputs && typeof card.inputs === 'object' ? Object.keys(card.inputs).sort() : [];
  if (JSON.stringify(inputs) !== JSON.stringify(['anchor', 'rosetta_corpus'])) fail('`inputs` must hold the Rosetta corpus and the anchor, and nothing else: this spec uses no slices.');
  const corpus = card.inputs?.rosetta_corpus;
  if (corpus?.id !== ROSETTA_CORPUS.id || corpus?.sha256 !== ROSETTA_CORPUS.sha256) fail("The Rosetta corpus doesn't match the spec.");
  const anchor = card.inputs?.anchor;
  if (anchor?.id !== ANCHOR.id || anchor?.theta0_sha256 !== ANCHOR.theta0_sha256) fail("The anchor doesn't match the spec.");
  if (!BACKENDS.includes(card.backend)) fail(`\`backend\` must be one of ${BACKENDS.join(', ')}.`);
  if (typeof card.tfjs !== 'string' || !/^\d+\.\d+\.\d+$/.test(card.tfjs)) fail('`tfjs` must be a version like 4.22.0.');
  if (typeof card.app !== 'string' || card.app.length > 20) fail('`app` must be a short version string.');
  if (card.parameters !== anchorParameters()) fail(`\`parameters\` must be ${anchorParameters()}, the anchor's.`);
  if (!Number.isFinite(card.seconds) || card.seconds < 0) fail('`seconds` must be a non-negative number.');
  if (typeof card.started !== 'string' || Number.isNaN(Date.parse(card.started))) fail('`started` must be a date.');
}

// ---------- trackb-merge/v1 (milestone 5): seeds against shards, and a warm start ----------
//
// A v1 unit runs three arms of 4 workers each and merges each arm five ways (v0's four, and
// DARE then TIES dividing by every worker's weight, which removes DARE's factor from TIES):
// - seeds: v0's unit exactly, 4 seeds from θ₀ on the whole training text;
// - shards: 4 workers from θ₀, each on its own quarter of the training text, in order;
// - warm_shards: the same shards, from a shared warm start: θ₀ trained 3 epochs on the whole
//   training text, keeping its best validation epoch. Their changes are taken from it.
// Each arm records the worker a volunteer would keep (the best on validation), so a merge can
// be compared with it, and the seeds arm's kept worker is the one trained on all the data.

export const MERGE_V1_CARD_VERSION = 1;
export const ARMS = ['seeds', 'shards', 'warm_shards'];
export const MERGES_V1 = [...MERGES, 'dare_ties_all'];
const SHARDS = 4;
const WARM_EPOCHS = 3;

export function mergeConfigV1(unit) {
  const seeds = (label) => Array.from({ length: WORKERS }, (_, i) => deriveSeed(unit, `${label}-${i}`));
  return {
    spec: MERGE_SPEC_V1,
    unit,
    anchor: ANCHOR.id,
    family: 'rosetta',
    data_seed: deriveSeed(unit, 'trackb-merge-data'),
    epochs: EPOCHS,
    shards: SHARDS,
    warm: { epochs: WARM_EPOCHS, seed: deriveSeed(unit, 'trackb-merge-warm') },
    arms: {
      seeds: { start: 'theta0', train: 'all', seeds: seeds('trackb-merge-worker') }, // v0's workers
      shards: { start: 'theta0', train: 'shard', seeds: seeds('trackb-merge-shard') },
      warm_shards: { start: 'warm', train: 'shard', seeds: seeds('trackb-merge-warm-shard') },
    },
    dare: { method: 'random', p: 0.9 },
    ties_keep: 0.2,
  };
}

// An arm's settings in the shape mergeDeltas and expectedSummary take.
export const armConfig = (config, arm) => ({ ...config, seeds: config.arms[arm].seeds });

// The training text cut into `count` contiguous shards at snippet boundaries, in order.
export function shardTrain(tokens, count = SHARDS) {
  const [stream] = tokens.train;
  const snippets = stream.length / T;
  const at = (i) => Math.floor((i * snippets) / count) * T;
  return Array.from({ length: count }, (_, i) => stream.subarray(at(i), at(i + 1)));
}

export const MERGE_V1_TARGETS = { ...MERGE_TARGETS, shards: [623, 623, 623, 639] };

// results: { targets, theta0, unigram, warm: { test, validation, bestEpoch }, arms: { [ARMS]:
// { workers: [{ test, validation, bestEpoch }], merges: { [MERGES_V1]: { test, validation } },
// stats } } } (stats from mergeDeltas).
export function mergeCardV1(unit, results, { backend, tfjs, seconds, started }) {
  const arm = (r) => {
    const pick = (key) => ({ workers: r.workers.map((w) => round(w[key])), ...Object.fromEntries(MERGES_V1.map((m) => [m, round(r.merges[m][key])])) });
    return {
      test: pick('test'),
      validation: pick('validation'),
      sparsity: Object.fromEntries(['dare_kept', ...MERGES_V1].map((k) => [k, round(r.stats.sparsity[k])])),
      signs: Object.fromEntries(Object.entries(r.stats.signs).map(([k, v]) => [k, round(v)])),
    };
  };
  return {
    format: 'eiathome-result',
    version: MERGE_V1_CARD_VERSION,
    spec: MERGE_SPEC_V1,
    unit,
    config: mergeConfigV1(unit),
    inputs: {
      rosetta_corpus: { id: ROSETTA_CORPUS.id, sha256: ROSETTA_CORPUS.sha256 },
      anchor: { id: ANCHOR.id, theta0_sha256: ANCHOR.theta0_sha256 },
    },
    backend,
    tfjs,
    app: APP_VERSION,
    parameters: anchorParameters(),
    seconds: round(seconds, 1),
    started,
    targets: { ...results.targets, shards: [...results.targets.shards] },
    warm_best_epoch: results.warm.bestEpoch,
    arms: Object.fromEntries(ARMS.map((a) => {
      const r = results.arms[a];
      return [a, { best_epochs: r.workers.map((w) => w.bestEpoch), best_worker: bestSeed(r.workers.map((w) => w.validation)), payload_bytes: r.stats.payloadBytes }];
    })),
    metrics: {
      theta0: round(results.theta0),
      unigram: round(results.unigram),
      warm: { test: round(results.warm.test), validation: round(results.warm.validation) },
      ...Object.fromEntries(ARMS.map((a) => [a, arm(results.arms[a])])),
    },
  };
}

// A v1 card's own checks, after the ones every merge card shares.
function v1Problems(card, config) {
  const errors = [];
  const fail = (message) => errors.push(message);
  const workers = WORKERS;
  if (JSON.stringify(card.targets) !== JSON.stringify(MERGE_V1_TARGETS)) fail("`targets` doesn't match the corpus's splits and shards.");
  if (!isInt(card.warm_best_epoch, 1, config.warm.epochs)) fail(`\`warm_best_epoch\` must be an epoch from 1 to ${config.warm.epochs}.`);
  const m = card.metrics;
  if (!m || typeof m !== 'object') return [...errors, '`metrics` is missing.'];
  const expectedUnigram = expectedSummary(armConfig(config, 'seeds')).unigram;
  if (!isBits(m.theta0) || !isBits(m.unigram) || !isBits(m.warm?.test) || !isBits(m.warm?.validation)) fail('`metrics` needs theta0, unigram and the warm start’s test and validation bits.');
  else if (Math.abs(m.unigram - expectedUnigram) > 1e-4) fail(`\`metrics.unigram\` must be ${expectedUnigram}, the corpus's unigram baseline.`);
  if (!card.arms || typeof card.arms !== 'object' || JSON.stringify(Object.keys(card.arms)) !== JSON.stringify(ARMS)) return [...errors, `\`arms\` must be ${ARMS.join(', ')}.`];
  for (const a of ARMS) {
    const meta = card.arms[a];
    if (!meta || typeof meta !== 'object') {
      fail(`\`arms.${a}\` must be an object.`);
      continue;
    }
    const ints = (v, lo, hi) => Array.isArray(v) && v.length === workers && v.every((x) => isInt(x, lo, hi));
    if (!ints(meta.best_epochs, 1, config.epochs)) fail(`\`arms.${a}.best_epochs\` needs ${workers} epochs from 1 to ${config.epochs}.`);
    const expected = expectedSummary(armConfig(config, a));
    if (JSON.stringify(meta.payload_bytes) !== JSON.stringify(expected.payloadBytes)) fail(`\`arms.${a}.payload_bytes\` must be ${expected.payloadBytes.join(', ')}: the sizes this arm's DARE masks give.`);
    const r = m[a];
    const scored = (x) => x && Array.isArray(x.workers) && x.workers.length === workers && x.workers.every(isBits) && MERGES_V1.every((k) => isBits(x[k]));
    if (!r || !scored(r.test) || !scored(r.validation)) {
      fail(`\`metrics.${a}\` needs test and validation bits for ${workers} workers and ${MERGES_V1.join(', ')}.`);
      continue;
    }
    if (!isInt(meta.best_worker, 0, workers - 1) || r.validation.workers[meta.best_worker] !== Math.min(...r.validation.workers)) fail(`\`arms.${a}.best_worker\` must be the worker with the lowest validation bits.`);
    if (!r.sparsity || !['dare_kept', ...MERGES_V1].every((k) => inRange(r.sparsity[k], 0, 1))) fail(`\`metrics.${a}.sparsity\` needs dare_kept, ${MERGES_V1.join(', ')}, from 0 to 1.`);
    else if (r.sparsity.dare_kept !== expected.dareKept) fail(`\`metrics.${a}.sparsity.dare_kept\` must be ${expected.dareKept}, the share this arm's DARE masks keep.`);
    if (!signsOk(r.signs)) fail(`\`metrics.${a}.signs\` needs agree and ties_conflicts from 0 to 1, and cosine from −1 to 1.`);
  }
  return errors;
}
