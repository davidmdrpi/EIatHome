// Track A engine: trains a small GRU to predict the next spectrogram frame, entirely in this
// Web Worker, and reports metrics as it goes. Used by both the Live Lab and work units.

import '../vendor/tfjs/tf.min.js';
import '../vendor/tfjs/tf-backend-webgpu.min.js';
import '../vendor/tfjs/tf-backend-wasm.min.js';
import { deriveSeed, makeRng } from './rng.js';
import { C, DEFAULT_FOFF_HZ, DEFAULT_TSAMP_S, FAMILIES, REAL_FAMILIES, T, driftToHzPerS, makeSnippet, normalise, shuffleFrames } from './signals.js';
import { canonicalOrder, datasetId, sha256Hex } from './slices.js';
import { sessionPools } from './splits.js';
import { ANCHOR, anchorTheta0, flatten, layoutProblems, unflatten } from './anchor.js';
import { attentionStates, buildTokenModel } from './layers.js';
import { ARMS, addDelta, armConfig, mergeCardV1, mergeConfigV1, mergeDeltas, mergeTokens, shardTrain, unigramBits } from './merge.js';
import { TOKEN_WINDOW, matchedSize } from './models.js';
import { APP_VERSION, RESULT_CARD_VERSION, ROSETTA_CORPUS, workUnitConfig } from './workunits.js';

const tf = globalThis.tf;
tf.wasm.setWasmPaths(new URL('../vendor/tfjs/', import.meta.url).href);

// "Auto" tries WebAssembly first: for models this small, GPU start-up and per-operation
// overhead usually cost more than the GPU saves. Users can pick WebGPU and compare ms/epoch.
const BACKENDS = ['wasm', 'webgpu', 'webgl', 'cpu'];
const N_EVAL = 48; // evaluation snippets per family
const N_MAP = 32; // hidden-state map snippets per family
const STEP = T * C;

let slices = [];
let rosettaText; // undefined = the built-in public-domain text
let stopRequested = false;
let busy = false;

const post = (msg, transfer) => self.postMessage(msg, transfer);
const pause = () => new Promise((resolve) => setTimeout(resolve, 0));

self.onmessage = async ({ data: msg }) => {
  try {
    if (msg.type === 'init') {
      const backend = await useBackend(msg.backend);
      post({ type: 'ready', backend, version: tf.version.tfjs });
    } else if (msg.type === 'slices') {
      slices = msg.slices;
    } else if (msg.type === 'text') {
      rosettaText = msg.text || undefined;
    } else if (msg.type === 'stop') {
      stopRequested = true;
    } else if (['train', 'workunit', 'token-lm', 'merge'].includes(msg.type)) {
      if (busy) throw new Error('Already running. Stop the current run first.');
      busy = true;
      stopRequested = false;
      try {
        if (msg.type === 'train') await liveTrain(msg.config);
        else if (msg.type === 'workunit') await runWorkUnit(msg.unit);
        else if (msg.type === 'merge') await runMergeUnit(msg.unit);
        else await runTokenModels(msg);
      } finally {
        busy = false;
      }
    }
  } catch (err) {
    post({ type: 'error', message: err?.message ?? String(err) });
  }
};

async function useBackend(preferred) {
  const order = preferred === 'auto' ? BACKENDS : [preferred, ...BACKENDS.filter((b) => b !== preferred)];
  for (const name of order) {
    try {
      if (await tf.setBackend(name)) {
        await tf.ready();
        return tf.getBackend();
      }
    } catch {
      // try the next backend
    }
  }
  throw new Error('No TensorFlow.js backend could start in this browser.');
}

// ---------- data ----------

// Everything a run reads, captured when it starts, so messages that arrive mid-run (new
// slices, new text) can't change its inputs.
function snapshotInputs({ text = rosettaText } = {}) {
  return { slices: canonicalOrder(slices), text };
}

// Real families need slices of their own; synthetic ones are always available.
function familiesAvailable(inputs) {
  return FAMILIES.filter((f) => !f.real || inputs.slices.some((s) => s.family === f.id)).map((f) => f.id);
}

// Which slices each split may use. Only the real family being trained on needs separate
// observing sessions; slices of any other family are unseen evaluation data.
function slicePools(target, inputs) {
  if (!REAL_FAMILIES.includes(target)) return { train: inputs.slices, eval: inputs.slices, heldOut: null };
  const others = inputs.slices.filter((s) => s.family !== target);
  const pools = sessionPools(inputs.slices.filter((s) => s.family === target));
  return { ...pools, train: pools.train.concat(others), eval: pools.eval.concat(others) };
}

function buildSet(family, n, rng, { region, shuffle = false, slices: pool, text }) {
  const x = new Float32Array(n * STEP);
  const labels = [];
  for (let i = 0; i < n; i++) {
    const { raw, label } = makeSnippet(family, rng, { region, slices: pool, text });
    let snippet = normalise(raw);
    if (shuffle) snippet = shuffleFrames(snippet, rng.permutation(T));
    x.set(snippet, i * STEP);
    labels.push(label);
  }
  return { x, n, labels, family };
}

function concatSets(sets) {
  const n = sets.reduce((a, s) => a + s.n, 0);
  const x = new Float32Array(n * STEP);
  let offset = 0;
  for (const s of sets) {
    x.set(s.x, offset * STEP);
    offset += s.n;
  }
  return { x, n, labels: sets.flatMap((s) => s.labels), families: sets.flatMap((s) => Array(s.n).fill(s.family)) };
}

// ---------- model ----------

// arch: 'gru' | 'lstm' | 'attention'. size: hidden units, or model width for attention.
function buildModel(arch, size, seed) {
  const input = tf.input({ shape: [null, C] });
  const recurrent = {
    units: size,
    returnSequences: true,
    kernelInitializer: tf.initializers.glorotUniform({ seed }),
    recurrentInitializer: tf.initializers.orthogonal({ seed: seed + 1 }),
  };
  let states;
  if (arch === 'gru') states = tf.layers.gru(recurrent).apply(input);
  else if (arch === 'lstm') states = tf.layers.lstm(recurrent).apply(input);
  else if (arch === 'attention') states = attentionStates(input, size, seed);
  else throw new Error(`Unknown architecture "${arch}"`);
  const prediction = tf.layers
    .dense({ units: C, kernelInitializer: tf.initializers.glorotUniform({ seed: seed + 2 }) })
    .apply(states);
  const predictor = tf.model({ inputs: input, outputs: prediction });
  const encoder = tf.model({ inputs: input, outputs: states });
  return { predictor, encoder, parameters: predictor.countParams(), arch, stateSize: size };
}

// Predictions of frames 1..T-1 from frames 0..T-2, as one flat array (n x (T-1) x C).
async function gruPredict(predictor, x, n) {
  const out = tf.tidy(() => predictor.apply(tf.tensor3d(x, [n, T, C]).slice([0, 0, 0], [n, T - 1, C])));
  const values = await out.data();
  out.dispose();
  return values;
}

// The hidden state after the last frame, for each snippet (n x hidden).
async function finalStates(encoder, x, n) {
  const out = tf.tidy(() => {
    const states = encoder.apply(tf.tensor3d(x, [n, T, C]));
    return states.slice([0, T - 1, 0], [n, 1, -1]).reshape([n, -1]);
  });
  const values = await out.data();
  out.dispose();
  return values;
}

// ---------- baselines and metrics ----------

function meanFrame(set) {
  const mean = new Float64Array(C);
  for (let i = 0; i < set.n; i++) {
    for (let t = 1; t < T; t++) for (let c = 0; c < C; c++) mean[c] += set.x[i * STEP + t * C + c];
  }
  return mean.map((v) => v / (set.n * (T - 1)));
}

// Least-squares linear map from one frame (plus a bias) to the next, with a small ridge.
function fitLinear(set) {
  const d = C + 1;
  const ata = new Float64Array(d * d);
  const atb = new Float64Array(d * C);
  const a = new Float64Array(d);
  for (let i = 0; i < set.n; i++) {
    for (let t = 0; t < T - 1; t++) {
      for (let p = 0; p < C; p++) a[p] = set.x[i * STEP + t * C + p];
      a[C] = 1;
      const next = i * STEP + (t + 1) * C;
      for (let p = 0; p < d; p++) {
        const ap = a[p];
        if (ap === 0) continue;
        for (let q = 0; q < d; q++) ata[p * d + q] += ap * a[q];
        for (let c = 0; c < C; c++) atb[p * C + c] += ap * set.x[next + c];
      }
    }
  }
  let trace = 0;
  for (let p = 0; p < d; p++) trace += ata[p * d + p];
  for (let p = 0; p < d; p++) ata[p * d + p] += (1e-3 * trace) / d;
  return solve(ata, atb, d, C);
}

// Solve A X = B for symmetric positive-definite A (d x d) by Cholesky; B is d x m.
function solve(A, B, d, m) {
  const L = new Float64Array(d * d);
  for (let i = 0; i < d; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * d + j];
      for (let k = 0; k < j; k++) s -= L[i * d + k] * L[j * d + k];
      L[i * d + j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j * d + j];
    }
  }
  const X = new Float64Array(d * m);
  for (let col = 0; col < m; col++) {
    const y = new Float64Array(d);
    for (let i = 0; i < d; i++) {
      let s = B[i * m + col];
      for (let k = 0; k < i; k++) s -= L[i * d + k] * y[k];
      y[i] = s / L[i * d + i];
    }
    for (let i = d - 1; i >= 0; i--) {
      let s = y[i];
      for (let k = i + 1; k < d; k++) s -= L[k * d + i] * X[k * m + col];
      X[i * m + col] = s / L[i * d + i];
    }
  }
  return X;
}

const baselines = {
  mean: (set, mean) => (i, t, c) => mean[c],
  persistence: (set) => (i, t, c) => set.x[i * STEP + (t - 1) * C + c],
  linear: (set, mean, W) => {
    const cache = new Map();
    return (i, t, c) => {
      const key = i * T + t;
      let row = cache.get(key);
      if (!row) {
        row = new Float64Array(C);
        const base = i * STEP + (t - 1) * C;
        for (let p = 0; p <= C; p++) {
          const ap = p < C ? set.x[base + p] : 1;
          for (let q = 0; q < C; q++) row[q] += ap * W[p * C + q];
        }
        cache.set(key, row);
      }
      return row[c];
    };
  },
};

// Mean squared error on frames 2..T-1, so every prediction has at least two frames of history.
function mse(set, predictFrame) {
  let s = 0;
  for (let i = 0; i < set.n; i++) {
    for (let t = 2; t < T; t++) {
      for (let c = 0; c < C; c++) {
        const diff = predictFrame(i, t, c) - set.x[i * STEP + t * C + c];
        s += diff * diff;
      }
    }
  }
  return s / (set.n * (T - 2) * C);
}

const fromGru = (pred) => (i, t, c) => pred[(i * (T - 1) + t - 1) * C + c];

// Prediction gain = 1 - MSE(model) / MSE(mean baseline). The baselines don't change while the
// GRU trains, so they're computed once.
function fixedBaselines(set, mean, W) {
  const floor = mse(set, baselines.mean(set, mean));
  return {
    floor,
    persistence: 1 - mse(set, baselines.persistence(set)) / floor,
    linear: 1 - mse(set, baselines.linear(set, mean, W)) / floor,
  };
}

async function gruGain(model, set, floor) {
  const pred = await gruPredict(model.predictor, set.x, set.n);
  return 1 - mse(set, fromGru(pred)) / floor;
}

function shuffledCopy(set, rng) {
  const x = new Float32Array(set.x.length);
  for (let i = 0; i < set.n; i++) x.set(shuffleFrames(set.x.subarray(i * STEP, (i + 1) * STEP), rng.permutation(T)), i * STEP);
  return { ...set, x };
}

// Top-2 principal components by power iteration with deflation.
function pca2(values, n, dim, previous) {
  const mean = new Float64Array(dim);
  for (let i = 0; i < n; i++) for (let d = 0; d < dim; d++) mean[d] += values[i * dim + d] / n;
  const cov = new Float64Array(dim * dim);
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < dim; p++) {
      const vp = values[i * dim + p] - mean[p];
      for (let q = 0; q < dim; q++) cov[p * dim + q] += (vp * (values[i * dim + q] - mean[q])) / n;
    }
  }
  const components = [];
  for (let k = 0; k < 2; k++) {
    let v = new Float64Array(dim).map((_, d) => Math.sin(d + 1 + k));
    let lambda = 0;
    for (let iter = 0; iter < 200; iter++) {
      const w = new Float64Array(dim);
      for (let p = 0; p < dim; p++) for (let q = 0; q < dim; q++) w[p] += cov[p * dim + q] * v[q];
      lambda = Math.hypot(...w) || 1;
      v = w.map((x) => x / lambda);
    }
    if (previous?.[k]) {
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += v[d] * previous[k][d];
      if (dot < 0) v = v.map((x) => -x); // keep the picture from flipping between epochs
    }
    components.push(v);
    for (let p = 0; p < dim; p++) for (let q = 0; q < dim; q++) cov[p * dim + q] -= lambda * v[p] * v[q];
  }
  const points = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 2; k++) {
      let s = 0;
      for (let d = 0; d < dim; d++) s += (values[i * dim + d] - mean[d]) * components[k][d];
      points[i * 2 + k] = s;
    }
  }
  return { points, components };
}

// Ridge regression from hidden states to drift rate: does the GRU's state encode drift?
function fitProbe(states, n, dim, targets) {
  const d = dim + 1;
  const ata = new Float64Array(d * d);
  const atb = new Float64Array(d);
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < d; p++) {
      const ap = p < dim ? states[i * dim + p] : 1;
      atb[p] += ap * targets[i];
      for (let q = 0; q < d; q++) ata[p * d + q] += ap * (q < dim ? states[i * dim + q] : 1);
    }
  }
  let trace = 0;
  for (let p = 0; p < d; p++) trace += ata[p * d + p];
  for (let p = 0; p < dim; p++) ata[p * d + p] += (1e-3 * trace) / d;
  const w = solve(ata, atb, d, 1);
  return (s, i) => {
    let y = w[dim];
    for (let p = 0; p < dim; p++) y += w[p] * s[i * dim + p];
    return y;
  };
}

// Synthetic drift rates are defined in channels per step; they're reported in Hz/s assuming
// Breakthrough Listen's high-resolution product. Voyager readouts use each slice's own units.
const SYNTHETIC_UNITS = { foffHz: DEFAULT_FOFF_HZ, tsamp: DEFAULT_TSAMP_S };

async function probeDrift(model, trainSet, evalSets) {
  const hidden = model.stateSize;
  const units = SYNTHETIC_UNITS;
  const toneIdx = trainSet.families.map((f, i) => (f === 'tone' ? i : -1)).filter((i) => i >= 0).slice(0, 600);
  if (toneIdx.length < 20 || !evalSets.tone) return null;
  const x = new Float32Array(toneIdx.length * STEP);
  toneIdx.forEach((src, j) => x.set(trainSet.x.subarray(src * STEP, (src + 1) * STEP), j * STEP));
  const trainStates = await finalStates(model.encoder, x, toneIdx.length);
  const probe = fitProbe(trainStates, toneIdx.length, hidden, toneIdx.map((i) => trainSet.labels[i].drift));

  const tone = evalSets.tone;
  const toneStates = await finalStates(model.encoder, tone.x, tone.n);
  const truth = tone.labels.map((l) => l.drift);
  const guess = truth.map((_, i) => probe(toneStates, i));
  const meanTruth = truth.reduce((a, b) => a + b, 0) / truth.length;
  let ssRes = 0;
  let ssTot = 0;
  let absErr = 0;
  truth.forEach((y, i) => {
    ssRes += (y - guess[i]) ** 2;
    ssTot += (y - meanTruth) ** 2;
    absErr += Math.abs(y - guess[i]);
  });
  const result = {
    r2: 1 - ssRes / ssTot,
    maeHzPerS: Math.abs(driftToHzPerS(absErr / truth.length, units.foffHz, units.tsamp)),
    points: truth.map((y, i) => [driftToHzPerS(y, units.foffHz, units.tsamp), driftToHzPerS(guess[i], units.foffHz, units.tsamp)]),
  };

  if (evalSets.voyager) {
    const v = evalSets.voyager;
    const states = await finalStates(model.encoder, v.x, v.n);
    const byGroup = {};
    v.labels.forEach((label, i) => {
      const g = (byGroup[label.group] ??= { measured: [], readouts: [] });
      g.measured.push(driftToHzPerS(label.drift, label.foffHz, label.tsamp));
      g.readouts.push(driftToHzPerS(probe(states, i), label.foffHz, label.tsamp));
    });
    const average = (values) => values.reduce((a, b) => a + b, 0) / values.length;
    result.voyager = Object.entries(byGroup).map(([group, { measured, readouts }]) => {
      const mean = average(readouts);
      const sd = Math.sqrt(average(readouts.map((r) => (r - mean) ** 2)));
      return { group, measured: average(measured), readout: mean, sd };
    });
  }
  return result;
}

// ---------- training ----------

async function train(config, onEpoch, inputs) {
  const { target, hidden, epochs, nTrain, batch, lr, seed, mixNoise, shuffle } = config;
  // Work units have no `arch`: they are GRU-only by spec. Other models get the GRU's parameter budget.
  const arch = config.arch ?? 'gru';
  const available = familiesAvailable(inputs);
  if (!available.includes(target)) throw new Error(`"${target}" isn't available. Load its slices first (see the Data tab).`);
  const pools = slicePools(target, inputs);
  const context = (region) => ({ region, text: inputs.text, slices: region === 'train' ? pools.train : pools.eval });
  // Nothing independent to hold out: the target's scores measure reconstruction. Holds the reason.
  const reconstruction = pools.heldOut === false ? pools.reason : null;

  const trainRng = makeRng(deriveSeed(seed, 'train'));
  const parts = [buildSet(target, mixNoise && target !== 'noise' ? nTrain / 2 : nTrain, trainRng, { ...context('train'), shuffle })];
  if (mixNoise && target !== 'noise') parts.push(buildSet('noise', nTrain / 2, trainRng, context('train')));
  const trainSet = concatSets(parts);

  const evalSets = {};
  for (const family of available) {
    const rng = makeRng(deriveSeed(seed, `eval-${family}`));
    evalSets[family] = buildSet(family, N_EVAL, rng, { ...context('eval'), shuffle: shuffle && family === target });
  }
  const mean = meanFrame(trainSet);
  const W = fitLinear(trainSet);
  const fixed = {};
  const shuffledSets = {};
  for (const family of available) {
    fixed[family] = fixedBaselines(evalSets[family], mean, W);
    const shuffled = shuffledCopy(evalSets[family], makeRng(deriveSeed(seed, `shuffle-${family}`)));
    shuffledSets[family] = { set: shuffled, floor: mse(shuffled, baselines.mean(shuffled, mean)) };
  }

  const model = buildModel(arch, matchedSize(arch, hidden), seed % 100000);
  const optimizer = tf.train.adam(lr);
  const order = makeRng(deriveSeed(seed, 'order'));
  const batchBuffer = new Float32Array(batch * STEP);
  const history = [];
  let previousComponents = null;

  try {
    // Control: how much drift a probe can read from the same GRU before any training.
    const untrainedProbe = await probeDrift(model, trainSet, evalSets);
    for (let epoch = 1; epoch <= epochs && !stopRequested; epoch++) {
      const started = performance.now();
      const perm = order.permutation(trainSet.n);
      let lossSum = 0;
      let seen = 0;
      for (let b = 0; b + batch <= perm.length && !stopRequested; b += batch) {
        for (let j = 0; j < batch; j++) batchBuffer.set(trainSet.x.subarray(perm[b + j] * STEP, (perm[b + j] + 1) * STEP), j * STEP);
        const xb = tf.tensor3d(batchBuffer, [batch, T, C]);
        const cost = optimizer.minimize(() => {
          const input = xb.slice([0, 0, 0], [batch, T - 1, C]);
          const target = xb.slice([0, 1, 0], [batch, T - 1, C]);
          return tf.losses.meanSquaredError(target, model.predictor.apply(input));
        }, true);
        lossSum += (await cost.data())[0] * batch;
        seen += batch;
        cost.dispose();
        xb.dispose();
      }
      if (stopRequested) break;
      const ms = performance.now() - started;

      const familyGains = {};
      const shuffledGains = {};
      for (const family of available) {
        const { persistence, linear, floor } = fixed[family];
        familyGains[family] = { persistence, linear, gru: await gruGain(model, evalSets[family], floor) };
        shuffledGains[family] = await gruGain(model, shuffledSets[family].set, shuffledSets[family].floor);
      }
      history.push({ epoch, loss: lossSum / seen, ms });
      await onEpoch({ epoch, epochs, loss: lossSum / seen, ms, gains: familyGains, shuffledGains, reconstruction, model, evalSets, previousComponents, setComponents: (c) => (previousComponents = c) });
      await pause();
    }
    const probe = stopRequested ? null : await probeDrift(model, trainSet, evalSets);
    return { history, probe, untrainedProbe, reconstruction, parameters: model.parameters, arch, stateSize: model.stateSize, stopped: stopRequested, available };
  } finally {
    model.predictor.dispose();
    optimizer.dispose();
  }
}

async function liveTrain(config) {
  const inputs = snapshotInputs();
  const pools = slicePools(config.target, inputs);
  const mapSets = {};
  for (const family of familiesAvailable(inputs)) {
    mapSets[family] = buildSet(family, N_MAP, makeRng(deriveSeed(config.seed, `map-${family}`)), {
      region: 'eval',
      slices: pools.eval,
      text: inputs.text,
      shuffle: config.shuffle && family === config.target,
    });
  }
  const mapAll = concatSets(Object.values(mapSets));

  const result = await train(config, async (state) => {
    const { model, evalSets } = state;
    const example = evalSets[config.target];
    const pred = await gruPredict(model.predictor, example.x.slice(0, STEP), 1);
    const actual = example.x.slice(2 * C, STEP);
    const predicted = pred.slice(C, (T - 1) * C);
    const residual = actual.map((v, i) => v - predicted[i]);

    const states = await finalStates(model.encoder, mapAll.x, mapAll.n);
    const { points, components } = pca2(states, mapAll.n, model.stateSize, state.previousComponents);
    state.setComponents(components);

    post({
      type: 'epoch',
      epoch: state.epoch,
      epochs: state.epochs,
      arch: state.model.arch,
      parameters: state.model.parameters,
      loss: state.loss,
      ms: state.ms,
      gains: state.gains,
      shuffledGains: state.shuffledGains,
      reconstruction: state.reconstruction,
      example: { actual, predicted, residual, rows: T - 2 },
      map: { points, families: mapAll.families },
      backend: tf.getBackend(),
    });
  }, inputs);
  post({
    type: 'trained',
    probe: result.probe,
    untrainedProbe: result.untrainedProbe,
    reconstruction: result.reconstruction,
    parameters: result.parameters,
    arch: result.arch,
    stateSize: result.stateSize,
    stopped: result.stopped,
  });
}

// ---------- Track B: token language model ----------

// Token streams arrive split into training, validation and test segments (see trackb.js).
// Examples never cross a segment boundary.
const TRAIN_STRIDE = 8; // a training window starts every 8 tokens of a segment
const SCORE_STRIDE = 16; // after its first window, a long segment is scored 16 new targets at a time
const TOKEN_EPOCHS = 10;
const MIN_EXAMPLES_PER_EPOCH = 160; // a small split (real slices) repeats its windows within an epoch
// Adam learning rates. The transformer overfits small splits within an epoch at the GRU's rate.
const TOKEN_LEARNING_RATE = { gru: 1e-2, transformer: 3e-3 };

// Integer tokens (shape [batch, time]) as float one-hot vectors ([batch, time, k]).
const oneHot = (tokens, shape, k) => tf.tidy(() => tf.oneHot(tf.tensor2d(tokens, shape, 'int32'), k).cast('float32'));

// Examples from segments: input tokens, next-token targets and a weight per target (0 for
// padding). With scoreOnce, a target an earlier window already scored also gets weight 0, so
// every target after the first token of a segment counts exactly once.
function segmentWindows(segments, stride, scoreOnce) {
  const examples = [];
  for (const tokens of segments) {
    let covered = 0; // targets 1..covered are already in an example
    for (let start = 0; start < tokens.length - 1; start += stride) {
      const length = Math.min(TOKEN_WINDOW, tokens.length - 1 - start);
      const x = new Int32Array(TOKEN_WINDOW);
      const y = new Int32Array(TOKEN_WINDOW);
      const w = new Float32Array(TOKEN_WINDOW);
      for (let j = 0; j < length; j++) {
        x[j] = tokens[start + j];
        y[j] = tokens[start + j + 1];
        w[j] = !scoreOnce || start + j + 1 > covered ? 1 : 0;
      }
      examples.push({ x, y, w });
      covered = start + length;
      if (covered >= tokens.length - 1) break;
    }
  }
  return examples;
}

function exampleTensors(examples, k) {
  const n = examples.length;
  const x = new Int32Array(n * TOKEN_WINDOW);
  const y = new Int32Array(n * TOKEN_WINDOW);
  const w = new Float32Array(n * TOKEN_WINDOW);
  examples.forEach((e, i) => {
    x.set(e.x, i * TOKEN_WINDOW);
    y.set(e.y, i * TOKEN_WINDOW);
    w.set(e.w, i * TOKEN_WINDOW);
  });
  const shape = [n, TOKEN_WINDOW];
  return { inputs: oneHot(x, shape, k), targets: oneHot(y, shape, k), weights: tf.tensor2d(w, shape), count: w.reduce((a, b) => a + b, 0) };
}

// Mean cross-entropy in bits over every target of the segments, each counted once.
async function heldOutBits(model, segments, k) {
  const examples = segmentWindows(segments, SCORE_STRIDE, true);
  let nats = 0;
  let targets = 0;
  for (let b = 0; b < examples.length; b += 64) {
    const { inputs, targets: onehot, weights, count } = exampleTensors(examples.slice(b, b + 64), k);
    const loss = tf.tidy(() => tf.logSoftmax(model.apply(inputs)).mul(onehot).sum(-1).mul(weights).sum().neg());
    nats += (await loss.data())[0];
    targets += count;
    tf.dispose([inputs, onehot, weights, loss]);
  }
  return { bits: targets ? nats / targets / Math.LN2 : NaN, targets };
}

// The token transformer at the merge anchor's θ₀ (anchor.js), after checking that the model's
// weights are laid out as the anchor's.
async function anchorModel(k) {
  if (k !== ANCHOR.k) throw new Error(`The anchor’s token transformer reads ${ANCHOR.k} token types, not ${k}.`);
  const model = buildTokenModel(k, 0, ANCHOR.arch);
  try {
    const problems = layoutProblems(model.weights);
    if (problems.length) throw new Error(`The token transformer doesn’t match ${ANCHOR.id}: ${problems[0]}`);
    setFlat(model, flatten(await anchorTheta0()));
    return model;
  } catch (err) {
    model.dispose();
    throw err;
  }
}

const getFlat = (model) => flatten(model.getWeights().map((w) => w.dataSync()));

function setFlat(model, flat) {
  const tensors = unflatten(flat).map((values, i) => tf.tensor(values, model.weights[i].shape));
  model.setWeights(tensors);
  tf.dispose(tensors);
}

// Trains on the training split with early stopping: the weights from the best validation
// epoch are kept. The model is then scored once on the test split, which it never saw.
// fromAnchor: start the transformer from the anchor's θ₀ instead of a seeded random start.
// start: weights to begin from instead (a flat vector in the anchor's layout, such as a warm start).
async function trainTokenModel(name, splits, k, seed, progress, arch, { fromAnchor = false, epochs = TOKEN_EPOCHS, start = null } = {}) {
  const model = fromAnchor ? await anchorModel(k) : buildTokenModel(k, seed, arch);
  if (start) setFlat(model, start);
  const optimizer = tf.train.adam(TOKEN_LEARNING_RATE[arch]);
  const rng = makeRng(deriveSeed(seed, `token-order-${name}`));
  const examples = segmentWindows(splits.train, TRAIN_STRIDE, false);
  const passes = Math.max(1, Math.ceil(MIN_EXAMPLES_PER_EPOCH / examples.length));
  let best = { bits: Infinity, epoch: 0, weights: null };
  try {
    for (let epoch = 1; epoch <= epochs && !stopRequested; epoch++) {
      const order = Array.from({ length: passes }, () => rng.permutation(examples.length)).flat();
      for (let b = 0; b < order.length && !stopRequested; b += 32) {
        const { inputs, targets, weights } = exampleTensors(order.slice(b, b + 32).map((i) => examples[i]), k);
        const cost = optimizer.minimize(() => tf.losses.softmaxCrossEntropy(targets, model.apply(inputs), weights), true);
        await cost.data();
        tf.dispose([cost, inputs, targets, weights]);
      }
      if (stopRequested) break;
      const { bits: validationBits } = await heldOutBits(model, splits.validation, k);
      if (validationBits < best.bits) {
        tf.dispose(best.weights ?? []);
        best = { bits: validationBits, epoch, weights: model.getWeights().map((w) => w.clone()) };
      }
      progress(name, epoch);
      await pause();
    }
    if (best.weights) model.setWeights(best.weights);
    const test = stopRequested ? { bits: NaN, targets: 0 } : await heldOutBits(model, splits.test, k);
    return { model, ...test, validationBits: best.bits, bestEpoch: best.epoch, optimizer };
  } catch (err) {
    model.dispose();
    optimizer.dispose();
    throw err;
  } finally {
    tf.dispose(best.weights ?? []);
  }
}

// Continue a prompt by sampling one token at a time.
async function sampleTokens(model, prompt, n, k, rng) {
  const tokens = Array.from(prompt);
  for (let i = 0; i < n; i++) {
    const probs = tf.tidy(() => {
      const logits = model.apply(oneHot(Int32Array.from(tokens), [1, tokens.length], k));
      return tf.softmax(logits.slice([0, tokens.length - 1, 0], [1, 1, k]).reshape([k]));
    });
    const p = await probs.data();
    probs.dispose();
    let r = rng.next();
    let pick = k - 1;
    for (let j = 0; j < k; j++) {
      r -= p[j];
      if (r <= 0) {
        pick = j;
        break;
      }
    }
    tokens.push(pick);
  }
  return tokens.slice(prompt.length);
}

// anchor: the tokens are the merge anchor's, so the transformer starts from its θ₀ (on every
// stream, the surrogates included); the GRU has no anchor and starts at random as usual.
async function runTokenModels({ id, k, splits, control, centroids, seed, arch = 'gru', anchor = false }) {
  const results = {};
  let controlResult = null;
  let sample = null;
  let parameters = null;
  const fromAnchor = anchor && arch === ANCHOR.arch;
  for (const name of ['real', 'shuffled', 'markov']) {
    const progress = (stream, epoch) => post({ type: 'token-lm-progress', id, stream, epoch, epochs: TOKEN_EPOCHS, arch });
    const { model, bits, targets, bestEpoch, optimizer } = await trainTokenModel(name, splits[name], k, seed % 100000, progress, arch, { fromAnchor });
    try {
      parameters = model.countParams();
      if (stopRequested) break;
      results[name] = { bits, targets, bestEpoch };
      if (name === 'real') {
        // Control: the real model on another family's frames, tokenized with this family's codebook.
        if (control?.length) controlResult = await heldOutBits(model, control, k);
        // Continue the first test segment from its first half (at most 16 tokens), next to the
        // real continuation. The transformer stops when its input would outgrow the window its
        // positions were trained on: it never slides a token onto an untrained position.
        const segment = splits.real.test.find((s) => s.length >= 2);
        if (segment) {
          const promptLength = Math.min(16, Math.floor(segment.length / 2));
          const capacity = arch === 'transformer' ? TOKEN_WINDOW + 1 - promptLength : Infinity;
          const count = Math.min(32, segment.length - promptLength, capacity);
          const generated = await sampleTokens(model, segment.subarray(0, promptLength), count, k, makeRng(deriveSeed(seed, 'token-sample')));
          const frames = (tokens) => {
            const out = new Float32Array(tokens.length * C);
            tokens.forEach((token, i) => out.set(centroids.subarray(token * C, (token + 1) * C), i * C));
            return out;
          };
          sample = { real: frames(Array.from(segment.subarray(promptLength, promptLength + count))), generated: frames(generated), rows: count };
        }
      }
    } finally {
      model.dispose();
      optimizer.dispose();
    }
  }
  post({ type: 'token-lm-result', id, stopped: stopRequested, results, control: controlResult, sample, epochs: TOKEN_EPOCHS, arch, parameters, fromAnchor });
}

// ---------- the merge work unit (merge.js) ----------

// trackb-merge/v1: three arms of 4 workers, each trained from θ₀ (or the shared warm start) on
// the frozen corpus or a shard of it, merged five ways, everything scored on the held-out text.
async function runMergeUnit(unit) {
  const config = mergeConfigV1(unit);
  const started = new Date();
  const t0 = performance.now();
  const corpusSha256 = await sha256Hex(new TextEncoder().encode(ROSETTA_CORPUS.text));
  if (corpusSha256 !== ROSETTA_CORPUS.sha256) {
    throw new Error(`The built-in Rosetta text no longer matches spec ${config.spec}. Define a new spec instead of editing the text.`);
  }
  const theta0 = flatten(await anchorTheta0());
  const splits = mergeTokens(config);
  const shards = shardTrain(splits, config.shards);
  const k = ANCHOR.k;
  const score = async (model) => ({ validation: (await heldOutBits(model, splits.validation, k)).bits, test: (await heldOutBits(model, splits.test, k)).bits });
  // Progress over the 13 trainings: the warm start, then 4 workers per arm.
  const jobs = 1 + ARMS.length * WORKERS_PER_ARM;
  let job = 0;
  const train = async (name, data, seed, epochs, start, label) => {
    const progress = (_, epoch) => post({ type: 'merge-progress', unit, label, job, jobs, epoch, epochs });
    const result = await trainTokenModel(name, data, k, seed, progress, ANCHOR.arch, { fromAnchor: true, epochs, start });
    job++;
    return result;
  };

  const scratch = await anchorModel(k); // scores θ₀, then each merge
  try {
    const start = await score(scratch);
    const warmRun = await train('merge-warm', splits, config.warm.seed, config.warm.epochs, null, 'the warm start');
    let thetaWarm;
    try {
      thetaWarm = getFlat(warmRun.model);
    } finally {
      warmRun.model.dispose();
      warmRun.optimizer.dispose();
    }
    if (stopRequested) return post({ type: 'merge-stopped', unit });
    const warm = { test: warmRun.bits, validation: warmRun.validationBits, bestEpoch: warmRun.bestEpoch };

    const arms = {};
    for (const arm of ARMS) {
      const { start: from, train: data, seeds } = config.arms[arm];
      const base = from === 'warm' ? thetaWarm : theta0;
      const workers = [];
      const thetas = [];
      for (const [i, seed] of seeds.entries()) {
        const own = data === 'all' ? splits : { ...splits, train: [shards[i]] };
        const label = `${ARM_LABELS[arm]}, worker ${i + 1} of ${seeds.length}`;
        // The seeds arm names its workers' order streams as v0 did, so it is v0's unit exactly.
        const run = await train(arm === 'seeds' ? `merge-${i}` : `merge-${arm}-${i}`, own, seed, config.epochs, from === 'warm' ? thetaWarm : null, label);
        try {
          if (stopRequested) break;
          workers.push({ test: run.bits, validation: run.validationBits, bestEpoch: run.bestEpoch });
          thetas.push(getFlat(run.model));
        } finally {
          run.model.dispose();
          run.optimizer.dispose();
        }
      }
      if (stopRequested) return post({ type: 'merge-stopped', unit });
      post({ type: 'merge-progress', unit, label: `${ARM_LABELS[arm]}: merging`, job, jobs, merging: true });
      const stats = mergeDeltas(base, thetas, armConfig(config, arm));
      const merges = {};
      for (const [name, delta] of Object.entries(stats.deltas)) {
        setFlat(scratch, addDelta(base, delta));
        merges[name] = await score(scratch);
      }
      arms[arm] = { workers, merges, stats };
    }
    const count = (segments) => segments.reduce((n, s) => n + s.length - 1, 0);
    const results = {
      targets: { ...Object.fromEntries(Object.entries(splits).map(([name, segments]) => [name, count(segments)])), shards: shards.map((s) => s.length - 1) },
      theta0: start.test,
      unigram: unigramBits(splits.train, splits.test, k),
      warm,
      arms,
    };
    const card = mergeCardV1(unit, results, { backend: tf.getBackend(), tfjs: tf.version.tfjs, seconds: (performance.now() - t0) / 1000, started: started.toISOString() });
    post({ type: 'merge-result', card });
  } finally {
    scratch.dispose();
  }
}

const WORKERS_PER_ARM = 4;
const ARM_LABELS = { seeds: 'Seeds on all the text', shards: 'Shards from θ₀', warm_shards: 'Shards from the warm start' };

const round = (v, digits = 4) => (Number.isFinite(v) ? Number(v.toFixed(digits)) : null);

async function runWorkUnit(unit) {
  const config = workUnitConfig(unit);
  const started = new Date();
  const t0 = performance.now();
  // Work units always use the spec's frozen corpus, never the Live Lab's custom text.
  const inputs = snapshotInputs({ text: ROSETTA_CORPUS.text });
  const corpusSha256 = await sha256Hex(new TextEncoder().encode(inputs.text));
  if (corpusSha256 !== ROSETTA_CORPUS.sha256) {
    throw new Error(`The built-in Rosetta text no longer matches spec ${config.spec}. Define a new spec instead of editing the text.`);
  }
  const inputRecord = {
    rosetta_corpus: { id: ROSETTA_CORPUS.id, sha256: corpusSha256 },
    slices: inputs.slices.length
      ? {
          dataset_id: await datasetId(inputs.slices),
          // The dataset id hashes every slice's bytes and metadata; the ids say which they were.
          slices: inputs.slices.map((s) => s.id),
        }
      : null,
  };
  let last = null;
  const result = await train(config, async (state) => {
    last = state;
    post({ type: 'wu-progress', unit, epoch: state.epoch, epochs: state.epochs, loss: state.loss });
  }, inputs);
  if (result.stopped || !last) {
    post({ type: 'wu-stopped', unit });
    return;
  }
  const metrics = { prediction_gain: {}, gru_gain_frames_shuffled: {} };
  for (const [family, g] of Object.entries(last.gains)) {
    metrics.prediction_gain[family] = { persistence: round(g.persistence), linear: round(g.linear), gru: round(g.gru) };
    metrics.gru_gain_frames_shuffled[family] = round(last.shuffledGains[family]);
  }
  metrics.final_train_loss = round(last.loss);
  if (result.probe) {
    const { probe, untrainedProbe } = result;
    metrics.drift_probe = {
      r2_synthetic: round(probe.r2),
      mae_hz_per_s_synthetic: round(probe.maeHzPerS),
      untrained_r2_synthetic: round(untrainedProbe?.r2),
      untrained_mae_hz_per_s_synthetic: round(untrainedProbe?.maeHzPerS),
      voyager: probe.voyager?.map((v, i) => ({
        group: v.group,
        measured_hz_per_s: round(v.measured),
        readout_hz_per_s: round(v.readout),
        readout_sd: round(v.sd),
        untrained_readout_hz_per_s: round(untrainedProbe?.voyager?.[i]?.readout),
      })),
    };
  }
  post({
    type: 'wu-result',
    card: {
      format: 'eiathome-result',
      version: RESULT_CARD_VERSION,
      spec: config.spec,
      unit,
      config,
      app: APP_VERSION,
      tfjs: tf.version.tfjs,
      backend: tf.getBackend(),
      parameters: result.parameters,
      families: result.available,
      inputs: inputRecord,
      started: started.toISOString(),
      seconds: round((performance.now() - t0) / 1000, 1),
      metrics,
    },
  });
}
