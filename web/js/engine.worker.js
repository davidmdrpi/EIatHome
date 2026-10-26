// Track A engine: trains a small GRU to predict the next spectrogram frame, entirely in this
// Web Worker, and reports metrics as it goes. Used by both the Live Lab and work units.

import '../vendor/tfjs/tf.min.js';
import '../vendor/tfjs/tf-backend-webgpu.min.js';
import '../vendor/tfjs/tf-backend-wasm.min.js';
import { deriveSeed, makeRng } from './rng.js';
import { C, DEFAULT_FOFF_HZ, DEFAULT_TSAMP_S, FAMILIES, T, driftToHzPerS, makeSnippet, normalise, shuffleFrames } from './signals.js';
import { canonicalOrder, datasetId, sha256Hex } from './slices.js';
import { voyagerPools } from './splits.js';
import { matchedSize } from './models.js';
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
    } else if (msg.type === 'train' || msg.type === 'workunit') {
      if (busy) throw new Error('Already running. Stop the current run first.');
      busy = true;
      stopRequested = false;
      try {
        if (msg.type === 'train') await liveTrain(msg.config);
        else await runWorkUnit(msg.unit);
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

function familiesAvailable(inputs) {
  return FAMILIES.filter((f) => f.id !== 'voyager' || inputs.slices.length).map((f) => f.id);
}

// Which Voyager slices each split may use. Only training on Voyager needs separate
// observing sessions; otherwise every slice is unseen evaluation data.
function slicePools(target, inputs) {
  if (target === 'voyager') return voyagerPools(inputs.slices);
  return { train: inputs.slices, eval: inputs.slices, heldOut: null };
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

// A learned position embedding, added to each time step (attention has no built-in order).
class PositionEmbedding extends tf.layers.Layer {
  static className = 'EIatHomePositionEmbedding';

  constructor(config) {
    super(config);
    this.maxLength = config.maxLength;
    this.seed = config.seed;
  }

  build(inputShape) {
    const width = inputShape[inputShape.length - 1];
    this.table = this.addWeight('table', [this.maxLength, width], 'float32', tf.initializers.randomNormal({ stddev: 0.02, seed: this.seed }));
    this.built = true;
  }

  call(inputs) {
    return tf.tidy(() => {
      const x = Array.isArray(inputs) ? inputs[0] : inputs;
      return x.add(this.table.read().slice([0, 0], [x.shape[1], -1]));
    });
  }

  computeOutputShape(inputShape) {
    return inputShape;
  }

  getConfig() {
    return { ...super.getConfig(), maxLength: this.maxLength, seed: this.seed };
  }
}

// Single-head causal self-attention: each time step attends only to itself and earlier steps.
class CausalSelfAttention extends tf.layers.Layer {
  static className = 'EIatHomeCausalSelfAttention';

  constructor(config) {
    super(config);
    this.seed = config.seed;
  }

  build(inputShape) {
    const d = inputShape[inputShape.length - 1];
    for (const [i, name] of ['q', 'k', 'v', 'o'].entries()) {
      this[`w${name}`] = this.addWeight(`w${name}`, [d, d], 'float32', tf.initializers.glorotUniform({ seed: this.seed + i }));
      this[`b${name}`] = this.addWeight(`b${name}`, [d], 'float32', tf.initializers.zeros());
    }
    this.built = true;
  }

  call(inputs) {
    return tf.tidy(() => {
      const x = Array.isArray(inputs) ? inputs[0] : inputs;
      const [, steps, d] = x.shape;
      const flat = x.reshape([-1, d]);
      const project = (w, b) => flat.matMul(w.read()).add(b.read()).reshape([-1, steps, d]);
      const q = project(this.wq, this.bq);
      const k = project(this.wk, this.bk);
      const v = project(this.wv, this.bv);
      const future = tf.sub(1, tf.linalg.bandPart(tf.ones([steps, steps]), -1, 0)); // 1 above the diagonal
      const scores = tf.matMul(q, k, false, true).div(Math.sqrt(d)).add(future.mul(-1e9));
      const mixed = tf.matMul(tf.softmax(scores, -1), v).reshape([-1, d]);
      return mixed.matMul(this.wo.read()).add(this.bo.read()).reshape([-1, steps, d]);
    });
  }

  computeOutputShape(inputShape) {
    return inputShape;
  }

  getConfig() {
    return { ...super.getConfig(), seed: this.seed };
  }
}

// One pre-norm transformer block: attention then a feed-forward layer, each with a residual.
function attentionStates(input, width, seed) {
  const glorot = (i) => tf.initializers.glorotUniform({ seed: seed + i });
  let h = tf.layers.dense({ units: width, kernelInitializer: glorot(10) }).apply(input);
  h = new PositionEmbedding({ maxLength: T, seed: seed + 11 }).apply(h);
  const attended = new CausalSelfAttention({ seed: seed + 12 }).apply(tf.layers.layerNormalization().apply(h));
  h = tf.layers.add().apply([h, attended]);
  const expanded = tf.layers.dense({ units: 2 * width, activation: 'relu', kernelInitializer: glorot(20) }).apply(tf.layers.layerNormalization().apply(h));
  const projected = tf.layers.dense({ units: width, kernelInitializer: glorot(21) }).apply(expanded);
  return tf.layers.add().apply([h, projected]);
}

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
  if (!available.includes(target)) throw new Error(`"${target}" isn't available. Load Voyager slices first.`);
  const pools = slicePools(target, inputs);
  const context = (region) => ({ region, text: inputs.text, slices: region === 'train' ? pools.train : pools.eval });
  // Nothing independent to hold out: Voyager scores measure reconstruction. Holds the reason.
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
    voyager: inputs.slices.length
      ? { dataset_id: await datasetId(inputs.slices), slices: inputs.slices.map((s) => ({ id: s.id, sha256: s.sha256, start_mjd: Number.isFinite(s.startMjd) ? s.startMjd : null })) }
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
