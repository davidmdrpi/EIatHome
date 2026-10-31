// The browser's merge anchor (issue #20, milestone 4): a pinned tokenizer and a fixed starting
// point θ₀ for Track B's token transformer. Runs that share both use token IDs that mean the
// same frames and start from the same weights, so their weight changes (task vectors,
// τ = θ − θ₀) line up and can be averaged or merged (merge.js).
//
// The browser model is TF.js's own architecture, on its own tokens (16-frame snippets,
// normalized with asinh; see signals.js), so it gets its own anchor rather than tracks/llm's.
//
// The tokenizer fits nothing. A frame is token 0 when no channel reaches the threshold;
// otherwise it's 1 plus the bin of its brightest channel. It's the peak rule of tracks/llm's
// anchor v3 (6 noise sigma, 2-channel bins) in normalise()'s asinh units, so every Rosetta
// symbol always becomes the same token, whatever the noise.
//
// θ₀ comes from uniform draws of rng.js's seeded stream in a fixed order, rounded to float32.
// That's plain IEEE arithmetic, so it rebuilds identically on every machine. Its SHA-256 is
// pinned, and a run refuses an anchor that doesn't rebuild to it.

import { TOKEN_WINDOW, matchedTokenWidth } from './models.js';
import { deriveSeed, makeRng } from './rng.js';
import { C } from './signals.js';
import { sha256Hex } from './slices.js';

const THRESHOLD = 2.491779852644912; // asinh(6): a 6-sigma peak after normalise()
const BIN = 2; // channels per token bin: Rosetta's tones, 3 channels apart, never share one
const K = 1 + C / BIN;

export const ANCHOR = {
  id: 'token-anchor/v1',
  tokenizer: { kind: 'peak', threshold: THRESHOLD, width: BIN, channels: C },
  k: K,
  arch: 'transformer',
  width: matchedTokenWidth(K, TOKEN_WINDOW),
  window: TOKEN_WINDOW,
  seed: 4,
  theta0_sha256: 'd3db8ed5daa3234df2c78001a670debb8fad0aad35ba204bd7ca7a566776273e',
};

// The token transformer's weights in TF.js's order (layers.js, buildTokenModel with
// 'transformer'): each one's name (the part after the layer's name), shape and initializer.
export function anchorLayout(k = ANCHOR.k, d = ANCHOR.width, window = ANCHOR.window) {
  const dense = (rows, cols) => [
    { name: 'kernel', shape: [rows, cols], init: 'glorot' },
    { name: 'bias', shape: [cols], init: 'zeros' },
  ];
  const norm = [
    { name: 'gamma', shape: [d], init: 'ones' },
    { name: 'beta', shape: [d], init: 'zeros' },
  ];
  return [
    ...dense(k, d), // token projection
    { name: 'table', shape: [window, d], init: 'position' },
    ...norm,
    ...['q', 'k', 'v', 'o'].flatMap((x) => [
      { name: `w${x}`, shape: [d, d], init: 'glorot' },
      { name: `b${x}`, shape: [d], init: 'zeros' },
    ]),
    ...norm,
    ...dense(d, 2 * d), // feed-forward
    ...dense(2 * d, d),
    ...dense(d, k), // next-token logits
  ];
}

export const tensorSize = (shape) => shape.reduce((a, b) => a * b, 1);

// Where each tensor starts in the flat parameter vector, and the vector's length last.
export const layoutBounds = (layout) => layout.reduce((bounds, t) => [...bounds, bounds.at(-1) + tensorSize(t.shape)], [0]);

// θ₀, one Float32Array per tensor of the layout. Glorot-uniform kernels (limit
// sqrt(6 / (fan in + fan out)), as TF.js's), positions uniform with standard deviation 0.02
// (the size of the Live Lab's random start), zero biases and offsets, unit scales.
export function buildTheta0(layout = anchorLayout(), seed = ANCHOR.seed) {
  const rng = makeRng(deriveSeed(seed, ANCHOR.id));
  return layout.map(({ shape, init }) => {
    const values = new Float32Array(tensorSize(shape));
    if (init === 'ones') values.fill(1);
    const limit = init === 'glorot' ? Math.sqrt(6 / (shape[0] + shape[1])) : init === 'position' ? 0.02 * Math.sqrt(3) : 0;
    if (limit) for (let i = 0; i < values.length; i++) values[i] = rng.uniform(-limit, limit);
    return values;
  });
}

export function flatten(tensors) {
  const out = new Float32Array(tensors.reduce((n, t) => n + t.length, 0));
  let offset = 0;
  for (const t of tensors) {
    out.set(t, offset);
    offset += t.length;
  }
  return out;
}

export const unflatten = (flat, layout = anchorLayout()) => {
  const bounds = layoutBounds(layout);
  return layout.map((_, i) => Float32Array.from(flat.subarray(bounds[i], bounds[i + 1])));
};

// The SHA-256 of a flat parameter vector as little-endian float32, the bytes the pin covers.
export function parametersSha256(flat) {
  const bytes = new DataView(new ArrayBuffer(flat.length * 4));
  flat.forEach((v, i) => bytes.setFloat32(i * 4, v, true));
  return sha256Hex(bytes.buffer);
}

// θ₀ as tensors, after checking it rebuilds to the pinned hash.
let verified = null;
export async function anchorTheta0() {
  verified ??= (async () => {
    const tensors = buildTheta0();
    const sha = await parametersSha256(flatten(tensors));
    if (sha !== ANCHOR.theta0_sha256) throw new Error(`The browser anchor ${ANCHOR.id} didn’t rebuild to its pinned θ₀ on this machine, so its runs couldn’t be merged with anyone else’s.`);
    return tensors;
  })();
  try {
    return await verified;
  } catch (err) {
    verified = null;
    throw err;
  }
}

// The problems with a TF.js model's weights as the anchor's layout: names and shapes in order.
export function layoutProblems(weights, layout = anchorLayout()) {
  if (weights.length !== layout.length) return [`The model has ${weights.length} weight tensors; the anchor has ${layout.length}.`];
  return layout.flatMap((t, i) => {
    const { name, shape } = weights[i];
    const ok = name.split('/').at(-1) === t.name && shape.length === t.shape.length && shape.every((n, j) => n === t.shape[j]);
    return ok ? [] : [`Weight ${i} is ${name} [${shape}]; the anchor expects ${t.name} [${t.shape}].`];
  });
}

// The pinned tokenizer: token 0 for a frame with no channel at the threshold, else 1 plus its
// brightest channel's bin. frames: Float32Array(n * dim), already normalized.
export function peakTokens(frames, n, dim = C, { threshold, width } = ANCHOR.tokenizer) {
  const tokens = new Uint16Array(n);
  for (let i = 0; i < n; i++) {
    let best = 0;
    for (let c = 1; c < dim; c++) if (frames[i * dim + c] > frames[i * dim + best]) best = c;
    tokens[i] = frames[i * dim + best] >= threshold ? 1 + Math.floor(best / width) : 0;
  }
  return tokens;
}

// The frame each token stands for, for drawing tokens: a line at twice the threshold in noise
// sigma (asinh(12)) in its bin, or nothing for token 0. Shaped like a k-means codebook (k x dim).
export function peakFrames({ width, channels } = ANCHOR.tokenizer) {
  const k = 1 + channels / width;
  const frames = new Float32Array(k * channels);
  for (let token = 1; token < k; token++) frames[token * channels + (token - 1) * width + Math.floor(width / 2)] = Math.asinh(12);
  return frames;
}

// The tokenizer as trackb.js's tokenize() takes it.
export const peakTokenizer = () => ({ k: ANCHOR.k, centroids: peakFrames(), encode: (frames, n) => peakTokens(frames, n) });
