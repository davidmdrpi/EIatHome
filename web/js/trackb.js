// Track B: turn frames into "radio tokens" and measure how language-like the token stream is.
// Every statistic is reported next to surrogates that destroy specific kinds of structure.
//
// Tokens come in segments: runs of consecutive frames, such as a snippet, a slice or a stretch
// of text. No statistic looks across a segment boundary. The data is split into training,
// validation and test segments before anything is fitted. The tokenizer and the surrogates
// learn from the training split only, and held-out scores use the test split.

// k-means with k-means++ initialization. frames: Float32Array(n * dim).
export function kmeans(frames, n, dim, k, rng, iterations = 30) {
  const centroids = new Float32Array(k * dim);
  const dist2 = (i, j) => {
    let s = 0;
    for (let d = 0; d < dim; d++) {
      const diff = frames[i * dim + d] - centroids[j * dim + d];
      s += diff * diff;
    }
    return s;
  };

  const first = rng.int(0, n);
  centroids.set(frames.subarray(first * dim, (first + 1) * dim), 0);
  const nearest = new Float64Array(n).fill(Infinity);
  for (let j = 1; j < k; j++) {
    let total = 0;
    for (let i = 0; i < n; i++) {
      nearest[i] = Math.min(nearest[i], dist2(i, j - 1));
      total += nearest[i];
    }
    let target = rng.next() * total;
    let pick = n - 1;
    for (let i = 0; i < n; i++) {
      target -= nearest[i];
      if (target <= 0) {
        pick = i;
        break;
      }
    }
    centroids.set(frames.subarray(pick * dim, (pick + 1) * dim), j * dim);
  }

  const labels = new Uint16Array(n);
  for (let iter = 0; iter < iterations; iter++) {
    let changed = 0;
    for (let i = 0; i < n; i++) {
      let best = 0;
      let bestD = Infinity;
      for (let j = 0; j < k; j++) {
        const d = dist2(i, j);
        if (d < bestD) {
          bestD = d;
          best = j;
        }
      }
      if (labels[i] !== best || iter === 0) changed++;
      labels[i] = best;
    }
    const sums = new Float64Array(k * dim);
    const counts = new Uint32Array(k);
    for (let i = 0; i < n; i++) {
      counts[labels[i]]++;
      for (let d = 0; d < dim; d++) sums[labels[i] * dim + d] += frames[i * dim + d];
    }
    for (let j = 0; j < k; j++) {
      if (!counts[j]) continue; // keep an empty cluster's old centroid
      for (let d = 0; d < dim; d++) centroids[j * dim + d] = sums[j * dim + d] / counts[j];
    }
    if (!changed) break;
  }
  return { labels, centroids };
}

// The nearest centroid of each frame: tokenizes frames the codebook wasn't fitted on.
export function assign(frames, n, dim, centroids, k) {
  const labels = new Uint16Array(n);
  for (let i = 0; i < n; i++) {
    let best = 0;
    let bestD = Infinity;
    for (let j = 0; j < k; j++) {
      let d2 = 0;
      for (let d = 0; d < dim; d++) {
        const diff = frames[i * dim + d] - centroids[j * dim + d];
        d2 += diff * diff;
      }
      if (d2 < bestD) {
        bestD = d2;
        best = j;
      }
    }
    labels[i] = best;
  }
  return labels;
}

const concat = (arrays, Type) => {
  const out = new Type(arrays.reduce((n, a) => n + a.length, 0));
  let offset = 0;
  for (const a of arrays) {
    out.set(a, offset);
    offset += a.length;
  }
  return out;
};

export const SPLITS = ['train', 'validation', 'test'];
const mapSplits = (splits, fn) => Object.fromEntries(SPLITS.map((name) => [name, fn(splits[name], name)]));

// Scored tokens: every token after the first of its segment.
export const targetCount = (segments) => segments.reduce((n, s) => n + Math.max(0, s.length - 1), 0);

// Fits the codebook on the training frames only, then encodes every split, and any extra
// segments, with it. frameSplits: { train, validation, test }, each an array of segments
// (Float32Array of frames x dim). A fixed tokenizer ({ centroids, encode(frames, n) }, such as
// the merge anchor's in anchor.js) is used as it is, with nothing fitted.
export function tokenize(frameSplits, k, dim, rng, extra = [], fixed = null) {
  let centroids = fixed?.centroids;
  if (!fixed) {
    const train = concat(frameSplits.train, Float32Array);
    const n = train.length / dim;
    if (n < k) throw new Error(`Only ${n} training frames, too few for ${k} token types. Choose a smaller K.`);
    ({ centroids } = kmeans(train, n, dim, k, rng));
  }
  const encode = (segments) => segments.map((f) => (fixed ? fixed.encode(f, f.length / dim) : assign(f, f.length / dim, dim, centroids, k)));
  return { centroids, tokens: mapSplits(frameSplits, encode), extra: encode(extra) };
}

function ngramCounts(segments, n, k) {
  const counts = new Map();
  let total = 0;
  for (const tokens of segments) {
    for (let i = 0; i + n <= tokens.length; i++) {
      let key = 0;
      for (let j = 0; j < n; j++) key = key * k + tokens[i + j];
      counts.set(key, (counts.get(key) ?? 0) + 1);
      total++;
    }
  }
  return { counts, total };
}

// Block entropy H_n in bits, with the Miller-Madow bias correction.
function blockEntropy(segments, n, k) {
  const { counts, total } = ngramCounts(segments, n, k);
  let h = 0;
  for (const c of counts.values()) h -= (c / total) * Math.log2(c / total);
  h += (counts.size - 1) / (2 * total * Math.LN2);
  return { h, distinct: counts.size, total };
}

// Conditional entropies h_n = H_n - H_(n-1): the uncertainty of the next token given the
// previous n-1. "reliable" is false when the average observed n-gram appears fewer than 5 times.
export function conditionalEntropies(segments, k, maxOrder = 4) {
  const out = [];
  let previous = 0;
  for (let n = 1; n <= maxOrder; n++) {
    const { h, distinct, total } = blockEntropy(segments, n, k);
    out.push({ n, h: h - previous, reliable: total / distinct >= 5 });
    previous = h;
  }
  return out;
}

// Held-out cross-entropy (bits per token) of a smoothed n-gram model. Plug-in entropies
// collapse when there are more possible contexts than tokens, because most n-grams are seen
// once. A held-out score can't be lowered by memorizing them, so a shuffled stream stays near
// its single-token entropy at every context length, and any drop means the context really
// predicts the next token.
//
// Smoothing is hierarchical (Dirichlet): the estimate for a context of length L backs off to
// the estimate for its last L-1 tokens, with concentration `beta`, down to a uniform 1/k.
// The counts come from the training split only, as the token model's weights do. The
// validation split only chooses beta for each context length, as it only chooses the token
// model's epoch.
const BETAS = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64];

function ngramTables(segments, maxContext, k) {
  const tables = [];
  for (let L = 0; L <= maxContext; L++) {
    const joint = new Map(); // context + next token
    const context = new Map(); // context followed by any token
    for (const tokens of segments) {
      for (let i = L; i < tokens.length; i++) {
        let key = 0;
        for (let j = i - L; j < i; j++) key = key * k + tokens[j];
        context.set(key, (context.get(key) ?? 0) + 1);
        const full = key * k + tokens[i];
        joint.set(full, (joint.get(full) ?? 0) + 1);
      }
    }
    tables.push({ joint, context });
  }
  return tables;
}

// Every token after the first of its segment is scored once, whatever the context length:
// near the start of a segment the model uses the shorter context it has.
function scoreBits(tables, segments, contextLength, k, beta) {
  let bits = 0;
  let n = 0;
  for (const tokens of segments) {
    for (let i = 1; i < tokens.length; i++) {
      let p = 1 / k;
      for (let L = 0; L <= Math.min(contextLength, i); L++) {
        let key = 0;
        for (let j = i - L; j < i; j++) key = key * k + tokens[j];
        const { joint, context } = tables[L];
        p = ((joint.get(key * k + tokens[i]) ?? 0) + beta * p) / ((context.get(key) ?? 0) + beta);
      }
      bits -= Math.log2(p);
      n++;
    }
  }
  return { bits: n ? bits / n : NaN, targets: n };
}

export function fitNgram({ train, validation }, k, maxContext = 3) {
  const fitted = ngramTables(train, maxContext, k);
  const betas = Array.from({ length: maxContext + 1 }, (_, L) => {
    let best = BETAS[0];
    let bestBits = Infinity;
    for (const beta of BETAS) {
      const { bits } = scoreBits(fitted, validation, L, k, beta);
      if (bits < bestBits) {
        bestBits = bits;
        best = beta;
      }
    }
    return best;
  });
  return { k, betas, tables: fitted };
}

// Bits per token for each context length on `segments`, which the model wasn't fitted on.
export const ngramBits = (model, segments) => model.betas.map((beta, L) => ({ context: L, beta, ...scoreBits(model.tables, segments, L, model.k, beta) }));

export const heldOutCrossEntropy = (splits, k, maxContext = 3) => ngramBits(fitNgram(splits, k, maxContext), splits.test);

// Rank-frequency of n-grams, normalized to frequencies (for Zipf plots).
export function rankFrequency(segments, n, k) {
  const counts = [...ngramCounts(segments, n, k).counts.values()].sort((a, b) => b - a);
  const total = counts.reduce((a, b) => a + b, 0);
  return counts.map((c) => c / total);
}

// Plug-in mutual information (bits) between tokens d steps apart in the same segment.
export function mutualInformation(segments, k, d) {
  const joint = new Float64Array(k * k);
  const left = new Float64Array(k);
  const right = new Float64Array(k);
  let total = 0;
  for (const tokens of segments) {
    for (let i = 0; i + d < tokens.length; i++) {
      joint[tokens[i] * k + tokens[i + d]]++;
      left[tokens[i]]++;
      right[tokens[i + d]]++;
      total++;
    }
  }
  if (!total) return NaN;
  let mi = 0;
  for (let a = 0; a < k; a++) {
    for (let b = 0; b < k; b++) {
      const p = joint[a * k + b] / total;
      if (p > 0) mi += p * Math.log2(p / ((left[a] / total) * (right[b] / total)));
    }
  }
  return mi;
}

// S1: the tokens of each segment in random order. Same token frequencies, no order; never
// moves a token into another segment or split.
export function shuffleWithin(segments, rng) {
  return segments.map((tokens) => {
    const out = Uint16Array.from(tokens);
    for (let i = out.length - 1; i > 0; i--) {
      const j = Math.floor(rng.next() * (i + 1));
      [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  });
}

// S2: a first-order Markov chain with add-one smoothing, fitted to training segments only. It
// keeps pair statistics and nothing longer.
export function fitMarkov(segments, k) {
  const unigram = new Float64Array(k).fill(1);
  const transitions = new Float64Array(k * k).fill(1);
  for (const tokens of segments) {
    for (let i = 0; i < tokens.length; i++) {
      unigram[tokens[i]]++;
      if (i + 1 < tokens.length) transitions[tokens[i] * k + tokens[i + 1]]++;
    }
  }
  return { k, unigram, transitions };
}

// Segments of the given lengths drawn from a fitted chain.
export function sampleMarkov({ k, unigram, transitions }, lengths, rng) {
  const draw = (weights, offset) => {
    let total = 0;
    for (let j = 0; j < k; j++) total += weights[offset + j];
    let target = rng.next() * total;
    for (let j = 0; j < k; j++) {
      target -= weights[offset + j];
      if (target <= 0) return j;
    }
    return k - 1;
  };
  return lengths.map((length) => {
    const out = new Uint16Array(length);
    for (let i = 0; i < length; i++) out[i] = i ? draw(transitions, out[i - 1] * k) : draw(unigram, 0);
    return out;
  });
}

// Compressed size in bits per token, using the browser's built-in DEFLATE. Each segment is
// compressed on its own, so no match or dictionary spans a boundary. Short segments pay
// DEFLATE's fixed overhead, the same for every stream: compare streams, not absolute sizes.
export async function compressedBitsPerToken(segments) {
  let bits = 0;
  let tokens = 0;
  for (const segment of segments) {
    const stream = new Blob([Uint8Array.from(segment)]).stream().pipeThrough(new CompressionStream('deflate-raw'));
    bits += (await new Response(stream).arrayBuffer()).byteLength * 8;
    tokens += segment.length;
  }
  return bits / tokens;
}

const MAX_LAG = 24;

// How a control family's frames fall on this family's codebook: its commonest token, the share
// of its frames on it, how common that token is in this family's training tokens, and whether
// it's the codebook's flattest frame (the smallest peak), the one plain noise is nearest.
export function controlProfile(controlSegments, trainSegments, centroids, k, dim) {
  const count = (segments) => {
    const c = new Array(k).fill(0);
    for (const s of segments) for (const token of s) c[token]++;
    return c;
  };
  const control = count(controlSegments);
  const train = count(trainSegments);
  const frames = control.reduce((a, b) => a + b, 0);
  const token = control.indexOf(Math.max(...control));
  const peaks = Array.from({ length: k }, (_, j) => Math.max(...centroids.subarray(j * dim, (j + 1) * dim).map(Math.abs)));
  return {
    token,
    frames,
    share: control[token] / frames,
    typesUsed: control.filter(Boolean).length,
    trainShare: train[token] / train.reduce((a, b) => a + b, 0),
    trainCommonest: train[token] === Math.max(...train),
    flattest: peaks[token] === Math.min(...peaks),
  };
}

// Everything Track B reports for one family, from frames already split into training,
// validation and test segments. `control` holds another family's frame segments, encoded with
// this family's codebook and scored with its n-gram model. `tokenizer` is a fixed tokenizer to
// use instead of fitting k-means (see tokenize).
export async function analyzeSplits(frameSplits, { k, dim, rng, control = [], tokenizer = null }) {
  const { centroids, tokens, extra } = tokenize(frameSplits, k, dim, rng, control, tokenizer);
  const chain = fitMarkov(tokens.train, k);
  const streams = {
    real: tokens,
    shuffled: mapSplits(tokens, (segments) => shuffleWithin(segments, rng)),
    markov: mapSplits(tokens, (segments) => sampleMarkov(chain, segments.map((s) => s.length), rng)),
  };
  const pooled = (stream) => SPLITS.flatMap((name) => stream[name]);
  const entropy = {};
  const crossEntropy = {};
  const compression = {};
  for (const [name, stream] of Object.entries(streams)) {
    entropy[name] = conditionalEntropies(pooled(stream), k);
    crossEntropy[name] = heldOutCrossEntropy(stream, k);
    compression[name] = await compressedBitsPerToken(pooled(stream));
  }
  const all = pooled(tokens);
  const longest = Math.max(...all.map((s) => s.length));
  const lags = Array.from({ length: Math.min(MAX_LAG, longest - 1) }, (_, i) => i + 1);
  return {
    centroids,
    streams,
    sizes: mapSplits(tokens, (segments) => ({ segments: segments.length, frames: segments.reduce((n, s) => n + s.length, 0), targets: targetCount(segments) })),
    nTokens: all.reduce((n, s) => n + s.length, 0),
    used: new Set(all.flatMap((s) => [...s])).size,
    entropy,
    crossEntropy,
    compression,
    control: extra.length
      ? { segments: extra, targets: targetCount(extra), crossEntropy: ngramBits(fitNgram(tokens, k), extra), profile: controlProfile(extra, tokens.train, centroids, k, dim) }
      : null,
    zipf: {
      unigram: { real: rankFrequency(all, 1, k), shuffled: rankFrequency(pooled(streams.shuffled), 1, k) },
      bigram: { real: rankFrequency(all, 2, k), shuffled: rankFrequency(pooled(streams.shuffled), 2, k) },
    },
    mi: {
      lags,
      real: lags.map((d) => mutualInformation(all, k, d)),
      shuffled: lags.map((d) => mutualInformation(pooled(streams.shuffled), k, d)),
    },
  };
}
