// Track B: turn frames into "radio tokens" and measure how language-like the token stream is.
// Every statistic is reported next to surrogates that destroy specific kinds of structure.

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

function ngramCounts(tokens, n, k) {
  const counts = new Map();
  for (let i = 0; i + n <= tokens.length; i++) {
    let key = 0;
    for (let j = 0; j < n; j++) key = key * k + tokens[i + j];
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return counts;
}

// Block entropy H_n in bits, with the Miller-Madow bias correction.
function blockEntropy(tokens, n, k) {
  const counts = ngramCounts(tokens, n, k);
  const total = tokens.length - n + 1;
  let h = 0;
  for (const c of counts.values()) h -= (c / total) * Math.log2(c / total);
  h += (counts.size - 1) / (2 * total * Math.LN2);
  return { h, distinct: counts.size, total };
}

// Conditional entropies h_n = H_n - H_(n-1): the uncertainty of the next token given the
// previous n-1. "reliable" is false when the average observed n-gram appears fewer than 5 times.
export function conditionalEntropies(tokens, k, maxOrder = 4) {
  const out = [];
  let previous = 0;
  for (let n = 1; n <= maxOrder; n++) {
    const { h, distinct, total } = blockEntropy(tokens, n, k);
    out.push({ n, h: h - previous, reliable: total / distinct >= 5 });
    previous = h;
  }
  return out;
}

// Held-out cross-entropy (bits per token) of a smoothed n-gram model: fitted on the first
// `trainFraction` of the stream and scored on the rest. Plug-in entropies collapse when there
// are more possible contexts than tokens, because most n-grams are seen once. A held-out score
// can't be lowered by memorizing them, so a shuffled stream stays near its single-token entropy
// at every context length, and any drop means the context really predicts the next token.
//
// Smoothing is hierarchical (Dirichlet): the estimate for a context of length L backs off to
// the estimate for its last L-1 tokens, with concentration `beta`, down to a uniform 1/k.
// For each context length, beta is chosen on the last 20% of the training part, then the model
// is refitted on the whole training part.
const BETAS = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64];

function ngramTables(tokens, maxContext, k) {
  const tables = [];
  for (let L = 0; L <= maxContext; L++) {
    const joint = new Map(); // context + next token
    const context = new Map(); // context followed by any token
    for (let i = L; i < tokens.length; i++) {
      let key = 0;
      for (let j = i - L; j < i; j++) key = key * k + tokens[j];
      context.set(key, (context.get(key) ?? 0) + 1);
      const full = key * k + tokens[i];
      joint.set(full, (joint.get(full) ?? 0) + 1);
    }
    tables.push({ joint, context });
  }
  return tables;
}

function scoreBits(tables, scored, contextLength, k, beta) {
  let bits = 0;
  let n = 0;
  for (let i = contextLength; i < scored.length; i++) {
    let p = 1 / k;
    for (let L = 0; L <= contextLength; L++) {
      let key = 0;
      for (let j = i - L; j < i; j++) key = key * k + scored[j];
      const { joint, context } = tables[L];
      p = ((joint.get(key * k + scored[i]) ?? 0) + beta * p) / ((context.get(key) ?? 0) + beta);
    }
    bits -= Math.log2(p);
    n++;
  }
  return bits / n;
}

export function heldOutCrossEntropy(tokens, k, maxContext = 3, trainFraction = 0.8) {
  const split = Math.floor(tokens.length * trainFraction);
  const train = tokens.subarray(0, split);
  const test = tokens.subarray(split);
  const devSplit = Math.floor(train.length * 0.8);
  const fit = ngramTables(train.subarray(0, devSplit), maxContext, k);
  const dev = train.subarray(devSplit);
  const full = ngramTables(train, maxContext, k);
  return Array.from({ length: maxContext + 1 }, (_, L) => {
    let best = BETAS[0];
    let bestBits = Infinity;
    for (const beta of BETAS) {
      const bits = scoreBits(fit, dev, L, k, beta);
      if (bits < bestBits) {
        bestBits = bits;
        best = beta;
      }
    }
    return { context: L, bits: scoreBits(full, test, L, k, best), beta: best };
  });
}

// Rank-frequency of n-grams, normalized to frequencies (for Zipf plots).
export function rankFrequency(tokens, n, k) {
  const counts = [...ngramCounts(tokens, n, k).values()].sort((a, b) => b - a);
  const total = counts.reduce((a, b) => a + b, 0);
  return counts.map((c) => c / total);
}

// Plug-in mutual information (bits) between tokens d steps apart.
export function mutualInformation(tokens, k, d) {
  const joint = new Float64Array(k * k);
  const left = new Float64Array(k);
  const right = new Float64Array(k);
  const total = tokens.length - d;
  for (let i = 0; i < total; i++) {
    joint[tokens[i] * k + tokens[i + d]]++;
    left[tokens[i]]++;
    right[tokens[i + d]]++;
  }
  let mi = 0;
  for (let a = 0; a < k; a++) {
    for (let b = 0; b < k; b++) {
      const p = joint[a * k + b] / total;
      if (p > 0) mi += p * Math.log2(p / ((left[a] / total) * (right[b] / total)));
    }
  }
  return mi;
}

// S1: same token frequencies, no order.
export function shuffleSurrogate(tokens, rng) {
  const out = Uint16Array.from(tokens);
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng.next() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

// S2: a first-order Markov chain fitted to the tokens: keeps pair statistics, nothing longer.
export function markovSurrogate(tokens, k, rng) {
  const transitions = new Float64Array(k * k);
  const unigram = new Float64Array(k);
  for (let i = 0; i < tokens.length; i++) {
    unigram[tokens[i]]++;
    if (i + 1 < tokens.length) transitions[tokens[i] * k + tokens[i + 1]]++;
  }
  const sample = (weights, offset, size) => {
    let total = 0;
    for (let j = 0; j < size; j++) total += weights[offset + j];
    let target = rng.next() * total;
    for (let j = 0; j < size; j++) {
      target -= weights[offset + j];
      if (target <= 0) return j;
    }
    return size - 1;
  };
  const out = new Uint16Array(tokens.length);
  out[0] = tokens[0];
  for (let i = 1; i < out.length; i++) {
    const row = out[i - 1] * k;
    let rowTotal = 0;
    for (let j = 0; j < k; j++) rowTotal += transitions[row + j];
    out[i] = rowTotal ? sample(transitions, row, k) : sample(unigram, 0, k);
  }
  return out;
}

// Compressed size in bits per token, using the browser's built-in DEFLATE.
export async function compressedBitsPerToken(tokens) {
  const bytes = Uint8Array.from(tokens);
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate-raw'));
  const compressed = await new Response(stream).arrayBuffer();
  return (compressed.byteLength * 8) / tokens.length;
}
