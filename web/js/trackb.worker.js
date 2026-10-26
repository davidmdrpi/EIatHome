// Track B worker: tokenizes a family's frames with k-means and measures language-like
// statistics against surrogates. Pure JavaScript, so it runs while the engine trains.

import { deriveSeed, makeRng } from './rng.js';
import { C, T, makeSnippet, normalise, rosettaStream } from './signals.js';
import {
  compressedBitsPerToken,
  conditionalEntropies,
  kmeans,
  markovSurrogate,
  mutualInformation,
  rankFrequency,
  shuffleSurrogate,
} from './trackb.js';

const MAX_LAG = 24;
let slices = [];

self.onmessage = async ({ data: msg }) => {
  if (msg.type === 'slices') {
    slices = msg.slices;
    return;
  }
  if (msg.type !== 'analyze') return;
  try {
    self.postMessage({ type: 'result', id: msg.id, ...(await analyze(msg)) });
  } catch (err) {
    self.postMessage({ type: 'error', id: msg.id, message: err?.message ?? String(err) });
  }
};

async function analyze({ family, k, nFrames, seed, text }) {
  const rng = makeRng(deriveSeed(seed, `trackb-${family}`));
  const nSnippets = Math.ceil(nFrames / T);
  const notes = [];
  let snippets;
  if (family === 'rosetta') {
    const stream = rosettaStream(rng, nSnippets, text);
    snippets = stream.snippets;
    if (stream.wrapped) notes.push('The text is shorter than the stream, so it repeats. Repeats add artificial structure: use a longer text or fewer frames.');
  } else {
    snippets = Array.from({ length: nSnippets }, () => makeSnippet(family, rng, { slices, region: 'eval' }).raw);
    if (family === 'voyager') notes.push(`Only ${slices.length} real signals are loaded, so these snippets are shifted views of the same few signals.`);
  }
  if (family !== 'rosetta') notes.push('Snippets are independent, so structure can only span 16 tokens.');

  const n = nSnippets * T;
  const frames = new Float32Array(n * C);
  snippets.forEach((raw, i) => frames.set(normalise(raw), i * T * C));
  const tokens = kmeans(frames, n, C, k, rng).labels;

  const streams = {
    real: tokens,
    shuffled: shuffleSurrogate(tokens, rng),
    markov: markovSurrogate(tokens, k, rng),
  };
  const entropy = {};
  const compression = {};
  for (const [name, stream] of Object.entries(streams)) {
    entropy[name] = conditionalEntropies(stream, k);
    compression[name] = await compressedBitsPerToken(stream);
  }
  const lags = Array.from({ length: MAX_LAG }, (_, i) => i + 1);
  return {
    family,
    k,
    nTokens: n,
    used: new Set(tokens).size,
    preview: Array.from(tokens.subarray(0, 48)),
    notes,
    entropy,
    compression,
    zipf: {
      unigram: { real: rankFrequency(streams.real, 1, k), shuffled: rankFrequency(streams.shuffled, 1, k) },
      bigram: { real: rankFrequency(streams.real, 2, k), shuffled: rankFrequency(streams.shuffled, 2, k) },
    },
    mi: {
      lags,
      real: lags.map((d) => mutualInformation(streams.real, k, d)),
      shuffled: lags.map((d) => mutualInformation(streams.shuffled, k, d)),
    },
  };
}
