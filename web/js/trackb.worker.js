// Track B worker: tokenizes a family's frames with k-means and measures language-like
// statistics against surrogates. Pure JavaScript, so it runs while the engine trains.
// The frames are split into training, validation and test segments before anything is fitted
// (see trackb-splits.js and trackb.js).

import { deriveSeed, makeRng } from './rng.js';
import { C, T, familyById } from './signals.js';
import { canonicalOrder } from './slices.js';
import { analyzeSplits } from './trackb.js';
import { realSplits, rosettaSplits, syntheticSplits } from './trackb-splits.js';

let slices = [];

self.onmessage = async ({ data: msg }) => {
  if (msg.type === 'slices') {
    slices = canonicalOrder(msg.slices); // sampling must not depend on manifest order
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
  const nSnippets = Math.ceil(nFrames / T);
  const notes = [];
  let frameSplits;
  let control = [];
  let controlFamily = null;
  let evaluation;
  if (family === 'rosetta') {
    const r = rosettaSplits(nSnippets, seed, text);
    frameSplits = r.frameSplits;
    evaluation = { unit: 'text', description: 'the text in order, about 70% for training, 10% for validation and 20% for testing, cut between snippets' };
    if (r.wrapped) notes.push('The text is shorter than the stream, so it repeats and the test split contains training text. Use a longer text or fewer frames.');
  } else if (familyById[family].real) {
    const r = realSplits(slices, family);
    ({ frameSplits, control, controlFamily } = r);
    const { unit, units, independent } = r.split;
    evaluation = {
      unit,
      independent,
      description: independent
        ? `whole observing sessions (${units} sessions)`
        : `whole scans (${units} scans, all from one observing session)`,
    };
    notes.push(`Real data: one fixed window per slice, holding out whole ${unit}s.`);
    if (!independent) notes.push('The scans share an observing session, so this is a within-session holdout. It doesn’t show that structure carries over to other sessions.');
  } else {
    frameSplits = syntheticSplits(family, nSnippets, seed);
    evaluation = { unit: 'snippet', description: 'independent snippets drawn separately for each split' };
    notes.push('Snippets are independent, so structure can only span 16 tokens.');
  }

  const rng = makeRng(deriveSeed(seed, `trackb-${family}`));
  const result = await analyzeSplits(frameSplits, { k, dim: C, rng, control });
  if (result.sizes.test.targets < 200) notes.push(`Only ${result.sizes.test.targets} test tokens are scored, so treat small differences as noise.`);
  if (result.control) {
    result.control.family = controlFamily;
    notes.push(`The ${familyById[controlFamily].label} row scores that family’s frames with this family’s tokenizer and models, as a control.`);
  }
  return {
    family,
    k,
    ...result,
    evaluation,
    preview: Array.from(result.streams.real.train.flatMap((s) => [...s]).slice(0, 48)),
    notes,
  };
}
