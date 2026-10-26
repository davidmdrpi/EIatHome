// Track B worker: tokenizes a family's frames with k-means and measures language-like
// statistics against surrogates. Pure JavaScript, so it runs while the engine trains.
//
// The frames are split into training, validation and test segments before anything is fitted
// (see trackb.js). How depends on the family:
// - synthetic signals: independent snippets, drawn separately for each split;
// - Rosetta: the text in order, so the test split is text the tokenizer never saw;
// - real data: one fixed central window per slice, holding out whole scans (or sessions).

import { deriveSeed, makeRng } from './rng.js';
import { C, REAL_FAMILIES, T, familyById, makeSnippet, normalise, rosettaStream } from './signals.js';
import { canonicalOrder } from './slices.js';
import { holdoutSplits } from './splits.js';
import { SPLITS, analyzeSplits } from './trackb.js';

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

// Synthetic families and Rosetta: 70% training, 10% validation, 20% test.
const FRACTIONS = { validation: 0.1, test: 0.2 };

function syntheticSplits(family, nSnippets, seed) {
  const nTest = Math.max(1, Math.round(nSnippets * FRACTIONS.test));
  const nValidation = Math.max(1, Math.round(nSnippets * FRACTIONS.validation));
  const counts = { train: nSnippets - nTest - nValidation, validation: nValidation, test: nTest };
  return Object.fromEntries(
    SPLITS.map((name) => {
      const rng = makeRng(deriveSeed(seed, `trackb-${family}-${name}`));
      return [name, Array.from({ length: counts[name] }, () => normalise(makeSnippet(family, rng).raw))];
    }),
  );
}

// One continuous transmission: contexts may span snippets, but each split is a separate part
// of the text.
function rosettaSplits(nSnippets, seed, text) {
  const { snippets, wrapped } = rosettaStream(makeRng(deriveSeed(seed, 'trackb-rosetta')), nSnippets, text);
  const frames = new Float32Array(nSnippets * T * C);
  snippets.forEach((raw, i) => frames.set(normalise(raw), i * T * C));
  const n = nSnippets * T;
  const trainEnd = Math.floor(n * (1 - FRACTIONS.validation - FRACTIONS.test));
  const validationEnd = Math.floor(n * (1 - FRACTIONS.test));
  return {
    frameSplits: {
      train: [frames.slice(0, trainEnd * C)],
      validation: [frames.slice(trainEnd * C, validationEnd * C)],
      test: [frames.slice(validationEnd * C)],
    },
    wrapped,
  };
}

// The same central C-channel window of every slice, normalized like a generated snippet.
function centralWindow(slice) {
  const first = Math.round(slice.width / 2 - C / 2);
  const raw = new Float32Array(T * C);
  for (let t = 0; t < T; t++) raw.set(slice.data.subarray(t * slice.width + first, t * slice.width + first + C), t * C);
  return normalise(raw);
}

function realSplits(family) {
  const split = holdoutSplits(slices, family);
  if (split.error) throw new Error(`${familyById[family].label}: ${split.error}`);
  // Control: the other real family's slices, without copies or windows this family also has.
  const own = new Set(slices.filter((s) => s.family === family).map((s) => s.sha256));
  const seen = new Set();
  const control = slices.filter((s) => s.family !== family && REAL_FAMILIES.includes(s.family) && !own.has(s.sha256) && !seen.has(s.sha256) && seen.add(s.sha256));
  return {
    frameSplits: Object.fromEntries(SPLITS.map((name) => [name, split[name].map(centralWindow)])),
    control: control.map(centralWindow),
    controlFamily: control.length ? control[0].family : null,
    split,
  };
}

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
    evaluation = { unit: 'text', description: 'the text in order: the first 70% for training, the next 10% for validation, the last 20% for testing' };
    if (r.wrapped) notes.push('The text is shorter than the stream, so it repeats and the test split contains training text. Use a longer text or fewer frames.');
  } else if (familyById[family].real) {
    const r = realSplits(family);
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
