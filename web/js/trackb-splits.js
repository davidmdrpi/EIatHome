// Track B's training, validation and test frames, split before anything is fitted.
//
// Each split is an array of segments (Float32Array of frames x C). Normalization is per
// snippet or slice, so a split boundary never falls inside one: no split's frames depend on
// another split's data. How the frames are split depends on the family:
// - synthetic signals: independent snippets, drawn separately for each split;
// - Rosetta: the text in order, cut between snippets, so the test split is text the tokenizer
//   never saw;
// - real data: one fixed central window per slice, holding out whole scans (or sessions).

import { deriveSeed, makeRng } from './rng.js';
import { C, REAL_FAMILIES, T, familyById, makeSnippet, normalise, rosettaStream } from './signals.js';
import { holdoutSplits } from './splits.js';
import { SPLITS } from './trackb.js';

// Synthetic families and Rosetta: 70% training, 10% validation, 20% test, in whole snippets.
function snippetCounts(nSnippets) {
  const test = Math.max(1, Math.round(nSnippets * 0.2));
  const validation = Math.max(1, Math.round(nSnippets * 0.1));
  return { train: nSnippets - test - validation, validation, test };
}

export function syntheticSplits(family, nSnippets, seed) {
  const counts = snippetCounts(nSnippets);
  return Object.fromEntries(
    SPLITS.map((name) => {
      const rng = makeRng(deriveSeed(seed, `trackb-${family}-${name}`));
      return [name, Array.from({ length: counts[name] }, () => normalise(makeSnippet(family, rng).raw))];
    }),
  );
}

// One continuous transmission: contexts may span snippets within a split, and each split is a
// separate stretch of the text.
export function rosettaSplits(nSnippets, seed, text) {
  const { snippets, wrapped } = rosettaStream(makeRng(deriveSeed(seed, 'trackb-rosetta')), nSnippets, text);
  const counts = snippetCounts(nSnippets);
  let next = 0;
  const frameSplits = {};
  for (const name of SPLITS) {
    const part = snippets.slice(next, next + counts[name]);
    next += counts[name];
    const frames = new Float32Array(part.length * T * C);
    part.forEach((raw, i) => frames.set(normalise(raw), i * T * C));
    frameSplits[name] = [frames];
  }
  return { frameSplits, wrapped };
}

// The same central C-channel window of every slice, normalized like a generated snippet.
export function centralWindow(slice) {
  const first = Math.round(slice.width / 2 - C / 2);
  const raw = new Float32Array(T * C);
  for (let t = 0; t < T; t++) raw.set(slice.data.subarray(t * slice.width + first, t * slice.width + first + C), t * C);
  return normalise(raw);
}

export function realSplits(slices, family) {
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
