// Track A model families and their parameter counts, so different architectures can be
// compared at (nearly) the same size. Pure arithmetic: the engine builds the actual models.

import { C, T } from './signals.js';

export const ARCHITECTURES = [
  { id: 'gru', label: 'GRU' },
  { id: 'lstm', label: 'LSTM' },
  { id: 'attention', label: 'Causal attention' },
];

export const architectureLabel = (id) => ARCHITECTURES.find((a) => a.id === id)?.label ?? id;

// Parameters for `size` hidden units (recurrent) or model width (attention), including the
// input projection and the output head that predicts the next C-channel frame.
export function parameterCount(arch, size) {
  const head = size * C + C;
  if (arch === 'gru') return 3 * size * (C + size + 1) + head;
  if (arch === 'lstm') return 4 * size * (C + size + 1) + head;
  if (arch === 'attention') {
    const d = size;
    // input projection, position table, two layer norms, Q/K/V/O, feed-forward (d -> 2d -> d)
    return C * d + d + T * d + 2 * d + 4 * (d * d + d) + 2 * d + (2 * d * d + 2 * d) + (2 * d * d + d) + head;
  }
  throw new Error(`Unknown architecture "${arch}"`);
}

// The size whose parameter count is closest to a GRU with `gruHidden` units.
export function matchedSize(arch, gruHidden) {
  if (arch === 'gru') return gruHidden;
  const target = parameterCount('gru', gruHidden);
  let best = 1;
  for (let size = 1; size <= 4 * gruHidden + 16; size++) {
    if (Math.abs(parameterCount(arch, size) - target) < Math.abs(parameterCount(arch, best) - target)) best = size;
  }
  return best;
}

// Track B token models: next-token logits over k token types, from one-hot tokens, over at
// most `window` tokens. The GRU has a bias-free 16-dimensional embedding and 32 hidden units;
// the transformer is one block of Track A's causal attention at width `size`.
export const TOKEN_ARCHITECTURES = [
  { id: 'gru', label: 'GRU' },
  { id: 'transformer', label: 'Transformer' },
];

export const tokenArchitectureLabel = (id) => TOKEN_ARCHITECTURES.find((a) => a.id === id)?.label ?? id;

export const TOKEN_GRU_UNITS = 32;
const TOKEN_EMBEDDING = 16;

export function tokenParameterCount(arch, size, k, window) {
  const head = size * k + k;
  if (arch === 'gru') return k * TOKEN_EMBEDDING + 3 * size * (TOKEN_EMBEDDING + size + 1) + head;
  if (arch === 'transformer') {
    const d = size;
    // token projection, position table, two layer norms, Q/K/V/O, feed-forward (d -> 2d -> d)
    return k * d + d + window * d + 2 * d + 4 * (d * d + d) + 2 * d + (2 * d * d + 2 * d) + (2 * d * d + d) + head;
  }
  throw new Error(`Unknown token architecture "${arch}"`);
}

// The transformer width whose parameter count is closest to the token GRU's.
export function matchedTokenWidth(k, window) {
  const target = tokenParameterCount('gru', TOKEN_GRU_UNITS, k, window);
  let best = 1;
  for (let d = 1; d <= 4 * TOKEN_GRU_UNITS; d++) {
    if (Math.abs(tokenParameterCount('transformer', d, k, window) - target) < Math.abs(tokenParameterCount('transformer', best, k, window) - target)) best = d;
  }
  return best;
}
