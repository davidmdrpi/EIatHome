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
