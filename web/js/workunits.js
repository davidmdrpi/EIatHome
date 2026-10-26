// Work-unit definitions. A work unit is fully determined by its spec and number, so anyone can
// re-run any unit and compare results: that's how replication works without a central server.

import { ROSETTA_TEXT } from './signals.js';

export const APP_VERSION = '0.2.0';
export const RESULT_CARD_VERSION = 2;
export const SPEC = 'track-a/v0';
const HIDDEN_GRID = [16, 32, 64];

// The Rosetta text is part of the spec, not a user setting: work units always evaluate on
// this corpus, whatever custom text the Live Lab is using. The engine checks the hash before
// running, so an edit to the built-in text can't silently change the spec.
export const ROSETTA_CORPUS = {
  id: 'en-gettysburg-bliss+us-preamble/v1',
  text: ROSETTA_TEXT,
  sha256: 'e2369a7f60ca396a402206a1f9b8d5d8bf2c375069debcc8527422552cc8e854',
};

// track-a/v0 replicates the Hello, Voyager notebook in the browser: train on noise and
// synthetic drifting tones only, then evaluate on every family, including Voyager if loaded.
export function workUnitConfig(unit) {
  return {
    spec: SPEC,
    unit,
    seed: unit,
    target: 'tone',
    mixNoise: true,
    shuffle: false,
    hidden: HIDDEN_GRID[unit % HIDDEN_GRID.length],
    epochs: 12,
    nTrain: 1200,
    batch: 64,
    lr: 3e-3,
    rosettaCorpus: ROSETTA_CORPUS.id,
  };
}

export const MAX_UNIT = 9999;
