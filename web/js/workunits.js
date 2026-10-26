// Work-unit definitions. A work unit is fully determined by its number, so anyone can re-run
// any unit and compare results: that's how replication works without a central server.

export const APP_VERSION = '0.1.0';
export const SPEC = 'track-a/v0';
const HIDDEN_GRID = [16, 32, 64];

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
  };
}

export const MAX_UNIT = 9999;
