// Blinded signal recovery: the Live Lab's recovery work unit (tracks/recovery, issue #13).
//
// A unit injects synthetic signals of known class and strength into real off-source windows,
// deals the windows to the detectors in shuffled order with the labels held back, and only then
// counts which were flagged, at thresholds set beforehand on a separate calibration scan. It's
// tracks/recovery/run.py's comparison on the published windows, with 18 injections per class
// and strength and a different draw for every unit. tracks/recovery/browser.py ports it to
// Python, and its tests check that both give the same signals, scores and counts.
//
// Everything is plain float64 arithmetic in a fixed order, so a unit's counts are identical on
// every machine.

import { MAX_UNIT, APP_VERSION } from './workunits.js';
import { deriveSeed, makeRng } from './rng.js';
import { sha256Hex } from './slices.js';

export const RECOVERY_SPEC = 'recovery/v0';
export const RECOVERY_CARD_VERSION = 1;
export const SIGNALS = ['tone', 'intermittent', 'wideband', 'sidebands', 'fsk'];
export const DECOYS = ['stationary', 'pulse']; // interference-like: zero drift, or one broadband frame
export const KINDS = [...SIGNALS, ...DECOYS];
export const STRENGTHS = [5, 7, 10, 14, 20, 28]; // the optimal matched-filter SNR: the signal's norm
export const RATES = ['0.01', '0.001']; // false alarms per signal-free window
const MAX_DRIFT = 1.2; // channels per frame
const MARGIN = 2; // channels kept free at each edge of the window
const PER_CELL = 18;

// The published windows (every 4th window of the test scan, in steps of 1/16 noise sigma) and
// the thresholds set on the whole calibration scan, quantized the same way: see
// web/data/recovery/index.json, written by python -m tracks.recovery.browser. They're part of
// the spec, so new windows or thresholds need a new spec.
export const RECOVERY_DATA = {
  url: 'data/recovery/test.i8',
  sha256: 'c26379abea93fc7e4e48f498b525131cb84d0df236717f7e7466e40cd4f96914',
  shape: [1024, 49, 64],
  step: 1 / 16,
  thresholds: {
    dedoppler: { '0.01': 4.9375, '0.001': 5.473214285714286 },
    energy: { '0.01': 3.6531712994267327, '0.001': 4.787324014731104 },
  },
};

// Everything that determines a unit's results besides the published windows.
export function recoveryConfig(unit) {
  return {
    spec: RECOVERY_SPEC,
    unit,
    seed: deriveSeed(unit, 'recovery'),
    per_cell: PER_CELL,
    kinds: KINDS,
    strengths: STRENGTHS,
    max_drift: MAX_DRIFT,
    margin: MARGIN,
    thresholds: RECOVERY_DATA.thresholds,
  };
}

// Python's round(): halves go to the even neighbour.
function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

// ---------- injections (tracks/recovery/inject.py) ----------

// One frame's power across channels for a line at a fractional channel, summing to 1. Narrow
// lines (width <= 1) split between the two nearest channels; wider ones are Gaussian with that
// full width at half maximum, cut off beyond 1.5 widths from the centre.
function profile(center, width, channels) {
  const out = new Float64Array(channels);
  if (width <= 1) {
    const low = Math.floor(center);
    const frac = center - low;
    out[low] += 1 - frac;
    if (frac) out[low + 1] += frac;
    return out;
  }
  const sigma = width / 2.3548;
  let sum = 0;
  for (let c = 0; c < channels; c++) {
    const distance = c - center;
    out[c] = Math.abs(distance) > 1.5 * width ? 0 : Math.exp(-0.5 * (distance / sigma) ** 2);
    sum += out[c];
  }
  for (let c = 0; c < channels; c++) out[c] /= sum;
  return out;
}

// A start channel whose track, plus extent either side, stays inside the window.
function trackStart(rng, drift, frames, channels, extent = 0) {
  const travel = drift * (frames - 1);
  const low = MARGIN + extent - Math.min(0, travel);
  const high = channels - 1 - MARGIN - extent - Math.max(0, travel);
  if (high < low) throw new Error("The track doesn't fit in the window");
  return rng.uniform(low, high);
}

// A random spec of this kind and strength, drawing exactly as inject.draw() does.
export function draw(rng, kind, strength, frames, channels) {
  const spec = { kind, strength };
  if (kind === 'none') return spec;
  if (kind === 'pulse') return { ...spec, frame: rng.int(0, frames) };
  if (kind === 'stationary') return { ...spec, drift: 0, start: trackStart(rng, 0, frames, channels), width: 1 };
  // The shape first, then a drift that keeps the whole signal inside the window.
  spec.width = 1;
  let extent = 0;
  if (kind === 'intermittent') {
    spec.period = rng.int(4, 13);
    spec.on = Math.max(1, roundHalfEven(spec.period * rng.uniform(0.25, 0.5)));
    spec.phase = rng.int(0, spec.period);
  } else if (kind === 'wideband') {
    spec.width = rng.uniform(3, 8);
    extent = 1.5 * spec.width;
  } else if (kind === 'sidebands') {
    spec.offset = rng.uniform(6, 14);
    spec.ratio = rng.uniform(0.2, 0.6);
    extent = spec.offset + 1;
  } else if (kind === 'fsk') {
    spec.symbol = rng.int(1, 5);
    spec.shift = rng.uniform(2, 6);
    spec.symbols = Array.from({ length: Math.ceil(frames / spec.symbol) }, () => rng.int(0, 2));
    extent = spec.shift + 1;
  } else if (kind !== 'tone') {
    throw new Error(`Unknown kind: ${kind}`);
  }
  const limit = Math.min(MAX_DRIFT, (channels - 1 - 2 * (MARGIN + extent)) / (frames - 1));
  spec.drift = rng.uniform(-limit, limit);
  spec.start = trackStart(rng, spec.drift, frames, channels, extent);
  return spec;
}

// The signal a spec describes, frames x channels, scaled to its strength (zero for "none").
export function render(spec, frames, channels) {
  const s = new Float64Array(frames * channels);
  if (spec.kind === 'none') return s;
  if (spec.kind === 'pulse') {
    s.fill(1, spec.frame * channels, (spec.frame + 1) * channels);
  } else {
    for (let t = 0; t < frames; t++) {
      let centre = spec.start + spec.drift * t;
      if (spec.kind === 'intermittent' && (t + spec.phase) % spec.period >= spec.on) continue;
      if (spec.kind === 'fsk') centre += spec.shift * spec.symbols[Math.floor(t / spec.symbol)];
      const row = profile(centre, spec.width, channels);
      if (spec.kind === 'sidebands') {
        for (const side of [-1, 1]) {
          const line = profile(centre + side * spec.offset, 1, channels);
          for (let c = 0; c < channels; c++) row[c] += spec.ratio * line[c];
        }
      }
      s.set(row, t * channels);
    }
  }
  let sum = 0;
  for (const v of s) sum += v * v;
  const scale = spec.strength / Math.sqrt(sum);
  for (let i = 0; i < s.length; i++) s[i] *= scale;
  return s;
}

// [kind, strength] for each of total windows: perCell of every kind and strength, and
// signal-free windows for the rest.
export function design(perCell, total) {
  const cells = KINDS.flatMap((kind) => STRENGTHS.flatMap((s) => Array.from({ length: perCell }, () => [kind, s])));
  if (cells.length > total) throw new Error(`${cells.length} injections don't fit in ${total} windows`);
  return [...cells, ...Array.from({ length: total - cells.length }, () => ['none', 0])];
}

// Injects one signal (or none) into each background window, in a random order that's
// independent of window and class. Returns the dealt windows, the only thing detectors see,
// and the key: each dealt window's spec and background window.
export function deal(backgrounds, n, frames, channels, cells, seed) {
  if (cells.length !== n) throw new Error('Need one design cell per background window');
  const rng = makeRng(seed);
  const size = frames * channels;
  const assignment = rng.permutation(n); // which design cell each background window gets
  const windows = new Float32Array(n * size);
  const key = [];
  rng.permutation(n).forEach((i, position) => {
    const [kind, strength] = cells[assignment[i]];
    const spec = draw(rng, kind, strength, frames, channels);
    const signal = render(spec, frames, channels);
    for (let j = 0; j < size; j++) windows[position * size + j] = backgrounds[i * size + j] + signal[j];
    key.push({ ...spec, background: i });
  });
  return { windows, key };
}

// ---------- detectors (tracks/recovery/detectors.py) ----------

// As numpy.arange(-maxDrift, maxDrift + 1e-9, 4 / frames): trials 4 / frames channels per
// frame apart, so a line between two trials ends at most 2 channels off over the window.
export function driftTrials(frames, maxDrift = 4) {
  const step = 4 / frames;
  const delta = -maxDrift + step - -maxDrift;
  return Array.from({ length: Math.ceil((maxDrift + 1e-9 - -maxDrift) / step) }, (_, i) => -maxDrift + i * delta);
}

// The conventional narrowband search: the strongest straight track, summed over frames and
// divided by sqrt(frames), over every drift trial and start channel. Tracks never wrap.
export function dedoppler(windows, n, frames, channels) {
  const best = new Float64Array(n).fill(-Infinity);
  for (const drift of driftTrials(frames)) {
    const offsets = Array.from({ length: frames }, (_, t) => roundHalfEven(drift * t));
    const first = -Math.min(...offsets);
    const end = channels - Math.max(...offsets);
    if (end <= first) continue; // the track leaves the window at any start
    const index = Int32Array.from(offsets, (o, t) => t * channels + o);
    for (let w = 0; w < n; w++) {
      const base = w * frames * channels;
      let top = best[w];
      for (let s = first; s < end; s++) {
        let sum = 0;
        for (let t = 0; t < frames; t++) sum += windows[base + s + index[t]];
        if (sum > top) top = sum;
      }
      best[w] = top;
    }
  }
  return best.map((v) => v / Math.sqrt(frames));
}

// A radiometer: the window's power above what unit Gaussian noise would give, in units of
// that noise's standard deviation.
export function energy(windows, n, frames, channels) {
  const size = frames * channels;
  const out = new Float64Array(n);
  for (let w = 0; w < n; w++) {
    let sum = 0;
    for (let j = w * size; j < (w + 1) * size; j++) sum += windows[j] * windows[j];
    out[w] = (sum - size) / Math.sqrt(2 * size);
  }
  return out;
}

export const DETECTORS = { dedoppler, energy };
export const DETECTOR_NAMES = { dedoppler: 'de-Doppler', energy: 'Energy' };

// ---------- the unit ----------

// The published windows, checked against the spec's SHA-256, in noise sigma.
export async function parseRecoveryWindows(buffer) {
  const [n, frames, channels] = RECOVERY_DATA.shape;
  if (buffer.byteLength !== n * frames * channels) throw new Error("The recovery windows' file size doesn't match the spec.");
  if ((await sha256Hex(buffer)) !== RECOVERY_DATA.sha256) throw new Error("The recovery windows don't match the spec's SHA-256.");
  const bytes = new Int8Array(buffer);
  const windows = new Float32Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) windows[i] = bytes[i] * RECOVERY_DATA.step;
  return { windows, n, frames, channels };
}

// Per detector and false-alarm rate: false alarms among the signal-free windows, and windows
// flagged per class and strength; per class and strength, those flagged by both detectors.
export function tally(key, scores, thresholds = RECOVERY_DATA.thresholds) {
  const names = Object.keys(thresholds);
  const results = { signal_free: key.filter((s) => s.kind === 'none').length, false_alarms: {}, recovery: {}, both: {} };
  for (const rate of RATES) {
    const flags = Object.fromEntries(names.map((name) => [name, scores[name].map((v) => v > thresholds[name][rate])]));
    const count = (flagged, kind, strength) => key.filter((s, i) => flagged(i) && s.kind === kind && (strength === undefined || s.strength === strength)).length;
    const cells = (flagged) => Object.fromEntries(KINDS.map((kind) => [kind, STRENGTHS.map((s) => count(flagged, kind, s))]));
    for (const name of names) {
      (results.false_alarms[name] ??= {})[rate] = count((i) => flags[name][i], 'none');
      (results.recovery[name] ??= {})[rate] = cells((i) => flags[name][i]);
    }
    results.both[rate] = cells((i) => names.every((name) => flags[name][i]));
  }
  return results;
}

// How many of the published windows each detector flags with nothing injected, per rate. Each
// unit's empty windows are a sample of these, so this is what combined false alarms converge to.
export function emptyWindowFlags(data, thresholds = RECOVERY_DATA.thresholds) {
  const { windows, n, frames, channels } = data;
  return Object.fromEntries(
    Object.entries(DETECTORS).map(([name, detect]) => {
      const scores = detect(windows, n, frames, channels);
      return [name, Object.fromEntries(RATES.map((rate) => [rate, scores.filter((v) => v > thresholds[name][rate]).length]))];
    }),
  );
}

// Runs unit `unit` on the published windows. The detectors see only the dealt windows; the
// key is joined to their scores afterwards.
export function runRecoveryUnit(unit, data) {
  const config = recoveryConfig(unit);
  const { n, frames, channels } = data;
  const { windows, key } = deal(data.windows, n, frames, channels, design(config.per_cell, n), config.seed);
  const scores = Object.fromEntries(Object.entries(DETECTORS).map(([name, detect]) => [name, detect(windows, n, frames, channels)]));
  return { results: tally(key, scores, config.thresholds), key, scores };
}

export function recoveryCard(unit, results, { seconds, started }) {
  return {
    format: 'eiathome-result',
    version: RECOVERY_CARD_VERSION,
    spec: RECOVERY_SPEC,
    unit,
    config: recoveryConfig(unit),
    inputs: { windows_sha256: RECOVERY_DATA.sha256 },
    app: APP_VERSION,
    seconds,
    started,
    results,
  };
}

// ---------- checking cards, and combining them ----------

function sameJSON(a, b) {
  if (Array.isArray(a)) return Array.isArray(b) && a.length === b.length && a.every((v, i) => sameJSON(v, b[i]));
  if (a && typeof a === 'object') {
    if (!b || typeof b !== 'object' || Array.isArray(b)) return false;
    const keys = Object.keys(a);
    return keys.length === Object.keys(b).length && keys.every((k) => Object.hasOwn(b, k) && sameJSON(a[k], b[k]));
  }
  return a === b;
}

const SIGNAL_FREE = RECOVERY_DATA.shape[0] - KINDS.length * STRENGTHS.length * PER_CELL;

// Checks a recovery card's form and that its counts are possible. Returns a list of problems
// (empty means valid). Ingestion also recomputes the unit: see verifyRecoveryCard.
export function validateRecoveryCard(card) {
  const errors = [];
  const fail = (message) => errors.push(message);
  if (card.format !== 'eiathome-result') fail('`format` must be "eiathome-result".');
  if (card.version !== RECOVERY_CARD_VERSION) fail(`\`version\` must be ${RECOVERY_CARD_VERSION}. Re-run the unit in the current Live Lab.`);
  if (card.spec !== RECOVERY_SPEC) fail(`\`spec\` must be "${RECOVERY_SPEC}".`);
  if (!Number.isInteger(card.unit) || card.unit < 0 || card.unit > MAX_UNIT) fail(`\`unit\` must be a whole number from 0 to ${MAX_UNIT}.`);
  else if (!sameJSON(card.config, recoveryConfig(card.unit))) fail("`config` doesn't match the spec for this unit.");
  if (card.inputs?.windows_sha256 !== RECOVERY_DATA.sha256) fail("The background windows don't match the spec's.");
  if (typeof card.app !== 'string' || card.app.length > 20) fail('`app` must be a short version string.');
  if (!Number.isFinite(card.seconds) || card.seconds < 0) fail('`seconds` must be a non-negative number.');
  if (typeof card.started !== 'string' || Number.isNaN(Date.parse(card.started))) fail('`started` must be a date.');

  const r = card.results;
  const count = (v, max) => Number.isInteger(v) && v >= 0 && v <= max;
  const row = (v) => Array.isArray(v) && v.length === STRENGTHS.length && v.every((k) => count(k, PER_CELL));
  const table = (v) => v && typeof v === 'object' && KINDS.every((kind) => row(v[kind])) && Object.keys(v).length === KINDS.length;
  const perRate = (v, ok) => v && typeof v === 'object' && Object.keys(v).length === RATES.length && RATES.every((rate) => ok(v[rate]));
  const perDetector = (v, ok) => v && typeof v === 'object' && Object.keys(v).length === Object.keys(DETECTORS).length && Object.keys(DETECTORS).every((name) => perRate(v[name], ok));
  if (!r || typeof r !== 'object' || Object.keys(r).length !== 4) return [...errors, '`results` must have signal_free, false_alarms, recovery and both.'];
  if (r.signal_free !== SIGNAL_FREE) fail(`\`results.signal_free\` must be ${SIGNAL_FREE}.`);
  if (!perDetector(r.false_alarms, (v) => count(v, SIGNAL_FREE))) fail('`results.false_alarms` needs a count per detector and false-alarm rate.');
  if (!perDetector(r.recovery, table)) fail(`\`results.recovery\` needs, per detector and false-alarm rate, ${STRENGTHS.length} counts of at most ${PER_CELL} for each class.`);
  else if (!perRate(r.both, table)) fail(`\`results.both\` needs, per false-alarm rate, ${STRENGTHS.length} counts for each class.`);
  else {
    const [a, b] = Object.keys(DETECTORS);
    const possible = RATES.every((rate) => KINDS.every((kind) => STRENGTHS.every((_, i) => {
      const [x, y, both] = [r.recovery[a][rate][kind][i], r.recovery[b][rate][kind][i], r.both[rate][kind][i]];
      return both <= Math.min(x, y) && both >= x + y - PER_CELL;
    })));
    if (!possible) fail('`results.both` must be consistent with each detector’s counts.');
  }
  return errors;
}

// Recomputes the card's unit on the published windows: [] if its results match, else why not.
export function verifyRecoveryCard(card, data) {
  return sameJSON(runRecoveryUnit(card.unit, data).results, card.results) ? [] : ["The card's results don't match a recomputation of the unit on the published windows."];
}

// 95% Wilson score interval for k successes in n trials.
export function wilson(k, n, z = 1.96) {
  if (!n) return [0, 1];
  const p = k / n;
  const centre = (p + (z * z) / (2 * n)) / (1 + (z * z) / n);
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / (1 + (z * z) / n);
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

// The counts of several units added up, one card per unit number (later cards for the same
// unit repeat its injections, so they would count them twice). Returns { units, perCell,
// signalFree, results } with results shaped like a card's.
export function combine(cards) {
  const byUnit = new Map();
  for (const card of cards) if (card.spec === RECOVERY_SPEC && !byUnit.has(card.unit)) byUnit.set(card.unit, card);
  const add = (a, b) => (Array.isArray(a) ? a.map((v, i) => v + b[i]) : typeof a === 'object' ? Object.fromEntries(Object.keys(a).map((k) => [k, add(a[k], b[k])])) : a + b);
  const units = [...byUnit.values()];
  const results = units.length ? units.map((c) => c.results).reduce(add) : null;
  return { units: units.length, perCell: units.length * PER_CELL, signalFree: units.length * SIGNAL_FREE, results };
}
