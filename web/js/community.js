// Community results: validation of submitted result cards, and replication summaries.
// Used by tools/ingest_result.mjs (in a GitHub Action) and by the Live Lab's Work units tab.

import { FAMILIES, REAL_FAMILIES } from './signals.js';
import { MAX_UNIT, RESULT_CARD_VERSION, ROSETTA_CORPUS, SPEC, workUnitConfig } from './workunits.js';

export const RESULTS_FORMAT = 'eiathome-community-results';
export const MAX_CARD_BYTES = 16 * 1024;
// Two runs of the same unit agree when every metric is within this absolute tolerance.
// Different backends (WebAssembly, WebGPU, WebGL) take different floating-point paths.
export const AGREEMENT_TOLERANCE = 0.02;
const BACKENDS = ['wasm', 'webgpu', 'webgl', 'cpu'];
const FAMILY_IDS = FAMILIES.map((f) => f.id);
const isNumberOrNull = (v) => v === null || Number.isFinite(v);

// GitHub refuses longer request URLs ("Whoa there! Your request URL is too long"): 8,180
// characters worked and 8,280 didn't. Stay well below, with room for a sign-in redirect.
export const MAX_ISSUE_URL = 7500;
const ISSUE_INTRO = 'Result card from the EI@Home Live Lab.';
// No backticks in the comment: extractCard reads the first json block.
const PASTE_BODY = `${ISSUE_INTRO}\n\n<!-- This card was too long to fill in automatically, so the Live Lab copied it to your clipboard. Paste it on the empty line of the json block below, then submit. -->\n\n\`\`\`json\n\n\`\`\`\n`;

// A link that opens a new result issue in repo, with the card in the body as compact JSON
// (pretty-printed, a card on the published slices is too long for a link). prefilled is false
// when even that is too long: the body then has an empty json block to paste the card into.
export function issueLink(card, repo) {
  const title = `Work unit result: ${card.spec} #${card.unit}`;
  const link = (body) => `${repo}/issues/new?${new URLSearchParams({ title, body, labels: 'work-unit' })}`;
  const url = link(`${ISSUE_INTRO}\n\n\`\`\`json\n${JSON.stringify(card)}\n\`\`\`\n`);
  return url.length <= MAX_ISSUE_URL ? { url, prefilled: true } : { url: link(PASTE_BODY), prefilled: false };
}

// The first ```json block of an issue body, parsed. Throws a readable error otherwise.
export function extractCard(body) {
  const match = /```json\s*([\s\S]*?)```/.exec(String(body ?? ''));
  if (!match) throw new Error('No ```json block with a result card was found.');
  if (new TextEncoder().encode(match[1]).length > MAX_CARD_BYTES) throw new Error(`The result card is larger than ${MAX_CARD_BYTES / 1024} KB.`);
  try {
    return JSON.parse(match[1]);
  } catch {
    throw new Error("The result card isn't valid JSON.");
  }
}

// The dataset a card used: its slices' dataset id, or null for synthetic data only. Runs of a
// unit are compared only within one dataset.
export const cardDataset = (card) => card?.inputs?.slices?.dataset_id ?? null;

// Checks a card against its spec. Returns a list of problems (empty means valid).
// published: the slices this site publishes, as { datasetId, families } (see describeDataset
// in slices.js), or null if none.
export function validateCard(card, { published = null } = {}) {
  const errors = [];
  const fail = (message) => errors.push(message);
  if (!card || typeof card !== 'object' || Array.isArray(card)) return ['The card is not a JSON object.'];
  if (card.format !== 'eiathome-result') fail('`format` must be "eiathome-result".');
  if (card.version !== RESULT_CARD_VERSION) fail(`\`version\` must be ${RESULT_CARD_VERSION}. Re-run the unit in the current Live Lab.`);
  if (card.spec !== SPEC) fail(`\`spec\` must be "${SPEC}".`);
  if (!Number.isInteger(card.unit) || card.unit < 0 || card.unit > MAX_UNIT) fail(`\`unit\` must be a whole number from 0 to ${MAX_UNIT}.`);
  else if (JSON.stringify(card.config) !== JSON.stringify(workUnitConfig(card.unit))) fail("`config` doesn't match the spec for this unit.");

  const corpus = card.inputs?.rosetta_corpus;
  if (corpus?.id !== ROSETTA_CORPUS.id || corpus?.sha256 !== ROSETTA_CORPUS.sha256) fail("The Rosetta corpus doesn't match the spec.");
  const slices = card.inputs?.slices;
  const usedPublished = slices != null && published != null && slices.dataset_id === published.datasetId;
  if (slices != null) {
    if (!published) fail('The card used real-data slices, but this site publishes none, so nobody could replicate it.');
    else if (!usedPublished) fail("The card used real-data slices that aren't the ones this site publishes, so nobody could replicate it.");
  }

  if (!BACKENDS.includes(card.backend)) fail(`\`backend\` must be one of ${BACKENDS.join(', ')}.`);
  if (typeof card.tfjs !== 'string' || !/^\d+\.\d+\.\d+$/.test(card.tfjs)) fail('`tfjs` must be a version like 4.22.0.');
  if (typeof card.app !== 'string' || card.app.length > 20) fail('`app` must be a short version string.');
  if (!Number.isInteger(card.parameters) || card.parameters <= 0) fail('`parameters` must be a positive whole number.');
  if (!Number.isFinite(card.seconds) || card.seconds < 0) fail('`seconds` must be a non-negative number.');
  if (typeof card.started !== 'string' || Number.isNaN(Date.parse(card.started))) fail('`started` must be a date.');

  const families = card.families;
  const expected = FAMILY_IDS.filter((f) => !REAL_FAMILIES.includes(f) || (usedPublished && published.families.includes(f)));
  if (!Array.isArray(families) || families.length !== expected.length || !expected.every((f) => families.includes(f))) {
    fail(`\`families\` must be exactly: ${expected.join(', ')}.`);
  }
  const m = card.metrics;
  if (!m || typeof m !== 'object') {
    fail('`metrics` is missing.');
  } else {
    for (const f of expected) {
      const g = m.prediction_gain?.[f];
      if (!g || !['persistence', 'linear', 'gru'].every((key) => Number.isFinite(g[key]))) fail(`\`metrics.prediction_gain.${f}\` needs numeric persistence, linear and gru values.`);
      if (!Number.isFinite(m.gru_gain_frames_shuffled?.[f])) fail(`\`metrics.gru_gain_frames_shuffled.${f}\` must be a number.`);
    }
    if (!Number.isFinite(m.final_train_loss)) fail('`metrics.final_train_loss` must be a number.');
    const probe = m.drift_probe;
    if (!probe || !['r2_synthetic', 'mae_hz_per_s_synthetic', 'untrained_r2_synthetic', 'untrained_mae_hz_per_s_synthetic'].every((key) => isNumberOrNull(probe[key]))) {
      fail('`metrics.drift_probe` needs numeric (or null) probe values.');
    }
  }
  return errors;
}

// Every numeric metric of a card, keyed by path, for comparing two runs.
export function metricValues(card) {
  const values = {};
  const walk = (value, path) => {
    if (Number.isFinite(value)) values[path] = value;
    else if (value && typeof value === 'object') for (const [key, inner] of Object.entries(value)) walk(inner, path ? `${path}.${key}` : key);
  };
  walk(card.metrics, '');
  return values;
}

export function agree(a, b, tolerance = AGREEMENT_TOLERANCE) {
  const va = metricValues(a);
  const vb = metricValues(b);
  const keys = Object.keys(va);
  return keys.length === Object.keys(vb).length && keys.every((key) => key in vb && Math.abs(va[key] - vb[key]) <= tolerance);
}

// Replication status per unit, and per-volunteer counts.
// results: [{ issue, submitted_by, submitted_at, card }]
export function summarize(results) {
  const byUnit = new Map();
  for (const r of results) {
    const key = `${r.card.spec}#${r.card.unit}|${cardDataset(r.card) ?? 'synthetic'}`;
    if (!byUnit.has(key)) byUnit.set(key, []);
    byUnit.get(key).push(r);
  }
  const units = [...byUnit.values()].map((runs) => {
    const volunteers = [...new Set(runs.map((r) => r.submitted_by))];
    // one run per volunteer: their first submission
    const firstRuns = volunteers.map((v) => runs.find((r) => r.submitted_by === v));
    let status = 'single';
    if (firstRuns.length > 1) status = firstRuns.every((r) => firstRuns.every((other) => agree(r.card, other.card))) ? 'replicated' : 'disagree';
    const { spec, unit, config } = runs[0].card;
    return {
      spec,
      unit,
      hidden: config.hidden,
      dataset_id: cardDataset(runs[0].card),
      runs: runs.length,
      volunteers,
      backends: [...new Set(runs.map((r) => r.card.backend))].sort(),
      status,
      issues: runs.map((r) => r.issue),
    };
  });
  units.sort((a, b) => a.unit - b.unit || (a.dataset_id ?? '').localeCompare(b.dataset_id ?? ''));

  const people = new Map();
  for (const u of units) {
    for (const v of u.volunteers) {
      const p = people.get(v) ?? { volunteer: v, units: 0, replicated: 0 };
      p.units += 1;
      if (u.status === 'replicated') p.replicated += 1;
      people.set(v, p);
    }
  }
  const leaderboard = [...people.values()].sort((a, b) => b.units - a.units || b.replicated - a.replicated || (a.volunteer < b.volunteer ? -1 : 1));
  return { units, leaderboard };
}
