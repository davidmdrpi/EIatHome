// Where each contribution stands, stage by stage, with the criterion that decides each stage.
// Two kinds of contribution, each read from the record that decides it:
//
// - Result cards made in this browser: Track A work units (track-a/v0), merge units
//   (trackb-merge/v0 and v1) and recovery units (recovery/v0). They carry measurements, never
//   weights. Computed here, submitted as a GitHub issue, validated and recorded by the automatic
//   check (web/data/community_results.json), then replicated by another volunteer's run of the
//   same unit, or for recovery units recomputed exactly; last, counted in the community results.
// - The checkpoint pilot's contributions (tracks/llm/PILOT.md), trained from a clone. They carry
//   weights. Submitted, validated by the coordinator's own evaluation and gate, replicated by the
//   coordinator's own run, then evaluated for merge. Read from web/data/pilot_status.json, which
//   `python -m tracks.llm.pilot export` makes from the pilot's ledger.
//
// No DOM: the Contributions tab renders what these return, and the tests run them in Node.

import { AGREEMENT_TOLERANCE, cardDataset, isRecoveryCard, metricValues, summarize, unitKey } from './community.js';
import { cardKey } from './contributions.js';
import { fixed, possessive } from './format.js';
import { isMergeCard } from './merge.js';

export const PILOT_STATUS_FORMAT = 'eiathome-pilot-status';
export const PILOT_STATUS_VERSION = 1;

// A stage is done, current (where the entry stands: the next step is someone's to take), pending
// (after the current one), failed, or skipped (doesn't apply to this entry).
export const STATE_LABELS = { done: 'Done', current: 'Now', pending: 'Later', failed: 'Failed', skipped: 'Not applicable' };

export const cardKind = (card) => (isRecoveryCard(card) ? 'recovery' : isMergeCard(card) ? 'merge' : 'lab');

const bits = (v) => fixed(v, 4);
const plural = (n, word, many = `${word}s`) => `${n.toLocaleString('en-US')} ${n === 1 ? word : many}`;
const day = (iso) => (Number.isNaN(Date.parse(iso)) ? 'an unknown date' : new Date(iso).toISOString().slice(0, 10));
const capital = (text) => text.charAt(0).toUpperCase() + text.slice(1);
const sentence = (text) => capital(text).replace(/\.?$/, '.');
const list = (items) => (items.length < 3 ? items.join(' and ') : `${items.slice(0, -1).join(', ')} and ${items.at(-1)}`);

// How far apart a unit's runs are on each metric: [{ metric, spread }], widest first. A metric
// some run lacks has an infinite spread (agree() fails on it too).
export function spreads(cards) {
  const values = cards.map(metricValues);
  const keys = [...new Set(values.flatMap(Object.keys))];
  return keys
    .map((metric) => {
      const xs = values.map((v) => v[metric]);
      return { metric, spread: xs.every(Number.isFinite) ? Math.max(...xs) - Math.min(...xs) : Infinity };
    })
    .sort((a, b) => b.spread - a.spread || (a.metric < b.metric ? -1 : 1));
}

// The community results by card text and by unit (spec, unit and dataset), with each unit's
// replication as summarize() decides it and how far its volunteers' first runs are apart.
export function communityIndex(results) {
  const byText = new Map(results.map((r) => [JSON.stringify(r.card), r]));
  const groups = new Map();
  for (const r of results.filter((r) => !isRecoveryCard(r.card))) {
    const key = unitKey(r.card);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  }
  const units = new Map();
  for (const [key, runs] of groups) {
    const [unit] = summarize(runs).units;
    const firstRuns = unit.volunteers.map((v) => runs.find((r) => r.submitted_by === v));
    const widths = spreads(firstRuns.map((r) => r.card));
    units.set(key, {
      ...unit,
      key,
      metrics: Object.keys(metricValues(runs[0].card)).length,
      spread: firstRuns.length > 1 ? Math.max(0, ...widths.map((w) => w.spread)) : null,
      over: firstRuns.length > 1 ? widths.filter((w) => w.spread > AGREEMENT_TOLERANCE) : [],
    });
  }
  const recovery = results.filter((r) => isRecoveryCard(r.card));
  return { byText, units, recovery };
}

const KIND_SPECS = { lab: 'Track A work unit', merge: 'merge unit', recovery: 'recovery unit' };

// What the automatic check (tools/ingest_result.mjs) requires of a card of this kind.
function validationCriteria(card, kind) {
  const settings = `its settings are unit ${card.unit}’s in ${card.spec}`;
  if (kind === 'recovery') {
    return `${settings}, and the check recomputes every injection and detection on the published windows: only a card whose counts match is recorded`;
  }
  if (kind === 'merge') {
    return `${settings}, its inputs are the pinned Rosetta corpus and the anchor’s θ₀, and its numbers hold together: the corpus’s split targets and unigram baseline, the payload sizes and kept share this unit’s DARE masks give, and a kept worker that is the best on validation`;
  }
  const slices = cardDataset(card) ? ', its slices the ones this site publishes' : '';
  return `${settings} (hidden size ${card.config?.hidden ?? '?'}), its inputs the pinned Rosetta corpus${slices}, and every metric is present and numeric`;
}

// A browser card's stages. marks: this browser's { [cardKey]: { saved?, opened? } }; index:
// communityIndex() of the community results.
export function cardLifecycle(card, { marks = {}, index }) {
  const kind = cardKind(card);
  const record = index.byText.get(JSON.stringify(card)) ?? null;
  const mark = marks[cardKey(card)] ?? {};
  const metrics = Object.keys(metricValues(card)).length;
  const stages = [];
  const stage = (id, name, state, text) => stages.push({ id, name, state, text: sentence(text) });

  const seconds = Number.isFinite(card.seconds) ? `, ${fixed(card.seconds, 0)} s` : '';
  stage('computed', 'Computed', 'done', `unit ${card.unit} of ${card.spec} ran in this browser on ${day(card.started)} (${card.backend ?? 'unknown backend'}${seconds})`);

  if (record) {
    stage('submitted', 'Submitted', 'done', `as issue #${record.issue}${record.submitted_by ? `, by ${record.submitted_by}` : ''}`);
  } else if (mark.opened) {
    stage('submitted', 'Submitted', 'current', `opened as a GitHub issue on ${day(mark.opened)}: it’s submitted once you post that issue. The automatic check then records it, or replies with what to fix and labels the issue work-unit-rejected`);
  } else {
    stage('submitted', 'Submitted', 'current', `not yet${mark.saved ? ' (saved to a file)' : ''}: Submit opens a GitHub issue with the card filled in, for you to post`);
  }

  const criteria = validationCriteria(card, kind);
  stage('validated', 'Validated', record ? 'done' : 'pending', record ? `recorded in the community results: ${criteria}` : `the automatic check records it only if ${criteria}`);

  let unit = null;
  if (kind === 'recovery') {
    stage('replicated', 'Recomputed', record ? 'done' : 'pending', record
      ? 'its counts were recomputed on the published windows and match exactly. Recovery units are checked by recomputation instead of a second volunteer’s run'
      : 'recovery units are checked by recomputation, exactly, instead of a second volunteer’s run');
    stage('counted', 'Combined', record ? 'done' : 'pending', record
      ? `combined with the community’s other recorded recovery units (${plural(index.recovery.length, 'unit')}) into the detection rates on the Recovery tab`
      : 'once recorded, it is combined with the other recorded recovery units into the community detection rates');
  } else {
    unit = record ? index.units.get(unitKey(card)) ?? null : null;
    const rule = `a different volunteer’s run of unit ${card.unit} on the same data agrees within ${AGREEMENT_TOLERANCE} on every metric (${plural(metrics, 'metric')} here)`;
    if (!unit) {
      stage('replicated', 'Replicated', 'pending', `it counts as replicated once ${rule}`);
      stage('counted', 'Counted', 'pending', 'replicated units are counted in the community table and in each volunteer’s tally');
    } else if (unit.status === 'single') {
      stage('replicated', 'Replicated', 'current', `${plural(unit.runs, 'run')} from one volunteer so far: it counts as replicated once ${rule}`);
      stage('counted', 'Counted', 'pending', 'listed in the community results as one volunteer’s run, and counted as replicated once it is');
    } else if (unit.status === 'replicated') {
      stage('replicated', 'Replicated', 'done', `${possessive(plural(unit.volunteers.length, 'volunteer'))} runs agree within ${AGREEMENT_TOLERANCE} on all ${plural(unit.metrics, 'metric')} (largest difference ${bits(unit.spread)})`);
      stage('counted', 'Counted', 'done', `counted as a replicated unit in the community table and in each of its ${possessive(plural(unit.volunteers.length, 'volunteer'))} tallies`);
    } else {
      const shown = unit.over.slice(0, 3).map((w) => `${w.metric} (${Number.isFinite(w.spread) ? `apart by ${bits(w.spread)}` : 'missing in a run'})`);
      const more = unit.over.length > 3 ? ` and ${plural(unit.over.length - 3, 'more')}` : '';
      stage('replicated', 'Replicated', 'failed', `${possessive(plural(unit.volunteers.length, 'volunteer'))} runs differ by more than ${AGREEMENT_TOLERANCE} on ${unit.over.length} of ${plural(unit.metrics, 'metric')}: ${list(shown)}${more}. That could be a backend difference (${unit.backends.join(', ')}) or a bug`);
      stage('counted', 'Counted', 'failed', 'listed as disagreeing, so not counted as replicated');
    }
  }
  return { kind, card, record, unit, stages, ...standing(stages) };
}

// Where an entry stands: its first stage that isn't done (or its last), and a rank for sorting
// what needs doing first: the earlier the stage it's held at, the sooner; failed after current;
// complete entries last, after any held at a later stage.
const COMPLETE = 1000;
function standing(stages) {
  const at = stages.findIndex((s) => s.state !== 'done');
  const stage = stages[at === -1 ? stages.length - 1 : at];
  const rank = at === -1 ? COMPLETE : 2 * at + (stage.state === 'failed' ? 1 : 0);
  return { stage: stage.id, state: at === -1 ? 'done' : stage.state, rank };
}

// This browser's cards, what needs doing first first: not submitted, then submitted but not
// recorded, then recorded but unreplicated, then disagreeing; complete ones last. Newest first
// within each.
export function sortLifecycles(lifecycles) {
  return [...lifecycles].sort((a, b) => a.rank - b.rank || String(b.card.started).localeCompare(String(a.card.started)));
}

// ---------- the checkpoint pilot ----------

// The problems with a status file, or none: the tab shows only a status it can read.
export function pilotStatusProblems(status) {
  if (!status || typeof status !== 'object') return ['not a JSON object'];
  const problems = [];
  if (status.format !== PILOT_STATUS_FORMAT || status.version !== PILOT_STATUS_VERSION) problems.push(`not a ${PILOT_STATUS_FORMAT} version ${PILOT_STATUS_VERSION} file`);
  for (const key of ['submissions', 'rounds', 'rollbacks', 'exclusions']) if (!Array.isArray(status[key])) problems.push(`no ${key} list`);
  if (!status.rules || !status.open || !status.current) problems.push('no rules, open round or current model');
  return problems;
}

// The round that merged a contribution into the current chain, and its entry there.
const liveRound = (status, id) => {
  for (const round of status.rounds) {
    const entry = round.live && round.contributors.find((c) => c.submission === id && c.weight > 0);
    if (entry) return { round, entry };
  }
  return null;
};

// A round's shards: how many contributions it merged on each, their training targets, and the
// weight the shard holds in all.
export function roundShards(round) {
  const merged = round.contributors.filter((c) => c.weight > 0);
  const total = [...new Map(merged.map((c) => [c.shard, c.training_targets])).values()].reduce((a, b) => a + b, 0);
  const shards = new Map();
  for (const c of merged) {
    const s = shards.get(c.shard) ?? { shard: c.shard, contributions: 0, training_targets: c.training_targets, weight: 0 };
    s.contributions += 1;
    s.weight += c.weight;
    shards.set(c.shard, s);
  }
  return { total, shards: [...shards.values()].sort((a, b) => a.shard - b.shard) };
}

// A pilot contribution's stages, from its ledger record (pilot_status.json's submissions).
export function pilotLifecycle(sub, status) {
  const { rules } = status;
  const ev = sub.evaluation;
  const rep = sub.replication;
  const stages = [];
  const stage = (id, name, state, text) => stages.push({ id, name, state, text: sentence(text) });

  const own = Number.isFinite(sub.volunteer_validation_bits) ? `; their own run scored ${bits(sub.volunteer_validation_bits)} bits on validation` : '';
  stage('submitted', 'Submitted', 'done', `issue #${sub.issue}, by ${sub.volunteer}: round ${sub.round}, shard ${sub.shard}, seed ${sub.seed}, a ${fixed(sub.bytes / 1024, 1)} KB delta artifact${own}`);

  const gate = ev.start_validation_bits + rules.gate_bits;
  const checks = 'it rebuilt the delta its digest names, named the pilot’s next round (anchor, tokenizer, dataset, split, start, an open shard and its slices, the training settings, a seed in range), counted once for its volunteer, shard and seed, and passed merge.py’s checks on the round, the start, the shard’s training targets and the bound on the delta';
  stage('validated', 'Validated', ev.status === 'validated' ? 'done' : 'failed', ev.status === 'validated'
    ? `${checks}. The coordinator scores the worker it makes at ${bits(ev.validation_bits)} bits per token on validation, against the current model’s ${bits(ev.start_validation_bits)}: within the ${rules.gate_bits}-bit gate, with ${bits(gate - ev.validation_bits)} to spare`
    : `the coordinator’s evaluation refused it: ${ev.reason}`);

  const tol = rules.replication.validation_bits;
  const cos = rules.replication.cosine;
  if (!rep) {
    stage('replicated', 'Replicated', 'skipped', 'only a validated contribution is retrained');
  } else {
    const apart = Math.abs(rep.validation_bits - ev.validation_bits);
    const run = `the coordinator retrained shard ${sub.shard} with seed ${sub.seed}: ${bits(rep.validation_bits)} bits on validation, ${bits(apart)} from the contribution (at most ${tol} allowed), and a cosine of ${fixed(rep.cosine, 3)} between the two runs’ deltas (at least ${cos} needed)`;
    const missed = [apart > tol && 'too far apart on validation', rep.cosine < cos && 'their deltas point different ways'].filter(Boolean);
    // The export decides from the ledger (the rule on the recorded numbers, and a record that holds together).
    const ok = sub.stage !== 'not replicated' && rep.replicated && !missed.length;
    stage('replicated', 'Replicated', ok ? 'done' : 'failed', ok ? run : `${run}. Not replicated: ${missed.length ? list(missed) : sub.why}`);
  }

  const min = rules.min_volunteers;
  const live = liveRound(status, sub.id);
  const ex = sub.exclusion;
  const excludedBy = ex ? `excluded from later merges by ${ex.by}: ${ex.reason}` : `excluded from later merges: ${sub.why}`;
  if (sub.stage === 'merged') {
    if (live) {
      const { total, shards } = roundShards(live.round);
      const mine = shards.find((s) => s.shard === sub.shard);
      stage('merge', 'Merged', 'done', `in round ${live.round.round}, with weight ${fixed(live.entry.weight, 3)}: shard ${sub.shard} holds its share of the round’s training targets (${mine.training_targets} of ${total}), split equally among the ${plural(mine.contributions, 'contribution')} merged on it`);
    } else {
      stage('merge', 'Merged', 'done', sub.why);
    }
    // Excluded after the merge took it: both facts hold until that round is rolled back.
    if (ex) stage('excluded', 'Excluded', 'failed', `${excludedBy}. Its weight stays in the current model until round ${live?.round.round ?? sub.round} is rolled back`);
  } else if (sub.stage === 'eligible') {
    const waiting = status.open.volunteers;
    stage('merge', 'Merge', 'current', `eligible for round ${sub.round}’s merge, which needs contributions from at least ${min} volunteers: ${waiting} so far. The merged model must pass the same ${rules.gate_bits}-bit gate against the current one`);
  } else if (sub.stage === 'not replicated') {
    stage('merge', 'Merge', 'skipped', 'not offered to a merge: only replicated contributions are');
  } else {
    const names = { excluded: 'Excluded', 'rolled back': 'Rolled back', superseded: 'Superseded', 'not merged': 'Not merged' };
    stage('merge', names[sub.stage] ?? 'Merge', 'failed', sub.stage === 'excluded' ? excludedBy : sub.why);
  }
  return { kind: 'pilot', sub, stages, ...standing(stages) };
}

// What a round decided, and why, in words.
export function roundText(round, rules) {
  const gate = round.gate_bits ?? rules.gate_bits;
  const decided = round.decision === 'accepted'
    ? `The merged model scores ${bits(round.merged_validation_bits)} bits per token on validation, against the current model’s ${bits(round.start_validation_bits)}: within the ${gate}-bit gate, so it became round ${round.round}.`
    : `Rejected${round.reason ? `: ${round.reason}` : ''}. The current model stayed as it was.`;
  const undone = round.rolled_back ? ` Rolled back to round ${round.rolled_back.to_round} by ${round.rolled_back.by}: ${round.rolled_back.reason}.` : '';
  return `Round ${round.round}, merged by ${round.by} from ${possessive(plural(round.volunteers.length, 'volunteer'))} contributions (${list(round.volunteers)}). ${decided}${undone}`;
}

// The open round's shards, the least covered first: where a contribution helps most.
export function openShards(status) {
  return [...status.open.shards].sort((a, b) => a.contributions - b.contributions || a.index - b.index);
}

// ---------- what needs doing next ----------

const UNIT_LIST = 5;

// The work that would move contributions along, most useful first: the pilot's missing shard
// coverage and volunteers, then units with one volunteer's run (unreplicated), then units whose
// runs disagree, then this browser's cards that aren't submitted. Each: { id, text }.
export function nextSteps({ index = null, pilot = null, awaiting = {} } = {}) {
  const steps = [];
  if (pilot) {
    const round = pilot.open.round;
    const empty = openShards(pilot).filter((s) => !s.contributions).map((s) => s.index);
    if (empty.length) {
      steps.push({ id: 'pilot-shards', text: `Checkpoint pilot, round ${round}: ${empty.length === 1 ? 'shard' : 'shards'} ${list(empty.map(String))} ${empty.length === 1 ? 'has' : 'have'} no contribution yet. \`python -m tracks.llm.pilot contribute\` trains the least-covered shard (tracks/llm/PILOT.md).` });
    }
    const need = pilot.rules.min_volunteers - pilot.open.volunteers;
    if (need > 0) steps.push({ id: 'pilot-volunteers', text: `Checkpoint pilot, round ${round}: a merge needs eligible contributions from ${plural(need, 'more volunteer')} (${pilot.open.volunteers} of ${pilot.rules.min_volunteers} so far).` });
  }
  if (index) {
    const units = [...index.units.values()];
    const single = units.filter((u) => u.status === 'single');
    if (single.length) {
      const names = single.slice(0, UNIT_LIST).map((u) => `${u.spec} #${u.unit}`);
      const more = single.length > UNIT_LIST ? ` and ${plural(single.length - UNIT_LIST, 'more')}` : '';
      steps.push({ id: 'unreplicated', text: `${plural(single.length, 'unit has', 'units have')} one volunteer’s run and no replication yet: ${list(names)}${more}. Running one of them yourself replicates it if your run agrees.` });
    }
    const disagree = units.filter((u) => u.status === 'disagree');
    if (disagree.length) {
      steps.push({ id: 'disagree', text: `${plural(disagree.length, 'unit’s runs disagree', 'units’ runs disagree')} by more than ${AGREEMENT_TOLERANCE}: ${list(disagree.slice(0, UNIT_LIST).map((u) => `${u.spec} #${u.unit}`))}. Another run shows which result holds.` });
    }
  }
  const unsent = Object.entries(awaiting).filter(([, n]) => n > 0);
  if (unsent.length) {
    steps.push({ id: 'unsubmitted', text: `In this browser, ${list(unsent.map(([kind, n]) => plural(n, KIND_SPECS[kind])))} ${unsent.reduce((a, [, n]) => a + n, 0) === 1 ? 'is' : 'are'} computed but not submitted.` });
  }
  return steps;
}
