// Where each contribution stands (web/js/lifecycle.js): browser result cards against the community
// results, and checkpoint-pilot contributions from the pilot's status file.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { AGREEMENT_TOLERANCE } from '../web/js/community.js';
import { cardKey } from '../web/js/contributions.js';
import {
  cardKind, cardLifecycle, communityIndex, nextSteps, openShards, pilotLifecycle, pilotStatusProblems, roundShards, roundText,
  sortLifecycles, spreads,
} from '../web/js/lifecycle.js';
import { MERGE_SPEC_V1 } from '../web/js/merge.js';
import { RECOVERY_SPEC } from '../web/js/recovery.js';
import { makeCard } from './fixtures.mjs';

const fixture = () => JSON.parse(readFileSync(new URL('./fixtures/pilot_status.json', import.meta.url), 'utf8'));
const published = () => JSON.parse(readFileSync(new URL('../web/data/pilot_status.json', import.meta.url), 'utf8'));
const record = (issue, by, card) => ({ issue, submitted_by: by, submitted_at: '2026-11-01T00:00:00Z', card_sha256: String(issue), card });
const states = (life) => Object.fromEntries(life.stages.map((s) => [s.id, s.state]));
const text = (life, id) => life.stages.find((s) => s.id === id).text;

test('a card made here moves from computed to submitted as the browser and the record say', async () => {
  const card = await makeCard(4);
  const index = communityIndex([]);
  const fresh = cardLifecycle(card, { index });
  assert.equal(fresh.kind, 'lab');
  assert.deepEqual(states(fresh), { computed: 'done', submitted: 'current', validated: 'pending', replicated: 'pending', counted: 'pending' });
  assert.match(text(fresh, 'computed'), /^Unit 4 of track-a\/v0 ran in this browser on 2026-10-26 \(wasm, 20 s\)\.$/);
  assert.match(text(fresh, 'submitted'), /^Not yet: Submit opens a GitHub issue/);
  // The criteria the automatic check applies, before it has.
  assert.match(text(fresh, 'validated'), /only if its settings are unit 4’s in track-a\/v0 \(hidden size \d+\), its inputs the pinned Rosetta corpus, and every metric is present and numeric\.$/);
  assert.match(text(fresh, 'replicated'), new RegExp(`a different volunteer’s run of unit 4 on the same data agrees within ${AGREEMENT_TOLERANCE} on every metric \\(\\d+ metrics here\\)`));
  assert.equal(fresh.stage, 'submitted');

  const saved = cardLifecycle(card, { index, marks: { [cardKey(card)]: { saved: '2026-11-02T00:00:00Z' } } });
  assert.match(text(saved, 'submitted'), /^Not yet \(saved to a file\)/);
  const opened = cardLifecycle(card, { index, marks: { [cardKey(card)]: { opened: '2026-11-03T10:00:00Z' } } });
  assert.equal(states(opened).submitted, 'current', 'opening an issue is a draft until it’s posted');
  assert.match(text(opened, 'submitted'), /^Opened as a GitHub issue on 2026-11-03: it’s submitted once you post that issue\..*work-unit-rejected\.$/);
});

test('a recorded card is single, replicated or disagreeing as its unit’s volunteers’ runs are', async () => {
  const mine = await makeCard(4);
  const single = communityIndex([record(10, 'alice', mine)]);
  let life = cardLifecycle(mine, { index: single });
  assert.deepEqual(states(life), { computed: 'done', submitted: 'done', validated: 'done', replicated: 'current', counted: 'pending' });
  assert.equal(text(life, 'submitted'), 'As issue #10, by alice.');
  assert.match(text(life, 'validated'), /^Recorded in the community results: its settings are unit 4’s/);
  assert.match(text(life, 'replicated'), /^1 run from one volunteer so far: it counts as replicated once a different volunteer’s run of unit 4/);

  const agreeing = communityIndex([record(10, 'alice', mine), record(11, 'bob', await makeCard(4, { shift: 0.015, backend: 'webgl' }))]);
  life = cardLifecycle(mine, { index: agreeing });
  assert.deepEqual(states(life), { computed: 'done', submitted: 'done', validated: 'done', replicated: 'done', counted: 'done' });
  assert.match(text(life, 'replicated'), /^2 volunteers’ runs agree within 0\.02 on all \d+ metrics \(largest difference 0\.0150\)\.$/);
  assert.equal(life.state, 'done');

  const apart = communityIndex([record(10, 'alice', mine), record(11, 'bob', await makeCard(4, { shift: 0.05, backend: 'webgpu' }))]);
  life = cardLifecycle(mine, { index: apart });
  assert.equal(states(life).replicated, 'failed');
  assert.equal(states(life).counted, 'failed');
  assert.match(text(life, 'replicated'), /^2 volunteers’ runs differ by more than 0\.02 on \d+ of \d+ metrics: .+ \(apart by 0\.0500\).* That could be a backend difference \(wasm, webgpu\) or a bug\.$/);
  // Runs on another dataset are another group: they don't replicate this one.
  const otherData = communityIndex([record(10, 'alice', mine), record(11, 'bob', await makeCard(4, { dataset: { datasetId: 'other', families: [] } }))]);
  assert.equal(states(cardLifecycle(mine, { index: otherData })).replicated, 'current');
});

test('spreads name each metric’s widest gap, and a metric one run lacks', () => {
  const a = { metrics: { x: 1, y: { z: 2 } } };
  const b = { metrics: { x: 1.5, y: { z: 2.01 }, w: 3 } };
  assert.deepEqual(spreads([a, b]), [{ metric: 'w', spread: Infinity }, { metric: 'x', spread: 0.5 }, { metric: 'y.z', spread: 0.009999999999999787 }]);
});

test('recovery units are recomputed and combined, and merge units replicated like Track A’s', () => {
  const recovery = { spec: RECOVERY_SPEC, unit: 3, started: '2026-10-27T00:00:00.000Z', backend: 'cpu', seconds: 0.6, results: {} };
  assert.equal(cardKind(recovery), 'recovery');
  let life = cardLifecycle(recovery, { index: communityIndex([]) });
  assert.deepEqual(life.stages.map((s) => s.name), ['Computed', 'Submitted', 'Validated', 'Recomputed', 'Combined']);
  assert.match(text(life, 'validated'), /the check recomputes every injection and detection on the published windows: only a card whose counts match is recorded\.$/);
  life = cardLifecycle(recovery, { index: communityIndex([record(20, 'alice', recovery)]) });
  assert.deepEqual(states(life), { computed: 'done', submitted: 'done', validated: 'done', replicated: 'done', counted: 'done' });
  assert.match(text(life, 'counted'), /\(1 unit\) into the detection rates on the Recovery tab\.$/);

  const merge = { spec: MERGE_SPEC_V1, unit: 2, started: '2026-10-28T00:00:00.000Z', backend: 'wasm', seconds: 60, config: {}, metrics: { theta0: 4.1, unigram: 3.2 } };
  assert.equal(cardKind(merge), 'merge');
  life = cardLifecycle(merge, { index: communityIndex([record(30, 'alice', merge)]) });
  assert.match(text(life, 'validated'), /a kept worker that is the best on validation\.$/);
  assert.match(text(life, 'replicated'), /on every metric \(2 metrics here\)\.$/);
});

test('this browser’s cards sort what needs doing first: unsubmitted, then unreplicated, then disagreeing', async () => {
  const [unsent, single, replicated, disagree] = await Promise.all([1, 2, 3, 5].map((u) => makeCard(u)));
  const index = communityIndex([
    record(10, 'alice', single),
    record(11, 'alice', replicated), record(12, 'bob', await makeCard(3, { shift: 0.01 })),
    record(13, 'alice', disagree), record(14, 'bob', await makeCard(5, { shift: 0.3 })),
  ]);
  const sorted = sortLifecycles([replicated, disagree, single, unsent].map((card) => cardLifecycle(card, { index })));
  assert.deepEqual(sorted.map((l) => l.card.unit), [1, 2, 5, 3]);
  assert.deepEqual(sorted.map((l) => l.state), ['current', 'current', 'failed', 'done']);
});

test('the pilot’s status file is read only when it is one', () => {
  assert.deepEqual(pilotStatusProblems(fixture()), []);
  assert.deepEqual(pilotStatusProblems(published()), []);
  assert.deepEqual(pilotStatusProblems({ ...fixture(), version: 2 }), ['not a eiathome-pilot-status version 1 file']);
  assert.deepEqual(pilotStatusProblems(null), ['not a JSON object']);
  assert.ok(pilotStatusProblems({ format: 'eiathome-pilot-status', version: 1 }).includes('no submissions list'));
});

test('each pilot contribution shows its gate, its replication and its merge, with the numbers', () => {
  const status = fixture();
  const by = Object.fromEntries(status.submissions.map((s) => [s.issue, pilotLifecycle(s, status)]));
  assert.deepEqual(states(by[101]), { submitted: 'done', validated: 'done', replicated: 'done', merge: 'done' });
  assert.equal(text(by[101], 'submitted'), 'Issue #101, by alice: round 2, shard 2, seed 11, a 40.0 KB delta artifact; their own run scored 1.5502 bits on validation.');
  assert.match(text(by[101], 'validated'), /The coordinator scores the worker it makes at 1\.5500 bits per token on validation, against the current model’s 1\.6200: within the 0\.02-bit gate, with 0\.0900 to spare\.$/);
  assert.equal(text(by[101], 'replicated'), 'The coordinator retrained shard 2 with seed 11: 1.5510 bits on validation, 0.0010 from the contribution (at most 0.02 allowed), and a cosine of 1.000 between the two runs’ deltas (at least 0.9 needed).');
  assert.equal(text(by[101], 'merge'), 'In round 2, with weight 0.250: shard 2 holds its share of the round’s training targets (225 of 450), split equally among the 2 contributions merged on it.');
  assert.equal(text(by[102], 'merge'), 'In round 2, with weight 0.500: shard 3 holds its share of the round’s training targets (225 of 450), split equally among the 1 contribution merged on it.');

  // A shard-2 run sent as shard 4: past the gate, but the coordinator's run of shard 4 isn't it.
  assert.deepEqual(states(by[106]), { submitted: 'done', validated: 'done', replicated: 'failed', merge: 'skipped' });
  assert.match(text(by[106], 'replicated'), /a cosine of 0\.130 .*\. Not replicated: their deltas point different ways\.$/);
  assert.equal(text(by[106], 'merge'), 'Not offered to a merge: only replicated contributions are.');

  assert.deepEqual(states(by[108]), { submitted: 'done', validated: 'done', replicated: 'done', merge: 'current' });
  assert.match(text(by[108], 'merge'), /^Eligible for round 3’s merge, which needs contributions from at least 2 volunteers: 1 so far\./);
  assert.equal(text(by[108], 'submitted'), 'Issue #108, by erin: round 3, shard 4, seed 21, a 40.0 KB delta artifact; their own run scored 1.5004 bits on validation.');
  assert.equal(states(by[109]).merge, 'failed');
  assert.equal(text(by[109], 'merge'), 'Excluded from later merges by maintainer: trained on a modified corpus.');

  // Excluded after the live round merged it: still merged, and excluded, until that round is rolled back.
  assert.deepEqual(states(by[103]), { submitted: 'done', validated: 'done', replicated: 'done', merge: 'done', excluded: 'failed' });
  assert.match(text(by[103], 'merge'), /^In round 2, with weight 0\.250:/);
  assert.equal(text(by[103], 'excluded'), 'Excluded from later merges by maintainer: withdrawn after corpus audit. Its weight stays in the current model until round 2 is rolled back.');
  assert.deepEqual([by[103].stage, by[103].state], ['excluded', 'failed'], 'not complete');
  assert.ok(by[103].rank < by[101].rank, 'listed before the complete ones');
  assert.doesNotMatch(text(by[109], 'submitted'), /their own run/, 'no score claimed, none shown');
});

test('a round shows its gate, its shard weights and a rollback', () => {
  const status = fixture();
  const [undone, live] = status.rounds;
  assert.equal(roundText(live, status.rules), 'Round 2, merged by maintainer from 3 volunteers’ contributions (alice, bob and carol). The merged model scores 1.5200 bits per token on validation, against the current model’s 1.6200: within the 0.02-bit gate, so it became round 2.');
  assert.match(roundText(undone, status.rules), / Rolled back to round 1 by maintainer: a test of rollback\.$/);
  assert.match(roundText({ ...live, decision: 'rejected', reason: 'the merged model scores 1.7000 bits on validation, worse than the incumbent’s 1.6200 by more than 0.02' }, status.rules),
    /Rejected: the merged model scores 1\.7000 bits .* The current model stayed as it was\.$/);
  const { total, shards } = roundShards(live);
  assert.equal(total, 450);
  assert.deepEqual(shards, [{ shard: 2, contributions: 2, training_targets: 225, weight: 0.5 }, { shard: 3, contributions: 1, training_targets: 225, weight: 0.5 }],
    'each shard holds its share however many contribute to it');
});

test('next steps put missing shard coverage and unreplicated units first', async () => {
  const empty = nextSteps({ pilot: published(), index: communityIndex([]) });
  assert.deepEqual(empty.map((s) => s.id), ['pilot-shards', 'pilot-volunteers']);
  assert.match(empty[0].text, /^Checkpoint pilot, round 2: shards 2, 3, 4 and 5 have no contribution yet\./);
  assert.match(empty[1].text, /needs eligible contributions from 2 more volunteers \(0 of 2 so far\)\.$/);

  const status = fixture();
  assert.deepEqual(openShards(status).map((s) => s.index), [2, 3, 5, 4], 'the least covered first');
  const cards = await Promise.all([7, 8].map((u) => makeCard(u)));
  const index = communityIndex([record(1, 'alice', cards[0]), record(2, 'alice', cards[1]), record(3, 'bob', await makeCard(8, { shift: 0.5 }))]);
  const steps = nextSteps({ pilot: status, index, awaiting: { lab: 3, recovery: 0, merge: 1 } });
  assert.deepEqual(steps.map((s) => s.id), ['pilot-shards', 'pilot-volunteers', 'unreplicated', 'disagree', 'unsubmitted']);
  assert.match(steps[0].text, /^Checkpoint pilot, round 3: shards 2, 3 and 5 have no contribution yet\./);
  assert.match(steps[1].text, /1 more volunteer \(1 of 2 so far\)/);
  assert.equal(steps[2].text, '1 unit has one volunteer’s run and no replication yet: track-a/v0 #7. Running one of them yourself replicates it if your run agrees.');
  assert.match(steps[3].text, /^1 unit’s runs disagree by more than 0\.02: track-a\/v0 #8\./);
  assert.equal(steps[4].text, 'In this browser, 3 Track A work units and 1 merge unit are computed but not submitted.');
  assert.deepEqual(nextSteps({}), []);
});
