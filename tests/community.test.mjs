import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

import { MAX_CARD_BYTES, MAX_ISSUE_URL, extractCard, issueLink, submissionText, summarize, validateCard } from '../web/js/community.js';
import { describeDataset } from '../web/js/slices.js';
import { makeCard, publishedSlices } from './fixtures.mjs';

test('a card produced by the spec validates', async () => {
  assert.deepEqual(validateCard(await makeCard(7)), []);
});

test('cards that would not replicate are rejected', async () => {
  const tampered = await makeCard(7);
  tampered.config.epochs = 50;
  assert.match(validateCard(tampered).join(' '), /config/);

  const otherText = await makeCard(7);
  otherText.inputs.rosetta_corpus.sha256 = '0'.repeat(64);
  assert.match(validateCard(otherText).join(' '), /Rosetta corpus/);

  const privateSlices = await makeCard(7, { dataset: { datasetId: 'abc', families: ['voyager'] } });
  assert.match(validateCard(privateSlices, { published: null }).join(' '), /publishes none/);
  assert.match(validateCard(privateSlices, { published: { datasetId: 'def', families: ['voyager'] } }).join(' '), /aren't the ones/);

  const brokenMetric = await makeCard(7);
  brokenMetric.metrics.prediction_gain.tone.gru = 'high';
  assert.match(validateCard(brokenMetric).join(' '), /prediction_gain\.tone/);

  const oldFormat = await makeCard(7);
  oldFormat.version = 1;
  assert.match(validateCard(oldFormat).join(' '), /version/);
});

test('a card on the published slices must cover each of their families', async () => {
  const published = { datasetId: 'abc', families: ['voyager', 'offsource'] };
  assert.deepEqual(validateCard(await makeCard(7, { dataset: published }), { published }), []);
  const missing = await makeCard(7, { dataset: { datasetId: 'abc', families: ['voyager'] } });
  assert.match(validateCard(missing, { published }).join(' '), /families.*offsource/);
});

test('result cards are read from the first json block of an issue', async () => {
  const card = await makeCard(3);
  assert.deepEqual(extractCard(`Result card.\n\n\`\`\`json\n${JSON.stringify(card)}\n\`\`\`\n`), card);
  assert.throws(() => extractCard('no card here'), /No ```json block/);
  assert.throws(() => extractCard('```json\n{not json}\n```'), /isn't valid JSON/);
  assert.throws(() => extractCard(`\`\`\`json\n"${'x'.repeat(20000)}"\n\`\`\``), /larger than/);
});

// A card as the Live Lab makes it on the published slices: every slice listed, Voyager readouts.
async function publishedCard(unit, extraSlices = 0) {
  const slices = await publishedSlices();
  const published = await describeDataset(slices);
  const card = await makeCard(unit, { dataset: published });
  const listed = slices.map((s) => ({ id: s.id, family: s.family, sha256: s.sha256, start_mjd: s.startMjd }));
  for (let i = 0; i < extraSlices; i++) listed.push({ ...listed[i % listed.length], id: `${listed[i % listed.length].id}-more-${i}` });
  card.inputs.slices.slices = listed;
  card.metrics.drift_probe.voyager = ['carrier', 'sideband', 'sideband'].map((group) => ({
    group, measured_hz_per_s: -0.4297, readout_hz_per_s: -0.3812, readout_sd: 0.0412, untrained_readout_hz_per_s: 0.0123,
  }));
  return { card, published };
}

const issueBody = (url) => new URL(url).searchParams.get('body');

test('the submit link carries a card on the published slices, and ingestion reads the same card back', async () => {
  const { card, published } = await publishedCard(4);
  const { url, prefilled } = issueLink(card, 'https://github.com/davidmdrpi/EIatHome');
  // Pretty-printed in the link, this card was over GitHub's limit ("Your request URL is too long").
  assert.ok(prefilled);
  assert.ok(url.length <= MAX_ISSUE_URL, `${url.length} characters`);
  const params = new URL(url).searchParams;
  assert.equal(params.get('title'), `Work unit result: ${card.spec} #4`);
  assert.equal(params.get('labels'), 'work-unit');
  assert.deepEqual(extractCard(params.get('body')), card);
  assert.deepEqual(validateCard(extractCard(params.get('body')), { published }), []);
});

test('a card too long for a link opens an issue with an empty json block to paste it into', async () => {
  // Ten more slices (a second session would add nine) push it over the limit.
  const { card, published } = await publishedCard(4, 10);
  const { url, prefilled } = issueLink(card, 'https://github.com/davidmdrpi/EIatHome');
  assert.equal(prefilled, false);
  assert.ok(url.length <= MAX_ISSUE_URL);
  const body = issueBody(url);
  assert.ok(!body.includes(card.inputs.slices.dataset_id), 'the card itself is not in the link');
  // Unpasted, the card is rejected with a reason; with its submission text pasted, it is read back.
  assert.throws(() => extractCard(body), /isn't valid JSON/);
  const pasted = body.replace('```json\n\n```', `\`\`\`json\n${submissionText(card)}\n\`\`\``);
  assert.notEqual(pasted, body);
  assert.deepEqual(extractCard(pasted), card);
  assert.deepEqual(validateCard(extractCard(pasted), { published }), []);
});

test('the text pasted for a large card is compact, so ingestion accepts a card it would reject pretty-printed', async () => {
  const { card } = await publishedCard(4, 60);
  const bytes = (text) => new TextEncoder().encode(text).length;
  assert.ok(bytes(JSON.stringify(card, null, 2)) > MAX_CARD_BYTES, 'pretty-printed, over the ingestion limit');
  assert.ok(bytes(submissionText(card)) < MAX_CARD_BYTES, 'compact, under it');
  const { url, prefilled } = issueLink(card, 'https://github.com/davidmdrpi/EIatHome');
  assert.equal(prefilled, false);
  // The exact issue a volunteer posts, through the ingestion script.
  const issue = (number, text) => ({ number, user: { login: 'alice' }, created_at: '2026-10-26T01:00:00Z',
    body: issueBody(url).replace('```json\n\n```', `\`\`\`json\n${text}\n\`\`\``) });
  const dir = mkdtempSync(join(tmpdir(), 'eiathome-ingest-'));
  const accepted = ingest(dir, issue(20, submissionText(card)));
  assert.equal(accepted.status, 'status=accepted', accepted.comment);
  assert.deepEqual(JSON.parse(readFileSync(join(dir, 'results.json'), 'utf8')).results[0].card, card);
  const pretty = ingest(dir, issue(21, JSON.stringify(card, null, 2)));
  assert.equal(pretty.status, 'status=rejected');
  assert.match(pretty.comment, /larger than 16 KB/);
});

test('replication counts volunteers whose runs agree', async () => {
  const run = async (issue, by, unit, shift = 0) => ({ issue, submitted_by: by, card: await makeCard(unit, { shift }) });
  const { units, leaderboard } = summarize([
    await run(1, 'alice', 4),
    await run(2, 'bob', 4, 0.01), // within tolerance
    await run(3, 'alice', 4, 0.5), // alice again: only her first run counts
    await run(4, 'carol', 5),
    await run(5, 'dave', 5, 0.1), // disagrees with carol
    await run(6, 'erin', 6),
  ]);
  const status = Object.fromEntries(units.map((u) => [u.unit, u.status]));
  assert.deepEqual(status, { 4: 'replicated', 5: 'disagree', 6: 'single' });
  assert.equal(units.find((u) => u.unit === 4).volunteers.length, 2);
  assert.deepEqual(leaderboard.find((p) => p.volunteer === 'alice'), { volunteer: 'alice', units: 1, replicated: 1 });
});

function ingest(dir, issue) {
  const event = join(dir, `event-${issue.number}.json`);
  const comment = join(dir, 'comment.md');
  const output = join(dir, 'output.txt');
  writeFileSync(event, JSON.stringify({ issue }));
  writeFileSync(output, '');
  const run = spawnSync(process.execPath, ['tools/ingest_result.mjs', event, comment, join(dir, 'results.json'), 'web/data/slices'], {
    env: { ...process.env, GITHUB_OUTPUT: output },
    encoding: 'utf8',
  });
  return { code: run.status, status: readFileSync(output, 'utf8').trim(), comment: readFileSync(comment, 'utf8') };
}

const issueWith = async (number, login, unit, shift = 0, started, dataset) => ({
  number,
  user: { login },
  created_at: '2026-10-26T01:00:00Z',
  body: `Result card from the EI@Home Live Lab.\n\n\`\`\`json\n${JSON.stringify(await makeCard(unit, { shift, started, dataset }), null, 2)}\n\`\`\`\n`,
});

test('the ingestion script records valid cards once and explains rejections', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'eiathome-ingest-'));
  const first = ingest(dir, await issueWith(10, 'alice', 4));
  assert.equal(first.status, 'status=accepted');
  assert.match(first.comment, /1 run\(s\) from 1 volunteer/);

  const again = ingest(dir, await issueWith(10, 'alice', 4));
  assert.equal(again.status, 'status=recorded', 'same issue: stored by an earlier run');
  assert.match(again.comment, /Unit 4 .* is recorded[\s\S]*1 run\(s\) from 1 volunteer/);
  assert.equal(ingest(dir, await issueWith(11, 'mallory', 4)).status, 'status=duplicate', 'identical card copied by someone else');
  assert.match(ingest(dir, await issueWith(12, 'alice', 4, 0, '2026-10-27T00:00:00.000Z')).comment, /already submitted unit 4/);

  const second = ingest(dir, await issueWith(13, 'bob', 4, 0.01, '2026-10-27T00:00:00.000Z'));
  assert.equal(second.status, 'status=accepted');
  assert.match(second.comment, /replicated/);

  const bad = await issueWith(14, 'carol', 4);
  bad.body = bad.body.replace('"epochs": 12', '"epochs": 99');
  const rejected = ingest(dir, bad);
  assert.equal(rejected.status, 'status=rejected');
  assert.match(rejected.comment, /config/);

  const stored = JSON.parse(readFileSync(join(dir, 'results.json'), 'utf8'));
  assert.deepEqual(stored.results.map((r) => [r.issue, r.submitted_by]), [[10, 'alice'], [13, 'bob']]);
});

test('the reply describes the accepted card’s own dataset, not an older one for the same unit', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'eiathome-ingest-'));
  const older = { datasetId: '0'.repeat(64), families: ['voyager'] };
  const stored = async (issue, by, shift) => ({ issue, submitted_by: by, submitted_at: '2026-10-01T00:00:00Z', card_sha256: String(issue), card: await makeCard(4, { shift, dataset: older }) });
  writeFileSync(join(dir, 'results.json'), JSON.stringify({ format: 'eiathome-community-results', version: 1, results: [await stored(1, 'alice', 0), await stored(2, 'bob', 0.01)] }));

  const current = await describeDataset(await publishedSlices());
  const reply = ingest(dir, await issueWith(30, 'carol', 4, 0, undefined, current));
  assert.equal(reply.status, 'status=accepted');
  assert.match(reply.comment, /1 run\(s\) from 1 volunteer/);
  assert.match(reply.comment, /first run of this unit/);

  const units = summarize(JSON.parse(readFileSync(join(dir, 'results.json'), 'utf8')).results).units;
  assert.deepEqual(units.map((u) => [u.dataset_id, u.runs, u.status]), [[older.datasetId, 2, 'replicated'], [current.datasetId, 1, 'single']]);
});
