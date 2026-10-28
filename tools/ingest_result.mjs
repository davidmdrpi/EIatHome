#!/usr/bin/env node
// Validates a work-unit result card submitted as a GitHub issue and, if it's valid and new,
// appends it to web/data/community_results.json. Run for each issue by tools/ingest_issues.mjs,
// which commits the result and replies on the issue.
//
// Statuses: accepted (appended), recorded (this issue's card was stored by an earlier run that
// didn't finish), duplicate, rejected, error. The comment for accepted and recorded is the same.
//
// Recovery cards (web/js/recovery.js) are also recomputed on the published windows: only a card
// whose counts match is recorded.
//
// Usage: node tools/ingest_result.mjs <event.json> <comment.md> [results.json] [slices folder] [recovery folder]
//
// The issue text is untrusted: this script only parses it as JSON and validates it. Nothing
// from the issue is executed, and the reply comment never repeats strings from the card.

import { createHash } from 'node:crypto';
import { appendFileSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { AGREEMENT_TOLERANCE, RESULTS_FORMAT, cardDataset, extractCard, isRecoveryCard, summarize, validateCard } from '../web/js/community.js';
import { RECOVERY_DATA, combine, parseRecoveryWindows, verifyRecoveryCard } from '../web/js/recovery.js';
import { describeDataset, parseSlice } from '../web/js/slices.js';

const [eventPath, commentPath, resultsPath = 'web/data/community_results.json', slicesDir = 'web/data/slices', recoveryDir = 'web/data/recovery'] = process.argv.slice(2);

const readArrayBuffer = (path) => {
  const bytes = readFileSync(path);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

async function publishedDataset() {
  const manifestPath = join(slicesDir, 'index.json');
  if (!existsSync(manifestPath)) return null;
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  const slices = await Promise.all((manifest.slices ?? []).map((entry) => parseSlice(entry, readArrayBuffer(join(slicesDir, entry.file)))));
  return describeDataset(slices);
}

const recoveryWindows = () => parseRecoveryWindows(readArrayBuffer(join(recoveryDir, RECOVERY_DATA.url.split('/').pop())));

function loadResults() {
  if (!existsSync(resultsPath)) return { format: RESULTS_FORMAT, version: 1, results: [] };
  return JSON.parse(readFileSync(resultsPath, 'utf8'));
}

const STATUS_TEXT = {
  single: "It's the first run of this unit. It counts as replicated once another volunteer's run agrees.",
  replicated: `It's replicated: runs from different volunteers agree within ${AGREEMENT_TOLERANCE} on every metric.`,
  disagree: `Runs from different volunteers disagree by more than ${AGREEMENT_TOLERANCE} on at least one metric. That's worth a look: it could be a compute-backend difference or a bug.`,
};

function finish(status, comment) {
  writeFileSync(commentPath, `${comment.trim()}\n`);
  if (process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, `status=${status}\n`);
  console.log(JSON.stringify({ status }));
}

// What the volunteer is told once their card is stored: the unit's replication status on the
// card's own dataset (runs on other datasets, such as slices published earlier, are separate).
function storedMessage(card, results) {
  if (isRecoveryCard(card)) {
    const recovery = results.filter((r) => isRecoveryCard(r.card));
    const volunteers = new Set(recovery.map((r) => r.submitted_by)).size;
    return `Thanks for running a recovery work unit! Unit ${card.unit} (\`${card.spec}\`) was recomputed on the published windows and its counts match, so it's recorded. It will appear in the Live Lab's Recovery tab shortly.\n\nThe community results now combine ${combine(recovery.map((r) => r.card)).units} unit(s) from ${volunteers} volunteer(s).`;
  }
  const unit = summarize(results).units.find((u) => u.spec === card.spec && u.unit === card.unit && u.dataset_id === cardDataset(card));
  return `Thanks for running a work unit! Unit ${card.unit} (\`${card.spec}\`, ${card.backend}) is recorded and will appear in the Live Lab's community results shortly.\n\nThis unit now has ${unit.runs} run(s) from ${unit.volunteers.length} volunteer(s). ${STATUS_TEXT[unit.status]}`;
}

async function main() {
  const { issue } = JSON.parse(readFileSync(eventPath, 'utf8'));
  const login = String(issue.user?.login ?? '');
  const record = (card) => ({ issue: issue.number, submitted_by: login, submitted_at: issue.created_at, card_sha256: createHash('sha256').update(JSON.stringify(card)).digest('hex'), card });

  // Checked first: the stored card stays valid even if the published slices changed since.
  const store = loadResults();
  const stored = store.results.find((r) => r.issue === issue.number);
  if (stored) return finish('recorded', storedMessage(stored.card, store.results));

  let card;
  try {
    card = extractCard(issue.body);
  } catch (err) {
    return finish('rejected', `Thanks for running a work unit! This issue couldn't be recorded: ${err.message}\n\nUse **Submit as a GitHub issue** in the Live Lab's Work units or Recovery tab, which fills in the card for you.`);
  }
  const errors = validateCard(card, { published: await publishedDataset() });
  if (!errors.length && isRecoveryCard(card)) errors.push(...verifyRecoveryCard(card, await recoveryWindows()));
  if (errors.length) {
    return finish('rejected', `Thanks for running a work unit! This card couldn't be recorded because it doesn't match its spec:\n\n${errors.map((e) => `- ${e}`).join('\n')}\n\nRe-run the unit in the current [Live Lab](https://davidmdrpi.github.io/EIatHome/) and submit the new card.`);
  }

  const entry = record(card);
  const sameDataset = (r) => cardDataset(r.card) === cardDataset(card);
  const copy = store.results.find((r) => r.card_sha256 === entry.card_sha256);
  if (copy) return finish('duplicate', `An identical card was already recorded from #${copy.issue}, so this one wasn't counted again.`);
  const again = store.results.find((r) => r.submitted_by === login && r.card.spec === card.spec && r.card.unit === card.unit && sameDataset(r));
  if (again) {
    const why = isRecoveryCard(card) ? 'Each unit counts once' : 'Replications count runs from different volunteers';
    return finish('duplicate', `Thanks! You already submitted unit ${card.unit} in #${again.issue}. ${why}, so this one wasn't added. Try a different unit number.`);
  }

  store.results.push(entry);
  store.results.sort((a, b) => a.issue - b.issue);
  writeFileSync(resultsPath, `${JSON.stringify(store, null, 2)}\n`);

  finish('accepted', storedMessage(card, store.results));
}

main().catch((err) => {
  finish('error', `Something went wrong while recording this card (${err.message}).`);
  process.exitCode = 1;
});
