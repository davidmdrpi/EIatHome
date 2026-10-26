#!/usr/bin/env node
// Records work-unit result issues: validates each card with tools/ingest_result.mjs, commits
// accepted cards to main, replies on the issue and labels it. Run from the repository root by
// .github/workflows/ingest-results.yml, with GH_TOKEN and GITHUB_REPOSITORY set.
//
// Usage: node tools/ingest_issues.mjs <issue number>...   record these issues
//        node tools/ingest_issues.mjs --pending          record every open result issue that
//                                                        isn't recorded or rejected yet
//
// --pending is the retry path. It picks up issues whose run was cancelled (a full concurrency
// queue), failed, or never started, and issues labelled work-unit-pending because saving failed.
// Issues already labelled work-unit or work-unit-rejected are skipped, so re-running is safe.
// An issue is labelled work-unit only after its card is on main and a Pages deploy was
// requested, so a run that fails after the push leaves the publication to the next run.
//
// The issue text is untrusted. It is only written to a file for ingest_result.mjs to parse as
// JSON. gh and git run without a shell, with issue numbers and fixed strings as arguments.
//
// It resets the working tree to origin/main before each issue, so it only runs in CI
// (GITHUB_ACTIONS=true), never in a working copy with changes to lose.

import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const TITLE_PREFIX = 'Work unit result:';
const RESULTS = 'web/data/community_results.json';
const SLICES = 'web/data/slices';
const LABELS = {
  recorded: ['work-unit', '1D76DB', 'Work-unit result card'],
  rejected: ['work-unit-rejected', 'D93F0B', 'Result card that could not be recorded'],
  pending: ['work-unit-pending', 'FBCA04', 'Valid result card waiting to be saved; retried automatically'],
};
const INGEST = fileURLToPath(new URL('./ingest_result.mjs', import.meta.url));
const SAVE_ATTEMPTS = 3;

const repo = process.env.GITHUB_REPOSITORY;
const work = mkdtempSync(join(tmpdir(), 'eiathome-ingest-'));
const run = (command, args) => execFileSync(command, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'] });
const tryRun = (command, args) => spawnSync(command, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'] }).status === 0;
const gh = (...args) => run('gh', args);
const git = (...args) => run('git', args);

const labelsOf = (issue) => (issue.labels ?? []).map((l) => (typeof l === 'string' ? l : l.name));
const isOpenResult = (issue) => issue.state.toLowerCase() === 'open' && !issue.pull_request && String(issue.title).startsWith(TITLE_PREFIX);
const isSettled = (issue) => labelsOf(issue).some((l) => l === LABELS.recorded[0] || l === LABELS.rejected[0]);

function pendingIssues() {
  const open = JSON.parse(gh('issue', 'list', '--repo', repo, '--state', 'open', '--limit', '500', '--json', 'number,title,labels,state'));
  return open.filter((i) => isOpenResult(i) && !isSettled(i)).map((i) => i.number);
}

// Validates the card against main as it is now. Returns { status, comment }.
function ingest(issue) {
  const event = join(work, 'event.json');
  const comment = join(work, 'comment.md');
  writeFileSync(event, JSON.stringify({ issue }));
  writeFileSync(comment, '');
  const env = { ...process.env, GITHUB_OUTPUT: '' };
  const result = spawnSync(process.execPath, [INGEST, event, comment, RESULTS, SLICES], { encoding: 'utf8', env, stdio: ['ignore', 'pipe', 'inherit'] });
  let status = 'error';
  try {
    status = JSON.parse(result.stdout.trim().split('\n').pop()).status ?? 'error';
  } catch {
    // the script crashed before reporting a status
  }
  return { status, comment: readFileSync(comment, 'utf8').trim() || 'Something went wrong while recording this card.' };
}

// Commits the accepted card and pushes it to main. If main moved on, starts again from the new
// main, so the card is validated against (and appended to) the latest results.
function ingestAndSave(issue) {
  for (let attempt = 1; ; attempt++) {
    git('fetch', '--quiet', 'origin', '+refs/heads/main:refs/remotes/origin/main');
    git('reset', '--quiet', '--hard', 'origin/main');
    const outcome = ingest(issue);
    if (outcome.status !== 'accepted') return outcome;
    git('add', RESULTS);
    git('commit', '--quiet', '-m', `Add work-unit result from #${issue.number}`);
    if (tryRun('git', ['push', '--quiet', 'origin', 'HEAD:main'])) return outcome;
    git('reset', '--quiet', '--hard', 'origin/main');
    if (attempt === SAVE_ATTEMPTS) return { status: 'unsaved', comment: outcome.comment };
  }
}

function reply(issue, file, text) {
  writeFileSync(file, text);
  gh('issue', 'comment', String(issue.number), '--repo', repo, '--body-file', file);
}

function label(issue, add) {
  const labels = labelsOf(issue);
  const args = ['issue', 'edit', String(issue.number), '--repo', repo, '--add-label', add];
  if (add !== LABELS.pending[0] && labels.includes(LABELS.pending[0])) args.push('--remove-label', LABELS.pending[0]);
  gh(...args);
}

// Replies, labels and (once recorded) closes the issue. Returns false if it still needs a retry.
function settle({ issue, status, comment }, published) {
  const file = join(work, `reply-${issue.number}.md`);
  if (status === 'accepted' || status === 'recorded') {
    if (!published) return false; // left open and unlabelled: the next run requests the deploy again
    reply(issue, file, comment);
    label(issue, LABELS.recorded[0]);
    gh('issue', 'close', String(issue.number), '--repo', repo, '--reason', 'completed');
    return true;
  }
  if (status === 'duplicate') {
    reply(issue, file, comment);
    label(issue, LABELS.recorded[0]);
    gh('issue', 'close', String(issue.number), '--repo', repo, '--reason', 'completed');
    return true;
  }
  if (status === 'rejected') {
    reply(issue, file, comment);
    label(issue, LABELS.rejected[0]);
    return true;
  }
  // Valid but not saved, or the script failed: retried by the next --pending run. Say so once.
  if (!labelsOf(issue).includes(LABELS.pending[0])) {
    const why = status === 'unsaved' ? 'The card is valid, but saving it failed.' : comment;
    reply(issue, file, `${why}\n\nIt will be retried automatically, so there's no need to submit it again.`);
    label(issue, LABELS.pending[0]);
  }
  return false;
}

async function main() {
  if (process.env.GITHUB_ACTIONS !== 'true') throw new Error('This resets the working tree to origin/main, so it only runs in CI (GITHUB_ACTIONS=true).');
  if (!repo) throw new Error('Set GITHUB_REPOSITORY (owner/name).');
  const args = process.argv.slice(2);
  const numbers = args.includes('--pending') ? pendingIssues() : args.filter((a) => /^\d+$/.test(a)).map(Number);
  for (const [name, color, description] of Object.values(LABELS)) gh('label', 'create', name, '--repo', repo, '--color', color, '--description', description, '--force');

  let failed = 0;
  const outcomes = [];
  for (const number of [...new Set(numbers)].sort((a, b) => a - b)) {
    try {
      const issue = JSON.parse(gh('api', `repos/${repo}/issues/${number}`));
      if (!isOpenResult(issue) || isSettled(issue)) {
        console.log(`#${number}: skipped (not an open, unrecorded result issue)`);
        continue;
      }
      const outcome = ingestAndSave(issue);
      console.log(`#${number}: ${outcome.status}`);
      outcomes.push({ issue, ...outcome });
    } catch (err) {
      failed++;
      console.error(`#${number}: ${err.message}`);
    }
  }

  // A stored card is only announced, and its issue settled, after a Pages deploy was requested
  // with the card on main. Until then the issue stays open and unlabelled, so whichever run
  // comes next finds the card already stored ("recorded") and requests the deploy again.
  // (Pushes made with the workflow token don't trigger other workflows, hence the request.)
  let published = false;
  if (outcomes.some((o) => o.status === 'accepted' || o.status === 'recorded')) {
    try {
      gh('workflow', 'run', 'pages.yml', '--repo', repo, '--ref', 'main');
      published = true;
    } catch (err) {
      console.error(`Couldn't start the Pages deploy (${err.message}). The next run requests it again.`);
    }
  }
  for (const outcome of outcomes) {
    try {
      if (!settle(outcome, published)) failed++;
    } catch (err) {
      failed++;
      console.error(`#${outcome.issue.number}: ${err.message}`);
    }
  }
  if (failed) {
    console.error(`${failed} issue(s) aren't finished and will be retried.`);
    process.exitCode = 1;
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
