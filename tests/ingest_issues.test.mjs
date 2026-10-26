// tools/ingest_issues.mjs end to end: real git against a local "origin", and a fake gh that
// keeps issues in a JSON file. Covers the retry path for cancelled runs and failed saves.
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { chmodSync, cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { makeCard } from './fixtures.mjs';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const REPO = 'volunteers/eiathome';

// Just the gh commands the driver uses.
const FAKE_GH = `#!/usr/bin/env node
const fs = require('node:fs');
const path = process.env.FAKE_GH_STATE;
const state = JSON.parse(fs.readFileSync(path, 'utf8'));
const args = process.argv.slice(2);
const opt = (name) => args[args.indexOf(name) + 1];
const view = (i, upper) => ({ ...i, state: upper ? i.state.toUpperCase() : i.state, labels: i.labels.map((name) => ({ name })) });
const [a, b] = args;
// Failure injection: state.fail lists commands ('issue comment', 'workflow', ...) to fail.
if ((state.fail ?? []).includes(a === 'issue' ? 'issue ' + b : a)) {
  console.error('fake gh: injected failure of ' + args.join(' '));
  process.exit(1);
}
if (a === 'label') {
} else if (a === 'api') {
  process.stdout.write(JSON.stringify(view(state.issues[b.split('/').pop()])));
} else if (a === 'issue' && b === 'list') {
  process.stdout.write(JSON.stringify(Object.values(state.issues).filter((i) => i.state === 'open').map((i) => view(i, true))));
} else if (a === 'issue') {
  const issue = state.issues[args[2]];
  if (b === 'comment') state.comments.push([issue.number, fs.readFileSync(opt('--body-file'), 'utf8')]);
  if (b === 'close') issue.state = 'closed';
  if (b === 'edit') {
    if (args.includes('--remove-label')) issue.labels = issue.labels.filter((l) => l !== opt('--remove-label'));
    if (args.includes('--add-label')) issue.labels.push(opt('--add-label'));
  }
} else if (a === 'workflow') {
  state.deploys += 1;
} else {
  console.error('fake gh: unsupported command ' + args.join(' '));
  process.exit(2);
}
fs.writeFileSync(path, JSON.stringify(state));
`;

// push.negotiate (if set globally) can't negotiate with a local remote and prints errors.
const git = (cwd, ...args) => execFileSync('git', ['-c', 'push.negotiate=false', ...args], { cwd, encoding: 'utf8' });

async function setup() {
  const dir = mkdtempSync(join(tmpdir(), 'eiathome-issues-'));
  const bin = join(dir, 'bin');
  mkdirSync(bin);
  writeFileSync(join(bin, 'gh'), FAKE_GH);
  chmodSync(join(bin, 'gh'), 0o755);

  const origin = join(dir, 'origin.git');
  git(dir, 'init', '--quiet', '--bare', '-b', 'main', origin);
  const seed = join(dir, 'seed');
  mkdirSync(join(seed, 'web', 'data'), { recursive: true });
  cpSync(join(ROOT, 'web', 'data', 'slices'), join(seed, 'web', 'data', 'slices'), { recursive: true });
  writeFileSync(join(seed, 'web', 'data', 'community_results.json'), JSON.stringify({ format: 'eiathome-community-results', version: 1, results: [] }));
  git(seed, 'init', '--quiet', '-b', 'main');
  git(seed, 'add', '.');
  git(seed, '-c', 'user.name=t', '-c', 'user.email=t@example.com', 'commit', '--quiet', '-m', 'seed');
  git(seed, 'push', '--quiet', origin, 'main');
  const work = join(dir, 'work');
  git(dir, 'clone', '--quiet', origin, work);
  git(work, 'config', 'user.name', 'bot');
  git(work, 'config', 'user.email', 'bot@example.com');
  git(work, 'config', 'push.negotiate', 'false'); // for the driver's own pushes

  // The fake gh's state file is the only copy: every change reads it and writes it back.
  const statePath = join(dir, 'state.json');
  writeFileSync(statePath, JSON.stringify({ issues: {}, comments: [], deploys: 0 }));
  const load = () => JSON.parse(readFileSync(statePath, 'utf8'));
  const issue = (number, login, title, card, labels = []) => {
    const state = load();
    const body = `Result card from the EI@Home Live Lab.\n\n\`\`\`json\n${JSON.stringify(card, null, 2)}\n\`\`\`\n`;
    state.issues[number] = { number, title, state: 'open', labels, user: { login }, created_at: '2026-10-26T01:00:00Z', body };
    writeFileSync(statePath, JSON.stringify(state));
  };
  const ingest = (...args) => {
    const env = { ...process.env, PATH: `${bin}:${process.env.PATH}`, GITHUB_ACTIONS: 'true', GITHUB_REPOSITORY: REPO, FAKE_GH_STATE: statePath };
    return spawnSync(process.execPath, [join(ROOT, 'tools', 'ingest_issues.mjs'), ...args], { cwd: work, env, encoding: 'utf8' });
  };
  const recorded = () => JSON.parse(git(origin, 'show', 'main:web/data/community_results.json')).results.map((r) => r.issue);
  const rejectPushes = (on) => {
    const hook = join(origin, 'hooks', 'pre-receive');
    if (on) {
      writeFileSync(hook, '#!/bin/sh\necho "push rejected by test" >&2\nexit 1\n');
      chmodSync(hook, 0o755);
    } else rmSync(hook);
  };
  const failGh = (...commands) => writeFileSync(statePath, JSON.stringify({ ...load(), fail: commands }));
  return { dir, work, origin, issue, load, ingest, recorded, rejectPushes, failGh };
}

test('result issues are recorded once, and cancelled or failed ones are retried', async () => {
  const t = await setup();
  const commentsOn = (n) => t.load().comments.filter(([number]) => number === n);
  try {
    const invalid = await makeCard(4);
    invalid.config.epochs = 99;
    await t.issue(10, 'alice', 'Work unit result: unit 4', await makeCard(4));
    await t.issue(11, 'bob', 'Work unit result: unit 4', await makeCard(4, { shift: 0.01 }), ['work-unit']); // already recorded
    await t.issue(12, 'carol', 'Work unit result: unit 4', invalid);
    await t.issue(13, 'dave', 'The Live Lab is slow on my phone', await makeCard(6));

    // A scheduled run picks up every open result issue that no run recorded (say, a cancelled one).
    const first = t.ingest('--pending');
    assert.equal(first.status, 0, first.stderr);
    let s = t.load();
    assert.deepEqual(t.recorded(), [10]);
    assert.deepEqual([s.issues[10].state, s.issues[10].labels], ['closed', ['work-unit']]);
    assert.deepEqual([s.issues[12].state, s.issues[12].labels], ['open', ['work-unit-rejected']]);
    assert.deepEqual(s.issues[11].labels, ['work-unit'], 'an issue already recorded is left alone');
    assert.deepEqual(s.issues[13].labels, [], 'other issues are left alone');
    assert.deepEqual(s.comments.map(([n]) => n), [10, 12]);
    assert.equal(s.deploys, 1);

    // Saving fails: the issue is told once that it will be retried, and nothing unpushed is left behind.
    t.rejectPushes(true);
    await t.issue(14, 'erin', 'Work unit result: unit 5', await makeCard(5));
    const failed = t.ingest('14');
    assert.equal(failed.status, 1);
    assert.deepEqual(t.load().issues[14].labels, ['work-unit-pending']);
    assert.match(commentsOn(14)[0][1], /saving it failed[\s\S]*retried automatically/);
    assert.equal(git(t.work, 'rev-parse', 'HEAD'), git(t.work, 'rev-parse', 'origin/main'));
    assert.equal(t.ingest('--pending').status, 1);
    assert.equal(commentsOn(14).length, 1, 'no repeated comment while saving keeps failing');

    // The next run after the problem is fixed records it.
    t.rejectPushes(false);
    const retried = t.ingest('--pending');
    assert.equal(retried.status, 0, retried.stderr);
    s = t.load();
    assert.deepEqual(t.recorded(), [10, 14]);
    assert.deepEqual([s.issues[14].state, s.issues[14].labels], ['closed', ['work-unit']]);
    assert.match(commentsOn(14)[1][1], /Unit 5 .* is recorded/);
    assert.equal(commentsOn(10).length, 1, 'recorded issues are never processed twice');
    assert.equal(s.deploys, 2);
  } finally {
    rmSync(t.dir, { recursive: true, force: true });
  }
});

test('the ingestion driver refuses to run outside CI, where it would reset a working copy', () => {
  const run = spawnSync(process.execPath, [join(ROOT, 'tools', 'ingest_issues.mjs'), '--pending'], { cwd: ROOT, env: { ...process.env, GITHUB_ACTIONS: '' }, encoding: 'utf8' });
  assert.equal(run.status, 1);
  assert.match(run.stderr, /only runs in CI/);
});

test('a card saved before a later step failed is still published and announced by the next run', async () => {
  const t = await setup();
  const commentsOn = (n) => t.load().comments.filter(([number]) => number === n);
  try {
    // The issue reply fails after the card reached main.
    await t.issue(20, 'alice', 'Work unit result: unit 4', await makeCard(4));
    t.failGh('issue comment');
    assert.equal(t.ingest('20').status, 1);
    assert.deepEqual(t.recorded(), [20]);
    assert.deepEqual(t.load().issues[20].labels, [], 'not settled, so the next run comes back to it');

    // Requesting the Pages deploy fails after the card reached main.
    await t.issue(21, 'bob', 'Work unit result: unit 5', await makeCard(5));
    t.failGh('workflow');
    assert.equal(t.ingest('21').status, 1);
    assert.deepEqual(t.recorded(), [20, 21]);
    assert.deepEqual([t.load().issues[21].labels, commentsOn(21).length], [[], 0], 'not announced before a deploy was requested');
    const deploysBefore = t.load().deploys;

    // Once gh works again, the scheduled run publishes both and settles them.
    t.failGh();
    const retry = t.ingest('--pending');
    assert.equal(retry.status, 0, retry.stderr);
    const s = t.load();
    assert.ok(s.deploys > deploysBefore, 'the retry requests a deploy although it added no new card');
    for (const [n, unit] of [[20, 4], [21, 5]]) {
      assert.deepEqual([s.issues[n].state, s.issues[n].labels], ['closed', ['work-unit']]);
      assert.equal(commentsOn(n).length, 1);
      assert.match(commentsOn(n)[0][1], new RegExp(`Unit ${unit} .* is recorded[\\s\\S]*1 run\\(s\\) from 1 volunteer`));
    }
    assert.deepEqual(t.recorded(), [20, 21], 'nothing is stored twice');
  } finally {
    rmSync(t.dir, { recursive: true, force: true });
  }
});
