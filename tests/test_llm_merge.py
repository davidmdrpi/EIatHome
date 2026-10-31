"""tracks/llm/merge.py: merge rounds of shard workers trained from a shared anchor (issue #20,
milestone 2): shards, round records, checks on deltas, the gate, the weighted average, DARE
payloads, TIES, the greedy soup, DiLoCo's outer step, and replaying a round."""

import copy
import itertools
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

from tracks.llm import merge, merge_bench
from tracks.llm.anchor import difference, flatten, load_anchor, weights_digest
from tracks.llm.data import load_slices, split_records
from tracks.llm.train import parser, run

ROOT = Path(__file__).resolve().parents[1]
SLICES = ROOT / 'tracks/llm/data/slices/index.json'


def worker(folder, name, shard=None, seed=42, start=None, epochs=2, family='voyager', manifest=SLICES):
    argv = ['--anchor', 'v1', '--manifest', str(manifest), '--allow-within-session', '--variants', 'real',
            '--family', family, '--epochs', str(epochs), '--seed', str(seed), '--output', str(folder / name)]
    argv += ['--shard', shard] if shard else []
    argv += ['--start', str(start)] if start else []
    run(parser().parse_args(argv))
    return folder / name / 'real.pt'


def edited(source, name, change):
    """A copy of a checkpoint with change(checkpoint) applied."""
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    change(checkpoint)
    torch.save(checkpoint, name)
    return Path(name)


class ShardTests(unittest.TestCase):
    def test_shards_are_disjoint_cover_the_split_and_ignore_manifest_order(self):
        partitions, _ = split_records(load_slices(SLICES), 'voyager', allow_within_session=True)
        train = partitions['train']
        shards = [merge.shard_records(train, i, 3) for i in (1, 2, 3)]
        self.assertEqual(sorted(r['sha256'] for s in shards for r in s), sorted(r['sha256'] for r in train))
        self.assertEqual([len(s) for s in shards], [1, 1, 1])
        self.assertEqual(merge.shard_records(list(reversed(train)), 2, 3), shards[1])
        self.assertEqual(len(merge.shard_records(train, 1, 2)) + len(merge.shard_records(train, 2, 2)), 3)
        with self.assertRaisesRegex(ValueError, '4 shards need at least 4 training slices'):
            merge.shard_records(train, 1, 4)
        for bad in ('0/3', '4/3', '2', 'a/b'):
            with self.assertRaises(ValueError, msg=bad):
                merge.parse_shard(bad)

    def test_workers_need_an_anchor_and_the_real_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = ['--manifest', str(SLICES), '--allow-within-session', '--epochs', '1', '--output', str(Path(tmp) / 'run')]
            with self.assertRaisesRegex(ValueError, 'needs --anchor'):
                run(parser().parse_args(base + ['--shard', '1/3']))
            with self.assertRaisesRegex(ValueError, '--variants must include real'):
                run(parser().parse_args(base + ['--anchor', 'v1', '--variants', 'shuffled']))


class AlgorithmTests(unittest.TestCase):
    """DARE, the payload, TIES, the greedy soup and the outer step, on synthetic deltas."""

    @classmethod
    def setUpClass(cls):
        cls.anchor = load_anchor('v1')
        cls.base = cls.anchor.spec['theta0_sha256']
        # A delta the anchor's size and shape, of the order a worker's is.
        cls.tau = merge.unflatten(cls.anchor, np.random.default_rng(0).normal(0, 0.02, 26_496))

    def test_dare_random_drop_is_unbiased(self):
        flat = np.random.default_rng(1).normal(0, 1, 2000)
        rebuilt = np.zeros((400, flat.size))
        for seed in range(400):
            kept, values = merge.dare(flat, 0.9, 'random', seed)
            np.testing.assert_allclose(values, flat[kept] * 10)  # survivors rescaled by 1/(1 − p)
            rebuilt[seed, kept] = values
        self.assertAlmostEqual(np.mean([len(np.flatnonzero(r)) for r in rebuilt]) / flat.size, 0.1, delta=0.005)
        # E[τ̃] = τ: the mean over 400 draws is within a few standard errors of every entry.
        error = (rebuilt.mean(axis=0) - flat) / (np.abs(flat) * 3 / np.sqrt(400) + 1e-12)
        self.assertLess(np.abs(error).max(), 5)
        self.assertLess(abs(rebuilt.mean(axis=0).sum() - flat.sum()), 0.05 * np.abs(flat).sum())

    def test_dare_by_magnitude_keeps_the_largest_unscaled(self):
        kept, values = merge.dare(np.array([0.1, -3.0, 0.5, 2.0, -0.2]), 0.6, 'magnitude', 0)
        self.assertEqual(kept.tolist(), [1, 3])
        self.assertEqual(values.tolist(), [-3.0, 2.0])

    def test_the_payload_round_trips_within_int8_rounding_and_fits_a_card(self):
        text = merge.encode_delta(self.anchor, self.tau, self.base, 0.9, 'random', 7)
        self.assertLessEqual(len(text.encode()), merge.CARD_BUDGET_BYTES)
        rebuilt = flatten(merge.decode_delta(self.anchor, text, self.base))
        kept, values = merge.dare(flatten(self.tau), 0.9, 'random', 7)
        dropped = np.setdiff1d(np.arange(rebuilt.size), kept)
        self.assertFalse(rebuilt[dropped].any(), 'dropped entries rebuild as zero')
        scales = np.array(json.loads(text)['scales'])
        tensor, _ = merge.tensor_of(self.anchor, kept)
        self.assertTrue((np.abs(rebuilt[kept] - values) <= scales[tensor] / 2 + 1e-12).all(), 'int8 rounds to half a step')
        # Without dropping, the same delta is far over a card.
        self.assertGreater(len(merge.encode_delta(self.anchor, self.tau, self.base, 0.0, 'random', 7)), merge.CARD_BUDGET_BYTES)

    def test_a_payload_for_another_start_or_malformed_is_refused(self):
        text = merge.encode_delta(self.anchor, self.tau, self.base, 0.9, 'random', 7)
        with self.assertRaisesRegex(ValueError, 'another anchor or start'):
            merge.decode_delta(self.anchor, text, '0' * 64)
        payload = json.loads(text)
        for change in (lambda p: p.update(kept=p['kept'] + 1), lambda p: p.update(scales=p['scales'][:-1]),
                       lambda p: p.update(values=p['values'][:-8]), lambda p: p.update(scales=[0.1] * len(p['scales']))):
            bad = copy.deepcopy(payload)
            change(bad)
            with self.assertRaisesRegex(ValueError, 'malformed'):
                merge.decode_delta(self.anchor, json.dumps(bad), self.base)

    def test_ties_elects_signs_and_averages_only_nonzero_agreeing_entries(self):
        a = np.array([1.0, -2.0, 0.1, 3.0, 2.0, 0.0])
        b = np.array([-1.0, 1.0, 0.2, 3.0, 0.0, 4.0])
        # Opposing signs that cancel elect nothing; a minority sign is dropped, not averaged in;
        # a zero entry doesn't halve the other delta's value there.
        np.testing.assert_allclose(merge.ties([a, b], [0.5, 0.5], 1.0), [0.0, -2.0, 0.15, 3.0, 2.0, 4.0])
        np.testing.assert_allclose(sum(0.5 * v for v in (a, b)), [0.0, -0.5, 0.15, 3.0, 1.0, 2.0])
        # Trimming keeps each delta's largest half by magnitude; weights tip the election.
        np.testing.assert_allclose(merge.trim(a, 0.5), [0.0, -2.0, 0.0, 3.0, 2.0, 0.0])
        np.testing.assert_allclose(merge.ties([a, b], [0.75, 0.25], 1.0), [1.0, -2.0, 0.125, 3.0, 2.0, 4.0])
        # The mean over all contributors: an agreeing entry's weighted sum over every weight, so a
        # zero or disagreeing entry counts as zero.
        np.testing.assert_allclose(merge.ties([a, b], [0.5, 0.5], 1.0, mean="all"), [0.0, -1.0, 0.15, 3.0, 1.0, 2.0])

    def test_after_dare_the_all_contributor_denominator_removes_the_extra_factor(self):
        # Three identical deltas: TIES's own mean keeps DARE's 1/(1 − p) on every survivor, while the
        # mean over all contributors keeps the expected merge at the delta itself.
        tau = np.random.default_rng(0).normal(size=4000)

        def dropped(seed):
            kept, values = merge.dare(tau, 0.9, 'random', seed)
            out = np.zeros_like(tau)
            out[kept] = values
            return out

        merged = {mean: np.mean([merge.ties([dropped(3 * draw + i) for i in range(3)], [1 / 3] * 3, 1.0, mean) for draw in range(60)], axis=0)
                  for mean in ('agreeing', 'all')}
        ratio = lambda m: float(m @ tau / (tau @ tau))
        self.assertAlmostEqual(ratio(merged['all']), 1.0, delta=0.05)
        self.assertAlmostEqual(ratio(merged['agreeing']), 10 * (1 - 0.9 ** 3), delta=0.2)

    def test_the_all_contributor_denominator_is_unbiased_only_where_signs_agree_and_nothing_is_trimmed(self):
        # The exact expectation over every DARE mask (p = 0.5, survivors doubled), entry by entry.
        def expected(deltas, mean, keep=1.0, p=0.5):
            deltas = [np.atleast_1d(np.asarray(d, dtype=np.float64)) for d in deltas]
            total = np.zeros_like(deltas[0])
            for mask in itertools.product((0, 1), repeat=sum(d.size for d in deltas)):
                kept = np.reshape(mask, (len(deltas), -1))
                chance = np.prod(np.where(kept, 1 - p, p))
                flats = [k * d / (1 - p) for k, d in zip(kept, deltas)]
                total += chance * merge.ties(flats, [1 / len(deltas)] * len(deltas), keep, mean)
            return total

        # Agreeing deltas: the weighted mean (2) in expectation, against 1.5 times it for TIES's own mean.
        np.testing.assert_allclose(expected([2.0, 2.0], 'all'), [2.0])
        np.testing.assert_allclose(expected([2.0, 2.0], 'agreeing'), [3.0])
        # Opposing deltas: sign election leans to whichever survives, so 0.75 against the mean of 0.5.
        np.testing.assert_allclose(expected([2.0, -1.0], 'all'), [0.75])
        # Identical, agreeing workers, trimmed after the drop to their larger half: the smaller entry
        # survives the trim mainly when the larger was dropped, so [2, 0.5] against [2, 1].
        np.testing.assert_allclose(expected([[2.0, 1.0], [2.0, 1.0]], 'all', keep=0.5), [2.0, 0.5])
        np.testing.assert_allclose(expected([[2.0, 1.0], [2.0, 1.0]], 'all', keep=1.0), [2.0, 1.0])

    def test_the_greedy_soup_tries_best_first_and_keeps_only_improvements(self):
        # Candidates by their own validation bits; a merge's score is looked up by its members.
        own = {'strong': 3.0, 'good': 3.1, 'near zero': 3.99, 'helps late': 3.5}
        merges = {('strong',): 3.0, ('good', 'strong'): 2.9, ('good', 'near zero', 'strong'): 3.2,
                  ('good', 'helps late', 'strong'): 2.8, ('good', 'helps late', 'near zero', 'strong'): 2.85}
        ranked = sorted(own, key=own.get)
        soup, best, trials = merge.greedy_soup(ranked, lambda members: merges[tuple(sorted(members))])
        self.assertEqual([t[0] for t in trials], ['strong', 'good', 'helps late', 'near zero'], 'best first')
        self.assertEqual(soup, ['strong', 'good', 'helps late'])
        self.assertEqual(best, 2.8)
        self.assertEqual([t[3] for t in trials], [True, True, True, False])
        # A worker that ties the soup doesn't join it: it has to improve validation.
        soup, _, _ = merge.greedy_soup(['strong', 'good'], lambda members: 3.0)
        self.assertEqual(soup, ['strong'])

    def test_the_outer_step_is_plain_merging_without_momentum_and_nesterov_with_it(self):
        state = self.anchor.theta0
        first = merge.unflatten(self.anchor, flatten(self.tau))
        second = merge.unflatten(self.anchor, -0.5 * flatten(self.tau))
        weights, momentum = merge.outer_step(state, first, None, 1.0, 0.0)
        self.assertIsNone(momentum)
        self.assertTrue(all(torch.equal(weights[k], merge.add(state, first)[k]) for k in state))
        # m₁ = τ₁, θ₁ = θ₀ + lr(τ₁ + μm₁); m₂ = μm₁ + τ₂, θ₂ = θ₁ + lr(τ₂ + μm₂).
        lr, mu = 0.7, 0.9
        one, m1 = merge.outer_step(state, first, None, lr, mu)
        two, m2 = merge.outer_step(one, second, m1, lr, mu)
        t1, t2 = flatten(first), flatten(second)
        np.testing.assert_allclose(flatten(m1), t1, rtol=1e-6)
        np.testing.assert_allclose(flatten(one), flatten(state) + lr * (t1 + mu * t1), atol=1e-6)
        expected_m2 = mu * flatten(m1).astype(np.float64) + t2
        np.testing.assert_allclose(flatten(m2), expected_m2, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(flatten(two), flatten(one) + lr * (t2 + mu * expected_m2), atol=1e-6)

    def test_rules_are_checked(self):
        self.assertEqual(merge.check_rule(), merge.RULE)
        self.assertEqual(merge.check_rule({'merge': 'ties', 'ties_mean': 'all'})['ties_mean'], 'all')
        for bad in ({'merge': 'median'}, {'ties_keep': 0.2}, {'merge': 'ties', 'ties_keep': 0}, {'dare': {'method': 'random', 'p': 1}},
                    {'ties_mean': 'all'}, {'merge': 'ties', 'ties_mean': 'median'},
                    {'soup': 'some'}, {'outer': {'lr': 0, 'momentum': 0}}, {'outer': {'lr': 1, 'momentum': 1}}, {'speed': 1}):
            with self.assertRaises(ValueError, msg=bad):
                merge.check_rule(bad)


class RoundTests(unittest.TestCase):
    """Round 1 from θ₀ with three shard workers, and round 2 from its merge."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        cls.anchor = load_anchor('v1')
        cls.cwd = os.getcwd()
        os.chdir(cls.folder)  # records keep the paths they were given; replay from the same folder
        cls.workers = [worker(Path('.'), f'w{i}', shard=f'{i}/3') for i in (1, 2, 3)]
        cls.record, merged = merge.merge_round(cls.anchor, cls.workers)
        cls.state = merged.state
        merge.run_round(Path('round1'), cls.workers)
        cls.offsource = worker(Path('.'), 'offsource-r1', shard='1/3', family='offsource')

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        cls.tmp.cleanup()

    def test_workers_record_their_round_start_shard_and_targets(self):
        for i, path in enumerate(self.workers, 1):
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['merge'], {'round': 1, 'start_sha256': self.anchor.spec['theta0_sha256'],
                                                   'shard': [i, 3], 'seed': 42, 'training_targets': 48})
            self.assertEqual(len(checkpoint['splits']['train']), 1)

    def test_the_round_averages_accepted_deltas_by_training_targets(self):
        record = self.record
        self.assertEqual(record['round'], 1)
        self.assertEqual([c['status'] for c in record['contributors']], ['accepted'] * 3)
        self.assertAlmostEqual(sum(c['weight'] for c in record['contributors']), 1.0)
        self.assertTrue(all(abs(c['weight'] - 1 / 3) < 1e-12 for c in record['contributors']))  # 48 targets each
        taus = [difference(torch.load(p, map_location='cpu', weights_only=True)['model'], self.anchor.theta0) for p in self.workers]
        for name, value in self.anchor.theta0.items():
            expected = (value.double() + sum(t[name] for t in taus) / 3).float()
            self.assertTrue(torch.allclose(self.state[name], expected, atol=1e-7), name)
        self.assertEqual(record['merged']['weights_sha256'], weights_digest(self.state))
        self.assertLessEqual(record['merged']['validation_bits'], record['start']['validation_bits'] + merge.TOLERANCE_BITS)
        self.assertNotIn('test', json.dumps(record['merged']), 'a round never scores the test split')
        # The selection control is the off-source validation scan; the reporting one (its test
        # scan) is never read by a round.
        self.assertEqual(record['control']['family'], 'offsource')
        self.assertTrue(all(e['family'] == 'offsource' for e in record['control']['selection']))
        reporting = merge.controls(merge.held_out(self.anchor, 'voyager', 'within_session_scan_holdout')[0], 'voyager',
                                   'within_session_scan_holdout')['reporting']
        self.assertFalse({json.dumps(r['entry']) for r in reporting} & {json.dumps(e) for e in record['control']['selection']})
        self.assertEqual(record['control']['role'], 'diagnostic')
        for scores in (record['start'], record['merged'], *record['contributors']):
            self.assertEqual(set(scores['control']), {'bits', 'order_gain_bits', 'order_p'})

    def test_the_merged_checkpoint_starts_the_next_round_and_replays(self):
        start = merge.round_start(self.anchor, Path('round1/merged.pt'))
        state, digest = start.state, start.weights_sha256
        self.assertEqual((start.round, digest), (1, self.record['merged']['weights_sha256']))
        self.assertEqual(start.lineage['family'], 'voyager')
        self.assertEqual(merge.replay(Path('round1/round.json')), json.loads(Path('round1/round.json').read_text()))
        scores = merge.score(Path('round1/merged.pt'))
        self.assertEqual(set(scores), {'validation', 'test', 'control'})

        # Round 2: workers start from the merge; a round-1 worker and a duplicate don't count.
        second = worker(Path('.'), 'r2w1', shard='1/3', start=Path('round1/merged.pt'))
        checkpoint = torch.load(second, map_location='cpu', weights_only=True)
        self.assertEqual((checkpoint['merge']['round'], checkpoint['merge']['start_sha256']), (2, digest))
        shutil.copy(second, 'duplicate.pt')
        record, merged = merge.merge_round(self.anchor, [second, Path('duplicate.pt'), self.workers[1]], Path('round1/merged.pt'))
        merged = merged.state
        reasons = [c['reason'] for c in record['contributors']]
        self.assertIsNone(reasons[0])
        self.assertRegex(reasons[1], 'duplicate')
        self.assertRegex(reasons[2], 'trained for round 1 from another start')
        self.assertEqual(record['round'], 2)
        expected = merge.add(state, difference(checkpoint['model'], state))
        self.assertTrue(all(torch.equal(merged[k], expected[k]) for k in expected), 'one accepted worker: the merge is that worker')

    def test_a_round_that_cant_be_replayed_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = json.loads(Path('round1/round.json').read_text())
            record['merged']['weights_sha256'] = '0' * 64
            edited = Path(tmp) / 'round.json'
            edited.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'gave a different result'):
                merge.replay(edited)
        original = Path(self.workers[0]).read_bytes()
        try:
            Path(self.workers[0]).write_bytes(original + b'x')
            with self.assertRaisesRegex(ValueError, "isn't the file the round used"):
                merge.replay(Path('round1/round.json'))
        finally:
            Path(self.workers[0]).write_bytes(original)

    def test_a_worker_is_held_to_its_shard_and_weighted_by_it(self):
        first, second, third = self.workers
        # The same slice under another shard's label, and an inflated target count, are refused.
        relabelled = edited(first, 'relabelled.pt', lambda c: c['merge'].update(shard=[2, 3]))
        inflated = edited(second, 'inflated.pt', lambda c: c['merge'].update(training_targets=48_000_000))
        record, merged = merge.merge_round(self.anchor, [first, relabelled, inflated])
        reasons = [c['reason'] for c in record['contributors']]
        self.assertIsNone(reasons[0])
        self.assertRegex(reasons[1], "its training slices aren't shard 2/3's")
        self.assertRegex(reasons[2], 'records 48000000 training targets, but shard 2/3 has 48')
        self.assertEqual([c['weight'] for c in record['contributors']], [1.0, 0.0, 0.0])

        # Two seeds on one shard share its weight, so each slice counts once.
        again = worker(Path('.'), 'w1-seed43', shard='1/3', seed=43)
        record, _ = merge.merge_round(self.anchor, [first, again, second, third])
        self.assertEqual([round(c['weight'], 12) for c in record['contributors']], [round(w, 12) for w in (1 / 6, 1 / 6, 1 / 3, 1 / 3)])

        # Workers split into different numbers of shards would overlap: refused.
        halves = edited(second, 'halves.pt', lambda c: c['merge'].update(shard=[1, 2]))
        with self.assertRaisesRegex(ValueError, 'one division into shards; these use 2 and 3 shards'):
            merge.merge_round(self.anchor, [first, halves])

    def test_a_round_chain_keeps_its_family_dataset_and_split(self):
        # Training another family from a Voyager round is refused before it trains.
        with self.assertRaisesRegex(ValueError, 'merged from voyager runs, and this one.s family, validation, test differ'):
            worker(Path('.'), 'offsource-r2', shard='1/3', family='offsource', start=Path('round1/merged.pt'))
        # So is merging another family's workers onto it.
        with self.assertRaisesRegex(ValueError, 'merged from voyager runs'):
            merge.merge_round(self.anchor, [self.offsource], Path('round1/merged.pt'))

        # Another manifest (here one entry gains a field) is another dataset: refused, both when
        # continuing from a round and when merging runs trained on the original.
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / 'slices'
            shutil.copytree(SLICES.parent, copy)
            manifest = json.loads((copy / 'index.json').read_text())
            manifest['slices'][0]['note'] = 'copied'
            (copy / 'index.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'this one.s dataset_sha256 differ'):
                worker(Path('.'), 'copy-r2', shard='1/3', start=Path('round1/merged.pt'), manifest=copy / 'index.json')
            with self.assertRaisesRegex(ValueError, "isn't the one the runs were trained on"):
                merge.merge_round(self.anchor, self.workers, manifest=copy / 'index.json')
            with self.assertRaisesRegex(ValueError, "isn't the one the runs were trained on"):
                merge.score(Path('round1/merged.pt'), manifest=copy / 'index.json')

    def test_the_greedy_soup_sets_aside_a_worker_that_barely_moved(self):
        first, second, third = self.workers
        # Shard 3's worker, moved 1% of the way: it passes the gate (it's the incumbent, nearly),
        # but averaged in it would dilute the others. Under the plain rule it's averaged in.
        theta0 = self.anchor.theta0
        faint = edited(third, 'faint.pt', lambda c: c.update(model=merge.add(theta0, {k: 0.01 * v for k, v in difference(c['model'], theta0).items()})))
        plain, plain_merge = merge.merge_round(self.anchor, [faint, second, first])
        soup, soup_merge = merge.merge_round(self.anchor, [faint, second, first], rule={'soup': 'greedy'})
        self.assertEqual([c['status'] for c in plain['contributors']], ['accepted'] * 3)
        statuses = {c['file']: c for c in soup['contributors']}
        self.assertEqual(statuses[str(faint)]['status'], 'set aside')
        self.assertRegex(statuses[str(faint)]['reason'], 'no better than')
        self.assertEqual(statuses[str(faint)]['soup']['rank'], 3, 'tried last: it scores worst on its own')
        self.assertEqual(statuses[str(faint)]['weight'], 0.0)
        self.assertLess(soup['merged']['validation_bits'], plain['merged']['validation_bits'])
        self.assertEqual(soup['merged']['soup_validation_bits'], soup['merged']['validation_bits'])

    def test_the_control_is_a_diagnostic_and_a_context_free_model_gains_nothing_from_order(self):
        records, _ = merge.held_out(self.anchor, 'voyager', 'within_session_scan_holdout')
        selection = [self.anchor.tokenizer.encode(r['frames'])
                     for r in merge.controls(records, 'voyager', 'within_session_scan_holdout')['selection']]
        # A context-free predictor of the control's own token frequencies: every weight zero but
        # the final norm's bias, which the head maps to the log frequencies, whatever the input.
        counts = np.bincount(np.concatenate([t[1:] for t in selection]), minlength=16)
        state = {k: torch.zeros_like(v) for k, v in self.anchor.theta0.items()}
        state['norm.bias'][0] = 1.0
        state['head.weight'][:, 0] = torch.from_numpy(np.log((counts + 1) / (counts.sum() + 16))).float()
        diagnostic = merge.order_gain(merge.model_from(self.anchor, state), selection)
        theta0 = merge.order_gain(self.anchor.model(), selection)
        self.assertLess(diagnostic['bits'], theta0['bits'] - 1, 'far better calibrated on the control than θ₀')
        self.assertAlmostEqual(diagnostic['order_gain_bits'], 0.0, places=5)  # the same targets in every order

        # As a worker (a loose validation tolerance, since it predicts Voyager badly), it isn't
        # refused for the control: its likelihood there is recorded, not gated on.
        calibrated = edited(self.workers[0], 'calibrated.pt', lambda c: c.update(model=state))
        record, merged = merge.merge_round(self.anchor, [calibrated], rule={'tolerance_bits': 5.0})
        entry = record['contributors'][0]
        self.assertEqual(entry['status'], 'accepted')
        self.assertAlmostEqual(entry['control']['bits'], diagnostic['bits'], places=9)
        self.assertLess(entry['control']['bits'], record['start']['control']['bits'] - 1)

    def test_a_rebuilt_delta_is_held_to_the_bound(self):
        # Every head weight moved by 9: within the bound of 10, but DARE's rescaling by 1/(1 − p)
        # rebuilds the kept ones near 90.
        theta0 = self.anchor.theta0
        nine = {k: (torch.full_like(v, 9.0) if k == 'head.weight' else torch.zeros_like(v)).double() for k, v in theta0.items()}
        moved = edited(self.workers[0], 'nine.pt', lambda c: c.update(model=merge.add(theta0, nine)))
        record, _ = merge.merge_round(self.anchor, [moved], rule={'dare': {'method': 'random', 'p': 0.9}})
        self.assertLessEqual(record['contributors'][0]['payload']['bytes'], merge.CARD_BUDGET_BYTES)
        self.assertRegex(record['contributors'][0]['reason'], r'its rebuilt delta reaches (89|90|91)\.?\d*, beyond the bound of 10')
        record, _ = merge.merge_round(self.anchor, [moved])
        self.assertNotRegex(record['contributors'][0]['reason'] or '', 'bound')

    def test_a_compressed_ties_round_fits_the_card_budget_and_replays(self):
        rule = {'merge': 'ties', 'dare': {'method': 'random', 'p': 0.9}, 'soup': 'greedy'}
        record = merge.run_round(Path('dare-ties'), self.workers, rule=rule)
        for c in record['contributors']:
            self.assertLessEqual(c['payload']['bytes'], merge.CARD_BUDGET_BYTES)
            self.assertEqual(c['payload']['seed'], int(weights_digest(torch.load(c['file'], weights_only=True)['model'])[:13], 16))
        self.assertEqual(merge.replay(Path('dare-ties/round.json'))['rule'], merge.check_rule(rule))
        # A record from before TIES's mean was a setting replays with the default mean.
        old = json.loads(Path('dare-ties/round.json').read_text())
        del old['rule']['ties_mean']
        Path('dare-ties/round.json').write_text(json.dumps(old))
        self.assertNotIn('ties_mean', merge.replay(Path('dare-ties/round.json'))['rule'])
        # The all-contributor denominator replays too, and merges differently.
        everyone = merge.run_round(Path('dare-ties-all'), self.workers, rule={**rule, 'ties_mean': 'all'})
        self.assertEqual(merge.replay(Path('dare-ties-all/round.json'))['rule']['ties_mean'], 'all')
        self.assertNotEqual(everyone['merged'].get('weights_sha256'), record['merged'].get('weights_sha256'))
        # Only TIES's mean may be missing: any other setting deleted from a record, even one at its
        # default, fails the replay.
        merge.run_round(Path('plain'), self.workers)
        original = Path('plain/round.json').read_text()
        for key in ('merge', 'max_abs_delta', 'soup'):
            edited = json.loads(original)
            del edited['rule'][key]
            Path('plain/round.json').write_text(json.dumps(edited))
            with self.assertRaisesRegex(ValueError, 'different result', msg=key):
                merge.replay(Path('plain/round.json'))
        # A budget no payload fits: every worker is refused.
        record, merged = merge.merge_round(self.anchor, self.workers, rule={'dare': {'method': 'random', 'p': 0.0}})
        self.assertTrue(all('over the 16,384-byte budget' in c['reason'] for c in record['contributors']))
        self.assertIsNone(merged)

    def test_outer_momentum_carries_from_round_to_round(self):
        rule = {'outer': {'lr': 0.7, 'momentum': 0.9}}
        merge.run_round(Path('diloco1'), self.workers, rule=rule)
        start = merge.round_start(self.anchor, Path('diloco1/merged.pt'))
        self.assertEqual(start.outer, rule['outer'])
        taus = [difference(torch.load(p, weights_only=True)['model'], self.anchor.theta0) for p in self.workers]
        tau_bar = {k: sum(t[k] for t in taus) / 3 for k in taus[0]}
        self.assertTrue(all(torch.allclose(start.momentum[k].double(), tau_bar[k], atol=1e-7) for k in tau_bar))

        later = worker(Path('.'), 'diloco-r2', shard='2/3', start=Path('diloco1/merged.pt'))
        record = merge.run_round(Path('diloco2'), [later], Path('diloco1/merged.pt'), rule)
        self.assertEqual(record['merged']['decision'], 'accepted')
        two = merge.round_start(self.anchor, Path('diloco2/merged.pt'))
        tau2 = difference(torch.load(later, weights_only=True)['model'], start.state)
        for k in tau2:
            m2 = 0.9 * start.momentum[k].double() + tau2[k]
            self.assertTrue(torch.allclose(two.momentum[k].double(), m2, atol=1e-6), k)
            self.assertTrue(torch.allclose(two.state[k].double(), start.state[k].double() + 0.7 * (tau2[k] + 0.9 * m2), atol=1e-6), k)
        merge.replay(Path('diloco2/round.json'))
        # The chain keeps its outer step, and its momentum can't be swapped.
        with self.assertRaisesRegex(ValueError, 'keeps one outer step'):
            merge.merge_round(self.anchor, [later], Path('diloco1/merged.pt'))
        checkpoint = torch.load('diloco1/merged.pt', weights_only=True)
        checkpoint['outer_momentum'] = {k: v * 2 for k, v in checkpoint['outer_momentum'].items()}
        torch.save(checkpoint, 'swapped.pt')
        with self.assertRaisesRegex(ValueError, "momentum doesn't match"):
            merge.round_start(self.anchor, Path('swapped.pt'))

    def test_bad_deltas_are_refused_and_a_tampered_merge_cant_start_a_round(self):
        variant = lambda name, change: edited(self.workers[0], name, change)

        rng = np.random.default_rng(0)
        noise = lambda c: c['model'].update({k: v + torch.from_numpy(rng.normal(0, 0.5, v.shape)).float() for k, v in c['model'].items()})
        poisoned = variant('poisoned.pt', noise)
        nan = variant('nan.pt', lambda c: c['model']['head.weight'].__setitem__((0, 0), float('nan')))
        huge = variant('huge.pt', lambda c: c['model']['head.weight'].__setitem__((0, 0), 50.0))
        surrogate = variant('surrogate.pt', lambda c: c.update(variant='shuffled'))
        record, merged = merge.merge_round(self.anchor, [poisoned, nan, huge, surrogate])
        reasons = [c['reason'] for c in record['contributors']]
        self.assertRegex(reasons[0], 'worse than the incumbent')
        self.assertRegex(reasons[1], "delta isn't finite")
        self.assertRegex(reasons[2], 'beyond the bound')
        self.assertRegex(reasons[3], 'shuffled surrogate')
        self.assertIsNone(merged)
        self.assertEqual(record['merged']['decision'], 'rejected')

        # Workers whose validation split isn't the round's, or another family's, can't join.
        other = variant('other.pt', lambda c: c['splits'].update(validation=c['splits']['test']))
        with self.assertRaisesRegex(ValueError, 'one family, on one dataset and split'):
            merge.merge_round(self.anchor, [self.workers[0], other])

        tampered = torch.load('round1/merged.pt', map_location='cpu', weights_only=True)
        tampered['model']['head.weight'][0, 0] += 1
        torch.save(tampered, 'tampered.pt')
        with self.assertRaisesRegex(ValueError, "weights don't match the digest its round recorded"):
            merge.round_start(self.anchor, Path('tampered.pt'))


class BenchmarkTests(unittest.TestCase):
    def test_the_benchmark_runs_every_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'bench'
            merge_bench.main(['--output', str(out), '--seeds', '42', '--epochs', '2', '--rounds', '2'])
            report = json.loads((out / 'report.json').read_text())
        expected = {'shard workers', 'best worker (by validation)', 'one worker on all the data',
                    "one worker on all the data, a shard worker's steps"}
        expected |= {f'one-shot: {name}' for name in merge_bench.ONE_SHOT} | {f'2 rounds: {name}' for name in merge_bench.CHAINS}
        self.assertEqual(set(report['summary']), expected)
        self.assertEqual(report['step_matched_epochs'], 1)
        seed = report['seeds']['42']
        for name in merge_bench.CHAINS:
            self.assertEqual(len(seed['rounds'][name]['history']), 2)
        self.assertEqual(set(report['selection_control']), {'theta0', 'round models', 'full-data workers', 'off-source workers'})
        self.assertGreater(report['selection_control']['round models']['n'], 0)
        payloads = seed['one_shot']['DARE-TIES']['payload_bytes']
        self.assertEqual(len(payloads), 3)
        self.assertLessEqual(max(payloads), merge.CARD_BUDGET_BYTES)

if __name__ == '__main__':
    unittest.main()
