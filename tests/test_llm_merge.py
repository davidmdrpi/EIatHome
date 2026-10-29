"""tracks/llm/merge.py: merge rounds of shard workers trained from a shared anchor (issue #20,
milestone 2): shards, round records, checks on deltas, the validation gate, the weighted
average, and replaying a round."""

import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

from tracks.llm import merge, merge_bench
from tracks.llm.anchor import difference, load_anchor, weights_digest
from tracks.llm.data import load_slices, split_records
from tracks.llm.train import parser, run

ROOT = Path(__file__).resolve().parents[1]
SLICES = ROOT / 'tracks/llm/data/slices/index.json'


def worker(folder, name, shard=None, seed=42, start=None, epochs=2):
    argv = ['--anchor', 'v1', '--manifest', str(SLICES), '--allow-within-session', '--variants', 'real',
            '--epochs', str(epochs), '--seed', str(seed), '--output', str(folder / name)]
    argv += ['--shard', shard] if shard else []
    argv += ['--start', str(start)] if start else []
    run(parser().parse_args(argv))
    return folder / name / 'real.pt'


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
        cls.record, cls.state = merge.merge_round(cls.anchor, cls.workers)
        merge.run_round(type('Args', (), dict(output=Path('round1'), anchor='v1', workers=cls.workers, start=None,
                                              tolerance=merge.TOLERANCE_BITS, max_abs_delta=merge.MAX_ABS_DELTA))())

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

    def test_the_merged_checkpoint_starts_the_next_round_and_replays(self):
        state, round_, digest = merge.load_round(self.anchor, Path('round1/merged.pt'))
        self.assertEqual((round_, digest), (1, self.record['merged']['weights_sha256']))
        self.assertEqual(merge.replay(Path('round1/round.json')), json.loads(Path('round1/round.json').read_text()))
        scores = merge.score(Path('round1/merged.pt'))
        self.assertEqual(set(scores), {'validation', 'test', 'control'})

        # Round 2: workers start from the merge; a round-1 worker and a duplicate don't count.
        second = worker(Path('.'), 'r2w1', shard='1/3', start=Path('round1/merged.pt'))
        checkpoint = torch.load(second, map_location='cpu', weights_only=True)
        self.assertEqual((checkpoint['merge']['round'], checkpoint['merge']['start_sha256']), (2, digest))
        shutil.copy(second, 'duplicate.pt')
        record, merged = merge.merge_round(self.anchor, [second, Path('duplicate.pt'), self.workers[1]], Path('round1/merged.pt'))
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

    def test_bad_deltas_are_refused_and_a_tampered_merge_cant_start_a_round(self):
        def variant(name, change):
            checkpoint = torch.load(self.workers[0], map_location='cpu', weights_only=True)
            change(checkpoint)
            torch.save(checkpoint, name)
            return Path(name)

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
        with self.assertRaisesRegex(ValueError, "validation or test split isn't this round's"):
            merge.merge_round(self.anchor, [self.workers[0], other])

        tampered = torch.load('round1/merged.pt', map_location='cpu', weights_only=True)
        tampered['model']['head.weight'][0, 0] += 1
        torch.save(tampered, 'tampered.pt')
        with self.assertRaisesRegex(ValueError, "weights don't match the digest its round recorded"):
            merge.load_round(self.anchor, Path('tampered.pt'))


class BenchmarkTests(unittest.TestCase):
    def test_the_benchmark_runs_every_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'bench'
            merge_bench.main(['--output', str(out), '--seeds', '42', '--epochs', '2', '--rounds', '2'])
            report = json.loads((out / 'report.json').read_text())
        self.assertEqual(set(report['summary']), {'shard workers', 'best worker (by validation)', 'one-shot merge',
                                                  'one worker on all the data', "one worker on all the data, a shard worker's steps",
                                                  '2 rounds of averaging'})
        self.assertEqual(report['step_matched_epochs'], 1)
        self.assertEqual(len(report['seeds']['42']['rounds']['history']), 2)


if __name__ == '__main__':
    unittest.main()
