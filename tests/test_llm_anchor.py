"""tracks/llm/anchor.py: the shared anchor that runs start from so they can be merged (issue #20,
milestone 1): a pinned codebook, a deterministic θ₀, and aligned task vectors."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

from tracks.llm import anchor as anchors
from tracks.llm.anchor import (ANCHORS, apply, build_codebook, build_theta0, checkpoint_task_vector, codebook_digest,
                               compare_codebook, flatten, load_anchor, task_vector, theta0_files)
from tracks.llm.data import load_slices, split_records
from tracks.llm.model import SignalLM, fit
from tracks.llm.train import model_config, parser, run

ROOT = Path(__file__).resolve().parents[1]
SLICES = ROOT / 'tracks/llm/data/slices/index.json'
SPEC = ANCHORS['v1']


class CommittedAnchorTests(unittest.TestCase):
    def test_the_committed_anchor_verifies_and_rebuilds(self):
        anchor = load_anchor('v1')
        # θ₀ is uniform draws and exactly rounded arithmetic: the same bytes anywhere.
        header, data = theta0_files(SPEC, build_theta0(SPEC))
        self.assertEqual(data, (anchors.ANCHOR_DIR / 'v1' / 'theta0.bin').read_bytes())
        self.assertEqual(header['sha256'], SPEC['theta0_sha256'])
        # k-means goes through numpy's SIMD kernels, which differ by CPU (AVX-512 or not) in a
        # float32 rounding: the rebuilt codebook agrees to that, with the same token for every
        # frame, but its digest can differ. The committed file is the pinned codebook.
        frames = np.concatenate([r['frames'] for r in load_slices(SLICES)])
        check = compare_codebook(build_codebook(SPEC)['tokenizer'], anchor, frames)
        self.assertTrue(check['same_tokens'])
        self.assertLessEqual(check['max_difference'], 1e-5)
        self.assertTrue(check['agrees'])
        model = anchor.model()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 26496)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, anchor.theta0[name]), name)

    def test_the_initializers(self):
        theta = load_anchor('v1').theta0
        self.assertTrue(torch.equal(theta['blocks.0.norm1.weight'], torch.ones(32)))
        self.assertTrue(torch.equal(theta['blocks.1.qkv.bias'], torch.zeros(96)))
        self.assertTrue(torch.equal(theta['blocks.1.out.bias'], torch.zeros(32)))
        self.assertLessEqual(theta['blocks.0.qkv.weight'].abs().max().item(), (6 / 128) ** 0.5)
        self.assertLessEqual(theta['blocks.0.feed.3.weight'].abs().max().item(), 128 ** -0.5)
        self.assertAlmostEqual(theta['token.weight'].var().item(), 1.0, delta=0.15)

    def test_the_codebook_is_fitted_on_training_scans_only(self):
        anchor = load_anchor('v1')
        records = load_slices(SLICES)
        fitted = {s['sha256'] for s in anchor.codebook['fit']['slices']}
        for family in ('voyager', 'offsource'):
            partitions, _ = split_records(records, family, allow_within_session=True)
            self.assertTrue({r['sha256'] for r in partitions['train']} <= fitted, family)
            held_out = {r['sha256'] for name in ('validation', 'test') for r in partitions[name]}
            self.assertFalse(held_out & fitted, family)
        self.assertEqual(anchor.codebook['fit']['frames'], 294)
        self.assertEqual(anchor.tokenizer.centroids.shape, (16, 64))

    def test_edited_files_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ('codebook.json', 'theta0.json', 'theta0.bin'):
                shutil.copy(anchors.ANCHOR_DIR / 'v1' / name, folder / name)
            load_anchor('v1', folder)  # an exact copy loads

            data = bytearray((folder / 'theta0.bin').read_bytes())
            data[100] ^= 1
            (folder / 'theta0.bin').write_bytes(bytes(data))
            with self.assertRaisesRegex(ValueError, 'θ₀ doesn.t match its pinned SHA-256'):
                load_anchor('v1', folder)
            shutil.copy(anchors.ANCHOR_DIR / 'v1' / 'theta0.bin', folder / 'theta0.bin')

            # A codebook file that's consistent with itself but isn't the pinned codebook.
            codebook = json.loads((folder / 'codebook.json').read_text())
            codebook['tokenizer']['centroids'][0][0] += 1e-3
            codebook['sha256'] = codebook_digest(codebook['tokenizer'])
            (folder / 'codebook.json').write_text(json.dumps(codebook))
            with self.assertRaisesRegex(ValueError, 'codebook doesn.t match its pinned SHA-256'):
                load_anchor('v1', folder)


class WorkerTests(unittest.TestCase):
    """Two workers train from the anchor with different seeds, as volunteers would."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        cls.anchor = load_anchor('v1')
        cls.reports = {}
        for seed in (42, 43):
            args = parser().parse_args(['--anchor', 'v1', '--manifest', str(SLICES), '--allow-within-session',
                                        '--epochs', '3', '--seed', str(seed), '--output', str(cls.folder / f'w{seed}')])
            cls.reports[seed] = run(args)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_runs_from_the_anchor_share_its_codebook_and_its_start(self):
        for seed, report in self.reports.items():
            with self.subTest(seed=seed):
                self.assertEqual({k: report['anchor'][k] for k in self.anchor.identity()}, self.anchor.identity())
                self.assertEqual(report['tokenizer'], {'fit_split': 'anchor', 'training_frames': 147, 'sha256': SPEC['codebook_sha256']})
                self.assertEqual(report['model'], SPEC['model'])
                self.assertEqual(report['parameters'], 26496)
                self.assertEqual(report['anchor']['codebook_frames_in_this_manifest'], 294)
                checkpoint = torch.load(self.folder / f'w{seed}' / 'real.pt', map_location='cpu', weights_only=True)
                self.assertEqual(checkpoint['anchor'], self.anchor.identity())
                self.assertEqual(codebook_digest(checkpoint['tokenizer']), SPEC['codebook_sha256'])

    def test_task_vectors_are_aligned_and_rebuild_the_weights(self):
        tau42, checkpoint = checkpoint_task_vector(self.folder / 'w42' / 'real.pt', self.anchor)
        tau43, _ = checkpoint_task_vector(self.folder / 'w43' / 'real.pt', self.anchor)
        self.assertEqual(list(tau42), list(self.anchor.theta0))
        flat42, flat43 = flatten(tau42), flatten(tau43)
        self.assertEqual(flat42.shape, (26496,))
        self.assertGreater(np.abs(flat42).max(), 0)
        self.assertFalse(np.array_equal(flat42, flat43), 'different seeds make different updates')
        rebuilt = apply(self.anchor, tau42)
        for name, value in checkpoint['model'].items():
            self.assertTrue(torch.equal(rebuilt[name], value), name)

    def test_the_delta_command_writes_tau_by_parameter(self):
        out = self.folder / 'tau42.npz'
        anchors.main(['delta', str(self.folder / 'w42' / 'real.pt'), '--output', str(out)])
        saved = np.load(out)
        self.assertEqual(json.loads(str(saved['__anchor__'])), self.anchor.identity())
        tau, _ = checkpoint_task_vector(self.folder / 'w42' / 'real.pt', self.anchor)
        for name, value in tau.items():
            np.testing.assert_array_equal(saved[name], value.float().numpy())

    def test_training_from_the_anchor_starts_at_theta0(self):
        # With a vanishing learning rate, the weights stay where training started.
        sequences = [np.arange(40, dtype=np.int64) % 16]
        model, _, _ = fit(sequences, sequences, SPEC['model'], 1, 8, 1e-12, 1, 7, 'cpu', init_state=self.anchor.theta0)
        self.assertLess(np.abs(flatten(task_vector(model.state_dict(), self.anchor))).max(), 1e-8)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def args(self, *extra, manifest=SLICES):
        return parser().parse_args(['--anchor', 'v1', '--manifest', str(manifest), '--allow-within-session',
                                    '--epochs', '1', '--output', str(self.folder / 'run'), *extra])

    def test_an_anchor_sets_the_model_size(self):
        anchor = load_anchor('v1')
        self.assertEqual(model_config(self.args(), anchor), SPEC['model'])
        self.assertEqual(model_config(self.args('--width', '32'), anchor), SPEC['model'])
        with self.assertRaisesRegex(ValueError, r'--width 64 conflicts with anchor v1 \(32\)'):
            run(self.args('--width', '64'))
        self.assertFalse((self.folder / 'run').exists())

    def test_another_data_product_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'defined on the 6_sec_integration product'):
            run(self.args(manifest=ROOT / 'web/data/slices/index.json'))

    def test_held_out_frames_the_codebook_saw_are_refused(self):
        # Reverse the scans' order in time: the scans the codebook was fitted on become the test scans.
        copy = self.folder / 'slices'
        shutil.copytree(SLICES.parent, copy)
        manifest = json.loads((copy / 'index.json').read_text())
        for entry in manifest['slices']:
            entry['source']['tstart_mjd'] = 2 * 59047 - entry['source']['tstart_mjd']
        (copy / 'index.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'codebook was fitted on frames this run holds out'):
            run(self.args(manifest=copy / 'index.json'))

    def test_only_runs_from_the_anchor_have_a_task_vector(self):
        # Without --anchor, a run fits its own codebook and starts from its own weights.
        free = parser().parse_args(['--manifest', str(SLICES), '--allow-within-session', '--epochs', '1', '--width', '32',
                                    '--layers', '2', '--heads', '4', '--position', 'rope', '--output', str(self.folder / 'free')])
        run(free)
        anchor = load_anchor('v1')
        with self.assertRaisesRegex(ValueError, "wasn't trained from anchor v1"):
            checkpoint_task_vector(self.folder / 'free' / 'real.pt', anchor)
        with self.assertRaisesRegex(ValueError, "don't have the anchor's parameters"):
            task_vector(SignalLM(width=64, layers=2, heads=4, position='rope').state_dict(), anchor)


if __name__ == '__main__':
    unittest.main()


class CodebookComparisonTests(unittest.TestCase):
    def test_a_different_codebook_doesnt_agree(self):
        anchor = load_anchor('v1')
        frames = np.concatenate([r['frames'] for r in load_slices(SLICES)])
        state = anchor.tokenizer.state()
        self.assertEqual(compare_codebook(state, anchor, frames), {'exact': True, 'max_difference': 0.0, 'same_tokens': True, 'agrees': True})
        nudged = json.loads(json.dumps(state))
        nudged['centroids'][3][7] += 2e-7  # a float32 rounding: agrees
        check = compare_codebook(nudged, anchor, frames)
        self.assertEqual((check['exact'], check['agrees']), (False, True))
        moved = json.loads(json.dumps(state))
        moved['centroids'][3] = moved['centroids'][5]  # a different codebook
        self.assertFalse(compare_codebook(moved, anchor, frames)['agrees'])
