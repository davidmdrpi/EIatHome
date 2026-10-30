"""Milestone 3 of issue #20: the Rosetta control in the Python track (tracks/llm/synthetic.py),
the three-family anchor v2, and upcycling family specialists into a mixture of experts
(tracks/llm/moe.py): routing, load balance, upcycling and router finetuning."""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import torch

from tracks.llm import moe, moe_bench, synthetic
from tracks.llm.anchor import ANCHORS, build_codebook, build_theta0, compare_codebook, difference, load_anchor, theta0_files
from tracks.llm.data import load_slices, split_records
from tracks.llm.merge import model_from
from tracks.llm.train import parser, run

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'tracks/llm/data/slices/index-rosetta.json'
FAMILIES = ('voyager', 'offsource', 'rosetta')


def specialist(folder, family, anchor='v2', epochs=2, seed=42, start=None):
    argv = ['--anchor', anchor, '--manifest', str(MANIFEST if anchor == 'v2' else ROOT / 'tracks/llm/data/slices/index.json'),
            '--family', family, '--allow-within-session', '--variants', 'real', '--epochs', str(epochs),
            '--seed', str(seed), '--output', str(folder / f'{anchor}-{family}-{seed}')]
    argv += ['--start', str(start)] if start else []
    run(parser().parse_args(argv))
    return folder / f'{anchor}-{family}-{seed}' / 'real.pt'


class RosettaTests(unittest.TestCase):
    def test_the_corpus_is_the_live_labs(self):
        js = (ROOT / 'web/js/signals.js').read_text()
        self.assertEqual(re.search(r'export const ROSETTA_TEXT = `(.*?)`;', js, re.S).group(1), synthetic.CORPUS)
        self.assertEqual(hashlib.sha256(synthetic.CORPUS.encode()).hexdigest(), synthetic.CORPUS_SHA256)
        self.assertIn(synthetic.CORPUS_SHA256, (ROOT / 'web/js/workunits.js').read_text())
        self.assertEqual(synthetic.text_symbols('Fo').tolist(), [4, 6, 6, 15])  # 'F' 0x46, 'o' 0x6f

    def test_the_committed_slices_regenerate_and_the_manifest_keeps_the_real_ones(self):
        committed = json.loads(MANIFEST.read_text())
        real = json.loads((ROOT / 'tracks/llm/data/slices/index.json').read_text())
        self.assertEqual([e for e in committed['slices'] if e['family'] != 'rosetta'], real['slices'])
        stored = {e['id']: e for e in committed['slices'] if e['family'] == 'rosetta'}
        for entry, frames in synthetic.generate():
            self.assertEqual(entry, stored[entry['id']])
            data = np.frombuffer((MANIFEST.parent / entry['file']).read_bytes(), dtype='<f4').reshape(frames.shape)
            self.assertLessEqual(np.abs(data - frames).max(), synthetic.TOLERANCE)
            # Every frame's brightest channel is its symbol's tone: the text is in the frames.
            self.assertTrue(np.array_equal(data.argmax(axis=1), synthetic.fsk_channel(synthetic.symbols_of(entry))))

    def test_the_scans_hold_different_text_in_order(self):
        partitions, independent = split_records(load_slices(MANIFEST), 'rosetta', allow_within_session=True)
        self.assertFalse(independent)
        spans = {name: sorted(r['entry']['source']['symbols'] for r in partitions[name]) for name in ('train', 'validation', 'test')}
        self.assertEqual([len(s) for s in spans.values()], [3, 3, 3])
        self.assertLess(spans['train'][-1][1], spans['validation'][0][0] + 1)
        self.assertLess(spans['validation'][-1][1], spans['test'][0][0] + 1)


class AnchorV2Tests(unittest.TestCase):
    def test_v2_rebuilds_and_covers_three_families(self):
        spec, anchor = ANCHORS['v2'], load_anchor('v2')
        header, data = theta0_files(spec, build_theta0(spec))
        self.assertEqual(header['sha256'], spec['theta0_sha256'])
        records = load_slices(MANIFEST)
        check = compare_codebook(build_codebook(spec)['tokenizer'], anchor, np.concatenate([r['frames'] for r in records]))
        self.assertTrue(check['agrees'])
        self.assertEqual(sum(p.numel() for p in anchor.model().parameters()), 27520)
        fitted = {s['sha256'] for s in anchor.codebook['fit']['slices']}
        for family in FAMILIES:
            partitions, _ = split_records(records, family, allow_within_session=True)
            self.assertTrue({r['sha256'] for r in partitions['train']} <= fitted, family)
            self.assertFalse({r['sha256'] for name in ('validation', 'test') for r in partitions[name]} & fitted, family)

    def test_the_codebook_size_rule(self):
        # 32 tokens: a Rosetta training frame's token gives its symbol for at least 95% of
        # frames, and neither real family is tokenized more coarsely than under v1.
        anchor, records = load_anchor('v2'), load_slices(MANIFEST)
        train = {f: split_records(records, f, allow_within_session=True)[0]['train'] for f in FAMILIES}
        tokens = np.concatenate([anchor.tokenizer.encode(r['frames']) for r in train['rosetta']])
        symbols = np.concatenate([synthetic.symbols_of(r['entry']) for r in train['rosetta']])
        self.assertGreaterEqual(sum(np.bincount(symbols[tokens == t]).max() for t in np.unique(tokens)) / len(symbols), 0.95)
        used = lambda family: len(np.unique(np.concatenate([anchor.tokenizer.encode(r['frames']) for r in train[family]])))
        self.assertGreaterEqual(used('voyager'), 12)
        self.assertGreaterEqual(used('offsource'), 6)


class MixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        cls.anchor = load_anchor('v2')
        cls.paths = [specialist(cls.folder, f) for f in FAMILIES]
        cls.states, cls.families, _, cls.partitions = moe.load_specialists(cls.anchor, cls.paths, MANIFEST)
        cls.x = torch.randint(0, 32, (3, 15), generator=torch.Generator().manual_seed(0))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_shapes_parameters_and_gradients(self):
        model = moe.upcycle(self.anchor, self.states)
        model.train()
        logits = model(self.x)
        self.assertEqual(tuple(logits.shape), (3, 15, 32))
        (logits.logsumexp(-1).mean() + model.balance_loss).backward()
        self.assertTrue(all(block.router.weight.grad.abs().sum() > 0 for block in model.blocks))
        self.assertTrue(all(e[0].weight.grad is not None for block in model.blocks for e in block.experts))
        # Dense 27,520; each block trades its feed-forward layer (8,352) for three of them and a
        # 32 x 3 router; top-2 leaves one expert per block unused by each token.
        ffn, layers = 32 * 128 + 128 + 128 * 32 + 32, 2
        self.assertEqual(sum(p.numel() for p in model.parameters()), 27520 + layers * (2 * ffn + 32 * 3))
        self.assertEqual(moe.active_parameters(model), 27520 + layers * (ffn + 32 * 3))

    def test_top2_gates_and_the_block_output(self):
        model = moe.upcycle(self.anchor, self.states).eval()
        block = model.blocks[0]
        x = model.token(self.x)
        with torch.no_grad():
            out = block(x)
            attended = block.attend(x)
            h = block.norm2(attended)
            top, chosen = block.router(h).topk(2, dim=-1)
            gates = top.softmax(-1)
            outputs = torch.stack([expert(h) for expert in block.experts], dim=-1)  # [b, n, width, experts]
            picked = lambda j: outputs.gather(-1, chosen[..., j, None, None].expand(3, 15, 32, 1))[..., 0]
            expected = attended + gates[..., :1] * picked(0) + gates[..., 1:] * picked(1)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))
        self.assertTrue(torch.equal(block.chosen, chosen))
        self.assertTrue(torch.allclose(gates.sum(-1), torch.ones(3, 15)))

    def test_the_load_balancing_loss(self):
        even = torch.zeros(10, 3)
        self.assertAlmostEqual(moe.balance_loss(even, even.topk(2, dim=-1)[1]).item(), 1.0, places=6)
        collapsed = torch.tensor([[10.0, 0.0, 0.0]]).repeat(10, 1)  # every token prefers expert 0
        self.assertAlmostEqual(moe.balance_loss(collapsed, collapsed.topk(2, dim=-1)[1]).item(), 1.5, places=3)
        # Padding positions don't count.
        mixed = torch.cat([even, collapsed])
        keep = torch.tensor([True] * 10 + [False] * 10)
        self.assertAlmostEqual(moe.balance_loss(mixed, mixed.topk(2, dim=-1)[1], keep).item(), 1.0, places=6)
        # In the model: a zero router gives every expert the same probability, so the loss is 1.
        model = moe.upcycle(self.anchor, self.states).eval()
        with torch.no_grad():
            for block in model.blocks:
                block.router.weight.zero_()
            model(self.x)
        self.assertAlmostEqual(model.balance_loss.item(), 1.0, places=6)

    def test_upcycling_keeps_each_specialist(self):
        # One expert: the mixture is exactly the specialist.
        one, dense = moe.upcycle(self.anchor, self.states[:1]).eval(), model_from(self.anchor, self.states[0]).eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(one(self.x), dense(self.x)))
        # Three: the backbone is θ₀ plus the specialists' mean delta, the experts are theirs.
        model = moe.upcycle(self.anchor, self.states)
        state = model.state_dict()
        taus = [difference(s, self.anchor.theta0) for s in self.states]
        for name in ('token.weight', 'blocks.1.qkv.weight', 'blocks.0.norm2.bias', 'head.weight'):
            expected = (self.anchor.theta0[name].double() + sum(t[name] for t in taus) / 3).float()
            self.assertTrue(torch.allclose(state[name], expected, atol=1e-7), name)
        for i, s in enumerate(self.states):
            self.assertTrue(torch.equal(state[f'blocks.1.experts.{i}.3.weight'], s['blocks.1.feed.3.weight']))
        # Identical specialists: whatever the router picks, the gates sum to one.
        same = moe.upcycle(self.anchor, [self.states[0]] * 3).eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(same(self.x), dense(self.x), atol=1e-5))
            same.force = 2
            self.assertTrue(torch.allclose(same(self.x), dense(self.x), atol=1e-5))

    def test_router_finetuning_freezes_the_experts(self):
        train, validation = moe.mixed(self.anchor, self.partitions, 'train'), moe.mixed(self.anchor, self.partitions, 'validation')
        for parts in (('router',), ('router', 'backbone')):
            before = moe.upcycle(self.anchor, self.states).state_dict()
            model, history, _ = moe.finetune(moe.upcycle(self.anchor, self.states), train, validation, 2, 32, 1e-2, 8, 0, parts)
            after = model.state_dict()
            changed = {name for name in before if not torch.equal(before[name], after[name])}
            self.assertFalse({n for n in changed if '.experts.' in n}, parts)
            self.assertTrue(any('.router.' in n for n in changed), parts)
            self.assertEqual(any(moe.is_backbone(n) for n in changed), 'backbone' in parts, parts)
            self.assertTrue(all(p.requires_grad for p in model.parameters()), 'restored for later use')
            self.assertIn('balance', history[-1])

    def test_only_family_specialists_from_theta0_upcycle(self):
        with self.assertRaisesRegex(ValueError, 'family of its own'):
            moe.load_specialists(self.anchor, [self.paths[0], specialist(self.folder, 'voyager', seed=43)], MANIFEST)
        with self.assertRaisesRegex(ValueError, 'not trained from anchor v2'):
            moe.load_specialists(self.anchor, [self.paths[0], specialist(self.folder, 'offsource', anchor='v1')], MANIFEST)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            moe.load_specialists(self.anchor, self.paths[:1], MANIFEST)
        with self.assertRaisesRegex(ValueError, "isn't the one the specialists were trained on"):
            moe.load_specialists(self.anchor, self.paths, ROOT / 'tracks/llm/data/slices/index.json')

    def test_the_upcycle_command(self):
        out = self.folder / 'cli'
        moe.main(['upcycle', '--output', str(out), '--epochs', '2', *map(str, self.paths)])
        report = json.loads((out / 'report.json').read_text())
        self.assertEqual(report['experts'], list(FAMILIES))
        self.assertEqual(set(report['test']), {*FAMILIES, 'mean'})
        self.assertEqual(len(report['routing']['voyager']['first_choice']), 2)
        checkpoint = torch.load(out / 'moe.pt', weights_only=True)
        self.assertEqual(checkpoint['config']['experts'], 3)


class BenchmarkTests(unittest.TestCase):
    def test_the_benchmark_runs_every_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            moe_bench.main(['--output', str(Path(tmp) / 'bench'), '--seeds', '42', '--epochs', '2'])
            report = json.loads((Path(tmp) / 'bench' / 'report.json').read_text())
        self.assertIn('mixture, router and backbone finetuned', report['summary'])
        self.assertIn('one dense model on all three families', report['summary'])
        self.assertEqual(set(report['summary']['mixture, router finetuned']['own_expert_share']), set(FAMILIES))


if __name__ == '__main__':
    unittest.main()
