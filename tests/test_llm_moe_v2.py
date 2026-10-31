"""Milestone 3b of issue #20: the peak tokenizer and anchor v3 (each Rosetta symbol exactly one
token), the longer Rosetta and the RFI sweep (synthetic.py), warm-start branching and its
lineage checks, and the parameter-matched dense baselines (tracks/llm/moe.py)."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tracks.llm import moe, moe_ablation, synthetic
from tracks.llm.anchor import ANCHORS, build_codebook, build_theta0, difference, load_anchor, theta0_files
from tracks.llm.data import PeakTokenizer, Tokenizer, load_slices, split_records, tokenizer_from_state
from tracks.llm.merge import model_from
from tracks.llm.model import SignalLM
from tracks.llm.train import parser, run

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'tracks/llm/data/slices/index-moe-v2.json'
FAMILIES = ('voyager', 'rosetta', 'sweep')


def specialist(folder, family, name, epochs=2, warm=None, start=None, anchor='v3', manifest=MANIFEST):
    argv = ['--anchor', anchor, '--manifest', str(manifest), '--family', family, '--allow-within-session', '--variants', 'real',
            '--epochs', str(epochs), '--seed', '42', '--output', str(folder / name)]
    argv += ['--warm', str(warm)] if warm else []
    argv += ['--start', str(start)] if start else []
    run(parser().parse_args(argv))
    return folder / name / 'real.pt'


def edited(source, name, change):
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    change(checkpoint)
    torch.save(checkpoint, name)
    return Path(name)


class PeakTokenizerTests(unittest.TestCase):
    def test_a_frames_token_is_its_brightest_channels_bin_or_no_line(self):
        tok = PeakTokenizer(6.0, 2)
        self.assertEqual(tok.vocab_size, 33)
        frames = np.zeros((4, 64), dtype=np.float32)
        frames[0, 10] = 7.0   # bin 5 -> token 6
        frames[1, 11] = 50.0  # the same bin
        frames[2, 63] = 6.0   # the threshold itself counts: bin 31 -> token 32
        frames[3, 5] = 5.9    # below it: no line
        self.assertEqual(tok.encode(frames).tolist(), [6, 6, 32, 0])
        self.assertEqual(tok.encode(tok.decode([0, 6, 32])).tolist(), [0, 6, 32])
        self.assertIsInstance(tokenizer_from_state(tok.state()), PeakTokenizer)
        self.assertIsInstance(tokenizer_from_state({'mean': 0.0, 'scale': 1.0, 'centroids': [[0.0] * 64]}), Tokenizer)
        for bad in (dict(threshold=0, width=2), dict(threshold=6, width=3), dict(threshold=6, width=2, kind='kmeans')):
            with self.assertRaises(ValueError, msg=bad):
                PeakTokenizer(**bad)


class DataTests(unittest.TestCase):
    def test_the_slices_regenerate_and_the_manifest_keeps_the_real_ones(self):
        committed = json.loads(MANIFEST.read_text())
        real = json.loads((ROOT / 'tracks/llm/data/slices/index.json').read_text())
        self.assertEqual([e for e in committed['slices'] if e['source'].get('product') != 'synthetic'], real['slices'])
        stored = {e['id']: e for e in committed['slices'] if e['source'].get('product') == 'synthetic'}
        generated = synthetic.generate_moe()
        self.assertEqual(len(generated), 18 + 9)
        for entry, frames in generated:
            self.assertEqual(entry, stored[entry['id']])
            data = np.frombuffer((MANIFEST.parent / entry['file']).read_bytes(), dtype='<f4').reshape(frames.shape)
            self.assertLessEqual(np.abs(data - frames).max(), synthetic.TOLERANCE)

    def test_each_rosetta_symbol_is_exactly_one_token(self):
        anchor, records = load_anchor('v3'), load_slices(MANIFEST)
        partitions, _ = split_records(records, 'rosetta', allow_within_session=True)
        self.assertEqual([len(partitions[n]) for n in ('train', 'validation', 'test')], [6, 6, 6])
        for name, parts in partitions.items():
            tokens = np.concatenate([anchor.tokenizer.encode(r['frames']) for r in parts])
            symbols = np.concatenate([synthetic.symbols_of(r['entry']) for r in parts])
            # H(token | symbol) = 0: the token is a function of the symbol, and vice versa.
            self.assertTrue(np.array_equal(tokens, 1 + synthetic.fsk_channel(symbols) // 2), name)
        spans = {n: sorted(r['entry']['source']['symbols'] for r in partitions[n]) for n in ('train', 'validation', 'test')}
        self.assertLessEqual(spans['train'][-1][1], spans['validation'][0][0])
        self.assertLessEqual(spans['validation'][-1][1], spans['test'][0][0])

    def test_the_framing_control_for_rosetta(self):
        # Byte statistics that know the UTF-8 framing, the control for "learned the language":
        # these match a reproduction made independently in review, to 6 decimals.
        anchor, records = load_anchor('v3'), load_slices(MANIFEST)
        partitions, _ = split_records(records, 'rosetta', allow_within_session=True)
        out = moe_ablation.baselines(anchor, {'rosetta': partitions})['rosetta']
        self.assertAlmostEqual(out['framing_tokens'], 2.539182, places=6)
        self.assertAlmostEqual(out['framing_symbols'], 2.298789, places=6)
        self.assertAlmostEqual(out['unigram'], 3.310620, places=6)
        self.assertAlmostEqual(out['markov1'], 3.182158, places=6)

    def test_anchor_v3_rebuilds(self):
        spec, anchor = ANCHORS['v3'], load_anchor('v3')
        self.assertEqual(theta0_files(spec, build_theta0(spec))[0]['sha256'], spec['theta0_sha256'])
        self.assertEqual(build_codebook(spec), anchor.codebook)
        self.assertEqual(anchor.codebook['fit']['slices'], [])  # a fixed rule: nothing fitted, nothing held out to guard
        self.assertEqual(sum(p.numel() for p in anchor.model().parameters()), 27584)


class WarmStartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        cls.anchor = load_anchor('v3')
        records = load_slices(MANIFEST)
        torch.save(moe.warmup(cls.anchor, records, ('voyager', 'rosetta'), epochs=2), cls.folder / 'warm.pt')
        cls.warm = cls.folder / 'warm.pt'
        cls.warm_specialists = [specialist(cls.folder, f, f'warm-{f}', warm=cls.warm) for f in ('voyager', 'rosetta')]
        cls.cold_specialists = [specialist(cls.folder, f, f'cold-{f}') for f in ('voyager', 'rosetta')]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_specialists_branch_from_the_warm_start_and_record_it(self):
        warm = moe.load_warm(self.anchor, self.warm, MANIFEST)
        self.assertEqual(warm.families, ['voyager', 'rosetta'])
        for path in self.warm_specialists:
            checkpoint = torch.load(path, weights_only=True)
            self.assertEqual(checkpoint['merge']['start_sha256'], warm.weights_sha256)
            self.assertEqual(json.loads((path.parent / 'report.json').read_text())['anchor']['init'], 'warm start')
        # The mixture's backbone is the warm start plus the specialists' mean delta from it.
        states, families, _, _ = moe.load_specialists(self.anchor, self.warm_specialists, MANIFEST, self.warm)
        model = moe.upcycle(self.anchor, states, base=warm.state)
        expected = (warm.state['blocks.0.qkv.weight'].double()
                    + sum(difference(s, warm.state)['blocks.0.qkv.weight'] for s in states) / 2).float()
        self.assertTrue(torch.allclose(model.state_dict()['blocks.0.qkv.weight'], expected, atol=1e-7))
        one = moe.upcycle(self.anchor, states[:1], base=warm.state).eval()
        x = torch.randint(0, 33, (2, 15))
        with torch.no_grad():
            self.assertTrue(torch.equal(one(x), model_from(self.anchor, states[0]).eval()(x)))

    def test_a_mixture_takes_specialists_of_one_start(self):
        with self.assertRaisesRegex(ValueError, 'from θ₀'):
            moe.load_specialists(self.anchor, self.warm_specialists, MANIFEST)
        with self.assertRaisesRegex(ValueError, 'from this warm start'):
            moe.load_specialists(self.anchor, self.cold_specialists, MANIFEST, self.warm)
        with self.assertRaisesRegex(ValueError, 'from this warm start'):
            moe.load_specialists(self.anchor, [self.warm_specialists[0], self.cold_specialists[1]], MANIFEST, self.warm)

    def test_a_warm_start_is_checked_before_anything_branches_from_it(self):
        with self.assertRaisesRegex(ValueError, 'warmed up on voyager, rosetta, not sweep'):
            specialist(self.folder, 'sweep', 'sweep-from-warm', warm=self.warm)
        with self.assertRaisesRegex(ValueError, 'two different starts'):
            specialist(self.folder, 'voyager', 'both', warm=self.warm, start=self.warm)
        tampered = edited(self.warm, self.folder / 'tampered.pt', lambda c: c['model']['head.weight'].add_(1.0))
        with self.assertRaisesRegex(ValueError, "weights don't match"):
            moe.load_warm(self.anchor, tampered, MANIFEST)
        # A warm start that trained on a held-out slice is refused, wherever it's used.
        held_out = edited(self.warm, self.folder / 'held-out.pt',
                          lambda c: c['splits'].update(train=c['splits']['train'] + c['splits']['test'][:1]))
        with self.assertRaisesRegex(ValueError, "train slices aren't its families' train splits"):
            moe.load_warm(self.anchor, held_out, MANIFEST)
        with self.assertRaisesRegex(ValueError, "train slices aren't its families' train splits"):
            specialist(self.folder, 'voyager', 'from-held-out', warm=held_out)
        with self.assertRaisesRegex(ValueError, 'not trained from anchor v3'):
            moe.load_warm(self.anchor, specialist(self.folder, 'rosetta', 'v2-rosetta', anchor='v2',
                                                  manifest=ROOT / 'tracks/llm/data/slices/index-rosetta.json'), MANIFEST)
        with self.assertRaisesRegex(ValueError, 'not a warm start'):
            moe.load_warm(self.anchor, self.cold_specialists[0], MANIFEST)


class MatchedDenseTests(unittest.TestCase):
    def test_a_dense_model_matched_to_the_mixtures_size(self):
        config = load_anchor('v3').spec['model']
        for target in (44480, 61184):  # the mixture's active and total parameters
            matched = moe.matched_config(config, target)
            self.assertLessEqual(abs(moe.dense_parameters(matched) - target) / target, 0.005, target)
            self.assertEqual(matched['width'] % 8, 0)
            model = SignalLM(**matched)
            self.assertEqual(model.config['ffn'], matched['ffn'])
            self.assertEqual(tuple(model(torch.randint(0, 33, (2, 15))).shape), (2, 15, 33))
            start = moe.dense_start(matched)
            self.assertTrue(all(torch.equal(start[k], v) for k, v in moe.dense_start(matched).items()))
            model.load_state_dict(start)
        self.assertNotIn('ffn', SignalLM(**config).config)  # the anchors' sizes, as before

    def test_the_ablation_runs_every_comparison_within_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            moe_ablation.main(['--output', str(Path(tmp) / 'ablation'), '--seeds', '42', '--epochs', '2'])
            report = json.loads((Path(tmp) / 'ablation' / 'report.json').read_text())
        rows = report['seeds']['42']['rows']
        budget = report['seeds']['42']['cost']['budget']
        for start in ('cold', 'warm'):
            for kind in ('oracle routing', 'router finetuned', 'router and backbone finetuned'):
                self.assertIn(f'mixture ({start}), {kind}', rows)
        for name, row in rows.items():
            if name.endswith("the mixtures' budget"):
                self.assertGreaterEqual(row['cost'], budget, name)
        self.assertLess(abs(rows["dense, the mixture's total parameters, the mixtures' budget"]['parameters'] - 61184), 300)
        bank = rows['specialist bank, dispatched by family label (warm)']
        self.assertEqual((bank['parameters'], bank['active_parameters']), (3 * 27584, 27584))  # all stored, one runs
        self.assertEqual(bank['cost'], rows['mixture (warm), oracle routing']['cost'])
        self.assertIn('framing_tokens', report['seeds']['42']['baselines']['rosetta'])


if __name__ == '__main__':
    unittest.main()
