"""Scientific validity and a complete CPU training/checkpoint cycle; unittest only."""

import copy
import json
import io
import os
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from tracks.llm.data import Tokenizer, digest, groups, load_slices, markov_probabilities, split_records, split_sizes, surrogate, SESSION_GAP_DAYS
from tracks.llm.model import RotaryBlock, SignalLM, evaluate, fit, generate, windows
from tracks.llm.sample import load_checkpoint, main as sample_main
from tracks.llm.train import parser, run, window_starts

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "web/data/slices/index.json"
torch.set_num_threads(1)


def fixture_manifest(folder, sessions=1, scans=3, slices=3):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    entries = []
    for session in range(sessions):
        for scan in range(scans):
            for family in ('voyager', 'offsource'):
                for window in range(slices):
                    frames = rng.normal(size=(16, 80)).astype('<f4')
                    if family == 'voyager':
                        frames[np.arange(16), (np.arange(16) + window * 4) % 40 + 20] += 8
                    payload = frames.tobytes()
                    name = f'{family}-{session}-{scan}-{window}'
                    (folder / f'{name}.bin').write_bytes(payload)
                    entries.append({'id': name, 'file': f'{name}.bin', 'family': family,
                                    'dtype': 'float32-le', 'shape': [16, 80], 'sha256': digest(payload),
                                    'source': {'tstart_mjd': 60000 + session * 2 + scan * 0.01
                                               + (0.002 if family == 'offsource' else 0)}})
    manifest = folder / 'index.json'
    manifest.write_text(json.dumps({'format': 'eiathome-slices', 'version': 1, 'slices': entries}))
    return manifest


class SignalFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.manifest = fixture_manifest(self.folder / 'fixture')

    def args(self, output='run', **changes):
        args = parser().parse_args(['--manifest', str(self.manifest), '--output', str(self.folder / output),
                                   '--allow-within-session', '--width', '16', '--heads', '2',
                                   '--layers', '1', '--epochs', '2', '--vocab-size', '4'])
        for key, value in changes.items():
            setattr(args, key, value)
        return args


class PublishedDataTests(unittest.TestCase):
    def test_current_manifest_integrity_without_assuming_size_or_sessions(self):
        entries = json.loads(MANIFEST.read_text())['slices']
        self.assertEqual(len(load_slices(MANIFEST)), len(entries))

    def test_long_window_manifest_integrity(self):
        # The 49-frame slices from the 6-second product, published for this track only.
        manifest = ROOT / 'tracks/llm/data/slices/index.json'
        entries = json.loads(manifest.read_text())['slices']
        records = load_slices(manifest)
        self.assertEqual(len(records), len(entries))
        for record in records:
            self.assertEqual(len(record['frames']), record['entry']['shape'][0])
            self.assertEqual(record['entry']['source']['product'], '6_sec_integration')


class SignalDataTests(SignalFixture):
    def test_session_rounding_and_reported_counts(self):
        for n, expected in [(3, [1, 1, 1]), (5, [3, 1, 1]), (9, [5, 2, 2]), (10, [6, 2, 2])]:
            with self.subTest(sessions=n):
                records = load_slices(fixture_manifest(self.folder / str(n), sessions=n, scans=1, slices=1))
                partitions, independent = split_records(records)
                self.assertTrue(independent)
                self.assertEqual([len(part) for part in partitions.values()], expected)
                sizes = split_sizes(records, partitions, independent)
                self.assertEqual([part['groups'] for part in sizes.values()], expected)
                self.assertEqual([part['targets'] for part in sizes.values()], [15 * n for n in expected])

    def test_fast_groups_match_pairwise_reference(self):
        rng = np.random.default_rng(3)
        records = [{'start': float(t), 'sha256': f's{i}', 'frame_hash': f'f{i}'}
                   for i, t in enumerate(rng.uniform(0, 40, 100))]
        records[70]['sha256'] = records[3]['sha256']
        records[50]['frame_hash'] = records[9]['frame_hash']
        # Small independent all-pairs oracle for transitive connectivity.
        components = [{i} for i in range(len(records))]
        for i, a in enumerate(records):
            for j, b in enumerate(records[:i]):
                if (abs(a['start'] - b['start']) <= SESSION_GAP_DAYS
                        or a['sha256'] == b['sha256'] or a['frame_hash'] == b['frame_hash']):
                    left = next(c for c in components if i in c)
                    right = next(c for c in components if j in c)
                    if left is not right:
                        left.update(right)
                        components.remove(right)
        ids = {id(r): i for i, r in enumerate(records)}
        expected = {frozenset(c) for c in components}
        for order in [records, records[::-1]]:
            self.assertEqual({frozenset(ids[id(r)] for r in g) for g in groups(order, SESSION_GAP_DAYS)}, expected)

    def test_session_gap_matches_other_tracks(self):
        import re
        for path in ['web/js/splits.js', 'tools/extract_slices.py']:
            text = (ROOT / path).read_text()
            value = re.search(r'SESSION_GAP_DAYS\s*=\s*([0-9.]+)', text).group(1)
            self.assertEqual(float(value), SESSION_GAP_DAYS)

    def test_matrix_nearest_matches_direct_distance(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=(300, 64)).astype(np.float32)
        centres = rng.normal(size=(40, 64)).astype(np.float32)
        tokenizer = Tokenizer(0, 1, centres)
        expected = ((x[:, None].astype(np.float64) - centres[None]) ** 2).sum(-1).argmin(-1)
        np.testing.assert_array_equal(tokenizer._nearest(x), expected)
        centres[1] = centres[0]
        self.assertEqual(Tokenizer(0, 1, centres)._nearest(centres[:1])[0], 0)

    def test_single_session_fixture_requires_explicit_smoke_mode(self):
        records = load_slices(self.manifest)
        self.assertEqual(len(records), 18)
        with self.assertRaisesRegex(ValueError, "3 independent sessions"):
            split_records(records)
        splits, independent = split_records(records, allow_within_session=True)
        self.assertFalse(independent)
        self.assertEqual([len(splits[k]) for k in splits], [3, 3, 3])
        starts = [{r["start"] for r in part} for part in splits.values()]
        self.assertEqual([len(s) for s in starts], [1, 1, 1])
        self.assertEqual(len(set.union(*starts)), 3)

    def test_sessions_transitive_and_copies_cannot_cross_splits(self):
        records = copy.deepcopy(load_slices(self.manifest))
        target = [r for r in records if r["entry"]["family"] == "voyager"]
        for i, record in enumerate(target):
            record["start"] = 60000 + 2 * (i // 3)
        splits, independent = split_records(target)
        self.assertTrue(independent)
        duplicate = copy.deepcopy(target[0])
        duplicate["start"] = target[-1]["start"]  # renamed copy with a conflicting time
        duplicate["entry"]["id"] = "renamed"
        duplicate["entry"]["file"] = "renamed.bin"
        with self.assertRaisesRegex(ValueError, "3 independent sessions"):
            split_records(target + [duplicate])
        chain = copy.deepcopy(target[:3])
        for r, t in zip(chain, [60000, 60000.4, 60000.8]):
            r["start"] = t
        self.assertEqual(len(groups(chain, 0.5)), 1)
        hashes = [{r["frame_hash"] for r in part} for part in splits.values()]
        self.assertFalse(hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])

    def test_loader_rejects_corruption_nonfinite_and_cross_family_copies(self):
        original = json.loads(self.manifest.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            entry = copy.deepcopy(original["slices"][0])
            payload = (self.manifest.parent / entry["file"]).read_bytes()
            file = folder / entry["file"]
            file.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
            manifest = folder / "index.json"
            data = {"format": "eiathome-slices", "version": 1, "slices": [entry]}
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_slices(manifest)
            bad = np.frombuffer(payload, dtype="<f4").copy()
            bad[0] = np.nan
            file.write_bytes(bad.tobytes())
            entry["sha256"] = digest(bad.tobytes())
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                load_slices(manifest)
            file.write_bytes(payload)
            entry["sha256"] = digest(payload)
            duplicate = copy.deepcopy(entry)
            duplicate["family"] = "offsource" if entry["family"] == "voyager" else "voyager"
            data["slices"].append(duplicate)
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "both signal and off-source"):
                load_slices(manifest)

    def test_tokenizer_fits_training_only_and_is_stable_at_inference(self):
        rng = np.random.default_rng(1)
        train = [rng.normal(size=(16, 64)).astype(np.float32)]
        tokenizer = Tokenizer.fit(train, 4, 7)
        state = tokenizer.state()
        held_out = rng.normal(100, 2, (16, 64)).astype(np.float32)
        tokens = tokenizer.encode(held_out)
        self.assertEqual(tokenizer.state(), state)
        np.testing.assert_array_equal(tokens, Tokenizer(**state).encode(held_out))
        np.testing.assert_allclose(tokenizer.decode(tokens), Tokenizer(**state).decode(tokens))
        self.assertEqual(tokenizer.decode([]).shape, (0, 64))

    def test_windows_never_cross_boundaries_or_duplicate_targets(self):
        seqs = [np.arange(18), np.arange(100, 105)]
        examples = windows(seqs, 5)
        targets = np.concatenate([y for _, y in examples])
        np.testing.assert_array_equal(targets, np.r_[np.arange(1, 18), np.arange(101, 105)])
        for x, y in examples:
            np.testing.assert_array_equal(x[:len(y)] + 1, y)

    def test_surrogates_do_not_pool_splits_and_markov_uses_training_only(self):
        train = [np.array([0, 1, 0, 1, 0, 1])]
        probabilities = markov_probabilities(train, 3)
        a = surrogate([np.array([2, 2, 2, 2])], "markov", 1, probabilities)
        b = surrogate([np.array([0, 0, 0, 0])], "markov", 1, probabilities)
        np.testing.assert_array_equal(a, b)  # test identities never affect the surrogate law
        shuffled = surrogate(train, "shuffled", 3, probabilities)
        np.testing.assert_array_equal(np.sort(shuffled[0]), np.sort(train[0]))


class SignalModelTests(SignalFixture):
    def test_generation_never_slides_the_prompt_and_is_bounded(self):
        model = SignalLM(vocab_size=4, context=4, width=16, layers=1, heads=2, dropout=0)
        prompt = [0, 1]
        with mock.patch.object(model, 'forward', wraps=model.forward) as forward:
            tokens = generate(model, prompt, seed=3)
            self.assertEqual(len(tokens), 3)
            self.assertEqual([call.args[0].shape[1] for call in forward.call_args_list], [2, 3, 4])
            for call in forward.call_args_list:
                self.assertEqual(call.args[0][0, :2].tolist(), prompt)
        with mock.patch.object(model, 'forward') as forward:
            with self.assertRaisesRegex(ValueError, 'at most 3 tokens'):
                generate(model, prompt, count=4)
            forward.assert_not_called()

    def test_tiny_positive_temperature_is_greedy_not_nan(self):
        model = SignalLM(vocab_size=4, context=4, width=16, layers=1, heads=2, dropout=0).eval()
        prompt = [0, 1]
        with torch.no_grad():
            expected = model(torch.tensor([prompt]))[0, -1].argmax().item()
        for temperature in [1e-40, 1e-320]:
            self.assertEqual(generate(model, prompt, 1, temperature=temperature), [expected])

    def test_invalid_model_settings_leave_output_available(self):
        for change in [dict(heads=3), dict(layers=0), dict(context=0), dict(dropout=1), dict(width=17)]:
            with self.subTest(change=change), mock.patch('tracks.llm.train.atomic_output') as publish:
                args = self.args(**change)
                with self.assertRaises(ValueError):
                    run(args)
                self.assertFalse(args.output.exists())
                publish.assert_not_called()
        # A numerically valid but untrained positional range must also be rejected.
        args = self.args(context=32)
        with self.assertRaisesRegex(ValueError, 'longest training sequence'):
            run(args)
        self.assertFalse(args.output.exists())
        run(self.args())  # corrected settings reuse the same final path

    def test_late_failure_cleans_staging_and_restores_process_state(self):
        args = self.args()
        old = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
               torch.is_deterministic_algorithms_warn_only_enabled(), os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
        calls = 0

        def fail_second(*a, **kw):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.assertTrue(list(self.folder.glob('.run-*/real.pt')))
                raise RuntimeError('injected later-variant failure')
            return fit(*a, **kw)

        try:
            torch.set_num_threads(2)
            torch.use_deterministic_algorithms(False, warn_only=True)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
            rng = torch.get_rng_state().clone()
            with mock.patch('tracks.llm.train.fit', side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, 'later-variant'):
                    run(args)
            self.assertFalse(args.output.exists())
            self.assertFalse(list(self.folder.glob('.run-*')))
            for retry in [False, True]:
                if retry:
                    run(args)
                self.assertEqual(torch.get_num_threads(), 2)
                self.assertFalse(torch.are_deterministic_algorithms_enabled())
                self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
                self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'], ':16:8')
                torch.testing.assert_close(torch.get_rng_state(), rng)
            self.assertTrue((args.output / 'report.json').exists())
        finally:
            torch.set_num_threads(old[0])
            torch.use_deterministic_algorithms(old[1], warn_only=old[2])
            if old[3] is None:
                os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
            else:
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = old[3]

    def test_duplicate_control_does_not_change_control_score(self):
        before = run(self.args())
        manifest = json.loads(self.manifest.read_text())
        record = copy.deepcopy(next(e for e in manifest['slices'] if e['family'] == 'offsource'))
        record['id'] = 'renamed-control'
        manifest['slices'].append(record)
        self.manifest.write_text(json.dumps(manifest))
        after = run(self.args(output='duplicated'))
        a = before['variants']['real']['other_family_control']
        b = after['variants']['real']['other_family_control']
        for key in ['targets', 'bits_per_token', 'perplexity', 'unigram_bits', 'markov1_bits']:
            self.assertEqual(a[key], b[key])
        self.assertEqual(len(a['slices']), len(b['slices']))

    def test_causal_mask_in_train_and_eval(self):
        a = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        b = a.clone()
        b[:, 4:] = 0
        for position in ['absolute', 'rope']:
            torch.manual_seed(5)
            model = SignalLM(vocab_size=8, context=10, width=16, layers=2, heads=2, dropout=0, position=position)
            for training in [True, False]:
                model.train(training)
                with self.subTest(position=position, training=training), torch.no_grad():
                    torch.testing.assert_close(model(a)[:, :4], model(b)[:, :4], rtol=0, atol=0)

    def test_padding_does_not_change_likelihood(self):
        torch.manual_seed(5)
        model = SignalLM(vocab_size=4, context=8, width=16, layers=1, heads=2, dropout=0)
        sequences = [np.array([0, 1, 2, 3, 0]), np.array([1, 2])]
        batched = evaluate(model, sequences, 2)
        single = evaluate(model, sequences, 1)
        self.assertEqual(batched["targets"], 5)
        self.assertAlmostEqual(batched["bits_per_token"], single["bits_per_token"], places=6)

    def test_training_learns_a_known_next_token_rule(self):
        seqs = [np.tile(np.arange(4), 4) for _ in range(8)]
        config = dict(vocab_size=4, context=15, width=16, layers=1, heads=2, dropout=0)
        model, history, best = fit(seqs, seqs[:1], config, 30, 8, 0.02, 8, 3, "cpu")
        self.assertLess(evaluate(model, seqs[:1])["bits_per_token"], 0.2)
        self.assertLess(history[-1]["train_bits"], history[0]["train_bits"])
        self.assertGreater(best, 0)

    def test_full_training_checkpoint_and_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            args = parser().parse_args(["--manifest", str(self.manifest), "--output", str(output), "--allow-within-session",
                                        "--width", "16", "--heads", "2", "--layers", "1",
                                        "--epochs", "2", "--vocab-size", "4"])
            report = run(args)
            self.assertEqual(report["evaluation"], "within_session_scan_holdout")
            self.assertEqual(report["tokenizer"]["training_frames"], 48)
            self.assertEqual(set(report["variants"]), {"real", "shuffled", "markov"})
            for variant in report["variants"]:
                self.assertEqual(report["variants"][variant]["test"]["targets"], 45)
            model, tokenizer, checkpoint = load_checkpoint(output / "real.pt")
            sample = report["variants"]["real"]["sample"]
            self.assertEqual(generate(model, sample["prompt"], seed=args.seed), sample["generated"])
            np.testing.assert_allclose(np.load(output / "generated_frames.npy"), tokenizer.decode(sample["generated"]))
            again, _, _ = load_checkpoint(output / "real.pt")
            with torch.no_grad():
                x = torch.tensor([sample["prompt"]])
                torch.testing.assert_close(model(x), again(x), rtol=0, atol=0)
            self.assertEqual(checkpoint["splits"], report["splits"])
            records = load_slices(self.manifest)
            partitions, _ = split_records(records, allow_within_session=True)
            expected = Tokenizer.fit([r["frames"] for r in partitions["train"]], 4, args.seed)
            self.assertEqual(tokenizer.state(), expected.state())
            self.assertEqual(json.loads((output / "report.json").read_text())["dataset_sha256"], report["dataset_sha256"])
            # Mutate only test samples. Tokenizer, early stopping and fitted weights must agree.
            modified = copy.deepcopy(json.loads(self.manifest.read_text()))
            test_ids = {r["entry"]["id"] for r in partitions["test"]}
            folder = Path(tmp) / "slices"
            folder.mkdir()
            for entry in modified["slices"]:
                payload = (self.manifest.parent / entry["file"]).read_bytes()
                if entry["id"] in test_ids:
                    values = np.frombuffer(payload, dtype="<f4").copy() * 3 + 100
                    payload = values.tobytes()
                    entry["sha256"] = digest(payload)
                (folder / entry["file"]).write_bytes(payload)
            args.manifest = folder / "index.json"
            args.manifest.write_text(json.dumps(modified))
            args.output = Path(tmp) / "changed-test"
            changed = run(args)
            for variant in report["variants"]:
                self.assertEqual(changed["variants"][variant]["history"], report["variants"][variant]["history"])
                before, _, _ = load_checkpoint(output / f"{variant}.pt")
                after, tok, _ = load_checkpoint(args.output / f"{variant}.pt")
                self.assertEqual(tok.state(), tokenizer.state())
                for key, value in before.state_dict().items():
                    torch.testing.assert_close(value, after.state_dict()[key], rtol=0, atol=0)
            # A single-session run cannot be silently promoted to independent evaluation.
            args.allow_within_session = False
            args.output = Path(tmp) / "must-refuse"
            with self.assertRaisesRegex(ValueError, "3 independent sessions"):
                run(args)


class PositionRobustnessTests(SignalFixture):
    """Issue #7: RoPE, offset training windows, mid-scan scoring and sliding generation."""

    def test_rope_attention_depends_only_on_relative_offsets(self):
        torch.manual_seed(7)
        block = RotaryBlock(16, 2, 0).eval()
        x = torch.randn(2, 9, 16)
        with torch.no_grad():
            reference = block(x)
            for start in [1, 37, 1000]:
                torch.testing.assert_close(block(x, start=start), reference, rtol=0, atol=2e-5)

    def test_rope_block_is_the_pytorch_layer_plus_rotation(self):
        # So comparing absolute and RoPE models compares position encodings, not block designs.
        torch.manual_seed(7)
        block = RotaryBlock(16, 2, 0).eval()
        layer = torch.nn.TransformerEncoderLayer(16, 2, 64, 0, activation='gelu', batch_first=True, norm_first=True).eval()
        with torch.no_grad():
            layer.self_attn.in_proj_weight.copy_(block.qkv.weight)
            layer.self_attn.in_proj_bias.copy_(block.qkv.bias)
            layer.self_attn.out_proj.weight.copy_(block.out.weight)
            layer.self_attn.out_proj.bias.copy_(block.out.bias)
            for ours, theirs in [(block.norm1, layer.norm1), (block.norm2, layer.norm2),
                                 (block.feed[0], layer.linear1), (block.feed[3], layer.linear2)]:
                theirs.load_state_dict(ours.state_dict())
            x = torch.randn(2, 9, 16)
            mask = torch.ones(9, 9, dtype=torch.bool).triu(1)
            with mock.patch('tracks.llm.model.rotary', lambda q, start=0: q):
                torch.testing.assert_close(block(x), layer(x, src_mask=mask), rtol=1e-5, atol=1e-5)

    def test_rope_continuations_slide_past_the_context_and_stay_in_vocabulary(self):
        torch.manual_seed(3)
        model = SignalLM(vocab_size=4, context=4, width=16, layers=1, heads=2, dropout=0, position='rope')
        prompt = [0, 1]
        with mock.patch.object(model, 'forward', wraps=model.forward) as forward:
            tokens = generate(model, prompt, count=40, seed=3)
        self.assertEqual(len(tokens), 40)
        self.assertTrue(all(0 <= t < 4 for t in tokens))
        sequence = prompt + tokens
        inputs = [call.args[0][0].tolist() for call in forward.call_args_list]
        self.assertEqual([len(x) for x in inputs[:4]], [2, 3, 4, 4])
        self.assertTrue(all(len(x) <= 4 for x in inputs))
        for step, x in enumerate(inputs):
            self.assertEqual(x, sequence[max(0, step + 2 - 4):step + 2], 'each step sees the last context tokens')
        # Within the context, RoPE generation is the same as before: no sliding.
        self.assertEqual(generate(model, prompt, seed=3), tokens[:3])

    def test_absolute_models_still_stop_at_the_context(self):
        model = SignalLM(vocab_size=4, context=4, width=16, layers=1, heads=2, dropout=0)
        with mock.patch.object(model, 'forward') as forward:
            with self.assertRaisesRegex(ValueError, 'at most 3 tokens .*RoPE'):
                generate(model, [0, 1], count=4)
            forward.assert_not_called()
        rope = SignalLM(vocab_size=4, context=4, width=16, layers=1, heads=2, dropout=0, position='rope')
        with self.assertRaisesRegex(ValueError, 'at most 0 tokens'):
            generate(rope, [0, 1, 2, 3, 0], count=1)

    def test_training_stride_overlaps_but_scoring_counts_each_target_once(self):
        seqs = [np.arange(12), np.arange(100, 105)]
        starts = lambda examples: [int(x[0]) for x, _ in examples]
        self.assertEqual(starts(windows(seqs, 5)), [0, 5, 10, 100])
        self.assertEqual(starts(windows(seqs, 5, stride=1)), list(range(11)) + [100, 101, 102, 103])
        for stride in [1, 2, 5]:
            for x, y in windows(seqs, 5, stride=stride):
                np.testing.assert_array_equal(x[:len(y)] + 1, y)
        with self.assertRaisesRegex(ValueError, 'stride must be positive'):
            windows(seqs, 5, stride=0)
        # Training on overlapping windows never changes how the test is scored: each target once.
        seqs = [np.arange(12) % 4, np.array([1, 2, 3, 0, 1])]
        config = dict(vocab_size=4, context=5, width=16, layers=1, heads=2, dropout=0, position='rope')
        model, _, _ = fit(seqs, seqs, config, 1, 8, 0.01, 1, 3, 'cpu', stride=1)
        self.assertEqual(evaluate(model, seqs)['targets'], 11 + 4)

    def test_window_start_scores_compare_equal_length_windows(self):
        torch.manual_seed(5)
        model = SignalLM(vocab_size=12, context=5, width=16, layers=1, heads=2, dropout=0, position='rope')
        seqs = [np.arange(12), np.arange(9) + 3, np.arange(4)]
        starts = window_starts(model, seqs, 8)
        # Full 6-frame windows: the 12-frame slice has starts 0-6, the 9-frame one 0-3, the
        # 4-frame one none.
        self.assertEqual([(s['start'], s['windows']) for s in starts['by_start']],
                         [(0, 2), (1, 2), (2, 2), (3, 2), (4, 1), (5, 1), (6, 1)])
        self.assertEqual(starts['mid_scan_windows'], 9)
        for s in starts['by_start']:
            chunks = [q[s['start']:s['start'] + 6] for q in seqs if len(q) >= s['start'] + 6]
            self.assertEqual(s['bits_per_token'], evaluate(model, chunks)['bits_per_token'])
        self.assertEqual(starts['scan_start_bits'], starts['by_start'][0]['bits_per_token'])
        pooled = sum(s['bits_per_token'] * s['windows'] for s in starts['by_start'][1:]) / 9
        self.assertAlmostEqual(starts['mid_scan_bits'], pooled, places=12)
        self.assertIsNone(window_starts(model, [np.arange(5)], 8), 'no room for a full window')
        self.assertIsNone(window_starts(model, [np.arange(6)], 8)['mid_scan_bits'], 'only a scan-start window')

    def test_fit_trains_on_the_requested_stride(self):
        seqs = [np.arange(12) % 4]
        config = dict(vocab_size=4, context=5, width=16, layers=1, heads=2, dropout=0, position='rope')
        with mock.patch('tracks.llm.model.windows', wraps=windows) as made:
            fit(seqs, seqs, config, 1, 8, 0.01, 1, 3, 'cpu', stride=1)
        self.assertEqual(made.call_args_list[0].args[2], 1, 'training windows use the stride')
        self.assertTrue(all(len(call.args) < 3 or call.args[2] is None for call in made.call_args_list[1:]),
                        'validation never overlaps')

    def test_rope_training_run_records_position_and_scores_mid_scan_starts(self):
        args = self.args(position='rope', train_stride=1, context=8)
        report = run(args)
        self.assertEqual(report['model']['position'], 'rope')
        self.assertEqual(report['training']['train_stride'], 1)
        for variant, result in report['variants'].items():
            starts = result['test_window_starts']
            # 3 test slices of 16 frames hold full 9-frame windows at starts 0-7.
            self.assertEqual([(s['start'], s['windows']) for s in starts['by_start']], [(i, 3) for i in range(8)])
            self.assertEqual(starts['mid_scan_windows'], 21)
            self.assertTrue(np.isfinite([starts['scan_start_bits'], starts['mid_scan_bits']]).all())
        control = report['variants']['real']['other_family_control']['window_starts']
        self.assertEqual(control['by_start'][0]['windows'], len(report['variants']['real']['other_family_control']['slices']))
        sample = report['variants']['real']['sample']
        self.assertEqual(len(sample['prompt']) + len(sample['generated']), 16, 'RoPE samples run to the slice length')
        model, _, checkpoint = load_checkpoint(args.output / 'real.pt')
        self.assertEqual(checkpoint['config']['position'], 'rope')
        self.assertEqual(generate(model, sample['prompt'], len(sample['generated']), seed=args.seed), sample['generated'])
        # Sampling past the context from the saved checkpoint.
        records = load_slices(self.manifest)
        output = self.folder / 'long-sample'
        sample_main([str(args.output / 'real.pt'), '--manifest', str(self.manifest), '--slice-id',
                     records[0]['entry']['id'], '--prompt-frames', '4', '--tokens', '30', '--output', str(output)])
        written = json.loads((output / 'sample.json').read_text())
        self.assertEqual((len(written['generated']), written['position']), (30, 'rope'))
        self.assertIn('Qualitative', written['note'])
        self.assertTrue(np.isfinite(np.load(output / 'frames.npy')).all())

    def test_invalid_position_settings_are_rejected_before_output(self):
        for change in [dict(position='rope', width=12, heads=4), dict(train_stride=0), dict(train_stride=16)]:
            with self.subTest(change=change):
                args = self.args(**change)
                with self.assertRaises(ValueError):
                    run(args)
                self.assertFalse(args.output.exists())

    def test_checkpoints_without_a_position_load_as_absolute(self):
        model = SignalLM(vocab_size=4, context=15, width=16, layers=1, heads=2, dropout=0)
        legacy = {k: v for k, v in model.config.items() if k != 'position'}
        records = load_slices(self.manifest)
        path = self.folder / 'legacy.pt'
        torch.save({'format': 'eiathome-signal-lm', 'version': 1, 'config': legacy, 'model': model.state_dict(),
                    'tokenizer': Tokenizer.fit([records[0]['frames']], 4, 3).state(),
                    'dataset_sha256': 'fixture', 'variant': 'real'}, path)
        loaded, _, _ = load_checkpoint(path)
        self.assertEqual(loaded.config['position'], 'absolute')
        x = torch.tensor([[0, 1, 2, 3]])
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), model.eval()(x), rtol=0, atol=0)


class SignalSamplingTests(SignalFixture):
    def checkpoint(self):
        records = load_slices(self.manifest)
        tokenizer = Tokenizer.fit([records[0]['frames']], 4, 3)
        model = SignalLM(vocab_size=4, context=15, width=16, layers=1, heads=2, dropout=0)
        path = self.folder / 'test.pt'
        torch.save({'format': 'eiathome-signal-lm', 'version': 1, 'config': model.config,
                    'model': model.state_dict(), 'tokenizer': tokenizer.state(),
                    'dataset_sha256': 'fixture', 'variant': 'real'}, path)
        return path, records[0]['entry']['id']

    def test_sampling_ignores_unrelated_bad_slices_and_bounds_default(self):
        checkpoint, selected = self.checkpoint()
        manifest = json.loads(self.manifest.read_text())
        manifest['slices'].append({'id': 'broken', 'file': 'missing.bin', 'family': 'unknown',
                                   'source': {'tstart_mjd': float('nan')}})
        # Even the selected entry needs no session metadata to be used as an inference prompt.
        next(e for e in manifest['slices'] if e['id'] == selected)['source'] = {}
        self.manifest.write_text(json.dumps(manifest))
        output = self.folder / 'sample'
        sample_main([str(checkpoint), '--manifest', str(self.manifest), '--slice-id', selected,
                     '--output', str(output), '--temperature', '1e-40'])
        frames = np.load(output / 'frames.npy')
        self.assertEqual(frames.shape, (8, 64))
        self.assertTrue(np.isfinite(frames).all())
        self.assertEqual(len(json.loads((output / 'sample.json').read_text())['generated']), 8)

    def test_reused_output_is_usage_error_before_loading_or_generating(self):
        output = self.folder / 'sample'
        output.mkdir()
        stderr = io.StringIO()
        with mock.patch('tracks.llm.sample.load_checkpoint') as load, redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as exc:
                sample_main(['missing.pt', '--manifest', str(self.manifest), '--slice-id', 'anything',
                             '--output', str(output)])
        self.assertEqual(exc.exception.code, 2)
        self.assertIn('Output already exists', stderr.getvalue())
        self.assertNotIn('Traceback', stderr.getvalue())
        load.assert_not_called()

    def test_corrupt_selected_slice_is_usage_error_and_leaves_no_output(self):
        checkpoint, selected = self.checkpoint()
        manifest = json.loads(self.manifest.read_text())
        entry = next(e for e in manifest['slices'] if e['id'] == selected)
        (self.manifest.parent / entry['file']).write_bytes(b'corrupt')
        output = self.folder / 'sample'
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as exc:
            sample_main([str(checkpoint), '--manifest', str(self.manifest), '--slice-id', selected,
                         '--output', str(output)])
        self.assertEqual(exc.exception.code, 2)
        self.assertIn('SHA-256 mismatch', stderr.getvalue())
        self.assertNotIn('Traceback', stderr.getvalue())
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
