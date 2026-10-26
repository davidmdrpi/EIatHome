"""Scientific validity and a complete CPU training/checkpoint cycle; unittest only."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tracks.llm.data import Tokenizer, digest, groups, load_slices, markov_probabilities, split_records, surrogate
from tracks.llm.model import SignalLM, evaluate, fit, generate, windows
from tracks.llm.sample import load_checkpoint
from tracks.llm.train import parser, run

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "web/data/slices/index.json"
torch.set_num_threads(1)


class SignalDataTests(unittest.TestCase):
    def test_published_data_requires_explicit_smoke_mode(self):
        records = load_slices(MANIFEST)
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
        records = copy.deepcopy(load_slices(MANIFEST))
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
        original = json.loads(MANIFEST.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            entry = copy.deepcopy(original["slices"][0])
            payload = (MANIFEST.parent / entry["file"]).read_bytes()
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


class SignalModelTests(unittest.TestCase):
    def test_causal_mask_in_train_and_eval(self):
        torch.manual_seed(5)
        model = SignalLM(vocab_size=8, context=10, width=16, layers=2, heads=2, dropout=0)
        a = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        b = a.clone()
        b[:, 4:] = 0
        for training in [True, False]:
            model.train(training)
            with torch.no_grad():
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
            args = parser().parse_args(["--output", str(output), "--allow-within-session",
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
            self.assertEqual(generate(model, sample["prompt"], 32, args.seed), sample["generated"])
            np.testing.assert_allclose(np.load(output / "generated_frames.npy"), tokenizer.decode(sample["generated"]))
            again, _, _ = load_checkpoint(output / "real.pt")
            with torch.no_grad():
                x = torch.tensor([sample["prompt"]])
                torch.testing.assert_close(model(x), again(x), rtol=0, atol=0)
            self.assertEqual(checkpoint["splits"], report["splits"])
            records = load_slices(MANIFEST)
            partitions, _ = split_records(records, allow_within_session=True)
            expected = Tokenizer.fit([r["frames"] for r in partitions["train"]], 4, args.seed)
            self.assertEqual(tokenizer.state(), expected.state())
            self.assertEqual(json.loads((output / "report.json").read_text())["dataset_sha256"], report["dataset_sha256"])
            # Mutate only test samples. Tokenizer, early stopping and fitted weights must agree.
            modified = copy.deepcopy(json.loads(MANIFEST.read_text()))
            test_ids = {r["entry"]["id"] for r in partitions["test"]}
            folder = Path(tmp) / "slices"
            folder.mkdir()
            for entry in modified["slices"]:
                payload = (MANIFEST.parent / entry["file"]).read_bytes()
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
            with self.assertRaisesRegex(ValueError, "3 independent sessions"):
                run(args)


if __name__ == "__main__":
    unittest.main()
