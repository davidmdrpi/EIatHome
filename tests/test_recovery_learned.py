"""tracks/recovery/learned.py: the learned detectors and their training protocol, with small
models on synthetic backgrounds. Needs torch (run with the signal-LM tests)."""

import math
import unittest
from unittest import mock

import numpy as np
import torch

from tracks.llm.model import evaluate as lm_evaluate
from tracks.recovery import run
from tracks.recovery.detectors import DETECTORS
from tracks.recovery.inject import STRENGTHS, draw, render
from tracks.recovery.learned import (digest, fit_gru, learned_detectors, state_digest, token_bits,
                                     track_a_normalise, training_windows)

FRAMES, CHANNELS = 49, 64
SMALL_GRU = dict(hidden=16, lr=3e-3, batch=64, epochs=3, patience=2)
SMALL_TOKENS = dict(vocab=8, width=16, layers=1, heads=2, dropout=0.0, lr=3e-3, batch=32, epochs=2, patience=1)
torch.set_num_threads(1)


def noise(n, seed):
    return np.random.default_rng(seed).normal(size=(n, FRAMES, CHANNELS)).astype(np.float32)


def js_normalise(raw):
    """A line-by-line port of normalise() in web/js/signals.js, for one snippet."""
    flat = np.asarray(raw, dtype=np.float32).ravel()
    sorted_ = np.sort(flat)
    median = sorted_[len(sorted_) >> 1]
    deviations = np.sort(np.abs(sorted_ - median))
    scale = 1.4826 * deviations[len(deviations) >> 1] or 1
    return np.arcsinh((flat - median) / scale).reshape(np.shape(raw))


class TrainingDataTests(unittest.TestCase):
    def test_fit_and_development_windows_split_the_train_scan_with_tones_in_half(self):
        train = noise(200, 1)
        fit, dev, info = training_windows(train, seed=4)
        self.assertEqual((len(fit), len(dev)), (150, 50))
        # Recover where each window came from: every window is a train window plus a tone or nothing.
        rng = np.random.default_rng(4)
        order = rng.permutation(len(train))
        for windows, idx in ((fit, order[:150]), (dev, order[150:])):
            added = windows - train[idx]
            with_tone = np.abs(added).sum(axis=(1, 2)) > 0
            self.assertEqual(with_tone.sum(), len(idx) // 2)
            norms = np.linalg.norm(added[with_tone].reshape(with_tone.sum(), -1), axis=1)
            self.assertTrue(set(np.round(norms, 3)) <= set(float(s) for s in STRENGTHS))
        self.assertEqual(set(order[:150]) & set(order[150:]), set())
        self.assertEqual(info["windows_sha256"], digest(fit, dev))
        again = training_windows(train, seed=4)
        np.testing.assert_array_equal(again[0], fit)
        self.assertNotEqual(training_windows(train, seed=5)[2]["windows_sha256"], info["windows_sha256"])

    def test_preprocessing_is_track_as(self):
        windows = noise(3, 2) * 3 + 1
        windows[1, 10] += 40  # a bright frame, compressed by asinh
        np.testing.assert_allclose(track_a_normalise(windows), np.stack([js_normalise(w) for w in windows]), rtol=1e-6, atol=1e-6)


class ModelTests(unittest.TestCase):
    def test_gru_training_is_seeded_and_keeps_the_best_development_epoch(self):
        fit, dev, _ = training_windows(noise(160, 3), seed=0)
        a, history, best = fit_gru(fit, dev, seed=7, **SMALL_GRU)
        b, _, _ = fit_gru(fit, dev, seed=7, **SMALL_GRU)
        self.assertEqual(state_digest(a), state_digest(b))
        self.assertEqual(best, 1 + int(np.argmin([h["dev_mse"] for h in history])))
        # With these settings the development loss rises after epoch 5: the weights returned
        # must be epoch 5's, not the last epoch's.
        model, history, best = fit_gru(fit, dev, seed=7, hidden=16, lr=3e-2, batch=64, epochs=6, patience=6)
        self.assertEqual((best, len(history)), (5, 6))
        z = torch.from_numpy(track_a_normalise(dev))
        with torch.no_grad():
            dev_mse = torch.nn.functional.mse_loss(model(z[:, :-1]), z[:, 1:]).item()
        self.assertAlmostEqual(dev_mse, history[best - 1]["dev_mse"], places=6)

    def test_token_scores_match_the_llm_tracks_own_evaluation(self):
        detectors, state = learned_detectors(noise(120, 4), seed=0, gru=SMALL_GRU, tokens=SMALL_TOKENS)
        token_detector = detectors["token-surprise"]
        windows = noise(5, 5)
        surprise = token_detector(windows)
        # Rebuild the model from the saved state and score one window with tracks/llm's evaluate().
        from tracks.llm.data import Tokenizer
        from tracks.llm.model import SignalLM
        tokenizer = Tokenizer(**state["tokenizer"])
        config = dict(vocab_size=SMALL_TOKENS["vocab"], context=FRAMES - 1, width=SMALL_TOKENS["width"],
                      layers=SMALL_TOKENS["layers"], heads=SMALL_TOKENS["heads"], dropout=0.0)
        model = SignalLM(**config)
        model.load_state_dict(state["token"])
        for i in (0, 3):
            tokens = tokenizer.encode(windows[i])
            self.assertAlmostEqual(surprise[i], lm_evaluate(model, [tokens])["bits_per_token"], places=5)
        # Gain: the same tokens' bits under the training token frequencies, minus the model's.
        unigram = np.array(state["unigram"])
        tokens = tokenizer.encode(windows[2])
        expected = -np.log2(unigram[tokens[1:]]).mean() - surprise[2]
        self.assertAlmostEqual(detectors["token-gain"](windows)[2], expected, places=6)
        model_bits, unigram_bits = token_bits(tokenizer, model, unigram, windows)
        np.testing.assert_allclose(model_bits, surprise, rtol=1e-6)

    def test_gru_statistics(self):
        detectors, state = learned_detectors(noise(120, 6), seed=0, gru=SMALL_GRU, tokens=SMALL_TOKENS)
        windows = noise(4, 7)
        z = track_a_normalise(windows)
        null = (z[:, 1:] ** 2).mean(axis=(1, 2))
        np.testing.assert_allclose(detectors["gru-gain"](windows), null - detectors["gru-surprise"](windows), rtol=1e-5, atol=1e-6)
        pulse = windows.copy()
        pulse[0] += render(draw(np.random.default_rng(0), "pulse", 60, FRAMES, CHANNELS), FRAMES, CHANNELS)
        self.assertGreater(detectors["gru-surprise"](pulse)[0], detectors["gru-surprise"](windows)[0], "a pulse is surprising")

    def test_each_detector_records_its_model_and_training(self):
        train = noise(120, 8)
        detectors, state = learned_detectors(train, seed=3, gru=SMALL_GRU, tokens=SMALL_TOKENS)
        fit, dev, info = training_windows(train, seed=3)
        self.assertEqual(set(detectors), {"gru-surprise", "gru-gain", "token-surprise", "token-gain"})
        for name, detect in detectors.items():
            config = detect.config
            self.assertEqual(config["training"]["windows_sha256"], digest(fit, dev), name)
            self.assertEqual(config["training"]["scan"], "train")
            self.assertGreater(config["parameters"], 0)
            self.assertIn("statistic", config)
            self.assertRegex(config["state_sha256"], r"^[0-9a-f]{64}$")
        gru_state = {k: v for k, v in state["gru"].items()}
        self.assertEqual(detectors["gru-gain"].config["state_sha256"],
                         digest(*(v.numpy() for v in gru_state.values())))
        self.assertEqual(state["training"], info)


class ProtocolTests(unittest.TestCase):
    def test_learned_detectors_train_on_the_train_scan_only(self):
        loaded = []

        def load(cache, role):
            loaded.append(role)
            return noise(40, {"train": 1, "calibration": 2, "test": 3}[role])

        with mock.patch.object(run.background, "load", side_effect=load), \
                mock.patch("tracks.recovery.learned.learned_detectors", return_value=({}, {})) as trained:
            detectors, _ = run.detectors_for("cache", learned=True, train_seed=9)
        self.assertEqual(loaded, ["train"])
        np.testing.assert_array_equal(trained.call_args.args[0], noise(40, 1))
        self.assertEqual(trained.call_args.kwargs["seed"], 9)
        self.assertEqual(set(detectors), set(DETECTORS))
        self.assertEqual(run.detectors_for("cache", learned=False), (dict(DETECTORS), None))

    def test_learned_detectors_are_compared_with_every_baseline(self):
        detectors, _ = learned_detectors(noise(120, 10), seed=0, gru=SMALL_GRU, tokens=SMALL_TOKENS)
        report, _ = run.evaluate(noise(300, 11), noise(300, 12), seed=0, per_cell=2, detectors={**DETECTORS, **detectors})
        names = list(DETECTORS) + list(detectors)
        self.assertEqual(len(report["paired"]["0.01"]), math.comb(len(names), 2))
        for model in detectors:
            for baseline in DETECTORS:
                self.assertIn(f"{baseline} vs {model}", report["paired"]["0.01"])
            self.assertEqual(report["detectors"][model]["training"]["scan"], "train")


if __name__ == "__main__":
    unittest.main()
