"""tracks/recovery: injections, blinded dealing, detectors and calibrated scoring, on synthetic
backgrounds (no download needed)."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tracks.recovery import background
from tracks.recovery.detectors import dedoppler, drift_trials, energy
from tracks.recovery.inject import DECOYS, KINDS, SIGNALS, STRENGTHS, deal, design, draw, render
from tracks.recovery.run import code_identity, evaluate, threshold, wilson

FRAMES, CHANNELS = 49, 64


def noise(n, seed):
    return np.random.default_rng(seed).normal(size=(n, FRAMES, CHANNELS)).astype(np.float32)


def specs(kind, count=40, strength=10):
    return [draw(np.random.default_rng(i), kind, strength, FRAMES, CHANNELS) for i in range(count)]


class InjectionTests(unittest.TestCase):
    def test_every_signal_has_its_strength_as_norm_and_stays_inside_the_window(self):
        for kind in KINDS:
            for spec in specs(kind):
                s = render(spec, FRAMES, CHANNELS)
                self.assertAlmostEqual(np.linalg.norm(s), spec["strength"] if kind != "none" else 0, places=9, msg=kind)
                if kind not in ("none", "pulse"):
                    # Nothing reaches the window's outer channels (the margin), whatever the drift.
                    self.assertEqual(np.abs(s[:, :2]).sum() + np.abs(s[:, -2:]).sum(), 0, kind)
                self.assertGreaterEqual(s.min(), 0, kind)

    def test_tracks_follow_their_drift(self):
        for kind in ("tone", "wideband", "stationary"):
            for spec in specs(kind):
                s = render(spec, FRAMES, CHANNELS)
                centroid = (s * np.arange(CHANNELS)).sum(axis=1) / s.sum(axis=1)
                slope, intercept = np.polyfit(np.arange(FRAMES), centroid, 1)
                self.assertAlmostEqual(slope, spec["drift"], delta=0.02, msg=kind)
                self.assertAlmostEqual(intercept, spec["start"], delta=0.5, msg=kind)
        self.assertTrue(all(s["drift"] == 0 for s in specs("stationary")))
        self.assertGreater(max(abs(s["drift"]) for s in specs("tone")), 1, "drifts span Voyager's")

    def test_class_shapes(self):
        for spec in specs("pulse"):
            s = render(spec, FRAMES, CHANNELS)
            self.assertEqual(np.flatnonzero(s.sum(axis=1)).tolist(), [spec["frame"]])
            self.assertEqual(np.count_nonzero(s[spec["frame"]]), CHANNELS, "broadband")
        for spec in specs("intermittent"):
            on = render(spec, FRAMES, CHANNELS).sum(axis=1) > 0
            expected = [(t + spec["phase"]) % spec["period"] < spec["on"] for t in range(FRAMES)]
            self.assertEqual(on.tolist(), expected)
            self.assertTrue(0.15 < on.mean() < 0.6)
        for spec in specs("sidebands"):
            frame = render(spec, FRAMES, CHANNELS)[0]
            peaks = sorted(np.argsort(frame)[-6:])  # three narrow lines, two channels each at most
            self.assertGreaterEqual(peaks[-1] - peaks[0], 2 * spec["offset"] - 2)
        for spec in specs("fsk"):
            s = render(spec, FRAMES, CHANNELS)
            centroid = (s * np.arange(CHANNELS)).sum(axis=1) / s.sum(axis=1)
            hop = centroid - (spec["start"] + spec["drift"] * np.arange(FRAMES))
            expected = [spec["shift"] * spec["symbols"][t // spec["symbol"]] for t in range(FRAMES)]
            np.testing.assert_allclose(hop, expected, atol=1e-9)
        self.assertEqual(np.abs(render({"kind": "none", "strength": 0.0}, FRAMES, CHANNELS)).sum(), 0)

    def test_draws_are_seeded(self):
        a = [draw(np.random.default_rng(5), k, 7, FRAMES, CHANNELS) for k in KINDS]
        b = [draw(np.random.default_rng(5), k, 7, FRAMES, CHANNELS) for k in KINDS]
        self.assertEqual(a, b)
        self.assertNotEqual(a, [draw(np.random.default_rng(6), k, 7, FRAMES, CHANNELS) for k in KINDS])
        json.dumps(a)  # specs are plain JSON

    def test_design_fills_every_cell_then_signal_free_windows(self):
        cells = design(3, 200)
        self.assertEqual(len(cells), 200)
        for kind in KINDS:
            for s in STRENGTHS:
                self.assertEqual(cells.count((kind, s)), 3)
        self.assertEqual(cells.count(("none", 0.0)), 200 - 3 * len(KINDS) * len(STRENGTHS))
        with self.assertRaises(ValueError):
            design(10, 100)


class DealerTests(unittest.TestCase):
    def test_detectors_get_arrays_with_opaque_ids_and_the_key_rebuilds_each_window(self):
        backgrounds = noise(120, 1)
        blind, key = deal(backgrounds, design(1, 120), seed=3)
        self.assertEqual(len(blind), 120)
        self.assertEqual(sorted(v["background"] for v in key.values()), list(range(120)), "each window used once")
        for window_id, window in blind:
            self.assertRegex(window_id, r"^w[0-9a-f]{16}$")
            self.assertIsInstance(window, np.ndarray)
            spec = key[window_id]
            np.testing.assert_allclose(window, backgrounds[spec["background"]] + render(spec, FRAMES, CHANNELS), atol=1e-5)
        # The dealing order says nothing about the kinds: signal-free windows are spread through it.
        none_positions = [i for i, (w, _) in enumerate(blind) if key[w]["kind"] == "none"]
        self.assertGreater(max(none_positions) - min(none_positions) + 1, len(none_positions), "not one block")
        # Nor do the kinds follow the windows' positions in the coarse channel (and its interference).
        none_backgrounds = sorted(v["background"] for v in key.values() if v["kind"] == "none")
        self.assertGreater(none_backgrounds[-1] - none_backgrounds[0] + 1, len(none_backgrounds), "signal-free windows aren't one block")
        backgrounds_in_order = [key[w]["background"] for w, _ in blind]
        self.assertNotEqual(backgrounds_in_order, sorted(backgrounds_in_order), "not dealt in window order")
        self.assertNotEqual([w for w, _ in blind], sorted(key), "not dealt in id order")

    def test_dealing_is_reproducible_and_seeded(self):
        backgrounds = noise(60, 2)
        a, key_a = deal(backgrounds, design(0, 60), seed=1)
        b, key_b = deal(backgrounds, design(0, 60), seed=1)
        self.assertEqual(key_a, key_b)
        self.assertEqual([w for w, _ in a], [w for w, _ in b])
        _, key_c = deal(backgrounds, design(0, 60), seed=2)
        self.assertNotEqual(key_a, key_c)


class DetectorTests(unittest.TestCase):
    def reference(self, window):
        """Every straight track inside the window, by explicit loops."""
        best = -np.inf
        for drift in drift_trials(FRAMES):
            offsets = [round(drift * t) for t in range(FRAMES)]
            for c in range(CHANNELS):
                if all(0 <= c + o < CHANNELS for o in offsets):
                    best = max(best, sum(window[t, c + o] for t, o in enumerate(offsets)))
        return best / np.sqrt(FRAMES)

    def test_dedoppler_is_the_best_track_and_never_wraps(self):
        windows = noise(3, 4).astype(np.float64)
        # A bright track that leaves the right edge and would continue at the left with wrapping.
        windows[2] = 0
        for t in range(FRAMES):
            windows[2, t, (50 + t) % CHANNELS] = 1
        scores = dedoppler(windows)
        for window, score in zip(windows, scores):
            self.assertAlmostEqual(score, self.reference(window), places=9)
        self.assertLess(scores[2], 14 / np.sqrt(FRAMES) + 1e-9, "only the 14 frames inside the window count")

    def test_dedoppler_recovers_a_tone_near_its_strength(self):
        tone = lambda drift, start: render({"kind": "tone", "strength": 10.0, "drift": drift, "start": start, "width": 1.0}, FRAMES, CHANNELS)
        # A tone on whole channels, on a trial (zero drift): all of its power, its full strength.
        self.assertAlmostEqual(dedoppler(tone(0.0, 30.0)[None])[0], 10.0, places=6)
        # Drifting through fractional channels, power splits between two channels per frame.
        drift = float(drift_trials(FRAMES)[59])
        self.assertAlmostEqual(drift, 0.816, places=3)
        self.assertTrue(8.5 < dedoppler(tone(drift, 10.0)[None])[0] <= 10.0)

    def test_energy_is_standardized_on_unit_noise_and_sees_pulses(self):
        scores = energy(noise(2000, 5))
        self.assertAlmostEqual(scores.mean(), 0, delta=0.1)
        self.assertAlmostEqual(scores.std(), 1, delta=0.1)
        pulse = render({"kind": "pulse", "strength": 28.0, "frame": 10}, FRAMES, CHANNELS)
        empty = np.zeros((1, FRAMES, CHANNELS))
        self.assertAlmostEqual(energy(pulse[None])[0] - energy(empty)[0], 28 ** 2 / np.sqrt(2 * FRAMES * CHANNELS), places=9)


class ScoringTests(unittest.TestCase):
    def test_thresholds_hold_the_false_alarm_rate_they_were_set_for(self):
        # Enough calibration scores for the method, not sampling noise, to decide: at 0.1%, the
        # threshold rests on about 50 exceedances here (4 with 4096 calibration windows).
        scores = np.random.default_rng(0).normal(size=50000)
        for target in (0.01, 0.001):
            cut = threshold(scores, target)
            self.assertLessEqual(np.mean(scores > cut), target)
            fresh = np.random.default_rng(1).normal(size=200000)
            self.assertAlmostEqual(np.mean(fresh > cut), target, delta=target * 0.6)

    def test_wilson_interval(self):
        low, high = wilson(10, 100)
        self.assertAlmostEqual(low, 0.0552, places=3)
        self.assertAlmostEqual(high, 0.1744, places=3)
        self.assertEqual(wilson(0, 0), [0.0, 1.0])
        self.assertEqual(wilson(0, 50)[0], 0.0)

    def test_blinded_evaluation_end_to_end(self):
        calibration, test = noise(1500, 6), noise(1500, 7)
        report, blinded = evaluate(calibration, test, seed=0, per_cell=12)
        self.assertEqual(report["windows"]["test_signal_free"], 1500 - 12 * len(KINDS) * len(STRENGTHS))
        for name, detect in (("dedoppler", dedoppler), ("energy", energy)):  # set on the calibration scan alone
            self.assertEqual(report["thresholds"][name], {f"{r:g}": threshold(detect(calibration), r) for r in (0.01, 0.001)})
        for name in ("dedoppler", "energy"):
            for label, fa in report["false_alarms"][name].items():
                self.assertLessEqual(fa["interval"][0], float(label) * 2, f"{name} {label}")
        dd, en = report["recovery"]["dedoppler"]["0.01"], report["recovery"]["energy"]["0.01"]
        # Well-known behaviour of the two baselines, as a check of the whole chain.
        for kind in SIGNALS:
            self.assertEqual(dd[kind]["28"]["fraction"], 1.0, kind)
        self.assertGreaterEqual(dd["tone"]["10"]["fraction"], 0.9)
        self.assertLessEqual(dd["tone"]["5"]["fraction"], 0.6)
        self.assertGreater(dd["tone"]["10"]["fraction"], en["tone"]["10"]["fraction"], "a matched search beats a radiometer on tones")
        self.assertEqual(dd["pulse"]["28"]["fraction"], 0.0, "a pulse has no track")
        self.assertEqual(en["pulse"]["28"]["fraction"], 1.0)
        self.assertIn("stationary", DECOYS)
        paired = report["paired"]["0.01"]["dedoppler vs energy"]["tone"]["10"]
        self.assertEqual(sum(paired.values()), 12)
        self.assertEqual(paired["both"] + paired["dedoppler only"], dd["tone"]["10"]["flagged"])
        # The key joins back: rescoring each blind window by id gives the stored score.
        ids, key = blinded["ids"], blinded["key"]
        i = ids.index(next(w for w in ids if key[w]["kind"] == "tone" and key[w]["strength"] == 10))
        window = test[key[ids[i]]["background"]] + render(key[ids[i]], FRAMES, CHANNELS)
        self.assertAlmostEqual(dedoppler(window[None])[0], blinded["scores"]["dedoppler"][i], places=4)


    def test_every_pair_of_detectors_is_compared_when_more_are_added(self):
        # The route to learned detectors: add one to the baselines. No pairing may be lost.
        def brightest(windows):
            return np.asarray(windows).max(axis=(1, 2))
        brightest.config = {"score": "brightest pixel"}

        calibration, test = noise(600, 8), noise(600, 9)
        two, _ = evaluate(calibration, test, seed=1, per_cell=4)
        three, _ = evaluate(calibration, test, seed=1, per_cell=4,
                            detectors={"dedoppler": dedoppler, "energy": energy, "brightest": brightest})
        for label in ("0.01", "0.001"):
            pairs = three["paired"][label]
            self.assertEqual(set(pairs), {"dedoppler vs energy", "dedoppler vs brightest", "energy vs brightest"})
            self.assertEqual(pairs["dedoppler vs energy"], two["paired"][label]["dedoppler vs energy"])
            for name, kinds in pairs.items():
                first, second = name.split(" vs ")
                for kind in KINDS:
                    for s in STRENGTHS:
                        cell = kinds[kind][f"{s:g}"]
                        self.assertEqual(sum(cell.values()), 4, f"{name} {kind} {s}")
                        flagged = three["recovery"][first][label][kind][f"{s:g}"]["flagged"]
                        self.assertEqual(cell["both"] + cell[f"{first} only"], flagged)
                        flagged = three["recovery"][second][label][kind][f"{s:g}"]["flagged"]
                        self.assertEqual(cell["both"] + cell[f"{second} only"], flagged)
        self.assertEqual(three["detectors"]["brightest"]["score"], "brightest pixel")

    def test_reports_record_the_code_and_each_detector(self):
        report, _ = evaluate(noise(300, 10), noise(300, 11), seed=0, per_cell=2)
        package = Path(__file__).resolve().parents[1] / "tracks" / "recovery"
        expected = hashlib.sha256(b"".join(f.read_bytes() for f in sorted(package.glob("*.py")))).hexdigest()
        self.assertEqual(report["code"]["sha256"], expected)
        self.assertEqual(report["code"], code_identity())
        if report["code"]["git"] is not None:
            self.assertRegex(report["code"]["git"]["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(report["detectors"]["dedoppler"]["function"], "tracks.recovery.detectors.dedoppler")
        self.assertIn("never wrapping", report["detectors"]["dedoppler"]["tracks"])
        self.assertIn("score", report["detectors"]["energy"])
        self.assertEqual(report["design"]["strengths"], list(STRENGTHS))
        self.assertEqual(report["design"]["window"], [FRAMES, CHANNELS])


class BackgroundTests(unittest.TestCase):
    def test_windows_skip_the_edges_and_dc_and_never_overlap(self):
        n_chan = 1 << 20
        starts = background.slots(n_chan)
        edge, mid = int(n_chan * background.EDGE_FRACTION), n_chan // 2
        self.assertGreaterEqual(starts.min(), edge)
        self.assertLessEqual(starts.max() + background.WIDTH, n_chan - edge)
        self.assertTrue(np.all(np.diff(starts) >= background.WIDTH))
        near_dc = (starts + background.WIDTH > mid - background.DC_GUARD) & (starts < mid + background.DC_GUARD)
        self.assertFalse(near_dc.any())
        self.assertGreater(len(starts), background.WINDOWS)

    def test_committed_manifest_is_complete_and_written_as_the_tool_writes_it(self):
        text = background.MANIFEST.read_text()
        manifest = json.loads(text)
        self.assertEqual(background.manifest_text(manifest), text)
        self.assertEqual(set(manifest["scans"]), set(background.ROLES))
        for role, scan in manifest["scans"].items():
            self.assertEqual(scan["url"], background.SCANS[role])
            self.assertEqual(scan["shape"], [background.WINDOWS, FRAMES, CHANNELS])
            self.assertEqual(len(scan["first_channels"]), background.WINDOWS)
            self.assertEqual(scan["first_channels"], sorted(set(scan["first_channels"])))
            self.assertTrue(set(scan["first_channels"]) <= set(background.slots(1 << 20).tolist()))
        starts = [s["tstart_mjd"] for s in manifest["scans"].values()]
        self.assertLess(max(starts) - min(starts), 0.5, "one session, as every report says")

    def test_cut_is_seeded_and_loading_checks_the_manifest(self):
        z = np.random.default_rng(0).normal(size=(FRAMES, 40000)).astype(np.float32)
        first, windows = background.cut(z, 50, seed=3)
        again, same = background.cut(z, 50, seed=3)
        np.testing.assert_array_equal(first, again)
        np.testing.assert_array_equal(windows, same)
        self.assertTrue(np.all(np.diff(first) > 0))
        np.testing.assert_array_equal(windows[7], z[:, first[7]:first[7] + background.WIDTH])
        with tempfile.TemporaryDirectory() as tmp:
            cache, manifest = Path(tmp), Path(tmp) / "manifest.json"
            manifest.write_text(json.dumps({"scans": {"test": {"shape": list(windows.shape), "sha256": background.digest(windows)}}}))
            with self.assertRaisesRegex(FileNotFoundError, "python -m tracks.recovery.background"):
                background.load(cache, "test", manifest)
            np.save(background.cache_path(cache, "test"), windows)
            np.testing.assert_array_equal(background.load(cache, "test", manifest), windows)
            windows[0, 0, 0] += 1
            np.save(background.cache_path(cache, "test"), windows)
            with self.assertRaisesRegex(ValueError, "doesn't match"):
                background.load(cache, "test", manifest)


if __name__ == "__main__":
    unittest.main()
