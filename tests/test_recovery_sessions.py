"""tracks/recovery/sessions.py: the benchmark across two observing sessions, on synthetic sessions
(no download needed), and the committed manifest of the real ones."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from tracks.recovery import background, sessions
from tracks.recovery.inject import KINDS, SIGNALS, STRENGTHS

FRAMES, CHANNELS = 16, 64


def windows(n, seed, scale=1.0):
    return (np.random.default_rng(seed).normal(size=(n, FRAMES, CHANNELS)) * scale).astype(np.float32)


class WindowTests(unittest.TestCase):
    def test_slots_skip_an_excluded_band_and_are_unchanged_without_one(self):
        n_chan = 1 << 20
        plain = background.slots(n_chan)
        guarded = background.slots(n_chan, exclude=[(700_000, 800_001)])
        self.assertFalse(np.any((guarded + background.WIDTH > 700_000) & (guarded < 800_001)))
        self.assertEqual(set(guarded), {s for s in plain if s + background.WIDTH <= 700_000 or s >= 800_001})
        z = np.random.default_rng(0).normal(size=(FRAMES, 40000)).astype(np.float32)
        np.testing.assert_array_equal(background.cut(z, 50, seed=3)[0], background.cut(z, 50, seed=3, exclude=())[0])
        first, _ = background.cut(z, 50, seed=3, exclude=[(10000, 20000)])
        self.assertFalse(np.any((first + background.WIDTH > 10000) & (first < 20000)))

    def test_committed_manifest_covers_both_sessions_as_the_tool_writes_it(self):
        text = sessions.MANIFEST.read_text()
        manifest = json.loads(text)
        self.assertEqual(background.manifest_text(manifest), text)
        self.assertEqual(set(manifest["scans"]), set(sessions.ROLES))
        scans = manifest["scans"]
        for role, scan in scans.items():
            self.assertEqual(scan["shape"], [sessions.WINDOWS, FRAMES, CHANNELS], role)
            self.assertEqual(scan["first_channels"], sorted(set(scan["first_channels"])), role)
            self.assertTrue(set(scan["first_channels"]) <= set(background.slots(1 << 20).tolist()), role)
            self.assertAlmostEqual(scan["foff_hz"], -2.7939677238464355, places=9)
            self.assertAlmostEqual(scan["tsamp_s"], 18.253611008, places=6)
        for part, url in sessions.SCANS_2020.items():
            self.assertEqual(scans[f"2020-{part}"]["url"], url)
        self.assertGreater(scans["2020-test"]["tstart_mjd"] - scans["2016-test"]["tstart_mjd"], 1000, "independent sessions, years apart")
        # The 2016 windows alternate along the band and stay clear of Voyager's carrier.
        cal, test = scans["2016-calibration"], scans["2016-test"]
        both = sorted(cal["first_channels"] + test["first_channels"])
        self.assertEqual(both[0::2], cal["first_channels"])
        self.assertEqual(both[1::2], test["first_channels"])
        guard = range(test["carrier_channel"] - test["guard_channels"] - CHANNELS + 1, test["carrier_channel"] + test["guard_channels"] + 1)
        self.assertFalse(set(both) & set(guard))
        self.assertAlmostEqual(test["guard_channels"] * abs(test["foff_hz"]) / 1e3, sessions.GUARD_KHZ, delta=0.01)
        self.assertAlmostEqual(test["carrier_mhz"], 8419.297, delta=0.001)
        self.assertEqual(test["file_sha256"], sessions.VOYAGER_2016["sha256"])

    def test_the_2016_file_is_downloaded_once_and_checked(self):
        payload = b"not really hdf5"
        response = mock.MagicMock()
        response.__enter__.return_value = io.BytesIO(payload)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(sessions.VOYAGER_2016, sha256=hashlib.sha256(payload).hexdigest()), \
                mock.patch("urllib.request.urlopen", return_value=response) as urlopen:
            path = sessions.fetch_2016(tmp)
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(sessions.fetch_2016(tmp), path)
            self.assertEqual(urlopen.call_count, 1, "the cached copy is reused")
            path.write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                sessions.fetch_2016(tmp)

    def test_missing_windows_name_the_build_command(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(FileNotFoundError, "tracks.recovery.sessions build"):
            sessions.load(tmp, "2016-test")


class AcrossSessionTests(unittest.TestCase):
    def test_mcnemar(self):
        self.assertEqual(sessions.mcnemar(0, 0), 1.0)
        self.assertEqual(sessions.mcnemar(5, 5), 1.0)
        self.assertAlmostEqual(sessions.mcnemar(0, 10), 2 / 1024)
        self.assertEqual(sessions.mcnemar(3, 9), sessions.mcnemar(9, 3))
        self.assertAlmostEqual(sessions.mcnemar(2, 8), 2 * (1 + 10 + 45) / 1024)

    def evaluate(self, scale_2016):
        data = {"2020-calibration": windows(1200, 1), "2020-test": windows(1200, 2),
                "2016-calibration": windows(1200, 3, scale_2016), "2016-test": windows(1200, 4, scale_2016)}
        return sessions.evaluate_sessions(data, seed=5, per_cell=10)

    def test_like_sessions_agree_and_every_twin_is_counted(self):
        pairs, across = self.evaluate(1.0)
        self.assertEqual(set(pairs), {f"thresholds {a}, test {b}" for a in sessions.SESSIONS for b in sessions.SESSIONS})
        self.assertEqual(len({r["key_sha256"] for r in pairs.values()}), 1, "the same injections in every pairing")
        for variant in ("own thresholds", "2020 thresholds"):
            for name, rates in across[variant].items():
                for rate, groups in rates.items():
                    signals = groups["signals"]
                    self.assertEqual(sum(v for k, v in signals.items() if k != "p"), len(SIGNALS) * len(STRENGTHS) * 10)
                    self.assertEqual(sum(groups["cells"]["tone"][f"{s:g}"]["both"] for s in STRENGTHS), groups["classes"]["tone"]["both"])
                    self.assertGreater(signals["p"], 0.001, f"{variant} {name} {rate}: like sessions shouldn't differ")
                    flagged = pairs["thresholds 2020, test 2020"]["recovery"][name][rate]["tone"]
                    self.assertEqual(sum(c["flagged"] for c in flagged.values()), groups["classes"]["tone"]["both"] + groups["classes"]["tone"]["2020 only"])
        self.assertEqual(set(across["own thresholds"]["energy"]["0.01"]["classes"]), set(KINDS))

    def test_a_noisier_session_shows_up_in_false_alarms_and_recovery(self):
        pairs, across = self.evaluate(1.3)
        fraction = lambda key, name: pairs[key]["false_alarms"][name]["0.01"]["fraction"]
        for name in ("dedoppler", "energy"):
            self.assertGreater(fraction("thresholds 2020, test 2016", name), 0.05, f"{name}: 2020's threshold doesn't hold on noisier data")
            self.assertLess(fraction("thresholds 2016, test 2016", name), 0.04, f"{name}: its own does")
        own = across["own thresholds"]["dedoppler"]["0.01"]["signals"]
        self.assertGreater(own["2020 only"], own["2016 only"])
        self.assertLess(own["p"], 1e-3, "at their own thresholds, the noisier session recovers less")

    def test_unlike_injections_are_refused(self):
        spec = {"kind": "tone", "strength": 5.0}
        flags = {"energy": {"0.01": np.array([True])}}
        with self.assertRaisesRegex(ValueError, "same injections"):
            sessions.across_sessions({"2020": ([spec], flags), "2016": ([{**spec, "strength": 7.0}], flags)})

    def test_the_report_tables_name_every_pairing(self):
        pairs, across = self.evaluate(1.0)
        text = sessions.table(pairs, across)
        for key in pairs:
            self.assertIn(f"## Recovery: {key}", text)
        self.assertIn("at each session's own thresholds", text)
        self.assertIn("at the 2020 thresholds", text)
        self.assertIn("McNemar p", text)


if __name__ == "__main__":
    unittest.main()
