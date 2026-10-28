"""tracks/recovery/browser.py: the Live Lab's recovery windows, and the Python port of its work
unit checked against the JavaScript one (web/js/recovery.js, run with Node). Needs numpy, and
Node for the cross-check."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from tracks.recovery import browser
from tracks.recovery.inject import KINDS, STRENGTHS

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "recovery"


def javascript_unit(unit):
    """The JavaScript unit's results, each dealt window's spec, and the scores."""
    run = subprocess.run(["node", str(ROOT / "tools" / "recovery_unit.mjs"), str(unit), "--details"],
                         capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(run.stdout)


@unittest.skipIf(shutil.which("node") is None and not os.environ.get("CI"), "needs Node")
class CrossCheckTests(unittest.TestCase):
    def test_python_and_javascript_units_deal_score_and_count_alike(self):
        windows, index = browser.published()
        for unit in (0, 41):
            js = javascript_unit(unit)
            results, key, scores = browser.run_unit(unit, windows, index["thresholds"])
            # The same random stream draws the same signals into the same windows, exactly.
            self.assertEqual(key, js["key"], unit)
            for name, values in scores.items():
                # Sums in a different order: equal to rounding.
                np.testing.assert_allclose(values, js["scores"][name], rtol=0, atol=1e-9, err_msg=name)
            self.assertEqual(results, js["results"], unit)
            self.assertEqual(results["signal_free"], len(windows) - len(KINDS) * len(STRENGTHS) * browser.PER_CELL)

    def test_the_random_stream_is_rng_js(self):
        script = ("import { makeRng, deriveSeed } from './web/js/rng.js';"
                  "const r = makeRng(deriveSeed(7, 'recovery'));"
                  "console.log(JSON.stringify([deriveSeed(7, 'recovery'), r.next(), r.uniform(-1.2, 1.2), r.int(4, 13), r.permutation(6)]));")
        run = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True, cwd=ROOT)
        seed = browser.derive_seed(7, "recovery")
        rng = browser.Mulberry32(seed)
        self.assertEqual([seed, rng.next(), rng.uniform(-1.2, 1.2), rng.integers(4, 13), rng.permutation(6)], json.loads(run.stdout))


class PublishedWindowTests(unittest.TestCase):
    def test_the_published_windows_match_their_index(self):
        windows, index = browser.published()
        self.assertEqual(list(windows.shape), index["shape"])
        self.assertEqual(index["shape"], [4096 // browser.STRIDE, 49, 64])
        self.assertEqual(index["test"]["sha256"], json.loads((ROOT / "tracks/recovery/backgrounds.json").read_text())["scans"]["test"]["sha256"])
        self.assertEqual(sorted(index["thresholds"]), ["dedoppler", "energy"])
        steps = windows / browser.STEP
        np.testing.assert_array_equal(steps, np.round(steps))
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(browser.PUBLISHED / "index.json", tmp)
            data = bytearray((browser.PUBLISHED / index["file"]).read_bytes())
            data[0] ^= 1
            (Path(tmp) / index["file"]).write_bytes(bytes(data))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                browser.published(Path(tmp))

    def test_quantizing_rounds_to_sixteenths_and_refuses_to_clip(self):
        x = np.array([0.0, 0.03, -0.04, 1.0, 7.9, -7.9], dtype=np.float32)
        np.testing.assert_array_equal(browser.dequantize(browser.quantize(x)), [0, 0, -0.0625, 1, 7.875, -7.875])
        with self.assertRaisesRegex(ValueError, "clip"):
            browser.quantize(np.array([8.0], dtype=np.float32))

    @unittest.skipUnless((CACHE / "background-test.npy").exists(), "needs the background cache (python -m tracks.recovery.background)")
    def test_the_published_files_are_rebuilt_exactly_from_the_backgrounds(self):
        data, index = browser.build(CACHE)
        self.assertEqual(data, (browser.PUBLISHED / browser.DATA_FILE).read_bytes())
        self.assertEqual(json.dumps(index, indent=1) + "\n", (browser.PUBLISHED / "index.json").read_text())
        self.assertEqual(hashlib.sha256(data).hexdigest(), index["sha256"])


if __name__ == "__main__":
    unittest.main()
