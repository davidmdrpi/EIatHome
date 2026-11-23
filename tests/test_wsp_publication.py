"""Publication coverage and attempt preservation; no API calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import wsp_publication as publication


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / "results"
        self.folder.mkdir()
        self.report = {"sessions_benchmark": "signal-lm-sessions-bench/v1",
                       "settings": {"methods": publication.METHODS, "seeds": publication.SEEDS}, "seeds": {}}
        for seed in publication.SEEDS:
            methods = {}
            for method in publication.METHODS:
                methods[method] = {}
                identity = {"seed": seed, "method": method}
                if method != "specialist":
                    self.write(f"edits/seed{seed}/{method}.json", identity)
                if method.startswith("llm"):
                    summary = {"stopped": "finished", "summary": "done", "turns": 1,
                               "editor_seconds": 1, "served_by": ["fixture"], "cost": {"dollars": 1}}
                    methods[method] = summary
                    self.write(f"transcripts/seed{seed}/{method}.json",
                               {**identity, **summary, "per_turn": [{}], "transcript": [{}]})
            self.report["seeds"][str(seed)] = {"methods": methods}
        self.write("report.json", self.report)

    def write(self, name, value):
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_complete(self):
        publication.complete(self.folder)

    def test_missing_seed_or_arm_even_when_settings_claim_complete(self):
        for change in (lambda r: r["seeds"].pop("44"),
                       lambda r: r["seeds"]["42"]["methods"].pop("llm")):
            report = copy.deepcopy(self.report)
            change(report)
            self.write("report.json", report)
            with self.assertRaises(ValueError):
                publication.complete(self.folder)

    def test_wrong_settings(self):
        self.report["settings"] = {"methods": ["random"], "seeds": [42]}
        self.write("report.json", self.report)
        with self.assertRaises(ValueError):
            publication.complete(self.folder)

    def test_missing_transcript(self):
        (self.folder / "transcripts/seed42/llm.json").unlink()
        with self.assertRaises(FileNotFoundError):
            publication.complete(self.folder)

    def test_transcript_mismatch(self):
        path = self.folder / "transcripts/seed42/llm.json"
        transcript = json.loads(path.read_text())
        transcript["cost"] = {"dollars": 2}
        path.write_text(json.dumps(transcript))
        with self.assertRaises(ValueError):
            publication.complete(self.folder)

    def test_missing_key_records_failure_without_executing_and_refuses_overwrite(self):
        output = Path(self.temp.name) / "attempt"
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "no arms executed"):
                publication.run(output)
        status = json.loads((output / "attempt.json").read_text())
        self.assertEqual(status["state"], "incomplete")
        self.assertEqual(status["failed_stage"], "preflight")
        self.assertFalse((output / "results").exists())
        with self.assertRaises(FileExistsError):
            publication.run(output)
        self.assertEqual(status, json.loads((output / "attempt.json").read_text()))

    def test_failed_run_preserves_attempt_and_does_not_publish(self):
        from tracks.llm import wsp_bench
        output = Path(self.temp.name) / "failed-api"
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fixture"}), \
             patch.object(wsp_bench, "run", side_effect=RuntimeError("private detail")), \
             patch.object(publication, "check") as check:
            with self.assertRaises(RuntimeError):
                publication.run(output)
        check.assert_not_called()
        text = (output / "attempt.json").read_text()
        self.assertNotIn("private detail", text)
        self.assertNotIn("fixture", text)
        self.assertEqual(json.loads(text)["failed_stage"], "running")

    def test_replay_failure_is_not_success(self):
        from tracks.llm import wsp_bench
        output = Path(self.temp.name) / "failed-replay"
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fixture"}), \
             patch.object(wsp_bench, "run"), \
             patch.object(publication, "check", side_effect=ValueError("mismatch")):
            with self.assertRaises(ValueError):
                publication.run(output)
        status = json.loads((output / "attempt.json").read_text())
        self.assertEqual(status["state"], "incomplete")
        self.assertEqual(status["failed_stage"], "verifying")
