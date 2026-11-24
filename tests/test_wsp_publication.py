"""Publication coverage and attempt preservation; no API calls."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
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

    def test_preflight_refuses_what_the_frozen_run_would_refuse_before_any_arm(self):
        cases = [({"ANTHROPIC_API_KEY": "fixture", "ANTHROPIC_AUTH_TOKEN": "other"}, {}, "ANTHROPIC_AUTH_TOKEN is set"),
                 ({"ANTHROPIC_API_KEY": "fixture"}, {"anthropic": None}, "anthropic package isn't installed")]
        for i, (env, modules, reason) in enumerate(cases):
            output = Path(self.temp.name) / f"preflight-{i}"
            with self.subTest(reason=reason), patch.dict("os.environ", env, clear=True), patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(RuntimeError, reason):
                    publication.run(output)
                status = json.loads((output / "attempt.json").read_text())
                self.assertEqual((status["state"], status["failed_stage"]), ("incomplete", "preflight"))
                self.assertNotIn("other", json.dumps(status), "no secret recorded")

    def test_a_publication_needs_its_verified_attempt_record(self):
        attempts = Path(self.temp.name) / "attempts"
        attempts.mkdir()
        with patch.object(publication, "check"):
            with self.assertRaisesRegex(ValueError, "No verified attempt record"):
                publication.check_publication(self.folder, attempts)
            digest = publication.report_sha256(self.folder)
            (attempts / "preflight-001.json").write_text(json.dumps({"state": "incomplete"}))
            (attempts / "attempt-002.json").write_text(json.dumps({"state": "incomplete", "report_sha256": digest}))
            with self.assertRaisesRegex(ValueError, "No verified attempt record"):
                publication.check_publication(self.folder, attempts)
            (attempts / "attempt-003.json").write_text(json.dumps({"state": "verified", "report_sha256": digest}))
            self.assertEqual(publication.check_publication(self.folder, attempts).name, "attempt-003.json")
            self.write("report.json", {**self.report, "edited": True})
            with self.assertRaisesRegex(ValueError, "No verified attempt record"):
                publication.check_publication(self.folder, attempts)

    def test_failed_run_preserves_attempt_and_does_not_publish(self):
        from tracks.llm import wsp_bench
        output = Path(self.temp.name) / "failed-api"
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fixture"}), \
             patch.object(publication, "preflight_problem", return_value=None), \
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
             patch.object(publication, "preflight_problem", return_value=None), \
             patch.object(wsp_bench, "run"), \
             patch.object(publication, "check", side_effect=ValueError("mismatch")):
            with self.assertRaises(ValueError):
                publication.run(output)
        status = json.loads((output / "attempt.json").read_text())
        self.assertEqual(status["state"], "incomplete")
        self.assertEqual(status["failed_stage"], "verifying")


def scripted_send():
    """A stand-in for the API: evaluates one program, then finishes. No request leaves the machine."""
    turns = []

    def message(*blocks):
        usage = SimpleNamespace(input_tokens=1000, cache_creation_input_tokens=0, cache_read_input_tokens=0, output_tokens=200)
        return SimpleNamespace(content=list(blocks), stop_reason="tool_use", model="claude-opus-5-5", usage=usage)

    def send(request):
        turns.append(request)
        if len(turns) % 2:
            call = SimpleNamespace(type="tool_use", id=f"e{len(turns)}", name="evaluate_program",
                                   input={"program": {"start": "average", "ops": []}})
        else:
            call = SimpleNamespace(type="tool_use", id=f"f{len(turns)}", name="finish", input={"summary": "scripted"})
        return message(call)
    return send


class ContractTests(unittest.TestCase):
    """The completeness check on a real wsp_bench run's output, not a hand-built fixture. On v0 only:
    a run on v1 would compute the controls' v1 test scores before the frozen experiment."""

    def test_a_real_run_passes_the_completeness_check(self):
        from tracks.llm import session_bench, wsp_bench
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results"
            args = SimpleNamespace(benchmark="v0", output=out, methods=publication.METHODS, seeds=[42], publish=False)
            with patch.object(wsp_bench, "anthropic_send", scripted_send), contextlib.redirect_stdout(io.StringIO()):
                wsp_bench.run(args)
            with patch.object(publication, "SEEDS", [42]):
                report = publication.complete(out, benchmark=session_bench.benchmark("v0").id)
            self.assertEqual(sorted(report["seeds"]["42"]["methods"]), sorted(publication.METHODS))
            # and the check refuses the same run once a transcript no longer matches the report
            path = out / "transcripts" / "seed42" / "llm.json"
            transcript = json.loads(path.read_text())
            transcript["turns"] += 1
            path.write_text(json.dumps(transcript))
            with patch.object(publication, "SEEDS", [42]), self.assertRaisesRegex(ValueError, "turns mismatch"):
                publication.complete(out, benchmark=session_bench.benchmark("v0").id)
