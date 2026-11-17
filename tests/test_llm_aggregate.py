"""The aggregation benchmark (tracks/llm/aggregate_bench.py): chains of merge rounds that restart
their outer momentum when the gate refuses a merge, and report their best round.

    python -m unittest discover -s tests -p 'test_llm_aggregate.py'
"""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tracks.llm import aggregate_bench as agg, session_bench as sessions, wsp_bench as wsp
from tracks.llm.anchor import load_anchor, weights_digest
from tracks.llm.merge import load_checkpoint, round_start

ANCHOR = load_anchor("v4")


class BestRoundTests(unittest.TestCase):
    def test_the_best_round_is_the_accepted_one_with_the_lowest_validation_bits(self):
        history = [{"round": None, "accepted": True, "validation_bits": 1.50},
                   {"round": 1, "accepted": True, "validation_bits": 1.40},
                   {"round": 2, "accepted": False, "validation_bits": 1.30},  # refused: never a model the chain had
                   {"round": 3, "accepted": True, "validation_bits": 1.40},  # a tie goes to the earlier
                   {"round": 4, "accepted": True, "validation_bits": 1.41}]
        self.assertEqual(agg.best_round(history)["round"], 1)

    def test_the_start_counts_when_no_round_improves_on_it(self):
        history = [{"round": None, "accepted": True, "validation_bits": 1.50},
                   {"round": 1, "accepted": True, "validation_bits": 1.51},
                   {"round": 2, "accepted": False, "validation_bits": None}]
        self.assertIsNone(agg.best_round(history)["round"])


class ScriptedChain:
    """session_bench's training, merging and scoring replaced by a script of each merge's outcome:
    (accepted, validation bits, whether the merged checkpoint carries momentum)."""

    def __init__(self, folder, script):
        self.folder, self.script = Path(folder), list(script)
        self.trained, self.merges = [], []

    def checkpoint(self, name, momentum):
        path = self.folder / name
        torch.save({"outer_momentum": {"w": torch.ones(1)} if momentum else None,
                    "merge": {"momentum_sha256": "recorded" if momentum else None}}, path)
        return path

    def train(self, output, seed, epochs, manifest, shard=None, start=None, base=None):
        self.trained.append((output.name, seed, epochs, Path(start).name))
        return {"checkpoint": output / "real.pt", "seconds": 1.0}

    def merge(self, output, workers, rule, start, manifest, anchor):
        accepted, bits, momentum = self.script.pop(0)
        self.merges.append((output.name, Path(start).name, [w.parent.name for w in workers], agg.carries_momentum(start)))
        record = {"decision": "accepted" if accepted else "rejected", "validation_bits": bits,
                  "reason": None if accepted else "worse than the incumbent"}
        return record, (self.checkpoint(f"{output.name}.pt", momentum) if accepted else None), 0.5

    def run(self, restart, rounds=3):
        start = self.checkpoint("start.pt", False)
        ctx = SimpleNamespace(anchor=ANCHOR)
        with mock.patch.object(sessions, "train", self.train), mock.patch.object(sessions, "merge", self.merge), \
                mock.patch.object(sessions, "outcome", lambda record, base: dict(record)), \
                mock.patch.object(sessions, "assess", lambda ctx, path: {"scored": Path(path).name}):
            return agg.chain(self.folder, start, 7, {"outer": agg.DILOCO}, restart, rounds, rounds * 10, None, ctx, 1.5)


class ChainTests(unittest.TestCase):
    def test_a_refused_momentum_merge_keeps_refusing_without_a_restart(self):
        """What the session benchmark's DiLoCo chain did on sessions-v1: the incumbent keeps the
        momentum that overshot, and every later round starts from it again."""
        with tempfile.TemporaryDirectory() as tmp:
            script = ScriptedChain(tmp, [(True, 1.40, True), (False, 1.45, True), (False, 1.46, True)])
            arm = script.run(restart=False)
        self.assertEqual([m[:2] for m in script.merges], [("round1", "start.pt"), ("round2", "round1.pt"), ("round3", "round1.pt")])
        self.assertEqual((arm["accepted_rounds"], arm["restarts"]), (1, 0))
        self.assertEqual(arm["last"], {"scored": "round1.pt"})

    def test_a_refused_momentum_merge_is_merged_again_without_the_momentum(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = ScriptedChain(tmp, [(True, 1.40, True), (False, 1.45, True), (True, 1.38, True), (True, 1.39, True)])
            arm = script.run(restart=True)
            restarted = torch.load(Path(tmp) / "round2-restart-start.pt", weights_only=True)
        self.assertEqual((restarted["outer_momentum"], restarted["merge"]["momentum_sha256"]), (None, None))
        # Round 2 is merged twice from the same workers; nothing is retrained for the restart.
        self.assertEqual(script.merges[1:3], [("round2", "round1.pt", ["r2-worker2", "r2-worker3", "r2-worker4", "r2-worker5"], True),
                                              ("round2-restart", "round2-restart-start.pt",
                                               ["r2-worker2", "r2-worker3", "r2-worker4", "r2-worker5"], False)])
        self.assertEqual(len(script.trained), 3 * len(sessions.WORKER_SHARDS))
        self.assertEqual({t[3] for t in script.trained if t[0].startswith("r3-")}, {"round2-restart.pt"}, "round 3 trains from the restarted merge")
        self.assertEqual([t[1:3] for t in script.trained[::4]], [(8, 10), (9, 10), (10, 10)], "round r trains with seed + r")
        self.assertEqual((arm["accepted_rounds"], arm["restarts"], arm["contributions"]), (3, 1, 12))
        self.assertEqual(arm["rounds"][1]["restarted_after"], {"validation_bits": 1.45, "reason": "worse than the incumbent"})
        self.assertEqual(arm["last"], {"scored": "round3.pt"})
        self.assertEqual(arm["best_round"], {"round": 2, "scored": "round2-restart.pt"}, "the best round isn't the last")

    def test_a_merge_refused_without_momentum_too_leaves_the_incumbent_with_its_momentum_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = ScriptedChain(tmp, [(True, 1.40, True), (False, 1.45, True), (False, 1.44, True), (False, 1.43, True)])
            arm = script.run(restart=True)
        # Round 3 starts from the incumbent without momentum; refused again, there's nothing to restart.
        self.assertEqual([m[:2] for m in script.merges], [("round1", "start.pt"), ("round2", "round1.pt"),
                                                          ("round2-restart", "round2-restart-start.pt"), ("round3", "round2-restart-start.pt")])
        self.assertEqual((arm["accepted_rounds"], arm["restarts"]), (1, 1))
        self.assertEqual(arm["last"], {"scored": "round2-restart-start.pt"})
        self.assertEqual(arm["best_round"]["round"], 1)

    def test_a_plain_chain_never_restarts(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = ScriptedChain(tmp, [(True, 1.40, False), (False, 1.45, False), (True, 1.39, False)])
            arm = script.run(restart=True)
        self.assertEqual((len(script.merges), arm["restarts"]), (3, 0))


class PublishedCohortTests(unittest.TestCase):
    """Short chains from v0's published seed-42 base: real training and merge.py's rounds."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.tmp.name) / "run"
        args = SimpleNamespace(benchmark="v0", output=cls.output, arms=list(agg.ARMS), seeds=[42], epochs=20, rounds=2, publish=False)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.report = agg.run(args)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_arm_runs_its_rounds_from_the_published_base(self):
        seed = self.report["seeds"]["42"]
        published = json.loads((sessions.FOLDER / "report.json").read_text())["seeds"]["42"]
        self.assertEqual(seed["base"]["weights_sha256"], published["base"]["weights_sha256"])
        self.assertEqual(list(seed["arms"]), list(agg.ARMS))
        for name, arm in seed["arms"].items():
            self.assertEqual([r["round"] for r in arm["rounds"] if r["decision"] == "accepted"],
                             list(range(2, 2 + arm["accepted_rounds"])), "the chains continue the base's chain")
            self.assertEqual((arm["contributions"], arm["bytes_sent"]), (8, 8 * 27_584 * 4))
            self.assertEqual(arm["rule"]["soup"], "greedy" if "soup" in name else "all")
            self.assertEqual(arm["rule"]["outer"], agg.DILOCO if "DiLoCo" in name else {"lr": 1.0, "momentum": 0.0})
            self.assertLessEqual(arm["best_round"]["validation"], arm["last"]["validation"], "the best round is no worse on validation")
            self.assertLessEqual(arm["best_round"]["validation"], seed["base"]["validation"])
        self.assertIsNone(seed["reference_problem"], "a shorter run isn't compared with the session benchmark's chains")
        self.assertEqual(set(self.report["summary"]["greedy soup"]["last"]["sessions_vs_average"]), {"better", "worse"})

    def test_clearing_the_momentum_keeps_the_weights_and_the_chain(self):
        merged = self.output / "seed42" / "arm1" / "round1" / "merged.pt"  # average + DiLoCo, its first round
        if not merged.exists():
            self.skipTest("the DiLoCo round was refused")
        self.assertTrue(agg.carries_momentum(merged))
        cleared = agg.without_momentum(merged, Path(self.tmp.name) / "cleared.pt")
        self.assertFalse(agg.carries_momentum(cleared))
        before, after = round_start(ANCHOR, merged), round_start(ANCHOR, cleared)  # round_start checks the recorded digests
        self.assertIsNone(after.momentum)
        self.assertEqual((after.weights_sha256, after.round, after.outer, after.lineage),
                         (before.weights_sha256, before.round, before.outer, before.lineage))
        self.assertEqual(weights_digest(load_checkpoint(cleared, ANCHOR)["model"]), before.weights_sha256)

    def test_publishing_is_only_for_the_benchmarks_settings(self):
        args = SimpleNamespace(benchmark="v0", output=Path(self.tmp.name) / "again", arms=list(agg.ARMS), seeds=[42], epochs=20,
                               rounds=2, publish=True)
        with self.assertRaisesRegex(ValueError, "--publish"):
            agg.run(args)


class PublishedReportTests(unittest.TestCase):
    def test_the_published_reports_verify(self):
        for name in ("v0", "v1"):
            self.assertEqual(agg.verify(name), ["42", "43", "44"])

    def test_on_v1_the_session_benchmarks_diloco_chain_was_refused_after_one_round_and_restarting_frees_it(self):
        report = json.loads((agg.FOLDERS["v1"] / "report.json").read_text())
        for seed in report["seeds"].values():
            stuck, restarting = seed["arms"]["average + DiLoCo"], seed["arms"]["average + DiLoCo, restarting"]
            self.assertEqual([r["decision"] for r in stuck["rounds"]], ["accepted"] + ["rejected"] * 9)
            self.assertGreater(restarting["accepted_rounds"], stuck["accepted_rounds"])
            self.assertGreaterEqual(restarting["restarts"], 1)

    def test_a_changed_report_is_refused(self):
        original = json.loads((agg.FOLDERS["v0"] / "report.json").read_text())

        def refused(change, message):
            report = copy.deepcopy(original)
            change(report)
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "report.json").write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, message):
                    agg.verify("v0", Path(tmp))

        arm = lambda r, name: r["seeds"]["42"]["arms"][name]
        refused(lambda r: arm(r, "greedy soup")["best_round"].update(round=1), "best round")
        refused(lambda r: arm(r, "average")["last"].update(weights_sha256="0" * 64), "isn't the session benchmark's")
        refused(lambda r: arm(r, "average")["rounds"][0].update(restarted_after={}), "restarts")
        refused(lambda r: r["seeds"]["42"]["base"].update(weights_sha256="0" * 64), "published base")
        refused(lambda r: r["summary"]["average"]["last"]["test"].update(mean=1.0), "summary")
        refused(lambda r: r.update(split_sha256="0" * 64), "another split")

    def test_the_wsp_protocol_is_still_the_pinned_one(self):
        """This benchmark adds a module; nothing the WSP protocol hashes may change with it."""
        self.assertEqual(wsp.protocol_sha256(), wsp.PROTOCOL_SHA256)


if __name__ == "__main__":
    unittest.main()
