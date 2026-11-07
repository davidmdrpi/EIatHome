"""The weight-space programming benchmark (tracks/llm/wsp_bench.py): the edit language and its
validator, exact replay, the budgeted oracle, the search controls, the specialist bank, and the
editor's loop with a scripted client (no API calls).

    python -m unittest discover -s tests -p 'test_llm_wsp.py'
"""

import contextlib
import copy
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tracks.llm import model as model_module, route_bench, session_bench as sessions, wsp_bench as wsp
from tracks.llm.anchor import load_anchor, weights_digest
from tracks.llm.merge import model_from
from tracks.llm.model import evaluate

ANCHOR = load_anchor("v4")


def combine(group, **weights):
    return {"op": "combine", "group": group, "weights": weights}


class Shared:
    """The published v0 cohort of seed 42 and its held-out sets, loaded once."""
    ctx = None
    cohort = None

    @classmethod
    def load(cls):
        if cls.ctx is None:
            cls.ctx = sessions.Context(ANCHOR, sessions.MANIFEST)
            cls.cohort = wsp.load_cohort("v0", 42)
        return cls.ctx, cls.cohort


class ProgramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx, cls.cohort = Shared.load()

    def refused(self, program, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            wsp.apply(self.cohort, program)

    def test_every_parameter_is_in_one_group(self):
        groups = {wsp.group_of(k) for k in self.cohort.base}
        self.assertEqual(groups, set(wsp.GROUPS))

    def test_the_references_replay_the_published_weights_bit_for_bit(self):
        """'average' is merge.py's plain average (the session benchmark's one-shot merge), and a
        worker alone is that worker."""
        report = json.loads((sessions.FOLDER / "report.json").read_text())
        average = wsp.apply(self.cohort, {"start": "average", "ops": []})
        self.assertEqual(weights_digest(average), report["seeds"]["42"]["one_shot"]["average (FedAvg)"]["weights_sha256"])
        for name, program in wsp.references(self.cohort).items():
            if name.endswith(" alone"):
                self.assertEqual(weights_digest(wsp.apply(self.cohort, program)), self.cohort.digests[name.split()[0]])
        self.assertEqual(weights_digest(wsp.apply(self.cohort, {"start": "base"})), self.cohort.digests["base"])

    def test_the_validator_refuses_what_breaks_a_rule(self):
        self.refused([], "A program is")
        self.refused({"start": "θ₀", "ops": []}, "A program is")
        self.refused({"start": "base", "ops": [], "note": "x"}, "A program is")
        self.refused({"start": "base", "ops": [{"op": "finetune"}]}, "op is")
        self.refused({"start": "base", "ops": [{"op": "reset", "group": "head", "keep": 1}]}, "takes exactly")
        self.refused({"start": "base", "ops": [{"op": "reset", "group": "blocks.0"}]}, "group is one of")
        self.refused({"start": "base", "ops": [combine("head", w2=2.5)]}, r"in \[-1.0, 2.0\]")
        self.refused({"start": "base", "ops": [combine("head", w6=1.0)]}, "maps workers")
        self.refused({"start": "base", "ops": [combine("head", w2=2.0, w3=2.0, w4=0.5)]}, "sum to at most 4")
        self.refused({"start": "base", "ops": [combine("head", w2=float("nan"))]}, "each weight")
        self.refused({"start": "base", "ops": [combine("head", w2=True)]}, "each weight")
        self.refused({"start": "base", "ops": [{"op": "trim", "group": "head", "keep": 0.01}]}, "keep is in")
        low_rank = {"op": "low_rank", "tensor": "head.weight", "source": "w5", "rank": 2, "scale": 1.0}
        self.refused({"start": "base", "ops": [{**low_rank, "tensor": "norm.weight"}]}, "names a weight matrix")
        self.refused({"start": "base", "ops": [{**low_rank, "source": "base"}]}, "source is one of")
        self.refused({"start": "base", "ops": [{**low_rank, "rank": 5}]}, "rank is an integer")
        self.refused({"start": "base", "ops": [{**low_rank, "rank": 2.0}]}, "rank is an integer")
        self.refused({"start": "base", "ops": [{**low_rank, "scale": -3}]}, "scale is in")
        self.refused({"start": "base", "ops": [{"op": "reset", "group": "head"}] * 33}, "at most 32")
        with mock.patch.dict(wsp.LIMITS, {"max_abs_delta": 1e-6}):
            self.refused({"start": "average", "ops": []}, "beyond the bound")

    def test_operations_do_what_they_say(self):
        base, w5 = self.cohort.base, self.cohort.workers["w5"]
        # combine and reset touch only their group
        state = wsp.apply(self.cohort, {"start": "base", "ops": [combine("head", w5=1.0)]})
        self.assertTrue(torch.equal(state["head.weight"], w5["head.weight"]))
        self.assertTrue(torch.equal(state["token.weight"], base["token.weight"]))
        state = wsp.apply(self.cohort, {"start": "average", "ops": [{"op": "reset", "group": "embed"}]})
        self.assertTrue(torch.equal(state["token.weight"], base["token.weight"]))
        # trim keeps the largest fraction of each tensor's delta, by magnitude
        state = wsp.apply(self.cohort, {"start": "base", "ops": [combine("head", w5=1.0), {"op": "trim", "group": "head", "keep": 0.05}]})
        delta = (state["head.weight"].double() - base["head.weight"].double()).flatten()
        full = (w5["head.weight"].double() - base["head.weight"].double()).flatten()
        kept = delta != 0
        self.assertEqual(int(kept.sum()), math.ceil(0.05 * full.numel()))
        self.assertGreaterEqual(full[kept].abs().min().item(), full[~kept].abs().max().item())
        trimmed_all = wsp.apply(self.cohort, {"start": "average", "ops": [{"op": "trim", "group": g, "keep": 1.0} for g in wsp.GROUPS]})
        self.assertEqual(weights_digest(trimmed_all), weights_digest(wsp.apply(self.cohort, {"start": "average", "ops": []})))
        # low_rank adds a rank-r piece of one worker's delta to one matrix
        op = {"op": "low_rank", "tensor": "blocks.1.qkv.weight", "source": "w4", "rank": 2, "scale": 1.0}
        state = wsp.apply(self.cohort, {"start": "base", "ops": [op]})
        added = state["blocks.1.qkv.weight"].double() - base["blocks.1.qkv.weight"].double()
        self.assertEqual(int(torch.linalg.matrix_rank(added, rtol=1e-5)), 2)
        changed = [k for k in base if not torch.equal(state[k], base[k])]
        self.assertEqual(changed, ["blocks.1.qkv.weight"])

    def test_the_same_program_makes_the_same_weights(self):
        program = {"start": "average", "ops": [combine("block1.feed-forward", w4=0.7, w5=0.6),
                                               {"op": "trim", "group": "embed", "keep": 0.3},
                                               {"op": "low_rank", "tensor": "head.weight", "source": "w5", "rank": 3, "scale": 0.5}]}
        self.assertEqual(weights_digest(wsp.apply(self.cohort, program)), weights_digest(wsp.apply(self.cohort, json.loads(json.dumps(program)))))


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx, cls.cohort = Shared.load()
        cls.refs = wsp.Oracle(cls.ctx, cls.cohort).references

    def test_the_references_are_free_and_refused_programs_are_not_counted(self):
        oracle = wsp.Oracle(self.ctx, self.cohort, budget=2, refs=self.refs)
        self.assertEqual(oracle.left, 2)
        with self.assertRaisesRegex(ValueError, "op is"):
            oracle.evaluate({"start": "base", "ops": [{"op": "nope"}]})
        self.assertEqual((oracle.left, len(oracle.refused)), (2, 1))
        oracle.evaluate({"start": "average", "ops": [combine("head", w5=1.0)]})
        oracle.evaluate({"start": "average", "ops": [combine("head", w4=1.0)]})
        with self.assertRaises(wsp.BudgetSpent):
            oracle.evaluate({"start": "average", "ops": []})
        self.assertEqual(len(oracle.queries), 2)

    def test_too_many_refusals_end_the_session(self):
        oracle = wsp.Oracle(self.ctx, self.cohort, budget=1, refs=self.refs)
        with self.assertRaises(ValueError):
            oracle.evaluate({})
        with self.assertRaises(wsp.BudgetSpent):
            oracle.evaluate({})

    def test_the_pick_is_the_best_on_validation_among_references_and_queries(self):
        oracle = wsp.Oracle(self.ctx, self.cohort, budget=3, refs=self.refs)
        best_reference = min(self.refs, key=lambda n: self.refs[n]["validation"])
        self.assertEqual(oracle.pick()[0], best_reference)
        worse = oracle.evaluate({"start": "base", "ops": [combine(g, w3=1.0) for g in wsp.GROUPS]})
        self.assertGreater(worse["validation"], self.refs[best_reference]["validation"])
        self.assertEqual(oracle.pick()[0], best_reference)
        # an exact copy of the best reference doesn't displace it (ties go to the earlier)
        oracle.evaluate(self.refs[best_reference]["program"])
        self.assertEqual(oracle.pick()[0], best_reference)

    def test_scores_are_the_session_benchmarks(self):
        report = json.loads((sessions.FOLDER / "report.json").read_text())["seeds"]["42"]
        best = report["best_worker"]
        name = f"w{[w for w in report['workers'] if w['weights_sha256'] == best['weights_sha256']][0]['shard']} alone"
        scores = wsp.score_state(self.ctx, wsp.apply(self.cohort, self.refs[name]["program"]), test=True)
        for key in ("validation", "test", "test_worst_session", "control", "test_order_gain_bits"):
            # recorded in the session benchmark's environment: equal up to another CPU's rounding
            self.assertAlmostEqual(scores[key], best[key], delta=wsp.SCORE_TOLERANCE)
        self.assertEqual(len(scores["validation_sessions"]), 7)

    def test_the_search_controls_spend_the_budget_and_replay(self):
        for search in (wsp.random_search, wsp.coordinate_search):
            runs = []
            for _ in range(2):
                oracle = wsp.Oracle(self.ctx, self.cohort, budget=6, refs=self.refs)
                search(oracle)
                runs.append(oracle)
            self.assertLessEqual(len(runs[0].queries), 6)
            self.assertEqual([q["weights_sha256"] for q in runs[0].queries], [q["weights_sha256"] for q in runs[1].queries])
            self.assertFalse(runs[0].refused)

    def test_the_specialist_bank_of_one_model_is_that_model(self):
        """To float32 rounding: evaluate() takes its log-softmax in float32, the bank in float64."""
        model = model_from(ANCHOR, self.cohort.workers["w5"])
        with torch.no_grad():
            alone = evaluate(model, self.ctx.test)["bits_per_token"]
            self.assertAlmostEqual(wsp.bank_bits([model], self.ctx.test), alone, places=6)
            self.assertAlmostEqual(wsp.bank_bits([model, model], self.ctx.test), wsp.bank_bits([model], self.ctx.test), places=12)


def message(*blocks, stop="tool_use", model="claude-opus-5-5", usage=(1000, 0, 0, 200)):
    i, w, r, o = usage
    return SimpleNamespace(content=list(blocks), stop_reason=stop, model=model,
                           usage=SimpleNamespace(input_tokens=i, cache_creation_input_tokens=w, cache_read_input_tokens=r, output_tokens=o))


def tool(name, args, id_):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=args)


class EditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx, cls.cohort = Shared.load()
        cls.refs = wsp.Oracle(cls.ctx, cls.cohort).references

    def test_the_loop_evaluates_programs_through_the_oracle_and_records_the_session(self):
        responses = [
            message(SimpleNamespace(type="text", text="Start with w5's head."),
                    tool("evaluate_program", {"program": {"start": "average", "ops": [combine("head", w5=1.0)]}}, "a")),
            message(tool("evaluate_program", {"program": {"start": "average", "ops": [combine("head", w5=3.0)]}}, "b"),
                    usage=(10, 2000, 500, 100)),
            message(tool("finish", {"summary": "Tried the head."}, "c")),
        ]
        script, requests = iter(responses), []

        def send(request):
            requests.append(request)
            return next(script)

        oracle = wsp.Oracle(self.ctx, self.cohort, refs=self.refs)
        session = wsp.run_editor(oracle, "context", send)
        self.assertEqual((session["stopped"], session["summary"], session["turns"]), ("finished", "Tried the head.", 3))
        self.assertEqual((len(oracle.queries), len(oracle.refused)), (1, 1))
        first = requests[0]
        self.assertEqual(first["model"], "claude-opus-5-5")
        self.assertEqual((first["fallbacks"], first["betas"]), ("default", ["server-side-fallback-2026-07-01"]))
        self.assertEqual(first["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("tool_choice", first)  # forced tool choice is refused on this model
        # the second request carries the first turn's content unchanged, then its tool result
        self.assertIs(requests[1]["messages"][1]["content"], responses[0].content)
        result = requests[1]["messages"][2]["content"][0]
        self.assertEqual(result["tool_use_id"], "a")
        self.assertEqual(json.loads(result["content"])["queries_left"], wsp.BUDGET - 1)
        refused = requests[2]["messages"][4]["content"][0]
        self.assertTrue(refused["is_error"])
        self.assertIn("Refused, not counted", refused["content"])
        tokens = session["cost"]["tokens"]
        self.assertEqual(tokens, {"input": 2010, "cache_write": 2000, "cache_read": 500, "output": 500})
        self.assertAlmostEqual(session["cost"]["dollars"], (2010 * 4 + 2000 * 5 + 500 * 0.2 + 500 * 20) / 1e6)
        self.assertEqual(session["transcript"][1]["content"][0], {"type": "text", "text": "Start with w5's head."})

    def test_the_editor_reads_tensors_without_spending_the_budget(self):
        read = lambda **args: json.loads(wsp.read_tensor(self.cohort, args))
        head = read(tensor="head.weight", source="w5", delta=True, rows=[0, 2])
        expected = (self.cohort.workers["w5"]["head.weight"].double() - self.cohort.base["head.weight"].double())[:2]
        self.assertEqual(head["shape"], [2, 32])
        self.assertTrue(torch.allclose(torch.tensor(head["values"], dtype=torch.float64), expected, atol=1e-5))
        self.assertFalse(read(tensor="norm.bias", source="base", delta=True)["delta"])
        for args, pattern in (({"tensor": "nope", "source": "w2"}, "tensor names"), ({"tensor": "head.weight", "source": "w9"}, "source is"),
                              ({"tensor": "head.weight", "source": "w2", "rows": [5, 2]}, "rows is"),
                              ({"tensor": "head.weight", "source": "w2", "rows": [0, 34]}, "rows is")):
            with self.assertRaisesRegex(ValueError, pattern):
                wsp.read_tensor(self.cohort, args)
        oracle = wsp.Oracle(self.ctx, self.cohort, refs=self.refs)
        script = iter([message(tool("read_tensor", {"tensor": "head.weight", "source": "w3"}, "a"),
                               tool("read_tensor", {"tensor": "x", "source": "w3"}, "b")),
                       message(tool("finish", {"summary": ""}, "c"))])
        session = wsp.run_editor(oracle, "context", lambda request: next(script))
        self.assertEqual((len(oracle.queries), oracle.left, session["stopped"]), (0, wsp.BUDGET, "finished"))
        results = session["transcript"][2]["content"]
        self.assertEqual(json.loads(results[0]["content"])["shape"], [33, 32])
        self.assertTrue(results[1]["is_error"])

    def test_a_truncated_or_refused_turn_runs_no_tool(self):
        for stop in ("max_tokens", "refusal"):
            oracle = wsp.Oracle(self.ctx, self.cohort, refs=self.refs)
            session = wsp.run_editor(oracle, "context", lambda request: message(
                tool("evaluate_program", {"program": {"start": "average", "ops": []}}, "a"), stop=stop))
            self.assertEqual((session["stopped"], len(oracle.queries)), (stop, 0))

    def test_the_session_ends_when_the_budget_is_spent(self):
        oracle = wsp.Oracle(self.ctx, self.cohort, budget=2, refs=self.refs)
        program = {"start": "average", "ops": []}
        session = wsp.run_editor(oracle, "context", lambda request: message(
            tool("evaluate_program", {"program": program}, "a"), tool("evaluate_program", {"program": program}, "b")))
        self.assertEqual((session["stopped"], session["turns"], len(oracle.queries)), ("budget spent", 1, 2))

    def test_finish_ends_the_session_whatever_its_input(self):
        oracle = wsp.Oracle(self.ctx, self.cohort, refs=self.refs)
        session = wsp.run_editor(oracle, "context", lambda request: message(tool("finish", None, "a")))
        self.assertEqual((session["stopped"], session["turns"], session["summary"]), ("finished", 1, ""))

    def test_an_unpriced_model_is_reported_not_guessed(self):
        oracle = wsp.Oracle(self.ctx, self.cohort, refs=self.refs)
        session = wsp.run_editor(oracle, "context", lambda request: message(
            tool("finish", {"summary": ""}, "a"), model="another-model"))
        self.assertIsNone(session["cost"]["dollars"])
        self.assertEqual(session["served_by"], ["another-model"])

    def test_the_editor_needs_the_experimenters_own_key(self):
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(SystemExit, "ANTHROPIC_API_KEY"):
            wsp.anthropic_send()
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k", "ANTHROPIC_AUTH_TOKEN": "t"}, clear=True), \
                self.assertRaisesRegex(SystemExit, "ANTHROPIC_AUTH_TOKEN"):
            wsp.anthropic_send()

    def test_the_anonymized_context_names_no_target_date_band_or_family(self):
        oracle = wsp.Oracle(self.ctx, self.cohort, refs=self.refs)
        named = wsp.editor_context(self.ctx, self.cohort, sessions.benchmark("v0"), oracle, anonymized=False)
        hidden = wsp.editor_context(self.ctx, self.cohort, sessions.benchmark("v0"), oracle, anonymized=True)
        for word in ("HIP ", "2016-", "2021-", "Green Bank", "stars", "blind-window", "\"band\"", "centre_mhz", "telescope"):
            self.assertNotIn(word, hidden)
        for word in ("HIP ", "2016-", "Green Bank", "stars", "\"band\""):
            self.assertIn(word, named)
        self.assertIn("w5 alone", hidden)  # the scores and statistics are the same

    def test_the_protocol_is_pinned(self):
        """Changing the edit language, the budget, the selection rule, the controls, the scoring
        code or the editor changes this hash; see WSP_RESULTS.md."""
        self.assertEqual(wsp.protocol_sha256(), wsp.PROTOCOL_SHA256)

    def test_the_fingerprint_covers_the_scoring_code_it_calls(self):
        """Replacing the evaluator, the windowing under it, or the bank's scoring and mixing helpers
        changes the fingerprint, wherever the replaced object is bound."""
        pinned = wsp.protocol_sha256()
        fake = lambda *args, **kwargs: None
        for target, name in ((wsp, "evaluate"), (model_module, "evaluate"), (model_module, "windows"), (wsp, "target_bits"),
                             (wsp, "mixture"), (wsp, "order_gain"), (wsp, "shuffled_gain"), (wsp, "score_state"),
                             (route_bench, "LOG2"), (sessions, "Context"), (wsp, "ORDER_SHUFFLES")):
            with self.subTest(name=f"{target.__name__}.{name}"), mock.patch.object(target, name, fake):
                self.assertNotEqual(wsp.protocol_sha256(), pinned)
        self.assertEqual(wsp.protocol_sha256(), pinned)


class RunTests(unittest.TestCase):
    """One small run (random search and the bank, seed 43), then copies of it tampered with."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.tmp.name) / "run"
        args = SimpleNamespace(benchmark="v0", output=cls.output, methods=["random", "specialist"], seeds=[43], publish=False)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.report = wsp.run(args)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def tampered(self, change, pattern):
        """verify refuses a copy of the run after `change(report, artifact)` edits it."""
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "run"
            shutil.copytree(self.output, folder)
            report = json.loads((folder / "report.json").read_text())
            path = folder / "edits" / "seed43" / "random.json"
            artifact = json.loads(path.read_text())
            change(report, artifact)
            (folder / "report.json").write_text(json.dumps(report))
            path.write_text(json.dumps(artifact))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, pattern):
                wsp.verify("v0", folder)

    def test_the_run_verifies(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(wsp.verify("v0", self.output), 2)
        seed = self.report["seeds"]["43"]
        self.assertEqual(seed["methods"]["random"]["queries"], wsp.BUDGET)
        self.assertIn("best worker (by validation)", self.report["summary"])
        artifact = json.loads((self.output / "edits" / "seed43" / "random.json").read_text())
        self.assertEqual((artifact["inputs"], len(artifact["queries"]), artifact["budget"]), (seed["inputs"], wsp.BUDGET, wsp.BUDGET))
        self.assertEqual(set(artifact["references"]), set(seed["references"]))

    def test_a_changed_program_is_refused(self):
        def change(report, artifact):
            artifact["queries"][0]["program"]["ops"][0]["weights"]["w2"] += 0.01
        self.tampered(change, "query 1 doesn't replay")

    def test_a_changed_score_is_refused(self):
        def change(report, artifact):
            artifact["queries"][0]["validation"] = -123.0
        self.tampered(change, "query 1: validation rescores")

        def reference(report, artifact):
            report["seeds"]["43"]["references"]["w4 alone"]["validation"] += 0.01
        self.tampered(reference, "w4 alone: validation rescores")

    def test_a_history_over_the_budget_or_out_of_order_is_refused(self):
        def extra(report, artifact):
            artifact["queries"].append({**copy.deepcopy(artifact["queries"][-1]), "query": wsp.BUDGET + 1})
        self.tampered(extra, "over the budget")

        def renumbered(report, artifact):
            artifact["queries"][0]["query"], artifact["queries"][1]["query"] = 2, 1
        self.tampered(renumbered, "numbered")

    def test_counts_the_report_gives_must_be_the_artifacts(self):
        def change(report, artifact):
            report["seeds"]["43"]["methods"]["random"]["queries"] = 31
        self.tampered(change, "report's queries")

    def test_a_valid_program_cant_be_recorded_as_refused(self):
        def change(report, artifact):
            artifact["refused"].append({"program": {"start": "average", "ops": []}, "reason": "made up"})
        self.tampered(change, "recorded as refused is valid")

    def test_a_pick_the_rule_doesnt_make_is_refused(self):
        def change(report, artifact):
            program = report["seeds"]["43"]["references"]["w2 alone"]["program"]
            artifact["pick"], artifact["program"] = "w2 alone", program
            result = report["seeds"]["43"]["methods"]["random"]
            result["pick"], result["program"] = "w2 alone", program
        self.tampered(change, "the rule picks w5 alone")

    def test_a_control_must_ask_for_what_its_procedure_asks_for(self):
        artifact = json.loads((self.output / "edits" / "seed43" / "random.json").read_text())
        cohort = wsp.load_cohort("v0", 43)
        refs = artifact["references"]
        queries = copy.deepcopy(artifact["queries"])
        queries[5]["program"]["ops"][2]["weights"]["w3"] = 0.5
        with self.assertRaisesRegex(ValueError, "query 6: the method's procedure asks for a program other"):
            wsp.random_search(wsp.Replay(cohort, refs, queries))
        replay = wsp.Replay(cohort, refs, artifact["queries"])
        wsp.random_search(replay)
        self.assertEqual(len(replay.queries), wsp.BUDGET)
        replay = wsp.Replay(cohort, refs, artifact["queries"][:10])
        with self.assertRaisesRegex(ValueError, "query 11"):
            wsp.random_search(replay)

    def test_a_run_made_with_another_protocol_is_refused(self):
        with mock.patch.object(wsp, "evaluate", lambda *args, **kwargs: None), contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(ValueError, "made with protocol"):
            wsp.verify("v0", self.output)

        def change(report, artifact):
            artifact["protocol_sha256"] = "0" * 64
        self.tampered(change, "protocol_sha256")

    def test_a_fresh_cohort_needs_the_pinned_protocol(self):
        args = SimpleNamespace(benchmark="v1", output=Path("unused"), methods=["random"], seeds=[42], publish=False)
        with mock.patch.object(wsp, "PROTOCOL_SHA256", "0" * 64), self.assertRaisesRegex(SystemExit, "scored once"):
            wsp.run(args)
        with mock.patch.object(wsp, "evaluate", lambda *args, **kwargs: None), self.assertRaisesRegex(SystemExit, "scored once"):
            wsp.run(args)


if __name__ == "__main__":
    unittest.main()
