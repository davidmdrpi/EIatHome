"""The volunteer checkpoint pilot (tracks/llm/pilot.py): delta artifacts and their identities,
the coordinator's checks, influence caps, merging, rollback and replay. Simulated volunteers
train real shards of the session benchmark (about a second each).

    python -m unittest discover -s tests -p 'test_llm_pilot.py'
"""

import base64
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

from tracks.llm import pilot as P
from tracks.llm.anchor import difference
from tracks.llm.merge import load_checkpoint


class Scenario:
    """One pilot in a temporary folder, with five simulated volunteers, run once."""
    built = None

    @classmethod
    def get(cls):
        if cls.built is None:
            cls.built = cls()
        return cls.built

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root = root
        self.pilot = P.Pilot(folder=root / "store")
        self.ledger = P.new_ledger()
        self.outcomes = {}
        self.bodies = {}
        self.submit("alice", 2, 11, 101)
        self.submit("bob", 3, 12, 102)
        self.submit("carol", 2, 13, 103)
        self.submit("alice", 2, 14, 104)  # alice again on shard 2
        self.submit("dave", 3, 12, 105)  # bob's shard and seed
        self.mislabelled(106)  # mallory: a shard-2 run sent as shard 4
        self.before_merge = copy.deepcopy(self.ledger)
        self.round = P.merge(self.pilot, self.ledger, "maintainer")
        P.save_ledger(self.ledger, self.pilot.folder)

    def issue(self, login, number, body):
        return {"number": number, "user": {"login": login}, "created_at": "2026-11-07T08:00:00Z", "body": body}

    def ingest(self, login, number, body):
        status, comment, text = P.ingest(self.pilot, self.issue(login, number, body), self.ledger)
        if status == "accepted":
            P.store(self.pilot.folder, self.ledger, self.ledger["events"][-1]["id"], text)
        self.outcomes[number] = (status, comment)
        return status, comment

    def submit(self, login, shard, seed, number):
        out = P.contribute(self.pilot, self.root / f"{login}-{number}", shard=shard, seed=seed, ledger=self.ledger)
        self.bodies[number] = out["issue"].read_text()
        return self.ingest(login, number, self.bodies[number])

    def mislabelled(self, number):
        head = P.state_of(self.ledger).head
        start = P.head_checkpoint(self.pilot, head)
        with tempfile.TemporaryDirectory() as tmp:
            run = P.train_worker(self.pilot, P.saved(start, tmp, "head"), 2, 15, tmp)
            worker = load_checkpoint(run["checkpoint"], self.pilot.anchor)
        tau = difference(worker["model"], start["model"])
        training = {"seed": 15, **P.TRAINING, "best_epoch": run["best_epoch"], "epochs_run": run["epochs_run"], "seconds": 1.0,
                    "environment": P.environment()}
        text, artifact_id = P.encode_artifact(self.pilot, head, 4, training, tau)
        artifact, _ = P.decode_artifact(self.pilot.anchor, text)
        _, body = P.issue_text(P.card_of(artifact_id, text, artifact, 1.7), text)
        self.bodies[number] = body
        return self.ingest("mallory", number, body)


def tampered_copy(scenario, change):
    """A copy of the scenario's store after `change(folder, ledger)`; returns (pilot, ledger, tmp)."""
    tmp = tempfile.TemporaryDirectory()
    folder = Path(tmp.name) / "store"
    shutil.copytree(scenario.pilot.folder, folder)
    pilot = copy.copy(scenario.pilot)
    pilot.folder = folder
    ledger = json.loads((folder / "ledger.json").read_text())
    change(folder, ledger)
    return pilot, ledger, tmp


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Scenario.get()
        cls.text = cls.s.bodies[101].split("```eiathome-delta\n", 1)[1].split("\n```", 1)[0]

    def test_an_artifact_carries_every_identity_and_fits_in_an_issue(self):
        artifact, tau = P.decode_artifact(self.s.pilot.anchor, self.text)
        identities = self.s.pilot.identities()
        for key in ("pilot", "anchor", "tokenizer", "dataset"):
            self.assertEqual(artifact[key], identities[key])
        self.assertEqual(artifact["round"], {"number": 2, "start_sha256": P.START_SHA256})
        self.assertEqual(artifact["shard"], self.s.pilot.shard(2))
        self.assertEqual(artifact["training"]["seed"], 11)
        self.assertLess(len(self.s.bodies[101]), P.ISSUE_LIMIT)
        self.assertEqual(sum(v.numel() for v in tau.values()), 27_584)

    def test_quantizing_costs_almost_nothing_on_validation(self):
        out = P.contribute(self.s.pilot, self.s.root / "quantize", shard=5, seed=3, ledger=P.new_ledger())
        start = P.head_checkpoint(self.s.pilot, P.state_of(P.new_ledger()).head)
        with tempfile.TemporaryDirectory() as tmp:
            run = P.train_worker(self.s.pilot, P.saved(start, tmp, "head"), 5, 3, tmp)
            full = self.s.pilot.bits(load_checkpoint(run["checkpoint"], self.s.pilot.anchor)["model"])
        self.assertLess(abs(out["card"]["validation_bits"] - full), 1e-3)

    def test_an_artifact_that_doesnt_rebuild_its_digest_or_is_malformed_is_refused(self):
        artifact = json.loads(self.text)
        refused = lambda change, pattern: self.assertRaisesRegex(ValueError, pattern, P.decode_artifact, self.s.pilot.anchor,
                                                                 json.dumps(change(copy.deepcopy(artifact))))
        values = bytearray(base64.b64decode(artifact["values"]))

        def flip(a):
            values[7] = (values[7] + 1) % 127
            a["values"] = base64.b64encode(bytes(values)).decode()
            return a
        refused(flip, "don't rebuild")
        refused(lambda a: {**a, "version": 1}, "version-2")
        refused(lambda a: {**a, "tensors": a["tensors"][:-1]}, "tensors")
        refused(lambda a: {**a, "scales": [*a["scales"][:-1], 0.1]}, "float32")
        refused(lambda a: {**a, "scales": [*a["scales"][:-1], -1.0]}, "non-negative")
        refused(lambda a: {**a, "values": a["values"][:-8]}, "one int8")
        refused(lambda a: {**a, "values": "not base64!"}, "base64")
        with self.assertRaisesRegex(ValueError, "isn't JSON"):
            P.decode_artifact(self.s.pilot.anchor, "{")

    def test_each_identity_must_be_the_pilots_next_rounds(self):
        artifact, _ = P.decode_artifact(self.s.pilot.anchor, self.text)
        head = P.state_of(P.new_ledger()).head
        self.assertEqual(P.identity_problems(self.s.pilot, artifact, head), [])
        cases = [
            (lambda a: a.update(pilot="another"), "pilot"),
            (lambda a: a["anchor"].update(theta0_sha256="0" * 64), "anchor"),
            (lambda a: a["tokenizer"].update(sha256="0" * 64), "tokenizer"),
            (lambda a: a["dataset"].update(split_sha256="0" * 64), "dataset"),
            (lambda a: a["round"].update(number=3), "next round"),
            (lambda a: a["round"].update(start_sha256="0" * 64), "next round"),
            (lambda a: a["shard"].update(index=1), "open shards"),
            (lambda a: a["shard"].update(slices_sha256="0" * 64), "slices or training targets"),
            (lambda a: a["shard"].update(training_targets=1), "slices or training targets"),
            (lambda a: a["training"].update(epochs=10), "settings"),
            (lambda a: a["training"].update(seed="11"), "settings"),
        ]
        for change, pattern in cases:
            changed = copy.deepcopy(artifact)
            change(changed)
            with self.subTest(pattern=pattern):
                self.assertRegex(" ".join(P.identity_problems(self.s.pilot, changed, head)), pattern)

    def test_an_issue_needs_one_card_and_one_artifact(self):
        body = self.s.bodies[101]
        card, text = P.parse_issue(body)
        self.assertEqual((card["shard"], card["seed"], text), (2, 11, self.text))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            P.parse_issue(body.split("```eiathome-delta")[0])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            P.parse_issue(body + "\n```eiathome-delta\n{}\n```\n")
        with self.assertRaisesRegex(ValueError, "card isn't"):
            P.parse_issue(body.replace('"format": "eiathome-checkpoint-card"', '"format": "other"'))


class IngestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Scenario.get()
        cls.events = {e["issue"]: e for e in cls.s.before_merge["events"] if e["event"] == "submission"}

    def test_honest_contributions_are_validated_replicated_and_receipted(self):
        for number in (101, 102, 103):
            status, comment = self.s.outcomes[number]
            event = self.events[number]
            self.assertEqual((status, event["status"]), ("accepted", "validated"))
            self.assertTrue(event["replication"]["replicated"])
            self.assertGreater(event["replication"]["cosine"], 0.99)
            self.assertIn(f"submissions/{event['id']}.json", comment)
            self.assertTrue((self.s.pilot.folder / event["artifact"]).exists())
        self.assertLess(self.events[101]["evaluation"]["validation_bits"], self.events[101]["evaluation"]["start_validation_bits"])

    def test_each_volunteer_counts_once_per_shard_and_the_same_run_once(self):
        self.assertEqual(self.s.outcomes[104][0], "duplicate")
        self.assertIn("once per shard per round", self.s.outcomes[104][1])
        self.assertEqual(self.s.outcomes[105][0], "duplicate")
        self.assertIn("seed 12", self.s.outcomes[105][1])
        self.assertNotIn(104, self.events)
        self.assertNotIn(105, self.events)

    def test_a_run_sent_as_another_shard_is_recorded_but_not_replicated_so_never_merged(self):
        event = self.events[106]
        self.assertEqual(event["status"], "validated")
        self.assertFalse(event["replication"]["replicated"])
        self.assertLess(event["replication"]["cosine"], P.REPLICATION["cosine"])
        self.assertIn("not replicated", self.s.outcomes[106][1])
        self.assertNotIn(event["id"], self.s.round["submissions"])

    def test_an_issue_recorded_before_is_announced_again_not_recorded_twice(self):
        ledger = copy.deepcopy(self.s.before_merge)
        status, comment, text = P.ingest(self.s.pilot, self.s.issue("alice", 101, self.s.bodies[101]), ledger)
        self.assertEqual((status, text, len(ledger["events"])), ("recorded", None, len(self.s.before_merge["events"])))
        self.assertIn("receipt", comment)

    def test_a_tampered_or_mismatched_contribution_is_rejected_without_echoing_it(self):
        body = self.s.bodies[102]
        ledger = copy.deepcopy(P.new_ledger())
        bad = [
            body.replace('"values":"', '"values":"A', 1),
            body.replace('"seed": 12', '"seed": 13'),
            body.replace('"shard": 3', '"shard": 4'),
            "no blocks here <script>",
        ]
        for i, text in enumerate(bad):
            status, comment, _ = P.ingest(self.s.pilot, self.s.issue("eve", 200 + i, text), ledger)
            with self.subTest(i=i):
                self.assertEqual(status, "rejected")
                self.assertNotIn("<script>", comment)
        self.assertEqual(ledger["events"], [])

    def test_a_contribution_worse_than_the_current_model_fails_the_gate(self):
        artifact, tau = P.decode_artifact(self.s.pilot.anchor, P.parse_issue(self.s.bodies[102])[1])
        head = P.state_of(P.new_ledger()).head
        # A uniform shift is nearly invisible (LayerNorm and the softmax drop it): random noise isn't.
        rng = np.random.default_rng(0)
        worse = {k: torch.from_numpy(rng.normal(0, 0.5, tuple(v.shape))) for k, v in tau.items()}
        training = artifact["training"]
        text, artifact_id = P.encode_artifact(self.s.pilot, head, 3, {**training, "seed": 99}, worse)
        _, body = P.issue_text(P.card_of(artifact_id, text, P.decode_artifact(self.s.pilot.anchor, text)[0], 1.0), text)
        status, comment, _ = P.ingest(self.s.pilot, self.s.issue("eve", 210, body), copy.deepcopy(P.new_ledger()))
        self.assertEqual(status, "rejected")
        self.assertIn("worse than the incumbent", comment)

    def test_the_command_line_ingest_reports_a_status_and_never_crashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            event, comment = Path(tmp) / "event.json", Path(tmp) / "comment.md"
            event.write_text("not json")
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                P.main(["ingest", str(event), str(comment)])
            self.assertEqual(json.loads(out.getvalue().strip().splitlines()[-1]), {"status": "error"})
            self.assertIn("retried", comment.read_text())


class MergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Scenario.get()

    def test_a_merge_needs_two_volunteers(self):
        ledger = copy.deepcopy(P.new_ledger())
        ledger["events"] = [e for e in self.s.before_merge["events"] if e.get("volunteer") == "alice"]
        with self.assertRaisesRegex(ValueError, "1 volunteer"):
            P.merge(self.s.pilot, ledger, "maintainer")

    def test_a_shard_weighs_its_share_however_many_contribute_to_it(self):
        record = self.s.round["record"]
        self.assertEqual(self.s.round["decision"], "accepted")
        weights = {}
        for entry in record["contributors"]:
            weights.setdefault(entry["shard"][0], []).append(entry["weight"])
        self.assertEqual({k: len(v) for k, v in weights.items()}, {2: 2, 3: 1})
        self.assertAlmostEqual(sum(weights[2]), sum(weights[3]))  # both shards have 225 training targets
        self.assertAlmostEqual(weights[2][0], weights[2][1])
        self.assertEqual(self.s.round["volunteers"], ["alice", "bob", "carol"])
        self.assertLessEqual(record["merged"]["validation_bits"], record["start"]["validation_bits"] + P.RULE["tolerance_bits"])

    def test_the_merge_becomes_the_current_model_and_old_contributions_go_stale(self):
        state = P.state_of(self.s.ledger)
        self.assertEqual((state.head.round, state.head.weights_sha256), (2, self.s.round["weights_sha256"]))
        self.assertTrue((self.s.pilot.folder / self.s.round["checkpoint"]).exists())
        self.assertEqual(state.eligible(), [])
        status, comment, _ = P.ingest(self.s.pilot, self.s.issue("frank", 300, self.s.bodies[102]), copy.deepcopy(self.s.ledger))
        self.assertEqual(status, "rejected")
        self.assertIn("round 3", comment)
        self.assertEqual(P.head_checkpoint(self.s.pilot, state.head)["merge"]["round"], 2)

    def test_rollback_and_exclusion_keep_history_and_change_what_merges_next(self):
        ledger = copy.deepcopy(self.s.ledger)
        P.rollback(ledger, 1, "a test", "maintainer")
        state = P.state_of(ledger)
        self.assertEqual(state.head.round, 1)
        self.assertEqual(sorted(s["issue"] for s in state.eligible()), [101, 102, 103], "merged into a rolled-back round: eligible again")
        alice = next(s for s in state.eligible() if s["issue"] == 101)
        P.exclude(ledger, alice["id"][:10], "a test", "maintainer")
        self.assertEqual(sorted(s["issue"] for s in P.state_of(ledger).eligible()), [102, 103])
        with self.assertRaisesRegex(ValueError, "already excluded"):
            P.exclude(ledger, alice["id"], "again", "maintainer")
        with self.assertRaisesRegex(ValueError, "names 0"):
            P.exclude(ledger, "0" * 12, "unknown", "maintainer")
        with self.assertRaisesRegex(ValueError, "isn't an earlier round"):
            P.rollback(ledger, 1, "already there", "maintainer")
        self.assertEqual(len(ledger["events"]), len(self.s.ledger["events"]) + 2)

        # An exclusion after the merge leaves the live round alone, and says so.
        live = copy.deepcopy(self.s.ledger)
        _, note = P.exclude(live, self.s.round["merged"][0], "found later", "maintainer")
        self.assertIn("until that round is rolled back", note)

    def test_events_that_dont_follow_from_the_ones_before_are_refused(self):
        ledger = copy.deepcopy(self.s.ledger)
        ledger["events"].append({**self.s.round, "start_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "doesn't follow"):
            P.state_of(ledger)
        ledger = copy.deepcopy(self.s.ledger)
        ledger["events"].append({"event": "rollback", "to_round": 5, "weights_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "isn't an earlier live one"):
            P.state_of(ledger)


class VerifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Scenario.get()

    def test_the_whole_ledger_replays(self):
        self.assertEqual(P.verify(self.s.pilot, self.s.ledger), len(self.s.ledger["events"]))

    def refused(self, change, pattern):
        pilot, ledger, tmp = tampered_copy(self.s, change)
        with tmp, self.assertRaisesRegex(ValueError, pattern):
            P.verify(pilot, ledger)

    def test_a_changed_artifact_score_or_round_is_refused(self):
        def artifact(folder, ledger):
            path = folder / ledger["events"][0]["artifact"]
            path.write_text(path.read_text().replace('"seed":11', '"seed":12'))
        self.refused(artifact, "isn't the artifact its id names")

        def score(folder, ledger):
            ledger["events"][0]["evaluation"]["validation_bits"] += 0.01
        self.refused(score, "evaluates as validated")

        def weights(folder, ledger):
            target = folder / ledger["events"][-1]["checkpoint"]
            data = bytearray(target.read_bytes())
            data[0] ^= 1
            target.write_bytes(bytes(data))
        self.refused(weights, "weights")

        def record(folder, ledger):
            ledger["events"][-1]["record"]["contributors"][0]["weight"] = 0.5
        self.refused(record, "weight")

        def inputs(folder, ledger):
            ledger["events"][-1]["submissions"] = ledger["events"][-1]["submissions"][:2]
        self.refused(inputs, "eligible then")

    def test_a_stored_file_the_ledger_doesnt_name_is_refused(self):
        def stray(folder, ledger):
            (folder / "submissions" / "stray.json").write_text("{}")
        self.refused(stray, "doesn't name")

    def test_the_published_pilot_starts_from_the_session_benchmarks_base(self):
        pilot = P.Pilot()
        ledger = P.load_ledger(pilot.folder)
        state = P.state_of(ledger)
        self.assertEqual((state.head.round, state.head.weights_sha256), (1, P.START_SHA256))
        self.assertEqual(P.head_checkpoint(pilot, state.head)["merge"]["weights_sha256"], P.START_SHA256)
        self.assertEqual(P.verify(pilot, ledger), len(ledger["events"]))


if __name__ == "__main__":
    unittest.main()
