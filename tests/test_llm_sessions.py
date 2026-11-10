"""The session benchmark (tracks/llm/session_bench.py): session shards, the base and its workers,
published checkpoints, and the frozen split.

    python -m unittest discover -s tests -p 'test_llm_sessions.py'
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from tracks.llm import merge, session_bench as bench
from tracks.llm.anchor import load_anchor, weights_digest
from tracks.llm.data import SESSION_GAP_DAYS, digest, groups, load_slices, split_records
from tracks.llm.train import parser, run

ANCHOR = load_anchor("v4")


def write_corpus(folder, sessions=12, hits=3, blind=1, seed=0):
    """A small sessions corpus in the published format: per session (a day apart), `hits` slices of
    a drifting line (family stars) and `blind` of noise (family blind-window), [16, 64] each."""
    rng = np.random.default_rng(seed)
    entries = []
    for s in range(sessions):
        for k in range(hits + blind):
            frames = rng.standard_normal((16, 64)).astype("<f4")
            if k < hits:
                start, drift = rng.integers(8, 56), rng.choice([-2, -1, 1, 2])
                for t in range(16):
                    frames[t, int(np.clip(start + drift * t, 0, 63))] += 25
            family = "stars" if k < hits else "blind-window"
            name = f"s{s:02d}-{family}-{k}"
            payload = frames.tobytes()
            (folder / f"{name}.bin").write_bytes(payload)
            entries.append({"id": name, "family": family, "group": "hit" if k < hits else "blind window", "file": f"{name}.bin",
                            "dtype": "float32-le", "shape": [16, 64], "sha256": digest(payload),
                            "source": {"tstart_mjd": 58000.0 + s + 0.01 * k, "product": "fine_resolution",
                                       "sky": {"target": f"HIP {s}"}}})
    (folder / "index.json").write_text(json.dumps({"format": "eiathome-slices", "version": 1, "slices": entries}))
    return folder / "index.json"


class DrawTests(unittest.TestCase):
    def test_a_later_draw_never_takes_a_session_an_earlier_pool_drew(self):
        from tracks.llm import sessions_corpus as corpus

        names = [f"AGBT{k:02d}A_999_01" for k in range(10)]
        rows = "".join(f'<tr><td><a href="{n}/">{n}/</a></td><td> HIP{k} </td></tr>' for k, n in enumerate(names))
        pages = {corpus.BASE: f"<table>{rows}</table>"}
        for k, n in enumerate(names):
            pages[f"{corpus.BASE}{n}/"] = f'<a href="blc00_guppi_58000_1000{k}_HIP{k}_0001.gpuspec.0000.h5">x</a>'
        quiet = lambda *a: None
        first = corpus.draw(sessions=4, seed=1, fetch_text=pages.__getitem__, log=quiet)
        self.assertEqual(first["id"], "llm-sessions/v1")
        self.assertNotIn("excluded_sessions", first["draw"])
        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp) / "pool.json"
            pool.write_text(json.dumps(first))
            skip = corpus.excluded_sessions([pool])
            second = corpus.draw(sessions=6, seed=1, fetch_text=pages.__getitem__, log=quiet, exclude=skip, pool_id="llm-sessions/v2")
        taken = lambda p: {t["session"] for t in p["targets"]}
        self.assertEqual(len(taken(first)), 4)
        self.assertFalse(taken(first) & taken(second))
        self.assertEqual(taken(first) | taken(second), set(names))
        self.assertEqual((second["id"], second["draw"]["excluded_sessions"], second["draw"]["candidates"]), ("llm-sessions/v2", 4, 6))


class SessionShardTests(unittest.TestCase):
    def test_session_shards_are_whole_disjoint_sessions_that_cover_the_training_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = load_slices(write_corpus(Path(tmp)))
            train = split_records(records, "stars")[0]["train"]
            shards = [merge.shard_records(train, i, 5, "session") for i in range(1, 6)]
            ids = [{r["entry"]["id"] for r in shard} for shard in shards]
            self.assertEqual(set().union(*ids), {r["entry"]["id"] for r in train})
            self.assertEqual(sum(map(len, ids)), len(train), "disjoint")
            for shard in shards:
                self.assertTrue(shard)
                days = {int(r["start"]) for r in shard}
                for other in shards:
                    if other is not shard:
                        self.assertFalse(days & {int(r["start"]) for r in other}, "no two shards share a session")
            # By slice, the same split's shards cut across sessions.
            by_slice = merge.shard_records(train, 1, 5)
            self.assertNotEqual({r["entry"]["id"] for r in by_slice}, ids[0])
            with self.assertRaises(ValueError):
                merge.shard_records(train, 1, len(groups(train, SESSION_GAP_DAYS)) + 1, "session")
            with self.assertRaises(ValueError):
                merge.shard_records(train, 1, 2, "scan")

    def test_a_session_another_familys_scan_holds_together_stays_in_one_shard(self):
        """Two hit scans 0.8 days apart are one session only through a blind window between them,
        as split_records counts sessions; sharding the hits alone mustn't split that session."""
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rng = np.random.default_rng(1)
            entries = []
            for name, family, mjd in (("a", "stars", 58000.0), ("bridge", "blind-window", 58000.4), ("b", "stars", 58000.8),
                                      ("c", "stars", 58002.0), ("d", "stars", 58004.0), ("e", "stars", 58006.0), ("f", "stars", 58008.0)):
                payload = rng.standard_normal((16, 64)).astype("<f4").tobytes()
                (folder / f"{name}.bin").write_bytes(payload)
                entries.append({"id": name, "family": family, "group": family, "file": f"{name}.bin", "dtype": "float32-le",
                                "shape": [16, 64], "sha256": digest(payload),
                                "source": {"tstart_mjd": mjd, "product": "fine_resolution"}})
            (folder / "index.json").write_text(json.dumps({"format": "eiathome-slices", "version": 1, "slices": entries}))
            records = load_slices(folder / "index.json")
            train = split_records(records, "stars")[0]["train"]
            self.assertEqual(sorted(r["entry"]["id"] for r in train), ["a", "b", "c", "d"])
            self.assertEqual(len(groups(train, SESSION_GAP_DAYS)), 4, "the hits alone look like four sessions")
            shards = [{r["entry"]["id"] for r in merge.shard_records(train, i, 2, "session")} for i in (1, 2)]
            self.assertEqual(shards[0] | shards[1], {"a", "b", "c", "d"})
            self.assertFalse(shards[0] & shards[1])
            self.assertTrue(any({"a", "b"} <= shard for shard in shards), f"a and b stay together: {shards}")
            listed = bench.sessions_of(train)
            self.assertEqual(len(listed), 3, "split.json lists them as one session too")
            self.assertEqual({e["id"] for e in listed[0]["slices"]}, {"a", "b"})

    def test_a_shard_unit_needs_a_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_corpus(Path(tmp))
            args = parser().parse_args(["--anchor", "v4", "--manifest", str(manifest), "--family", "stars", "--variants", "real",
                                        "--shard-unit", "session", "--epochs", "1", "--output", str(Path(tmp) / "run")])
            with self.assertRaisesRegex(ValueError, "--shard-unit"):
                run(args)


class BenchmarkTests(unittest.TestCase):
    """The whole benchmark on a small corpus, once: every arm, then publishing and verifying it."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / "data").mkdir()
        cls.manifest = write_corpus(root / "data")
        args = bench.argparse.Namespace(output=root / "run", manifest=cls.manifest, seeds=[1], epochs=2, rounds=2, publish=False)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.report = bench.run(args)
        cls.output = args.output

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_cohort_shares_a_base_and_no_session(self):
        seed = self.report["seeds"]["1"]
        base, workers = bench.cohort_paths(self.output, 1)
        base_cp = merge.load_checkpoint(base, ANCHOR)
        self.assertEqual(base_cp["merge"]["round"], 1)
        self.assertEqual(base_cp["merge"]["start_sha256"], ANCHOR.spec["theta0_sha256"], "the base trains from θ₀")
        sessions = []
        for k, path in workers.items():
            cp = merge.load_checkpoint(path, ANCHOR)
            self.assertEqual(cp["merge"]["start_sha256"], base_cp["merge"]["weights_sha256"], "every worker branches from the base")
            self.assertEqual((cp["merge"]["shard"], cp["merge"]["shard_unit"]), ([k, 5], "session"))
            sessions.append({round(e["source"]["tstart_mjd"]) for e in cp["splits"]["train"]})
        base_run = json.loads((self.output / "seed1" / "base-worker" / "report.json").read_text())
        sessions.append({round(e["source"]["tstart_mjd"]) for e in base_run["splits"]["train"]})
        held = {round(e["source"]["tstart_mjd"]) for name in ("validation", "test") for e in base_cp["splits"][name]}
        for i, a in enumerate(sessions):
            self.assertFalse(a & held, "nothing trains on a held-out session")
            for b in sessions[i + 1:]:
                self.assertFalse(a & b, "no two contributors share a session")
        self.assertEqual(len(seed["workers"]), 4)
        self.assertEqual(seed["best_worker"]["validation"], min(w["validation"] for w in seed["workers"]))

    def test_every_arm_is_scored_on_the_held_out_sessions(self):
        rows = self.report["summary"]
        self.assertEqual(list(rows), [name for name, _ in bench.ARMS])
        for name, row in rows.items():
            self.assertEqual(row["runs"], 4 if name.startswith("workers") else 1, name)
            for key in ("validation", "test", "test_worst_session", "control"):
                self.assertTrue(np.isfinite(row[key]["mean"]), (name, key))
            self.assertGreaterEqual(row["test_worst_session"]["mean"], row["test"]["min"] - 1e-9, "the worst session is no better than the mean")
        seed = self.report["seeds"]["1"]
        for name in bench.CHAINS:
            chain = seed["chains"][name]
            self.assertEqual([h["round"] for h in chain["history"]], [2, 3], "the chains continue the base's chain")
            self.assertEqual(chain["contributions"], 8)
            self.assertEqual(chain["bytes_sent"], 8 * 27_584 * 4)
        self.assertEqual(seed["chains"]["average + DiLoCo"]["rule"]["outer"], bench.DILOCO)
        for name in bench.ONE_SHOT:
            self.assertEqual(seed["one_shot"][name]["round"]["round"], 2)
        self.assertEqual(self.report["environment"]["torch"], str(torch.__version__))
        self.assertIn("python -m tracks.llm.train --anchor v4", seed["workers"][0]["training"]["command"])
        self.assertIn("--shard 2/5 --shard-unit session", seed["workers"][0]["training"]["command"])

    def test_a_chain_may_take_up_an_outer_rule_after_plain_rounds_but_keeps_it(self):
        base, workers = bench.cohort_paths(self.output, 1)
        diloco = self.output / "seed1" / "chain1-round1" / "merged.pt"
        if not diloco.exists():
            self.skipTest("the DiLoCo round was rejected on this small corpus")
        later = sorted((self.output / "seed1").glob("chain1-r2-worker*/real.pt"))
        with self.assertRaisesRegex(ValueError, "keeps one outer step"):
            merge.merge_round(ANCHOR, [str(p) for p in later], str(diloco), {}, self.manifest)

    def test_a_round_refuses_workers_sharded_by_slice_and_by_session_together(self):
        base, workers = bench.cohort_paths(self.output, 1)
        by_slice = Path(self.tmp.name) / "slice-worker"
        with contextlib.redirect_stdout(io.StringIO()):
            run(parser().parse_args(["--anchor", "v4", "--manifest", str(self.manifest), "--family", "stars", "--variants", "real",
                                     "--epochs", "1", "--shard", "2/5", "--start", str(base), "--output", str(by_slice)]))
        self.assertNotIn("shard_unit", merge.load_checkpoint(by_slice / "real.pt", ANCHOR)["merge"], "slice shards are recorded as before")
        with self.assertRaisesRegex(ValueError, "divide"):
            merge.merge_round(ANCHOR, [str(workers[3]), str(by_slice / "real.pt")], str(base), {}, self.manifest)

    def test_published_checkpoints_rebuild_and_verify_and_a_changed_byte_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            split = bench.build_split(self.manifest)
            (folder / "split.json").write_text(bench.split_text(split))
            pin = digest(bench.split_text(split).encode())
            self.assertEqual(self.report["split_sha256"], pin)
            with mock.patch.object(bench, "SPLIT_SHA256", pin):
                bench.publish(self.report, {1: bench.cohort_paths(self.output, 1)}, ANCHOR, folder)
                files = sorted(p.name for p in (folder / "checkpoints" / "seed1").iterdir())
                self.assertEqual(files, sorted(f"{n}.{x}" for n in ("base", "worker2", "worker3", "worker4", "worker5") for x in ("bin", "json")))
                for path in (folder / "checkpoints" / "seed1").glob("*.bin"):
                    self.assertEqual(path.stat().st_size, 27_584 * 4)
                # A rebuilt worker has the original's weights and training slices.
                base, workers = bench.cohort_paths(self.output, 1)
                rebuilt = bench.load_published(ANCHOR, folder / "checkpoints" / "seed1" / "worker3.bin", self.manifest)
                original = merge.load_checkpoint(workers[3], ANCHOR)
                self.assertEqual(weights_digest(rebuilt["model"]), weights_digest(original["model"]))
                self.assertEqual(rebuilt["splits"], original["splits"])
                self.assertEqual(rebuilt["merge"], original["merge"])
                self.assertEqual(bench.verify(self.manifest, folder, seeds=(1,)), [1])

                worker = folder / "checkpoints" / "seed1" / "worker4.bin"
                data = bytearray(worker.read_bytes())
                data[100] ^= 1
                worker.write_bytes(bytes(data))
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    bench.verify(self.manifest, folder, seeds=(1,))
            with mock.patch.object(bench, "SPLIT_SHA256", "0" * 64), self.assertRaisesRegex(ValueError, "pinned"):
                bench.check_split(self.manifest, folder)

    def test_verify_needs_every_seed_and_every_checkpoint(self):
        """An incomplete publication fails: no checkpoints at all, a missing seed or file, or an extra one."""
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            split = bench.build_split(self.manifest)
            (folder / "split.json").write_text(bench.split_text(split))
            pin = digest(bench.split_text(split).encode())
            with mock.patch.object(bench, "SPLIT_SHA256", pin):
                (folder / "report.json").write_text(json.dumps(self.report))
                with self.assertRaisesRegex(ValueError, "holds nothing; the benchmark publishes seed1"):
                    bench.verify(self.manifest, folder, seeds=(1,))
                bench.publish(self.report, {1: bench.cohort_paths(self.output, 1)}, ANCHOR, folder)
                self.assertEqual(bench.verify(self.manifest, folder, seeds=(1,)), [1])
                with self.assertRaisesRegex(ValueError, "covers seeds 1; the benchmark's are 1, 2"):
                    bench.verify(self.manifest, folder, seeds=(1, 2))
                with self.assertRaisesRegex(ValueError, "holds seed1; the benchmark publishes seed1, seed2"):
                    bench.materialize(folder / "out", self.manifest, folder, seeds=(1, 2))
                seed = folder / "checkpoints" / "seed1"
                for name in ("worker5.bin", "base.json"):
                    moved = seed / name
                    moved.rename(folder / name)
                    with self.assertRaisesRegex(ValueError, f"missing {name}"):
                        bench.verify(self.manifest, folder, seeds=(1,))
                    (folder / name).rename(moved)
                (seed / "worker6.bin").write_bytes(b"")
                with self.assertRaisesRegex(ValueError, "unexpected worker6.bin"):
                    bench.verify(self.manifest, folder, seeds=(1,))
                (seed / "worker6.bin").unlink()
                for path in seed.iterdir():
                    path.unlink()
                seed.rmdir()
                with self.assertRaisesRegex(ValueError, "holds nothing"):
                    bench.verify(self.manifest, folder, seeds=(1,))

    def test_publishing_is_only_for_the_frozen_settings(self):
        args = bench.argparse.Namespace(output=Path(self.tmp.name) / "again", manifest=self.manifest, seeds=[1], epochs=2, rounds=2,
                                        publish=True)
        with self.assertRaisesRegex(ValueError, "frozen benchmark"):
            bench.run(args)


class FrozenSplitTests(unittest.TestCase):
    def test_the_committed_split_is_pinned_and_is_what_the_corpus_splits_into(self):
        split = bench.check_split()
        self.assertEqual(split["benchmark"], bench.BENCHMARK)
        sessions = {name: [s["start_mjd"] for s in part["sessions"]] for name, part in split["splits"].items()}
        self.assertLess(max(sessions["train"]), min(sessions["validation"]), "chronological: training sessions come first")
        self.assertLess(max(sessions["validation"]), min(sessions["test"]))
        self.assertGreaterEqual(len(sessions["test"]), 3)
        shards = split["cohort"]["shard_sessions"]
        self.assertEqual(sorted(s for v in shards.values() for s in v), sorted(sessions["train"]), "the shards are the training sessions")

    def test_v1_is_a_fresh_draw_split_the_same_way_and_pinned(self):
        """sessions-v1: drawn after the WSP protocol was pinned, from sessions v0 never drew, split by
        the same rule, its split pinned like v0's."""
        v1 = bench.benchmark("v1")
        split = bench.check_split(bench=v1)
        self.assertEqual((split["benchmark"], split["family"], split["evaluation"]), (bench.BENCHMARK_V1, bench.FAMILY, bench.EVALUATION))
        sessions = {name: [s["start_mjd"] for s in part["sessions"]] for name, part in split["splits"].items()}
        self.assertLess(max(sessions["train"]), min(sessions["validation"]))
        self.assertLess(max(sessions["validation"]), min(sessions["test"]))
        self.assertGreater(len(sessions["test"]), len(bench.check_split()["splits"]["test"]["sessions"]),
                           "more independent test sessions than v0's: the point of drawing more")
        shards = split["cohort"]["shard_sessions"]
        self.assertEqual(sorted(s for v in shards.values() for s in v), sorted(sessions["train"]))

        # Nothing of v0's: no session drawn, no slice, no scan.
        pool = lambda folder: json.loads((Path(bench.ROOT) / folder / "pool.json").read_text())
        p0, p1 = pool("tracks/llm/data/sessions"), pool("tracks/llm/data/sessions-v1")
        self.assertEqual((p1["id"], p1["draw"]["excluded_sessions"]), ("llm-sessions/v2", len(p0["targets"])))
        self.assertFalse({t["session"] for t in p0["targets"]} & {t["session"] for t in p1["targets"]})
        r0, r1 = load_slices(bench.MANIFEST), load_slices(v1.manifest)
        self.assertFalse({r["sha256"] for r in r0} & {r["sha256"] for r in r1})
        self.assertFalse({r["entry"]["source"]["url"] for r in r0} & {r["entry"]["source"]["url"] for r in r1})
        self.assertEqual({r["entry"]["source"]["pool"]["id"] for r in r1}, {"llm-sessions/v2"})
        # Every session v1 is scored on (validation, test, the control's) is another observing session
        # than every one of v0's, by the corpus's own rule (more than 0.5 days apart). Only training
        # sessions may share a night with v0's: v1 trains its own cohort from θ₀, so that leaks nothing.
        def starts(split):
            return ([s["start_mjd"] for part in split["splits"].values() for s in part["sessions"]]
                    + [s["start_mjd"] for name in ("selection", "reporting") for s in split["control"][name]["sessions"]])
        v0 = starts(bench.check_split())
        held_out = {"validation": sessions["validation"], "test": sessions["test"],
                    **{f"control {name}": [s["start_mjd"] for s in split["control"][name]["sessions"]] for name in ("selection", "reporting")}}
        for name, part in held_out.items():
            for start in part:
                self.assertGreater(min(abs(start - s) for s in v0), SESSION_GAP_DAYS, f"v1 {name} session at MJD {start}")


if __name__ == "__main__":
    unittest.main()
