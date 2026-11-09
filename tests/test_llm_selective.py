"""Selective specialist execution (tracks/llm/selective.py): the committed routers run selectively
reproduce route_bench.py's routers, count what they run, and the energy and memory readers work.

    python -m unittest discover -s tests -p 'test_llm_selective.py'
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tracks.llm import selective as sel
from tracks.llm.anchor import load_anchor
from tracks.llm.merge import model_from
from tracks.llm.route_bench import ROUTERS, evidence, hard, target_bits

ANCHOR = load_anchor("v3")


def specialists(count=3):
    """Three different models: θ₀ with seeded noise, so their predictions (and the router's
    evidence) differ and don't tie."""
    out = []
    for k in range(count):
        g = torch.Generator().manual_seed(k)
        out.append(model_from(ANCHOR, {n: v + 0.3 * torch.randn(v.shape, generator=g) for n, v in ANCHOR.theta0.items()}))
    return out


def sequences(lengths, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 33, size=n) for n in lengths]


def logfreq(seed=0):
    p = np.random.default_rng(seed).dirichlet(np.ones(33), size=3)
    return np.log2(p)


class WindowTests(unittest.TestCase):
    def test_windows_score_as_target_bits_does_batched_or_not(self):
        """One window at a time, exactly; batched, to float32 rounding (a batch of other windows
        changes the matrix shapes)."""
        model = specialists(1)[0]
        seqs = sequences([2, 5, 15, 16, 17, 31, 49])
        reference = target_bits(model, seqs)
        for batched, tolerance in ((False, 0), (True, 1e-5)):
            for a, b in zip(sel.window_bits(model, seqs, batched), reference):
                np.testing.assert_allclose(a, b, rtol=0, atol=tolerance)

    def test_the_work_is_counted(self):
        model = specialists(1)[0]
        work = sel.Work()
        sel.window_bits(model, sequences([49, 49]), batched=False, work=work)
        self.assertEqual((work.calls, work.positions), (8, 96))  # 4 windows each: 15, 15, 15, 3 positions
        work = sel.Work()
        sel.window_bits(model, sequences([49, 49]), batched=True, work=work)
        self.assertEqual((work.calls, work.positions), (2, 96))  # one batch of the 15s, one of the 3s


class CommittedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = specialists()
        cls.logfreq = logfreq()

    def test_selective_routing_reproduces_route_benchs_committed_routers(self):
        seqs = sequences([2, 3, 8, 9, 16, 30, 49, 49, 49], seed=1)
        per_model = [target_bits(m, seqs) for m in self.models]
        for n in (1, 2, 4, 8, 20, 60):
            # route_bench's routers for 4 and 8 tokens, and its rule for any other n
            router = ROUTERS.get(f"specialists' likelihood, committed after {n} tokens") or (
                lambda tokens, bits, f, n=n: hard(bits, evidence(tokens, bits, f)[1][:, np.minimum(np.arange(bits.shape[1]), n - 1)]))
            selective = sel.committed(self.models, self.logfreq, seqs, n)
            for i, s in enumerate(seqs):
                ref_bits, ref_weights = router(s, np.stack([p[i] for p in per_model]), self.logfreq)
                bits, choice = selective[i]
                with self.subTest(n=n, length=len(s)):
                    np.testing.assert_allclose(bits, ref_bits, rtol=0, atol=1e-5)
                    np.testing.assert_array_equal(choice, ref_weights.argmax(axis=0))

    def test_only_the_committed_specialist_runs_after_the_evidence(self):
        works = [sel.Work() for _ in self.models]
        sel.committed(self.models, self.logfreq, sequences([49]), 8, batched=False, works=works)
        calls = [w.calls for w in works]
        positions = [w.positions for w in works]
        self.assertEqual(sorted(calls), [1, 1, 5])  # the prefix on each, then the committed one's 4 windows
        self.assertEqual(sum(positions), 3 * 7 + 48)

        works = [sel.Work() for _ in self.models]
        sel.bank(self.models, self.logfreq, sequences([49]), batched=False, works=works)
        self.assertEqual(([w.calls for w in works], sum(w.positions for w in works)), ([4, 4, 4], 144))

    def test_windows_the_evidence_completed_are_not_run_again(self):
        """The committed specialist resumes at the window its first target falls in: windows of
        15 targets, a 49-token sequence (48 targets)."""
        for n, extra in ((8, 7), (16, 0), (20, 4)):  # evidence ends inside window 1, at its end, inside window 2
            works = [sel.Work() for _ in self.models]
            (bits, _), = sel.committed(self.models, self.logfreq, sequences([49], seed=2), n, batched=False, works=works)
            evidence = n - 1
            with self.subTest(n=n):
                self.assertEqual(len(bits), 48)
                self.assertEqual(sum(w.positions for w in works), 3 * evidence + (48 - evidence) + extra)
                self.assertEqual(max(w.positions for w in works) - evidence, 48 - evidence + extra)

    def test_a_sequence_shorter_than_the_evidence_never_commits(self):
        works = [sel.Work() for _ in self.models]
        (bits, choice), = sel.committed(self.models, self.logfreq, sequences([5]), 8, works=works)
        self.assertEqual(len(bits), 4)
        self.assertEqual([w.calls for w in works], [1, 1, 1])  # the prefix only: no specialist runs alone


class MeasurementTests(unittest.TestCase):
    def test_rapl_counters_are_read_where_present_and_wrap_around(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(sel.rapl_domains(root), [])
            for name, energy in (("intel-rapl:0", "package-0"), ("intel-rapl:1", "package-1"), ("intel-rapl:0:0", "core")):
                d = root / name
                d.mkdir()
                (d / "name").write_text(energy + "\n")
                (d / "energy_uj").write_text("1000\n")
                (d / "max_energy_range_uj").write_text("5000\n")
            domains = sel.rapl_domains(root)
            self.assertEqual(len(domains), 2, "packages only, not their subdomains")
            before = sel.energy_reading(domains)
            (root / "intel-rapl:0" / "energy_uj").write_text("4000\n")
            (root / "intel-rapl:1" / "energy_uj").write_text("500\n")  # wrapped past 5000
            self.assertAlmostEqual(sel.energy_joules(domains, before, sel.energy_reading(domains)), (3000 + 4500) / 1e6)

    def test_timing_reports_no_energy_without_counters(self):
        calls = []
        out = sel.timed({"a": lambda: calls.append("a"), "b": lambda: calls.append("b")}, repeats=3, domains=[])
        self.assertEqual(calls, ["a", "b", "a", "b", "b", "a", "a", "b"], "a warm-up each, then turns in a rotating order")
        out = out["a"]
        self.assertIsNone(out["energy_j"])
        self.assertGreaterEqual(out["wall_s"], 0)
        self.assertEqual(len(out["wall_s_quartiles"]), 2)

    @unittest.skipUnless(Path("/proc/self/status").exists(), "Linux only")
    def test_peak_memory_is_read(self):
        memory = sel.peak_memory_kib()
        self.assertGreater(memory["VmHWM"], 0)
        self.assertGreaterEqual(memory["VmHWM"], memory["VmRSS"])


class PublishedReportTests(unittest.TestCase):
    def test_the_published_measurement_reproduced_the_routers_exactly(self):
        report = json.loads((sel.FOLDER / "report.json").read_text())
        self.assertEqual(sorted(report["seeds"]), ["42", "43", "44"])
        for seed in report["seeds"].values():
            for n in sel.COMMIT_AFTER:
                exact = seed["exactness"][f"committed after {n}"]
                self.assertEqual(exact["targets_routed_differently"], 0)
                self.assertLess(exact["max_abs_bits_difference"], 1e-5)
            arms = seed["arms"]
            self.assertEqual(arms["bank"]["forward_calls_per_sequence"], 12)
            self.assertAlmostEqual(arms["committed after 8"]["positions_per_target"], 69 / 48)
            self.assertEqual(arms["committed after 8"]["weights_resident_bytes"], arms["bank"]["weights_resident_bytes"])
            # batched: one call per (specialist, window length) in each batch
            self.assertEqual([arms[a]["batched"]["forward_calls"] for a in sel.ARMS], [6, 9, 9, 6])
            self.assertEqual(arms["bank"]["batched_per_family"]["forward_calls"], 18)
            for arm in ("committed after 8", "committed after 4"):
                groups = sum(len(to) for to in arms[arm]["committed_to"].values())  # (family, specialist) pairs
                self.assertEqual(arms[arm]["batched_per_family"]["forward_calls"], 9 + 2 * groups)
        self.assertTrue(report["summary"]["committed after 8"]["latency_vs_bank"] < 1)


if __name__ == "__main__":
    unittest.main()
