"""Milestone 5's sequence routers over the specialist bank (tracks/llm/route_bench.py)."""

import unittest

import numpy as np

from tracks.llm import route_bench as rb
from tracks.llm.anchor import load_anchor
from tracks.llm.data import load_slices
from tracks.llm.model import evaluate
from tracks.llm.moe import family_splits


class RouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.anchor = load_anchor(rb.ANCHOR)
        records = load_slices(rb.MANIFEST)
        cls.partitions = family_splits(records, rb.FAMILIES, "within_session_scan_holdout")
        cls.rng = np.random.default_rng(0)

    def sequences(self, family, split="test"):
        return [self.anchor.tokenizer.encode(r["frames"]) for r in self.partitions[family][split]]

    def test_target_bits_score_each_target_as_evaluate_does(self):
        model = self.anchor.model()
        sequences = self.sequences("voyager")
        bits = rb.target_bits(model, sequences)
        self.assertEqual([len(b) for b in bits], [len(s) - 1 for s in sequences])
        self.assertAlmostEqual(float(np.concatenate(bits).mean()), evaluate(model, sequences)["bits_per_token"], places=5)

    def test_a_route_for_a_target_never_reads_that_target_or_later_tokens(self):
        k, v, n = 3, 33, 49
        tokens = self.rng.integers(0, v, n)
        bits = self.rng.uniform(0.5, 6, (k, n - 1))
        logfreq = np.log2(self.rng.dirichlet(np.ones(v), k))
        for cut in (1, 5, 20, 47):  # change target `cut + 1` (column `cut`) and everything after it
            later_tokens, later_bits = tokens.copy(), bits.copy()
            later_tokens[cut + 1:] = self.rng.integers(0, v, n - cut - 1)
            later_bits[:, cut:] = self.rng.uniform(0.5, 6, (k, n - 1 - cut))
            for a, b in zip(rb.evidence(tokens, bits, logfreq), rb.evidence(later_tokens, later_bits, logfreq)):
                np.testing.assert_array_equal(a[:, :cut + 1], b[:, :cut + 1])
            for name, router in rb.ROUTERS.items():
                _, choice_a = router(tokens, bits, logfreq)
                _, choice_b = router(later_tokens, later_bits, logfreq)
                np.testing.assert_allclose(choice_a[:, :cut + 1], choice_b[:, :cut + 1], err_msg=name)

    def test_routers_send_each_target_to_the_specialist_the_evidence_favours(self):
        # Specialist 1 predicts every target far better, and the first token is commonest in its family.
        k, v, n = 3, 33, 49
        tokens = np.full(n, 7)
        bits = np.full((k, n - 1), 5.0)
        bits[1] = 0.5
        logfreq = np.log2(np.full((k, v), 1 / v))
        logfreq[1, 7] = np.log2(0.5)
        for name, router in rb.ROUTERS.items():
            routed, choice = router(tokens, bits, logfreq)
            np.testing.assert_array_equal(choice.argmax(axis=0), np.ones(n - 1), err_msg=name)
            if name != "posterior mixture":
                np.testing.assert_allclose(routed, bits[1], err_msg=name)
        # The mixture approaches the favoured specialist as its evidence grows.
        routed, weights = rb.ROUTERS["posterior mixture"](tokens, bits, logfreq)
        self.assertGreater(weights[1, -1], 0.999)
        self.assertAlmostEqual(routed[-1], 0.5, places=3)

    def test_committing_freezes_the_route_after_that_many_tokens(self):
        k, v, n = 3, 33, 49
        tokens = self.rng.integers(0, v, n)
        bits = np.full((k, n - 1), 3.0)
        bits[0, :3] = 0.1  # specialist 0 leads early
        bits[2, 3:] = 0.1  # specialist 2 is better from then on
        logfreq = np.log2(np.full((k, v), 1 / v))
        _, followed = rb.ROUTERS["specialists' likelihood"](tokens, bits, logfreq)
        _, committed = rb.ROUTERS["specialists' likelihood, committed after 4 tokens"](tokens, bits, logfreq)
        self.assertEqual(followed.argmax(axis=0)[-1], 2)
        self.assertTrue((committed.argmax(axis=0)[3:] == 0).all(), "decided on the first 4 tokens, then fixed")

    def test_active_parameters_count_what_runs(self):
        single, router = 27_584, 99
        self.assertEqual(rb.active("token frequencies", 48, single, router), single + router)
        self.assertEqual(rb.active("posterior mixture", 48, single, router), 3 * single + router)
        # Committed after 4 tokens: all three predict targets 1–3, which decide; one runs on the other 45.
        self.assertAlmostEqual(rb.active("specialists' likelihood, committed after 4 tokens", 48, single, router),
                               (3 * 3 * single + 45 * single) / 48 + router)

    def test_the_equal_family_validation_counts_each_family_once(self):
        sequences = rb.equal_family(self.anchor, self.partitions, "validation")
        model = self.anchor.model()
        per_family = [evaluate(model, self.sequences(f, "validation"))["bits_per_token"] for f in rb.FAMILIES]
        self.assertAlmostEqual(evaluate(model, sequences)["bits_per_token"], float(np.mean(per_family)), places=5)
        counts = {f: sum(len(s) - 1 for s in self.sequences(f, "validation")) for f in rb.FAMILIES}
        self.assertEqual(sum(len(s) - 1 for s in sequences), 3 * max(counts.values()))


if __name__ == "__main__":
    unittest.main()
