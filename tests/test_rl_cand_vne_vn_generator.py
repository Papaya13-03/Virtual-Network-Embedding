import unittest
import random
from algorithms.rl_cand_vne.vn_generator import generate_random_vn_with_domains


class TestGenerateRandomVnWithDomains(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        self.domain_ids = ["d1", "d2", "d3", "d4"]

    def test_basic_shape(self):
        vn = generate_random_vn_with_domains(
            min_nodes=3, max_nodes=5,
            min_cpu=1.0, max_cpu=10.0,
            min_bw=5.0, max_bw=20.0,
            link_prob=0.5,
            domain_ids=self.domain_ids,
            p_all=0.5, p_single=0.3, p_subset=0.2,
            subset_min=2, subset_max=3,
        )
        self.assertGreaterEqual(len(vn.nodes), 3)
        self.assertLessEqual(len(vn.nodes), 5)

    def test_allowed_domains_distribution(self):
        counts = {"all": 0, "single": 0, "subset": 0}
        for _ in range(500):
            vn = generate_random_vn_with_domains(
                min_nodes=3, max_nodes=3,
                min_cpu=1.0, max_cpu=10.0,
                min_bw=5.0, max_bw=20.0,
                link_prob=0.5,
                domain_ids=self.domain_ids,
                p_all=0.5, p_single=0.3, p_subset=0.2,
                subset_min=2, subset_max=3,
            )
            for node in vn.nodes.values():
                if not node.allowed_domains:
                    counts["all"] += 1
                elif len(node.allowed_domains) == 1:
                    counts["single"] += 1
                else:
                    counts["subset"] += 1
        total = sum(counts.values())
        self.assertAlmostEqual(counts["all"] / total, 0.5, delta=0.07)
        self.assertAlmostEqual(counts["single"] / total, 0.3, delta=0.07)
        self.assertAlmostEqual(counts["subset"] / total, 0.2, delta=0.07)

    def test_allowed_domains_are_valid(self):
        for _ in range(50):
            vn = generate_random_vn_with_domains(
                min_nodes=2, max_nodes=4,
                min_cpu=1.0, max_cpu=10.0,
                min_bw=5.0, max_bw=20.0,
                link_prob=0.5,
                domain_ids=self.domain_ids,
                p_all=0.3, p_single=0.4, p_subset=0.3,
                subset_min=2, subset_max=3,
            )
            for node in vn.nodes.values():
                for d in node.allowed_domains:
                    self.assertIn(d, self.domain_ids)


if __name__ == "__main__":
    unittest.main()
