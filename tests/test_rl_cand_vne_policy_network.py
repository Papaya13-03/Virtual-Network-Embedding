import unittest
import torch
from algorithms.rl_cand_vne.policy_network import (
    GCNEncoder, plackett_luce_topk,
)


class TestGCNEncoder(unittest.TestCase):
    def test_shape(self):
        enc = GCNEncoder(in_dim=5, hidden=32)
        X = torch.randn(4, 5)
        A = torch.eye(4)
        out = enc(X, A)
        self.assertEqual(out.shape, (4, 32))


class TestPlackettLuceTopK(unittest.TestCase):
    def test_returns_k_distinct_with_logprobs(self):
        torch.manual_seed(0)
        logits = torch.tensor([2.0, 1.0, 0.5, -1.0, 0.1])
        indices, log_probs = plackett_luce_topk(logits, k=3)
        self.assertEqual(len(indices), 3)
        self.assertEqual(len(log_probs), 3)
        self.assertEqual(len(set(indices)), 3)
        for lp in log_probs:
            self.assertLessEqual(lp.item(), 0.0)

    def test_k_capped_at_n(self):
        logits = torch.tensor([1.0, 2.0])
        indices, log_probs = plackett_luce_topk(logits, k=5)
        self.assertEqual(len(indices), 2)
        self.assertEqual(len(log_probs), 2)


if __name__ == "__main__":
    unittest.main()
