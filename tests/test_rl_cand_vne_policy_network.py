import unittest
import torch
from algorithms.rl_cand_vne.policy_network import (
    GCNEncoder, plackett_luce_topk, DomainHead, SNodeHead, PolicyNetwork,
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


class TestDomainHead(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        torch.manual_seed(0)
        head = DomainHead(hidden=16)
        h_A = torch.randn(16)
        g_domains = torch.randn(3, 16)
        logits = head(h_A, g_domains)
        probs = torch.softmax(logits, dim=0)
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)
        self.assertEqual(logits.shape, (3,))


class TestSNodeHead(unittest.TestCase):
    def test_mask_zeros_infeasible(self):
        torch.manual_seed(0)
        head = SNodeHead(hidden=16)
        h_A = torch.randn(16)
        g_d = torch.randn(16)
        e_snodes = torch.randn(4, 16)
        avail_cpu = torch.tensor([10.0, 1.0, 10.0, 0.5])
        demand = 5.0
        logits = head(h_A, g_d, e_snodes, avail_cpu, demand)
        probs = torch.softmax(logits, dim=0)
        self.assertAlmostEqual(probs[1].item(), 0.0, places=6)
        self.assertAlmostEqual(probs[3].item(), 0.0, places=6)
        self.assertGreater(probs[0].item(), 0.0)
        self.assertGreater(probs[2].item(), 0.0)

    def test_full_mask_fallback(self):
        torch.manual_seed(0)
        head = SNodeHead(hidden=16)
        h_A = torch.randn(16)
        g_d = torch.randn(16)
        e_snodes = torch.randn(3, 16)
        avail_cpu = torch.tensor([0.1, 0.2, 0.3])
        demand = 5.0
        logits = head(h_A, g_d, e_snodes, avail_cpu, demand)
        self.assertTrue(torch.all(torch.isfinite(logits)))
        probs = torch.softmax(logits, dim=0)
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)


class TestPolicyNetworkForward(unittest.TestCase):
    def test_toy_forward(self):
        torch.manual_seed(0)
        pn = PolicyNetwork(vnode_feat_size=5, snode_feat_size=5, hidden=32, K=2)
        vnode_feats = torch.randn(2, 5)
        vn_adj = torch.eye(2)
        domain_inputs_per_vnode = [
            [  # vnode 0: allowed domains = [d1, d2]
                (torch.randn(3, 5), torch.eye(3), torch.tensor([5.0, 10.0, 50.0])),
                (torch.randn(4, 5), torch.eye(4), torch.tensor([5.0, 5.0, 5.0, 50.0])),
            ],
            [  # vnode 1: allowed domains = [d2]
                (torch.randn(4, 5), torch.eye(4), torch.tensor([5.0, 5.0, 5.0, 50.0])),
            ],
        ]
        cpu_demands = [3.0, 4.0]
        result = pn(
            vnode_feats=vnode_feats,
            vn_adj_norm=vn_adj,
            domain_inputs_per_vnode=domain_inputs_per_vnode,
            cpu_demands=cpu_demands,
            sample=True,
        )
        self.assertEqual(len(result["chosen_domains"]), 2)
        self.assertEqual(len(result["chosen_snodes"]), 2)
        self.assertEqual(len(result["chosen_snodes"][0]), 2)
        self.assertGreaterEqual(len(result["chosen_snodes"][1]), 1)
        self.assertEqual(len(result["domain_log_probs"]), 2)
        self.assertEqual(len(result["snode_log_probs_per_vnode"]), 2)


if __name__ == "__main__":
    unittest.main()
