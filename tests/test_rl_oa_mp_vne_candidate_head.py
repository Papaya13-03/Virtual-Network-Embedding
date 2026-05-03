import unittest

import torch

from algorithms.rl_oa_mp_vne.policy_network import CandidateHead, PolicyNetwork
from algorithms.rl_oa_mp_vne.rl_oa_mp_vne import RLOAMPVNE
from problem.request import VirtualNetworkRequest
from problem.substrate_network import SubstrateLink, SubstrateNetwork, SubstrateNode
from problem.virtual_network import VirtualLink, VirtualNetwork, VirtualNode


class TestCandidateHead(unittest.TestCase):
    def _inputs(self, n=6, vnode_feat=5, gcn_hidden=32):
        return (
            torch.randn(vnode_feat),
            torch.randn(n, gcn_hidden),
            torch.tensor([10.0, 20.0, -5.0, 3.0, -1.0, 7.0]),  # 2 infeasible
        )

    def test_shape(self):
        head = CandidateHead(vnode_feat_size=5, gcn_hidden=32, hidden_size=64)
        vf, sf, slack = self._inputs()
        scores = head(vf, sf, slack)
        self.assertEqual(scores.shape, (6,))

    def test_infeasible_masked_to_neg_inf(self):
        head = CandidateHead(vnode_feat_size=5, gcn_hidden=32, hidden_size=64)
        vf, sf, slack = self._inputs()
        scores = head(vf, sf, slack)
        # Indices 2 and 4 have negative slack.
        self.assertTrue(torch.isinf(scores[2]) and scores[2] < 0)
        self.assertTrue(torch.isinf(scores[4]) and scores[4] < 0)
        # Feasible entries must be finite.
        for i in [0, 1, 3, 5]:
            self.assertTrue(torch.isfinite(scores[i]))

    def test_differentiable(self):
        head = CandidateHead(vnode_feat_size=5, gcn_hidden=32, hidden_size=64)
        vf, sf, slack = self._inputs()
        scores = head(vf, sf, slack)
        # Backprop through the finite entries only (sum of feasible scores).
        loss = scores[torch.isfinite(scores)].sum()
        loss.backward()
        got_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in head.parameters()
        )
        self.assertTrue(got_grad)

    def test_all_feasible_all_finite(self):
        head = CandidateHead(vnode_feat_size=5, gcn_hidden=32, hidden_size=64)
        vf = torch.randn(5)
        sf = torch.randn(4, 32)
        slack = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scores = head(vf, sf, slack)
        self.assertTrue(torch.all(torch.isfinite(scores)))

    def test_all_infeasible_all_neg_inf(self):
        head = CandidateHead(vnode_feat_size=5, gcn_hidden=32, hidden_size=64)
        vf = torch.randn(5)
        sf = torch.randn(3, 32)
        slack = torch.tensor([-1.0, -2.0, -3.0])
        scores = head(vf, sf, slack)
        self.assertTrue(torch.all(torch.isneginf(scores)))


class TestPlackettLuceTopK(unittest.TestCase):
    def test_returns_k_distinct_indices(self):
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        chosen, lps = RLOAMPVNE._plackett_luce_topk(scores, k=3)
        self.assertEqual(len(chosen), 3)
        self.assertEqual(len(lps), 3)
        self.assertEqual(len(set(chosen)), 3)
        for c in chosen:
            self.assertIn(c, range(5))

    def test_respects_infeasible_mask(self):
        # 5 entries, only 2 finite
        scores = torch.tensor([float("-inf"), 2.0, float("-inf"), float("-inf"), 5.0])
        chosen, _ = RLOAMPVNE._plackett_luce_topk(scores, k=5)
        self.assertEqual(set(chosen), {1, 4})

    def test_k_larger_than_feasible(self):
        scores = torch.tensor([1.0, 2.0, float("-inf")])
        chosen, lps = RLOAMPVNE._plackett_luce_topk(scores, k=10)
        # Only 2 feasible, so returns 2.
        self.assertEqual(len(chosen), 2)
        self.assertEqual(len(lps), 2)

    def test_k_zero(self):
        chosen, lps = RLOAMPVNE._plackett_luce_topk(torch.tensor([1.0, 2.0]), k=0)
        self.assertEqual(chosen, [])
        self.assertEqual(lps, [])

    def test_all_infeasible_returns_empty(self):
        scores = torch.tensor([float("-inf"), float("-inf")])
        chosen, lps = RLOAMPVNE._plackett_luce_topk(scores, k=3)
        self.assertEqual(chosen, [])
        self.assertEqual(lps, [])

    def test_log_probs_are_differentiable_through_scores(self):
        raw = torch.randn(5, requires_grad=True)
        _, lps = RLOAMPVNE._plackett_luce_topk(raw, k=3)
        loss = sum(lps)
        loss.backward()
        self.assertIsNotNone(raw.grad)
        self.assertGreater(raw.grad.abs().sum().item(), 0.0)


class TestPolicyNetworkWithCandidateHead(unittest.TestCase):
    def test_forward_returns_candidate_scores(self):
        net = PolicyNetwork(
            vnode_feat_size=5, vlink_feat_size=5,
            gcn_node_feat_size=5, gcn_hidden=32, hidden_size=64,
        )
        sub_X = torch.randn(4, 5)
        sub_A = torch.eye(4)
        vnode_feats = torch.randn(3, 5)
        vlink_feats = torch.randn(2, 5)
        domain_Xs = [(sub_X,)] * 3
        domain_As = [(sub_A,)] * 3
        # All feasible for simplicity
        slacks = [torch.tensor([10.0, 10.0, 10.0, 10.0])] * 3

        node_s, link_s, cand_s = net(
            vnode_feats, vlink_feats, domain_Xs, domain_As,
            per_vnode_cpu_slacks=slacks,
        )
        self.assertEqual(node_s.shape, (3,))
        self.assertEqual(link_s.shape, (2,))
        self.assertEqual(len(cand_s), 3)
        for s in cand_s:
            self.assertEqual(s.shape, (4,))

    def test_candidate_head_respects_mask(self):
        net = PolicyNetwork(
            vnode_feat_size=5, vlink_feat_size=5,
            gcn_node_feat_size=5, gcn_hidden=32, hidden_size=64,
        )
        sub_X = torch.randn(3, 5)
        sub_A = torch.eye(3)
        vnode_feats = torch.randn(1, 5)
        vlink_feats = torch.randn(1, 5)
        # vnode 0: snode 1 is infeasible
        slacks = [torch.tensor([5.0, -1.0, 2.0])]

        _, _, cand_s = net(
            vnode_feats, vlink_feats, [(sub_X,)], [(sub_A,)],
            per_vnode_cpu_slacks=slacks,
        )
        self.assertTrue(torch.isinf(cand_s[0][1]) and cand_s[0][1] < 0)
        self.assertTrue(torch.isfinite(cand_s[0][0]))
        self.assertTrue(torch.isfinite(cand_s[0][2]))


class TestRLOAMPVNECandidateIntegration(unittest.TestCase):
    def setUp(self):
        self.substrate = SubstrateNetwork()
        for i in range(1, 7):
            self.substrate.nodes[f"s{i}"] = SubstrateNode(
                f"s{i}", cpu_capacity=100.0, cpu_price=1.0, processing_delay=1.0,
            )
        # Chain topology
        for i in range(1, 6):
            self.substrate.links[(f"s{i}", f"s{i + 1}")] = SubstrateLink(
                f"s{i}", f"s{i + 1}",
                bandwidth_capacity=50.0, bandwidth_price=1.0, transmission_delay=1.0,
            )

    def test_rank_all_nn_returns_topk_candidates(self):
        algo = RLOAMPVNE()
        algo._init_controller(self.substrate)
        algo.config["candidates"]["K"] = 3

        vn = VirtualNetwork(id="vn")
        vn.nodes["v1"] = VirtualNode("v1", cpu_demand=10.0)
        vn.nodes["v2"] = VirtualNode("v2", cpu_demand=10.0)
        vn.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=5.0)

        ordered_vnodes, ordered_vlinks, candidates, lps = algo.rank_all_nn(vn, sample=True)

        self.assertEqual(len(ordered_vnodes), 2)
        self.assertEqual(len(ordered_vlinks), 1)
        self.assertEqual(len(candidates), 2)
        for cand_list in candidates:
            # At most K candidates per vnode; all feasible (sufficient CPU).
            self.assertLessEqual(len(cand_list), 3)
            self.assertGreater(len(cand_list), 0)
            for sn in cand_list:
                self.assertGreaterEqual(sn.cpu_capacity, 10.0)

    def test_candidates_exclude_infeasible_snodes(self):
        algo = RLOAMPVNE()
        algo._init_controller(self.substrate)
        # Squeeze s3 so it cannot host v1 (needs 50 CPU)
        algo.global_controller.snetwork.domains["domain_1"].network.nodes["s3"].available_cpu = 5.0
        algo.config["candidates"]["K"] = 6  # ask for all

        vn = VirtualNetwork(id="vn")
        vn.nodes["v1"] = VirtualNode("v1", cpu_demand=50.0)
        vn.links[("v1", "v1")] = VirtualLink("v1", "v1", bandwidth_demand=1.0)  # self-link placeholder

        ordered_vnodes, _, candidates, _ = algo.rank_all_nn(vn, sample=True)
        cand_ids = {sn.id for sn in candidates[0]}
        self.assertNotIn("s3", cand_ids)

    def test_end_to_end_solve_still_succeeds(self):
        vnr = VirtualNetwork(id="vn_simple")
        vnr.nodes["v1"] = VirtualNode("v1", cpu_demand=5.0)
        vnr.nodes["v2"] = VirtualNode("v2", cpu_demand=5.0)
        vnr.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=10.0)
        request = VirtualNetworkRequest(
            id="req1", virtual_network=vnr, arrival_time=0.0, lifetime=50.0,
        )

        algo = RLOAMPVNE()
        algo.config["training"]["pretrain_episodes"] = 3  # fast
        algo.config["candidates"]["K"] = 3
        solution = algo.solve(self.substrate, request)

        self.assertTrue(solution.is_successful)
        self.assertEqual(len(solution.node_mapping), 2)


if __name__ == "__main__":
    unittest.main()
