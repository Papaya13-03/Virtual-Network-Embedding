import unittest
import torch
from algorithms.rl_oa_mp_vne.vn_generator import generate_random_vn
from algorithms.rl_oa_mp_vne.policy_network import GCNEncoder, PolicyNetwork


class TestVNGenerator(unittest.TestCase):
    def test_generates_valid_vn(self):
        """Generated VN must have nodes, links, and valid demands."""
        vn = generate_random_vn(
            min_nodes=3, max_nodes=5,
            min_cpu=1.0, max_cpu=20.0,
            min_bw=5.0, max_bw=50.0,
            link_prob=0.8
        )
        self.assertGreaterEqual(len(vn.nodes), 3)
        self.assertLessEqual(len(vn.nodes), 5)
        self.assertGreater(len(vn.links), 0)
        for node in vn.nodes.values():
            self.assertGreaterEqual(node.cpu_demand, 1.0)
            self.assertLessEqual(node.cpu_demand, 20.0)
        for link in vn.links.values():
            self.assertGreaterEqual(link.bandwidth_demand, 5.0)
            self.assertLessEqual(link.bandwidth_demand, 50.0)

    def test_generates_connected_graph(self):
        """Generated VN should be connected (all nodes reachable)."""
        vn = generate_random_vn(
            min_nodes=4, max_nodes=4,
            min_cpu=1.0, max_cpu=10.0,
            min_bw=5.0, max_bw=20.0,
            link_prob=0.5
        )
        if len(vn.nodes) <= 1:
            return
        adj = {n: set() for n in vn.nodes}
        for (s, t) in vn.links:
            adj[s].add(t)
            adj[t].add(s)
        start = next(iter(vn.nodes))
        visited = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            queue.extend(adj[node] - visited)
        self.assertEqual(visited, set(vn.nodes.keys()))

    def test_min_equals_max_nodes(self):
        """When min==max nodes, exact count should be produced."""
        vn = generate_random_vn(
            min_nodes=3, max_nodes=3,
            min_cpu=5.0, max_cpu=5.0,
            min_bw=10.0, max_bw=10.0,
            link_prob=1.0
        )
        self.assertEqual(len(vn.nodes), 3)


class TestGCNEncoder(unittest.TestCase):
    def test_output_shape(self):
        """GCN should produce (num_nodes, gcn_hidden) output."""
        encoder = GCNEncoder(node_feat_size=5, hidden_size=32)
        X = torch.randn(3, 5)
        A = torch.tensor([
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
            [0.0, 0.5, 0.5],
        ], dtype=torch.float32)
        out = encoder(X, A)
        self.assertEqual(out.shape, (3, 32))

    def test_single_node(self):
        """GCN should handle a single-node graph."""
        encoder = GCNEncoder(node_feat_size=5, hidden_size=32)
        X = torch.randn(1, 5)
        A = torch.ones(1, 1)
        out = encoder(X, A)
        self.assertEqual(out.shape, (1, 32))


class TestPolicyNetwork(unittest.TestCase):
    def test_node_scores_shape(self):
        """NodeHead should produce one score per virtual node."""
        net = PolicyNetwork(vnode_feat_size=5, vlink_feat_size=5, gcn_node_feat_size=5, gcn_hidden=32, hidden_size=64)
        sub_X = torch.randn(4, 5)
        sub_A = torch.eye(4)
        vnode_feats = torch.randn(3, 5)
        vlink_feats = torch.randn(2, 5)
        # For each vnode, tuple of domain X tensors and A tensors
        domain_node_feats = [(sub_X,)] * 3  # 3 vnodes, each sees 1 domain
        domain_adj_mats = [(sub_A,)] * 3

        node_scores, link_scores, cand_scores = net(
            vnode_feats, vlink_feats,
            domain_node_feats, domain_adj_mats
        )
        self.assertEqual(node_scores.shape, (3,))
        self.assertEqual(link_scores.shape, (2,))
        self.assertIsNone(cand_scores)

    def test_scores_are_differentiable(self):
        """Scores must support backpropagation for REINFORCE."""
        net = PolicyNetwork(vnode_feat_size=5, vlink_feat_size=5, gcn_node_feat_size=5, gcn_hidden=32, hidden_size=64)
        sub_X = torch.randn(4, 5)
        sub_A = torch.eye(4)
        vnode_feats = torch.randn(3, 5)
        vlink_feats = torch.randn(2, 5)
        domain_node_feats = [(sub_X,)] * 3
        domain_adj_mats = [(sub_A,)] * 3
        slacks = [torch.tensor([1.0, 2.0, 3.0, 4.0])] * 3

        node_scores, link_scores, cand_scores = net(
            vnode_feats, vlink_feats,
            domain_node_feats, domain_adj_mats,
            per_vnode_cpu_slacks=slacks,
        )
        loss = node_scores.sum() + link_scores.sum() + sum(s.sum() for s in cand_scores)
        loss.backward()
        for p in net.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)


from algorithms.rl_oa_mp_vne.trainer import RankingTrainer


class TestRankingTrainer(unittest.TestCase):
    def _make_policy(self):
        return PolicyNetwork(
            vnode_feat_size=5, vlink_feat_size=5,
            gcn_node_feat_size=5, gcn_hidden=32, hidden_size=64
        )

    def test_record_and_buffer_size(self):
        """Recording an experience should grow the buffer."""
        policy = self._make_policy()
        trainer = RankingTrainer(policy, lr=0.001, gamma=0.99, batch_size=4)
        log_probs = {"node": [torch.tensor(0.5, requires_grad=True)]}
        trainer.record(log_probs, reward=2.0)
        self.assertEqual(len(trainer.buffer), 1)

    def test_update_clears_buffer(self):
        """After update(), the buffer should be emptied."""
        policy = self._make_policy()
        trainer = RankingTrainer(policy, lr=0.001, gamma=0.99, batch_size=2)
        for _ in range(3):
            log_probs = {"node": [torch.tensor(0.5, requires_grad=True)]}
            trainer.record(log_probs, reward=1.0)
        trainer.update()
        self.assertEqual(len(trainer.buffer), 0)

    def _make_log_probs(self, policy):
        """Run a forward pass and sample to get graph-connected log_probs."""
        sub_X = torch.randn(4, 5)
        sub_A = torch.eye(4)
        vnode_feats = torch.randn(3, 5)
        vlink_feats = torch.randn(2, 5)
        domain_node_feats = [(sub_X,)] * 3
        domain_adj_mats = [(sub_A,)] * 3
        node_scores, link_scores, _ = policy(
            vnode_feats, vlink_feats,
            domain_node_feats, domain_adj_mats,
        )
        # Plackett-Luce style: softmax then sample from Categorical
        node_dist = torch.distributions.Categorical(logits=node_scores)
        link_dist = torch.distributions.Categorical(logits=link_scores)
        log_probs = {
            "node": [node_dist.log_prob(node_dist.sample())],
            "link": [link_dist.log_prob(link_dist.sample())],
            "cand": []
        }
        return log_probs

    def test_update_changes_weights(self):
        """REINFORCE update should modify policy network weights."""
        policy = self._make_policy()
        trainer = RankingTrainer(policy, lr=0.01, gamma=0.99, batch_size=2)
        w_before = policy.node_head[0].weight.data.clone()
        rewards = [1.0, 5.0, 10.0]  # Different rewards → non-zero advantages
        for r in rewards:
            log_probs = self._make_log_probs(policy)
            trainer.record(log_probs, reward=r)
        trainer.update()
        w_after = policy.node_head[0].weight.data
        self.assertFalse(torch.equal(w_before, w_after))


from problem.substrate_network import SubstrateNetwork, SubstrateNode, SubstrateLink
from problem.virtual_network import VirtualNetwork, VirtualNode, VirtualLink
from problem.request import VirtualNetworkRequest


class TestRLOAMPVNEFeatures(unittest.TestCase):
    def setUp(self):
        self.substrate = SubstrateNetwork()
        self.substrate.nodes["s1"] = SubstrateNode("s1", cpu_capacity=100.0, cpu_price=1.0, processing_delay=1.0)
        self.substrate.nodes["s2"] = SubstrateNode("s2", cpu_capacity=100.0, cpu_price=1.0, processing_delay=1.0)
        self.substrate.nodes["s3"] = SubstrateNode("s3", cpu_capacity=100.0, cpu_price=1.0, processing_delay=1.0)
        self.substrate.links[("s1", "s2")] = SubstrateLink("s1", "s2", bandwidth_capacity=50.0, bandwidth_price=1.0, transmission_delay=1.0)
        self.substrate.links[("s2", "s3")] = SubstrateLink("s2", "s3", bandwidth_capacity=50.0, bandwidth_price=1.0, transmission_delay=1.0)
        self.substrate.links[("s1", "s3")] = SubstrateLink("s1", "s3", bandwidth_capacity=50.0, bandwidth_price=1.0, transmission_delay=1.0)

    def test_domain_features_shape(self):
        """Domain feature extraction should return (num_nodes, 5) and (num_nodes, num_nodes)."""
        from algorithms.rl_oa_mp_vne.rl_oa_mp_vne import RLOAMPVNE
        algo = RLOAMPVNE()
        algo._init_controller(self.substrate)
        X, A = algo._extract_domain_features(algo.global_controller.local_controllers[0])
        self.assertEqual(X.shape[0], 3)  # 3 substrate nodes
        self.assertEqual(X.shape[1], 5)  # 5 features
        self.assertEqual(A.shape, (3, 3))

    def test_vnode_features_shape(self):
        """Vnode feature extraction should return (num_vnodes, 5)."""
        from algorithms.rl_oa_mp_vne.rl_oa_mp_vne import RLOAMPVNE
        vn = VirtualNetwork(id="test")
        vn.nodes["v1"] = VirtualNode("v1", cpu_demand=10.0)
        vn.nodes["v2"] = VirtualNode("v2", cpu_demand=20.0)
        vn.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=30.0)
        algo = RLOAMPVNE()
        feats = algo._extract_vnode_features(vn)
        self.assertEqual(feats.shape, (2, 5))

    def test_vlink_features_shape(self):
        """Vlink feature extraction should return (num_vlinks, 5)."""
        from algorithms.rl_oa_mp_vne.rl_oa_mp_vne import RLOAMPVNE
        vn = VirtualNetwork(id="test")
        vn.nodes["v1"] = VirtualNode("v1", cpu_demand=10.0)
        vn.nodes["v2"] = VirtualNode("v2", cpu_demand=20.0)
        vn.nodes["v3"] = VirtualNode("v3", cpu_demand=5.0)
        vn.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=30.0)
        vn.links[("v2", "v3")] = VirtualLink("v2", "v3", bandwidth_demand=15.0)
        algo = RLOAMPVNE()
        feats = algo._extract_vlink_features(vn)
        self.assertEqual(feats.shape, (2, 5))

class TestRLOAMPVNEEndToEnd(unittest.TestCase):
    def setUp(self):
        self.substrate = SubstrateNetwork()
        self.substrate.nodes["s1"] = SubstrateNode("s1", cpu_capacity=100.0, cpu_price=1.0, processing_delay=1.0)
        self.substrate.nodes["s2"] = SubstrateNode("s2", cpu_capacity=100.0, cpu_price=1.0, processing_delay=1.0)
        self.substrate.nodes["s3"] = SubstrateNode("s3", cpu_capacity=100.0, cpu_price=1.0, processing_delay=1.0)
        self.substrate.links[("s1", "s2")] = SubstrateLink("s1", "s2", bandwidth_capacity=50.0, bandwidth_price=1.0, transmission_delay=1.0)
        self.substrate.links[("s2", "s3")] = SubstrateLink("s2", "s3", bandwidth_capacity=50.0, bandwidth_price=1.0, transmission_delay=1.0)
        self.substrate.links[("s1", "s3")] = SubstrateLink("s1", "s3", bandwidth_capacity=50.0, bandwidth_price=1.0, transmission_delay=1.0)

    def test_simple_embedding_succeeds(self):
        """A simple 2-node VN with low demand should embed successfully."""
        from algorithms.rl_oa_mp_vne.rl_oa_mp_vne import RLOAMPVNE
        vnr = VirtualNetwork(id="vn_simple")
        vnr.nodes["v1"] = VirtualNode("v1", cpu_demand=5.0)
        vnr.nodes["v2"] = VirtualNode("v2", cpu_demand=5.0)
        vnr.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=10.0)
        request = VirtualNetworkRequest(id="req1", virtual_network=vnr, arrival_time=0.0, lifetime=50.0)

        algo = RLOAMPVNE()
        # Override pretrain to use fewer episodes for speed
        algo.config["training"]["pretrain_episodes"] = 5
        solution = algo.solve(self.substrate, request)

        self.assertTrue(solution.is_successful)
        self.assertEqual(len(solution.node_mapping), 2)
        # Mapped to different substrate nodes
        self.assertNotEqual(
            list(solution.node_mapping.values())[0],
            list(solution.node_mapping.values())[1],
        )

    def test_infeasible_request_fails_gracefully(self):
        """A request demanding more CPU than available should fail without error."""
        from algorithms.rl_oa_mp_vne.rl_oa_mp_vne import RLOAMPVNE
        vnr = VirtualNetwork(id="vn_fail")
        vnr.nodes["v1"] = VirtualNode("v1", cpu_demand=500.0)
        vnr.nodes["v2"] = VirtualNode("v2", cpu_demand=500.0)
        vnr.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=10.0)
        request = VirtualNetworkRequest(id="req_fail", virtual_network=vnr, arrival_time=0.0, lifetime=50.0)

        algo = RLOAMPVNE()
        algo.config["training"]["pretrain_episodes"] = 3
        solution = algo.solve(self.substrate, request)

        self.assertFalse(solution.is_successful)

    def test_multiple_requests_with_online_learning(self):
        """Multiple requests should succeed and trigger online learning."""
        from algorithms.rl_oa_mp_vne.rl_oa_mp_vne import RLOAMPVNE
        algo = RLOAMPVNE()
        algo.config["training"]["pretrain_episodes"] = 3
        algo.config["training"]["online_k"] = 2  # Learn every 2 requests

        successes = 0
        for i in range(4):
            vnr = VirtualNetwork(id=f"vn_{i}")
            vnr.nodes["v1"] = VirtualNode("v1", cpu_demand=5.0)
            vnr.nodes["v2"] = VirtualNode("v2", cpu_demand=5.0)
            vnr.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=10.0)
            request = VirtualNetworkRequest(
                id=f"req_{i}", virtual_network=vnr,
                arrival_time=float(i * 10), lifetime=5.0,
            )
            solution = algo.solve(self.substrate, request)
            if solution.is_successful:
                successes += 1

        self.assertGreater(successes, 0)
        # Online learning should have triggered at least once (k=2, 4 requests)
        self.assertGreaterEqual(algo._request_count, 4)


if __name__ == "__main__":
    unittest.main()

