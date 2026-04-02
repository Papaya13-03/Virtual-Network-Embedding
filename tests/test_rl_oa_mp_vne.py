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

        node_scores, link_scores = net(
            vnode_feats, vlink_feats,
            domain_node_feats, domain_adj_mats
        )
        self.assertEqual(node_scores.shape, (3,))
        self.assertEqual(link_scores.shape, (2,))

    def test_scores_are_differentiable(self):
        """Scores must support backpropagation for REINFORCE."""
        net = PolicyNetwork(vnode_feat_size=5, vlink_feat_size=5, gcn_node_feat_size=5, gcn_hidden=32, hidden_size=64)
        sub_X = torch.randn(4, 5)
        sub_A = torch.eye(4)
        vnode_feats = torch.randn(3, 5)
        vlink_feats = torch.randn(2, 5)
        domain_node_feats = [(sub_X,)] * 3
        domain_adj_mats = [(sub_A,)] * 3

        node_scores, link_scores = net(
            vnode_feats, vlink_feats,
            domain_node_feats, domain_adj_mats
        )
        loss = node_scores.sum() + link_scores.sum()
        loss.backward()
        for p in net.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)


if __name__ == "__main__":
    unittest.main()
