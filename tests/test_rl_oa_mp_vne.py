import unittest
from algorithms.rl_oa_mp_vne.vn_generator import generate_random_vn


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


if __name__ == "__main__":
    unittest.main()
