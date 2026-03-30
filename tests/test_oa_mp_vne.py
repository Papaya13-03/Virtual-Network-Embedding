import unittest

from problem.virtual_network import VirtualNetwork, VirtualNode, VirtualLink


class TestOAMPVNENodeOrdering(unittest.TestCase):
    def _make_vnetwork(self):
        """
        Build a virtual network with 4 nodes:
          v1 -- v2 -- v3
                |
                v4
        v2 has degree=3, v1/v3/v4 have degree=1.
        v3 has the highest CPU demand.
        v1 has the highest adjacent BW.
        """
        vn = VirtualNetwork(id="vn_test")
        vn.nodes["v1"] = VirtualNode("v1", cpu_demand=5.0)
        vn.nodes["v2"] = VirtualNode("v2", cpu_demand=10.0)
        vn.nodes["v3"] = VirtualNode("v3", cpu_demand=20.0)
        vn.nodes["v4"] = VirtualNode("v4", cpu_demand=2.0)

        vn.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=100.0)
        vn.links[("v2", "v3")] = VirtualLink("v2", "v3", bandwidth_demand=30.0)
        vn.links[("v2", "v4")] = VirtualLink("v2", "v4", bandwidth_demand=10.0)
        return vn

    def test_node_ordering_returns_all_nodes(self):
        """Ordering must return all virtual nodes, no duplicates."""
        from algorithms.oa_mp_vne.oa_mp_vne import OAMPVNE
        algo = OAMPVNE()
        vn = self._make_vnetwork()
        ordered = algo.rank_virtual_nodes(vn)
        self.assertEqual(len(ordered), 4)
        self.assertEqual(set(n.id for n in ordered), {"v1", "v2", "v3", "v4"})

    def test_highest_degree_node_ranked_first(self):
        """v2 (degree=3) should rank higher than degree-1 nodes with default weights."""
        from algorithms.oa_mp_vne.oa_mp_vne import OAMPVNE
        algo = OAMPVNE()
        vn = self._make_vnetwork()
        ordered = algo.rank_virtual_nodes(vn)
        # v2 has the highest degree (3 links) — it should be first
        self.assertEqual(ordered[0].id, "v2")

    def test_single_node_network(self):
        """A network with one node should return that node."""
        from algorithms.oa_mp_vne.oa_mp_vne import OAMPVNE
        algo = OAMPVNE()
        vn = VirtualNetwork(id="single")
        vn.nodes["v1"] = VirtualNode("v1", cpu_demand=5.0)
        ordered = algo.rank_virtual_nodes(vn)
        self.assertEqual(len(ordered), 1)
        self.assertEqual(ordered[0].id, "v1")


if __name__ == "__main__":
    unittest.main()
