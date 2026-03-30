import unittest

from problem.virtual_network import VirtualNetwork, VirtualNode, VirtualLink
from problem.substrate_network import SubstrateNetwork, SubstrateNode, SubstrateLink
from problem.request import VirtualNetworkRequest


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


class TestOAMPVNELinkOrdering(unittest.TestCase):
    def test_links_sorted_by_bandwidth_descending(self):
        """Virtual links should be ordered highest BW demand first."""
        from algorithms.oa_mp_vne.oa_mp_vne import OAMPVNE
        algo = OAMPVNE()

        vn = VirtualNetwork(id="vn_link")
        vn.nodes["v1"] = VirtualNode("v1", cpu_demand=5.0)
        vn.nodes["v2"] = VirtualNode("v2", cpu_demand=5.0)
        vn.nodes["v3"] = VirtualNode("v3", cpu_demand=5.0)

        vn.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=10.0)
        vn.links[("v2", "v3")] = VirtualLink("v2", "v3", bandwidth_demand=50.0)
        vn.links[("v1", "v3")] = VirtualLink("v1", "v3", bandwidth_demand=30.0)

        ranked = algo.rank_virtual_links(vn)
        bw_values = [vlink.bandwidth_demand for _, vlink in ranked]
        self.assertEqual(bw_values, [50.0, 30.0, 10.0])

    def test_single_link(self):
        """A network with one link returns that link."""
        from algorithms.oa_mp_vne.oa_mp_vne import OAMPVNE
        algo = OAMPVNE()

        vn = VirtualNetwork(id="vn_single")
        vn.nodes["v1"] = VirtualNode("v1", cpu_demand=5.0)
        vn.nodes["v2"] = VirtualNode("v2", cpu_demand=5.0)
        vn.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=25.0)

        ranked = algo.rank_virtual_links(vn)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0][1].bandwidth_demand, 25.0)


class TestOAMPVNEEndToEnd(unittest.TestCase):
    def setUp(self):
        self.substrate = SubstrateNetwork()
        self.substrate.nodes["s1"] = SubstrateNode("s1", cpu_capacity=100.0, cpu_price=1.0, processing_delay=1.0)
        self.substrate.nodes["s2"] = SubstrateNode("s2", cpu_capacity=100.0, cpu_price=1.0, processing_delay=1.0)
        self.substrate.nodes["s3"] = SubstrateNode("s3", cpu_capacity=100.0, cpu_price=1.0, processing_delay=1.0)

        self.substrate.links[("s1", "s3")] = SubstrateLink("s1", "s3", bandwidth_capacity=50.0, bandwidth_price=1.0, transmission_delay=1.0)
        self.substrate.links[("s1", "s2")] = SubstrateLink("s1", "s2", bandwidth_capacity=50.0, bandwidth_price=1.0, transmission_delay=1.0)
        self.substrate.links[("s2", "s3")] = SubstrateLink("s2", "s3", bandwidth_capacity=50.0, bandwidth_price=1.0, transmission_delay=1.0)

    def test_multipath_splitting(self):
        """Same test as MP-VNE: 80 BW demand forces multi-path split."""
        from algorithms.oa_mp_vne.oa_mp_vne import OAMPVNE

        vnr = VirtualNetwork(id="vn1")
        vnr.nodes["v1"] = VirtualNode("v1", cpu_demand=10.0)
        vnr.nodes["v2"] = VirtualNode("v2", cpu_demand=10.0)
        vnr.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=80.0)

        request = VirtualNetworkRequest(id="req1", virtual_network=vnr, arrival_time=0.0, lifetime=100.0)

        # Force v1->s1, v2->s3 by disabling s2 CPU
        self.substrate.nodes["s2"].cpu_capacity = 0.0

        algo = OAMPVNE()
        solution = algo.solve(self.substrate, request)

        self.assertTrue(solution.is_successful)
        link_maps = solution.link_mapping[("v1", "v2")]
        self.assertGreaterEqual(len(link_maps), 2)
        total_bw = sum(bw for _, bw in link_maps)
        self.assertAlmostEqual(total_bw, 80.0)

    def test_simple_embedding_succeeds(self):
        """A simple 2-node virtual network with low demand should always succeed."""
        from algorithms.oa_mp_vne.oa_mp_vne import OAMPVNE

        vnr = VirtualNetwork(id="vn_simple")
        vnr.nodes["v1"] = VirtualNode("v1", cpu_demand=5.0)
        vnr.nodes["v2"] = VirtualNode("v2", cpu_demand=5.0)
        vnr.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=10.0)

        request = VirtualNetworkRequest(id="req_simple", virtual_network=vnr, arrival_time=0.0, lifetime=50.0)

        algo = OAMPVNE()
        solution = algo.solve(self.substrate, request)

        self.assertTrue(solution.is_successful)
        self.assertEqual(len(solution.node_mapping), 2)
        self.assertNotEqual(
            list(solution.node_mapping.values())[0],
            list(solution.node_mapping.values())[1]
        )

    def test_infeasible_request_fails_gracefully(self):
        """A request demanding more CPU than available should fail."""
        from algorithms.oa_mp_vne.oa_mp_vne import OAMPVNE

        vnr = VirtualNetwork(id="vn_fail")
        vnr.nodes["v1"] = VirtualNode("v1", cpu_demand=500.0)
        vnr.nodes["v2"] = VirtualNode("v2", cpu_demand=500.0)
        vnr.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=10.0)

        request = VirtualNetworkRequest(id="req_fail", virtual_network=vnr, arrival_time=0.0, lifetime=50.0)

        algo = OAMPVNE()
        solution = algo.solve(self.substrate, request)

        self.assertFalse(solution.is_successful)


if __name__ == "__main__":
    unittest.main()
