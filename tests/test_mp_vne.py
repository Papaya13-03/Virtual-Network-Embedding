import unittest

from problem.substrate_network import SubstrateNetwork, SubstrateNode, SubstrateLink
from problem.virtual_network import VirtualNetwork, VirtualNode, VirtualLink
from problem.request import VirtualNetworkRequest
from algorithms.mp_vne.legacy import MPVNELegacy

class TestMPVNE(unittest.TestCase):
    def setUp(self):
        # Create a simple substrate network
        self.substrate = SubstrateNetwork()
        
        # 3 Nodes
        self.substrate.nodes["s1"] = SubstrateNode("s1", cpu_capacity=100.0, cpu_price=1.0, processing_delay=1.0)
        self.substrate.nodes["s2"] = SubstrateNode("s2", cpu_capacity=100.0, cpu_price=1.0, processing_delay=1.0)
        self.substrate.nodes["s3"] = SubstrateNode("s3", cpu_capacity=100.0, cpu_price=1.0, processing_delay=1.0)
        
        # Two paths from s1 to s3:
        # Path A: s1 -> s3 (Capacity: 50, Delay: 1.0)
        # Path B: s1 -> s2 -> s3 (Capacity: 50, Delay: 2.0 total)
        self.substrate.links[("s1", "s3")] = SubstrateLink("s1", "s3", bandwidth_capacity=50.0, bandwidth_price=1.0, transmission_delay=1.0)
        self.substrate.links[("s1", "s2")] = SubstrateLink("s1", "s2", bandwidth_capacity=50.0, bandwidth_price=1.0, transmission_delay=1.0)
        self.substrate.links[("s2", "s3")] = SubstrateLink("s2", "s3", bandwidth_capacity=50.0, bandwidth_price=1.0, transmission_delay=1.0)

    def test_multipath_splitting(self):
        # Virtual Request asking for 80 units of bandwidth between two nodes
        # This forces the algorithm to split traffic (50 on Path A, 30 on Path B)
        vnr = VirtualNetwork(id="vn1")
        vnr.nodes["v1"] = VirtualNode("v1", cpu_demand=10.0)
        vnr.nodes["v2"] = VirtualNode("v2", cpu_demand=10.0)
        
        # We need 80 BW. Single path max is 50.
        vnr.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=80.0)
        
        request = VirtualNetworkRequest(id="req1", virtual_network=vnr, arrival_time=0.0, lifetime=100.0)
        
        # Act
        mp_vne = MPVNELegacy()
        
        # Force the algorithm to map v1 to s1 and v2 to s3 
        # (Since they are the only ones big enough if we restrict CPU dynamically, but our greedy will hit them anyway)
        # To guarantee mapping, let's temporarily modify s2 CPU to 0
        self.substrate.nodes["s2"].cpu_capacity = 0.0
        
        solution = mp_vne.solve(self.substrate, request)
        
        # Assert
        self.assertTrue(solution.is_successful)
        link_maps = solution.link_mapping[("v1", "v2")]
        
        # Check that it split into at least 2 paths
        self.assertGreaterEqual(len(link_maps), 2)
        
        total_allocated_bw = sum(bw for path, bw in link_maps)
        
        # We requested 80, we should have gotten exactly 80
        self.assertAlmostEqual(total_allocated_bw, 80.0)

if __name__ == "__main__":
    unittest.main()
