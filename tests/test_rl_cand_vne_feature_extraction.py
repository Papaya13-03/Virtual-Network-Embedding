import unittest
import torch
from problem.substrate_network import SubstrateNetwork, SubstrateNode, SubstrateLink
from problem.domain import PhysicalDomain
from problem.virtual_network import VirtualNetwork, VirtualNode, VirtualLink
from algorithms.rl_cand_vne.feature_extraction import (
    extract_domain_features,
    extract_vnode_features,
    build_vn_adjacency,
)


def _toy_domain():
    net = SubstrateNetwork()
    for nid in ["s1", "s2", "s3"]:
        node = SubstrateNode(id=nid, cpu_capacity=100.0, cpu_price=2.0, processing_delay=1.0)
        node.available_cpu = 50.0
        net.nodes[nid] = node
    for (u, v, bw) in [("s1", "s2", 1000.0), ("s2", "s3", 1000.0)]:
        link = SubstrateLink(source=u, target=v, bandwidth_capacity=bw,
                             bandwidth_price=1.0, transmission_delay=0.5)
        link.available_bw = 800.0
        net.links[(u, v)] = link
    return PhysicalDomain(id="d1", network=net)


def _toy_vn():
    vn = VirtualNetwork(id="vn1")
    vn.nodes = {
        "v1": VirtualNode(id="v1", cpu_demand=5.0),
        "v2": VirtualNode(id="v2", cpu_demand=10.0),
    }
    vn.links = {
        ("v1", "v2"): VirtualLink(source="v1", target="v2", bandwidth_demand=20.0),
    }
    return vn


class TestExtractDomainFeatures(unittest.TestCase):
    def test_shapes_and_values(self):
        d = _toy_domain()
        X, A = extract_domain_features(d)
        self.assertEqual(X.shape, (3, 5))
        self.assertEqual(A.shape, (3, 3))
        self.assertTrue(torch.all(X[:, 0] >= 0) and torch.all(X[:, 0] <= 1))
        row_sums = A.sum(dim=1)
        self.assertTrue(torch.all(row_sums > 0))


class TestExtractVnodeFeatures(unittest.TestCase):
    def test_shape(self):
        vn = _toy_vn()
        feats = extract_vnode_features(vn)
        self.assertEqual(feats.shape, (2, 5))


class TestBuildVnAdjacency(unittest.TestCase):
    def test_symmetric_with_self_loops(self):
        vn = _toy_vn()
        A = build_vn_adjacency(vn)
        self.assertEqual(A.shape, (2, 2))
        self.assertTrue(torch.allclose(A, A.t(), atol=1e-6))
        self.assertTrue(torch.all(torch.diag(A) > 0))


if __name__ == "__main__":
    unittest.main()
