import unittest
import random
from problem.substrate_network import SubstrateNetwork, SubstrateNode, SubstrateLink
from problem.domain import PhysicalDomain
from algorithms.rl_cand_vne.state_sampler import (
    fractional_drop,
    warmup_embed,
    sample_substrate_state,
)
from algorithms.oa_mp_vne.global_controller import GlobalController
from problem.domain import MultiDomainNetwork


def _build_mini_substrate() -> MultiDomainNetwork:
    """Build a small 2-domain MultiDomainNetwork. GlobalController expects this shape."""
    md = MultiDomainNetwork()
    for did in ["d1", "d2"]:
        sn = SubstrateNetwork()
        for j in range(3):
            nid = f"{did}_n{j}"
            node = SubstrateNode(id=nid, cpu_capacity=100.0, cpu_price=2.0, processing_delay=1.0)
            node.available_cpu = 100.0
            sn.nodes[nid] = node
        intra_pairs = [(f"{did}_n0", f"{did}_n1"), (f"{did}_n1", f"{did}_n2")]
        for u, v in intra_pairs:
            lk = SubstrateLink(source=u, target=v, bandwidth_capacity=1000.0,
                               bandwidth_price=1.0, transmission_delay=0.5)
            lk.available_bw = 1000.0
            sn.links[(u, v)] = lk
        md.domains[did] = PhysicalDomain(id=did, network=sn, boundary_nodes={f"{did}_n2" if did == "d1" else f"{did}_n0"})
    inter = SubstrateLink(source="d1_n2", target="d2_n0", bandwidth_capacity=1000.0,
                          bandwidth_price=1.0, transmission_delay=0.5)
    inter.available_bw = 1000.0
    md.inter_domain_links[("d1_n2", "d2_n0")] = inter
    return md


class TestFractionalDrop(unittest.TestCase):
    def test_available_within_bounds(self):
        random.seed(0)
        md = _build_mini_substrate()
        gc = GlobalController(md)
        fractional_drop(gc, u_max_cpu=0.8, u_max_bw=0.6)
        for lc in gc.local_controllers:
            for node in lc.domain.network.nodes.values():
                self.assertGreaterEqual(node.available_cpu, node.cpu_capacity * (1 - 0.8) - 1e-6)
                self.assertLessEqual(node.available_cpu, node.cpu_capacity + 1e-6)
            for link in lc.domain.network.links.values():
                self.assertGreaterEqual(link.available_bw, link.bandwidth_capacity * (1 - 0.6) - 1e-6)
                self.assertLessEqual(link.available_bw, link.bandwidth_capacity + 1e-6)


class TestWarmupEmbed(unittest.TestCase):
    def test_no_over_allocation(self):
        random.seed(0)
        md = _build_mini_substrate()
        gc = GlobalController(md)
        warmup_embed(gc, md, M_max=5, vn_kwargs={
            "min_nodes": 2, "max_nodes": 3,
            "min_cpu": 1.0, "max_cpu": 5.0,
            "min_bw": 5.0, "max_bw": 20.0,
            "link_prob": 0.5,
        })
        for lc in gc.local_controllers:
            for node in lc.domain.network.nodes.values():
                self.assertGreaterEqual(node.available_cpu, -1e-6)
                self.assertLessEqual(node.available_cpu, node.cpu_capacity + 1e-6)
            for link in lc.domain.network.links.values():
                self.assertGreaterEqual(link.available_bw, -1e-6)
                self.assertLessEqual(link.available_bw, link.bandwidth_capacity + 1e-6)


class TestSampleSubstrateState(unittest.TestCase):
    def test_routes_to_both_modes(self):
        random.seed(0)
        md = _build_mini_substrate()
        gc = GlobalController(md)
        modes = set()
        for _ in range(50):
            mode = sample_substrate_state(gc, md, warmup_fraction=0.5,
                                          u_max_cpu=0.8, u_max_bw=0.8, M_max=3,
                                          vn_kwargs={
                                              "min_nodes": 2, "max_nodes": 2,
                                              "min_cpu": 1.0, "max_cpu": 5.0,
                                              "min_bw": 5.0, "max_bw": 20.0,
                                              "link_prob": 0.5,
                                          })
            modes.add(mode)
        self.assertEqual(modes, {"fractional_drop", "warmup_embed"})


if __name__ == "__main__":
    unittest.main()
