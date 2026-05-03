import unittest

from algorithms.mp_vne.global_controller import GlobalController
from problem.domain import MultiDomainNetwork, PhysicalDomain
from problem.substrate_network import SubstrateLink, SubstrateNetwork, SubstrateNode
from problem.virtual_network import VirtualLink, VirtualNetwork, VirtualNode


def _snode(id_, cpu=100.0, price=1.0):
    return SubstrateNode(id=id_, cpu_capacity=cpu, cpu_price=price, processing_delay=1.0)


def _slink(src, dst, bw=50.0, price=1.0, delay=1.0):
    return SubstrateLink(
        source=src, target=dst,
        bandwidth_capacity=bw, bandwidth_price=price, transmission_delay=delay,
    )


def _build_multi_domain():
    """Two domains connected by one inter-domain link.

        domain_A:  a1 --10bw-- a2 (boundary)
        domain_B:  b1 (boundary) --10bw-- b2
        inter:     a2 --5bw-- b1
    """
    net_a = SubstrateNetwork()
    net_a.nodes["a1"] = _snode("a1", cpu=100.0, price=1.0)
    net_a.nodes["a2"] = _snode("a2", cpu=100.0, price=2.0)
    net_a.links[("a1", "a2")] = _slink("a1", "a2", bw=10.0, price=1.0)

    net_b = SubstrateNetwork()
    net_b.nodes["b1"] = _snode("b1", cpu=100.0, price=3.0)
    net_b.nodes["b2"] = _snode("b2", cpu=100.0, price=4.0)
    net_b.links[("b1", "b2")] = _slink("b1", "b2", bw=10.0, price=1.0)

    mdn = MultiDomainNetwork(
        domains={
            "domain_A": PhysicalDomain(id="domain_A", network=net_a),
            "domain_B": PhysicalDomain(id="domain_B", network=net_b),
        },
        inter_domain_links={
            ("a2", "b1"): _slink("a2", "b1", bw=5.0, price=1.0, delay=2.0),
        },
    )
    return mdn


class TestGlobalControllerSingleDomainWrapping(unittest.TestCase):
    def test_single_substrate_wrapped_as_multidomain(self):
        substrate = SubstrateNetwork()
        substrate.nodes["s1"] = _snode("s1")
        substrate.nodes["s2"] = _snode("s2")
        substrate.links[("s1", "s2")] = _slink("s1", "s2")

        gc = GlobalController(substrate)

        self.assertIsInstance(gc.snetwork, MultiDomainNetwork)
        self.assertIn("domain_1", gc.snetwork.domains)
        self.assertEqual(gc.snetwork.inter_domain_links, {})
        self.assertEqual(len(gc.local_controllers), 1)

    def test_inter_domain_link_available_bw_initialized(self):
        mdn = _build_multi_domain()
        gc = GlobalController(mdn)

        inter = gc.snetwork.inter_domain_links[("a2", "b1")]
        self.assertEqual(inter.available_bw, inter.bandwidth_capacity)


class TestGlobalControllerBoundaryIndex(unittest.TestCase):
    def test_boundary_index_built_from_inter_domain_links(self):
        mdn = _build_multi_domain()
        gc = GlobalController(mdn)

        self.assertEqual(gc._boundary_of["domain_A"], {"a2"})
        self.assertEqual(gc._boundary_of["domain_B"], {"b1"})
        self.assertEqual(gc._boundary_between[("domain_A", "domain_B")], {"a2"})
        self.assertEqual(gc._boundary_between[("domain_B", "domain_A")], {"b1"})


class TestGlobalControllerProcessRequest(unittest.TestCase):
    def setUp(self):
        self.gc = GlobalController(_build_multi_domain())

    def test_candidates_respect_allowed_domains(self):
        vn = VirtualNetwork(id="vn1")
        vn.nodes["v1"] = VirtualNode("v1", cpu_demand=10.0, allowed_domains=["domain_A"])
        vn.nodes["v2"] = VirtualNode("v2", cpu_demand=10.0, allowed_domains=["domain_B"])
        vn.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=1.0)

        candidates = self.gc.process_request(vn, top_k=5)

        v1_ids = {n.id for n in candidates[0]}
        v2_ids = {n.id for n in candidates[1]}
        self.assertTrue(v1_ids.issubset({"a1", "a2"}))
        self.assertTrue(v2_ids.issubset({"b1", "b2"}))

    def test_top_k_limits_per_domain(self):
        vn = VirtualNetwork(id="vn2")
        vn.nodes["v1"] = VirtualNode("v1", cpu_demand=10.0)  # allowed=all
        # No vlinks => PreCost degenerates to node_term; top_k still applies

        candidates = self.gc.process_request(vn, top_k=1)

        self.assertEqual(len(candidates), 1)
        # top_k=1 per domain, 2 domains => <= 2 total
        self.assertLessEqual(len(candidates[0]), 2)

    def test_precost_prefers_cheaper_cpu_node_with_no_vlinks(self):
        vn = VirtualNetwork(id="vn3")
        vn.nodes["v1"] = VirtualNode(
            "v1", cpu_demand=10.0, allowed_domains=["domain_A"],
        )

        candidates = self.gc.process_request(vn, top_k=1)

        # domain_A has a1 (price 1.0) and a2 (price 2.0); a1 should be picked.
        self.assertEqual(len(candidates[0]), 1)
        self.assertEqual(candidates[0][0].id, "a1")


class TestGlobalControllerShortestPath(unittest.TestCase):
    def setUp(self):
        self.gc = GlobalController(_build_multi_domain())

    def test_intra_domain_path_delegated_to_local(self):
        src = self.gc.snetwork.domains["domain_A"].network.nodes["a1"]
        dst = self.gc.snetwork.domains["domain_A"].network.nodes["a2"]

        path = self.gc.shortest_path(src, dst, bw_required=0.0)

        self.assertEqual(len(path), 1)
        self.assertEqual((path[0].source, path[0].target), ("a1", "a2"))

    def test_inter_domain_path_crosses_boundary(self):
        src = self.gc.snetwork.domains["domain_A"].network.nodes["a1"]
        dst = self.gc.snetwork.domains["domain_B"].network.nodes["b2"]

        path = self.gc.shortest_path(src, dst, bw_required=0.0)

        self.assertGreater(len(path), 0)
        visited_nodes = [path[0].source] + [l.target for l in path]
        self.assertEqual(visited_nodes[0], "a1")
        self.assertEqual(visited_nodes[-1], "b2")
        self.assertIn("a2", visited_nodes)
        self.assertIn("b1", visited_nodes)

    def test_inter_domain_path_respects_bw_requirement(self):
        src = self.gc.snetwork.domains["domain_A"].network.nodes["a1"]
        dst = self.gc.snetwork.domains["domain_B"].network.nodes["b2"]

        # Inter-domain link capacity is 5 — request more than that.
        path = self.gc.shortest_path(src, dst, bw_required=999.0, use_cache=False)

        self.assertEqual(path, [])


class TestGlobalControllerCommitRelease(unittest.TestCase):
    def setUp(self):
        self.gc = GlobalController(_build_multi_domain())

        self.vn = VirtualNetwork(id="vn")
        self.vn.nodes["v1"] = VirtualNode("v1", cpu_demand=20.0)
        self.vn.nodes["v2"] = VirtualNode("v2", cpu_demand=30.0)
        self.vn.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=5.0)

        self.mapping = {"v1": "a1", "v2": "b2"}

    def _snapshot_resources(self):
        cpu = {}
        bw = {}
        for lc in self.gc.local_controllers:
            for n in lc.domain.network.nodes.values():
                cpu[n.id] = n.available_cpu
            for key, l in lc.domain.network.links.items():
                bw[key] = l.available_bw
        for key, l in self.gc.snetwork.inter_domain_links.items():
            bw[key] = l.available_bw
        return cpu, bw

    def test_commit_then_release_restores_resources(self):
        cpu_before, bw_before = self._snapshot_resources()

        vlink_paths = self.gc.commit_mapping(self.mapping, self.vn)
        self.assertIn(("v1", "v2"), vlink_paths)

        cpu_during, bw_during = self._snapshot_resources()
        self.assertEqual(cpu_during["a1"], cpu_before["a1"] - 20.0)
        self.assertEqual(cpu_during["b2"], cpu_before["b2"] - 30.0)
        self.assertLess(bw_during[("a2", "b1")], bw_before[("a2", "b1")])

        self.gc.release_mapping(self.mapping, self.vn, vlink_paths)

        cpu_after, bw_after = self._snapshot_resources()
        self.assertEqual(cpu_after, cpu_before)
        for k in bw_before:
            self.assertAlmostEqual(bw_after[k], bw_before[k], places=6)

    def test_commit_rollback_on_insufficient_bandwidth(self):
        # Inter-domain capacity is 5; demand 100 cannot be satisfied even
        # with splitting across 5 paths.
        self.vn.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=100.0)

        cpu_before, bw_before = self._snapshot_resources()

        with self.assertRaises(ValueError):
            self.gc.commit_mapping(self.mapping, self.vn)

        cpu_after, bw_after = self._snapshot_resources()
        # CPU must have been restored (rollback path)
        self.assertEqual(cpu_after, cpu_before)
        for k in bw_before:
            self.assertAlmostEqual(bw_after[k], bw_before[k], places=6)

    def test_commit_rollback_on_missing_snode(self):
        bad_mapping = {"v1": "a1", "v2": "does_not_exist"}
        cpu_before, _ = self._snapshot_resources()

        with self.assertRaises(ValueError):
            self.gc.commit_mapping(bad_mapping, self.vn)

        cpu_after, _ = self._snapshot_resources()
        self.assertEqual(cpu_after, cpu_before)

    def test_commit_rollback_on_insufficient_cpu(self):
        # Squeeze a1 so it can't fit v1's cpu_demand.
        a1 = self.gc.snetwork.domains["domain_A"].network.nodes["a1"]
        a1.available_cpu = 5.0  # v1 needs 20
        cpu_before, bw_before = self._snapshot_resources()

        with self.assertRaises(ValueError):
            self.gc.commit_mapping(self.mapping, self.vn)

        cpu_after, bw_after = self._snapshot_resources()
        self.assertEqual(cpu_after, cpu_before)
        for k in bw_before:
            self.assertAlmostEqual(bw_after[k], bw_before[k], places=6)


class TestGlobalControllerMultiPathSplitting(unittest.TestCase):
    def test_multipath_splitting_inside_single_domain(self):
        substrate = SubstrateNetwork()
        substrate.nodes["s1"] = _snode("s1")
        substrate.nodes["s2"] = _snode("s2")
        substrate.nodes["s3"] = _snode("s3")
        # Two parallel paths s1 -> s3 (direct) and s1 -> s2 -> s3.
        substrate.links[("s1", "s3")] = _slink("s1", "s3", bw=50.0)
        substrate.links[("s1", "s2")] = _slink("s1", "s2", bw=50.0)
        substrate.links[("s2", "s3")] = _slink("s2", "s3", bw=50.0)

        gc = GlobalController(substrate)

        vn = VirtualNetwork(id="vn")
        vn.nodes["v1"] = VirtualNode("v1", cpu_demand=10.0)
        vn.nodes["v2"] = VirtualNode("v2", cpu_demand=10.0)
        vn.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=80.0)

        paths = gc.commit_mapping({"v1": "s1", "v2": "s3"}, vn)

        allocated = paths[("v1", "v2")]
        self.assertGreaterEqual(len(allocated), 2)
        total = sum(bw for _, bw in allocated)
        self.assertAlmostEqual(total, 80.0, places=4)


class TestGlobalControllerResetAndCache(unittest.TestCase):
    def test_reset_restores_all_capacities(self):
        gc = GlobalController(_build_multi_domain())

        # Mutate available_bw and available_cpu directly.
        for lc in gc.local_controllers:
            for n in lc.domain.network.nodes.values():
                n.available_cpu = 0.0
            for l in lc.domain.network.links.values():
                l.available_bw = 0.0
        for l in gc.snetwork.inter_domain_links.values():
            l.available_bw = 0.0

        gc.reset_allocations()

        for lc in gc.local_controllers:
            for n in lc.domain.network.nodes.values():
                self.assertEqual(n.available_cpu, n.cpu_capacity)
            for l in lc.domain.network.links.values():
                self.assertEqual(l.available_bw, l.bandwidth_capacity)
        for l in gc.snetwork.inter_domain_links.values():
            self.assertEqual(l.available_bw, l.bandwidth_capacity)

    def test_clear_caches_wipes_floyd_warshall_cache(self):
        gc = GlobalController(_build_multi_domain())
        src = gc.snetwork.domains["domain_A"].network.nodes["a1"]
        dst = gc.snetwork.domains["domain_B"].network.nodes["b2"]
        gc.shortest_path(src, dst, bw_required=1.0)
        self.assertTrue(gc._id_fw_cache)

        gc.clear_caches()

        self.assertEqual(gc._id_fw_cache, {})


if __name__ == "__main__":
    unittest.main()
