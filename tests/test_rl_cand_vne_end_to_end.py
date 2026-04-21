import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
import torch

from problem.substrate_network import SubstrateNetwork, SubstrateNode, SubstrateLink
from problem.domain import PhysicalDomain, MultiDomainNetwork
from problem.virtual_network import VirtualNetwork, VirtualNode, VirtualLink
from problem.request import VirtualNetworkRequest
from algorithms.rl_cand_vne.rl_cand_vne import RLCandVNE


def _build_sn() -> MultiDomainNetwork:
    """Build a 2-domain MultiDomainNetwork (the type GlobalController expects)."""
    md = MultiDomainNetwork()
    for did, boundary in [("d1", "d1_n2"), ("d2", "d2_n0")]:
        sn = SubstrateNetwork()
        for j in range(3):
            nid = f"{did}_n{j}"
            n = SubstrateNode(id=nid, cpu_capacity=100.0, cpu_price=2.0, processing_delay=1.0)
            n.available_cpu = 100.0
            sn.nodes[nid] = n
        for u, v in [(f"{did}_n0", f"{did}_n1"), (f"{did}_n1", f"{did}_n2")]:
            lk = SubstrateLink(source=u, target=v, bandwidth_capacity=1000.0,
                               bandwidth_price=1.0, transmission_delay=0.5)
            lk.available_bw = 1000.0
            sn.links[(u, v)] = lk
        md.domains[did] = PhysicalDomain(id=did, network=sn, boundary_nodes={boundary})
    inter = SubstrateLink(source="d1_n2", target="d2_n0", bandwidth_capacity=1000.0,
                          bandwidth_price=1.0, transmission_delay=0.5)
    inter.available_bw = 1000.0
    md.inter_domain_links[("d1_n2", "d2_n0")] = inter
    return md


def _build_vn():
    vn = VirtualNetwork(id="vn1")
    vn.nodes = {
        "v1": VirtualNode(id="v1", cpu_demand=5.0),
        "v2": VirtualNode(id="v2", cpu_demand=5.0),
    }
    vn.links = {
        ("v1", "v2"): VirtualLink(source="v1", target="v2", bandwidth_demand=20.0),
    }
    return vn


class TestRLCandVNESolve(unittest.TestCase):
    def test_solve_produces_valid_solution(self):
        random.seed(0)
        torch.manual_seed(0)
        sn = _build_sn()
        vn = _build_vn()
        req = VirtualNetworkRequest(id="r1", virtual_network=vn,
                                    arrival_time=0.0, lifetime=100.0)
        algo = RLCandVNE()
        algo.config["training"]["inline_pretrain_episodes"] = 0
        solution = algo.solve(sn, req)
        self.assertEqual(solution.vnr_id, "r1")
        if solution.is_successful:
            # Collect all substrate node ids from all domains
            all_snodes = {}
            for domain in sn.domains.values():
                all_snodes.update(domain.network.nodes)
            for v_id, s_id in solution.node_mapping.items():
                self.assertIn(v_id, vn.nodes)
                self.assertIn(s_id, all_snodes)


class TestCheckpointIO(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        random.seed(0); torch.manual_seed(0)
        algo = RLCandVNE()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            algo.save_checkpoint(path, substrate_hash="abc")
            self.assertTrue(os.path.exists(path))

            algo2 = RLCandVNE()
            ok = algo2.load_checkpoint(path, expected_hash="abc")
            self.assertTrue(ok)

    def test_hash_mismatch_warns_but_loads(self):
        random.seed(0); torch.manual_seed(0)
        algo = RLCandVNE()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            algo.save_checkpoint(path, substrate_hash="abc")
            algo2 = RLCandVNE()
            # Different expected hash -> with require_hash_match=False this still loads.
            ok = algo2.load_checkpoint(path, expected_hash="different")
            self.assertTrue(ok)


class TestInlinePretrain(unittest.TestCase):
    def test_inline_pretrain_runs_without_error(self):
        random.seed(0); torch.manual_seed(0)
        sn = _build_sn()
        algo = RLCandVNE()
        algo.config["training"]["inline_pretrain_episodes"] = 5
        algo.config["training"]["batch_size"] = 2
        algo.config["training"]["warmup_fraction"] = 0.0  # keep the test fast
        algo.pretrain_inline(sn)
        self.assertTrue(algo._pretrained)


class TestOnlineLearning(unittest.TestCase):
    def test_online_updates_fire_without_errors(self):
        random.seed(0); torch.manual_seed(0)
        sn = _build_sn()
        algo = RLCandVNE()
        algo.config["training"]["inline_pretrain_episodes"] = 0
        algo.config["training"]["online_k"] = 5
        algo.config["training"]["batch_size"] = 5
        for i in range(10):
            vn = VirtualNetwork(id=f"v{i}")
            vn.nodes = {
                "a": VirtualNode(id="a", cpu_demand=3.0),
                "b": VirtualNode(id="b", cpu_demand=3.0),
            }
            vn.links = {("a", "b"): VirtualLink(source="a", target="b", bandwidth_demand=10.0)}
            req = VirtualNetworkRequest(id=f"r{i}", virtual_network=vn,
                                        arrival_time=float(i), lifetime=1000.0)
            algo.solve(sn, req)
        self.assertGreaterEqual(algo._request_count, 10)


class TestRegistry(unittest.TestCase):
    def test_registry_returns_rl_cand_vne(self):
        from algorithms.registry import get_algorithm
        algo = get_algorithm("rl_cand_vne")
        self.assertEqual(algo.__class__.__name__, "RLCandVNE")


def _dump_substrate(md, path):
    """Dump MultiDomainNetwork to the JSON schema read_substrate() parses."""
    data = {"domains": [], "inter_domain_links": []}
    for d in md.domains.values():
        data["domains"].append({
            "id": d.id,
            "nodes": [{"id": n.id, "cpu_capacity": n.cpu_capacity,
                       "cpu_price": n.cpu_price, "processing_delay": n.processing_delay}
                      for n in d.network.nodes.values()],
            "links": [{"source": lk.source, "target": lk.target,
                       "bandwidth_capacity": lk.bandwidth_capacity,
                       "bandwidth_price": lk.bandwidth_price,
                       "transmission_delay": lk.transmission_delay}
                      for lk in d.network.links.values()],
        })
    for lk in md.inter_domain_links.values():
        data["inter_domain_links"].append({
            "source": lk.source, "target": lk.target,
            "bandwidth_capacity": lk.bandwidth_capacity,
            "bandwidth_price": lk.bandwidth_price,
            "transmission_delay": lk.transmission_delay,
        })
    with open(path, "w") as f:
        json.dump(data, f)


class TestOfflineTrainingToSolveIntegration(unittest.TestCase):
    def test_offline_train_then_solve(self):
        random.seed(0)
        torch.manual_seed(0)
        # Write a tiny substrate JSON for the offline script.
        sn = _build_sn()
        with tempfile.TemporaryDirectory() as tmp:
            sn_path = os.path.join(tmp, "substrate.json")
            _dump_substrate(sn, sn_path)
            ckpt = os.path.join(tmp, "ckpt.pt")
            log_dir = os.path.join(tmp, "log")
            cfg_path = os.path.join(tmp, "cfg.yaml")
            import yaml as _y
            cfg = _y.safe_load(open("configs/rl_cand_vne.yaml"))
            cfg["training"]["warmup_fraction"] = 0.0
            cfg["training"]["batch_size"] = 5
            cfg["training"]["vn_max_nodes"] = 3
            with open(cfg_path, "w") as f:
                _y.safe_dump(cfg, f)

            res = subprocess.run(
                [sys.executable, "scripts/train_rl_cand_vne.py",
                 "--substrate", sn_path, "--config", cfg_path,
                 "--episodes", "20", "--checkpoint", ckpt, "--log-dir", log_dir,
                 "--seed", "0"],
                capture_output=True, text=True, timeout=120,
                cwd="/Users/duvannguyen/Workspace/Studies/Virtual-Network-Embedding",
            )
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertTrue(os.path.exists(ckpt))
            log_file = os.path.join(log_dir, "train.jsonl")
            self.assertTrue(os.path.exists(log_file))
            with open(log_file) as f:
                lines = f.readlines()
            self.assertGreaterEqual(len(lines), 1)
            for line in lines:
                rec = json.loads(line)
                for key in ["loss_total", "reward_mean", "success_rate",
                            "cost_per_revenue_mean", "baseline"]:
                    self.assertIn(key, rec)

            algo = RLCandVNE()
            algo.config["checkpoint"]["path"] = ckpt
            algo.config["training"]["inline_pretrain_episodes"] = 0
            vn = _build_vn()
            req = VirtualNetworkRequest(id="r1", virtual_network=vn,
                                        arrival_time=0.0, lifetime=100.0)
            solution = algo.solve(sn, req)
            self.assertEqual(solution.vnr_id, "r1")


if __name__ == "__main__":
    unittest.main()
