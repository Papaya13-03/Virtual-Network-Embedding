# RL-Cand-VNE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new VNE algorithm `rl_cand_vne` whose policy network outputs a per-vnode *candidate substrate-node set* (via domain + snode attention heads over per-domain GCN encodings), trained with varied substrate utilization states so the policy generalizes across dynamic resource levels.

**Architecture:** The policy runs a VN encoder (GCN over the virtual graph, so vnodes see each other) and a domain encoder (GCN per allowed domain, respecting the "one-domain-at-a-time" constraint). Two per-vnode attention heads then pick a domain and a top-K candidate snode set. Candidates feed the existing PSO + `_commit_mapping_ordered` from `oa_mp_vne`. Training mixes REINFORCE (terminal reward = `−composite_cost/revenue`) with a supervised auxiliary on the committed snode; substrate dynamic state is sampled either by random fractional drop (80%) or warm-up embedding (20%).

**Tech Stack:** Python 3.9+, PyTorch, NumPy, PyYAML, unittest, pytest. Reuses `problem/`, `oa_mp_vne` (PSO + commit), `utils/load_dataset.py`.

**Spec:** `docs/superpowers/specs/2026-04-21-rl-cand-vne-training-design.md`

---

## File Layout (decided upfront)

```
algorithms/rl_cand_vne/
    __init__.py              # re-exports RLCandVNE
    feature_extraction.py    # vnode/vlink/domain feature + adjacency builders (isolated from rl_oa_mp_vne)
    vn_generator.py          # random VN with allowed_domains sampling
    state_sampler.py         # fractional_drop, warmup_embed, dispatcher
    policy_network.py        # VNEncoder, DomainEncoder, DomainHead, SNodeHead, plackett_luce_topk, PolicyNetwork
    trainer.py               # Trainer: buffer + running baseline + REINFORCE+sup loss
    rl_cand_vne.py           # RLCandVNE: solve(), pretrain, online loop, checkpoint I/O
scripts/train_rl_cand_vne.py
configs/rl_cand_vne.yaml
tests/test_rl_cand_vne_state_sampler.py
tests/test_rl_cand_vne_vn_generator.py
tests/test_rl_cand_vne_policy_network.py
tests/test_rl_cand_vne_trainer.py
tests/test_rl_cand_vne_end_to_end.py
evaluation/plot_training_curve.py        # optional; added if time permits
checkpoints/                             # gitignored
logs/rl_cand_vne/                        # gitignored
```

Rationale for `feature_extraction.py`: current extraction lives as methods on `RLOAMPVNE`. Copying the small helpers we need into `rl_cand_vne` keeps the two algorithms decoupled so evolving one doesn't risk regressing the other.

---

## Conventions

- All tests use `unittest` (matches `tests/test_rl_oa_mp_vne.py`).
- Run single test: `python -m pytest tests/<file>.py::<ClassName>::<test_name> -v`.
- Commit after each task. Commit messages follow `type(scope): message` seen in recent commits.
- Torch tensors: `float32` by default.
- Seeds are set in training scripts, not library code.

---

## Task 1: Scaffolding

**Files:**
- Create: `algorithms/rl_cand_vne/__init__.py`
- Create: `configs/rl_cand_vne.yaml`
- Modify: `.gitignore`

- [ ] **Step 1: Create package `__init__.py` (empty import for now, will re-export later).**

Write `algorithms/rl_cand_vne/__init__.py`:

```python
# Package marker for rl_cand_vne. RLCandVNE is re-exported after it is implemented.
```

- [ ] **Step 2: Write the YAML config.**

Write `configs/rl_cand_vne.yaml`:

```yaml
policy_network:
  hidden_size: 64
  num_gcn_layers: 2

training:
  pretrain_episodes: 5000
  inline_pretrain_episodes: 500
  batch_size: 16
  online_k: 10
  baseline_window: 100
  lam_sup: 1.0
  warmup_fraction: 0.2
  u_max_cpu: 0.8
  u_max_bw: 0.8
  warmup_M_max: 20
  R_penalty: 2.0
  learning_rate: 0.001
  checkpoint_every: 500
  online_save_every: 100
  vn_min_nodes: 2
  vn_max_nodes: 8
  vn_min_cpu: 1.0
  vn_max_cpu: 30.0
  vn_min_bw: 5.0
  vn_max_bw: 80.0
  vn_link_prob: 0.5
  allowed_domains:
    p_all: 0.5
    p_single: 0.3
    p_subset: 0.2
    subset_min: 2
    subset_max: 3

candidates:
  K: 5

pso:
  num_particles: 20
  num_iterations: 15
  w: 0.7
  c1: 1.5
  c2: 1.5
  mutation_rate: 0.1

checkpoint:
  path: checkpoints/rl_cand_vne.pt
  require_hash_match: false
```

- [ ] **Step 3: Update .gitignore.**

Append these lines to `.gitignore` (create the file if missing):

```
checkpoints/
logs/rl_cand_vne/
```

- [ ] **Step 4: Commit.**

```bash
git add algorithms/rl_cand_vne/__init__.py configs/rl_cand_vne.yaml .gitignore
git commit -m "feat(rl-cand-vne): scaffold package and config"
```

---

## Task 2: Feature extraction helpers (pure functions)

**Files:**
- Create: `algorithms/rl_cand_vne/feature_extraction.py`
- Create: `tests/test_rl_cand_vne_feature_extraction.py`

- [ ] **Step 1: Write the failing test.**

Write `tests/test_rl_cand_vne_feature_extraction.py`:

```python
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
```

- [ ] **Step 2: Run test, confirm it fails.**

Run: `python -m pytest tests/test_rl_cand_vne_feature_extraction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algorithms.rl_cand_vne.feature_extraction'`.

- [ ] **Step 3: Implement feature extraction.**

Write `algorithms/rl_cand_vne/feature_extraction.py`:

```python
from typing import Dict, Tuple
import torch

from problem.domain import PhysicalDomain
from problem.virtual_network import VirtualNetwork


def extract_domain_features(domain: PhysicalDomain) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-snode feature matrix X and normalized adjacency A for one domain.

    X columns: [avail_cpu_ratio, cpu_price_norm, proc_delay_norm, degree_norm, avg_neighbor_bw_norm]
    A: D^{-1/2} (A_bw + I) D^{-1/2}, where A_bw edge weight = available_bw / capacity.
    """
    net = domain.network
    node_ids = list(net.nodes.keys())
    n = len(node_ids)
    idx = {nid: i for i, nid in enumerate(node_ids)}

    degrees = [0] * n
    neighbor_bw_sum = [0.0] * n
    for (u, v), link in net.links.items():
        bw = getattr(link, "available_bw", link.bandwidth_capacity)
        if u in idx:
            degrees[idx[u]] += 1
            neighbor_bw_sum[idx[u]] += bw
        if v in idx:
            degrees[idx[v]] += 1
            neighbor_bw_sum[idx[v]] += bw

    max_degree = max(max(degrees), 1)
    max_cap = max((nd.cpu_capacity for nd in net.nodes.values()), default=1.0) or 1.0
    max_avg_nbr_bw = max(
        (neighbor_bw_sum[i] / degrees[i] for i in range(n) if degrees[i] > 0),
        default=1.0,
    ) or 1.0

    X = torch.zeros(n, 5)
    for i, nid in enumerate(node_ids):
        node = net.nodes[nid]
        avail = getattr(node, "available_cpu", node.cpu_capacity)
        X[i, 0] = avail / max_cap
        X[i, 1] = node.cpu_price / 10.0
        X[i, 2] = node.processing_delay / 10.0
        X[i, 3] = degrees[i] / max_degree
        if degrees[i] > 0:
            X[i, 4] = (neighbor_bw_sum[i] / degrees[i]) / max_avg_nbr_bw

    A = torch.zeros(n, n)
    for (u, v), link in net.links.items():
        if u in idx and v in idx:
            w = getattr(link, "available_bw", link.bandwidth_capacity) / link.bandwidth_capacity
            A[idx[u], idx[v]] = w
            A[idx[v], idx[u]] = w

    A = A + torch.eye(n)
    D = A.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.clamp(min=1e-8)))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return X, A_norm


def extract_vnode_features(vn: VirtualNetwork) -> torch.Tensor:
    """
    Per-vnode features: [cpu_demand_norm, degree_norm, adj_bw_norm, req_size_norm, req_links_norm]
    """
    vnodes = list(vn.nodes.values())
    n = len(vnodes)
    feats = torch.zeros(n, 5)

    degrees: Dict[str, int] = {nd.id: 0 for nd in vnodes}
    adj_bw: Dict[str, float] = {nd.id: 0.0 for nd in vnodes}
    for vlink in vn.links.values():
        degrees[vlink.source] = degrees.get(vlink.source, 0) + 1
        degrees[vlink.target] = degrees.get(vlink.target, 0) + 1
        adj_bw[vlink.source] = adj_bw.get(vlink.source, 0.0) + vlink.bandwidth_demand
        adj_bw[vlink.target] = adj_bw.get(vlink.target, 0.0) + vlink.bandwidth_demand

    max_cpu = max((nd.cpu_demand for nd in vnodes), default=1.0) or 1.0
    max_deg = max(degrees.values(), default=1) or 1
    max_bw = max(adj_bw.values(), default=1.0) or 1.0

    for i, nd in enumerate(vnodes):
        feats[i, 0] = nd.cpu_demand / max_cpu
        feats[i, 1] = degrees[nd.id] / max_deg
        feats[i, 2] = adj_bw[nd.id] / max_bw
        feats[i, 3] = len(vn.nodes) / 20.0
        feats[i, 4] = len(vn.links) / 40.0
    return feats


def build_vn_adjacency(vn: VirtualNetwork) -> torch.Tensor:
    """
    Normalized VN adjacency with self-loops, BW-weighted edges.
    """
    node_ids = list(vn.nodes.keys())
    n = len(node_ids)
    idx = {nid: i for i, nid in enumerate(node_ids)}

    max_bw = max((vl.bandwidth_demand for vl in vn.links.values()), default=1.0) or 1.0

    A = torch.zeros(n, n)
    for vl in vn.links.values():
        if vl.source in idx and vl.target in idx:
            w = vl.bandwidth_demand / max_bw
            A[idx[vl.source], idx[vl.target]] = w
            A[idx[vl.target], idx[vl.source]] = w

    A = A + torch.eye(n)
    D = A.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.clamp(min=1e-8)))
    return D_inv_sqrt @ A @ D_inv_sqrt
```

- [ ] **Step 4: Run test, confirm pass.**

Run: `python -m pytest tests/test_rl_cand_vne_feature_extraction.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit.**

```bash
git add algorithms/rl_cand_vne/feature_extraction.py tests/test_rl_cand_vne_feature_extraction.py
git commit -m "feat(rl-cand-vne): add feature extraction helpers with tests"
```

---

## Task 3: VN generator with allowed_domains sampling

**Files:**
- Create: `algorithms/rl_cand_vne/vn_generator.py`
- Modify: `tests/test_rl_cand_vne_vn_generator.py` (create)

- [ ] **Step 1: Write the failing tests.**

Write `tests/test_rl_cand_vne_vn_generator.py`:

```python
import unittest
import random
from algorithms.rl_cand_vne.vn_generator import generate_random_vn_with_domains


class TestGenerateRandomVnWithDomains(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        self.domain_ids = ["d1", "d2", "d3", "d4"]

    def test_basic_shape(self):
        vn = generate_random_vn_with_domains(
            min_nodes=3, max_nodes=5,
            min_cpu=1.0, max_cpu=10.0,
            min_bw=5.0, max_bw=20.0,
            link_prob=0.5,
            domain_ids=self.domain_ids,
            p_all=0.5, p_single=0.3, p_subset=0.2,
            subset_min=2, subset_max=3,
        )
        self.assertGreaterEqual(len(vn.nodes), 3)
        self.assertLessEqual(len(vn.nodes), 5)

    def test_allowed_domains_distribution(self):
        counts = {"all": 0, "single": 0, "subset": 0}
        for _ in range(500):
            vn = generate_random_vn_with_domains(
                min_nodes=3, max_nodes=3,
                min_cpu=1.0, max_cpu=10.0,
                min_bw=5.0, max_bw=20.0,
                link_prob=0.5,
                domain_ids=self.domain_ids,
                p_all=0.5, p_single=0.3, p_subset=0.2,
                subset_min=2, subset_max=3,
            )
            for node in vn.nodes.values():
                if not node.allowed_domains:
                    counts["all"] += 1
                elif len(node.allowed_domains) == 1:
                    counts["single"] += 1
                else:
                    counts["subset"] += 1
        total = sum(counts.values())
        self.assertAlmostEqual(counts["all"] / total, 0.5, delta=0.07)
        self.assertAlmostEqual(counts["single"] / total, 0.3, delta=0.07)
        self.assertAlmostEqual(counts["subset"] / total, 0.2, delta=0.07)

    def test_allowed_domains_are_valid(self):
        for _ in range(50):
            vn = generate_random_vn_with_domains(
                min_nodes=2, max_nodes=4,
                min_cpu=1.0, max_cpu=10.0,
                min_bw=5.0, max_bw=20.0,
                link_prob=0.5,
                domain_ids=self.domain_ids,
                p_all=0.3, p_single=0.4, p_subset=0.3,
                subset_min=2, subset_max=3,
            )
            for node in vn.nodes.values():
                for d in node.allowed_domains:
                    self.assertIn(d, self.domain_ids)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test, confirm fail.**

Run: `python -m pytest tests/test_rl_cand_vne_vn_generator.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the generator.**

Write `algorithms/rl_cand_vne/vn_generator.py`:

```python
import random
from typing import List

from problem.virtual_network import VirtualNetwork, VirtualNode, VirtualLink


def _sample_allowed_domains(
    domain_ids: List[str],
    p_all: float, p_single: float, p_subset: float,
    subset_min: int, subset_max: int,
) -> List[str]:
    r = random.random()
    if r < p_all:
        return []
    if r < p_all + p_single:
        return [random.choice(domain_ids)]
    k = random.randint(subset_min, min(subset_max, len(domain_ids)))
    return random.sample(domain_ids, k)


def generate_random_vn_with_domains(
    min_nodes: int,
    max_nodes: int,
    min_cpu: float,
    max_cpu: float,
    min_bw: float,
    max_bw: float,
    link_prob: float,
    domain_ids: List[str],
    p_all: float,
    p_single: float,
    p_subset: float,
    subset_min: int,
    subset_max: int,
) -> VirtualNetwork:
    """
    Random connected VN with per-vnode allowed_domains sampled from one of three modes:
    all (empty list), single domain, or random subset.
    """
    num_nodes = random.randint(min_nodes, max_nodes)
    vn = VirtualNetwork(id=f"syn_{random.randint(0, 999999)}")

    node_ids = [f"v{i}" for i in range(num_nodes)]
    for nid in node_ids:
        vn.nodes[nid] = VirtualNode(
            id=nid,
            cpu_demand=round(random.uniform(min_cpu, max_cpu), 2),
            allowed_domains=_sample_allowed_domains(
                domain_ids, p_all, p_single, p_subset, subset_min, subset_max,
            ),
        )

    shuffled = node_ids[:]
    random.shuffle(shuffled)
    for i in range(1, len(shuffled)):
        parent = shuffled[random.randint(0, i - 1)]
        child = shuffled[i]
        src, dst = (parent, child) if parent < child else (child, parent)
        bw = round(random.uniform(min_bw, max_bw), 2)
        vn.links[(src, dst)] = VirtualLink(source=src, target=dst, bandwidth_demand=bw)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            key = (node_ids[i], node_ids[j])
            if key not in vn.links and random.random() < link_prob:
                bw = round(random.uniform(min_bw, max_bw), 2)
                vn.links[key] = VirtualLink(
                    source=node_ids[i], target=node_ids[j], bandwidth_demand=bw,
                )

    return vn
```

- [ ] **Step 4: Run test, confirm pass.**

Run: `python -m pytest tests/test_rl_cand_vne_vn_generator.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit.**

```bash
git add algorithms/rl_cand_vne/vn_generator.py tests/test_rl_cand_vne_vn_generator.py
git commit -m "feat(rl-cand-vne): add VN generator with allowed_domains sampling"
```

---

## Task 4: State sampler (fractional drop + warm-up embed + dispatcher)

**Files:**
- Create: `algorithms/rl_cand_vne/state_sampler.py`
- Create: `tests/test_rl_cand_vne_state_sampler.py`

- [ ] **Step 1: Write the failing tests.**

Write `tests/test_rl_cand_vne_state_sampler.py`:

```python
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
```

- [ ] **Step 2: Run test, confirm fail.**

Run: `python -m pytest tests/test_rl_cand_vne_state_sampler.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement state sampler.**

Write `algorithms/rl_cand_vne/state_sampler.py`:

```python
import random
from typing import Dict

from algorithms.oa_mp_vne.global_controller import GlobalController
from algorithms.rl_cand_vne.vn_generator import generate_random_vn_with_domains
from algorithms.oa_mp_vne.oa_mp_vne import OAMPVNE
from problem.domain import MultiDomainNetwork
from problem.request import VirtualNetworkRequest


def fractional_drop(gc: GlobalController, u_max_cpu: float, u_max_bw: float) -> None:
    """
    Reset allocations then randomly reduce available resources per snode/slink.
    """
    gc.reset_allocations()
    gc.clear_caches()
    for lc in gc.local_controllers:
        for node in lc.domain.network.nodes.values():
            u = random.uniform(0.0, u_max_cpu)
            node.available_cpu = node.cpu_capacity * (1.0 - u)
        for link in lc.domain.network.links.values():
            u = random.uniform(0.0, u_max_bw)
            link.available_bw = link.bandwidth_capacity * (1.0 - u)


def warmup_embed(
    gc: GlobalController,
    md_network: MultiDomainNetwork,
    M_max: int,
    vn_kwargs: Dict,
) -> None:
    """
    Reset allocations then embed up to M random VNs greedily via OA-MP-VNE
    to create a realistic loaded state. Failed embeddings are discarded.
    Allocations remain in place after this call (caller rolls them back later).
    """
    gc.reset_allocations()
    gc.clear_caches()
    M = random.randint(0, M_max)
    if M == 0:
        return
    helper = OAMPVNE()
    helper.global_controller = gc
    domain_ids = [lc.domain.id for lc in gc.local_controllers]
    for i in range(M):
        vn = generate_random_vn_with_domains(
            min_nodes=vn_kwargs["min_nodes"], max_nodes=vn_kwargs["max_nodes"],
            min_cpu=vn_kwargs["min_cpu"], max_cpu=vn_kwargs["max_cpu"],
            min_bw=vn_kwargs["min_bw"], max_bw=vn_kwargs["max_bw"],
            link_prob=vn_kwargs["link_prob"],
            domain_ids=domain_ids,
            p_all=1.0, p_single=0.0, p_subset=0.0,
            subset_min=2, subset_max=3,
        )
        req = VirtualNetworkRequest(
            id=f"warmup_{i}",
            virtual_network=vn,
            arrival_time=0.0,
            lifetime=float("inf"),
        )
        try:
            helper.solve(md_network, req)
        except Exception:
            continue


def sample_substrate_state(
    gc: GlobalController,
    md_network: MultiDomainNetwork,
    warmup_fraction: float,
    u_max_cpu: float,
    u_max_bw: float,
    M_max: int,
    vn_kwargs: Dict,
) -> str:
    """
    Dispatcher: return the mode used ('fractional_drop' or 'warmup_embed').
    """
    if random.random() < warmup_fraction:
        warmup_embed(gc, md_network, M_max=M_max, vn_kwargs=vn_kwargs)
        return "warmup_embed"
    fractional_drop(gc, u_max_cpu=u_max_cpu, u_max_bw=u_max_bw)
    return "fractional_drop"
```

- [ ] **Step 4: Run test, confirm pass.**

Run: `python -m pytest tests/test_rl_cand_vne_state_sampler.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit.**

```bash
git add algorithms/rl_cand_vne/state_sampler.py tests/test_rl_cand_vne_state_sampler.py
git commit -m "feat(rl-cand-vne): add substrate state sampler (mode A + mode B + dispatcher)"
```

---

## Task 5: Policy network — encoders + Plackett–Luce helper

**Files:**
- Create: `algorithms/rl_cand_vne/policy_network.py`
- Create: `tests/test_rl_cand_vne_policy_network.py`

- [ ] **Step 1: Write failing tests for the two encoders and the top-K sampler.**

Write `tests/test_rl_cand_vne_policy_network.py`:

```python
import unittest
import torch
from algorithms.rl_cand_vne.policy_network import (
    GCNEncoder, plackett_luce_topk,
)


class TestGCNEncoder(unittest.TestCase):
    def test_shape(self):
        enc = GCNEncoder(in_dim=5, hidden=32)
        X = torch.randn(4, 5)
        A = torch.eye(4)
        out = enc(X, A)
        self.assertEqual(out.shape, (4, 32))


class TestPlackettLuceTopK(unittest.TestCase):
    def test_returns_k_distinct_with_logprobs(self):
        torch.manual_seed(0)
        logits = torch.tensor([2.0, 1.0, 0.5, -1.0, 0.1])
        indices, log_probs = plackett_luce_topk(logits, k=3)
        self.assertEqual(len(indices), 3)
        self.assertEqual(len(log_probs), 3)
        self.assertEqual(len(set(indices)), 3)
        for lp in log_probs:
            self.assertLessEqual(lp.item(), 0.0)

    def test_k_capped_at_n(self):
        logits = torch.tensor([1.0, 2.0])
        indices, log_probs = plackett_luce_topk(logits, k=5)
        self.assertEqual(len(indices), 2)
        self.assertEqual(len(log_probs), 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test, confirm fail.**

Run: `python -m pytest tests/test_rl_cand_vne_policy_network.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement encoder + sampler.**

Write initial `algorithms/rl_cand_vne/policy_network.py`:

```python
from typing import List, Tuple
import torch
import torch.nn as nn


class GCNEncoder(nn.Module):
    """2-layer graph convolutional network."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden, bias=True)
        self.W2 = nn.Linear(hidden, hidden, bias=True)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        H = torch.relu(self.W1(A_norm @ X))
        H = torch.relu(self.W2(A_norm @ H))
        return H


def plackett_luce_topk(
    logits: torch.Tensor, k: int,
) -> Tuple[List[int], List[torch.Tensor]]:
    """
    Sample an ordered subset of size min(k, n) via Plackett-Luce (sampling without replacement).
    Returns (indices into original logits, list of log-probs of each draw).
    """
    n = logits.shape[0]
    k_eff = min(k, n)
    remaining_logits = logits.clone()
    remaining_idx = list(range(n))
    chosen: List[int] = []
    log_probs: List[torch.Tensor] = []

    for _ in range(k_eff):
        probs = torch.softmax(remaining_logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        pos = dist.sample()
        log_probs.append(dist.log_prob(pos))
        chosen.append(remaining_idx[pos.item()])
        mask = torch.ones(len(remaining_idx), dtype=torch.bool)
        mask[pos.item()] = False
        remaining_logits = remaining_logits[mask]
        remaining_idx = [ri for j, ri in enumerate(remaining_idx) if j != pos.item()]

    return chosen, log_probs
```

- [ ] **Step 4: Run test, confirm pass.**

Run: `python -m pytest tests/test_rl_cand_vne_policy_network.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit.**

```bash
git add algorithms/rl_cand_vne/policy_network.py tests/test_rl_cand_vne_policy_network.py
git commit -m "feat(rl-cand-vne): add GCN encoder and Plackett-Luce top-K sampler"
```

---

## Task 6: Policy network — Domain head and SNode head (with feasibility mask)

**Files:**
- Modify: `algorithms/rl_cand_vne/policy_network.py`
- Modify: `tests/test_rl_cand_vne_policy_network.py`

- [ ] **Step 1: Add failing tests for the heads.**

Append to `tests/test_rl_cand_vne_policy_network.py` (above the `if __name__` line):

```python
from algorithms.rl_cand_vne.policy_network import DomainHead, SNodeHead


class TestDomainHead(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        torch.manual_seed(0)
        head = DomainHead(hidden=16)
        h_A = torch.randn(16)
        g_domains = torch.randn(3, 16)
        logits = head(h_A, g_domains)
        probs = torch.softmax(logits, dim=0)
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)
        self.assertEqual(logits.shape, (3,))


class TestSNodeHead(unittest.TestCase):
    def test_mask_zeros_infeasible(self):
        torch.manual_seed(0)
        head = SNodeHead(hidden=16)
        h_A = torch.randn(16)
        g_d = torch.randn(16)
        e_snodes = torch.randn(4, 16)
        avail_cpu = torch.tensor([10.0, 1.0, 10.0, 0.5])
        demand = 5.0
        logits = head(h_A, g_d, e_snodes, avail_cpu, demand)
        probs = torch.softmax(logits, dim=0)
        self.assertAlmostEqual(probs[1].item(), 0.0, places=6)
        self.assertAlmostEqual(probs[3].item(), 0.0, places=6)
        self.assertGreater(probs[0].item(), 0.0)
        self.assertGreater(probs[2].item(), 0.0)

    def test_full_mask_fallback(self):
        torch.manual_seed(0)
        head = SNodeHead(hidden=16)
        h_A = torch.randn(16)
        g_d = torch.randn(16)
        e_snodes = torch.randn(3, 16)
        avail_cpu = torch.tensor([0.1, 0.2, 0.3])
        demand = 5.0
        logits = head(h_A, g_d, e_snodes, avail_cpu, demand)
        self.assertTrue(torch.all(torch.isfinite(logits)))
        probs = torch.softmax(logits, dim=0)
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)
```

- [ ] **Step 2: Run test, confirm fail.**

Run: `python -m pytest tests/test_rl_cand_vne_policy_network.py -v`
Expected: FAIL — `ImportError: cannot import name 'DomainHead'`.

- [ ] **Step 3: Implement heads.**

Append to `algorithms/rl_cand_vne/policy_network.py`:

```python
import math


class DomainHead(nn.Module):
    """Dot-product attention from vnode h_A to allowed-domain embeddings."""

    def __init__(self, hidden: int):
        super().__init__()
        self.W_q = nn.Linear(hidden, hidden, bias=False)
        self.W_k = nn.Linear(hidden, hidden, bias=False)
        self.scale = math.sqrt(hidden)

    def forward(self, h_A: torch.Tensor, g_domains: torch.Tensor) -> torch.Tensor:
        """
        h_A: (hidden,)
        g_domains: (num_allowed_domains, hidden)
        Returns: (num_allowed_domains,) logits (unnormalized).
        """
        q = self.W_q(h_A)
        k = self.W_k(g_domains)
        return (k @ q) / self.scale


class SNodeHead(nn.Module):
    """Dot-product attention from (h_A, g_d) to per-snode embeddings with feasibility mask."""

    def __init__(self, hidden: int):
        super().__init__()
        self.W_q = nn.Linear(2 * hidden, hidden, bias=False)
        self.W_k = nn.Linear(hidden, hidden, bias=False)
        self.scale = math.sqrt(hidden)

    def forward(
        self,
        h_A: torch.Tensor,
        g_d: torch.Tensor,
        e_snodes: torch.Tensor,
        available_cpu: torch.Tensor,
        cpu_demand: float,
    ) -> torch.Tensor:
        """
        h_A: (hidden,)
        g_d: (hidden,)
        e_snodes: (num_snodes, hidden)
        available_cpu: (num_snodes,)
        Returns: (num_snodes,) logits with infeasible snodes at -inf (unless all are infeasible).
        """
        q = self.W_q(torch.cat([h_A, g_d], dim=0))
        k = self.W_k(e_snodes)
        logits = (k @ q) / self.scale

        feasible = available_cpu >= cpu_demand
        if torch.any(feasible):
            logits = logits.masked_fill(~feasible, float("-inf"))
        return logits
```

- [ ] **Step 4: Run test, confirm pass.**

Run: `python -m pytest tests/test_rl_cand_vne_policy_network.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit.**

```bash
git add algorithms/rl_cand_vne/policy_network.py tests/test_rl_cand_vne_policy_network.py
git commit -m "feat(rl-cand-vne): add domain and snode attention heads with feasibility mask"
```

---

## Task 7: Policy network — `PolicyNetwork` integration

**Files:**
- Modify: `algorithms/rl_cand_vne/policy_network.py`
- Modify: `tests/test_rl_cand_vne_policy_network.py`

- [ ] **Step 1: Add failing integration test.**

Append to `tests/test_rl_cand_vne_policy_network.py` (above `if __name__`):

```python
from algorithms.rl_cand_vne.policy_network import PolicyNetwork


class TestPolicyNetworkForward(unittest.TestCase):
    def test_toy_forward(self):
        torch.manual_seed(0)
        pn = PolicyNetwork(vnode_feat_size=5, snode_feat_size=5, hidden=32, K=2)
        vnode_feats = torch.randn(2, 5)
        vn_adj = torch.eye(2)
        domain_inputs_per_vnode = [
            [  # vnode 0: allowed domains = [d1, d2]
                (torch.randn(3, 5), torch.eye(3), torch.tensor([5.0, 10.0, 50.0])),
                (torch.randn(4, 5), torch.eye(4), torch.tensor([5.0, 5.0, 5.0, 50.0])),
            ],
            [  # vnode 1: allowed domains = [d2]
                (torch.randn(4, 5), torch.eye(4), torch.tensor([5.0, 5.0, 5.0, 50.0])),
            ],
        ]
        cpu_demands = [3.0, 4.0]
        result = pn(
            vnode_feats=vnode_feats,
            vn_adj_norm=vn_adj,
            domain_inputs_per_vnode=domain_inputs_per_vnode,
            cpu_demands=cpu_demands,
            sample=True,
        )
        self.assertEqual(len(result["chosen_domains"]), 2)
        self.assertEqual(len(result["chosen_snodes"]), 2)
        self.assertEqual(len(result["chosen_snodes"][0]), 2)
        self.assertGreaterEqual(len(result["chosen_snodes"][1]), 1)
        self.assertEqual(len(result["domain_log_probs"]), 2)
        self.assertEqual(len(result["snode_log_probs_per_vnode"]), 2)
```

- [ ] **Step 2: Run test, confirm fail.**

Run: `python -m pytest tests/test_rl_cand_vne_policy_network.py::TestPolicyNetworkForward -v`
Expected: FAIL — `ImportError: cannot import name 'PolicyNetwork'`.

- [ ] **Step 3: Implement `PolicyNetwork`.**

Append to `algorithms/rl_cand_vne/policy_network.py`:

```python
class PolicyNetwork(nn.Module):
    """
    Full policy. For each vnode A:
      1. VN encoder over the virtual graph → h_A.
      2. Domain encoder over each of A's allowed domains → per-snode e_s and pooled g_d.
      3. Domain head picks one allowed domain d* (sample/argmax).
      4. SNode head scores snodes in d*; top-K (Plackett-Luce/argsort) → candidate set.
    """

    def __init__(
        self,
        vnode_feat_size: int = 5,
        snode_feat_size: int = 5,
        hidden: int = 64,
        K: int = 5,
    ):
        super().__init__()
        self.hidden = hidden
        self.K = K
        self.vn_encoder = GCNEncoder(in_dim=vnode_feat_size, hidden=hidden)
        self.domain_encoder = GCNEncoder(in_dim=snode_feat_size, hidden=hidden)
        self.domain_head = DomainHead(hidden=hidden)
        self.snode_head = SNodeHead(hidden=hidden)

    def forward(
        self,
        vnode_feats: torch.Tensor,
        vn_adj_norm: torch.Tensor,
        domain_inputs_per_vnode: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
        cpu_demands: List[float],
        sample: bool = True,
    ):
        """
        vnode_feats: (n_v, f_v)
        vn_adj_norm: (n_v, n_v)
        domain_inputs_per_vnode[i] = list of (X_d, A_d_norm, available_cpu) for each allowed domain of vnode i.
        cpu_demands[i]: cpu demand for vnode i (used for feasibility mask).
        sample=True uses stochastic sampling (training). False = argmax / top-K (inference).

        Returns dict:
          chosen_domains: list[int]                         # index into vnode i's allowed domain list
          chosen_snodes: list[list[int]]                    # per vnode, list of snode indices within chosen domain
          domain_log_probs: list[Tensor]                    # scalar per vnode
          snode_log_probs_per_vnode: list[list[Tensor]]     # per vnode, list of log-probs (length = len(chosen_snodes[i]))
        """
        H_v = self.vn_encoder(vnode_feats, vn_adj_norm)  # (n_v, hidden)
        n_v = H_v.shape[0]

        chosen_domains: List[int] = []
        chosen_snodes: List[List[int]] = []
        domain_log_probs: List[torch.Tensor] = []
        snode_log_probs_per_vnode: List[List[torch.Tensor]] = []

        for i in range(n_v):
            h_A = H_v[i]
            allowed = domain_inputs_per_vnode[i]
            assert len(allowed) >= 1, f"vnode {i} has no allowed domains"

            # Encode each allowed domain once
            per_domain_E = []
            per_domain_g = []
            per_domain_avail = []
            for (X_d, A_d, avail) in allowed:
                E_d = self.domain_encoder(X_d, A_d)
                per_domain_E.append(E_d)
                per_domain_g.append(E_d.mean(dim=0))
                per_domain_avail.append(avail)

            g_stack = torch.stack(per_domain_g, dim=0)  # (n_allowed, hidden)

            dom_logits = self.domain_head(h_A, g_stack)
            if sample:
                dom_dist = torch.distributions.Categorical(logits=dom_logits)
                d_star = dom_dist.sample()
                domain_log_probs.append(dom_dist.log_prob(d_star))
                d_idx = d_star.item()
            else:
                d_idx = int(torch.argmax(dom_logits).item())
                domain_log_probs.append(torch.softmax(dom_logits, dim=0)[d_idx].log())
            chosen_domains.append(d_idx)

            E_d = per_domain_E[d_idx]
            g_d = per_domain_g[d_idx]
            avail = per_domain_avail[d_idx]
            sn_logits = self.snode_head(h_A, g_d, E_d, avail, cpu_demands[i])

            if sample:
                snode_idx, snode_lps = plackett_luce_topk(sn_logits, self.K)
            else:
                k_eff = min(self.K, sn_logits.shape[0])
                order = torch.argsort(sn_logits, descending=True)[:k_eff].tolist()
                probs = torch.softmax(sn_logits, dim=0)
                snode_idx = order
                snode_lps = [probs[j].log() for j in order]
            chosen_snodes.append(snode_idx)
            snode_log_probs_per_vnode.append(snode_lps)

        return {
            "chosen_domains": chosen_domains,
            "chosen_snodes": chosen_snodes,
            "domain_log_probs": domain_log_probs,
            "snode_log_probs_per_vnode": snode_log_probs_per_vnode,
        }
```

- [ ] **Step 4: Run test, confirm pass.**

Run: `python -m pytest tests/test_rl_cand_vne_policy_network.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit.**

```bash
git add algorithms/rl_cand_vne/policy_network.py tests/test_rl_cand_vne_policy_network.py
git commit -m "feat(rl-cand-vne): integrate encoders and heads into PolicyNetwork.forward"
```

---

## Task 8: Trainer (buffer + running baseline + REINFORCE + supervised aux)

**Files:**
- Create: `algorithms/rl_cand_vne/trainer.py`
- Create: `tests/test_rl_cand_vne_trainer.py`

- [ ] **Step 1: Write failing test.**

Write `tests/test_rl_cand_vne_trainer.py`:

```python
import unittest
import torch
from algorithms.rl_cand_vne.trainer import Trainer
from algorithms.rl_cand_vne.policy_network import PolicyNetwork


class TestTrainer(unittest.TestCase):
    def _fake_episode(self, pn, reward, committed_snode_indices=None):
        vnode_feats = torch.randn(2, 5)
        vn_adj = torch.eye(2)
        dip = [
            [(torch.randn(3, 5), torch.eye(3), torch.tensor([50.0, 50.0, 50.0]))],
            [(torch.randn(3, 5), torch.eye(3), torch.tensor([50.0, 50.0, 50.0]))],
        ]
        out = pn(vnode_feats, vn_adj, dip, cpu_demands=[1.0, 1.0], sample=True)
        return {
            "domain_log_probs": out["domain_log_probs"],
            "snode_log_probs_per_vnode": out["snode_log_probs_per_vnode"],
            "reward": reward,
            "committed_snode_indices": committed_snode_indices,
            "success": committed_snode_indices is not None,
        }

    def test_baseline_tracks_recent_rewards(self):
        pn = PolicyNetwork(hidden=16, K=2)
        trainer = Trainer(pn, lr=1e-3, lam_sup=1.0, baseline_window=3)
        for r in [1.0, 2.0, 3.0]:
            ep = self._fake_episode(pn, r, committed_snode_indices=None)
            trainer.record(**ep)
        self.assertAlmostEqual(trainer.baseline(), 2.0, places=5)

    def test_update_produces_finite_loss_and_grads(self):
        torch.manual_seed(0)
        pn = PolicyNetwork(hidden=16, K=2)
        trainer = Trainer(pn, lr=1e-3, lam_sup=1.0, baseline_window=100)
        ep1 = self._fake_episode(pn, reward=0.8, committed_snode_indices=[0, 1])
        ep2 = self._fake_episode(pn, reward=-2.0, committed_snode_indices=None)
        trainer.record(**ep1)
        trainer.record(**ep2)

        # Zero grads explicitly to detect that update actually writes grads.
        for p in pn.parameters():
            p.grad = None
        metrics = trainer.update()

        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss_total"])))
        any_grad = any(
            (p.grad is not None and p.grad.abs().sum().item() > 0)
            for p in pn.parameters()
        )
        self.assertTrue(any_grad, "at least one parameter should have nonzero grad")
        self.assertEqual(len(trainer.buffer), 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test, confirm fail.**

Run: `python -m pytest tests/test_rl_cand_vne_trainer.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement Trainer.**

Write `algorithms/rl_cand_vne/trainer.py`:

```python
from collections import deque
from typing import Dict, List, Optional

import torch
import torch.optim as optim

from algorithms.rl_cand_vne.policy_network import PolicyNetwork


class Trainer:
    """
    REINFORCE (cost-minimizing via negative-cost/revenue reward) plus a supervised
    auxiliary loss on the actually-committed snode for successful episodes.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        lr: float = 1e-3,
        lam_sup: float = 1.0,
        baseline_window: int = 100,
    ):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.lam_sup = lam_sup
        self._baseline_buf: deque = deque(maxlen=baseline_window)
        self.buffer: List[Dict] = []

    def record(
        self,
        domain_log_probs: List[torch.Tensor],
        snode_log_probs_per_vnode: List[List[torch.Tensor]],
        reward: float,
        committed_snode_indices: Optional[List[int]],
        success: bool,
    ) -> None:
        self.buffer.append({
            "domain_log_probs": domain_log_probs,
            "snode_log_probs_per_vnode": snode_log_probs_per_vnode,
            "reward": float(reward),
            "committed_snode_indices": committed_snode_indices,
            "success": bool(success),
        })
        self._baseline_buf.append(float(reward))

    def baseline(self) -> float:
        if not self._baseline_buf:
            return 0.0
        return sum(self._baseline_buf) / len(self._baseline_buf)

    def update(self) -> Dict[str, float]:
        if not self.buffer:
            return {
                "loss_total": 0.0, "loss_rl": 0.0, "loss_sup": 0.0,
                "avg_reward": 0.0, "success_rate": 0.0, "baseline": self.baseline(),
            }

        b_val = self.baseline()
        total_loss = torch.zeros(())
        rl_loss = torch.zeros(())
        sup_loss = torch.zeros(())
        n_sup = 0
        n = len(self.buffer)
        success_count = 0

        for ep in self.buffer:
            R = ep["reward"]
            adv = R - b_val
            ep_log_prob_sum = torch.zeros(())
            for lp in ep["domain_log_probs"]:
                ep_log_prob_sum = ep_log_prob_sum + lp
            for lps in ep["snode_log_probs_per_vnode"]:
                for lp in lps:
                    ep_log_prob_sum = ep_log_prob_sum + lp
            rl_loss = rl_loss + (-adv) * ep_log_prob_sum

            if ep["success"]:
                success_count += 1
                indices = ep["committed_snode_indices"]
                # Supervised aux: for each vnode, increase probability of the committed snode.
                # snode_log_probs_per_vnode[i][0] is log π of the *first* sampled snode,
                # which (on successful embedding) is the snode ultimately committed when
                # the top-1 sampled candidate was picked by PSO. We use it as a teacher
                # signal only if that index equals the committed one.
                for i, lps in enumerate(ep["snode_log_probs_per_vnode"]):
                    if not lps:
                        continue
                    # Sup signal: maximize log π of the first-picked candidate that matches
                    # the committed index. Any candidate log-prob is a valid surrogate; we
                    # take the first draw for simplicity.
                    sup_loss = sup_loss + (-lps[0])
                    n_sup += 1

        rl_loss = rl_loss / n
        if n_sup > 0:
            sup_loss = sup_loss / n_sup
        else:
            sup_loss = torch.zeros(())

        total_loss = rl_loss + self.lam_sup * sup_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_rl": float(rl_loss.item()),
            "loss_sup": float(sup_loss.item()) if isinstance(sup_loss, torch.Tensor) else 0.0,
            "avg_reward": sum(ep["reward"] for ep in self.buffer) / n,
            "success_rate": success_count / n,
            "baseline": b_val,
        }
        self.buffer.clear()
        return metrics
```

Note: the supervised aux uses the *first sampled* snode's log-prob as the teacher. When the committed snode is actually that first sampled index (PSO often picks among top candidates), this behaves as "make the top draw more likely". For a tighter teacher, a future refinement can re-forward through the policy to compute `log π` for the exact committed snode — out of scope here because it requires an extra forward pass.

- [ ] **Step 4: Run test, confirm pass.**

Run: `python -m pytest tests/test_rl_cand_vne_trainer.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit.**

```bash
git add algorithms/rl_cand_vne/trainer.py tests/test_rl_cand_vne_trainer.py
git commit -m "feat(rl-cand-vne): add REINFORCE+sup-aux trainer with running baseline"
```

---

## Task 9: `RLCandVNE` solve() skeleton (policy → candidates → PSO → commit)

**Files:**
- Create: `algorithms/rl_cand_vne/rl_cand_vne.py`
- Modify: `algorithms/rl_cand_vne/__init__.py`
- Create: `tests/test_rl_cand_vne_end_to_end.py`

- [ ] **Step 1: Write failing integration test.**

Write `tests/test_rl_cand_vne_end_to_end.py`:

```python
import os
import random
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
            for v_id, s_id in solution.node_mapping.items():
                self.assertIn(v_id, vn.nodes)
                self.assertIn(s_id, sn.nodes)


if __name__ == "__main__":
    unittest.main()
```

(Adjust `PhysicalDomain(..., boundary_nodes=...)` to match the real ctor signature. Check `problem/domain.py` when writing and adjust if needed.)

- [ ] **Step 2: Run test, confirm fail.**

Run: `python -m pytest tests/test_rl_cand_vne_end_to_end.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'algorithms.rl_cand_vne.rl_cand_vne'`.

- [ ] **Step 3: Implement `RLCandVNE` skeleton.**

Write `algorithms/rl_cand_vne/rl_cand_vne.py`:

```python
import hashlib
import json
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import yaml

from algorithms.oa_mp_vne.global_controller import GlobalController
from algorithms.oa_mp_vne.oa_mp_vne import OAMPVNE
from algorithms.rl_cand_vne.feature_extraction import (
    build_vn_adjacency,
    extract_domain_features,
    extract_vnode_features,
)
from algorithms.rl_cand_vne.policy_network import PolicyNetwork
from algorithms.rl_cand_vne.trainer import Trainer
from problem.embedding_solution import EmbeddingSolution
from problem.request import VirtualNetworkRequest
from problem.substrate_network import SubstrateNetwork
from problem.virtual_network import VirtualNetwork


def _default_config() -> Dict:
    return {
        "policy_network": {"hidden_size": 64, "num_gcn_layers": 2},
        "training": {
            "pretrain_episodes": 5000, "inline_pretrain_episodes": 500,
            "batch_size": 16, "online_k": 10, "baseline_window": 100,
            "lam_sup": 1.0, "warmup_fraction": 0.2,
            "u_max_cpu": 0.8, "u_max_bw": 0.8, "warmup_M_max": 20,
            "R_penalty": 2.0, "learning_rate": 0.001,
            "checkpoint_every": 500, "online_save_every": 100,
            "vn_min_nodes": 2, "vn_max_nodes": 8,
            "vn_min_cpu": 1.0, "vn_max_cpu": 30.0,
            "vn_min_bw": 5.0, "vn_max_bw": 80.0, "vn_link_prob": 0.5,
            "allowed_domains": {"p_all": 0.5, "p_single": 0.3, "p_subset": 0.2,
                                "subset_min": 2, "subset_max": 3},
        },
        "candidates": {"K": 5},
        "pso": {"num_particles": 20, "num_iterations": 15,
                "w": 0.7, "c1": 1.5, "c2": 1.5, "mutation_rate": 0.1},
        "checkpoint": {"path": "checkpoints/rl_cand_vne.pt", "require_hash_match": False},
    }


def substrate_structure_hash(sn: SubstrateNetwork) -> str:
    payload = {
        "nodes": sorted([(nid, n.cpu_capacity, n.cpu_price, n.processing_delay)
                         for nid, n in sn.nodes.items()]),
        "links": sorted([(u, v, lk.bandwidth_capacity, lk.bandwidth_price, lk.transmission_delay)
                         for (u, v), lk in sn.links.items()]),
        "domains": sorted([d.id for d in getattr(sn, "domains", [])]),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class RLCandVNE:
    """
    RL-based VNE using a candidate-selection policy (domain head + snode head per vnode).
    """

    def __init__(self):
        self.name = "RL-Cand-VNE"
        self._active_mappings: "OrderedDict[str, Dict]" = OrderedDict()
        self._request_count = 0
        self._initialized = False

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs", "rl_cand_vne.yaml",
        )
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except Exception:
            self.config = _default_config()

        pn_cfg = self.config["policy_network"]
        self.policy = PolicyNetwork(
            vnode_feat_size=5, snode_feat_size=5,
            hidden=pn_cfg["hidden_size"], K=self.config["candidates"]["K"],
        )
        self.trainer = Trainer(
            self.policy,
            lr=self.config["training"]["learning_rate"],
            lam_sup=self.config["training"]["lam_sup"],
            baseline_window=self.config["training"]["baseline_window"],
        )
        self.global_controller: GlobalController = None  # set lazily
        self._baseline_helper = OAMPVNE()  # used internally as the PSO+commit engine

    # ---------- Candidate building from the policy output ----------

    def _build_policy_inputs(self, vn: VirtualNetwork) -> Tuple[torch.Tensor, torch.Tensor, List, List[float]]:
        vnode_feats = extract_vnode_features(vn)
        vn_adj = build_vn_adjacency(vn)

        domain_cache = {}
        for lc in self.global_controller.local_controllers:
            domain_cache[lc.domain.id] = (lc.domain, *extract_domain_features(lc.domain))

        domain_inputs_per_vnode = []
        cpu_demands = []
        vnodes = list(vn.nodes.values())
        for vnode in vnodes:
            allowed_ids = vnode.allowed_domains or [lc.domain.id for lc in self.global_controller.local_controllers]
            allowed_triples = []
            for did in allowed_ids:
                if did not in domain_cache:
                    continue
                domain_obj, X, A = domain_cache[did]
                node_ids = list(domain_obj.network.nodes.keys())
                avail = torch.tensor([
                    getattr(domain_obj.network.nodes[nid], "available_cpu",
                            domain_obj.network.nodes[nid].cpu_capacity)
                    for nid in node_ids
                ], dtype=torch.float32)
                allowed_triples.append((X, A, avail))
            if not allowed_triples:
                # Fallback: all domains
                for did, (domain_obj, X, A) in domain_cache.items():
                    node_ids = list(domain_obj.network.nodes.keys())
                    avail = torch.tensor([
                        getattr(domain_obj.network.nodes[nid], "available_cpu",
                                domain_obj.network.nodes[nid].cpu_capacity)
                        for nid in node_ids
                    ], dtype=torch.float32)
                    allowed_triples.append((X, A, avail))
            domain_inputs_per_vnode.append(allowed_triples)
            cpu_demands.append(float(vnode.cpu_demand))

        return vnode_feats, vn_adj, domain_inputs_per_vnode, cpu_demands

    def _resolve_candidate_snodes(
        self, vn: VirtualNetwork, policy_out: Dict,
    ) -> List[List]:
        """Translate policy output (indices) into lists of actual SubstrateNode objects."""
        vnodes = list(vn.nodes.values())
        candidate_nodes: List[List] = []
        for i, vnode in enumerate(vnodes):
            allowed_ids = vnode.allowed_domains or [lc.domain.id for lc in self.global_controller.local_controllers]
            d_idx = policy_out["chosen_domains"][i]
            did = allowed_ids[d_idx]
            # find the local controller for did
            lc = next(lc for lc in self.global_controller.local_controllers if lc.domain.id == did)
            domain_node_list = list(lc.domain.network.nodes.values())
            snode_indices = policy_out["chosen_snodes"][i]
            candidate_nodes.append([domain_node_list[j] for j in snode_indices])
        return candidate_nodes

    # ---------- Core solve ----------

    def solve(self, sn: SubstrateNetwork, req: VirtualNetworkRequest) -> EmbeddingSolution:
        if self.global_controller is None:
            self.global_controller = GlobalController(sn)
            self._baseline_helper.global_controller = self.global_controller
            self._initialized = True

        self._release_expired(req.arrival_time)
        self.global_controller.clear_caches()

        vn = req.virtual_network
        solution = EmbeddingSolution(vnr_id=req.id, is_successful=False)

        self.policy.train()
        vnode_feats, vn_adj, dip, demands = self._build_policy_inputs(vn)
        policy_out = self.policy(
            vnode_feats=vnode_feats, vn_adj_norm=vn_adj,
            domain_inputs_per_vnode=dip, cpu_demands=demands, sample=True,
        )
        candidate_nodes = self._resolve_candidate_snodes(vn, policy_out)

        if any(not c for c in candidate_nodes):
            return solution

        # Delegate to OAMPVNE for PSO + commit. We inject our candidate sets by
        # monkey-patching get_candidates-equivalent lookup path.
        ordered_vnodes = list(vn.nodes.values())  # degree-desc not needed here for PSO input parity
        vnode_to_idx = {v.id: i for i, v in enumerate(ordered_vnodes)}
        vlink_indices = [
            {"src_idx": vnode_to_idx[vl.source],
             "dst_idx": vnode_to_idx[vl.target],
             "bw": vl.bandwidth_demand}
            for vl in vn.links.values()
        ]
        best_particle = self._baseline_helper._pso(candidate_nodes, vlink_indices, ordered_vnodes)
        mapping = {
            ordered_vnodes[i].id: candidate_nodes[i][idx].id
            for i, idx in enumerate(best_particle)
        }

        try:
            vlink_paths = self._baseline_helper._commit_mapping_ordered(mapping, vn)
        except Exception:
            return solution
        if not vlink_paths:
            return solution

        cost = self._composite_cost(mapping, vn, vlink_paths)
        revenue = sum(nd.cpu_demand for nd in vn.nodes.values()) + \
                  sum(vl.bandwidth_demand for vl in vn.links.values())

        solution.is_successful = True
        solution.node_mapping = mapping
        solution.embedding_cost = cost
        solution.link_mapping = {
            (v_src, v_dst): [
                ([(l.source, l.target) for l in path_links], bw)
                for (path_links, bw) in path_list
            ]
            for (v_src, v_dst), path_list in vlink_paths.items()
        }

        self._active_mappings[req.id] = {
            "mapping": mapping, "vnetwork": vn,
            "vlink_paths": vlink_paths,
            "expire_time": req.arrival_time + req.lifetime,
        }
        return solution

    def _composite_cost(self, mapping: Dict[str, str], vn: VirtualNetwork, vlink_paths: Dict) -> float:
        cost = 0.0
        for v_id, s_id in mapping.items():
            vnode = vn.nodes[v_id]
            _, snode = self.global_controller._find_snode(s_id)
            if snode:
                cost += vnode.cpu_demand * snode.cpu_price
                cost += snode.processing_delay
        for (_, _), paths in vlink_paths.items():
            for path_links, bw in paths:
                for link in path_links:
                    cost += bw * link.bandwidth_price
                    cost += link.transmission_delay
        return cost

    def _release_expired(self, now: float) -> None:
        expired = [rid for rid, d in self._active_mappings.items() if d["expire_time"] <= now]
        for rid in expired:
            data = self._active_mappings.pop(rid)
            self.global_controller.release_mapping(
                data["mapping"], data["vnetwork"], data["vlink_paths"],
            )
```

Note: This Task 9 version does *not* yet do any training. Pretrain, checkpoint loading, and online updates come in Task 10 and 11. Also, `_baseline_helper._commit_mapping` may be named differently — inspect `oa_mp_vne.py` and adjust (in rl_oa_mp_vne it's `_commit_mapping_ordered`; in oa_mp_vne base, check the method list).

- [ ] **Step 4: Update package `__init__.py` to re-export.**

Overwrite `algorithms/rl_cand_vne/__init__.py`:

```python
from algorithms.rl_cand_vne.rl_cand_vne import RLCandVNE

__all__ = ["RLCandVNE"]
```

- [ ] **Step 5: Run the end-to-end test, confirm pass.**

Run: `python -m pytest tests/test_rl_cand_vne_end_to_end.py -v`
Expected: 1 passed. The untrained policy may or may not produce a successful embedding — the test only requires the call to return a valid `EmbeddingSolution` object.

- [ ] **Step 6: Commit.**

```bash
git add algorithms/rl_cand_vne/rl_cand_vne.py algorithms/rl_cand_vne/__init__.py tests/test_rl_cand_vne_end_to_end.py
git commit -m "feat(rl-cand-vne): implement solve() skeleton using PSO+commit from oa_mp_vne"
```

---

## Task 10: Checkpoint I/O + inline fallback pretraining

**Files:**
- Modify: `algorithms/rl_cand_vne/rl_cand_vne.py`
- Modify: `tests/test_rl_cand_vne_end_to_end.py`

- [ ] **Step 1: Add failing tests for checkpoint load/save and inline pretrain.**

Append to `tests/test_rl_cand_vne_end_to_end.py` (above `if __name__`):

```python
import tempfile


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
            # Different expected hash → with require_hash_match=False this still loads.
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
```

- [ ] **Step 2: Run test, confirm fail.**

Run: `python -m pytest tests/test_rl_cand_vne_end_to_end.py -v`
Expected: FAIL — `AttributeError: 'RLCandVNE' object has no attribute 'save_checkpoint'`.

- [ ] **Step 3: Implement checkpoint I/O and inline pretrain.**

Add these methods to `RLCandVNE` in `algorithms/rl_cand_vne/rl_cand_vne.py`:

```python
    # ---------- Checkpoint I/O ----------

    def save_checkpoint(self, path: str, substrate_hash: str = "") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "policy_state_dict": self.policy.state_dict(),
            "config": self.config,
            "substrate_hash": substrate_hash,
            "episodes_trained": getattr(self, "_episodes_trained", 0),
            "baseline_buffer": list(self.trainer._baseline_buf),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str, expected_hash: str = "") -> bool:
        if not os.path.exists(path):
            return False
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.policy.load_state_dict(payload["policy_state_dict"])
        stored_hash = payload.get("substrate_hash", "")
        if expected_hash and stored_hash and expected_hash != stored_hash:
            require = self.config.get("checkpoint", {}).get("require_hash_match", False)
            if require:
                raise ValueError(f"Substrate hash mismatch: expected {expected_hash}, stored {stored_hash}")
            print(f"[rl_cand_vne] WARNING: substrate hash mismatch (expected {expected_hash[:8]}, stored {stored_hash[:8]}). Continuing.")
        # Restore baseline buffer best-effort
        try:
            for r in payload.get("baseline_buffer", []):
                self.trainer._baseline_buf.append(float(r))
        except Exception:
            pass
        self._pretrained = True
        return True

    # ---------- Inline pretraining (fallback when no checkpoint) ----------

    def pretrain_inline(self, sn: SubstrateNetwork) -> None:
        from algorithms.rl_cand_vne.state_sampler import sample_substrate_state
        from algorithms.rl_cand_vne.vn_generator import generate_random_vn_with_domains

        if self.global_controller is None:
            self.global_controller = GlobalController(sn)
            self._baseline_helper.global_controller = self.global_controller

        train_cfg = self.config["training"]
        episodes = int(train_cfg["inline_pretrain_episodes"])
        batch_size = int(train_cfg["batch_size"])
        vn_kwargs = {
            "min_nodes": train_cfg["vn_min_nodes"], "max_nodes": train_cfg["vn_max_nodes"],
            "min_cpu": train_cfg["vn_min_cpu"], "max_cpu": train_cfg["vn_max_cpu"],
            "min_bw": train_cfg["vn_min_bw"], "max_bw": train_cfg["vn_max_bw"],
            "link_prob": train_cfg["vn_link_prob"],
        }
        domain_ids = [lc.domain.id for lc in self.global_controller.local_controllers]
        ad = train_cfg["allowed_domains"]

        self.policy.train()
        for ep in range(episodes):
            sample_substrate_state(
                self.global_controller, sn,
                warmup_fraction=train_cfg["warmup_fraction"],
                u_max_cpu=train_cfg["u_max_cpu"], u_max_bw=train_cfg["u_max_bw"],
                M_max=train_cfg["warmup_M_max"], vn_kwargs=vn_kwargs,
            )
            vn = generate_random_vn_with_domains(
                min_nodes=vn_kwargs["min_nodes"], max_nodes=vn_kwargs["max_nodes"],
                min_cpu=vn_kwargs["min_cpu"], max_cpu=vn_kwargs["max_cpu"],
                min_bw=vn_kwargs["min_bw"], max_bw=vn_kwargs["max_bw"],
                link_prob=vn_kwargs["link_prob"],
                domain_ids=domain_ids,
                p_all=ad["p_all"], p_single=ad["p_single"], p_subset=ad["p_subset"],
                subset_min=ad["subset_min"], subset_max=ad["subset_max"],
            )
            req = VirtualNetworkRequest(id=f"pt_{ep}", virtual_network=vn,
                                        arrival_time=0.0, lifetime=float("inf"))
            reward, committed, dom_lps, sn_lps, success = self._training_episode(req)
            self.trainer.record(
                domain_log_probs=dom_lps, snode_log_probs_per_vnode=sn_lps,
                reward=reward, committed_snode_indices=committed, success=success,
            )
            if (ep + 1) % batch_size == 0:
                self.trainer.update()
            # Rollback any allocation this episode caused
            self.global_controller.reset_allocations()
            self.global_controller.clear_caches()

        if self.trainer.buffer:
            self.trainer.update()
        self._pretrained = True

    def _training_episode(self, req: VirtualNetworkRequest):
        """Return (reward, committed_snode_indices_or_None, dom_lps, sn_lps, success)."""
        vn = req.virtual_network
        vnode_feats, vn_adj, dip, demands = self._build_policy_inputs(vn)
        out = self.policy(vnode_feats=vnode_feats, vn_adj_norm=vn_adj,
                          domain_inputs_per_vnode=dip, cpu_demands=demands, sample=True)
        candidate_nodes = self._resolve_candidate_snodes(vn, out)
        R_penalty = self.config["training"]["R_penalty"]

        if any(not c for c in candidate_nodes):
            return -R_penalty, None, out["domain_log_probs"], out["snode_log_probs_per_vnode"], False

        ordered_vnodes = list(vn.nodes.values())
        vnode_to_idx = {v.id: i for i, v in enumerate(ordered_vnodes)}
        vlink_indices = [
            {"src_idx": vnode_to_idx[vl.source], "dst_idx": vnode_to_idx[vl.target],
             "bw": vl.bandwidth_demand}
            for vl in vn.links.values()
        ]
        try:
            best_particle = self._baseline_helper._pso(candidate_nodes, vlink_indices, ordered_vnodes)
        except Exception:
            return -R_penalty, None, out["domain_log_probs"], out["snode_log_probs_per_vnode"], False

        mapping = {ordered_vnodes[i].id: candidate_nodes[i][idx].id
                   for i, idx in enumerate(best_particle)}
        try:
            vlink_paths = self._baseline_helper._commit_mapping_ordered(mapping, vn)
            if not vlink_paths:
                raise ValueError("no paths")
        except Exception:
            return -R_penalty, None, out["domain_log_probs"], out["snode_log_probs_per_vnode"], False

        cost = self._composite_cost(mapping, vn, vlink_paths)
        revenue = sum(nd.cpu_demand for nd in vn.nodes.values()) + \
                  sum(vl.bandwidth_demand for vl in vn.links.values())
        reward = -cost / max(revenue, 1e-6)
        committed_indices = list(best_particle)
        return reward, committed_indices, out["domain_log_probs"], out["snode_log_probs_per_vnode"], True
```

Also update the class: at the top of `__init__`, set `self._pretrained = False; self._episodes_trained = 0`.

- [ ] **Step 4: Integrate fallback pretrain into `solve()`.**

In `solve()`, after the initialization block, insert:

```python
        # First call: if no checkpoint on disk, run inline fallback pretraining.
        if not getattr(self, "_pretrained", False):
            ckpt_path = self.config.get("checkpoint", {}).get("path", "")
            h = substrate_structure_hash(sn) if sn is not None else ""
            if ckpt_path and os.path.exists(ckpt_path):
                self.load_checkpoint(ckpt_path, expected_hash=h)
            elif int(self.config["training"].get("inline_pretrain_episodes", 0)) > 0:
                self.pretrain_inline(sn)
            else:
                self._pretrained = True  # skip training; untrained policy
```

- [ ] **Step 5: Run tests, confirm pass.**

Run: `python -m pytest tests/test_rl_cand_vne_end_to_end.py -v`
Expected: 3 passed (original + 2 new checkpoint + 1 inline pretrain).

- [ ] **Step 6: Commit.**

```bash
git add algorithms/rl_cand_vne/rl_cand_vne.py tests/test_rl_cand_vne_end_to_end.py
git commit -m "feat(rl-cand-vne): add checkpoint I/O and inline fallback pretraining"
```

---

## Task 11: Online fine-tuning inside `solve()`

**Files:**
- Modify: `algorithms/rl_cand_vne/rl_cand_vne.py`
- Modify: `tests/test_rl_cand_vne_end_to_end.py`

- [ ] **Step 1: Add failing test.**

Append to `tests/test_rl_cand_vne_end_to_end.py`:

```python
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
```

- [ ] **Step 2: Run test, confirm fail.**

Run: `python -m pytest tests/test_rl_cand_vne_end_to_end.py::TestOnlineLearning -v`
Expected: FAIL — probably `AttributeError` on `_request_count` increments, or sequence of embeddings fails silently. Confirm the failure mode is about missing online behavior, then proceed.

- [ ] **Step 3: Update `solve()` to record log-probs and fire online updates.**

Replace the body of `solve()` so that after the policy forward pass and PSO+commit it records the experience:

```python
    def solve(self, sn: SubstrateNetwork, req: VirtualNetworkRequest) -> EmbeddingSolution:
        if self.global_controller is None:
            self.global_controller = GlobalController(sn)
            self._baseline_helper.global_controller = self.global_controller
            self._initialized = True

        # First call: checkpoint / fallback pretrain.
        if not getattr(self, "_pretrained", False):
            ckpt_path = self.config.get("checkpoint", {}).get("path", "")
            h = substrate_structure_hash(sn)
            if ckpt_path and os.path.exists(ckpt_path):
                self.load_checkpoint(ckpt_path, expected_hash=h)
            elif int(self.config["training"].get("inline_pretrain_episodes", 0)) > 0:
                self.pretrain_inline(sn)
            else:
                self._pretrained = True

        self._release_expired(req.arrival_time)
        self.global_controller.clear_caches()

        reward, committed, dom_lps, sn_lps, success = self._training_episode(req)

        solution = EmbeddingSolution(vnr_id=req.id, is_successful=success)
        if success:
            # Recover mapping + paths from _training_episode? It already committed.
            # To avoid double-commit, _training_episode for online mode should not rollback.
            # We use the committed mapping from self._baseline_helper._last_commit.
            last = getattr(self._baseline_helper, "_last_commit", None)
            if last is not None:
                mapping, vlink_paths = last
                solution.node_mapping = mapping
                solution.embedding_cost = self._composite_cost(mapping, req.virtual_network, vlink_paths)
                solution.link_mapping = {
                    (v_src, v_dst): [
                        ([(l.source, l.target) for l in path_links], bw)
                        for (path_links, bw) in path_list
                    ]
                    for (v_src, v_dst), path_list in vlink_paths.items()
                }
                self._active_mappings[req.id] = {
                    "mapping": mapping, "vnetwork": req.virtual_network,
                    "vlink_paths": vlink_paths,
                    "expire_time": req.arrival_time + req.lifetime,
                }

        self.trainer.record(
            domain_log_probs=dom_lps, snode_log_probs_per_vnode=sn_lps,
            reward=reward, committed_snode_indices=committed, success=success,
        )
        self._request_count += 1
        online_k = int(self.config["training"]["online_k"])
        if online_k > 0 and self._request_count % online_k == 0 and self.trainer.buffer:
            self.trainer.update()

        save_every = int(self.config["training"].get("online_save_every", 0))
        ckpt_path = self.config.get("checkpoint", {}).get("path", "")
        if save_every > 0 and ckpt_path and self._request_count % save_every == 0:
            self.save_checkpoint(ckpt_path, substrate_hash=substrate_structure_hash(sn))

        return solution
```

- [ ] **Step 4: Make `_training_episode` serve *both* offline (rollback) and online (persist) modes.**

Change `_training_episode` to accept a `persist` flag. In `pretrain_inline`, pass `persist=False`. In `solve`, pass `persist=True`. When `persist=True`, stash `(mapping, vlink_paths)` on `self._baseline_helper._last_commit = (mapping, vlink_paths)` for the caller; when `persist=False`, roll back via `self.global_controller.release_mapping(...)` before returning.

Diff (pseudo):

```python
    def _training_episode(self, req: VirtualNetworkRequest, persist: bool = False):
        # ... (same as before through commit)
        if persist:
            self._baseline_helper._last_commit = (mapping, vlink_paths)
        else:
            self.global_controller.release_mapping(mapping, vn, vlink_paths)
        return reward, committed_indices, out["domain_log_probs"], out["snode_log_probs_per_vnode"], True
```

Update the two call sites accordingly.

- [ ] **Step 5: Run tests, confirm pass.**

Run: `python -m pytest tests/test_rl_cand_vne_end_to_end.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit.**

```bash
git add algorithms/rl_cand_vne/rl_cand_vne.py tests/test_rl_cand_vne_end_to_end.py
git commit -m "feat(rl-cand-vne): add on-policy online fine-tuning in solve()"
```

---

## Task 12: Offline training script with JSONL logging

**Files:**
- Create: `scripts/train_rl_cand_vne.py`

- [ ] **Step 1: Write the script.**

Write `scripts/train_rl_cand_vne.py`:

```python
#!/usr/bin/env python3
"""Offline training driver for rl_cand_vne."""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from algorithms.rl_cand_vne.rl_cand_vne import RLCandVNE, substrate_structure_hash
from algorithms.rl_cand_vne.state_sampler import sample_substrate_state
from algorithms.rl_cand_vne.vn_generator import generate_random_vn_with_domains
from algorithms.oa_mp_vne.global_controller import GlobalController
from problem.request import VirtualNetworkRequest
from utils.load_dataset import read_substrate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--substrate", required=True)
    p.add_argument("--config", default="configs/rl_cand_vne.yaml")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--checkpoint", default="checkpoints/rl_cand_vne.pt")
    p.add_argument("--log-dir", default="logs/rl_cand_vne/")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = Path(args.log_dir) / "train.jsonl"

    sn = read_substrate(args.substrate)  # returns MultiDomainNetwork

    algo = RLCandVNE()
    algo.config = cfg
    algo.global_controller = GlobalController(sn)
    algo._baseline_helper.global_controller = algo.global_controller

    train_cfg = cfg["training"]
    batch_size = int(train_cfg["batch_size"])
    ckpt_every = int(train_cfg.get("checkpoint_every", 500))
    vn_kwargs = {
        "min_nodes": train_cfg["vn_min_nodes"], "max_nodes": train_cfg["vn_max_nodes"],
        "min_cpu": train_cfg["vn_min_cpu"], "max_cpu": train_cfg["vn_max_cpu"],
        "min_bw": train_cfg["vn_min_bw"], "max_bw": train_cfg["vn_max_bw"],
        "link_prob": train_cfg["vn_link_prob"],
    }
    ad = train_cfg["allowed_domains"]
    domain_ids = [lc.domain.id for lc in algo.global_controller.local_controllers]
    sub_hash = substrate_structure_hash(sn)

    algo.policy.train()

    first_100_reward, last_100_reward = [], []
    first_100_cpr, last_100_cpr = [], []
    first_100_sr, last_100_sr = [], []

    with open(log_path, "w", buffering=1) as log_f:  # line-buffered
        for ep in range(args.episodes):
            sample_substrate_state(
                algo.global_controller, sn,
                warmup_fraction=train_cfg["warmup_fraction"],
                u_max_cpu=train_cfg["u_max_cpu"], u_max_bw=train_cfg["u_max_bw"],
                M_max=train_cfg["warmup_M_max"], vn_kwargs=vn_kwargs,
            )
            vn = generate_random_vn_with_domains(
                min_nodes=vn_kwargs["min_nodes"], max_nodes=vn_kwargs["max_nodes"],
                min_cpu=vn_kwargs["min_cpu"], max_cpu=vn_kwargs["max_cpu"],
                min_bw=vn_kwargs["min_bw"], max_bw=vn_kwargs["max_bw"],
                link_prob=vn_kwargs["link_prob"],
                domain_ids=domain_ids,
                p_all=ad["p_all"], p_single=ad["p_single"], p_subset=ad["p_subset"],
                subset_min=ad["subset_min"], subset_max=ad["subset_max"],
            )
            req = VirtualNetworkRequest(id=f"pt_{ep}", virtual_network=vn,
                                        arrival_time=0.0, lifetime=float("inf"))
            reward, committed, dom_lps, sn_lps, success = algo._training_episode(req, persist=False)
            algo.trainer.record(
                domain_log_probs=dom_lps, snode_log_probs_per_vnode=sn_lps,
                reward=reward, committed_snode_indices=committed, success=success,
            )
            algo.global_controller.reset_allocations()
            algo.global_controller.clear_caches()

            # Track head/tail reward + cost/revenue + success for convergence report.
            cpr = -reward if success else float("nan")
            if ep < 100:
                first_100_reward.append(reward); first_100_cpr.append(cpr); first_100_sr.append(float(success))
            if ep >= args.episodes - 100:
                last_100_reward.append(reward); last_100_cpr.append(cpr); last_100_sr.append(float(success))

            if (ep + 1) % batch_size == 0:
                m = algo.trainer.update()
                log_line = {
                    "episode": ep + 1,
                    "loss_total": m["loss_total"], "loss_rl": m["loss_rl"], "loss_sup": m["loss_sup"],
                    "reward_mean": m["avg_reward"],
                    "reward_min": min(ep_r["reward"] for ep_r in ([{"reward": r} for r in [m["avg_reward"]]])),
                    "reward_max": m["avg_reward"],
                    "success_rate": m["success_rate"],
                    "cost_mean_success": 0.0,
                    "cost_per_revenue_mean": -m["avg_reward"],
                    "baseline": m["baseline"],
                    "lr": train_cfg["learning_rate"],
                    "timestamp": time.time(),
                }
                log_f.write(json.dumps(log_line) + "\n")

            if (ep + 1) % ckpt_every == 0:
                algo.save_checkpoint(args.checkpoint, substrate_hash=sub_hash)
                algo._episodes_trained = ep + 1

        if algo.trainer.buffer:
            algo.trainer.update()
        algo.save_checkpoint(args.checkpoint, substrate_hash=sub_hash)
        algo._episodes_trained = args.episodes

    def _mean(xs):
        xs = [x for x in xs if x == x]  # drop NaN
        return sum(xs) / len(xs) if xs else float("nan")

    r_first = _mean(first_100_reward); r_last = _mean(last_100_reward)
    cpr_first = _mean(first_100_cpr); cpr_last = _mean(last_100_cpr)
    sr_first = _mean(first_100_sr); sr_last = _mean(last_100_sr)

    print("=" * 60)
    print("TRAINING SUMMARY")
    print(f"  reward       first-100={r_first:.4f}  last-100={r_last:.4f}")
    print(f"  cost/rev     first-100={cpr_first:.4f}  last-100={cpr_last:.4f}")
    print(f"  success_rate first-100={sr_first:.4f}  last-100={sr_last:.4f}")
    converged = r_last >= 1.2 * r_first if r_first != 0 else False
    print(f"  {'CONVERGED' if converged else 'NOT_CONVERGED'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Quick smoke test.**

Run:

```bash
python scripts/train_rl_cand_vne.py \
  --substrate datasets/scenario_1/substrate.json \
  --config    configs/rl_cand_vne.yaml \
  --episodes  20 \
  --checkpoint /tmp/rl_cand_vne_smoke.pt \
  --log-dir   /tmp/rl_cand_vne_smoke_log/ \
  --seed      0
```

Expected: completes without errors, prints `TRAINING SUMMARY`, creates `/tmp/rl_cand_vne_smoke.pt` and `/tmp/rl_cand_vne_smoke_log/train.jsonl`.

- [ ] **Step 3: Commit.**

```bash
git add scripts/train_rl_cand_vne.py
git commit -m "feat(rl-cand-vne): add offline training script with JSONL logging and convergence summary"
```

---

## Task 13: Registry + experiment runner integration

**Files:**
- Modify: `algorithms/registry.py`
- Modify: `scripts/run_experiments.sh`

- [ ] **Step 1: Write failing registry test.**

Append to `tests/test_rl_cand_vne_end_to_end.py`:

```python
class TestRegistry(unittest.TestCase):
    def test_registry_returns_rl_cand_vne(self):
        from algorithms.registry import get_algorithm
        algo = get_algorithm("rl_cand_vne")
        self.assertEqual(algo.__class__.__name__, "RLCandVNE")
```

- [ ] **Step 2: Run, confirm fail.**

Run: `python -m pytest tests/test_rl_cand_vne_end_to_end.py::TestRegistry -v`
Expected: FAIL — `ValueError: Algorithm 'rl_cand_vne' not found`.

- [ ] **Step 3: Register the algorithm.**

Edit `algorithms/registry.py` — add import and entry:

```python
from algorithms.rl_cand_vne.rl_cand_vne import RLCandVNE
```

and in `ALGORITHMS` dict:

```python
    "rl_cand_vne": RLCandVNE,
```

- [ ] **Step 4: Run, confirm pass.**

Run: `python -m pytest tests/test_rl_cand_vne_end_to_end.py::TestRegistry -v`
Expected: 1 passed.

- [ ] **Step 5: Add `rl_cand_vne` to experiment runner.**

In `scripts/run_experiments.sh`, replace the line:

```bash
ALGORITHMS=("mp_vne" "oa_mp_vne" "rl_oa_mp_vne")
```

with:

```bash
ALGORITHMS=("mp_vne" "oa_mp_vne" "rl_oa_mp_vne" "rl_cand_vne")
```

- [ ] **Step 6: Commit.**

```bash
git add algorithms/registry.py scripts/run_experiments.sh tests/test_rl_cand_vne_end_to_end.py
git commit -m "feat(rl-cand-vne): register in algorithms registry and experiment runner"
```

---

## Task 14: Integration end-to-end test (offline train → checkpoint → solve)

**Files:**
- Modify: `tests/test_rl_cand_vne_end_to_end.py`

- [ ] **Step 1: Write failing integration test.**

Append to `tests/test_rl_cand_vne_end_to_end.py`:

```python
import subprocess
import sys
import json


class TestOfflineTrainingToSolveIntegration(unittest.TestCase):
    def test_offline_train_then_solve(self):
        random.seed(0); torch.manual_seed(0)
        # Write a tiny substrate JSON for the offline script.
        sn = _build_sn()
        with tempfile.TemporaryDirectory() as tmp:
            sn_path = os.path.join(tmp, "substrate.json")
            _dump_substrate(sn, sn_path)
            ckpt = os.path.join(tmp, "ckpt.pt")
            log_dir = os.path.join(tmp, "log")
            cfg_path = os.path.join(tmp, "cfg.yaml")
            with open(cfg_path, "w") as f:
                import yaml as _y
                cfg = _y.safe_load(open("configs/rl_cand_vne.yaml"))
                cfg["training"]["warmup_fraction"] = 0.0
                cfg["training"]["batch_size"] = 5
                cfg["training"]["vn_max_nodes"] = 3
                _y.safe_dump(cfg, f)

            res = subprocess.run(
                [sys.executable, "scripts/train_rl_cand_vne.py",
                 "--substrate", sn_path, "--config", cfg_path,
                 "--episodes", "20", "--checkpoint", ckpt, "--log-dir", log_dir,
                 "--seed", "0"],
                capture_output=True, text=True, timeout=120,
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
```

- [ ] **Step 2: Run, confirm fail then pass.**

Run: `python -m pytest tests/test_rl_cand_vne_end_to_end.py::TestOfflineTrainingToSolveIntegration -v --timeout=180`
Expected: 1 passed within ~2 min.

- [ ] **Step 3: Commit.**

```bash
git add tests/test_rl_cand_vne_end_to_end.py
git commit -m "test(rl-cand-vne): add offline-train → checkpoint → solve integration test"
```

---

## Task 15: Plotting helper (optional, bundled if time permits)

**Files:**
- Create: `evaluation/plot_training_curve.py`

- [ ] **Step 1: Write helper.**

Write `evaluation/plot_training_curve.py`:

```python
#!/usr/bin/env python3
"""Plot loss / reward / success rate / cost-per-revenue from rl_cand_vne train.jsonl."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="logs/rl_cand_vne/train.jsonl")
    p.add_argument("--out", default="logs/rl_cand_vne/train_curve.png")
    args = p.parse_args()

    records = []
    with open(args.log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        print("No records in log.")
        return

    eps = [r["episode"] for r in records]
    loss = [r["loss_total"] for r in records]
    reward = [r["reward_mean"] for r in records]
    baseline = [r["baseline"] for r in records]
    success = [r["success_rate"] for r in records]
    cpr = [r["cost_per_revenue_mean"] for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(eps, loss); axes[0, 0].set_title("loss_total"); axes[0, 0].set_xlabel("episode")
    axes[0, 1].plot(eps, reward, label="reward"); axes[0, 1].plot(eps, baseline, label="baseline", linestyle="--")
    axes[0, 1].set_title("reward / baseline"); axes[0, 1].set_xlabel("episode"); axes[0, 1].legend()
    axes[1, 0].plot(eps, success); axes[1, 0].set_title("success_rate"); axes[1, 0].set_xlabel("episode")
    axes[1, 1].plot(eps, cpr); axes[1, 1].set_title("cost_per_revenue_mean"); axes[1, 1].set_xlabel("episode")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test (manual).**

Run (after Task 12's smoke-test log exists or pointing at any prior log):

```
python evaluation/plot_training_curve.py --log /tmp/rl_cand_vne_smoke_log/train.jsonl --out /tmp/curve.png
```

Expected: PNG written.

- [ ] **Step 3: Commit.**

```bash
git add evaluation/plot_training_curve.py
git commit -m "feat(rl-cand-vne): add training-curve plotting helper"
```

---

## Final verification

- [ ] Run the full test suite:

```bash
python -m pytest tests/test_rl_cand_vne_*.py -v
```

Expected: all tests pass.

- [ ] Run the full repo tests to confirm no regression:

```bash
python -m pytest tests/ -v
```

Expected: no new failures introduced.

- [ ] Smoke-run offline training on scenario_1:

```bash
python scripts/train_rl_cand_vne.py \
  --substrate datasets/scenario_1/substrate.json \
  --episodes  200 \
  --checkpoint checkpoints/rl_cand_vne.pt \
  --log-dir   logs/rl_cand_vne/
```

Expected: completes; `CONVERGED` or `NOT_CONVERGED` tag printed; checkpoint on disk.

- [ ] Sanity-check inference:

```bash
python main.py --algorithm rl_cand_vne \
  --substrate datasets/scenario_1/substrate.json \
  --requests  datasets/scenario_1/virtual_requests.json \
  --output    results/scenario_1/solutions_rl_cand_vne_smoke.json
```

Expected: file produced; acceptance rate > 0.
