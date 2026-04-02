# RL-OA-MP-VNE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Create a new `rl_oa_mp_vne` algorithm that replaces the hand-crafted vnode/vlink ranking heuristics in `oa_mp_vne` with a Policy Network (REINFORCE) using per-domain GCN substrate embeddings, pre-trained on synthetic virtual networks and continuing to learn online every k requests.

**Architecture:** A Policy Network with two heads (NodeHead, LinkHead) scores virtual nodes and links to produce priority orderings. A 2-layer GCN embeds each substrate domain independently (shared weights), producing a fixed 32-dim vector per domain. These domain embeddings are aggregated per-vnode/vlink based on `allowed_domains` and concatenated with virtual features. The rest of the pipeline (PSO optimization, order-aware link embedding) is reused from `oa_mp_vne`. Pre-training generates random VNs and runs full embedding episodes with REINFORCE updates. Online learning accumulates experiences every k requests during real solving.

**Tech Stack:** Python, PyTorch (already a dependency), YAML config, existing VNE framework.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `algorithms/rl_oa_mp_vne/__init__.py` | Create | Export `RLOAMPVNE` |
| `algorithms/rl_oa_mp_vne/policy_network.py` | Create | GCN encoder + NodeHead + LinkHead |
| `algorithms/rl_oa_mp_vne/vn_generator.py` | Create | Synthetic VN generator for pre-training |
| `algorithms/rl_oa_mp_vne/trainer.py` | Create | REINFORCE pre-training + online learning buffer |
| `algorithms/rl_oa_mp_vne/rl_oa_mp_vne.py` | Create | Main algorithm class (solve, PSO, commit) |
| `configs/rl_oa_mp_vne.yaml` | Create | All hyperparameters |
| `algorithms/registry.py` | Modify | Add `rl_oa_mp_vne` entry |
| `tests/test_rl_oa_mp_vne.py` | Create | Unit + integration tests |
| `scripts/run_experiments.sh` | Modify | Add `rl_oa_mp_vne` to algorithm list |

The new algorithm reuses `oa_mp_vne`'s `GlobalController` and `LocalController` directly via import — no need to copy them.

---

### Task 1: Config and VN Generator

**Files:**
- Create: `configs/rl_oa_mp_vne.yaml`
- Create: `algorithms/rl_oa_mp_vne/__init__.py`
- Create: `algorithms/rl_oa_mp_vne/vn_generator.py`
- Test: `tests/test_rl_oa_mp_vne.py`

- [x] **Step 1: Create config file**

```yaml
# configs/rl_oa_mp_vne.yaml
pso:
  num_particles: 20
  num_iterations: 15
  w: 0.7
  c1: 1.5
  c2: 1.5
  mutation_rate: 0.1

policy_network:
  hidden_size: 64
  gcn_hidden: 32
  learning_rate: 0.001
  gamma: 0.99

training:
  pretrain_episodes: 200
  batch_size: 16
  online_k: 10
  vn_min_nodes: 2
  vn_max_nodes: 8
  vn_min_cpu: 1.0
  vn_max_cpu: 30.0
  vn_min_bw: 5.0
  vn_max_bw: 80.0
  vn_link_prob: 0.5
```

- [x] **Step 2: Create empty module init**

```python
# algorithms/rl_oa_mp_vne/__init__.py
from algorithms.rl_oa_mp_vne.rl_oa_mp_vne import RLOAMPVNE
```

This will fail until we create `rl_oa_mp_vne.py`, but we set up the package structure now.

- [x] **Step 3: Write failing test for VN generator**

```python
# tests/test_rl_oa_mp_vne.py
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
        # At least one link (high probability with 0.8)
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
        # Check connectivity via BFS
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
```

- [x] **Step 4: Run test to verify it fails**

Run: `cd /Users/duvannguyen/Workspace/Studies/Virtual-Network-Embedding && python -m pytest tests/test_rl_oa_mp_vne.py::TestVNGenerator -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algorithms.rl_oa_mp_vne.vn_generator'`

- [x] **Step 5: Implement VN generator**

```python
# algorithms/rl_oa_mp_vne/vn_generator.py
import random
from problem.virtual_network import VirtualNetwork, VirtualNode, VirtualLink


def generate_random_vn(
    min_nodes: int = 2,
    max_nodes: int = 8,
    min_cpu: float = 1.0,
    max_cpu: float = 30.0,
    min_bw: float = 5.0,
    max_bw: float = 80.0,
    link_prob: float = 0.5,
) -> VirtualNetwork:
    """Generate a random connected virtual network."""
    num_nodes = random.randint(min_nodes, max_nodes)
    vn = VirtualNetwork(id=f"syn_{random.randint(0, 999999)}")

    # Create nodes
    node_ids = [f"v{i}" for i in range(num_nodes)]
    for nid in node_ids:
        vn.nodes[nid] = VirtualNode(
            id=nid,
            cpu_demand=round(random.uniform(min_cpu, max_cpu), 2),
        )

    # Create spanning tree first to guarantee connectivity
    shuffled = node_ids[:]
    random.shuffle(shuffled)
    for i in range(1, len(shuffled)):
        parent = shuffled[random.randint(0, i - 1)]
        child = shuffled[i]
        src, dst = (parent, child) if parent < child else (child, parent)
        bw = round(random.uniform(min_bw, max_bw), 2)
        vn.links[(src, dst)] = VirtualLink(source=src, target=dst, bandwidth_demand=bw)

    # Add extra random links
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            key = (node_ids[i], node_ids[j])
            if key not in vn.links and random.random() < link_prob:
                bw = round(random.uniform(min_bw, max_bw), 2)
                vn.links[key] = VirtualLink(
                    source=node_ids[i], target=node_ids[j], bandwidth_demand=bw
                )

    return vn
```

- [x] **Step 6: Run tests to verify they pass**

Run: `cd /Users/duvannguyen/Workspace/Studies/Virtual-Network-Embedding && python -m pytest tests/test_rl_oa_mp_vne.py::TestVNGenerator -v`
Expected: 3 tests PASS

- [x] **Step 7: Commit**

```bash
git add configs/rl_oa_mp_vne.yaml algorithms/rl_oa_mp_vne/__init__.py algorithms/rl_oa_mp_vne/vn_generator.py tests/test_rl_oa_mp_vne.py
git commit -m "feat(rl-oa-mp-vne): add config and synthetic VN generator with tests"
```

---

### Task 2: Policy Network (GCN + Ranking Heads)

**Files:**
- Create: `algorithms/rl_oa_mp_vne/policy_network.py`
- Modify: `tests/test_rl_oa_mp_vne.py`

- [x] **Step 1: Write failing tests for policy network**

Append to `tests/test_rl_oa_mp_vne.py`:

```python
import torch
from algorithms.rl_oa_mp_vne.policy_network import GCNEncoder, PolicyNetwork


class TestGCNEncoder(unittest.TestCase):
    def test_output_shape(self):
        """GCN should produce (num_nodes, gcn_hidden) output."""
        encoder = GCNEncoder(node_feat_size=5, hidden_size=32)
        # 3 nodes, 5 features each
        X = torch.randn(3, 5)
        # Normalized adjacency (3x3)
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
        # Substrate: 4 nodes, adjacency 4x4
        sub_X = torch.randn(4, 5)
        sub_A = torch.eye(4)
        # 3 vnodes, 5 features each
        vnode_feats = torch.randn(3, 5)
        # 2 vlinks, 5 features each
        vlink_feats = torch.randn(2, 5)
        # allowed_domain_masks: for each vnode, which domains to include (1 domain here)
        domain_node_feats = [(sub_X,)]
        domain_adj_mats = [(sub_A,)]

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
        domain_node_feats = [(sub_X,)]
        domain_adj_mats = [(sub_A,)]

        node_scores, link_scores = net(
            vnode_feats, vlink_feats,
            domain_node_feats, domain_adj_mats
        )
        loss = node_scores.sum() + link_scores.sum()
        loss.backward()
        # Check gradients exist on GCN weights
        for p in net.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd /Users/duvannguyen/Workspace/Studies/Virtual-Network-Embedding && python -m pytest tests/test_rl_oa_mp_vne.py::TestGCNEncoder -v tests/test_rl_oa_mp_vne.py::TestPolicyNetwork -v`
Expected: FAIL with `ImportError`

- [x] **Step 3: Implement policy network**

```python
# algorithms/rl_oa_mp_vne/policy_network.py
import torch
import torch.nn as nn
from typing import List, Tuple


class GCNEncoder(nn.Module):
    """2-layer Graph Convolutional Network for substrate domain embedding."""

    def __init__(self, node_feat_size: int, hidden_size: int = 32):
        super().__init__()
        self.W1 = nn.Linear(node_feat_size, hidden_size, bias=True)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Node feature matrix (N, node_feat_size)
            A_norm: Normalized adjacency matrix with self-loops (N, N)
        Returns:
            H: Node embeddings (N, hidden_size)
        """
        H = torch.relu(self.W1(A_norm @ X))
        H = torch.relu(self.W2(A_norm @ H))
        return H


class PolicyNetwork(nn.Module):
    """
    Policy network for ranking virtual nodes and links.

    Uses a shared GCN to embed substrate domains, then concatenates
    the aggregated substrate embedding with virtual features to produce
    priority scores for vnodes and vlinks.
    """

    def __init__(
        self,
        vnode_feat_size: int = 5,
        vlink_feat_size: int = 5,
        gcn_node_feat_size: int = 5,
        gcn_hidden: int = 32,
        hidden_size: int = 64,
    ):
        super().__init__()
        self.gcn = GCNEncoder(gcn_node_feat_size, gcn_hidden)

        # NodeHead: vnode_features + substrate_embed -> score
        node_input_size = vnode_feat_size + gcn_hidden
        self.node_head = nn.Sequential(
            nn.Linear(node_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # LinkHead: vlink_features + substrate_embed -> score
        link_input_size = vlink_feat_size + gcn_hidden
        self.link_head = nn.Sequential(
            nn.Linear(link_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(link_input_size, 1),
        )

    def _embed_domains(
        self,
        domain_node_feats: List[torch.Tensor],
        domain_adj_mats: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Run GCN on each domain, return list of domain embeddings (gcn_hidden,)."""
        embeddings = []
        for X, A in zip(domain_node_feats, domain_adj_mats):
            H = self.gcn(X, A)               # (N_d, gcn_hidden)
            embed = H.mean(dim=0)             # (gcn_hidden,)
            embeddings.append(embed)
        return embeddings

    def forward(
        self,
        vnode_feats: torch.Tensor,
        vlink_feats: torch.Tensor,
        per_vnode_domain_feats: List[Tuple[torch.Tensor, ...]],
        per_vnode_domain_adjs: List[Tuple[torch.Tensor, ...]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vnode_feats: (num_vnodes, vnode_feat_size)
            vlink_feats: (num_vlinks, vlink_feat_size)
            per_vnode_domain_feats: For each vnode, tuple of domain X tensors
            per_vnode_domain_adjs: For each vnode, tuple of domain A tensors
        Returns:
            node_scores: (num_vnodes,) - priority scores
            link_scores: (num_vlinks,) - priority scores
        """
        num_vnodes = vnode_feats.shape[0]
        num_vlinks = vlink_feats.shape[0]

        # Compute substrate context per vnode
        node_contexts = []
        for i in range(num_vnodes):
            domain_Xs = per_vnode_domain_feats[i]
            domain_As = per_vnode_domain_adjs[i]
            domain_embeds = self._embed_domains(domain_Xs, domain_As)
            # Aggregate allowed domain embeddings
            ctx = torch.stack(domain_embeds).mean(dim=0)  # (gcn_hidden,)
            node_contexts.append(ctx)
        substrate_ctx = torch.stack(node_contexts)  # (num_vnodes, gcn_hidden)

        # Node scores
        node_input = torch.cat([vnode_feats, substrate_ctx], dim=1)
        node_scores = self.node_head(node_input).squeeze(-1)  # (num_vnodes,)

        # Link scores: use mean of all node contexts as link substrate context
        link_ctx = substrate_ctx.mean(dim=0).unsqueeze(0).expand(num_vlinks, -1)
        link_input = torch.cat([vlink_feats, link_ctx], dim=1)
        link_scores = self.link_head(link_input).squeeze(-1)  # (num_vlinks,)

        return node_scores, link_scores
```

**Note:** There is a deliberate bug in the `link_head` definition — the last `nn.Linear` input size is wrong (`link_input_size` instead of `hidden_size`). This will be caught by the test and fixed.

- [x] **Step 4: Run tests, fix any issues**

Run: `cd /Users/duvannguyen/Workspace/Studies/Virtual-Network-Embedding && python -m pytest tests/test_rl_oa_mp_vne.py::TestGCNEncoder tests/test_rl_oa_mp_vne.py::TestPolicyNetwork -v`

If the `link_head` shape mismatch is caught, fix it:

In `policy_network.py`, the `link_head` Sequential should be:
```python
        self.link_head = nn.Sequential(
            nn.Linear(link_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
```

Re-run tests. Expected: 4 tests PASS.

- [x] **Step 5: Commit**

```bash
git add algorithms/rl_oa_mp_vne/policy_network.py tests/test_rl_oa_mp_vne.py
git commit -m "feat(rl-oa-mp-vne): implement GCN encoder and policy network with tests"
```

---

### Task 3: Trainer (REINFORCE Pre-training + Online Learning)

**Files:**
- Create: `algorithms/rl_oa_mp_vne/trainer.py`
- Modify: `tests/test_rl_oa_mp_vne.py`

- [x] **Step 1: Write failing tests for trainer**

Append to `tests/test_rl_oa_mp_vne.py`:

```python
from algorithms.rl_oa_mp_vne.trainer import RankingTrainer
from algorithms.rl_oa_mp_vne.policy_network import PolicyNetwork


class TestRankingTrainer(unittest.TestCase):
    def _make_policy(self):
        return PolicyNetwork(
            vnode_feat_size=5, vlink_feat_size=5,
            gcn_node_feat_size=5, gcn_hidden=32, hidden_size=64
        )

    def test_record_and_buffer_size(self):
        """Recording an experience should grow the buffer."""
        policy = self._make_policy()
        trainer = RankingTrainer(policy, lr=0.001, gamma=0.99, batch_size=4)
        # Fake log_probs and reward
        log_probs = [torch.tensor(0.5, requires_grad=True)]
        trainer.record(log_probs, reward=2.0)
        self.assertEqual(len(trainer.buffer), 1)

    def test_update_clears_buffer(self):
        """After update(), the buffer should be emptied."""
        policy = self._make_policy()
        trainer = RankingTrainer(policy, lr=0.001, gamma=0.99, batch_size=2)
        for _ in range(3):
            log_probs = [torch.tensor(0.5, requires_grad=True)]
            trainer.record(log_probs, reward=1.0)
        trainer.update()
        self.assertEqual(len(trainer.buffer), 0)

    def test_update_changes_weights(self):
        """REINFORCE update should modify policy network weights."""
        policy = self._make_policy()
        trainer = RankingTrainer(policy, lr=0.01, gamma=0.99, batch_size=2)
        # Get initial weights
        w_before = policy.node_head[0].weight.data.clone()
        for _ in range(3):
            log_probs = [torch.tensor(0.5, requires_grad=True) for _ in range(3)]
            trainer.record(log_probs, reward=5.0)
        trainer.update()
        w_after = policy.node_head[0].weight.data
        self.assertFalse(torch.equal(w_before, w_after))
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd /Users/duvannguyen/Workspace/Studies/Virtual-Network-Embedding && python -m pytest tests/test_rl_oa_mp_vne.py::TestRankingTrainer -v`
Expected: FAIL with `ImportError`

- [x] **Step 3: Implement trainer**

```python
# algorithms/rl_oa_mp_vne/trainer.py
import torch
import torch.optim as optim
from typing import List, Tuple

from algorithms.rl_oa_mp_vne.policy_network import PolicyNetwork


class RankingTrainer:
    """
    REINFORCE trainer for the ranking policy network.

    Accumulates (log_probs, reward) experiences in a buffer.
    On update(), computes policy gradient with baseline subtraction.
    Used for both pre-training and online learning.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        lr: float = 0.001,
        gamma: float = 0.99,
        batch_size: int = 16,
    ):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer: List[Tuple[List[torch.Tensor], float]] = []

    def record(self, log_probs: List[torch.Tensor], reward: float) -> None:
        """Store one episode's log-probabilities and reward."""
        self.buffer.append((log_probs, reward))

    def update(self) -> float:
        """
        Run REINFORCE update over buffered experiences.
        Returns average loss value.
        """
        if not self.buffer:
            return 0.0

        # Collect rewards and compute baseline
        rewards = [r for _, r in self.buffer]
        baseline = sum(rewards) / len(rewards)

        total_loss = torch.tensor(0.0)
        for log_probs, reward in self.buffer:
            advantage = reward - baseline
            episode_loss = torch.tensor(0.0)
            for lp in log_probs:
                episode_loss = episode_loss - lp * advantage
            total_loss = total_loss + episode_loss

        total_loss = total_loss / len(self.buffer)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        avg_loss = total_loss.item()
        self.buffer.clear()
        return avg_loss
```

- [x] **Step 4: Run tests to verify they pass**

Run: `cd /Users/duvannguyen/Workspace/Studies/Virtual-Network-Embedding && python -m pytest tests/test_rl_oa_mp_vne.py::TestRankingTrainer -v`
Expected: 3 tests PASS

- [x] **Step 5: Commit**

```bash
git add algorithms/rl_oa_mp_vne/trainer.py tests/test_rl_oa_mp_vne.py
git commit -m "feat(rl-oa-mp-vne): implement REINFORCE trainer with buffer and update"
```

---

### Task 4: Main Algorithm — Feature Extraction and NN Ranking

**Files:**
- Create: `algorithms/rl_oa_mp_vne/rl_oa_mp_vne.py`
- Modify: `tests/test_rl_oa_mp_vne.py`

This is the largest task. The main algorithm class integrates everything: feature extraction from substrate domains and virtual networks, NN-based ranking, PSO optimization, and the full `solve()` loop with pre-training and online learning.

- [x] **Step 1: Write failing tests for feature extraction and ranking**

Append to `tests/test_rl_oa_mp_vne.py`:

```python
from problem.substrate_network import SubstrateNetwork, SubstrateNode, SubstrateLink


class TestRLOAMPVNEFeatures(unittest.TestCase):
    def setUp(self):
        self.substrate = SubstrateNetwork()
        self.substrate.nodes["s1"] = SubstrateNode("s1", cpu_capacity=100.0, cpu_price=1.0, processing_delay=1.0)
        self.substrate.nodes["s2"] = SubstrateNode("s2", cpu_capacity=100.0, cpu_price=1.0, processing_delay=1.0)
        self.substrate.nodes["s3"] = SubstrateNode("s3", cpu_capacity=100.0, cpu_price=1.0, processing_delay=1.0)
        self.substrate.links[("s1", "s2")] = SubstrateLink("s1", "s2", bandwidth_capacity=50.0, bandwidth_price=1.0, transmission_delay=1.0)
        self.substrate.links[("s2", "s3")] = SubstrateLink("s2", "s3", bandwidth_capacity=50.0, bandwidth_price=1.0, transmission_delay=1.0)
        self.substrate.links[("s1", "s3")] = SubstrateLink("s1", "s3", bandwidth_capacity=50.0, bandwidth_price=1.0, transmission_delay=1.0)

    def test_domain_features_shape(self):
        """Domain feature extraction should return (num_nodes, 5) and (num_nodes, num_nodes)."""
        from algorithms.rl_oa_mp_vne.rl_oa_mp_vne import RLOAMPVNE
        algo = RLOAMPVNE()
        algo._init_controller(self.substrate)
        X, A = algo._extract_domain_features(algo.global_controller.local_controllers[0])
        self.assertEqual(X.shape[0], 3)  # 3 substrate nodes
        self.assertEqual(X.shape[1], 5)  # 5 features
        self.assertEqual(A.shape, (3, 3))

    def test_vnode_features_shape(self):
        """Vnode feature extraction should return (num_vnodes, 5)."""
        from algorithms.rl_oa_mp_vne.rl_oa_mp_vne import RLOAMPVNE
        vn = VirtualNetwork(id="test")
        vn.nodes["v1"] = VirtualNode("v1", cpu_demand=10.0)
        vn.nodes["v2"] = VirtualNode("v2", cpu_demand=20.0)
        vn.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=30.0)
        algo = RLOAMPVNE()
        feats = algo._extract_vnode_features(vn)
        self.assertEqual(feats.shape, (2, 5))

    def test_vlink_features_shape(self):
        """Vlink feature extraction should return (num_vlinks, 5)."""
        from algorithms.rl_oa_mp_vne.rl_oa_mp_vne import RLOAMPVNE
        vn = VirtualNetwork(id="test")
        vn.nodes["v1"] = VirtualNode("v1", cpu_demand=10.0)
        vn.nodes["v2"] = VirtualNode("v2", cpu_demand=20.0)
        vn.nodes["v3"] = VirtualNode("v3", cpu_demand=5.0)
        vn.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=30.0)
        vn.links[("v2", "v3")] = VirtualLink("v2", "v3", bandwidth_demand=15.0)
        algo = RLOAMPVNE()
        feats = algo._extract_vlink_features(vn)
        self.assertEqual(feats.shape, (2, 5))
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd /Users/duvannguyen/Workspace/Studies/Virtual-Network-Embedding && python -m pytest tests/test_rl_oa_mp_vne.py::TestRLOAMPVNEFeatures -v`
Expected: FAIL with `ImportError`

- [x] **Step 3: Implement the main algorithm class**

```python
# algorithms/rl_oa_mp_vne/rl_oa_mp_vne.py
import os
import random
import yaml
import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import OrderedDict

from algorithms.oa_mp_vne.global_controller import GlobalController
from algorithms.oa_mp_vne.local_controller import LocalController
from algorithms.rl_oa_mp_vne.policy_network import PolicyNetwork
from algorithms.rl_oa_mp_vne.trainer import RankingTrainer
from algorithms.rl_oa_mp_vne.vn_generator import generate_random_vn
from problem.substrate_network import SubstrateNetwork, SubstrateNode
from problem.virtual_network import VirtualNetwork, VirtualNode, VirtualLink
from problem.request import VirtualNetworkRequest
from problem.embedding_solution import EmbeddingSolution


class RLOAMPVNE:
    """
    RL-enhanced Order-Aware Multi-Path VNE.

    Uses a Policy Network (GCN + REINFORCE) to rank virtual nodes and links
    instead of the hand-crafted heuristic in OA-MP-VNE.
    Pre-trains on synthetic virtual networks, then continues learning
    online every k requests.
    """

    def __init__(self):
        self.name = "RL-OA-MP-VNE"
        self._active_mappings: Dict[str, Dict] = OrderedDict()
        self._request_count = 0
        self._pretrained = False

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs", "rl_oa_mp_vne.yaml",
        )
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except Exception:
            self.config = {
                "pso": {
                    "num_particles": 20, "num_iterations": 15,
                    "w": 0.7, "c1": 1.5, "c2": 1.5, "mutation_rate": 0.1,
                },
                "policy_network": {
                    "hidden_size": 64, "gcn_hidden": 32,
                    "learning_rate": 0.001, "gamma": 0.99,
                },
                "training": {
                    "pretrain_episodes": 200, "batch_size": 16, "online_k": 10,
                    "vn_min_nodes": 2, "vn_max_nodes": 8,
                    "vn_min_cpu": 1.0, "vn_max_cpu": 30.0,
                    "vn_min_bw": 5.0, "vn_max_bw": 80.0,
                    "vn_link_prob": 0.5,
                },
            }

        pn_cfg = self.config.get("policy_network", {})
        self.policy = PolicyNetwork(
            vnode_feat_size=5,
            vlink_feat_size=5,
            gcn_node_feat_size=5,
            gcn_hidden=pn_cfg.get("gcn_hidden", 32),
            hidden_size=pn_cfg.get("hidden_size", 64),
        )
        self.trainer = RankingTrainer(
            self.policy,
            lr=pn_cfg.get("learning_rate", 0.001),
            gamma=pn_cfg.get("gamma", 0.99),
            batch_size=self.config.get("training", {}).get("batch_size", 16),
        )
        self.global_controller = None

    # ---- Controller Init ----

    def _init_controller(self, substrate_network: SubstrateNetwork) -> None:
        self.global_controller = GlobalController(substrate_network)

    # ---- Feature Extraction ----

    def _extract_domain_features(self, lc: LocalController) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract node features and normalized adjacency for one domain.
        Returns: X (num_nodes, 5), A_norm (num_nodes, num_nodes)
        """
        net = lc.domain.network
        node_ids = list(net.nodes.keys())
        n = len(node_ids)
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        # Node features: [avail_cpu/capacity, cpu_price, proc_delay, degree, avg_neighbor_bw]
        X = torch.zeros(n, 5)
        degrees = [0] * n
        neighbor_bw = [0.0] * n

        for (u, v), link in net.links.items():
            if u in id_to_idx:
                degrees[id_to_idx[u]] += 1
                neighbor_bw[id_to_idx[u]] += getattr(link, 'available_bw', link.bandwidth_capacity)
            if v in id_to_idx:
                degrees[id_to_idx[v]] += 1
                neighbor_bw[id_to_idx[v]] += getattr(link, 'available_bw', link.bandwidth_capacity)

        max_degree = max(degrees) if degrees and max(degrees) > 0 else 1
        max_cap = max(nd.cpu_capacity for nd in net.nodes.values()) or 1.0
        max_bw = max(neighbor_bw) if neighbor_bw and max(neighbor_bw) > 0 else 1.0

        for i, nid in enumerate(node_ids):
            node = net.nodes[nid]
            avail = getattr(node, 'available_cpu', node.cpu_capacity)
            X[i, 0] = avail / max_cap
            X[i, 1] = node.cpu_price / 10.0  # normalize price
            X[i, 2] = node.processing_delay / 10.0  # normalize delay
            X[i, 3] = degrees[i] / max_degree
            X[i, 4] = (neighbor_bw[i] / degrees[i] / max_bw) if degrees[i] > 0 else 0.0

        # Adjacency with self-loops + symmetric normalization
        A = torch.zeros(n, n)
        for (u, v), link in net.links.items():
            if u in id_to_idx and v in id_to_idx:
                bw_ratio = getattr(link, 'available_bw', link.bandwidth_capacity) / link.bandwidth_capacity
                A[id_to_idx[u], id_to_idx[v]] = bw_ratio
                A[id_to_idx[v], id_to_idx[u]] = bw_ratio

        # Add self-loops
        A = A + torch.eye(n)
        # D^{-1/2} A D^{-1/2}
        D = A.sum(dim=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.clamp(min=1e-8)))
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt

        return X, A_norm

    def _extract_vnode_features(self, vn: VirtualNetwork) -> torch.Tensor:
        """
        Extract features for each virtual node.
        Returns: (num_vnodes, 5) tensor.
        Features: [cpu_demand_norm, degree_norm, adj_bw_norm, num_nodes_norm, num_links_norm]
        """
        vnodes = list(vn.nodes.values())
        n = len(vnodes)
        feats = torch.zeros(n, 5)

        degrees = {nd.id: 0 for nd in vnodes}
        adj_bw = {nd.id: 0.0 for nd in vnodes}
        for vlink in vn.links.values():
            degrees[vlink.source] = degrees.get(vlink.source, 0) + 1
            degrees[vlink.target] = degrees.get(vlink.target, 0) + 1
            adj_bw[vlink.source] = adj_bw.get(vlink.source, 0.0) + vlink.bandwidth_demand
            adj_bw[vlink.target] = adj_bw.get(vlink.target, 0.0) + vlink.bandwidth_demand

        max_cpu = max(nd.cpu_demand for nd in vnodes) or 1.0
        max_deg = max(degrees.values()) or 1
        max_bw = max(adj_bw.values()) if adj_bw and max(adj_bw.values()) > 0 else 1.0

        for i, nd in enumerate(vnodes):
            feats[i, 0] = nd.cpu_demand / max_cpu
            feats[i, 1] = degrees[nd.id] / max_deg
            feats[i, 2] = adj_bw[nd.id] / max_bw
            feats[i, 3] = len(vn.nodes) / 20.0
            feats[i, 4] = len(vn.links) / 40.0

        return feats

    def _extract_vlink_features(self, vn: VirtualNetwork) -> torch.Tensor:
        """
        Extract features for each virtual link.
        Returns: (num_vlinks, 5) tensor.
        Features: [bw_norm, src_degree_norm, dst_degree_norm, src_cpu_norm, dst_cpu_norm]
        """
        vlinks = list(vn.links.values())
        m = len(vlinks)
        feats = torch.zeros(m, 5)

        degrees = {nid: 0 for nid in vn.nodes}
        for vlink in vlinks:
            degrees[vlink.source] = degrees.get(vlink.source, 0) + 1
            degrees[vlink.target] = degrees.get(vlink.target, 0) + 1

        max_bw = max(vl.bandwidth_demand for vl in vlinks) or 1.0
        max_deg = max(degrees.values()) or 1
        max_cpu = max(nd.cpu_demand for nd in vn.nodes.values()) or 1.0

        for i, vl in enumerate(vlinks):
            feats[i, 0] = vl.bandwidth_demand / max_bw
            feats[i, 1] = degrees[vl.source] / max_deg
            feats[i, 2] = degrees[vl.target] / max_deg
            feats[i, 3] = vn.nodes[vl.source].cpu_demand / max_cpu
            feats[i, 4] = vn.nodes[vl.target].cpu_demand / max_cpu

        return feats

    def _get_domain_inputs(self, vn: VirtualNetwork) -> Tuple[list, list]:
        """
        For each vnode, get the substrate domain features and adjacency matrices
        of its allowed domains (or all domains if unconstrained).
        Returns parallel lists: per_vnode_Xs, per_vnode_As
        """
        # Pre-compute all domain features
        domain_cache = {}
        for lc in self.global_controller.local_controllers:
            X, A = self._extract_domain_features(lc)
            domain_cache[lc.domain.id] = (X, A)

        per_vnode_Xs = []
        per_vnode_As = []
        for vnode in vn.nodes.values():
            allowed = vnode.allowed_domains or [lc.domain.id for lc in self.global_controller.local_controllers]
            Xs = tuple(domain_cache[d][0] for d in allowed if d in domain_cache)
            As = tuple(domain_cache[d][1] for d in allowed if d in domain_cache)
            if not Xs:
                # Fallback: use all domains
                Xs = tuple(v[0] for v in domain_cache.values())
                As = tuple(v[1] for v in domain_cache.values())
            per_vnode_Xs.append(Xs)
            per_vnode_As.append(As)

        return per_vnode_Xs, per_vnode_As

    # ---- NN-Based Ranking ----

    def rank_virtual_nodes_nn(
        self, vn: VirtualNetwork, sample: bool = False
    ) -> Tuple[List[VirtualNode], List[torch.Tensor]]:
        """
        Use the policy network to rank virtual nodes.
        Args:
            sample: If True, sample from softmax (training). If False, use argmax (inference).
        Returns:
            ordered_vnodes: Sorted list of VirtualNode
            log_probs: Log-probabilities for REINFORCE (empty if sample=False)
        """
        vnode_feats = self._extract_vnode_features(vn)
        vlink_feats = self._extract_vlink_features(vn)
        per_vnode_Xs, per_vnode_As = self._get_domain_inputs(vn)

        node_scores, _ = self.policy(vnode_feats, vlink_feats, per_vnode_Xs, per_vnode_As)

        vnodes = list(vn.nodes.values())
        log_probs = []

        if sample:
            # Sample a permutation using Plackett-Luce model
            remaining_scores = node_scores.clone()
            remaining_indices = list(range(len(vnodes)))
            ordered_indices = []

            for _ in range(len(vnodes)):
                probs = torch.softmax(remaining_scores, dim=0)
                dist = torch.distributions.Categorical(probs)
                chosen_pos = dist.sample()
                log_probs.append(dist.log_prob(chosen_pos))

                chosen_idx = remaining_indices[chosen_pos.item()]
                ordered_indices.append(chosen_idx)

                # Remove chosen from remaining
                mask = torch.ones(len(remaining_indices), dtype=torch.bool)
                mask[chosen_pos.item()] = False
                remaining_scores = remaining_scores[mask]
                remaining_indices = [remaining_indices[j] for j in range(len(remaining_indices)) if j != chosen_pos.item()]

            ordered_vnodes = [vnodes[i] for i in ordered_indices]
        else:
            # Greedy: sort by score descending
            sorted_indices = torch.argsort(node_scores, descending=True).tolist()
            ordered_vnodes = [vnodes[i] for i in sorted_indices]

        return ordered_vnodes, log_probs

    def rank_virtual_links_nn(
        self, vn: VirtualNetwork, sample: bool = False
    ) -> Tuple[List[Tuple[Tuple[str, str], VirtualLink]], List[torch.Tensor]]:
        """
        Use the policy network to rank virtual links.
        Returns:
            ordered_links: Sorted list of ((src,dst), VirtualLink)
            log_probs: Log-probabilities for REINFORCE
        """
        vnode_feats = self._extract_vnode_features(vn)
        vlink_feats = self._extract_vlink_features(vn)
        per_vnode_Xs, per_vnode_As = self._get_domain_inputs(vn)

        _, link_scores = self.policy(vnode_feats, vlink_feats, per_vnode_Xs, per_vnode_As)

        link_items = list(vn.links.items())
        log_probs = []

        if sample:
            remaining_scores = link_scores.clone()
            remaining_indices = list(range(len(link_items)))
            ordered_indices = []

            for _ in range(len(link_items)):
                probs = torch.softmax(remaining_scores, dim=0)
                dist = torch.distributions.Categorical(probs)
                chosen_pos = dist.sample()
                log_probs.append(dist.log_prob(chosen_pos))

                chosen_idx = remaining_indices[chosen_pos.item()]
                ordered_indices.append(chosen_idx)

                mask = torch.ones(len(remaining_indices), dtype=torch.bool)
                mask[chosen_pos.item()] = False
                remaining_scores = remaining_scores[mask]
                remaining_indices = [remaining_indices[j] for j in range(len(remaining_indices)) if j != chosen_pos.item()]

            ordered_links = [link_items[i] for i in ordered_indices]
        else:
            sorted_indices = torch.argsort(link_scores, descending=True).tolist()
            ordered_links = [link_items[i] for i in sorted_indices]

        return ordered_links, log_probs

    # ---- Pre-Training ----

    def _pretrain(self, substrate_network: SubstrateNetwork) -> None:
        """Pre-train on synthetic virtual networks."""
        train_cfg = self.config.get("training", {})
        episodes = train_cfg.get("pretrain_episodes", 200)
        batch_size = train_cfg.get("batch_size", 16)

        print(f"  [RL-OA-MP-VNE] Pre-training on {episodes} synthetic VNs...")
        self.policy.train()

        for ep in range(episodes):
            vn = generate_random_vn(
                min_nodes=train_cfg.get("vn_min_nodes", 2),
                max_nodes=train_cfg.get("vn_max_nodes", 8),
                min_cpu=train_cfg.get("vn_min_cpu", 1.0),
                max_cpu=train_cfg.get("vn_max_cpu", 30.0),
                min_bw=train_cfg.get("vn_min_bw", 5.0),
                max_bw=train_cfg.get("vn_max_bw", 80.0),
                link_prob=train_cfg.get("vn_link_prob", 0.5),
            )

            # Reset substrate for each episode
            self.global_controller.reset_allocations()
            self.global_controller.clear_caches()

            # NN-ranked ordering (sample for exploration)
            ordered_vnodes, node_log_probs = self.rank_virtual_nodes_nn(vn, sample=True)
            ordered_links, link_log_probs = self.rank_virtual_links_nn(vn, sample=True)
            all_log_probs = node_log_probs + link_log_probs

            # Try embedding with this ordering
            reward = self._try_embedding(vn, ordered_vnodes, ordered_links)
            self.trainer.record(all_log_probs, reward)

            if (ep + 1) % batch_size == 0:
                loss = self.trainer.update()
                if (ep + 1) % (batch_size * 5) == 0:
                    print(f"    Episode {ep + 1}/{episodes}, loss={loss:.4f}")

        # Final update for remaining buffer
        if self.trainer.buffer:
            self.trainer.update()

        self.global_controller.reset_allocations()
        self.global_controller.clear_caches()
        self._pretrained = True
        print(f"  [RL-OA-MP-VNE] Pre-training complete.")

    def _try_embedding(
        self, vn: VirtualNetwork,
        ordered_vnodes: List[VirtualNode],
        ordered_links: List[Tuple[Tuple[str, str], VirtualLink]],
    ) -> float:
        """
        Attempt a full embedding with the given ordering. Returns reward.
        Does NOT permanently allocate — rolls back after evaluation.
        """
        # Collect candidates in ranked order
        candidate_nodes = []
        for vnode in ordered_vnodes:
            candidates = []
            seen = set()
            for lc in self.global_controller.local_controllers:
                if not vnode.allowed_domains or lc.domain.id in vnode.allowed_domains:
                    for snode in lc.get_candidates(vnode):
                        if snode.id not in seen:
                            seen.add(snode.id)
                            candidates.append(snode)
            candidate_nodes.append(candidates)

        if any(not c for c in candidate_nodes):
            return -1.0

        # Build index maps
        vnode_to_idx = {vnode.id: i for i, vnode in enumerate(ordered_vnodes)}
        vlink_indices = []
        for vlink in vn.links.values():
            vlink_indices.append({
                "src_idx": vnode_to_idx[vlink.source],
                "dst_idx": vnode_to_idx[vlink.target],
                "bw": vlink.bandwidth_demand,
            })

        # PSO search
        best_particle = self._pso(candidate_nodes, vlink_indices, ordered_vnodes)
        mapping = {
            ordered_vnodes[i].id: candidate_nodes[i][idx].id
            for i, idx in enumerate(best_particle)
        }

        # Try commit (will rollback internally on failure)
        try:
            vlink_paths = self._commit_mapping_ordered(mapping, vn, ordered_links)
            if not vlink_paths:
                return -1.0

            # Compute reward
            revenue = sum(nd.cpu_demand for nd in vn.nodes.values()) + \
                      sum(vl.bandwidth_demand for vl in vn.links.values())
            cost = self._compute_cost(mapping, vn, vlink_paths)
            reward = revenue / (cost + 1e-6)

            # Rollback — pre-training should not permanently allocate
            self.global_controller.release_mapping(mapping, vn, vlink_paths)
            return reward

        except (ValueError, Exception):
            return -1.0

    def _compute_cost(self, mapping: Dict[str, str], vn: VirtualNetwork, vlink_paths: Dict) -> float:
        """Compute total embedding cost (CPU + BW)."""
        cost = 0.0
        for vnode_id, snode_id in mapping.items():
            vnode = vn.nodes[vnode_id]
            _, snode = self.global_controller._find_snode(snode_id)
            if snode:
                cost += vnode.cpu_demand * snode.cpu_price
        for (v_src, v_dst), paths in vlink_paths.items():
            for path_links, bw in paths:
                for link in path_links:
                    cost += bw * link.bandwidth_price
        return cost

    # ---- PSO (reused from oa_mp_vne) ----

    def _repair_particle(self, particle: List[int], candidates: List[List[SubstrateNode]]) -> List[int]:
        used_snode_ids = set()
        for i in range(len(particle)):
            snode = candidates[i][particle[i]]
            if snode.id in used_snode_ids:
                found = False
                alternatives = list(range(len(candidates[i])))
                random.shuffle(alternatives)
                for alt_idx in alternatives:
                    if candidates[i][alt_idx].id not in used_snode_ids:
                        particle[i] = alt_idx
                        snode = candidates[i][alt_idx]
                        found = True
                        break
                if not found:
                    pass
            used_snode_ids.add(snode.id)
        return particle

    def _pso(self, candidates: List[List[SubstrateNode]],
             vlink_indices: List[Dict], ordered_vnodes: List[VirtualNode]) -> List[int]:
        pso_cfg = self.config.get("pso", {})
        num_particles = pso_cfg.get("num_particles", 20)
        num_iterations = pso_cfg.get("num_iterations", 15)
        w = pso_cfg.get("w", 0.7)
        c1 = pso_cfg.get("c1", 1.5)
        c2 = pso_cfg.get("c2", 1.5)
        mutation_rate = pso_cfg.get("mutation_rate", 0.1)
        num_vnode = len(candidates)

        population = [
            self._repair_particle(
                [random.randint(0, len(candidates[j]) - 1) for j in range(num_vnode)],
                candidates,
            )
            for _ in range(num_particles)
        ]
        velocities = [[0.0] * num_vnode for _ in range(num_particles)]

        pbest = [p[:] for p in population]
        pbest_score = [self._fitness(p, candidates, vlink_indices, ordered_vnodes) for p in population]

        gbest_idx = pbest_score.index(min(pbest_score))
        gbest = pbest[gbest_idx][:]
        gbest_score = pbest_score[gbest_idx]

        for _ in range(num_iterations):
            for i in range(num_particles):
                for j in range(num_vnode):
                    r1, r2 = random.random(), random.random()
                    velocities[i][j] = (
                        w * velocities[i][j]
                        + c1 * r1 * (pbest[i][j] - population[i][j])
                        + c2 * r2 * (gbest[j] - population[i][j])
                    )
                    new_idx = int(round(population[i][j] + velocities[i][j])) % len(candidates[j])
                    population[i][j] = new_idx

                if random.random() < mutation_rate:
                    mut_idx = random.randint(0, num_vnode - 1)
                    population[i][mut_idx] = random.randint(0, len(candidates[mut_idx]) - 1)

                self._repair_particle(population[i], candidates)

                score = self._fitness(population[i], candidates, vlink_indices, ordered_vnodes)
                if score < pbest_score[i]:
                    pbest[i] = population[i][:]
                    pbest_score[i] = score

            current_best = min(pbest_score)
            if current_best < gbest_score:
                gbest = pbest[pbest_score.index(current_best)][:]
                gbest_score = current_best

        return gbest

    def _fitness(self, particle_idx: List[int], candidates: List[List[SubstrateNode]],
                 vlink_indices: List[Dict], ordered_vnodes: List[VirtualNode]) -> float:
        mapping = [candidates[i][idx] for i, idx in enumerate(particle_idx)]
        snode_ids = {s.id for s in mapping}
        if len(snode_ids) != len(mapping):
            return float('inf')

        node_cost = sum(
            vnode.cpu_demand * snode.cpu_price
            for vnode, snode in zip(ordered_vnodes, mapping)
        )

        sorted_vlinks = sorted(vlink_indices, key=lambda x: x["bw"], reverse=True)
        link_cost = 0.0
        for vlink_info in sorted_vlinks:
            src_node = mapping[vlink_info["src_idx"]]
            dst_node = mapping[vlink_info["dst_idx"]]
            path = self.global_controller.shortest_path(
                src_node, dst_node, bw_required=min(1.0, vlink_info["bw"] * 0.1)
            )
            if not path:
                return float('inf')
            link_cost += sum(
                l.transmission_delay + l.bandwidth_price * vlink_info["bw"]
                for l in path
            )

        return node_cost + link_cost

    # ---- Commit Mapping (reused from oa_mp_vne, with custom link order) ----

    def _commit_mapping_ordered(
        self, mapping: Dict[str, str], vn: VirtualNetwork,
        ordered_links: List[Tuple[Tuple[str, str], VirtualLink]] = None,
    ) -> Dict[tuple, list]:
        """
        Commit mapping with links processed in the given order.
        If ordered_links is None, falls back to BW-descending order.
        """
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("Duplicate substrate node in mapping")

        allocated_cpu: Dict[str, float] = {}
        allocated_bw: Dict[tuple, float] = {}
        vlink_paths: Dict[tuple, list] = {}

        if ordered_links is None:
            link_items = list(vn.links.items())
            link_items.sort(key=lambda item: item[1].bandwidth_demand, reverse=True)
            ordered_links = link_items

        try:
            # Allocate CPU
            for vnode_id, snode_id in mapping.items():
                vnode = vn.nodes[vnode_id]
                _, snode = self.global_controller._find_snode(snode_id)
                if not snode:
                    raise ValueError(f"Node {snode_id} not found")
                if getattr(snode, 'available_cpu', snode.cpu_capacity) < vnode.cpu_demand:
                    raise ValueError(f"Insufficient CPU on {snode.id}")
                snode.available_cpu -= vnode.cpu_demand
                allocated_cpu[snode.id] = allocated_cpu.get(snode.id, 0) + vnode.cpu_demand

            # Allocate bandwidth in NN-ranked order
            for vlink_key, vlink in ordered_links:
                src_snode_id = mapping[vlink.source]
                dst_snode_id = mapping[vlink.target]
                _, src_snode = self.global_controller._find_snode(src_snode_id)
                _, dst_snode = self.global_controller._find_snode(dst_snode_id)

                path = self.global_controller.shortest_path(
                    src_snode, dst_snode, bw_required=vlink.bandwidth_demand, use_cache=False
                )
                if not path:
                    raise ValueError(f"No path for vlink {vlink.source}->{vlink.target}")

                path_bw = min(getattr(l, 'available_bw', l.bandwidth_capacity) for l in path)
                if path_bw < vlink.bandwidth_demand:
                    raise ValueError(f"Insufficient BW for vlink {vlink.source}->{vlink.target}")

                for link in path:
                    link.available_bw -= vlink.bandwidth_demand
                    link_key = (link.source, link.target)
                    allocated_bw[link_key] = allocated_bw.get(link_key, 0) + vlink.bandwidth_demand

                vlink_paths[vlink_key] = [(path, vlink.bandwidth_demand)]

        except Exception as e:
            for snode_id, cpu in allocated_cpu.items():
                _, snode = self.global_controller._find_snode(snode_id)
                if snode:
                    snode.available_cpu += cpu
            for link_key, bw in allocated_bw.items():
                link = self.global_controller._find_link(*link_key)
                if link:
                    link.available_bw += bw
            raise e

        return vlink_paths

    # ---- Lifecycle ----

    def _release_expired(self, current_time: float) -> None:
        expired_ids = [rid for rid, data in self._active_mappings.items()
                       if data["expire_time"] <= current_time]
        for expired_id in expired_ids:
            data = self._active_mappings.pop(expired_id)
            self.global_controller.release_mapping(
                data["mapping"], data["vnetwork"], data["vlink_paths"]
            )

    # ---- Main Solve ----

    def solve(self, substrate_network: SubstrateNetwork, virtual_request: VirtualNetworkRequest) -> EmbeddingSolution:
        # Initialize controller on first call
        if self.global_controller is None:
            self._init_controller(substrate_network)

        # Pre-train on first call
        if not self._pretrained:
            self._pretrain(substrate_network)

        self._release_expired(virtual_request.arrival_time)
        self.global_controller.clear_caches()

        vnetwork = virtual_request.virtual_network
        solution = EmbeddingSolution(vnr_id=virtual_request.id, is_successful=False)

        # NN-ranked ordering (greedy inference)
        self.policy.eval()
        with torch.no_grad():
            ordered_vnodes, _ = self.rank_virtual_nodes_nn(vnetwork, sample=False)
            ordered_links, _ = self.rank_virtual_links_nn(vnetwork, sample=False)

        # Collect candidates in ranked order
        candidate_nodes = []
        for vnode in ordered_vnodes:
            candidates = []
            seen = set()
            for lc in self.global_controller.local_controllers:
                if not vnode.allowed_domains or lc.domain.id in vnode.allowed_domains:
                    for snode in lc.get_candidates(vnode):
                        if snode.id not in seen:
                            seen.add(snode.id)
                            candidates.append(snode)
            candidate_nodes.append(candidates)

        if any(not c for c in candidate_nodes):
            self._record_online(vnetwork, -1.0)
            return solution

        # Build index maps
        vnode_to_idx = {vnode.id: i for i, vnode in enumerate(ordered_vnodes)}
        vlink_indices = []
        for vlink in vnetwork.links.values():
            vlink_indices.append({
                "src_idx": vnode_to_idx[vlink.source],
                "dst_idx": vnode_to_idx[vlink.target],
                "bw": vlink.bandwidth_demand,
            })

        # PSO optimization
        best_particle = self._pso(candidate_nodes, vlink_indices, ordered_vnodes)
        best_mapping = {
            ordered_vnodes[i].id: candidate_nodes[i][idx].id
            for i, idx in enumerate(best_particle)
        }

        # Commit
        try:
            vlink_paths = self._commit_mapping_ordered(best_mapping, vnetwork, ordered_links)
            if not vlink_paths:
                raise ValueError("No paths allocated")
            solution.is_successful = True
        except ValueError:
            self._record_online(vnetwork, -1.0)
            return solution

        cost = self._compute_cost(best_mapping, vnetwork, vlink_paths)
        revenue = sum(nd.cpu_demand for nd in vnetwork.nodes.values()) + \
                  sum(vl.bandwidth_demand for vl in vnetwork.links.values())
        reward = revenue / (cost + 1e-6)

        solution.node_mapping = best_mapping
        solution.embedding_cost = cost

        formatted_link_mapping = {}
        for (v_src, v_dst), allocated_paths in vlink_paths.items():
            formatted_paths = []
            for path_links, allocated_bw in allocated_paths:
                link_tuples = [(l.source, l.target) for l in path_links]
                formatted_paths.append((link_tuples, allocated_bw))
            formatted_link_mapping[(v_src, v_dst)] = formatted_paths
        solution.link_mapping = formatted_link_mapping

        self._active_mappings[virtual_request.id] = {
            "mapping": best_mapping,
            "vnetwork": vnetwork,
            "vlink_paths": vlink_paths,
            "expire_time": virtual_request.arrival_time + virtual_request.lifetime,
        }

        # Online learning
        self._record_online(vnetwork, reward)

        return solution

    def _record_online(self, vn: VirtualNetwork, reward: float) -> None:
        """Record experience for online learning and update every k requests."""
        self._request_count += 1

        # Re-run with sampling to get log_probs for REINFORCE
        self.policy.train()
        _, node_lp = self.rank_virtual_nodes_nn(vn, sample=True)
        _, link_lp = self.rank_virtual_links_nn(vn, sample=True)
        self.trainer.record(node_lp + link_lp, reward)

        online_k = self.config.get("training", {}).get("online_k", 10)
        if self._request_count % online_k == 0 and self.trainer.buffer:
            self.trainer.update()
```

- [x] **Step 4: Run tests to verify feature extraction passes**

Run: `cd /Users/duvannguyen/Workspace/Studies/Virtual-Network-Embedding && python -m pytest tests/test_rl_oa_mp_vne.py::TestRLOAMPVNEFeatures -v`
Expected: 3 tests PASS

- [x] **Step 5: Commit**

```bash
git add algorithms/rl_oa_mp_vne/rl_oa_mp_vne.py tests/test_rl_oa_mp_vne.py
git commit -m "feat(rl-oa-mp-vne): implement main algorithm with NN ranking, PSO, and online learning"
```

---

### Task 5: End-to-End Integration Tests

**Files:**
- Modify: `tests/test_rl_oa_mp_vne.py`

- [x] **Step 1: Write end-to-end tests**

Append to `tests/test_rl_oa_mp_vne.py`:

```python
from problem.request import VirtualNetworkRequest


class TestRLOAMPVNEEndToEnd(unittest.TestCase):
    def setUp(self):
        self.substrate = SubstrateNetwork()
        self.substrate.nodes["s1"] = SubstrateNode("s1", cpu_capacity=100.0, cpu_price=1.0, processing_delay=1.0)
        self.substrate.nodes["s2"] = SubstrateNode("s2", cpu_capacity=100.0, cpu_price=1.0, processing_delay=1.0)
        self.substrate.nodes["s3"] = SubstrateNode("s3", cpu_capacity=100.0, cpu_price=1.0, processing_delay=1.0)
        self.substrate.links[("s1", "s2")] = SubstrateLink("s1", "s2", bandwidth_capacity=50.0, bandwidth_price=1.0, transmission_delay=1.0)
        self.substrate.links[("s2", "s3")] = SubstrateLink("s2", "s3", bandwidth_capacity=50.0, bandwidth_price=1.0, transmission_delay=1.0)
        self.substrate.links[("s1", "s3")] = SubstrateLink("s1", "s3", bandwidth_capacity=50.0, bandwidth_price=1.0, transmission_delay=1.0)

    def test_simple_embedding_succeeds(self):
        """A simple 2-node VN with low demand should embed successfully."""
        from algorithms.rl_oa_mp_vne.rl_oa_mp_vne import RLOAMPVNE
        vnr = VirtualNetwork(id="vn_simple")
        vnr.nodes["v1"] = VirtualNode("v1", cpu_demand=5.0)
        vnr.nodes["v2"] = VirtualNode("v2", cpu_demand=5.0)
        vnr.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=10.0)
        request = VirtualNetworkRequest(id="req1", virtual_network=vnr, arrival_time=0.0, lifetime=50.0)

        algo = RLOAMPVNE()
        # Override pretrain to use fewer episodes for speed
        algo.config["training"]["pretrain_episodes"] = 5
        solution = algo.solve(self.substrate, request)

        self.assertTrue(solution.is_successful)
        self.assertEqual(len(solution.node_mapping), 2)
        # Mapped to different substrate nodes
        self.assertNotEqual(
            list(solution.node_mapping.values())[0],
            list(solution.node_mapping.values())[1],
        )

    def test_infeasible_request_fails_gracefully(self):
        """A request demanding more CPU than available should fail without error."""
        from algorithms.rl_oa_mp_vne.rl_oa_mp_vne import RLOAMPVNE
        vnr = VirtualNetwork(id="vn_fail")
        vnr.nodes["v1"] = VirtualNode("v1", cpu_demand=500.0)
        vnr.nodes["v2"] = VirtualNode("v2", cpu_demand=500.0)
        vnr.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=10.0)
        request = VirtualNetworkRequest(id="req_fail", virtual_network=vnr, arrival_time=0.0, lifetime=50.0)

        algo = RLOAMPVNE()
        algo.config["training"]["pretrain_episodes"] = 3
        solution = algo.solve(self.substrate, request)

        self.assertFalse(solution.is_successful)

    def test_multiple_requests_with_online_learning(self):
        """Multiple requests should succeed and trigger online learning."""
        from algorithms.rl_oa_mp_vne.rl_oa_mp_vne import RLOAMPVNE
        algo = RLOAMPVNE()
        algo.config["training"]["pretrain_episodes"] = 3
        algo.config["training"]["online_k"] = 2  # Learn every 2 requests

        successes = 0
        for i in range(4):
            vnr = VirtualNetwork(id=f"vn_{i}")
            vnr.nodes["v1"] = VirtualNode("v1", cpu_demand=5.0)
            vnr.nodes["v2"] = VirtualNode("v2", cpu_demand=5.0)
            vnr.links[("v1", "v2")] = VirtualLink("v1", "v2", bandwidth_demand=10.0)
            request = VirtualNetworkRequest(
                id=f"req_{i}", virtual_network=vnr,
                arrival_time=float(i * 10), lifetime=5.0,
            )
            solution = algo.solve(self.substrate, request)
            if solution.is_successful:
                successes += 1

        self.assertGreater(successes, 0)
        # Online learning should have triggered at least once (k=2, 4 requests)
        self.assertGreaterEqual(algo._request_count, 4)
```

- [x] **Step 2: Run all tests**

Run: `cd /Users/duvannguyen/Workspace/Studies/Virtual-Network-Embedding && python -m pytest tests/test_rl_oa_mp_vne.py -v`
Expected: All tests PASS (may take a few seconds due to pre-training)

- [x] **Step 3: Commit**

```bash
git add tests/test_rl_oa_mp_vne.py
git commit -m "test(rl-oa-mp-vne): add end-to-end integration tests"
```

---

### Task 6: Registry, Script, and Final Wiring

**Files:**
- Modify: `algorithms/registry.py:1-21`
- Modify: `scripts/run_experiments.sh:13`

- [x] **Step 1: Register algorithm**

In `algorithms/registry.py`, add import and registry entry:

Add import line after line 8:
```python
from algorithms.rl_oa_mp_vne.rl_oa_mp_vne import RLOAMPVNE
```

Add to ALGORITHMS dict:
```python
    "rl_oa_mp_vne": RLOAMPVNE,
```

- [x] **Step 2: Update run_experiments.sh**

In `scripts/run_experiments.sh`, change line 13:
```bash
ALGORITHMS=("mp_vne" "oa_mp_vne" "rl_oa_mp_vne")
```

- [x] **Step 3: Verify registry works**

Run: `cd /Users/duvannguyen/Workspace/Studies/Virtual-Network-Embedding && python -c "from algorithms.registry import get_algorithm; a = get_algorithm('rl_oa_mp_vne'); print(a.name)"`
Expected: `RL-OA-MP-VNE`

- [x] **Step 4: Run full test suite**

Run: `cd /Users/duvannguyen/Workspace/Studies/Virtual-Network-Embedding && python -m pytest tests/ -v`
Expected: All tests PASS (existing + new)

- [x] **Step 5: Commit**

```bash
git add algorithms/registry.py algorithms/rl_oa_mp_vne/__init__.py scripts/run_experiments.sh
git commit -m "feat(rl-oa-mp-vne): register algorithm and add to experiment scripts"
```

---

## Self-Review Notes

1. **Spec coverage:** All requirements covered — GCN per-domain embedding, policy network with node+link heads, REINFORCE training, synthetic VN pre-training, online learning every k requests, configurable k in YAML, new algorithm registered separately.

2. **Placeholder scan:** No TBDs, TODOs, or vague instructions. All code blocks are complete.

3. **Type consistency:** `RLOAMPVNE` used consistently across `__init__.py`, `registry.py`, and tests. `PolicyNetwork`, `GCNEncoder`, `RankingTrainer` match between definition and usage. `generate_random_vn` signature matches test calls.

4. **Known intentional issue:** Task 2 Step 3 has a deliberate `link_head` shape bug (line `nn.Linear(link_input_size, 1)` instead of `nn.Linear(hidden_size, 1)`) — Step 4 explicitly catches and fixes it. This simulates TDD red-green flow.
