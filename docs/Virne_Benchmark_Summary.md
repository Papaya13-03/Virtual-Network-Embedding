# Virne: Comprehensive Benchmark for Deep RL-based NFV Resource Allocation

> **Paper:** "Virne: A Comprehensive Benchmark for Deep RL-based Network Resource Allocation in NFV"
> **Authors:** Tianfu Wang, Liwei Deng, Xi Chen, Junyang Wang, Huiguo He, Leilei Ding, Wei Wu, Qilin Fan, Hui Xiong
> **Published:** ICLR 2026 | arXiv: 2507.19234
> **Code:** [github.com/GeminiLight/virne](https://github.com/GeminiLight/virne)

---

## 1. Problem Description

*(Same VNE/NFV-RA problem - see MP_VNE_Summary.md for full formulation)*

Virtual networks $G_v = (N_v, L_v, \omega, \varpi)$ mapped onto physical networks $G_p = (N_p, L_p)$.

**Constraints:**
- One-to-one node placement
- Resource sufficiency: $C(n_v) \leq C(n_p)$
- Link routing with bandwidth requirements, no loops

**Objective:** Maximize Revenue-to-Cost ratio: $R2C(S) = \varkappa \cdot REV(S) / COST(S)$

---

## 2. Motivation

Existing VNE/NFV research suffers from **fragmentation**:
- Different simulators, metrics, and setups across papers
- Most benchmarks limited to cloud-only, 1-5 algorithms, no RL support
- No standardized comparison framework

Virne addresses this with a **unified benchmark** supporting 30+ algorithms across cloud, edge, and 5G scenarios.

---

## 3. Framework Architecture (5 Modules)

```
┌─────────────────────────────────────────────────────┐
│                    Virne Framework                    │
├──────────────┬──────────────┬───────────────────────┤
│  Simulation  │   Network    │     Algorithm          │
│  Config      │   System     │     Implementation     │
│              │              │                        │
│ - Topologies │ - Event-     │ - 10+ RL methods       │
│ - Resources  │   driven     │ - PPO, A3C, PG, MCTS  │
│ - Cloud/Edge │   simulator  │ - MLP, CNN, GCN, GAT   │
│   /5G        │ - Constraint │ - DualGCN, HeteroGAT   │
│              │   validation │ - Action masking        │
├──────────────┴──────────────┴───────────────────────┤
│  Auxiliary Utilities  │  Evaluation Infrastructure   │
│  - Controller         │  - Effectiveness metrics     │
│  - Monitoring         │  - Generalization testing    │
│  - Visualization      │  - Scalability analysis      │
│                       │  - Solvability assessment    │
└───────────────────────┴─────────────────────────────┘
```

---

## 4. MDP Formulation for RL

| Component | Description |
|-----------|-------------|
| **State** $S$ | Embedding progress + resource availability |
| **Action** $A$ | Physical node selection for placement |
| **Transition** $P$ | State update upon placement attempt |
| **Reward** $R$ | Intermediate + final embedding reward |
| **Objective** | $\max_{\pi_\theta} \mathbb{E}_{\pi_\theta} \left[ \sum_{i=0}^{T} \gamma^i R(s_t, a_t) \right]$ |

### Unified RL Pipeline
```
Feature Constructor → Neural Policy Network (θ) → Action
        ↑                                           │
        └──── Experience Memory ←── Environment ←───┘
```

Modular design allows swapping any component (encoder, RL algorithm, features).

---

## 5. Key Implementation Findings (Ablation)

### Intermediate Rewards
| Strategy | Effect |
|----------|--------|
| Fixed reward = 0.1 per step | **Best** - consistent across architectures |
| Adaptive normalization | Worse performance |
| No intermediate reward | Slowest convergence |

### Feature Engineering
- **Status features only**: Baseline
- **Topological features only** (degree, betweenness, eigenvector): Moderate
- **Combined status + topological**: **Best performance**

### Action Masking
- Preventing infeasible placements improves acceptance rate by **up to 5.3%**
- Should always be enabled

---

## 6. Experimental Setup

### Topologies
| Network | Type | Nodes |
|---------|------|-------|
| WX100 | Synthetic | 100 |
| GEANT | Real | Intermediate |
| BRAIN | Real | Complex |

### Simulation Parameters
- 1000 VN requests per simulation
- VN sizes: Uniform [2, 10] nodes
- CPU demands: $\mathcal{U}(0, 20)$, BW demands: $\mathcal{U}(0, 50)$
- Lifetime: Exponential (mean 500)
- Arrival: Poisson process

### Metrics
| Metric | Description |
|--------|-------------|
| **RAC** | Request Acceptance Rate |
| **LRC** | Long-term Revenue-to-Cost ratio |
| **LAR** | Long-term Average Revenue |
| **AST** | Average Solving Time |

---

## 7. Results

### Effectiveness (Best RL vs Heuristics)

| Topology | Method | RAC | LRC | LAR |
|----------|--------|-----|-----|-----|
| WX100 | **PPO-DualGAT** | **78.10%** | **0.74** | **14,938** |
| GEANT | **PPO-DualGCN** | **60.40%** | **0.75** | - |
| BRAIN | PPO-DualGAT | **58.90%** | **0.75** | - |
| All | GRC-Rank (heuristic) | Lower | 0.56 | Lower |

**Key insight:** RL methods significantly outperform heuristics in quality but are slower (heuristics: 0.01-0.04s vs RL: sub-second but higher).

### Generalization
- Policies trained on specific traffic patterns **degrade** under different arrival rates
- Limited generalization to unseen demand distributions
- **Open challenge** for the field

### Scalability
- Performance degrades with network size
- Sub-second inference achievable for medium-scale networks
- Large-scale remains challenging

---

## 8. Extended Scenarios

| Scenario | Key Addition |
|----------|-------------|
| **Heterogeneous Resources** | Multiple resource types (CPU, GPU, memory) with simultaneous constraints |
| **Latency-Aware Edge** | Delay constraints: $D(\rho_p) \leq D(l_v)$ |
| **Energy-Efficient** | Multi-objective: $\max -w_a \sum E(n_p) + w_b \cdot R2C(S)$ |

---

## 9. Comparison with Previous Benchmarks

| Feature | VNE-Sim, ALEVIN, ALib | **Virne** |
|---------|----------------------|-----------|
| Scenarios | Cloud only | Cloud + Edge + 5G |
| Algorithms | 1-5 | **30+** |
| RL support | No | **Yes (Gym-style)** |
| Evaluation | Effectiveness only | **Multi-perspective** |
| GNN architectures | None | **MLP, CNN, GCN, GAT, DualGCN, HeteroGAT** |

---

## 10. Open Challenges Identified

1. **Representation learning** for cross-graph constraints
2. **Generalizable policies** across network scales and dynamics
3. **Robust frameworks** for conflicting operational constraints
4. **Scalability** to extremely large infrastructure

---

## 11. Takeaways for Slides

1. Virne is the **first comprehensive benchmark** for RL-based VNE/NFV (30+ algorithms, ICLR 2026)
2. GNN-based RL (PPO-DualGAT) consistently beats heuristics in quality
3. Implementation details matter: fixed intermediate rewards + combined features + action masking = best practice
4. **Generalization is the biggest open problem** - trained policies don't transfer well
5. Supports cloud, edge, and 5G scenarios in one unified framework
6. Practical tool for comparing new VNE algorithms fairly
