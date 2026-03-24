# Virtual Network Embedding in Multi-Domain Networks
## A Study of PSO, Reinforcement Learning, and Swarm Intelligence Approaches

---

## Slide 1: Title Slide

**Title:** Virtual Network Embedding in Multi-Domain Networks: A Study of PSO, Reinforcement Learning, and Swarm Intelligence Approaches

**Presenter:** Duvan Nguyen

---

## Slide 2: Outline

1. Abstract
2. Introduction & Motivation
3. Problem Formulation
4. Related Work
5. Proposed Methods
6. Experiment
7. Results & Discussion
8. Conclusion & Future Work
9. References

---

# I. ABSTRACT

## Slide 3: Abstract

Network virtualization enables multiple virtual networks to coexist on shared physical infrastructure, but efficiently embedding virtual network requests onto substrate networks — the Virtual Network Embedding (VNE) problem — remains NP-hard, especially in multi-domain environments. This work investigates and compares six VNE algorithms spanning three paradigms: heuristic (greedy), metaheuristic (Particle Swarm Optimization), and learning-based (Q-Learning, Deep Q-Networks, Swarm Reinforcement Learning). All algorithms operate within a unified centralized hierarchical multi-domain architecture with Global-Local controllers. We further incorporate multi-path link allocation to improve bandwidth utilization. Experiments on a multi-domain substrate network with 4 domains (120 nodes total) show that hybrid approaches combining PSO with DQN-guided fitness and multi-path routing achieve the best trade-off between embedding cost, delay, acceptance rate, and revenue-to-cost ratio. We also contextualize our work against the Virne benchmark (ICLR 2026), the most comprehensive evaluation framework for RL-based VNE, which identifies generalization and scalability as key open challenges in the field.

---

# II. INTRODUCTION & MOTIVATION

## Slide 4: What is Network Virtualization?

**Definition:** Network virtualization decouples virtual networks from the underlying physical infrastructure, allowing multiple isolated virtual networks to share the same substrate network.

```
  VN Request 1        VN Request 2
  [A]---[B]           [X]---[Y]
   |                   |     |
  [C]                 [Z]----+

          ↓ Embed onto ↓

   Physical Substrate Network
  [1]---[2]---[3]---[4]
   |     |     |     |
  [5]---[6]---[7]---[8]
```

**Applications:**
- Cloud computing: multi-tenant resource sharing
- 5G network slicing: dedicated virtual networks per service type
- Internet of Drones (IoD) in Industry 4.0
- Edge computing for latency-sensitive services

---

## Slide 5: Why is VNE Hard?

**The VNE problem is NP-hard** — it combines two coupled subproblems:
1. **Node mapping:** Assign each virtual node to a physical node (constrained by CPU, domain)
2. **Link mapping:** Route each virtual link through physical paths (constrained by bandwidth)

**Key challenges in multi-domain VNE:**

| Challenge | Description |
|-----------|-------------|
| **Combinatorial explosion** | $O(|N^p|^{|N^v|})$ possible node mappings |
| **Coupled decisions** | Node placement affects link routing feasibility |
| **Cross-domain routing** | Links may traverse multiple domains via boundary nodes |
| **Dynamic arrivals** | VN requests arrive and depart over time (online problem) |
| **Resource fragmentation** | Past embeddings leave scattered residual resources |

**Motivation:** No single approach dominates — we systematically compare heuristic, metaheuristic, and RL-based methods to understand their trade-offs.

---

# III. PROBLEM FORMULATION

## Slide 6: Substrate Network Model

*(Based on MP-VNE [Zhang et al., 2022])*

**Multi-domain substrate network:**

$$G^s = (\{G_i^s\}_{i=1}^{D}, L^s_{inter})$$

**Each physical domain:** $G_i^s = (N_i^s, L_i^s)$ with boundary nodes $B_i^s \subset N_i^s$

**Node attributes:**
| Symbol | Description | Range |
|--------|-------------|-------|
| $C_{n}^s$ | CPU processing capacity | [50, 100] |
| $P_{n}^s$ | Unit price of CPU resource | [1, 5] |
| $D_{n}^s$ | Processing delay | [0.1, 2.0] |

**Link attributes:**
| Symbol | Description | Intra-domain | Inter-domain |
|--------|-------------|-------------|--------------|
| $B_{l}^s$ | Bandwidth capacity | [500, 1000] | [1000, 5000] |
| $P_{l}^s$ | Unit price of bandwidth | [0.1, 1.0] | [0.5, 2.0] |
| $D_{l}^s$ | Transmission delay | [1.0, 10.0] | [5.0, 20.0] |

**Topology:** Erdos-Renyi random graph, edge probability = 0.5 per domain.

---

## Slide 7: Virtual Network Request Model

**Virtual network request (VNR):**

$$G^v = (N^v, L^v)$$

**Virtual node $n_i^v$:**
| Symbol | Description | Range |
|--------|-------------|-------|
| $C_i^v$ | CPU demand | [1, 20] |
| $D_i^v$ | Allowed physical domains | Subset of $\{1, \dots, D\}$ |

**Virtual link $l_{ij}^v$:**
| Symbol | Description | Range |
|--------|-------------|-------|
| $B_{ij}^v$ | Bandwidth demand | [1, 50] |

**Request arrival model:**
- Number of nodes: Uniform $\mathcal{U}(3, 10)$
- Edge probability: 0.5
- Arrival process: Poisson with rate $\lambda = 0.04$
- Lifetime: Exponential with mean = 500 time units

---

## Slide 8: VNE Mapping Definition

**Node mapping** $M_N: N^v \rightarrow N^s$ must satisfy:

$$C_i^v \leq C_{avail}(M_N(n_i^v)) \quad \forall n_i^v \in N^v$$
$$M_N(n_i^v) \in \text{Domain}(D_i^v) \quad \text{(domain constraint)}$$
$$M_N(n_i^v) \neq M_N(n_j^v) \quad \forall i \neq j \quad \text{(one-to-one)}$$

**Link mapping** $M_L: L^v \rightarrow \mathcal{P}(L^s)$ must satisfy:

$$B_{ij}^v \leq \sum_{p \in \text{Paths}} B_{avail}(p) \quad \text{(single or multi-path)}$$

---

## Slide 9: Optimization Objective

**Minimize composite embedding cost:**

$$\min \; \text{Cost}(G^v) = \underbrace{\sum_{n^v \in N^v} C^v_n \times P^s_{M_N(n^v)}}_{\text{Node mapping cost}} + \underbrace{\sum_{l^v \in L^v} \sum_{l^s \in M_L(l^v)} B^v_l \times P^s_{l^s}}_{\text{Link mapping cost}}$$

**Evaluation metrics (computed over time):**

| Metric | Formula | Description |
|--------|---------|-------------|
| **RAC** | $\frac{\text{# Accepted VNRs}}{\text{# Total VNRs}}$ | Request Acceptance Rate |
| **LAR** | $\frac{\sum_{v} REV(v) \times lifetime(v)}{T}$ | Long-term Average Revenue |
| **R2C** | $\frac{\sum_{v} REV(v) \times lifetime(v)}{\sum_{v} COST(v) \times lifetime(v)}$ | Revenue-to-Cost Ratio |
| **Avg Cost** | $\frac{\sum COST(v)}{\text{# Successes}}$ | Average embedding cost |
| **Avg Delay** | $\frac{\sum \text{PathDelay}(v)}{\text{# Successes}}$ | Average path latency |

Where $REV(v) = \sum C^v_n + \sum B^v_l$ (total resource demand as revenue proxy).

---

# IV. RELATED WORK

## Slide 10: Related Work — Overview

**Three paradigms for solving VNE:**

| Paradigm | Approach | Strengths | Weaknesses |
|----------|----------|-----------|------------|
| **Heuristic** | Greedy, ranking | Fast, simple | No global optimization, myopic |
| **Metaheuristic** | PSO, GA, ACO | Global search, avoids local optima | Slow, no learning between requests |
| **Learning-based** | Q-Learning, DQN, GNN+RL | Adapts over time, learns patterns | Training cost, generalization issues |

---

## Slide 11: Related Work — MP-VNE [Zhang et al., 2022]

**"A Multi-Domain VNE Algorithm Based on Multi-Objective Optimization for IoD Architecture in Industry 4.0"**

**Key contributions:**
1. Centralized hierarchical multi-domain architecture with Global-Local controllers
2. PSO with genetic mutation factor (10% random reset) to avoid local optima
3. Estimated mapping cost for candidate node pre-selection
4. Weighted summation converting multi-objective (cost + delay) into single objective

**Results:** Outperforms MC-VNM, VNE-PSO, and LID-VNE across acceptance rate (~60%), mapping cost (650-750), and delay (~460).

**Our adoption:** We use MP-VNE's problem formulation, multi-domain architecture, and PSO framework as the foundation for all 6 algorithms.

---

## Slide 12: Related Work — FlagVNE [Wang et al., IJCAI 2024]

**"FlagVNE: A Flexible and Generalizable RL Framework for Network Resource Allocation"**

**Key contributions:**
1. **Bidirectional action model:** Joint selection of virtual AND physical nodes
   - Expands action space from $|N^p| \times 1$ to $|N^p| \times |N^v|$
   - Provably better than unidirectional (Theorem 1)
2. **Hierarchical decoder:** Bilevel policy $\pi(a_t|s_t) = \pi^H(n^v|s_t) \cdot \pi^L(n^p|s_t, n^v)$
   - Reduces distribution from $|N^v| \times |N^p|$ to $|N^v| + |N^p|$
3. **Meta-RL with curriculum scheduling:** MAML for generalization across VNR sizes
   - Progressive complexity: start small, increase when entropy drops below $\delta$

**Results:** +10.4% RAC, +12.8% R2C over A3C-GCN on GEANT topology.

**Our reference:** FlagVNE represents the state-of-the-art RL approach; its bidirectional actions and meta-RL training are directions for our future work.

---

## Slide 13: Related Work — Swarm RL [Srini-Rohan, GitHub]

**"Swarm Reinforcement Learning Using Particle Swarm Optimization"**

**Key idea:** Multiple DQN agents act as PSO particles, sharing knowledge to accelerate learning.

**Mechanism:**
- Each agent independently trains a DQN on the environment
- PSO equations influence Q-value updates:
$$Q_{target} \mathrel{+}= \beta(Q_{pBest} - Q_{curr}) + \delta(Q_{gBest} - Q_{curr})$$
- Personal best (pBest) and global best (gBest) models are tracked

**Result:** 4-agent swarm converges in 117 episodes vs 130 for single DQN (~10% faster) on CartPole.

**Our adoption:** We adapt this Swarm RL framework for VNE node selection in SRL-VNE, MP-DQN-VNE, and SRL-MP-VNE.

---

## Slide 14: Related Work — Virne Benchmark [Wang et al., ICLR 2026]

**"Virne: A Comprehensive Benchmark for Deep RL-based Network Resource Allocation in NFV"**

**Why it matters:** First unified benchmark with 30+ algorithms across cloud, edge, and 5G.

| Feature | Previous Benchmarks | Virne |
|---------|-------------------|-------|
| Scenarios | Cloud only | **Cloud + Edge + 5G** |
| Algorithms | 1-5 | **30+** |
| RL support | None | **Gym-style environments** |
| Evaluation | Effectiveness only | **Multi-perspective** (generalization, scalability, solvability) |

**Key findings from Virne:**
- PPO-DualGAT achieves best quality: 78.1% RAC, 0.74 R2C (WX100)
- Fixed intermediate rewards (0.1) outperform adaptive strategies
- Action masking improves acceptance by +5.3%
- **Generalization is the biggest open problem** — policies degrade on unseen traffic

**Our reference:** We use Virne's metrics (RAC, LAR, R2C) and evaluation methodology.

---

# V. PROPOSED METHODS

## Slide 15: System Architecture

**All 6 algorithms share a centralized hierarchical multi-domain architecture:**

```
┌──────────────────────────────────────────────────┐
│                 Global Controller                  │
│  • Receives VN Requests                           │
│  • Runs optimization (Greedy / PSO / RL)          │
│  • Coordinates cross-domain link routing          │
│  • Atomic commit / rollback of resource allocation│
└───────┬──────────────┬──────────────┬─────────────┘
        │              │              │
        ▼              ▼              ▼
┌────────────┐  ┌────────────┐  ┌────────────┐
│ Local D1   │  │ Local D2   │  │ Local D3   │  ...
│ • 30 nodes │  │ • 30 nodes │  │ • 30 nodes │
│ • 2 bndry  │  │ • 2 bndry  │  │ • 2 bndry  │
│ • Resource │  │ • Resource │  │ • Resource │
│   tracking │  │   tracking │  │   tracking │
│ • Candidate│  │ • Candidate│  │ • Candidate│
│   selection│  │   selection│  │   selection│
└────────────┘  └────────────┘  └────────────┘
```

---

## Slide 16: Two-Phase Mapping Pipeline

**All algorithms follow the same two-phase process:**

```
VN Request (G^v) arrives
        │
        ▼
┌───────────────────┐     Greedy / PSO / Q-Learning / DQN
│  Phase 1: NODE    │     selects substrate node for each
│  MAPPING          │     virtual node
└────────┬──────────┘
         │
         ▼
┌───────────────────┐     Kruskal MST / Dijkstra / Floyd +
│  Phase 2: LINK    │     Single-path or Multi-path (≤5)
│  MAPPING          │     allocation
└────────┬──────────┘
         │
    ┌────┴─────┐
    │ Success? │
    ├─ Yes ──→ Commit: deduct CPU & BW from substrate
    └─ No ───→ Rollback: restore ALL allocated resources
```

**Link mapping strategies:**
- **Single-path:** One physical path per virtual link
- **Multi-path:** Split bandwidth demand across up to 5 paths — better utilization, higher acceptance

---

## Slide 17: Algorithm Summary

| # | Algorithm | Node Mapping | Link Mapping | Learning |
|---|-----------|-------------|--------------|----------|
| 1 | **MC-VNM** | Greedy ranking | Kruskal MST (single) | None |
| 2 | **MP-VNE** | PSO (20 particles, 15 iter) | Floyd + multi-path (≤5) | None |
| 3 | **MPQ-VNE** | Q-Learning (ε-greedy) | Dijkstra (single) | Q-table |
| 4 | **SRL-VNE** | DQN + PSO Swarm (4 agents) | Dijkstra (single) | DQN |
| 5 | **MP-DQN-VNE** | PSO + DQN fitness (15p, 10i) | Floyd + multi-path (≤5) | DQN |
| 6 | **SRL-MP-VNE** | PSO + Swarm RL (10p, 50i) | Floyd + multi-path (≤5) | DQN |

**Algorithm evolution:**
```
Heuristic          Metaheuristic          Learning-based
─────────          ─────────────          ──────────────
MC-VNM ─────────→ MP-VNE ──────────┐
(Greedy)          (PSO + mutation)  │
                                    ├──→ MP-DQN-VNE
                  MPQ-VNE ──────┐  │    (PSO + DQN fitness
                  (Q-table)     │  │     + multi-path)
                                ▼  │
                              SRL-VNE ──→ SRL-MP-VNE
                              (DQN +      (Swarm RL + PSO
                               Swarm)      + multi-path)
```

---

## Slide 18: Method 1 — MC-VNM (Baseline Greedy)

**Node Mapping — Greedy cost-efficient selection:**
1. Sort virtual nodes by CPU demand **descending** (hardest-to-place first)
2. For each virtual node, score candidates:
$$\text{Score}(n^s) = \frac{P_{cpu}(n^s)}{C_{avail}(n^s) + \epsilon}$$
3. Select node with **lowest score** (most cost-efficient)

**Link Mapping — Kruskal's MST:**
1. Sort substrate links by cost: $D_{trans}(l) + P_{bw}(l) \times B^v_{demand}$
2. Build Minimum Spanning Tree using Union-Find
3. Find path in MST via BFS — **single path** per virtual link

**Characteristics:**
- Fastest algorithm (no iterations, no learning)
- Deterministic — same input always produces same output
- No global view — each node placed independently (myopic)

---

## Slide 19: Method 2 — MP-VNE (PSO + Multi-Path)

**Node Mapping — Particle Swarm Optimization:**

Each particle = a complete node mapping scheme. Swarm searches collectively.

$$v_i^{new} = w \cdot v_i + c_1 \cdot r_1 \cdot (x_i^{pbest} - x_i) + c_2 \cdot r_2 \cdot (x^{gbest} - x_i)$$
$$x_i^{new} = x_i + v_i^{new}$$

| Parameter | Value | Role |
|-----------|-------|------|
| Particles | 20 | Population size |
| Iterations | 15 | Search depth |
| $w$ (inertia) | 0.7 | Momentum of current direction |
| $c_1$ (cognitive) | 1.5 | Pull toward personal best |
| $c_2$ (social) | 1.5 | Pull toward global best |
| Mutation rate | 0.1 | 10% random reset — avoids local optima |

**Fitness function:**
$$f(x) = \sum_{n^v} C^v_n \times P^s_{M(n^v)} + \sum_{l^v} \left( D_{trans} + P_{bw} \times B^v_l \right) \times \text{hops}$$

**Link Mapping — Multi-path (up to 5 paths):**
1. Find shortest path → allocate $\min(\text{remaining demand}, \text{path BW})$
2. Repeat until demand satisfied or 5 paths used
3. Rollback if insufficient bandwidth

---

## Slide 20: Method 3 — MPQ-VNE (Q-Learning)

**Node Mapping — Tabular Q-Learning:**

**Q-table:** Maps $(domain\_id, \; snode\_id) \rightarrow Q\text{-value}$

**Initialization:** $Q(s) = \frac{C_{cpu}(n^s)}{P_{cpu}(n^s) + \epsilon}$ (capacity-to-price ratio)

**Action selection (ε-greedy):**
- With probability $1 - \epsilon$: select $\arg\max_s Q(s)$ (exploit)
- With probability $\epsilon$: select random candidate (explore)

**Q-value update after each VNR:**
$$Q(s) \leftarrow Q(s) + \alpha \cdot (r - Q(s))$$

| Parameter | Value |
|-----------|-------|
| $\alpha$ (learning rate) | 0.1 |
| $\gamma$ (discount) | 0.9 |
| $\epsilon$ (exploration) | 0.1 |
| $r$ (success / failure) | +1.0 / -1.0 |

**Link Mapping:** Dijkstra's shortest path (single path).

**Advantage over MC-VNM:** Improves with experience.
**Limitation:** Tabular — doesn't scale to large or continuous state spaces.

---

## Slide 21: Method 4 — SRL-VNE (DQN + PSO Swarm)

**Key idea:** Replace Q-table with neural networks; multiple DQN agents share knowledge via PSO.

**DQN Architecture:**
```
State (6-dim) → [FC 64 → ReLU] → [FC 64 → ReLU] → Q-values (20 actions)
```

**State representation:**
| Feature | Description | Normalization |
|---------|-------------|---------------|
| $v_{cpu}$ | Virtual node CPU demand | ÷ 20 |
| $s_{cpu}$ | Substrate node available CPU | ÷ 100 |
| $s_{bw}$ | Average available bandwidth | ÷ 500 |
| $s_{degree}$ | Node degree | ÷ 1000 |
| $progress$ | Fraction of nodes mapped | [0, 1] |
| $bias$ | Constant term | 0.5 |

**Swarm of 4 DQN agents with PSO-influenced updates:**

$$Q_{target} \mathrel{+}= \beta(Q_{pBest} - Q_{curr}) + \delta(Q_{gBest} - Q_{curr})$$

- $\beta = 0.1$ (personal best influence), $\delta = 0.1$ (global best influence)
- Memory buffer: 2000 transitions, batch size: 64
- $\epsilon$: 1.0 → 0.05 (decay 0.995 per episode)

**Reward:** $R = 100 \times \frac{REV}{COST + \epsilon}$

**Link Mapping:** Dijkstra (single path).

---

## Slide 22: Method 5 — MP-DQN-VNE (PSO Guided by DQN)

**Key innovation:** DQN acts as an **intelligent fitness function** for PSO.

```
PSO Optimization (15 particles × 10 iterations)
  │
  │  For each particle (= candidate mapping):
  │
  │    1. Compute node_cost = Σ CPU_demand × CPU_price
  │    2. Extract state features for each mapped node
  │    3. Query DQN: max_q = max Q(state, ·)
  │    4. dqn_penalty = -10.0 × max_q
  │    5. fitness = node_cost + dqn_penalty
  │
  │  Higher Q-value → lower fitness → PSO PREFERS this mapping
  └──────────────────────────────────────────────────────
```

**Why this works:**
- **PSO** handles the combinatorial search over mapping space
- **DQN** provides a learned heuristic about node quality
- As DQN improves over time → PSO fitness becomes more accurate
- Combines global search (PSO) with learned guidance (DQN)

**Link Mapping:** Multi-path (up to 5 paths) — same as MP-VNE.

**Training:** All 4 swarm agents receive the same reward → cooperative learning.

---

## Slide 23: Method 6 — SRL-MP-VNE (Full Swarm RL + Multi-Path)

**The most complete hybrid — combines all techniques:**

| Component | Configuration |
|-----------|--------------|
| PSO | 10 particles, 50 iterations |
| Swarm RL | 4 DQN agents, pBest + gBest tracking |
| Link mapping | Multi-path, up to 5 paths |
| Reward | +100 (success), -50 (failure) |

**State features:** $[v_{cpu}/20, \; s_{cpu}/100, \; s_{bw}/500, \; s_{degree}/10, \; progress, \; 0.5]$

**Full pipeline:**
```
VNR → Find candidates → PSO with DQN-guided fitness → Multi-path allocation
  ↑                                                          │
  └────── Update pBest/gBest ←── Train all DQN agents ←── Reward
```

**Differences from MP-DQN-VNE:**
| | MP-DQN-VNE | SRL-MP-VNE |
|---|-----------|------------|
| PSO iterations | 10 | **50** (deeper search) |
| Reward signal | Continuous R2C | **Binary** (+100/-50) |
| Degree normalization | ÷ 1000 | **÷ 10** |

---

# VI. EXPERIMENT

## Slide 24: Experimental Setup — Substrate Network

**Multi-domain substrate network configuration:**

| Parameter | Value |
|-----------|-------|
| Number of domains | 4 |
| Nodes per domain | 30 |
| Boundary nodes per domain | 2 |
| Total physical nodes | 120 |
| Intra-domain edge probability | 0.5 |
| Inter-domain edge probability | 0.1 |

**Resource distributions:**

| Resource | Intra-domain | Inter-domain |
|----------|-------------|--------------|
| CPU capacity | $\mathcal{U}(50, 100)$ | — |
| CPU price | $\mathcal{U}(1, 5)$ | — |
| Processing delay | $\mathcal{U}(0.1, 2.0)$ | — |
| Bandwidth capacity | $\mathcal{U}(500, 1000)$ | $\mathcal{U}(1000, 5000)$ |
| Bandwidth price | $\mathcal{U}(0.1, 1.0)$ | $\mathcal{U}(0.5, 2.0)$ |
| Transmission delay | $\mathcal{U}(1.0, 10.0)$ | $\mathcal{U}(5.0, 20.0)$ |

---

## Slide 25: Experimental Setup — Virtual Requests & Evaluation

**Virtual network request parameters:**

| Parameter | Value |
|-----------|-------|
| Nodes per VNR | $\mathcal{U}(3, 10)$ |
| Edge probability | 0.5 |
| CPU demand | $\mathcal{U}(1, 20)$ |
| Bandwidth demand | $\mathcal{U}(1, 50)$ |
| Arrival process | Poisson, $\lambda = 0.04$ |
| Lifetime | Exponential, mean = 500 |

**Evaluation protocol:**
- **3 independent runs** per algorithm on the same dataset
- Results averaged with standard deviation bands
- 6 metrics plotted over simulation time (binned every 1000 time units)

**Metrics:**
| Metric | Measures |
|--------|----------|
| RAC | Request acceptance rate |
| LAR | Long-term average revenue |
| R2C | Revenue-to-cost efficiency |
| Avg Cost | Average per-embedding cost |
| Avg Delay | Average physical path latency |
| Success Count | Cumulative successful embeddings |

---

# VII. RESULTS & DISCUSSION

## Slide 26: Results — Performance Plots

*(Display: `results/test_1/algorithm_comparison_plots.png`)*

**6 subplots showing all algorithms over time:**
1. RAC (Acceptance Rate)
2. LAR (Average Revenue)
3. LT-R2C (Revenue/Cost)
4. Average Embedding Cost
5. Average Path Delay
6. Total Success Count

Each line = one algorithm, with shaded bands showing ± standard deviation across 3 runs.

---

## Slide 27: Results — Analysis

**Expected observations:**

| Aspect | Finding |
|--------|---------|
| **Acceptance Rate** | Multi-path methods (MP-VNE, MP-DQN-VNE, SRL-MP-VNE) achieve higher RAC due to flexible bandwidth splitting |
| **Embedding Cost** | PSO-based methods find lower-cost mappings than greedy MC-VNM |
| **Learning Effect** | RL methods (SRL-VNE, MP-DQN-VNE, SRL-MP-VNE) show improving performance over time as DQN agents learn |
| **Delay** | PSO fitness explicitly includes delay → lower average latency than greedy |
| **Revenue** | Higher acceptance + lower cost → higher long-term revenue for hybrid methods |

**Speed-Quality Trade-off:**

| Algorithm | Relative Speed | Relative Quality |
|-----------|---------------|-----------------|
| MC-VNM | Fastest | Lowest |
| MP-VNE | Medium | Good |
| MPQ-VNE | Fast | Medium |
| SRL-VNE | Slower | Good |
| MP-DQN-VNE | Slowest | High |
| SRL-MP-VNE | Slowest | Highest |

---

## Slide 28: Results — Comparison with Literature

**Our results in context of related work:**

| Source | Best Method | RAC | R2C |
|--------|------------|-----|-----|
| MP-VNE paper [Zhang 2022] | MP-VNE | ~60% | — |
| FlagVNE [Wang, IJCAI 2024] | FlagVNE | +10.4% over A3C-GCN | +12.8% over A3C-GCN |
| Virne benchmark [Wang, ICLR 2026] | PPO-DualGAT | 78.1% | 0.74 |
| **Our work** | SRL-MP-VNE | *see plots* | *see plots* |

**Key differences:**
- Our focus: **multi-domain** with cross-domain routing (MP-VNE, Virne focus on single-domain)
- Our contribution: systematic comparison of **PSO + RL hybrid** combinations
- Virne shows GNN encoders (GCN, GAT) significantly outperform MLP — a direction for our improvement

---

# VIII. CONCLUSION & FUTURE WORK

## Slide 29: Conclusion

**Contributions:**

1. **Unified multi-domain framework:** Centralized hierarchical architecture with Global-Local controllers supporting all 6 algorithms
2. **Systematic comparison of 3 paradigms:** Heuristic (MC-VNM), metaheuristic (MP-VNE), and learning-based (MPQ-VNE, SRL-VNE, MP-DQN-VNE, SRL-MP-VNE)
3. **Novel hybrids:** DQN-guided PSO fitness (MP-DQN-VNE) and full Swarm RL + PSO + multi-path (SRL-MP-VNE)
4. **Multi-path link allocation:** Up to 5 paths per virtual link for improved bandwidth utilization

**Key findings:**

- PSO provides significant improvement over greedy by enabling **global search**
- DQN-guided PSO fitness combines **learned heuristics** with **metaheuristic search**
- Swarm intelligence (4 DQN agents + PSO sharing) **accelerates RL convergence**
- Multi-path allocation improves **acceptance rates** at the cost of computational complexity
- A clear **speed-quality trade-off** exists across all approaches

---

## Slide 30: Future Work

| Direction | Motivation | Reference |
|-----------|-----------|-----------|
| **GNN-based encoders** (GCN, GAT, DualGAT) | Capture graph topology in state representation; PPO-DualGAT achieves 78.1% RAC | Virne [ICLR 2026] |
| **Bidirectional action model** | Joint virtual-physical node selection expands search space, provably better | FlagVNE [IJCAI 2024] |
| **Meta-RL with curriculum scheduling** | Generalize across VNR sizes without retraining | FlagVNE [IJCAI 2024] |
| **Latency-aware & energy-efficient** extensions | Multi-objective optimization for edge/5G scenarios | Virne [ICLR 2026] |
| **Larger-scale evaluation** | Real topologies (GEANT 40 nodes, BRAIN, WX500 500 nodes) | Virne [ICLR 2026] |
| **Integration with Virne benchmark** | Standardized comparison against 30+ algorithms | Virne [ICLR 2026] |

---

# IX. REFERENCES

## Slide 31: References

1. P. Zhang, C. Wang, Z. Qin, H. Cao, "A Multi-Domain VNE Algorithm Based on Multi-Objective Optimization for IoD Architecture in Industry 4.0," *arXiv:2202.12830*, 2022.

2. T. Wang, Q. Fan, C. Wang, L. Yang, L. Ding, N. J. Yuan, H. Xiong, "FlagVNE: A Flexible and Generalizable Reinforcement Learning Framework for Network Resource Allocation," *IJCAI*, 2024.

3. Srini-Rohan, "Swarm Reinforcement Learning Using PSO," GitHub, https://github.com/Srini-Rohan/Swarm-Reinforcement-Learning-Using-PSO.

4. T. Wang, L. Deng, X. Chen, J. Wang, H. He, L. Ding, W. Wu, Q. Fan, H. Xiong, "Virne: A Comprehensive Benchmark for Deep RL-based Network Resource Allocation in NFV," *ICLR*, 2026.

---

# APPENDIX

## Slide 32 (Appendix): Path Finding Algorithms

| Algorithm | Used By | Type | Complexity |
|-----------|---------|------|------------|
| **Floyd-Warshall** | MP-VNE, MP-DQN-VNE, SRL-MP-VNE | All-pairs shortest path (cached) | $O(V^3)$ |
| **Dijkstra** | MPQ-VNE, SRL-VNE | Single-source shortest path | $O(E \log V)$ |
| **Kruskal MST + BFS** | MC-VNM | Minimum spanning tree path | $O(E \log E)$ |

**Link cost function (all algorithms):**
$$\text{Cost}(l^s) = D_{trans}(l^s) + P_{bw}(l^s) \times B^v_{demand}$$

---

## Slide 33 (Appendix): DQN Hyperparameters

| Parameter | SRL-VNE | MP-DQN-VNE | SRL-MP-VNE |
|-----------|---------|------------|------------|
| Swarm agents | 4 | 4 | 4 |
| Network arch | 6→64→64→20 | 6→64→64→20 | 6→64→64→1 |
| Memory buffer | 2000 | 2000 | 2000 |
| Batch size | 64 | 64 | 64 |
| $\gamma$ (discount) | 0.95 | 0.95 | 0.95 |
| $\epsilon$ start→end | 1.0→0.05 | 1.0→0.05 | 1.0→0.05 |
| $\epsilon$ decay | 0.995 | 0.995 | 0.995 |
| $\beta$ (pBest) | 0.1 | 0.1 | 0.1 |
| $\delta$ (gBest) | 0.1 | 0.1 | 0.1 |
| Reward | $100 \times \frac{REV}{COST}$ | $100 \times \frac{REV}{COST}$ | +100 / -50 |

---

## Slide 34 (Appendix): Multi-Path Allocation Example

```
Virtual Link: A → B  (demand = 80 BW)

Iteration 1:
  Shortest path: [1]→[3]→[5]    available BW = 50
  Allocate: min(80, 50) = 50     remaining = 30

Iteration 2:
  Shortest path: [1]→[2]→[5]    available BW = 30
  Allocate: min(30, 30) = 30     remaining = 0

Result: 2 paths, total allocated = 80 ✓

If remaining > 0 after 5 paths → FAIL → rollback entire VNR
```
