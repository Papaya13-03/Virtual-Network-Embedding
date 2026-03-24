# MP-VNE: Multi-Domain VNE Based on Multi-Objective Optimization

> **Paper:** "A Multi-Domain VNE Algorithm Based on Multi-Objective Optimization for IoD Architecture in Industry 4.0"
> **Authors:** Peiying Zhang, Chao Wang, Zeyu Qin, Haotong Cao
> **Published:** 2022 | arXiv: 2202.12830

---

## 1. Problem Description

### What is Virtual Network Embedding (VNE)?

Network virtualization allows multiple **Virtual Networks (VNs)** to coexist on a shared **Substrate (Physical) Network (SN)**. The VNE problem is: **how to efficiently map virtual network requests onto the physical infrastructure** while satisfying resource constraints and optimizing cost/delay.

### Substrate Network Model

The physical infrastructure is modeled as a weighted undirected graph:

$$G^s = (G_i^s, L^s)$$

- $G_i^s = (N_i^s, L_i^s)$: Individual physical domain with nodes and intra-domain links
- $L^s$: Inter-domain links connecting boundary nodes across domains

**Node attributes:**
| Symbol | Meaning |
|--------|---------|
| $C_{N_i^s}$ | CPU processing capacity |
| $P_{N_i^s}$ | Unit price of node resource |
| $D_{N_i^s}$ | Processing delay |

**Link attributes:**
| Symbol | Meaning |
|--------|---------|
| $B_{L_i^s}$ | Bandwidth capacity |
| $P_{L_i^s}$ | Resource unit price |
| $D_{L_i^s}$ | Transmission delay |

### Virtual Network Model

Virtual requests are modeled as:

$$G^v = (N^v, L^v)$$

- **Virtual nodes**: CPU demand ($C^v$), candidate domain constraints ($D^v$)
- **Virtual links**: Bandwidth requirements ($B_l^v$)

### Objective

Minimize a **composite cost** combining resource expenses and latency:

$$\text{Cost} = \sum \left[ CPU(n^v) \times Cost(n_v^s) \right] + \sum \left[ BW(l^v) \times Cost(l_v^s) \right]$$

This is a **multi-objective optimization** problem (minimize cost AND delay), converted to single-objective via **weighted summation**.

---

## 2. Multi-Domain Architecture

MP-VNE uses a **centralized hierarchical** design:

```
                    ┌─────────────────┐
                    │ Global Controller│
                    │  - Receives VNRs │
                    │  - PSO Optimizer │
                    └───────┬─────────┘
              ┌─────────────┼─────────────┐
              v             v             v
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │Local Ctrl D1 │ │Local Ctrl D2 │ │Local Ctrl D3 │
     │- Candidate   │ │- Candidate   │ │- Candidate   │
     │  Selection   │ │  Selection   │ │  Selection   │
     │- Execute Map │ │- Execute Map │ │- Execute Map │
     └──────────────┘ └──────────────┘ └──────────────┘
```

- **Global Controller**: Receives VNRs, decomposes into subgraphs, integrates pseudo-topologies, runs PSO
- **Local Controllers**: Select candidate nodes, upload aggregated info, execute final embedding

---

## 3. Proposed Algorithm: MP-VNE

### Step 1: Candidate Node Selection (Local Controllers)

Each local controller estimates mapping cost:

$$PreCost_{i,j,k} = CPU(n_i^v) \times C(n_k^s) + \frac{\sum_{\text{links}} \sum_{\text{candiDomain}} BW(l) \times C_{k,b}}{NoL}$$

- First term: node CPU-cost product
- Second term: average link costs based on boundary node distances
- Select nodes with **lowest estimated costs** as candidates

### Step 2: PSO with Genetic Variation (Global Controller)

**Particle Swarm Optimization** finds optimal node mapping:

**Velocity update:**
$$v_i^{new} = v_i + c_1 \cdot rand_1 \cdot (x_i^{pb} - x_i) + c_2 \cdot rand_2 \cdot (x^{gb} - x_i)$$

**Position update:**
$$x_i^{new} = x_i + v_i^{new}$$

**Key innovation - Genetic Mutation:**
- 10% probability of random position reset
- Prevents convergence to **local optima**

**Parameters:** 10 particles, 50 iterations, $c_1 = 0.3$, $c_2 = 0.3$, $\gamma = 0.4$

- **Particle position** = complete node mapping scheme
- **Fitness function** = composite cost of mapping

### Step 3: Link Mapping

- **Floyd shortest path** algorithm finds optimal physical paths
- For cross-domain links: route through boundary nodes and inter-domain connections

---

## 4. Key Innovations

| Innovation | Description |
|-----------|-------------|
| Multi-objective to single-objective | Weighted summation converts cost + delay into unified fitness |
| Estimated cost candidate selection | Reduces search space before PSO optimization |
| Genetic mutation in PSO | 10% random reset prevents local optima |
| Centralized hierarchical architecture | Partial topology awareness + scalability |

---

## 5. Experimental Setup

### Physical Network
- **4 domains**, 30 nodes each (2 boundary nodes per domain)
- Node CPU: Uniform [100-300], Cost: [1-10]
- Link BW: [1000-3000], Cost: [1-10]
- 50% connection probability

### Virtual Requests
- 6 nodes per request, CPU demand: [1-10]
- Bandwidth: [1-10] per link
- Arrival: Poisson (avg 10 per 100 time units)
- Lifetime: Exponential (avg 1000 time units)

### Baselines
| Algorithm | Approach |
|-----------|----------|
| MC-VNM | Kruskal MST with greedy link prioritization |
| VNE-PSO | Standard PSO with boundary node preference |
| LID-VNE | Distributed architecture, bandwidth matrix decomposition |

---

## 6. Results

### Acceptance Rate
- **MP-VNE & MC-VNM**: Steady ~60%
- **VNE-PSO & LID-VNE**: Decline to ~30%

### Average Mapping Cost
| Algorithm | Cost Range |
|-----------|-----------|
| **MP-VNE** | **650-750 (best)** |
| VNE-PSO | Mid-range |
| LID-VNE | ~1150 |
| MC-VNM | ~1400 (worst) |

### Mapping Delay
| Algorithm | Delay |
|-----------|-------|
| **MP-VNE** | **~460 (minimum)** |
| VNE-PSO | ~600 |
| LID-VNE | 650-700 |
| MC-VNM | ~800 |

### Composite Cost
- **MP-VNE < 650** (best)
- VNE-PSO ~950, LID-VNE ~1000, MC-VNM > 1250

---

## 7. Takeaways for Slides

1. VNE in multi-domain is harder due to cross-domain coordination
2. PSO alone gets stuck in local optima -> add genetic mutation
3. Candidate pre-selection drastically reduces search space
4. Centralized architecture enables global optimization with local execution
5. MP-VNE achieves best cost, delay, AND acceptance rate simultaneously
