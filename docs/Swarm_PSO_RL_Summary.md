# Swarm Reinforcement Learning Using PSO

> **Repository:** [Srini-Rohan/Swarm-Reinforcement-Learning-Using-PSO](https://github.com/Srini-Rohan/Swarm-Reinforcement-Learning-Using-PSO)
> **Method:** Combining Deep Q-Networks (DQN) with Particle Swarm Optimization (PSO)

---

## 1. Problem Description

*(Same VNE context as MP-VNE - see MP_VNE_Summary.md for full formulation)*

**Core problem addressed:** Traditional single-agent reinforcement learning converges slowly. Can **PSO's information-sharing mechanisms** among multiple Q-learning agents accelerate convergence to optimal policies?

---

## 2. Core Method: PSO + Deep Q-Networks

The framework replaces traditional tabular Q-learning with **neural networks** for continuous state spaces, and uses **PSO communication** among multiple DQN agents.

```
┌─────────────────────────────────────────────┐
│              DQN_PSO_Agent (Swarm)           │
│                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │ Agent 1  │ │ Agent 2  │ │ Agent 3  │ .. │
│  │  (DQN)   │ │  (DQN)   │ │  (DQN)   │    │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘    │
│       │             │             │          │
│       └─────────────┼─────────────┘          │
│                     ▼                        │
│         PSO Information Sharing              │
│    - Personal best (pbest) models            │
│    - Global best (gbest) model               │
│    - Velocity/position updates               │
└─────────────────────────────────────────────┘
```

### Key Idea
- Each agent = a **particle** in PSO
- Agents share their **best-performing models** (not just scalar values)
- After training episodes, Q-values are updated using PSO equations
- Collective intelligence accelerates convergence

---

## 3. Architecture Components

### DQN_PSO_Agent (Swarm Manager)
- Orchestrates multiple Q-learning agents
- Maintains **experience replay** memory across swarm
- Tracks **individual best** (pbest) and **global best** (gbest) models
- Applies PSO velocity/position updates to Q-value parameters

### DQNAgent (Individual Particle)
- Neural network-based Q-function approximation
- Standard DQN components: action selection, target network, experience buffer
- Each agent explores independently but shares discoveries

### PSO Update Mechanism

After each episode, agent parameters are updated:

$$v_i^{new} = w \cdot v_i + c_1 \cdot r_1 \cdot (pbest_i - x_i) + c_2 \cdot r_2 \cdot (gbest - x_i)$$

$$x_i^{new} = x_i + v_i^{new}$$

Where $x_i$ represents the neural network weights of agent $i$.

---

## 4. Algorithm Flow

```
1. Initialize N agents (particles) with random DQN weights
2. For each episode:
   a. Each agent independently plays the environment
   b. Each agent trains its DQN via experience replay
   c. Evaluate each agent's performance
   d. Update personal best (pbest) if improved
   e. Update global best (gbest) if any agent improved
   f. Apply PSO velocity + position updates to DQN weights
3. Return global best agent's policy
```

---

## 5. Experimental Setup

- **Environment:** OpenAI Gym CartPole-v1
- **Task:** Balance a pole on a moving cart
- **Dependencies:** Python 3.8, PyTorch, Gym, NumPy, Matplotlib

### Configurations
| Parameter | Single DQN | Swarm PSO-DQN |
|-----------|-----------|---------------|
| Agents | 1 | 4 |
| Architecture | Standard DQN | DQN + PSO updates |

---

## 6. Results

| Experiment | Convergence (episodes) |
|-----------|----------------------|
| Single DQN Agent | **130 episodes** |
| 4-Agent PSO Swarm (best agent) | **117 episodes** |

**Improvement: ~10% faster convergence** through collaborative learning.

### Key Observations
- Multi-agent collaboration outperforms isolated single-agent learning
- PSO enables agents to leverage each other's discoveries
- Global best model benefits from diverse exploration strategies

---

## 7. Relevance to VNE

While this implementation uses CartPole, the **PSO + RL combination** is directly applicable to VNE:

| Aspect | CartPole Demo | VNE Application |
|--------|--------------|-----------------|
| State | Cart position/velocity | Network resource state |
| Action | Left/right | Node/link mapping decisions |
| Reward | Pole balanced | Embedding cost/revenue |
| PSO benefit | Faster convergence | Better mapping solutions |

The MP-VNE algorithm uses a similar PSO approach but applied directly to the **node mapping optimization** rather than to RL weight updates.

---

## 8. Takeaways for Slides

1. PSO can enhance RL by enabling **multi-agent knowledge sharing**
2. Each agent explores differently -> swarm finds better solutions faster
3. Neural networks replace tabular Q-learning for continuous state spaces
4. ~10% convergence speedup demonstrated on CartPole benchmark
5. Same PSO principles used in MP-VNE for optimizing node mappings
6. Trade-off: more compute (N agents) but faster/better convergence
