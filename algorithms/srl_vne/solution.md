# Swarm Reinforcement Learning using PSO for Virtual Network Embedding (VNE)

## 1. Overview

Bài toán Virtual Network Embedding (VNE) yêu cầu ánh xạ một mạng ảo (Virtual Network - VN) lên mạng vật lý (Substrate Network - SN) sao cho tối ưu tài nguyên và thỏa mãn các ràng buộc.

Trong bài này, chúng tôi đề xuất một phương pháp **Swarm Reinforcement Learning (SRL)** kết hợp:

- Deep Q-Network (DQN)
- Particle Swarm Optimization (PSO)

Ý tưởng chính:

- Mỗi agent học embedding như một RL agent
- Nhiều agent hợp thành swarm
- PSO giúp chia sẻ tri thức giữa các agent → tăng tốc hội tụ

---

## 2. Motivation

### Vấn đề của VNE truyền thống:

- Greedy → dễ mắc kẹt local optimum
- RL đơn agent → hội tụ chậm
- Multi-agent RL → khó phối hợp

### Insight:

- PSO có khả năng tìm global optimum nhờ chia sẻ thông tin swarm :contentReference[oaicite:0]{index=0}
- RL có khả năng học policy từ dữ liệu

→ Kết hợp:

> RL = learning  
> PSO = global search + knowledge sharing

---

## 3. Related Work

### 3.1 PSO (Particle Swarm Optimization)

- Particle = nghiệm (mapping VN → SN)
- Update dựa trên:
  - pBest (cá nhân)
  - gBest (toàn swarm)

### 3.2 RL + PSO

- RL cải thiện khả năng thích nghi của PSO :contentReference[oaicite:1]{index=1}
- Multi-swarm + RL giúp tránh local optimum :contentReference[oaicite:2]{index=2}

### 3.3 Swarm RL (Repo base)

Repo sử dụng:

- Multi DQN agents
- PSO để update Q-values và model weights :contentReference[oaicite:3]{index=3}

---

## 4. Problem Formulation

### Input:

- Substrate Network:
  - Graph \( G_s = (N_s, E_s) \)
- Virtual Network request:
  - Graph \( G_v = (N_v, E_v) \)

### Output:

- Mapping:
  - Node mapping: \( N_v \rightarrow N_s \)
  - Link mapping: \( E_v \rightarrow Paths \)

### Objective:

Maximize:

- Revenue
- Acceptance ratio

Minimize:

- Cost
- Resource fragmentation

---

## 5. Proposed Method

## 5.1 Architecture

+----------------------+
| Multiple DQN Agents |
+----------+-----------+
|
v
Swarm Controller (PSO)
|
v
Shared Knowledge (gBest)

---

## 5.2 Agent Design

### State:

- Remaining CPU, bandwidth
- VN node features
- Current partial embedding

### Action:

- Chọn node vật lý cho 1 VN node

### Reward:

- - nếu mapping thành công
- - nếu fail constraint
- - tối ưu tài nguyên

---

## 5.3 Swarm Learning (PSO Integration)

Mỗi agent có:

- Local best (pBest)
- Global best (gBest)

Update:

- Thay vì chỉ update Q-learning:
  → thêm influence từ swarm:

Q_new = Q + α \* (reward + γ max(Q') - Q)

- β \* (pBest - current)
- δ \* (gBest - current)

Ý nghĩa:

- RL → học từ experience
- PSO → học từ swarm

---

## 5.4 Training Process

1. Initialize N agents
2. Với mỗi episode:
   - Mỗi agent embed VN
   - Lưu experience
   - Train DQN
3. Cập nhật:
   - pBest (agent tốt nhất của mỗi agent)
   - gBest (agent tốt nhất toàn swarm)
4. Apply PSO update
5. Lặp lại

---

## 6. Innovation Ideas (Quan trọng cho bài của bạn)

### Idea 1: Dynamic Swarm Size

- Tăng agent khi khó
- Giảm agent khi dễ

### Idea 2: Topology-aware PSO

- Local vs Global topology
- RL chọn topology (inspired by research) :contentReference[oaicite:4]{index=4}

### Idea 3: Multi-objective VNE

- Reward gồm:
  - latency
  - energy
  - load balance

### Idea 4: Hierarchical Swarm

- Level 1: node mapping
- Level 2: link mapping

### Idea 5: Hybrid với heuristic

- Agent học nhưng vẫn dùng:
  - shortest path
  - BFS pruning

---

## 7. Evaluation

### Baselines:

- Greedy VNE
- DQN single agent
- PSO only
- RL only

### Metrics:

- Acceptance ratio
- Revenue/Cost ratio
- Convergence speed

---

## 8. Expected Results

- Swarm RL hội tụ nhanh hơn single RL
- Tránh local optimum tốt hơn
- Tăng acceptance ratio

(Repo gốc cho thấy multi-agent converge nhanh hơn single agent) :contentReference[oaicite:5]{index=5}

---

## 9. Future Work

- Apply PPO thay vì DQN
- Graph Neural Network cho state encoding
- Real-world dataset (ISP topology)

---

## 10. Conclusion

Phương pháp Swarm RL + PSO:

- Kết hợp learning + optimization
- Phù hợp với bài toán NP-hard như VNE
- Có tiềm năng outperform các phương pháp truyền thống

---

## 11. Keywords

- VNE
- Reinforcement Learning
- Particle Swarm Optimization
- Multi-agent systems
- Network virtualization
