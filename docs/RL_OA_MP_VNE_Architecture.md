# RL-OA-MP-VNE — Kiến trúc chi tiết

Tài liệu mô tả từng phần của thuật toán `rl_oa_mp_vne` (RL-enhanced Order-Aware Multi-Path VNE), tập trung vào cấu trúc mạng, data flow, và training loop.

**Files liên quan:**
- `algorithms/rl_oa_mp_vne/rl_oa_mp_vne.py` — orchestrator class
- `algorithms/rl_oa_mp_vne/policy_network.py` — 3 neural network heads
- `algorithms/rl_oa_mp_vne/trainer.py` — REINFORCE trainer
- `algorithms/rl_oa_mp_vne/vn_generator.py` — synthetic VN cho pretrain
- `configs/rl_oa_mp_vne.yaml` — hyperparams

**Hyperparameters mặc định:**

```yaml
policy_network:
  hidden_size: 64       # MLP hidden trong các head
  gcn_hidden: 32        # chiều embedding GCN
  learning_rate: 0.001
  gamma: 0.99

candidates:
  K: 10                 # top-K snode / vnode

training:
  pretrain_episodes: 800
  batch_size: 16
  online_k: 10          # update mỗi 10 request thật
  ...
```

---

## 1. Input / Output cấp thuật toán

### Input — `solve(substrate_network, virtual_request)`

| Tham số | Type | Mô tả |
|---|---|---|
| `substrate_network` | `SubstrateNetwork` hoặc `MultiDomainNetwork` | Mạng vật lý. Nếu single-domain, tự động wrap vào MultiDomainNetwork 1-domain. |
| `virtual_request` | `VirtualNetworkRequest` | `{id, arrival_time, lifetime, virtual_network}` |

`SubstrateNode` có: `id, cpu_capacity, cpu_price, processing_delay` (+ runtime: `available_cpu`).
`SubstrateLink` có: `source, target, bandwidth_capacity, bandwidth_price, transmission_delay` (+ runtime: `available_bw`).
`VirtualNode` có: `id, cpu_demand, allowed_domains`.
`VirtualLink` có: `source, target, bandwidth_demand`.

### Output — `EmbeddingSolution`

```python
@dataclass
class EmbeddingSolution:
    vnr_id: str
    is_successful: bool
    node_mapping: Dict[str, str]       # vnode_id → snode_id
    link_mapping: Dict[
        Tuple[str, str],
        List[Tuple[List[Tuple[str, str]], float]]
    ]                                   # vlink_key → [(path, bw), ...]
    embedding_cost: float
```

Link mapping là **multi-path**: mỗi vlink có thể được chia thành nhiều path vật lý (`Σ bw_i = bandwidth_demand`).

---

## 2. Feature Extraction

Có 3 loại feature, mỗi loại 5-chiều, được normalize về ~[0,1].

### 2.1 Domain features — `_extract_domain_features(lc)` (rl_oa_mp_vne.py:85-136)

**Cho mỗi domain**, extract:

**Node feature matrix `X` shape `(N_d, 5)`:**
```python
X[i] = [
    avail_cpu[i] / max_capacity,                       # CPU headroom
    cpu_price[i] / 10.0,                                # giá CPU
    processing_delay[i] / 10.0,                         # delay xử lý
    degree[i] / max_degree,                             # bậc đồ thị
    (neighbor_bw[i] / degree[i]) / max_bw,              # avg BW kề
]
```

**Normalized adjacency `A_norm` shape `(N_d, N_d)`:**
```python
A[u][v] = available_bw(u,v) / bandwidth_capacity(u,v)   # weighted adjacency (bw_ratio)
A[v][u] = A[u][v]                                        # symmetric
A += I                                                   # self-loop
A_norm = D^(-1/2) · A · D^(-1/2)                        # Kipf-Welling normalization
```

**Điểm đáng chú ý:** `A` không phải 0/1 mà là `bw_ratio`. Link đã cạn BW (ratio = 0) bị loại khỏi đồ thị về mặt propagation, cho encoder biết topology "sống".

### 2.2 Vnode features — `_extract_vnode_features(vn)` (rl_oa_mp_vne.py:138-167)

**Shape `(V, 5)`** với V = số vnode:
```python
vnode_feat[i] = [
    cpu_demand[i] / max_cpu_demand,        # CPU yêu cầu
    degree[i] / max_degree,                # bậc trong VN
    adj_bw[i] / max_adj_bw,                # tổng BW demand các vlink kề
    num_nodes / 20,                        # kích thước VN (context scalar)
    num_links / 40,                        # density VN (context scalar)
]
```

### 2.3 Vlink features — `_extract_vlink_features(vn)` (rl_oa_mp_vne.py:169-195)

**Shape `(L, 5)`** với L = số vlink:
```python
vlink_feat[i] = [
    bw_demand[i] / max_bw,                  # BW yêu cầu
    degree[src] / max_degree,               # bậc src
    degree[dst] / max_degree,               # bậc dst
    cpu_demand[src] / max_cpu,              # CPU src
    cpu_demand[dst] / max_cpu,              # CPU dst
]
```

### 2.4 Candidate pool — `_get_domain_inputs(vn)` (rl_oa_mp_vne.py:197-240)

Cho **mỗi vnode**, build pool các snode được phép:

```python
for vnode in vn.nodes.values():
    allowed = vnode.allowed_domains or all_domain_ids

    pool[i]       = [snode for d in allowed for snode in domain[d].nodes]
    cpu_slack[i]  = [avail_cpu(snode) - vnode.cpu_demand for snode in pool[i]]
    Xs[i], As[i]  = (X_d, A_d) of each allowed domain
```

Trả về 4 list song song:
- `per_vnode_Xs[i]`: tuple domain X tensors (để GCN encode)
- `per_vnode_As[i]`: tuple domain A tensors
- `per_vnode_pools[i]`: list `SubstrateNode` — index-align với candidate_scores
- `per_vnode_slacks[i]`: tensor `(N_total_i,)` — CPU slack

**Độ dài pool của mỗi vnode** phụ thuộc `allowed_domains`:
- Rỗng/None → gộp tất cả domain
- 1 domain → chỉ node của domain đó
- 2 domain → gộp 2 domain

---

## 3. Encoder — `GCNEncoder`

### 3.1 Code (policy_network.py:6-17)

```python
class GCNEncoder(nn.Module):
    def __init__(self, node_feat_size: int, hidden_size: int = 32):
        super().__init__()
        self.W1 = nn.Linear(node_feat_size, hidden_size, bias=True)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, X, A_norm):
        H = torch.relu(self.W1(A_norm @ X))
        H = torch.relu(self.W2(A_norm @ H))
        return H
```

### 3.2 Công thức toán

Vanilla 2-layer GCN (Kipf & Welling, ICLR'17):

```
H^(l+1) = σ( Ã · H^(l) · W^(l)ᵀ + b^(l) )
```
Trong đó:
- `Ã = D^(-1/2)·(A + I)·D^(-1/2)` — adjacency chuẩn hóa đối xứng, có self-loop
- `H^(0) = X`
- `σ = ReLU`
- `L = 2` → mỗi node "thấy" 2-hop neighbors

### 3.3 Input / Output

**Input:**
- `X: (N, 5)` — node features một domain
- `A_norm: (N, N)` — normalized weighted adjacency

**Output:**
- `H: (N, 32)` — embedding 32-chiều cho mỗi snode

### 3.4 Layer-by-layer (concrete với `N=4`)

```
Input:  X (4, 5), A_norm (4, 4)

─── Layer 1 ───
  A_norm @ X                   (4, 4) × (4, 5) = (4, 5)
  W1(result)                   (4, 5) × (5, 32) + (32,) = (4, 32)
  ReLU                         (4, 32)
  H_1 = (4, 32)

─── Layer 2 ───
  A_norm @ H_1                 (4, 4) × (4, 32) = (4, 32)
  W2(result)                   (4, 32) × (32, 32) + (32,) = (4, 32)
  ReLU                         (4, 32)
  H_2 = (4, 32)

Output: H_2 (4, 32)
```

### 3.5 Parameters

| Module | Shape | Params |
|---|---|---|
| `W1.weight` | (32, 5) | 160 |
| `W1.bias` | (32,) | 32 |
| `W2.weight` | (32, 32) | 1,024 |
| `W2.bias` | (32,) | 32 |
| **Total** | | **1,248** |

### 3.6 Role trong PolicyNetwork

Encoder là **shared**: cùng 1 instance được gọi cho mỗi domain trong mỗi forward pass, output được dùng bởi **cả 3 head**.

- `node_head` và `link_head` dùng **mean-pooled** `H.mean(dim=0)` làm substrate context
- `candidate_head` dùng **full** `H` làm keys cho attention

### 3.7 Hạn chế

| Vấn đề | Ảnh hưởng |
|---|---|
| Chỉ 2 layer | Snode cách >2 hop không share info |
| Uniform neighbor aggregation (chỉ weighted bởi bw_ratio, không attention) | Không phân biệt neighbor quan trọng cho từng vnode |
| Giả định undirected | OK với project hiện tại |
| Không batch domain | Forward tuần tự từng domain |

Upgrade path: GAT (attention weights học được), GIN (expressive hơn).

---

## 4. Node Head — ranking MLP cho vnode

### 4.1 Code (policy_network.py:80-87)

```python
self.node_head = nn.Sequential(
    nn.Linear(vnode_feat_size + gcn_hidden, hidden_size),   # 37 → 64
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),                    # 64 → 64
    nn.ReLU(),
    nn.Linear(hidden_size, 1),                              # 64 → 1
)
```

### 4.2 Input / Output

**Input:**
- `vnode_feats: (V, 5)` — 5-dim features của V vnodes
- `substrate_ctx: (V, 32)` — per-vnode substrate context (GCN embeddings pooled qua allowed domains)

**Concat:** `(V, 37)` — mỗi hàng là `[vnode_feat[i] ‖ substrate_ctx[i]]`

**Output:**
- `node_scores: (V,)` — 1 logit cho mỗi vnode

### 4.3 Substrate context

```python
# policy_network.py:132-146
per_vnode_pools = []
node_contexts = []
for i in range(num_vnodes):
    full_list = self._embed_domains(Xs[i], As[i])     # list of (N_d, 32)
    pool = torch.cat(full_list, dim=0)                # (N_total_i, 32)
    per_vnode_pools.append(pool)
    node_contexts.append(pool.mean(dim=0))            # (32,)

substrate_ctx = torch.stack(node_contexts)            # (V, 32)
```

Mỗi vnode có substrate context riêng = **mean GCN embedding** qua các allowed domains. Vnode với `allowed_domains=[d_0]` thấy context khác vnode với `allowed_domains=[d_0, d_1]`.

### 4.4 Parameters

| Layer | Shape | Params |
|---|---|---|
| `node_head.0.weight` | (64, 37) | 2,368 |
| `node_head.0.bias` | (64,) | 64 |
| `node_head.2.weight` | (64, 64) | 4,096 |
| `node_head.2.bias` | (64,) | 64 |
| `node_head.4.weight` | (1, 64) | 64 |
| `node_head.4.bias` | (1,) | 1 |
| **Total** | | **6,657** |

### 4.5 Score dùng làm gì

Score → softmax → Plackett-Luce sample **permutation** của V vnodes:

```python
ordered_vnodes, node_lp = _plackett_luce_sample(node_scores, vnodes)
# ordered_vnodes: List[VirtualNode] — full permutation
# node_lp: List[Tensor] — log_probs (length V) cho REINFORCE
```

Thứ tự này dùng cho:
- PSO search mapping (vnode mức priority cao xử lý trước)
- Commit mapping order

---

## 5. Link Head — ranking MLP cho vlink

### 5.1 Code (policy_network.py:89-96)

```python
self.link_head = nn.Sequential(
    nn.Linear(vlink_feat_size + gcn_hidden, hidden_size),   # 37 → 64
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),                    # 64 → 64
    nn.ReLU(),
    nn.Linear(hidden_size, 1),                              # 64 → 1
)
```

### 5.2 Input / Output

**Input:**
- `vlink_feats: (L, 5)`
- `link_ctx: (L, 32)` — **global** substrate context (mean qua tất cả vnode contexts)

**Concat:** `(L, 37)`

**Output:**
- `link_scores: (L,)`

### 5.3 Link context (policy_network.py:149-151)

```python
link_ctx = substrate_ctx.mean(dim=0).unsqueeze(0).expand(num_vlinks, -1)
# shape: (L, 32)
```

Khác node_head ở chỗ: link_ctx **giống nhau cho mọi vlink** — chỉ 1 global substrate signature. Có lý do đơn giản: vlink không có khái niệm `allowed_domains` riêng.

### 5.4 Parameters

Giống hệt `node_head`: **6,657**.

### 5.5 Score dùng làm gì

Score → softmax → Plackett-Luce sample **permutation** của L vlinks:

```python
ordered_links, link_lp = _plackett_luce_sample(link_scores, link_items)
```

Thứ tự này dùng khi **commit bandwidth** (vlink mức priority cao được cấp BW trước — giảm khả năng vlink lớn bị "kẹt" không có path đủ).

---

## 6. Candidate Head — cross-attention scorer (phần NEW)

### 6.1 Motivation

Thay vì đưa **toàn bộ** snode feasible vào PSO (search space O(N^V)), ta học cách chọn **top-K ứng viên** cho từng vnode dựa trên:
- Đặc tính bản thân vnode (cpu_demand, degree, etc.)
- Đặc tính snode (GCN embedding)
- Feasibility (CPU slack)

Cảm hứng: **Attention Model** (Kool et al., ICLR'19) — SOTA cho combinatorial optimization.

### 6.2 Code (policy_network.py:20-67)

```python
class CandidateHead(nn.Module):
    def __init__(self, vnode_feat_size, gcn_hidden, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale = hidden_size ** 0.5
        self.query_proj = nn.Linear(vnode_feat_size, hidden_size, bias=True)
        self.key_proj = nn.Linear(gcn_hidden + 1, hidden_size, bias=True)
        self.residual = nn.Sequential(
            nn.Linear(vnode_feat_size + gcn_hidden + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, vnode_feat, snode_embeds, cpu_slack):
        n = snode_embeds.shape[0]
        denom = cpu_slack.abs().max().clamp(min=1.0)
        slack_norm = (cpu_slack / denom).unsqueeze(-1)             # (N, 1)

        key_input = torch.cat([snode_embeds, slack_norm], dim=-1)  # (N, 33)
        q = self.query_proj(vnode_feat)                             # (hidden,)
        k = self.key_proj(key_input)                                # (N, hidden)
        attn = (k @ q) / self.scale                                 # (N,)

        q_expand = vnode_feat.unsqueeze(0).expand(n, -1)
        mlp_input = torch.cat([q_expand, snode_embeds, slack_norm], dim=-1)
        residual = self.residual(mlp_input).squeeze(-1)             # (N,)

        scores = attn + residual
        mask = cpu_slack < 0
        scores = scores.masked_fill(mask, float("-inf"))
        return scores
```

### 6.3 Công thức toán

```
Input cho 1 vnode:
    vnode_feat    ∈ ℝ^5
    snode_embeds  ∈ ℝ^(N × 32)     (từ GCN, N = |allowed pool|)
    cpu_slack     ∈ ℝ^N

slack_norm = cpu_slack / max(|cpu_slack|, 1)      ∈ ℝ^N

─── Branch 1: scaled dot-product attention ───
Query:  q = W_q · vnode_feat + b_q               ∈ ℝ^64
Keys:   K = W_k · [snode_embeds ‖ slack_norm] + b_k   ∈ ℝ^(N×64)
Attn:   a_j = (K_j · q) / √64                    ∈ ℝ

─── Branch 2: residual MLP ───
input_j = [vnode_feat ‖ snode_embeds_j ‖ slack_norm_j]     ∈ ℝ^38
r_j     = MLP(input_j)                                     ∈ ℝ

─── Combine + mask ───
score_j = a_j + r_j
if cpu_slack_j < 0: score_j = -∞
```

### 6.4 Tại sao cần cả attention VÀ residual MLP?

| Branch | Lợi thế | Hạn chế một mình |
|---|---|---|
| **Scaled dot-product** | O(N·d) complexity, vectorize tốt, theory sound | Bilinear trong (q,k), không expressive cho interaction phức tạp |
| **Residual MLP** | Có thể học interaction nonlinear bất kỳ | O(N·d^2) params theo hidden, không có scaling guarantee |

Kết hợp → có "baseline signal" từ attention + "correction" từ MLP.

### 6.5 Feasibility mask

Dòng `scores.masked_fill(cpu_slack < 0, -inf)` là **hard constraint**:
- `cpu_slack < 0` nghĩa là `available_cpu(snode) < cpu_demand(vnode)` → snode KHÔNG thể host vnode
- Sau softmax, `-inf` → `e^(-inf) = 0` → xác suất 0 → không bao giờ được sample

Ưu điểm hơn "mask sau sample": không cần reject sampling, không bias gradient.

### 6.6 Input / Output

**Input (per vnode):**

| Tên | Shape | Nghĩa |
|---|---|---|
| `vnode_feat` | (5,) | 5 features của 1 vnode |
| `snode_embeds` | (N, 32) | GCN-encoded snode (N = |allowed pool|) |
| `cpu_slack` | (N,) | `available_cpu - cpu_demand` |

**Output:**
- `scores: (N,)` — logit cho mỗi snode; `-inf` chỗ infeasible

### 6.7 Parameters

| Layer | Shape | Params |
|---|---|---|
| `query_proj.weight` | (64, 5) | 320 |
| `query_proj.bias` | (64,) | 64 |
| `key_proj.weight` | (64, 33) | 2,112 |
| `key_proj.bias` | (64,) | 64 |
| `residual.0.weight` | (64, 38) | 2,432 |
| `residual.0.bias` | (64,) | 64 |
| `residual.2.weight` | (1, 64) | 64 |
| `residual.2.bias` | (1,) | 1 |
| **Total** | | **5,121** |

### 6.8 Score dùng làm gì

Sau khi có `candidate_scores[i]: (N_i,)` cho mỗi vnode i:

```python
K = config["candidates"]["K"]                     # mặc định 10
picked_indices, log_probs = _plackett_luce_topk(scores_i, K)
candidate_nodes[i] = [pool[i][j] for j in picked_indices]   # top-K snode
```

Top-K candidate được đưa vào PSO search (thu hẹp search space drastic).

---

## 7. PolicyNetwork.forward — tích hợp 3 head

### 7.1 Code (policy_network.py:113-162)

```python
def forward(self, vnode_feats, vlink_feats,
            per_vnode_domain_feats, per_vnode_domain_adjs,
            per_vnode_cpu_slacks=None):

    num_vnodes = vnode_feats.shape[0]
    num_vlinks = vlink_feats.shape[0]

    # Step 1: GCN encode per vnode's allowed domains, build pools
    per_vnode_pools = []
    node_contexts = []
    for i in range(num_vnodes):
        full_list = self._embed_domains(per_vnode_domain_feats[i],
                                         per_vnode_domain_adjs[i])
        pool = torch.cat(full_list, dim=0)                   # (N_total_i, 32)
        per_vnode_pools.append(pool)
        node_contexts.append(pool.mean(dim=0))               # (32,)

    substrate_ctx = torch.stack(node_contexts)               # (V, 32)

    # Step 2: node_head
    node_input = torch.cat([vnode_feats, substrate_ctx], dim=1)   # (V, 37)
    node_scores = self.node_head(node_input).squeeze(-1)     # (V,)

    # Step 3: link_head
    link_ctx = substrate_ctx.mean(dim=0).unsqueeze(0).expand(num_vlinks, -1)
    link_input = torch.cat([vlink_feats, link_ctx], dim=1)   # (L, 37)
    link_scores = self.link_head(link_input).squeeze(-1)     # (L,)

    # Step 4: candidate_head (optional)
    candidate_scores = None
    if per_vnode_cpu_slacks is not None:
        candidate_scores = []
        for i in range(num_vnodes):
            scores = self.candidate_head(vnode_feats[i],
                                          per_vnode_pools[i],
                                          per_vnode_cpu_slacks[i])
            candidate_scores.append(scores)                  # (N_i,)

    return node_scores, link_scores, candidate_scores
```

### 7.2 Tổng params toàn mạng

| Component | Params |
|---|---|
| GCN Encoder | 1,248 |
| node_head | 6,657 |
| link_head | 6,657 |
| candidate_head | 5,121 |
| **Grand total** | **19,683** |

~77 KB nếu save float32. Nhỏ gọn → train nhanh, dễ deploy.

---

## 8. Plackett-Luce Sampling

### 8.1 Công thức toán

Plackett-Luce model cho permutation:

```
P(π | s) = Π  exp(s_{π_k}) / Σ_{j∈remaining}  exp(s_j)
          k=1..K
```

Tức là sample phần tử đầu theo softmax toàn bộ, loại khỏi pool, rồi sample phần tử kế theo softmax của pool còn lại, lặp cho đến khi sample xong (hoặc cắt ở K).

### 8.2 Full permutation — `_plackett_luce_sample` (rl_oa_mp_vne.py:265-285)

Dùng cho `node_scores` và `link_scores`:

```python
for _ in range(len(items)):
    dist = Categorical(logits=remaining_scores)
    pos = dist.sample()
    log_probs.append(dist.log_prob(pos))
    # remove chosen from pool
    mask = ones(len(remaining)); mask[pos] = False
    remaining_scores = remaining_scores[mask]
    remaining_indices = [...]
```

Output: `(permutation, log_probs)` với `len(log_probs) = len(items)`.

### 8.3 Top-K with mask — `_plackett_luce_topk` (rl_oa_mp_vne.py:287-319)

Dùng cho `candidate_scores` — cắt sớm tại K và skip `-inf`:

```python
finite_count = isfinite(scores).sum()
k = min(k, finite_count)           # cắt theo feasible count
if k == 0: return [], []

for _ in range(k):
    dist = Categorical(logits=remaining)
    pos = dist.sample()
    log_probs.append(dist.log_prob(pos))
    # remove from pool
    ...
```

**Tại sao `Categorical(logits=remaining)` xử lý `-inf` đúng:**
- `softmax(logits)` với logit `-inf` → prob = 0 → không bao giờ sample.

---

## 9. Training — REINFORCE with baseline

### 9.1 Trainer code (trainer.py)

```python
class RankingTrainer:
    def __init__(self, policy, lr=0.001, gamma=0.99, batch_size=16):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.buffer: List[Tuple[List[Tensor], float]] = []

    def record(self, log_probs, reward):
        self.buffer.append((log_probs, reward))

    def update(self):
        rewards = [r for _, r in self.buffer]
        baseline = sum(rewards) / len(rewards)              # running mean

        total_loss = tensor(0.0)
        for log_probs, reward in self.buffer:
            advantage = reward - baseline
            for lp in log_probs:
                total_loss -= lp * advantage

        total_loss /= len(self.buffer)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.buffer.clear()
```

### 9.2 REINFORCE với baseline

Policy gradient:

```
∇_θ J(θ) = E[ (R - b) · ∇_θ log π_θ(a|s) ]
```

- `R` = reward (revenue/cost)
- `b` = baseline (running mean của batch)
- Subtract baseline giảm variance mà không bias estimator

Loss:

```
L = -E[ (R - b) · log π(a|s) ]
  = -(1/N) · Σ_epi  (R_epi - b) · Σ_step  log π(a_step | s_step)
```

### 9.3 Credit assignment

Với 3 head, **cùng 1 reward**, **cùng 1 advantage**:

```python
# Trong 1 episode:
all_log_probs = node_lp + link_lp + cand_lp       # concat

# Trong update():
advantage = reward - baseline
for lp in all_log_probs:
    loss -= lp * advantage
```

Hệ quả:
- Head nào contribute nhiều hơn → gradient lớn hơn (qua backprop tự nhiên)
- Tuy nhiên, "credit assignment" là **implicit** — nếu reward xấu, cả 3 head đều bị "phạt" như nhau
- Giảm variance qua baseline, không giải quyết credit assignment

### 9.4 On-policy sampling

Quan trọng: log_probs phải match **action đã execute**. Code trong `solve()`:

```python
self.policy.train()
ordered_vnodes, ordered_links, candidate_nodes, log_probs = \
    self.rank_all_nn(vnetwork, sample=True)

# ... embed thật với chính action đó ...

reward = revenue / cost
self._record_online(log_probs, reward)
```

Không re-sample, không forward lại — log_probs chính là từ policy tại thời điểm quyết định.

---

## 10. End-to-End Flow — `solve()`

### 10.1 Lifecycle

```
solve(substrate, request):
    (lần đầu tiên:)
        _init_controller(substrate)        # wrap thành MultiDomainNetwork
        _pretrain(substrate)                # 800 synthetic episodes
            for ep in range(800):
                generate_random_vn
                reset substrate allocations
                ordered_v, ordered_l, cand, lps = rank_all_nn(sample=True)
                reward = _try_embedding(...) (rollback sau)
                trainer.record(lps, reward)
                mỗi 16 episode: trainer.update()

    _release_expired(request.arrival_time)
    global_controller.clear_caches()

    # forward + sample on-policy
    ordered_v, ordered_l, cand_nodes, log_probs = rank_all_nn(vn, sample=True)

    # PSO search mapping từ top-K candidates
    best_mapping = _pso(cand_nodes, vlink_indices, ordered_v)

    # Commit với multi-path splitting
    try:
        vlink_paths = _commit_mapping_ordered(best_mapping, vn, ordered_l)
        solution.is_successful = True
    except ValueError:
        self._record_online(log_probs, -1.0)
        return solution (failure)

    # Reward + online learning
    reward = revenue / cost
    self._record_online(log_probs, reward)
    # mỗi 10 request: trainer.update()

    return solution
```

### 10.2 PSO (`_pso`, rl_oa_mp_vne.py:471-526)

- 20 particles, 15 iterations
- Mỗi particle = list indices `[idx_0, idx_1, ...]` với `candidate_nodes[i][idx_i]` là snode được chọn cho vnode i
- Velocity update standard PSO: `v = w·v + c1·r1·(pbest - x) + c2·r2·(gbest - x)`
- Mutation random để tránh local optima
- `_repair_particle` đảm bảo injective (không 2 vnode cùng snode)
- Fitness: node_cost + link_cost (với link_cost = hop_count × bw, match evaluation metric)

### 10.3 Multi-path commit (`_commit_mapping_ordered`)

Port từ `mp_vne.commit_mapping`:

```python
for vlink_key, vlink in ordered_links:          # thứ tự từ link_head
    demand_remaining = vlink.bandwidth_demand
    allocated_paths = []
    max_paths = 5

    while demand_remaining > 0.001 and len(allocated_paths) < max_paths:
        min_required = min(demand_remaining * 0.1, 1.0, demand_remaining)
        path = global_controller.shortest_path(src, dst, bw_required=min_required)
        if not path: break
        path_bw = min(link.available_bw for link in path)
        allocated = min(demand_remaining, path_bw)
        for link in path:
            link.available_bw -= allocated
        allocated_paths.append((path, allocated))
        demand_remaining -= allocated

    if demand_remaining > 0.001:
        raise ValueError(...)   # trigger rollback
```

Max 5 path per vlink, tự động chuyển path nếu bottleneck đầy.

### 10.4 Reward formula (rl_oa_mp_vne.py:400-415)

```python
revenue = Σ cpu_demand + Σ bw_demand
cost = Σ cpu_demand · cpu_price + Σ bw · len(path)       # match visualize metric
reward = revenue / (cost + 1e-6)
```

Cost là **hop-count based** cho link (không phải price-based) — align với metric evaluation trong `evaluation/visualize_results.py`.

---

## 11. Tóm tắt kiến trúc

```
                           PolicyNetwork (19,683 params)
                  ┌──────────────────────────────────────┐
                  │     GCNEncoder (1,248 params)         │
                  │     shared — 2 layers, hidden 32      │
                  │              │                        │
                  │    ┌─────────┼─────────┐              │
                  │    ▼         ▼         ▼              │
                  │ node_head  link_head  candidate_head  │
                  │ 6,657       6,657       5,121         │
                  └──────────────────────────────────────┘

                  Input:  vn + substrate_state
                  Output: (V,) + (L,) + List[(N_i,)]   logits
                            │      │         │
                            ▼      ▼         ▼
                         perm_V  perm_L   top-K   (Plackett-Luce sample)
                            └──────┴─────────┘
                                   ▼
                         log_probs (concat)
                                   ▼
                              PSO + commit
                                   ▼
                           reward (rev / cost)
                                   ▼
                        REINFORCE update toàn mạng
```

**Đặc điểm chính:**
- 1 mạng, 3 head, encoder chia sẻ
- Shared reward signal (không credit assignment explicit)
- End-to-end differentiable qua log_probs của Plackett-Luce sampling
- Feasibility mask cứng trong candidate head (không sample snode thiếu CPU)
- Multi-path commit (port từ mp_vne) — cho phép 1 vlink ảo dùng tối đa 5 path vật lý

**Kết quả so với mp_vne baseline (scenario_large 500 VNRs):**
- R2C: 0.412 vs 0.410 (+0.5%)
- Avg cost: 966.71 vs 978.21 (−1.2%)
- Avg delay: 13.99 vs 13.03 (+7.4% — do reward không include delay)
