# Design — Chương 3 "Phương pháp đề xuất" (RL-OA-MP-VNE)

Date: 2026-06-04
Target file: `thesis/Chuong/3_De_xuat.tex` (~15 pages)
Approach: **C — committed 5-section skeleton + foregrounded design rationale.**
Language: chapter written in Vietnamese (matches Ch1–2); this spec is in English for precision.

---

## 0. Guiding decisions (from brainstorming)

1. **Truth source:** Ch3 describes the *actual V19 code*, not the outdated Ch1–2 narrative. After Ch3, a
   separate small task patches Ch1 §1.3/§1.4 and Ch2 §2.3 for consistency.
2. **Headline result:** 100-node multi-domain (V19 = 25.9% acc vs R2 baseline 24.3%, +1.6pp; revenue 928
   vs ~900; mp_vne ceiling 32.8%), with 50-node / scaling and lifetime studies as Ch5 generalization.
3. **Three foregrounded novelties:**
   - (N1) Hierarchical **multi-domain** graph encoding (intra-domain + inter-domain GCN).
   - (N2) **Freeze the ordering heads, RL-train only the candidate head** — candidate selection is the
     lever; ordering RL was flat; random ordering-head gradients corrupt the shared encoder.
   - (N3) Two-stage IL warm-start → **actor-critic** fine-tune with KL-anchor to the IL reference.

## 1. CRITICAL terminology corrections (must be reflected throughout)

These are facts verified in code; they override Ch1–2 and the first Explore agent's "TRUE PPO" claim.

| Claim in Ch1–2 / loose naming | Verified reality (cite) |
|---|---|
| "REINFORCE + **running-mean** baseline" | **Learned critic** `value_head` V(s); `--baseline=critic`, adv-norm. (`ppo_finetune.py` run-log header: `baseline=critic V(s)`) |
| "PPO fine-tune" (script name `ppo_finetune.py`) | Default `--ppo-mode reinforce` = **REINFORCE + critic + KL + entropy = vanilla actor-critic**. The headline 100-node and 50-node runs used this mode (run-log prints no clip-ε / no K-epochs). (`ppo_finetune.py:393-394`) |
| Clipped PPO is the method | Clipped PPO exists only as the **optional** `--ppo-mode ppo` variant (clip ε=0.2, K epochs, importance ratio). Not the headline run. Used in Ch5 as the "PPO vs actor-critic" comparison. (`ppo_finetune.py:393-406,521-524`) |
| flat GCN, ~20k params | **Hierarchical** GCN, **55,687 params** (50-node run: 19,394 frozen + 36,293 trainable). |
| single-domain | fundamentally **multi-domain** (boundary nodes, allowed-domain sets, per-domain candidates). |
| Plackett-Luce IS the RL policy | PL is used only for **ordering**; ordering heads are **frozen** during RL. RL lever = candidate head. |

**Teacher's question mapping:** "xem PPO có bắt kịp actor-critic không" = compare `--ppo-mode ppo`
(clipped PPO) vs `--ppo-mode reinforce` (the actual headline actor-critic). The headline method *is* the
actor-critic; this becomes a Ch5 ablation/curve.

## 2. Method name

Keep **RL-OA-MP-VNE** (already in Ch1). Gloss in §3.1: "OA-MP" = the Order-Aware Multi-Path link-mapping
substrate inherited from OA-MP-VNE (multi-path splitting at commit); "RL" = the actor-critic-fine-tuned
candidate head on top of an IL warm-start.

---

## 3. Section-by-section design

### §3.1 Tổng quan giải pháp (~2 pg)
- **Design-principles paragraph (foregrounded):** state N1/N2/N3 explicitly as the chapter's thesis.
- **System block diagram** `fig:arch`: Graph Encoder (substrate hierarchical + VNR) → {node-rank head,
  link-rank head, candidate head, critic V(s)} → RL Environment (substrate state + reward). Mark which
  blocks are frozen vs trained during Stage 2.
- **Agent–environment loop for one VNR:** encode → order vnodes (PL) → per-vnode per-domain candidate
  decode → link-map (shortest path) → commit (OA multi-path) → reward.
- Forward-reference: complexity in Ch4, experiments in Ch5.

### §3.2 Bộ mã hoá đồ thị (~3.5 pg)
- **§3.2.1 Mã hoá mạng vật lý (hierarchical, N1):**
  - Intra-domain GCN: 3 layers, hidden 32, per domain. Adjacency D^{-1/2}(A+I)D^{-1/2}, bandwidth-weighted
    edges. LayerNorm + residual. (`il_mp_vne_v6.py:181-194`, `policy_network.py:54-60`)
  - Inter-domain GCN: 2 layers, hidden 32, over (D×D) domain-summary graph (max-pool of intra embeddings).
    (`il_mp_vne_v6.py:342-369`)
  - 7 substrate node features: avail_cpu, cpu_price, proc_delay, degree, avg_neighbor_bw,
    avg_neighbor_bw_price, is_boundary. (`il_mp_vne_v6.py:128-196`)
- **§3.2.2 Mã hoá yêu cầu mạng ảo:** VN-GCN 2 layers (hidden 32) + 4-head self-attention across vnodes
  (Transformer block). 7 vnode features incl. cross_domain_vlink_ratio, n_allowed_domains_ratio.
  (`policy_network.py:63-82`, `il_mp_vne_v6.py:198-243`). Separate weights from substrate encoder.
- **§3.2.3 Trộn biểu diễn substrate–VNR:** per-vnode max-pooled substrate context (intra+inter = 64-dim)
  fused with vnode embedding. (`policy_network.py:222-230`)
- **§3.2.4 Phân tích thiết kế:** GCN vs GAT (GCN chosen for compactness; note V21/V22 explore GAT); why
  hierarchical for multi-domain (no global snode visibility, respects locality); param budget ~55.7k.

### §3.3 Bộ giải mã chọn ứng viên (~3.5 pg) — CORE (N2)
- **§3.3.1 Sinh tập ứng viên:** per allowed domain, hard CPU-feasibility filter, top-K; **K=1 per domain**
  in V17/V19 strict (paper-aligned Algorithm 1). (`il_mp_vne_v16.py:86-120`)
- **§3.3.2 Candidate head (cross-attention + explicit cost prior):** query=vnode (proj to 64),
  key=snode-context+aux(4: cpu_slack_norm, cpu_cost, is_boundary, link_cost), 4 heads (head dim 16),
  residual MLP, **learned cost prior** `−cost_bias·cpu_cost − link_cost_bias·link_cost + boundary_bias·is_boundary`,
  final = attn + residual + prior. **Hard feasibility mask** (cpu_slack<0 → −1e4).
  (`policy_network.py:128-203`)
- **§3.3.3 Stochastic vs deterministic selection (the RL enabler):** training `sample_cand=True` →
  per-domain `Categorical` sample (K=1) or Gumbel-top-K (K>1) → exploration → learnable RL signal.
  Eval = per-domain argmax. State the key insight: argmax-only path gives no exploration → no learning;
  stochastic candidate selection is what lets RL beat the IL ceiling.
  (`il_mp_vne_v6.py:717-815`)
- **§3.3.4 Ánh xạ liên kết:** vlink order = PL-sampled (empirically best vs bw-desc/original).
  Decode/feasibility: **single continuous shortest path**, no splitting (fails otherwise)
  (`il_mp_vne_v6.py:1095,1155-1162`). Commit: **OA multi-path splitting**, up to 5 sub-paths
  (`oa_mp_vne/global_controller.py:66-97`). Collision masking forbids snode reuse.
- **§3.3.5 Xác suất hành động hợp thành:** total log-prob = Σ(node-order PL) + Σ(link-order PL) +
  Σ(per-vnode candidate). During Stage 2 only the candidate term carries gradients (others frozen).

### §3.4 Môi trường học tăng cường (~2.5 pg)
- **§3.4.1 Không gian trạng thái:** evolving substrate (CPU/BW availability) + current VNR + domain
  compatibility; on-policy persistent commits.
- **§3.4.2 Không gian hành động:** node-order permutation (PL), link-order permutation (PL), per-vnode
  per-domain candidate (Categorical). (Ordering frozen in Stage 2.)
- **§3.4.3 Hàm phần thưởng:** two variants. Default `cost_rel`: success = `success_bonus + λ·(cost_EMA −
  cost)/cost_EMA`, fail = negative const (100-node: bonus 1.0, λ 0.3, fail −1.0; 50-node cost-focused:
  α 0.5, λ 1.0, fail −0.5). Alt `rev_cost`: size-invariant clipped revenue/cost (V22).
  (`ppo_finetune.py:653-687`)
- **§3.4.4 Xử lý hành động không hợp lệ:** CPU mask (−1e4), collision mask (−inf), empty-domain skip.

### §3.5 Quy trình huấn luyện hai giai đoạn (~3.5 pg) — N3
- **§3.5.1 Stage 1 — IL pre-training (warm-start = R2/V17):** candidate head imitates MP-VNE/OA-MP-VNE
  via RankingTrainer (advantage-normalized actor-critic with entropy annealing 0.01→0.001 on synthetic
  VNs). Produces the R2 baseline (24.3% @100-node). (`il_mp_vne_v6.py:882-926`)
- **§3.5.2 Stage 2 — Actor-critic fine-tuning (the actual headline method):** present as actor-critic =
  policy-gradient on the candidate head + **learned critic** V(s) baseline + advantage normalization +
  **KL-anchor to frozen IL reference (β_KL=0.1)** + entropy bonus (β_H=0.01). Loss
  `L = L_pol + 0.5·L_value − 0.01·H + 0.1·KL`. Direct-decoding rollout (`--rollout direct`,
  `--target cand`). Then: **clipped-PPO variant** (`--ppo-mode ppo`: importance ratio, clip ε=0.2,
  K epochs, substrate snapshot + replay) presented as an *optional* objective — the one Ch5 compares
  against the actor-critic. (`ppo_finetune.py:186-294,719-754`)
- **§3.5.3 Freeze discipline (named subsection, N2):** freeze node/link heads; train candidate head +
  encoder + critic. *Why:* ordering RL signal is too weak/flat; random ordering-head gradients corrupt
  the shared encoder and blow up KL → success drops. (`ppo_finetune.py:443-448`; memory:
  feedback-ppo-freeze-discipline)
- **§3.5.4 Mã giả thuật toán huấn luyện (algorithm2e):** end-to-end two-stage loop.
- **§3.5.5 Thủ thuật ổn định:** grad clip 5.0, AdamW lr 3e-4 wd 1e-4, KL anchor, entropy bonus,
  per-batch advantage normalization.

### Kết chương
Bridge to Ch4 (complexity + convergence) and Ch5 (experiments incl. PPO-vs-actor-critic, ablations,
lifetime/topology scenarios).

---

## 4. Figures / algorithms to produce
- `fig:arch` — system block diagram (encoder → heads + critic → env), frozen/trained annotation.
- `fig:candidate_decode` (optional) — per-domain candidate cross-attention + mask.
- `alg:two_stage` — algorithm2e pseudocode of IL→actor-critic two-stage training.
- A small table mapping each component → its Ch5 ablation (encoder GCN/GAT, with/without IL warm-start,
  with/without candidate exploration, actor-critic vs PPO, reward cost_rel vs rev_cost).

## 5. Follow-up tasks (after Ch3 drafted)
- **Patch Ch1 §1.3/§1.4:** replace "REINFORCE + running-mean baseline + Plackett-Luce" framing with
  "IL warm-start + actor-critic fine-tune of the candidate head; PL used for ordering"; reflect
  multi-domain + hierarchical encoder; keep 3-contribution list but correct claims.
- **Patch Ch2 §2.3:** keep MDP/policy-gradient/REINFORCE background, but add a §2.3.x on **actor-critic**
  (critic baseline → advantage) and PPO clipping, since those are now the methods used. Correct the
  "đồ án dùng running-mean baseline" sentence (it's a critic baseline). Verify Plackett-Luce subsection
  is framed as ordering-only.
- **Ch5 §5.2.1:** expand baseline descriptions (MP-VNE, MP-VNE-V3/V4 paper-faithful PSO, R2/V17, V19);
  note MC-VNM/FlagVNE are cited-not-implemented unless added.

## 6. Open items to confirm while writing
- Exact 100-node reward constants vs 50-node (documented above; confirm from 100-node run log header).
- Whether `fig:arch` should be hand-drawn (TikZ) or generated; check `Hinh_ve/` for existing assets.
- Confirm the inter-domain summary pooling op (max vs mean) at `il_mp_vne_v6.py:342-369` when writing.
