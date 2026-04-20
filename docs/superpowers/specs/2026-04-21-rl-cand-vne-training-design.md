# RL-Cand-VNE: Candidate-Selection Policy with Resource-State-Aware Training

**Status:** Design approved, pending implementation plan
**Date:** 2026-04-21
**Author:** Du Van Nguyen (with Claude)
**Supersedes:** Nothing — new sibling algorithm, existing `rl_oa_mp_vne` untouched

---

## 1. Motivation

The current `rl_oa_mp_vne` policy has two limitations:

1. **Static training state.** Pre-training calls `reset_allocations()` before every episode, so the policy only ever sees a fresh, full-capacity substrate. It cannot learn how to behave under partial utilization — which is the state it actually faces at inference.
2. **Ranking-only output.** The network outputs priority scores for ordering vnodes/vlinks; final snode selection is still delegated to a hand-crafted `get_candidates()` heuristic + PSO. The network never learns *which snodes* to place onto.

We want a policy that (a) learns a general strategy across many resource-utilization states and (b) directly outputs **a candidate set of substrate nodes per virtual node**, so the neural network is responsible for candidate selection, not just ordering.

**Constraint.** The substrate network is large enough that the GCN encoder cannot process the whole substrate in one forward pass. The encoder must work **one domain at a time**.

---

## 2. Scope

In scope:
- New algorithm `rl_cand_vne` living at `algorithms/rl_cand_vne/`.
- New policy network: per-vnode domain + snode attention heads over per-domain encodings.
- Offline training script with varied resource states and a saveable checkpoint.
- Online fine-tuning inside `solve()` on real VNRs.
- Loss = REINFORCE (cost-minimizing) + supervised auxiliary on committed snodes.
- JSONL training logs + plotting helper.
- Unit and integration tests.
- Registry entry and YAML config.

Out of scope (explicitly):
- Multi-substrate / multi-topology training (single substrate per training run; design is extensible later).
- Replacing `rl_oa_mp_vne` (kept as a comparison baseline).
- Replacing PSO (new candidate sets still feed the existing PSO + commit pipeline from `oa_mp_vne`).

---

## 3. Architecture Overview

On each request, the policy runs three stages:

1. **VN encoder** — 2-layer GCN over the virtual graph. Produces contextual embedding `h_A ∈ R^d` for every vnode A. This is how "vnode A watches other vnodes".
2. **Domain encoder** — 2-layer GCN, run **once per allowed domain** of the request. For each domain d: per-snode embeddings `E_d ∈ R^{n_d × d}` and pooled `g_d = mean(E_d)`. Respects the "one-domain-at-a-time" constraint; no cross-domain message passing inside the encoder.
3. **Per-vnode heads** — for each vnode A:
   - **Domain head**: dot-product attention with query `h_A`, keys/values `{g_d : d ∈ allowed_domains(A)}` → distribution over A's allowed domains. Sample (training) / argmax (inference) → `d*`.
   - **Snode head**: dot-product attention with query `[h_A ; g_{d*}]`, keys/values `{e_s : s ∈ d*}` → logits over snodes in `d*`. Plackett–Luce top-K sampling (training) / deterministic top-K (inference) → candidate set of size K.

The candidate sets feed the **existing PSO + `_commit_mapping_ordered`** from `oa_mp_vne` (imported, not copy-pasted). Processing order remains hand-crafted: vnodes by degree-desc, vlinks by BW-desc. K defaults to 5 and is config-tunable.

### 3.1 Rationale

- Separates the expensive part (per-domain encoding) from cheap per-vnode heads.
- Two decisions (domain + snode) give two cleaner REINFORCE credit signals rather than one tangled ranking signal.
- Feasibility mask in the snode head: snodes with `available_cpu < vnode.cpu_demand` are masked out (logit `= -inf`) before softmax. If all snodes are masked, mask is dropped and the REINFORCE penalty teaches the domain head to avoid that state.
- Approximate parameter count < 100k with `d=64` — small.

---

## 4. Network Internals

Hidden dim `d = 64` (tunable via config). Single-head dot-product attention everywhere.

**VN encoder input:** `vnode_feats ∈ R^{n_v × f_v}` with features `[cpu_demand_norm, degree_norm, adj_bw_norm, req_size_feat_1, req_size_feat_2]` (reuse `_extract_vnode_features` from `rl_oa_mp_vne`). VN adjacency is BW-weighted, symmetrically normalized with self-loops.

**Domain encoder input:** `X_d ∈ R^{n_d × f_s}` with features `[avail_cpu_ratio, cpu_price_norm, proc_delay_norm, degree_norm, avg_neighbor_bw_norm]` (reuse `_extract_domain_features` from `rl_oa_mp_vne`). Adjacency weighted by `available_bw / capacity`, symmetrically normalized with self-loops.

**Heads.**

```
Domain head (per vnode A):
  q_A = W_q_dom · h_A
  k_d = W_k_dom · g_d                for d ∈ allowed_domains(A)
  score(A, d) = (q_A · k_d) / sqrt(d)
  π_dom(A, ·) = softmax over allowed domains(A)

Snode head (per vnode A, given d*):
  q_A = W_q_sn · concat(h_A, g_{d*})
  k_s = W_k_sn · e_s                 for s ∈ snodes(d*)
  score(A, s) = (q_A · k_s) / sqrt(d)
  masked_score(A, s) = -inf if available_cpu(s) < cpu_demand(A) else score(A, s)
  π_snode(A, ·) = softmax over snodes(d*)         # falls back to unmasked if all masked
```

Plackett–Luce sampling without replacement yields a size-K ordered tuple; log-probs are summed for REINFORCE.

---

## 5. Training Data Generation

Fixed across all episodes: substrate topology, domain partitioning, `cpu_price`, `bandwidth_price`, `processing_delay`, `transmission_delay`, `cpu_capacity`, `bandwidth_capacity`.

Varied per episode: `available_cpu` per snode, `available_bw` per slink, synthetic VN request.

### 5.1 Substrate state sampler (`state_sampler.py`)

Per episode, pick one mode by draw `u ~ Uniform(0, 1)` vs `warmup_fraction`:

- **Mode A — random fractional drop** (default ~80%).
  - `reset_allocations()`, then for each snode: `u_s ~ Uniform(0, u_max_cpu)` → `available_cpu = capacity · (1 − u_s)`.
  - For each slink: `u_l ~ Uniform(0, u_max_bw)` → `available_bw = capacity · (1 − u_l)`.
  - Defaults: `u_max_cpu = 0.8`, `u_max_bw = 0.8`.
- **Mode B — warm-up embedding** (default ~20%).
  - `reset_allocations()`, then sample `M ~ Uniform{0, …, M_max}` (default `M_max = 20`).
  - Generate M random VNs via `generate_random_vn`, embed each greedily via OA-MP-VNE's deterministic path, keep successful ones allocated.
  - Then train on the target VN against this realistic loaded state.
  - Slower; isolated in its own function for easy replacement.

After the reward for the training episode is computed, the episode's commit is rolled back (offline training must not permanently allocate). Mode B's warm-up allocations are also rolled back before the next episode starts.

### 5.2 VN generator

Reuse `algorithms/rl_oa_mp_vne/vn_generator.py` logic. One addition: randomly assign `allowed_domains` per vnode — empty (all domains), single domain, or small subset — with configurable probabilities. Without this, the domain head would always see the same "all domains allowed" input and never learn selective behavior.

Defaults for allowed_domains sampling:
- `p_all = 0.5` (empty list → all domains allowed)
- `p_single = 0.3` (exactly one random domain)
- `p_subset = 0.2` (random 2–3 domains)

### 5.3 Substrate source

Offline training accepts a single substrate JSON path via CLI (`--substrate`). Single-substrate training satisfies the "keep the structure" requirement. The design permits passing a list later for multi-topology generalization, but that extension is out of scope.

---

## 6. Loss & Trainer

### 6.1 Reward

Per episode, after PSO + commit:

```
revenue = Σ vnode.cpu_demand  +  Σ vlink.bandwidth_demand
if success: R = −composite_cost / revenue
else:       R = −R_penalty              # default 2.0, config-tunable
```

`composite_cost` = full weighted resource cost + delay cost, matching the paper's objective (same weights used by the evaluation `composite_cost` metric).

Framing is negated so "higher reward = lower cost"; dimensionless (cost/revenue is unit-free) so the running-mean baseline behaves well across varying VN sizes.

### 6.2 Loss

Per batch of B episodes:

```
L_rl  = − (1/B) · Σ_episodes  (R − baseline) · Σ_actions log π(action)
L_sup = − (1/B_succ) · Σ_successful  Σ_A  log π_snode(A, committed_snode_A)
L     = L_rl + λ · L_sup             # λ default 1.0, config-tunable
```

Where:
- `Σ_actions log π(action)` = domain pick log-prob (one per vnode) + K snode pick log-probs (from Plackett–Luce) per vnode.
- `baseline` = running mean of reward over last `baseline_window` episodes (default 100). More stable than per-batch mean for small batches.
- `B_succ` = number of successful episodes in the batch. If 0, `L_sup` contribution is 0 for that update.
- Supervised aux only fires on successes → cannot reinforce bad decisions. It provides dense short-horizon gradients on the snode head; REINFORCE alone converges slowly with top-K sampling.

### 6.3 Trainer class (`trainer.py`)

- Buffers per episode: `(domain_log_probs, snode_log_probs, reward, committed_snode_indices_or_None, success_flag)`.
- Running-mean baseline via `collections.deque(maxlen=baseline_window)`.
- `update()` runs every `batch_size` episodes; returns dict with `loss_total`, `loss_rl`, `loss_sup`, `avg_reward`, `avg_cost_per_revenue`, `success_rate`, `baseline`.
- On-policy invariant: log-probs come from the exact same sampled actions that drove PSO/commit. Never re-sample between action and gradient (this is the regression rl_oa_mp_vne hit recently; we avoid repeating it).

---

## 7. Offline Training Pipeline

New script: `scripts/train_rl_cand_vne.py`.

### 7.1 CLI

```
python scripts/train_rl_cand_vne.py \
    --substrate  datasets/scenario_1/substrate.json \
    --config     configs/rl_cand_vne.yaml \
    --episodes   5000 \
    --checkpoint checkpoints/rl_cand_vne.pt \
    --log-dir    logs/rl_cand_vne/ \
    --seed       42
```

### 7.2 Flow

1. Load substrate + config; build `PolicyNetwork`, `Trainer`, `GlobalController`.
2. Loop for `--episodes`:
   1. Choose state-sampling mode (A or B) per `warmup_fraction`.
   2. Generate synthetic VN (random `allowed_domains`).
   3. Forward pass → collect log-probs; sample candidate sets.
   4. Run PSO on candidate sets → commit; record success, cost, committed snodes.
   5. Compute reward → `trainer.record(...)`.
   6. Roll back commit (offline training must not permanently allocate).
   7. Every `batch_size` episodes: `trainer.update()` → append JSONL log line.
   8. Every `checkpoint_every` episodes: save checkpoint.
3. Final checkpoint save; convergence summary printed to stdout.

### 7.3 Checkpoint format

```python
{
  "policy_state_dict": ...,
  "config": {...},            # frozen copy of the YAML used
  "substrate_hash": "<sha256>",
  "episodes_trained": int,
  "baseline_buffer": [...],   # last baseline_window rewards
  "training_finished_at": "<iso timestamp>",
}
```

`substrate_hash` is computed from topology + prices + delays + capacities (NOT available resources). On load, `solve()` verifies the hash matches the current substrate; mismatch → warning log, continue anyway (useful for experimenting on similar substrates).

### 7.4 Logging

JSONL at `logs/rl_cand_vne/train.jsonl`, one line per batch, flushed on each write (`tail -f`-friendly):

```json
{
  "episode": 160,
  "loss_total": 0.42, "loss_rl": 0.30, "loss_sup": 0.12,
  "reward_mean": -0.48, "reward_min": -2.0, "reward_max": -0.12,
  "success_rate": 0.88,
  "cost_mean_success": 1050.3,
  "cost_per_revenue_mean": 0.48,
  "baseline": -0.55,
  "lr": 0.001
}
```

### 7.5 Convergence summary

At the end of training, the script prints:
- Mean reward first 100 vs last 100 episodes.
- Mean cost_per_revenue first 100 vs last 100 episodes.
- Mean success_rate first 100 vs last 100 episodes.
- `CONVERGED` / `NOT_CONVERGED` tag based on reward threshold (e.g., last-100 ≥ 1.2 × first-100).

### 7.6 Plotting helper

`evaluation/plot_training_curve.py` (optional, added in the same PR if time permits):
- Reads `logs/rl_cand_vne/train.jsonl`.
- Produces a single PNG with 4 subplots: `loss_total`, `reward_mean` (with `baseline` overlay), `success_rate`, `cost_per_revenue_mean`.
- Re-runnable mid-training.

### 7.7 Reproducibility

`--seed` sets Python, NumPy, and Torch seeds. Config file is serialized into the checkpoint.

---

## 8. Online Fine-Tuning & Integration

### 8.1 `RLCandVNE.solve()` flow

1. First call: init `GlobalController`. Try to load checkpoint at `config.checkpoint.path`; if missing, run inline `_pretrain()` (same logic as offline script, shorter episode budget).
2. Release expired mappings, clear caches.
3. Forward pass on the real VN + current substrate state → sampled domain + K snode candidates per vnode, plus log-probs.
4. Feed candidate sets into existing PSO + `_commit_mapping_ordered` from `oa_mp_vne` (reused, unchanged).
5. Compute reward; call `trainer.record(log_probs, reward, committed_snodes)`.
6. Every `online_k` requests (default 10): `trainer.update()`. Every `online_save_every` updates (default 100): re-save checkpoint.

### 8.2 Registry

`algorithms/registry.py`:

```python
from algorithms.rl_cand_vne.rl_cand_vne import RLCandVNE
ALGORITHMS["rl_cand_vne"] = RLCandVNE
```

### 8.3 Config (`configs/rl_cand_vne.yaml`)

```yaml
policy_network:
  hidden_size: 64
  num_gcn_layers: 2

training:
  pretrain_episodes: 5000      # offline script default
  inline_pretrain_episodes: 500  # fallback when no checkpoint
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
  require_hash_match: false    # warn on mismatch, don't fail
```

### 8.4 File layout

```
algorithms/rl_cand_vne/
    __init__.py             # re-export RLCandVNE
    rl_cand_vne.py          # RLCandVNE class, solve(), online loop
    policy_network.py       # VNEncoder, DomainEncoder, DomainHead, SNodeHead, PolicyNetwork
    trainer.py              # Buffer, running baseline, REINFORCE + sup aux update
    state_sampler.py        # Mode A (fractional drop) + Mode B (warm-up embed)
    vn_generator.py         # Random VN w/ allowed_domains sampling
scripts/train_rl_cand_vne.py
configs/rl_cand_vne.yaml
checkpoints/                # gitignored; created on first save
logs/rl_cand_vne/           # gitignored
evaluation/plot_training_curve.py   # optional, same PR if time
```

PSO and commit reused from `oa_mp_vne` by import; not copy-pasted.

---

## 9. Evaluation & Tests

### 9.1 Training-time sanity

- Loss trends down; rolling success-rate rises; rolling `cost_per_revenue` trends down.
- Random-policy baseline check: run 100 episodes with a freshly initialized network → record mean cost_per_revenue → trained policy must beat that by a clear margin before a checkpoint is shipped.

### 9.2 Algorithm-level evaluation

- Add `rl_cand_vne` to `scripts/run_experiments.sh` alongside `mp_vne`, `oa_mp_vne`, `rl_oa_mp_vne`.
- Four paper metrics: acceptance rate, avg mapping cost, mapping delay, composite cost.
- Primary success criterion: on `scenario_1`, `rl_cand_vne` beats `oa_mp_vne` on composite cost without losing ≥5% acceptance rate.
- Stretch: also beats `rl_oa_mp_vne` on composite cost — that's the core claim (candidate-selection learning > ranking learning).

### 9.3 Unit tests (`tests/algorithms/test_rl_cand_vne.py`)

1. `PolicyNetwork.forward` produces correct shapes for a 2-domain × 3-snode / 2-vnode / 2-vlink toy input.
2. Feasibility mask: snodes with `available_cpu < cpu_demand` get softmax probability 0 (unless all are masked — fallback verified).
3. Plackett–Luce top-K sampler returns K distinct items and K log-probs whose sum equals `log P(ordered subset)`.
4. `state_sampler.fractional_drop`: `available_cpu ∈ [capacity·(1−u_max), capacity]` for every snode post-sample.
5. `state_sampler.warmup_embed`: `available_cpu <= capacity` always; state reachable via legal embeddings (no over-allocation).
6. `Trainer.update`: with two dummy episodes (one success, one failure), loss is finite and `policy.parameters()` receive non-zero gradients (guards the gradient-flow regression).
7. `Trainer`: running-mean baseline tracks last-N correctly.

### 9.4 Integration tests (`tests/integration/test_rl_cand_vne_end_to_end.py`)

1. Offline training for 50 episodes on a tiny substrate (2 domains × 5 nodes) → checkpoint saved; file contents match the documented format.
2. `RLCandVNE().solve()` with that checkpoint on a held-out VN → valid `EmbeddingSolution`; resource constraints respected on success.
3. Online fine-tune over 20 requests — no exceptions; checkpoint can be re-saved; gradients non-zero through at least one update.

### 9.5 Runtime targets

- Unit tests < 5s total.
- Integration tests < 60s total (tiny substrate + low episode counts).

---

## 10. Risks & Open Questions

- **K too small starves PSO.** Default K=5 matches the VN max of 8 nodes. If acceptance rate drops noticeably vs OA-MP-VNE, bump K to 8 or make it adaptive to domain size.
- **REINFORCE variance with top-K sampling.** Mitigated by the supervised auxiliary and running-mean baseline. If still unstable, fall back to entropy regularization (easy add: `+ α · H(π)` in the loss).
- **Warm-up mode cost.** Mode B is ~M× slower than Mode A per episode. Default 20% fraction keeps total training cost manageable; tune down if slow.
- **Substrate hash mismatch.** If someone regenerates the substrate with the same seed, hashes match; if they regenerate with a different seed, hashes differ and a warning appears. Intended behavior.
- **Online fine-tune overwrites offline progress.** Online updates happen every `online_k` requests. If online reward distribution differs from offline, the policy can drift. Config flag `online_learning_enabled` could disable this; decide during implementation if needed.

---

## 11. Deliverables

- Files under `algorithms/rl_cand_vne/`.
- `scripts/train_rl_cand_vne.py`.
- `configs/rl_cand_vne.yaml`.
- Registry entry in `algorithms/registry.py`.
- `rl_cand_vne` added to `scripts/run_experiments.sh`.
- Unit and integration tests.
- Optional: `evaluation/plot_training_curve.py`.
- This spec + implementation plan in `docs/superpowers/`.
