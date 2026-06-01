"""V20 — V19 recipe applied at 200-node scale (encoder learned on 200-node).

Hypothesis behind V20: V19's encoder is over-specialised to 100-node graph
structure (IL + RL both on 100-node). Transferred or fine-tuned on 200-node it
yields sub-optimal substrate features for the larger graph (10 domains × 20
snodes each instead of 10×10), capping acceptance at ~12.5%.

V20 fixes that by initialising from a 200-node-IL pretrain so the encoder sees
200-node statistics from the start, then applying the same direct-decoding
cand-RL recipe that worked for V19.

Recipe (200-node):
  1. IL pretrain on 200-node training set → cand_head imitates mp_vne, encoder
     learns 200-node graph features. Checkpoint:
     checkpoints/il_mp_vne_v17_200nodes.pt (~11.6% acceptance, K=1 PSO).
  2. PPO-direct cand-RL (scripts/ppo_finetune.py --rollout direct --target cand)
     initialised from step 1; same hyperparams as V19 (β_KL=0.1, β_H=0.01,
     value-coef 0.5, success-bonus 1, cost-lambda 0.3, 10k VNRs from
     scenario_200nodes_train). KL anchors cand to the 200-IL init.
  3. Deploy via PSO (`il_mp_vne_v20_pso`).

Same architecture as V17/V19; the difference is the training data scale and
the checkpoint (checkpoints/il_mp_vne_v20_200nodes.pt). See V19 docstring for
the underlying cand-exploration mechanism — V20 is the apples-to-apples
200-node version of that recipe.
"""
from algorithms.il_mp_vne_v17.il_mp_vne_v17 import (
    ILMPVNEV17 as _V17Base,
    ILMPVNEV17Direct as _V17Direct,
    ILMPVNEV17PSO as _V17PSO,
)


class ILMPVNEV20(_V17Base):
    def __init__(self):
        super().__init__()
        self.name = "IL-MP-VNE-V20"


class ILMPVNEV20Direct(_V17Direct):
    def __init__(self):
        super().__init__()
        self.name = "IL-MP-VNE-V20-Direct"


class ILMPVNEV20PSO(_V17PSO):
    def __init__(self):
        super().__init__()
        self.name = "IL-MP-VNE-V20-PSO"
