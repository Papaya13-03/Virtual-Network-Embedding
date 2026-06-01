"""V19 — V17 architecture with cand_head fine-tuned by direct-decoding PPO.

Same network architecture as V17 (top-1 candidate per allowed domain at PSO
inference). The difference is the trained cand_head: instead of pure imitation
of mp_vne (V17/R2), the cand_head is RL-fine-tuned to PICK candidates that lead
to successful, cheap embeddings — learned from real outcome feedback.

Winning recipe (100-node, this is the saved baseline for V19):
  1. IL pretrain (R2): cand_head imitates mp_vne.                → 24.3% (PSO)
  2. PPO fine-tune via DIRECT decoding (scripts/ppo_finetune.py --rollout direct):
       - rank_direct samples 1 snode/vnode → real exploration over candidates
       - reward = success_bonus + cost term; fail = -1
       - freeze ordering heads (node/link); train cand_head + encoder + value
       - KL-anchor cand distribution to R2 (β_KL=0.1) → refine, not collapse
       - critic V(s) baseline + advantage normalisation (low variance)
     → cand_head learns to pick better snodes (KL drift 0.53, entropy sharpens).
  3. Deploy via PSO (top-1 per domain): the improved cand_head proposes better
     candidates.                                                  → 25.9% (PSO)

Result: +1.6pp acceptance (776 vs 729 / 3000) and higher revenue (928 vs ~900)
over the R2-pso baseline — the exploration the strict K=1 argmax path lacked.

Checkpoint: checkpoints/il_mp_vne_v19_100nodes.pt (== il_mp_vne_v17_ppo_direct).
"""
from algorithms.il_mp_vne_v17.il_mp_vne_v17 import (
    ILMPVNEV17 as _V17Base,
    ILMPVNEV17Direct as _V17Direct,
    ILMPVNEV17PSO as _V17PSO,
)


# Link-order at commit: A/B tested bw-desc, original VN order, and PL-sampled
# (from random link_head) on seed=42. PL-sampled stayed best (780 / 766 / 713
# accepted respectively). Fixed heuristics concentrate `shortest_path` traffic
# and lose feasibility on later vlinks; PL's pseudo-random spread happens to
# avoid that. Optimal order is state-dependent — would need a trained link_head
# (RL signal too weak to learn it cleanly given the small lever). Keeping the
# inherited V17 behaviour.


class ILMPVNEV19(_V17Base):
    def __init__(self):
        super().__init__()
        self.name = "IL-MP-VNE-V19"


class ILMPVNEV19Direct(_V17Direct):
    def __init__(self):
        super().__init__()
        self.name = "IL-MP-VNE-V19-Direct"


class ILMPVNEV19PSO(_V17PSO):
    def __init__(self):
        super().__init__()
        self.name = "IL-MP-VNE-V19-PSO"
