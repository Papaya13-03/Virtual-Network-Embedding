"""V22 — V21 architecture + new multi-objective reward (acceptance + rev/cost).

Same encoder/cand-head as V21. The difference is the PPO TRAINING reward:

  V19/V21 reward (cost-relative):
    reward_success = success_bonus + cost_lambda · (cost_EMA − cost) / cost_EMA
    reward_fail    = fail_reward (= −1)
  → Cost EMA is *size-dependent* — a big VNR can get reward < fail when cost
    spikes beyond the EMA built from small VNRs. Policy gets perverse incentive
    to refuse big VNRs.

  V22 reward (rev/cost ratio, clipped):
    ratio          = vn_revenue / actual_cost      (size-invariant)
    ratio_EMA      = 0.95·ratio_EMA + 0.05·ratio
    rel            = clip((ratio − ratio_EMA) / ratio_EMA, −1, +1)
    reward_success = α + β · rel                   ∈ [α − β, α + β]
    reward_fail    = −α
  with α=1.0, β=0.5 (default) → success always ∈ [0.5, 1.5], always > fail
    (−1.0). Policy optimises BOTH acceptance and rev/cost efficiency.

The checkpoint shape is identical to V21 — V22 just selects the
``--reward-mode rev_cost`` path in scripts/ppo_finetune.py with this class
as the algorithm. Deploy via PSO like V19/V21 (per-domain top-1).
"""
from algorithms.il_mp_vne_v21.il_mp_vne_v21 import (
    ILMPVNEV21 as _V21Base,
    ILMPVNEV21Direct as _V21Direct,
    ILMPVNEV21PSO as _V21PSO,
)


class ILMPVNEV22(_V21Base):
    def __init__(self):
        super().__init__()
        self.name = "IL-MP-VNE-V22"


class ILMPVNEV22Direct(_V21Direct):
    def __init__(self):
        super().__init__()
        self.name = "IL-MP-VNE-V22-Direct"


class ILMPVNEV22PSO(_V21PSO):
    def __init__(self):
        super().__init__()
        self.name = "IL-MP-VNE-V22-PSO"
