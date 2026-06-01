from typing import Dict, Type

from algorithms.mp_vne.mp_vne import MPVNE
from algorithms.mp_vne_v2.mp_vne_v2 import MPVNEV2
from algorithms.mp_vne_mr.mp_vne_mr import MPVNEMR
from algorithms.mp_vne_v3.mp_vne_v3 import MPVNEV3
from algorithms.mp_vne_v4.mp_vne_v4 import MPVNEV4
from algorithms.il_mp_vne_v17.il_mp_vne_v17 import (
    ILMPVNEV17,
    ILMPVNEV17Direct,
    ILMPVNEV17PSO,
)
from algorithms.il_mp_vne_v19.il_mp_vne_v19 import (
    ILMPVNEV19,
    ILMPVNEV19Direct,
    ILMPVNEV19PSO,
)
from algorithms.il_mp_vne_v20.il_mp_vne_v20 import (
    ILMPVNEV20,
    ILMPVNEV20Direct,
    ILMPVNEV20PSO,
)
from algorithms.il_mp_vne_v22.il_mp_vne_v22 import (
    ILMPVNEV22,
    ILMPVNEV22Direct,
    ILMPVNEV22PSO,
)
from algorithms.il_mp_vne_v21.il_mp_vne_v21 import (
    ILMPVNEV21,
    ILMPVNEV21Direct,
    ILMPVNEV21PSO,
)
from algorithms.il_mp_vne.il_mp_vne import (
    ILMPVNE,
    ILMPVNEDirect,
    ILMPVNEPSO,
)
from algorithms.il_mp_vne_v3.il_mp_vne_v3 import (
    ILMPVNEV3,
    ILMPVNEV3Direct,
    ILMPVNEV3PSO,
)
from algorithms.il_mp_vne_v5.il_mp_vne_v5 import (
    ILMPVNEV5,
    ILMPVNEV5Direct,
    ILMPVNEV5PSO,
)
from algorithms.il_mp_vne_v6.il_mp_vne_v6 import (
    ILMPVNEV6,
    ILMPVNEV6Direct,
    ILMPVNEV6PSO,
)
from algorithms.il_mp_vne_v7.il_mp_vne_v7 import (
    ILMPVNEV7,
    ILMPVNEV7Direct,
    ILMPVNEV7PSO,
)
from algorithms.il_mp_vne_v9.il_mp_vne_v9 import (
    ILMPVNEV9,
    ILMPVNEV9Direct,
    ILMPVNEV9PSO,
)
from algorithms.il_mp_vne_v10.il_mp_vne_v10 import (
    ILMPVNEV10,
    ILMPVNEV10Direct,
    ILMPVNEV10PSO,
)
from algorithms.il_mp_vne_v11.il_mp_vne_v11 import (
    ILMPVNEV11,
    ILMPVNEV11Direct,
    ILMPVNEV11PSO,
)
from algorithms.il_mp_vne_v12.il_mp_vne_v12 import (
    ILMPVNEV12,
    ILMPVNEV12Direct,
    ILMPVNEV12PSO,
)
from algorithms.il_mp_vne_v14.il_mp_vne_v14 import (
    ILMPVNEV14,
    ILMPVNEV14Direct,
    ILMPVNEV14PSO,
)
from algorithms.il_mp_vne_v16.il_mp_vne_v16 import (
    ILMPVNEV16,
    ILMPVNEV16Direct,
    ILMPVNEV16PSO,
)

ALGORITHMS: Dict[str, Type] = {
    "mp_vne": MPVNE,
    "mp_vne_v2": MPVNEV2,
    "mp_vne_mr": MPVNEMR,
    "mp_vne_v3": MPVNEV3,
    "mp_vne_v4": MPVNEV4,
    # v2 — per-vnode independent cand_head
    "il_mp_vne": ILMPVNE,
    "il_mp_vne_direct": ILMPVNEDirect,
    "il_mp_vne_pso": ILMPVNEPSO,
    # v3 — VN-GCN + cross-vnode self-attention (joint coordination)
    "il_mp_vne_v3": ILMPVNEV3,
    "il_mp_vne_v3_direct": ILMPVNEV3Direct,
    "il_mp_vne_v3_pso": ILMPVNEV3PSO,
    # v5 — explicit PreCost-aware cost features in cand_head + boundary/bw_price
    "il_mp_vne_v5": ILMPVNEV5,
    "il_mp_vne_v5_direct": ILMPVNEV5Direct,
    "il_mp_vne_v5_pso": ILMPVNEV5PSO,
    # v6 — adds explicit link_cost estimate (full PreCost-equivalent prior)
    "il_mp_vne_v6": ILMPVNEV6,
    "il_mp_vne_v6_direct": ILMPVNEV6Direct,
    "il_mp_vne_v6_pso": ILMPVNEV6PSO,
    # v8 — V6 architecture + RL fine-tune (alias to V6 classes)
    "il_mp_vne_v8": ILMPVNEV6,
    "il_mp_vne_v8_pso": ILMPVNEV6PSO,
    "il_mp_vne_v8_direct": ILMPVNEV6Direct,
    # v7 — adds forward-looking features (util_after, global state)
    "il_mp_vne_v7": ILMPVNEV7,
    "il_mp_vne_v7_direct": ILMPVNEV7Direct,
    "il_mp_vne_v7_pso": ILMPVNEV7PSO,
    # v9 — GAT for substrate intra + slack-aware + domain availability + calibrated priors
    "il_mp_vne_v9": ILMPVNEV9,
    "il_mp_vne_v9_direct": ILMPVNEV9Direct,
    "il_mp_vne_v9_pso": ILMPVNEV9PSO,
    # v10 — V6 model + multi-restart PSO inference (no retraining needed)
    "il_mp_vne_v10": ILMPVNEV10,
    "il_mp_vne_v10_direct": ILMPVNEV10Direct,
    "il_mp_vne_v10_pso": ILMPVNEV10PSO,
    # v11 — V6 + neighbor-conditional link cost (4th cost feature)
    "il_mp_vne_v11": ILMPVNEV11,
    "il_mp_vne_v11_direct": ILMPVNEV11Direct,
    "il_mp_vne_v11_pso": ILMPVNEV11PSO,
    # v12 — V6 + per-slink stats (max_bw, min_price, avg_delay, std_price)
    "il_mp_vne_v12": ILMPVNEV12,
    "il_mp_vne_v12_direct": ILMPVNEV12Direct,
    "il_mp_vne_v12_pso": ILMPVNEV12PSO,
    # v13 — V6 architecture + IL on improved targets (v10 self-distillation).
    # Eval class = same as v10 (V6 + multi-restart PSO inference); only ckpt differs.
    "il_mp_vne_v13": ILMPVNEV10,
    "il_mp_vne_v13_direct": ILMPVNEV10Direct,
    "il_mp_vne_v13_pso": ILMPVNEV10PSO,
    # v14 — V10 + hybrid top-K candidates (NN top ∪ PreCost top) at inference.
    "il_mp_vne_v14": ILMPVNEV14,
    "il_mp_vne_v14_direct": ILMPVNEV14Direct,
    "il_mp_vne_v14_pso": ILMPVNEV14PSO,
    # v15 — V6 architecture + RL fine-tune (PPO/REINFORCE + KL). Uses V10 wrapper at eval.
    "il_mp_vne_v15": ILMPVNEV10,
    "il_mp_vne_v15_direct": ILMPVNEV10Direct,
    "il_mp_vne_v15_pso": ILMPVNEV10PSO,
    # v16 — V6 model + per-domain top-K + mp_vne-style fitness + multi-restart.
    "il_mp_vne_v16": ILMPVNEV16,
    "il_mp_vne_v16_direct": ILMPVNEV16Direct,
    "il_mp_vne_v16_pso": ILMPVNEV16PSO,
    # v17 — V16 with PER_DOMAIN_K=1 (strict paper interpretation, one candidate per domain).
    "il_mp_vne_v17": ILMPVNEV17,
    "il_mp_vne_v17_direct": ILMPVNEV17Direct,
    "il_mp_vne_v17_pso": ILMPVNEV17PSO,
    # v19 — V17 arch + cand_head RL-fine-tuned via direct-decoding PPO. Deploy via PSO.
    #        ckpt: checkpoints/il_mp_vne_v19_100nodes.pt → 25.9% (vs R2-pso 24.3%).
    "il_mp_vne_v19": ILMPVNEV19,
    "il_mp_vne_v19_direct": ILMPVNEV19Direct,
    "il_mp_vne_v19_pso": ILMPVNEV19PSO,
    # v20 — V19 recipe applied at 200-node scale: 200-IL init + PPO-direct cand-RL.
    #        ckpt: checkpoints/il_mp_vne_v20_200nodes.pt
    "il_mp_vne_v20": ILMPVNEV20,
    "il_mp_vne_v20_direct": ILMPVNEV20Direct,
    "il_mp_vne_v20_pso": ILMPVNEV20PSO,
    # v21 — Stronger GAT encoder + multi-layer cross-attn cand head, no
    #        node/link heads. Designed to scale 100 → 200 → 500 substrate nodes.
    "il_mp_vne_v21": ILMPVNEV21,
    "il_mp_vne_v21_direct": ILMPVNEV21Direct,
    "il_mp_vne_v21_pso": ILMPVNEV21PSO,
    # v22 — V21 arch + new rev/cost ratio reward (size-invariant, clipped).
    #        Optimises acceptance AND rev/cost simultaneously.
    "il_mp_vne_v22": ILMPVNEV22,
    "il_mp_vne_v22_direct": ILMPVNEV22Direct,
    "il_mp_vne_v22_pso": ILMPVNEV22PSO,
}


def get_algorithm(name: str):
    algo_class = ALGORITHMS.get(name.lower())
    if not algo_class:
        raise ValueError(
            f"Algorithm '{name}' not found. Available algorithms: {list(ALGORITHMS.keys())}"
        )
    return algo_class()
