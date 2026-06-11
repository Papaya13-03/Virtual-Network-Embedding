"""V10 — V6 architecture + multi-restart PSO at inference.

V6 is the best IL architecture we found (rev/cost 0.360 vs mp_vne 0.340).
V10 keeps V6 weights/model AS-IS, only changes the inference pipeline:
the PSO call inside solve() now runs K times with different RNG seeds and
picks the lowest-cost mapping.

Why this should improve ALL metrics where V6 lost:
  - Acceptance ↑: K independent PSO attempts → higher chance ≥1 succeeds
  - Avg cost ↓: best of K → lower expected cost (variance reduction)
  - Avg delay ↓: correlated with cost via shorter paths
  - rev/cost ↑: maintains V6's lead, amplified by better PSO output

PSO hyperparams per call are UNCHANGED (20 particles × 15 iterations) —
each individual PSO matches the baseline. We just run it K times.

V10 inherits everything from V6 (weights, features, priors). The ONLY
change is `_pso()` override.
"""
import random

from algorithms.carl_vne.il_mp_vne_v6 import (
    ILMPVNEV6 as ILMPVNE_V6_BASE,
    ILMPVNEV6Direct as _V6Direct,
    ILMPVNEV6PSO as _V6PSO,
)


# Default number of PSO restarts at inference. Larger = better quality, slower.
DEFAULT_NUM_RESTARTS = 3


class _MultiRestartMixin:
    """Override `_pso` to run K independent PSO calls and pick best."""

    NUM_RESTARTS = DEFAULT_NUM_RESTARTS

    def _pso(self, candidates, vlink_indices, ordered_vnodes, cand_weights=None):
        # Capture parent's _pso (V6's PSO implementation).
        parent_pso = super()._pso
        best_particle = None
        best_score = float("inf")
        # Master seed (set externally via run_eval --seed). Defaults to 42.
        master_seed = getattr(self, "_master_seed", 42)
        for k in range(self.NUM_RESTARTS):
            # Different RNG seed per restart so each call has fresh stochasticity.
            random.seed(k * 1337 + master_seed)
            particle = parent_pso(candidates, vlink_indices, ordered_vnodes, cand_weights)
            score = self._fitness(particle, candidates, vlink_indices,
                                  ordered_vnodes, cand_weights)
            if score < best_score:
                best_score = score
                best_particle = particle
        return best_particle if best_particle is not None else particle


class ILMPVNEV10(_MultiRestartMixin, ILMPVNE_V6_BASE):
    """V10 base class — V6 weights + K-restart inference."""

    def __init__(self):
        super().__init__()
        self.name = "IL-MP-VNE-V10"


class ILMPVNEV10Direct(_MultiRestartMixin, _V6Direct):
    """Direct mode (no PSO) — multi-restart is a no-op here. Kept for API symmetry."""

    def __init__(self):
        super().__init__()
        self.name = "IL-MP-VNE-V10-Direct"


class ILMPVNEV10PSO(_MultiRestartMixin, _V6PSO):
    """V10 PSO inference — V6's policy + K-restart PSO search."""

    def __init__(self):
        super().__init__()
        self.name = "IL-MP-VNE-V10-PSO"
