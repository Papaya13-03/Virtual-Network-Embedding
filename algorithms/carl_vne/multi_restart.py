"""Multi-restart PSO at inference.

Keeps the policy weights/model AS-IS, only changes the inference pipeline:
the PSO call inside solve() runs K times with different RNG seeds and picks
the lowest-cost mapping.

Why this improves metrics:
  - Acceptance ↑: K independent PSO attempts → higher chance ≥1 succeeds
  - Avg cost ↓: best of K → lower expected cost (variance reduction)
  - Avg delay ↓: correlated with cost via shorter paths

PSO hyperparams per call are UNCHANGED (20 particles × 15 iterations) —
each individual PSO matches the baseline. We just run it K times.
"""
import random


# Default number of PSO restarts at inference. Larger = better quality, slower.
DEFAULT_NUM_RESTARTS = 3


class MultiRestartMixin:
    """Override `_pso` to run K independent PSO calls and pick best."""

    NUM_RESTARTS = DEFAULT_NUM_RESTARTS

    def _pso(self, candidates, vlink_indices, ordered_vnodes, cand_weights=None):
        # Capture parent's _pso (BaseVNE's PSO implementation).
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
