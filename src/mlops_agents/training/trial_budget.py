"""Deterministic per-candidate Optuna trial budget.

The trial count is fixed in code, independent of the dataset and of the planner
LLM (which no longer emits a budget). Rationale: the number of trials needed is a
function of the *search space* you explore, not the dataset — and per-trial cost on
large series is already bounded by the rolling-window validation policy.
"""
from __future__ import annotations

from mlops_agents.models.loader import get_model

# Calibrated to the magnitude the pipeline already used (~3-6 trials per ML model),
# so existing timing results carry over. Raise for a more thorough search.
TRIALS_PER_ML_MODEL = 5


def deterministic_trials(model_key: str) -> int:
    """Fixed Optuna trial count for a candidate.

    - 0-parameter models (e.g. ``naive``): 1 — nothing to tune.
    - categorical-only search spaces (``season_length`` for seasonal_naive/ets/
      auto_arima): the GridSampler in ``executor._make_sampler`` sweeps the grid
      exhaustively regardless, so this value is only an upper bound.
    - multi-parameter ML forecasters: ``TRIALS_PER_ML_MODEL`` via TPE.
    """
    n_params = len(get_model(model_key).search_space.params)
    return 1 if n_params == 0 else TRIALS_PER_ML_MODEL
