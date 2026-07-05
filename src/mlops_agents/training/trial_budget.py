"""Deterministic per-candidate Optuna trial budget and sampler selection.

The trial count is fixed in code, independent of the dataset and of the planner
LLM (which no longer emits a budget). Rationale: the number of trials needed is a
function of the *search space* you explore, not the dataset — and per-trial cost on
large series is already bounded by the rolling-window validation policy.
"""
from __future__ import annotations

from typing import Any

import optuna

from mlops_agents.models.loader import get_model

# Calibrated to the magnitude the pipeline already used (~3-6 trials per ML model),
# so existing timing results carry over. Raise for a more thorough search.
TRIALS_PER_ML_MODEL = 5

# Above this many combinations, a fully-categorical space falls back to TPE sampling
# instead of an exhaustive GridSampler — defensive guard against a future model whose
# Cartesian product blows up (e.g. 8*8*8=512). No effect today (max grid is 5).
GRID_SAMPLER_MAX_COMBOS = 50


def deterministic_trials(model_key: str) -> int:
    """Fixed Optuna trial count for a candidate.

    - 0-parameter models (e.g. ``naive``): 1 — nothing to tune.
    - categorical-only search spaces (``season_length`` for seasonal_naive/ets/
      auto_arima): the GridSampler in ``make_sampler`` sweeps the grid
      exhaustively regardless, so this value is only an upper bound.
    - multi-parameter ML forecasters: ``TRIALS_PER_ML_MODEL`` via TPE.
    """
    n_params = len(get_model(model_key).search_space.params)
    return 1 if n_params == 0 else TRIALS_PER_ML_MODEL


def make_sampler(narrowed: Any, n_trials: int) -> tuple[optuna.samplers.BaseSampler, int]:
    """Pick an Optuna sampler + trial count for a candidate's (narrowed) search space.

    When the space is small and fully enumerable (every param categorical, and the
    Cartesian product is at most GRID_SAMPLER_MAX_COMBOS), use an exhaustive
    GridSampler so *every* configuration is always evaluated, regardless of n_trials.
    For statistical forecasters this sweeps the frequency-narrowed season_length grid,
    usually a single canonical value such as 7 for daily data or 52 for weekly data.
    Larger int/float spaces or a categorical grid that would explode past the cap keep
    the seeded TPE sampler with the deterministic trial count (see
    trial_budget.deterministic_trials).

    Returns (sampler, effective_n_trials).
    """
    params = getattr(narrowed, "params", {})
    if params and all(p.type == "categorical" and p.choices for p in params.values()):
        grid = {name: list(p.choices) for name, p in params.items()}
        n_grid = 1
        for choices in grid.values():
            n_grid *= len(choices)
        if n_grid <= GRID_SAMPLER_MAX_COMBOS:
            return optuna.samplers.GridSampler(grid, seed=42), n_grid
    return optuna.samplers.TPESampler(seed=42), n_trials
