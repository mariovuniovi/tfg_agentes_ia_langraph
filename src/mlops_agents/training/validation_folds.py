"""Generate (train_idx, val_idx) pairs given a ValidationStrategy.

This module is single-series for now. Multi-target panel folds are out of
scope for v1 — the executor refuses panel plans that try to use exog
strategies other than `naive_carry`.
"""
from __future__ import annotations

from typing import Iterator

import pandas as pd

from mlops_agents.contracts.training import ValidationStrategy


def iter_folds(
    train_pool: pd.DataFrame,
    strategy: ValidationStrategy,
    dt_col: str,
    sid_cols: list[str],
) -> Iterator[tuple[pd.Index, pd.Index]]:
    """Yield (train_idx, val_idx) pairs in chronological order."""
    if sid_cols:
        raise NotImplementedError(
            "Panel multi-target fold iteration deferred to v2"
        )

    # Sort by date; keep original index so .loc[] works on the caller's DataFrame.
    sorted_index = train_pool.sort_values(dt_col).index
    horizon = strategy.horizon
    step = strategy.step_size or horizon
    n_folds = strategy.n_folds

    n = len(sorted_index)
    if strategy.type == "single_split":
        train_end = n - horizon
        yield sorted_index[:train_end], sorted_index[train_end:n]
        return

    # Compute val_start for each fold so the last fold ends at row n.
    # For k in [0, n_folds-1]: val_start[k] = n - step*(n_folds-1-k) - horizon
    # Example: n=100, horizon=10, step=10, n_folds=3 → [70, 80, 90]
    val_starts = [n - step * (n_folds - 1 - k) - horizon for k in range(n_folds)]

    for val_start in val_starts:
        val_end = val_start + horizon
        if strategy.type == "expanding_window":
            train_start = 0
        elif strategy.type == "rolling_window":
            window = strategy.window_size or 0
            train_start = max(0, val_start - window)
        else:
            raise ValueError(f"Unknown validation strategy type: {strategy.type}")
        yield sorted_index[train_start:val_start], sorted_index[val_start:val_end]
