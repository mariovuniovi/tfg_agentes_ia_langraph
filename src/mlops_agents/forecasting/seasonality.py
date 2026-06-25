"""Frequency-aware seasonal-period policy for forecasting models.

The pipeline treats season length as a deterministic property of the data
cadence, not as something the planner LLM should guess. Keep this module free of
executor/model imports so factories, exogenous extension, and search-space
narrowing all share one policy without creating dependency cycles.
"""

from __future__ import annotations

import re
from typing import Any

DEFAULT_SEASON_LENGTH = 1

_CANONICAL_SEASON_LENGTH_BY_UNIT: dict[str, int] = {
    "H": 24,
    "D": 7,
    "W": 52,
    "M": 12,
    "Q": 4,
    "Y": 1,
}

_FREQ_UNIT_ALIASES: dict[str, str] = {
    "BH": "H",
    "B": "D",
    "CBH": "H",
    "C": "D",
    "MS": "M",
    "ME": "M",
    "BM": "M",
    "BMS": "M",
    "BME": "M",
    "QS": "Q",
    "QE": "Q",
    "BQ": "Q",
    "BQS": "Q",
    "BQE": "Q",
    "YS": "Y",
    "YE": "Y",
    "A": "Y",
    "AS": "Y",
    "AE": "Y",
    "BA": "Y",
    "BAS": "Y",
    "BAE": "Y",
    "BY": "Y",
    "BYS": "Y",
    "BYE": "Y",
}


def normalize_frequency(freq: Any | None) -> str | None:
    """Return the base seasonal unit for a pandas-style frequency.

    Examples:
    - ``"h"`` -> ``"H"``
    - ``"W-MON"`` -> ``"W"``
    - ``"MS"`` / ``"ME"`` -> ``"M"``

    Unknown frequencies are still normalized as far as possible and returned to
    the caller; they simply will not map to a canonical season length.
    """
    if freq is None:
        return None

    raw = getattr(freq, "freqstr", freq)
    text = str(raw).strip()
    if not text:
        return None

    base = text.upper().split("-", 1)[0]
    base = re.sub(r"^\d+", "", base)
    return _FREQ_UNIT_ALIASES.get(base, base)


def canonical_season_length(freq: Any | None) -> int | None:
    """Return the deterministic seasonal period for a known frequency."""
    unit = normalize_frequency(freq)
    if unit is None:
        return None
    return _CANONICAL_SEASON_LENGTH_BY_UNIT.get(unit)


def default_season_length(freq: Any | None) -> int:
    """Return the canonical period, falling back to non-seasonal length 1."""
    return canonical_season_length(freq) or DEFAULT_SEASON_LENGTH


# Candidate seasonal periods per model-family "tier", keyed by base frequency unit.
# seasonal_naive fits are ~free -> richest grid; ETS is moderate -> skip huge m;
# AutoARIMA is the most expensive per fit -> tightest grid. 1 = non-seasonal floor.
_SEASON_GRID_BY_TIER: dict[str, dict[str, list[int]]] = {
    "rich":   {"H": [1, 24, 168], "D": [1, 7, 30], "W": [1, 4, 13, 52], "M": [1, 3, 12], "Q": [1, 4], "Y": [1]},
    "modest": {"H": [1, 24],      "D": [1, 7],     "W": [1, 13, 52],    "M": [1, 3, 12], "Q": [1, 4], "Y": [1]},
    "tight":  {"H": [1, 24],      "D": [1, 7],     "W": [1, 52],        "M": [1, 12],    "Q": [1, 4], "Y": [1]},
}

_MODEL_GRID_TIER: dict[str, str] = {
    "seasonal_naive": "rich",
    "ets": "modest",
    "auto_arima": "tight",
}

_MIN_CYCLES = 2  # need >= 2 full seasonal cycles to estimate a period


def season_length_grid(model_key: str, freq: Any | None, n_obs: int) -> list[int] | None:
    """Candidate seasonal periods for ``model_key`` at frequency ``freq``.

    The grid is chosen by model family (cost), then pruned to periods estimable
    from ``n_obs`` observations (``m == 1`` or ``n_obs >= 2*m``). The non-seasonal
    period 1 is always retained as a floor. Returns ``None`` for unknown
    frequencies so the caller keeps the model's original search space.
    """
    unit = normalize_frequency(freq)
    if unit is None:
        return None
    tier = _MODEL_GRID_TIER.get(model_key, "tight")
    grid = _SEASON_GRID_BY_TIER[tier].get(unit)
    if grid is None:
        return None
    pruned = [m for m in grid if m == 1 or n_obs >= _MIN_CYCLES * m]
    return pruned or [1]


def max_season_length(freq: Any | None) -> int | None:
    """Largest seasonal period any model might request at this frequency.

    Used to size the validation window so it can support every candidate's
    seasonality. Returns ``None`` for unknown frequencies.
    """
    unit = normalize_frequency(freq)
    grid = _SEASON_GRID_BY_TIER["rich"].get(unit) if unit else None
    return max(grid) if grid else None
