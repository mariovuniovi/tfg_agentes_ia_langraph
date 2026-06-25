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
