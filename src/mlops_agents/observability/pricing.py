import re
from functools import lru_cache
from pathlib import Path
from typing import cast

import yaml

_PRICING_FILE = Path(__file__).parent / "model_pricing.yaml"
_DATE_SUFFIX_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")


@lru_cache(maxsize=1)
def _load() -> dict[str, dict[str, float]]:
    # Cached for the lifetime of the process — restart required after editing model_pricing.yaml
    with open(_PRICING_FILE) as f:
        return cast(dict[str, dict[str, float]], yaml.safe_load(f))


def _normalize(model: str) -> str:
    """Strip date suffix and provider prefix: 'openai/gpt-5.4-mini-2025-11-01' → 'gpt-5.4-mini'."""
    key = model.split("/")[-1]
    return _DATE_SUFFIX_RE.sub("", key)


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float | None:
    """Return estimated USD cost. Returns None for unknown models (display as 'Unknown')."""
    p = _load().get(_normalize(model))
    if not p:
        return None
    return (
        input_tokens * p.get("input_per_1m", 0) / 1_000_000
        + output_tokens * p.get("output_per_1m", 0) / 1_000_000
        + cached_input_tokens * p.get("cached_input_per_1m", 0) / 1_000_000
    )
