"""Tests for retrieval helper functions."""
from mlops_agents.experience.retrieval import derive_relevance_tier


def test_high_tier_ge_0_7():
    assert derive_relevance_tier(0.7) == "high"
    assert derive_relevance_tier(0.95) == "high"


def test_medium_tier_range():
    assert derive_relevance_tier(0.4) == "medium"
    assert derive_relevance_tier(0.69) == "medium"


def test_low_tier_below_0_4():
    assert derive_relevance_tier(0.0) == "low"
    assert derive_relevance_tier(0.39) == "low"
