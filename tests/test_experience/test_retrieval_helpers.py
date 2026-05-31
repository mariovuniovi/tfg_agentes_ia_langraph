"""Tests for retrieval helper functions."""
from mlops_agents.experience.retrieval import derive_relevance_tier, compare_target_scales


def test_high_tier_ge_0_7():
    assert derive_relevance_tier(0.7) == "high"
    assert derive_relevance_tier(0.95) == "high"


def test_medium_tier_range():
    assert derive_relevance_tier(0.4) == "medium"
    assert derive_relevance_tier(0.69) == "medium"


def test_low_tier_below_0_4():
    assert derive_relevance_tier(0.0) == "low"
    assert derive_relevance_tier(0.39) == "low"


def test_similar_scales_returns_none():
    assert compare_target_scales(
        profile_target_std=2.0, experience_target_std=2.3,
    ) is None


def test_one_order_of_magnitude_returns_note():
    note = compare_target_scales(profile_target_std=25.0, experience_target_std=2.0)
    assert note is not None
    assert "×" in note or "x" in note.lower()


def test_missing_either_side_returns_none():
    assert compare_target_scales(profile_target_std=None, experience_target_std=2.0) is None
    assert compare_target_scales(profile_target_std=2.0, experience_target_std=None) is None
    assert compare_target_scales(profile_target_std=None, experience_target_std=None) is None


def test_zero_target_std_returns_none():
    assert compare_target_scales(profile_target_std=0.0, experience_target_std=2.0) is None
