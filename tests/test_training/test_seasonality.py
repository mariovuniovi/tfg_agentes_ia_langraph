"""Shared frequency -> season_length policy."""

from mlops_agents.forecasting.seasonality import (
    canonical_season_length,
    default_season_length,
    normalize_frequency,
)


def test_normalize_frequency_collapses_pandas_aliases():
    assert normalize_frequency("h") == "H"
    assert normalize_frequency("W-MON") == "W"
    assert normalize_frequency("MS") == "M"
    assert normalize_frequency("ME") == "M"
    assert normalize_frequency("QS") == "Q"
    assert normalize_frequency("YS") == "Y"


def test_canonical_season_length_for_supported_frequencies():
    assert canonical_season_length("H") == 24
    assert canonical_season_length("D") == 7
    assert canonical_season_length("W-MON") == 52
    assert canonical_season_length("MS") == 12
    assert canonical_season_length("QS") == 4
    assert canonical_season_length("YS") == 1


def test_unknown_frequency_has_no_canonical_period_but_default_is_nonseasonal():
    assert canonical_season_length(None) is None
    assert canonical_season_length("weird") is None
    assert default_season_length(None) == 1
    assert default_season_length("weird") == 1
