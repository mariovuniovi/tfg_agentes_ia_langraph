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


from mlops_agents.forecasting.seasonality import max_season_length, season_length_grid  # noqa: E402


def test_season_length_grid_tiers_by_model_family():
    # daily, ample history: seasonal_naive rich, ets modest, auto_arima tight
    assert season_length_grid("seasonal_naive", "D", 10000) == [1, 7, 30]
    assert season_length_grid("ets", "D", 10000) == [1, 7]
    assert season_length_grid("auto_arima", "D", 10000) == [1, 7]


def test_season_length_grid_weekly_differs_by_tier():
    assert season_length_grid("seasonal_naive", "W", 10000) == [1, 4, 13, 52]
    assert season_length_grid("ets", "W", 10000) == [1, 13, 52]
    assert season_length_grid("auto_arima", "W", 10000) == [1, 52]


def test_season_length_grid_prunes_periods_without_two_cycles():
    # 60 weekly obs: 52 needs >=104 -> dropped; 13 needs >=26 -> kept; 4 kept
    assert season_length_grid("seasonal_naive", "W", 60) == [1, 4, 13]


def test_season_length_grid_always_keeps_nonseasonal_floor():
    # tiny series: every seasonal period pruned, 1 survives
    assert season_length_grid("auto_arima", "W", 10) == [1]


def test_season_length_grid_unknown_freq_returns_none():
    assert season_length_grid("ets", None, 1000) is None
    assert season_length_grid("ets", "weird", 1000) is None


def test_season_length_grid_handles_pandas_aliases():
    assert season_length_grid("auto_arima", "W-MON", 10000) == [1, 52]
    assert season_length_grid("ets", "h", 10000) == [1, 24]


def test_season_length_grid_unknown_model_defaults_to_tight():
    assert season_length_grid("some_future_model", "D", 10000) == [1, 7]


def test_max_season_length_uses_largest_candidate_period():
    assert max_season_length("h") == 168   # hourly rich grid max
    assert max_season_length("D") == 30
    assert max_season_length("W") == 52
    assert max_season_length(None) is None
