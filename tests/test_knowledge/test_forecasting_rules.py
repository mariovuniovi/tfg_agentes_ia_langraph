"""The new forecasting rules load and match correctly."""
from mlops_agents.knowledge.reader import load_rules


def test_six_new_forecasting_rules_load():
    rules = load_rules()
    ids = {r.rule_id for r in rules}
    expected = {
        "forecasting_short_history_single_split",
        "forecasting_medium_long_expanding_window",
        "forecasting_high_drift_rolling_window",
        "exog_calendar_known_future",
        "exog_unknown_default_naive_carry",
        "exog_slow_macro_auto_arima",
    }
    assert expected.issubset(ids)


def test_recommend_field_present_on_new_rules():
    rules = {r.rule_id: r for r in load_rules()}
    r = rules["forecasting_short_history_single_split"]
    assert r.recommend == {"validation_strategy": "single_split"}


from mlops_agents.knowledge.reader import match_rules


def test_match_rules_short_history_recommends_single_split():
    """At least one new recommend-only rule fires via match_rules()."""
    profile = {
        "problem_type": "forecasting",
        "n_rows": "medium",
        "n_features": "small",
        "missing_rate": "none",
        "n_categorical_features": "none",
        "n_numerical_features": "few",
        "history_length": "short",
    }
    rules = match_rules(profile)
    ids = {r.rule_id for r in rules}
    assert "forecasting_short_history_single_split" in ids
