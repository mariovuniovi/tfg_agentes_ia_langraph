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
