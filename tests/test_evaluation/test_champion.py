from mlops_agents.evaluation.champion import resolve_champion_model_name


def test_uses_audit_when_present():
    state = {"evaluation_report_audit": {"champion_model": "lightgbm"}}
    assert resolve_champion_model_name(state) == "lightgbm"


def test_falls_back_to_champion_candidate():
    state = {"champion_candidate": {"model_key": "ets"}}
    assert resolve_champion_model_name(state) == "ets"


def test_falls_back_to_training_plan():
    state = {"training_plan": {"selected_model": "auto_arima"}}
    assert resolve_champion_model_name(state) == "auto_arima"


def test_final_fallback_truncates_run_id():
    state = {"training_run_id": "abcdef1234567890"}
    assert resolve_champion_model_name(state) == "abcdef12"


def test_returns_unknown_on_total_emptiness():
    assert resolve_champion_model_name({}) == "unknown"


def test_audit_takes_precedence_over_others():
    state = {
        "evaluation_report_audit": {"champion_model": "lightgbm"},
        "champion_candidate": {"model_key": "ets"},
        "training_plan": {"selected_model": "auto_arima"},
        "training_run_id": "abcd1234",
    }
    assert resolve_champion_model_name(state) == "lightgbm"
