from mlops_agents.planning.context import build_planner_validation_context


def test_context_problem_type_propagated():
    ctx = build_planner_validation_context({}, {}, "forecasting")
    assert ctx.problem_type == "forecasting"
    assert len(ctx.available_model_keys) > 0
    assert len(ctx.available_model_specs) == len(ctx.available_model_keys)


def test_context_is_deterministic():
    ctx1 = build_planner_validation_context({"x": 1}, {"y": 2}, "regression")
    ctx2 = build_planner_validation_context({"x": 1}, {"y": 2}, "regression")
    assert ctx1.available_model_keys == ctx2.available_model_keys
    # rules_by_id deterministic too
    assert set(ctx1.rules_by_id.keys()) == set(ctx2.rules_by_id.keys())


def test_context_rules_by_id_lookup():
    ctx = build_planner_validation_context({"history_length_bucket": "short"},
                                            {"problem_type": "forecasting"},
                                            "forecasting")
    for rule_id, rule in ctx.rules_by_id.items():
        assert rule.get("rule_id") == rule_id
