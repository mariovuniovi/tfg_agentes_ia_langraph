"""Smoke test: the shipped ml_rules.yaml loads and validates cleanly."""
from mlops_agents.knowledge.reader import load_rules


def test_starter_rules_load_without_error():
    rules = load_rules()
    assert len(rules) >= 10


def test_starter_rules_all_have_reason():
    for rule in load_rules():
        assert rule.reason.strip(), f"Rule {rule.rule_id} has empty reason"


def test_starter_rules_all_have_prefer_or_avoid():
    for rule in load_rules():
        assert rule.prefer or rule.avoid_or_deprioritize, \
            f"Rule {rule.rule_id} has neither prefer nor avoid_or_deprioritize"
