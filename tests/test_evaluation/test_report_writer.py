from unittest.mock import MagicMock, patch


def test_evaluation_report_schema_fields():
    from mlops_agents.evaluation.report_writer import EvaluationReport
    fields = EvaluationReport.model_fields
    expected = {
        "summary",
        "champion_model",
        "why_champion_won",
        "planner_alignment",
        "deviations_from_planner_expectations",
        "evidence_consistency_warnings",
        "risks_and_warnings",
        "promotion_decision_explanation",
        "human_review_notes",
    }
    assert expected.issubset(set(fields.keys()))


def test_evaluation_report_required_minimum_construction():
    from mlops_agents.evaluation.report_writer import EvaluationReport
    rep = EvaluationReport(
        summary="ok",
        champion_model="lightgbm",
        why_champion_won="best f1",
        planner_alignment="aligned",
        deviations_from_planner_expectations=[],
        evidence_consistency_warnings=[],
        risks_and_warnings=[],
        promotion_decision_explanation="passes thresholds",
        human_review_notes=[],
    )
    assert rep.summary == "ok"


def test_build_report_writer_returns_structured_llm():
    with patch("mlops_agents.evaluation.report_writer.get_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = "STRUCTURED"
        mock_get_llm.return_value = fake_llm

        from mlops_agents.evaluation.report_writer import build_report_writer
        result = build_report_writer()

    mock_get_llm.assert_called_once_with("report_writer")
    fake_llm.with_structured_output.assert_called_once()
    assert result == "STRUCTURED"


def test_run_report_writer_ok_on_first_try():
    from mlops_agents.evaluation.report_writer import EvaluationReport, run_report_writer

    fake_report = EvaluationReport(
        summary="ok",
        champion_model="lightgbm",
        why_champion_won="best f1",
        planner_alignment="aligned",
        deviations_from_planner_expectations=[],
        evidence_consistency_warnings=[],
        risks_and_warnings=[],
        promotion_decision_explanation="passes",
        human_review_notes=[],
    )
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = fake_report

    state = {
        "evaluation_passed": True,
        "candidate_metrics": {"macro_f1": 0.82},
        "champion_metrics": {"macro_f1": 0.78},
        "thresholds_applied": {"accuracy_min": 0.80, "macro_f1_min": 0.75},
        "_planner_output_record": {"plan_summary": {"candidate_models": ["lightgbm"]}},
        "training_plan": {"candidates": [{"model_key": "lightgbm"}]},
        "champion_candidate": {"model_key": "lightgbm"},
    }
    with patch("mlops_agents.evaluation.report_writer.get_report_writer_agent", return_value=fake_agent):
        result = run_report_writer(state)

    assert result["evaluation_report_audit_status"] == "ok"
    assert result["evaluation_report_audit"]["champion_model"] == "lightgbm"


def test_run_report_writer_retry_then_stub_on_repeated_failure():
    from mlops_agents.evaluation.report_writer import run_report_writer

    fake_agent = MagicMock()
    fake_agent.invoke.side_effect = [RuntimeError("boom"), RuntimeError("boom2")]
    state = {
        "evaluation_passed": False,
        "candidate_metrics": {},
        "champion_metrics": {},
        "thresholds_applied": {},
        "_planner_output_record": {},
        "training_plan": {},
        "champion_candidate": {},
    }
    with patch("mlops_agents.evaluation.report_writer.get_report_writer_agent", return_value=fake_agent):
        result = run_report_writer(state)

    assert result["evaluation_report_audit_status"] == "stub"
    assert "unavailable" in result["evaluation_report_audit"]["summary"].lower()
    assert fake_agent.invoke.call_count == 2
