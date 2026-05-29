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
