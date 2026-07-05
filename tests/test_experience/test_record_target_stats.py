"""Tests for Optional target stats fields on ExperienceRecord."""
from mlops_agents.experience.schema import ExperienceRecord


def test_record_accepts_target_stats():
    rec = ExperienceRecord(
        task_id="t1",
        problem_type="regression",
        dataset_profile={"n_rows": 100},
        metric_to_optimize="rmse",
        target_mean=5.5,
        target_std=2.1,
        target_min=1.0,
        target_max=10.0,
    )
    assert rec.target_mean == 5.5


def test_record_legacy_compat_target_stats_default_none():
    rec = ExperienceRecord(
        task_id="t2",
        problem_type="classification",
        dataset_profile={"n_rows": 50},
        metric_to_optimize="accuracy",
    )
    assert rec.target_mean is None
    assert rec.target_std is None
    assert rec.target_min is None
    assert rec.target_max is None
