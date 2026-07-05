"""Tests for RetrievalView bucket/tier/scale fields (Task 2.3)."""
from __future__ import annotations

import pytest

from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.experience.schema import ExperienceRecord


@pytest.fixture
def seeded_pool(tmp_path):
    db = tmp_path / "exp.db"
    pool = ExperiencePool(db)
    pool.insert_from_record(ExperienceRecord(
        task_id="match",
        problem_type="forecasting",
        experience_summary="weekly",
        dataset_profile={
            "history_length_bucket": "short",
            "frequency_bucket": "weekly",
            "target_std": 2.0,
        },
        selected_solution={"model_key": "ets", "validation_score": 1.0, "complexity_rank": 1},
        models_tested=[],
        metric_to_optimize="rmse",
        target_mean=10.0, target_std=2.0, target_min=5.0, target_max=15.0,
    ))
    pool.insert_from_record(ExperienceRecord(
        task_id="mismatch",
        problem_type="forecasting",
        experience_summary="daily different scale",
        dataset_profile={
            "history_length_bucket": "long",
            "frequency_bucket": "daily",
            "target_std": 200.0,
        },
        selected_solution={"model_key": "naive", "validation_score": 50.0, "complexity_rank": 1},
        models_tested=[],
        metric_to_optimize="rmse",
        target_mean=1000.0, target_std=200.0, target_min=500.0, target_max=1500.0,
    ))
    return pool


def test_retrieval_view_has_matched_and_mismatched_buckets(seeded_pool):
    profile = {
        "history_length_bucket": "short",
        "frequency_bucket": "weekly",
        "target_std": 2.0,
        "target_mean": 10.0,
    }
    views = seeded_pool.find_similar(profile, "forecasting", k=2)
    match_view = next(v for v in views if v.task_id == "match")
    assert "history_length_bucket" in match_view.matched_buckets or "frequency_bucket" in match_view.matched_buckets


def test_retrieval_view_target_scale_note_present_for_mismatch(seeded_pool):
    profile = {
        "history_length_bucket": "short",
        "frequency_bucket": "weekly",
        "target_std": 2.0,
        "target_mean": 10.0,
    }
    views = seeded_pool.find_similar(profile, "forecasting", k=2)
    mismatch_view = next(v for v in views if v.task_id == "mismatch")
    assert mismatch_view.target_scale_note is not None
