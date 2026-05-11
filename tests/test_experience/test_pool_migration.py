"""Migration adds five JSON columns; inserting a record populates them."""
import json
import sqlite3
from pathlib import Path

from mlops_agents.experience.pool import ExperiencePool


def test_migration_adds_five_columns(tmp_path):
    db = tmp_path / "test.db"
    pool = ExperiencePool(db, audit_dir=tmp_path / "audit")
    conn = sqlite3.connect(db)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(experiences)")}
    conn.close()
    for c in [
        "validation_strategy_json",
        "exog_availability_json",
        "exog_strategies_json",
        "per_fold_metrics_json",
        "exog_fit_failures_json",
    ]:
        assert c in cols


def test_migration_is_idempotent(tmp_path):
    db = tmp_path / "test.db"
    ExperiencePool(db, audit_dir=tmp_path / "audit")
    # Re-init: should not error
    ExperiencePool(db, audit_dir=tmp_path / "audit")


def test_insert_record_with_new_fields_round_trips(tmp_path, minimal_experience_record):
    db = tmp_path / "test.db"
    pool = ExperiencePool(db, audit_dir=tmp_path / "audit")
    record = minimal_experience_record(
        validation_strategy={"type": "single_split", "horizon": 6, "n_folds": 1},
        exog_availability={"oil": "unknown_future"},
        exog_strategies={"oil": "naive_carry"},
        per_fold_metrics=[{"fold_id": 0, "rmse": 1.23}],
        exog_fit_failures=[],
    )
    pool.insert_from_record(record)

    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT validation_strategy_json, exog_availability_json, exog_strategies_json, "
        "per_fold_metrics_json, exog_fit_failures_json FROM experiences WHERE task_id=?",
        (record.task_id,),
    ).fetchone()
    conn.close()

    vs, ea, es, pfm, ef = row
    assert json.loads(vs)["type"] == "single_split"
    assert json.loads(ea)["oil"] == "unknown_future"
    assert json.loads(es)["oil"] == "naive_carry"
    assert json.loads(pfm)[0]["rmse"] == 1.23
    assert json.loads(ef) == []


def test_pool_get_returns_new_fields(tmp_path, minimal_experience_record):
    db = tmp_path / "test.db"
    pool = ExperiencePool(db, audit_dir=tmp_path / "audit")
    record = minimal_experience_record(
        validation_strategy={"type": "rolling_window", "n_folds": 3, "horizon": 6},
        exog_strategies={"oil": "ets"},
        per_fold_metrics=[{"fold_id": 0, "rmse": 1.0}, {"fold_id": 1, "rmse": 1.5}],
    )
    pool.insert_from_record(record)

    fetched = pool.get(record.task_id)
    assert fetched is not None
    assert fetched.validation_strategy == {"type": "rolling_window", "n_folds": 3, "horizon": 6}
    assert fetched.exog_strategies == {"oil": "ets"}
    assert len(fetched.per_fold_metrics) == 2
