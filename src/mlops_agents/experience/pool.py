"""ExperiencePool — SQLite-backed read/write layer for experience records."""
from __future__ import annotations
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from mlops_agents.experience.migrations._runner import apply_pending_migrations
from mlops_agents.experience.schema import ExperienceRecord, RetrievalView
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


class ExperiencePool:
    def __init__(self, db_path: Path, audit_dir: Path | None = None):
        apply_pending_migrations(db_path)
        self._db_path = db_path
        self._audit_dir = audit_dir

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def insert_from_record(self, record: ExperienceRecord) -> None:
        """Insert into all 3 tables atomically, then write JSON audit copy."""
        created_at = datetime.now(UTC).isoformat()
        sol = record.selected_solution
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO experiences
                (task_id, problem_type, dataset_name, dataset_profile_json,
                 training_plan_json, selected_model_key, metric_to_optimize,
                 metric_direction, validation_score, validation_std,
                 experience_summary, mlflow_parent_run_id, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (record.task_id, record.problem_type, record.dataset_name,
                 json.dumps(record.dataset_profile), json.dumps(record.training_plan_input),
                 sol.model_key if sol else None, record.metric_to_optimize,
                 record.metric_direction,
                 sol.validation_score if sol else None, sol.validation_std if sol else None,
                 record.experience_summary, record.mlflow.get("parent_run_id"), created_at),
            )
            conn.execute("DELETE FROM candidate_results WHERE task_id = ?", (record.task_id,))
            for cand in record.models_tested:
                conn.execute(
                    """INSERT INTO candidate_results
                    (task_id, model_key, status, best_params_json, best_score,
                     best_score_std, n_trials_used, duration_s, complexity_rank,
                     mlflow_run_id, error_type, error_message)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (record.task_id, cand.model_key, cand.status,
                     json.dumps(cand.best_params) if cand.best_params else None,
                     cand.best_score, cand.best_score_std, cand.n_trials_used,
                     cand.duration_s, cand.complexity_rank, cand.mlflow_run_id,
                     cand.error_type, cand.error_message),
                )
            conn.execute("DELETE FROM model_artifacts WHERE task_id = ?", (record.task_id,))
            if sol:
                conn.execute(
                    """INSERT INTO model_artifacts
                    (task_id, model_key, mlflow_run_id, is_champion, metric_name, metric_value, created_at)
                    VALUES (?,?,?,1,?,?,?)""",
                    (record.task_id, sol.model_key, record.mlflow.get("parent_run_id"),
                     record.metric_to_optimize, sol.validation_score, created_at),
                )
        if self._audit_dir is not None:
            try:
                self._audit_dir.mkdir(parents=True, exist_ok=True)
                (self._audit_dir / f"{record.task_id}.json").write_text(
                    json.dumps(record.model_dump(), default=str, indent=2)
                )
            except Exception as e:
                logger.warning(f"Failed to write audit JSON for {record.task_id}: {e}")

    def get(self, task_id: str) -> ExperienceRecord:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM experiences WHERE task_id = ?", (task_id,)).fetchone()
            if row is None:
                raise KeyError(f"No experience found for task_id={task_id!r}")
            cand_rows = conn.execute(
                "SELECT * FROM candidate_results WHERE task_id = ?", (task_id,)
            ).fetchall()
        candidates = [
            {"model_key": r["model_key"], "status": r["status"],
             "best_params": json.loads(r["best_params_json"]) if r["best_params_json"] else None,
             "best_score": r["best_score"], "best_score_std": r["best_score_std"],
             "n_trials_used": r["n_trials_used"], "duration_s": r["duration_s"],
             "complexity_rank": r["complexity_rank"], "mlflow_run_id": r["mlflow_run_id"],
             "error_type": r["error_type"], "error_message": r["error_message"]}
            for r in cand_rows
        ]
        sol = None
        if row["selected_model_key"]:
            sol = {"model_key": row["selected_model_key"],
                   "validation_score": row["validation_score"],
                   "validation_std": row["validation_std"]}
        return ExperienceRecord(
            task_id=row["task_id"], problem_type=row["problem_type"],
            dataset_name=row["dataset_name"],
            dataset_profile=json.loads(row["dataset_profile_json"]),
            training_plan_input=json.loads(row["training_plan_json"]),
            mlflow={"parent_run_id": row["mlflow_parent_run_id"] or ""},
            metric_to_optimize=row["metric_to_optimize"],
            metric_direction=row["metric_direction"],
            models_tested=candidates, selected_solution=sol,
            experience_summary=row["experience_summary"],
        )

    def count(self, problem_type: str | None = None) -> int:
        with self._conn() as conn:
            if problem_type:
                return conn.execute(
                    "SELECT COUNT(*) FROM experiences WHERE problem_type = ?", (problem_type,)
                ).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]

    def find_similar(self, profile: dict[str, Any], problem_type: str, k: int = 5) -> list[RetrievalView]:
        """Weighted-overlap retrieval — implemented in Task 4."""
        from mlops_agents.experience.retrieval import find_similar_impl
        return find_similar_impl(self, profile, problem_type, k)
