"""Print a JSON snapshot of the current experience pool state.

Used by the run-benchmark skill to compare pool before/after a benchmark run.

Usage:
    uv run python scripts/pool_snapshot.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mlops_agents.config.settings import settings


def snapshot() -> dict:
    db_path = settings.experience_db_path
    if not db_path.exists():
        return {"total": 0, "by_problem_type": {}, "entries": []}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]

    by_type: dict[str, int] = {}
    for row in conn.execute(
        "SELECT problem_type, COUNT(*) as cnt FROM experiences GROUP BY problem_type"
    ):
        by_type[row["problem_type"]] = row["cnt"]

    entries = []
    for row in conn.execute(
        "SELECT task_id, dataset_name, problem_type, selected_model_key, "
        "validation_score, validation_std, created_at FROM experiences ORDER BY created_at DESC"
    ):
        entries.append({
            "task_id": row["task_id"],
            "dataset_name": row["dataset_name"],
            "problem_type": row["problem_type"],
            "champion_model": row["selected_model_key"],
            "validation_score": row["validation_score"],
            "validation_std": row["validation_std"],
            "created_at": row["created_at"],
        })

    conn.close()
    return {"total": total, "by_problem_type": by_type, "entries": entries}


if __name__ == "__main__":
    print(json.dumps(snapshot(), indent=2))
