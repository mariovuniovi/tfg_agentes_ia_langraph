"""Experience record writer: builds the long-term JSON dump per pipeline run."""
from __future__ import annotations
import json
from datetime import date
from pathlib import Path
from typing import Any


def build_task_id(dataset_stem: str, problem_type: str, run_idx: int = 1) -> str:
    """Format: <stem>_<problem_type>_<YYYY-MM-DD>_<NNN>."""
    today = date.today().strftime("%Y-%m-%d")
    return f"{dataset_stem}_{problem_type}_{today}_{run_idx:03d}"


def write_experience_record(record: dict[str, Any], output_dir: Path) -> Path:
    """Write the JSON record. Returns the file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{record['task_id']}.json"
    out_path.write_text(json.dumps(record, default=str, indent=2))
    return out_path
