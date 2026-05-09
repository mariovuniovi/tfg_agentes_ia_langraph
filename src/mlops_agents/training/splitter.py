"""Train/pool/test split for classification, regression, and forecasting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_agents.config.settings import settings


def split_dataset(
    canonical_path: Path,
    task_metadata: dict[str, Any],
    output_dir: Path,
    test_size: float | None = None,
    random_state: int = 42,
) -> tuple[Path, Path, Path]:
    """Write train_pool, test, and split_metadata files. Returns the three paths."""
    test_size = test_size if test_size is not None else settings.train_test_split_ratio
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = canonical_path.stem
    train_pool_path = output_dir / f"{stem}_train_pool.csv"
    test_path = output_dir / f"{stem}_test.csv"
    metadata_path = output_dir / f"{stem}_split_metadata.json"

    problem_type = task_metadata["problem_type"]
    target = task_metadata["target_column"]
    df = pd.read_csv(canonical_path)

    if problem_type == "classification":
        train_pool, test = train_test_split(
            df, test_size=test_size, stratify=df[target], random_state=random_state,
        )
        metadata = {
            "split_kind": "stratified",
            "n_train_pool": len(train_pool),
            "n_test": len(test),
            "test_size_ratio": test_size,
            "forecast_horizon": None,
            "n_series_total": None,
            "n_series_dropped": 0,
            "dropped_series": [],
            "random_state": random_state,
        }
    elif problem_type == "regression":
        train_pool, test = train_test_split(
            df, test_size=test_size, shuffle=True, random_state=random_state,
        )
        metadata = {
            "split_kind": "random_shuffle",
            "n_train_pool": len(train_pool),
            "n_test": len(test),
            "test_size_ratio": test_size,
            "forecast_horizon": None,
            "n_series_total": None,
            "n_series_dropped": 0,
            "dropped_series": [],
            "random_state": random_state,
        }
    elif problem_type == "forecasting":
        dt_col = task_metadata["datetime_column"]
        sid_cols = task_metadata.get("series_id_columns") or []
        horizon = int(task_metadata["forecast_horizon"])
        df[dt_col] = pd.to_datetime(df[dt_col])

        # Series length guard: each series needs horizon + horizon + min_train_points
        min_required = 2 * horizon + settings.forecasting_min_train_points
        if sid_cols:
            grouped = df.groupby(sid_cols)
            short = [(name, len(g)) for name, g in grouped if len(g) < min_required]
            if len(short) > grouped.ngroups / 2:
                raise ValueError(
                    f"More than half of series ({len(short)}/{grouped.ngroups}) are too short "
                    f"(need >= {min_required} obs). Reduce forecast_horizon or filter dataset."
                )
            short_keys = {tuple(name) if isinstance(name, tuple) else (name,) for name, _ in short}
            df = df[~df[sid_cols].apply(tuple, axis=1).isin(short_keys)]
            n_total = grouped.ngroups
            dropped = [
                {"series_id": dict(zip(sid_cols, name)) if isinstance(name, tuple) else {sid_cols[0]: name},
                 "n_obs": n}
                for name, n in short
            ]
        else:
            if len(df) < min_required:
                raise ValueError(
                    f"Series too short ({len(df)} < {min_required} obs). "
                    f"Reduce forecast_horizon."
                )
            n_total = 1
            dropped = []

        # Per-series temporal split: last `horizon` rows of each series → test
        df = df.sort_values(sid_cols + [dt_col]) if sid_cols else df.sort_values(dt_col)
        if sid_cols:
            test = df.groupby(sid_cols).tail(horizon)
            train_pool = df.drop(test.index)
        else:
            test = df.tail(horizon)
            train_pool = df.iloc[:-horizon]

        metadata = {
            "split_kind": "temporal_per_series",
            "n_train_pool": len(train_pool),
            "n_test": len(test),
            "test_size_ratio": None,
            "forecast_horizon": horizon,
            "n_series_total": n_total,
            "n_series_dropped": len(dropped),
            "dropped_series": dropped,
            "random_state": random_state,
        }
    else:
        raise ValueError(f"Unknown problem_type: {problem_type!r}")

    train_pool.to_csv(train_pool_path, index=False)
    test.to_csv(test_path, index=False)
    metadata_path.write_text(json.dumps(metadata, default=str, indent=2))

    return train_pool_path, test_path, metadata_path
