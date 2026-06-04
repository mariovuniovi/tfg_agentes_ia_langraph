"""Join discovery tools — profiling, candidate evaluation, and plan execution."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from mlops_agents.config.settings import settings
from mlops_agents.contracts.join_discovery import ColumnProfile, RawDatasetProfile
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


def profile_raw_datasets(raw_paths: dict[str, str]) -> list[RawDatasetProfile]:
    """Profile raw datasets deterministically. Called before the agent starts — not a tool.

    Stats are computed on the full dataset. head_rows contains the first
    settings.data_validator_profile_nrows rows for the agent to inspect.

    Args:
        raw_paths: mapping of dataset_name → file path

    Returns:
        List of RawDatasetProfile objects.
    """
    head_n = settings.data_validator_profile_nrows
    profiles: list[RawDatasetProfile] = []

    for dataset_name, path_str in raw_paths.items():
        path = Path(path_str)
        if not path.exists():
            logger.warning(f"[profile] file not found: {path_str}")
            continue

        df = pd.read_csv(path)
        n_rows = len(df)
        col_profiles: list[ColumnProfile] = []

        for col in df.columns:
            series = df[col]
            non_null = series.dropna()
            null_rate = round(float(series.isnull().sum()) / max(n_rows, 1), 4)
            unique_count = int(non_null.nunique())
            unique_ratio = round(unique_count / max(len(non_null), 1), 4)

            min_val = max_val = None
            if (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series)) and not non_null.empty:
                min_val = str(non_null.min())
                max_val = str(non_null.max())

            col_profiles.append(ColumnProfile(
                column_name=col,
                dtype=str(series.dtype),
                non_null_count=int(len(non_null)),
                null_rate=null_rate,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                min_value=min_val,
                max_value=max_val,
            ))

        profiles.append(RawDatasetProfile(
            dataset_name=dataset_name,
            path=path_str,
            n_rows=n_rows,
            n_columns=len(df.columns),
            columns=col_profiles,
            head_rows=df.head(head_n).to_dict(orient="records"),
        ))

    logger.info(f"[profile] profiled {len(profiles)} datasets, {sum(p.n_columns for p in profiles)} total columns")
    return profiles
