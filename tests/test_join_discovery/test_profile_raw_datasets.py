from __future__ import annotations

import pytest
from mlops_agents.tools.join_discovery_tools import profile_raw_datasets


@pytest.fixture
def sample_csvs(tmp_path):
    energy = tmp_path / "energy.csv"
    weather = tmp_path / "weather.csv"
    energy.write_text("week_date,kwh_consumed\n2024-01-01,100\n2024-01-08,120\n")
    weather.write_text("week_date,avg_temp_c\n2024-01-01,10\n2024-01-08,12\n")
    return {"energy": str(energy), "weather": str(weather)}


def test_profile_returns_both_datasets(sample_csvs) -> None:
    profiles = profile_raw_datasets(sample_csvs)
    assert len(profiles) == 2
    names = {p.dataset_name for p in profiles}
    assert names == {"energy", "weather"}


def test_profile_column_stats(sample_csvs) -> None:
    profiles = profile_raw_datasets(sample_csvs)
    energy = next(p for p in profiles if p.dataset_name == "energy")
    assert energy.n_rows == 2
    assert energy.n_columns == 2
    week_col = next(c for c in energy.columns if c.column_name == "week_date")
    assert week_col.null_rate == 0.0
    assert week_col.unique_count == 2
    assert week_col.unique_ratio == 1.0


def test_profile_head_rows(sample_csvs) -> None:
    profiles = profile_raw_datasets(sample_csvs)
    energy = next(p for p in profiles if p.dataset_name == "energy")
    assert len(energy.head_rows) == 2  # only 2 rows in fixture
    assert "week_date" in energy.head_rows[0]
    assert "kwh_consumed" in energy.head_rows[0]


def test_profile_skips_missing_file() -> None:
    profiles = profile_raw_datasets({"missing": "/nonexistent/path/file.csv"})
    assert profiles == []
