"""Unit tests for new _dataset_sources handlers.

All tests are offline — they mock HTTP and use only statsmodels (static data).
"""
from __future__ import annotations
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# statsmodels source handlers
# ---------------------------------------------------------------------------

def test_fetch_statsmodels_co2():
    from scripts._dataset_sources import fetch_dataset
    entry = {"source": "statsmodels", "source_id": "co2", "dataset_id": "co2_mauna_loa"}
    df = fetch_dataset(entry)
    assert "date" in df.columns
    assert "co2" in df.columns
    assert len(df) >= 250
    # monthly dates — all first-of-month
    assert pd.to_datetime(df["date"]).dt.day.eq(1).all()


def test_fetch_statsmodels_sunspots():
    from scripts._dataset_sources import fetch_dataset
    entry = {"source": "statsmodels", "source_id": "sunspots", "dataset_id": "sunspots"}
    df = fetch_dataset(entry)
    assert "year" in df.columns
    assert "sunspots" in df.columns
    assert len(df) >= 200


def test_fetch_statsmodels_nile():
    from scripts._dataset_sources import fetch_dataset
    entry = {"source": "statsmodels", "source_id": "nile", "dataset_id": "nile"}
    df = fetch_dataset(entry)
    assert "year" in df.columns
    assert "volume" in df.columns
    assert len(df) >= 90


def test_fetch_statsmodels_air_passengers():
    from scripts._dataset_sources import fetch_dataset
    entry = {"source": "statsmodels", "source_id": "air_passengers",
             "dataset_id": "air_passengers"}
    df = fetch_dataset(entry)
    assert "month" in df.columns
    assert "passengers" in df.columns
    assert len(df) == 144


def test_fetch_statsmodels_unknown_raises():
    from scripts._dataset_sources import fetch_dataset
    with pytest.raises(ValueError, match="Unknown statsmodels"):
        fetch_dataset({"source": "statsmodels", "source_id": "nope", "dataset_id": "nope"})


# ---------------------------------------------------------------------------
# uci_url handler — mock HTTP to avoid network in unit tests
# ---------------------------------------------------------------------------

def _make_metro_raw() -> str:
    """Minimal Metro Traffic Volume CSV content."""
    rows = []
    rows.append("holiday,temp,rain_1h,snow_1h,clouds_all,weather_main,"
                "weather_description,date_time,traffic_volume")
    for i in range(50):
        dt = pd.Timestamp("2018-01-01") + pd.Timedelta(hours=i)
        holiday = "None" if i % 24 != 0 else "New Years Day"
        rows.append(f"{holiday},280.0,0.0,0.0,40,Clouds,broken clouds,"
                    f"{dt.strftime('%Y-%m-%d %H:%M:%S')},3000")
    return "\n".join(rows)


def test_fetch_uci_url_metro(monkeypatch):
    import io
    import pandas as pd
    from scripts._dataset_sources import fetch_dataset

    raw_csv = _make_metro_raw()
    _real_read_csv = pd.read_csv  # capture before patching

    def _mock_read_csv(url, **kwargs):
        kwargs.pop("parse_dates", None)
        df = _real_read_csv(io.StringIO(raw_csv))
        df["date_time"] = pd.to_datetime(df["date_time"])
        return df

    monkeypatch.setattr("pandas.read_csv", _mock_read_csv)

    url = "https://example.com/metro.csv.gz"
    entry = {"source": "uci_url", "source_id": url,
             "dataset_id": "metro_traffic_volume"}
    df = fetch_dataset(entry)

    assert "date" in df.columns
    assert "traffic_volume" in df.columns
    assert "is_holiday" in df.columns
    assert len(df) >= 1  # at least one daily row


def test_fetch_uci_url_unknown_raises():
    from scripts._dataset_sources import fetch_dataset
    with pytest.raises(ValueError, match="No uci_url handler"):
        fetch_dataset({"source": "uci_url", "source_id": "https://x.com/f.csv",
                       "dataset_id": "unknown_dataset"})


# ---------------------------------------------------------------------------
# Family-check unit tests
# ---------------------------------------------------------------------------

def test_family_check_statistical() -> None:
    from scripts.run_benchmark import _check_family
    assert _check_family("ets", "statistical") is True
    assert _check_family("auto_arima", "statistical") is True
    assert _check_family("naive", "statistical") is True
    assert _check_family("lightgbm_forecaster", "statistical") is False


def test_family_check_random_walk() -> None:
    from scripts.run_benchmark import _check_family
    assert _check_family("naive", "random_walk") is True
    assert _check_family("ets", "random_walk") is False      # only naive qualifies
    assert _check_family("seasonal_naive", "random_walk") is False


def test_family_check_supervised() -> None:
    from scripts.run_benchmark import _check_family
    assert _check_family("lightgbm_forecaster", "supervised") is True
    assert _check_family("random_forest_forecaster", "supervised") is True
    assert _check_family("ets", "supervised") is False
    assert _check_family("naive", "supervised") is False


def test_reset_forecasting_experiences(tmp_path) -> None:
    from mlops_agents.experience.pool import ExperiencePool
    pool = ExperiencePool(tmp_path / "test.db", audit_dir=tmp_path)
    with pool._conn() as conn:
        for task_id, problem_type in [("fc1", "forecasting"), ("cls1", "classification")]:
            conn.execute(
                """INSERT INTO experiences
                   (task_id, problem_type, dataset_profile_json,
                    training_plan_json, created_at)
                   VALUES (?, ?, '{}', '{}', datetime('now'))""",
                (task_id, problem_type),
            )
    assert pool.count("forecasting") == 1
    assert pool.count("classification") == 1

    n = pool.reset_forecasting_experiences()

    assert n == 1
    assert pool.count("forecasting") == 0
    assert pool.count("classification") == 1  # untouched
