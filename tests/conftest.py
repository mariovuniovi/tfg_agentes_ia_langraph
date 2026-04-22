"""Shared pytest fixtures — check here before creating new fixtures."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    """Minimal 5-row CSV with a 'target' column. Use for data_tools tests."""
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature_2": [0.5, 1.5, 2.5, 3.5, 4.5],
        "target": [0, 1, 0, 1, 0],
    })
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def sample_csv_with_missing(tmp_path: Path) -> Path:
    """CSV where feature_2 has 2 missing values (40%)."""
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature_2": [0.5, None, 2.5, None, 4.5],
        "target": [0, 1, 0, 1, 0],
    })
    path = tmp_path / "missing_data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def larger_csv(tmp_path: Path) -> Path:
    """60-row balanced binary classification CSV — enough for train/val split + CV.

    Use for training_tools tests that require sklearn cross-validation.
    """
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame({
        "feature_1": rng.normal(0, 1, n),
        "feature_2": rng.normal(1, 1, n),
        "feature_3": rng.uniform(0, 10, n),
        "target": ([0] * (n // 2) + [1] * (n // 2)),
    })
    path = tmp_path / "larger_data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def mock_llm():
    """Mock ChatOpenAI to avoid real LLM calls in unit tests."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Mocked LLM response", tool_calls=[])
    return mock


@pytest.fixture()
def iris_schema_file(tmp_path: Path) -> Path:
    """Iris schema JSON file for validate_against_schema tests."""
    import json
    schema = {
        "name": "iris_classification",
        "columns": [
            {"name": "sepal_length", "dtype": "float", "required": True, "nullable": False, "min": 0.0, "max": 30.0},
            {"name": "sepal_width",  "dtype": "float", "required": True, "nullable": False, "min": 0.0, "max": 30.0},
            {"name": "petal_length", "dtype": "float", "required": True, "nullable": False, "min": 0.0, "max": 30.0},
            {"name": "petal_width",  "dtype": "float", "required": True, "nullable": False, "min": 0.0, "max": 30.0},
            {"name": "sample_id",    "dtype": "int",   "required": True, "nullable": False, "is_key": True},
            {"name": "target",       "dtype": "str",   "required": True, "nullable": False,
             "allowed_values": ["setosa", "versicolor", "virginica"]},
        ],
    }
    path = tmp_path / "iris_classification.json"
    path.write_text(json.dumps(schema))
    return path


@pytest.fixture()
def canonical_iris_csv(tmp_path: Path) -> Path:
    """Valid canonical iris CSV — all constraints satisfied."""
    df = pd.DataFrame({
        "sepal_length": [5.1, 4.9, 4.7],
        "sepal_width":  [3.5, 3.0, 3.2],
        "petal_length": [1.4, 1.4, 1.3],
        "petal_width":  [0.2, 0.2, 0.2],
        "sample_id":    [1, 2, 3],
        "target":       ["setosa", "setosa", "setosa"],
    })
    path = tmp_path / "canonical.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def measurements_csv(tmp_path: Path) -> Path:
    """Raw iris measurements CSV with 'Id' as key column."""
    df = pd.DataFrame({
        "Id": [1, 2, 3],
        "SepalLengthCm": [5.1, 4.9, 4.7],
        "SepalWidthCm":  [3.5, 3.0, 3.2],
        "PetalLengthCm": [1.4, 1.4, 1.3],
        "PetalWidthCm":  [0.2, 0.2, 0.2],
    })
    path = tmp_path / "measurements.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def labels_csv(tmp_path: Path) -> Path:
    """Raw iris labels CSV with 'sample_id' as key column."""
    df = pd.DataFrame({
        "sample_id": [1, 2, 3],
        "species":   ["setosa", "setosa", "setosa"],
    })
    path = tmp_path / "labels.csv"
    df.to_csv(path, index=False)
    return path
