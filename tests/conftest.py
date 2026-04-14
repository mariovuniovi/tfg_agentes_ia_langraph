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
