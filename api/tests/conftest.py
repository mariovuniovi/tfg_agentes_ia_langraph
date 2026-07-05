"""Shared fixtures for api tests."""
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient


async def aiter(items):
    for item in items:
        yield item


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature_2": [0.5, 1.5, 2.5, 3.5, 4.5],
        "target": [0, 1, 0, 1, 0],
    })


@pytest.fixture()
def mock_graph():
    """Mock LangGraph graph with astream() and aget_state()."""
    graph = MagicMock()
    graph.astream = AsyncMock(return_value=aiter([]))
    graph.aget_state = AsyncMock()
    graph.aget_state.return_value.values = {
        "processed_dataset_path": "",
        "training_metrics": {},
        "validation_report": {},
    }
    graph.aget_state.return_value.next = ("supervisor",)
    return graph


@pytest.fixture()
async def async_client():
    from api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
