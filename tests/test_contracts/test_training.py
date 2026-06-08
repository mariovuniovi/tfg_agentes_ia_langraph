"""Unit tests for training contracts (exog strategies, validation settings)."""

import pytest
from pydantic import ValidationError

from mlops_agents.contracts.training import ExogStrategySettings


def test_drop_strategy_is_rejected_default():
    with pytest.raises(ValidationError):
        ExogStrategySettings(default_unknown_future="drop")


def test_drop_strategy_is_rejected_per_column():
    with pytest.raises(ValidationError):
        ExogStrategySettings(per_column={"temp": "drop"})
