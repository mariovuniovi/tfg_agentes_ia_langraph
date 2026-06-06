"""Unit tests for node→state update contracts."""

import pytest
from pydantic import ValidationError

from mlops_agents.contracts.outputs import StateUpdate


class _Sample(StateUpdate):
    foo: str = "x"


def test_to_update_returns_plain_dict():
    assert _Sample(foo="hello").to_update() == {"foo": "hello"}


def test_to_update_merges_messages_when_provided():
    assert _Sample(foo="hello").to_update(messages=["m1"]) == {"foo": "hello", "messages": ["m1"]}


def test_to_update_omits_messages_key_when_none():
    assert "messages" not in _Sample(foo="hello").to_update()


def test_extra_keys_are_forbidden():
    with pytest.raises(ValidationError):
        _Sample(foo="hello", bogus=1)
