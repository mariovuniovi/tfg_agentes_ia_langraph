"""Regression tests for the loguru handler bug (E8-1)."""

import contextlib
from io import StringIO

from loguru import logger as _logger


def test_get_logger_does_not_remove_external_sinks():
    """
    Regression: each get_logger() call must NOT call logger.remove().

    With the bug: get_logger() strips ALL sinks (including externally added ones).
    With the fix: get_logger() never touches global handler state after module load.
    """
    from mlops_agents.utils.logging import get_logger

    buf = StringIO()
    sink_id = _logger.add(buf, format="{message}", colorize=False)

    try:
        # Simulate multiple module-level get_logger calls (as happens on import)
        get_logger("module_a")
        get_logger("module_b")
        get_logger("module_c")

        # If the bug exists, our sink was removed by the get_logger calls above
        _logger.info("probe_message")
        assert "probe_message" in buf.getvalue(), (
            "get_logger() called logger.remove() and stripped the external sink"
        )
    finally:
        # sink may already be gone if the bug removed it — suppress the error
        with contextlib.suppress(Exception):
            _logger.remove(sink_id)


def test_get_logger_returns_logger_bound_with_name():
    """get_logger(name) must return a logger with 'name' in its extra context."""
    from mlops_agents.utils.logging import get_logger

    log = get_logger("my_module")

    # Test behaviorally: the bound logger should not raise when used
    buf = StringIO()
    sink_id = _logger.add(buf, format="{extra[name]}: {message}", colorize=False)
    try:
        log.info("hello")
        assert "my_module: hello" in buf.getvalue()
    finally:
        _logger.remove(sink_id)
