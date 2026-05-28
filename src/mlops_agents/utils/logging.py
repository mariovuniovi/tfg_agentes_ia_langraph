"""Loguru-based logging setup.

Handler registration happens once at module level. get_logger() only
binds a name into the context — it never touches global handler state.
"""

import sys
from pathlib import Path
from typing import Any

from loguru import logger
from mlops_agents.config.settings import settings

_LOG_FILE = Path("logs/pipeline.log")

logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[name]}</cyan> | "
        "{message}"
    ),
)
logger.add(
    _LOG_FILE,
    level="DEBUG",
    rotation="10 MB",
    retention=3,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[name]} | {message}",
    encoding="utf-8",
)


def get_logger(name: str) -> Any:
    """Return a loguru logger bound with the module name.

    Safe to call many times — does not alter the global handler list.
    """
    return logger.bind(name=name)
