"""Loguru-based logging setup.

Handler registration happens once at module level. get_logger() only
binds a name into the context — it never touches global handler state.
"""

import sys
from typing import Any

from loguru import logger
from mlops_agents.config.settings import settings

# Configure once — any subsequent import of this module reuses the same handlers
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


def get_logger(name: str) -> Any:
    """Return a loguru logger bound with the module name.

    Safe to call many times — does not alter the global handler list.
    """
    return logger.bind(name=name)
