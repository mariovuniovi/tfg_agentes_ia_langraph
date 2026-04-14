"""Loguru-based logging setup."""

import sys
from loguru import logger
from mlops_agents.config.settings import settings


def get_logger(name: str):
    """Return a loguru logger bound with the module name."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[name]}</cyan> | {message}",
    )
    return logger.bind(name=name)
