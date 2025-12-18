"""Logging configuration for the project.

This module provides structured logging capabilities, following the
Constitution's guidelines for observability and error handling.
"""

import logging
import sys

# Constants
# Simple clean format: [TIME] [LEVEL] MESSAGE
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logger(
    name: str = "ai_core_extractor", level: int = logging.INFO
) -> logging.Logger:
    """Configures and returns a logger instance.

    Args:
        name (str, optional): Name of the logger. Defaults to "ai_core_extractor".
        level (int, optional): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger

    handler = logging.StreamHandler(sys.stdout)

    # Use simple human-readable formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False  # Prevent double logging in some environments

    return logger
