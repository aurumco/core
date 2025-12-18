"""Logging configuration for the project.

This module provides structured logging capabilities, following the
Constitution's guidelines for observability and error handling.
"""

import logging
import sys
import os

# Constants
# Minimalist Format: [TIME] [LEVEL] MESSAGE
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
DATE_FORMAT = "%H:%M:%S"


class ContextFilter(logging.Filter):
    """Filters out noisy logs from external libraries."""

    def filter(self, record: logging.LogRecord) -> bool:
        # List of noisy libraries to suppress
        noisy_loggers = [
            "transformers",
            "unsloth",
            "torch",
            "accelerate",
            "bitsandbytes",
        ]
        # Suppress external info/warnings, keep only critical errors
        if any(lib in record.name for lib in noisy_loggers):
            return record.levelno >= logging.ERROR
        return True


def setup_logger(name: str = "ai_core", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a logger instance."""

    # Silence C++ level warnings from TensorFlow/XLA
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        return logger

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handler.setFormatter(formatter)

    # Add the noise filter
    handler.addFilter(ContextFilter())

    logger.addHandler(handler)
    logger.propagate = False

    # Force silence on root logger of external libs explicitly
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("unsloth").setLevel(logging.ERROR)

    return logger
