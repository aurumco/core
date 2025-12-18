"""Logging configuration for the project.

This module provides structured logging capabilities, following the
Constitution's guidelines for observability and error handling.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict

# Constants
LOG_FORMAT = "%(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


class JsonFormatter(logging.Formatter):
    """Formatter to output logs in JSON format for production environments.

    Adheres to Constitution Article XIV.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record as a JSON string.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: JSON string representation of the log.
        """
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).strftime(DATE_FORMAT),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "path": record.pathname,
            "line": record.lineno,
        }

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Include any extra attributes
        if hasattr(record, "context"):
            log_obj["context"] = getattr(record, "context")

        return json.dumps(log_obj)


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

    # Use JSON formatter for structured logging
    formatter = JsonFormatter(LOG_FORMAT)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
