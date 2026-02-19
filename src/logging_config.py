"""
Logging configuration for the Sentiment-Forecasting pipeline.

Provides structured logging setup with console and file handlers.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

from src.config import OUTPUT_DIR

# Default log format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log levels
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_configured = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Parameters
    ----------
    name : str
        Logger name (typically __name__ from the calling module)

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    return logging.getLogger(name)


def configure_logging(
    level: LogLevel = "INFO",
    log_file: str | Path | None = None,
    console: bool = True,
) -> None:
    """
    Configure the root logger for the pipeline.

    Parameters
    ----------
    level : LogLevel
        Minimum log level to capture
    log_file : str | Path | None
        Optional path to log file. If None, logs only to console.
    console : bool
        Whether to log to console (default: True)
    """
    global _configured

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))

    # Clear existing handlers if reconfiguring
    if _configured:
        root_logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _configured = True


def configure_pipeline_logging(
    level: LogLevel = "INFO",
    enable_file_log: bool = False,
) -> None:
    """
    Configure logging specifically for pipeline runs.

    Parameters
    ----------
    level : LogLevel
        Minimum log level
    enable_file_log : bool
        Whether to also log to a file in the output directory
    """
    log_file = None
    if enable_file_log:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        log_file = OUTPUT_DIR / "pipeline.log"

    configure_logging(level=level, log_file=log_file, console=True)


# Configure basic logging on import (can be reconfigured later)
if not _configured:
    configure_logging(level="WARNING", console=True)
