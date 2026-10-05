"""
Shared Structured Logging Module.

Provides a consistent logging setup across all components (edge, cloud, training).
Uses RotatingFileHandler for production durability + console output for development.

Usage:
    from server.utils.logger import get_logger
    logger = get_logger("inference_server")
    logger.info("Model loaded", extra={"model": "classifier_fp32.h5", "params": 122184})
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Project root — works from any CWD
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# ─── Formatters ───────────────────────────────────────────────────────────────

_FILE_FORMAT = logging.Formatter(
    fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

_CONSOLE_FORMAT = logging.Formatter(
    fmt='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)

# Cache to avoid duplicate handlers
_loggers = {}


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Get or create a named logger with console + rotating file handlers.

    Args:
        name: Logger name (used as filename too, e.g. "inference_server")
        level: Minimum log level
        log_to_file: Whether to write to a rotating log file
        max_bytes: Max size per log file (default 10 MB)
        backup_count: Number of backup files to keep (default 5)

    Returns:
        Configured logging.Logger instance
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate logs from root logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(_CONSOLE_FORMAT)
    logger.addHandler(console_handler)

    # Rotating file handler
    if log_to_file:
        log_file = os.path.join(LOG_DIR, f'{name}.log')
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(_FILE_FORMAT)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger
