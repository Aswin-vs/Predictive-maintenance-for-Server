"""
Edge Logging Module (lightweight, no server.utils dependency).

Standalone logging setup for RPi4 edge components. Mirrors the
server's logger.py but doesn't require server.utils to be importable.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler

_EDGE_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_EDGE_DIR)
LOG_DIR = os.path.join(_BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

_FILE_FORMAT = logging.Formatter(
    fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

_CONSOLE_FORMAT = logging.Formatter(
    fmt='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)

_loggers = {}


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get or create a named logger with console + rotating file handlers."""
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(_CONSOLE_FORMAT)
    logger.addHandler(console_handler)

    log_file = os.path.join(LOG_DIR, f'{name}.log')
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(_FILE_FORMAT)
    logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger
