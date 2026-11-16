"""
CanineVision AI — Centralised Logging Setup.

Why this matters:
    Without centralised logging, every module prints to stdout
    differently. A single logger configuration means all modules
    write consistent, timestamped, levelled log entries to both
    the console and an optional log file.

    In production systems, logs are the primary debugging tool.
    You cannot attach a debugger to a running server.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Dataset loaded: 20580 images")
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: Path | None = None, level: str = "INFO") -> logging.Logger:
    """
    Create and return a named logger with consistent formatting.

    Args:
        name:     Module name — pass __name__ from the calling module.
        log_file: Optional path to write log entries to a file as well.
        level:    Logging level string — "DEBUG", "INFO", "WARNING", "ERROR".

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # --- Formatter: timestamp | module | level | message ---
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Console handler: always present ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File handler: only when a path is provided ---
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent log records from propagating to the root logger (avoids duplicate output)
    logger.propagate = False

    return logger
