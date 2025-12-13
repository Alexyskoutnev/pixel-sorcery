"""
Global logging configuration for pixel-sorcery.

Usage:
    from src.utils.logging import logger

    logger.info("Training started")
    logger.debug("Batch processed")
    logger.warning("Low memory")
    logger.error("Failed to load image")
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "pixel-sorcery",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up and return a configured logger.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to also log to
        format_string: Custom format string

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Default format
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


# Global logger instance
logger = setup_logger()


def set_log_level(level: str) -> None:
    """
    Set the global log level.

    Args:
        level: "DEBUG", "INFO", "WARNING", or "ERROR"
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    if level.upper() not in level_map:
        raise ValueError(f"Invalid log level: {level}")

    logger.setLevel(level_map[level.upper()])
    for handler in logger.handlers:
        handler.setLevel(level_map[level.upper()])


def add_file_logging(log_file: str) -> None:
    """
    Add file logging to the global logger.

    Args:
        log_file: Path to log file
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logger.level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    )
    logger.addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")
