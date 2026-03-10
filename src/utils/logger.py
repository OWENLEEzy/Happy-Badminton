"""
Logging configuration module using loguru.
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logger(
    log_file: str = "logs/badminton.log",
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "30 days",
):
    """
    Configure the loguru logger with console and file handlers.

    Args:
        log_file: Path to the log file.
        level: Logging level (e.g. "INFO", "DEBUG").
        rotation: Log rotation threshold (e.g. "10 MB").
        retention: How long to keep rotated logs (e.g. "30 days").
    """
    # Ensure the log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove the default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
    )

    # Add file handler
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level=level,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )

    logger.info(f"Logger initialised: {log_file}")
    return logger
