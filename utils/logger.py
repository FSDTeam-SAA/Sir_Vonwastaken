"""
utils/logger.py

Logging configuration using loguru with rotation and structured output.

Every module imports `logger` from here rather than configuring its own logging.
Supports both console output and file rotation for production deployments.
"""
from __future__ import annotations

import sys

from loguru import logger as loguru_logger

from config.settings import settings

# Remove default handler
loguru_logger.remove()

# Console handler with color
log_level = settings.log_level.upper() if hasattr(settings, 'log_level') else 'INFO'

loguru_logger.add(
    sys.stdout,
    format="<level>{level: <8}</level> | <cyan>{name}:{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=log_level,
    colorize=True,
)

# File handler with rotation (for production)
# Rotates daily, keeps 7 days of logs
loguru_logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    level=log_level,
    rotation="00:00",  # Rotate at midnight
    retention="7 days",  # Keep 7 days of logs
    compression="zip",  # Compress old logs
)

# JSON structured logging for log aggregation services (optional)
if settings.app_env == "production":
    loguru_logger.add(
        "logs/app_structured_{time:YYYY-MM-DD}.jsonl",
        format="{message}",
        level=log_level,
        rotation="00:00",
        retention="7 days",
        compression="zip",
        serialize=True,  # JSON output
    )

logger = loguru_logger
