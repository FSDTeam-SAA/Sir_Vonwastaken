"""
utils/logger.py

Single shared logger configuration for the whole app. Every module does
`from utils.logger import logger` instead of configuring its own logging.
"""
from __future__ import annotations

import logging
import sys

from config.settings import settings


def _build_logger() -> logging.Logger:
    log = logging.getLogger("content_trend_assistant")
    if log.handlers:
        return log  # already configured (avoids duplicate handlers on reload)

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    log.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.propagate = False
    return log


logger = _build_logger()
