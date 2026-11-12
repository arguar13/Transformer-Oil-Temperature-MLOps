"""Structured logging: JSON in production, human-readable in a local TTY.

One JSON line per event is what CloudWatch Logs Insights / Elasticsearch need to
filter and aggregate by field (`event`, `level`, `logger`, timestamp) without
parsing free text. In an interactive terminal (local Docker Compose,
`poetry run`) it renders in colour instead.
"""

from __future__ import annotations

import logging
import sys
from typing import cast

import structlog


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structlog once, at the process entrypoint.

    Idempotent: calling it again just replaces the global configuration with
    the same one.
    """
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: structlog.typing.Processor = (
        structlog.dev.ConsoleRenderer()
        if sys.stderr.isatty()
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    return cast("structlog.typing.FilteringBoundLogger", structlog.get_logger(name))
