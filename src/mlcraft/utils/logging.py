"""Logging utilities for mlcraft."""

from __future__ import annotations

import logging
from typing import Any

PACKAGE_LOGGER_NAME = "mlcraft"
_VERBOSITY_TO_LEVEL = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
    3: logging.DEBUG,
}


def _resolve_level(verbose: int) -> int:
    if verbose <= 0:
        return logging.WARNING
    if verbose == 1:
        return logging.INFO
    return logging.DEBUG


def configure_logging(
    verbose: int = 0,
    *,
    logger_name: str = PACKAGE_LOGGER_NAME,
    logger: logging.Logger | None = None,
    handler: logging.Handler | None = None,
    fmt: str | None = None,
) -> logging.Logger:
    """Configure and return a package logger."""

    target = logger or logging.getLogger(logger_name)
    target.setLevel(_resolve_level(verbose))
    if not target.handlers:
        stream_handler = handler or logging.StreamHandler()
        formatter = logging.Formatter(fmt or "%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        stream_handler.setFormatter(formatter)
        target.addHandler(stream_handler)
    target.propagate = False
    return target


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger inside the package namespace."""

    if not name:
        return logging.getLogger(PACKAGE_LOGGER_NAME)
    if name.startswith(PACKAGE_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")


def set_verbosity(verbose: int = 0, *, logger_name: str = PACKAGE_LOGGER_NAME) -> logging.Logger:
    """Update the level of an existing logger and its handlers."""

    logger = logging.getLogger(logger_name)
    level = _resolve_level(verbose)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
    return logger


def inject_logger(logger: logging.Logger | None, name: str | None = None) -> logging.Logger:
    """Return a custom logger or fall back to a package logger."""

    return logger or get_logger(name)


def log_kv(logger: logging.Logger, message: str, **kwargs: Any) -> None:
    """Helper for consistent key-value logging."""

    if not kwargs:
        logger.info(message)
        return
    payload = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
    logger.info("%s | %s", message, payload)

