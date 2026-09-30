"""Logging setup.

One rule matters here: the user-facing UI shows friendly messages, while the
full detail of every failure goes to the log. ``log_exception`` exists so the
UI can render a short message while still capturing the traceback.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Final

_CONFIGURED = False
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"


def setup_logging(level: str | int | None = None) -> None:
    """Configure root logging once, writing to stderr.

    Streamlit captures stderr and shows it in the terminal, which is exactly
    where a developer debugging a local run wants it. Nothing here writes to
    the browser.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = level or os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(resolved, str):
        resolved = getattr(logging, resolved.upper(), logging.INFO)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    root = logging.getLogger()
    root.setLevel(resolved)
    root.handlers = [handler]
    _quiet_noisy_libraries(resolved)
    _CONFIGURED = True


#: Third-party loggers that are either far too chatty at INFO or actively
#: unhelpful. Silencing them keeps the terminal readable during a demo.
_NOISY_LOGGERS: Final[tuple[str, ...]] = (
    "streamlit",
    "httpx",
    "httpcore",
    "urllib3",
    "requests",
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
    "torch",
    "faiss",
    "numexpr",
    "PIL",
    "openai",
    "ollama",
    "filelock",
    "matplotlib",
)


def _quiet_noisy_libraries(level: int) -> None:
    noisy_level = max(level, logging.WARNING)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(noisy_level)


def get_logger(name: str) -> logging.Logger:
    """Get a namespaced logger, configuring logging on first use."""
    setup_logging()
    return logging.getLogger(name)


def log_exception(logger: logging.Logger, message: str, exc: BaseException) -> None:
    """Log ``exc`` with a full traceback under a short ``message``.

    Called from the UI error boundary: the user sees ``message``, the terminal
    gets the traceback.
    """
    logger.error("%s: %s: %s", message, type(exc).__name__, exc, exc_info=exc)


__all__ = ["setup_logging", "get_logger", "log_exception", "LOG_FORMAT"]
