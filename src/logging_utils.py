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
_CONFIGURED_LEVEL: int | None = None
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"


def _apply(root: logging.Logger, level: int) -> None:
    root.setLevel(level)
    # Only install our handler when the root logger has none. Replacing
    # `root.handlers` outright discards the logging config uvicorn sets up,
    # which silently removes its access and error logs.
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.handlers = [handler]
    _quiet_noisy_libraries(level)


def setup_logging(level: str | int | None = None) -> None:
    """Configure root logging, writing to stderr.

    Streamlit captures stderr and shows it in the terminal, which is exactly
    where a developer debugging a local run wants it. Nothing here writes to
    the browser.

    Every module calls ``get_logger`` at import time, which is *before* ``.env``
    is read by ``Settings.from_env()``. A plain "configure once" guard therefore
    latched the level from the process environment and silently discarded
    ``LOG_LEVEL`` from ``.env``, making the later explicit call a no-op. A
    subsequent call that passes an explicit level is now honoured.
    """
    global _CONFIGURED, _CONFIGURED_LEVEL

    resolved = level or os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(resolved, str):
        resolved = getattr(logging, resolved.upper(), logging.INFO)

    root = logging.getLogger()
    if _CONFIGURED:
        if level and resolved != _CONFIGURED_LEVEL:
            _apply(root, resolved)
            _CONFIGURED_LEVEL = resolved
        return

    _apply(root, resolved)
    _CONFIGURED = True
    _CONFIGURED_LEVEL = resolved


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

    For an error boundary: the user sees ``message``, the terminal gets the
    traceback.
    """
    logger.error("%s: %s: %s", message, type(exc).__name__, exc, exc_info=exc)


__all__ = ["setup_logging", "get_logger", "log_exception", "LOG_FORMAT"]
