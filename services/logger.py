"""
services/logger.py
──────────────────
Centralised LoggerService for the LogSentry project.

Features
--------
* Logs to both the console (stdout) AND a rotating file inside logs/.
* Professional, structured format:
      2026-03-01 00:15:09 | INFO     | services.logger | Starting pipeline
* Log files are rotated at 5 MB; the last 5 rotated files are kept.
* A single call to `get_logger(name)` returns a pre-configured logger so
  every module can share the same logging setup without extra boilerplate.
* Thread-safe: uses standard-library logging which is inherently thread-safe.

Usage
-----
    from services.logger import get_logger

    log = get_logger(__name__)
    log.info("Pipeline started")
    log.warning("Checkpoint not found – running from scratch")
    log.error("Training failed: %s", exc)
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

# ── Constants ──────────────────────────────────────────────────────────────────

LOG_DIR: str = "logs"  # directory that holds log files
LOG_FILE: str = "logsentry.log"  # base filename
LOG_MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB per file
LOG_BACKUP_COUNT: int = 5  # keep 5 rotated files

# Format shared by all handlers
_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# Root logger name for the project (all child loggers inherit from it)
_ROOT_LOGGER_NAME = "logsentry"

# Internal flag so we only set up handlers once
_initialised: bool = False


# ── Internal setup ─────────────────────────────────────────────────────────────


def _setup_root_logger(log_dir: str = LOG_DIR, level: int = logging.DEBUG) -> None:
    """Configure the project root logger exactly once."""
    global _initialised
    if _initialised:
        return

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, LOG_FILE)

    formatter = logging.Formatter(fmt=_FMT, datefmt=_DATE_FMT)

    # ── File handler (DEBUG and above, rotated) ────────────────────────────────
    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # ── Stream handler (INFO and above to stdout) ──────────────────────────────
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(level)

    # Guard against duplicate handlers if someone calls this twice somehow
    if not root.handlers:
        root.addHandler(file_handler)
        root.addHandler(stream_handler)

    # Prevent log records from bubbling to the Python root logger (which
    # would cause duplicate output if any other code calls basicConfig).
    root.propagate = False

    _initialised = True


# ── Public API ─────────────────────────────────────────────────────────────────


def get_logger(
    name: Optional[str] = None, level: int = logging.DEBUG
) -> logging.Logger:
    """
    Return a logger that is a child of the project root logger.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module, e.g. ``"training.pretrain"``.
        When *None* the root project logger is returned directly.
    level:
        Minimum severity captured by this specific child logger.
        Defaults to DEBUG so that the file handler receives everything.

    Returns
    -------
    logging.Logger
        A fully configured logger shared across the whole process.
    """
    _setup_root_logger()

    if name is None or name == _ROOT_LOGGER_NAME:
        logger = logging.getLogger(_ROOT_LOGGER_NAME)
    else:
        # Nest under the project root so level/handler inheritance works
        qualified = (
            f"{_ROOT_LOGGER_NAME}.{name}"
            if not name.startswith(_ROOT_LOGGER_NAME)
            else name
        )
        logger = logging.getLogger(qualified)
        logger.setLevel(level)

    return logger


# ── Convenience wrapper (mirrors the original LoggerService interface) ──────────


class LoggerService:
    """
    Thin OO wrapper around :func:`get_logger`.

    Instantiate once at module level or pass around as a dependency.  Each
    instance binds to a named logger so log records carry the correct module
    path in the formatted output.

    Example
    -------
        from services.logger import LoggerService

        class MyTrainer:
            def __init__(self):
                self._log = LoggerService(__name__)

            def train(self):
                self._log.info("Training started")
    """

    def __init__(self, name: Optional[str] = None, level: int = logging.DEBUG) -> None:
        self._logger = get_logger(name=name, level=level)

    # ── Standard severity helpers ──────────────────────────────────────────────

    def debug(self, message: str, *args, **kwargs) -> None:
        self._logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        self._logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        self._logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        self._logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        self._logger.critical(message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs) -> None:
        """Log ERROR with full traceback (call from inside an except block)."""
        self._logger.exception(message, *args, **kwargs)

    # ── Legacy aliases (backwards-compat with the old __init__.py interface) ───

    def log(self, message: str, *args, **kwargs) -> None:
        self._logger.info(message, *args, **kwargs)

    def log_error(self, message: str, *args, **kwargs) -> None:
        self._logger.error(message, *args, **kwargs)

    def log_warning(self, message: str, *args, **kwargs) -> None:
        self._logger.warning(message, *args, **kwargs)

    def log_debug(self, message: str, *args, **kwargs) -> None:
        self._logger.debug(message, *args, **kwargs)

    def log_critical(self, message: str, *args, **kwargs) -> None:
        self._logger.critical(message, *args, **kwargs)
