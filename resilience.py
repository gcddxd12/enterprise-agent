"""Resilience layer: retry, circuit breaker, structured errors, logging.

Provides production-grade error handling for the agent system:
- Structured error taxonomy (retryable / fatal / degraded)
- Exponential backoff with jitter for transient failures
- Graceful degradation from primary to fallback implementations
- Structured logging to replace print() diagnostics

No external dependencies beyond Python stdlib.
"""

from __future__ import annotations

import functools
import logging
import os
import random
import time
from enum import Enum
from typing import Any, Callable, Optional, Tuple, Type


# ============================================================================
# Structured Error Types
# ============================================================================


class ErrorCategory(Enum):
    RETRYABLE = "retryable"   # transient, safe to retry (timeout, rate limit)
    FATAL = "fatal"            # unrecoverable (auth failure, bad config)
    DEGRADED = "degraded"      # partial failure, continue with fallback


class AgentError(Exception):
    """Base error type for the agent system."""
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.FATAL,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.category = category
        self.original_error = original_error


class LLMTimeoutError(AgentError):
    """LLM call timed out."""
    def __init__(self, message: str = "LLM request timed out", original: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.RETRYABLE, original)


class LLMRateLimitError(AgentError):
    """LLM rate limit hit."""
    def __init__(self, message: str = "LLM rate limit exceeded", original: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.RETRYABLE, original)


class ToolExecutionError(AgentError):
    """A tool failed during execution."""
    def __init__(self, tool_name: str, original: Optional[Exception] = None):
        msg = f"Tool '{tool_name}' execution failed"
        super().__init__(msg, ErrorCategory.RETRYABLE, original)


class RetrievalDegradedError(AgentError):
    """Primary retrieval failed but fallback succeeded."""
    def __init__(self, message: str = "Primary retrieval degraded to fallback", original: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.DEGRADED, original)


class InvalidInputError(AgentError):
    """User input or tool parameter validation failed."""
    def __init__(self, message: str):
        super().__init__(message, ErrorCategory.FATAL)


# ============================================================================
# Error Classification
# ============================================================================


def classify_llm_error(exception: Exception) -> AgentError:
    """Classify a raw LLM exception into a structured AgentError.

    Patterns matched (case-insensitive):
    - timeout / timed out / connection → LLMTimeoutError (RETRYABLE)
    - rate limit / too many requests / 429 → LLMRateLimitError (RETRYABLE)
    - auth / unauthorized / 401 / 403 → AgentError(FATAL)
    - everything else → AgentError(RETRYABLE) — safer to assume transient
    """
    msg = str(exception).lower()
    if any(kw in msg for kw in ("timeout", "timed out", "connection", "timed_out")):
        return LLMTimeoutError(original=exception)
    if any(kw in msg for kw in ("rate limit", "too many requests", "429", "rate_limit")):
        return LLMRateLimitError(original=exception)
    if any(kw in msg for kw in ("auth", "unauthorized", "401", "403")):
        return AgentError(str(exception), ErrorCategory.FATAL, exception)
    return AgentError(str(exception), ErrorCategory.RETRYABLE, exception)


# ============================================================================
# Retry with Exponential Backoff
# ============================================================================


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (AgentError,),
):
    """Decorator: retry a function with exponential backoff + optional jitter.

    Delay sequence (base=1s, factor=2): 1s, 2s, 4s
    Jitter adds ±25% random variation to each delay.
    Only retries on exceptions whose category == RETRYABLE (for AgentError)
    or any instance of a retryable_exceptions type.
    Re-raises immediately on FATAL errors.

    Example:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def call_llm(prompt):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = logging.getLogger("cmcc_agent.resilience")
            last_error: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    # Determine if retryable
                    is_retryable = False
                    if isinstance(e, AgentError):
                        is_retryable = e.category == ErrorCategory.RETRYABLE
                    elif isinstance(e, retryable_exceptions):
                        is_retryable = True

                    if not is_retryable:
                        logger.error(
                            "Fatal error in %s: %s (category=%s)",
                            func.__name__, e,
                            e.category if isinstance(e, AgentError) else "unknown",
                        )
                        raise

                    if attempt < max_retries:
                        delay = base_delay * (backoff_factor ** attempt)
                        if jitter:
                            delay *= 0.75 + random.random() * 0.5  # 0.75x to 1.25x
                        logger.warning(
                            "%s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                            func.__name__, attempt + 1, max_retries, e, delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "%s exhausted %d retries: %s",
                            func.__name__, max_retries, e,
                        )

            # All retries exhausted
            raise last_error  # type: ignore[misc]

        return wrapper
    return decorator


# ============================================================================
# Graceful Degradation
# ============================================================================


def with_fallback(fallback_func: Callable):
    """Decorator: if primary function raises, call fallback instead.

    The fallback receives the same *args, **kwargs.
    Returns a (result, degraded: bool) tuple.
    If the fallback also raises, the original exception is re-raised.

    Example:
        @with_fallback(fallback=keyword_search)
        def vector_search(query):
            ...
    """
    def decorator(primary_func: Callable) -> Callable:
        @functools.wraps(primary_func)
        def wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, bool]:
            logger = logging.getLogger("cmcc_agent.resilience")
            try:
                result = primary_func(*args, **kwargs)
                return result, False
            except Exception as e:
                logger.warning(
                    "%s failed, degrading to %s: %s",
                    primary_func.__name__, fallback_func.__name__, e,
                )
                try:
                    fallback_result = fallback_func(*args, **kwargs)
                    return fallback_result, True
                except Exception as fb_e:
                    logger.error(
                        "Fallback %s also failed: %s",
                        fallback_func.__name__, fb_e,
                    )
                    raise RetrievalDegradedError(
                        f"Both primary and fallback failed: {e} | {fb_e}",
                        original=e,
                    ) from fb_e
        return wrapper
    return decorator


# ============================================================================
# Structured Logging Setup
# ============================================================================


_loggers_initialized: set = set()


def setup_logging(
    name: str = "cmcc_agent",
    level: int = logging.DEBUG,
    log_dir: Optional[str] = None,
) -> logging.Logger:
    """Initialize structured logging for the agent system.

    Console handler: INFO+ (for production visibility)
    File handler: DEBUG+ (for forensic debugging, optional)

    Args:
        name: Root logger name.
        level: Log level for the root logger.
        log_dir: Optional directory path for log file output (from LOG_DIR env or explicit).

    Returns:
        The root logger instance.
    """
    root = logging.getLogger(name)
    if name in _loggers_initialized:
        return root

    root.setLevel(level)
    root.handlers.clear()

    fmt = logging.Formatter(
        "[%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler (optional)
    effective_log_dir = log_dir or os.environ.get("LOG_DIR", "")
    if effective_log_dir:
        os.makedirs(effective_log_dir, exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(effective_log_dir, "agent.log"),
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    _loggers_initialized.add(name)
    return root


def get_logger(name: str = "cmcc_agent") -> logging.Logger:
    """Get a named child logger.

    Ensures the root logger is initialized on first call.
    Safe to call from any module.

    Args:
        name: Logger name. If not "cmcc_agent", returns a child logger.

    Returns:
        A logging.Logger instance.
    """
    setup_logging()  # idempotent
    if name == "cmcc_agent":
        return logging.getLogger("cmcc_agent")
    return logging.getLogger(name)
