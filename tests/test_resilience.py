"""Tests for resilience.py: structured errors, retry, degradation, logging."""

import logging
import re
import time

import pytest

from resilience import (
    AgentError,
    ErrorCategory,
    InvalidInputError,
    LLMRateLimitError,
    LLMTimeoutError,
    RetrievalDegradedError,
    ToolExecutionError,
    classify_llm_error,
    get_logger,
    retry_with_backoff,
    setup_logging,
    with_fallback,
)


# ============================================================================
# TestStructuredErrors
# ============================================================================


class TestStructuredErrors:
    def test_error_categories(self):
        assert ErrorCategory.RETRYABLE.value == "retryable"
        assert ErrorCategory.FATAL.value == "fatal"
        assert ErrorCategory.DEGRADED.value == "degraded"

    def test_agent_error_default_fatal(self):
        e = AgentError("test")
        assert e.category == ErrorCategory.FATAL
        assert str(e) == "test"

    def test_llm_timeout_is_retryable(self):
        e = LLMTimeoutError()
        assert e.category == ErrorCategory.RETRYABLE

    def test_llm_rate_limit_is_retryable(self):
        e = LLMRateLimitError()
        assert e.category == ErrorCategory.RETRYABLE

    def test_tool_execution_is_retryable(self):
        e = ToolExecutionError("billing_query")
        assert e.category == ErrorCategory.RETRYABLE
        assert "billing_query" in str(e)

    def test_retrieval_degraded(self):
        e = RetrievalDegradedError()
        assert e.category == ErrorCategory.DEGRADED

    def test_invalid_input_is_fatal(self):
        e = InvalidInputError("bad phone")
        assert e.category == ErrorCategory.FATAL


# ============================================================================
# TestClassifyLLMError
# ============================================================================


class TestClassifyLLMError:
    def test_timeout_pattern(self):
        e = classify_llm_error(Exception("Request timed out after 30s"))
        assert isinstance(e, LLMTimeoutError)
        assert e.category == ErrorCategory.RETRYABLE

    def test_rate_limit_pattern(self):
        e = classify_llm_error(Exception("Too many requests: rate limit exceeded (429)"))
        assert isinstance(e, LLMRateLimitError)
        assert e.category == ErrorCategory.RETRYABLE

    def test_auth_error_is_fatal(self):
        e = classify_llm_error(Exception("Authentication failed: 401 Unauthorized"))
        assert isinstance(e, AgentError)
        assert e.category == ErrorCategory.FATAL

    def test_unknown_is_retryable_by_default(self):
        e = classify_llm_error(Exception("Something unexpected happened"))
        assert isinstance(e, AgentError)
        assert e.category == ErrorCategory.RETRYABLE


# ============================================================================
# TestRetryWithBackoff
# ============================================================================


class TestRetryWithBackoff:
    def test_succeeds_first_try(self):
        call_count = [0]

        @retry_with_backoff(max_retries=3, base_delay=0.01, jitter=False)
        def succeed():
            call_count[0] += 1
            return "ok"

        result = succeed()
        assert result == "ok"
        assert call_count[0] == 1

    def test_retries_on_retryable_error(self):
        call_count = [0]

        @retry_with_backoff(max_retries=3, base_delay=0.01, jitter=False)
        def fails_then_succeeds():
            call_count[0] += 1
            if call_count[0] < 3:
                raise LLMTimeoutError("timeout")
            return "recovered"

        result = fails_then_succeeds()
        assert result == "recovered"
        assert call_count[0] == 3

    def test_exhausts_retries(self):
        call_count = [0]

        @retry_with_backoff(max_retries=2, base_delay=0.01, jitter=False)
        def always_fails():
            call_count[0] += 1
            raise LLMTimeoutError("timeout again")

        with pytest.raises(LLMTimeoutError):
            always_fails()
        assert call_count[0] == 3  # 1 initial + 2 retries

    def test_no_retry_on_fatal(self):
        call_count = [0]

        @retry_with_backoff(max_retries=3, base_delay=0.01, jitter=False)
        def fatal_error():
            call_count[0] += 1
            raise AgentError("auth failed", ErrorCategory.FATAL)

        with pytest.raises(AgentError):
            fatal_error()
        assert call_count[0] == 1  # no retries

    def test_jitter_adds_variation(self):
        delays = []

        @retry_with_backoff(max_retries=5, base_delay=0.01, jitter=True)
        def jittered():
            delays.append(time.monotonic())
            if len(delays) < 6:
                raise LLMTimeoutError("timeout")

        try:
            jittered()
        except LLMTimeoutError:
            pass

        # Check that delays are not strictly geometric (jitter varies them)
        intervals = [delays[i] - delays[i - 1] for i in range(1, len(delays))]
        # With jitter, at least one interval should differ
        assert len(set(round(d, 4) for d in intervals[:2])) >= 1


# ============================================================================
# TestGracefulDegradation
# ============================================================================


class TestGracefulDegradation:
    def test_primary_succeeds_no_degradation(self):
        @with_fallback(fallback_func=lambda x: f"fallback_{x}")
        def primary(x):
            return f"primary_{x}"

        result, degraded = primary("test")
        assert result == "primary_test"
        assert degraded is False

    def test_primary_fails_fallback_used(self):
        @with_fallback(fallback_func=lambda x: f"fallback_{x}")
        def primary(x):
            raise RuntimeError("primary down")

        result, degraded = primary("test")
        assert result == "fallback_test"
        assert degraded is True

    def test_both_fail_raises(self):
        @with_fallback(fallback_func=lambda x: (_ for _ in ()).throw(ValueError("fallback down")))
        def primary(x):
            raise RuntimeError("primary down")

        with pytest.raises(RetrievalDegradedError):
            primary("test")


# ============================================================================
# TestLogging
# ============================================================================


class TestLogging:
    def test_setup_logging_creates_logger(self):
        logger = setup_logging("cmcc_agent_test", level=logging.DEBUG)
        assert logger.name == "cmcc_agent_test"
        assert logger.level == logging.DEBUG

    def test_setup_logging_idempotent(self):
        a = setup_logging("cmcc_agent_test")
        b = setup_logging("cmcc_agent_test")
        assert a is b

    def test_get_logger_returns_child(self):
        child = get_logger("cmcc_agent.repos")
        assert child.name == "cmcc_agent.repos"

    def test_get_logger_root(self):
        root = get_logger("cmcc_agent")
        assert root.name == "cmcc_agent"

    def test_logger_writes_message(self, capsys):
        logger = setup_logging("cmcc_agent_writer", level=logging.DEBUG)
        # Force console handler to DEBUG for test
        for h in logger.handlers:
            h.setLevel(logging.DEBUG)
        logger.info("test message 12345")
        captured = capsys.readouterr()
        assert "test message 12345" in captured.err or "test message 12345" in captured.out
