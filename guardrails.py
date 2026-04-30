"""Production guardrails: input validation, content safety, rate limiting.

Designed for integration into the agent's preprocess_node.
All components are zero-dependency (Python stdlib only).
"""

from __future__ import annotations

import re
import time
from threading import Lock
from typing import Dict, Optional, Tuple


# ============================================================================
# Input Validation
# ============================================================================

PHONE_PATTERN = re.compile(r"^1[3-9]\d{9}$")
TICKET_ID_PATTERN = re.compile(r"^TK-\d{6,}$", re.IGNORECASE)


def validate_phone(phone: str) -> Tuple[bool, str]:
    """Validate Chinese mobile phone number format: 1[3-9]XXXXXXXXX.

    Args:
        phone: Raw phone number string.

    Returns:
        (is_valid, message) tuple.
    """
    if not phone or not isinstance(phone, str):
        return False, "请输入有效的手机号码"
    phone = phone.strip()
    if not PHONE_PATTERN.match(phone):
        return False, "手机号格式不正确，请输入11位大陆手机号（如13812345678）"
    return True, phone


def validate_ticket_id(ticket_id: str) -> Tuple[bool, str]:
    """Validate ticket ID format: TK-XXXXXX (at least 6 digits).

    Args:
        ticket_id: Raw ticket ID string.

    Returns:
        (is_valid, message_or_validated_id) tuple.
    """
    if not ticket_id or not isinstance(ticket_id, str):
        return False, "请提供有效的工单号（如 TK-123456）"
    ticket_id = ticket_id.strip()
    if not TICKET_ID_PATTERN.match(ticket_id):
        return False, "工单号格式不正确，应为 TK- 开头后跟6位以上数字（如 TK-123456）"
    return True, ticket_id.upper()


def validate_query_length(query: str, max_length: int = 2000) -> Tuple[bool, str]:
    """Reject excessively long queries (potential abuse or accidental paste).

    Args:
        query: User query text.
        max_length: Maximum allowed characters.

    Returns:
        (is_valid, message) tuple.
    """
    if not query or not isinstance(query, str):
        return False, "请输入有效的问题"
    if len(query) > max_length:
        return False, f"输入内容过长，请控制在{max_length}字以内（当前{len(query)}字）"
    return True, query.strip()


def sanitize_input(text: str) -> str:
    """Strip control characters, normalize whitespace.

    Args:
        text: Raw user input.

    Returns:
        Cleaned text.
    """
    if not text:
        return ""
    # Remove control characters except common whitespace
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalize whitespace
    cleaned = re.sub(r"[\t ]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ============================================================================
# Content Safety
# ============================================================================


# Patterns matching obviously harmful content.
# This is a lightweight keyword check — not a full content moderation system.
_HARMFUL_PATTERNS: list = [
    # Security / hacking
    re.compile(
        r"(?:hack|exploit|inject|sql\s*injection|xss|csrf|crack|破解|攻击|漏洞|注入|0day)",
        re.IGNORECASE,
    ),
    # Self-harm / violence
    re.compile(
        r"(?:自杀|自残|杀人|爆炸|恐怖|毒品|赌博|诈骗|裸聊|招嫖|色情)",
        re.IGNORECASE,
    ),
    # Prompt injection
    re.compile(
        r"(?:prompt\s*injection|ignore\s*(?:previous|all)\s*instructions|"
        r"system\s*prompt|忘记.*指令|忽略.*规则|你现在.*角色)",
        re.IGNORECASE,
    ),
    # PII fishing
    re.compile(
        r"(?:身份证号|银行卡号|密码|cvv|验证码|短信验证码)",
        re.IGNORECASE,
    ),
]


def content_safety_check(text: str) -> Tuple[bool, Optional[str]]:
    """Check text for harmful content patterns.

    Args:
        text: User input text to check.

    Returns:
        (is_safe, reason_if_unsafe) tuple. If safe, reason is None.
    """
    if not text:
        return True, None

    for pattern in _HARMFUL_PATTERNS:
        match = pattern.search(text)
        if match:
            return False, "输入内容包含不适当关键词，无法处理。如需帮助请联系10086人工客服。"

    return True, None


# ============================================================================
# Rate Limiter (Token Bucket)
# ============================================================================


class TokenBucketRateLimiter:
    """Simple token bucket rate limiter for per-session throttling.

    Default: 10 requests per second, burst capacity of 20.
    Thread-safe via internal Lock.

    Example:
        limiter = TokenBucketRateLimiter(rate=10.0, burst=20)
        if limiter.acquire():
            process_request()
        else:
            return "rate limited"
    """

    def __init__(self, rate: float = 10.0, burst: int = 20):
        self.rate = float(rate)
        self.burst = float(burst)
        self.tokens = float(burst)
        self.last_refill = time.monotonic()
        self._lock = Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_refill = now

    def acquire(self, tokens: float = 1.0) -> bool:
        """Try to acquire tokens. Returns True if allowed, False if rate limited.

        Args:
            tokens: Number of tokens to consume (default 1 per request).

        Returns:
            True if request is allowed, False if rate limited.
        """
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def reset(self) -> None:
        """Reset to full burst capacity."""
        with self._lock:
            self.tokens = self.burst
            self.last_refill = time.monotonic()


# Global per-session rate limiters
_session_limiters: Dict[str, TokenBucketRateLimiter] = {}


def check_rate_limit(session_id: str = "default") -> bool:
    """Check if the session is within rate limits.

    Creates a rate limiter for new sessions automatically.
    Default: 10 requests/second, burst of 20 per session.

    Args:
        session_id: Unique session identifier.

    Returns:
        True if request is allowed, False if rate limited.
    """
    if session_id not in _session_limiters:
        _session_limiters[session_id] = TokenBucketRateLimiter(rate=10.0, burst=20)
    return _session_limiters[session_id].acquire()


def reset_rate_limit(session_id: str = "default") -> None:
    """Reset rate limit for a session."""
    if session_id in _session_limiters:
        _session_limiters[session_id].reset()
