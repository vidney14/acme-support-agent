"""Error classification + retry with exponential backoff and jitter.

The key production idea: *not every error deserves a retry.*
- 429 / 5xx / timeouts are transient, so retry with backoff.
- 400 / 401 / 403 / 404 are our bug or config, so retrying just burns quota.
- A 429 whose server-suggested retry delay is longer than we are willing to make a
  user wait is treated as "give up on this model now" so the caller can fall back to
  a different model (with its own quota) instead of stalling the conversation.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, TypeVar

log = logging.getLogger(__name__)
T = TypeVar("T")


class ErrorKind(str, Enum):
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    SERVER = "server_error"
    CLIENT = "client_error"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class ProviderError(Exception):
    def __init__(self, kind: ErrorKind, message: str, retryable: bool, retry_after_s: float | None = None,
                 status_code: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.retry_after_s = retry_after_s
        self.status_code = status_code

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "message": str(self), "retryable": self.retryable,
                "retry_after_s": self.retry_after_s, "status_code": self.status_code}


_RETRY_DELAY = re.compile(r"retry(?:Delay|_delay)?['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", re.IGNORECASE)


def classify(exc: BaseException) -> ProviderError:
    """Map any exception (SDK, network, asyncio) onto a ProviderError."""
    if isinstance(exc, ProviderError):
        return exc
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return ProviderError(ErrorKind.TIMEOUT, "Upstream request timed out", retryable=True)

    code = getattr(exc, "code", None)
    if isinstance(code, int):
        msg = str(exc)
        if code == 429:
            m = _RETRY_DELAY.search(msg)
            return ProviderError(ErrorKind.RATE_LIMIT, "Upstream rate limit / quota exhausted", True,
                                 retry_after_s=float(m.group(1)) if m else None, status_code=429)
        if code in (500, 502, 503, 504):
            return ProviderError(ErrorKind.SERVER, f"Upstream server error ({code})", True, status_code=code)
        if code == 408:
            return ProviderError(ErrorKind.TIMEOUT, "Upstream request timed out", True, status_code=code)
        return ProviderError(ErrorKind.CLIENT, f"Upstream rejected the request ({code}): {msg[:300]}",
                             retryable=False, status_code=code)

    name = type(exc).__name__.lower()
    if any(k in name for k in ("connect", "network", "remoteprotocol", "readerror")):
        return ProviderError(ErrorKind.UNAVAILABLE, f"Network error: {exc}", retryable=True)
    return ProviderError(ErrorKind.UNKNOWN, f"{type(exc).__name__}: {exc}", retryable=False)


@dataclass
class RetryPolicy:
    max_retries: int = 2
    base_delay_s: float = 0.6
    max_delay_s: float = 6.0          # never make the user wait longer than this per retry

    def delay_for(self, attempt: int, err: ProviderError) -> float | None:
        """Seconds to wait before retry `attempt` (1-based), or None to stop retrying."""
        if not err.retryable or attempt > self.max_retries:
            return None
        if err.retry_after_s is not None:
            if err.retry_after_s > self.max_delay_s:
                return None  # quota window too long: let the caller fall back instead
            return err.retry_after_s
        exp = self.base_delay_s * (2 ** (attempt - 1))
        return min(self.max_delay_s, exp) * (0.5 + random.random())  # full jitter


OnRetry = Callable[[int, ProviderError, float], Awaitable[None]]


async def with_retries(fn: Callable[[], Awaitable[T]], policy: RetryPolicy, label: str = "call",
                       on_retry: OnRetry | None = None) -> T:
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - we classify everything
            err = classify(exc)
            attempt += 1
            delay = policy.delay_for(attempt, err)
            if delay is None:
                raise err from exc
            log.warning("%s failed (%s), retry %d in %.2fs", label, err.kind.value, attempt, delay)
            if on_retry:
                await on_retry(attempt, err, delay)
            await asyncio.sleep(delay)
