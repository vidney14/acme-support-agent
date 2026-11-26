"""Token-bucket rate limiting.

Why it matters here: the demo runs on a *free-tier* Gemini key that is shared by every
visitor. Without limits, one person hammering the chat would exhaust the daily quota
for everyone. Each client (IP) gets a bucket that refills at `rate_per_minute` and holds
up to `burst` tokens, so short bursts are fine but sustained spam gets a 429 with a
Retry-After header (which the UI turns into a friendly countdown).

In-memory = per-instance. With several instances, move the buckets to Redis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from starlette.requests import HTTPConnection as Request


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    def __init__(self, rate_per_minute: float, burst: int):
        self.rate = rate_per_minute / 60.0
        self.burst = burst
        self._buckets: dict[str, _Bucket] = {}

    def check(self, key: str, cost: float = 1.0) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds)."""
        now = time.monotonic()
        b = self._buckets.get(key)
        if b is None:
            b = self._buckets[key] = _Bucket(tokens=float(self.burst), updated=now)
        b.tokens = min(self.burst, b.tokens + (now - b.updated) * self.rate)
        b.updated = now
        if b.tokens >= cost:
            b.tokens -= cost
            return True, 0.0
        if len(self._buckets) > 10_000:  # bound memory under abuse
            self._buckets = {k: v for k, v in self._buckets.items() if now - v.updated < 300}
        return False, round((cost - b.tokens) / self.rate, 1)


def client_key(request: Request) -> str:
    # Behind Render/Railway/Vercel the real client is the first X-Forwarded-For hop.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
