"""Per-session fault injection, so resilience can be *demonstrated*, not just claimed.

The demo panel arms a fault ("the next 2 primary-model calls return 429"), and the real
code paths (retry, fallback, tool retry, graceful degradation) handle it exactly as they
would a genuine upstream failure. Faults are counters that are consumed as they fire, so
the system recovers on its own afterwards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


class InjectedRateLimit(Exception):
    code = 429  # looks like an SDK APIError to resilience.classify()

    def __str__(self) -> str:
        return "Injected fault: 429 RESOURCE_EXHAUSTED (retryDelay: 30s)"


class InjectedTimeout(TimeoutError):
    def __str__(self) -> str:
        return "Injected fault: upstream timeout"


class InjectedToolFailure(Exception):
    def __str__(self) -> str:
        return "Injected fault: CRM returned 503"


@dataclass
class FaultConfig:
    llm_rate_limit: int = 0     # next N primary-model calls raise 429 (long retry delay -> fallback)
    llm_timeout: int = 0        # next N model calls time out (retried with backoff)
    tool_flaky: int = 0         # next N CRM tool attempts fail once (retry succeeds)
    crm_outage: bool = False    # every CRM tool attempt fails until switched off

    def take(self, name: str) -> bool:
        n = getattr(self, name)
        if isinstance(n, bool):
            return n
        if n > 0:
            setattr(self, name, n - 1)
            return True
        return False

    def to_dict(self) -> dict:
        return asdict(self)

    def update(self, data: dict) -> None:
        for k, v in data.items():
            if k in self.__dataclass_fields__:
                setattr(self, k, bool(v) if k == "crm_outage" else max(0, int(v)))
