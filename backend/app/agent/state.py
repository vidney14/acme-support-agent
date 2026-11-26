"""Session state: the single source of truth shared by chat and voice.

Memory has two layers:
- **Transcript** (`messages`): every user/assistant utterance from either modality,
  tagged with where it came from. Both the chat agent and the voice agent are seeded
  from it, which is what makes "start in chat, continue by voice" work.
- **Structured facts** (`customer_id`, `tickets`, `actions`): things the agent has
  *established* through tools. These are injected into the system prompt as a compact
  "what we know so far" block, so the model doesn't have to re-derive them from a long
  transcript, and so they survive a model switch or a modality switch.

Each session also gets its own forked copy of the mock CRM, so demo actions (refunds,
profile edits) are isolated per visitor and every new session starts clean.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from app.faults import FaultConfig
from app.observability.tracer import Trace
from app.tools.mock_crm import MockCRM


@dataclass
class Message:
    role: str                       # user | assistant
    text: str
    modality: str = "text"          # text | voice
    turn_id: str = ""
    ts: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)   # citations, grounding, interrupted...

    def to_dict(self) -> dict:
        return {"role": self.role, "text": self.text, "modality": self.modality, "turn_id": self.turn_id,
                "ts": self.ts, "meta": self.meta}


@dataclass
class Session:
    session_id: str
    crm: MockCRM
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    messages: list[Message] = field(default_factory=list)
    customer_id: str | None = None
    verify_failures: int = 0
    identity_locked: bool = False
    status: str = "active"          # active | escalated
    tickets: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)     # audit log of side effects
    traces: deque = field(default_factory=lambda: deque(maxlen=60))
    faults: FaultConfig = field(default_factory=FaultConfig)
    voice_connected: bool = False
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def touch(self) -> None:
        self.last_active = time.time()

    @property
    def customer(self) -> dict | None:
        return self.crm.get(self.customer_id) if self.customer_id else None

    def add_message(self, role: str, text: str, modality: str, turn_id: str, **meta: Any) -> Message:
        m = Message(role=role, text=text, modality=modality, turn_id=turn_id, meta=meta)
        self.messages.append(m)
        self.touch()
        return m

    def add_trace(self, trace: Trace) -> None:
        self.traces.append(trace)

    def facts_block(self) -> str:
        """Compact, structured working memory injected into every system prompt."""
        lines = []
        c = self.customer
        if c:
            lines.append(f"- Identity VERIFIED: {c['profile']['display_name']} ({c['customer_id']}), "
                         f"plan={c['tier']}, billing={c.get('billing_cycle')}, contract_status={c['contract_status']}")
        elif self.identity_locked:
            lines.append("- Identity verification LOCKED after too many failed attempts. Do not retry; escalate.")
        else:
            lines.append(f"- Identity NOT verified yet (failed attempts: {self.verify_failures}).")
        for t in self.tickets:
            lines.append(f"- Escalation ticket {t['ticket_id']} open: {t['category']}/{t['priority']}, "
                         f"human response target {t['response_target']}. Summary: {t['summary']}")
        for a in self.actions[-6:]:
            lines.append(f"- Action taken: {a['summary']}")
        if self.status == "escalated":
            lines.append("- Session is ESCALATED to a human. Keep helping, but remind the customer a human will follow up.")
        return "\n".join(lines)

    def snapshot(self) -> dict:
        c = self.customer
        return {
            "session_id": self.session_id,
            "status": self.status,
            "verified": bool(c),
            "customer": None if not c else {
                "customer_id": c["customer_id"], "name": c["profile"]["display_name"], "tier": c["tier"],
                "contract_status": c["contract_status"]},
            "identity_locked": self.identity_locked,
            "tickets": self.tickets,
            "actions": self.actions,
            "faults": self.faults.to_dict(),
            "voice_connected": self.voice_connected,
            "message_count": len(self.messages),
        }


class SessionStore:
    """In-memory session store with TTL eviction.

    Trade-off: single-instance only (sessions live in process memory). For horizontal
    scaling, back this interface with Redis; nothing else in the app would change.
    """

    def __init__(self, crm_template: MockCRM, ttl_minutes: int = 120, max_sessions: int = 2000):
        self._sessions: dict[str, Session] = {}
        self._crm_template = crm_template
        self._ttl = ttl_minutes * 60
        self._max = max_sessions

    def get(self, session_id: str) -> Session | None:
        s = self._sessions.get(session_id)
        if s and time.time() - s.last_active > self._ttl:
            del self._sessions[session_id]
            return None
        return s

    def get_or_create(self, session_id: str | None) -> Session:
        if session_id:
            s = self.get(session_id)
            if s:
                s.touch()
                return s
        self._evict()
        sid = session_id if (session_id and len(session_id) <= 64) else uuid.uuid4().hex
        s = Session(session_id=sid, crm=self._crm_template.fork())
        self._sessions[sid] = s
        return s

    def _evict(self) -> None:
        now = time.time()
        for sid in [k for k, v in self._sessions.items() if now - v.last_active > self._ttl]:
            del self._sessions[sid]
        if len(self._sessions) >= self._max:  # drop least recently active
            oldest = sorted(self._sessions.values(), key=lambda s: s.last_active)[: len(self._sessions) // 10 + 1]
            for s in oldest:
                self._sessions.pop(s.session_id, None)

    def __len__(self) -> int:
        return len(self._sessions)
