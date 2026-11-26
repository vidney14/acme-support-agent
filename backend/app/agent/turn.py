"""Per-turn working state + the deterministic end-of-turn guards.

Tools write facts here as they run (which KB searches returned "escalate", which refund
needed a human, which tickets were opened). At the end of every turn, in chat AND in
voice, `enforce_turn_guards` checks the invariants the LLM was *asked* to follow and
fixes anything it didn't:

  * KB said "escalate" but no ticket was opened        -> open one automatically
  * a refund needs human approval but no ticket exists -> open a P2 billing ticket
  * identity got locked but nobody escalated           -> open an account ticket

So the model gets the first chance to handle things gracefully in its own words, and
code guarantees the outcome either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.tools.registry import ToolContext


@dataclass
class TurnState:
    turn_id: str
    modality: str
    user_text: str = ""
    kb_searches: list[dict] = field(default_factory=list)   # {query, decision, score, citations}
    retrieved_ids: set[str] = field(default_factory=set)
    used_account_data: bool = False
    refund_outcomes: list[dict] = field(default_factory=list)
    tickets_created: list[str] = field(default_factory=list)
    identity_locked_now: bool = False
    guard_actions: list[str] = field(default_factory=list)

    @property
    def kb_escalations(self) -> list[dict]:
        return [s for s in self.kb_searches if s["decision"] == "escalate"]


async def enforce_turn_guards(ctx: "ToolContext") -> list[dict]:
    """Returns tickets created by the guard (empty if the model already complied)."""
    from app.tools.support_tools import open_ticket  # local import: avoid cycle

    turn, session = ctx.turn, ctx.session
    created: list[dict] = []
    has_ticket = bool(turn.tickets_created) or bool(session.tickets)

    if turn.kb_escalations and not has_ticket:
        q = turn.kb_escalations[-1]["query"]
        created.append(await open_ticket(ctx, summary=f"Question not covered by the knowledge base: \"{q}\"",
                                         category="other", priority="P3",
                                         attempted_steps="Searched the knowledge base; confidence below threshold.",
                                         source="guardrail"))
        has_ticket = True

    if any(r["outcome"] == "needs_human" for r in turn.refund_outcomes) and not any(
            t["category"] == "billing" for t in session.tickets):
        r = next(r for r in turn.refund_outcomes if r["outcome"] == "needs_human")
        created.append(await open_ticket(
            ctx, summary=f"Refund of ${r['amount']:.2f} on {r['invoice_id']} needs human approval "
                         f"({', '.join(h['code'] for h in r['rules'])}).",
            category="billing", priority="P2", attempted_steps="Policy engine evaluated the refund.",
            source="guardrail"))

    if turn.identity_locked_now and not any(t["category"] == "account" for t in session.tickets):
        created.append(await open_ticket(ctx, summary="Identity verification failed repeatedly; manual verification needed.",
                                         category="account", priority="P2",
                                         attempted_steps="Email + PIN verification attempted.", source="guardrail"))

    for t in created:
        turn.guard_actions.append(f"auto-escalated: {t['ticket_id']}")
    return created
