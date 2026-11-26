"""Post-answer grounding check (the second line of defence after the retrieval gate).

The retrieval gate prevents answering when the KB has nothing relevant. This check
verifies the answer that was actually produced:

- every cited tag must be one that retrieval returned *in this turn* (a citation to a
  passage the model never saw is a hallucinated source);
- an answer that states policy-like specifics (amounts, durations, percentages, plan
  rules) with no citation and no account/tool evidence is flagged as unsupported.

It runs after streaming (we don't hold tokens back from the user), so the result is
shown as a badge on the message: "Grounded · 2 sources", or a warning that offers a human.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.agent.turn import TurnState

CITATION = re.compile(r"\[([a-z0-9\-]+#\d+)\]")
POLICY_CLAIM = re.compile(
    r"(\$\s?\d|\d+\s?(%|percent|days?|hours?|business days|GB|TB)\b|\b(refundable|eligible|within \d|"
    r"service credit|uptime|grace period|automated approval limit)\b)", re.IGNORECASE)


@dataclass
class Grounding:
    status: str                      # grounded | account_data | escalated | no_claims | unsupported | invalid_citation
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict:
        return {"status": self.status, "citations": self.citations, "invalid_citations": self.invalid_citations,
                "detail": self.detail}


def check_grounding(answer: str, turn: TurnState) -> Grounding:
    cited = list(dict.fromkeys(CITATION.findall(answer)))
    valid = [c for c in cited if c in turn.retrieved_ids]
    invalid = [c for c in cited if c not in turn.retrieved_ids]
    used_tools = turn.used_account_data or bool(turn.refund_outcomes) or bool(turn.tickets_created)

    if invalid:
        return Grounding("invalid_citation", valid, invalid,
                         f"Cited {len(invalid)} source(s) that were not retrieved this turn.")
    if valid:
        return Grounding("grounded", valid, [], f"Backed by {len(valid)} knowledge base passage(s).")
    if turn.kb_escalations and turn.tickets_created:
        return Grounding("escalated", [], [], "Knowledge base didn't cover it; handed to a human instead of guessing.")
    if used_tools:
        return Grounding("account_data", [], [], "Based on account data and tool results.")
    if POLICY_CLAIM.search(answer) and not turn.kb_escalations:
        return Grounding("unsupported", [], [], "States policy specifics without a knowledge base source.")
    return Grounding("no_claims", [], [], "No factual policy claims.")
