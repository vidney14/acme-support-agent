"""Deterministic policy engine for refunds.

The LLM *proposes* a refund; this code *decides*. Every rule returns a machine-readable
code, so the agent can explain the outcome precisely and the trace shows exactly which
rule fired. Two kinds of rule:

- DENY      -> policy says no (e.g. outside the refund window). The agent explains and
               may offer a human review.
- APPROVAL  -> allowed by policy but above the agent's authority (e.g. over the $50
               automated limit). The agent must open a P2 escalation ticket.

Outcome: any DENY -> "deny"; else any APPROVAL -> "needs_human"; else "approve".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RuleHit:
    code: str
    kind: str          # deny | approval
    message: str

    def to_dict(self) -> dict:
        return {"code": self.code, "kind": self.kind, "message": self.message}


@dataclass
class RefundDecision:
    outcome: str                                  # approve | needs_human | deny
    hits: list[RuleHit] = field(default_factory=list)
    automated_limit: float = 0.0
    window_days: int | None = None
    days_since_charge: int | None = None

    def to_dict(self) -> dict:
        return {"outcome": self.outcome, "rules": [h.to_dict() for h in self.hits],
                "automated_limit_usd": self.automated_limit, "refund_window_days": self.window_days,
                "days_since_charge": self.days_since_charge}


class PolicyRules:
    def __init__(self, raw: dict):
        self.raw = raw
        r = raw["refunds"]
        self.limits: dict[str, float] = r["automated_limit_usd"]
        self.windows: dict[str, dict[str, int]] = r["window_days"]
        self.frequency_days: int = r["automated_frequency_days"]
        self.dup_ignores_window: bool = r.get("duplicate_charge_ignores_window", True)
        self.response_targets: dict = raw["escalation"]["response_targets"]
        self.updatable_fields: dict = raw["account"]["updatable_fields"]
        self.max_failed_attempts: int = raw["identity"]["max_failed_attempts"]

    @classmethod
    def load(cls, path: Path) -> "PolicyRules":
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")))

    def evaluate_refund(self, customer: dict, invoice: dict, amount: float, days_since_charge: int,
                        recent_automated_refunds: int) -> RefundDecision:
        tier = customer["tier"]
        limit = float(self.limits.get(tier, 0))
        window = self.windows.get(tier, {}).get(invoice.get("cycle") or "monthly")
        d = RefundDecision(outcome="approve", automated_limit=limit, window_days=window,
                           days_since_charge=days_since_charge)
        deny = lambda c, m: d.hits.append(RuleHit(c, "deny", m))          # noqa: E731
        needs = lambda c, m: d.hits.append(RuleHit(c, "approval", m))     # noqa: E731

        remaining = round(invoice["amount"] - invoice.get("refunded_amount", 0.0), 2)
        if invoice["status"] == "refunded":
            deny("ALREADY_REFUNDED", "This invoice has already been refunded in full.")
        elif invoice["status"] not in ("paid", "partially_refunded"):
            deny("INVOICE_NOT_PAID", f"Invoice status is '{invoice['status']}', so there is no payment to refund.")
        if amount <= 0:
            deny("INVALID_AMOUNT", "Refund amount must be positive.")
        elif amount > remaining + 1e-9:
            deny("AMOUNT_EXCEEDS_INVOICE", f"Amount exceeds the refundable balance of ${remaining:.2f}.")
        if tier == "free":
            deny("FREE_PLAN", "Free plan accounts have no charges to refund.")
        if customer["contract_status"] == "past_due":
            deny("PAST_DUE_BALANCE",
                 f"Account is past due (${customer.get('outstanding_balance', 0):.2f} outstanding); "
                 "refunds are blocked until the balance is paid.")

        is_duplicate = bool(invoice.get("duplicate_of"))
        if window is not None and days_since_charge > window and not (is_duplicate and self.dup_ignores_window):
            deny("OUTSIDE_REFUND_WINDOW",
                 f"Charged {days_since_charge} days ago; the {tier} {invoice.get('cycle')} refund window is {window} days.")

        if tier == "enterprise":
            needs("ENTERPRISE_MSA", "Enterprise refunds are governed by the MSA and handled by the account manager.")
        elif amount > limit:
            needs("ABOVE_AUTOMATED_LIMIT",
                  f"${amount:.2f} is above the ${limit:.0f} automated approval limit for the {tier} plan.")
        if recent_automated_refunds > 0:
            needs("FREQUENCY_LIMIT",
                  f"An automated refund was already issued in the last {self.frequency_days} days.")

        if any(h.kind == "deny" for h in d.hits):
            d.outcome = "deny"
        elif d.hits:
            d.outcome = "needs_human"
        return d
