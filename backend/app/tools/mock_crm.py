"""Mock CRM + billing system.

Loaded once from data/customers.json; each session gets a deep-copied fork so a refund
issued in one visitor's session never leaks into another's (and demos are repeatable).
Relative dates (days_ago) are resolved against "today" at load time.
"""

from __future__ import annotations

import copy
import json
import random
from datetime import date, timedelta
from pathlib import Path


class MockCRM:
    def __init__(self, customers: dict[str, dict], today: date | None = None):
        self._by_id = customers
        self._by_email = {c["email"].lower(): cid for cid, c in customers.items()}
        self.today = today or date.today()

    @classmethod
    def load(cls, path: Path, today: date | None = None) -> "MockCRM":
        today = today or date.today()
        raw = json.loads(path.read_text(encoding="utf-8"))
        customers: dict[str, dict] = {}
        for c in raw["customers"]:
            c = copy.deepcopy(c)
            for inv in c.get("invoices", []):
                inv["date"] = (today - timedelta(days=inv.pop("days_ago"))).isoformat()
                inv.setdefault("currency", "USD")
                inv.setdefault("refunded_amount", 0.0)
            for rf in c.get("refunds", []):
                rf["date"] = (today - timedelta(days=rf.pop("days_ago"))).isoformat()
            if "contract" in c and "renewal_days_from_now" in c["contract"]:
                c["contract"]["renewal_date"] = (today + timedelta(days=c["contract"].pop("renewal_days_from_now"))).isoformat()
            if "expired_days_ago" in c:
                c["expired_on"] = (today - timedelta(days=c.pop("expired_days_ago"))).isoformat()
            customers[c["customer_id"]] = c
        return cls(customers, today)

    def fork(self) -> "MockCRM":
        return MockCRM(copy.deepcopy(self._by_id), self.today)

    # --- reads -------------------------------------------------------------------
    def get(self, customer_id: str) -> dict | None:
        return self._by_id.get(customer_id)

    def find_by_email(self, email: str) -> dict | None:
        cid = self._by_email.get(email.strip().lower())
        return self._by_id.get(cid) if cid else None

    def invoice(self, customer_id: str, invoice_id: str) -> dict | None:
        c = self.get(customer_id)
        if not c:
            return None
        return next((i for i in c["invoices"] if i["invoice_id"].upper() == invoice_id.strip().upper()), None)

    def days_since(self, iso: str) -> int:
        return (self.today - date.fromisoformat(iso)).days

    def demo_accounts(self) -> list[dict]:
        """Public cheat-sheet for the demo UI (mock data only)."""
        return [{"name": c["profile"]["display_name"], "email": c["email"], "pin": c["support_pin"], "tier": c["tier"],
                 "contract_status": c["contract_status"], "scenario": c.get("demo_scenario", "")}
                for c in self._by_id.values()]

    # --- writes ------------------------------------------------------------------
    def record_refund(self, customer_id: str, invoice_id: str, amount: float, reason: str) -> dict:
        c = self._by_id[customer_id]
        inv = self.invoice(customer_id, invoice_id)
        refund = {"refund_id": f"RF-{random.randint(10000, 99999)}", "invoice_id": inv["invoice_id"],
                  "amount": round(amount, 2), "date": self.today.isoformat(), "approved_by": "automated",
                  "reason": reason[:200]}
        c["refunds"].append(refund)
        inv["refunded_amount"] = round(inv.get("refunded_amount", 0.0) + amount, 2)
        inv["status"] = "refunded" if inv["refunded_amount"] >= inv["amount"] else "partially_refunded"
        return refund

    def update_profile(self, customer_id: str, field: str, value) -> tuple[object, object]:
        prof = self._by_id[customer_id]["profile"]
        old = prof.get(field)
        prof[field] = value
        return old, value
