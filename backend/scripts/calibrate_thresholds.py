"""Calibrate the confidence-gate thresholds against real Gemini embeddings.

    cd backend && python -m scripts.calibrate_thresholds

Runs labelled questions through retrieval (no LLM calls, embeddings only) and prints the
best dense similarity for each. In-scope questions should score above HIGH; out-of-scope
ones below LOW. The grey zone in between is where the verifier earns its keep.
Paste the suggested values into backend/.env (CONFIDENCE_HIGH / CONFIDENCE_LOW).
"""

from __future__ import annotations

import asyncio
import statistics

from app.config import get_settings
from app.container import build_container

IN_SCOPE = [
    ("How long do I have to request a refund on Pro monthly?", "pro"),
    ("I was charged twice this month", "pro"),
    ("What does error E-1042 mean?", "free"),
    ("How do I fix a sync that says authentication expired?", "pro"),
    ("What happens to my files when my subscription expires?", "business"),
    ("What service credits do we get if uptime drops below 99%?", "enterprise"),
    ("Can you change my phone number?", "pro"),
    ("I lost my phone and can't log in with 2FA", "pro"),
    ("How do I export all my data?", "free"),
    ("How much storage does the Business plan include?", "business"),
    ("My card payment failed, what happens now?", "pro"),
    ("How do I cancel auto renewal on my annual plan?", "pro"),
]
OUT_OF_SCOPE = [
    ("Do you offer a student discount?", "pro"),
    ("Can I pay with Bitcoin?", "pro"),
    ("Is there an on-premise version of AcmeCloud?", "business"),
    ("What's the weather in Boston tomorrow?", None),
    ("Write me a poem about clouds", None),
    ("Do you have a referral program?", "pro"),
    ("Can I get a refund in gift cards?", "pro"),  # KB says no: answerable, but adversarial
    ("Who is the CEO of AcmeCloud?", None),
]


async def main() -> None:
    s = get_settings()
    c = build_container(s)
    status = await c.kb.ingest()
    assert status.ready, status.error
    c.kb.gate.provider = None  # measure similarity only; don't call the verifier

    async def score(q, tier):
        res, _ = await c.kb.retriever.retrieve(q, tier, top_k=s.retrieval_top_k), None
        return res.top_dense, res.chunks[0].breadcrumb if res.chunks else "-"

    print(f"Embedding model: {s.embedding_model} (dim {s.embedding_dim})\n")
    ins, outs = [], []
    for label, rows, bucket in (("IN SCOPE", IN_SCOPE, ins), ("OUT OF SCOPE", OUT_OF_SCOPE, outs)):
        print(label)
        for q, tier in rows:
            sc, top = await score(q, tier)
            bucket.append(sc)
            print(f"  {sc:.3f}  {q[:60]:60}  -> {top[:60]}")
            await asyncio.sleep(0.3)  # be gentle with free-tier embedding quota
        print()

    print(f"in-scope:     min {min(ins):.3f}  median {statistics.median(ins):.3f}")
    print(f"out-of-scope: max {max(outs):.3f}  median {statistics.median(outs):.3f}")
    low = round(max(statistics.median(outs) + 0.02, min(ins) - 0.05), 2)
    high = round(max(low + 0.05, statistics.median(ins) - 0.02), 2)
    print(f"\nSuggested: CONFIDENCE_LOW={low}  CONFIDENCE_HIGH={high}")
    print(f"Current:   CONFIDENCE_LOW={s.confidence_low}  CONFIDENCE_HIGH={s.confidence_high}")
    print("Anything scoring between LOW and HIGH goes to the verifier model, so overlap there is fine.")


if __name__ == "__main__":
    asyncio.run(main())
