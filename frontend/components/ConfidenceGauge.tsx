"use client";

import type { Confidence } from "@/lib/types";

/**
 * Shows where the retrieval score landed relative to the two guardrail thresholds:
 *   below `low` -> hand off to a human, between -> verifier decides, above `high` -> answer.
 */
export function ConfidenceGauge({ c, low, high }: { c: Confidence; low: number; high: number }) {
  const min = Math.max(0, low - 0.25);
  const max = Math.min(1, high + 0.25);
  const pos = (v: number) => `${Math.min(100, Math.max(0, ((v - min) / (max - min)) * 100))}%`;
  const exact = c.reasons.some((r) => r.startsWith("exact identifier"));

  const colour =
    c.decision === "answer" ? "bg-ok" : c.decision === "answer_with_caution" ? "bg-work" : "bg-handoff";

  let sentence: string;
  if (exact) sentence = `Exact match on ${c.reasons[0].match(/\(([^)]+)\)/)?.[1] ?? "an identifier"}, answered from the knowledge base.`;
  else if (c.decision === "escalate")
    sentence =
      c.score < low
        ? `Confidence ${c.score.toFixed(2)} is below ${low}, so the agent won't guess.`
        : `Confidence ${c.score.toFixed(2)} is in the grey zone${c.verifier ? ` and the verifier said "${c.verifier.answerable}"` : ""}, so the agent won't guess.`;
  else if (c.band === "medium")
    sentence = c.verifier
      ? `Confidence ${c.score.toFixed(2)} is in the grey zone; the verifier said "${c.verifier.answerable}".`
      : `Confidence ${c.score.toFixed(2)} is in the grey zone; answering with caution.`;
  else sentence = `Confidence ${c.score.toFixed(2)}, above the ${high} answer threshold.`;

  return (
    <div className="mb-3 max-w-xl">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
        <div className="relative h-5 w-48 shrink-0" role="img" aria-label={sentence}>
          <div className="absolute inset-x-0 top-[7px] flex h-1.5 overflow-hidden rounded-full">
            <span className="bg-fault/25" style={{ width: pos(low) }} />
            <span className="bg-work/25" style={{ width: `calc(${pos(high)} - ${pos(low)})` }} />
            <span className="flex-1 bg-ok/25" />
          </div>
          <span className="absolute top-[3px] h-3.5 w-px bg-slate/60" style={{ left: pos(low) }} />
          <span className="absolute top-[3px] h-3.5 w-px bg-slate/60" style={{ left: pos(high) }} />
          <span
            className={`absolute top-[5px] h-2.5 w-2.5 -translate-x-1/2 rounded-full ring-2 ring-paper ${colour}`}
            style={{ left: pos(c.score) }}
          />
        </div>
        <p className="text-caption text-slate">{sentence}</p>
      </div>
      <details className="group mt-1">
        <summary className="cursor-pointer list-none text-caption text-mist hover:text-slate">
          <span className="group-open:hidden">Show retrieved passages ({c.results.length})</span>
          <span className="hidden group-open:inline">Hide retrieved passages</span>
          <span className="text-mist">
            {" "}
            for &ldquo;{c.query}&rdquo;, filtered to {c.tier_filter}
          </span>
        </summary>
        <ul className="mt-2 space-y-2 border-l border-rule pl-3">
          {c.results.map((r) => (
            <li key={r.chunk_id} className="text-caption">
              <div className="tnum flex flex-wrap gap-x-3 text-slate">
                <span className="text-ink">{r.breadcrumb}</span>
                <span>dense {r.dense_score?.toFixed(3) ?? "n/a"}</span>
                <span>bm25 {r.bm25_score.toFixed(2)}</span>
                {r.exact_match && <span className="text-ok">exact match</span>}
              </div>
              <p className="mt-0.5 line-clamp-2 text-mist">{r.text}</p>
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}
