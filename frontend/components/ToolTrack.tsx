"use client";

import type { Step, Turn } from "@/lib/types";

/**
 * The agent's work for one turn, drawn as a rail of tool nodes that fill in as they run:
 * amber while working, green once settled, red if a step failed. A "Thinking" node trails
 * the rail while the model plans its next move. A vertical rail keeps long chains legible
 * on any width (a horizontal track wraps awkwardly past three steps).
 */
export function ToolTrack({ turn }: { turn: Turn }) {
  const streaming = turn.status === "streaming";
  const phase = turn.phase?.phase;
  const running = turn.steps.some((s) => s.status === "started" || s.status === "retrying");
  const thinking = streaming && !running && (phase === "plan" || phase === "verify" || phase === "guard");
  if (!turn.steps.length && !thinking) return null;

  const rows = turn.steps.length + (thinking ? 1 : 0);
  return (
    <ol className="mb-4 max-w-xl rounded-xl border border-rule-soft bg-surface/70 px-3 py-2.5 backdrop-blur" aria-label="Agent steps">
      {turn.steps.map((s, i) => (
        <Row key={s.step_id} step={s} last={i === rows - 1} />
      ))}
      {thinking && (
        <li className="relative pl-6">
          <span className="node-working absolute left-0 top-[7px] h-2.5 w-2.5 rounded-full bg-work" />
          <span className="shimmer-text text-small">{turn.phase?.label ?? "Thinking"}</span>
        </li>
      )}
    </ol>
  );
}

const ERROR_LABELS: Record<string, string> = {
  UPSTREAM_UNAVAILABLE: "account system unavailable",
  IDENTITY_REQUIRED: "needs identity verification first",
  IDENTITY_LOCKED: "verification locked",
  INVALID_ARGUMENTS: "invalid arguments, model will correct",
  INVOICE_NOT_FOUND: "invoice not found",
  FIELD_NOT_UPDATABLE: "field can't be changed here",
  INVALID_VALUE: "invalid value",
  KB_UNAVAILABLE: "knowledge base unavailable",
};

function Row({ step, last }: { step: Step; last: boolean }) {
  const working = step.status === "started" || step.status === "retrying";
  const dot =
    step.status === "completed" ? "bg-ok" : step.status === "failed" ? "bg-fault" : "bg-work node-working";
  const code = typeof step.error === "string" ? step.error : step.error?.code;
  const err = code ? (ERROR_LABELS[code] ?? code.toLowerCase().replaceAll("_", " ")) : undefined;
  const detail =
    step.status === "retrying"
      ? `retrying, attempt ${step.attempt ?? 2}`
      : step.status === "failed"
        ? err ?? "failed"
        : step.summary;
  const detailTone = step.status === "failed" ? "text-fault" : step.status === "retrying" ? "text-work" : "text-mist";

  return (
    <li className={`relative pl-6 ${last ? "" : "pb-1.5"}`}>
      {!last && (
        <span aria-hidden className={`absolute left-[4.5px] top-[14px] bottom-[-6px] w-px ${working ? "bg-rule" : "bg-ok/50"}`} />
      )}
      <span className={`absolute left-0 top-[7px] h-2.5 w-2.5 rounded-full ${dot}`} />
      <div className="flex flex-wrap items-baseline gap-x-2">
        <span className={`text-small ${working ? "text-ink" : "text-slate"}`}>
          {step.label}
          {working && <span className="sr-only"> (in progress)</span>}
        </span>
        {detail && <span className={`text-caption ${detailTone}`}>{detail}</span>}
        {step.duration_ms != null && !working && (
          <span className="tnum ml-auto text-caption text-mist">
            {step.duration_ms >= 1000 ? `${(step.duration_ms / 1000).toFixed(1)}s` : `${Math.round(step.duration_ms)}ms`}
            {(step.attempts ?? 1) > 1 ? `, ${step.attempts} attempts` : ""}
          </span>
        )}
      </div>
    </li>
  );
}
