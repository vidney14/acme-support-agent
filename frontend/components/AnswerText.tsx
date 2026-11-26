"use client";

import { Fragment, type ReactNode } from "react";

const CITE = /\[([a-z0-9-]+#\d+)\]/g;

/** Order in which citation tags first appear: that's the number shown on the chip. */
export function citationOrder(text: string): string[] {
  const seen: string[] = [];
  for (const m of text.matchAll(CITE)) if (!seen.includes(m[1])) seen.push(m[1]);
  return seen;
}

/**
 * Minimal, safe markdown: paragraphs, bullet/numbered lists, **bold**, and citation tags
 * rendered as numbered chips. No HTML injection: everything goes through React text nodes.
 */
export function AnswerText({
  text,
  sources,
  streaming,
}: {
  text: string;
  sources: Record<string, string>;
  streaming: boolean;
}) {
  const order = citationOrder(text);
  const blocks = text.split(/\n{2,}/);

  const inline = (s: string, key: string): ReactNode[] => {
    const out: ReactNode[] = [];
    // split on citations and **bold**
    const parts = s.split(/(\[[a-z0-9-]+#\d+\]|\*\*[^*]+\*\*)/g);
    parts.forEach((p, i) => {
      const cite = p.match(/^\[([a-z0-9-]+#\d+)\]$/);
      if (cite) {
        const n = order.indexOf(cite[1]) + 1;
        const label = sources[cite[1]] ?? cite[1];
        out.push(
          <sup key={`${key}-${i}`} className="mx-0.5">
            <span
              title={label}
              aria-label={`Source ${n}: ${label}`}
              className="tnum inline-flex h-4 min-w-4 items-center justify-center rounded bg-ok-soft px-1 text-[0.65rem] font-semibold text-ok"
            >
              {n}
            </span>
          </sup>,
        );
      } else if (p.startsWith("**") && p.endsWith("**") && p.length > 4) {
        out.push(
          <strong key={`${key}-${i}`} className="font-semibold">
            {p.slice(2, -2)}
          </strong>,
        );
      } else if (p) {
        out.push(<Fragment key={`${key}-${i}`}>{p}</Fragment>);
      }
    });
    return out;
  };

  return (
    <div className={`max-w-[68ch] space-y-3 text-body ${streaming ? "streaming-caret" : ""}`}>
      {blocks.map((block, bi) => {
        const lines = block.split("\n");
        const isList = lines.length > 0 && lines.every((l) => /^\s*([-*]|\d+\.)\s+/.test(l));
        if (isList) {
          const ordered = /^\s*\d+\./.test(lines[0]);
          const Tag = ordered ? "ol" : "ul";
          return (
            <Tag key={bi} className={`space-y-1 pl-5 ${ordered ? "list-decimal" : "list-disc"} marker:text-mist`}>
              {lines.map((l, li) => (
                <li key={li}>{inline(l.replace(/^\s*([-*]|\d+\.)\s+/, ""), `${bi}-${li}`)}</li>
              ))}
            </Tag>
          );
        }
        return (
          <p key={bi} className="whitespace-pre-wrap">
            {inline(block, `${bi}`)}
          </p>
        );
      })}
      {order.length > 0 && !streaming && (
        <ol className="space-y-0.5 border-t border-rule-soft pt-2 text-caption text-slate">
          {order.map((id, i) => (
            <li key={id} className="flex gap-2">
              <span className="tnum font-semibold text-ok">{i + 1}</span>
              <span>{sources[id] ?? id}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
