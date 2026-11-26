"use client";

import { useEffect, useRef } from "react";
import { CloseIcon } from "./icons";
import type { AppConfig, Faults } from "@/lib/types";

export const SUGGESTIONS = [
  "I was charged twice this month. Can I get a refund?",
  "My desktop app shows error E-1042 when syncing",
  "What happens to my files if my subscription expires?",
  "Do you offer a student discount?",
  "Please change my phone number to +1 617 555 0199",
  "I'd like to talk to a human",
];

interface Props {
  config: AppConfig;
  faults: Faults | undefined;
  onClose: () => void;
  onUseAccount: (text: string) => void;
  onPrompt: (text: string) => void;
  onFaults: (f: Partial<Faults>) => void;
}

export function DemoGuide({ config, faults, onClose, onUseAccount, onPrompt, onFaults }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;
    ref.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      prev?.focus();
    };
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-40 flex" role="dialog" aria-modal="true" aria-labelledby="guide-title">
      <button className="absolute inset-0 bg-ink/25" aria-label="Close guide" onClick={onClose} />
      <div
        ref={ref}
        tabIndex={-1}
        className="scroll-thin relative z-10 h-full w-full max-w-[34rem] overflow-y-auto bg-surface px-6 py-5 shadow-xl focus:outline-none"
      >
        <div className="flex items-start justify-between">
          <div>
            <h2 id="guide-title" className="font-display text-display">
              Demo guide
            </h2>
            <p className="mt-1 max-w-[46ch] text-small text-slate">
              Every account below is fake. Each one triggers a different path through the policy engine.
            </p>
          </div>
          <button onClick={onClose} aria-label="Close guide" className="rounded-lg p-1.5 text-slate hover:bg-paper">
            <CloseIcon />
          </button>
        </div>

        <section className="mt-6">
          <h3 className="font-display text-title">Test accounts</h3>
          <p className="mt-1 text-small text-slate">
            The agent asks for an email and a 4-digit support PIN before it touches an account.
          </p>
          <ul className="mt-3 divide-y divide-rule-soft">
            {config.demo_accounts.map((a) => (
              <li key={a.email} className="flex items-start gap-3 py-3">
                <div className="min-w-0 flex-1">
                  <p className="text-small text-ink">
                    <span className="font-semibold">{a.name}</span>
                    <span className="text-slate">, {a.tier} plan{a.contract_status !== "active" ? `, ${a.contract_status.replace("_", " ")}` : ""}</span>
                  </p>
                  <p className="tnum text-caption text-slate">
                    {a.email}, PIN {a.pin}
                  </p>
                  <p className="mt-0.5 text-caption text-mist">{a.scenario}</p>
                </div>
                <button
                  onClick={() => onUseAccount(`My email is ${a.email} and my PIN is ${a.pin}`)}
                  className="shrink-0 rounded-lg border border-rule px-2.5 py-1 text-small text-ink hover:bg-paper"
                >
                  Use
                </button>
              </li>
            ))}
          </ul>
        </section>

        <section className="mt-6">
          <h3 className="font-display text-title">Things to try</h3>
          <div className="mt-2 flex flex-wrap gap-2">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => onPrompt(s)}
                className="rounded-full border border-rule px-3 py-1.5 text-left text-small text-ink hover:border-slate"
              >
                {s}
              </button>
            ))}
          </div>
        </section>

        {config.fault_injection && faults && (
          <section className="mt-6 pb-4">
            <h3 className="font-display text-title">Simulate failures</h3>
            <p className="mt-1 text-small text-slate">
              Arm a fault, then send a message. Recovery runs through the real retry, fallback and escalation code.
            </p>
            <ul className="mt-3 space-y-2">
              <FaultRow
                label="Rate limit on the next model call"
                help="The primary model returns 429, so the agent switches to the fallback model."
                armed={faults.llm_rate_limit > 0}
                onArm={() => onFaults({ llm_rate_limit: faults.llm_rate_limit > 0 ? 0 : 1 })}
              />
              <FaultRow
                label="Timeout on the next model call"
                help="The call times out and is retried with backoff."
                armed={faults.llm_timeout > 0}
                onArm={() => onFaults({ llm_timeout: faults.llm_timeout > 0 ? 0 : 1 })}
              />
              <FaultRow
                label="Account system fails once"
                help="The next account tool call fails, then succeeds on retry."
                armed={faults.tool_flaky > 0}
                onArm={() => onFaults({ tool_flaky: faults.tool_flaky > 0 ? 0 : 1 })}
              />
              <FaultRow
                label="Account system outage"
                help="Every account tool fails until you switch this off; the agent explains and offers a ticket."
                armed={faults.crm_outage}
                onArm={() => onFaults({ crm_outage: !faults.crm_outage })}
              />
            </ul>
          </section>
        )}
      </div>
    </div>
  );
}

function FaultRow({ label, help, armed, onArm }: { label: string; help: string; armed: boolean; onArm: () => void }) {
  return (
    <li className="flex items-start gap-3">
      <button
        role="switch"
        aria-checked={armed}
        onClick={onArm}
        className={`mt-0.5 flex h-5 w-9 shrink-0 items-center rounded-full p-0.5 transition-colors ${armed ? "bg-fault" : "bg-rule"}`}
      >
        <span className={`h-4 w-4 rounded-full bg-surface transition-transform ${armed ? "translate-x-4" : ""}`} />
        <span className="sr-only">{label}</span>
      </button>
      <div>
        <p className="text-small text-ink">{label}</p>
        <p className="text-caption text-mist">{help}</p>
      </div>
    </li>
  );
}
