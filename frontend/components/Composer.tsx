"use client";

import { useEffect, useRef, useState } from "react";
import { MicIcon, SendIcon } from "./icons";

interface Props {
  disabled: boolean;
  busy: boolean;
  cooldownUntil: number;
  voiceAvailable: boolean;
  draft: string;
  onDraftChange: (v: string) => void;
  onSend: (text: string) => void;
  onStartVoice: () => void;
}

export function Composer({ disabled, busy, cooldownUntil, voiceAvailable, draft, onDraftChange, onSend, onStartVoice }: Props) {
  const ref = useRef<HTMLTextAreaElement>(null);
  // `now` starts at 0 (no clock reads during render/prerender) and ticks only during a cooldown.
  const [now, setNow] = useState(0);
  const cooling = cooldownUntil > now;

  useEffect(() => {
    if (cooldownUntil <= 0) return;
    const tick = () => {
      const t = Date.now();
      setNow(t);
      if (t >= cooldownUntil) clearInterval(timer);
    };
    const timer = setInterval(tick, 250);
    const first = setTimeout(tick, 0);
    return () => {
      clearInterval(timer);
      clearTimeout(first);
    };
  }, [cooldownUntil]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [draft]);

  const canSend = Boolean(draft.trim()) && !busy && !disabled && !cooling;
  const submit = () => {
    if (!canSend) return;
    onSend(draft);
    onDraftChange("");
  };

  const hint = cooling
    ? `Try again in ${Math.ceil((cooldownUntil - now) / 1000)}s`
    : busy
      ? "The agent is working on your last message"
      : "Enter to send, Shift+Enter for a new line";

  return (
    <div className="composer-glow rounded-2xl border border-rule bg-surface/95 p-2 shadow-[0_12px_32px_-20px_var(--color-ink)] backdrop-blur">
      <label htmlFor="composer" className="sr-only">
        Message
      </label>
      <textarea
        id="composer"
        ref={ref}
        rows={1}
        value={draft}
        disabled={disabled}
        onChange={(e) => onDraftChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            submit();
          }
        }}
        placeholder="Ask about billing, a refund, a sync error or your account"
        className="block w-full resize-none bg-transparent px-2 py-1.5 text-body text-ink placeholder:text-mist focus:outline-none focus-visible:outline-none disabled:opacity-50"
      />
      <div className="flex items-center justify-between gap-2 pl-2 pt-1">
        <span className="text-caption text-mist" aria-live="polite">
          {hint}
        </span>
        <div className="flex items-center gap-1.5">
          <button
            onClick={onStartVoice}
            disabled={!voiceAvailable || disabled}
            title={voiceAvailable ? "Start a voice call" : "Voice needs a Gemini API key on the server"}
            className="flex items-center gap-1.5 rounded-xl border border-rule px-3 py-2 text-small font-semibold text-ink transition-colors hover:border-brand/40 hover:bg-brand-soft hover:text-brand disabled:cursor-not-allowed disabled:border-transparent disabled:bg-transparent disabled:text-mist"
          >
            <MicIcon />
            <span className="hidden sm:inline">Voice call</span>
          </button>
          <button
            onClick={submit}
            disabled={!canSend}
            aria-label="Send message"
            className={`flex h-9 w-9 items-center justify-center rounded-xl transition-transform ${
              canSend
                ? "bg-brand-gradient text-surface shadow-[0_6px_16px_-6px_var(--color-brand)] hover:scale-105 active:scale-95"
                : "bg-rule text-mist"
            }`}
          >
            <SendIcon />
          </button>
        </div>
      </div>
    </div>
  );
}
