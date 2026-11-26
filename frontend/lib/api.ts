import type { AppConfig, Faults, SessionSnapshot, StoredMessage, Trace } from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export function voiceSocketUrl(sessionId: string): string {
  const base = API_URL.replace(/^http/, "ws");
  return `${base}/ws/voice?session_id=${encodeURIComponent(sessionId)}`;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public retryAfterS?: number,
  ) {
    super(message);
  }
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(body.message || body.detail || `Request failed (${res.status})`, res.status, body.retry_after_s);
  }
  return res.json() as Promise<T>;
}

export const api = {
  config: () => json<AppConfig>("/api/config"),
  createSession: () => json<SessionSnapshot>("/api/sessions", { method: "POST" }),
  session: (id: string) => json<SessionSnapshot & { messages: StoredMessage[] }>(`/api/sessions/${id}`),
  traces: (id: string) => json<{ traces: Trace[] }>(`/api/sessions/${id}/traces`),
  setFaults: (id: string, faults: Partial<Faults>) =>
    json<Faults>(`/api/sessions/${id}/faults`, { method: "POST", body: JSON.stringify(faults) }),
};

/**
 * POST + Server-Sent Events. EventSource only supports GET, so we read the response
 * body stream ourselves and split it into `event:` / `data:` frames.
 * Resolves `true` if the server sent `done`, `false` if the stream ended early.
 */
export async function streamChat(
  body: { session_id?: string; message: string; retry?: boolean },
  onEvent: (type: string, data: Record<string, unknown>) => void,
  signal?: AbortSignal,
): Promise<boolean> {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => ({}));
    throw new ApiError(err.message || `Request failed (${res.status})`, res.status, err.retry_after_s);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let sawDone = false;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      let event = "message";
      const dataLines: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith(":")) continue; // heartbeat comment
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
      }
      if (!dataLines.length) continue;
      try {
        const data = JSON.parse(dataLines.join("\n"));
        if (event === "done") sawDone = true;
        onEvent(event, data);
      } catch {
        // ignore malformed frame
      }
    }
  }
  return sawDone;
}
