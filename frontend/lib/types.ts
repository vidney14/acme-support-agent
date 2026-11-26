// Mirrors the backend event vocabulary (backend/app/schemas/events.py).
// SSE chat turns and voice WebSocket messages share these shapes.

export type Modality = "text" | "voice";
export type StepStatus = "started" | "retrying" | "completed" | "failed";

export interface Step {
  step_id: string;
  tool: string;
  label: string;
  status: StepStatus;
  summary?: string;
  duration_ms?: number;
  attempts?: number;
  attempt?: number;
  error?: { code?: string; message?: string } | string | null;
  args?: Record<string, unknown>;
}

export interface RetrievedChunk {
  chunk_id: string;
  breadcrumb: string;
  text: string;
  tiers: string[];
  dense_score: number | null;
  bm25_score: number;
  rrf_score: number;
  exact_match: boolean;
}

export interface Confidence {
  kind: "retrieval_confidence";
  query: string;
  score: number;
  band: "high" | "medium" | "low";
  decision: "answer" | "answer_with_caution" | "escalate";
  reasons: string[];
  verifier?: { answerable: string; reason: string; latency_ms?: number } | null;
  tier_filter: string;
  results: RetrievedChunk[];
}

export type GroundingStatus = "grounded" | "account_data" | "escalated" | "no_claims" | "unsupported" | "invalid_citation";

export interface Grounding {
  status: GroundingStatus;
  citations: string[];
  invalid_citations: string[];
  detail: string;
}

export interface Ticket {
  ticket_id: string;
  category: string;
  priority: string;
  summary: string;
  response_target: string;
  created_by: string;
  plan?: string;
}

export interface Notice {
  kind: string;
  message: string;
}

export interface TurnError {
  message: string;
  recoverable: boolean;
  kind?: string;
  retry_after_s?: number;
}

export interface Turn {
  id: string; // client-side id
  serverTurnId?: string;
  modality: Modality;
  user: string;
  agent: string;
  steps: Step[];
  confidences: Confidence[];
  grounding?: Grounding;
  sources: Record<string, string>;
  tickets: Ticket[];
  notices: Notice[];
  phase?: { phase: string; label: string };
  status: "streaming" | "done" | "error";
  error?: TurnError;
  interrupted?: boolean;
  traceId?: string;
  latencyMs?: number | null;
  restored?: boolean;
  startedAt: number;
  finishedAt?: number;
}

export interface Span {
  span_id: string;
  parent_id: string | null;
  name: string;
  kind: string;
  start_ms: number;
  duration_ms: number | null;
  status: string;
  attrs: Record<string, unknown>;
}

export interface Trace {
  trace_id: string;
  session_id: string;
  modality: Modality;
  user_input: string;
  started_at: number;
  ttft_ms: number | null;
  total_ms: number | null;
  status: string;
  totals: {
    input_tokens: number;
    output_tokens: number;
    cost_usd: number;
    llm_calls: number;
    tool_calls: number;
  };
  spans: Span[];
}

export interface SessionSnapshot {
  session_id: string;
  status: "active" | "escalated";
  verified: boolean;
  customer: { customer_id: string; name: string; tier: string; contract_status: string } | null;
  identity_locked: boolean;
  tickets: Ticket[];
  actions: { type: string; summary: string }[];
  faults: Faults;
  voice_connected: boolean;
  message_count: number;
}

export interface Faults {
  llm_rate_limit: number;
  llm_timeout: number;
  tool_flaky: number;
  crm_outage: boolean;
}

export interface DemoAccount {
  name: string;
  email: string;
  pin: string;
  tier: string;
  contract_status: string;
  scenario: string;
}

export interface AppConfig {
  provider: string;
  models: Record<string, string>;
  guardrails: { confidence_high: number; confidence_low: number };
  fault_injection: boolean;
  demo_accounts: DemoAccount[];
  voice_enabled: boolean;
  voice_simulated?: boolean;
}

export interface StoredMessage {
  role: "user" | "assistant";
  text: string;
  modality: Modality;
  turn_id: string;
  ts: number;
  meta: {
    grounding?: Grounding;
    sources?: Record<string, string>;
    interrupted?: boolean;
    tickets?: string[];
  };
}

export type AgentEvent = { type: string; data: Record<string, unknown> };
