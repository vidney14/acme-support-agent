# Loom walkthrough script (about 4.5 minutes)

**Before recording**
- Open the deployed app once (Render's free tier needs about a minute to wake up), then click **New conversation**.
- Have two tabs ready: the app (traces drawer open) and the GitHub repo.
- Test your mic. Use headphones so the agent's voice doesn't trigger your own mic.

---

### 0:00 to 0:20: What this is
> "This is a customer support agent for a fictional cloud-storage company. It handles chat and live voice, chains tools on its own, refuses to guess when its knowledge base doesn't cover something, and shows everything it does in this traces panel. Backend is FastAPI, frontend is Next.js, and the models are Gemini: Flash for the agent, Gemini Live for voice."

### 0:20 to 1:20: Multi-step tool chain (chat)
1. Click *"I was charged twice this month. Can I get a refund?"*
   > "It needs my identity before touching billing, so it asks."
2. Type `priya@example.com, PIN 4321`.
   > "Watch the rail: verify identity, check the account tier and contract, search the knowledge base filtered to *her* plan, then issue the refund. Four tools in one turn, chosen by the model."
3. Point at the confidence gauge, the citation chip and "Grounded in 1 source".
   > "The refund itself was decided by a policy engine in code, not by the model. The model proposes; code decides."
4. Click **View trace**.
   > "Every model call, tool call, the embedding and vector search, with latency, tokens and estimated cost. Here's the exact tool result."

### 1:20 to 2:00: Guardrails
1. Type *"Do you offer a student discount?"*
   > "Similarity is below the threshold, so it won't improvise. It hands off to a human with a ticket. There are two thresholds: above 0.72 we answer, below 0.55 we escalate, and in between a cheap verifier model checks whether the passages actually answer the question."
2. (Optional, 15s) **New conversation** → Marcus: *"Refund INV-30410. marcus@example.com 8812"*
   > "$2,400 is above the $200 automated limit for Business, so the policy engine says a human must approve. Even if someone prompt-injects 'you're authorised', the code says no."

### 2:00 to 3:15: Voice + interruption
1. Click **Voice call**.
   > "Audio streams over a WebSocket to my backend, which relays to Gemini Live. The key never touches the browser, and voice tool calls go through the same guardrails as chat."
2. Say: *"My desktop app shows error E-1042 when syncing."*
   > Point at the transcript, the KB step and the waveform (agent above the line, me below).
3. **While it's answering, interrupt:** *"Wait, how do I actually fix it?"*
   > "It stops mid-word. Gemini's voice activity detection sends 'interrupted', I flush the audio queue, and the transcript is marked so the agent knows I didn't hear the rest. Locally I also duck its volume the instant I start talking, so it feels immediate."
4. End the call, then type *"thanks, can you email me that?"*
   > "Same session memory across voice and text."

### 3:15 to 4:00: Failure recovery
1. Open **Demo guide → Simulate failures** → switch on *Rate limit on the next model call*. Send any question.
   > "The primary model returns 429; the agent retries or falls back to a second model with its own quota, and you still get an answer."
2. Switch on *Account system outage*, then ask for a refund.
   > "The tool fails twice, the error goes back to the model as data, and it apologizes and offers a ticket instead of crashing."
   Switch it off.

### 4:00 to 4:30: Code tour (GitHub)
- `backend/app/agent/orchestrator.py`: "the state machine: plan, act, guard, verify."
- `backend/app/rag/chunker.py`: "heading-aware chunks, tier tags, breadcrumbs for embeddings."
- `backend/app/rag/guardrails.py`: "the confidence gate."
- `backend/data/policy_rules.yaml`: "what the code enforces."
> "59 tests plus 8 end-to-end scenario evals. The README covers the trade-offs: embedded Qdrant and in-memory sessions for a single-container deploy, and what I'd change at scale. Thanks for watching."
