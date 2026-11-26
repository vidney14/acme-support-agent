"""Composition root: builds the object graph from Settings.

Everything is constructed here and passed down explicitly (no hidden globals), which
is what makes each layer unit-testable with fakes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.agent.orchestrator import AgentOrchestrator
from app.agent.state import SessionStore
from app.config import Settings
from app.llm.base import LLMProvider
from app.rag.embeddings import Embedder, GeminiEmbedder, HashingEmbedder
from app.rag.guardrails import ConfidenceGate
from app.rag.knowledge_base import KnowledgeBase
from app.rag.store import VectorStore
from app.tools.mock_crm import MockCRM
from app.tools.policy import PolicyRules
from app.tools.registry import ToolRegistry
from app.tools.support_tools import build_registry

log = logging.getLogger(__name__)

# The offline hashing embedder has a very different similarity scale from Gemini's,
# so fake mode uses its own (empirically chosen) thresholds.
FAKE_THRESHOLDS = (0.26, 0.20)


@dataclass
class Container:
    settings: Settings
    provider: LLMProvider
    genai_client: object | None
    kb: KnowledgeBase
    rules: PolicyRules
    registry: ToolRegistry
    sessions: SessionStore
    crm: MockCRM
    orchestrator: AgentOrchestrator
    live_connect: object | None = None   # test hook: replaces client.aio.live.connect


def build_container(settings: Settings, provider: LLMProvider | None = None,
                    embedder: Embedder | None = None) -> Container:
    genai_client = None
    fake = settings.llm_provider == "fake"
    if provider is None:
        if fake:
            from app.llm.fake import RuleBasedProvider
            provider = RuleBasedProvider()
        else:
            if not settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is not set. Set it, or run with LLM_PROVIDER=fake for offline mode.")
            from app.llm.gemini import GeminiProvider, make_client
            genai_client = make_client(settings.gemini_api_key)
            provider = GeminiProvider(genai_client)

    if embedder is None:
        embedder = (HashingEmbedder(settings.embedding_dim) if fake or genai_client is None
                    else GeminiEmbedder(genai_client, settings.embedding_model, settings.embedding_dim))
    high, low = (FAKE_THRESHOLDS if isinstance(embedder, HashingEmbedder)
                 else (settings.confidence_high, settings.confidence_low))

    gate = ConfidenceGate(high=high, low=low, provider=provider, verifier_model=settings.verifier_model,
                          verifier_thinking=settings.fallback_thinking_level)
    store = VectorStore(settings.collection_name, settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_path)
    kb = KnowledgeBase(settings.data_dir / "kb", store, embedder, gate, settings.chunk_max_tokens,
                       settings.chunk_overlap_tokens, settings.retrieval_top_k)
    rules = PolicyRules.load(settings.data_dir / "policy_rules.yaml")
    crm = MockCRM.load(settings.data_dir / "customers.json")
    registry = build_registry()
    sessions = SessionStore(crm, ttl_minutes=settings.session_ttl_minutes)
    orchestrator = AgentOrchestrator(provider, registry, kb, rules, settings)
    live_connect = None
    if fake:
        from app.voice.fake_live import demo_connect
        live_connect = demo_connect()  # scripted voice call, so the voice UI works offline
    return Container(settings, provider, genai_client, kb, rules, registry, sessions, crm, orchestrator,
                     live_connect)
