"""Central configuration. Every tunable lives here and can be overridden via env vars.

Keeping thresholds, model names and limits in one typed object makes the system
easy to reason about ("what is the guardrail threshold in prod?") and easy to test
(tests construct a Settings with different values).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    # --- Provider -----------------------------------------------------------
    # "gemini" for real use; "fake" runs a deterministic offline provider (tests/dev).
    llm_provider: str = "gemini"
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    chat_model: str = "gemini-3.8-flash"
    chat_thinking_level: str = "low"          # low latency matters more than depth here
    fallback_model: str = "gemini-3.5-flash-lite"
    fallback_thinking_level: str = "minimal"
    verifier_model: str = "gemini-3.5-flash-lite"
    live_model: str = "gemini-3.8-live"
    live_voice: str = "Kore"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768                  # MRL-truncated; plenty for a small KB

    # --- Resilience ---------------------------------------------------------
    llm_timeout_s: float = 25.0               # per LLM request (time to first byte + stream)
    llm_max_retries: int = 2                  # retries on the primary before falling back
    tool_timeout_s: float = 6.0
    tool_max_retries: int = 1
    max_agent_steps: int = 8                  # hard cap on reasoning-loop iterations per turn

    # --- RAG + guardrails ---------------------------------------------------
    qdrant_url: str = ""                      # empty => embedded Qdrant (in-process)
    qdrant_api_key: str = ""
    qdrant_path: str = ""                     # empty => in-memory; set a path to persist
    collection_name: str = "acme_kb"
    chunk_max_tokens: int = 220
    chunk_overlap_tokens: int = 40
    retrieval_top_k: int = 4
    # Confidence bands on cosine similarity of the best dense match.
    # >= high: answer.  [low, high): ask the verifier model.  < low: escalate.
    confidence_high: float = 0.72
    confidence_low: float = 0.55

    # --- API / production ---------------------------------------------------
    cors_origins: str = "http://localhost:3000"
    rate_limit_per_minute: int = 12           # chat turns per client per minute
    rate_limit_burst: int = 4
    voice_sessions_per_minute: int = 3
    max_concurrent_voice_sessions: int = 3
    session_ttl_minutes: int = 120
    enable_fault_injection: bool = True       # demo panel toggles; disable in real prod

    data_dir: Path = DATA_DIR

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
