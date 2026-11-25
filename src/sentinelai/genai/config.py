"""Settings, read only from environment variables. Secrets are never stored, logged or defaulted."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping


class MissingSecretError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    provider: str = "template"            # template (offline, no LLM) | anthropic | fake
    anthropic_model: str = "claude-opus-5-5"
    effort: str = "medium"
    max_tokens: int = 16000
    embedder: str = "minilm"              # minilm (local model) | hash (offline, tests)
    embed_dim: int = 384
    database_url: str | None = field(default=None, repr=False)   # may contain a password
    top_k: int = 6

    @staticmethod
    def from_env(env: Mapping[str, str] | None = None) -> "Settings":
        e = os.environ if env is None else env
        return Settings(
            provider=e.get("SENTINEL_LLM_PROVIDER", "template"),
            anthropic_model=e.get("SENTINEL_ANTHROPIC_MODEL", "claude-opus-5-5"),
            effort=e.get("SENTINEL_EFFORT", "medium"),
            max_tokens=int(e.get("SENTINEL_MAX_TOKENS", "16000")),
            embedder=e.get("SENTINEL_EMBEDDER", "minilm"),
            embed_dim=int(e.get("SENTINEL_EMBED_DIM", "384")),
            database_url=e.get("DATABASE_URL") or None,
            top_k=int(e.get("SENTINEL_TOP_K", "6")),
        )


def require_api_key(env: Mapping[str, str] | None = None) -> str:
    """The Anthropic API key, from ANTHROPIC_API_KEY only. Raises a clear error if it is missing."""
    key = (os.environ if env is None else env).get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise MissingSecretError("ANTHROPIC_API_KEY is not set. Put it in your environment (see .env.example); "
                                 "it is never read from files in the repository.")
    return key
