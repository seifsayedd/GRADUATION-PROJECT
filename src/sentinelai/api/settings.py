"""API settings, read only from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

DEFAULT_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


@dataclass(frozen=True)
class ApiSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: tuple[str, ...] = DEFAULT_ORIGINS
    max_question_chars: int = 500

    @staticmethod
    def from_env(env: Mapping[str, str] | None = None) -> "ApiSettings":
        e = os.environ if env is None else env
        raw = e.get("SENTINEL_API_CORS_ORIGINS", ",".join(DEFAULT_ORIGINS))
        origins = tuple(o.strip() for o in raw.split(",") if o.strip())
        return ApiSettings(host=e.get("SENTINEL_API_HOST", "127.0.0.1"),
                           port=int(e.get("SENTINEL_API_PORT", "8000")),
                           cors_origins=origins,
                           max_question_chars=int(e.get("SENTINEL_API_MAX_QUESTION", "500")))
