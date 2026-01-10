from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    mongodb_uri: str | None
    mongodb_db: str
    app_env: str
    app_base_url: str
    demo_mode: bool
    council_mode: str
    openai_api_key: str | None
    openai_model: str
    agent_timeout_seconds: float
    agent_max_retries: int
    context_pack_token_budget: int


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


def load_settings() -> Settings:
    return Settings(
        mongodb_uri=os.getenv("MONGODB_URI") or None,
        mongodb_db=os.getenv("MONGODB_DB", "babyhandoff"),
        app_env=os.getenv("APP_ENV", "dev"),
        app_base_url=os.getenv("APP_BASE_URL", "http://127.0.0.1:8000"),
        demo_mode=_parse_bool(os.getenv("DEMO_MODE"), default=True),
        council_mode=os.getenv("COUNCIL_MODE", "heuristic").strip().lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.2"),
        agent_timeout_seconds=_parse_float(os.getenv("AGENT_TIMEOUT_SECONDS"), default=25.0),
        agent_max_retries=_parse_int(os.getenv("AGENT_MAX_RETRIES"), default=2),
        context_pack_token_budget=_parse_int(os.getenv("CONTEXT_PACK_TOKEN_BUDGET"), default=4000),
    )
