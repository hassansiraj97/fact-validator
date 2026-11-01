"""
Central configuration.

All paths are resolved relative to the project, so the code works on any machine
(no hard-coded C:\\ paths). Secrets are read from environment variables / .env only.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# services/api/app/config.py -> services/api
API_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = API_DIR.parent.parent

load_dotenv(API_DIR / ".env")
load_dotenv()  # also allow a .env in the current working directory


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


DATA_DIR = Path(_env("FACT_VALIDATOR_DATA_DIR", str(API_DIR / "data")))
DB_PATH = Path(_env("FACT_VALIDATOR_DB", str(DATA_DIR / "fact_validator.db")))
DOMAIN_CACHE_PATH = Path(_env("FACT_VALIDATOR_DOMAIN_CACHE", str(DATA_DIR / "domain_cache.json")))
SEARCH_CACHE_DIR = Path(_env("FACT_VALIDATOR_SEARCH_CACHE", str(DATA_DIR / "search_cache")))
LOG_DIR = Path(_env("FACT_VALIDATOR_LOG_DIR", str(API_DIR / "logs")))

# External services (never hard-code keys in source code)
SERPAPI_API_KEY = _env("SERPAPI_API_KEY")
GROQ_API_KEY = _env("GROQ_API_KEY")
# llama-3.1-8b-instant was retired from Groq free/developer tiers on 16 Aug 2026; gpt-oss-20b is Groq's replacement
GROQ_MODEL = _env("GROQ_MODEL", "openai/gpt-oss-20b")
# low | medium | high (only used by reasoning models such as gpt-oss)
GROQ_REASONING_EFFORT = _env("GROQ_REASONING_EFFORT", "low")
GROQ_BASE_URL = _env("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# "groq" (default) or "mock" (deterministic offline LLM for tests / dry runs)
LLM_PROVIDER = _env("LLM_PROVIDER", "groq").lower()

# Set FACT_VALIDATOR_DEBUG_LOG=1 to write NDJSON debug traces to logs/debug.log
DEBUG_LOG = _env("FACT_VALIDATOR_DEBUG_LOG", "0") in ("1", "true", "yes")

# Comma-separated list of allowed frontend origins for CORS
CORS_ORIGINS = [o.strip() for o in _env("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()]
