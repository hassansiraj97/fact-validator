"""
LLM access layer.

* Provider "groq": OpenAI-compatible Groq API (default model openai/gpt-oss-20b).
* Provider "mock": deterministic offline stand-in, used by the unit tests and by
  `run_eval.py --dry-run` so the whole pipeline can be checked without an API key.

Every call is recorded in the active `UsageTracker` (number of calls, prompt/completion
tokens, wall-clock latency), which the evaluation uses for the efficiency metrics.
"""
from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import random
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from app import config


# ----------------------------------------------------------------------------
# Usage tracking
# ----------------------------------------------------------------------------

@dataclass
class UsageTracker:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_seconds: float = 0.0
    retries: int = 0
    events: List[Dict] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> Dict:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_seconds": round(self.llm_seconds, 3),
            "retries": self.retries,
        }


_current: contextvars.ContextVar[Optional[UsageTracker]] = contextvars.ContextVar("llm_usage", default=None)


@contextmanager
def track_usage() -> Iterator[UsageTracker]:
    tracker = UsageTracker()
    token = _current.set(tracker)
    try:
        yield tracker
    finally:
        _current.reset(token)


def _record(prompt_tokens: int, completion_tokens: int, seconds: float, retries: int, role: str) -> None:
    t = _current.get()
    if t is None:
        return
    t.calls += 1
    t.prompt_tokens += int(prompt_tokens or 0)
    t.completion_tokens += int(completion_tokens or 0)
    t.llm_seconds += seconds
    t.retries += retries
    t.events.append({"role": role, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "seconds": round(seconds, 3)})


# ----------------------------------------------------------------------------
# Groq provider
# ----------------------------------------------------------------------------

_client = None


def is_reasoning_model(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith("openai/gpt-oss") or m.startswith("qwen/qwen3")


def _groq_client():
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set (add it to services/api/.env).")
        from openai import AsyncOpenAI

        _client = AsyncOpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL, timeout=60.0, max_retries=0)
    return _client


async def _groq_chat(messages, temperature, max_tokens, json_mode, model, role) -> str:
    from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

    client = _groq_client()
    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if is_reasoning_model(model):
        # Reasoning models (gpt-oss, qwen3) think before answering and the thinking counts towards
        # max_tokens: keep the effort low and leave room for the answer itself.
        kwargs["max_tokens"] = max(int(max_tokens) * 3, 1500)
        kwargs["extra_body"] = {"reasoning_effort": config.GROQ_REASONING_EFFORT}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    retries = 0
    delay = 2.0
    while True:
        t0 = time.perf_counter()
        try:
            resp = await client.chat.completions.create(**kwargs)
            dt = time.perf_counter() - t0
            usage = getattr(resp, "usage", None)
            _record(getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0), dt, retries, role)
            return resp.choices[0].message.content or ""
        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            err = e
        except APIStatusError as e:
            # JSON mode can be rejected if the model fails to produce valid JSON -> retry without it
            if e.status_code == 400 and "response_format" in kwargs:
                kwargs.pop("response_format", None)
                retries += 1
                continue
            # A model that does not accept reasoning_effort -> retry without it
            if e.status_code == 400 and "extra_body" in kwargs:
                kwargs.pop("extra_body", None)
                retries += 1
                continue
            if e.status_code == 404:
                raise RuntimeError(
                    f"Groq model '{model}' is not available to your account. Set GROQ_MODEL in "
                    "services/api/.env to a current model (see https://console.groq.com/docs/models)."
                ) from e
            if e.status_code < 500:
                raise
            err = e
        retries += 1
        if retries > 6:
            raise RuntimeError(f"Groq request failed after {retries} attempts: {err}")
        wait = delay
        resp_obj = getattr(err, "response", None)
        if resp_obj is not None:
            try:
                wait = max(wait, float(resp_obj.headers.get("retry-after", 0)))
            except Exception:
                pass
        await asyncio.sleep(wait + random.uniform(0, 0.5))
        delay = min(delay * 2, 60)


# ----------------------------------------------------------------------------
# Mock provider (deterministic, offline)
# ----------------------------------------------------------------------------

_NEG = re.compile(r"\b(no|not|false|fake|never|hoax|incorrect|misleading|debunk\w*|denied|refut\w*)\b", re.IGNORECASE)


def _mock_reply(messages: List[Dict[str, str]], json_mode: bool, temperature: float) -> str:
    prompt = "\n".join(m["content"] for m in messages)
    evid = prompt.split("EVIDENCE", 1)[1] if "EVIDENCE" in prompt else ""
    has_evidence = bool(re.search(r"(snippet|text)\W+\s*\S{3,}", evid)) and "[]" not in evid[:40]
    neg = len(_NEG.findall(evid))
    if not has_evidence:
        verdict = "NEI"
    elif neg >= 2:
        verdict = "REFUTED"
    else:
        verdict = "SUPPORTED"
    h = int(hashlib.md5((prompt + str(temperature)).encode()).hexdigest(), 16)
    conf = round(0.55 + (h % 40) / 100, 2)

    if "Role: PROVER" in prompt:
        obj = {"role": "support", "datum": "mock datum", "warrant": "mock warrant", "claim": "mock supported conclusion", "confidence": conf}
    elif "Role: SKEPTIC" in prompt:
        obj = {"role": "attack", "attack_type": "rebuttal" if verdict == "REFUTED" else "undermining", "datum": "mock datum", "warrant": "mock warrant", "claim": "mock attack conclusion", "confidence": conf}
    elif "Role: JUDGE" in prompt:
        obj = {"verdict": verdict, "confidence": conf, "accepted_arguments": ["support"] if verdict == "SUPPORTED" else ["attack"],
               "defeated_arguments": ["attack"] if verdict == "SUPPORTED" else ["support"], "summary": f"Mock judge decided {verdict}.",
               "formal_argument": {"support": {"datum": "", "warrant": "", "claim": ""}, "attack": {"attack_type": "", "datum": "", "warrant": "", "claim": ""}}}
    else:
        obj = {"reasoning": "Mock step-by-step reasoning.", "verdict": verdict, "confidence": conf}
    if not json_mode and ("Role: PROVER" in prompt or "Role: SKEPTIC" in prompt) and "free-form" in prompt:
        return "Mock free-form argument."
    return json.dumps(obj)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

async def chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 450,
    json_mode: bool = False,
    model: Optional[str] = None,
    role: str = "llm",
) -> str:
    msgs = [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]
    provider = config.LLM_PROVIDER
    if provider == "mock":
        t0 = time.perf_counter()
        out = _mock_reply(msgs, json_mode, temperature)
        prompt_tokens = sum(len(m["content"]) for m in msgs) // 4
        _record(prompt_tokens, len(out) // 4, time.perf_counter() - t0, 0, role)
        return out
    if provider == "groq":
        return await _groq_chat(msgs, temperature, max_tokens, json_mode, model or config.GROQ_MODEL, role)
    raise RuntimeError(f"Unknown LLM_PROVIDER '{provider}' (use 'groq' or 'mock').")


def safe_json_parse(s: str) -> Dict:
    """Parse a JSON object from an LLM reply, tolerating code fences / surrounding text."""
    s = (s or "").strip()
    if not s:
        return {}
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        try:
            v = json.loads(s[start : end + 1])
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}
    return {}


def model_name() -> str:
    return "mock" if config.LLM_PROVIDER == "mock" else config.GROQ_MODEL
