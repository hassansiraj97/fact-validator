from __future__ import annotations

from typing import List, Dict, Any, Literal, Tuple
import os
import json
import httpx

Verdict = Literal["SUPPORTED", "REFUTED", "NEI"]


def _env_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()


def _env_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.1:8b").strip()


def _compact_evidence(evidence: List[Dict[str, Any]], max_items: int = 2) -> List[Dict[str, Any]]:
    """
    Keep evidence small and deterministic to reduce Ollama latency.
    The caller may pass many evidence items, but we only keep top-N.
    """
    out: List[Dict[str, Any]] = []
    for e in evidence[:max_items]:
        out.append(
            {
                "domain": e.get("domain"),
                "domain_score": e.get("domain_score"),
                "snippet": (e.get("snippet") or "")[:450],
                "url": e.get("url"),
            }
        )
    return out


def _safe_json_parse(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        pass

    # If the model wrapped JSON with extra text, salvage the first {...} block.
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except Exception:
            return {}
    return {}


async def _ollama_chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    num_ctx: int = 1536,
    num_predict: int = 220,
) -> str:
    """
    Calls Ollama /api/chat with conservative settings to avoid timeouts.
    - trust_env=False avoids Windows proxy env vars breaking localhost calls
    - long read timeout accommodates slower machines
    - num_predict limits output length (major speed-up)
    """
    base_url = _env_ollama_base_url()
    model = _env_ollama_model()

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }

    timeout = httpx.Timeout(connect=30.0, read=900.0, write=120.0, pool=120.0)

    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        r = await client.post(f"{base_url}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return (data.get("message", {}) or {}).get("content", "") or ""


async def llm_debate_verdict(
    claim_text: str,
    evidence_items: List[Dict[str, Any]],
) -> Tuple[Verdict, float, str, Dict[str, Any]]:
    """
    Prover vs Skeptic vs Judge debate.
    Returns:
      verdict, confidence, short_summary, debug_trace
    """
    compact = _compact_evidence(evidence_items, max_items=2)

    system = (
        "You are part of a fact-checking debate system.\n"
        "You must ONLY use the provided evidence snippets.\n"
        "If evidence is insufficient or ambiguous, choose NEI.\n"
        "Do NOT invent sources.\n"
        "Be concise.\n"
    )

    prover = await _ollama_chat(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "Role: PROVER\n"
                    "Goal: Argue the claim is SUPPORTED using ONLY evidence.\n"
                    "Output 3-5 concise bullet points referencing evidence by (domain, domain_score).\n"
                    "If you cannot support, write: cannot support.\n\n"
                    f"CLAIM:\n{claim_text}\n\n"
                    f"EVIDENCE(JSON):\n{json.dumps(compact, ensure_ascii=False, indent=2)}\n"
                ),
            },
        ],
        temperature=0.2,
        num_ctx=1536,
        num_predict=220,
    )

    skeptic = await _ollama_chat(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "Role: SKEPTIC\n"
                    "Goal: Argue the claim is REFUTED or at least NOT proven using ONLY evidence.\n"
                    "Output 3-5 concise bullet points referencing evidence by (domain, domain_score).\n"
                    "Point out missing support, ambiguity, or weak sources.\n\n"
                    f"CLAIM:\n{claim_text}\n\n"
                    f"EVIDENCE(JSON):\n{json.dumps(compact, ensure_ascii=False, indent=2)}\n"
                ),
            },
        ],
        temperature=0.2,
        num_ctx=1536,
        num_predict=220,
    )

    judge_prompt = (
        "Role: JUDGE\n"
        "Decide the verdict based ONLY on evidence.\n"
        "Return STRICT JSON ONLY, no markdown, no extra text.\n"
        "Schema:\n"
        "{\n"
        '  "verdict": "SUPPORTED|REFUTED|NEI",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "summary": "one short sentence justification",\n'
        '  "key_domains": ["domain1","domain2"]\n'
        "}\n\n"
        f"CLAIM:\n{claim_text}\n\n"
        f"PROVER_ARGUMENT:\n{prover}\n\n"
        f"SKEPTIC_ARGUMENT:\n{skeptic}\n\n"
        f"EVIDENCE(JSON):\n{json.dumps(compact, ensure_ascii=False, indent=2)}\n"
    )

    judge_raw = await _ollama_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": judge_prompt}],
        temperature=0.1,
        num_ctx=1536,
        num_predict=180,
    )

    j = _safe_json_parse(judge_raw)

    verdict: Verdict = j.get("verdict", "NEI")
    confidence = j.get("confidence", 0.55)
    summary = j.get("summary", "Insufficient evidence to decide.")
    key_domains = j.get("key_domains", [])

    if verdict not in ("SUPPORTED", "REFUTED", "NEI"):
        verdict = "NEI"

    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.55

    confidence = max(0.05, min(confidence, 0.95))

    debug = {
        "provider": "ollama",
        "base_url": _env_ollama_base_url(),
        "model": _env_ollama_model(),
        "compact_evidence": compact,
        "prover": prover[:4000],
        "skeptic": skeptic[:4000],
        "judge_raw": judge_raw[:4000],
        "judge_json": j,
        "key_domains": key_domains,
    }

    return verdict, confidence, summary, debug
