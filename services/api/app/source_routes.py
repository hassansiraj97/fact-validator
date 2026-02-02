from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import tldextract
from fastapi import APIRouter, HTTPException

# IMPORTANT: your backend already uses this name in main.py
# and earlier you saw: "from app.credibility import score_domain_rubric"
from app.credibility import score_domain_rubric

router = APIRouter(tags=["source"])


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def base_domain(domain: str) -> str:
    d = (domain or "").lower().strip().replace("www.", "")
    ext = tldextract.extract(d)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return d


@router.get("/source/{domain}")
def source_score(domain: str) -> Dict[str, Any]:
    d = (domain or "").strip().lower().replace("www.", "")
    if not d:
        raise HTTPException(status_code=400, detail="domain is required")

    bd = base_domain(d)

    # credibility.py returns a dataclass-like object with score/label/reasons
    cs = score_domain_rubric(d)

    return {
        "domain": d,
        "base_domain": bd,
        "score": int(cs.score),
        "label": str(cs.label),
        "reasons": dict(cs.reasons or {}),
        "timestamp_utc": utc_now_iso(),
        "disclaimer": "Credibility score is heuristic; treat it as a risk signal, not ground truth.",
    }
