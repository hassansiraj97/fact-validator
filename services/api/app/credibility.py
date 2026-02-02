from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import json
import os
import time
import tldextract


CACHE_PATH = os.getenv(
    "FACTVALIDATOR_DOMAIN_CACHE",
    r"C:\Fact_Validator\services\api\data\domain_cache.json"
)
CACHE_TTL_SECONDS = 60 * 60 * 24 * 14  # 14 days


@dataclass
class CredibilityScore:
    score: int
    label: str
    reasons: Dict[str, str]


def base_domain(domain: str) -> str:
    ext = tldextract.extract(domain or "")
    if not ext.domain or not ext.suffix:
        return (domain or "").lower()
    return f"{ext.domain}.{ext.suffix}".lower()


def _load_cache() -> Dict[str, Dict]:
    try:
        if not os.path.exists(CACHE_PATH):
            return {}
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: Dict[str, Dict]) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _label(score: int) -> str:
    if score >= 80:
        return "HIGH"
    if score >= 50:
        return "MEDIUM"
    return "LOW"


def score_domain_rubric(domain: str) -> CredibilityScore:
    """
    Transparent rubric inspired by source credibility checklists.
    This is NOT NewsGuard; it's a thesis-safe, explainable rubric.
    """
    d = (domain or "").lower().strip()
    bd = base_domain(d)

    cache = _load_cache()
    now = int(time.time())

    cached = cache.get(bd)
    if cached and (now - int(cached.get("ts", 0))) < CACHE_TTL_SECONDS:
        return CredibilityScore(
            score=int(cached["score"]),
            label=cached["label"],
            reasons=cached["reasons"],
        )

    # Rubric components (0..100)
    score = 50
    reasons: Dict[str, str] = {}

    # Strong signals
    if bd.endswith(".gov") or bd.endswith(".edu"):
        score += 35
        reasons["tld"] = "Government/Education domain suffix is a strong trust signal (+35)."

    # Known reputable references (extend over time)
    whitelist = {
        # Institutions / references
        "who.int": 35,
        "nih.gov": 35,
        "cdc.gov": 35,
        "ipcc.ch": 30,
        "un.org": 30,
        "oecd.org": 30,
        "worldbank.org": 30,
        "ec.europa.eu": 30,

        # Science
        "nature.com": 25,
        "science.org": 25,
        "sciencedirect.com": 15,

        # Data publishers
        "ourworldindata.org": 20,
        "iea.org": 20,
        "unep.org": 20,

        # Encyclopedia / general reference
        "britannica.com": 20,
        "wikipedia.org": 15,

        # Major news outlets / wires (added)
        "bbc.com": 30,
        "bbc.co.uk": 30,
        "reuters.com": 30,
        "apnews.com": 25,
        "theguardian.com": 20,
        "nytimes.com": 20,
        "washingtonpost.com": 20,
    }

    if bd in whitelist:
        score += whitelist[bd]
        reasons["whitelist"] = f"Domain appears in local reputable-source list (+{whitelist[bd]})."

    # Weak signals / risks
    risk_markers = [
        ("blogspot.", -20, "User-generated blog hosting is higher risk (-20)."),
        ("wordpress.", -20, "User-generated blog hosting is higher risk (-20)."),
        ("medium.com", -10, "Open publishing platform: quality varies (-10)."),
        ("substack.com", -10, "Newsletter platform: quality varies (-10)."),
        ("facebook.com", -20, "Social platform: high repost/misinformation risk (-20)."),
        ("tiktok.com", -20, "Social platform: high repost/misinformation risk (-20)."),
        ("x.com", -20, "Social platform: high repost/misinformation risk (-20)."),
        ("twitter.com", -20, "Social platform: high repost/misinformation risk (-20)."),
        ("reddit.com", -15, "Forum platform: quality varies (-15)."),
        ("youtube.com", -10, "Video platform: varies widely (-10)."),
    ]
    for marker, delta, msg in risk_markers:
        if marker in d:
            score += delta
            reasons["platform_risk"] = msg
            break

    # Obvious spammy/low-quality keyword markers
    low_markers = [
        ("hoax", -30),
        ("clickbait", -25),
        ("rumor", -25),
        ("conspiracy", -25),
    ]
    for marker, delta in low_markers:
        if marker in d:
            score += delta
            reasons["keyword_risk"] = f"Domain contains marker '{marker}' indicating higher risk ({delta})."
            break

    score = max(0, min(score, 100))
    label = _label(score)

    cache[bd] = {"score": score, "label": label, "reasons": reasons, "ts": now}
    _save_cache(cache)

    return CredibilityScore(score=score, label=label, reasons=reasons)
