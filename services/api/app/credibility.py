"""
Source-credibility module: a transparent, rule-based score (0-100) per web domain.

Score = 50 (neutral prior)
        + max(TLD bonus, curated-list bonus)        (the two are not added together)
        + platform penalty                          (user-generated / social platforms)
        + keyword penalty                           (e.g. "hoax", "conspiracy" in the domain)
clipped to [0, 100].  Labels: HIGH >= 80, MEDIUM >= 50, LOW < 50.

This is NOT NewsGuard / MBFC; it is an explainable heuristic and should be read as a risk
signal. eval/validate_credibility.py compares it against the aggregated expert ratings of
Lin et al. (2023, PNAS Nexus).
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from app import config
from app.text_utils import base_domain, domain_matches

# Bump this whenever the rules below change: cached scores from older versions are ignored,
# so experiments are always computed with the current rubric.
RUBRIC_VERSION = "3"
CACHE_TTL_SECONDS = 60 * 60 * 24 * 14  # 14 days

NEUTRAL_PRIOR = 50

TLD_BONUS = {
    ".gov": 35, ".edu": 35, ".mil": 25, ".int": 25,
    ".gov.uk": 35, ".ac.uk": 30, ".gov.it": 35, ".gov.pk": 30, ".edu.pk": 30,
}

# Curated reputable sources. Matched on the full domain *or* any parent domain,
# so "ec.europa.eu" and "news.bbc.co.uk" are recognised.
CURATED_SOURCES: Dict[str, int] = {
    # Institutions / references
    "who.int": 35, "nih.gov": 35, "cdc.gov": 35, "ipcc.ch": 30, "un.org": 30,
    "oecd.org": 30, "worldbank.org": 30, "ec.europa.eu": 30, "europa.eu": 25,
    "imf.org": 30, "ecdc.europa.eu": 30,
    # Science
    "nature.com": 25, "science.org": 25, "thelancet.com": 25, "nejm.org": 25,
    "sciencedirect.com": 15,
    # Data publishers
    "ourworldindata.org": 20, "iea.org": 20, "unep.org": 20, "statista.com": 10,
    # Encyclopedias / general reference
    "britannica.com": 20, "wikipedia.org": 15,
    # News wires and major outlets
    "reuters.com": 30, "apnews.com": 30, "afp.com": 30, "bbc.com": 30, "bbc.co.uk": 30,
    "theguardian.com": 20, "nytimes.com": 20, "washingtonpost.com": 20, "ft.com": 20,
    "economist.com": 20, "npr.org": 20, "aljazeera.com": 15, "dw.com": 20, "ansa.it": 20,
    # Professional fact-checkers (IFCN signatories)
    "snopes.com": 25, "politifact.com": 25, "factcheck.org": 25, "fullfact.org": 25,
    "checkyourfact.com": 20, "leadstories.com": 20, "africacheck.org": 25,
}

# (domain, delta, reason) - matched with domain_matches(), i.e. exact domain or sub-domain only.
# (The old version used substring matching, so e.g. "netflix.com" was penalised as "x.com".)
PLATFORM_PENALTIES = [
    ("blogspot.com", -20, "User-generated blog hosting is higher risk"),
    ("wordpress.com", -20, "User-generated blog hosting is higher risk"),
    ("medium.com", -10, "Open publishing platform: quality varies"),
    ("substack.com", -10, "Newsletter platform: quality varies"),
    ("facebook.com", -20, "Social platform: high repost/misinformation risk"),
    ("instagram.com", -20, "Social platform: high repost/misinformation risk"),
    ("tiktok.com", -20, "Social platform: high repost/misinformation risk"),
    ("x.com", -20, "Social platform: high repost/misinformation risk"),
    ("twitter.com", -20, "Social platform: high repost/misinformation risk"),
    ("reddit.com", -15, "Forum platform: quality varies"),
    ("quora.com", -15, "Q&A platform: quality varies"),
    ("youtube.com", -10, "Video platform: quality varies widely"),
    ("linkedin.com", -10, "Professional social network: self-published posts"),
    # Homework / essay-sharing sites: user-uploaded student material, not primary sources
    ("studocu.com", -25, "User-uploaded study material, not a primary source"),
    ("coursehero.com", -25, "User-uploaded study material, not a primary source"),
    ("quizlet.com", -25, "User-made flashcards, not a primary source"),
    ("cliffsnotes.com", -20, "Study-guide site, not a primary source"),
    ("chegg.com", -25, "Homework-help site, not a primary source"),
    ("brainly.com", -25, "Homework Q&A site, not a primary source"),
    ("bartleby.com", -20, "Essay/homework site, not a primary source"),
    ("ukessays.com", -25, "Essay-writing site, not a primary source"),
    ("ipl.org", -20, "Essay-sharing site, not a primary source"),
    ("scribd.com", -15, "User-uploaded documents: quality varies"),
]

KEYWORD_PENALTIES = [("hoax", -30), ("clickbait", -25), ("rumor", -25), ("conspiracy", -25)]


@dataclass
class CredibilityScore:
    score: int
    label: str
    reasons: Dict[str, str] = field(default_factory=dict)


def label_for(score: int) -> str:
    if score >= 80:
        return "HIGH"
    if score >= 50:
        return "MEDIUM"
    return "LOW"


def _curated_bonus(domain: str) -> Optional[tuple]:
    """Most specific curated entry that matches the domain (or one of its parent domains)."""
    best = None
    for src, bonus in CURATED_SOURCES.items():
        if domain_matches(domain, src) and (best is None or len(src) > len(best[0])):
            best = (src, bonus)
    return best


def _tld_bonus(domain: str) -> Optional[tuple]:
    best = None
    for suffix, bonus in TLD_BONUS.items():
        if domain.endswith(suffix) and (best is None or len(suffix) > len(best[0])):
            best = (suffix, bonus)
    return best


def compute_score(domain: str) -> CredibilityScore:
    """Pure function (no cache) - used by tests and by eval/validate_credibility.py."""
    d = (domain or "").lower().strip().strip(".")
    if d.startswith("www."):
        d = d[4:]
    score = NEUTRAL_PRIOR
    reasons: Dict[str, str] = {}

    tld = _tld_bonus(d)
    cur = _curated_bonus(d)
    tld_val = tld[1] if tld else 0
    cur_val = cur[1] if cur else 0
    # Take the larger of the two bonuses instead of adding them (no double counting for e.g. nih.gov)
    if cur and cur_val >= tld_val:
        score += cur_val
        reasons["curated_list"] = f"'{cur[0]}' is in the curated reputable-source list (+{cur_val})."
    elif tld:
        score += tld_val
        reasons["tld"] = f"Institutional domain suffix '{tld[0]}' (+{tld_val})."

    for target, delta, msg in PLATFORM_PENALTIES:
        if domain_matches(d, target):
            score += delta
            reasons["platform_risk"] = f"{msg} ({delta})."
            break

    for marker, delta in KEYWORD_PENALTIES:
        if marker in d:
            score += delta
            reasons["keyword_risk"] = f"Domain contains '{marker}', indicating higher risk ({delta})."
            break

    if not reasons:
        reasons["neutral"] = f"No rule matched: neutral prior ({NEUTRAL_PRIOR})."

    score = max(0, min(score, 100))
    return CredibilityScore(score=score, label=label_for(score), reasons=reasons)


# ----------------------------------------------------------------------------
# Cached access (keyed by full domain + rubric version)
# ----------------------------------------------------------------------------

_lock = threading.Lock()
_mem_cache: Optional[Dict[str, Dict]] = None


def _load_cache() -> Dict[str, Dict]:
    global _mem_cache
    if _mem_cache is None:
        try:
            with open(config.DOMAIN_CACHE_PATH, "r", encoding="utf-8") as f:
                _mem_cache = json.load(f)
        except Exception:
            _mem_cache = {}
    return _mem_cache


def _save_cache(cache: Dict[str, Dict]) -> None:
    try:
        config.DOMAIN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = config.DOMAIN_CACHE_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        tmp.replace(config.DOMAIN_CACHE_PATH)
    except Exception:
        pass  # the cache is only an optimisation


def score_domain_rubric(domain: str, use_cache: bool = True) -> CredibilityScore:
    d = (domain or "").lower().strip()
    if d.startswith("www."):
        d = d[4:]
    if not use_cache:
        return compute_score(d)

    now = int(time.time())
    with _lock:
        cached = _load_cache().get(d)
        if (
            cached
            and cached.get("version") == RUBRIC_VERSION
            and (now - int(cached.get("ts", 0))) < CACHE_TTL_SECONDS
        ):
            return CredibilityScore(score=int(cached["score"]), label=cached["label"], reasons=cached["reasons"])

    res = compute_score(d)
    with _lock:
        cache = _load_cache()
        cache[d] = {"score": res.score, "label": res.label, "reasons": res.reasons, "ts": now, "version": RUBRIC_VERSION}
        _save_cache(cache)
    return res


__all__ = ["CredibilityScore", "score_domain_rubric", "compute_score", "label_for", "base_domain", "RUBRIC_VERSION"]
