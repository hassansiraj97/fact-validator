"""Text, URL and claim-extraction helpers shared by the API and the evaluation code."""
from __future__ import annotations

import re
from typing import List, Optional
from urllib.parse import urlparse

import tldextract

# tldextract without network access: use the bundled public-suffix snapshot
_TLD = tldextract.TLDExtract(suffix_list_urls=())

_ARCHIVE_RE = re.compile(r"^https?://web\.archive\.org/web/[^/]+/(.+)$", re.IGNORECASE)


# ----------------------------------------------------------------------------
# URLs and domains
# ----------------------------------------------------------------------------

def normalize_url(u: Optional[str]) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    if "://" not in u:
        u = "https://" + u
    return u


def unwrap_archive_url(u: Optional[str]) -> str:
    """https://web.archive.org/web/2020.../https://bbc.com/x -> https://bbc.com/x"""
    u = (u or "").strip()
    m = _ARCHIVE_RE.match(u)
    if m:
        inner = m.group(1)
        return inner if "://" in inner else "https://" + inner
    return u


def extract_domain(url: Optional[str]) -> Optional[str]:
    """Host name without 'www.' (archive.org links are unwrapped first)."""
    try:
        u = normalize_url(unwrap_archive_url(url))
        if not u:
            return None
        host = (urlparse(u).netloc or "").lower().split("@")[-1].split(":")[0]
        if host.startswith("www."):
            host = host[4:]
        return host or None
    except Exception:
        return None


def base_domain(domain: Optional[str]) -> str:
    """news.bbc.co.uk -> bbc.co.uk ; ec.europa.eu -> europa.eu"""
    d = (domain or "").lower().strip()
    if d.startswith("www."):
        d = d[4:]
    ext = _TLD(d)
    if not ext.domain or not ext.suffix:
        return d
    return f"{ext.domain}.{ext.suffix}"


def domain_matches(domain: str, target: str) -> bool:
    """True if `domain` is `target` or a sub-domain of it (x.com matches x.com / m.x.com, NOT netflix.com)."""
    d = (domain or "").lower().strip(".")
    t = (target or "").lower().strip(".")
    return bool(d) and bool(t) and (d == t or d.endswith("." + t))


# ----------------------------------------------------------------------------
# Tokenisation / lexical overlap
# ----------------------------------------------------------------------------

STOPWORDS = {
    "this", "that", "with", "from", "were", "have", "has", "been", "into", "also",
    "their", "they", "them", "than", "more", "most", "such", "some", "many", "will",
    "would", "could", "should", "about", "there", "these", "those", "which", "what",
    "when", "where", "while", "after", "before", "over", "under", "said", "says",
}


def tokenize_for_overlap(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if (len(t) >= 4 or t.isdigit()) and t not in STOPWORDS]
    return toks[:80]


def compute_overlap(claim: str, text: str) -> int:
    """Number of distinct content tokens shared by claim and text."""
    claim_toks = set(tokenize_for_overlap(claim))
    if not claim_toks:
        return 0
    return len(claim_toks & set(tokenize_for_overlap(text)))


def overlap_ratio(claim: str, text: str) -> float:
    """Fraction of the claim's content tokens that also appear in text (0..1)."""
    claim_toks = set(tokenize_for_overlap(claim))
    if not claim_toks:
        return 0.0
    return len(claim_toks & set(tokenize_for_overlap(text))) / len(claim_toks)


# ----------------------------------------------------------------------------
# Claim extraction (heuristic check-worthiness ranking)
# ----------------------------------------------------------------------------

BOILERPLATE_MARKERS = [
    "this topic page can be cited as", "published online at", "cite this work",
    "reuse this work", "license terms", "creative commons", "open access",
    "data produced by third parties", "the underlying data for this chart",
    "sign up for", "subscribe to", "follow us on", "all rights reserved",
    "click here", "read more", "cookie", "newsletter", "share this",
    "image caption", "image source", "getty images", "watch:", "listen:",
]

# Live-blog / page furniture fragments
_NOISE_PATTERNS = [
    re.compile(r"^\s*(posted|updated|published)\s+\d", re.IGNORECASE),
    re.compile(r"^\s*\d{1,2}:\d{2}\s*(gmt|bst|cet|utc)?\b", re.IGNORECASE),  # "14:05 GMT ..."
    re.compile(r"\bexplore data\b", re.IGNORECASE),
    re.compile(r"\bresearch\s*&\s*writing\b", re.IGNORECASE),
]

_RELATION_TOKENS = [" is ", " are ", " was ", " were ", " has ", " have ", " had ", " will ", " won ", " killed ",
                    " increased ", " decreased ", " rose ", " fell ", " announced ", " said ", " caused "]


_HYPOTHETICAL = re.compile(
    r"^\W*(now\s+)?(suppose|imagine|picture|think of|consider|let'?s say|say you|what if|if you)\b", re.IGNORECASE
)


def heuristic_claim_score(s: str) -> float:
    """Simple check-worthiness proxy: medium length, contains numbers/entities and a factual relation."""
    s = s.strip()
    if not s:
        return 0.0
    length_score = 1.0 - min(abs(len(s) - 160) / 160.0, 1.0)
    number_bonus = 0.15 if any(ch.isdigit() for ch in s) else 0.0
    relation_bonus = 0.10 if any(tok in f" {s.lower()} " for tok in _RELATION_TOKENS) else 0.0
    # capitalised words after the first token ~ named entities
    entity_bonus = 0.05 if len(re.findall(r"\s[A-Z][a-z]+", s)) >= 2 else 0.0
    question_penalty = -0.3 if s.endswith("?") else 0.0
    # illustrations and hypotheticals ("Now suppose...", "Imagine...") are not checkable facts
    hypothetical_penalty = -0.4 if _HYPOTHETICAL.match(s) else 0.0
    score = 0.55 * length_score + number_bonus + relation_bonus + entity_bonus + question_penalty + hypothetical_penalty
    return max(0.05, min(score, 0.95))


def clean_text_for_claims(text: str) -> str:
    """Normalise whitespace. NOTE: dates are intentionally kept - they are often the fact being checked."""
    t = (text or "").strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def is_noise_sentence(s: str) -> bool:
    low = s.lower()
    if any(m in low for m in BOILERPLATE_MARKERS):
        return True
    return any(p.search(s) for p in _NOISE_PATTERNS)


def _sentences(block: str) -> List[str]:
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(block)
    except Exception:
        # Fallback splitter if NLTK data is unavailable
        return re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", block)


def extract_claim_candidates(text: str, max_claims: int = 6) -> List[str]:
    text = clean_text_for_claims(text)
    if not text:
        return []

    blocks = [b.strip() for b in text.split("\n") if b.strip()]
    candidates: List[str] = []
    seen = set()

    for b in blocks[:250]:
        for s in _sentences(b):
            s2 = " ".join(s.split()).strip()
            if len(s2) < 60 or len(s2) > 280:
                continue
            if is_noise_sentence(s2):
                continue
            key = s2.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(s2)
            if len(candidates) >= 120:
                break
        if len(candidates) >= 120:
            break

    if not candidates:
        # Short input (e.g. a single pasted claim such as "Paris is the capital of France"):
        # the 60-character minimum is meant for news articles, so accept shorter sentences here.
        for b in blocks[:50]:
            for s in _sentences(b):
                s2 = " ".join(s.split()).strip()
                if 10 <= len(s2) <= 280 and not is_noise_sentence(s2) and s2.lower() not in seen:
                    seen.add(s2.lower())
                    candidates.append(s2)

    ranked = sorted(candidates, key=heuristic_claim_score, reverse=True)
    return ranked[:max_claims]
