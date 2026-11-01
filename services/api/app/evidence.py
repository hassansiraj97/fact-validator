"""
Evidence handling: label sets, evidence filtering (independence / circularity), and
ranking + formatting of evidence for the LLM verifiers.

An evidence item is a dict:
    {"url", "title", "snippet", "domain", "domain_score", "overlap", ...}
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.credibility import score_domain_rubric
from app.text_utils import (
    base_domain,
    compute_overlap,
    domain_matches,
    extract_domain,
    overlap_ratio,
    tokenize_for_overlap,
)

# ----------------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------------

SUPPORTED, REFUTED, NEI, CONFLICTING = "SUPPORTED", "REFUTED", "NEI", "CONFLICTING"
LABELS_3: Tuple[str, ...] = (SUPPORTED, REFUTED, NEI)
LABELS_4: Tuple[str, ...] = (SUPPORTED, REFUTED, NEI, CONFLICTING)

_LABEL_ALIASES = {
    "SUPPORTED": SUPPORTED, "SUPPORTS": SUPPORTED, "SUPPORT": SUPPORTED, "TRUE": SUPPORTED,
    "REFUTED": REFUTED, "REFUTES": REFUTED, "REFUTE": REFUTED, "FALSE": REFUTED,
    "NEI": NEI, "NOT ENOUGH INFO": NEI, "NOT ENOUGH INFORMATION": NEI, "NOT ENOUGH EVIDENCE": NEI,
    "INSUFFICIENT": NEI, "UNVERIFIABLE": NEI,
    "CONFLICTING": CONFLICTING, "CONFLICTING EVIDENCE/CHERRYPICKING": CONFLICTING,
    "CONFLICTING EVIDENCE": CONFLICTING, "CHERRYPICKING": CONFLICTING, "CHERRY-PICKING": CONFLICTING, "MIXED": CONFLICTING,
}


def normalize_label(raw: Any, labels: Sequence[str] = LABELS_3) -> str:
    """Map free-form label strings to the canonical set; unknown -> NEI.
    If CONFLICTING is not part of the label set it is mapped to NEI."""
    s = str(raw or "").strip().upper().replace("_", " ")
    lab = _LABEL_ALIASES.get(s)
    if lab is None:
        for k, v in _LABEL_ALIASES.items():
            if k in s:
                lab = v
                break
    lab = lab or NEI
    if lab not in labels:
        lab = NEI
    return lab


# ----------------------------------------------------------------------------
# Evidence construction / filtering for live web search
# ----------------------------------------------------------------------------

BLOCKED_DOMAINS = {
    "facebook.com", "x.com", "twitter.com", "tiktok.com", "instagram.com", "reddit.com",
    "pinterest.com", "worldarticledatabase.com", "wecanfigurethisout.org",
}

# Sites that mostly republish other outlets' articles: not independent evidence
AGGREGATOR_DOMAINS = {
    "msn.com", "aol.com", "yahoo.com", "news.google.com", "flipboard.com", "newsbreak.com",
    "inkl.com", "headtopics.com", "newsnow.co.uk", "pressreader.com", "ground.news",
}

# Copies: if a snippet contains this share of the claim's content words it is treated as a
# re-publication of the same text, not as independent confirmation.
NEAR_COPY_RATIO = 0.9
NEAR_COPY_MIN_TOKENS = 6


def is_blocked_domain(domain: str) -> bool:
    return any(domain_matches(domain, b) for b in BLOCKED_DOMAINS)


def is_aggregator(domain: str) -> bool:
    return any(domain_matches(domain, a) for a in AGGREGATOR_DOMAINS)


def make_evidence_item(claim: str, url: str, snippet: str, title: Optional[str] = None,
                       use_credibility_cache: bool = True, **extra: Any) -> Dict[str, Any]:
    dom = extract_domain(url) or ""
    cred = score_domain_rubric(dom, use_cache=use_credibility_cache) if dom else None
    item = {
        "url": url,
        "title": title,
        "snippet": (snippet or "").strip(),
        "domain": dom,
        "base_domain": base_domain(dom),
        "domain_score": int(cred.score) if cred else 50,
        "overlap": int(compute_overlap(claim, snippet or "")),
    }
    item.update(extra)
    return item


def filter_search_results(
    claim: str,
    raw_results: List[Dict[str, Any]],
    input_domain: Optional[str] = None,
    max_per_base_domain: int = 2,
    min_snippet_chars: int = 40,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Turn raw search hits into evidence items and drop non-independent ones.

    Returns (kept, excluded); every excluded item carries an 'excluded_reason'.
    Rules:
      * same site as the article being checked  -> circular evidence
      * aggregator / syndication sites            -> re-publish the same story
      * near-verbatim copies of the claim         -> same text, not independent confirmation
      * social media / blocked domains, very short snippets, > N results per site
    """
    input_bd = base_domain(input_domain) if input_domain else ""
    kept: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    per_bd: Dict[str, int] = {}

    for rr in raw_results:
        link = rr.get("link") or rr.get("url") or ""
        snippet = (rr.get("snippet") or "").strip()
        item = make_evidence_item(claim, link, snippet, rr.get("title"))
        dom, bd = item["domain"], item["base_domain"]

        reason = None
        if not dom:
            reason = "no_domain"
        elif input_bd and bd == input_bd:
            reason = "same_site_as_input"
        elif is_blocked_domain(dom):
            reason = "blocked_domain"
        elif is_aggregator(dom):
            reason = "aggregator_or_syndication"
        elif len(snippet) < min_snippet_chars:
            reason = "snippet_too_short"
        elif (len(set(tokenize_for_overlap(claim))) >= NEAR_COPY_MIN_TOKENS
              and overlap_ratio(claim, snippet) >= NEAR_COPY_RATIO):
            reason = "near_verbatim_copy_of_claim"
        elif per_bd.get(bd, 0) >= max(1, int(max_per_base_domain)):
            reason = "too_many_from_same_site"

        if reason:
            item["excluded_reason"] = reason
            excluded.append(item)
            continue
        per_bd[bd] = per_bd.get(bd, 0) + 1
        kept.append(item)

    return kept, excluded


# ----------------------------------------------------------------------------
# Ranking / formatting for the LLM
# ----------------------------------------------------------------------------

def rank_for_llm(evidence: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    """Most relevant first, credibility as tie-breaker.
    Relevance = explicit 'relevance' score if the retriever provided one (e.g. BM25, annotator order),
    otherwise the lexical overlap with the claim."""
    def key(e: Dict[str, Any]):
        rel = e.get("relevance")
        rel = float(rel) if rel is not None else float(e.get("overlap", 0))
        return (rel, int(e.get("domain_score", 0)))
    return sorted(evidence, key=key, reverse=True)[:k]


def format_evidence_for_llm(evidence: List[Dict[str, Any]], show_credibility: bool = True, max_chars: int = 600) -> str:
    rows = []
    for i, e in enumerate(evidence, 1):
        row: Dict[str, Any] = {"id": i, "source": e.get("domain") or "unknown"}
        if show_credibility:
            row["source_credibility_0_100"] = e.get("domain_score")
        row["text"] = (e.get("snippet") or "")[:max_chars]
        rows.append(row)
    return json.dumps(rows, ensure_ascii=False, indent=1)


def label_definitions(labels: Sequence[str]) -> str:
    lines = [
        "SUPPORTED: the evidence clearly confirms the claim.",
        "REFUTED: the evidence clearly contradicts the claim or shows it is false or fabricated.",
        "NEI: the evidence is missing, unrelated or too weak to decide. Evidence that simply does not mention "
        "the claim is NEI, NOT REFUTED.",
    ]
    if CONFLICTING in labels:
        lines.append("CONFLICTING: the evidence both supports and contradicts the claim, or the claim is "
                     "technically true but misleading / cherry-picked.")
    return "\n".join("- " + l for l in lines)
