from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Literal, Dict, Any, Tuple
from datetime import datetime
from urllib.parse import urlparse
import os
import re

import httpx
import trafilatura
from dotenv import load_dotenv
import tldextract

import nltk
from nltk.tokenize import sent_tokenize

from app.credibility import score_domain_rubric
from app.storage import save_run, list_runs, get_run, export_runs
from app.debate import llm_debate_verdict

 

# Load environment variables early
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "").strip()


def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass

    try:
        nltk.data.find("tokenizers/punkt_tab/english.pickle")
    except Exception:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass


_ensure_nltk()

app = FastAPI(title="Fact Validator API", version="0.8.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





# ----------------------------
# Models
# ----------------------------

class AnalyzeRequest(BaseModel):
    url: Optional[str] = None
    text: Optional[str] = None
    mode: Literal["live", "snapshot"] = "live"
    verifier: Literal["baseline", "debate"] = "baseline"  # debate can fallback

    # optional Step 11 knobs (safe if unused by UI)
    enable_weighted_confidence: bool = False
    min_source_score: int = 50
    require_independent_domains: bool = True
    min_overlap: int = 6
    max_sources_per_base_domain: int = 2

    max_claims: int = 6
    max_evidence_per_claim: int = 5
    max_debate_claims: int = 2


class EvidenceItem(BaseModel):
    url: str
    title: Optional[str] = None
    snippet: str
    domain: str
    domain_score: int
    overlap: int = 0


class ClaimResult(BaseModel):
    claim_text: str
    verdict: Literal["SUPPORTED", "REFUTED", "NEI"]
    confidence: float
    evidence: List[EvidenceItem]
    debate_summary: Optional[str] = None

    # optional Step 11 outputs
    adjusted_verdict: Optional[Literal["SUPPORTED", "REFUTED", "NEI"]] = None
    adjusted_confidence: Optional[float] = None
    evidence_summary: Optional[Dict[str, Any]] = None


class AnalyzeResponse(BaseModel):
    input_type: Literal["url", "text"]
    domain: Optional[str] = None
    extracted_text_chars: int
    extracted_text_preview: str
    domain_score: int
    domain_label: Literal["HIGH", "MEDIUM", "LOW"]
    final_misinformation_likelihood: float
    claims: List[ClaimResult]
    timestamp_utc: str
    metadata: Dict[str, Any]


# ----------------------------
# Helpers
# ----------------------------

def normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    if "://" not in u:
        u = "https://" + u
    return u


def extract_domain(url: str) -> Optional[str]:
    try:
        u = normalize_url(url)
        if not u:
            return None
        parsed = urlparse(u)
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host or None
    except Exception:
        return None


def domain_from_any_url(url: str) -> str:
    return extract_domain(url) or ""


def base_domain(domain: str) -> str:
    ext = tldextract.extract(domain or "")
    if not ext.domain or not ext.suffix:
        return (domain or "").lower()
    return f"{ext.domain}.{ext.suffix}".lower()


def is_blocked_domain(domain: str) -> bool:
    bd = base_domain((domain or "").lower())
    blocked = {
        "facebook.com", "x.com", "twitter.com", "tiktok.com",
        "instagram.com", "reddit.com", "pinterest.com",
        "worldarticledatabase.com", "wecanfigurethisout.org",
    }
    return bd in blocked


async def fetch_html(url: str) -> Tuple[str, str]:
    try:
        headers = {
            "User-Agent": "FactValidatorBot/0.8.2 (thesis demo)",
            "Accept": "text/html,application/xhtml+xml",
        }
        timeout = httpx.Timeout(connect=20.0, read=30.0, write=20.0, pool=30.0)
        async with httpx.AsyncClient(
            headers=headers,
            follow_redirects=True,
            timeout=timeout,
            trust_env=False,
        ) as client:
            r = await client.get(url)
            if r.status_code >= 400:
                return "", ""
            return str(r.url), r.text
    except Exception:
        return "", ""


def extract_readable_text_from_html(html: str, url: str = "") -> str:
    try:
        extracted = trafilatura.extract(
            html,
            url=url or None,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
        )
        return (extracted or "").strip()
    except Exception:
        return ""


def heuristic_claim_score(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.0
    length = len(s)
    length_score = 1.0 - min(abs(length - 160) / 160.0, 1.0)
    has_number = any(ch.isdigit() for ch in s)
    number_bonus = 0.15 if has_number else 0.0
    has_relation = any(tok in s.lower() for tok in [" is ", " are ", " was ", " were ", " has ", " have ", " with "])
    relation_bonus = 0.10 if has_relation else 0.0
    score = 0.55 * length_score + number_bonus + relation_bonus
    return max(0.05, min(score, 0.95))


def clean_text_for_claims(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\bExplore Data\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bResearch\s*&\s*Writing\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
        " ",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def extract_claim_candidates(text: str, max_claims: int = 6) -> List[str]:
    text = clean_text_for_claims(text)
    if not text:
        return []

    blocks = [b.strip() for b in text.split("\n") if b.strip()]
    candidates: List[str] = []
    seen = set()

    boilerplate_markers = [
        "this topic page can be cited as",
        "published online at",
        "cite this work",
        "reuse this work",
        "license terms",
        "creative commons",
        "open access",
        "data produced by third parties",
        "the underlying data for this chart",
    ]

    for b in blocks[:250]:
        for s in sent_tokenize(b):
            s2 = " ".join(s.split()).strip()
            if len(s2) < 60 or len(s2) > 280:
                continue
            low = s2.lower()
            if any(m in low for m in boilerplate_markers):
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

    ranked = sorted(candidates, key=heuristic_claim_score, reverse=True)
    return ranked[:max_claims]


async def serpapi_search(query: str, num: int = 5) -> List[Dict[str, Any]]:
    if not SERPAPI_API_KEY:
        return []

    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": str(num),
        "hl": "en",
        "gl": "us",
        "no_cache": "true",
    }

    try:
        timeout = httpx.Timeout(connect=20.0, read=45.0, write=20.0, pool=45.0)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            r = await client.get("https://serpapi.com/search", params=params)
            r.raise_for_status()
            data = r.json()
            organic = data.get("organic_results", []) or []
            out = []
            for item in organic[:num]:
                out.append(
                    {
                        "title": item.get("title"),
                        "link": item.get("link"),
                        "snippet": item.get("snippet") or "",
                    }
                )
            return out
    except Exception:
        return []


def tokenize_for_overlap(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if len(t) >= 4]
    stop = {
        "this", "that", "with", "from", "were", "have", "has", "been", "into", "also",
        "their", "they", "them", "than", "more", "most", "such", "some", "many"
    }
    toks = [t for t in toks if t not in stop]
    return toks[:80]


def compute_overlap(claim: str, snippet: str) -> int:
    claim_toks = set(tokenize_for_overlap(claim))
    if not claim_toks:
        return 0
    snip_toks = set(tokenize_for_overlap(snippet))
    return len(claim_toks.intersection(snip_toks))


def baseline_verdict(
    claim: str, evidence: List[EvidenceItem]
) -> Tuple[Literal["SUPPORTED", "REFUTED", "NEI"], float, str]:
    if not evidence:
        return "NEI", 0.55, "No evidence retrieved."

    claim_toks = set(tokenize_for_overlap(claim))
    if not claim_toks:
        return "NEI", 0.55, "Claim tokenization empty."

    neg_cues = ["false", "hoax", "debunk", "misleading", "not true", "no evidence", "incorrect"]
    overlaps = []
    for e in evidence:
        ev_toks = set(tokenize_for_overlap(e.snippet))
        overlap = len(claim_toks.intersection(ev_toks))
        overlaps.append((overlap, e.domain_score, base_domain(e.domain), (e.snippet or "").lower()))

    for ov, ds, bd, snip_low in overlaps:
        if ds >= 65 and any(cue in snip_low for cue in neg_cues):
            return "REFUTED", 0.70, "Evidence snippet contains refutation cue from a medium/high source."

    strong_support_domains = {bd for (ov, ds, bd, _) in overlaps if ds >= 65 and ov >= 6}
    if len(strong_support_domains) >= 2:
        return "SUPPORTED", 0.75, "At least two distinct medium/high domains have strong keyword overlap."

    best = max(overlaps, key=lambda x: (x[0], x[1]))
    if best[1] >= 90 and best[0] >= 7:
        return "SUPPORTED", 0.72, "Single high-score domain has strong keyword overlap."

    return "NEI", 0.56, "Evidence retrieved but insufficient strength for a decision (baseline)."


def estimate_misinformation_likelihood(claims: List[ClaimResult]) -> float:
    if not claims:
        return 0.5
    score = 0.5
    for c in claims:
        if c.verdict == "REFUTED":
            score += 0.10
        elif c.verdict == "SUPPORTED":
            score -= 0.07
    return float(max(0.05, min(score, 0.95)))


def _label(score: int) -> str:
    if score >= 80:
        return "HIGH"
    if score >= 50:
        return "MEDIUM"
    return "LOW"


# ----------------------------
# Routes
# ----------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# Step 9: Source credibility endpoint
@app.get("/source/{domain}")
def source_report(domain: str):
    d = (domain or "").strip().lower()
    rep = score_domain_rubric(d)
    return {
        "domain": d,
        "base_domain": base_domain(d) if d else d,
        "score": rep.score,
        "label": rep.label,
        "reasons": rep.reasons,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "disclaimer": "Credibility score is heuristic and should be interpreted as a risk signal, not ground truth.",
    }


@app.get("/runs")
def runs(limit: int = 50):
    return {"items": list_runs(limit=limit)}


@app.get("/runs/{run_id}")
def run_detail(run_id: int):
    r = get_run(run_id)
    if not r:
        return JSONResponse(status_code=404, content={"detail": "Run not found"})
    return r


@app.get("/runs-export")
def runs_export(limit: int = 500):
    items = export_runs(limit=limit)
    return {"items": items, "exported_at_utc": datetime.utcnow().isoformat() + "Z"}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    has_url = bool(req.url and req.url.strip())
    has_text = bool(req.text and req.text.strip())

    input_type: Literal["url", "text"] = "url" if has_url else "text"
    domain = extract_domain(req.url) if has_url else None

    extracted_text = ""
    final_url = None

    if has_url:
        url_norm = normalize_url(req.url or "")
        final_url, html = await fetch_html(url_norm)
        if html:
            extracted_text = extract_readable_text_from_html(html, url=final_url or url_norm)

    if not extracted_text and has_text:
        extracted_text = (req.text or "").strip()

    preview = extracted_text[:400].replace("\n", " ").strip()
    chars = len(extracted_text)

    claim_texts = extract_claim_candidates(extracted_text, max_claims=req.max_claims)

    claims: List[Dict[str, Any]] = []
    debate_meta: Dict[str, Any] = {
        "enabled": req.verifier == "debate",
        "claims_debated": 0,
        "max_debate_claims": int(req.max_debate_claims),
        "items": [],
    }

    for ct in claim_texts:
        raw_results = await serpapi_search(ct[:220], num=req.max_evidence_per_claim)

        ev_items: List[Dict[str, Any]] = []
        seen_base_domains = {}
        for rr in raw_results:
            link = rr.get("link") or ""
            dom = domain_from_any_url(link)
            bd = base_domain(dom)

            if not dom:
                continue
            if is_blocked_domain(dom):
                continue

            # limit per base domain
            seen_base_domains.setdefault(bd, 0)
            if seen_base_domains[bd] >= max(1, int(req.max_sources_per_base_domain)):
                continue
            seen_base_domains[bd] += 1

            snippet = (rr.get("snippet") or "").strip()
            if len(snippet) < 40:
                continue

            cred = score_domain_rubric(dom)
            ov = compute_overlap(ct, snippet)

            ev_items.append(
                {
                    "url": link,
                    "title": rr.get("title"),
                    "snippet": snippet,
                    "domain": dom,
                    "domain_score": int(cred.score),
                    "overlap": int(ov),
                    "base_domain": bd,
                }
            )

        # credibility-first sorting
        ev_items.sort(key=lambda e: (e["domain_score"], e["overlap"]), reverse=True)

        # baseline verdict uses all evidence (always computed as fallback)
        ev_for_baseline = [
            EvidenceItem(**{k: v for k, v in e.items() if k in EvidenceItem.model_fields})
            for e in ev_items
        ]
        verdict, conf, summary = baseline_verdict(ct, ev_for_baseline)

        # Step 11: adjusted verdict (optional)
        adjusted_verdict = None
        adjusted_conf = None
        evidence_summary = None

        if req.enable_weighted_confidence:
            filtered = [
                e for e in ev_items
                if int(e["domain_score"]) >= int(req.min_source_score) and int(e["overlap"]) >= int(req.min_overlap)
            ]
            strong_domains = sorted({e["base_domain"] for e in filtered})
            distinct_domains = len(strong_domains)

            reason = "ok"
            if not filtered:
                reason = "no_evidence_after_filters"
            if req.require_independent_domains and distinct_domains < 2:
                reason = "insufficient_independent_domains"

            # re-run baseline decision on filtered list
            ev_for_adjusted = [EvidenceItem(**{k: v for k, v in e.items() if k in EvidenceItem.model_fields}) for e in filtered]
            av, ac, atext = baseline_verdict(ct, ev_for_adjusted)

            # enforce independence if requested
            if req.require_independent_domains and distinct_domains < 2:
                av = "NEI"
                ac = max(0.50, float(ac))
                atext = atext + " | independence constraint triggered"

            adjusted_verdict = av
            adjusted_conf = round(float(ac), 2)
            evidence_summary = {
                "min_source_score": int(req.min_source_score),
                "min_overlap": int(req.min_overlap),
                "require_independent_domains": bool(req.require_independent_domains),
                "distinct_base_domains": distinct_domains,
                "strong_base_domains": strong_domains,
                "reason": reason,
                "note": atext,
            }

        # Optional LLM debate verifier (PROVER vs SKEPTIC vs JUDGE)
        debate_summary: Optional[str] = summary
        if req.verifier == "debate" and ev_items and debate_meta["claims_debated"] < req.max_debate_claims:
            try:
                d_verdict, d_conf, d_summary, d_debug = await llm_debate_verdict(
                    claim_text=ct,
                    evidence_items=ev_items,
                )
                # If debate returns a sensible verdict, prefer it; otherwise keep baseline.
                verdict = d_verdict or verdict
                conf = float(d_conf) if d_conf is not None else conf
                debate_summary = d_summary or summary

                debate_meta["claims_debated"] += 1
                debate_meta["items"].append(
                    {
                        "claim_text": ct,
                        "verdict": d_verdict,
                        "confidence": d_conf,
                        "summary": d_summary,
                        "debug": d_debug,
                    }
                )
            except Exception as e:
                # If Ollama debate fails, fall back to baseline and record minimal error info.
                debate_meta["items"].append(
                    {
                        "claim_text": ct,
                        "error": str(e),
                        "note": "Debate verifier failed; baseline verdict used.",
                    }
                )
                debate_summary = summary

        claims.append(
            {
                "claim_text": ct,
                "verdict": verdict,
                "confidence": round(float(conf), 2),
                "evidence": [{k: v for k, v in e.items() if k != "base_domain"} for e in ev_items],
                "debate_summary": debate_summary,
                "adjusted_verdict": adjusted_verdict,
                "adjusted_confidence": adjusted_conf,
                "evidence_summary": evidence_summary,
            }
        )

    # input domain score
    if domain:
        input_cred = score_domain_rubric(domain)
        input_domain_score = int(input_cred.score)
        input_domain_label = str(input_cred.label)
        input_domain_reasons = input_cred.reasons
    else:
        input_domain_score = 65
        input_domain_label = "MEDIUM"
        input_domain_reasons = {}

    final_like = estimate_misinformation_likelihood([ClaimResult(**c) for c in claims])

    response_dict: Dict[str, Any] = {
        "input_type": input_type,
        "domain": domain,
        "extracted_text_chars": chars,
        "extracted_text_preview": preview,
        "domain_score": input_domain_score,
        "domain_label": input_domain_label,
        "final_misinformation_likelihood": round(float(final_like), 2),
        "claims": claims,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "metadata": {
            "mode": req.mode,
            "final_url": final_url,
            "extraction_success": bool(extracted_text) and chars > 0,
            "claims_extracted": len(claims),
            "serpapi_enabled": bool(SERPAPI_API_KEY),
            "domain_score_reasons": input_domain_reasons,
            "verifier": req.verifier,
            "debate": debate_meta,
            "note": "Runs are saved to SQLite; list at GET /runs and export at GET /runs-export.",
        },
    }

    run_id = save_run(
        input_type=input_type,
        url=req.url,
        text=req.text,
        domain=domain,
        mode=req.mode,
        verifier=req.verifier,
        response=response_dict,
    )
    response_dict["metadata"]["run_id"] = run_id

    return response_dict
