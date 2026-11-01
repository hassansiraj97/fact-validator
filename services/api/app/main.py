"""
Fact Validator API.

Pipeline for POST /analyze:
  1. text extraction from a URL (trafilatura) or pasted text
  2. claim detection (heuristic check-worthiness ranking)
  3. evidence retrieval (SerpAPI, cached) + independence filtering
  4. source-credibility scoring (rule-based rubric)
  5. verdict: heuristic baseline, optionally the Toulmin LLM debate, with fallback
  6. article-level misinformation likelihood; the run is stored in SQLite
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app import config, llm
from app.credibility import score_domain_rubric
from app.evidence import LABELS_3, filter_search_results
from app.retrieval import extract_readable_text_from_html, fetch_html, serpapi_search
from app.storage import export_runs, export_runs_csv, get_run, list_runs, save_run, utc_now_iso
from app.text_utils import base_domain, extract_claim_candidates, extract_domain, normalize_url
from app.verifiers import heuristic_verdict, misinformation_likelihood, prepare_evidence, debate as run_debate

APP_VERSION = "1.0.0"


# ----------------------------------------------------------------------------
# Debug logging (off by default; FACT_VALIDATOR_DEBUG_LOG=1 to enable)
# ----------------------------------------------------------------------------

def debug_log(event: str, **data: Any) -> None:
    if not config.DEBUG_LOG:
        return
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"ts": utc_now_iso(), "event": event, **data}, ensure_ascii=False, default=str)
        with open(config.LOG_DIR / "debug.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _ensure_nltk() -> None:
    try:
        import nltk

        for res, pkg in (("tokenizers/punkt", "punkt"), ("tokenizers/punkt_tab", "punkt_tab")):
            try:
                nltk.data.find(res)
            except LookupError:
                nltk.download(pkg, quiet=True)
    except Exception:
        pass  # text_utils falls back to a regex sentence splitter


_ensure_nltk()

app = FastAPI(title="Fact Validator API", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    # any port on this computer (Next.js moves to 3001, 3002... when 3000 is busy)
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    url: Optional[str] = None
    text: Optional[str] = None
    mode: Literal["live", "snapshot"] = "live"  # kept for backward compatibility; only "live" is implemented
    verifier: Literal["baseline", "debate"] = "baseline"

    # Evidence-policy knobs ("adjusted verdict")
    enable_weighted_confidence: bool = False
    min_source_score: int = Field(50, ge=0, le=100)
    require_independent_domains: bool = True
    min_overlap: int = Field(6, ge=0, le=50)
    max_sources_per_base_domain: int = Field(2, ge=1, le=10)

    max_claims: int = Field(6, ge=1, le=12)
    max_evidence_per_claim: int = Field(5, ge=1, le=10)
    max_debate_claims: int = Field(6, ge=0, le=12)
    debate_evidence_k: int = Field(3, ge=1, le=10)


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "serpapi_configured": bool(config.SERPAPI_API_KEY),
        "llm_provider": config.LLM_PROVIDER,
        "llm_configured": config.LLM_PROVIDER == "mock" or bool(config.GROQ_API_KEY),
        "model": llm.model_name(),
    }


@app.get("/source/{domain}")
def source_report(domain: str):
    d = (domain or "").strip().lower()
    if d.startswith("www."):
        d = d[4:]
    if not d:
        raise HTTPException(status_code=400, detail="domain is required")
    rep = score_domain_rubric(d)
    return {
        "domain": d,
        "base_domain": base_domain(d),
        "score": rep.score,
        "label": rep.label,
        "reasons": rep.reasons,
        "timestamp_utc": utc_now_iso(),
        "disclaimer": "Credibility score is heuristic and should be interpreted as a risk signal, not ground truth.",
    }


@app.get("/runs")
def runs(limit: int = 50):
    return {"items": list_runs(limit=limit)}


@app.get("/runs/{run_id}")
def run_detail(run_id: int):
    r = get_run(run_id)
    if not r:
        raise HTTPException(status_code=404, detail="Run not found")
    return r


@app.get("/runs-export")
def runs_export(limit: int = 500, format: Literal["json", "csv"] = "json"):
    if format == "csv":
        return PlainTextResponse(
            export_runs_csv(limit=limit),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="fact_validator_runs.csv"'},
        )
    return {"items": export_runs(limit=limit), "exported_at_utc": utc_now_iso()}


def _adjusted_verdict(claim: str, ev_items: List[Dict[str, Any]], req: AnalyzeRequest) -> Dict[str, Any]:
    """Stricter evidence policy: keep only credible + relevant evidence, optionally require
    two independent sites, then re-run the heuristic decision."""
    filtered = [e for e in ev_items if e["domain_score"] >= req.min_source_score and e["overlap"] >= req.min_overlap]
    strong = sorted({e["base_domain"] for e in filtered})
    res = heuristic_verdict(claim, filtered, LABELS_3)
    reason = "ok"
    if not filtered:
        reason = "no_evidence_after_filters"
    elif req.require_independent_domains and len(strong) < 2:
        reason = "insufficient_independent_domains"
    verdict, conf, note = res.verdict, res.confidence, res.summary
    if req.require_independent_domains and len(strong) < 2:
        verdict, conf, note = "NEI", max(0.50, conf), note + " | independence constraint triggered"
    return {
        "adjusted_verdict": verdict,
        "adjusted_confidence": round(float(conf), 2),
        "evidence_summary": {
            "min_source_score": req.min_source_score,
            "min_overlap": req.min_overlap,
            "require_independent_domains": req.require_independent_domains,
            "distinct_base_domains": len(strong),
            "strong_base_domains": strong,
            "reason": reason,
            "note": note,
        },
    }


async def _process_claim(idx: int, claim: str, input_domain: Optional[str], req: AnalyzeRequest,
                         sem: asyncio.Semaphore) -> Dict[str, Any]:
    async with sem:
        t0 = time.perf_counter()
        raw = await serpapi_search(claim[:220], num=req.max_evidence_per_claim + 3)
        kept, excluded = filter_search_results(
            claim, raw, input_domain=input_domain, max_per_base_domain=req.max_sources_per_base_domain
        )
        kept = kept[: req.max_evidence_per_claim]
        kept.sort(key=lambda e: (e["domain_score"], e["overlap"]), reverse=True)

        base = heuristic_verdict(claim, kept, LABELS_3)
        verdict, conf, summary, source = base.verdict, base.confidence, base.summary, "heuristic"
        debate_info: Optional[Dict[str, Any]] = None
        usage: Dict[str, Any] = {}

        if req.verifier == "debate":
            usable = prepare_evidence(kept, use_credibility=True, min_source_score=req.min_source_score)
            if idx >= req.max_debate_claims:
                source = "heuristic (debate limit reached)"
            elif not usable:
                # Fallback 1: insufficient evidence -> no debate, heuristic verdict kept (NEI)
                source = "heuristic (fallback: insufficient evidence)"
            else:
                try:
                    with llm.track_usage() as tracker:
                        r = await run_debate(claim, usable, LABELS_3, None, req.debate_evidence_k, "full", True)
                    usage = tracker.as_dict()
                    verdict, conf, summary, source = r.verdict, r.confidence, r.summary, "debate"
                    d = r.debug
                    debate_info = {
                        "prover": d.get("prover"),
                        "skeptic": d.get("skeptic"),
                        "judge": d.get("judge_json"),
                        "formal_argument": d.get("formal_argument"),
                        "accepted_arguments": d.get("accepted_arguments"),
                        "defeated_arguments": d.get("defeated_arguments"),
                        "evidence_shown": d.get("evidence_shown"),
                        "consistency_rule": d.get("consistency_rule"),
                        "model": d.get("model"),
                    }
                except Exception as e:
                    # Fallback 2: model failure -> heuristic verdict
                    source = "heuristic (fallback: debate failed)"
                    debate_info = {"error": f"{type(e).__name__}: {e}"}
                    debug_log("debate_error", claim_index=idx, error=str(e))

        out: Dict[str, Any] = {
            "claim_text": claim,
            "verdict": verdict,
            "confidence": round(float(conf), 2),
            "verdict_source": source,
            "debate_summary": summary,
            "baseline_verdict": base.verdict,
            "baseline_confidence": round(float(base.confidence), 2),
            "evidence": [{k: v for k, v in e.items() if k != "base_domain"} for e in kept],
            "excluded_evidence": [
                {"domain": e["domain"], "url": e["url"], "reason": e["excluded_reason"]} for e in excluded
            ],
            "debate": debate_info,
            "adjusted_verdict": None,
            "adjusted_confidence": None,
            "evidence_summary": None,
            "usage": usage,
            "seconds": round(time.perf_counter() - t0, 3),
        }
        if req.enable_weighted_confidence:
            out.update(_adjusted_verdict(claim, kept, req))
        debug_log("claim_done", claim_index=idx, verdict=verdict, source=source, n_evidence=len(kept),
                  n_excluded=len(excluded))
        return out


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    t_start = time.perf_counter()
    has_url = bool(req.url and req.url.strip())
    has_text = bool(req.text and req.text.strip())
    if not has_url and not has_text:
        raise HTTPException(status_code=400, detail="Provide a URL or some text to analyse.")

    input_type: Literal["url", "text"] = "url" if has_url else "text"
    domain = extract_domain(req.url) if has_url else None

    extracted_text, final_url, fetch_error = "", None, None
    if has_url:
        url_norm = normalize_url(req.url)
        final_url, html, fetch_error = await fetch_html(url_norm)
        if html:
            extracted_text = extract_readable_text_from_html(html, url=final_url or url_norm)
            if not extracted_text:
                fetch_error = "Page downloaded but no readable article text could be extracted."
        if final_url:
            domain = extract_domain(final_url) or domain

    if not extracted_text and has_text:
        extracted_text = (req.text or "").strip()

    if not extracted_text:
        raise HTTPException(status_code=422, detail=f"Could not get text from the URL ({fetch_error or 'unknown error'}).")

    claim_texts = extract_claim_candidates(extracted_text, max_claims=req.max_claims)
    debug_log("claims_extracted", n=len(claim_texts), verifier=req.verifier, has_url=has_url)

    sem = asyncio.Semaphore(3)  # bounded parallelism (SerpAPI / Groq rate limits)
    claims = await asyncio.gather(*[_process_claim(i, c, domain, req, sem) for i, c in enumerate(claim_texts)])

    if domain:
        cred = score_domain_rubric(domain)
        input_score, input_label, input_reasons = int(cred.score), cred.label, cred.reasons
    else:
        input_score, input_label, input_reasons = None, None, {}

    total_usage = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for c in claims:
        for k in total_usage:
            total_usage[k] += int((c.get("usage") or {}).get(k, 0))

    response: Dict[str, Any] = {
        "input_type": input_type,
        "domain": domain,
        "extracted_text_chars": len(extracted_text),
        "extracted_text_preview": extracted_text[:400].replace("\n", " ").strip(),
        "domain_score": input_score,
        "domain_label": input_label,
        "final_misinformation_likelihood": misinformation_likelihood(claims),
        "claims": claims,
        "timestamp_utc": utc_now_iso(),
        "metadata": {
            "version": APP_VERSION,
            "mode": req.mode,
            "final_url": final_url,
            "fetch_error": fetch_error,
            "claims_extracted": len(claims),
            "serpapi_enabled": bool(config.SERPAPI_API_KEY),
            "domain_score_reasons": input_reasons,
            "verifier": req.verifier,
            "llm_model": llm.model_name() if req.verifier == "debate" else None,
            "llm_usage": total_usage,
            "seconds_total": round(time.perf_counter() - t_start, 3),
            "misinformation_likelihood_formula": "mean over claims of P(false): REFUTED=conf, SUPPORTED=1-conf, NEI=0.5",
            "warning": None if claims else "No checkable claim was found in the input.",
        },
    }

    run_id = save_run(input_type=input_type, url=req.url, text=req.text, domain=domain, mode=req.mode,
                      verifier=req.verifier, response=response)
    response["metadata"]["run_id"] = run_id
    return response
