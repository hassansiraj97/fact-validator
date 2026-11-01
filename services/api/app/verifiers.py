"""
Claim verifiers.

  heuristic        B1  keyword-overlap + source-credibility rules (no LLM)
  llm_no_evidence  B2  single LLM call, parametric knowledge only
  llm_rag          B3  single LLM call with the same evidence the debate sees
  llm_cot_sc       B4  chain-of-thought + self-consistency (3 samples, majority vote) - same
                       number of LLM calls as the debate, i.e. a compute-matched baseline
  debate           proposed: Toulmin PROVER-SKEPTIC-JUDGE debate (+ credibility)
  debate_no_skeptic / debate_freeform / debate_no_cred   ablations

All evidence-based methods share the same evidence preparation and the same
insufficient-evidence fallback, so differences come from the verification step only.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from app import llm
from app.debate import llm_debate_verdict
from app.evidence import (
    CONFLICTING,
    LABELS_3,
    NEI,
    REFUTED,
    SUPPORTED,
    format_evidence_for_llm,
    label_definitions,
    normalize_label,
    rank_for_llm,
)


@dataclass
class VerifierResult:
    verdict: str
    confidence: float
    summary: str
    source: str  # which component produced the verdict
    debug: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# B1: heuristic baseline
# ----------------------------------------------------------------------------

REFUTATION_CUES = ["false", "hoax", "debunk", "misleading", "not true", "no evidence", "incorrect", "fabricated", "fake"]


def heuristic_verdict(
    claim: str,
    evidence: List[Dict[str, Any]],
    labels: Sequence[str] = LABELS_3,
    strong_source: int = 65,
    strong_overlap: int = 6,
    refute_min_overlap: int = 3,
) -> VerifierResult:
    """Rule-based verdict.

    REFUTED   : a credible source (>= strong_source) that is *about the claim* (overlap >= refute_min_overlap)
                contains a refutation cue.
    SUPPORTED : >= 2 independent credible sites with strong overlap, or one very credible site (>= 90) with overlap >= 7.
    CONFLICTING (4-label mode only): both of the above.
    otherwise NEI.
    """
    if not evidence:
        return VerifierResult(NEI, 0.55, "No evidence retrieved.", "heuristic")

    refuting, support_sites, single_high = [], set(), False
    for e in evidence:
        ds, ov = int(e.get("domain_score", 50)), int(e.get("overlap", 0))
        snip = (e.get("snippet") or "").lower()
        if ds >= strong_source and ov >= refute_min_overlap and any(c in snip for c in REFUTATION_CUES):
            refuting.append(e)
            continue
        if ds >= strong_source and ov >= strong_overlap:
            support_sites.add(e.get("base_domain") or e.get("domain"))
        if ds >= 90 and ov >= 7:
            single_high = True

    supported = len(support_sites) >= 2 or single_high
    if refuting and supported:
        if CONFLICTING in labels:
            return VerifierResult(CONFLICTING, 0.60, "Credible sources both support and refute the claim.", "heuristic")
        return VerifierResult(NEI, 0.55, "Credible sources disagree; no decision.", "heuristic")
    if refuting:
        return VerifierResult(REFUTED, 0.70, "A credible, relevant source contains a refutation cue.", "heuristic")
    if len(support_sites) >= 2:
        return VerifierResult(SUPPORTED, 0.75, "At least two independent credible sites strongly overlap with the claim.", "heuristic")
    if single_high:
        return VerifierResult(SUPPORTED, 0.72, "One highly credible site strongly overlaps with the claim.", "heuristic")
    return VerifierResult(NEI, 0.56, "Evidence retrieved but too weak for a decision.", "heuristic")


# ----------------------------------------------------------------------------
# Evidence preparation shared by the LLM methods
# ----------------------------------------------------------------------------

def prepare_evidence(evidence: List[Dict[str, Any]], use_credibility: bool, min_source_score: int = 50) -> List[Dict[str, Any]]:
    """With credibility on, evidence from LOW-credibility sources (< min_source_score) is dropped."""
    if not use_credibility:
        return list(evidence)
    return [e for e in evidence if int(e.get("domain_score", 50)) >= min_source_score]


# ----------------------------------------------------------------------------
# B2-B4: single-LLM baselines
# ----------------------------------------------------------------------------

def _clip(x: Any, default: float = 0.55) -> float:
    try:
        v = float(x)
    except Exception:
        v = default
    return max(0.05, min(v, 0.95))


def _claim_block(claim: str, claim_date: Optional[str]) -> str:
    return f"CLAIM:\n{claim}\n" + (f"(claim made on: {claim_date})\n" if claim_date else "")


SYSTEM = "You are a careful, neutral fact-checker. Be concise."


async def llm_no_evidence(claim: str, labels: Sequence[str] = LABELS_3, claim_date: Optional[str] = None) -> VerifierResult:
    prompt = (
        "Decide the verdict for the CLAIM using only your own knowledge (no evidence is provided).\n\n"
        f"Verdict definitions:\n{label_definitions(labels)}\n\n"
        f'Return a JSON object: {{"verdict": "{"|".join(labels)}", "confidence": <number between 0 and 1>, '
        '"explanation": "<one sentence>"}\n\n' + _claim_block(claim, claim_date)
    )
    raw = await llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                         temperature=0.0, max_tokens=200, json_mode=True, role="single")
    j = llm.safe_json_parse(raw)
    return VerifierResult(normalize_label(j.get("verdict"), labels), _clip(j.get("confidence")),
                          str(j.get("explanation") or ""), "llm_no_evidence", {"raw": raw[:2000]})


def _rag_prompt(claim: str, ev_json: str, labels: Sequence[str], claim_date: Optional[str], cot: bool) -> str:
    keys = ('"reasoning": "<step-by-step reasoning>", ' if cot else "")
    instr = ("Think step by step: first reason about what each evidence item says about the claim, then decide.\n"
             if cot else "")
    return (
        "Decide the verdict for the CLAIM using ONLY the evidence below.\n" + instr + "\n"
        f"Verdict definitions:\n{label_definitions(labels)}\n\n"
        f'Return a JSON object: {{{keys}"verdict": "{"|".join(labels)}", "confidence": <number between 0 and 1>, '
        '"explanation": "<one sentence>"}\n\n'
        + _claim_block(claim, claim_date) + f"\nEVIDENCE (JSON list):\n{ev_json}\n"
    )


async def llm_rag(claim: str, evidence: List[Dict[str, Any]], labels: Sequence[str] = LABELS_3,
                  claim_date: Optional[str] = None, k: int = 3, show_credibility: bool = True) -> VerifierResult:
    chosen = rank_for_llm(evidence, k)
    raw = await llm.chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": _rag_prompt(claim, format_evidence_for_llm(chosen, show_credibility), labels, claim_date, cot=False)}],
        temperature=0.0, max_tokens=250, json_mode=True, role="single",
    )
    j = llm.safe_json_parse(raw)
    return VerifierResult(normalize_label(j.get("verdict"), labels), _clip(j.get("confidence")),
                          str(j.get("explanation") or ""), "llm_rag", {"raw": raw[:2000]})


async def llm_cot_sc(claim: str, evidence: List[Dict[str, Any]], labels: Sequence[str] = LABELS_3,
                     claim_date: Optional[str] = None, k: int = 3, n_samples: int = 3,
                     temperature: float = 0.7, show_credibility: bool = True) -> VerifierResult:
    chosen = rank_for_llm(evidence, k)
    prompt = _rag_prompt(claim, format_evidence_for_llm(chosen, show_credibility), labels, claim_date, cot=True)
    samples = []
    for _ in range(n_samples):
        raw = await llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                             temperature=temperature, max_tokens=500, json_mode=True, role="cot_sample")
        j = llm.safe_json_parse(raw)
        samples.append((normalize_label(j.get("verdict"), labels), _clip(j.get("confidence")), str(j.get("explanation") or "")))

    votes = Counter(v for v, _, _ in samples)
    top = max(votes.values())
    tied = [lab for lab, c in votes.items() if c == top]
    if len(tied) > 1:
        mean_conf = {lab: sum(c for v, c, _ in samples if v == lab) / votes[lab] for lab in tied}
        best = max(mean_conf.values())
        winners = [lab for lab in tied if mean_conf[lab] == best]
        verdict = winners[0] if len(winners) == 1 else NEI
    else:
        verdict = tied[0]
    confidence = _clip(votes.get(verdict, 0) / n_samples)
    summary = next((s for v, _, s in samples if v == verdict), "")
    return VerifierResult(verdict, confidence, summary, "llm_cot_sc",
                          {"votes": dict(votes), "samples": [{"verdict": v, "confidence": c} for v, c, _ in samples]})


async def debate(claim: str, evidence: List[Dict[str, Any]], labels: Sequence[str] = LABELS_3,
                 claim_date: Optional[str] = None, k: int = 3, variant: str = "full",
                 show_credibility: bool = True, consistency_rule: bool = True) -> VerifierResult:
    v, c, s, dbg = await llm_debate_verdict(claim, evidence, labels=labels, variant=variant,
                                             show_credibility=show_credibility, max_evidence=k, claim_date=claim_date,
                                             consistency_rule=consistency_rule)
    return VerifierResult(v, c, s, f"debate_{variant}", dbg)


# ----------------------------------------------------------------------------
# Method registry (used by eval/run_eval.py)
# ----------------------------------------------------------------------------

MethodFn = Callable[..., Awaitable[VerifierResult]]


@dataclass
class MethodSpec:
    name: str
    description: str
    uses_evidence: bool
    uses_credibility: bool
    run: MethodFn


async def _m_heuristic(claim, evidence, labels, claim_date, k):
    return heuristic_verdict(claim, evidence, labels)


async def _m_no_ev(claim, evidence, labels, claim_date, k):
    return await llm_no_evidence(claim, labels, claim_date)


async def _m_rag(claim, evidence, labels, claim_date, k):
    return await llm_rag(claim, evidence, labels, claim_date, k, show_credibility=True)


async def _m_cot_sc(claim, evidence, labels, claim_date, k):
    return await llm_cot_sc(claim, evidence, labels, claim_date, k, show_credibility=True)


async def _m_debate(claim, evidence, labels, claim_date, k):
    return await debate(claim, evidence, labels, claim_date, k, "full", show_credibility=True)


async def _m_debate_no_skeptic(claim, evidence, labels, claim_date, k):
    return await debate(claim, evidence, labels, claim_date, k, "no_skeptic", show_credibility=True)


async def _m_debate_freeform(claim, evidence, labels, claim_date, k):
    return await debate(claim, evidence, labels, claim_date, k, "freeform", show_credibility=True)


async def _m_debate_no_rule(claim, evidence, labels, claim_date, k):
    return await debate(claim, evidence, labels, claim_date, k, "full", show_credibility=True, consistency_rule=False)


async def _m_debate_no_cred(claim, evidence, labels, claim_date, k):
    return await debate(claim, evidence, labels, claim_date, k, "full", show_credibility=False)


METHODS: Dict[str, MethodSpec] = {
    "heuristic": MethodSpec("heuristic", "B1: keyword-overlap + credibility rules", True, True, _m_heuristic),
    "llm_no_evidence": MethodSpec("llm_no_evidence", "B2: single LLM, no evidence", False, False, _m_no_ev),
    "llm_rag": MethodSpec("llm_rag", "B3: single LLM with evidence (RAG)", True, True, _m_rag),
    "llm_cot_sc": MethodSpec("llm_cot_sc", "B4: CoT + self-consistency (3 samples)", True, True, _m_cot_sc),
    "debate": MethodSpec("debate", "Proposed: Toulmin Prover-Skeptic-Judge debate", True, True, _m_debate),
    "debate_no_skeptic": MethodSpec("debate_no_skeptic", "Ablation: no Skeptic", True, True, _m_debate_no_skeptic),
    "debate_freeform": MethodSpec("debate_freeform", "Ablation: free-form (no Toulmin)", True, True, _m_debate_freeform),
    "debate_no_cred": MethodSpec("debate_no_cred", "Ablation: no credibility filtering/scores", True, False, _m_debate_no_cred),
    "debate_no_rule": MethodSpec("debate_no_rule", "Ablation: no rebuttal-only REFUTED rule", True, True, _m_debate_no_rule),
}


async def run_method(name: str, claim: str, evidence: List[Dict[str, Any]], labels: Sequence[str] = LABELS_3,
                     claim_date: Optional[str] = None, k: int = 3, min_source_score: int = 50) -> VerifierResult:
    """Run one method with the shared evidence preparation and insufficient-evidence fallback."""
    spec = METHODS[name]
    ev = prepare_evidence(evidence, spec.uses_credibility, min_source_score) if spec.uses_evidence else []
    if spec.uses_evidence and name != "heuristic" and not ev:
        # Fallback: nothing to reason about -> do not call the LLM, answer NEI
        r = heuristic_verdict(claim, [], labels)
        r.source = "fallback_insufficient_evidence"
        return r
    return await spec.run(claim, ev, labels, claim_date, k)


# ----------------------------------------------------------------------------
# Article-level score used by the API
# ----------------------------------------------------------------------------

def misinformation_likelihood(claims: List[Dict[str, Any]]) -> Optional[float]:
    """Mean probability that a claim is false.

    Per claim: REFUTED -> confidence, SUPPORTED -> 1 - confidence, NEI / CONFLICTING -> 0.5.
    Uses the adjusted (policy) verdict when it is present. None when there are no claims
    (nothing was checked, so no score is shown).
    """
    if not claims:
        return None
    ps = []
    for c in claims:
        v = c.get("adjusted_verdict") or c.get("verdict")
        conf = c.get("adjusted_confidence") if c.get("adjusted_verdict") else c.get("confidence")
        conf = _clip(conf, 0.5)
        if v == REFUTED:
            ps.append(conf)
        elif v == SUPPORTED:
            ps.append(1.0 - conf)
        else:
            ps.append(0.5)
    return round(sum(ps) / len(ps), 4)
