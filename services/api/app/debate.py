"""
Multi-agent LLM debate verifier: PROVER vs SKEPTIC, decided by a JUDGE.

Arguments follow a Toulmin-style structure (datum, warrant, claim). The Skeptic's attack is
typed as rebuttal / undermining / undercutting (Pollock-style attack types).

Variants (used for the ablation study):
  * "full"        PROVER -> SKEPTIC -> JUDGE, structured Toulmin arguments   (proposed method)
  * "no_skeptic"  PROVER -> JUDGE                                            (ablation: no attack)
  * "freeform"    PROVER -> SKEPTIC -> JUDGE, unstructured free-text arguments (ablation: no Toulmin)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from app import config, llm
from app.evidence import (
    LABELS_3,
    NEI,
    REFUTED,
    format_evidence_for_llm,
    label_definitions,
    normalize_label,
    rank_for_llm,
)

VARIANTS = ("full", "no_skeptic", "freeform")

SYSTEM_PROMPT = (
    "You are part of a fact-checking debate system.\n"
    "Use ONLY the provided evidence. Do NOT use outside knowledge and do NOT invent sources.\n"
    "If the evidence is insufficient or ambiguous, say so.\n"
    "Be concise."
)


def _clip_conf(x: Any, default: float = 0.55) -> float:
    try:
        v = float(x)
    except Exception:
        v = default
    return max(0.05, min(v, 0.95))


def _claim_block(claim_text: str, claim_date: Optional[str]) -> str:
    s = f"CLAIM:\n{claim_text}\n"
    if claim_date:
        s += f"(claim made on: {claim_date})\n"
    return s


def _prover_prompt(claim_block: str, evidence_json: str, structured: bool) -> str:
    if structured:
        return (
            "Role: PROVER\n"
            "Build the strongest argument that the CLAIM is TRUE, in Toulmin format, using ONLY the evidence.\n"
            "If no evidence supports the claim, say so in the datum and give a low confidence.\n\n"
            "Return a JSON object with exactly these keys:\n"
            '{"role": "support", "datum": "<evidence fact (cite evidence id)>", '
            '"warrant": "<why this datum supports the claim>", '
            '"claim": "<conclusion about the original claim>", "confidence": <number between 0 and 1>}\n\n'
            f"{claim_block}\nEVIDENCE (JSON list):\n{evidence_json}\n"
        )
    return (
        "Role: PROVER\n"
        "Write a short free-form paragraph (max 120 words) arguing that the CLAIM is TRUE, using ONLY the evidence.\n\n"
        f"{claim_block}\nEVIDENCE (JSON list):\n{evidence_json}\n"
    )


def _skeptic_prompt(claim_block: str, evidence_json: str, prover: str, structured: bool) -> str:
    rules = (
        "Attack the claim through contradiction (rebuttal), weak or unreliable evidence (undermining), "
        "or a broken inference from evidence to claim (undercutting).\n"
        "Only attack points that are necessary to verify the claim; do not attack irrelevant details.\n"
        "Your attack must directly weaken the original claim, not a side issue.\n"
        "Use ONLY the evidence.\n"
    )
    if structured:
        return (
            "Role: SKEPTIC\n"
            "Build the strongest argument AGAINST the CLAIM (or against the Prover's argument) in Toulmin format.\n"
            + rules
            + "\nReturn a JSON object with exactly these keys:\n"
            '{"role": "attack", "attack_type": "rebuttal|undermining|undercutting", '
            '"datum": "<evidence fact or missing evidence (cite evidence id)>", '
            '"warrant": "<why this weakens the claim>", '
            '"claim": "<refuted or not-proven conclusion>", "confidence": <number between 0 and 1>}\n\n'
            f"{claim_block}\nPROVER ARGUMENT:\n{prover}\n\nEVIDENCE (JSON list):\n{evidence_json}\n"
        )
    return (
        "Role: SKEPTIC\n"
        "Write a short free-form paragraph (max 120 words) arguing AGAINST the CLAIM or the Prover's argument.\n"
        + rules
        + f"\n{claim_block}\nPROVER ARGUMENT:\n{prover}\n\nEVIDENCE (JSON list):\n{evidence_json}\n"
    )


def _judge_prompt(claim_block: str, evidence_json: str, prover: str, skeptic: Optional[str],
                  labels: Sequence[str], structured: bool) -> str:
    label_opts = "|".join(labels)
    attack_part = f"ATTACK ARGUMENT:\n{skeptic}\n\n" if skeptic is not None else "ATTACK ARGUMENT:\n(none)\n\n"
    formal = (
        ', "formal_argument": {"support": {"datum": "", "warrant": "", "claim": ""}, '
        '"attack": {"attack_type": "", "datum": "", "warrant": "", "claim": ""}}'
        if structured else ""
    )
    return (
        "Role: JUDGE\n"
        "Decide the verdict for the CLAIM by weighing the support and attack arguments against the evidence.\n"
        "Reject any argument that is not grounded in the evidence, targets an irrelevant detail, "
        "or changes the meaning of the original claim.\n"
        "Choose REFUTED only if a specific evidence item explicitly CONTRADICTS the claim (cite its id in the "
        "summary). If the evidence is silent, off-topic, or about a different person, event or number, the "
        "verdict is NEI: missing evidence is never a refutation.\n\n"
        f"Verdict definitions:\n{label_definitions(labels)}\n\n"
        "Return a JSON object with exactly these keys:\n"
        f'{{"verdict": "{label_opts}", "confidence": <number between 0 and 1>, '
        '"accepted_arguments": ["support" and/or "attack"], "defeated_arguments": ["support" and/or "attack"], '
        f'"summary": "<one or two sentences>"{formal}}}\n\n'
        f"{claim_block}\nSUPPORT ARGUMENT:\n{prover}\n\n{attack_part}EVIDENCE (JSON list):\n{evidence_json}\n"
    )


async def llm_debate_verdict(
    claim_text: str,
    evidence_items: List[Dict[str, Any]],
    *,
    labels: Sequence[str] = LABELS_3,
    variant: str = "full",
    show_credibility: bool = True,
    max_evidence: int = 3,
    claim_date: Optional[str] = None,
    consistency_rule: bool = True,
) -> Tuple[str, float, str, Dict[str, Any]]:
    """Returns (verdict, confidence, summary, debug)."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown debate variant '{variant}'. Use one of {VARIANTS}.")
    structured = variant != "freeform"

    chosen = rank_for_llm(evidence_items, k=max_evidence)
    ev_json = format_evidence_for_llm(chosen, show_credibility=show_credibility)
    cb = _claim_block(claim_text, claim_date)
    sys = {"role": "system", "content": SYSTEM_PROMPT}

    prover = await llm.chat(
        [sys, {"role": "user", "content": _prover_prompt(cb, ev_json, structured)}],
        temperature=0.2, max_tokens=400, json_mode=structured, role="prover",
    )

    skeptic: Optional[str] = None
    if variant != "no_skeptic":
        skeptic = await llm.chat(
            [sys, {"role": "user", "content": _skeptic_prompt(cb, ev_json, prover, structured)}],
            temperature=0.2, max_tokens=400, json_mode=structured, role="skeptic",
        )

    judge_raw = await llm.chat(
        [sys, {"role": "user", "content": _judge_prompt(cb, ev_json, prover, skeptic, labels, structured)}],
        temperature=0.1, max_tokens=550, json_mode=True, role="judge",
    )
    j = llm.safe_json_parse(judge_raw)

    verdict = normalize_label(j.get("verdict", NEI), labels)
    confidence = _clip_conf(j.get("confidence", 0.55))
    summary = str(j.get("summary") or "Insufficient evidence to decide.")

    # Argument-type consistency (structured variants only). In argumentation terms, an *undermining*
    # attack (weak or missing evidence) or an *undercutting* attack (the inference fails) defeats the
    # support argument but does not establish that the claim is false: the result is NEI. Only an
    # accepted *rebuttal* (evidence that contradicts the claim) can justify REFUTED.
    consistency_note = None
    if consistency_rule and structured and verdict == REFUTED:
        sk = llm.safe_json_parse(skeptic or "") if skeptic is not None else {}
        attack_type = str(sk.get("attack_type") or "").strip().lower()
        accepted = [str(a).lower() for a in (j.get("accepted_arguments") or [])]
        if attack_type != "rebuttal" or "attack" not in accepted:
            consistency_note = (f"REFUTED changed to NEI: the accepted attack was '{attack_type or 'none'}', "
                                "not a rebuttal backed by contradicting evidence.")
            verdict = NEI
            confidence = min(confidence, 0.7)
            summary = summary + " [Consistency rule: missing or weak evidence is not a refutation -> NEI.]"

    debug = {
        "provider": config.LLM_PROVIDER,
        "model": llm.model_name(),
        "variant": variant,
        "evidence_shown": [{k: e.get(k) for k in ("domain", "domain_score", "overlap", "url")} for e in chosen],
        "prover_raw": prover[:4000],
        "prover": llm.safe_json_parse(prover) if structured else prover[:4000],
        "skeptic_raw": (skeptic or "")[:4000],
        "skeptic": (llm.safe_json_parse(skeptic or "") if structured else (skeptic or "")[:4000]) if skeptic is not None else None,
        "judge_raw": judge_raw[:4000],
        "judge_json": j,
        "formal_argument": j.get("formal_argument", {}),
        "accepted_arguments": j.get("accepted_arguments", []),
        "defeated_arguments": j.get("defeated_arguments", []),
        "consistency_rule": consistency_note,
    }
    return verdict, confidence, summary, debug
