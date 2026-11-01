"""
Build the results tables from the prediction files of a run.

  python -m eval.report eval/results/averitec_gold            # re-generate report.md / results.csv

Outputs in the run folder:
  metrics_<method>.json   all metrics for one method
  results.csv             one row per method (for plotting / LaTeX)
  report.md               paper-style tables: main results, per-class F1, efficiency, significance
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))

from eval.metrics import mcnemar_exact, summarize  # noqa: E402

METHOD_LABELS = {
    "heuristic": "B1 Heuristic (overlap + credibility)",
    "llm_no_evidence": "B2 LLM, no evidence",
    "llm_rag": "B3 LLM + evidence (RAG)",
    "llm_cot_sc": "B4 LLM CoT + self-consistency (n=3)",
    "debate": "Proposed: Toulmin debate",
    "debate_no_skeptic": "  - without Skeptic",
    "debate_freeform": "  - without Toulmin structure",
    "debate_no_cred": "  - without credibility",
    "debate_no_rule": "  - without the rebuttal rule",
}
ORDER = list(METHOD_LABELS)


def _load_preds(path: Path) -> Dict[str, Dict]:
    """Last record per id wins (re-runs overwrite earlier errors)."""
    recs: Dict[str, Dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r["id"] in recs and recs[r["id"]].get("pred") and r.get("error"):
                continue
            recs[r["id"]] = r
    return recs


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def build_report(out_dir: Path, labels: Sequence[str], reference: str = "debate",
                 price_in: Optional[float] = None, price_out: Optional[float] = None) -> str:
    out_dir = Path(out_dir)
    files = sorted(out_dir.glob("predictions_*.jsonl"))
    preds = {f.stem.replace("predictions_", ""): _load_preds(f) for f in files}
    methods = sorted(preds, key=lambda m: ORDER.index(m) if m in ORDER else 99)

    # Paired comparison only on claims every method answered
    common = set.intersection(*[{i for i, r in p.items() if not r.get("error")} for p in preds.values()]) if preds else set()

    summaries = {}
    for m in methods:
        recs = [preds[m][i] for i in sorted(common, key=lambda x: (len(x), x))]
        s = summarize(recs, labels)
        errs = sum(1 for r in preds[m].values() if r.get("error"))
        s["n_errors_total"] = errs
        if price_in is not None and price_out is not None:
            pin = sum(int((r.get("usage") or {}).get("prompt_tokens", 0)) for r in recs)
            pout = sum(int((r.get("usage") or {}).get("completion_tokens", 0)) for r in recs)
            s["cost_usd_total"] = pin / 1e6 * price_in + pout / 1e6 * price_out
            s["cost_usd_per_claim"] = s["cost_usd_total"] / max(1, len(recs))
        summaries[m] = s
        (out_dir / f"metrics_{m}.json").write_text(json.dumps(s, indent=2), encoding="utf-8")

    sig = {}
    if reference in preds:
        ids = sorted(common)
        ref_ok = [preds[reference][i]["pred"] == preds[reference][i]["gold"] for i in ids]
        for m in methods:
            if m == reference:
                continue
            ok = [preds[m][i]["pred"] == preds[m][i]["gold"] for i in ids]
            sig[m] = mcnemar_exact(ref_ok, ok)

    # CSV
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "n", "accuracy", "acc_ci_low", "acc_ci_high", "macro_f1", "f1_ci_low", "f1_ci_high",
                    *[f"f1_{lab}" for lab in labels], "ece", "latency_mean_s", "latency_median_s",
                    "llm_calls_per_claim", "tokens_per_claim", "fallback_rate", f"mcnemar_p_vs_{reference}"])
        for m in methods:
            s = summaries[m]
            w.writerow([m, s["n"], round(s["accuracy"], 4), round(s["ci95"]["accuracy"][0], 4), round(s["ci95"]["accuracy"][1], 4),
                        round(s["macro_f1"], 4), round(s["ci95"]["macro_f1"][0], 4), round(s["ci95"]["macro_f1"][1], 4),
                        *[round(s["per_class"][lab]["f1"], 4) for lab in labels], round(s["ece"], 4),
                        round(s["latency_mean_s"], 3), round(s["latency_median_s"], 3), round(s["llm_calls_per_claim"], 2),
                        round(s["tokens_per_claim"], 1), round(s["fallback_rate"], 4),
                        round(sig[m]["p_value"], 4) if m in sig else ""])

    # Markdown report
    cfg = {}
    if (out_dir / "run_config.json").exists():
        cfg = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
    L: List[str] = []
    L.append(f"# Results: {cfg.get('dataset', '?')} ({cfg.get('evidence', '?')} evidence)\n")
    L.append(f"Model: `{cfg.get('model', '?')}` | claims evaluated (paired): **{len(common)}** | labels: {', '.join(labels)}"
             f" | k = {cfg.get('k', '?')} | credibility rubric v{cfg.get('credibility_rubric_version', '?')}\n")
    if cfg.get("llm_provider") == "mock":
        L.append("> **DRY RUN with the mock LLM - these numbers are NOT real results.**\n")

    L.append("## Table 1 - Effectiveness (accuracy and macro-F1 in %, 95% bootstrap CI)\n")
    L.append("| Method | Accuracy | Macro-F1 | ECE | McNemar p vs " + reference + " |")
    L.append("|---|---|---|---|---|")
    for m in methods:
        s = summaries[m]
        ci_a, ci_f = s["ci95"]["accuracy"], s["ci95"]["macro_f1"]
        p = f"{sig[m]['p_value']:.3f}" if m in sig else "-"
        L.append(f"| {METHOD_LABELS.get(m, m)} | {_pct(s['accuracy'])} [{_pct(ci_a[0])}, {_pct(ci_a[1])}] | "
                 f"{_pct(s['macro_f1'])} [{_pct(ci_f[0])}, {_pct(ci_f[1])}] | {s['ece']:.3f} | {p} |")

    L.append("\n## Table 2 - Per-class F1 (%)\n")
    L.append("| Method | " + " | ".join(labels) + " |")
    L.append("|---|" + "---|" * len(labels))
    for m in methods:
        s = summaries[m]
        L.append(f"| {METHOD_LABELS.get(m, m)} | " + " | ".join(_pct(s["per_class"][lab]["f1"]) for lab in labels) + " |")

    L.append("\n## Table 3 - Efficiency (per claim)\n")
    cost_col = " Cost (USD) |" if price_in is not None and price_out is not None else ""
    L.append("| Method | LLM calls | Tokens | Mean latency (s) | Median latency (s) | Fallback rate |" + cost_col)
    L.append("|---|---|---|---|---|---|" + ("---|" if cost_col else ""))
    for m in methods:
        s = summaries[m]
        cost = f" {s['cost_usd_per_claim']:.5f} |" if cost_col else ""
        L.append(f"| {METHOD_LABELS.get(m, m)} | {s['llm_calls_per_claim']:.2f} | {s['tokens_per_claim']:.0f} | "
                 f"{s['latency_mean_s']:.2f} | {s['latency_median_s']:.2f} | {_pct(s['fallback_rate'])}% |{cost}")

    if reference in summaries:
        L.append(f"\n## Confusion matrix - {METHOD_LABELS.get(reference, reference)} (rows = gold, columns = predicted)\n")
        cm = summaries[reference]["confusion_matrix"]
        L.append("| gold \\ pred | " + " | ".join(labels) + " |")
        L.append("|---|" + "---|" * len(labels))
        for g in labels:
            L.append(f"| {g} | " + " | ".join(str(cm[g][p]) for p in labels) + " |")

    errs = {m: summaries[m]["n_errors_total"] for m in methods if summaries[m]["n_errors_total"]}
    if errs:
        L.append(f"\n> Claims with errors (excluded, re-run the command to retry): {errs}")

    text = "\n".join(L) + "\n"
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    import argparse

    from app.evidence import LABELS_3, LABELS_4

    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--reference", default="debate")
    ap.add_argument("--price-in", type=float, default=None)
    ap.add_argument("--price-out", type=float, default=None)
    a = ap.parse_args()
    cfg_path = a.run_dir / "run_config.json"
    labs = LABELS_4
    if cfg_path.exists():
        labs = tuple(json.loads(cfg_path.read_text(encoding="utf-8")).get("labels", LABELS_4))
    print(build_report(a.run_dir, labs, a.reference, a.price_in, a.price_out))
