"""Evaluation metrics (pure Python, no numpy/scipy needed).

Effectiveness : accuracy, macro-F1, per-class precision/recall/F1, confusion matrix
Calibration   : expected calibration error (ECE, 10 equal-width bins)
Efficiency    : latency, LLM calls, tokens (aggregated in run_eval.py)
Significance  : exact McNemar test, paired bootstrap confidence intervals
"""
from __future__ import annotations

import math
import random
from statistics import mean, median
from typing import Dict, List, Sequence


def confusion_matrix(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> Dict[str, Dict[str, int]]:
    cm = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, pred):
        if g in cm and p in cm[g]:
            cm[g][p] += 1
    return cm


def per_class_prf(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for lab in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def accuracy(gold: Sequence[str], pred: Sequence[str]) -> float:
    return sum(1 for g, p in zip(gold, pred) if g == p) / len(gold) if gold else 0.0


def macro_f1(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> float:
    prf = per_class_prf(gold, pred, labels)
    present = [lab for lab in labels if prf[lab]["support"] > 0]
    return mean(prf[lab]["f1"] for lab in present) if present else 0.0


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> float:
    """ECE = sum_b |B_b|/N * |acc(B_b) - conf(B_b)| over equal-width confidence bins."""
    if not confidences:
        return 0.0
    bins: List[List[int]] = [[] for _ in range(n_bins)]
    for i, c in enumerate(confidences):
        b = min(int(float(c) * n_bins), n_bins - 1)
        bins[b].append(i)
    n = len(confidences)
    ece = 0.0
    for idx in bins:
        if not idx:
            continue
        acc = sum(1 for i in idx if correct[i]) / len(idx)
        conf = sum(float(confidences[i]) for i in idx) / len(idx)
        ece += len(idx) / n * abs(acc - conf)
    return ece


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> Dict[str, float]:
    """Exact (binomial) McNemar test on paired predictions.
    b = A right & B wrong, c = A wrong & B right; two-sided p-value."""
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if (not x) and y)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return {"b": b, "c": c, "p_value": min(1.0, 2 * p)}


def bootstrap_ci(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str], n_boot: int = 1000,
                 seed: int = 13, alpha: float = 0.05) -> Dict[str, List[float]]:
    rng = random.Random(seed)
    n = len(gold)
    if n == 0:
        return {"accuracy": [0.0, 0.0], "macro_f1": [0.0, 0.0]}
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [gold[i] for i in idx]
        p = [pred[i] for i in idx]
        accs.append(accuracy(g, p))
        f1s.append(macro_f1(g, p, labels))
    accs.sort()
    f1s.sort()
    lo, hi = int(alpha / 2 * n_boot), int((1 - alpha / 2) * n_boot) - 1
    return {"accuracy": [accs[lo], accs[hi]], "macro_f1": [f1s[lo], f1s[hi]]}


def summarize(records: List[Dict], labels: Sequence[str]) -> Dict:
    """records: prediction rows written by run_eval.py"""
    ok = [r for r in records if not r.get("error")]
    gold = [r["gold"] for r in ok]
    pred = [r["pred"] for r in ok]
    conf = [float(r.get("confidence", 0.5)) for r in ok]
    correct = [g == p for g, p in zip(gold, pred)]
    lat = [float(r.get("seconds", 0.0)) for r in ok]
    calls = [int((r.get("usage") or {}).get("llm_calls", 0)) for r in ok]
    toks = [int((r.get("usage") or {}).get("total_tokens", 0)) for r in ok]
    fallbacks = sum(1 for r in ok if str(r.get("source", "")).startswith("fallback"))
    return {
        "n": len(ok),
        "n_errors": len(records) - len(ok),
        "accuracy": accuracy(gold, pred),
        "macro_f1": macro_f1(gold, pred, labels),
        "per_class": per_class_prf(gold, pred, labels),
        "confusion_matrix": confusion_matrix(gold, pred, labels),
        "ece": expected_calibration_error(conf, correct),
        "ci95": bootstrap_ci(gold, pred, labels),
        "latency_mean_s": mean(lat) if lat else 0.0,
        "latency_median_s": median(lat) if lat else 0.0,
        "llm_calls_per_claim": mean(calls) if calls else 0.0,
        "tokens_per_claim": mean(toks) if toks else 0.0,
        "tokens_total": sum(toks),
        "fallback_rate": fallbacks / len(ok) if ok else 0.0,
        "pred_distribution": {lab: pred.count(lab) for lab in labels},
        "gold_distribution": {lab: gold.count(lab) for lab in labels},
    }
