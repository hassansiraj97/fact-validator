"""
Validate the rule-based credibility rubric against expert news-domain ratings.

Reference: Lin, H., Lasser, J., Lewandowsky, S., et al. (2023). "High level of correspondence
across different news domain quality rating sets." PNAS Nexus 2(9), pgad286.
Their aggregate rating (first principal component, "pc1", 0..1) covers ~11k news domains.

Download their domain-rating CSV (see the paper's data availability statement / GitHub repo),
then run from services/api:

  python -m eval.validate_credibility path/to/domain_pc1.csv --domain-col domain --score-col pc1

Reports: coverage of the curated list, Spearman rank correlation between our score and the
expert rating, and agreement of LOW/MEDIUM/HIGH bands with expert tertiles.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))

from app.credibility import CURATED_SOURCES, compute_score  # noqa: E402


def _rank(xs: List[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: List[float], b: List[float]) -> float:
    ra, rb = _rank(a), _rank(b)
    ma, mb = mean(ra), mean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ratings_csv", type=Path)
    ap.add_argument("--domain-col", default="domain")
    ap.add_argument("--score-col", default="pc1")
    ap.add_argument("--out", type=Path, default=API_DIR / "eval" / "results" / "credibility_validation.json")
    a = ap.parse_args()

    expert: Dict[str, float] = {}
    with open(a.ratings_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                d = row[a.domain_col].strip().lower()
                d = d[4:] if d.startswith("www.") else d
                expert[d] = float(row[a.score_col])
            except (KeyError, ValueError):
                continue
    if not expert:
        raise SystemExit("No ratings read - check --domain-col / --score-col.")

    domains = sorted(expert)
    ours = [compute_score(d).score for d in domains]
    theirs = [expert[d] for d in domains]
    rho_all = spearman(ours, theirs)

    # Only domains where a rule fires (the rest get the neutral prior 50)
    fired = [i for i, d in enumerate(domains) if "neutral" not in compute_score(d).reasons]
    rho_fired = spearman([ours[i] for i in fired], [theirs[i] for i in fired]) if len(fired) > 2 else None

    t = sorted(theirs)
    lo, hi = t[len(t) // 3], t[2 * len(t) // 3]
    band = lambda x: "LOW" if x < lo else ("HIGH" if x >= hi else "MEDIUM")  # noqa: E731
    our_band = [compute_score(d).label for d in domains]
    agree = sum(1 for i in range(len(domains)) if our_band[i] == band(theirs[i])) / len(domains)

    curated_in = [d for d in CURATED_SOURCES if d in expert]
    res = {
        "n_expert_domains": len(expert),
        "spearman_all": round(rho_all, 4),
        "n_domains_with_rule": len(fired),
        "spearman_rule_domains": round(rho_fired, 4) if rho_fired is not None else None,
        "band_agreement_vs_expert_tertiles": round(agree, 4),
        "curated_list_coverage": f"{len(curated_in)}/{len(CURATED_SOURCES)} curated domains are rated by experts",
        "curated_mean_expert_score": round(mean(expert[d] for d in curated_in), 4) if curated_in else None,
        "all_mean_expert_score": round(mean(theirs), 4),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
