"""
Run the experiments: every selected method on the same claims with the same evidence.

Examples (run from services/api):
  # check the whole pipeline offline with a fake LLM (no API key, no cost)
  python -m eval.run_eval --dataset averitec --evidence gold --methods all --limit 20 --dry-run

  # main experiment: AVeriTeC dev, gold evidence, all baselines + debate + ablations
  python -m eval.run_eval --dataset averitec --evidence gold --methods all

  # realistic retrieval setting (needs the AVeriTeC dev knowledge store)
  python -m eval.run_eval --dataset averitec --evidence knowledge_store --methods heuristic,llm_rag,llm_cot_sc,debate

  # FEVER, 1000-claim stratified sample, gold evidence (needs FEVER dev + wiki-pages)
  python -m eval.run_eval --dataset fever --data-path eval/data/fever/shared_task_dev.jsonl \
      --fever-wiki-dir eval/data/fever/wiki-pages --limit 1000 --methods all

Runs are resumable: predictions are appended to <out>/predictions_<method>.jsonl and
already-finished claims are skipped when the command is run again.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fact Validator evaluation runner")
    p.add_argument("--dataset", choices=["averitec", "fever", "jsonl"], default="averitec")
    p.add_argument("--data-path", type=Path, default=None, help="dataset file (default: eval/data/averitec/dev.json)")
    p.add_argument("--evidence", choices=["gold", "knowledge_store", "search"], default="gold")
    p.add_argument("--knowledge-store-dir", type=Path, default=None)
    p.add_argument("--fever-wiki-dir", type=Path, default=None)
    p.add_argument("--labels", type=int, choices=[3, 4], default=None,
                   help="label set size (default: 4 for AVeriTeC, 3 otherwise)")
    p.add_argument("--methods", default="all",
                   help="comma list, 'all', 'baselines' or 'ablations' (see app/verifiers.py METHODS)")
    p.add_argument("--limit", type=int, default=None, help="stratified sample of N claims")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=5, help="max evidence items shown to the LLM")
    p.add_argument("--min-source-score", type=int, default=50, help="credibility filter threshold")
    p.add_argument("--reference", default="debate", help="method compared against the others (McNemar)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--concurrency", type=int, default=1, help="claims processed in parallel (mind rate limits)")
    p.add_argument("--save-debug", action="store_true", help="store full LLM transcripts in the predictions")
    p.add_argument("--dry-run", action="store_true", help="use the offline mock LLM (pipeline check, no cost)")
    p.add_argument("--price-in", type=float, default=None, help="USD per 1M prompt tokens (for cost estimate)")
    p.add_argument("--price-out", type=float, default=None, help="USD per 1M completion tokens")
    return p.parse_args(argv)


GROUPS = {
    "baselines": ["heuristic", "llm_no_evidence", "llm_rag", "llm_cot_sc"],
    "ablations": ["debate_no_skeptic", "debate_freeform", "debate_no_cred", "debate_no_rule"],
}

# Fact-checking sites are excluded in "search" mode: they would reveal the gold verdict
FACTCHECK_DOMAINS = {
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org", "checkyourfact.com", "leadstories.com",
    "africacheck.org", "factcheck.afp.com", "boomlive.in", "altnews.in", "vishvasnews.com",
    "newschecker.in", "factly.in", "healthfeedback.org", "sciencefeedback.co", "climatefeedback.org",
    "truthorfiction.com", "misbar.com", "maldita.es", "newtral.es", "pagellapolitica.it", "facta.news",
    "polygraph.info", "factcheck.thequint.com", "factchecker.in",
}


def _resolve_methods(spec: str) -> List[str]:
    from app.verifiers import METHODS

    if spec == "all":
        return list(METHODS)
    names: List[str] = []
    for part in [s.strip() for s in spec.split(",") if s.strip()]:
        names.extend(GROUPS.get(part, [part]))
    unknown = [n for n in names if n not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown method(s): {unknown}. Available: {list(METHODS)}")
    return list(dict.fromkeys(names))


async def _search_evidence(ex, k: int) -> List[Dict[str, Any]]:
    from app.evidence import filter_search_results
    from app.retrieval import serpapi_search
    from app.text_utils import domain_matches, extract_domain

    raw = await serpapi_search(ex.claim[:220], num=k + 5)
    fc_dom = extract_domain((ex.meta or {}).get("fact_checking_article") or "")
    raw = [
        r for r in raw
        if not any(domain_matches(extract_domain(r.get("link")) or "", d) for d in FACTCHECK_DOMAINS)
        and not (fc_dom and domain_matches(extract_domain(r.get("link")) or "", fc_dom))
    ]
    kept, _ = filter_search_results(ex.claim, raw)
    return kept[:k + 3]


def _load(args, labels):
    from eval import datasets as ds

    if args.dataset == "averitec":
        exs = ds.load_averitec(args.data_path, evidence=args.evidence, knowledge_store_dir=args.knowledge_store_dir,
                               labels=labels)
        return ds.stratified_sample(exs, args.limit, args.seed)
    if args.dataset == "fever":
        if not args.data_path:
            raise SystemExit("--data-path is required for FEVER (e.g. eval/data/fever/shared_task_dev.jsonl)")
        if args.evidence == "knowledge_store":
            raise SystemExit("FEVER supports --evidence gold or search")
        return ds.load_fever(args.data_path, evidence=args.evidence, wiki_dir=args.fever_wiki_dir,
                             sample=args.limit, seed=args.seed)
    if not args.data_path:
        raise SystemExit("--data-path is required for --dataset jsonl")
    return ds.stratified_sample(ds.load_jsonl(args.data_path, labels), args.limit, args.seed)


def _done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if not r.get("error"):
                    ids.add(r["id"])
            except Exception:
                continue
    return ids


async def _run(args) -> Path:
    if args.dry_run:
        os.environ["LLM_PROVIDER"] = "mock"
    from app import config

    if args.dry_run:
        config.LLM_PROVIDER = "mock"

    from app import llm
    from app.credibility import RUBRIC_VERSION
    from app.evidence import LABELS_3, LABELS_4
    from app.verifiers import run_method
    from eval.report import build_report

    n_labels = args.labels or (4 if args.dataset == "averitec" else 3)
    labels = LABELS_4 if n_labels == 4 else LABELS_3
    methods = _resolve_methods(args.methods)

    if config.LLM_PROVIDER != "mock" and any(m != "heuristic" for m in methods) and not config.GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY is not set. Add it to services/api/.env, or use --dry-run.")
    if args.evidence == "search" and not config.SERPAPI_API_KEY:
        raise SystemExit("SERPAPI_API_KEY is not set (needed for --evidence search).")

    out = args.out or (API_DIR / "eval" / "results" / f"{args.dataset}_{args.evidence}{'_dryrun' if args.dry_run else ''}")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} ({args.evidence} evidence)...", flush=True)
    examples = _load(args, labels)
    print(f"{len(examples)} claims | labels={list(labels)} | methods={methods} | model={llm.model_name()}", flush=True)

    # Evidence snapshot: identical evidence for every method, saved for the thesis appendix
    ev_path = out / "evidence.jsonl"
    cached_ev: Dict[str, List[Dict[str, Any]]] = {}
    if ev_path.exists():
        for line in ev_path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                cached_ev[r["id"]] = r["evidence"]
            except Exception:
                pass
    with open(ev_path, "a", encoding="utf-8") as f:
        for ex in examples:
            if ex.id in cached_ev:
                ex.evidence = cached_ev[ex.id]
                continue
            if args.evidence == "search":
                ex.evidence = await _search_evidence(ex, args.k)
            f.write(json.dumps({"id": ex.id, "claim": ex.claim, "gold": ex.label, "evidence": ex.evidence},
                               ensure_ascii=False) + "\n")

    run_cfg = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset, "data_path": str(args.data_path or "default"), "evidence": args.evidence,
        "labels": list(labels), "n_claims": len(examples), "methods": methods, "k": args.k,
        "min_source_score": args.min_source_score, "seed": args.seed, "limit": args.limit,
        "llm_provider": config.LLM_PROVIDER, "model": llm.model_name(), "credibility_rubric_version": RUBRIC_VERSION,
    }
    (out / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    sem = asyncio.Semaphore(max(1, args.concurrency))

    for method in methods:
        pred_path = out / f"predictions_{method}.jsonl"
        done = _done_ids(pred_path)
        todo = [ex for ex in examples if ex.id not in done]
        print(f"\n== {method}: {len(done)} done, {len(todo)} to go", flush=True)
        lock = asyncio.Lock()
        counter = {"n": 0}

        async def one(ex, method=method, pred_path=pred_path, lock=lock, counter=counter, total=len(todo)):
            async with sem:
                rec: Dict[str, Any] = {"id": ex.id, "claim": ex.claim, "gold": ex.label, "method": method}
                t0 = time.perf_counter()
                try:
                    with llm.track_usage() as usage:
                        r = await run_method(method, ex.claim, ex.evidence, labels, ex.claim_date, args.k,
                                             args.min_source_score)
                    rec.update(pred=r.verdict, confidence=r.confidence, source=r.source, summary=r.summary[:500],
                               usage=usage.as_dict())
                    if args.save_debug:
                        rec["debug"] = r.debug
                except Exception as e:
                    rec.update(error=f"{type(e).__name__}: {e}")
                rec["seconds"] = round(time.perf_counter() - t0, 3)
                rec["n_evidence"] = len(ex.evidence)
                async with lock:
                    with open(pred_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    counter["n"] += 1
                    if counter["n"] % 10 == 0 or counter["n"] == total:
                        print(f"  {method}: {counter['n']}/{total}", flush=True)

        await asyncio.gather(*[one(ex) for ex in todo])

    report = build_report(out, labels, reference=args.reference, price_in=args.price_in, price_out=args.price_out)
    print("\n" + report)
    print(f"\nResults written to {out}")
    return out


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
