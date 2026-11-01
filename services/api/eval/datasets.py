"""
Benchmark loaders. Every loader returns a list of `Example`s whose evidence items use the
same format as the live system (see app/evidence.py), so all methods see identical evidence.

Supported:
  * AVeriTeC (Schlichtkrull et al., NeurIPS 2023) - real-world claims, 4 labels
      evidence = "gold"            annotator question-answer pairs (upper bound / oracle setting)
               = "knowledge_store" BM25 over the official per-claim knowledge store (realistic, reproducible)
               = "search"          live Google search via SerpAPI (cached), like the deployed system
  * FEVER (Thorne et al., NAACL 2018) - Wikipedia claims, 3 labels
      evidence = "gold"   annotated Wikipedia sentences (needs the wiki-pages dump); NEI claims get
                          randomly sampled sentences ("RandomS", Thorne et al. 2018)
               = "search" live Google search via SerpAPI (cached)
  * Generic JSONL: {"id", "claim", "label", "evidence": [str | {"text", "url"}], "claim_date"?}
"""
from __future__ import annotations

import json
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from app.evidence import CONFLICTING, LABELS_3, LABELS_4, NEI, make_evidence_item, normalize_label
from eval.bm25 import BM25

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "data"


@dataclass
class Example:
    id: str
    claim: str
    label: str  # canonical label (SUPPORTED / REFUTED / NEI / CONFLICTING)
    claim_date: Optional[str] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


def _ev(claim: str, text: str, url: str = "", relevance: Optional[float] = None, **extra) -> Dict[str, Any]:
    # use_credibility_cache=False: scores always come from the current rubric (reproducible)
    item = make_evidence_item(claim, url or "", text, title=None, use_credibility_cache=False, **extra)
    if relevance is not None:
        item["relevance"] = relevance
    return item


def stratified_sample(examples: List[Example], n: Optional[int], seed: int = 42) -> List[Example]:
    """Sample n examples keeping the label distribution (deterministic for a given seed)."""
    if not n or n >= len(examples):
        return examples
    rng = random.Random(seed)
    by_label: Dict[str, List[Example]] = {}
    for ex in examples:
        by_label.setdefault(ex.label, []).append(ex)
    out: List[Example] = []
    for lab, items in sorted(by_label.items()):
        k = max(1, round(n * len(items) / len(examples)))
        out.extend(rng.sample(items, min(k, len(items))))
    rng.shuffle(out)
    return out[:n]


# ----------------------------------------------------------------------------
# AVeriTeC
# ----------------------------------------------------------------------------

AVERITEC_LABELS = {
    "Supported": "SUPPORTED",
    "Refuted": "REFUTED",
    "Not Enough Evidence": "NEI",
    "Conflicting Evidence/Cherrypicking": "CONFLICTING",
}


def _averitec_gold_evidence(claim: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
    evid = []
    qs = item.get("questions") or []
    for qi, q in enumerate(qs):
        for a in q.get("answers") or []:
            atype = a.get("answer_type")
            ans = (a.get("answer") or "").strip()
            if atype == "Unanswerable":
                ans = "No answer could be found."
            if a.get("boolean_explanation"):
                ans = f"{ans}. {a['boolean_explanation']}"
            text = f"Q: {q.get('question', '').strip()} A: {ans}"
            # annotator order as relevance: earlier questions first
            evid.append(_ev(claim, text, a.get("source_url") or "", relevance=float(len(qs) - qi)))
    return evid


def _read_ks_file(path: Path) -> List[Dict[str, Any]]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return docs


def _averitec_ks_evidence(claim: str, ks_file: Path, top_k: int = 10) -> List[Dict[str, Any]]:
    """BM25 over all sentences of the claim's knowledge-store documents.
    Each hit is returned with its neighbouring sentences for context."""
    sents, urls, ctx = [], [], []
    for doc in _read_ks_file(ks_file):
        s = doc.get("url2text") or []
        if isinstance(s, str):
            s = re.split(r"(?<=[.!?])\s+", s)
        s = [x.strip() for x in s if isinstance(x, str) and len(x.strip()) > 20]
        for i, sent in enumerate(s):
            sents.append(sent)
            urls.append(doc.get("url", ""))
            ctx.append(" ".join(s[max(0, i - 1): i + 2]))
    if not sents:
        return []
    bm = BM25(sents)
    out, seen = [], set()
    for idx, score in bm.top_k(claim, top_k * 3):
        key = ctx[idx][:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(_ev(claim, ctx[idx][:800], urls[idx], relevance=round(score, 4)))
        if len(out) >= top_k:
            break
    return out


def load_averitec(path: Optional[Path] = None, evidence: str = "gold", knowledge_store_dir: Optional[Path] = None,
                  labels: Sequence[str] = LABELS_4, ks_top_k: int = 10) -> List[Example]:
    path = Path(path or DATA_DIR / "averitec" / "dev.json")
    if not path.exists():
        raise FileNotFoundError(f"AVeriTeC file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if evidence == "knowledge_store":
        knowledge_store_dir = Path(knowledge_store_dir or DATA_DIR / "averitec" / "dev_knowledge_store")
        if not knowledge_store_dir.exists():
            raise FileNotFoundError(
                f"Knowledge store folder not found: {knowledge_store_dir}. Download and unzip the AVeriTeC "
                "dev knowledge store there (see services/api/README.md)."
            )
    out = []
    for i, item in enumerate(data):
        claim = item["claim"].strip()
        gold = AVERITEC_LABELS.get(item.get("label"), normalize_label(item.get("label"), LABELS_4))
        if gold == CONFLICTING and CONFLICTING not in labels:
            gold = NEI
        if evidence == "gold":
            ev = _averitec_gold_evidence(claim, item)
        elif evidence == "knowledge_store":
            f = knowledge_store_dir / f"{i}.json"
            ev = _averitec_ks_evidence(claim, f, ks_top_k) if f.exists() else []
        elif evidence == "search":
            ev = []  # filled at run time
        else:
            raise ValueError("evidence must be gold | knowledge_store | search")
        out.append(Example(
            id=str(i), claim=claim, label=gold, claim_date=item.get("claim_date"), evidence=ev,
            meta={"speaker": item.get("speaker"), "claim_types": item.get("claim_types"),
                  "fact_checking_article": item.get("fact_checking_article")},
        ))
    return out


# ----------------------------------------------------------------------------
# FEVER
# ----------------------------------------------------------------------------

FEVER_LABELS = {"SUPPORTS": "SUPPORTED", "REFUTES": "REFUTED", "NOT ENOUGH INFO": "NEI"}

_FEVER_BRACKETS = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}", "-COLON-": ":"}


def _fever_clean(s: str) -> str:
    """Undo FEVER's PTB-style tokenisation: '-LRB- Campania -RRB- .' -> '(Campania).'"""
    for k, v in _FEVER_BRACKETS.items():
        s = s.replace(k, v)
    s = s.replace("_", " ")
    s = re.sub(r"\s+([.,;:!?%)\]}])", r"\1", s)
    s = re.sub(r"([(\[{])\s+", r"\1", s)
    s = s.replace(" 's", "'s").replace("`` ", '"').replace(" ''", '"')
    return s.strip()


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_wiki_pages(wiki_dir: Path, wanted: set) -> Dict[str, List[str]]:
    """Load only the pages we need from the FEVER wiki-pages dump (wiki-*.jsonl)."""
    pages: Dict[str, List[str]] = {}
    wanted_n = {unicodedata.normalize("NFD", w) for w in wanted}
    for f in sorted(wiki_dir.glob("wiki-*.jsonl")):
        for doc in _read_jsonl(f):
            pid = unicodedata.normalize("NFD", doc.get("id", ""))
            if pid not in wanted_n:
                continue
            lines = []
            for row in (doc.get("lines") or "").split("\n"):
                parts = row.split("\t")
                lines.append(parts[1] if len(parts) > 1 else "")
            pages[pid] = lines
        if len(pages) >= len(wanted_n):
            break
    return pages


def load_fever(path: Path, evidence: str = "gold", wiki_dir: Optional[Path] = None,
               sample: Optional[int] = None, seed: int = 42) -> List[Example]:
    rows = list(_read_jsonl(Path(path)))
    examples = [
        Example(id=str(r.get("id", i)), claim=r["claim"].strip(),
                label=FEVER_LABELS.get(r.get("label"), normalize_label(r.get("label"), LABELS_3)),
                meta={"raw_evidence": r.get("evidence")})
        for i, r in enumerate(rows)
    ]
    examples = stratified_sample(examples, sample, seed)

    if evidence == "search":
        for ex in examples:
            ex.meta.pop("raw_evidence", None)
        return examples
    if evidence != "gold":
        raise ValueError("FEVER evidence must be gold | search")

    # Pre-processed files may already contain evidence text
    if rows and isinstance(rows[0].get("evidence"), list) and rows[0]["evidence"] and isinstance(rows[0]["evidence"][0], str):
        for ex in examples:
            ex.evidence = [_ev(ex.claim, t, "https://en.wikipedia.org", relevance=float(-j))
                           for j, t in enumerate(ex.meta.pop("raw_evidence") or [])]
        return examples

    if not wiki_dir or not Path(wiki_dir).exists():
        raise FileNotFoundError("FEVER gold evidence needs --fever-wiki-dir pointing to the wiki-pages folder.")

    wanted = set()
    for ex in examples:
        for ev_set in ex.meta.get("raw_evidence") or []:
            for ann in ev_set:
                if len(ann) >= 4 and ann[2] is not None:
                    wanted.add(ann[2])
    pages = _load_wiki_pages(Path(wiki_dir), wanted)

    # NEI claims have no annotated evidence. Following the "RandomS" setting of Thorne et al. (2018),
    # they receive randomly sampled Wikipedia sentences, so NEI cannot be predicted from "no evidence" alone.
    rng = random.Random(seed)
    pool = [(p, i, s) for p, sents in pages.items() for i, s in enumerate(sents) if len(s) > 30]

    for ex in examples:
        if ex.label == NEI and pool:
            ex.evidence = [
                _ev(ex.claim, f"{_fever_clean(p)}: {_fever_clean(s)}", f"https://en.wikipedia.org/wiki/{p}", relevance=float(-j))
                for j, (p, i, s) in enumerate(rng.sample(pool, min(2, len(pool))))
            ]
            ex.meta.pop("raw_evidence", None)
            continue
        items, seen = [], set()
        for ev_set in ex.meta.get("raw_evidence") or []:
            for ann in ev_set:
                if len(ann) < 4 or ann[2] is None:
                    continue
                page, line = unicodedata.normalize("NFD", ann[2]), ann[3]
                if (page, line) in seen:
                    continue
                seen.add((page, line))
                sents = pages.get(page) or []
                if 0 <= line < len(sents) and sents[line]:
                    text = f"{_fever_clean(page)}: {_fever_clean(sents[line])}"
                    items.append(_ev(ex.claim, text, f"https://en.wikipedia.org/wiki/{page}", relevance=float(-len(items))))
        ex.evidence = items
        ex.meta.pop("raw_evidence", None)
    return examples


# ----------------------------------------------------------------------------
# Generic JSONL
# ----------------------------------------------------------------------------

def load_jsonl(path: Path, labels: Sequence[str] = LABELS_3) -> List[Example]:
    out = []
    for i, r in enumerate(_read_jsonl(Path(path))):
        ev = []
        for j, e in enumerate(r.get("evidence") or []):
            if isinstance(e, str):
                ev.append(_ev(r["claim"], e, "", relevance=float(-j)))
            elif isinstance(e, dict):
                ev.append(_ev(r["claim"], e.get("text") or e.get("snippet") or "", e.get("url") or "", relevance=float(-j)))
        out.append(Example(id=str(r.get("id", i)), claim=r["claim"], label=normalize_label(r.get("label"), labels),
                           claim_date=r.get("claim_date"), evidence=ev))
    return out
