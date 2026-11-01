"""Minimal Okapi BM25 (no external dependency) used to retrieve evidence sentences
from the AVeriTeC knowledge store and from FEVER Wikipedia pages."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Sequence, Tuple

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were", "be", "by",
    "with", "that", "this", "it", "as", "at", "from", "has", "have", "had", "not", "but", "its", "his", "her",
}


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs_tok = [tokenize(d) for d in docs]
        self.N = len(self.docs_tok)
        self.avgdl = (sum(len(d) for d in self.docs_tok) / self.N) if self.N else 0.0
        df: Counter = Counter()
        for d in self.docs_tok:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tf = [Counter(d) for d in self.docs_tok]

    def score(self, query: str) -> List[float]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs_tok[i])
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            scores.append(s)
        return scores

    def top_k(self, query: str, k: int) -> List[Tuple[int, float]]:
        sc = self.score(query)
        order = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)
        return [(i, sc[i]) for i in order[:k] if sc[i] > 0]
