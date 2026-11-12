"""Dependency-free Okapi BM25 index."""

from __future__ import annotations

import math
from collections import Counter


class BM25Index:
    def __init__(self, docs_tokens: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[Counter[str]] = [Counter(toks) for toks in docs_tokens]
        self.doc_lens = [len(toks) for toks in docs_tokens]
        self.n_docs = len(docs_tokens)
        self.avgdl = (sum(self.doc_lens) / self.n_docs) if self.n_docs else 0.0
        df: Counter[str] = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())
        self.idf = {
            term: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5)) for term, n in df.items()
        }
        # IDF assigned to terms that never occur in the corpus (used for coverage).
        self.unseen_idf = math.log(1 + (self.n_docs + 0.5) / 0.5) if self.n_docs else 1.0

    def term_idf(self, term: str) -> float:
        return self.idf.get(term, self.unseen_idf)

    def score(self, query_tokens: list[str], doc_index: int) -> float:
        freqs = self.doc_freqs[doc_index]
        dl = self.doc_lens[doc_index]
        total = 0.0
        for term in set(query_tokens):
            tf = freqs.get(term, 0)
            if not tf:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            total += self.idf[term] * tf * (self.k1 + 1) / denom
        return total

    def search(self, query_tokens: list[str], top_n: int) -> list[tuple[int, float]]:
        scored = [(i, self.score(query_tokens, i)) for i in range(self.n_docs)]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda s: (-s[1], s[0]))
        return scored[:top_n]

    def coverage(self, query_tokens: list[str], doc_index: int) -> float:
        """IDF-weighted share of query terms that appear in the given chunk."""
        terms = set(query_tokens)
        if not terms:
            return 0.0
        total = sum(self.term_idf(t) for t in terms)
        hit = sum(self.term_idf(t) for t in terms if self.doc_freqs[doc_index].get(t))
        return hit / total if total else 0.0
