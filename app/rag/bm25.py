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

    def score(
        self, query_tokens: list[str], doc_index: int, weights: dict[str, float] | None = None
    ) -> float:
        freqs = self.doc_freqs[doc_index]
        dl = self.doc_lens[doc_index]
        total = 0.0
        for term in set(query_tokens):
            tf = freqs.get(term, 0)
            if not tf:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            w = weights.get(term, 1.0) if weights else 1.0
            total += w * self.idf[term] * tf * (self.k1 + 1) / denom
        return total

    def search(
        self, query_tokens: list[str], top_n: int, weights: dict[str, float] | None = None
    ) -> list[tuple[int, float]]:
        scored = [(i, self.score(query_tokens, i, weights)) for i in range(self.n_docs)]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda s: (-s[1], s[0]))
        return scored[:top_n]

    def coverage(
        self,
        query_tokens: list[str],
        doc_index: int,
        alternatives: dict[str, set[str]] | None = None,
        unseen: set[str] | None = None,
        alternative_weight: float = 1.0,
    ) -> float:
        """IDF-weighted share of query terms that appear in the given chunk.

        A query term matched only through one of its ``alternatives`` (soft
        matches: corpus definitions, shared prefix) earns ``alternative_weight``
        of its IDF instead of the full value.
        """
        terms = set(query_tokens)
        if not terms:
            return 0.0
        freqs = self.doc_freqs[doc_index]
        alternatives = alternatives or {}
        unseen = unseen or set()

        def idf(t: str) -> float:
            # ``unseen`` terms exist only in chunks excluded from this search.
            return self.unseen_idf if t in unseen else self.term_idf(t)

        total = sum(idf(t) for t in terms)
        hit = 0.0
        for t in terms:
            if freqs.get(t):
                hit += idf(t)
            elif any(freqs.get(a) for a in alternatives.get(t, ())):
                hit += alternative_weight * idf(t)
        return hit / total if total else 0.0
