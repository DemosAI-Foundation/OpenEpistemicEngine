"""Pure-Python Okapi BM25 lexical channel (no sklearn, no external services)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .textutil import tokenize

K1 = 1.5
B = 0.75


class LexicalIndex:
    def __init__(self):
        self._doc_ids: list[str] = []
        self._df: Counter[str] = Counter()
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        self._lengths: dict[int, int] = {}
        self._index_of: dict[str, int] = {}
        self._avg = 0.0
        self._dirty = False

    def __len__(self) -> int:
        return len(self._doc_ids)

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._index_of:
            return
        idx = len(self._doc_ids)
        self._doc_ids.append(doc_id)
        self._index_of[doc_id] = idx
        terms = tokenize(text)
        counts = Counter(terms)
        for term, n in counts.items():
            self._postings[term][idx] = n
            self._df[term] += 1
        self._lengths[idx] = max(1, len(terms))
        self._avg = sum(self._lengths.values()) / len(self._lengths)
        self._dirty = True

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if not self._doc_ids:
            return []
        q_terms = tokenize(query)
        if not q_terms:
            return []
        n = len(self._doc_ids)
        scores: Counter[int] = Counter()
        seen: set[str] = set()
        for term in q_terms:
            if term in seen:
                continue
            seen.add(term)
            postings = self._postings.get(term)
            if not postings:
                continue
            df = self._df[term]
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for idx, tf in postings.items():
                dl = self._lengths[idx]
                denom = tf + K1 * (1.0 - B + B * dl / max(self._avg, 1.0))
                scores[idx] += idf * (tf * (K1 + 1.0)) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [(self._doc_ids[i], float(s)) for i, s in ranked]

    def explain(self, query: str, doc_id: str) -> dict:
        terms = tokenize(query)
        idx = self._index_of.get(doc_id)
        if idx is None:
            return {"doc_id": doc_id, "matched": [], "score": 0.0}
        matched = []
        total = 0.0
        for term in dict.fromkeys(terms):
            tf = self._postings.get(term, {}).get(idx)
            if tf:
                df = self._df[term]
                idf = math.log(1.0 + (len(self._doc_ids) - df + 0.5) / (df + 0.5))
                contrib = idf * (tf * (K1 + 1.0)) / (
                    tf + K1 * (1.0 - B + B * self._lengths[idx] / max(self._avg, 1.0))
                )
                total += contrib
                matched.append({"term": term, "tf": tf, "idf": round(idf, 4),
                                "contrib": round(contrib, 4)})
        return {"doc_id": doc_id, "matched": matched[:12], "score": round(total, 4)}
