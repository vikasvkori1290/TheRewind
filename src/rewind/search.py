"""Keyword search primitives: tokenizer, BM25 ranking and query coverage.

BM25 ranks candidates. Coverage (the share of meaningful query terms found in a
document) decides whether a hit is good enough: unlike raw BM25 scores it does
not depend on corpus size, so one threshold works for small and large archives.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9_]+(?:\.[0-9]+)*")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")

STOPWORDS = frozenset(
    """a an and are as at be but by can could did do does for from had has have how i
    in is it its me my of on or our please should so that the their them then there
    these this those to us was we were what when where which who why will with would
    you your earlier exact exactly again show tell remind previous before said told""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase tokens that keep identifiers (get_user_id) and numbers (1,237 -> 1237, 0.75)."""
    return _TOKEN.findall(_THOUSANDS.sub("", text.lower()))


def query_terms(query: str) -> list[str]:
    """Distinct meaningful terms of a query, in order."""
    seen: dict[str, None] = {}
    for token in tokenize(query):
        if token not in STOPWORDS:
            seen.setdefault(token)
    return list(seen)


def coverage(terms: list[str], doc_tokens: set[str]) -> float:
    if not terms:
        return 0.0
    return sum(t in doc_tokens for t in terms) / len(terms)


class BM25:
    """Okapi BM25 with the always-positive IDF used by Lucene."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avg_len = (sum(self.lengths) / len(docs)) if docs else 0.0
        df: Counter = Counter()
        for counts in self.tf:
            df.update(counts.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, terms: list[str]) -> list[float]:
        out = []
        for counts, length in zip(self.tf, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
            s = 0.0
            for t in terms:
                f = counts.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out
