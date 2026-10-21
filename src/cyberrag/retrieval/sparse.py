"""Dependency-free keyword vectors for hybrid search.

Documents get BM25-style term-frequency weights; Qdrant multiplies them by
IDF at query time (the collection uses Modifier.IDF). This catches exact
terms that dense embeddings blur: product names, versions, function names.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter

from cyberrag.schemas import SparseVec

SPARSE_VERSION = "bm25-crc32-v1"  # stored in collection metadata; bump if tokenizing changes

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._\-/][a-z0-9]+)*")
_SPLIT_RE = re.compile(r"[._\-/]")
_STOPWORDS = frozenset(
    """a an and are as at be by can could do does for from has have how i if in into is it its
    may of on or that the their there these this to was were what when where which who why will
    with would you your via not no""".split()
)


def tokenize(text: str) -> list[str]:
    """'Log4j 2.14.1 CVE-2021-44228' -> ['log4j', '2.14.1', '2', '14', '1', 'cve-2021-44228',
    'cve', '2021', '44228']: compound terms are kept whole and also split into parts."""
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text.lower()):
        if match not in _STOPWORDS:
            tokens.append(match)
        if _SPLIT_RE.search(match):
            tokens.extend(p for p in _SPLIT_RE.split(match) if p and p not in _STOPWORDS)
    return tokens


def _index(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF


class SparseEncoder:
    def __init__(self, k1: float = 1.2, b: float = 0.75, avg_len: float = 120.0) -> None:
        self.k1, self.b, self.avg_len = k1, b, avg_len

    def encode_document(self, text: str) -> SparseVec:
        tokens = tokenize(text)
        if not tokens:
            return SparseVec(indices=[], values=[])
        counts = Counter(_index(t) for t in tokens)
        norm = self.k1 * (1 - self.b + self.b * len(tokens) / self.avg_len)
        indices = sorted(counts)
        values = [counts[i] * (self.k1 + 1) / (counts[i] + norm) for i in indices]
        return SparseVec(indices=indices, values=values)

    def encode_query(self, text: str) -> SparseVec:
        indices = sorted({_index(t) for t in tokenize(text)})
        return SparseVec(indices=indices, values=[1.0] * len(indices))
