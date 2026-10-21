"""Find the documents that answer a question.

Order of results (most important first, which matters when the context
budget is tight):
  1. CVEs named in the question (from the index, or fetched live from NVD)
  2. the CWE overview for each of those CVEs
  3. the most relevant sections of CWEs named in the question
  4. hybrid search results, CVEs and CWEs interleaved
  5. the CWE overview for CVEs found by search
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from itertools import chain, zip_longest
from typing import Any

from cyberrag.config import Settings
from cyberrag.errors import CyberRAGError, IndexMissingError
from cyberrag.providers.embeddings.base import BaseEmbedder
from cyberrag.retrieval.query import parse_query
from cyberrag.retrieval.sparse import SparseEncoder
from cyberrag.schemas import Document, Hit
from cyberrag.store.base import Filters, VectorStore

log = logging.getLogger("cyberrag.retrieval")

CVE_INDEXES = {
    "cve_id": "keyword",
    "severity": "keyword",
    "base_score": "float",
    "published": "datetime",
    "year": "integer",
    "cwe_ids": "keyword",
    "vendors": "keyword",
    "products": "keyword",
    "known_exploited": "bool",
}
CWE_INDEXES = {
    "cwe_id": "keyword",
    "section": "keyword",
    "abstraction": "keyword",
    "observed_cves": "keyword",
}

LiveLookup = Callable[[str], Document | None]
MAX_CARRIED = 2  # per kind (CVE, CWE) carried over from the previous exchange


def _dedupe(hits: Iterable[Hit]) -> list[Hit]:
    seen: set[str] = set()
    out = []
    for hit in hits:
        if hit.id not in seen:
            seen.add(hit.id)
            out.append(hit)
    return out


def _interleave(a: list[Hit], b: list[Hit]) -> list[Hit]:
    return [h for pair in zip_longest(a, b) for h in pair if h is not None]


class Retriever:
    def __init__(
        self,
        settings: Settings,
        embedder: BaseEmbedder,
        store: VectorStore,
        sparse: SparseEncoder | None = None,
        live_lookup: LiveLookup | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.sparse = sparse or SparseEncoder()
        self.live_lookup = live_lookup

    @property
    def cve_collection(self) -> str:
        return self.store.collection_name("cve", self.embedder.slug)

    @property
    def cwe_collection(self) -> str:
        return self.store.collection_name("cwe", self.embedder.slug)

    def _cwe_overviews(self, cwe_ids: Iterable[str], via: str) -> list[Hit]:
        ids = list(dict.fromkeys(cwe_ids))
        hits = self.store.get(self.cwe_collection, [f"{c}#overview" for c in ids])
        for h in hits:
            h.via = via  # type: ignore[assignment]
        return hits

    def _named_cves(self, cve_ids: list[str], available: bool) -> list[Hit]:
        found = self.store.get(self.cve_collection, cve_ids) if available else []
        found_ids = {h.id for h in found}
        live: list[Hit] = []
        if self.live_lookup:
            for cve_id in [c for c in cve_ids if c not in found_ids][:3]:
                try:
                    doc = self.live_lookup(cve_id)
                except CyberRAGError as exc:
                    log.warning("Live NVD lookup for %s failed: %s", cve_id, exc)
                    continue
                if doc:
                    live.append(Hit(**doc.model_dump(), score=1.0, via="live"))
        order = {c: i for i, c in enumerate(cve_ids)}
        return sorted(found + live, key=lambda h: order.get(h.id, 99))

    def retrieve(
        self,
        question: str,
        filters: Filters | None = None,
        carry_ids: Sequence[str] = (),
    ) -> list[Hit]:
        """`carry_ids`: CVE/CWE IDs from earlier in a conversation. They are used as if the
        question had named them, but only when it names none itself, so a follow-up such
        as "how do I fix it?" stays about the same CVE."""
        rc = self.settings.retrieval
        pq = parse_query(question)
        if carry_ids and not pq.cve_ids and not pq.cwe_ids:
            pq.cve_ids = [i for i in carry_ids if i.startswith("CVE-")][:MAX_CARRIED]
            pq.cwe_ids = [i for i in carry_ids if i.startswith("CWE-")][:MAX_CARRIED]
        cve_filters: dict[str, Any] = {**pq.filters, **(filters or {})}

        has_cve = self.store.exists(self.cve_collection)
        has_cwe = self.store.exists(self.cwe_collection)
        if not has_cve and not has_cwe:
            raise IndexMissingError(
                f"Nothing is indexed for embedder '{self.embedder.name}'. "
                "Run: cyberrag fetch cwe && cyberrag index cwe   (and the same for cve)"
            )

        named_cves = self._named_cves(pq.cve_ids, has_cve) if pq.cve_ids else []
        named_cve_cwes: list[Hit] = []
        if has_cwe and rc.enrich_cwe and named_cves:
            cwe_ids = chain.from_iterable(h.payload.get("cwe_ids", []) for h in named_cves)
            named_cve_cwes = self._cwe_overviews(cwe_ids, via="enrich")

        dense = self.embedder.embed_query(pq.text)
        sparse = self.sparse.encode_query(pq.text) if rc.hybrid else None

        named_cwes: list[Hit] = []
        if pq.cwe_ids and has_cwe:
            overviews = self._cwe_overviews(pq.cwe_ids, via="exact")
            sections = self.store.search(
                self.cwe_collection, dense, sparse,
                limit=max(3, rc.top_k_cwe), filters={"cwe_id": pq.cwe_ids},
                prefetch_k=rc.prefetch_k,
            )
            for h in sections:
                h.via = "exact"
            named_cwes = overviews + sections

        # Named IDs already cover the core of the question: search much less widely and
        # skip enrichment for incidental hits. Less noise, and a shorter prompt is faster.
        focused = bool(named_cves or named_cwes)
        k_cve = min(2, rc.top_k_cve) if focused else rc.top_k_cve
        k_cwe = min(1, rc.top_k_cwe) if focused else rc.top_k_cwe

        found_cves = (
            self.store.search(
                self.cve_collection, dense, sparse,
                limit=k_cve, filters=cve_filters or None, prefetch_k=rc.prefetch_k,
            )
            if has_cve else []
        )
        found_cwes = (
            self.store.search(
                self.cwe_collection, dense, sparse, limit=k_cwe, prefetch_k=rc.prefetch_k
            )
            if has_cwe else []
        )
        found_cve_cwes: list[Hit] = []
        if has_cwe and rc.enrich_cwe and found_cves and not focused:
            cwe_ids = chain.from_iterable(h.payload.get("cwe_ids", [])[:2] for h in found_cves)
            found_cve_cwes = self._cwe_overviews(cwe_ids, via="enrich")

        return _dedupe(
            chain(
                named_cves,
                named_cve_cwes,
                named_cwes,
                _interleave(found_cves, found_cwes),
                found_cve_cwes,
            )
        )
