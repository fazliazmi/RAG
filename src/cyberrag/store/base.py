"""Vector store interface.

Filters use a small store-neutral format so retrieval code never builds
Qdrant (or any other) query objects directly:

    {"severity": ["HIGH", "CRITICAL"],      # any of
     "known_exploited": True,               # equals
     "base_score": {"gte": 7.0},            # numeric range
     "published": {"gte": "2024-01-01"}}    # date range (ISO strings)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cyberrag.config import VectorStoreConfig
from cyberrag.schemas import Document, Hit, SourceKind, SparseVec

Filters = dict[str, Any]


class VectorStore(ABC):
    def __init__(self, cfg: VectorStoreConfig) -> None:
        self.cfg = cfg

    def collection_name(self, kind: SourceKind, embedder_slug: str) -> str:
        """One collection per data source and embedding model, e.g. 'cve__nomic-embed-text'.
        Changing the embedder therefore never mixes incompatible vectors."""
        return f"{self.cfg.collection_prefix}{kind}__{embedder_slug}"

    @abstractmethod
    def exists(self, name: str) -> bool: ...

    @abstractmethod
    def ensure_collection(
        self, name: str, dim: int, metadata: dict[str, Any], indexes: dict[str, str]
    ) -> None:
        """Create the collection if missing; if it exists, verify it matches `metadata`.
        `indexes` maps payload field -> keyword | float | integer | bool | datetime."""

    @abstractmethod
    def metadata(self, name: str) -> dict[str, Any]: ...

    @abstractmethod
    def upsert(
        self,
        name: str,
        docs: Sequence[Document],
        dense: Sequence[Sequence[float]],
        sparse: Sequence[SparseVec] | None = None,
    ) -> None: ...

    @abstractmethod
    def delete(self, name: str, doc_ids: Sequence[str]) -> None: ...

    @abstractmethod
    def search(
        self,
        name: str,
        dense: Sequence[float],
        sparse: SparseVec | None = None,
        *,
        limit: int = 10,
        filters: Filters | None = None,
        prefetch_k: int = 40,
    ) -> list[Hit]: ...

    @abstractmethod
    def get(self, name: str, doc_ids: Sequence[str]) -> list[Hit]: ...

    @abstractmethod
    def find(self, name: str, filters: Filters, limit: int = 20) -> list[Hit]:
        """Payload-only lookup, no vectors involved."""

    @abstractmethod
    def count(self, name: str) -> int: ...

    @abstractmethod
    def delete_collection(self, name: str) -> None: ...

    @abstractmethod
    def list_collections(self) -> list[str]: ...

    @abstractmethod
    def export_snapshot(self, name: str, dest_dir: Path) -> Path: ...

    @abstractmethod
    def import_snapshot(self, name: str, path: Path) -> None: ...

    @abstractmethod
    def health(self) -> tuple[bool, str]: ...
