from __future__ import annotations

import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
from qdrant_client import QdrantClient, models

from cyberrag.config import VectorStoreConfig
from cyberrag.errors import EmbedderMismatchError, IndexMissingError, ProviderError
from cyberrag.providers.registry import register_store
from cyberrag.schemas import Document, Hit, SparseVec
from cyberrag.store.base import Filters, VectorStore

_NAMESPACE = uuid.UUID("6f1c6c1e-2f0c-4bde-9a3e-6a52b9f0c0de")
_DENSE = "dense"
_SPARSE = "sparse"
_INDEX_TYPES = {
    "keyword": models.PayloadSchemaType.KEYWORD,
    "float": models.PayloadSchemaType.FLOAT,
    "integer": models.PayloadSchemaType.INTEGER,
    "bool": models.PayloadSchemaType.BOOL,
    "datetime": models.PayloadSchemaType.DATETIME,
}
# Must match between the collection and the data in it.
_CHECKED_METADATA = ("embedder", "dim", "sparse")


def point_id(doc_id: str) -> str:
    """Stable ID, so re-indexing the same CVE overwrites instead of duplicating."""
    return str(uuid.uuid5(_NAMESPACE, doc_id))


def to_qdrant_filter(filters: Filters | None) -> models.Filter | None:
    if not filters:
        return None
    must: list[models.Condition] = []
    for key, cond in filters.items():
        if cond is None:
            continue
        if isinstance(cond, dict):
            bounds = {k: v for k, v in cond.items() if k in ("gt", "gte", "lt", "lte")}
            if any(isinstance(v, str) for v in bounds.values()):
                must.append(models.FieldCondition(key=key, range=models.DatetimeRange(**bounds)))
            else:
                must.append(models.FieldCondition(key=key, range=models.Range(**bounds)))
        elif isinstance(cond, list | tuple | set):
            must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(cond))))
        else:
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=cond)))
    return models.Filter(must=must) if must else None


def _to_hit(point: Any, via: str = "search") -> Hit:
    payload = dict(point.payload or {})
    return Hit(
        id=payload.pop("doc_id"),
        source=payload.pop("source"),
        text=payload.pop("text", ""),
        payload=payload,
        score=float(getattr(point, "score", 0.0) or 0.0),
        via=via,
    )


@register_store("qdrant")
class QdrantStore(VectorStore):
    def __init__(self, cfg: VectorStoreConfig, client: QdrantClient | None = None) -> None:
        super().__init__(cfg)
        self._api_key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
        self.client = client or QdrantClient(
            url=cfg.url, api_key=self._api_key, timeout=int(cfg.timeout_s)
        )

    # ---- collections -------------------------------------------------------
    def exists(self, name: str) -> bool:
        return self.client.collection_exists(name)

    def _require(self, name: str) -> None:
        if not self.exists(name):
            raise IndexMissingError(f"Collection '{name}' does not exist yet")

    def ensure_collection(
        self, name: str, dim: int, metadata: dict[str, Any], indexes: dict[str, str]
    ) -> None:
        if self.exists(name):
            current = self.metadata(name)
            info = self.client.get_collection(name)
            vectors = info.config.params.vectors
            actual_dim = vectors[_DENSE].size if isinstance(vectors, dict) else None
            if actual_dim is not None and actual_dim != dim:
                raise EmbedderMismatchError(
                    f"Collection '{name}' has {actual_dim}-dim vectors but the embedder makes "
                    f"{dim}-dim ones. Rebuild it with --rebuild."
                )
            for key in _CHECKED_METADATA:
                if key in current and key in metadata and current[key] != metadata[key]:
                    raise EmbedderMismatchError(
                        f"Collection '{name}' was built with {key}={current[key]!r}, "
                        f"current setting is {metadata[key]!r}. Rebuild it with --rebuild."
                    )
            return

        quantization = None
        if self.cfg.quantization == "int8":
            quantization = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
            )
        self.client.create_collection(
            collection_name=name,
            vectors_config={
                _DENSE: models.VectorParams(
                    size=dim, distance=models.Distance.COSINE, on_disk=self.cfg.on_disk
                )
            },
            sparse_vectors_config={_SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            quantization_config=quantization,
            on_disk_payload=self.cfg.on_disk,
            metadata=metadata,
        )
        for field, kind in indexes.items():
            self.client.create_payload_index(name, field, field_schema=_INDEX_TYPES[kind])

    def metadata(self, name: str) -> dict[str, Any]:
        self._require(name)
        return dict(self.client.get_collection(name).config.metadata or {})

    def count(self, name: str) -> int:
        if not self.exists(name):
            return 0
        return self.client.count(name, exact=True).count

    def delete_collection(self, name: str) -> None:
        if self.exists(name):
            self.client.delete_collection(name)

    def list_collections(self) -> list[str]:
        return sorted(c.name for c in self.client.get_collections().collections)

    # ---- data --------------------------------------------------------------
    def upsert(
        self,
        name: str,
        docs: Sequence[Document],
        dense: Sequence[Sequence[float]],
        sparse: Sequence[SparseVec] | None = None,
    ) -> None:
        if len(docs) != len(dense) or (sparse is not None and len(sparse) != len(docs)):
            raise ValueError("docs, dense and sparse must have the same length")
        points = []
        for i, doc in enumerate(docs):
            vector: dict[str, Any] = {_DENSE: list(dense[i])}
            if sparse is not None and sparse[i].indices:
                vector[_SPARSE] = models.SparseVector(
                    indices=sparse[i].indices, values=sparse[i].values
                )
            payload = {**doc.payload, "doc_id": doc.id, "source": doc.source, "text": doc.text}
            points.append(models.PointStruct(id=point_id(doc.id), vector=vector, payload=payload))
        self.client.upsert(name, points=points, wait=True)

    def delete(self, name: str, doc_ids: Sequence[str]) -> None:
        if doc_ids and self.exists(name):
            self.client.delete(
                name, points_selector=models.PointIdsList(points=[point_id(d) for d in doc_ids])
            )

    def search(
        self,
        name: str,
        dense: Sequence[float],
        sparse: SparseVec | None = None,
        *,
        limit: int = 10,
        filters: Filters | None = None,
        prefetch_k: int = 40,
    ) -> list[Hit]:
        self._require(name)
        qfilter = to_qdrant_filter(filters)
        if sparse is not None and sparse.indices:
            res = self.client.query_points(
                name,
                prefetch=[
                    models.Prefetch(
                        query=list(dense), using=_DENSE, limit=prefetch_k, filter=qfilter
                    ),
                    models.Prefetch(
                        query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                        using=_SPARSE,
                        limit=prefetch_k,
                        filter=qfilter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=qfilter,
                limit=limit,
                with_payload=True,
            )
        else:
            res = self.client.query_points(
                name, query=list(dense), using=_DENSE, query_filter=qfilter, limit=limit,
                with_payload=True,
            )
        return [_to_hit(p) for p in res.points]

    def get(self, name: str, doc_ids: Sequence[str]) -> list[Hit]:
        if not doc_ids or not self.exists(name):
            return []
        points = self.client.retrieve(name, ids=[point_id(d) for d in doc_ids], with_payload=True)
        by_id = {p.payload["doc_id"]: p for p in points if p.payload}
        return [_to_hit(by_id[d], via="exact") for d in doc_ids if d in by_id]

    def find(self, name: str, filters: Filters, limit: int = 20) -> list[Hit]:
        if not self.exists(name):
            return []
        points, _ = self.client.scroll(
            name, scroll_filter=to_qdrant_filter(filters), limit=limit, with_payload=True
        )
        return [_to_hit(p, via="exact") for p in points]

    # ---- snapshots ---------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {"api-key": self._api_key} if self._api_key else {}

    def export_snapshot(self, name: str, dest_dir: Path) -> Path:
        self._require(name)
        dest_dir.mkdir(parents=True, exist_ok=True)
        snap = self.client.create_snapshot(name, wait=True)
        if snap is None:
            raise ProviderError(f"Qdrant did not create a snapshot for '{name}'")
        target = dest_dir / f"{name}.snapshot"
        url = f"{self.cfg.url.rstrip('/')}/collections/{name}/snapshots/{snap.name}"
        try:
            with httpx.stream("GET", url, headers=self._headers(), timeout=None) as r:
                r.raise_for_status()
                with target.open("wb") as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        fh.write(chunk)
        finally:
            self.client.delete_snapshot(name, snap.name, wait=True)
        return target

    def import_snapshot(self, name: str, path: Path) -> None:
        url = f"{self.cfg.url.rstrip('/')}/collections/{name}/snapshots/upload"
        with path.open("rb") as fh:
            r = httpx.post(
                url,
                params={"priority": "snapshot", "wait": "true"},
                files={"snapshot": (path.name, fh, "application/octet-stream")},
                headers=self._headers(),
                timeout=None,
            )
        if r.status_code >= 400:
            raise ProviderError(f"Snapshot upload for '{name}' failed: {r.status_code} {r.text}")

    def health(self) -> tuple[bool, str]:
        try:
            names = self.list_collections()
            return True, f"{self.cfg.url} ({len(names)} collections)"
        except Exception as exc:  # noqa: BLE001
            return False, f"cannot reach Qdrant at {self.cfg.url} ({exc}); run: docker compose up -d"
