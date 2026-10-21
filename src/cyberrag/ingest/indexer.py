"""Embed raw data and write it to the vector store, with checkpoints so a long
run (the full NVD on a CPU can take hours) can be stopped and resumed."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from cyberrag.config import Settings
from cyberrag.errors import CyberRAGError
from cyberrag.ingest.fetch import cve_raw_files, latest_cwe_file
from cyberrag.ingest.normalize_cve import is_rejected, normalize_cve
from cyberrag.ingest.normalize_cwe import normalize_cwe
from cyberrag.ingest.state import delete_state, load_state, read_json_gz, save_state
from cyberrag.providers.embeddings.base import BaseEmbedder
from cyberrag.retrieval.retriever import CVE_INDEXES, CWE_INDEXES
from cyberrag.retrieval.sparse import SPARSE_VERSION, SparseEncoder
from cyberrag.schemas import Document, SourceKind
from cyberrag.store.base import VectorStore

T = TypeVar("T")
SCHEMA_VERSION = 1
# Typical document length in tokens, for BM25 length normalisation.
_AVG_LEN = {"cve": 150.0, "cwe": 260.0}


def batched(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def index_state_name(collection: str) -> str:
    return f"index__{collection}"


@dataclass
class IndexStats:
    collection: str
    docs: int = 0
    deleted: int = 0
    files_done: int = 0
    files_pending: int = 0
    seconds: float = 0.0
    message: str = ""

    @property
    def rate(self) -> float:
        return self.docs / self.seconds if self.seconds > 0 else 0.0


class Indexer:
    def __init__(
        self,
        settings: Settings,
        embedder: BaseEmbedder,
        store: VectorStore,
        upsert_batch: int = 256,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.upsert_batch = upsert_batch

    def collection(self, kind: SourceKind) -> str:
        return self.store.collection_name(kind, self.embedder.slug)

    def metadata(self, kind: SourceKind) -> dict[str, Any]:
        return {
            "kind": kind,
            "embedder": self.embedder.name,
            "dim": self.embedder.dim,
            "doc_prefix": self.embedder.cfg.doc_prefix,
            "sparse": SPARSE_VERSION,
            "schema": SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
        }

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.settings.raw_dir).as_posix()

    def ensure(self, kind: SourceKind) -> str:
        name = self.collection(kind)
        indexes = CVE_INDEXES if kind == "cve" else CWE_INDEXES
        self.store.ensure_collection(name, self.embedder.dim, self.metadata(kind), indexes)
        return name

    def _write(self, name: str, docs: Sequence[Document], encoder: SparseEncoder) -> None:
        dense = self.embedder.embed_documents([d.text for d in docs])
        sparse = [encoder.encode_document(d.text) for d in docs]
        self.store.upsert(name, docs, dense, sparse)

    def _write_all(
        self,
        name: str,
        docs: Sequence[Document],
        encoder: SparseEncoder,
        stats: IndexStats,
        progress: Callable[[int], None] | None,
    ) -> None:
        for chunk in batched(docs, self.upsert_batch):
            self._write(name, chunk, encoder)
            stats.docs += len(chunk)
            if progress:
                progress(len(chunk))

    # ---- CWE ---------------------------------------------------------------
    def cwe_documents(self) -> list[Document]:
        path = latest_cwe_file(self.settings)
        if path is None:
            raise CyberRAGError("No CWE data downloaded yet. Run: cyberrag fetch cwe")
        weaknesses: Iterable[dict] = read_json_gz(path).get("Weaknesses", [])
        return [doc for w in weaknesses for doc in normalize_cwe(w)]

    def index_cwe(
        self,
        rebuild: bool = False,
        on_start: Callable[[int], None] | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> IndexStats:
        path = latest_cwe_file(self.settings)
        if path is None:
            raise CyberRAGError("No CWE data downloaded yet. Run: cyberrag fetch cwe")
        rel = path.relative_to(self.settings.raw_dir).as_posix()
        name = self.collection("cwe")
        state_name = index_state_name(name)
        state = load_state(self.settings, state_name)
        stats = IndexStats(collection=name)

        up_to_date = state.get("file") == rel and self.store.count(name) > 0
        if up_to_date and not rebuild:
            stats.message = f"{name} is already up to date ({rel})"
            return stats
        # A new CWE release can drop or rename sections: start clean.
        self.store.delete_collection(name)
        delete_state(self.settings, state_name)
        self.ensure("cwe")

        docs = self.cwe_documents()
        if on_start:
            on_start(len(docs))
        started = time.perf_counter()
        self._write_all(name, docs, SparseEncoder(avg_len=_AVG_LEN["cwe"]), stats, progress)
        stats.seconds = time.perf_counter() - started
        stats.files_done = 1
        save_state(self.settings, state_name, {
            "file": rel, "docs": stats.docs,
            "updated_at": datetime.now(UTC).isoformat(),
        })
        stats.message = f"Indexed {stats.docs} CWE sections into {name}"
        return stats

    # ---- CVE ---------------------------------------------------------------
    def index_cve(
        self,
        limit: int | None = None,
        rebuild: bool = False,
        on_start: Callable[[int], None] | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> IndexStats:
        files = cve_raw_files(self.settings)
        name = self.collection("cve")
        state_name = index_state_name(name)
        stats = IndexStats(collection=name)
        if rebuild:
            self.store.delete_collection(name)
            delete_state(self.settings, state_name)
        if not files:
            if self.store.count(name) > 0:
                stats.message = f"{name} has data but no new raw CVE files to index"
                return stats
            raise CyberRAGError("No CVE data downloaded yet. Run: cyberrag fetch cve")

        self.ensure("cve")
        state = load_state(self.settings, state_name)
        if state.get("done") and self.store.count(name) == 0:
            state = {}  # the collection was wiped (e.g. new Docker volume): start over
        # done maps file -> size when indexed; a file that changed size is indexed again
        done: dict[str, int] = dict(state.get("done", {}))
        pending = [f for f in files if done.get(self._rel(f)) != f.stat().st_size]
        stats.files_pending = len(pending)
        if not pending:
            stats.message = f"{name} is up to date ({len(done)} files indexed)"
            return stats
        if on_start:
            page = self.settings.sources.nvd.page_size
            estimate = len(pending) * page
            on_start(min(estimate, limit) if limit else estimate)

        encoder = SparseEncoder(avg_len=_AVG_LEN["cve"])
        skip_rejected = self.settings.sources.nvd.skip_rejected
        started = time.perf_counter()
        for path in pending:
            vulns = read_json_gz(path).get("vulnerabilities") or []
            docs: list[Document] = []
            rejected: list[str] = []
            for item in vulns:
                cve = item.get("cve") or {}
                doc = normalize_cve(cve, skip_rejected=skip_rejected)
                if doc:
                    docs.append(doc)
                elif cve.get("id") and is_rejected(cve):
                    rejected.append(cve["id"])

            partial = False
            if limit is not None:
                room = limit - stats.docs
                if len(docs) > room:
                    docs, partial = docs[:room], True

            self._write_all(name, docs, encoder, stats, progress)
            if rejected and not partial:
                self.store.delete(name, rejected)
                stats.deleted += len(rejected)
            if partial:
                break
            done[self._rel(path)] = path.stat().st_size
            stats.files_done += 1
            save_state(self.settings, state_name, {
                "done": done,
                "updated_at": datetime.now(UTC).isoformat(),
            })
            if limit is not None and stats.docs >= limit:
                break
        stats.seconds = time.perf_counter() - started
        stats.message = (
            f"Indexed {stats.docs} CVEs into {name} ({stats.rate:.1f} docs/s); "
            f"{stats.files_pending - stats.files_done} raw files still pending"
        )
        return stats
