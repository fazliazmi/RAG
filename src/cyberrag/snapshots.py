"""Move a built index to another machine without re-embedding.

export: data/snapshots/<time>/{manifest.json, <collection>.snapshot, ...}
import: uploads each snapshot to Qdrant and restores the checkpoints, so
        `cyberrag sync` keeps working on the new machine.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cyberrag import __version__
from cyberrag.config import Settings
from cyberrag.errors import CyberRAGError, EmbedderMismatchError
from cyberrag.ingest.fetch import CWE_FETCH_STATE, NVD_FETCH_STATE, NVD_SYNC_STATE
from cyberrag.ingest.indexer import index_state_name
from cyberrag.ingest.state import load_state, save_state
from cyberrag.providers.embeddings.base import BaseEmbedder
from cyberrag.schemas import SourceKind
from cyberrag.store.base import VectorStore

MANIFEST = "manifest.json"


def export_snapshots(
    settings: Settings,
    embedder: BaseEmbedder,
    store: VectorStore,
    out_dir: Path | None = None,
    kinds: Sequence[SourceKind] = ("cwe", "cve"),
) -> Path:
    names = [
        store.collection_name(kind, embedder.slug)
        for kind in kinds
        if store.exists(store.collection_name(kind, embedder.slug))
    ]
    if not names:
        raise CyberRAGError(f"Nothing indexed for embedder '{embedder.name}' to export")
    out = out_dir or settings.snapshots_dir / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "format": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "cyberrag_version": __version__,
        "embedder": embedder.name,
        "collections": {},
        "state": {},
    }
    for name in names:
        path = store.export_snapshot(name, out)
        manifest["collections"][name] = {
            "file": path.name,
            "points": store.count(name),
            "metadata": store.metadata(name),
        }
        manifest["state"][index_state_name(name)] = load_state(settings, index_state_name(name))
    for state_name in (NVD_FETCH_STATE, NVD_SYNC_STATE, CWE_FETCH_STATE):
        manifest["state"][state_name] = load_state(settings, state_name)
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def latest_snapshot_dir(settings: Settings) -> Path | None:
    dirs = sorted(p.parent for p in settings.snapshots_dir.glob(f"*/{MANIFEST}"))
    return dirs[-1] if dirs else None


def import_snapshots(
    settings: Settings,
    embedder: BaseEmbedder,
    store: VectorStore,
    src_dir: Path | None = None,
    force: bool = False,
) -> dict[str, int]:
    src = src_dir or latest_snapshot_dir(settings)
    if src is None or not (src / MANIFEST).is_file():
        raise CyberRAGError(
            f"No snapshot found. Put the exported folder in {settings.snapshots_dir} "
            "or pass --from <folder>"
        )
    manifest = json.loads((src / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("embedder") != embedder.name:
        raise EmbedderMismatchError(
            f"This snapshot was built with '{manifest.get('embedder')}' but your config uses "
            f"'{embedder.name}'. Set embedding_profile in config/default.yaml to the matching profile."
        )
    for name in manifest["collections"]:
        if store.exists(name) and not force:
            raise CyberRAGError(f"Collection '{name}' already exists. Use --force to replace it.")

    imported: dict[str, int] = {}
    for name, info in manifest["collections"].items():
        store.delete_collection(name)
        store.import_snapshot(name, src / info["file"])
        imported[name] = store.count(name)
    for state_name, data in manifest.get("state", {}).items():
        if data:
            save_state(settings, state_name, data)
    return imported
