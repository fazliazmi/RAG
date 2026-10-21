"""Build components from settings. This is the only place that turns config
into concrete classes; everything else works with the base interfaces."""

from __future__ import annotations

from cyberrag.config import Settings, get_settings
from cyberrag.ingest.normalize_cve import normalize_cve
from cyberrag.providers.embeddings.base import BaseEmbedder
from cyberrag.providers.llm.base import BaseLLM
from cyberrag.providers.registry import build_embedder, build_llm, build_store, load_plugins
from cyberrag.retrieval.retriever import LiveLookup, Retriever
from cyberrag.schemas import Document
from cyberrag.sources.nvd import NVDClient
from cyberrag.store.base import VectorStore


def make_embedder(settings: Settings | None = None) -> BaseEmbedder:
    settings = settings or get_settings()
    load_plugins(settings.plugins)
    return build_embedder(settings.embeddings)


def make_store(settings: Settings | None = None) -> VectorStore:
    settings = settings or get_settings()
    load_plugins(settings.plugins)
    return build_store(settings.vector_store)


def make_llm(settings: Settings | None = None, profile: str | None = None) -> BaseLLM:
    settings = settings or get_settings()
    load_plugins(settings.plugins)
    return build_llm(settings.llm_config(profile))


def make_live_lookup(settings: Settings) -> LiveLookup | None:
    if not settings.sources.nvd.live_lookup:
        return None
    client: NVDClient | None = None

    def lookup(cve_id: str) -> Document | None:
        nonlocal client
        client = client or NVDClient(settings.sources.nvd, api_key=settings.nvd_key)
        cve = client.get_cve(cve_id)
        # Keep rejected records here: "this CVE was rejected" is a useful answer.
        return normalize_cve(cve, skip_rejected=False) if cve else None

    return lookup


def make_retriever(
    settings: Settings | None = None,
    embedder: BaseEmbedder | None = None,
    store: VectorStore | None = None,
) -> Retriever:
    settings = settings or get_settings()
    return Retriever(
        settings,
        embedder or make_embedder(settings),
        store or make_store(settings),
        live_lookup=make_live_lookup(settings),
    )
