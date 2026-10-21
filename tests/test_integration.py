"""End-to-end checks against the real services. Run with:  pytest -m integration
Needs: docker compose up -d, Ollama running, and `cyberrag index cwe` done."""

import pytest

from cyberrag.config import Settings
from cyberrag.factory import make_embedder, make_llm, make_store
from cyberrag.generation.rag import RAGPipeline

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_settings() -> Settings:
    return Settings()


def test_services_are_up(real_settings):
    for component in (make_llm(real_settings), make_embedder(real_settings), make_store(real_settings)):
        ok, detail = component.health()
        assert ok, detail


def test_cwe_question_cites_the_cwe(real_settings):
    store, embedder = make_store(real_settings), make_embedder(real_settings)
    if store.count(store.collection_name("cwe", embedder.slug)) == 0:
        pytest.skip("CWE index is empty; run: cyberrag fetch cwe && cyberrag index cwe")
    answer = RAGPipeline.from_settings(real_settings).ask("How do I prevent CWE-89?")
    assert answer.sources[0].ref == "CWE-89"
    assert "CWE-89" in answer.answer


def test_snapshot_round_trip(real_settings, tmp_path):
    from cyberrag.snapshots import export_snapshots

    store, embedder = make_store(real_settings), make_embedder(real_settings)
    name = store.collection_name("cwe", embedder.slug)
    if store.count(name) == 0:
        pytest.skip("CWE index is empty")
    folder = export_snapshots(real_settings, embedder, store, tmp_path, kinds=("cwe",))
    assert (folder / "manifest.json").is_file()
    assert (folder / f"{name}.snapshot").stat().st_size > 0
