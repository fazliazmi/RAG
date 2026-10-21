import pytest

from cyberrag.errors import EmbedderMismatchError, IndexMissingError
from cyberrag.generation.rag import RAGPipeline
from cyberrag.ingest.indexer import Indexer
from cyberrag.ingest.normalize_cve import normalize_cve
from cyberrag.retrieval.retriever import Retriever
from tests.conftest import FakeEmbedder, FakeLLM, cve_fixture


@pytest.fixture
def indexed(raw_data, embedder, store):
    indexer = Indexer(raw_data, embedder, store, upsert_batch=8)
    cwe_stats = indexer.index_cwe()
    cve_stats = indexer.index_cve()
    return indexer, cwe_stats, cve_stats


def test_indexing_counts_and_resume(indexed, store):
    indexer, cwe_stats, cve_stats = indexed
    assert cwe_stats.docs > 10
    assert cve_stats.docs == 3  # the rejected CVE is not indexed
    assert store.count(indexer.collection("cve")) == 3
    # second run: nothing to do
    assert indexer.index_cve().docs == 0
    assert "up to date" in indexer.index_cwe().message


def test_index_limit_and_rebuild(raw_data, embedder, store):
    indexer = Indexer(raw_data, embedder, store)
    assert indexer.index_cve(limit=2).docs == 2
    # the partly indexed file is not marked done, so the next run finishes it
    assert indexer.index_cve().docs == 3
    assert indexer.index_cve(rebuild=True).docs == 3


def test_collection_records_embedder(indexed, store):
    indexer, *_ = indexed
    meta = store.metadata(indexer.collection("cve"))
    assert meta["embedder"] == "fake:fake-embed" and meta["dim"] == 128
    with pytest.raises(EmbedderMismatchError):
        Indexer(indexer.settings, FakeEmbedder(dim=64), store).ensure("cve")


def test_named_cve_comes_first(indexed, raw_data, embedder, store):
    hits = Retriever(raw_data, embedder, store).retrieve("What is CVE-2024-3094?")
    assert hits[0].id == "CVE-2024-3094" and hits[0].via == "exact"


def test_cve_is_enriched_with_its_cwe(indexed, raw_data, embedder, store):
    hits = Retriever(raw_data, embedder, store).retrieve("Explain CVE-2099-0001")
    assert [h.id for h in hits[:2]] == ["CVE-2099-0001", "CWE-89#overview"]
    assert hits[1].via == "enrich"


def test_named_cwe_sections(indexed, raw_data, embedder, store):
    hits = Retriever(raw_data, embedder, store).retrieve("How do I prevent CWE-89?")
    assert hits[0].id == "CWE-89#overview"
    exact = [h for h in hits if h.via == "exact"]
    assert all(h.payload["cwe_id"] == "CWE-89" for h in exact)
    assert any("mitigations" in h.id for h in exact)


def test_filters_reach_the_store(indexed, raw_data, embedder, store):
    retriever = Retriever(raw_data, embedder, store)
    hits = retriever.retrieve("remote code execution", filters={"known_exploited": True})
    assert [h.id for h in hits if h.source == "cve"] == ["CVE-2021-44228"]


def test_live_lookup_for_unindexed_cve(indexed, raw_data, embedder, store):
    calls = []

    def live(cve_id):
        calls.append(cve_id)
        doc = normalize_cve(cve_fixture("cve-2024-3094.json"))
        return doc.model_copy(update={"id": cve_id})

    hits = Retriever(raw_data, embedder, store, live_lookup=live).retrieve("What is CVE-2030-1111?")
    assert calls == ["CVE-2030-1111"]
    assert hits[0].id == "CVE-2030-1111" and hits[0].via == "live"


def test_no_index_gives_clear_error(settings, embedder, store):
    with pytest.raises(IndexMissingError, match="cyberrag fetch cwe"):
        Retriever(settings, embedder, store).retrieve("anything")


def test_pipeline_answer_has_sources_and_prompt(indexed, raw_data, embedder, store):
    llm = FakeLLM()
    pipeline = RAGPipeline(raw_data, llm, Retriever(raw_data, embedder, store))
    answer = pipeline.ask("What is CVE-2024-3094?", include_context=True)
    assert answer.answer == "It is a backdoor in xz [CVE-2024-3094]."
    assert answer.sources[0].ref == "CVE-2024-3094"
    assert answer.sources[0].url.endswith("CVE-2024-3094")
    messages = llm.calls[0]
    assert messages[0].role == "system" and "CyberRAG" in messages[0].content
    assert "[CVE-2024-3094] (CVE record)" in messages[-1].content
    assert "Question: What is CVE-2024-3094?" in messages[-1].content


def test_bigger_context_window_gets_more_context(indexed, raw_data, embedder, store):
    retriever = Retriever(raw_data, embedder, store)
    small = RAGPipeline(raw_data, FakeLLM(num_ctx=1800), retriever)
    large = RAGPipeline(raw_data, FakeLLM(num_ctx=32000), retriever)
    q = "How do I prevent CWE-89 SQL injection?"
    assert len(large.prepare(q).hits) > len(small.prepare(q).hits)


def test_changed_raw_file_is_indexed_again(indexed, raw_data, embedder, store):
    from cyberrag.ingest.fetch import cve_raw_files
    from cyberrag.ingest.state import read_json_gz, write_json_gz

    indexer, *_ = indexed
    path = cve_raw_files(raw_data)[0]
    data = read_json_gz(path)
    data["vulnerabilities"] = data["vulnerabilities"][:1]  # file content changes
    write_json_gz(path, data)
    assert indexer.index_cve().docs == 1


def test_cited_sources_are_flagged_and_listed_first(indexed, raw_data, embedder, store):
    llm = FakeLLM(reply="Parameterized queries stop it [CWE-89].")
    pipeline = RAGPipeline(raw_data, llm, Retriever(raw_data, embedder, store))
    answer = pipeline.ask("Explain CVE-2099-0001")
    assert answer.sources[0].ref == "CWE-89" and answer.sources[0].cited
    assert not any(s.cited for s in answer.sources[1:])


def test_follow_up_question_keeps_the_previous_ids(indexed, raw_data, embedder, store):
    from cyberrag.schemas import Message

    llm = FakeLLM(reply="Use parameterized queries [CWE-89].")
    pipeline = RAGPipeline(raw_data, llm, Retriever(raw_data, embedder, store))
    history = [
        Message(role="user", content="Explain CVE-2099-0001"),
        Message(role="assistant", content="It is SQL injection in ExampleShop [CVE-2099-0001] [CWE-89]."),
    ]
    prep = pipeline.prepare("How do I fix it?", history=history)
    ids = [h.id for h in prep.hits]
    assert ids[0] == "CVE-2099-0001"
    assert "CWE-89#overview" in ids
    # a question with its own ID ignores the history
    own = pipeline.prepare("What is CVE-2024-3094?", history=history)
    assert own.hits[0].id == "CVE-2024-3094"


def test_ids_not_in_the_context_are_flagged(indexed, raw_data, embedder, store):
    from cyberrag.generation.rag import unsupported_ids

    llm = FakeLLM(reply="See [CVE-2099-0001], [CWE-89] and also CVE-1999-9999.")
    pipeline = RAGPipeline(raw_data, llm, Retriever(raw_data, embedder, store))
    answer = pipeline.ask("Explain CVE-2099-0001")
    assert answer.unsupported == ["CVE-1999-9999"]
    assert unsupported_ids("CWE-079 is XSS", "context mentions CWE-79") == []
