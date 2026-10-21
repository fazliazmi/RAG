import json

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cyberrag.api.app import create_app
from cyberrag.cli import app as cli_app
from cyberrag.errors import ConfigError
from cyberrag.generation.rag import RAGPipeline
from cyberrag.ingest.indexer import Indexer
from cyberrag.retrieval.retriever import Retriever
from tests.conftest import FakeLLM


@pytest.fixture
def client(raw_data, embedder, store):
    Indexer(raw_data, embedder, store).index_cwe()
    Indexer(raw_data, embedder, store).index_cve()

    def factory(profile):
        if profile == "bad":
            raise ConfigError("LLM profile 'bad' not found")
        return RAGPipeline(raw_data, FakeLLM(), Retriever(raw_data, embedder, store))

    return TestClient(create_app(raw_data, factory))


def test_query(client):
    r = client.post("/query", json={"question": "What is CVE-2024-3094?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("It is a backdoor")
    assert body["sources"][0]["ref"] == "CVE-2024-3094"


def test_query_stream(client):
    r = client.post("/query", json={"question": "What is CVE-2024-3094?", "stream": True})
    assert r.status_code == 200
    events = [line for line in r.text.splitlines() if line.startswith("event:")]
    assert events[0] == "event: sources" and events[-1] == "event: done"
    tokens = [json.loads(line[6:]) for line in r.text.splitlines()
              if line.startswith("data:") and line[6:].startswith('"')]
    assert "".join(tokens) == "It is a backdoor in xz [CVE-2024-3094]."


def test_bad_profile_is_400(client):
    r = client.post("/query", json={"question": "hi there", "llm_profile": "bad"})
    assert r.status_code == 400 and "not found" in r.json()["detail"]


def test_lookup_endpoints(client):
    assert client.get("/cve/cve-2024-3094").json()["id"] == "CVE-2024-3094"
    assert client.get("/cve/CVE-1999-0001").status_code == 404
    sections = [h["payload"]["section"] for h in client.get("/cwe/89").json()]
    assert sections[0] == "overview" and "mitigations" in sections
    assert client.get("/cwe/not-an-id").status_code == 400


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["indexed_documents"]["cve"] == 3


def test_cli_profiles_and_help(settings):
    runner = CliRunner()
    result = runner.invoke(cli_app, ["profiles"])
    assert result.exit_code == 0, result.output
    assert "llama3.2-3b" in result.output and "default" in result.output
    assert runner.invoke(cli_app, ["--help"]).exit_code == 0


def test_cli_reports_errors_without_traceback(settings):
    result = CliRunner().invoke(cli_app, ["ask", "hello", "--llm", "missing-profile"])
    assert result.exit_code == 1
    assert "Error:" in result.output and "Traceback" not in result.output


def test_stream_done_event_lists_cited_ids(client):
    r = client.post("/query", json={"question": "What is CVE-2024-3094?", "stream": True})
    done = r.text.split("event: done\ndata: ")[1].strip()
    assert json.loads(done) == {"cited": ["CVE-2024-3094"], "unsupported": []}


def test_api_key_is_required_when_configured(raw_data, embedder, store):
    from pydantic import SecretStr

    raw_data.api.key = SecretStr("s3cret")
    def factory(profile):
        return RAGPipeline(raw_data, FakeLLM(), Retriever(raw_data, embedder, store))

    c = TestClient(create_app(raw_data, factory))
    assert c.get("/profiles").status_code == 401
    assert c.get("/profiles", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/profiles", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert c.post("/query", json={"question": "hello there"}).status_code == 401
    assert c.get("/openapi.json").status_code == 200  # docs stay readable
