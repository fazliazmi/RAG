from __future__ import annotations

import json
import math
import shutil
import zlib
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from cyberrag.config import EmbeddingsConfig, LLMConfig, Settings, VectorStoreConfig, set_settings
from cyberrag.ingest.state import write_json_gz
from cyberrag.providers.embeddings.base import BaseEmbedder
from cyberrag.providers.llm.base import BaseLLM
from cyberrag.retrieval.sparse import tokenize
from cyberrag.schemas import Message
from cyberrag.store.qdrant import QdrantStore

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def cve_fixture(name: str) -> dict:
    return load_fixture(name)["vulnerabilities"][0]["cve"]


SYNTHETIC_SQLI_CVE = {
    "id": "CVE-2099-0001",
    "published": "2099-01-02T10:00:00.000",
    "lastModified": "2099-01-03T10:00:00.000",
    "vulnStatus": "Analyzed",
    "descriptions": [{"lang": "en", "value": "SQL injection in the login form of ExampleShop "
                      "3.2 allows remote attackers to read the user database."}],
    "metrics": {"cvssMetricV31": [{"type": "Primary", "cvssData": {
        "version": "3.1", "baseScore": 9.8, "baseSeverity": "CRITICAL",
        "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}}]},
    "weaknesses": [{"type": "Primary", "description": [{"lang": "en", "value": "CWE-89"}]}],
    "configurations": [{"nodes": [{"cpeMatch": [{
        "vulnerable": True, "criteria": "cpe:2.3:a:example:exampleshop:3.2:*:*:*:*:*:*:*"}]}]}],
    "references": [{"url": "https://example.com/advisory"}],
}

REJECTED_CVE = {
    "id": "CVE-2099-0002",
    "published": "2099-01-02T10:00:00.000",
    "vulnStatus": "Rejected",
    "descriptions": [{"lang": "en", "value": "** REJECT ** Duplicate."}],
}


class FakeEmbedder(BaseEmbedder):
    """Deterministic hashed bag-of-words vectors: similar words, similar vectors."""

    def __init__(self, dim: int = 128) -> None:
        super().__init__(EmbeddingsConfig(provider="fake", model="fake-embed", batch_size=16))
        self._n = dim

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self._n
            for tok in tokenize(text):
                vec[zlib.crc32(tok.encode()) % self._n] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out

    def health(self) -> tuple[bool, str]:
        return True, "fake"


class FakeLLM(BaseLLM):
    def __init__(self, reply: str = "It is a backdoor in xz [CVE-2024-3094].", num_ctx: int = 4096):
        super().__init__(LLMConfig(profile="fake", provider="fake", model="fake-llm", num_ctx=num_ctx,
                                   max_tokens=256))
        self.reply = reply
        self.calls: list[list[Message]] = []

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        self.calls.append(list(messages))
        words = self.reply.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")

    def health(self) -> tuple[bool, str]:
        return True, "fake"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for key in list(os.environ):
        if key.startswith("CYBERRAG") or key == "NVD_API_KEY":
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def settings(tmp_path: Path) -> Iterator[Settings]:
    shutil.copytree(REPO / "config", tmp_path / "config")
    shutil.copytree(REPO / "prompts", tmp_path / "prompts")
    s = Settings(
        home=tmp_path,
        sources={
            "nvd": {"delay_no_key_s": 0, "delay_with_key_s": 0, "retry_wait_min_s": 0,
                    "retry_wait_max_s": 0, "max_retries": 3, "live_lookup": False},
            "cwe": {"retry_wait_min_s": 0, "max_retries": 2},
        },
    )
    set_settings(s)
    yield s
    set_settings(None)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store() -> QdrantStore:
    return QdrantStore(VectorStoreConfig(quantization="none"), client=QdrantClient(":memory:"))


@pytest.fixture
def raw_data(settings: Settings) -> Settings:
    """Write fixture data where `fetch` would have put it."""
    cwe = load_fixture("cwe-79-89.json")
    write_json_gz(settings.raw_dir / "cwe" / "cwe-weaknesses-4.20.json.gz",
                  {"version": {"ContentVersion": "4.20"}, "Weaknesses": cwe["Weaknesses"]})
    vulns = [
        {"cve": cve_fixture("cve-2024-3094.json")},
        {"cve": cve_fixture("cve-2021-44228.json")},
        {"cve": SYNTHETIC_SQLI_CVE},
        {"cve": REJECTED_CVE},
    ]
    write_json_gz(settings.raw_dir / "nvd" / "full" / "page-0000000.json.gz",
                  {"totalResults": len(vulns), "vulnerabilities": vulns})
    return settings
