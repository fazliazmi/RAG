"""Data shapes shared by every layer. Nothing here depends on a specific provider."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SourceKind = Literal["cve", "cwe"]


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class SparseVec(BaseModel):
    indices: list[int]
    values: list[float]


class Document(BaseModel):
    """One unit that gets embedded: a whole CVE, or one section of a CWE entry."""

    id: str  # "CVE-2024-3094" or "CWE-89#mitigations"
    source: SourceKind
    text: str
    payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def ref(self) -> str:
        """Citation tag: the CVE/CWE ID without the section suffix."""
        return self.id.split("#", 1)[0]


class Hit(Document):
    """A retrieved document with its score and how it was found."""

    score: float = 0.0
    via: Literal["exact", "search", "enrich", "live"] = "search"


class Source(BaseModel):
    ref: str
    source: SourceKind
    title: str = ""
    url: str = ""
    score: float = 0.0
    via: str = "search"
    cited: bool = False  # the answer mentions this ID


class Answer(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    unsupported: list[str] = Field(
        default_factory=list,
        description="CVE/CWE IDs in the answer that were not in the retrieved data",
    )
    llm: str
    embedder: str
    timings_ms: dict[str, float] = Field(default_factory=dict)
    context: str | None = None
