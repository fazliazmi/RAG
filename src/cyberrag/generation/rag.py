"""Question -> retrieve -> prompt -> LLM -> answer with sources."""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from cyberrag.config import Settings, get_settings
from cyberrag.errors import ConfigError
from cyberrag.factory import make_llm, make_retriever
from cyberrag.providers.llm.base import BaseLLM
from cyberrag.retrieval.context import build_context, estimate_tokens
from cyberrag.retrieval.query import extract_cve_ids, extract_cwe_ids
from cyberrag.retrieval.retriever import Retriever
from cyberrag.schemas import Answer, Hit, Message, Source
from cyberrag.store.base import Filters

MAX_HISTORY_MESSAGES = 4
_SAFETY_MARGIN_TOKENS = 64


def load_prompt(prompts_dir: Path, name: str) -> str:
    path = prompts_dir / name
    if not path.is_file():
        raise ConfigError(f"Prompt file missing: {path}")
    return path.read_text(encoding="utf-8").strip()


def to_sources(hits: Sequence[Hit]) -> list[Source]:
    """One source per CVE/CWE ID, in the order they were given to the LLM."""
    seen: dict[str, Source] = {}
    for hit in hits:
        if hit.ref in seen:
            continue
        seen[hit.ref] = Source(
            ref=hit.ref,
            source=hit.source,
            title=str(hit.payload.get("title") or hit.payload.get("name") or ""),
            url=str(hit.payload.get("url") or ""),
            score=hit.score,
            via=hit.via,
        )
    return list(seen.values())


def mark_cited(sources: list[Source], answer: str) -> list[Source]:
    """Flag the sources the answer mentions and list them first."""
    upper = answer.upper()
    for s in sources:
        s.cited = s.ref.upper() in upper
    return sorted(sources, key=lambda s: not s.cited)  # stable: keeps retrieval order


def _ids(text: str) -> list[str]:
    return extract_cve_ids(text) + extract_cwe_ids(text)


def unsupported_ids(answer: str, context: str) -> list[str]:
    """CVE/CWE IDs the answer mentions that never appeared in the context given to the
    LLM. Small models sometimes invent IDs; these are the ones to double-check."""
    known = set(_ids(context))
    return [i for i in _ids(answer) if i not in known]


def carried_ids(question: str, history: Sequence[Message]) -> list[str]:
    """IDs from the last exchange, for a follow-up question that names none itself.
    The user's own earlier question comes first, then what the answer mentioned."""
    if _ids(question) or not history:
        return []
    last_user = next((m.content for m in reversed(history) if m.role == "user"), "")
    last_answer = next((m.content for m in reversed(history) if m.role == "assistant"), "")
    return list(dict.fromkeys(_ids(last_user) + _ids(last_answer)))


@dataclass
class Prepared:
    messages: list[Message]
    hits: list[Hit]
    context: str
    timings_ms: dict[str, float]


class RAGPipeline:
    def __init__(self, settings: Settings, llm: BaseLLM, retriever: Retriever) -> None:
        self.settings = settings
        self.llm = llm
        self.retriever = retriever
        self.system_prompt = load_prompt(settings.prompts_path, "system.md")
        self.answer_template = load_prompt(settings.prompts_path, "answer.md")

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, llm_profile: str | None = None
    ) -> RAGPipeline:
        settings = settings or get_settings()
        return cls(settings, make_llm(settings, llm_profile), make_retriever(settings))

    def _render(self, context: str, question: str) -> str:
        # str.replace, not str.format: CWE code examples contain braces.
        return self.answer_template.replace("{context}", context).replace("{question}", question)

    def context_budget(self, question: str, history: Sequence[Message]) -> int:
        cpt = self.settings.generation.chars_per_token
        fixed = (
            estimate_tokens(self.system_prompt, cpt)
            + estimate_tokens(self._render("", question), cpt)
            + sum(estimate_tokens(m.content, cpt) + 4 for m in history)
        )
        return (
            self.llm.context_window
            - self.settings.generation.answer_reserve_tokens
            - fixed
            - _SAFETY_MARGIN_TOKENS
        )

    def prepare(
        self,
        question: str,
        filters: Filters | None = None,
        history: Sequence[Message] | None = None,
    ) -> Prepared:
        history = list(history or [])[-MAX_HISTORY_MESSAGES:]
        t0 = time.perf_counter()
        hits = self.retriever.retrieve(question, filters, carry_ids=carried_ids(question, history))
        t1 = time.perf_counter()
        budget = self.context_budget(question, history)
        context, used = build_context(hits, budget, self.settings.generation.chars_per_token)
        if not used:
            context = "(no relevant documents found)"
        messages = [
            Message(role="system", content=self.system_prompt),
            *history,
            Message(role="user", content=self._render(context, question)),
        ]
        return Prepared(messages, used, context, {"retrieve": (t1 - t0) * 1000})

    def ask(
        self,
        question: str,
        filters: Filters | None = None,
        history: Sequence[Message] | None = None,
        include_context: bool = False,
    ) -> Answer:
        prep = self.prepare(question, filters, history)
        t0 = time.perf_counter()
        text = self.llm.chat(prep.messages)
        prep.timings_ms["generate"] = (time.perf_counter() - t0) * 1000
        return Answer(
            question=question,
            answer=text.strip(),
            sources=mark_cited(to_sources(prep.hits), text),
            unsupported=unsupported_ids(text, prep.context),
            llm=self.llm.name,
            embedder=self.retriever.embedder.name,
            timings_ms={k: round(v, 1) for k, v in prep.timings_ms.items()},
            context=prep.context if include_context else None,
        )

    def ask_stream(
        self,
        question: str,
        filters: Filters | None = None,
        history: Sequence[Message] | None = None,
    ) -> tuple[Iterator[str], list[Source], Prepared]:
        prep = self.prepare(question, filters, history)
        return self.llm.stream(prep.messages), to_sources(prep.hits), prep
