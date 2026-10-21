from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

from cyberrag.config import LLMConfig
from cyberrag.schemas import Message


class BaseLLM(ABC):
    """Chat model interface. Prompts are plain role/content messages; each
    provider applies its model's own chat template, so nothing in the RAG
    code is tied to one model family."""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg

    @property
    def name(self) -> str:
        return f"{self.cfg.provider}:{self.cfg.model}"

    @property
    def context_window(self) -> int:
        return self.cfg.num_ctx

    @abstractmethod
    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        """Yield the answer piece by piece."""

    def chat(self, messages: Sequence[Message]) -> str:
        return "".join(self.stream(messages))

    @abstractmethod
    def health(self) -> tuple[bool, str]:
        """(ok, human-readable detail)."""
