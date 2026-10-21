from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

from cyberrag.config import EmbeddingsConfig


def slugify(text: str) -> str:
    """'nomic-embed-text:latest' -> 'nomic-embed-text'. Used in collection names."""
    text = text.lower().removesuffix(":latest")
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


class BaseEmbedder(ABC):
    """Text embedding interface.

    Prefixes and truncation come from config, so switching to a model with
    different conventions (bge, e5, ...) is a config change only.
    """

    def __init__(self, cfg: EmbeddingsConfig) -> None:
        self.cfg = cfg
        self._dim: int | None = None

    @property
    def name(self) -> str:
        return f"{self.cfg.provider}:{self.cfg.model}"

    @property
    def slug(self) -> str:
        return slugify(self.cfg.model)

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self._embed(["dimension probe"])[0])
        return self._dim

    @abstractmethod
    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of already-prepared texts."""

    @abstractmethod
    def health(self) -> tuple[bool, str]: ...

    def _prepare(self, text: str, prefix: str) -> str:
        return prefix + text[: self.cfg.max_chars]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        size = max(1, self.cfg.batch_size)
        for i in range(0, len(texts), size):
            batch = [self._prepare(t, self.cfg.doc_prefix) for t in texts[i : i + size]]
            out.extend(self._embed(batch))
        if out and self._dim is None:
            self._dim = len(out[0])
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self._prepare(text, self.cfg.query_prefix)])[0]
