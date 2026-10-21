from __future__ import annotations

from collections.abc import Sequence

import ollama

from cyberrag.config import EmbeddingsConfig
from cyberrag.errors import ProviderError
from cyberrag.providers.embeddings.base import BaseEmbedder
from cyberrag.providers.llm.ollama import DEFAULT_HOST, model_available
from cyberrag.providers.registry import register_embedder


@register_embedder("ollama")
class OllamaEmbedder(BaseEmbedder):
    def __init__(self, cfg: EmbeddingsConfig) -> None:
        super().__init__(cfg)
        self._client = ollama.Client(host=cfg.base_url or DEFAULT_HOST, timeout=cfg.timeout_s)

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            resp = self._client.embed(
                model=self.cfg.model,
                input=list(texts),
                truncate=True,
                keep_alive=self.cfg.extra.get("keep_alive", "10m"),
            )
        except ollama.ResponseError as exc:
            raise ProviderError(f"Ollama embedding error ({self.cfg.model}): {exc}") from exc
        except ConnectionError as exc:
            raise ProviderError(f"Cannot reach Ollama at {self.cfg.base_url}: {exc}") from exc
        return [list(v) for v in resp.embeddings]

    def health(self) -> tuple[bool, str]:
        try:
            if not model_available(self._client, self.cfg.model):
                return False, f"model not pulled; run: ollama pull {self.cfg.model}"
            return True, f"{self.cfg.model} available (dim {self.dim})"
        except Exception as exc:  # noqa: BLE001
            return False, f"cannot reach Ollama at {self.cfg.base_url or DEFAULT_HOST} ({exc})"
