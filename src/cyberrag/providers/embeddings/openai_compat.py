from __future__ import annotations

from collections.abc import Sequence

from cyberrag.config import EmbeddingsConfig
from cyberrag.errors import ProviderError
from cyberrag.providers.embeddings.base import BaseEmbedder
from cyberrag.providers.llm.openai_compat import make_openai_client
from cyberrag.providers.registry import register_embedder


@register_embedder("openai_compat")
class OpenAICompatEmbedder(BaseEmbedder):
    """Any server with an OpenAI-style /embeddings endpoint (OpenAI, vLLM,
    LM Studio, Text Embeddings Inference, Ollama /v1, ...)."""

    def __init__(self, cfg: EmbeddingsConfig) -> None:
        super().__init__(cfg)
        self._client = make_openai_client(cfg.base_url, cfg.api_key_env, cfg.timeout_s)

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            resp = self._client.embeddings.create(
                model=self.cfg.model, input=list(texts), **self.cfg.extra.get("params", {})
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Embedding error from {self.cfg.base_url}: {exc}") from exc
        return [list(d.embedding) for d in sorted(resp.data, key=lambda d: d.index)]

    def health(self) -> tuple[bool, str]:
        try:
            return True, f"{self.cfg.model} via {self.cfg.base_url} (dim {self.dim})"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
