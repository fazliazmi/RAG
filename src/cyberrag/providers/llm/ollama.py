from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import ollama

from cyberrag.config import LLMConfig
from cyberrag.errors import ProviderError
from cyberrag.providers.llm.base import BaseLLM
from cyberrag.providers.registry import register_llm
from cyberrag.schemas import Message

DEFAULT_HOST = "http://127.0.0.1:11434"


def model_available(client: ollama.Client, model: str) -> bool:
    names = {m.model for m in client.list().models if m.model}
    return model in names or f"{model}:latest" in names


@register_llm("ollama")
class OllamaLLM(BaseLLM):
    """Native Ollama API. Extra options in the profile's `extra:` block:
    `options` (dict of Ollama model options), `keep_alive`, `think`."""

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__(cfg)
        self._client = ollama.Client(host=cfg.base_url or DEFAULT_HOST, timeout=cfg.timeout_s)

    def _kwargs(self) -> dict[str, Any]:
        extra = self.cfg.extra
        options = {
            "temperature": self.cfg.temperature,
            "num_ctx": self.cfg.num_ctx,
            "num_predict": self.cfg.max_tokens,
            **extra.get("options", {}),
        }
        kwargs: dict[str, Any] = {"model": self.cfg.model, "options": options}
        if "keep_alive" in extra:
            kwargs["keep_alive"] = extra["keep_alive"]
        if "think" in extra:
            kwargs["think"] = extra["think"]
        return kwargs

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        try:
            for part in self._client.chat(
                messages=[m.model_dump() for m in messages], stream=True, **self._kwargs()
            ):
                if part.message and part.message.content:
                    yield part.message.content
        except ollama.ResponseError as exc:
            raise ProviderError(f"Ollama error for model '{self.cfg.model}': {exc}") from exc
        except ConnectionError as exc:
            raise ProviderError(f"Cannot reach Ollama at {self.cfg.base_url}: {exc}") from exc

    def health(self) -> tuple[bool, str]:
        try:
            if not model_available(self._client, self.cfg.model):
                return False, f"model not pulled; run: ollama pull {self.cfg.model}"
            return True, f"{self.cfg.model} available"
        except Exception as exc:  # noqa: BLE001 - any failure means unhealthy
            return False, f"cannot reach Ollama at {self.cfg.base_url or DEFAULT_HOST} ({exc})"
