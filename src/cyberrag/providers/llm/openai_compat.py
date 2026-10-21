from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import Any

from cyberrag.config import LLMConfig
from cyberrag.errors import ConfigError, ProviderError
from cyberrag.providers.llm.base import BaseLLM
from cyberrag.providers.registry import register_llm
from cyberrag.schemas import Message


def make_openai_client(base_url: str | None, api_key_env: str | None, timeout: float) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ConfigError(
            "The openai_compat provider needs the openai package: pip install -e \".[openai]\""
        ) from exc
    api_key = os.environ.get(api_key_env) if api_key_env else None
    if api_key_env and not api_key:
        raise ConfigError(f"Environment variable {api_key_env} is not set (needed by this profile)")
    # Local servers (Ollama /v1, vLLM, LM Studio) accept any key.
    return OpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=timeout)


@register_llm("openai_compat")
class OpenAICompatLLM(BaseLLM):
    """Any server that speaks the OpenAI chat-completions API.

    Profile `extra:` options:
      max_tokens_param: name of the length parameter (default "max_tokens";
                        newer OpenAI models use "max_completion_tokens")
      params:           dict passed straight to chat.completions.create
    """

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__(cfg)
        self._client = make_openai_client(cfg.base_url, cfg.api_key_env, cfg.timeout_s)

    def _kwargs(self) -> dict[str, Any]:
        length_param = self.cfg.extra.get("max_tokens_param", "max_tokens")
        return {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            length_param: self.cfg.max_tokens,
            **self.cfg.extra.get("params", {}),
        }

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        try:
            chunks = self._client.chat.completions.create(
                messages=[m.model_dump() for m in messages], stream=True, **self._kwargs()
            )
            for chunk in chunks:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except ConfigError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise SDK errors
            raise ProviderError(f"{self.cfg.base_url} ({self.cfg.model}): {exc}") from exc

    def health(self) -> tuple[bool, str]:
        try:
            ids = {m.id for m in self._client.models.list()}
            if ids and self.cfg.model not in ids:
                return False, f"server does not list model '{self.cfg.model}'"
            return True, f"{self.cfg.model} via {self.cfg.base_url}"
        except Exception as exc:  # noqa: BLE001
            return False, f"cannot reach {self.cfg.base_url} ({exc})"
