"""Provider registry: maps the `provider:` name in config to a class.

To add a backend, write one class and decorate it:

    @register_llm("my_backend")
    class MyLLM(BaseLLM): ...

Built-in providers are imported lazily, so an optional dependency (e.g. the
`openai` package) is only needed when that provider is actually used.
Third-party modules listed under `plugins:` in config are imported too.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, TypeVar

from cyberrag.errors import ConfigError

if TYPE_CHECKING:
    from cyberrag.config import EmbeddingsConfig, LLMConfig, VectorStoreConfig
    from cyberrag.providers.embeddings.base import BaseEmbedder
    from cyberrag.providers.llm.base import BaseLLM
    from cyberrag.store.base import VectorStore

T = TypeVar("T")

_LLMS: dict[str, type] = {}
_EMBEDDERS: dict[str, type] = {}
_STORES: dict[str, type] = {}

# provider name -> module that registers it
_BUILTIN_LLMS = {
    "ollama": "cyberrag.providers.llm.ollama",
    "openai_compat": "cyberrag.providers.llm.openai_compat",
}
_BUILTIN_EMBEDDERS = {
    "ollama": "cyberrag.providers.embeddings.ollama",
    "openai_compat": "cyberrag.providers.embeddings.openai_compat",
}
_BUILTIN_STORES = {
    "qdrant": "cyberrag.store.qdrant",
}


def _register(table: dict[str, type], name: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        table[name] = cls
        return cls

    return decorator


def register_llm(name: str) -> Callable[[type[T]], type[T]]:
    return _register(_LLMS, name)


def register_embedder(name: str) -> Callable[[type[T]], type[T]]:
    return _register(_EMBEDDERS, name)


def register_store(name: str) -> Callable[[type[T]], type[T]]:
    return _register(_STORES, name)


def load_plugins(modules: Iterable[str]) -> None:
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as exc:
            raise ConfigError(f"Could not import plugin module '{module}': {exc}") from exc


def _resolve(kind: str, name: str, table: dict[str, type], builtins: dict[str, str]) -> type:
    if name not in table and name in builtins:
        importlib.import_module(builtins[name])
    if name not in table:
        known = sorted(set(table) | set(builtins))
        raise ConfigError(f"Unknown {kind} provider '{name}'. Available: {', '.join(known)}")
    return table[name]


def build_llm(cfg: LLMConfig) -> BaseLLM:
    return _resolve("LLM", cfg.provider, _LLMS, _BUILTIN_LLMS)(cfg)


def build_embedder(cfg: EmbeddingsConfig) -> BaseEmbedder:
    return _resolve("embedding", cfg.provider, _EMBEDDERS, _BUILTIN_EMBEDDERS)(cfg)


def build_store(cfg: VectorStoreConfig) -> VectorStore:
    return _resolve("vector store", cfg.provider, _STORES, _BUILTIN_STORES)(cfg)
