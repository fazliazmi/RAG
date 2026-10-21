class CyberRAGError(Exception):
    """Base class for errors that should be shown to the user as a clear message."""


class ConfigError(CyberRAGError):
    """Bad or missing configuration (unknown profile, unknown provider, ...)."""


class ProviderError(CyberRAGError):
    """A model or service (Ollama, OpenAI-compatible server, Qdrant) failed or is unreachable."""


class IndexMissingError(CyberRAGError):
    """The vector collection for the current embedder does not exist yet."""


class EmbedderMismatchError(CyberRAGError):
    """A collection was built with a different embedding model or dimension."""
