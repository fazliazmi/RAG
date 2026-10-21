import pytest

from cyberrag.config import LLMConfig, Settings, VectorStoreConfig
from cyberrag.errors import ConfigError
from cyberrag.providers.llm.base import BaseLLM
from cyberrag.providers.registry import build_llm, build_store, load_plugins, register_llm


def test_default_profile_is_llama(settings: Settings):
    cfg = settings.llm_config()
    assert cfg.profile == "llama3.2-3b"
    assert cfg.provider == "ollama" and cfg.model == "llama3.2:3b"
    assert "qwen3-8b" in settings.list_llm_profiles()


def test_unknown_profile_lists_available(settings: Settings):
    with pytest.raises(ConfigError, match="Available: .*llama3.2-3b"):
        settings.llm_config("does-not-exist")


def test_env_overrides(settings: Settings, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CYBERRAG__LLM__MODEL", "llama3.2:1b")
    monkeypatch.setenv("CYBERRAG__LLM__NUM_CTX", "4096")
    monkeypatch.setenv("CYBERRAG__EMBEDDINGS__MODEL", "bge-m3")
    monkeypatch.setenv("NVD_API_KEY", "secret")
    s = Settings(home=settings.root)
    cfg = s.llm_config()
    assert cfg.model == "llama3.2:1b" and cfg.num_ctx == 4096
    # an explicitly chosen profile is used exactly as written
    assert s.llm_config("qwen3-8b").model == "qwen3:8b"
    assert s.embeddings.model == "bge-m3"
    assert s.nvd_key == "secret"


def test_profile_switch_by_env(settings: Settings, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CYBERRAG__LLM_PROFILE", "qwen3-8b")
    assert Settings(home=settings.root).llm_config().model == "qwen3:8b"


def test_unknown_provider():
    with pytest.raises(ConfigError, match="Unknown LLM provider 'nope'.*ollama"):
        build_llm(LLMConfig(provider="nope"))
    with pytest.raises(ConfigError, match="vector store"):
        build_store(VectorStoreConfig(provider="nope"))


def test_custom_provider_registers_without_code_changes():
    @register_llm("echo-test")
    class EchoLLM(BaseLLM):
        def stream(self, messages):
            yield messages[-1].content

        def health(self):
            return True, "ok"

    llm = build_llm(LLMConfig(provider="echo-test", model="echo"))
    assert llm.name == "echo-test:echo"


def test_bad_plugin_module():
    with pytest.raises(ConfigError, match="plugin"):
        load_plugins(["cyberrag_missing_plugin_xyz"])


def test_embedding_profiles(settings: Settings, monkeypatch: pytest.MonkeyPatch):
    assert settings.embedding_profile == "nomic-embed-text"
    assert settings.embeddings.model == "nomic-embed-text"
    assert settings.embeddings.doc_prefix == "search_document: "
    assert settings.embeddings.batch_size == 64  # from default.yaml, on top of the profile
    assert "arctic-embed-xs" in settings.list_embedding_profiles()

    monkeypatch.setenv("CYBERRAG__EMBEDDING_PROFILE", "arctic-embed-xs")
    s = Settings(home=settings.root)
    assert s.embeddings.model == "snowflake-arctic-embed:22m"
    assert s.embeddings.doc_prefix == ""

    monkeypatch.setenv("CYBERRAG__EMBEDDINGS__MAX_CHARS", "500")
    assert Settings(home=settings.root).embeddings.max_chars == 500  # override beats profile

    with pytest.raises(ConfigError, match="Available: .*nomic-embed-text"):
        Settings(home=settings.root, embedding_profile="nope")
