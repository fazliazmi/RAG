"""Settings loading.

Priority, highest first:
  explicit arguments > environment variables > .env file > config/default.yaml > defaults below

LLM settings live in profile files (config/llm/<name>.yaml) so a model can be
swapped by picking another profile instead of editing code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, BaseModel, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from cyberrag.errors import ConfigError


def find_project_root() -> Path:
    """Folder that holds config/default.yaml.

    Order: $CYBERRAG_HOME, then the current folder or any parent of it,
    then the folder this package was installed from (editable installs).
    """
    env = os.environ.get("CYBERRAG_HOME")
    if env:
        return Path(env).expanduser().resolve()
    cwd = Path.cwd().resolve()
    for folder in (cwd, *cwd.parents):
        if (folder / "config" / "default.yaml").is_file():
            return folder
    return Path(__file__).resolve().parents[2]


class LLMConfig(BaseModel):
    profile: str = "custom"
    provider: str = "ollama"
    model: str = "llama3.2:3b"
    base_url: str | None = None
    api_key_env: str | None = None  # name of the env var that holds the key
    temperature: float = 0.1
    num_ctx: int = 8192
    max_tokens: int = 1024
    timeout_s: float = 300.0
    extra: dict[str, Any] = Field(default_factory=dict)  # provider-specific options


class EmbeddingsConfig(BaseModel):
    provider: str = "ollama"
    model: str = "nomic-embed-text"
    base_url: str | None = "http://127.0.0.1:11434"
    api_key_env: str | None = None
    batch_size: int = 64
    max_chars: int = 2000
    doc_prefix: str = ""
    query_prefix: str = ""
    timeout_s: float = 300.0
    extra: dict[str, Any] = Field(default_factory=dict)


class VectorStoreConfig(BaseModel):
    provider: str = "qdrant"
    url: str = "http://127.0.0.1:6333"
    api_key_env: str | None = None
    quantization: Literal["int8", "none"] = "int8"
    on_disk: bool = True
    collection_prefix: str = ""
    timeout_s: float = 120.0


class NVDConfig(BaseModel):
    base_url: str = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    feeds_url: str = "https://nvd.nist.gov/feeds/json/cve/2.0"
    full_download: Literal["feeds", "api"] = "feeds"
    page_size: int = 2000
    delay_no_key_s: float = 6.0
    delay_with_key_s: float = 0.7
    skip_rejected: bool = True
    live_lookup: bool = True
    timeout_s: float = 300.0
    max_retries: int = 8
    retry_wait_min_s: float = 6.0
    retry_wait_max_s: float = 120.0


class CWEConfig(BaseModel):
    base_url: str = "https://cwe-api.mitre.org/api/v1"
    timeout_s: float = 300.0
    max_retries: int = 5
    retry_wait_min_s: float = 3.0


class SourcesConfig(BaseModel):
    nvd: NVDConfig = Field(default_factory=NVDConfig)
    cwe: CWEConfig = Field(default_factory=CWEConfig)


class RetrievalConfig(BaseModel):
    top_k_cve: int = 8
    top_k_cwe: int = 4
    prefetch_k: int = 40
    hybrid: bool = True
    enrich_cwe: bool = True


class GenerationConfig(BaseModel):
    answer_reserve_tokens: int = 1024
    chars_per_token: float = 3.5


class APIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    # When set (CYBERRAG__API__KEY), every route except /health and the docs requires
    # the header `X-API-Key: <key>`. Set one before serving on anything but 127.0.0.1.
    key: SecretStr | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CYBERRAG__",
        env_nested_delimiter="__",
        extra="ignore",
    )

    home: Path | None = None
    llm_profile: str = "llama3.2-3b"
    llm: dict[str, Any] = Field(default_factory=dict)  # per-field overrides for the default profile
    plugins: list[str] = Field(default_factory=list)
    embedding_profile: str | None = None  # a file in config/embeddings/
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    data_dir: Path = Path("data")
    prompts_dir: Path = Path("prompts")
    nvd_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("NVD_API_KEY", "CYBERRAG__NVD_API_KEY")
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        init_kwargs = getattr(init_settings, "init_kwargs", {}) or {}
        root = Path(init_kwargs["home"]) if init_kwargs.get("home") else find_project_root()
        config_file = Path(os.environ.get("CYBERRAG_CONFIG", root / "config" / "default.yaml"))
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            DotEnvSettingsSource(settings_cls, env_file=root / ".env"),
        ]
        if config_file.is_file():
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=config_file))
        return tuple(sources)

    @model_validator(mode="before")
    @classmethod
    def _apply_embedding_profile(cls, data: Any) -> Any:
        """The embedding profile file is the base; keys under `embeddings:` (YAML or
        CYBERRAG__EMBEDDINGS__*) override it."""
        if not isinstance(data, dict) or not data.get("embedding_profile"):
            return data
        root = Path(data["home"]) if data.get("home") else find_project_root()
        folder = root / "config" / "embeddings"
        name = data["embedding_profile"]
        path = folder / f"{name}.yaml"
        if not path.is_file():
            available = ", ".join(sorted(p.stem for p in folder.glob("*.yaml"))) or "(none)"
            raise ConfigError(f"Embedding profile '{name}' not found in {folder}. Available: {available}")
        profile = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        overrides = data.get("embeddings") or {}
        if isinstance(overrides, BaseModel):
            overrides = overrides.model_dump(exclude_unset=True)
        return {**data, "embeddings": {**profile, **overrides}}

    @model_validator(mode="after")
    def _resolve_home(self) -> Settings:
        if self.home is None:
            self.home = find_project_root()
        return self

    # ---- paths -------------------------------------------------------------
    @property
    def root(self) -> Path:
        assert self.home is not None
        return Path(self.home)

    def path(self, p: Path | str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.root / p

    @property
    def data_path(self) -> Path:
        return self.path(self.data_dir)

    @property
    def raw_dir(self) -> Path:
        return self.data_path / "raw"

    @property
    def state_dir(self) -> Path:
        return self.data_path / "state"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_path / "snapshots"

    @property
    def prompts_path(self) -> Path:
        return self.path(self.prompts_dir)

    @property
    def llm_profiles_dir(self) -> Path:
        return self.root / "config" / "llm"

    # ---- LLM profiles ------------------------------------------------------
    def list_llm_profiles(self) -> list[str]:
        return sorted(p.stem for p in self.llm_profiles_dir.glob("*.yaml"))

    def list_embedding_profiles(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "config" / "embeddings").glob("*.yaml"))

    def llm_config(self, profile: str | None = None, **overrides: Any) -> LLMConfig:
        """Load an LLM profile.

        Env overrides (CYBERRAG__LLM__*) apply to the default profile only, so
        `--llm other-profile` gives exactly what that profile file says.
        """
        name = profile or self.llm_profile
        path = self.llm_profiles_dir / f"{name}.yaml"
        if not path.is_file():
            available = ", ".join(self.list_llm_profiles()) or "(none)"
            raise ConfigError(
                f"LLM profile '{name}' not found in {self.llm_profiles_dir}. Available: {available}"
            )
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"{path} must contain a YAML mapping")
        env_overrides = self.llm if name == self.llm_profile else {}
        merged = {**data, **env_overrides, **overrides, "profile": name}
        return LLMConfig.model_validate(merged)

    @property
    def nvd_key(self) -> str | None:
        if self.nvd_api_key is None:
            return None
        value = self.nvd_api_key.get_secret_value().strip()
        return value or None


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(settings: Settings | None) -> None:
    """Replace the cached settings (used by tests and the CLI)."""
    global _settings
    _settings = settings
