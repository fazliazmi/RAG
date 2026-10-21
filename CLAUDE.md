# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CyberRAG: a local RAG system that answers questions about CVEs (NVD) and CWE weaknesses (MITRE), citing the IDs it used. It's plain Python behind small interfaces, with no LangChain or LlamaIndex. The LLM, the embedding model and the vector store are each picked by config. The defaults are Ollama `llama3.2:3b`, Ollama `nomic-embed-text` and Qdrant in Docker. `README.md` is the user-facing guide and holds measured performance numbers.

## Commands

The dev machine is Windows; `python` is not on PATH there, so use `py -3` or the venv's interpreter directly.

```bash
# setup
py -3 -m venv .venv                       # macOS/Linux: python3 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.lock && .venv/Scripts/python -m pip install -e . --no-deps

# checks (run both before finishing a change)
.venv/Scripts/ruff check src tests
.venv/Scripts/python -m pytest -q                                   # unit tests, no services needed
.venv/Scripts/python -m pytest -q tests/test_index_retrieve.py::test_named_cve_comes_first
.venv/Scripts/python -m pytest -m integration -q                    # needs Qdrant + Ollama + CWE index

# services
docker compose up -d                      # Qdrant on 127.0.0.1:6333 (named volume rag_qdrant_storage)
.venv/Scripts/cyberrag doctor             # checks LLM, embedder, Qdrant, indexes, raw data

# app
.venv/Scripts/cyberrag ask "How do I prevent CWE-89?" [--llm PROFILE] [--show-context] [--json]
.venv/Scripts/cyberrag index cve --limit 2000     # never run the full CVE index casually: hours on CPU
.venv/Scripts/cyberrag eval --retrieval-only      # hit rate / MRR on eval/questions.yaml
.venv/Scripts/cyberrag serve                       # FastAPI on 127.0.0.1:8000
```

Ruff: line length 110, rules E/F/W/I/B/UP, target Python 3.11. Don't use syntax newer than 3.11. CI (`.github/workflows/tests.yml`) runs ruff and the unit tests on Windows, macOS and Linux with Python 3.11 and 3.13.

## Architecture

Two paths share two services:
- The **build path** (`fetch` → `data/raw/` → `normalize` → `Indexer`) writes to Qdrant through the embedder.
- The **query path** (CLI/API → `RAGPipeline` → `Retriever` → LLM) reads from Qdrant through the same embedder.

**Config to objects.**
- `config.py` `Settings` merges, lowest priority first: `config/default.yaml`, then `.env`, then `CYBERRAG__SECTION__KEY` env vars, then init kwargs.
- LLM settings live in profiles (`config/llm/<name>.yaml`) loaded by `Settings.llm_config(profile)`. `CYBERRAG__LLM__*` overrides apply to the default profile only.
- Embedding profiles (`config/embeddings/<name>.yaml`) are merged in a `mode="before"` validator: the profile is the base and `embeddings:` keys override it.
- `factory.py` is the only place config becomes concrete classes. It uses `providers/registry.py`, which imports built-in providers lazily (so `openai` is optional) and loads `plugins:` modules.
- New backends are one class with `@register_llm` / `@register_embedder` / `@register_store` plus a profile. Don't add provider-specific code outside `providers/` and `store/`.

**Interfaces.**
- `BaseLLM.stream()` is abstract; `chat()` joins the stream.
- `BaseEmbedder._embed()` is abstract. The base class applies the profile's `doc_prefix`/`query_prefix` and `max_chars`.
- `VectorStore` takes a store-neutral filter dict (`{"severity": [...], "published": {"gte": "..."}}`), which `store/qdrant.py` translates. Retrieval code never builds Qdrant objects.

**Collections and IDs.**
- Collections are named `{kind}__{embedder slug}` (e.g. `cve__nomic-embed-text`), so every embedding model has its own pair.
- `ensure_collection` checks the stored metadata (`embedder`, `dim`, `sparse`) and raises `EmbedderMismatchError`.
- If keyword tokenization changes in `retrieval/sparse.py`, bump `SPARSE_VERSION`. It's stored in the collection metadata.
- Point IDs are `uuid5(doc id)`, so re-indexing overwrites instead of duplicating.
- Doc IDs are `CVE-2024-3094` or `CWE-89#mitigations[-N]`. `Document.ref` drops the section and is the citation tag.

**Build path details.**
- Raw files are the source of truth. Every CVE file has the API-page shape `{"vulnerabilities": [{"cve": ...}]}` with at most `page_size` (2,000) records. The CWE file is a single JSON.
- `ingest/fetch.cve_raw_files()` sets indexing order:
  1. yearly feeds, newest year first (a CVE appears in exactly one yearly feed)
  2. API full pages
  3. `updates/` runs, oldest first (so the newest version of a record wins)
- The indexer checkpoint `data/state/index__<collection>.json` maps each raw file's relative path to its size. Files whose size changed are indexed again.
- Rejected CVEs are skipped, and deleted when an update marks them rejected.

**Query path details.**
- `Retriever.retrieve` (`retrieval/retriever.py`) builds results in a fixed priority order (see its docstring):
  1. named CVEs, with a live NVD lookup if missing
  2. the CWE overviews of those CVEs
  3. sections of named CWEs
  4. hybrid search (dense + sparse, RRF fusion) on both collections
  5. CWE overviews for CVEs found by search
- When the question names IDs, search narrows to 2 CVEs + 1 CWE and step 5 is skipped.
- Chat follow-ups pass `carry_ids` taken from the last exchange.
- `RAGPipeline.prepare` fits hits into `num_ctx - answer_reserve - prompt`, with tokens estimated at 3.5 characters per token.
- Prompts come from `prompts/*.md` and are filled with `str.replace`, not `format`, because CWE code examples contain braces.
- `mark_cited` / `unsupported_ids` compare the answer against the sources and the context.

**Errors.** Subclass `CyberRAGError`. The CLI's `friendly_errors` prints it as one line, and the API maps it to an HTTP status in `api/app.py`.

## Tests

`tests/conftest.py` provides:
- `settings`: a temp home with copies of `config/` and `prompts/`, zero retry waits and live lookup off.
- `raw_data`: real CVE/CWE fixtures written into `data/raw/`.
- `FakeEmbedder` (hashed bag of words), `FakeLLM`, and `store` (a `QdrantStore` on `QdrantClient(":memory:")`).
- An autouse fixture that clears `CYBERRAG*` and `NVD_API_KEY` from the environment.

HTTP is mocked with `respx`. Integration tests use `@pytest.mark.integration` and are excluded by default (`addopts` in `pyproject.toml`).

## Gotchas (all observed on the dev machine)

- **Use `127.0.0.1`, never `localhost`,** in URLs. On Windows, `localhost` added ~2 s to every Qdrant call.
- **NVD API is slow.** It took ~3 min per 2,000-record page, so the full download uses the yearly JSON feeds (`sources/nvd_feeds.py`):
  - Downloads resume with HTTP Range requests because the server drops long transfers.
  - Feeds are parsed with the streaming `iter_array`; never `json.load` a whole feed (up to 300 MB).
  - `sync` uses the API in ≤120-day `lastMod` windows, starting from `data/state/nvd_sync.json`.
  - NVD rebuilds a yearly feed only when it changes, so the first sync starts a day before the *newest* feed date.
- **MITRE's `/cwe/weakness/all` response gets truncated.** `sources/cwe.py` walks view 1000's descendants and fetches 40 IDs per request.
- **CPU embedding is slow.** `nomic-embed-text` does ~5 docs/s (full NVD ≈ 23 h) and `arctic-embed-xs` ~30 docs/s. `cyberrag sync` deliberately won't start the first CVE index.
- **Ollama's OpenAI-compatible `/v1` ignores `num_ctx`** (server default 4096), so `openai_compat` profiles must not claim a larger `num_ctx`.
- **Leave the integrated GPU off.** Enabling it (`OLLAMA_IGPU_ENABLE`) made `llama3.2:3b` produce garbage.
- **Rich markup eats `[...]` text.** Wrap derived strings with `rich.markup.escape` (a profile name in brackets once disappeared). Write raw JSON to `sys.stdout.buffer` as UTF-8 for Windows consoles.
- **`requirements.lock` is a `pip freeze` with hand-added markers:**
  - `pywin32` and `colorama` get `sys_platform == "win32"`.
  - `numpy` is split by `python_version` because numpy ≥ 2.5 needs Python 3.12.
  - Keep these markers when regenerating.
- **Don't commit** `data/` (git-ignored; raw data, checkpoints, snapshots), `.env` or `.venv/`.

## Keep in sync

- **README:** when commands, options, config keys or measured numbers change, update `README.md`, including the "Known limitations" section.
- **Retrieval or prompt changes:** run `cyberrag eval --retrieval-only` before and after to compare.
