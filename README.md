# CyberRAG

Ask questions about software vulnerabilities and get answers grounded in official data, with sources:

- **CVE** records from the NVD (all ~412,000 of them): bulk-loaded from the [NVD JSON 2.0 yearly feeds](https://nvd.nist.gov/vuln/data-feeds) and kept current with the [NVD CVE API 2.0](https://nvd.nist.gov/developers/vulnerabilities)
- **CWE** weakness types from the [MITRE CWE REST API](https://github.com/CWE-CAPEC/REST-API-wg)

It runs fully locally by default: [Ollama](https://ollama.com) with `llama3.2:3b` for answers, `nomic-embed-text` for embeddings and [Qdrant](https://qdrant.tech) as the vector database. **The LLM, the embedding model and the vector store can each be swapped through config, without code changes** (see [Changing the model](#changing-the-model)).

```text
$ cyberrag ask "What is CVE-2024-3094 and how serious is it?"
ollama:llama3.2:3b | 5 documents in context

CVE-2024-3094 is a critical vulnerability (CVSS 3.1: 10.0 CRITICAL) in the xz
library, specifically in versions 5.6.0 and 5.6.1 [...]. It is classified as
CWE-506, which is a Embedded Malicious Code vulnerability.

This vulnerability allows malicious code to be embedded in the liblzma library,
which can be used by any software linked against this library, intercepting and
modifying data interaction with this library. [...]

Cited sources
 ID             Title                                         Found by  Link
 CVE-2024-3094  Malicious code was discovered in the upstr…   live      https://nvd.nist.gov/vuln/detail/CVE-2024-3094
 CWE-506        Embedded Malicious Code                       enrich    https://cwe.mitre.org/data/definitions/506.html
+3 more retrieved but not cited (see --json or --show-context)
```

## How it works

```text
 NVD feeds + CVE API ──┐                         ┌─ CVE/CWE IDs in the question → exact lookup
                       ├─► data/raw/ ─► embed ─► Qdrant ─┤
 MITRE CWE API ────────┘   (gzip JSON)            └─ hybrid search (meaning + keywords)
                                                     │
                     CWE info for each CVE found ◄───┘
                                                     ▼
                         fit to the model's context window ─► LLM ─► answer + sources
```

- **One document per CVE.** The score, CWE IDs, affected products and CISA "known exploited" status sit next to the description.
- **One document per CWE section:** overview, consequences, mitigations, detection, examples and observed CVEs.
- **Hybrid search.** Dense vectors find similar meaning; BM25-style keyword vectors match exact terms such as `log4j 2.14.1`. Qdrant combines the two.
- **Exact ID lookup.** `CVE-…` and `CWE-…` IDs in a question are fetched directly. If a CVE isn't indexed yet, it is fetched live from NVD.
- **Raw data stays in `data/raw/`.** Changing the embedding model re-embeds from those files and never downloads again.

## Requirements

| | Minimum | Notes |
|---|---|---|
| Python | 3.11+ | Tested on 3.13 |
| [Docker](https://docs.docker.com/get-docker/) | Docker Desktop (Windows/macOS) or Docker Engine (Linux) | Runs Qdrant |
| [Ollama](https://ollama.com/download) | 0.4+ | Runs the LLM and the embedding model |
| [Git](https://git-scm.com/downloads) | any | |
| RAM | 8 GB (16 GB recommended for the full CVE index) | |
| Disk | ~10 GB free | Models ~2.3 GB, raw NVD data ~1 GB, Qdrant index a few GB |

A GPU is optional. On CPU only, everything works, but embedding all CVEs takes hours (see step 5 for a faster embedding profile). You can skip that by importing a snapshot from a machine that has already built the index (Option B below).

On a laptop CPU, an answer takes roughly 20–60 seconds: `llama3.2:3b` reads the retrieved text at about 100 tokens/s before it starts writing. A GPU or a hosted model profile makes this much faster.

## Quick start on a new machine

### 1. Get the code

```bash
git clone https://github.com/fazliazmi/RAG.git
cd RAG
```

### 2. Create a Python environment and install

**Windows (PowerShell)**

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e ".[api]"
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[api]"
```

This installs the `cyberrag` command. Run `cyberrag --help` to see every command. (If you'd rather not activate the venv, call `.venv\Scripts\cyberrag` or `.venv/bin/cyberrag` directly.)

**To get the exact versions this project was tested with** (recommended, so a future release of a dependency can't break the install), use the lock file instead of the `pip install` line above:

```bash
pip install -r requirements.lock
pip install -e . --no-deps
```

### 3. Configure (optional)

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Everything in `.env` is optional. A free **NVD API key** ([request one here](https://nvd.nist.gov/developers/request-an-api-key)) raises NVD's API rate limit tenfold, which speeds up `cyberrag sync` and live CVE lookups.

### 4. Start the services

```bash
docker compose up -d                 # Qdrant on 127.0.0.1:6333
ollama pull llama3.2:3b              # the LLM (2.0 GB)
ollama pull nomic-embed-text         # the embedding model (274 MB)
cyberrag doctor                      # checks everything is reachable
```

`doctor` should show **yes** for Python, LLM, Embedder and Vector store. On a new machine the data rows show `-` until step 5.

### 5. Load the data: build it, or copy it from another machine

#### Option A: build from scratch

```bash
cyberrag fetch cwe      # ~2 min: 944 CWE weaknesses
cyberrag index cwe      # 5,160 sections: ~25 min on a laptop CPU with the default embedder
cyberrag fetch cve      # ~400 MB: all ~412k CVEs from the NVD yearly feeds
cyberrag sync           # CVE changes since the feeds were built (via the NVD API)
cyberrag index cve      # the slow step: see below
```

Every step **can be stopped (Ctrl+C) and resumed** by running the same command again. Progress is saved after each file of 2,000 CVEs. Downloads also resume, because NVD's server sometimes drops long transfers.

`fetch cve` uses NVD's yearly bulk files by default. How long it takes depends on your connection; it was 10–40 minutes during testing. To page through the NVD API instead, use `cyberrag fetch cve --method api`. That takes 1–10 hours depending on how busy NVD is (during testing, NVD needed about 3 minutes per page of 2,000 CVEs).

`index cve` embeds every CVE, newest year first, so a partial index already covers recent CVEs. **Embedding speed decides how long this takes.** Measure your machine first:

```bash
cyberrag index cve --limit 2000     # indexes 2,000 CVEs and prints docs/s
```

With the default `nomic-embed-text` on a laptop CPU (~5 CVEs/s), the full NVD takes about a day. **On a machine without a GPU, switch to the faster `arctic-embed-xs` profile before indexing** (~30 CVEs/s, ~3.5 hours, with the same results on the CWE evaluation):

```bash
ollama pull snowflake-arctic-embed:22m
# then set  embedding_profile: arctic-embed-xs  in config/default.yaml
# (or CYBERRAG__EMBEDDING_PROFILE=arctic-embed-xs in .env)
cyberrag index cwe && cyberrag index cve
```

You can ask questions as soon as `index cwe` is done. CVE questions get better as more CVEs are indexed, and any CVE you mention by ID is fetched live from NVD if it isn't indexed yet.

#### Option B: copy a finished index from another machine (skips hours of embedding)

On the machine that already has the index:

```bash
cyberrag snapshot export            # writes data/snapshots/<timestamp>/
```

Copy that whole folder (it holds `manifest.json` and one `.snapshot` file per collection, a few GB for the full CVE index) into `data/snapshots/` on the new machine. Then run:

```bash
cyberrag snapshot import            # loads the newest folder in data/snapshots/
cyberrag doctor                     # Index rows now show document counts
```

Both machines must use the same `embedding_profile`; the import checks this and stops with a clear message if they differ. Use `--only cwe` or `--only cve` to export just one index, and `snapshot import --force` to replace an existing one.

### 6. Ask questions

```bash
cyberrag ask "How do I prevent SQL injection?"
cyberrag ask "What is Log4Shell and which versions are affected?"
cyberrag ask "Remote code execution in VPN appliances" --severity CRITICAL --kev --since 2023
cyberrag ask "Which CWE does CVE-2021-44228 map to?" --show-context
cyberrag chat                       # interactive; /clear resets, /exit quits
```

In `chat`, a follow-up that names no ID ("how would I detect it?") stays on the CVEs and CWEs of the previous exchange.

| Option | Effect |
|---|---|
| `--severity CRITICAL` | Only CVEs with this CVSS severity (repeatable) |
| `--kev` | Only CVEs in CISA's Known Exploited Vulnerabilities list |
| `--since 2023` | Only CVEs published in or after this year |
| `--product log4j` | Only CVEs affecting this product |
| `--llm qwen3-8b` | Use another LLM profile for this question |
| `--show-context` | Print what was retrieved and sent to the LLM |
| `--json` | Print the full answer, sources and timings as JSON |

Clear phrases in the question work as filters too: "critical vulnerabilities", "actively exploited", "in 2024", "since 2022".

If an answer names a CVE or CWE that wasn't in the retrieved data, the CLI prints a "Not in the retrieved data" warning. Small models sometimes invent IDs, so check those before relying on them. The same list is in the `unsupported` field of `--json` and API answers.

## REST API

```bash
cyberrag serve                      # http://127.0.0.1:8000, interactive docs at /docs
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/query` | Ask a question. Body: `{"question": "...", "filters": {...}, "llm_profile": "...", "stream": false}` |
| `GET` | `/cve/{id}` | One CVE record (from the index, or live from NVD) |
| `GET` | `/cwe/{id}` | All sections of one CWE |
| `GET` | `/health` | Status of the LLM, embedder, Qdrant and index sizes |
| `GET` | `/profiles` | Available LLM profiles |

```bash
# macOS / Linux
curl -s 127.0.0.1:8000/query -H "Content-Type: application/json" \
  -d '{"question": "How is CSRF prevented?"}'
```

```powershell
# Windows PowerShell
Invoke-RestMethod http://127.0.0.1:8000/query -Method Post -ContentType "application/json" `
  -Body '{"question": "How is CSRF prevented?"}'
```

With `"stream": true` the response is server-sent events: one `sources` event, many `token` events, then `done` with `{"cited": [...], "unsupported": [...]}`.

**Before serving beyond your own machine**, set an API key: `CYBERRAG__API__KEY=<long random string>` in `.env`. Every route except `/health` and the docs then requires the header `X-API-Key: <key>`. The interactive docs have an **Authorize** button for it. `cyberrag serve --host 0.0.0.0` warns if no key is set. There is no rate limiting or HTTPS, so put a reverse proxy in front for anything public.

## Changing the model

Each LLM is a small YAML **profile** in `config/llm/`:

```yaml
# config/llm/llama3.2-3b.yaml (the default)
provider: ollama
model: llama3.2:3b
base_url: http://127.0.0.1:11434
temperature: 0.1
num_ctx: 8192        # context window; the retrieval budget scales with it
max_tokens: 1024
```

To try another model:

1. Pull it, e.g. `ollama pull qwen3:8b`.
2. Add a profile: copy a file in `config/llm/` and change `model` (and `num_ctx` if you like).
3. Use it once with `cyberrag ask --llm qwen3-8b "..."`, or make it the default by setting `llm_profile:` in `config/default.yaml` or `CYBERRAG__LLM_PROFILE=qwen3-8b` in `.env`.

`cyberrag profiles` lists the LLM and embedding profiles available. LLM profiles that ship with the project:

| Profile | What it shows |
|---|---|
| `llama3.2-3b` | Default local model via Ollama |
| `qwen3-8b` | A larger local model with a bigger context window |
| `llama3.2-3b-openai-api` | The same model through an OpenAI-compatible API |
| `openai` | A hosted API (needs `OPENAI_API_KEY` and `pip install -e ".[openai]"`) |

The `openai_compat` provider talks to anything with an OpenAI-style API: vLLM, LM Studio, llama.cpp server, OpenAI, Groq, OpenRouter, and so on. Point `base_url` at the server and set `api_key_env` to the name of the environment variable that holds the key.

**Comparing models:** `cyberrag eval --llm <profile>` runs 20 fixed questions and reports retrieval hit rate, MRR and how often the answer cites the right ID. Results are saved in `data/eval/`.

### Changing the embedding model

Embedding models also have profiles, in `config/embeddings/`. Each one sets the model, the server and the task prefixes the model expects. Pick one with `embedding_profile:` in `config/default.yaml` or with `CYBERRAG__EMBEDDING_PROFILE=...` in `.env`.

| Profile | Model | Dims | CPU speed* | Full NVD on CPU* | CWE eval (hit rate / MRR) |
|---|---|---|---|---|---|
| `nomic-embed-text` (default) | nomic-embed-text v1.5, 137M | 768 | ~5 CVEs/s | ~23 hours | 1.00 / 0.96 |
| `arctic-embed-s` | snowflake-arctic-embed:33m | 384 | ~19 CVEs/s | ~6 hours | not measured |
| `arctic-embed-xs` | snowflake-arctic-embed:22m | 384 | ~30 CVEs/s | ~3.5 hours | 1.00 / 1.00 |

\* Measured on an 8-core laptop CPU (Ryzen 7 6800H) with no GPU. With a GPU, every row is much faster.

Each embedding model gets its own Qdrant collections (e.g. `cve__nomic-embed-text`), so models never mix and you can switch back and forth. After switching to a new one, run `cyberrag index cwe` and `cyberrag index cve`. They re-embed from `data/raw/` without downloading again.

For a model that isn't listed, copy a profile, change `model`, and set `doc_prefix`/`query_prefix` to what the model card says (empty if none). Compare it with `cyberrag eval --retrieval-only`.

### Adding a new kind of backend

A new backend is one class and one decorator, with no other changes:

```python
# my_llm.py; then add "my_llm" under `plugins:` in config/default.yaml
from cyberrag.providers import register_llm
from cyberrag.providers.llm.base import BaseLLM

@register_llm("my_backend")
class MyLLM(BaseLLM):
    def stream(self, messages):
        yield "..."            # call your model here
    def health(self):
        return True, "ok"
```

Embedders (`register_embedder`) and vector stores (`register_store`) work the same way.

## Keeping CVE data up to date

```bash
cyberrag sync       # downloads CVEs added or changed since the last fetch/sync, then indexes them
```

NVD revises a lot of records, often thousands per day, so a weekly sync can bring in tens of thousands of updated CVEs. `sync` indexes them only if a CVE index already exists; it never starts the multi-hour first index on its own. If NVD marks a CVE as rejected, it is removed from the index.

To run it automatically every night at 03:00 (Docker and Ollama must be running at that time):

```powershell
# Windows: run once in PowerShell, with your own path to the project
schtasks /Create /SC DAILY /ST 03:00 /TN "CyberRAG sync" /TR "\"C:\path\to\RAG\.venv\Scripts\cyberrag.exe\" sync"
```

```bash
# macOS / Linux: add this line with `crontab -e`
0 3 * * * cd /path/to/RAG && .venv/bin/cyberrag sync >> data/sync.log 2>&1
```

MITRE publishes a new CWE release a few times a year. To pick it up, run `cyberrag fetch cwe` and `cyberrag index cwe`.

## Configuration

Settings come from `config/default.yaml`. Any value can be overridden with an environment variable (or a line in `.env`) named `CYBERRAG__<SECTION>__<KEY>`:

| Setting | Default | Override example |
|---|---|---|
| LLM profile | `llama3.2-3b` | `CYBERRAG__LLM_PROFILE=qwen3-8b` |
| A field of the default profile | (from the profile) | `CYBERRAG__LLM__NUM_CTX=16384` |
| Embedding profile | `nomic-embed-text` | `CYBERRAG__EMBEDDING_PROFILE=arctic-embed-xs` |
| A field of the embedding profile | (from the profile) | `CYBERRAG__EMBEDDINGS__BATCH_SIZE=32` |
| Ollama address (embeddings) | `http://127.0.0.1:11434` | `CYBERRAG__EMBEDDINGS__BASE_URL=http://gpu-box:11434` |
| Qdrant address | `http://127.0.0.1:6333` | `CYBERRAG__VECTOR_STORE__URL=http://qdrant:6333` |
| Results per search | 8 CVEs, 4 CWE sections | `CYBERRAG__RETRIEVAL__TOP_K_CVE=12` |
| Live NVD lookup | on | `CYBERRAG__SOURCES__NVD__LIVE_LOOKUP=false` |
| NVD API key | none | `NVD_API_KEY=...` |
| REST API key | none (open on 127.0.0.1) | `CYBERRAG__API__KEY=...` |

If Ollama runs on another machine, change `base_url` in both your LLM profile and your embedding profile. `CYBERRAG_HOME` points the CLI at the project folder when you run it from somewhere else.

## Project layout

```text
config/default.yaml       all settings
config/llm/*.yaml         one file per LLM (swap models here)
config/embeddings/*.yaml  one file per embedding model
prompts/                  system prompt and answer template
src/cyberrag/
  config.py               settings loading (YAML + .env + environment)
  providers/              LLM and embedding adapters + registry (ollama, openai_compat)
  store/                  vector store interface + Qdrant implementation
  sources/                NVD API + yearly feeds, CWE API (rate limits, retries, streaming)
  ingest/                 fetch -> normalize -> embed -> index, with checkpoints
  retrieval/              ID extraction, keyword vectors, hybrid retrieval, context budget
  generation/rag.py       the question -> answer pipeline
  api/app.py              FastAPI app
  cli.py                  the `cyberrag` command
  snapshots.py            export/import of the index
eval/questions.yaml       evaluation questions
tests/                    unit tests (no services needed) + integration tests
requirements.lock         exact tested versions
.github/workflows/        CI: lint + unit tests on Windows, macOS and Linux
CLAUDE.md                 guidance for AI coding assistants working on this repo
data/                     downloaded data, checkpoints, snapshots (git-ignored)
```

## Development

```bash
pip install -r requirements.lock && pip install -e . --no-deps   # or: pip install -e ".[api,openai,dev]"
pytest                      # unit tests: no Ollama, Qdrant or internet needed
pytest -m integration       # end-to-end tests against the running services
ruff check src tests
```

On GitHub, every push runs lint and the unit tests on Windows, macOS and Linux with Python 3.11 and 3.13 (`.github/workflows/tests.yml`).

## Known limitations

- **Answer quality is limited by the model.** `llama3.2:3b` is small: it sometimes gives shallow answers, adds links that don't exist, or stresses the wrong point. Use `--show-context` to see what it was given, or try a larger profile.
- **No counting or statistics.** RAG retrieves the most relevant documents; it can't answer "how many critical CVEs were published in 2024?".
- **CVSS scores are mixed.** The `severity` field uses the newest CVSS version NVD has for each CVE: about 30% of 2026 CVEs use CVSS 4.0, most others 3.1. 2–8% of CVEs per year have no score at all, and `--severity` leaves those out.
- **NVD analyses only some CVEs.** About 37% of 2025–26 CVEs are "Deferred" by NVD, so their affected products come only from the reporting organisation's data. Product lists are cut at 60 per CVE.
- **Limited sources.** Only NVD CVEs and CWE weaknesses. There's no CAPEC, ATT&CK, EPSS, full CISA KEV catalog or exploit data.
- **Updates are manual** unless you schedule `cyberrag sync` (see above).
- **Slow on CPU.** Answers take 20–35 s on a laptop CPU, and building the full CVE index takes hours (see [Changing the embedding model](#changing-the-embedding-model)).
- **Context size is estimated** at 3.5 characters per token. If a server's real context window is smaller than the profile's `num_ctx`, the prompt is cut off without an error (see `config/llm/llama3.2-3b-openai-api.yaml`).
- **The API is minimal:** an optional key, no rate limiting, no HTTPS.
- **Testing:** the full workflow was tested on Windows 11. macOS and Linux are covered by CI unit tests only.
- **Snapshots upload in one piece.** A multi-GB snapshot import needs a stable connection to Qdrant.

## Troubleshooting

| Problem | Fix |
|---|---|
| `doctor`: cannot reach Ollama | Start the Ollama app (or run `ollama serve`) |
| `doctor`: model not pulled | Run the `ollama pull ...` command it prints |
| `doctor`: cannot reach Qdrant | `docker compose up -d`; check `docker ps` and that port 6333 is free |
| NVD warnings `HTTP 403` / `503`, retrying | Normal under NVD's rate limit; the client waits and retries. Add `NVD_API_KEY` to go faster |
| "built with a different embedder" | You changed the embedding model; run `cyberrag index cwe --rebuild` (and `cve`) |
| "Nothing is indexed for embedder ..." | Run `cyberrag fetch cwe` and `cyberrag index cwe` first |
| Answers are vague or miss details | Try a bigger model (`--llm qwen3-8b`), or `--show-context` to see what was retrieved |
| Every question takes 10+ seconds before the answer starts | Use `127.0.0.1` rather than `localhost` in URLs. On Windows, `localhost` adds ~2 s to each Qdrant call (the defaults already use `127.0.0.1`) |
| `py` / `python` not found on Windows | Install Python from python.org and tick "Add to PATH" |
| PowerShell won't run `Activate.ps1` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

Qdrant data lives in a named Docker volume (`rag_qdrant_storage`). `docker compose down` keeps it; `docker compose down -v` deletes it. Don't switch it to a folder bind mount on Windows or macOS, because Qdrant storage on those file systems is slow and can get corrupted.

## Data sources and terms

- This product uses the NVD API but is not endorsed or certified by the NVD. See the [NVD terms of use](https://nvd.nist.gov/developers/terms-of-use).
- CWE™ content is © The MITRE Corporation and used under the [CWE terms of use](https://cwe.mitre.org/about/termsofuse.html).
- The system is intended for defensive security work. Always check critical details against the linked sources.

## License

Apache 2.0. See [LICENSE](LICENSE).
