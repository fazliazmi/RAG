"""Command-line interface: `cyberrag --help`."""

from __future__ import annotations

import functools
import json
import logging
import platform
import sys
import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from cyberrag import __version__
from cyberrag.config import Settings, get_settings
from cyberrag.errors import CyberRAGError
from cyberrag.generation.rag import mark_cited, unsupported_ids
from cyberrag.schemas import Message, Source

app = typer.Typer(
    help="CyberRAG: ask questions about CVE vulnerabilities and CWE weaknesses.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
snapshot_app = typer.Typer(help="Move a built index to another machine.", no_args_is_help=True)
app.add_typer(snapshot_app, name="snapshot")

console = Console()
F = TypeVar("F", bound=Callable[..., Any])


class SourceName(StrEnum):
    cve = "cve"
    cwe = "cwe"


def friendly_errors(fn: F) -> F:
    """Show CyberRAGError as a one-line message instead of a traceback."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except CyberRAGError as exc:
            console.print(f"[bold red]Error:[/] {escape(str(exc))}")
            raise typer.Exit(1) from None
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped.[/] Progress is saved; run the same command to resume.")
            raise typer.Exit(130) from None

    return wrapper  # type: ignore[return-value]


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show debug logs.")] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=False)],
    )
    for noisy in ("httpx", "httpcore", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[rate]}"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        console=console,
    )


def _sources_table(sources: list[Source], title: str = "Sources") -> Table:
    table = Table(title=title, show_lines=False, title_justify="left")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Title")
    table.add_column("Found by", style="dim")
    table.add_column("Link", style="dim")
    for s in sources:
        table.add_row(s.ref, escape(s.title[:70]), s.via, s.url)
    return table


# ---- setup ----------------------------------------------------------------
@app.command()
@friendly_errors
def doctor() -> None:
    """Check that Ollama, the models and Qdrant are reachable."""
    from cyberrag.factory import make_embedder, make_llm, make_store
    from cyberrag.ingest.fetch import cve_raw_files, latest_cwe_file

    s = get_settings()
    table = Table(title=f"CyberRAG {__version__} doctor", title_justify="left")
    table.add_column("Check")
    table.add_column("OK", justify="center")
    table.add_column("Detail")
    failures = 0

    def row(name: str, ok: bool | None, detail: str) -> None:
        nonlocal failures
        mark = "[green]yes[/]" if ok else ("[yellow]-[/]" if ok is None else "[red]no[/]")
        failures += ok is False
        table.add_row(escape(name), mark, escape(detail))

    row("Python", sys.version_info >= (3, 11), f"{platform.python_version()} ({sys.executable})")
    row("Project folder", (s.root / "config" / "default.yaml").is_file(), str(s.root))
    try:
        llm = make_llm(s)
        row(f"LLM [{s.llm_profile}]", *llm.health())
    except CyberRAGError as exc:
        row(f"LLM [{s.llm_profile}]", False, str(exc))
    embedder = make_embedder(s)
    emb_ok, emb_detail = embedder.health()
    row(f"Embedder [{s.embedding_profile or 'custom'}]", emb_ok, f"{embedder.name}: {emb_detail}")
    store = make_store(s)
    store_ok, store_detail = store.health()
    row("Vector store", store_ok, store_detail)
    if store_ok:
        for kind in ("cwe", "cve"):
            name = store.collection_name(kind, embedder.slug)  # type: ignore[arg-type]
            count = store.count(name)
            row(f"Index {kind}", count > 0 or None, f"{name}: {count:,} documents")
    cwe_file = latest_cwe_file(s)
    row("Raw CWE data", bool(cwe_file) or None, str(cwe_file) if cwe_file else "not downloaded")
    pages = cve_raw_files(s)
    row("Raw CVE data", bool(pages) or None, f"{len(pages)} page files" if pages else "not downloaded")
    row("NVD API key", bool(s.nvd_key) or None,
        "set" if s.nvd_key else "not set (optional: speeds up sync and live CVE lookups)")
    console.print(table)
    if failures:
        raise typer.Exit(1)


@app.command()
@friendly_errors
def profiles() -> None:
    """List LLM profiles (config/llm/, switch with --llm NAME) and embedding profiles
    (config/embeddings/, switch with embedding_profile in config/default.yaml)."""
    s = get_settings()
    table = Table(title="LLM profiles", title_justify="left")
    for col in ("Profile", "Provider", "Model", "Context", ""):
        table.add_column(col)
    for name in s.list_llm_profiles():
        try:
            cfg = s.llm_config(name)
            table.add_row(name, cfg.provider, cfg.model, f"{cfg.num_ctx:,}",
                          "[green]default[/]" if name == s.llm_profile else "")
        except Exception as exc:  # noqa: BLE001
            table.add_row(name, "[red]invalid[/]", escape(str(exc)), "", "")
    console.print(table)

    emb = Table(title="Embedding profiles", title_justify="left")
    for col in ("Profile", "Provider", "Model", ""):
        emb.add_column(col)
    for name in s.list_embedding_profiles():
        try:
            cfg = Settings(home=s.root, embedding_profile=name).embeddings
            emb.add_row(name, cfg.provider, cfg.model,
                        "[green]active[/]" if name == s.embedding_profile else "")
        except Exception as exc:  # noqa: BLE001
            emb.add_row(name, "[red]invalid[/]", escape(str(exc)), "")
    console.print(emb)


# ---- data -----------------------------------------------------------------
@app.command()
@friendly_errors
def fetch(
    source: Annotated[SourceName, typer.Argument(help="cve or cwe")],
    restart: Annotated[bool, typer.Option(help="CVE: start the full download over.")] = False,
    force: Annotated[bool, typer.Option(help="CWE: download again even if present.")] = False,
    method: Annotated[str | None, typer.Option(
        help="CVE: 'feeds' (yearly bulk files, minutes) or 'api' (paged API, hours). "
             "Default from config.")] = None,
) -> None:
    """Download raw data from NVD (cve) or MITRE (cwe) into data/raw/."""
    from cyberrag.ingest.fetch import fetch_cve, fetch_cwe

    s = get_settings()
    if source is SourceName.cwe:
        with _progress() as progress:
            task = progress.add_task("Fetching CWEs", total=None, rate="")
            result = fetch_cwe(s, force=force, progress=lambda d, t: progress.update(
                task, completed=d, total=t))
        console.print(f"[green]{result.message}[/]")
        return

    if method:
        if method not in ("feeds", "api"):
            raise CyberRAGError("--method must be 'feeds' or 'api'")
        s.sources.nvd.full_download = method  # type: ignore[assignment]
    by_api = s.sources.nvd.full_download == "api"
    if by_api:
        console.print("Downloading through the NVD API. This takes 1-10 hours depending on NVD "
                      "load (an NVD_API_KEY helps). It can be stopped and resumed.")
    with _progress() as progress:
        label = "Fetching CVEs (API)" if by_api else "Fetching CVE feeds (years)"
        task = progress.add_task(label, total=None, rate="")

        def update(done: int, total: int) -> None:
            progress.update(task, completed=done, total=total, rate="")

        result = fetch_cve(s, restart=restart, progress=update)
    console.print(f"[green]{result.message}[/]")


@app.command()
@friendly_errors
def index(
    source: Annotated[SourceName, typer.Argument(help="cve or cwe")],
    limit: Annotated[int | None, typer.Option(help="CVE: stop after N documents (benchmark).")] = None,
    rebuild: Annotated[bool, typer.Option(help="Delete the collection and re-embed everything.")] = False,
    batch_size: Annotated[int | None, typer.Option(help="Override embeddings.batch_size.")] = None,
) -> None:
    """Embed downloaded data into the vector store (resumable)."""
    from cyberrag.factory import make_embedder, make_store
    from cyberrag.ingest.indexer import Indexer

    s = get_settings()
    if batch_size:
        s.embeddings.batch_size = batch_size
    embedder = make_embedder(s)
    ok, detail = embedder.health()
    if not ok:
        raise CyberRAGError(f"Embedder not ready: {detail}")
    indexer = Indexer(s, embedder, make_store(s))

    with _progress() as progress:
        task = progress.add_task(f"Indexing {source.value.upper()}", total=None, rate="")
        state = {"done": 0, "t0": None}

        def on_start(total: int) -> None:
            state["t0"] = time.perf_counter()
            progress.update(task, total=total)

        def on_docs(n: int) -> None:
            state["done"] += n
            elapsed = time.perf_counter() - (state["t0"] or time.perf_counter())
            rate = f"{state['done'] / elapsed:.1f} docs/s" if elapsed > 0 else ""
            progress.update(task, advance=n, rate=rate)

        if source is SourceName.cwe:
            stats = indexer.index_cwe(rebuild=rebuild, on_start=on_start, progress=on_docs)
        else:
            stats = indexer.index_cve(limit=limit, rebuild=rebuild, on_start=on_start,
                                      progress=on_docs)
    console.print(f"[green]{stats.message}[/]")


@app.command()
@friendly_errors
def sync(
    no_index: Annotated[bool, typer.Option("--no-index", help="Only download.")] = False,
) -> None:
    """Download CVEs added or changed since the last fetch/sync, then index them."""
    from cyberrag.ingest.fetch import sync_cve

    s = get_settings()
    with _progress() as progress:
        task = progress.add_task("Syncing CVEs", total=None, rate="")
        result = sync_cve(s, progress=lambda d, t: progress.update(task, completed=d, total=t))
    console.print(f"[green]{result.message}[/]")
    if not result.files or no_index:
        return
    from cyberrag.factory import make_embedder, make_store

    store, embedder = make_store(s), make_embedder(s)
    if store.count(store.collection_name("cve", embedder.slug)) == 0:
        # Nothing indexed yet: don't silently start the multi-hour first index.
        console.print("No CVE index yet. Run [bold]cyberrag index cve[/] to build it.")
        return
    index(SourceName.cve, limit=None, rebuild=False, batch_size=None)


@app.command()
@friendly_errors
def stats() -> None:
    """Show what is downloaded and indexed."""
    from cyberrag.factory import make_embedder, make_store
    from cyberrag.ingest.fetch import NVD_FEEDS_STATE, NVD_FETCH_STATE, NVD_SYNC_STATE, cve_raw_files
    from cyberrag.ingest.state import load_state

    s = get_settings()
    store = make_store(s)
    embedder = make_embedder(s)
    table = Table(title="Collections", title_justify="left")
    for col in ("Collection", "Documents", "Embedder", "Created"):
        table.add_column(col)
    for name in store.list_collections():
        meta = store.metadata(name)
        marker = " [green](active)[/]" if name.endswith(f"__{embedder.slug}") else ""
        table.add_row(name + marker, f"{store.count(name):,}", str(meta.get("embedder", "?")),
                      str(meta.get("created_at", ""))[:19])
    console.print(table)
    feeds = load_state(s, NVD_FEEDS_STATE)
    api_state = load_state(s, NVD_FETCH_STATE)
    console.print(f"CVE raw page files: {len(cve_raw_files(s))}")
    if feeds:
        records = sum(int(v.get("records", 0)) for v in feeds.values())
        console.print(f"Yearly feeds: {len(feeds)} years, {records:,} CVEs")
    if api_state:
        console.print(f"API download: {'complete' if api_state.get('completed') else 'incomplete'} "
                      f"({api_state.get('next_start', 0):,}/{api_state.get('total', 0):,})")
    console.print(f"Last CVE sync: {load_state(s, NVD_SYNC_STATE).get('last_sync', 'never')}")


# ---- questions ------------------------------------------------------------
def _filters(
    severity: list[str] | None, kev: bool, since: int | None, product: str | None
) -> dict[str, Any] | None:
    f: dict[str, Any] = {}
    if severity:
        f["severity"] = [x.upper() for x in severity]
    if kev:
        f["known_exploited"] = True
    if since:
        f["published"] = {"gte": f"{since}-01-01T00:00:00Z"}
    if product:
        f["products"] = [product.lower()]
    return f or None


LlmOpt = Annotated[str | None, typer.Option("--llm", help="LLM profile from config/llm/.")]
SeverityOpt = Annotated[list[str] | None, typer.Option("--severity", "-s",
                                                       help="CRITICAL/HIGH/MEDIUM/LOW (repeatable).")]
KevOpt = Annotated[bool, typer.Option("--kev", help="Only CVEs in CISA's known-exploited list.")]
SinceOpt = Annotated[int | None, typer.Option("--since", help="Only CVEs published from this year.")]
ProductOpt = Annotated[str | None, typer.Option("--product", help="Only CVEs for this product, "
                                                "e.g. log4j.")]


@app.command()
@friendly_errors
def ask(
    question: Annotated[str, typer.Argument(help="Your question.")],
    llm: LlmOpt = None,
    severity: SeverityOpt = None,
    kev: KevOpt = False,
    since: SinceOpt = None,
    product: ProductOpt = None,
    show_context: Annotated[bool, typer.Option(help="Print the retrieved context.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full answer as JSON.")] = False,
) -> None:
    """Ask one question."""
    from cyberrag.generation.rag import RAGPipeline

    pipeline = RAGPipeline.from_settings(get_settings(), llm)
    filters = _filters(severity, kev, since, product)
    if as_json:
        answer = pipeline.ask(question, filters, include_context=show_context)
        # bytes, not print(): a redirected Windows console may not be UTF-8
        sys.stdout.buffer.write(answer.model_dump_json(indent=2).encode("utf-8") + b"\n")
        sys.stdout.flush()
        return
    with console.status("Searching..."):
        tokens, sources, prep = pipeline.ask_stream(question, filters)
    if show_context:
        console.rule("Context")
        console.print(escape(prep.context))
        console.rule()
    console.print(f"[dim]{pipeline.llm.name} | {len(prep.hits)} documents in context[/]\n")
    parts = []
    for token in tokens:
        parts.append(token)
        console.print(token, end="", markup=False, highlight=False, soft_wrap=True)
    console.print("\n")
    sources = mark_cited(sources, "".join(parts))
    cited = [s for s in sources if s.cited]
    console.print(_sources_table(cited or sources, "Cited sources" if cited else "Sources"))
    others = len(sources) - len(cited)
    if cited and others:
        console.print(f"[dim]+{others} more retrieved but not cited (see --json or --show-context)[/]")
    _warn_unsupported("".join(parts), prep.context)


def _warn_unsupported(answer: str, context: str) -> None:
    unsupported = unsupported_ids(answer, context)
    if unsupported:
        console.print(f"[yellow]Not in the retrieved data:[/] {', '.join(unsupported)}. "
                      "The model named these without a source; check them before relying on them.")


@app.command()
@friendly_errors
def chat(llm: LlmOpt = None) -> None:
    """Interactive chat. Type /exit to quit, /clear to forget the conversation."""
    from cyberrag.generation.rag import RAGPipeline

    pipeline = RAGPipeline.from_settings(get_settings(), llm)
    console.print(f"[bold]CyberRAG chat[/] using {pipeline.llm.name}. /exit to quit, /clear to reset.")
    history: list[Message] = []
    while True:
        try:
            question = console.input("\n[bold green]you>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question in ("/exit", "/quit"):
            break
        if question == "/clear":
            history.clear()
            console.print("[dim]Conversation cleared.[/]")
            continue
        try:
            tokens, sources, prep = pipeline.ask_stream(question, history=history)
            console.print("[bold cyan]cyberrag>[/] ", end="")
            parts = []
            for token in tokens:
                parts.append(token)
                console.print(token, end="", markup=False, highlight=False, soft_wrap=True)
            console.print()
            sources = mark_cited(sources, "".join(parts))
            shown = [s for s in sources if s.cited] or sources
            if shown:
                console.print("[dim]Sources: " + ", ".join(s.ref for s in shown) + "[/]")
            _warn_unsupported("".join(parts), prep.context)
            history += [Message(role="user", content=question),
                        Message(role="assistant", content="".join(parts))]
        except CyberRAGError as exc:
            console.print(f"[red]{escape(str(exc))}[/]")


@app.command("eval")
@friendly_errors
def evaluate(
    llm: LlmOpt = None,
    questions: Annotated[Path | None, typer.Option(help="YAML file of test questions.")] = None,
    retrieval_only: Annotated[bool, typer.Option(help="Skip the LLM; score retrieval only.")] = False,
) -> None:
    """Score retrieval and answers on a fixed question set (compare models with --llm)."""
    from cyberrag.evaluation import load_cases, run_eval, save_results, summarize
    from cyberrag.generation.rag import RAGPipeline

    s = get_settings()
    cases = load_cases(questions or s.root / "eval" / "questions.yaml")
    pipeline = RAGPipeline.from_settings(s, llm)
    console.print(f"Evaluating {len(cases)} questions with {pipeline.llm.name} "
                  f"and {pipeline.retriever.embedder.name}")

    def show(row: Any) -> None:
        if row.skipped:
            console.print(f"[dim]skip[/] {escape(row.question)} ({row.skipped})")
            return
        hit = f"[green]rank {row.rank}[/]" if row.rank else "[red]miss[/]"
        cited = "" if row.cited is None else ("[green]cited[/]" if row.cited else "[red]not cited[/]")
        console.print(f"{hit} {cited} {row.seconds:5.1f}s  {escape(row.question)}")

    rows = run_eval(pipeline, cases, retrieval_only=retrieval_only, on_row=show)
    summary = summarize(rows)
    path = save_results(s, pipeline, rows, summary)
    console.print_json(json.dumps(summary))
    console.print(f"[dim]Saved to {path}[/]")


@app.command()
@friendly_errors
def serve(
    host: Annotated[str | None, typer.Option(help="Default from config (127.0.0.1).")] = None,
    port: Annotated[int | None, typer.Option(help="Default from config (8000).")] = None,
) -> None:
    """Start the REST API (docs at http://HOST:PORT/docs)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise CyberRAGError('The API needs extra packages: pip install -e ".[api]"') from exc
    s = get_settings()
    host = host or s.api.host
    has_key = bool(s.api.key and s.api.key.get_secret_value().strip())
    if host not in ("127.0.0.1", "localhost", "::1"):
        if has_key:
            console.print(f"Serving on {host}: requests need the X-API-Key header.")
        else:
            console.print(f"[bold yellow]Warning:[/] serving on {host} with no API key, so anyone "
                          "who can reach this machine can use the API. Set CYBERRAG__API__KEY.")
    uvicorn.run("cyberrag.api.app:create_app", factory=True, host=host, port=port or s.api.port)


# ---- snapshots ------------------------------------------------------------
@snapshot_app.command("export")
@friendly_errors
def snapshot_export(
    out: Annotated[Path | None, typer.Option(help="Folder to write to.")] = None,
    only: Annotated[SourceName | None, typer.Option(help="Export only cve or only cwe.")] = None,
) -> None:
    """Export the index (and checkpoints) to data/snapshots/<time>/."""
    from cyberrag.factory import make_embedder, make_store
    from cyberrag.snapshots import export_snapshots

    s = get_settings()
    kinds = (only.value,) if only else ("cwe", "cve")
    with console.status("Creating snapshots (large indexes take a few minutes)..."):
        folder = export_snapshots(s, make_embedder(s), make_store(s), out, kinds=kinds)
    console.print(f"[green]Snapshot written to[/] {folder}\nCopy this folder to the other "
                  "machine's data/snapshots/ and run: cyberrag snapshot import")


@snapshot_app.command("import")
@friendly_errors
def snapshot_import(
    from_: Annotated[Path | None, typer.Option("--from", help="Snapshot folder "
                                               "(default: newest in data/snapshots/).")] = None,
    force: Annotated[bool, typer.Option(help="Replace collections that already exist.")] = False,
) -> None:
    """Load an exported index into Qdrant on this machine."""
    from cyberrag.factory import make_embedder, make_store
    from cyberrag.snapshots import import_snapshots

    s = get_settings()
    with console.status("Uploading snapshots to Qdrant..."):
        imported = import_snapshots(s, make_embedder(s), make_store(s), from_, force=force)
    for name, count in imported.items():
        console.print(f"[green]Imported[/] {name}: {count:,} documents")
