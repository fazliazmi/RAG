"""Download raw data to data/raw/. Raw files are the source of truth: changing
the embedder only re-embeds them, it never re-downloads.

  data/raw/cwe/cwe-weaknesses-<version>.json.gz
  data/raw/nvd/feeds/<year>/page-<n>.json.gz              (full download from yearly feeds)
  data/raw/nvd/full/page-<startIndex>.json.gz            (full download through the API)
  data/raw/nvd/updates/<run time>/w<NN>-page-<n>.json.gz  (incremental syncs)

Every CVE file has the API page shape {"vulnerabilities": [{"cve": {...}}, ...]}
with at most `page_size` records, so indexing can checkpoint after each file.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cyberrag.config import Settings
from cyberrag.errors import CyberRAGError
from cyberrag.ingest.state import load_state, save_state, write_json_gz
from cyberrag.sources import nvd_feeds
from cyberrag.sources.cwe import CWEClient
from cyberrag.sources.http import make_http_client
from cyberrag.sources.nvd import MAX_DATE_RANGE_DAYS, NVDClient, nvd_datetime

log = logging.getLogger("cyberrag.fetch")

Progress = Callable[[int, int], None]  # (records done, records total)

NVD_FETCH_STATE = "nvd_fetch"
NVD_FEEDS_STATE = "nvd_feeds"
NVD_SYNC_STATE = "nvd_sync"
CWE_FETCH_STATE = "cwe_fetch"
_FEED_SYNC_MARGIN = timedelta(days=1)


@dataclass
class FetchResult:
    files: list[Path] = field(default_factory=list)
    records: int = 0
    total: int = 0
    message: str = ""


def nvd_client(settings: Settings) -> NVDClient:
    return NVDClient(settings.sources.nvd, api_key=settings.nvd_key)


# ---- CWE -------------------------------------------------------------------
def cwe_dir(settings: Settings) -> Path:
    return settings.raw_dir / "cwe"


def latest_cwe_file(settings: Settings) -> Path | None:
    state = load_state(settings, CWE_FETCH_STATE)
    if state.get("file") and (settings.raw_dir / state["file"]).is_file():
        return settings.raw_dir / state["file"]
    files = sorted(cwe_dir(settings).glob("cwe-weaknesses-*.json.gz"))
    return files[-1] if files else None


def fetch_cwe(
    settings: Settings, force: bool = False, progress: Progress | None = None
) -> FetchResult:
    client = CWEClient(settings.sources.cwe)
    try:
        version = client.version()
        content_version = str(version.get("ContentVersion", "unknown"))
        path = cwe_dir(settings) / f"cwe-weaknesses-{content_version}.json.gz"
        if path.is_file() and not force:
            return FetchResult(
                files=[path], total=int(version.get("TotalWeaknesses", 0)),
                message=f"CWE {content_version} already downloaded",
            )
        weaknesses = client.all_weaknesses(progress=progress)
    finally:
        client.close()
    if not weaknesses:
        raise CyberRAGError("The CWE API returned no weaknesses; try again later")
    write_json_gz(path, {"version": version, "Weaknesses": weaknesses})
    save_state(settings, CWE_FETCH_STATE, {
        "version": content_version,
        "file": path.relative_to(settings.raw_dir).as_posix(),
        "count": len(weaknesses),
        "fetched_at": datetime.now(UTC).isoformat(),
    })
    return FetchResult(files=[path], records=len(weaknesses), total=len(weaknesses),
                       message=f"Downloaded CWE {content_version}: {len(weaknesses)} weaknesses")


# ---- CVE -------------------------------------------------------------------
def nvd_full_dir(settings: Settings) -> Path:
    return settings.raw_dir / "nvd" / "full"


def nvd_updates_dir(settings: Settings) -> Path:
    return settings.raw_dir / "nvd" / "updates"


def nvd_feeds_dir(settings: Settings) -> Path:
    return settings.raw_dir / "nvd" / "feeds"


def cve_raw_files(settings: Settings) -> list[Path]:
    """All downloaded CVE pages in indexing order.

    Yearly feeds come newest year first (each CVE is in exactly one yearly feed), so a
    partly built index already covers recent CVEs. API pages and sync updates follow,
    oldest first, so the newest version of a record is written last and wins."""
    files: list[Path] = []
    for year_dir in sorted((p for p in nvd_feeds_dir(settings).glob("*") if p.is_dir()), reverse=True):
        files.extend(sorted(year_dir.glob("page-*.json.gz")))
    files.extend(sorted(nvd_full_dir(settings).glob("page-*.json.gz")))
    for run in sorted(p for p in nvd_updates_dir(settings).glob("*") if p.is_dir()):
        files.extend(sorted(run.glob("*.json.gz")))
    return files


def fetch_cve(
    settings: Settings, restart: bool = False, progress: Progress | None = None
) -> FetchResult:
    """Full CVE download using the method set in sources.nvd.full_download."""
    if settings.sources.nvd.full_download == "api":
        return fetch_cve_full(settings, restart=restart, progress=progress)
    return fetch_cve_feeds(settings, force=restart, progress=progress)


def _split_feed(src: Path, dest_dir: Path, page_size: int) -> tuple[list[Path], int]:
    """Stream a yearly feed into API-shaped page files of `page_size` records."""
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    files: list[Path] = []
    batch: list[dict] = []
    count = 0

    def flush() -> None:
        path = dest_dir / f"page-{count - len(batch):07d}.json.gz"
        write_json_gz(path, {"vulnerabilities": batch})
        files.append(path)

    for item in nvd_feeds.iter_array(src):
        batch.append(item)
        count += 1
        if len(batch) == page_size:
            flush()
            batch = []
    if batch:
        flush()
    return files, count


def fetch_cve_feeds(
    settings: Settings,
    force: bool = False,
    progress: Progress | None = None,
    now: datetime | None = None,
) -> FetchResult:
    """Download the NVD yearly feeds. Years whose feed has not changed since the last
    run are skipped. Afterwards `cyberrag sync` fetches anything newer via the API."""
    cfg = settings.sources.nvd
    state = {} if force else load_state(settings, NVD_FEEDS_STATE)
    years = list(range(nvd_feeds.FIRST_YEAR, (now or datetime.now(UTC)).year + 1))
    result = FetchResult(total=len(years))
    feed_dates: list[datetime] = []
    downloads = settings.raw_dir / "nvd" / "_downloads"
    with make_http_client(cfg.timeout_s) as client:
        for done, year in enumerate(years, 1):
            meta = nvd_feeds.get_meta(client, cfg.feeds_url, year)
            if meta is None:
                continue
            feed_dates.append(nvd_feeds.meta_datetime(meta))
            previous = state.get(str(year), {})
            year_dir = nvd_feeds_dir(settings) / str(year)
            unchanged = (
                previous.get("last_modified") == meta.get("lastModifiedDate")
                and previous.get("gz_size") == meta.get("gzSize")
                and year_dir.is_dir()
            )
            if not unchanged:
                archive = downloads / f"{nvd_feeds.feed_name(year)}.json.gz"
                nvd_feeds.download(client, cfg.feeds_url, year, archive, int(meta.get("gzSize", 0)))
                files, count = _split_feed(archive, year_dir, cfg.page_size)
                archive.unlink(missing_ok=True)
                state[str(year)] = {
                    "last_modified": meta.get("lastModifiedDate"),
                    "gz_size": meta.get("gzSize"),
                    "records": count,
                    "pages": len(files),
                }
                save_state(settings, NVD_FEEDS_STATE, state)
                result.files.extend(files)
                result.records += count
            if progress:
                progress(done, len(years))
    shutil.rmtree(downloads, ignore_errors=True)
    if not feed_dates:
        raise CyberRAGError(f"No NVD feeds found at {cfg.feeds_url}")

    sync = load_state(settings, NVD_SYNC_STATE)
    if not sync.get("last_sync"):
        # NVD rebuilds a yearly feed only when a CVE of that year changes, so the whole
        # set is current as of the newest rebuild (an old year's feed can be months old
        # simply because nothing in it changed). Start syncing a little before that.
        sync["last_sync"] = (max(feed_dates) - _FEED_SYNC_MARGIN).astimezone(UTC).isoformat()
        save_state(settings, NVD_SYNC_STATE, sync)
    total = sum(int(v.get("records", 0)) for v in state.values())
    result.message = (
        f"{result.records:,} CVEs downloaded ({total:,} total from yearly feeds). "
        "Run `cyberrag sync` to add changes newer than the feeds."
    )
    return result


def _parse_nvd_timestamp(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def fetch_cve_full(
    settings: Settings, restart: bool = False, progress: Progress | None = None
) -> FetchResult:
    """Download every CVE, one page per file. Safe to stop and re-run: it resumes
    from the last saved page."""
    state = {} if restart else load_state(settings, NVD_FETCH_STATE)
    if state.get("completed"):
        return FetchResult(total=state.get("total", 0),
                           message="Full CVE download already complete (use --restart to redo)")
    start = int(state.get("next_start", 0))
    started_at = state.get("started_at")
    result = FetchResult()
    client = nvd_client(settings)
    try:
        for start_index, data in client.iter_pages(start):
            vulns = data.get("vulnerabilities") or []
            total = int(data.get("totalResults", 0))
            started_at = started_at or data.get("timestamp")
            if vulns:
                path = nvd_full_dir(settings) / f"page-{start_index:07d}.json.gz"
                write_json_gz(path, data)
                result.files.append(path)
            result.records += len(vulns)
            result.total = total
            save_state(settings, NVD_FETCH_STATE, {
                "next_start": start_index + len(vulns),
                "total": total,
                "started_at": started_at,
                "completed": False,
            })
            if progress:
                progress(start_index + len(vulns), total)
    finally:
        client.close()

    save_state(settings, NVD_FETCH_STATE, {
        "next_start": result.total, "total": result.total,
        "started_at": started_at, "completed": True,
    })
    sync = load_state(settings, NVD_SYNC_STATE)
    if not sync.get("last_sync") and started_at:
        # Anything changed after the download began will be picked up by `sync`.
        sync["last_sync"] = _parse_nvd_timestamp(started_at).isoformat()
        save_state(settings, NVD_SYNC_STATE, sync)
    result.message = f"Downloaded {result.records} CVE records in {len(result.files)} pages"
    return result


def sync_cve(
    settings: Settings, progress: Progress | None = None, now: datetime | None = None
) -> FetchResult:
    """Download CVEs added or changed since the last full download or sync."""
    sync = load_state(settings, NVD_SYNC_STATE)
    if not sync.get("last_sync"):
        raise CyberRAGError("No previous CVE download found. Run: cyberrag fetch cve")
    since = _parse_nvd_timestamp(sync["last_sync"])
    until = now or datetime.now(UTC)
    run_dir = nvd_updates_dir(settings) / until.strftime("%Y%m%dT%H%M%SZ")
    result = FetchResult()
    client = nvd_client(settings)
    try:
        window, w_start = 0, since
        while w_start < until:
            w_end = min(w_start + timedelta(days=MAX_DATE_RANGE_DAYS), until)
            params = {"lastModStartDate": nvd_datetime(w_start),
                      "lastModEndDate": nvd_datetime(w_end)}
            for start_index, data in client.iter_pages(0, **params):
                vulns = data.get("vulnerabilities") or []
                if vulns:
                    path = run_dir / f"w{window:02d}-page-{start_index:07d}.json.gz"
                    write_json_gz(path, data)
                    result.files.append(path)
                result.records += len(vulns)
                result.total = max(result.total, int(data.get("totalResults", 0)))
                if progress:
                    progress(result.records, max(result.total, result.records))
            window, w_start = window + 1, w_end
    finally:
        client.close()
    sync["last_sync"] = until.isoformat()
    save_state(settings, NVD_SYNC_STATE, sync)
    result.message = f"{result.records} CVEs added or changed since {since:%Y-%m-%d %H:%M} UTC"
    return result
