import gzip
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx

from cyberrag.errors import CyberRAGError
from cyberrag.ingest.fetch import (
    NVD_FEEDS_STATE,
    NVD_FETCH_STATE,
    NVD_SYNC_STATE,
    cve_raw_files,
    fetch_cve_feeds,
    fetch_cve_full,
    fetch_cwe,
    latest_cwe_file,
    sync_cve,
)
from cyberrag.ingest.state import load_state, read_json_gz, save_state
from cyberrag.sources.nvd_feeds import iter_array

NVD = "https://services.nvd.nist.gov/rest/json/cves/2.0"
CWE = "https://cwe-api.mitre.org/api/v1"


def page(ids, total, start=0):
    return {"totalResults": total, "startIndex": start, "resultsPerPage": len(ids),
            "timestamp": "2026-10-21T03:56:13.268",
            "vulnerabilities": [{"cve": {"id": i}} for i in ids]}


def nvd_router(pages_by_start, fail_first=0):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= fail_first:
            return httpx.Response(503)
        start = int(request.url.params["startIndex"])
        return httpx.Response(200, json=pages_by_start[start])

    return handler, calls


@respx.mock
def test_full_fetch_pages_and_state(settings):
    handler, calls = nvd_router({0: page(["CVE-1", "CVE-2"], 3), 2: page(["CVE-3"], 3, 2)})
    respx.get(NVD).mock(side_effect=handler)
    result = fetch_cve_full(settings)
    assert result.records == 3 and len(result.files) == 2
    assert [p.name for p in cve_raw_files(settings)] == ["page-0000000.json.gz",
                                                         "page-0000002.json.gz"]
    assert load_state(settings, NVD_FETCH_STATE)["completed"] is True
    assert load_state(settings, NVD_SYNC_STATE)["last_sync"].startswith("2026-10-21T03:56:13")
    # already complete: no new requests
    fetch_cve_full(settings)
    assert calls["n"] == 2


@respx.mock
def test_full_fetch_resumes_from_checkpoint(settings):
    save_state(settings, NVD_FETCH_STATE, {"next_start": 2, "total": 3, "completed": False,
                                           "started_at": "2026-10-20T00:00:00"})
    handler, calls = nvd_router({2: page(["CVE-3"], 3, 2)})
    respx.get(NVD).mock(side_effect=handler)
    result = fetch_cve_full(settings)
    assert calls["n"] == 1 and result.records == 1


@respx.mock
def test_retries_when_nvd_is_busy(settings):
    handler, calls = nvd_router({0: page(["CVE-1"], 1)}, fail_first=2)
    respx.get(NVD).mock(side_effect=handler)
    assert fetch_cve_full(settings).records == 1
    assert calls["n"] == 3


@respx.mock
def test_sync_splits_into_120_day_windows(settings):
    now = datetime(2026, 10, 21, tzinfo=UTC)
    save_state(settings, NVD_SYNC_STATE, {"last_sync": (now - timedelta(days=200)).isoformat()})
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.params["lastModStartDate"], request.url.params["lastModEndDate"]))
        return httpx.Response(200, json=page([f"CVE-{len(seen)}"], 1))

    respx.get(NVD).mock(side_effect=handler)
    result = sync_cve(settings, now=now)
    assert len(seen) == 2 and result.records == 2
    assert seen[0][0].endswith("Z") and seen[1][1] == "2026-10-21T00:00:00.000Z"
    assert load_state(settings, NVD_SYNC_STATE)["last_sync"].startswith("2026-10-21")
    # update files sort after the full download
    assert all("updates" in str(p) for p in cve_raw_files(settings))


def test_sync_needs_a_first_download(settings):
    with pytest.raises(CyberRAGError, match="cyberrag fetch cve"):
        sync_cve(settings)


def _tree(cwe_id, *children):
    return {"Data": {"Type": "base_weakness", "ID": str(cwe_id)}, "Children": list(children) or None}


@respx.mock
def test_fetch_cwe_walks_the_view_and_batches(settings):
    respx.get(f"{CWE}/cwe/version").mock(return_value=httpx.Response(
        200, json={"ContentVersion": "4.20", "TotalWeaknesses": 3}))
    respx.get(f"{CWE}/cwe/view/1000").mock(return_value=httpx.Response(
        200, json={"Views": [{"ID": "1000", "Members": [{"CweID": "74"}]}]}))
    respx.get(f"{CWE}/cwe/74/descendants").mock(return_value=httpx.Response(
        200, json=[_tree(74, _tree(79), _tree(89, _tree(564)))]))
    respx.get(f"{CWE}/cwe/weakness/74,79,89,564").mock(return_value=httpx.Response(
        404, text='"for weakness: cwe (564) not found"'))
    for i in (74, 79, 89):
        respx.get(f"{CWE}/cwe/weakness/{i}").mock(return_value=httpx.Response(
            200, json={"Weaknesses": [{"ID": str(i), "Name": f"W{i}"}]}))
    respx.get(f"{CWE}/cwe/weakness/564").mock(return_value=httpx.Response(404, text="nope"))

    result = fetch_cwe(settings)
    assert result.records == 3  # the bad ID is skipped, the rest of its batch is kept
    path = latest_cwe_file(settings)
    assert path is not None and path.name == "cwe-weaknesses-4.20.json.gz"
    assert [w["ID"] for w in read_json_gz(path)["Weaknesses"]] == ["74", "79", "89"]
    assert "already downloaded" in fetch_cwe(settings).message


# ---- yearly feeds ------------------------------------------------------------

FEEDS = "https://nvd.nist.gov/feeds/json/cve/2.0"


def _feed_bytes(ids):
    body = {"resultsPerPage": len(ids), "format": "NVD_CVE", "version": "2.0",
            "vulnerabilities": [{"cve": {"id": i, "descriptions": [{"lang": "en", "value": "x ] { }"}]}}
                                for i in ids]}
    return gzip.compress(json.dumps(body).encode())


def test_iter_array_streams_across_chunk_boundaries(tmp_path):
    path = tmp_path / "feed.json.gz"
    path.write_bytes(_feed_bytes([f"CVE-2024-{i:04d}" for i in range(50)]))
    items = list(iter_array(path, chunk_size=7))
    assert [i["cve"]["id"] for i in items] == [f"CVE-2024-{i:04d}" for i in range(50)]
    empty = tmp_path / "empty.json.gz"
    empty.write_bytes(gzip.compress(b'{"vulnerabilities": []}'))
    assert list(iter_array(empty)) == []


@respx.mock
def test_fetch_feeds_splits_pages_and_skips_unchanged(settings):
    settings.sources.nvd.page_size = 2
    feeds = {2002: _feed_bytes(["CVE-1999-0001", "CVE-2002-0001", "CVE-2002-0002"]),
             2003: _feed_bytes(["CVE-2003-0001"])}
    for year in range(2002, 2027):
        if year in feeds:
            # 2003 has not changed for weeks; 2002 was rebuilt on Oct 16
            date = "2026-10-16T03:01:38-04:00" if year == 2002 else "2026-08-28T03:04:38-04:00"
            meta = f"lastModifiedDate:{date}\r\nsize:1\r\ngzSize:{len(feeds[year])}\r\n"
            respx.get(f"{FEEDS}/nvdcve-2.0-{year}.meta").mock(return_value=httpx.Response(200, text=meta))
            respx.get(f"{FEEDS}/nvdcve-2.0-{year}.json.gz").mock(
                return_value=httpx.Response(200, content=feeds[year]))
        else:
            respx.get(f"{FEEDS}/nvdcve-2.0-{year}.meta").mock(return_value=httpx.Response(404))

    now = datetime(2026, 10, 21, tzinfo=UTC)
    result = fetch_cve_feeds(settings, now=now)
    assert result.records == 4
    names = [p.relative_to(settings.raw_dir).as_posix() for p in cve_raw_files(settings)]
    assert names == ["nvd/feeds/2003/page-0000000.json.gz", "nvd/feeds/2002/page-0000000.json.gz",
                     "nvd/feeds/2002/page-0000002.json.gz"]
    assert read_json_gz(cve_raw_files(settings)[2])["vulnerabilities"][0]["cve"]["id"] == "CVE-2002-0002"
    assert load_state(settings, NVD_FEEDS_STATE)["2002"]["records"] == 3
    # sync starts a day before the newest feed rebuild (07:01 UTC on Oct 16)
    assert load_state(settings, NVD_SYNC_STATE)["last_sync"] == "2026-10-15T07:01:38+00:00"
    # unchanged feeds are not downloaded again
    assert fetch_cve_feeds(settings, now=now).records == 0


@respx.mock
def test_feed_download_resumes_after_dropped_connection(tmp_path):
    from cyberrag.sources.nvd_feeds import download

    payload = b"0123456789" * 100
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("Range")
        calls.append(rng)
        if rng is None:  # first attempt: the connection drops after 300 bytes
            return httpx.Response(200, stream=_Dropping(payload[:300]))
        start = int(rng.split("=")[1].rstrip("-"))
        return httpx.Response(206, content=payload[start:])

    respx.get(f"{FEEDS}/nvdcve-2.0-2024.json.gz").mock(side_effect=handler)
    dest = tmp_path / "feed.json.gz"
    with httpx.Client() as client:
        download(client, FEEDS, 2024, dest, expected_size=len(payload))
    assert dest.read_bytes() == payload
    assert calls == [None, "bytes=300-"]


class _Dropping(httpx.SyncByteStream):
    def __init__(self, data: bytes) -> None:
        self.data = data

    def __iter__(self):
        yield self.data
        raise httpx.RemoteProtocolError("peer closed connection")
