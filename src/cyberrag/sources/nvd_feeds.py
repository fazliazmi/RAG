"""NVD JSON 2.0 yearly data feeds: https://nvd.nist.gov/vuln/data-feeds

Each feed (nvdcve-2.0-<year>.json.gz) has the same shape as an API response
page, so the same parser handles both. Feeds are static files, so a full
download takes minutes, compared with hours of paging through the API when
NVD is slow. NVD rebuilds them about once a week; `cyberrag sync` uses the API
for anything newer.
"""

from __future__ import annotations

import gzip
import json
import os
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from cyberrag.errors import ProviderError

FIRST_YEAR = 2002  # the 2002 feed also holds all CVEs from 1999-2001
_WS = re.compile(r"[\s,]*")


def feed_name(year: int) -> str:
    return f"nvdcve-2.0-{year}"


def get_meta(client: httpx.Client, base_url: str, year: int) -> dict[str, str] | None:
    """Parse the small .meta file; None if NVD has no feed for this year."""
    resp = client.get(f"{base_url.rstrip('/')}/{feed_name(year)}.meta")
    if resp.status_code == 404:
        return None
    if resp.status_code >= 400:
        raise ProviderError(f"NVD feed meta {year}: HTTP {resp.status_code}")
    meta = {}
    for line in resp.text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    return meta


def meta_datetime(meta: dict[str, str]) -> datetime:
    return datetime.fromisoformat(meta["lastModifiedDate"])


def download(
    client: httpx.Client, base_url: str, year: int, dest: Path, expected_size: int, attempts: int = 10
) -> None:
    """Download with resume: NVD's server sometimes drops long transfers, so each retry
    continues from the bytes already received (HTTP Range request)."""
    url = f"{base_url.rstrip('/')}/{feed_name(year)}.json.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.unlink(missing_ok=True)
    last_error: Exception | None = None
    for _ in range(attempts):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code == 200 and have:
                    have = 0  # server ignored the range: start over
                elif resp.status_code not in (200, 206):
                    raise ProviderError(f"{url}: HTTP {resp.status_code}")
                with tmp.open("ab" if have else "wb") as fh:
                    for chunk in resp.iter_bytes():  # unbuffered: keep every byte received
                        fh.write(chunk)
        except (httpx.HTTPError, ProviderError) as exc:
            last_error = exc
            continue
        size = tmp.stat().st_size
        if not expected_size or size == expected_size:
            os.replace(tmp, dest)
            return
        if size > expected_size:
            tmp.unlink()
        last_error = ProviderError(f"got {size} bytes, expected {expected_size}")
    tmp.unlink(missing_ok=True)
    raise ProviderError(f"Could not download {url} after {attempts} attempts: {last_error}")


def iter_array(path: Path, key: str = "vulnerabilities", chunk_size: int = 1 << 20) -> Iterator[Any]:
    """Stream the items of one top-level JSON array from a .json.gz file, so a
    300 MB feed never has to be loaded into memory at once."""
    decoder = json.JSONDecoder()
    marker = f'"{key}"'
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        buf = ""
        while True:  # find the start of the array
            idx = buf.find(marker)
            bracket = buf.find("[", idx + len(marker)) if idx >= 0 else -1
            if bracket >= 0:
                buf, pos = buf, bracket + 1
                break
            chunk = fh.read(chunk_size)
            if not chunk:
                return
            buf += chunk
        while True:
            pos = _WS.match(buf, pos).end()
            if pos < len(buf) and buf[pos] == "]":
                return
            try:
                obj, pos = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                chunk = fh.read(chunk_size)
                if not chunk:
                    raise
                buf, pos = buf[pos:] + chunk, 0
                continue
            yield obj
