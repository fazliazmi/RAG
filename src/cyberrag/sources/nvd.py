"""NVD CVE API 2.0 client. Docs: https://nvd.nist.gov/developers/vulnerabilities"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import httpx

from cyberrag.config import NVDConfig
from cyberrag.sources.http import RateLimiter, get_json, make_http_client

MAX_DATE_RANGE_DAYS = 120  # NVD limit for lastModStartDate/lastModEndDate


def nvd_datetime(dt: datetime) -> str:
    """Format a datetime the way the NVD API expects (UTC)."""
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


class NVDClient:
    def __init__(
        self, cfg: NVDConfig, api_key: str | None = None, client: httpx.Client | None = None
    ) -> None:
        self.cfg = cfg
        headers = {"apiKey": api_key} if api_key else {}
        self._http = client or make_http_client(cfg.timeout_s, headers)
        self.limiter = RateLimiter(cfg.delay_with_key_s if api_key else cfg.delay_no_key_s)

    def close(self) -> None:
        self._http.close()

    def _get(self, params: dict[str, Any], label: str) -> dict[str, Any]:
        return get_json(
            self._http,
            self.cfg.base_url,
            params=params,
            limiter=self.limiter,
            max_retries=self.cfg.max_retries,
            wait_min_s=self.cfg.retry_wait_min_s,
            wait_max_s=self.cfg.retry_wait_max_s,
            label=label,
        )

    def page(self, start_index: int, **params: Any) -> dict[str, Any]:
        query = {"startIndex": start_index, "resultsPerPage": self.cfg.page_size, **params}
        return self._get(query, f"NVD startIndex={start_index}")

    def iter_pages(self, start_index: int = 0, **params: Any) -> Iterator[tuple[int, dict]]:
        """Yield (start_index, page) until all results are read."""
        start = start_index
        while True:
            data = self.page(start, **params)
            yield start, data
            count = len(data.get("vulnerabilities") or [])
            start += count
            if count == 0 or start >= int(data.get("totalResults", 0)):
                break

    def get_cve(self, cve_id: str) -> dict[str, Any] | None:
        data = self._get({"cveId": cve_id}, f"NVD {cve_id}")
        vulns = data.get("vulnerabilities") or []
        return vulns[0]["cve"] if vulns else None
