"""MITRE CWE REST API client. Docs: https://github.com/CWE-CAPEC/REST-API-wg

The single `/cwe/weakness/all` response (~6 MB) is often cut off by the
server, so the full list is built from smaller requests instead: walk the
"Research Concepts" view (CWE-1000, which contains every weakness) to collect
the IDs, then fetch the weaknesses in batches.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import httpx

from cyberrag.config import CWEConfig
from cyberrag.errors import ProviderError
from cyberrag.sources.http import get_json, make_http_client

RESEARCH_VIEW = "1000"


def _walk(nodes: list[dict[str, Any]] | None, out: set[int]) -> None:
    for node in nodes or []:
        data = node.get("Data") or {}
        if str(data.get("Type", "")).endswith("weakness") and str(data.get("ID", "")).isdigit():
            out.add(int(data["ID"]))
        _walk(node.get("Children"), out)


class CWEClient:
    def __init__(self, cfg: CWEConfig, client: httpx.Client | None = None) -> None:
        self.cfg = cfg
        self._http = client or make_http_client(cfg.timeout_s)

    def close(self) -> None:
        self._http.close()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return get_json(
            self._http,
            f"{self.cfg.base_url.rstrip('/')}/{path.lstrip('/')}",
            params=params,
            max_retries=self.cfg.max_retries,
            wait_min_s=self.cfg.retry_wait_min_s,
            label=f"CWE {path}",
        )

    def version(self) -> dict[str, Any]:
        return self._get("cwe/version")

    def weakness_ids(self) -> list[int]:
        view = self._get(f"cwe/view/{RESEARCH_VIEW}")
        view = view.get("Views", [view])[0] if isinstance(view, dict) else view
        ids: set[int] = set()
        for member in view.get("Members", []):
            tree = self._get(f"cwe/{member['CweID']}/descendants", {"view": RESEARCH_VIEW})
            _walk(tree, ids)
        return sorted(ids)

    def weaknesses(self, ids: Iterable[int | str]) -> list[dict[str, Any]]:
        joined = ",".join(str(i).upper().removeprefix("CWE-") for i in ids)
        return self._get(f"cwe/weakness/{joined}").get("Weaknesses", [])

    def all_weaknesses(
        self, batch_size: int = 40, progress: Callable[[int, int], None] | None = None
    ) -> list[dict[str, Any]]:
        ids = self.weakness_ids()
        out: list[dict[str, Any]] = []
        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            try:
                out.extend(self.weaknesses(batch))
            except ProviderError:
                # One bad ID fails the whole batch: retry the batch one by one.
                for cwe_id in batch:
                    try:
                        out.extend(self.weaknesses([cwe_id]))
                    except ProviderError:
                        continue
            if progress:
                progress(min(i + batch_size, len(ids)), len(ids))
        return out
