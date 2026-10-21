"""Shared HTTP helpers: a polite rate limiter and JSON GET with retries."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from cyberrag import __version__
from cyberrag.errors import ProviderError

log = logging.getLogger("cyberrag.http")

USER_AGENT = f"cyberrag/{__version__} (+https://github.com/fazliazmi/RAG)"
# NVD answers 403 when you go over the rate limit, and 503 when it is overloaded.
RETRY_STATUS = {403, 408, 429, 500, 502, 503, 504}


class RetryableHTTPError(Exception):
    pass


class RateLimiter:
    """Keeps at least `min_interval_s` between requests."""

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = max(0.0, min_interval_s)
        self._last = 0.0

    def wait(self) -> None:
        delay = self._last + self.min_interval_s - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()


def make_http_client(timeout_s: float, headers: dict[str, str] | None = None) -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT, **(headers or {})},
        timeout=httpx.Timeout(timeout_s, connect=30.0),
        follow_redirects=True,
    )


def get_json(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    limiter: RateLimiter | None = None,
    max_retries: int = 5,
    wait_min_s: float = 3.0,
    wait_max_s: float = 120.0,
    label: str = "",
) -> Any:
    label = label or url

    def attempt() -> Any:
        if limiter:
            limiter.wait()
        resp = client.get(url, params=params)
        if resp.status_code in RETRY_STATUS:
            raise RetryableHTTPError(f"HTTP {resp.status_code} {resp.headers.get('message', '')}")
        if resp.status_code >= 400:
            detail = resp.headers.get("message") or resp.text[:300]
            raise ProviderError(f"{label}: HTTP {resp.status_code}: {detail}")
        return resp.json()

    def before_sleep(state: Any) -> None:
        log.warning(
            "%s: %s, retrying (attempt %d/%d)",
            label, state.outcome.exception(), state.attempt_number, max_retries,
        )

    retrying = Retrying(
        retry=retry_if_exception_type(
            (httpx.TransportError, RetryableHTTPError, json.JSONDecodeError)
        ),
        wait=wait_exponential(multiplier=max(wait_min_s, 0.0), min=wait_min_s, max=wait_max_s),
        stop=stop_after_attempt(max_retries),
        before_sleep=before_sleep,
        reraise=True,
    )
    try:
        return retrying(attempt)
    except (httpx.TransportError, RetryableHTTPError, json.JSONDecodeError) as exc:
        raise ProviderError(f"{label}: failed after {max_retries} attempts ({exc})") from exc
