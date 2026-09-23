"""Shared HTTP client with retries.

External APIs fail in ordinary ways: timeouts, brief outages, rate limits.
We retry those automatically with exponential backoff, and fail fast on
errors that retrying cannot fix (for example 404 Not Found).
"""

from typing import Any

import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from gridcast import __version__
from gridcast.config import Settings
from gridcast.errors import SourceError
from gridcast.logging import get_logger

log = get_logger(__name__)

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def make_client(settings: Settings) -> httpx.Client:
    """Create an HTTP client. Use it as a context manager so connections close."""
    return httpx.Client(
        timeout=settings.http_timeout_s,
        follow_redirects=True,
        headers={"User-Agent": f"gridcast/{__version__}"},
    )


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):  # timeouts, connection resets, DNS
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return False


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    log.warning("http.retry", attempt=state.attempt_number, error=repr(exc))


def get_with_retry(
    client: httpx.Client,
    url: str,
    settings: Settings,
    params: dict[str, Any] | None = None,
) -> httpx.Response:
    """GET a URL, retrying transient failures.

    Raises SourceError (with the server's message) once retries are exhausted
    or immediately for errors that retrying cannot fix, such as 400 or 404.
    """

    def _get() -> httpx.Response:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response

    retrying = Retrying(
        stop=stop_after_attempt(settings.http_max_retries),
        wait=wait_exponential(multiplier=settings.http_backoff_s, max=30),
        retry=retry_if_exception(_is_retryable),
        before_sleep=_log_retry,
        reraise=True,
    )
    try:
        return retrying(_get)
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:300]
        raise SourceError(f"{exc.response.status_code} from {url}: {body}") from exc
    except httpx.TransportError as exc:
        raise SourceError(f"Could not reach {url}: {exc!r}") from exc
