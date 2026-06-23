"""Generic HTTP/JSON serving adapter — call a REST inference endpoint per batch.

Sends ``{column: [values...]}`` as JSON to `url` and reads ``{column: [values...]}``
back. Works with any service that speaks columnar JSON (a custom Flask/FastAPI model
server, a KServe v2-style REST shim). Requests retry with exponential backoff on
transient failures (connection errors, 429, 5xx) — serving endpoints are flaky at
scale. For LLM HTTP endpoints, see `batcher.ml.llm.http_engine`.

JSON is fine for scalar/text features; for **tensor** inputs (decoded images,
embeddings) it is slow and bloated — use `batcher.ml.serving.triton_client`, which
sends binary tensors. This adapter warns (once) if asked to JSON-encode a tensor.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import BackendError
from batcher.ml.serving.base import serving_udf

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

__all__ = ["http_client", "post_json"]


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout: float,
    retries: int = 3,
    backoff: float = 0.5,
) -> dict[str, Any]:
    """POST `payload` as JSON to `url` and return the parsed JSON response.

    Retries up to `retries` times with exponential backoff on connection errors and
    retryable HTTP status codes (408/425/429/5xx); other 4xx errors fail immediately
    (a bad request won't get better by retrying). Raises `BackendError` on exhaustion.
    """
    import json
    import urllib.error
    import urllib.request

    body = json.dumps(payload).encode()
    last: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (408, 425, 429, 500, 502, 503, 504):
                raise BackendError(f"inference endpoint {url} returned {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
        if attempt < retries:
            time.sleep(backoff * (2**attempt))
    raise BackendError(f"inference endpoint {url} failed after {retries + 1} attempts: {last}")


def http_client(
    url: str,
    *,
    input_columns: Sequence[str],
    output_columns: Sequence[str],
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    retries: int = 3,
) -> type:
    """A `map_batches` class UDF posting each batch to a JSON inference endpoint.

    Args:
        url: the inference endpoint (receives ``{column: list}``, returns the same).
        input_columns / output_columns: columns sent and appended.
        headers: optional HTTP headers (e.g. an auth token).
        timeout: per-request timeout in seconds.
        retries: retry attempts with backoff on transient failures.
    """

    def connect() -> _HttpServingClient:
        return _HttpServingClient(url, headers or {}, timeout, retries)

    return serving_udf(connect, input_columns=input_columns, output_columns=output_columns)


class _HttpServingClient:
    """Posts columnar JSON to a REST endpoint (with retry) and parses the response."""

    def __init__(self, url: str, headers: dict[str, str], timeout: float, retries: int) -> None:
        self._url = url
        self._headers = {"Content-Type": "application/json", **headers}
        self._timeout = timeout
        self._retries = retries
        self._warned_tensor = False

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, Any]:
        import numpy as np

        self._warn_on_tensor_inputs(inputs)
        payload = {name: arr.tolist() for name, arr in inputs.items()}
        body = post_json(
            self._url,
            payload,
            headers=self._headers,
            timeout=self._timeout,
            retries=self._retries,
        )
        return {name: np.asarray(values) for name, values in body.items()}

    def _warn_on_tensor_inputs(self, inputs: dict[str, np.ndarray]) -> None:
        if self._warned_tensor:
            return
        if any(getattr(arr, "ndim", 1) > 1 for arr in inputs.values()):
            import warnings

            from batcher._internal.errors import PerformanceWarning

            warnings.warn(
                "http_client is JSON-encoding a multi-dimensional (tensor) input; this "
                "is slow and bloated. Use batcher.ml.serving.triton_client for binary "
                "tensor transport.",
                PerformanceWarning,
                stacklevel=3,
            )
            self._warned_tensor = True
