"""Generic HTTP/JSON serving adapter — call a REST inference endpoint per batch.

Sends ``{column: [values...]}`` as JSON to `url` and reads ``{column: [values...]}``
back. Works with any service that speaks columnar JSON (a custom Flask/FastAPI model
server, a KServe v2-style REST shim). Requests retry with exponential backoff on
transient failures (connection errors, 429, 5xx) — serving endpoints are flaky at
scale. For LLM HTTP endpoints, see `batcher.ml.llm.http_engine`.

JSON is fine for scalar/text features. For **tensor** inputs (decoded images,
embeddings) nested JSON lists are slow and bloated, and building them costs an
``O(rows x dims)`` walk through Python objects — the one thing a control plane must
never do. So a tensor is sent as a compact **binary envelope** instead:

.. code-block:: json

    {"__tensor__": true, "dtype": "<f4", "shape": [3, 4], "data": "<base64>"}

The bytes come straight from the array buffer, so encoding is a C-level `tobytes` plus
a base64 pass, not a per-element conversion. The same envelope is decoded on the way
back, so a server may reply in binary too. `tensor_encoding` selects the policy:
``"auto"`` (binary for rank > 1, plain lists otherwise), ``"binary"``, or ``"json"``
for the legacy nested-list shape — which still warns, because it is still slow. A
server that speaks neither wants `batcher.ml.serving.triton_client`.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import BackendError
from batcher.ml.serving.base import _jittered_backoff, serving_udf

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

__all__ = ["http_client", "post_json"]

#: Marker key identifying a binary tensor envelope in the request/response JSON.
_TENSOR_KEY = "__tensor__"

_ENCODINGS = ("auto", "binary", "json")


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout: float,
    retries: int = 3,
    backoff: float = 0.5,
    max_retry_after: float = 60.0,
) -> dict[str, Any]:
    """POST `payload` as JSON to `url` and return the parsed JSON response.

    Retries up to `retries` times with jittered exponential backoff on connection errors
    and retryable HTTP status codes (408/425/429/5xx); other 4xx errors fail immediately
    (a bad request won't get better by retrying). Raises `BackendError` on exhaustion.

    A ``Retry-After`` header wins over the computed backoff. Every hosted inference API
    sends one on a 429 or a 503, and it is the *only* number that reflects when the quota
    actually refills — a client-side ``0.5 * 2**attempt`` is a guess that is either too
    short (retrying into the same rejection, which several providers count against the
    quota, so the retries fund their own starvation) or too long. It is clamped to
    `max_retry_after` so a provider naming a five-minute wait cannot stall a worker for
    longer than the caller is willing to wait.
    """
    import json
    import urllib.error
    import urllib.request

    body = json.dumps(payload).encode()
    last: Exception | None = None
    for attempt in range(retries + 1):
        wait: float | None = None
        req = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            try:
                return json.loads(raw)
            except ValueError as exc:
                # A 200 with a non-JSON body is almost always a proxy/load-balancer error
                # page or an HTML redirect; surface that as an actionable BackendError
                # rather than an opaque JSONDecodeError far from the call site.
                raise BackendError(
                    f"inference endpoint {url} returned a non-JSON body "
                    f"(first 200 bytes: {raw[:200]!r})"
                ) from exc
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (408, 425, 429, 500, 502, 503, 504):
                raise BackendError(
                    f"inference endpoint {url} returned {exc.code}: {_error_detail(exc)}"
                ) from exc
            wait = _retry_after(exc, max_retry_after)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
        if attempt < retries:
            time.sleep(wait if wait is not None else _jittered_backoff(backoff, attempt))
    raise BackendError(f"inference endpoint {url} failed after {retries + 1} attempts: {last}")


def _retry_after(exc: Any, ceiling: float) -> float | None:
    """The server's ``Retry-After`` delay in seconds, clamped, or `None` if it named none.

    The header carries either a delay in seconds or an HTTP date; both spellings are in the
    RFC and both are in use, so both are read. A value that does not parse is `None` rather
    than an error — a malformed header is the server's problem and must not become a failed
    inference job.
    """
    header = None
    try:
        header = exc.headers.get("Retry-After") if exc.headers is not None else None
    except Exception:
        return None
    if not header:
        return None
    text = str(header).strip()
    try:
        return min(ceiling, max(0.0, float(text)))
    except ValueError:
        pass
    from email.utils import parsedate_to_datetime

    try:
        import datetime as _dt

        target = parsedate_to_datetime(text)
        now = _dt.datetime.now(tz=target.tzinfo)
        return min(ceiling, max(0.0, (target - now).total_seconds()))
    except Exception:
        return None


def _error_detail(exc: Any) -> str:
    """A short excerpt of an HTTP error body, for a message that says what went wrong.

    A hosted API puts the actionable part of a 4xx in the body — which model name it did not
    recognize, which field was malformed, which quota was exceeded — and the status code alone
    sends the caller to the wrong place. Best-effort: an unreadable body yields the empty
    string rather than masking the original failure with a second one.
    """
    try:
        return exc.read()[:400].decode("utf-8", "replace")
    except Exception:
        return ""


def _encode_value(arr: np.ndarray, encoding: str) -> Any:
    """One input array → its JSON payload: a binary envelope, or a plain nested list.

    The binary path is ``tobytes`` + base64, both C-level passes over the buffer. The
    ``tolist`` path it replaces built one Python object per element, which for a batch
    of decoded images is millions of allocations in the control plane per request.
    """
    if encoding == "json" or (encoding == "auto" and getattr(arr, "ndim", 1) <= 1):
        return arr.tolist()
    import base64

    import numpy as np

    contiguous = np.ascontiguousarray(arr)
    return {
        _TENSOR_KEY: True,
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def _decode_value(value: Any) -> np.ndarray:
    """One response value → NumPy, unwrapping a binary envelope if the server sent one.

    Detection is by shape (``dtype``/``shape``/``data``) rather than the marker key, so
    a server that emits the envelope without the marker still round-trips.
    """
    import numpy as np

    if _is_tensor_envelope(value):
        import base64

        raw = base64.b64decode(value["data"])
        return np.frombuffer(raw, dtype=np.dtype(value["dtype"])).reshape(value["shape"])
    return np.asarray(value)


def _is_tensor_envelope(value: Any) -> bool:
    """Whether `value` is a binary tensor envelope, not just a dict that resembles one.

    The `__tensor__` marker is authoritative. Without it, a marker-less server is still
    accepted, but only when the three fields have the *shapes* an envelope has — a base64
    ``data`` string, a list/tuple ``shape``, and a string ``dtype`` — so a legitimate model
    output dict that happens to carry ``dtype``/``shape``/``data`` keys with other value
    types is no longer decoded as a tensor and mangled.
    """
    if not isinstance(value, dict):
        return False
    if value.get(_TENSOR_KEY) is True:
        return True
    return (
        {"dtype", "shape", "data"} <= set(value)
        and isinstance(value["data"], str)
        and isinstance(value["shape"], (list, tuple))
        and isinstance(value["dtype"], str)
    )


def http_client(
    url: str,
    *,
    input_columns: Sequence[str],
    output_columns: Sequence[str],
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    retries: int = 3,
    tensor_encoding: str = "auto",
    max_batch_size: int | None = None,
    pipeline_depth: int = 1,
) -> type:
    """A `map_batches` class UDF posting each batch to a JSON inference endpoint.

    Examples:
        .. doctest::

            >>> from batcher.ml import http_client  # doctest: +SKIP
            >>> udf = http_client(  # doctest: +SKIP
            ...     "http://host:8080/predict",
            ...     input_columns=["feature"],
            ...     output_columns=["score"],
            ... )
            >>> ds.ml.map_batches(udf, concurrency=4).collect()  # doctest: +SKIP

    Args:
        url: the inference endpoint (receives ``{column: list}``, returns the same).
        input_columns: the columns sent to the endpoint, in order.
        output_columns: the result columns appended to each batch.
        headers: optional HTTP headers (e.g. an auth token).
        timeout: per-request timeout in seconds.
        retries: retry attempts with backoff on transient failures.
        tensor_encoding: how multi-dimensional inputs are sent. ``"auto"`` uses the
            binary envelope for rank > 1 and plain lists otherwise, ``"binary"`` always
            uses the envelope, and ``"json"`` keeps the legacy nested-list shape (and
            warns, because it costs an ``O(rows x dims)`` Python conversion).
        max_batch_size: rows per request. An engine batch is thousands of rows, and an
            endpoint sized for a serving batch answers one of those with a 413, a timeout,
            or an out-of-memory error on its own GPU rather than with predictions. Set this
            to the window the endpoint was built for and a large batch is split into
            requests it can hold. ``None`` posts the batch whole, which is right only for
            an endpoint that declared no window.
        pipeline_depth: how many of those requests to keep in flight. Above 1 the endpoint
            keeps working while this worker encodes and decodes, which is what stops a
            remote GPU idling between requests. Results stay in input order.

    Returns:
        A class for ``ds.ml.map_batches(...)`` — the client connects once per worker.

    Raises:
        BackendError: if `tensor_encoding` is not one of ``auto``/``binary``/``json``.
    """
    if tensor_encoding not in _ENCODINGS:
        raise BackendError(
            f"tensor_encoding must be one of {list(_ENCODINGS)}, got {tensor_encoding!r}"
        )

    def connect() -> _HttpServingClient:
        return _HttpServingClient(url, headers or {}, timeout, retries, tensor_encoding)

    # `retries=0` here: `post_json` already owns the HTTP retry (it can distinguish a
    # retryable status from a fatal 4xx, which the generic wrapper cannot). Letting both
    # layers retry would multiply the attempts and the tail latency.
    return serving_udf(
        connect,
        input_columns=input_columns,
        output_columns=output_columns,
        max_batch_size=max_batch_size,
        pipeline_depth=pipeline_depth,
        retries=0,
    )


class _HttpServingClient:
    """Posts columnar JSON to a REST endpoint (with retry) and parses the response."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str],
        timeout: float,
        retries: int,
        tensor_encoding: str = "auto",
    ) -> None:
        self._url = url
        self._headers = {"Content-Type": "application/json", **headers}
        self._timeout = timeout
        self._retries = retries
        self._encoding = tensor_encoding
        self._warned_tensor = False

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, Any]:
        self._warn_on_json_tensor_inputs(inputs)
        payload = {name: _encode_value(arr, self._encoding) for name, arr in inputs.items()}
        body = post_json(
            self._url,
            payload,
            headers=self._headers,
            timeout=self._timeout,
            retries=self._retries,
        )
        return {name: _decode_value(values) for name, values in body.items()}

    def _warn_on_json_tensor_inputs(self, inputs: dict[str, np.ndarray]) -> None:
        """Warn only when a tensor is *actually* JSON-encoded — i.e. under ``"json"``.

        The warning used to fire and then do the slow thing anyway. Now the default
        (`"auto"`) sends binary and says nothing; the warning is reserved for the
        opt-in legacy encoding, where it is advice the caller can act on.
        """
        if self._warned_tensor or self._encoding != "json":
            return
        if any(getattr(arr, "ndim", 1) > 1 for arr in inputs.values()):
            import warnings

            from batcher._internal.errors import PerformanceWarning

            warnings.warn(
                "http_client is JSON-encoding a multi-dimensional (tensor) input because "
                "tensor_encoding='json'; this is slow and bloated. Use the default "
                "tensor_encoding='auto' for binary transport, or "
                "batcher.ml.serving.triton_client.",
                PerformanceWarning,
                stacklevel=3,
            )
            self._warned_tensor = True
