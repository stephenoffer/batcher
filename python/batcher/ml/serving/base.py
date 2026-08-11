"""Serving clients + the load-once `map_batches` adapter they share.

A serving backend (Triton, TorchServe, an HTTP endpoint) is reached through a
`ServingClient`: ``predict({column: ndarray}) -> {column: ndarray}``. `serving_udf`
wraps a *connect* function (run once per worker) into a class UDF for
``ds.ml.map_batches`` — it extracts the input columns as NumPy (tensor columns keep
their shape), calls the server, and appends the outputs. Because it returns a
*class*, ``map_batches`` instantiates it once per worker (connection + warm model),
the load-once pattern; only `batches` cross the wire, never per-row Python.

Three things here are what separate a demo from a production client, and they apply to
**every** backend because they live above the `ServingClient`, not inside one:

* **Warmup.** If a client exposes an optional ``warmup()``, `serving_udf` calls it once
  at connect (under the same retry). A worker then fails fast against a server that is
  down, instead of on its first real batch after the read stage has already run.
* **Bounded retry with jittered backoff.** Serving fleets are flaky; a lockstep
  exponential backoff turns one blip into a synchronized retry storm, so the delay is
  half-to-full jittered. Errors that cannot improve by retrying (`TypeError`,
  `ValueError`, `KeyError` — a shape or schema bug) fail immediately.
* **Splitting and in-flight pipelining.** A batch larger than the server's own
  ``max_batch_size`` is split, and up to `pipeline_depth` sub-batches are kept in
  flight, so the server is not idle during the client's own encode/decode. Output order
  always matches input order.
"""

from __future__ import annotations

import contextlib
import random
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from batcher._internal.errors import BackendError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    import numpy as np
    import pyarrow as pa

__all__ = ["ServingClient", "serving_udf"]

# Errors a retry cannot fix: they are bugs in the request, not in the network. Retrying
# one only multiplies the latency before the same failure surfaces.
_FATAL = (TypeError, ValueError, KeyError, AttributeError, IndexError)


@runtime_checkable
class ServingClient(Protocol):
    """A connected inference backend: a batch of named arrays in, named arrays out.

    Implement it to teach `serving_udf` a backend the built-in clients
    (`http_client`, `triton_client`, `torchserve_client`) do not cover.

    Examples:
        .. doctest::

            >>> import numpy as np
            >>> from batcher.ml import ServingClient
            >>> class Doubler:
            ...     def predict(self, inputs):
            ...         return {"y": inputs["x"] * 2}
            >>> isinstance(Doubler(), ServingClient)  # a structural, runtime-checked match
            True
            >>> Doubler().predict({"x": np.array([1, 2])})
            {'y': array([2, 4])}
    """

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference on one batch of input arrays, returning output arrays.

        Examples:
            .. doctest::

                >>> client.predict({"image": images})  # doctest: +SKIP
                {'logits': array([[0.1, 0.9]])}

        Args:
            inputs: the input columns as NumPy arrays, keyed by the backend's input
                names (a tensor column keeps its ``(N, *shape)`` form).

        Returns:
            The backend's output arrays, keyed by its output names.
        """
        ...


def _reject_non_numeric_inputs(feed: dict[str, np.ndarray], batch: pa.RecordBatch) -> None:
    """Fail early and by name on a serving input that has no numeric array form.

    A string/nested column converts to an ``object``-dtype array, which a Triton dtype
    map rejects deep in the client and an HTTP client base64-mangles into garbage. Naming
    the offending column here turns that into an actionable message at the batch edge.
    """
    for name, arr in feed.items():
        if getattr(arr, "dtype", None) is not None and arr.dtype == object:
            from batcher._internal.errors import BackendError

            raise BackendError(
                f"serving input column {name!r} is {batch.schema.field(name).type}, which "
                "has no numeric/tensor array form for an inference endpoint. Select or cast "
                "it to a numeric or tensor column before serving."
            )


def _column_to_numpy(column: pa.Array) -> np.ndarray:
    """A batch column as NumPy — tensor columns keep their ``(N, *shape)`` form.

    Delegates to the one converter (`ml.converters`) so serving inputs are shaped
    identically to the training/loader path: a `FixedShapeTensor` **and** a numeric
    ``FixedSizeList<T, W>`` feature/embedding column both restore to their full
    ``(N, W...)`` array (null-safe), rather than a serving model silently receiving an
    opaque per-row object array for the fixed-size-list case.
    """
    from batcher.ml.converters import _column_to_numpy as _convert

    return _convert(column)


def _array_from_numpy(values: np.ndarray) -> pa.Array:
    """An output array → Arrow: 1-D stays scalar, higher-rank becomes a tensor column."""
    import pyarrow as pa

    if values.ndim <= 1:
        return pa.array(values)
    from batcher.io.formats.ml.tensor import to_tensor_column

    return to_tensor_column(values)


def _jittered_backoff(base: float, attempt: int) -> float:
    """Seconds to wait before retry `attempt` — **full jitter**, uniform in ``[0, ceiling]``.

    This is the one backoff implementation for the serving package (`http._retry_delay`
    delegates here), because a second copy would inevitably drift from this one.

    A deterministic ``base * 2**attempt`` is the classic thundering herd: every worker
    throttled by the same 429 sleeps the same duration, retries in the same millisecond,
    and re-triggers the 429 it was backing off from. Randomizing across the whole
    interval spreads the fleet out, and is what AWS's backoff guidance recommends over
    equal jitter.
    """
    return random.uniform(0.0, base * (2**attempt))


def _call_with_retry(fn: Callable[[], Any], retries: int, backoff: float, what: str) -> Any:
    """Run `fn`, retrying transient failures with jittered backoff; raise `BackendError`.

    A `_FATAL` error (a shape/schema bug) propagates untouched on the first attempt, so
    a real bug still fails fast and is not disguised as a flaky endpoint.
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except _FATAL:
            raise
        except Exception as exc:
            last = exc
            if attempt >= retries:
                break
            time.sleep(_jittered_backoff(backoff, attempt))
    raise BackendError(f"serving backend failed {what} after {retries + 1} attempt(s): {last}")


def _split_feed(
    feed: dict[str, np.ndarray], num_rows: int, max_batch_size: int | None
) -> list[dict[str, np.ndarray]]:
    """Slice the input arrays into sub-batches of at most `max_batch_size` rows.

    Returns the feed unsliced (no copy, no allocation) when it already fits, so the
    common case pays nothing for the capability.
    """
    if max_batch_size is None or max_batch_size <= 0 or num_rows <= max_batch_size:
        return [feed]

    def _slice(arr: np.ndarray, start: int) -> np.ndarray:
        # A per-row input is sliced; a broadcast/shared input (leading dim != num_rows,
        # e.g. a single (1, ...) value for the whole batch) is passed through whole —
        # slicing it would empty every sub-batch after the first and corrupt the request.
        if getattr(arr, "shape", (0,))[:1] == (num_rows,):
            return arr[start : start + max_batch_size]
        return arr

    return [
        {name: _slice(arr, start) for name, arr in feed.items()}
        for start in range(0, num_rows, max_batch_size)
    ]


def _pipelined(
    fn: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]],
    chunks: list[dict[str, np.ndarray]],
    depth: int,
) -> Iterator[dict[str, np.ndarray]]:
    """Apply `fn` to `chunks` with at most `depth` calls in flight, yielding **in order**.

    The window is bounded rather than submitting everything at once: an unbounded
    submit would hold every sub-batch of a large morsel in memory and flood the server,
    which is the backpressure failure this engine avoids everywhere else.
    """
    if depth <= 1 or len(chunks) == 1:
        for chunk in chunks:
            yield fn(chunk)
        return
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=depth, thread_name_prefix="batcher-serving") as pool:
        pending: deque = deque()
        for chunk in chunks:
            pending.append(pool.submit(fn, chunk))
            if len(pending) == depth:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def _merge_results(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate per-key output arrays from each sub-batch, preserving row order."""
    if len(parts) == 1:
        return parts[0]
    import numpy as np

    # `atleast_1d` so a 0-d (scalar-per-sub-batch) output — e.g. a per-batch summary from
    # a split large batch — concatenates instead of raising "zero-dimensional arrays
    # cannot be concatenated".
    return {name: np.concatenate([np.atleast_1d(p[name]) for p in parts]) for name in parts[0]}


def _require_input_columns(batch: Any, inputs: list[str]) -> None:
    """Reject an input column the batch does not have, naming what it does have.

    `serving_udf` builds a callable rather than a plan node, so there is no dataset to check
    against when it is constructed — the first batch is the earliest honest moment. Before
    this, a mistyped `input_columns` reached Arrow and returned
    ``KeyError: 'Field "nope" does not exist in schema'`` from inside a worker.
    """
    missing = [name for name in inputs if name not in batch.schema.names]
    if not missing:
        return
    from batcher._internal.errors import ColumnNotFoundError

    raise ColumnNotFoundError(
        f"serving_udf input_columns={missing} are not in the batch; it has "
        f"{list(batch.schema.names)}."
    )


def _check_response(result: Any, rows: int, outputs: list[str] | None) -> None:
    """Reject a backend response that cannot be aligned with the batch it answered.

    A serving backend that under-returns — a truncated response, a partially-failed batch —
    is a real production failure, and the only signal was
    ``ValueError: Arrays were not all the same length: 1 vs 3`` raised while assembling the
    output. That names neither the backend nor which output was short, and it is the error
    that stands between a caller and a **misaligned** result, so it is worth stating plainly.
    """
    from batcher._internal.errors import BackendError

    if not isinstance(result, dict):
        raise BackendError(
            f"serving backend returned {type(result).__name__}, but predict() must return a "
            f"{{name: array}} dict of output arrays."
        )
    for name in outputs if outputs is not None else list(result):
        if name not in result:
            raise BackendError(
                f"serving backend response is missing output {name!r}; it returned "
                f"{sorted(result)}."
            )
        length = getattr(result[name], "__len__", None)
        if length is not None and len(result[name]) != rows:
            raise BackendError(
                f"serving backend returned {len(result[name])} rows for output {name!r} but "
                f"was sent {rows}. A response that does not line up row-for-row would pair "
                f"predictions with the wrong inputs."
            )


def _declared_batch_window(client: object) -> int | None:
    """The batch window a client's server declares, or `None` when it declares none.

    Optional: a client that cannot ask its server simply does not define `batch_window`, and
    the batch is sent whole exactly as before. A failure to ask is not a failure to serve
    either — the window is an optimization over what the caller could have passed by hand, so
    an unreachable config endpoint degrades to the old behavior rather than failing the worker
    before it has run a row.
    """
    ask = getattr(client, "batch_window", None)
    if not callable(ask):
        return None
    try:
        window = ask()
    except Exception as exc:
        from batcher._internal.logging import note_suppressed

        note_suppressed("ml", "read the serving model's declared batch window", exc)
        return None
    return window if isinstance(window, int) and window > 0 else None


def serving_udf(
    connect: Callable[[], ServingClient],
    *,
    input_columns: Sequence[str],
    output_columns: Sequence[str] | None = None,
    max_batch_size: int | None = None,
    pipeline_depth: int = 1,
    retries: int = 2,
    retry_backoff: float = 0.5,
) -> type:
    """Build a load-once class UDF that runs `input_columns` through a serving backend.

    Examples:
        .. doctest::

            >>> from batcher.ml import serving_udf  # doctest: +SKIP
            >>> udf = serving_udf(connect, input_columns=["image"])  # doctest: +SKIP
            >>> ds.ml.map_batches(udf, concurrency=4).collect()  # doctest: +SKIP

    Args:
        connect: a zero-arg callable returning a connected `ServingClient`; run once
            per worker (the model/connection is reused across batches). If the client
            also defines ``warmup()``, it is called once here, under the same retry.
        input_columns: the columns sent to the server, in order.
        output_columns: the appended result columns (defaults to the server's keys).
        max_batch_size: the server's own batch window. A larger Arrow batch is split
            into requests of at most this many rows. ``None`` asks the client for the
            window the server declares (an optional ``batch_window()``), and sends the
            batch whole only when there is nothing to ask or nothing declared.
        pipeline_depth: how many sub-batch requests to keep in flight at once. Above 1
            the server keeps working while the client encodes and decodes; results are
            still emitted in input order.
        retries: retry attempts for a transient backend failure, with jittered backoff.
            A shape or schema error is never retried.
        retry_backoff: the base backoff in seconds, doubled and jittered per attempt.

    Returns:
        A class for ``ds.ml.map_batches(...)`` — instantiate-once-per-worker inference.
    """
    inputs = list(input_columns)
    outputs = None if output_columns is None else list(output_columns)
    depth = max(1, pipeline_depth)

    class _ServingUDF:
        def __init__(self) -> None:
            self._client = connect()
            warmup = getattr(self._client, "warmup", None)
            if callable(warmup):
                # Under retry: a worker that starts while the server is still rolling
                # should wait for it, not fail the whole job on a cold endpoint.
                _call_with_retry(warmup, retries, retry_backoff, "warmup")
            # A server's batch window is a property of the deployed model, and the server
            # knows it. An engine batch is orders of magnitude larger than the window a
            # serving config typically declares, so leaving it unasked meant the first real
            # batch was rejected whole — a configuration error surfacing as a failed job.
            self._window = max_batch_size or _declared_batch_window(self._client)

        def _predict(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            return _call_with_retry(
                lambda: self._client.predict(feed), retries, retry_backoff, "predict"
            )

        def close(self) -> None:
            """Release the backend connection when the worker is done with it.

            `close` is the teardown contract `core.udf.lifecycle` and `InferencePool` look
            for, and they look for it on *this* object. Without it a client holding a real
            connection — a Triton HTTP pool or gRPC channel, a database handle — was released
            only whenever the garbage collector happened to reach it, so a script running two
            serving stages back to back held both generations of connections at once. A client
            that needs no teardown simply defines no `close`, and this is a no-op.

            Best-effort by the same contract as `teardown_udf`: the rows are already produced,
            so a failing `close` must not fail the query.
            """
            close = getattr(self._client, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()

        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            import pyarrow as pa

            _require_input_columns(batch, inputs)
            feed = {name: _column_to_numpy(batch.column(name)) for name in inputs}
            _reject_non_numeric_inputs(feed, batch)
            chunks = _split_feed(feed, batch.num_rows, self._window)
            result = _merge_results(list(_pipelined(self._predict, chunks, depth)))
            _check_response(result, batch.num_rows, outputs)
            keep = [batch.column(i) for i in range(batch.num_columns)]
            names = list(batch.schema.names)
            for name in outputs if outputs is not None else list(result):
                array = _array_from_numpy(result[name])
                # An output whose name the batch already carries **replaces** it rather than
                # being appended beside it. Arrow permits duplicate field names, so appending
                # produced a batch with two columns of one name: `to_pydict()` keeps only the
                # last, every expression resolves the first, and nothing raises. That happens
                # whenever a server echoes an input back or a pipeline re-scores into the
                # column it read — and the model-id inference path already handles it.
                if name in names:
                    keep[names.index(name)] = array
                else:
                    keep.append(array)
                    names.append(name)
            return pa.RecordBatch.from_arrays(keep, names=names)

    return _ServingUDF
