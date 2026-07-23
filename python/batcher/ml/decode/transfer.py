"""Moving bytes in and out of a dataset — the ends of a multimodal pipeline.

`download_dataset` turns a table of URLs into a table of bytes, and `upload_dataset`
writes bytes back out to object storage. Neither decodes anything; they are the transfer
stages either side of the decoders in `media` and `video`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.decode.stage import _bounded_map, _shared_pool, _with_column

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["download_dataset", "upload_dataset"]


def upload_dataset(
    ds: Dataset,
    *,
    data_column: str,
    directory: str,
    output_column: str = "path",
    name_column: str | None = None,
    extension: str = "",
    max_concurrency: int = 16,
) -> Dataset:
    """Write each row's bytes to a file under `directory`, appending the written path.

    The counterpart to `download_dataset` (cf. Daft's ``url.upload``) — write decoded
    or transformed media back to ``s3://`` / ``gs://`` / ``az://`` / local storage.
    File names come from `name_column` (plus `extension`) or a content-addressed hash
    when no name column is given (collision-free across distributed workers). Writes
    each batch's rows concurrently and parallelizes across the cluster.

    Args:
        data_column: the binary column to write.
        directory: the destination directory/prefix.
        output_column: the appended column of written paths.
        name_column: optional column of file names (else a content hash is used).
        extension: appended to the file name (e.g. ``".jpg"``).
        max_concurrency: concurrent writes per batch.
    """
    base = directory.rstrip("/")

    def _write(fs: Any, name: str, data: bytes | None) -> str | None:
        if data is None:
            return None
        path = f"{base}/{name}{extension}"
        with fs.atomic_writer(path) as handle:
            handle.write(data)
        return path

    def _udf(batch: Any) -> Any:
        import hashlib
        from functools import partial

        import pyarrow as pa

        from batcher.io.filesystem import resolve_filesystem

        data = batch.column(data_column).to_pylist()
        if name_column is not None:
            names = [str(n) for n in batch.column(name_column).to_pylist()]
        else:
            names = [hashlib.sha1(b).hexdigest() if b is not None else "" for b in data]
        # One `directory` means one filesystem for every row: resolve it once here rather
        # than once per row inside the thread map, where it was re-derived (and, for a
        # local path, reconstructed) for every single file written.
        fs = resolve_filesystem(base)
        pool = _shared_pool(max_concurrency)
        paths = list(pool.map(partial(_write, fs), names, data))
        return _with_column(batch, output_column, pa.array(paths, type=pa.large_string()))

    out_cols = list(ds.columns) if output_column in ds.columns else [*ds.columns, output_column]
    return ds.map_batches(_udf, output_columns=out_cols)


def download_dataset(
    ds: Dataset,
    *,
    url_column: str,
    output_column: str = "bytes",
    max_concurrency: int = 16,
    on_error: str = "raise",
    retries: int = 2,
    retry_backoff: float = 0.2,
    timeout: float | None = None,
    error_column: str | None = None,
) -> Dataset:
    """Fetch the bytes at each URL/path into a ``large_binary`` column.

    The entry point of a multimodal pipeline (URL table → bytes → decode → model), the
    counterpart to Daft's ``col(url).url.download()``. Reads ``s3://`` / ``gs://`` /
    ``az://`` / ``http(s)://`` / local paths through the shared filesystem resolver,
    fetching each batch's rows **concurrently** on a per-process thread pool reused
    across batches, as a `map_batches` stage that parallelizes across the cluster.

    At scale a fetch fails for reasons that are not the URL's fault — a throttled object
    store, a dropped connection, a slow origin — so it is retried with exponential
    backoff and full jitter, the jitter being what stops a batch of simultaneous failures
    from retrying in lockstep and re-throttling the store. `on_error="null"` turns a
    still-failing fetch into a null, and `error_column` records *why*, which is the
    difference between a diagnosable partial result and a silently short dataset.

    `timeout` bounds one attempt, and abandons it rather than cancelling it: a blocking
    read inside a filesystem client cannot be interrupted, so its thread stays busy until
    the socket gives up. Treat it as "stop waiting", not "stop transferring".

    Args:
        url_column: the column of URLs/paths to fetch.
        output_column: the appended (or replaced) bytes column.
        max_concurrency: concurrent fetches per batch (I/O-bound, GIL-releasing).
        on_error: ``"raise"`` (default) or ``"null"``.
        retries: extra attempts after the first failure. ``0`` disables retrying.
        retry_backoff: base seconds for the jittered exponential backoff.
        timeout: seconds to wait for one attempt, or ``None`` to wait indefinitely.
        error_column: appended ``large_string`` column of per-row error messages
            (null where the fetch succeeded). Requires ``on_error="null"``.

    Raises:
        PlanError: on an invalid `on_error`, a negative `retries`, or an
            `error_column` without ``on_error="null"``.
    """
    if on_error not in ("raise", "null"):
        raise PlanError(f"download on_error must be 'raise' or 'null', got {on_error!r}")
    if retries < 0:
        raise PlanError(f"download retries must be >= 0, got {retries}")
    if error_column is not None and on_error != "null":
        raise PlanError("download error_column= requires on_error='null'")

    def _attempt(url: str) -> bytes:
        from batcher.io.filesystem import resolve_filesystem

        with resolve_filesystem(url).open(url) as handle:
            return handle.read()

    def _fetch(url: str | None) -> tuple[bytes | None, str | None]:
        if url is None:
            return None, None
        import random
        import time

        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return _attempt(url), None
            except Exception as exc:
                last = exc
                if attempt < retries:
                    time.sleep(random.uniform(0, retry_backoff * (2**attempt)))
        if on_error == "null":
            return None, f"{type(last).__name__}: {last}"
        raise PlanError(f"download failed for {url!r} after {retries + 1} attempts") from last

    def _timed_out() -> tuple[None, str]:
        message = f"TimeoutError: fetch exceeded timeout={timeout}s"
        if on_error == "raise":
            raise PlanError(message)
        return None, message

    def _udf(batch: Any) -> Any:
        import pyarrow as pa

        urls = batch.column(url_column).to_pylist()
        results = list(
            _bounded_map(_fetch, urls, max_concurrency, timeout=timeout, on_timeout=_timed_out)
        )
        col = pa.array([data for data, _ in results], type=pa.large_binary())
        out = _with_column(batch, output_column, col)
        if error_column is None:
            return out
        errors = pa.array([err for _, err in results], type=pa.large_string())
        return _with_column(out, error_column, errors)

    out_cols = list(ds.columns) if output_column in ds.columns else [*ds.columns, output_column]
    if error_column is not None and error_column not in out_cols:
        out_cols = [*out_cols, error_column]
    return ds.map_batches(_udf, output_columns=out_cols)
