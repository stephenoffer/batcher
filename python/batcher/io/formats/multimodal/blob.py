"""Blob-by-reference: offload large per-row payloads to a content-addressed store.

Multi-GB payloads (video/audio/pdf bytes) carried inline in an Arrow column blow up
every shuffle and spill buffer they cross. This module is the **write-side dual** of
reference mode (``read.video(materialize_bytes=False)``): `offload_blob_bytes` writes
each row's payload to a content-addressed file and replaces the bytes column with a
tiny URI handle, so only the handle (a short string) rides through the pipeline.
`io.formats.multimodal.media.read_blob_bytes` materializes the payloads back on
demand — the two are inverses.

Content addressing (SHA-256 of the payload) makes offload idempotent and dedupes
identical payloads, and means a re-read after a spill/recompute fetches the same
bytes. Writes go through ``pyarrow.fs`` so a local NVMe scratch dir and a remote
object store (``s3://``/``gs://``/…) are the same code path; reads reuse the
read-only filesystem façade. It never touches a row outside an explicit
`map_batches` blob step.
"""

from __future__ import annotations

import hashlib
import tempfile

import pyarrow as pa
import pyarrow.fs as pafs

from batcher.io._concurrent import read_each_file
from batcher.io.filesystem import resolve_filesystem

__all__ = [
    "BLOB_URI_COLUMN",
    "default_blob_root",
    "materialize_and_drop_handle",
    "offload_blob_bytes",
    "read_blob_bytes",
]

# The default name of the URI-handle column an offload produces / a materialize reads.
BLOB_URI_COLUMN = "uri"


def materialize_and_drop_handle(
    batch: pa.RecordBatch, *, uri_col: str = BLOB_URI_COLUMN, into: str = "bytes"
) -> pa.RecordBatch:
    """Read offloaded payloads back into `into` and drop the `uri_col` handle.

    The exact inverse used by automatic offload: it restores the original schema (the
    payload back in `into`, the temporary handle column gone), so an offload→breaker→
    materialize rewrite is schema-transparent end to end.
    """
    out = read_blob_bytes(batch, uri_col=uri_col, into=into)
    if uri_col in out.schema.names:
        out = out.drop_columns([uri_col])
    return out


def default_blob_root() -> str:
    """The content-addressed blob store root from config: the remote spill URI if set
    (so handles are reachable cluster-wide), else the local spill scratch dir."""
    from batcher.config import active_config

    mem = active_config().memory
    base = mem.spill_remote_uri or mem.spill_dir or tempfile.gettempdir()
    return f"{base.rstrip('/')}/batcher-blobs"


def _fs_for(root: str) -> tuple[pafs.FileSystem, str]:
    """The pyarrow filesystem and base path for a blob-store `root` (URI or local path)."""
    if "://" in root:
        fs, path = pafs.FileSystem.from_uri(root)
        return fs, path.rstrip("/")
    return pafs.LocalFileSystem(), root.rstrip("/")


def offload_blob_bytes(
    batch: pa.RecordBatch,
    *,
    root: str,
    src: str = "bytes",
    uri_col: str = BLOB_URI_COLUMN,
) -> pa.RecordBatch:
    """Offload each row's ``src`` payload to `root` and replace it with a URI handle.

    Each non-null payload is written to ``{root}/{sha256}`` (skipped if it already
    exists — content addressing dedupes), the ``src`` column is nulled out (the
    payload now lives out of line), and a ``uri_col`` string column of handles is
    added. Designed to run inside `map_batches` with a small ``batch_size`` so only a
    few payloads are resident while writing. The inverse of `read_blob_bytes`.
    """
    fs, base = _fs_for(root)
    payloads = batch.column(src).to_pylist()
    # Local writes need the directory to exist; remote object stores ignore it. Once per
    # batch, not once per row: it was inside the loop, so a batch of a thousand payloads
    # made a thousand `create_dir` syscalls for a directory that exists after the first.
    if isinstance(fs, pafs.LocalFileSystem):
        fs.create_dir(base, recursive=True)

    # Content addressing dedupes *within* the batch too: two rows carrying the same
    # payload hash the same, so the write happens once and both get the same handle.
    digests = [None if data is None else hashlib.sha256(data).hexdigest() for data in payloads]
    pending: dict[str, bytes] = {}
    for digest, data in zip(digests, payloads, strict=True):
        if digest is not None:
            pending.setdefault(digest, data)

    def _write(_fs: pafs.FileSystem, path: str) -> None:
        digest = path.rsplit("/", 1)[-1]
        if fs.get_file_info(path).type == pafs.FileType.NotFound:
            with fs.open_output_stream(path) as stream:
                stream.write(pending[digest])

    # Each payload is one round trip against the store, and the write releases the GIL, so
    # a serial loop leaves an offload latency-bound on a single connection — on the one
    # path whose entire purpose is moving *large* payloads. `read_each_file` owns the
    # remote-concurrent / local-serial split, measured once and shared.
    read_each_file(fs, [f"{base}/{d}" for d in pending], _write)
    uris: list[str | None] = [None if d is None else _handle(root, d) for d in digests]

    out = batch
    # Null the payload column (the bytes are out of line now), then add the handles.
    null_src = pa.nulls(batch.num_rows, type=batch.schema.field(src).type)
    out = out.set_column(out.schema.get_field_index(src), src, null_src)
    uri_arr = pa.array(uris, pa.string())
    if uri_col in out.schema.names:
        return out.set_column(out.schema.get_field_index(uri_col), uri_col, uri_arr)
    return out.append_column(uri_col, uri_arr)


def _handle(root: str, digest: str) -> str:
    """The URI handle stored in the column — what `read_blob_bytes` reads back."""
    return f"{root.rstrip('/')}/{digest}"


def read_blob_bytes(
    batch: pa.RecordBatch, *, uri_col: str = "uri", into: str = "bytes"
) -> pa.RecordBatch:
    """Materialize file payloads for a batch of reference handles.

    Reads each row's `uri_col` file and writes its bytes into the `into` column
    (replacing it if present, else appending). Intended to run inside `map_batches`
    *after* filtering/sampling reference-mode handles, so only the surviving rows'
    payloads are ever read — and with a small ``batch_size``, only a few payloads
    are resident at once. The bytes land in a `large_binary` column, so a batch of
    GB-scale payloads cannot overflow 32-bit offsets.

    Examples:
        .. doctest::

            >>> import batcher as bt  # doctest: +SKIP
            >>> from batcher import col  # doctest: +SKIP
            >>> from batcher.io import read_blob_bytes  # doctest: +SKIP
            >>> ds = bt.read.video("s3://clips/", materialize_bytes=False)  # doctest: +SKIP
            >>> big = ds.filter(col("size") < 500_000_000)  # doctest: +SKIP
            >>> # ... metadata pruned first, so only surviving payloads are read.
            >>> decoded = big.map_batches(read_blob_bytes, batch_size=4)  # doctest: +SKIP

    Args:
        batch: A batch of reference handles, one row per file.
        uri_col: Column holding each row's file URI.
        into: Column the payload bytes are written to.

    Returns:
        `batch` with `into` holding each row's file contents (null where the URI
        is null).
    """
    uris = batch.column(uri_col).to_pylist()
    present = [u for u in uris if u is not None]
    # One filesystem for the batch, resolved from the first handle rather than per row.
    # Handles in a batch share a store by construction — they came from one `offload` with
    # one root — and re-resolving per row rebuilt a client (and re-read credentials) for
    # every payload.
    fs = resolve_filesystem(present[0]) if present else None

    def _read(_fs: object, uri: str) -> bytes:
        with fs.open(uri) as fh:  # type: ignore[union-attr]
            return fh.read()

    # Concurrent for a remote store: each payload is a round trip, and a serial loop over a
    # batch of handles is the read half of the same bottleneck the offload had. Local reads
    # stay serial, which `read_each_file` measured and owns.
    fetched = dict(zip(present, read_each_file(fs, present, _read), strict=True)) if present else {}
    blobs: list[bytes | None] = [None if u is None else fetched[u] for u in uris]
    # `large_binary` (64-bit offsets) so a batch of GB payloads can't overflow the
    # 2 GB limit of 32-bit `binary` — the whole point is large per-row payloads.
    arr = pa.array(blobs, pa.large_binary())
    if into in batch.schema.names:
        return batch.set_column(batch.schema.get_field_index(into), into, arr)
    return batch.append_column(into, arr)
