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

__all__ = [
    "BLOB_URI_COLUMN",
    "default_blob_root",
    "materialize_and_drop_handle",
    "offload_blob_bytes",
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
    from batcher.io.formats.multimodal.media import read_blob_bytes

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
    uris: list[str | None] = []
    for data in payloads:
        if data is None:
            uris.append(None)
            continue
        digest = hashlib.sha256(data).hexdigest()
        path = f"{base}/{digest}"
        # Local writes need the directory to exist; remote object stores ignore it.
        if isinstance(fs, pafs.LocalFileSystem):
            fs.create_dir(base, recursive=True)
        if fs.get_file_info(path).type == pafs.FileType.NotFound:
            with fs.open_output_stream(path) as out:
                out.write(data)
        uris.append(_handle(root, digest))

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
