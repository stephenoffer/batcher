"""Native Rust Parquet reads (via `bc_io` through `batcher._native`), with PyArrow fallback.

`bc_io` fetches a file's projected column chunks concurrently from object storage and
decodes them in Rust, returning zero-copy Arrow batches — no Python-handle round trip and
no FFI copy. Measured 3-4x faster than PyArrow on S3 (100 small files 243ms vs 943ms; one
8.4M-row file 143ms vs 484ms). Every function returns ``None`` on any unsupported
scheme/feature (or a missing extension) so the caller falls back to PyArrow and the result
is byte-identical either way.
"""

from __future__ import annotations

import json

import pyarrow as pa

from batcher._internal.native import engine

__all__ = ["NATIVE_READ_BATCH", "read_many", "read_one", "read_row_groups_filtered"]

# Read batch size handed to the native reader; the engine re-morselizes downstream, so a
# larger read batch just trades a few big Arrow batches for better decode throughput.
NATIVE_READ_BATCH = 65536


def read_one(uri: str, projection: list[str] | None) -> list[pa.RecordBatch] | None:
    """One whole Parquet file's batches via the native reader, or ``None`` to fall back."""
    try:
        _native = engine()
        return _native.read_parquet(uri, [], projection, NATIVE_READ_BATCH)
    except Exception:
        return None


def read_row_groups_filtered(
    uri: str,
    row_groups: list[int],
    projection: list[str] | None,
    predicate: dict | None,
    batch_size: int = NATIVE_READ_BATCH,
) -> list[pa.RecordBatch] | None:
    """Read `row_groups` with a pushed `predicate` applied as native row-group pruning.

    `predicate` is the IR dict; its pushable subset is translated (`to_native_predicate`)
    to the reader's compact form and used to skip row-groups whose footer statistics prove
    no row can match — the reader never fetches or decodes those column chunks. Pruning is
    superset-safe (the engine keeps the `Filter`), so a non-pushable predicate reads every
    requested row-group. Returns ``None`` on any failure (caller falls back to PyArrow).
    """
    try:
        _native = engine()
        from batcher.io.predicate import to_native_predicate

        native_pred = to_native_predicate(predicate) if predicate is not None else None
        if native_pred is None:
            return _native.read_parquet(uri, row_groups, projection, batch_size)
        return _native.read_parquet_filtered(
            uri, row_groups, projection, batch_size, json.dumps(native_pred)
        )
    except Exception:
        return None


def read_many(uris: list[str], projection: list[str] | None) -> list[list[pa.RecordBatch]] | None:
    """Many whole Parquet files in one native pass (per-file batch lists), or ``None``.

    The many-small-files throughput path: one GIL release + one runtime pass overlaps every
    file's footer + column-chunk GETs, instead of a per-file call (and FFI round trip) each.
    """
    try:
        _native = engine()
        return _native.read_parquet_many(uris, projection, NATIVE_READ_BATCH)
    except Exception:
        return None
