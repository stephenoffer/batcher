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
from dataclasses import dataclass

import pyarrow as pa

from batcher._internal.native import engine

__all__ = [
    "NATIVE_READ_BATCH",
    "FooterStats",
    "file_manifest",
    "footer_stats",
    "read_many",
    "read_one",
    "read_row_groups_filtered",
]

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


@dataclass(frozen=True)
class FooterStats:
    """Aggregated Parquet footer statistics for a set of files, computed natively.

    `bounds` is a 2-row Arrow table — **row 0 = min, row 1 = max** — with one column per
    entry of `columns`, each keeping its own Arrow type, so a bound is read out with
    `bounds.column(name)[0].as_py()` and never passes through a string. A null bound means
    *unknown*, never *no value*.

    `files_read` below the number of files requested means at least one footer was
    unreadable: the row count then covers only the files that were read, so a caller must
    not publish it as exact.
    """

    columns: tuple[tuple[str, bool, int, bool, bool, int | None], ...]
    bounds: pa.Table
    total_rows: int
    total_bytes: int
    row_group_count: int
    files_read: int
    sort_declared: bool


def footer_stats(uris: list[str]) -> FooterStats | None:
    """Aggregate `uris`' Parquet footers natively, or ``None`` to fall back to PyArrow.

    Replaces a per-column-chunk Python walk (O(files x row_groups x columns) pybind11
    objects on the driver, before any data page is read) with one native pass over footers
    the reader has usually already parsed and cached.
    """
    if not uris:
        return None
    try:
        batch, columns, rows, nbytes, rgs, files_read, sorted_decl = engine().parquet_footer_stats(
            uris
        )
    except Exception:
        return None
    return FooterStats(
        columns=tuple(columns),
        bounds=pa.Table.from_batches([batch]),
        total_rows=rows,
        total_bytes=nbytes,
        row_group_count=rgs,
        files_read=files_read,
        sort_declared=sorted_decl,
    )


def file_manifest(uris: list[str], columns: list[str]) -> pa.Table | None:
    """Per-file bounds for `columns` in the add-action layout, or ``None`` to fall back.

    ``path | num_records | min.<col> | max.<col> | null_count.<col>`` — one row per file, in
    URI order, built natively from footers the statistics pass has usually already cached.
    A NULL bound means *unknown* (keep the file), never *no match*.
    """
    if not uris or not columns:
        return None
    try:
        return pa.Table.from_batches([engine().parquet_file_manifest(uris, columns)])
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
