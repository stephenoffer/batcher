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

from batcher._internal.logging import note_suppressed
from batcher._internal.native import engine

__all__ = [
    "NATIVE_READ_BATCH",
    "NATIVE_READ_TARGET_BYTES",
    "FooterStats",
    "file_manifest",
    "footer_stats",
    "native_read_batch",
    "read_many",
    "read_one",
    "read_row_groups_filtered",
]

# Read batch size handed to the native reader; the engine re-morselizes downstream, so a
# larger read batch just trades a few big Arrow batches for better decode throughput.
NATIVE_READ_BATCH = 65536

# Bytes one decoded batch is aimed at, capping `NATIVE_READ_BATCH` for a wide row.
#
# The row count alone is not a memory bound, and the difference is not marginal on the shape
# it misses. A Parquet decode's working set is several times the batch it produces — the
# physical column buffer (a `uint8` tensor child is INT32 on disk, so 4x), the definition
# levels beside it, and the cast to the Arrow type — so the batch size *is* the reader's
# residency. At 65,536 rows that is ~4 MB for an ordinary 64-byte row and **9.6 GB** for a
# decoded 224x224x3 image, which is why a GPU inference actor holding one 31 GB node's worth
# of images was OOM-killed by the kernel rather than by anything the engine could see.
#
# Measured on 4,000 such images (0.56 GB) read through `collect()`: **10.16 GB peak RSS in
# 5.82 s at 65,536 rows against 2.34 GB in 4.89 s at 64** — 4.3x the memory *and* 1.2x the
# time, for the same result and the same batch count out (the engine re-morselizes either
# way). Smaller is not a trade here; it is strictly better once a row is wide.
#
# 16 MiB is deliberately generous: it is what 65,536 rows of 256 bytes cost, so it does not
# bind on any ordinary table (TPC-H `lineitem` is ~130 bytes/row) and only shrinks the read
# for genuinely wide rows. The cap never raises the caller's request.
NATIVE_READ_TARGET_BYTES = 16 << 20


def native_read_batch(
    schema: pa.Schema | None,
    projection: list[str] | None = None,
    ceiling: int = NATIVE_READ_BATCH,
) -> int:
    """Rows to decode at once: `ceiling`, capped so one batch stays near the byte target.

    Returns `ceiling` unchanged when the schema is unknown or the row is narrow enough that
    the cap does not bind, so this can be dropped in wherever the flat row count was used.

    Args:
        schema: The Arrow schema being read, or ``None`` when it isn't known here.
        projection: Columns actually read, or ``None`` for all of them.
        ceiling: The largest batch the caller wants; never raised.

    Returns:
        A row count in ``[1, ceiling]``.
    """
    if schema is None:
        return ceiling
    from batcher.plan.types.widths import projected_row_bytes

    row_bytes = projected_row_bytes(schema, projection)
    if row_bytes <= 0:
        return ceiling
    return max(1, min(ceiling, int(NATIVE_READ_TARGET_BYTES // row_bytes)))


def read_one(
    uri: str, projection: list[str] | None, schema: pa.Schema | None = None
) -> list[pa.RecordBatch] | None:
    """One whole Parquet file's batches via the native reader, or ``None`` to fall back.

    `schema` sizes the decode batch by bytes rather than rows (`native_read_batch`); without
    it the flat row ceiling stands, which is only safe for a narrow row.
    """
    try:
        _native = engine()
        return _native.read_parquet(uri, [], projection, native_read_batch(schema, projection))
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
    no row can match — the reader never fetches or decodes those column chunks. The reader
    may *also* apply the predicate row-by-row during decode when it measures that worth doing,
    so the result is anywhere between the exactly-matching rows and every requested row-group.
    Both are correct: `to_native_predicate` is all-or-nothing, so a predicate that reaches the
    reader is a complete translation of the `Filter`, and the engine keeps that `Filter`
    regardless (`core.scan_only` declines its shortcut whenever a predicate was pushed).
    Returns ``None`` on any failure (caller falls back to PyArrow).
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
    except Exception as exc:
        note_suppressed("io", "read parquet natively", exc)
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


def read_many(
    uris: list[str], projection: list[str] | None, schema: pa.Schema | None = None
) -> list[list[pa.RecordBatch]] | None:
    """Many whole Parquet files in one native pass (per-file batch lists), or ``None``.

    The many-small-files throughput path: one GIL release + one runtime pass overlaps every
    file's footer + column-chunk GETs, instead of a per-file call (and FFI round trip) each.
    """
    try:
        _native = engine()
        return _native.read_parquet_many(uris, projection, native_read_batch(schema, projection))
    except Exception:
        return None
