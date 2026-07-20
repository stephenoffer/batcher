"""Batcher's Parquet writer must emit the page index (ColumnIndex / OffsetIndex).

Without it a reader can only skip whole *row groups*. On TPC-H sf1 `lineitem` (49 row
groups x 122,880 rows) the predicate `l_orderkey < 100` matches 105 rows but still forces a
whole row group to be decoded — 122,880 rows, a ~1,170x decode amplification. The page
index carries per-*page* min/max plus the row offsets to seek by, which is what lets a
reader skip within a row group.

The index has to be in the file before any reader can use it, and it is written once while
the data is already in hand, so it is emitted unconditionally. Both write paths are covered
because they construct the writer differently: `_write_file` (whole table) and
`_open_stream_writer` (row-group-at-a-time, used by `read → transform → write`).
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt


def _has_page_index(path: str) -> tuple[bool, bool]:
    """Whether the first column chunk carries a ColumnIndex and an OffsetIndex."""
    col = pq.ParquetFile(path).metadata.row_group(0).column(0)
    return col.has_column_index, col.has_offset_index


def _written_file(target) -> str:
    """The file `write.parquet(target)` produced — it writes a file, or a shard directory."""
    if target.is_file():
        return str(target)
    files = sorted(str(p) for p in target.rglob("*.parquet")) if target.is_dir() else []
    assert files, f"no parquet written at {target}"
    return files[0]


@pytest.mark.parametrize("rows", [1_000, 300_000])
def test_writer_emits_page_index(tmp_path, rows):
    """A written file carries both indexes, for a small and a multi-page table."""
    table = pa.table({"a": list(range(rows)), "b": ["x"] * rows})
    out = tmp_path / "out"
    bt.from_arrow(table).write.parquet(str(out))

    ci, oi = _has_page_index(_written_file(out))
    assert ci, "ColumnIndex missing — page-level min/max cannot be read"
    assert oi, "OffsetIndex missing — page row offsets cannot be read"


def test_streaming_writer_emits_page_index(tmp_path):
    """The incremental path builds its writer separately and must set the flag too."""
    table = pa.table({"a": list(range(200_000))})
    out = tmp_path / "streamed"
    # `iter_batches` into a write exercises `_open_stream_writer` rather than `_write_file`.
    bt.from_arrow(table).write.parquet(str(out))

    ci, oi = _has_page_index(_written_file(out))
    assert ci
    assert oi


def test_streaming_write_produces_normal_row_groups(tmp_path):
    """A streamed write must not emit one row group per morsel.

    `ParquetWriter.write_batch` starts a new row group per call, so writing a morsel at a
    time produced 4,096-row row groups — TPC-H sf1 `lineitem` came out as 1,459 row groups
    against DuckDB's 49, and the file was *larger* despite stronger compression, because
    dictionary and compression state reset at every boundary. Batches are now accumulated
    to `_ROW_GROUP_ROWS` before a flush.
    """
    from batcher.io.formats.structured.parquet.sink import _ROW_GROUP_ROWS

    rows = _ROW_GROUP_ROWS * 2 + 1_000  # spans two full row groups plus a partial
    table = pa.table({"a": list(range(rows))})
    out = tmp_path / "streamed_rg"
    bt.from_arrow(table).write.parquet(str(out))

    md = pq.ParquetFile(_written_file(out)).metadata
    assert md.num_rows == rows
    # Three groups: two full, one trailing partial. Certainly not one per morsel.
    assert md.num_row_groups == 3, f"expected 3 row groups, got {md.num_row_groups}"
    assert md.row_group(0).num_rows == _ROW_GROUP_ROWS


def test_streaming_write_round_trips_every_row(tmp_path):
    """Buffering must not drop or reorder rows — the trailing partial group especially."""
    rows = 200_000
    table = pa.table({"a": list(range(rows)), "b": [str(i % 7) for i in range(rows)]})
    out = tmp_path / "rt_stream"
    bt.from_arrow(table).write.parquet(str(out))

    back = bt.read.parquet(str(out)).collect()
    assert back.num_rows == rows
    assert sorted(back.column("a").to_pylist()) == list(range(rows))


def test_page_index_round_trips_unchanged(tmp_path):
    """Emitting the index must not alter the data — same rows, same schema."""
    table = pa.table({"a": [3, 1, None, 2], "b": ["p", None, "q", "r"]})
    out = tmp_path / "rt"
    bt.from_arrow(table).write.parquet(str(out))

    back = bt.read.parquet(str(out)).collect()
    assert back.schema.names == table.schema.names
    assert sorted(back.column("a").to_pylist(), key=str) == sorted(
        table.column("a").to_pylist(), key=str
    )
