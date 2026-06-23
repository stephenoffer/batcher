"""I/O format coverage: JSON round-trip and multi-file Parquet reads.

These exercise the source/sink layer directly (not the full engine), so they run
without the native engine. JSON goes out via `JSONSink` and back via the
`read.json` session entry point; multi-file Parquet exercises directory/glob
expansion, projection pushdown, footer-summed row counts, and streaming.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from batcher.api.io_namespace import read
from batcher.io.sink import JSONSink
from batcher.io.source import JSONSource, ParquetSource


def _sorted_pydict(table: pa.Table) -> dict:
    """Sort rows by all columns so row order is irrelevant to equality checks."""
    if table.num_rows == 0:
        return table.to_pydict()
    rows = sorted(table.to_pylist(), key=lambda r: tuple(r[c] for c in table.column_names))
    return pa.Table.from_pylist(rows, schema=table.schema).to_pydict()


def test_json_roundtrip_via_sink_and_read_json(tmp_path):
    table = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"], "v": [1.5, 2.5, 3.5]})
    path = str(tmp_path / "data.json")

    JSONSink().write(table, path)

    # `read.json` builds a Dataset over a `JSONSource`; read through the source
    # directly so the round-trip test does not require the native engine.
    assert read.json(path) is not None
    src = JSONSource(path)
    batches = src.read()
    got = pa.Table.from_batches(batches)

    assert _sorted_pydict(got) == _sorted_pydict(table)
    assert src.schema().names == table.schema.names


def test_json_source_projection(tmp_path):
    table = pa.table({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    path = str(tmp_path / "p.json")
    JSONSink().write(table, path)

    src = JSONSource(path)
    got = pa.Table.from_batches(src.read(projection=["a", "c"]))
    assert got.column_names == ["a", "c"]
    assert _sorted_pydict(got) == _sorted_pydict(table.select(["a", "c"]))


def _write_three_parquet(dir_path) -> tuple[list[str], pa.Table]:
    files = []
    tables = []
    for i in range(3):
        t = pa.table(
            {
                "k": [i * 10 + 0, i * 10 + 1],
                "name": [f"r{i}a", f"r{i}b"],
                "extra": [i, i],
            }
        )
        f = str(dir_path / f"part-{i}.parquet")
        # Two row groups per file so iter_batches yields more than one batch/file.
        pq.write_table(t, f, row_group_size=1)
        files.append(f)
        tables.append(t)
    return files, pa.concat_tables(tables)


def test_parquet_directory_reads_all_files(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    _files, combined = _write_three_parquet(d)

    src = ParquetSource(str(d))
    assert src.row_count() == combined.num_rows == 6

    got = pa.Table.from_batches(src.read())
    assert _sorted_pydict(got) == _sorted_pydict(combined)
    assert src.schema().names == combined.schema.names


def test_parquet_glob_reads_all_files(tmp_path):
    d = tmp_path / "globds"
    d.mkdir()
    _files, combined = _write_three_parquet(d)

    src = ParquetSource(str(d / "*.parquet"))
    assert src.row_count() == 6
    got = pa.Table.from_batches(src.read())
    assert _sorted_pydict(got) == _sorted_pydict(combined)


def test_parquet_multifile_projection_reads_subset(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    _files, combined = _write_three_parquet(d)

    src = ParquetSource(str(d))
    got = pa.Table.from_batches(src.read(projection=["k"]))
    assert got.column_names == ["k"]
    assert _sorted_pydict(got) == _sorted_pydict(combined.select(["k"]))


def test_parquet_multifile_iter_batches_streams_and_matches_read(tmp_path):
    d = tmp_path / "stream"
    d.mkdir()
    _files, combined = _write_three_parquet(d)

    src = ParquetSource(str(d))
    streamed = list(src.iter_batches())
    # 3 files x 2 row groups each = at least several batches (streaming, not one).
    assert len(streamed) > 1

    streamed_tbl = pa.Table.from_batches(streamed)
    read_tbl = pa.Table.from_batches(src.read())
    assert _sorted_pydict(streamed_tbl) == _sorted_pydict(read_tbl)
    assert _sorted_pydict(streamed_tbl) == _sorted_pydict(combined)


def test_parquet_single_file_behavior_unchanged(tmp_path):
    t = pa.table({"x": [1, 2, 3]})
    f = str(tmp_path / "single.parquet")
    pq.write_table(t, f)

    src = ParquetSource(f)
    assert src.row_count() == 3
    assert pa.Table.from_batches(src.read()).to_pydict() == t.to_pydict()
    assert src.identity() == f"parquet:{f}"
