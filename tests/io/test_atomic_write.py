"""Atomic single-file writes — a crash mid-write never destroys prior data.

Ray Data's `write_parquet` overwrite is non-atomic: a failure mid-write can leave
the target truncated and the old data gone (ray#62019). Batcher writes a local
file via a temp sibling + atomic rename, so an interrupted write leaves any prior
file intact and leaves no temp litter.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.io.formats import ParquetSink


class _CrashingParquetSink(ParquetSink):
    """A sink that writes some bytes then crashes — simulating an interrupted write."""

    def _write_file(self, table: pa.Table, fh) -> None:  # type: ignore[override]
        fh.write(b"corrupt-partial-bytes")
        raise RuntimeError("simulated crash mid-write")


def test_crash_preserves_prior_file_and_leaves_no_temp(tmp_path):
    path = str(tmp_path / "out.parquet")
    ParquetSink().write(pa.table({"x": [1, 2, 3]}), path)
    assert pq.read_table(path).num_rows == 3

    with pytest.raises(RuntimeError, match="simulated crash"):
        _CrashingParquetSink().write(pa.table({"x": [9, 9, 9, 9]}), path)

    # Prior file intact (not truncated to the partial bytes), no `.tmp` litter.
    assert pq.read_table(path).num_rows == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.parquet"]


def test_atomic_write_happy_path(tmp_path):
    path = str(tmp_path / "out.parquet")
    wf = ParquetSink().write(pa.table({"x": list(range(10))}), path)
    assert wf.rows == 10
    assert pq.read_table(path).num_rows == 10
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.parquet"]


def test_partitioned_shards_are_atomic(tmp_path):
    # write_partitioned routes through the same atomic write per shard.
    path = str(tmp_path / "ds")
    tbl = pa.table({"g": [1, 1, 2], "v": [10, 20, 30]})
    written = ParquetSink().write_partitioned(tbl, path, partition_by=["g"])
    assert len(written) == 2
    # No temp files anywhere under the partitioned output.
    leftovers = [p.name for p in tmp_path.rglob("*.tmp*")]
    assert leftovers == []
