"""Row-capped writes — `max_rows_per_file` (H5).

Closes Ray Data's file-size pains: no single giant output file, a bounded row count
per file, and the cap honored *per partition* even with `partition_by` (where Ray
ignores `min_rows_per_file`).
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt


def test_max_rows_per_file_splits_into_parts(tmp_path):
    path = str(tmp_path / "out")
    bt.from_arrow(pa.table({"v": list(range(1000))})).write.parquet(path, max_rows_per_file=300)
    files = sorted(p for p in tmp_path.rglob("*.parquet"))
    assert len(files) == 4  # 300, 300, 300, 100
    total = sum(pq.read_table(str(f)).num_rows for f in files)
    assert total == 1000
    assert max(pq.read_table(str(f)).num_rows for f in files) <= 300


def test_cap_honored_per_partition(tmp_path):
    # Each partition independently respects the row cap (the Ray partition_cols bug).
    path = str(tmp_path / "ds")
    tbl = pa.table({"g": [0] * 250 + [1] * 250, "v": list(range(500))})
    bt.from_arrow(tbl).write.parquet(path, partition_by=["g"], max_rows_per_file=100)
    ds = tmp_path / "ds"
    g0 = sorted(ds.glob("g=0/*.parquet"))
    g1 = sorted(ds.glob("g=1/*.parquet"))
    assert len(g0) == 3 and len(g1) == 3  # 100,100,50 each
    for f in [*g0, *g1]:
        assert pq.read_table(str(f)).num_rows <= 100


def test_no_cap_writes_single_file(tmp_path):
    # Default (no cap, no partitioning) stays a single file — unchanged behavior.
    path = str(tmp_path / "out.parquet")
    bt.from_arrow(pa.table({"v": [1, 2, 3]})).write.parquet(path)
    assert pq.read_table(path).num_rows == 3


def test_degenerate_max_rows_per_file_rejected():
    # 0 would crash opaquely (range step zero); a negative value would silently write
    # NOTHING (empty range → data loss). Both must raise a clear error.
    from batcher._internal.errors import PlanError

    ds = bt.from_arrow(pa.table({"v": [1, 2, 3]}))
    for bad in (0, -5):
        with pytest.raises(PlanError, match="max_rows_per_file"):
            ds.write.parquet("/tmp/bt_bad_cap", max_rows_per_file=bad)
