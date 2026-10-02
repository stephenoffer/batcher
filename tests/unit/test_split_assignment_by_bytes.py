"""Scan splits are balanced by decode work when their bytes are known, not by row count.

Rows are a proxy for read work that fails exactly where a scan is heterogeneous: a row group
of a wide table and one of a narrow table can hold the same number of rows and differ by
two orders of magnitude in what they cost to decode. `RowGroupSplit` records the run's
uncompressed bytes from the footer at planning time, so balancing by them is free.
"""

from __future__ import annotations

import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.dist.executors.partition_io.assignment import _balance
from batcher.io.splits.parquet import RowGroupSplit, parquet_row_group_splits

pytestmark = pytest.mark.unit


def _loads(groups):
    return sorted(sum(s.nbytes for s in g) for g in groups)


def test_wide_and_narrow_splits_balance_by_bytes():
    # Four splits of equal rows: one wide (100 MB), three narrow (1 MB each). By rows the
    # packer sees four equal splits and pairs the wide one with a narrow one; by bytes it
    # isolates the wide split, which is the only balanced answer.
    splits = [RowGroupSplit(f"f{i}.parquet", (0,), 1000, b) for i, b in enumerate([100, 1, 1, 1])]
    groups = _balance(splits, 2)
    assert _loads(groups) == [3, 100]


def test_without_bytes_on_every_split_it_falls_back_to_rows():
    """All-or-nothing: bytes on one split and rows on another are not comparable units."""
    splits = [
        RowGroupSplit("a.parquet", (0,), 10, 100),
        RowGroupSplit("b.parquet", (0,), 30, None),
        RowGroupSplit("c.parquet", (0,), 10, 100),
        RowGroupSplit("d.parquet", (0,), 10, 100),
    ]
    groups = _balance(splits, 2)
    rows = sorted(sum(s.rows for s in g) for g in groups)
    assert rows == [30, 30]


def test_planned_row_group_splits_carry_their_bytes(tmp_path):
    path = os.path.join(tmp_path, "t.parquet")
    table = pa.table({"x": list(range(10_000)), "s": ["y" * 50] * 10_000})
    pq.write_table(table, path, row_group_size=2_500)
    splits = parquet_row_group_splits(path, None)
    meta = pq.ParquetFile(path).metadata
    assert splits and all(isinstance(s.nbytes, int) and s.nbytes > 0 for s in splits)
    assert sum(s.nbytes for s in splits) == sum(
        meta.row_group(i).total_byte_size for i in range(meta.num_row_groups)
    )
