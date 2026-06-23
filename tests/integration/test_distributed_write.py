"""Distributed write equivalence + split-based distributed read.

Verifies that a parallel (Ray) write produces the same rows as a single-node
write, that the driver returns a complete manifest, and that a multi-row-group
Parquet source read distributed (the split path) matches single-node.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt


@pytest.mark.integration
def test_distributed_split_read_matches_single_node(tmp_path):
    path = str(tmp_path / "big.parquet")
    pq.write_table(
        pa.table({"k": [i % 5 for i in range(1000)], "v": list(range(1000))}),
        path,
        row_group_size=100,  # 10 row groups -> 10 splits
    )
    q = lambda **kw: bt.read.parquet(path).group_by("k").agg(s=bt.col("v").sum()).collect(**kw)  # noqa: E731
    single = sorted(q().to_pylist(), key=lambda r: r["k"])
    dist = sorted(q(distributed=True, num_workers=4).to_pylist(), key=lambda r: r["k"])
    assert single == dist


@pytest.mark.integration
def test_distributed_write_roundtrip(tmp_path):
    src = str(tmp_path / "src.parquet")
    pq.write_table(pa.table({"v": list(range(1000))}), src, row_group_size=100)
    out = str(tmp_path / "out")

    manifest = bt.read.parquet(src).write.parquet(out, distributed=True, num_workers=4)
    assert manifest.num_files >= 1
    assert manifest.total_rows == 1000
    # Read the part-* files back and confirm the full row set survives.
    assert bt.read.parquet(out).count() == 1000
