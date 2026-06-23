"""Distributed reads of a partitioned Parquet dataset (distributed listing).

`ParquetDatasetSource` splits per top-level partition directory so each worker
lists only its own subtree; the distributed result must equal single-node, with
projection + predicate pushed and partition columns recovered.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


def _write_dataset(tmp_path) -> str:
    out = str(tmp_path / "p")
    bt.from_arrow(
        pa.table({"k": [i % 8 for i in range(2000)], "v": list(range(2000))})
    ).write.parquet(out, partition_by=["k"])
    return out


@pytest.mark.integration
def test_dataset_distributed_scan_matches_single_node(tmp_path):
    out = _write_dataset(tmp_path)
    single = bt.read.parquet_dataset(out).collect().num_rows
    dist = bt.read.parquet_dataset(out).collect(distributed=True, num_workers=4).num_rows
    assert single == dist == 2000


@pytest.mark.integration
def test_dataset_distributed_filter_aggregate_matches_single_node(tmp_path):
    out = _write_dataset(tmp_path)

    def q(**kw):
        ds = (
            bt.read.parquet_dataset(out)
            .filter(bt.col("v") >= 1000)
            .group_by("k")
            .agg(s=bt.col("v").sum())
        )
        return sorted(ds.collect(**kw).to_pylist(), key=lambda r: r["k"])

    assert q() == q(distributed=True, num_workers=4)


@pytest.mark.integration
def test_dataset_partition_column_recovered(tmp_path):
    out = _write_dataset(tmp_path)
    table = bt.read.parquet_dataset(out).filter(bt.col("k") == 3).collect()
    assert set(table.column("k").to_pylist()) == {3}
