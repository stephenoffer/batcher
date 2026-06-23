"""Distributed aggregate/join where a filter empties some partitions.

Regression: a partition fully eliminated by the filter (or by predicate pushdown)
left the native partial-aggregate with no schema ("aggregation over empty input").
`read_partition` now backfills a schema-only batch for an emptied partition, so
the distributed result equals the single-node result.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt


def _write(tmp_path) -> str:
    path = str(tmp_path / "t.parquet")
    # 10 row groups; a filter of x>800 empties row-groups 0..7 entirely.
    pq.write_table(
        pa.table({"x": list(range(1000)), "k": [i % 5 for i in range(1000)]}),
        path,
        row_group_size=100,
    )
    return path


@pytest.mark.integration
def test_distributed_global_aggregate_with_emptying_filter(tmp_path):
    path = _write(tmp_path)
    single = bt.read.parquet(path).filter(bt.col("x") > 800).group_by().agg(n=bt.count()).collect()
    dist = (
        bt.read.parquet(path)
        .filter(bt.col("x") > 800)
        .group_by()
        .agg(n=bt.count())
        .collect(distributed=True, num_workers=4)
    )
    assert single.to_pydict() == dist.to_pydict() == {"n": [199]}


@pytest.mark.integration
def test_distributed_grouped_aggregate_with_emptying_filter(tmp_path):
    path = _write(tmp_path)

    def q(**kw):
        ds = bt.read.parquet(path).filter(bt.col("x") > 800).group_by("k").agg(s=bt.col("x").sum())
        return sorted(ds.collect(**kw).to_pylist(), key=lambda r: r["k"])

    assert q() == q(distributed=True, num_workers=4)
