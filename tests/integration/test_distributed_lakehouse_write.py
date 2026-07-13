"""Distributed lakehouse writes must be correct and lose no data.

The scalable lakehouse write has each worker write its shard as a real Parquet file
(shared-nothing, bounded memory) and the driver commit them in one transaction — Delta
streams the staged files into a single ``write_deltalake``; Iceberg ``add_files`` registers
them. These guard the regression where the old buffer-in-memory design silently wrote
nothing on the distributed path (a worker's buffer never reached the committing driver
sink), and check the distributed result equals the single-node result.
"""

from __future__ import annotations

import os

import pyarrow as pa
import pytest

import batcher as bt


def _src(tmp_path) -> tuple[str, int, int]:
    src = str(tmp_path / "src")
    os.makedirs(src)
    import pyarrow.parquet as pq

    total_rows = 0
    total_sum = 0
    for i in range(4):
        vals = list(range(i * 100, (i + 1) * 100))
        pq.write_table(pa.table({"a": vals, "r": [chr(65 + i % 3)] * 100}), f"{src}/p{i}.parquet")
        keep = [v for v in vals if v >= 50]
        total_rows += len(keep)
        total_sum += sum(keep)
    return src, total_rows, total_sum


@pytest.mark.integration
def test_delta_distributed_write_matches_single_node(tmp_path):
    pytest.importorskip("deltalake")
    src, rows, total = _src(tmp_path)

    def write(out: str, **kw) -> None:
        bt.read.parquet(f"{src}/*.parquet").filter(bt.col("a") >= 50).write.delta(out, **kw)

    single = str(tmp_path / "d_single")
    dist = str(tmp_path / "d_dist")
    write(single)
    write(dist, distributed=True, num_workers=3)

    for out in (single, dist):
        agg = (
            bt.read.delta(out).agg(c=bt.col("a").count(), s=bt.col("a").sum()).collect().to_pydict()
        )
        assert agg["c"][0] == rows, out
        assert agg["s"][0] == total, out


@pytest.mark.integration
def test_delta_distributed_partitioned_write(tmp_path):
    pytest.importorskip("deltalake")
    src, _rows, _total = _src(tmp_path)
    out = str(tmp_path / "d_part")
    bt.read.parquet(f"{src}/*.parquet").write.delta(
        out, partition_by=["r"], distributed=True, num_workers=3
    )

    assert bt.read.delta(out).agg(c=bt.col("a").count()).collect().to_pydict()["c"][0] == 400
    assert sorted(d for d in os.listdir(out) if d.startswith("r=")) == ["r=A", "r=B", "r=C"]
    # Staging scratch is removed after a successful commit.
    assert not os.path.exists(os.path.join(out, "_batcher_staging"))


@pytest.mark.integration
def test_iceberg_distributed_write_and_append(tmp_path):
    pytest.importorskip("pyiceberg")
    from batcher.io.catalog import resolve_catalog

    src, rows, total = _src(tmp_path)
    wh = str(tmp_path / "wh")
    os.makedirs(wh)
    cat = {"type": "sql", "uri": f"sqlite:///{tmp_path}/c.db", "warehouse": f"file://{wh}"}
    resolve_catalog(cat).create_namespace("ns")

    def q():
        return bt.read.parquet(f"{src}/*.parquet").filter(bt.col("a") >= 50)

    q().write.iceberg("ns.t", catalog=cat, distributed=True, num_workers=3)
    agg = bt.read.iceberg("ns.t", catalog=cat).agg(c=bt.col("a").count(), s=bt.col("a").sum())
    got = agg.collect().to_pydict()
    assert (got["c"][0], got["s"][0]) == (rows, total)

    # A second distributed append uses a fresh write token, so it adds without clobbering
    # the first snapshot's referenced files.
    q().write.iceberg("ns.t", catalog=cat, mode="append", distributed=True, num_workers=2)
    c2 = (
        bt.read.iceberg("ns.t", catalog=cat)
        .agg(c=bt.col("a").count())
        .collect()
        .to_pydict()["c"][0]
    )
    assert c2 == 2 * rows
