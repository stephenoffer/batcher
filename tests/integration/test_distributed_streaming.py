"""Distributed streaming drain — parallel cluster backfill of a stream.

A `distributed=True` streaming write with an `available_now`/`once` trigger over a
splittable source fans read+transform+write across workers (Spark `Trigger.AvailableNow`).
The mergeable/shared-nothing write means the result is identical to the single-node
drain. This proves that equivalence over a real, splittable Parquet source (multiple
row-groups → multiple splits, read in parallel under Ray local) and that the unsupported
distributed streaming shapes fail loudly instead of silently degrading to single-node.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.integration


def _splittable_parquet(path: str, n: int) -> str:
    # Small row groups so the source yields several splits (fanned across workers).
    pq.write_table(pa.table({"id": list(range(n))}), path, row_group_size=max(1, n // 6))
    return path


def _ids(path: str) -> list[int]:
    return sorted(bt.read(path, format="parquet").to_pydict()["id"])


def test_distributed_drain_matches_single_node(tmp_path):
    src = _splittable_parquet(str(tmp_path / "in.parquet"), 300)
    single = str(tmp_path / "single")
    dist = str(tmp_path / "dist")

    bt.read(src, format="parquet").write(
        single, format="parquet", trigger=bt.Trigger.available_now()
    ).await_termination()
    bt.read(src, format="parquet").write(
        dist, format="parquet", trigger=bt.Trigger.available_now(), distributed=True, num_workers=3
    ).await_termination()

    # Every row drained exactly once; the two paths agree on content.
    assert _ids(single) == list(range(300))
    assert _ids(dist) == list(range(300))


def test_distributed_drain_with_filter_matches_single_node(tmp_path):
    src = _splittable_parquet(str(tmp_path / "in.parquet"), 200)
    single = str(tmp_path / "single")
    dist = str(tmp_path / "dist")
    pred = bt.col("id") % 2 == 0  # a stateless transform fanned across workers

    bt.read(src, format="parquet").filter(pred).write(
        single, format="parquet", trigger=bt.Trigger.available_now()
    ).await_termination()
    bt.read(src, format="parquet").filter(pred).write(
        dist, format="parquet", trigger=bt.Trigger.available_now(), distributed=True, num_workers=4
    ).await_termination()

    assert _ids(single) == _ids(dist) == list(range(0, 200, 2))


def test_distributed_drain_returns_streaming_query(tmp_path):
    src = _splittable_parquet(str(tmp_path / "in.parquet"), 120)
    q = bt.read(src, format="parquet").write(
        str(tmp_path / "o"),
        format="parquet",
        trigger=bt.Trigger.available_now(),
        distributed=True,
    )
    assert not q.is_active  # a drain runs to completion before the handle returns
    assert q.await_termination() is True
    assert q.last_progress is not None and q.last_progress.num_output_rows == 120


def test_distributed_continuous_trigger_raises(tmp_path):
    # A non-drain distributed streaming request must fail loudly, not silently run local.
    src = _splittable_parquet(str(tmp_path / "in.parquet"), 10)
    with pytest.raises(PlanError, match="available_now"):
        bt.read(src, format="parquet").write(
            str(tmp_path / "o"),
            format="parquet",
            trigger=bt.Trigger.processing_time(0),
            distributed=True,
        )


def test_distributed_checkpoint_raises(tmp_path):
    src = _splittable_parquet(str(tmp_path / "in.parquet"), 10)
    with pytest.raises(PlanError, match="checkpoint"):
        bt.read(src, format="parquet").write(
            str(tmp_path / "o"),
            format="parquet",
            trigger=bt.Trigger.available_now(),
            distributed=True,
            checkpoint=str(tmp_path / "ckpt"),
        )
