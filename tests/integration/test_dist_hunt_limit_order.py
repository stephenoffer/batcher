"""Distributed `LIMIT` / `with_row_index` must preserve the source's row order.

Regression for a bug hunt. Both paths run the input through `_distributed_map` and then
either slice (`LIMIT`) or number (`with_row_index`) the concatenation of the per-partition
results, assuming that concatenation reproduces the source's global row order. It did not:
`partition_descriptors` assigned splits to partitions with `_balance` (largest-first, for
even load), so when a source has more splits than partitions a single partition holds
NON-adjacent splits (worker 0 got row-groups ``[0, 4]``). A per-partition ``LIMIT`` then
interleaved rows from different parts of the source, and the driver-side slice / row
numbering diverged from single-node:

* ``limit(1500)`` over a monotonic ``0..7999`` returned ``{0..999, 4000..4499}`` instead of
  ``{0..1499}`` — a different row set.
* ``with_row_index`` gave the row with ``id == 1000`` the index ``2000`` instead of ``1000``.

Both are distributed-only wrong answers (the single-node == distributed invariant). The fix
assigns splits as contiguous source-ordered runs (`_contiguous`) for these two order-sensitive
paths, so the partition-index-assembled concatenation is the source's row order again.

`target_rows_per_task` is forced small so the fan-out (4) is below the split count (8) — the
condition that makes a partition span non-adjacent splits. It is the everyday large-data
shape (more row-groups than workers), reproduced here at small scale.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    from conftest import init_test_ray, shutdown_test_ray

    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


@pytest.fixture
def _fewer_partitions_than_splits():
    """Force the map fan-out below the split count so a partition spans non-adjacent splits."""
    from batcher.config import active_config, set_config

    cfg = active_config()
    set_config(
        dataclasses.replace(
            cfg, optimizer=dataclasses.replace(cfg.optimizer, target_rows_per_task=4000)
        )
    )
    try:
        yield
    finally:
        set_config(cfg)


@pytest.fixture
def _monotonic_parquet(tmp_path):
    """An 8-row-group Parquet (id == 0..7999, monotonic) on worker-readable storage.

    On a multi-node cluster the workers read the file directly, so it must live on a shared
    mount; on a single-node laptop / CI box `tmp_path` is reachable everywhere.
    """
    from batcher.dist.shuffle_io import shared_scratch_root

    root = shared_scratch_root()
    if root is not None:
        base = os.path.join(root, f"limit_order_{os.getpid()}")
        os.makedirs(base, exist_ok=True)
    else:
        base = str(tmp_path)
    path = os.path.join(base, "mono.parquet")
    ids = np.arange(8000, dtype="int64")
    pq.write_table(pa.table({"id": ids, "v": ids * 2}), path, row_group_size=1000)
    return path


@pytest.mark.integration
@pytest.mark.parametrize("lim", [1500, 2500, 3500])
def test_distributed_limit_preserves_source_order(
    _monotonic_parquet, _fewer_partitions_than_splits, lim
):
    single = bt.read.parquet(_monotonic_parquet).limit(lim).collect().column("id").to_pylist()
    dist = (
        bt.read.parquet(_monotonic_parquet)
        .limit(lim)
        .collect(distributed=True, num_workers=4)
        .column("id")
        .to_pylist()
    )
    # The limit is order-sensitive: the first `lim` rows of the monotonic source are 0..lim-1.
    assert single == list(range(lim))
    assert dist == single


@pytest.mark.integration
def test_distributed_limit_offset_preserves_source_order(
    _monotonic_parquet, _fewer_partitions_than_splits
):
    single = (
        bt.read.parquet(_monotonic_parquet)
        .limit(1200, offset=800)
        .collect()
        .column("id")
        .to_pylist()
    )
    dist = (
        bt.read.parquet(_monotonic_parquet)
        .limit(1200, offset=800)
        .collect(distributed=True, num_workers=4)
        .column("id")
        .to_pylist()
    )
    assert single == list(range(800, 2000))
    assert dist == single


@pytest.mark.integration
def test_distributed_with_row_index_preserves_source_order(
    _monotonic_parquet, _fewer_partitions_than_splits
):
    single = bt.read.parquet(_monotonic_parquet).with_row_index("idx").collect()
    dist = (
        bt.read.parquet(_monotonic_parquet)
        .with_row_index("idx")
        .collect(distributed=True, num_workers=4)
    )
    # ids are 0..7999 in order, so the global row index must equal the id for every row.
    single_map = dict(
        zip(single.column("id").to_pylist(), single.column("idx").to_pylist(), strict=True)
    )
    dist_map = dict(zip(dist.column("id").to_pylist(), dist.column("idx").to_pylist(), strict=True))
    assert single_map == {i: i for i in range(8000)}
    assert dist_map == single_map
