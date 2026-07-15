"""Distributed sort must size its range boundaries by the *reducer* count.

Regression for a bug hunt: the disk and Flight distributed sorts computed
``n_buckets = shuffle_partitions(workers)`` (which can be strictly less than the
mapper fan-out `workers` — the ``max_shuffle_partitions`` cap on a large cluster,
or the learned shuffle fan-out) but sampled boundaries with
``merge_boundaries(grids, workers)``. That emits up to ``workers - 1`` boundaries
for only ``n_buckets`` buckets, so the range partitioner routes rows into bucket
ids past the last bucket and the Rust ``range_partition_batches`` panics with an
out-of-bounds index — the whole distributed sort crashes. The fix sizes the
boundaries by ``n_buckets``. This test drives a real distributed sort with the
reducer cap forced below the worker fan-out and asserts it matches single-node.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    from conftest import init_test_ray, shutdown_test_ray

    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


@pytest.fixture
def _capped_shuffle():
    """Force `shuffle_partitions(4) < 4` by capping `max_shuffle_partitions` at 2."""
    from batcher.config import active_config, set_config

    cfg = active_config()
    set_config(
        dataclasses.replace(
            cfg, distributed=dataclasses.replace(cfg.distributed, max_shuffle_partitions=2)
        )
    )
    try:
        yield
    finally:
        set_config(cfg)


@pytest.mark.parametrize("transport", ["disk", "flight"])
@pytest.mark.parametrize("descending", [False, True])
def test_distributed_sort_reducer_cap_below_workers(_capped_shuffle, transport, descending):
    rng = np.random.default_rng(3)
    n = 5000
    t = pa.table({"k": rng.integers(0, 1000, n).astype("int64"), "v": np.arange(n).astype("int64")})

    single = bt.from_arrow(t).sort("k", descending=descending).collect()
    distrib = (
        bt.from_arrow(t)
        .sort("k", descending=descending)
        .collect(distributed=True, num_workers=4, transport=transport)
    )
    # Order-dependent: the sort key column must match position-by-position.
    assert single.column("k").to_pylist() == distrib.column("k").to_pylist()
    assert single.num_rows == distrib.num_rows


def test_distributed_topn_reducer_cap_below_workers(_capped_shuffle):
    rng = np.random.default_rng(5)
    n = 5000
    t = pa.table({"k": rng.integers(0, 1000, n).astype("int64"), "v": np.arange(n).astype("int64")})
    single = bt.from_arrow(t).sort("k").limit(20).collect().column("k").to_pylist()
    distrib = (
        bt.from_arrow(t)
        .sort("k")
        .limit(20)
        .collect(distributed=True, num_workers=4, transport="flight")
        .column("k")
        .to_pylist()
    )
    assert single == distrib


def test_distributed_sort_computed_key_reducer_cap(_capped_shuffle):
    """The computed-key path (hidden `__sort_key` column) also range-partitions."""
    rng = np.random.default_rng(9)
    n = 4000
    t = pa.table(
        {
            "a": rng.integers(0, 500, n).astype("int64"),
            "b": rng.integers(0, 500, n).astype("int64"),
        }
    )
    single = bt.from_arrow(t).sort(col("a") + col("b")).collect()
    distrib = (
        bt.from_arrow(t)
        .sort(col("a") + col("b"))
        .collect(distributed=True, num_workers=4, transport="disk")
    )
    key_single = [r["a"] + r["b"] for r in single.to_pylist()]
    key_distrib = [r["a"] + r["b"] for r in distrib.to_pylist()]
    assert key_single == key_distrib
