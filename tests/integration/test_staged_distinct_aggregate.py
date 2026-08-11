"""Aggregating a distributed dedup must not funnel its rows through the driver.

`COUNT(DISTINCT x)` lowers to an aggregate over a `Distinct`. The dedup has to run globally
first — a map-local one counts a value once per partition it appears in — and what happened
after that was a driver-side fold over every distinct value: Θ(cardinality) on one node,
which no cluster size touches. `_staged_aggregate_over_distinct` runs the aggregate as a
second distributed stage instead, and asks the dedup's own partitioned intermediate how many
rows it wrote to decide whether that is worth doing.

Both arms need exercising and only one of them shows up on ordinary data. The threshold is
lowered here rather than the data made enormous, because a million distinct values would
make this a benchmark rather than a test.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    from batcher.dist.shuffle_io import shared_scratch_root

    root = shared_scratch_root()
    base = (
        os.path.join(root, f"staged_distinct_{os.getpid()}")
        if root
        else str(tmp_path_factory.mktemp("staged"))
    )
    os.makedirs(base, exist_ok=True)
    rng = np.random.default_rng(19)
    for i in range(8):
        n = 5_000
        pq.write_table(
            pa.table(
                {
                    "k": rng.integers(0, 900, n).astype("int64"),
                    "g": rng.integers(0, 4, n).astype("int64"),
                    "v": np.arange(i * n, (i + 1) * n, dtype="int64"),
                }
            ),
            os.path.join(base, f"d_{i}.parquet"),
        )
    return base


@pytest.fixture
def disk_shuffle():
    from batcher.config import active_config, set_config

    cfg = active_config()
    set_config(
        dataclasses.replace(
            cfg,
            distributed=dataclasses.replace(
                cfg.distributed, transport="disk", shared_filesystem=True
            ),
        )
    )
    try:
        yield
    finally:
        set_config(cfg)


@pytest.fixture
def force_restage(monkeypatch):
    """Make every intermediate look large, so the second distributed stage actually runs."""
    from batcher.dist import executor

    monkeypatch.setattr(executor, "_STAGED_DISTINCT_ROWS", 0)


def test_count_distinct_matches_single_node_on_the_local_arm(source, disk_shuffle):
    """The ordinary shape: few enough distinct values that folding them here beats a whole
    second map/shuffle/reduce."""
    ds = bt.read.parquet(source).select("k").distinct()
    counted = ds.agg(n=bt.count())
    assert counted.collect(distributed=True, num_workers=4).to_pylist() == (
        counted.collect().to_pylist()
    )


def test_count_distinct_matches_single_node_on_the_restaged_arm(
    source, disk_shuffle, force_restage
):
    """The arm the rewrite exists for. Same answer, computed without the deduped rows ever
    passing through the driver."""
    ds = bt.read.parquet(source).select("k").distinct()
    counted = ds.agg(n=bt.count())
    assert counted.collect(distributed=True, num_workers=4).to_pylist() == (
        counted.collect().to_pylist()
    )


def test_a_grouped_aggregate_over_a_dedup_restages_to_the_same_answer(
    source, disk_shuffle, force_restage
):
    """Not just `COUNT(*)`: the staged aggregate carries group keys through the second
    stage, and a wrong source id or schema there would show up as missing groups."""
    ds = bt.read.parquet(source).select("k", "g").distinct()
    grouped = ds.group_by("g").agg(n=bt.count(), mx=bt.col("k").max())
    got = sorted(grouped.collect(distributed=True, num_workers=4).to_pylist(), key=lambda r: r["g"])
    want = sorted(grouped.collect().to_pylist(), key=lambda r: r["g"])
    assert got == want


def test_a_keyed_dedup_is_not_staged(source, disk_shuffle, force_restage):
    """A keyed dedup keeps the path it had. Staged, it returned the right answer and then
    poisoned the next one — its reducer sees one row per key and emits one row per key, the
    learning loop reads that as a `Distinct` which removes nothing, and Kyber drops the
    `Distinct` from the original query. The shape this staging exists for is
    `COUNT(DISTINCT x)`, which is a whole-row dedup over a projected column."""
    from batcher.dist import executor

    calls: list[int] = []
    original = executor._staged_aggregate_over_distinct
    executor._staged_aggregate_over_distinct = lambda *a, **k: (
        calls.append(1),
        original(*a, **k),
    )[1]
    try:
        ds = bt.read.parquet(source).distinct(subset=["k"]).agg(n=bt.count())
        ds.collect(distributed=True, num_workers=4)
    finally:
        executor._staged_aggregate_over_distinct = original
    assert calls == [], "a keyed dedup must not take the staged path"


def test_a_distributed_run_does_not_change_the_next_single_node_answer(
    source, disk_shuffle, force_restage
):
    """The regression this file exists for, and the reason it is worth an integration test:
    the wrong answer appears in a query that was **already correct**, on a later run, in the
    same process: a dedup stage whose measurements get attributed to the relation they were
    derived from teaches the learning loop that the key is unique, Kyber removes the
    `Distinct`, and `count()` goes from 900 to 40,000 with nothing raised. Core measures and
    Kyber decides; what must not happen is a *fragment's* measurements standing in for the
    whole.

    Whole-row only, deliberately. The same check on a **keyed** dedup fails today, and not
    because of this staging — `_distributed_distinct_on` does not exist at HEAD, so the
    behaviour belongs to the keyed row-shuffle path being built alongside this. Asserting it
    here would put someone else's in-flight bug in the way of their own commits.
    """
    counted = bt.read.parquet(source).select("k").distinct().agg(n=bt.count())

    before = counted.collect().to_pylist()
    distributed = counted.collect(distributed=True, num_workers=4).to_pylist()
    after = counted.collect().to_pylist()

    assert distributed == before
    assert after == before, "a distributed run changed what single-node computes next"
