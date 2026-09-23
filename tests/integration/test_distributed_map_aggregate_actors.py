"""`map_batches(...).agg(...)` on real splittable data, over persistent workers.

`_distributed_map_aggregate` is the route a batch-inference job ends on: run the model over
each partition, partial-aggregate in place, combine on the driver. It now dispatches to a
session-warm actor pool addressed by partition index rather than to a fresh stateless task,
so the partition a worker reads is the one that worker cached. Warm on TPC-H sf100 that is
1.4-1.8x (`benchmarks/BENCHMARK_RESULTS.md`).

**No existing test reaches this route.** Every `map_batches` + aggregate equivalence test in
the suite builds its input with `from_pydict`, and an in-memory source is not splittable, so
`_unsupported` runs the plan on one node and returns the right answer -- the distributed path
is never entered and a defect in it is invisible. That is the same trap
`test_diff_distributed_map_stage.py` documents for the staging route. So the source here is a
real multi-row-group Parquet file, and the first test asserts the route was taken before any
other test's agreement means anything.

Both routes are exercised on purpose: `partial -> combine -> finalize` must give the same
answer whichever kind of worker computed the partials, and the stateless-task path is still
what runs whenever the pool declines.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.integration

pytest.importorskip("ray", reason="the distributed path needs Ray")
pytest.importorskip("batcher._native", reason="native engine not built")

import sys  # noqa: E402

import ray  # noqa: E402

from batcher.dist.executors import map as M  # noqa: E402

ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])

_N = 4000
_W = 4


@pytest.fixture(scope="module")
def parquet_path(cluster_tmp_dir):
    """A multi-row-group Parquet file, so the source really splits."""
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "g": pa.array([f"g{i % 5}" for i in range(_N)]),
            "v": pa.array([float(i % 97) * 0.5 for i in range(_N)], pa.float64()),
        }
    )
    path = cluster_tmp_dir / "mapagg_actors.parquet"
    pq.write_table(table, path, row_group_size=100)
    return str(path)


def _doubled(batch):
    """The UDF prefix. It derives a new column, so a stage that dropped it is caught."""
    v = batch.column("v").to_pylist()
    return pa.table({"g": batch.column("g"), "d": pa.array([x * 2.0 for x in v])})


def _global(path: str):
    return (
        bt.read.parquet(path)
        .map_batches(_doubled, output_columns=["g", "d"])
        .agg(s=col("d").sum(), n=col("d").count())
    )


def _grouped(path: str):
    return (
        bt.read.parquet(path)
        .map_batches(_doubled, output_columns=["g", "d"])
        .group_by("g")
        .agg(s=col("d").sum())
    )


@pytest.fixture
def route(monkeypatch):
    """Record what `_distributed_map_aggregate` asked for, and what it got.

    `_agg_actor_pool` is called from that one place, so a non-empty answer establishes the
    route *and* the pool in a single observation.
    """
    seen: dict[str, object] = {}
    original = M._agg_actor_pool

    def spy(plan0, workers):
        pool = original(plan0, workers)
        seen["actors"] = len(pool) if pool else 0
        seen["workers"] = workers
        return pool

    monkeypatch.setattr(M, "_agg_actor_pool", spy)
    return seen


def test_the_stage_really_takes_the_actor_route(parquet_path, route):
    """The positive control. Every agreement below is vacuous over a single-node fallback."""
    _global(parquet_path).collect(distributed=True, num_workers=_W)

    assert route.get("workers"), "`_distributed_map_aggregate` was never reached"
    assert route["actors"] > 0, "the stage ran stateless tasks, not the pool under test"


def test_the_actor_route_matches_single_node(parquet_path, route):
    """The mergeable contract: only partials leave the worker, whoever the worker is."""
    single = _global(parquet_path).collect().to_pydict()
    dist = _global(parquet_path).collect(distributed=True, num_workers=_W).to_pydict()

    assert route["actors"] > 0
    assert dist["n"] == single["n"] == [_N]
    assert dist["s"] == pytest.approx(single["s"])


def test_a_grouped_aggregate_matches_single_node(parquet_path, route):
    """Group keys travel through `partial_aggregate` differently from a global reduction."""
    one = _grouped(parquet_path).collect().sort_by("g").to_pydict()
    many = _grouped(parquet_path).collect(distributed=True, num_workers=_W).sort_by("g").to_pydict()

    assert route["actors"] > 0
    assert many["g"] == one["g"] == [f"g{i}" for i in range(5)]
    assert many["s"] == pytest.approx(one["s"])


def test_both_routes_produce_the_same_answer(parquet_path, route, monkeypatch):
    """The stateless-task path still runs whenever the pool declines; it must still agree."""
    with_actors = _global(parquet_path).collect(distributed=True, num_workers=_W).to_pydict()
    assert route["actors"] > 0, "the actor arm did not take the actor route"

    # Hand back the warm pool first. Declining the pool by monkeypatch leaves *this plan's*
    # actors alive and holding the cluster's cores -- a state production never reaches, since
    # a pipeline only ever declines a pool it does not hold -- and `release_foreign_agg_pools`
    # deliberately spares a pipeline's own pool, so the stateless tasks could never place and
    # the arm hung to the timeout. The idle timer is what returns them in production.
    M._shutdown_pools(M._AGG_POOLS)
    monkeypatch.setattr(M, "_agg_actor_pool", lambda plan0, workers: None)
    with_tasks = _global(parquet_path).collect(distributed=True, num_workers=_W).to_pydict()

    assert with_tasks["n"] == with_actors["n"] == [_N]
    assert with_tasks["s"] == pytest.approx(with_actors["s"])
