"""An over-partitioned map stage answers exactly what a per-worker one does.

The shuffle's map stage cuts its input into `workers x map_partition_multiplier` pieces
rather than one per worker, so the unit of scheduling and of recovery is a fraction of a
node's share instead of all of it. That is a *scheduling* change and nothing else: the
partitions are disjoint, each is a deterministic function of its own descriptor, and the
mergeable algebra makes any partitioning of the map side produce the same aggregate. If the
result ever moves, the partitioning is not disjoint — which is a dropped or duplicated row,
not an error — so these compare against single-node rather than against themselves.

The equivalence is checked across the shapes where a partitioning bug would actually surface:
the flat reduce and the combiner tree (the tree's leaf count is the *source* count, which
stops equalling the worker count here), with and without a worker loss, and against the
per-worker unit the multiplier of 1 pins.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray
from batcher import col, count
from batcher.config import (
    Config,
    DistributedConfig,
    FlowControlConfig,
    config_context,
)

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

WORKERS = 4
#: CPUs the local test cluster starts with. Above `WORKERS` on purpose: the fleet
#: reserves one core per worker actor, so a cluster sized exactly to the fan-out leaves
#: nothing schedulable for the driver-side tasks and the acquire times out.
CPUS = 6


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(CPUS)
    yield
    shutdown_test_ray(started)


@pytest.fixture(scope="module")
def split_source(tmp_path_factory):
    """A Parquet file of 32 row-groups — enough splits to fill a 4x over-partitioned map.

    Over-partitioning is bounded by the splits a source actually has, so an in-memory
    source (one `WholeSourceSplit`) would silently stay at one partition per worker and the
    test would pass while measuring the old path.
    """
    import pyarrow.parquet as pq

    rng = np.random.default_rng(101)
    n = 160_000
    table = pa.table(
        {
            "k": rng.integers(0, 50, n).astype("int64"),
            "v": rng.integers(0, 100, n).astype("int64"),
        }
    )
    path = str(tmp_path_factory.mktemp("granularity") / "t.parquet")
    pq.write_table(table, path, row_group_size=5_000)
    return path


def _norm(table: pa.Table) -> set:
    return {
        tuple(round(v, 6) if isinstance(v, float) else v for v in row.values())
        for row in table.to_pylist()
    }


def _granularity(multiplier: int, fan_in: int = 64, replication: int = 1):
    return config_context(
        Config().replace(
            distributed=DistributedConfig(
                map_partition_multiplier=multiplier, shuffle_replication=replication
            ),
            flow_control=FlowControlConfig(shuffle_fan_in=fan_in),
        )
    )


def _agg(path: str):
    return (
        bt.read.parquet(path)
        .group_by("k")
        .agg(s=col("v").sum(), n=count(), m=col("v").max(), a=col("v").mean())
    )


def _run(path, **kw):
    from batcher.dist.flight_aggregate import execute_aggregate_flight

    ds = _agg(path)
    return execute_aggregate_flight([], ds._plan, ds._sources, workers=WORKERS, **kw)


def test_over_partitioned_map_matches_single_node(split_source):
    expected = _agg(split_source).collect()
    with _granularity(4):
        got = _run(split_source)
    assert _norm(got) == _norm(expected)


def test_over_partitioned_map_actually_makes_more_partitions(split_source):
    """Without this the equivalence tests above would pass on the unchanged path.

    The count is a ceiling bounded by the source's splits, so this asserts "more than one
    per worker", not an exact number — the exact number is the split planner's business and
    pinning it here would fail on an unrelated coalescing change.
    """
    from batcher.dist.executors.partition_io import partition_descriptors
    from batcher.dist.executors.ray_runtime import map_partitions

    source = _agg(split_source)._sources[0]
    with _granularity(4):
        many = partition_descriptors(source, WORKERS, max_partitions=map_partitions(WORKERS))
    with _granularity(1):
        one_each = partition_descriptors(source, WORKERS, max_partitions=map_partitions(WORKERS))

    assert len(one_each) == WORKERS
    assert len(many) > WORKERS


def test_over_partitioned_tree_reduce_matches_single_node(split_source):
    # `workers > fan_in` builds the combiner tree — the trigger stays the peer count, since
    # several sources on one worker are one Flight server. What changes is the tree's LEAF
    # count, which is now the source count: its frontier, its per-level fallbacks and its
    # recovery all have to be sized by `len(leaf_addrs)` rather than by `workers`.
    expected = _agg(split_source).collect()
    with _granularity(4, fan_in=2):
        got = _run(split_source)
    assert _norm(got) == _norm(expected)


@pytest.mark.parametrize("killed", [{1}, {0, 2}])
def test_over_partitioned_map_survives_worker_loss(split_source, killed):
    # The payoff and the risk in one test. A dead worker held SEVERAL sources, so recovery
    # has to regenerate all of them; recomputing "the source with the dead worker's id"
    # leaves the rest unreachable, and an unreachable ticket reads back as an empty bucket
    # rather than an error — a silently short answer, which is what the comparison catches.
    expected = _agg(split_source).collect()
    with _granularity(4):
        got = _run(split_source, _fault_inject=killed)
    assert _norm(got) == _norm(expected)


def test_over_partitioned_tree_survives_worker_loss(split_source):
    expected = _agg(split_source).collect()
    with _granularity(4, fan_in=2):
        got = _run(split_source, _fault_inject={1})
    assert _norm(got) == _norm(expected)


def test_worker_loss_during_the_map_barrier_redeals_its_sources(split_source):
    # `_fault_inject_map` kills before the barrier, so the dead worker's share is re-dealt
    # to survivors as the barrier runs rather than recovered afterwards.
    expected = _agg(split_source).collect()
    with _granularity(4):
        got = _run(split_source, _fault_inject_map={2})
    assert _norm(got) == _norm(expected)


def test_the_per_worker_unit_is_still_available(split_source):
    # A multiplier of 1 pins the old task unit exactly, which is the escape hatch for a
    # deployment where the extra streams cost more than the finer recovery buys.
    expected = _agg(split_source).collect()
    with _granularity(1):
        got = _run(split_source)
    assert _norm(got) == _norm(expected)


# ---------------------------------------------------------------------------
# The other three shuffles. Each maps its own input, so each has its own task unit —
# and the sort and the join have a wrinkle the aggregate does not. The sort samples
# quantiles through a barrier before it partitions, so its sample now merges one grid
# per source. The join maps BOTH sides through one barrier under a single source id, so
# the two sides' partition lists have to be the same length even though each side's
# count is bounded by its own splits.
# ---------------------------------------------------------------------------


def test_over_partitioned_sort_matches_single_node(split_source):
    ds = bt.read.parquet(split_source).sort("v")
    expected = ds.collect()
    with _granularity(4):
        got = ds.collect(distributed=True, num_workers=WORKERS, transport="flight")
    # A sort is ORDER-sensitive, so compare the sequence — `_norm` is a set and would pass
    # on a correctly-partitioned but wrongly-ordered result.
    assert got.column("v").to_pylist() == expected.column("v").to_pylist()


def test_over_partitioned_window_matches_single_node(split_source):
    ds = (
        bt.read.parquet(split_source)
        .with_columns(r=col("v").sum().over(partition_by="k"))
        .group_by("k")
        .agg(r=col("r").max())
    )
    expected = ds.collect()
    with _granularity(4):
        got = ds.collect(distributed=True, num_workers=WORKERS, transport="flight")
    assert _norm(got) == _norm(expected)


def test_over_partitioned_join_matches_single_node(split_source, tmp_path_factory):
    # Deliberately lopsided: a small dimension against the big fact, which is the shape
    # where the two sides' achievable partition counts differ and the shorter list is
    # padded. Getting the padding wrong drops the unpadded side's tail rows.
    import pyarrow.parquet as pq

    dim = pa.table({"k": list(range(50)), "label": [f"g{i}" for i in range(50)]})
    dim_path = str(tmp_path_factory.mktemp("granularity_dim") / "d.parquet")
    pq.write_table(dim, dim_path)

    def q():
        return (
            bt.read.parquet(split_source)
            .join(bt.read.parquet(dim_path), on="k")
            .group_by("label")
            .agg(n=count())
        )

    expected = q().collect()
    with _granularity(4):
        got = q().collect(distributed=True, num_workers=WORKERS, transport="flight")
    assert _norm(got) == _norm(expected)
