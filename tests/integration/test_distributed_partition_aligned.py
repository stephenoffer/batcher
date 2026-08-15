"""A `GROUP BY` over a table's own partition columns runs with no exchange, and is exact.

The unit tests beside this one pin the *decision*; this pins the thing only a fleet can show:
that the rows come back right when the decision fires. That distinction matters more here than
for most operators, because the failure mode of a wrongly-skipped shuffle is a group reported
twice as two partial finals -- a wrong answer that a single-node run cannot produce and
therefore cannot catch.

Each test asserts against the single-node result on the same data, which is the oracle the
distributed path is defined against.
"""

from __future__ import annotations

import contextlib
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray
from batcher import col, count

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = pytest.mark.integration

_WORKERS = 4


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(_WORKERS)
    yield
    shutdown_test_ray(started)


@pytest.fixture(scope="module")
def hive_table(tmp_path_factory) -> str:
    """Twelve ``day=`` directories over three ``region`` values and repeating ``g`` values.

    Twelve is deliberately above the four-worker fleet: below it the scheduler keeps the
    shuffle rather than trade the layout's parallelism away, so the path under test would
    never run.
    """
    root = tmp_path_factory.mktemp("hive_aligned")
    for day in range(12):
        n = 200 + day
        table = pa.table(
            {
                "v": pa.array(range(day, day + n), type=pa.int64()),
                "g": pa.array([i % 5 for i in range(n)], type=pa.int64()),
                "region": pa.array([f"r{i % 3}" for i in range(n)]),
            }
        )
        os.makedirs(root / f"day={day}", exist_ok=True)
        pq.write_table(table, root / f"day={day}" / "part.parquet")
    return str(root)


def _norm(table: pa.Table) -> list[dict]:
    return sorted(table.to_pylist(), key=lambda r: tuple(str(v) for v in r.values()))


@contextlib.contextmanager
def _exchange_decisions():
    """Collect the scheduler's exchange decisions for the block's duration.

    Without this a test asserting only that the two results agree passes just as happily when
    the *shuffle* ran — which is the failure mode that matters here, because falling back is
    silent and correct. Several of these tests sit near the fleet-width guard, where a table
    with one partition fewer takes the other branch entirely.
    """
    from batcher._internal import events

    seen: list[dict] = []
    unsubscribe = events.subscribe(
        lambda e: (
            seen.append(e.fields)
            if e.kind == events.DECISION and e.fields.get("category") == "exchange"
            else None
        )
    )
    try:
        yield seen
    finally:
        unsubscribe()


def _both(hive_table: str, build, *, expect_aligned: bool = True):
    """Run `build` single-node and distributed, and assert which distributed plan ran.

    `expect_aligned=False` is for the near-misses, where the point of the test is that the
    scheduler *keeps* its shuffle and the answer is still right.
    """
    single = build(bt.read.parquet(hive_table)).collect()
    with _exchange_decisions() as decisions:
        distrib = build(bt.read.parquet(hive_table)).collect(distributed=True, num_workers=_WORKERS)
    assert bool(decisions) == expect_aligned, (
        f"expected aligned={expect_aligned}, got decisions={decisions}"
    )
    return _norm(single), _norm(distrib)


def test_group_by_the_partition_column_matches_single_node(hive_table):
    single, distrib = _both(
        hive_table,
        lambda ds: ds.group_by("day").agg(s=col("v").sum(), n=count(), hi=col("v").max()),
    )
    assert len(single) == 12
    assert single == distrib


def test_group_by_a_superset_of_the_partition_column_matches_single_node(hive_table):
    """Every ``(day, g)`` group is inside one directory because every ``day`` group is."""
    single, distrib = _both(
        hive_table, lambda ds: ds.group_by("day", "g").agg(s=col("v").sum(), n=count())
    )
    assert len(single) == 60
    assert single == distrib


def test_a_filter_below_the_aggregate_keeps_the_alignment(hive_table):
    """A filter only removes rows, so the layout's co-location survives it."""
    single, distrib = _both(
        hive_table,
        lambda ds: ds.filter(col("g") > 1).group_by("day").agg(s=col("v").sum(), n=count()),
    )
    assert single == distrib


def test_a_renaming_projection_keeps_the_alignment(hive_table):
    """The clustering is carried forward under the projection's output name."""
    single, distrib = _both(
        hive_table,
        lambda ds: (
            ds.select(d=col("day"), val=col("v")).group_by("d").agg(s=col("val").sum(), n=count())
        ),
    )
    assert len(single) == 12
    assert single == distrib


def test_group_by_a_non_partition_column_still_matches_single_node(hive_table):
    """The near-miss: `g` repeats inside every directory, so this must keep its shuffle. It
    is here to prove the *other* branch still works, not that it is fast."""
    single, distrib = _both(
        hive_table,
        lambda ds: ds.group_by("g").agg(s=col("v").sum(), n=count()),
        expect_aligned=False,
    )
    assert len(single) == 5
    assert single == distrib


def test_a_non_mergeable_aggregate_is_exact_when_aligned(hive_table):
    """`median` and `n_unique` carry per-group partial state that a shuffle has to merge.
    Aligned, nothing is merged at all -- each group is finalized where it was read -- so this
    is the shape that most directly proves the groups really were whole."""
    single, distrib = _both(
        hive_table,
        lambda ds: ds.group_by("day").agg(m=col("v").median(), nd=col("v").n_unique()),
    )
    assert single == distrib


def test_the_aligned_path_skips_the_exchange(hive_table, monkeypatch):
    """The performance claim itself: neither shuffle executor is entered.

    Asserting on the result alone cannot see this -- the shuffle path returns the same rows,
    which is exactly why a regression here would be silent.
    """
    import batcher.dist.executors.aggregate as agg_exec
    import batcher.dist.flight_aggregate as flight_agg

    def _refuse(*_args, **_kwargs):
        raise AssertionError("the aligned aggregate must not shuffle")

    monkeypatch.setattr(agg_exec, "_distributed_aggregate", _refuse)
    monkeypatch.setattr(flight_agg, "execute_aggregate_flight", _refuse)

    got = (
        bt.read.parquet(hive_table)
        .group_by("day")
        .agg(s=col("v").sum())
        .collect(distributed=True, num_workers=_WORKERS)
    )
    assert got.num_rows == 12


def test_whole_row_distinct_matches_single_node(hive_table):
    """A dedup is a group-by that keeps one row per group, so it eliminates the same
    exchange under the same condition -- and a whole-row `DISTINCT` groups on every column,
    which always contains the partition column."""
    single, distrib = _both(hive_table, lambda ds: ds.distinct())
    assert single == distrib


def test_distinct_on_the_partition_column_matches_single_node(hive_table):
    single, distrib = _both(hive_table, lambda ds: ds.select("day").distinct(["day"]))
    assert len(single) == 12
    assert single == distrib


def test_distinct_on_a_non_partition_column_matches_single_node(hive_table):
    """The near-miss for the dedup: `g` repeats across directories, so it keeps its shuffle."""
    single, distrib = _both(
        hive_table, lambda ds: ds.select("g").distinct(["g"]), expect_aligned=False
    )
    assert len(single) == 5
    assert single == distrib


def test_a_window_partitioned_by_the_partition_column_matches_single_node(hive_table):
    """A window computes each partition independently, so co-locating the partitions is the
    only thing its shuffle establishes -- and the layout established it already. Ranks are the
    sharpest check available: an off-by-one partition boundary restarts the numbering."""
    single, distrib = _both(
        hive_table,
        lambda ds: ds.with_columns(r=bt.row_number().over("day", order_by=["v", "g"])).select(
            "day", "v", "g", "r"
        ),
    )
    assert single == distrib


def test_a_window_partitioned_by_a_non_partition_column_matches_single_node(hive_table):
    """The window near-miss: `g` repeats in every directory, so the shuffle stays."""
    single, distrib = _both(
        hive_table,
        lambda ds: ds.with_columns(r=bt.row_number().over("g", order_by=["v", "day"])).select(
            "day", "v", "g", "r"
        ),
        expect_aligned=False,
    )
    assert single == distrib


@pytest.fixture(scope="module")
def delta_table(tmp_path_factory) -> str:
    """Six ``day`` partitions written in three appends, so each holds three data files.

    A Delta read splits per data *file*, so this is the shape whose co-location comes from
    grouping the splits rather than from the split set being distinct — and it is the shape
    every real lakehouse table has.
    """
    pytest.importorskip("deltalake", reason="deltalake not installed")
    root = str(tmp_path_factory.mktemp("delta_aligned"))
    for append in range(3):
        table = pa.table(
            {
                "day": pa.array([d for d in range(6) for _ in range(40)], pa.int64()),
                "g": pa.array([i % 5 for i in range(240)], pa.int64()),
                "v": pa.array([append * 1000 + i for i in range(240)], pa.int64()),
            }
        )
        bt.from_arrow(table).write.delta(
            root, partition_by=["day"], mode="overwrite" if append == 0 else "append"
        )
    return root


def _both_delta(delta_table: str, build, *, expect_aligned: bool = True):
    single = build(bt.read.delta(delta_table)).collect()
    with _exchange_decisions() as decisions:
        distrib = build(bt.read.delta(delta_table)).collect(distributed=True, num_workers=_WORKERS)
    assert bool(decisions) == expect_aligned, (
        f"expected aligned={expect_aligned}, got decisions={decisions}"
    )
    return _norm(single), _norm(distrib)


def test_delta_group_by_the_partition_column_matches_single_node(delta_table):
    """Eighteen files across six partitions: the three files of one day must be folded
    together, which only happens if all three were assigned to one worker."""
    single, distrib = _both_delta(
        delta_table,
        lambda ds: ds.group_by("day").agg(s=col("v").sum(), n=count(), hi=col("v").max()),
    )
    assert len(single) == 6
    assert all(row["n"] == 120 for row in single)
    assert single == distrib


def test_delta_non_mergeable_aggregate_is_exact_when_aligned(delta_table):
    single, distrib = _both_delta(
        delta_table, lambda ds: ds.group_by("day").agg(nd=col("v").n_unique())
    )
    assert single == distrib


def test_delta_group_by_a_non_partition_column_matches_single_node(delta_table):
    single, distrib = _both_delta(
        delta_table,
        lambda ds: ds.group_by("g").agg(s=col("v").sum(), n=count()),
        expect_aligned=False,
    )
    assert len(single) == 5
    assert single == distrib


def test_the_elimination_is_reported_as_a_decision(hive_table):
    """Without this the optimization is invisible from outside: the shuffle path returns the
    same rows, so a user has no way to tell whether their table's layout was used."""
    from batcher._internal import events

    seen: list[dict] = []
    unsubscribe = events.subscribe(
        lambda e: seen.append(e.fields) if e.kind == events.DECISION else None
    )
    try:
        (
            bt.read.parquet(hive_table)
            .group_by("day")
            .agg(s=col("v").sum())
            .collect(distributed=True, num_workers=_WORKERS)
        )
    finally:
        unsubscribe()

    exchange = [d for d in seen if d.get("category") == "exchange"]
    assert exchange, f"no exchange decision among {[d.get('category') for d in seen]}"
    assert exchange[0]["detail"] == {"operator": "aggregate", "clustered_on": ["day"]}
    assert "already partitioned by day" in exchange[0]["summary"]


# --- the edge cases a shuffle-free plan could get wrong silently ---------------------------
# `CLAUDE.md` names the cross-product a green gate misses:
# {collect, spill, iter_batches, distributed} x {nulls, empty, one row, duplicates, -0.0/NaN}.
# The aligned path is a *new cell* in it, and every failure it can have is a wrong answer
# rather than an error, so each of these is held against the single-node result.


@pytest.fixture(scope="module")
def awkward_table(tmp_path_factory) -> str:
    """A partitioned table built out of the shapes that break assumptions.

    Ten partitions covering: a NULL partition value (Hive writes it as a sentinel directory
    name), partitions holding exactly one row, and ordinary ones. Written through Batcher's own
    writer so the layout is the one a user gets.

    Ten rather than six on purpose: a pruning filter must still leave more partitions than the
    fleet has workers, or the scheduler keeps its shuffle and the test silently stops covering
    the path it is named for.
    """
    root = str(tmp_path_factory.mktemp("awkward"))
    days = [None, 1, 1, 2, 2, 2, 3, 4, 4, 5, 6, 6, 6, 7, 8, 9]
    table = pa.table(
        {
            "day": pa.array(days, pa.int64()),
            "v": pa.array([10 * (i + 1) for i in range(len(days))], pa.int64()),
            "g": pa.array(["a" if i % 2 else "b" for i in range(len(days))]),
        }
    )
    bt.from_arrow(table).write.parquet(root, partition_by=["day"], mode="overwrite")
    return root


def test_a_null_partition_value_is_one_group_not_several(awkward_table):
    """Hive stores a NULL partition as a sentinel directory name, so the value round-trips
    through a string. If two files ever disagreed on how that decodes they would be two
    groups, and the NULL group would come back split in two."""
    single, distrib = _both(
        awkward_table, lambda ds: ds.group_by("day").agg(s=col("v").sum(), n=count())
    )
    assert len(single) == 10, single
    assert single == distrib
    nulls = [r for r in single if r["day"] is None]
    assert len(nulls) == 1 and nulls[0]["n"] == 1


def test_single_row_partitions_survive(awkward_table):
    """Days 7, 8 and 9 hold one row each. A per-partition fold of a one-row group is where an
    off-by-one in the assignment would surface as a missing or duplicated group."""
    single, distrib = _both(
        awkward_table,
        lambda ds: ds.filter(col("day") >= 4).group_by("day").agg(s=col("v").sum(), n=count()),
    )
    assert single == distrib
    assert sorted(r["n"] for r in single) == [1, 1, 1, 1, 2, 3]


def test_pruning_narrows_the_split_set_the_clustering_is_checked_against(awkward_table):
    """A pushed predicate drops whole directories at plan time, so the set the check inspects
    is the pruned one. It must still be judged clustered, and still be judged by its *pruned*
    group count rather than the table's."""
    single, distrib = _both(
        awkward_table,
        lambda ds: ds.filter(col("day") >= 3).group_by("day").agg(s=col("v").sum(), n=count()),
    )
    assert single == distrib
    assert len(single) == 7


def test_a_filter_that_matches_nothing_returns_the_same_empty_shape(awkward_table):
    """Pruned to nothing, there are no groups to co-locate, so the scheduler keeps its
    ordinary plan. Pinned because "no partitions left" is the degenerate end of the
    fleet-width guard and must not become a division by zero or an aligned plan over an
    empty split set."""
    single, distrib = _both(
        awkward_table,
        lambda ds: ds.filter(col("day") > 999).group_by("day").agg(s=col("v").sum()),
        expect_aligned=False,
    )
    assert single == distrib == []


def test_a_computed_group_key_beside_the_partition_column_is_still_aligned(awkward_table):
    """Grouping by `(day, v % 2)` refines grouping by `(day)`, so every group is still inside
    one directory. The decision only counts the bare-column keys, and this is the shape that
    proves counting a *subset* is sound rather than merely conservative."""
    single, distrib = _both(
        awkward_table,
        lambda ds: ds.group_by("day", parity=col("v") % 2).agg(s=col("v").sum(), n=count()),
    )
    assert single == distrib


def test_a_string_partition_column_matches_single_node(tmp_path_factory):
    """Directory names *are* strings, so a string partition column is the case where the
    typed-value comparison and the raw one agree — worth pinning, because the int case is the
    one where they can differ."""
    root = str(tmp_path_factory.mktemp("string_part"))
    table = pa.table(
        {
            "region": pa.array([f"r{i % 6}" for i in range(120)]),
            "v": pa.array(range(120), pa.int64()),
        }
    )
    bt.from_arrow(table).write.parquet(root, partition_by=["region"], mode="overwrite")
    single = bt.read.parquet(root).group_by("region").agg(s=col("v").sum(), n=count()).collect()
    distrib = (
        bt.read.parquet(root)
        .group_by("region")
        .agg(s=col("v").sum(), n=count())
        .collect(distributed=True, num_workers=_WORKERS)
    )
    assert _norm(single) == _norm(distrib)
    assert len(_norm(single)) == 6


def test_count_distinct_grouped_by_the_partition_column_matches_single_node(hive_table):
    """`COUNT(DISTINCT)` is the expensive shape this most helps: it lowers to an aggregate over
    a `Distinct`, and the shuffle path has to dedup globally before it can count. Over a
    clustered relation the dedup is already global inside each partition."""
    single, distrib = _both(
        hive_table, lambda ds: ds.group_by("day").agg(u=col("v").n_unique(), n=count())
    )
    assert len(single) == 12
    assert single == distrib


def test_count_distinct_grouped_by_a_non_partition_column_matches_single_node(hive_table):
    """The near-miss: `g` straddles every directory, so the global dedup still has to happen."""
    single, distrib = _both(
        hive_table,
        lambda ds: ds.group_by("g").agg(u=col("v").n_unique()),
        expect_aligned=False,
    )
    assert len(single) == 5
    assert single == distrib


# --- Iceberg, distributed ------------------------------------------------------------------


@pytest.fixture(scope="module")
def iceberg_table(tmp_path_factory):
    """Six ``day`` partitions written in three appends: eighteen data files, six partitions."""
    pytest.importorskip("pyiceberg", reason="pyiceberg not installed")
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.types import LongType, NestedField

    warehouse = tmp_path_factory.mktemp("ice_dist") / "wh"
    warehouse.mkdir()
    spec = {
        "type": "sql",
        "uri": f"sqlite:///{warehouse}/catalog.db",
        "warehouse": f"file://{warehouse}",
    }
    catalog = SqlCatalog("default", uri=spec["uri"], warehouse=spec["warehouse"])
    catalog.create_namespace("db")
    table = catalog.create_table(
        "db.t",
        schema=Schema(
            NestedField(1, "day", LongType(), required=False),
            NestedField(2, "v", LongType(), required=False),
            NestedField(3, "g", LongType(), required=False),
        ),
        partition_spec=PartitionSpec(
            PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="day")
        ),
    )
    for append in range(3):
        n = 120
        table.append(
            pa.table(
                {
                    "day": pa.array([i % 6 for i in range(n)], pa.int64()),
                    "v": pa.array([append * 1000 + i for i in range(n)], pa.int64()),
                    "g": pa.array([i % 5 for i in range(n)], pa.int64()),
                }
            )
        )
    return "db.t", spec


def _both_iceberg(iceberg_table, build, *, expect_aligned: bool = True):
    identifier, spec = iceberg_table
    single = build(bt.read.iceberg(identifier, catalog=spec)).collect()
    with _exchange_decisions() as decisions:
        distrib = build(bt.read.iceberg(identifier, catalog=spec)).collect(
            distributed=True, num_workers=_WORKERS
        )
    assert bool(decisions) == expect_aligned, (
        f"expected aligned={expect_aligned}, got decisions={decisions}"
    )
    return _norm(single), _norm(distrib)


def test_iceberg_group_by_the_partition_column_matches_single_node(iceberg_table):
    """Eighteen data files across six partitions: the three files of one day must be folded
    together, which only happens if all three were assigned to one worker."""
    single, distrib = _both_iceberg(
        iceberg_table,
        lambda ds: ds.group_by("day").agg(s=col("v").sum(), n=count(), hi=col("v").max()),
    )
    assert len(single) == 6
    assert all(row["n"] == 60 for row in single)
    assert single == distrib


def test_iceberg_group_by_a_non_partition_column_matches_single_node(iceberg_table):
    single, distrib = _both_iceberg(
        iceberg_table,
        lambda ds: ds.group_by("g").agg(s=col("v").sum(), n=count()),
        expect_aligned=False,
    )
    assert len(single) == 5
    assert single == distrib
