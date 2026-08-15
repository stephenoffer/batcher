"""Exchange elimination against the layout a table already has on disk.

A Hive-partitioned table read one directory per split has already done what a shuffle would
do: every row with a given partition value is inside one split, and a split is assigned to a
worker whole. So a ``GROUP BY`` covering the partition columns needs no exchange at all --
each worker folds its own directories to final groups and the driver concatenates.

The reason this file is careful rather than brief is the failure mode. Over-claiming the
guarantee does not make a query slow, it makes it **wrong**: a group split across two workers
comes back as two partial groups, each labelled final. That is invisible on one node, invisible
in any test that does not run a fleet, and silent at PB scale. So the tests below pin both
directions -- the layouts that do guarantee co-location, and every near-miss that does not.
"""

from __future__ import annotations

import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher import col
from batcher.dist.executor import _partition_aligned_aggregate
from batcher.dist.executors.map import scan_clustering_for
from batcher.io.splits import declared_clustering, group_by_clustering
from batcher.kyber.properties import PhysicalProperties, clustered_on, satisfies
from batcher.plan.expr_ir import Binary, Col, Lit
from batcher.plan.logical import Distinct, Filter, Limit, Project, Projection, Scan, Sort
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit


class _Split:
    """A split declaring the clustering protocol, with no read behind it."""

    def __init__(self, columns, value):
        self.clustering_columns = columns
        self.clustering_value = value


# --- the split-set guarantee ------------------------------------------------------------


def test_splits_agreeing_on_a_column_declare_a_clustering():
    splits = [_Split(("day",), (d,)) for d in range(4)]
    assert declared_clustering(splits) == ("day",)


def test_splits_disagreeing_on_the_clustering_column_guarantee_nothing():
    assert declared_clustering([_Split(("day",), (1,)), _Split(("hour",), (2,))]) == ()


def test_a_split_that_declares_nothing_is_unclustered():
    assert declared_clustering([_Split(("day",), (1,)), object()]) == ()
    assert declared_clustering([object()]) == ()


def test_an_empty_split_set_is_unclustered():
    assert declared_clustering([]) == ()


def test_a_malformed_declaration_is_ignored_rather_than_trusted():
    """Columns and values of different lengths cannot be compared pairwise, so the split
    declares nothing rather than something half-checked."""
    assert declared_clustering([_Split(("day", "hour"), (1,))]) == ()


def test_many_splits_per_value_group_into_one_assignable_unit():
    """The Delta/Iceberg shape: one split per data *file*, several files per partition. The
    split set alone proves nothing -- three splits say ``day=3`` -- and grouping is what makes
    the value land on one worker anyway."""
    splits = [_Split(("day",), (3,)), _Split(("day",), (4,)), _Split(("day",), (3,))]
    groups = group_by_clustering(splits)
    assert groups is not None
    assert [len(g) for g in groups] == [2, 1]


def test_grouping_refuses_a_set_that_declares_nothing():
    assert group_by_clustering([_Split(("day",), (1,)), object()]) is None


# --- propagation through the plan -------------------------------------------------------


def _scan(*names: str) -> Scan:
    schema = SchemaRef.from_arrow(pa.schema([pa.field(n, pa.int64()) for n in names]))
    return Scan(source_id=0, schema=schema)


def test_clustering_survives_row_removing_operators():
    """Filter, Limit and Distinct only *remove* rows; removing one never moves another to a
    different worker, so the clustering the read delivered still holds above them."""
    scan = _scan("day", "v")
    mapping = {0: ("day",)}
    kept = Filter(input=scan, predicate=Binary(">", Col("v"), Lit(1)))
    assert clustered_on(kept, mapping) == ("day",)
    assert clustered_on(Limit(input=scan, n=5), mapping) == ("day",)
    assert clustered_on(Distinct(input=scan), mapping) == ("day",)


def test_clustering_survives_a_projection_under_the_output_name():
    scan = _scan("day", "v")
    proj = Project(
        input=scan,
        items=(Projection("d", Col("day")), Projection("v", Col("v"))),
    )
    assert clustered_on(proj, {0: ("day",)}) == ("d",)


def test_a_projection_that_drops_the_clustering_column_unclaims_it():
    """The rows are still physically co-located, but the property can no longer be *named*,
    so it is not claimed -- costing at most a needless shuffle, never a wrong answer."""
    scan = _scan("day", "v")
    proj = Project(input=scan, items=(Projection("v", Col("v")),))
    assert clustered_on(proj, {0: ("day",)}) == ()


def test_a_sort_unclaims_the_clustering():
    """A sort is not row-preserving *per worker* in the distributed path: it range-partitions
    on its key, which moves rows between workers on a key unrelated to `day`."""
    scan = _scan("day", "v")
    assert clustered_on(Sort(input=scan, keys=(), limit=None), {0: ("day",)}) == ()


def test_a_scan_of_an_unlisted_source_is_unclustered():
    assert clustered_on(_scan("day"), {}) == ()


# --- the containment direction ----------------------------------------------------------


def test_clustering_satisfies_a_grouping_that_covers_it():
    have = PhysicalProperties(clustered_on=("day",))
    assert satisfies(have, PhysicalProperties(hash_partitioned_on=("day", "region")))


def test_clustering_does_not_satisfy_a_grouping_it_does_not_cover():
    """Rows sharing `region` but differing in `day` sit in different directories, so a
    `GROUP BY region` straddles workers however the table is laid out."""
    have = PhysicalProperties(clustered_on=("day",))
    assert not satisfies(have, PhysicalProperties(hash_partitioned_on=("region",)))


def test_the_two_partitionings_are_tested_separately_not_unioned():
    """Hash-partitioned on `a` and clustered on `b` satisfies a grouping by `(a, c)` through
    the hash alone. Unioning the delivered keys into `{a, b}` would wrongly demand `b`."""
    have = PhysicalProperties(hash_partitioned_on=("a",), clustered_on=("b",))
    assert satisfies(have, PhysicalProperties(hash_partitioned_on=("a", "c")))


# --- the scheduling decision, over a real Hive table -------------------------------------


@pytest.fixture(scope="module")
def hive_table(tmp_path_factory) -> str:
    """Eight one-file ``day=`` directories, each holding the same three ``g`` values."""
    root = tmp_path_factory.mktemp("hive")
    for day in range(8):
        table = pa.table({"v": list(range(day, day + 30)), "g": [i % 3 for i in range(30)]})
        os.makedirs(root / f"day={day}", exist_ok=True)
        pq.write_table(table, root / f"day={day}" / "part.parquet")
    return str(root)


def _agg(path: str, *keys: str):
    ds = bt.read.parquet(path).group_by(*keys).agg(s=col("v").sum())
    return ds._plan, list(ds._sources)


def test_a_hive_table_reports_its_partition_column_as_its_clustering(hive_table):
    ds = bt.read.parquet(hive_table)
    assert scan_clustering_for(ds._plan, list(ds._sources), 4) == ("day",)


def test_grouping_on_the_partition_column_needs_no_exchange(hive_table):
    plan, sources = _agg(hive_table, "day")
    assert _partition_aligned_aggregate(plan, sources, 4, None)


def test_grouping_on_a_superset_of_the_partition_column_needs_no_exchange(hive_table):
    plan, sources = _agg(hive_table, "day", "g")
    assert _partition_aligned_aggregate(plan, sources, 4, None)


def test_grouping_on_a_non_partition_column_still_shuffles(hive_table):
    """`g` repeats inside every directory, so its groups straddle every worker."""
    plan, sources = _agg(hive_table, "g")
    assert not _partition_aligned_aggregate(plan, sources, 4, None)


def test_a_global_aggregate_still_combines(hive_table):
    """One group spanning every worker is the one shape that always needs a combine."""
    ds = bt.read.parquet(hive_table).agg(s=col("v").sum())
    assert not _partition_aligned_aggregate(ds._plan, list(ds._sources), 4, None)


def test_a_fleet_wider_than_a_directory_tree_still_aligns(hive_table):
    """A Hive tree splits one-per-partition, so grouping costs it *nothing*: the shuffle plan
    would have had the same eight tasks. Measured at 0.97x-3.9x across the range, never a
    meaningful loss, so the fleet being wider than the layout is not on its own a reason to
    shuffle. The previous rule refused here and gave up a measured 2.0x at half the fleet."""
    plan, sources = _agg(hive_table, "day")
    assert _partition_aligned_aggregate(plan, sources, 16, None) == ("day",)


def test_an_unpartitioned_table_is_unclustered(tmp_path):
    path = str(tmp_path / "flat.parquet")
    pq.write_table(pa.table({"day": [1, 1, 2], "v": [1, 2, 3]}), path)
    ds = bt.read.parquet(path).group_by("day").agg(s=col("v").sum())
    assert not _partition_aligned_aggregate(ds._plan, list(ds._sources), 1, None)


# --- the same elimination for a dedup ----------------------------------------------------


def test_whole_row_distinct_over_a_clustered_table_needs_no_exchange(hive_table):
    """A whole-row `DISTINCT` groups on every column, which contains the partition column,
    so any clustered layout aligns it."""
    from batcher.dist.executor import _partition_aligned_distinct

    ds = bt.read.parquet(hive_table).distinct()
    assert _partition_aligned_distinct(ds._plan, list(ds._sources), 4, None)


def test_distinct_on_the_partition_column_needs_no_exchange(hive_table):
    from batcher.dist.executor import _partition_aligned_distinct

    ds = bt.read.parquet(hive_table).distinct(["day"])
    assert _partition_aligned_distinct(ds._plan, list(ds._sources), 4, None)


def test_distinct_on_a_non_partition_column_still_shuffles(hive_table):
    from batcher.dist.executor import _partition_aligned_distinct

    ds = bt.read.parquet(hive_table).distinct(["g"])
    assert not _partition_aligned_distinct(ds._plan, list(ds._sources), 4, None)


def test_a_limited_distinct_is_never_aligned(hive_table):
    """`n` rows per partition concatenated is `n x partitions` rows, not `n`."""
    import dataclasses

    from batcher.dist.executor import _partition_aligned_distinct

    ds = bt.read.parquet(hive_table).distinct()
    limited = dataclasses.replace(ds._plan, limit=5)
    assert not _partition_aligned_distinct(limited, list(ds._sources), 4, None)


# --- the same elimination for a window ---------------------------------------------------


def _window_of(ds):
    """The `Window` node in `ds`'s plan, wherever `with_columns` put it."""
    from batcher.plan.logical import Window
    from batcher.plan.visitor import walk

    windows = [n for n in walk(ds._plan) if isinstance(n, Window)]
    assert len(windows) == 1, f"expected one Window, found {len(windows)}"
    return windows[0]


def test_a_window_partitioned_by_the_partition_column_needs_no_exchange(hive_table):
    """`ROW_NUMBER() OVER (PARTITION BY day ...)` over a directory-per-day table: each window
    partition is a directory, so each is already whole on one worker."""
    from batcher.dist.executor import _partition_aligned_window

    ds = bt.read.parquet(hive_table).with_columns(r=bt.row_number().over("day", order_by="v"))
    window = _window_of(ds)
    assert _partition_aligned_window(window, list(ds._sources), 4, None)


def test_a_window_partitioned_by_a_non_partition_column_still_shuffles(hive_table):
    from batcher.dist.executor import _partition_aligned_window

    ds = bt.read.parquet(hive_table).with_columns(r=bt.row_number().over("g", order_by="v"))
    window = _window_of(ds)
    assert not _partition_aligned_window(window, list(ds._sources), 4, None)


def test_an_unpartitioned_window_is_never_aligned(hive_table):
    """One partition over every row has nothing to co-locate it by."""
    from batcher.dist.executor import _partition_aligned_window

    ds = bt.read.parquet(hive_table).with_columns(r=bt.row_number().over(order_by="v"))
    window = _window_of(ds)
    assert not _partition_aligned_window(window, list(ds._sources), 4, None)


# --- a file-per-split reader: the shape grouping exists for --------------------------------


@pytest.fixture(scope="module")
def delta_table(tmp_path_factory) -> str:
    """Six ``day`` partitions written in three appends, so each holds three data files.

    This is the shape a Hive directory read never produces and every lakehouse read does: the
    split set alone proves nothing, because three splits say ``day=3``. Co-location comes from
    grouping them, not from the splits being distinct.
    """
    pytest.importorskip("deltalake", reason="deltalake not installed")
    root = str(tmp_path_factory.mktemp("delta_aligned"))
    for append in range(3):
        table = pa.table(
            {
                "day": pa.array([d for d in range(6) for _ in range(20)], pa.int64()),
                "v": pa.array([append * 1000 + i for i in range(120)], pa.int64()),
            }
        )
        bt.from_arrow(table).write.delta(
            root, partition_by=["day"], mode="overwrite" if append == 0 else "append"
        )
    return root


def test_a_delta_split_declares_the_partition_it_belongs_to(delta_table):
    src = bt.read.delta(delta_table)._sources[0]
    splits = src.splits()
    assert len(splits) == 18
    assert declared_clustering(splits) == ("day",)


def test_delta_splits_group_to_one_unit_per_partition(delta_table):
    """Eighteen files, six partitions, three files each — and the group is what gets assigned,
    so a partition cannot straddle two workers however many files it holds."""
    groups = group_by_clustering(bt.read.delta(delta_table)._sources[0].splits())
    assert groups is not None
    assert sorted(len(g) for g in groups) == [3, 3, 3, 3, 3, 3]


def test_grouping_not_split_count_decides_the_parallelism(delta_table):
    """Eighteen splits collapse to six assignable units, and it is the six that bound the
    aligned plan's fan-out. What decides is how much of the shuffle's parallelism that gives
    up: at six workers both plans run six tasks, and at eight the aligned plan keeps six of
    eight -- comfortably above the measured quarter where it stops paying."""
    ds = bt.read.delta(delta_table).group_by("day").agg(s=col("v").sum())
    assert _partition_aligned_aggregate(ds._plan, list(ds._sources), 6, None) == ("day",)
    assert _partition_aligned_aggregate(ds._plan, list(ds._sources), 8, None) == ("day",)


@pytest.fixture(scope="module")
def one_partition_delta(tmp_path_factory) -> str:
    """A single ``day`` partition written in eight appends: one group, eight data files.

    The measured loss case. The aligned plan has one assignable unit and runs on one worker;
    the shuffle reads the eight files across eight. Measured at **0.63x** — an actual
    regression, which is what the retention rule exists to refuse.
    """
    pytest.importorskip("deltalake", reason="deltalake not installed")
    root = str(tmp_path_factory.mktemp("delta_one_part"))
    for append in range(8):
        table = pa.table(
            {
                "day": pa.array([1] * 50, pa.int64()),
                "v": pa.array([append * 100 + i for i in range(50)], pa.int64()),
            }
        )
        bt.from_arrow(table).write.delta(
            root, partition_by=["day"], mode="overwrite" if append == 0 else "append"
        )
    return root


def test_a_layout_that_would_run_the_query_on_one_worker_shuffles(one_partition_delta):
    """Eight splits collapsing to a single assignable unit makes the whole query serial while
    the shuffle reads them across the fleet. Measured at **0.62x** — an actual regression, and
    the reason a ratio alone is not the rule: this layout keeps a *quarter* of the shuffle's
    parallelism, the same quarter that wins at two partitions."""
    ds = bt.read.delta(one_partition_delta).group_by("day").agg(s=col("v").sum())
    assert len(ds._sources[0].splits()) == 8
    assert _partition_aligned_aggregate(ds._plan, list(ds._sources), 8, None) == ()


def test_it_is_still_serial_on_a_small_fleet_and_still_refused(one_partition_delta):
    """The fleet does not rescue it. Two workers means the shuffle runs two tasks and the
    aligned plan still runs one, which is the same serial plan against a parallel one."""
    ds = bt.read.delta(one_partition_delta).group_by("day").agg(s=col("v").sum())
    assert _partition_aligned_aggregate(ds._plan, list(ds._sources), 2, None) == ()


def test_a_single_split_source_aligns_because_the_shuffle_is_serial_too(hive_table, tmp_path):
    """The one place a single-task aligned plan is right: when the read has one split, the
    shuffle is equally serial and its exchange buys nothing. The minimum-task rule is written
    against what the shuffle would *actually* have run, not a flat floor of two."""
    root = str(tmp_path / "one_dir")
    table = pa.table({"day": pa.array([7] * 40, pa.int64()), "v": pa.array(range(40), pa.int64())})
    bt.from_arrow(table).write.parquet(root, partition_by=["day"], mode="overwrite")
    ds = bt.read.parquet(root).group_by("day").agg(s=col("v").sum())
    assert len(ds._sources[0].splits()) == 1
    assert _partition_aligned_aggregate(ds._plan, list(ds._sources), 8, None) == ("day",)


def test_a_delta_group_by_a_non_partition_column_still_shuffles(delta_table):
    ds = bt.read.delta(delta_table).group_by("v").agg(n=bt.count())
    assert _partition_aligned_aggregate(ds._plan, list(ds._sources), 4, None) == ()


def test_an_unpartitioned_delta_table_is_unclustered(tmp_path):
    pytest.importorskip("deltalake", reason="deltalake not installed")
    root = str(tmp_path / "flat_delta")
    bt.from_arrow(pa.table({"day": [1, 1, 2], "v": [1, 2, 3]})).write.delta(root, mode="overwrite")
    assert declared_clustering(bt.read.delta(root)._sources[0].splits()) == ()


def test_the_assignment_never_makes_more_buckets_than_there_are_partitions():
    """An indivisible group cannot fill two buckets, so a surplus bucket would be *empty* --
    and an empty partition still costs a task, a CPU reservation and a schema-only round trip.
    Sixty-four splits over sixteen partitions asked for sixty-four buckets would run 48 no-op
    tasks."""
    from batcher.dist.executors.partition_io.assignment import assign_clustered_splits

    splits = [_Split(("day",), (d,)) for d in range(16) for _ in range(4)]
    buckets = assign_clustered_splits(splits, 64)
    assert len(buckets) == 16
    assert all(buckets)
    assert sum(len(b) for b in buckets) == 64


def test_a_fleet_smaller_than_the_layout_still_packs_every_group():
    """The other direction: four buckets for sixteen partitions is four workers each holding
    several whole groups, which is the assignment doing its ordinary job."""
    from batcher.dist.executors.partition_io.assignment import assign_clustered_splits

    splits = [_Split(("day",), (d,)) for d in range(16)]
    buckets = assign_clustered_splits(splits, 4)
    assert len(buckets) == 4
    assert sum(len(b) for b in buckets) == 16


def test_the_executor_refuses_splits_that_no_longer_declare_what_the_plan_needs(hive_table):
    """The second check, at the point the splits are real. A silent fallback here would be a
    wrong answer, because the plan chosen against the first check has no combine in it."""
    from batcher._internal.errors import ExecutionError
    from batcher.dist.executors.partition_io import partition_descriptors

    source = bt.read.parquet(hive_table)._sources[0]
    with pytest.raises(ExecutionError, match="clustering"):
        partition_descriptors(source, 4, cluster_by=("not_a_column",))


# --- what may sit between the scan and the operator ----------------------------------------


def test_a_dedup_below_the_aggregate_is_partition_local(hive_table):
    """`COUNT(DISTINCT x) GROUP BY day` lowers to an aggregate over a `Distinct`. A dedup only
    collapses rows that agree, and rows that agree on `day` are already on one worker, so the
    whole thing folds with no shuffle."""
    ds = bt.read.parquet(hive_table).group_by("day").agg(n=col("v").n_unique())
    assert _partition_aligned_aggregate(ds._plan, list(ds._sources), 4, None) == ("day",)


def test_a_limit_below_the_aggregate_is_not_partition_local(hive_table):
    """The counterexample the chain check exists for, and the reason the *clustering* property
    is not enough on its own. A `Limit` does not move rows between workers, so the relation is
    still clustered — but `limit(100).group_by(day)` run per partition keeps a hundred rows on
    every one of them, which is a different query."""
    from batcher.dist.executor import _partition_local_chain

    ds = bt.read.parquet(hive_table).limit(100).group_by("day").agg(s=col("v").sum())
    agg = ds._plan
    assert clustered_on(agg.input, {0: ("day",)}) == ("day",)  # still clustered...
    assert not _partition_local_chain(agg.input)  # ...but not safe per partition
    assert _partition_aligned_aggregate(agg, list(ds._sources), 4, None) == ()


def test_a_limited_dedup_below_the_aggregate_is_not_partition_local(hive_table):
    """A `Distinct` carrying a limit is a limit, and is refused as one."""
    import dataclasses

    from batcher.dist.executor import _partition_local_chain

    ds = bt.read.parquet(hive_table).distinct()
    limited = dataclasses.replace(ds._plan, limit=5)
    assert not _partition_local_chain(limited)


# --- Iceberg: the same guarantee, recorded a third way -------------------------------------


def _iceberg_catalog(tmp_path, name: str = "default"):
    """A local SQL catalog. The catalog *name* is part of a SqlCatalog table identity, so it
    must match the one `bt.read.iceberg` builds from the same spec or the table is invisible."""
    pytest.importorskip("pyiceberg", reason="pyiceberg not installed")
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = tmp_path / "wh"
    warehouse.mkdir(exist_ok=True)
    spec = {
        "type": "sql",
        "uri": f"sqlite:///{warehouse}/catalog.db",
        "warehouse": f"file://{warehouse}",
    }
    return SqlCatalog(name, uri=spec["uri"], warehouse=spec["warehouse"]), spec


def _iceberg_schema():
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField

    return Schema(
        NestedField(1, "day", LongType(), required=False),
        NestedField(2, "v", LongType(), required=False),
    )


@pytest.fixture(scope="module")
def iceberg_table(tmp_path_factory):
    """Three ``day`` partitions written in two appends: six data files, three partitions."""
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import IdentityTransform

    catalog, spec = _iceberg_catalog(tmp_path_factory.mktemp("ice"))
    catalog.create_namespace("db")
    table = catalog.create_table(
        "db.t",
        schema=_iceberg_schema(),
        partition_spec=PartitionSpec(
            PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="day")
        ),
    )
    rows = pa.table(
        {
            "day": pa.array([1, 1, 2, 2, 3, 3], pa.int64()),
            "v": pa.array(range(6), pa.int64()),
        }
    )
    for _ in range(2):
        table.append(rows)
    return "db.t", spec


def test_an_iceberg_split_declares_the_partition_source_column(iceberg_table):
    """The *source* column, not the partition field name. Iceberg records
    ``transform(column)``, and every transform is a deterministic function of the column, so
    equal column values always land in one file group — which is what a `GROUP BY` on that
    column needs. Verified here on the identity transform; the argument is what covers
    `days(ts)` and `bucket(16, id)`, which this environment cannot write (`pyiceberg_core`).
    """
    identifier, spec = iceberg_table
    splits = bt.read.iceberg(identifier, catalog=spec)._sources[0].splits()
    assert len(splits) == 6
    assert declared_clustering(splits) == ("day",)


def test_iceberg_splits_group_to_one_unit_per_partition(iceberg_table):
    identifier, spec = iceberg_table
    groups = group_by_clustering(bt.read.iceberg(identifier, catalog=spec)._sources[0].splits())
    assert groups is not None
    assert sorted(len(g) for g in groups) == [2, 2, 2]


def test_an_iceberg_group_by_the_partition_column_needs_no_exchange(iceberg_table):
    identifier, spec = iceberg_table
    ds = bt.read.iceberg(identifier, catalog=spec).group_by("day").agg(s=col("v").sum())
    assert _partition_aligned_aggregate(ds._plan, list(ds._sources), 2, None) == ("day",)


def test_a_file_written_under_an_older_partition_spec_refuses_the_whole_set(tmp_path_factory):
    """Partition evolution is the hazard that makes Iceberg different from Delta and Hive.

    An older file's partition record holds the *old* spec's fields, so reading it against the
    current spec's columns groups by the wrong thing entirely — and would put rows sharing a
    value on different workers while the plan claims they cannot be. The old file therefore
    declares nothing, which makes `declared_clustering` refuse the set rather than half-trust
    it.
    """
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import IdentityTransform

    catalog, spec = _iceberg_catalog(tmp_path_factory.mktemp("ice_evolved"))
    catalog.create_namespace("db")
    table = catalog.create_table(
        "db.evolved",
        schema=_iceberg_schema(),
        partition_spec=PartitionSpec(
            PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="day")
        ),
    )
    table.append(
        pa.table(
            {"day": pa.array([1, 1, 2, 2], pa.int64()), "v": pa.array([10, 11, 12, 13], pa.int64())}
        )
    )
    with table.update_spec() as evolve:
        evolve.add_field("v", IdentityTransform(), "v_id")
    table = catalog.load_table("db.evolved")
    table.append(
        pa.table({"day": pa.array([3, 3], pa.int64()), "v": pa.array([20, 21], pa.int64())})
    )

    splits = bt.read.iceberg("db.evolved", catalog=spec)._sources[0].splits()
    assert len({s._task.file.spec_id for s in splits}) == 2, "the table must span two specs"
    assert [s.clustering_columns for s in splits].count(()) == 2, "old-spec files declare nothing"
    assert declared_clustering(splits) == ()


# --- the cost of asking ---------------------------------------------------------------------


def test_an_unpartitioned_source_is_rejected_without_planning_a_read(tmp_path, monkeypatch):
    """The check must not cost a split plan on a table it is going to decline.

    Planning a read over a large dataset is real driver time — measured at 25 ms for 500 flat
    Parquet files, growing with the file count — and on an unpartitioned table every
    millisecond of it is thrown away. Every distributed aggregate reaches this check, so that
    waste would be charged to queries that gain nothing.

    Enforced by making the split plan *raise*: if the fast path ever stops short-circuiting,
    this fails loudly instead of quietly costing seconds on a 50,000-file table.
    """
    import batcher.dist.executors.partition_io._sources as sources_mod

    root = str(tmp_path / "flat")
    os.makedirs(root, exist_ok=True)
    pq.write_table(pa.table({"k": [1, 2, 3], "v": [1, 2, 3]}), f"{root}/part.parquet")
    ds = bt.read.parquet(root).group_by("k").agg(s=col("v").sum())

    def _explode(*_args, **_kwargs):
        raise AssertionError("the check planned a read on a source it was going to decline")

    monkeypatch.setattr(sources_mod, "_scan_splits", _explode)
    assert _partition_aligned_aggregate(ds._plan, list(ds._sources), 4, None) == ()


def test_a_partitioned_source_declares_its_clustering_without_planning_a_read(hive_table):
    """The other half: the fast path answers from metadata the source already holds — one
    memoized, non-recursive directory listing — so it is a filter rather than a first read."""
    from batcher.dist.executors.map import _source_clustering_columns

    source = bt.read.parquet(hive_table)._sources[0]
    assert _source_clustering_columns(source) == ("day",)


def test_a_source_that_cannot_answer_is_treated_as_unpartitioned():
    """Duck-typed: a connector that declares no `partition_columns` is simply not clustered,
    which costs a missed optimization and never a wrong answer."""
    from batcher.dist.executors.map import _source_clustering_columns

    class _Mute:
        pass

    class _Broken:
        def clustering_columns(self):
            raise RuntimeError("catalog unreachable")

    assert _source_clustering_columns(_Mute()) == ()
    assert _source_clustering_columns(_Broken()) == ()


# --- a nested partition tree --------------------------------------------------------------


@pytest.fixture(scope="module")
def nested_hive(tmp_path_factory) -> str:
    """A ``year=/month=`` tree: two years, four months each."""
    root = str(tmp_path_factory.mktemp("nested"))
    table = pa.table(
        {
            "year": pa.array([2023] * 8 + [2024] * 8, pa.int64()),
            "month": pa.array([1, 1, 2, 2, 3, 3, 4, 4] * 2, pa.int64()),
            "v": pa.array(range(16), pa.int64()),
        }
    )
    bt.from_arrow(table).write.parquet(root, partition_by=["year", "month"], mode="overwrite")
    return root


def test_a_nested_tree_is_clustered_on_its_top_level_column(nested_hive):
    """One split per top-level ``year=`` directory, so the year is what a split holds constant.

    This is the *complete* guarantee, not a partial one — see the two tests below.
    """
    splits = bt.read.parquet(nested_hive)._sources[0].splits()
    assert len(splits) == 2
    assert declared_clustering(splits) == ("year",)


def test_grouping_on_a_deeper_partition_column_alongside_the_top_one_is_aligned(nested_hive):
    """`(year, month)` groups are inside `year` groups, which are inside one directory. The
    containment does the work, so nothing has to be claimed about the month."""
    ds = bt.read.parquet(nested_hive).group_by("year", "month").agg(s=col("v").sum())
    assert _partition_aligned_aggregate(ds._plan, list(ds._sources), 2, None) == ("year",)


def test_grouping_on_a_deeper_partition_column_alone_is_correctly_refused(nested_hive):
    """And this is why the top-level claim is not a limitation to be lifted later.

    `month=1` exists under *every* year, so its rows are spread across every top-level
    directory. No split granularity fixes that: splitting per leaf directory would make a
    split's value `(year, month)`, and `month` alone still straddles them. A `GROUP BY month`
    over a year-partitioned table has to shuffle, and always will.
    """
    ds = bt.read.parquet(nested_hive).group_by("month").agg(n=bt.count())
    assert _partition_aligned_aggregate(ds._plan, list(ds._sources), 2, None) == ()
    # The rows say the same thing: every month straddles both year directories.
    rows = ds.collect().to_pylist()
    assert sorted(r["n"] for r in rows) == [4, 4, 4, 4]
