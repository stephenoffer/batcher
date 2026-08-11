"""Plan-time partition pruning: a directory a predicate rules out never becomes a task.

Partitioning a table buys exactly one thing — the ability to *not read* most of it — and
that saving is only realized if the elimination happens while splits are planned. A split
that reaches a worker has already cost a listing, a scheduling slot, and a read. These
tests hold the two halves of that: `ParquetDatasetSource.splits(predicate=…)` drops the
directories, and `io.stats.file_skipping` can decide a date-typed bound at all (it
silently decided nothing before, so every date-partitioned table was read whole).

Pruning must never change the answer, so the partitioned read is also compared against the
same query over a flat copy of the same rows.
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.io.formats.structured.parquet.dataset import ParquetDatasetSource
from batcher.io.stats.file_skipping import surviving_files

DAYS = ("2024-01-01", "2024-01-02", "2024-01-03")
DATES = [datetime.date.fromisoformat(d) for d in DAYS]

#: Every predicate shape below, with the directories it must leave standing. `None` means
#: nothing is provable, which must keep all of them.
PREDICATES = [
    (bt.col("day") == "2024-01-02", {"day=2024-01-02"}),
    (bt.col("day") == DATES[1], {"day=2024-01-02"}),
    (bt.col("day") >= DATES[1], {"day=2024-01-02", "day=2024-01-03"}),
    (bt.col("day") < DATES[1], {"day=2024-01-01"}),
    (bt.col("day") != DATES[1], {"day=2024-01-01", "day=2024-01-03"}),
    (
        (bt.col("day") == DATES[0]) | (bt.col("day") == DATES[2]),
        {"day=2024-01-01", "day=2024-01-03"},
    ),
    # A conjunction prunes on the half it can decide and keeps the other.
    ((bt.col("day") == DATES[1]) & (bt.col("v") > 1), {"day=2024-01-02"}),
    # Nothing in a directory name decides a data column, so every directory survives.
    (bt.col("v") > 1, None),
    # A disjunction with one undecidable side is undecidable as a whole.
    ((bt.col("day") == DATES[0]) | (bt.col("v") > 1), None),
]


@pytest.fixture
def tree(tmp_path):
    """A three-day Hive-partitioned Parquet tree, one file per day."""
    root = tmp_path / "events"
    for i, day in enumerate(DAYS):
        (root / f"day={day}").mkdir(parents=True)
        pq.write_table(pa.table({"v": [i, i + 1, i + 2]}), root / f"day={day}" / "part.parquet")
    return str(root)


@pytest.fixture
def flat(tmp_path):
    """The same rows in one unpartitioned file — the reference a pruned read must match."""
    path = tmp_path / "flat.parquet"
    pq.write_table(
        pa.table(
            {
                "v": [i + k for i in range(3) for k in range(3)],
                "day": [d for d in DATES for _ in range(3)],
            }
        ),
        path,
    )
    return str(path)


def _kept(source: ParquetDatasetSource, predicate) -> set[str]:
    """The partition-directory names `predicate` leaves standing."""
    ir = None if predicate is None else predicate.to_ir()
    return {s.subdir.rstrip("/").rsplit("/", 1)[-1] for s in source.splits(predicate=ir)}


@pytest.mark.parametrize(("predicate", "expected"), PREDICATES)
def test_predicate_prunes_partition_directories(tree, predicate, expected):
    want = expected if expected is not None else {f"day={d}" for d in DAYS}
    assert _kept(ParquetDatasetSource(tree), predicate) == want


def test_no_predicate_keeps_every_directory(tree):
    assert _kept(ParquetDatasetSource(tree), None) == {f"day={d}" for d in DAYS}


@pytest.mark.parametrize(("predicate", "_expected"), PREDICATES)
def test_pruning_never_changes_the_answer(tree, flat, predicate, _expected):
    """The pruned partitioned read equals the same filter over an unpartitioned copy."""
    partitioned = bt.read.parquet(tree).filter(predicate).select("v", "day").collect()
    reference = bt.read.parquet(flat).filter(predicate).select("v", "day").collect()
    assert sorted(zip(*partitioned.to_pydict().values(), strict=True)) == sorted(
        zip(*reference.to_pydict().values(), strict=True)
    )


# ---- the file-skipping half, independent of any format --------------------------------


def _manifest(values: list, value_type: pa.DataType) -> pa.Table:
    """A one-column add-action manifest: a partition value is its own min and max."""
    return pa.table(
        {
            "path": pa.array([f"f{i}" for i in range(len(values))], pa.string()),
            "partition.day": pa.array(values, value_type),
        }
    )


@pytest.mark.parametrize(
    ("value_type", "values"),
    [
        # A Hive tree types the column; a lakehouse log records it as text. Both must prune.
        (pa.date32(), DATES),
        (pa.string(), list(DAYS)),
    ],
)
def test_date_bounds_prune_whatever_the_manifest_records(value_type, values):
    manifest = _manifest(values, value_type)
    predicate = (bt.col("day") == DATES[1]).to_ir()
    assert surviving_files(predicate, manifest) == ["f1"]


def test_timestamp_literal_decides_a_date_bound_at_midnight():
    """A date bound widens to that day's midnight — the same cast the engine applies.

    ``day >= 2024-01-02T12:00`` over a `date32` column is false for 2024-01-02 itself,
    because a date compares as midnight. The pruning must reach the same conclusion the
    engine's own `Filter` does, and the assertion below is checked against it.
    """
    engine = (
        bt.from_pydict({"day": DATES})
        .filter(bt.col("day") >= datetime.datetime(2024, 1, 2, 12, 0))
        .collect()
        .to_pydict()["day"]
    )
    assert engine == [DATES[2]], "the reference the prune is held to"
    manifest = _manifest(DATES, pa.date32())
    predicate = (bt.col("day") >= datetime.datetime(2024, 1, 2, 12, 0)).to_ir()
    assert surviving_files(predicate, manifest) == ["f2"]


def test_uncomparable_literal_prunes_nothing():
    """A literal that will not cast to the bound's type keeps every file, never drops one."""
    manifest = _manifest(["a", "b", "c"], pa.string())
    predicate = (bt.col("day") == DATES[1]).to_ir()
    assert surviving_files(predicate, manifest) is None


def test_weighing_a_partition_split_does_not_sweep_its_subtree(tree):
    """Split assignment must not ask a question that costs a footer per file behind it.

    `_balance` weighs splits by row count so the load packs evenly. `PartitionDirSplit`
    stands for a whole partition *directory*, and answering that question re-lists the
    subtree and opens every footer in it — so weighing the splits of a date-partitioned
    petabyte would sweep the entire table on the driver, one partition at a time, which is
    exactly the cost the distributed-listing reader exists to remove. They pack by equal
    count instead, as every split above the weighing cap already does.
    """
    from batcher.dist.executors.partition_io.assignment import _balance, split_weights

    splits = ParquetDatasetSource(tree).splits()
    asked: list[str] = []
    original = type(splits[0]).row_count
    type(splits[0]).row_count = lambda self: (asked.append(self.subdir), original(self))[1]
    try:
        assert split_weights(splits) == [1] * len(splits)
        groups = _balance(splits, 2)
    finally:
        type(splits[0]).row_count = original

    assert asked == [], f"weighing swept {len(asked)} partition subtree(s)"
    assert sum(len(g) for g in groups) == len(splits), "assignment must still cover every split"
    # The count is still available to a caller that genuinely wants it.
    assert splits[0].row_count() == 3
