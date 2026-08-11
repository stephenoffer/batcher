"""A Hive partition column carries statistics, so a join can prune on it.

The column a table is partitioned by is the one a query most wants to prune on, and it was
the one column with no statistics at all: its values live in directory names, so the footer
sweep that stats every other column cannot see it.

That was not merely a worse estimate, it disabled an optimization.
`kyber.rules.joins.runtime_join_filter` pushes ``key BETWEEN other_min AND other_max`` onto a
join's prunable side **only when both sides' ranges are known** — so a star join against a
partitioned fact table, the shape its own docstring calls "dynamic partition pruning", could
never fire on the layout it exists for. A ten-day fact table joined to a two-day dimension
read all ten days.

Two properties are pinned here and they pull against each other. The bounds must be *present*,
or the pruning cannot happen. And they must not be *exact*, because a partition directory can
outlive its rows — which is precisely what `io.filesystem.prune_empty_dirs` exists to clean up
— so the extremes may not be attained, and an exact tag would let a metadata shortcut answer
``MIN`` with a value no row has.
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.plan.stats import Provenance

DAYS = [datetime.date(2024, 1, d) for d in range(1, 11)]


@pytest.fixture
def facts(tmp_path):
    """A ten-day partitioned fact table, three rows a day."""
    root = tmp_path / "facts"
    for day in DAYS:
        (root / f"day={day.isoformat()}").mkdir(parents=True)
        pq.write_table(pa.table({"v": [1, 2, 3]}), root / f"day={day.isoformat()}" / "p.parquet")
    return str(root)


@pytest.fixture
def dim(tmp_path):
    """A two-day dimension — the side whose range implies the partitions worth reading."""
    path = tmp_path / "dim.parquet"
    pq.write_table(pa.table({"day": [DAYS[2], DAYS[3]], "label": ["a", "b"]}), path)
    return str(path)


def _stats(path: str):
    """The `day` column's statistics as the optimizer would collect them."""
    from batcher.api.source_stats import collect_source_stats

    dataset = bt.read.parquet(path)
    collected = collect_source_stats(list(dataset._sources), None)
    return collected[0].columns.get("day")


def test_the_partition_column_has_bounds_at_all(facts):
    stat = _stats(facts)
    assert stat is not None, "the partitioned column reached the optimizer with no statistics"
    assert stat.min == DAYS[0]
    assert stat.max == DAYS[-1]
    assert stat.ndv == float(len(DAYS))


def test_the_bounds_are_not_claimed_exact(facts):
    """They are bounds, not attained values — a directory can outlive its rows."""
    assert _stats(facts).provenance is not Provenance.EXACT


def test_an_empty_partition_directory_does_not_corrupt_an_exact_answer(facts, tmp_path):
    """A directory whose rows were deleted widens the bounds; `MIN` must ignore it.

    This is the case that decides the provenance tag. `prune_empty_dirs` exists because a
    rewrite leaves `dt=x` standing after deleting its data files, so the widened bound is a
    real shape and not a contrived one.
    """
    (tmp_path / "facts" / "day=2023-12-01").mkdir(parents=True)
    assert _stats(facts).min == datetime.date(2023, 12, 1), "the bound widens, as it must"
    answered = bt.read.parquet(facts).agg(m=bt.col("day").min()).collect().to_pydict()
    assert answered["m"] == [DAYS[0]], "an unattained bound answered an exact MIN"


def test_a_join_prunes_the_partitions_its_other_side_rules_out(facts, dim):
    """Dynamic partition pruning: the dimension's range decides what the fact scan reads."""
    from batcher.api.source_stats import collect_source_stats
    from batcher.dist.executors.partition_io import source_pushdown
    from batcher.kyber import optimize_full

    joined = bt.read.parquet(facts).join(bt.read.parquet(dim), on="day", how="inner")
    sources = list(joined._sources)
    # Exactly what the conductor does (`api.orchestration.run._optimize`): collect the
    # sources' statistics, then hand them to the optimizer. The rule under test reads them.
    source_stats = collect_source_stats(sources, None)
    _phys, logical, _ = optimize_full(joined._plan, sources=sources, source_stats=source_stats)
    _projection, predicate = source_pushdown(logical, 0)
    assert predicate is not None, "no predicate reached the fact scan"

    kept = {s.subdir.rstrip("/").rsplit("/", 1)[-1] for s in sources[0].splits(predicate=predicate)}
    assert kept == {f"day={DAYS[2].isoformat()}", f"day={DAYS[3].isoformat()}"}, kept


def test_pruning_by_a_join_does_not_change_the_join(facts, dim):
    """The whole point: fewer partitions read, identical rows out."""
    joined = bt.read.parquet(facts).join(bt.read.parquet(dim), on="day", how="inner")
    got = joined.collect().to_pydict()
    assert sorted(got["day"]) == sorted([DAYS[2]] * 3 + [DAYS[3]] * 3)
    assert sorted(got["label"]) == ["a", "a", "a", "b", "b", "b"]


def test_an_unpartitioned_tree_reports_no_partition_bounds(tmp_path):
    path = tmp_path / "flat.parquet"
    pq.write_table(pa.table({"day": DAYS[:2], "v": [1, 2]}), path)
    from batcher.io.formats.structured.parquet.partitions import partition_bounds

    assert partition_bounds([], pa.schema([])) == {}


def test_a_segment_decodes_the_same_way_for_reading_and_for_bounding():
    """One decoding, three uses — a segment that decodes two ways drops rows.

    The value a worker appends to its rows, the value a predicate is pruned against, and the
    value that bounds the column all come from `typed_partition_value`.
    """
    from batcher.io.formats.structured.parquet.partitions import (
        HIVE_NULL,
        partition_bounds,
        typed_partition_value,
    )

    schema = pa.schema([pa.field("k", pa.string())])
    dirs = [("d/k=x%2Fy", ("k", "x%2Fy")), ("d/k=z", ("k", "z"))]
    assert typed_partition_value("x%2Fy", pa.string()) == "x/y", "the writer URL-encodes"
    assert partition_bounds(dirs, schema)["k"].min == "x/y"
    assert typed_partition_value(HIVE_NULL, pa.string()) is None
