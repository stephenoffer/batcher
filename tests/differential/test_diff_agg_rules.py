"""`agg_rules` rewrites preserve results vs DuckDB.

Every rule runs end to end through the full optimizer and is compared against DuckDB.
Because these rules fire only on *proven* statistics, the inputs are the two sources that
actually carry them: a **Parquet** file (whose footer declares EXACT min/max/null-count/row
count) and a source that declares an EXACT unique key the way a catalog declares a primary
key. `_optimized` re-uses the conductor's own statistics collection (`collect_source_stats`
with `column_bounds_needed`), so an assertion that a rule fired is an assertion about the
plan `.collect()` really runs.

Coverage is the edges the rules reason about: NULL groups, an all-NULL column, duplicate
keys, a unique key, a single-row relation, and an empty one — where a grouped aggregate emits
no row but a global one must still emit ``(COUNT = 0, everything else NULL)``.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher import col
from batcher.api.dataset import Dataset
from batcher.api.orchestration import collect_source_stats
from batcher.api.source_stats import column_bounds_needed
from batcher.io.source import InMemorySource
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.rules.extra import agg_rules as _agg_rules  # noqa: F401  (registers the rules)
from batcher.plan.logical import Aggregate, Scan
from batcher.plan.schema import SchemaRef
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance
from batcher.plan.visitor import walk
from conftest import assert_same

# `id` is unique; `g` has a NULL group and duplicate keys; `k` is a non-null constant; `x`
# carries a NULL; `n` is entirely NULL.
_DATA = pa.table(
    {
        "id": [1, 2, 3, 4, 5],
        "g": ["a", "a", "b", "b", None],
        "k": [7, 7, 7, 7, 7],
        "x": [10, 20, None, 40, 50],
        "n": pa.array([None] * 5, pa.int64()),
    }
)
_ONE = _DATA.slice(0, 1)
_EMPTY = _DATA.slice(0, 0)
_TABLES = {"t": _DATA, "one": _ONE, "e": _EMPTY}


class _KeyedSource(InMemorySource):
    """An in-memory source that declares an EXACT unique key (a catalog primary key).

    `resident = False` so the conductor takes the full `statistics()` path rather than the
    resident fast path (which reports only a row count and the bounds a predicate needs) —
    otherwise the declaration would never reach the optimizer.
    """

    __slots__ = ("_key",)
    resident = False

    def __init__(self, batches, key: str) -> None:
        super().__init__(batches)
        self._key = key

    def statistics(self) -> SourceStatistics:
        base = InMemorySource.statistics(self)
        columns = dict(base.columns)
        stat = columns.get(self._key, ColumnStat())
        columns[self._key] = ColumnStat(
            min=stat.min,
            max=stat.max,
            null_count=0,
            ndv=self.row_count(),
            provenance=Provenance.EXACT,
        )
        return SourceStatistics(row_count=self.row_count(), columns=columns)


@pytest.fixture(scope="module")
def pq_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("agg_rules")
    for name, table in _TABLES.items():
        pq.write_table(table, str(path / f"{name}.parquet"))
    return path


@pytest.fixture(autouse=True)
def _register(duck):
    for name, table in _TABLES.items():
        duck.register(name, table)


@pytest.fixture
def t(pq_dir):
    return bt.read.parquet(str(pq_dir / "t.parquet"))


@pytest.fixture
def one(pq_dir):
    return bt.read.parquet(str(pq_dir / "one.parquet"))


@pytest.fixture
def empty(pq_dir):
    return bt.read.parquet(str(pq_dir / "e.parquet"))


def _keyed(table, key="id"):
    src = _KeyedSource(table.to_batches(), key)
    return Dataset(Scan(source_id=0, schema=SchemaRef.from_arrow(src.schema())), [src])


def _optimized(ds):
    """The optimized plan — over exactly the statistics the conductor collects for it."""
    stats = collect_source_stats(ds._sources, None, need_columns=column_bounds_needed(ds._plan))
    return Optimizer(sources=ds._sources, source_stats=stats).logical_rewrite(ds._plan)


def _aggs(ds) -> set[str]:
    """The aggregate functions the plan `.collect()` runs still computes."""
    return {
        spec.agg.func
        for node in walk(_optimized(ds))
        if isinstance(node, Aggregate)
        for spec in node.aggregates
    }


def _group_keys(ds) -> list[int]:
    return [len(n.group_keys) for n in walk(_optimized(ds)) if isinstance(n, Aggregate)]


# --- drop_group_key_functionally_determined_by_another ------------------------


def test_group_by_unique_key_and_another(duck):
    ds = _keyed(_DATA).group_by("id", "g").agg(s=col("x").sum())
    assert _group_keys(ds) == [1]  # `g` is functionally determined by the unique `id`
    assert_same(ds.collect(), duck.sql("SELECT id, g, sum(x) AS s FROM t GROUP BY id, g"))


def test_group_by_unique_key_with_null_group(duck):
    # The NULL `g` row must survive as its own group, carried through MIN(g) as NULL.
    ds = _keyed(_DATA).group_by("id", "g", "k").agg(n=col("x").count())
    assert _group_keys(ds) == [1]
    assert_same(ds.collect(), duck.sql("SELECT id, g, k, count(x) AS n FROM t GROUP BY id, g, k"))


def test_group_by_duplicate_keys_is_untouched(duck, t):
    # `g` is *not* unique — grouping by (g, k) must keep both keys and merge duplicates.
    ds = t.group_by("g", "k").agg(s=col("x").sum())
    assert _group_keys(ds) == [2]
    assert_same(ds.collect(), duck.sql("SELECT g, k, sum(x) AS s FROM t GROUP BY g, k"))


# --- count_distinct_of_unique_column ------------------------------------------


def test_count_distinct_of_unique_key(duck):
    ds = _keyed(_DATA).group_by("g").agg(n=col("id").n_unique())
    assert "count_distinct" not in _aggs(ds)
    assert_same(ds.collect(), duck.sql("SELECT g, count(DISTINCT id) AS n FROM t GROUP BY g"))


def test_count_distinct_of_unique_key_global(duck):
    ds = _keyed(_DATA).group_by().agg(n=col("id").n_unique())
    assert "count_distinct" not in _aggs(ds)
    assert_same(ds.collect(), duck.sql("SELECT count(DISTINCT id) AS n FROM t"))


def test_count_distinct_of_duplicate_column_is_untouched(duck, t):
    ds = t.group_by("g").agg(n=col("k").n_unique())  # `k` repeats — the distinct count stands
    assert_same(ds.collect(), duck.sql("SELECT g, count(DISTINCT k) AS n FROM t GROUP BY g"))


# --- count_of_non_null_column --------------------------------------------------


def test_count_of_null_free_column(duck, t):
    ds = t.group_by("g").agg(n=col("k").count())
    assert _aggs(ds) == {"count_star"}  # COUNT(k) → COUNT(*)
    assert_same(ds.collect(), duck.sql("SELECT g, count(k) AS n FROM t GROUP BY g"))


def test_count_of_null_bearing_column_is_untouched(duck, t):
    ds = t.group_by("g").agg(n=col("x").count(), m=bt.count())
    assert "count" in _aggs(ds)  # `x` holds a NULL — COUNT(x) is not COUNT(*)
    assert_same(ds.collect(), duck.sql("SELECT g, count(x) AS n, count(*) AS m FROM t GROUP BY g"))


def test_count_of_all_null_column(duck, t):
    ds = t.group_by("g").agg(n=col("n").count())
    assert_same(ds.collect(), duck.sql("SELECT g, count(n) AS n FROM t GROUP BY g"))


# --- constant-column folds (min/max/mean/sum) ----------------------------------


def test_min_max_of_constant_column(duck, t):
    ds = t.group_by("g").agg(lo=col("k").min(), hi=col("k").max())
    assert not {"min", "max"} & _aggs(ds)
    assert_same(ds.collect(), duck.sql("SELECT g, min(k) AS lo, max(k) AS hi FROM t GROUP BY g"))


def test_mean_of_constant_column(duck, t):
    ds = t.group_by("g").agg(avg=col("k").mean())
    assert "mean" not in _aggs(ds)
    assert_same(ds.collect(), duck.sql("SELECT g, avg(k) AS avg FROM t GROUP BY g"))


def test_sum_of_constant_column(duck, t):
    ds = t.group_by("g").agg(s=col("k").sum())
    assert "sum" not in _aggs(ds)  # → 7 * count(*)
    assert_same(ds.collect(), duck.sql("SELECT g, sum(k) AS s FROM t GROUP BY g"))


def test_min_max_of_non_constant_column_is_untouched(duck, t):
    ds = t.group_by("g").agg(lo=col("x").min(), hi=col("x").max())
    assert {"min", "max"} <= _aggs(ds)
    assert_same(ds.collect(), duck.sql("SELECT g, min(x) AS lo, max(x) AS hi FROM t GROUP BY g"))


def test_constant_column_folds_over_empty_input(duck, empty):
    # Grouped: no rows either way. Global SUM/MIN over 0 rows is NULL — never 0, never 7.
    assert_same(
        empty.group_by("g").agg(s=col("k").sum(), lo=col("k").min()).collect(),
        duck.sql("SELECT g, sum(k) AS s, min(k) AS lo FROM e GROUP BY g"),
    )
    assert_same(
        empty.group_by().agg(s=col("k").sum(), lo=col("k").min()).collect(),
        duck.sql("SELECT sum(k) AS s, min(k) AS lo FROM e"),
    )


# --- global aggregates answered from exact metadata -----------------------------


def test_global_min_max_from_bounds(duck, t):
    ds = t.group_by().agg(lo=col("x").min(), hi=col("x").max())
    assert not _aggs(ds)  # both answered from the footer's bounds
    assert_same(ds.collect(), duck.sql("SELECT min(x) AS lo, max(x) AS hi FROM t"))


def test_global_min_of_all_null_column(duck, t):
    # An all-NULL column has no bound, so the rule stands down and MIN stays NULL.
    ds = t.group_by().agg(lo=col("n").min())
    assert_same(ds.collect(), duck.sql("SELECT min(n) AS lo FROM t"))


def test_global_min_max_over_empty(duck, empty):
    assert_same(
        empty.group_by().agg(lo=col("x").min()).collect(), duck.sql("SELECT min(x) AS lo FROM e")
    )


def test_grouped_min_max_is_untouched(duck, t):
    ds = t.group_by("g").agg(lo=col("x").min())  # a per-group extreme is not the relation's
    assert "min" in _aggs(ds)
    assert_same(ds.collect(), duck.sql("SELECT g, min(x) AS lo FROM t GROUP BY g"))


def test_global_count_star_from_cardinality(duck, t):
    ds = t.group_by().agg(n=bt.count())
    assert not _aggs(ds)  # answered from the footer's exact row count
    assert_same(ds.collect(), duck.sql("SELECT count(*) AS n FROM t"))


def test_global_count_star_over_empty(duck, empty):
    # The keyless aggregate must still emit its one row: COUNT(*) = 0, SUM = NULL.
    assert_same(
        empty.group_by().agg(n=bt.count(), s=col("x").sum()).collect(),
        duck.sql("SELECT count(*) AS n, sum(x) AS s FROM e"),
    )
    assert_same(
        empty.group_by().agg(n=bt.count()).collect(), duck.sql("SELECT count(*) AS n FROM e")
    )


# --- drop_aggregate_over_single_row_input --------------------------------------


def test_aggregate_over_single_row(duck, one):
    ds = one.group_by("g").agg(lo=col("x").min(), n=col("x").count(), c=bt.count())
    assert not _aggs(ds)  # one row → one group of one row → a projection
    assert_same(
        ds.collect(),
        duck.sql("SELECT g, min(x) AS lo, count(x) AS n, count(*) AS c FROM one GROUP BY g"),
    )


def test_aggregate_over_single_null_row(duck, tmp_path):
    table = pa.table({"g": ["a"], "x": pa.array([None], pa.int64())})
    duck.register("one_null", table)
    path = str(tmp_path / "one_null.parquet")
    pq.write_table(table, path)
    ds = bt.read.parquet(path).group_by("g").agg(lo=col("x").min(), n=col("x").count())
    assert_same(
        ds.collect(), duck.sql("SELECT g, min(x) AS lo, count(x) AS n FROM one_null GROUP BY g")
    )


def test_grouped_aggregate_over_empty(duck, empty):
    ds = empty.group_by("g").agg(lo=col("x").min(), n=bt.count())
    assert_same(ds.collect(), duck.sql("SELECT g, min(x) AS lo, count(*) AS n FROM e GROUP BY g"))


# --- merge_adjacent_aggregates_when_second_is_over_group_keys ------------------


def test_regroup_by_the_same_keys(duck, t):
    ds = t.group_by("g").agg(s=col("x").sum()).group_by("g").agg(hi=col("s").max())
    assert len([n for n in walk(_optimized(ds)) if isinstance(n, Aggregate)]) == 1
    assert_same(
        ds.collect(),
        duck.sql(
            "SELECT g, max(s) AS hi FROM (SELECT g, sum(x) AS s FROM t GROUP BY g) GROUP BY g"
        ),
    )


def test_regroup_by_a_subset_of_keys_is_untouched(duck, t):
    # Grouping the result by *fewer* keys really does merge groups — it must not fold.
    ds = t.group_by("g", "k").agg(s=col("x").sum()).group_by("g").agg(hi=col("s").max())
    assert_same(
        ds.collect(),
        duck.sql(
            "SELECT g, max(s) AS hi FROM (SELECT g, k, sum(x) AS s FROM t GROUP BY g, k) GROUP BY g"
        ),
    )


# --- drop_dead_aggregate_output -------------------------------------------------


def test_unread_aggregate_output_is_dropped(duck, t):
    ds = t.group_by("g").agg(s=col("x").sum(), dead=col("x").n_unique()).select("g", "s")
    assert "count_distinct" not in _aggs(ds)
    assert_same(ds.collect(), duck.sql("SELECT g, sum(x) AS s FROM t GROUP BY g"))


def test_read_aggregate_outputs_are_kept(duck, t):
    ds = t.group_by("g").agg(s=col("x").sum(), d=col("x").n_unique())
    assert "count_distinct" in _aggs(ds)
    assert_same(
        ds.collect(), duck.sql("SELECT g, sum(x) AS s, count(DISTINCT x) AS d FROM t GROUP BY g")
    )
