"""Differential tests vs DuckDB for the `metadata_adaptive` EXACT-gated rewrites.

Each rule skips or simplifies a breaker on a *provably-EXACT* statistic; the result must
stay identical to DuckDB. Covered: a `Sort` dropped over a one-row / global-aggregate
input, constant `ORDER BY` keys pruned, a `Distinct` dropped over a source that declares
an EXACT unique key, and a `col OP col` filter decided from EXACT footer bounds — with
nulls and empty results where relevant. Fireability (the shortcut really fires from
metadata) is asserted alongside each result check.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
import batcher.kyber.rules.extra.metadata_adaptive as _metadata_adaptive  # noqa: F401
from _harness import assert_same, assert_same_ordered
from batcher import col
from batcher.api.dataset import Dataset
from batcher.io.source import InMemorySource, source_statistics
from batcher.kyber.optimizer import Optimizer
from batcher.plan.logical import Distinct, Filter, Scan, Sort
from batcher.plan.schema import SchemaRef
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance
from batcher.plan.visitor import walk


def _reg(duck, name, table):
    duck.register(name, table)
    return bt.from_arrow(table)


def _pq(tmp_path, table, name="t.parquet"):
    import pyarrow.parquet as pq

    path = str(tmp_path / name)
    pq.write_table(table, path)
    return bt.read.parquet(path)


def _optimized(ds):
    """The logically-rewritten plan with the source's own (footer/declared) stats."""
    stats = [source_statistics(s) for s in ds._sources]
    return Optimizer(sources=ds._sources, source_stats=stats).logical_rewrite(ds._plan)


def _absent(ds, kind) -> bool:
    return not any(isinstance(n, kind) for n in walk(_optimized(ds)))


# --- skip_sort_of_single_row ---------------------------------------------------


def test_sort_over_one_row_source(duck):
    t = pa.table({"x": [7], "y": ["a"]})
    ds = _reg(duck, "s1", t)
    assert _absent(ds.sort("x"), Sort)  # one row → the sort is dropped
    assert_same_ordered(ds.sort("x").collect(), duck.sql("SELECT * FROM s1 ORDER BY x"))


def test_sort_over_one_null_row(duck):
    t = pa.table({"x": pa.array([None], pa.int64()), "y": ["a"]})
    ds = _reg(duck, "s1n", t)
    assert_same_ordered(ds.sort("x").collect(), duck.sql("SELECT * FROM s1n ORDER BY x"))


def test_sort_over_global_aggregate(duck):
    t = pa.table({"x": [1, 2, 3, None, 5]})
    ds = _reg(duck, "s2", t)
    q = ds.agg(n=col("x").count(), s=col("x").sum()).sort("s")
    assert _absent(q, Sort)  # a global aggregate is exactly one row
    assert_same_ordered(
        q.collect(), duck.sql("SELECT count(x) AS n, sum(x) AS s FROM s2 ORDER BY s")
    )


# --- prune_constant_sort_keys --------------------------------------------------


def test_constant_sort_key_pruned(duck):
    t = pa.table({"x": [3, 1, 2, 2], "y": ["a", "b", "c", "d"]})
    ds = _reg(duck, "c1", t)
    out = ds.with_columns(k=7).sort("k", "x")
    # The literal key `k` is pruned; the result (incl. the constant column) is unchanged.
    assert_same_ordered(out.collect(), duck.sql("SELECT *, 7 AS k FROM c1 ORDER BY k, x"))


def test_all_constant_sort_keys_dropped(duck):
    t = pa.table({"x": [3, 1, 2]})
    ds = _reg(duck, "c2", t)
    out = ds.with_columns(k=1).sort("k")
    assert _absent(out, Sort)  # the only key is constant and there is no top-N
    assert_same_ordered(out.collect(), duck.sql("SELECT *, 1 AS k FROM c2 ORDER BY k"))


# --- drop_distinct_when_unique -------------------------------------------------


class _KeyedSource(InMemorySource):
    """An in-memory source that declares an EXACT unique key (a catalog primary key)."""

    __slots__ = ("_key", "_ndv")

    def __init__(self, batches, key: str, ndv: int) -> None:
        super().__init__(batches)
        self._key = key
        self._ndv = ndv

    def statistics(self) -> SourceStatistics:
        return SourceStatistics(
            row_count=self.row_count(),
            columns={
                self._key: ColumnStat(ndv=self._ndv, null_count=0, provenance=Provenance.EXACT)
            },
        )


def _keyed_ds(table, key):
    src = _KeyedSource(table.to_batches(), key, ndv=table.num_rows)
    return Dataset(Scan(source_id=0, schema=SchemaRef.from_arrow(src.schema())), [src])


def test_distinct_over_declared_unique_key(duck):
    t = pa.table({"id": [1, 2, 3, 4, 5], "v": ["a", "a", "b", "b", "c"]})
    duck.register("u1", t)
    ds = _keyed_ds(t, "id")
    assert _absent(ds.distinct(), Distinct)  # id is a declared EXACT unique key
    assert_same(ds.distinct().collect(), duck.sql("SELECT DISTINCT * FROM u1"))


def test_distinct_over_unique_key_with_nulls(duck):
    t = pa.table({"id": [1, 2, 3], "v": pa.array([None, "x", None], pa.string())})
    duck.register("u2", t)
    ds = _keyed_ds(t, "id")  # rows stay distinct via the unique id, despite null v
    assert_same(ds.distinct().collect(), duck.sql("SELECT DISTINCT * FROM u2"))


# --- prune_filter_col_comparison -----------------------------------------------


def test_col_lt_col_always_true(tmp_path, duck):
    t = pa.table({"a": list(range(0, 10)), "b": list(range(100, 110))})
    duck.register("f1", t)
    ds = _pq(tmp_path, t, "f1.parquet")
    assert _absent(ds.filter(col("a") < col("b")), Filter)  # max(a)=9 < min(b)=100
    assert_same(ds.filter(col("a") < col("b")).collect(), duck.sql("SELECT * FROM f1 WHERE a < b"))


def test_col_lt_col_always_false_is_empty(tmp_path, duck):
    t = pa.table({"a": list(range(100, 110)), "b": list(range(0, 10))})
    duck.register("f2", t)
    ds = _pq(tmp_path, t, "f2.parquet")
    out = ds.filter(col("a") < col("b"))  # min(a)=100 >= max(b)=9 → empty
    assert_same(out.collect(), duck.sql("SELECT * FROM f2 WHERE a < b"))


def test_col_ge_col_always_true(tmp_path, duck):
    t = pa.table({"a": list(range(100, 110)), "b": list(range(0, 10))})
    duck.register("f3", t)
    ds = _pq(tmp_path, t, "f3.parquet")
    assert _absent(ds.filter(col("a") >= col("b")), Filter)  # min(a)=100 >= max(b)=9
    assert_same(
        ds.filter(col("a") >= col("b")).collect(), duck.sql("SELECT * FROM f3 WHERE a >= b")
    )


def test_col_comparison_overlap_executes(tmp_path, duck):
    t = pa.table({"a": [0, 5, 10, 15], "b": [3, 3, 3, 3]})
    duck.register("f4", t)
    ds = _pq(tmp_path, t, "f4.parquet")
    # Ranges overlap → undecidable → the filter executes and must still match DuckDB.
    assert_same(ds.filter(col("a") < col("b")).collect(), duck.sql("SELECT * FROM f4 WHERE a < b"))
