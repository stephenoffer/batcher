"""Differential tests vs DuckDB for the `join_elim` (join-elimination) rules.

Removing a join is the rewrite most able to silently change an answer, so every case
here runs the *whole* optimizer (the rule really fires — asserted on the plan shape) and
compares the executed result against DuckDB computing the same logical join. The tables
carry the inputs that break a naive elimination: NULL keys (which match nothing, not even
themselves), duplicate keys on the build side (fan-out), left rows with no match at all
(the filtering half a plain inner join can never be relieved of), and empty sides.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
import batcher.kyber.rules.extra.join_elim
from _harness import assert_same
from batcher import col
from batcher.api.dataset import Dataset
from batcher.io.source import InMemorySource, source_statistics
from batcher.kyber.optimizer import Optimizer
from batcher.plan.expr_ir import lit
from batcher.plan.logical import Join, Limit, Scan
from batcher.plan.schema import SchemaRef
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance
from batcher.plan.visitor import walk

# --- fixtures ----------------------------------------------------------------


def _left() -> pa.Table:
    # A duplicate key, a NULL key, and key 3 which has no match on the right.
    return pa.table({"k": [1, 2, 2, None, 3], "v": [10, 20, 21, 99, 30]})


def _right() -> pa.Table:
    # Key 1 is duplicated (a left join to it fans out) and key 9 matches nothing.
    return pa.table({"k": [1, 1, 2, 9, None], "w": [5, 6, 7, 8, 9]})


def _reg(duck, name: str, table: pa.Table):
    duck.register(name, table)
    return bt.from_arrow(table)


def _pq(tmp_path, table: pa.Table, name: str):
    import pyarrow.parquet as pq

    path = str(tmp_path / name)
    pq.write_table(table, path)
    return bt.read.parquet(path)


def _optimized(ds):
    """The logically-rewritten plan, with the sources' own (footer/declared) statistics."""
    stats = [source_statistics(s) for s in ds._sources]
    return Optimizer(sources=ds._sources, source_stats=stats).logical_rewrite(ds._plan)


def _no_join(ds) -> bool:
    return not any(isinstance(n, Join) for n in walk(_optimized(ds)))


def _joins(ds) -> list[Join]:
    return [n for n in walk(_optimized(ds)) if isinstance(n, Join)]


def _empty_marked(ds) -> bool:
    return any(isinstance(n, Limit) and n.n == 0 for n in walk(_optimized(ds)))


class _KeyedSource(InMemorySource):
    """An in-memory source declaring a primary key: EXACT ndv == row count, no nulls."""

    __slots__ = ("_key",)

    def __init__(self, batches, key: str) -> None:
        super().__init__(batches)
        self._key = key

    def statistics(self) -> SourceStatistics:
        rows = self.row_count()
        return SourceStatistics(
            row_count=rows,
            columns={self._key: ColumnStat(ndv=rows, null_count=0, provenance=Provenance.EXACT)},
        )


def _keyed(table: pa.Table, key: str) -> Dataset:
    src = _KeyedSource(table.to_batches(), key)
    return Dataset(Scan(source_id=0, schema=SchemaRef.from_arrow(src.schema())), [src])


# --- eliminate_left_join_under_distinct --------------------------------------


def test_left_join_under_distinct_eliminated(duck):
    left, right = _left(), _right()
    _reg(duck, "l", left), _reg(duck, "r", right)
    ds = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="left").select("k", "v")
    out = ds.distinct()
    assert _no_join(out)  # DISTINCT undoes the fan-out → the join is dead weight
    assert_same(out.collect(), duck.sql("SELECT DISTINCT l.k, l.v FROM l LEFT JOIN r ON l.k = r.k"))


def test_left_join_under_distinct_empty_right(duck):
    left = _left()
    empty = pa.table({"k": pa.array([], pa.int64()), "w": pa.array([], pa.int64())})
    _reg(duck, "l2", left), _reg(duck, "r2", empty)
    out = (
        bt.from_arrow(left)
        .join(bt.from_arrow(empty), on="k", how="left")
        .select("k", "v")
        .distinct()
    )
    assert_same(
        out.collect(), duck.sql("SELECT DISTINCT l2.k, l2.v FROM l2 LEFT JOIN r2 ON l2.k = r2.k")
    )


def test_right_join_under_distinct_eliminated(duck):
    left, right = _left(), _right()
    _reg(duck, "l3", left), _reg(duck, "r3", right)
    out = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="right").select("w").distinct()
    assert _no_join(out)  # the right side is the preserved one
    assert_same(
        out.collect(), duck.sql("SELECT DISTINCT r3.w FROM l3 RIGHT JOIN r3 ON l3.k = r3.k")
    )


def test_left_join_under_distinct_using_right_column_still_joins(duck):
    left, right = _left(), _right()
    _reg(duck, "l4", left), _reg(duck, "r4", right)
    out = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="left").select("k", "w")
    assert not _no_join(out.distinct())  # `w` is read → the join must stay
    assert_same(
        out.distinct().collect(),
        duck.sql("SELECT DISTINCT l4.k, r4.w FROM l4 LEFT JOIN r4 ON l4.k = r4.k"),
    )


# --- inner_join_to_semi_when_right_unique ------------------------------------


def test_inner_join_to_semi_on_unique_right(duck):
    left, right = _left(), _right()
    _reg(duck, "l5", left), _reg(duck, "r5", right)
    keys = bt.from_arrow(right).select("k").distinct()  # provably unique on `k`
    out = bt.from_arrow(left).join(keys, on="k", how="inner").select("k", "v")
    assert [j.join_type for j in _joins(out)] == ["semi"]  # reduced, NOT eliminated
    assert_same(
        out.collect(),
        duck.sql("SELECT l5.k, l5.v FROM l5 JOIN (SELECT DISTINCT k FROM r5) d ON l5.k = d.k"),
    )


def test_inner_join_to_semi_empty_right(duck):
    left = _left()
    empty = pa.table({"k": pa.array([], pa.int64())})
    _reg(duck, "l6", left), _reg(duck, "r6", empty)
    out = bt.from_arrow(left).join(bt.from_arrow(empty).distinct(), on="k", how="inner")
    assert_same(
        out.select("k", "v").collect(),
        duck.sql("SELECT l6.k, l6.v FROM l6 JOIN (SELECT DISTINCT k FROM r6) d ON l6.k = d.k"),
    )


# --- self_join_elimination ---------------------------------------------------


def test_self_join_on_primary_key_eliminated(duck):
    t = pa.table({"id": [1, 2, 3, 4], "v": [10, 20, 30, 40]})
    duck.register("pk", t)
    ds = _keyed(t, "id")
    out = ds.join(ds, on="id", how="inner").select("id", "v")
    assert _no_join(out)  # unique + non-null key → every row matches itself, once
    assert_same(out.collect(), duck.sql("SELECT a.id, a.v FROM pk a JOIN pk b ON a.id = b.id"))


def test_self_join_without_key_proof_still_joins(duck):
    # Same query over a plain source: uniqueness is unproven, so the join stays — and the
    # answer must still be right (key 2 is duplicated, so it fans out 2x2).
    t = pa.table({"id": [1, 2, 2, None], "v": [10, 20, 21, 99]})
    duck.register("dup", t)
    ds = bt.from_arrow(t)
    out = ds.join(ds, on="id", how="inner").select("id", "v")
    assert not _no_join(out)
    assert_same(out.collect(), duck.sql("SELECT a.id, a.v FROM dup a JOIN dup b ON a.id = b.id"))


# --- self_semi_join_to_filter / self_anti_join_to_null_keys ------------------


def test_self_semi_join_becomes_null_filter(duck):
    t = pa.table({"k": [1, 2, 2, None, 3], "v": [10, 20, 21, 99, 30]})
    duck.register("s1", t)
    ds = bt.from_arrow(t)
    out = ds.join(ds, on="k", how="semi")
    assert _no_join(out)  # every row is its own witness — unless its key is NULL
    assert_same(
        out.collect(),
        duck.sql("SELECT * FROM s1 a WHERE EXISTS (SELECT 1 FROM s1 b WHERE b.k = a.k)"),
    )


def test_self_anti_join_keeps_only_null_keys(duck):
    t = pa.table({"k": [1, 2, None, None], "v": [10, 20, 30, 40]})
    duck.register("s2", t)
    ds = bt.from_arrow(t)
    out = ds.join(ds, on="k", how="anti")
    assert _no_join(out)
    assert_same(
        out.collect(),
        duck.sql("SELECT * FROM s2 a WHERE NOT EXISTS (SELECT 1 FROM s2 b WHERE b.k = a.k)"),
    )


def test_self_anti_join_on_non_null_key_is_empty(duck):
    t = pa.table({"id": [1, 2, 3], "v": [10, 20, 30]})
    duck.register("s3", t)
    ds = _keyed(t, "id")  # id is declared non-null → no row can fail to match itself
    out = ds.join(ds, on="id", how="anti")
    assert _empty_marked(out) and _no_join(out)
    assert_same(
        out.collect(),
        duck.sql("SELECT * FROM s3 a WHERE NOT EXISTS (SELECT 1 FROM s3 b WHERE b.id = a.id)"),
    )


def test_self_semi_join_over_empty_source(duck):
    t = pa.table({"k": pa.array([], pa.int64()), "v": pa.array([], pa.int64())})
    duck.register("s4", t)
    ds = bt.from_arrow(t)
    assert_same(
        ds.join(ds, on="k", how="semi").collect(),
        duck.sql("SELECT * FROM s4 a WHERE EXISTS (SELECT 1 FROM s4 b WHERE b.k = a.k)"),
    )


# --- cartesian joins ---------------------------------------------------------


def test_cross_join_of_scalar_subquery_eliminated(duck):
    a = pa.table({"x": [1, 2, 3]})
    s = pa.table({"y": [5, 6, 7]})
    _reg(duck, "ca", a), _reg(duck, "cs", s)
    scalar = bt.from_arrow(s).agg(m=col("y").max())  # exactly one row (EXACT)
    out = bt.from_arrow(a).cross_join(scalar).select("x")
    assert _no_join(out)  # a constant key cannot filter; one row cannot duplicate
    assert_same(out.collect(), duck.sql("SELECT x FROM ca, (SELECT max(y) AS m FROM cs)"))


def test_cross_join_of_scalar_subquery_empty_left(duck):
    a = pa.table({"x": pa.array([], pa.int64())})
    s = pa.table({"y": [5, 6]})
    _reg(duck, "ce", a), _reg(duck, "cs2", s)
    out = bt.from_arrow(a).cross_join(bt.from_arrow(s).agg(m=col("y").max())).select("x")
    assert_same(out.collect(), duck.sql("SELECT x FROM ce, (SELECT max(y) AS m FROM cs2)"))


def test_cross_join_of_multi_row_side_still_joins(duck):
    a = pa.table({"x": [1, 2, 3]})
    s = pa.table({"y": [5, 6]})
    _reg(duck, "ma", a), _reg(duck, "ms", s)
    out = bt.from_arrow(a).cross_join(bt.from_arrow(s)).select("x")
    assert not _no_join(out)  # 2 rows on the other side → each `x` comes out twice
    assert_same(out.collect(), duck.sql("SELECT x FROM ma, ms"))


def _cartesian(left: pa.Table, right: pa.Table, how: str):
    ck = "ck"
    return (
        bt.from_arrow(left)
        .with_columns(**{ck: lit(1)})
        .join(bt.from_arrow(right).with_columns(**{ck: lit(1)}), on=ck, how=how)
        .select("x")
    )


def test_semi_join_of_nonempty_cartesian(duck):
    a, b = pa.table({"x": [1, 2, None]}), pa.table({"y": [7, 8]})
    _reg(duck, "ea", a), _reg(duck, "eb", b)
    out = _cartesian(a, b, "semi")
    assert _no_join(out)  # `b` is non-empty → every row of `a` has a match
    assert_same(out.collect(), duck.sql("SELECT x FROM ea WHERE EXISTS (SELECT 1 FROM eb)"))


def test_anti_join_of_nonempty_cartesian_is_empty(duck):
    a, b = pa.table({"x": [1, 2, None]}), pa.table({"y": [7, 8]})
    _reg(duck, "aa", a), _reg(duck, "ab", b)
    out = _cartesian(a, b, "anti")
    assert _no_join(out)
    assert_same(out.collect(), duck.sql("SELECT x FROM aa WHERE NOT EXISTS (SELECT 1 FROM ab)"))


def test_cartesian_semi_anti_over_empty_right(duck):
    a = pa.table({"x": [1, 2, None]})
    b = pa.table({"y": pa.array([], pa.int64())})
    _reg(duck, "za", a), _reg(duck, "zb", b)
    # The rules stand down (an empty right proves the *opposite* verdict); results must hold.
    assert_same(
        _cartesian(a, b, "semi").collect(),
        duck.sql("SELECT x FROM za WHERE EXISTS (SELECT 1 FROM zb)"),
    )
    assert_same(
        _cartesian(a, b, "anti").collect(),
        duck.sql("SELECT x FROM za WHERE NOT EXISTS (SELECT 1 FROM zb)"),
    )


# --- provably-disjoint key ranges (EXACT parquet footer bounds) ---------------


def _disjoint(tmp_path, duck):
    left = pa.table({"k": [1, 2, 3, None], "v": [10, 20, 30, 40]})
    right = pa.table({"k": [100, 101], "w": [7, 8]})
    duck.register("dl", left)
    duck.register("dr", right)
    return _pq(tmp_path, left, "dl.parquet"), _pq(tmp_path, right, "dr.parquet")


def test_inner_join_disjoint_keys_is_empty(tmp_path, duck):
    left, right = _disjoint(tmp_path, duck)
    out = left.join(right, on="k", how="inner").select("k", "w")
    assert _empty_marked(out)  # [1,3] and [100,101] cannot overlap → nothing matches
    assert_same(out.collect(), duck.sql("SELECT dl.k, dr.w FROM dl JOIN dr ON dl.k = dr.k"))


def test_semi_join_disjoint_keys_is_empty(tmp_path, duck):
    left, right = _disjoint(tmp_path, duck)
    out = left.join(right, on="k", how="semi")
    assert _empty_marked(out)
    assert_same(
        out.collect(),
        duck.sql("SELECT * FROM dl a WHERE EXISTS (SELECT 1 FROM dr b WHERE b.k = a.k)"),
    )


def test_left_join_disjoint_keys_is_the_left_side(tmp_path, duck):
    left, right = _disjoint(tmp_path, duck)
    out = left.join(right, on="k", how="left").select("k", "v")
    assert _no_join(out)  # nothing matches → every left row, once, unchanged
    assert_same(out.collect(), duck.sql("SELECT dl.k, dl.v FROM dl LEFT JOIN dr ON dl.k = dr.k"))


def test_anti_join_disjoint_keys_is_the_left_side(tmp_path, duck):
    left, right = _disjoint(tmp_path, duck)
    out = left.join(right, on="k", how="anti")
    assert _no_join(out)
    assert_same(
        out.collect(),
        duck.sql("SELECT * FROM dl a WHERE NOT EXISTS (SELECT 1 FROM dr b WHERE b.k = a.k)"),
    )


def test_overlapping_key_ranges_still_join(tmp_path, duck):
    left = pa.table({"k": [1, 2, 3], "v": [10, 20, 30]})
    right = pa.table({"k": [3, 4], "w": [7, 8]})
    duck.register("ol", left)
    duck.register("orr", right)
    dsl, dsr = _pq(tmp_path, left, "ol.parquet"), _pq(tmp_path, right, "orr.parquet")
    out = dsl.join(dsr, on="k", how="inner").select("k", "w")
    assert not _empty_marked(out)  # the ranges touch at 3 → a match is possible
    assert_same(out.collect(), duck.sql("SELECT ol.k, orr.w FROM ol JOIN orr ON ol.k = orr.k"))


def test_sketched_ndv_is_not_a_uniqueness_proof(duck):
    """A measured (HyperLogLog) ndv must never license a uniqueness-gated rewrite.

    `_right_unique_on_keys` gated on the column bundle's `provenance`, but a scanned
    column carries EXACT *bounds* alongside a *sketched* ndv — so the bundle reads EXACT
    while the count itself is a ~1%-error estimate. HLL overestimates above ~50k distinct
    values, making `ndv >= rows` true for a table that is not unique, which then licensed
    `eliminate_left_join`, `pre_aggregation_through_join`, and
    `inner_join_to_semi_when_right_unique`.

    The visible damage: this LEFT JOIN dropped one of key 0's two matches, and the
    grouped aggregate below it halved SUM and COUNT. The gate must read the ndv's own
    tag (`ndv_is_exact`), as every other consumer of these stats does.

    The row count has to clear HLL's overestimate threshold (~50k) for the sketch to
    exceed the true ndv, which is why this needs a large fixture rather than a toy one.
    """
    n = 60_000
    # Key 0 appears twice, so the right side is NOT unique on `k`.
    right = pa.table({"k": [*range(n - 1), 0], "c": [1.0] * n})
    left = pa.table({"k": [0] * 40 + [1, 2], "a": list(range(42))})
    duck.register("u_right", right)
    duck.register("u_left", left)
    dsl, dsr = bt.from_arrow(left), bt.from_arrow(right)

    # The LEFT JOIN must keep both of key 0's matches for every left row.
    assert_same(
        dsl.join(dsr, on="k", how="left").select("k", "a").collect(),
        duck.sql("SELECT l.k, l.a FROM u_left l LEFT JOIN u_right r ON l.k = r.k"),
    )
    # And an aggregate must not be pushed below the fan-out.
    assert_same(
        dsl.join(dsr, on="k", how="inner")
        .group_by("k")
        .agg(sa=bt.sum(col("a")), n=bt.count())
        .collect(),
        duck.sql(
            "SELECT k, sum(a) AS sa, count(*) AS n FROM u_left JOIN u_right USING (k) GROUP BY k"
        ),
    )
