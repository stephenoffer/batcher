"""Correctness for the `pushdown_gaps` rules — every rewrite preserves the result.

Importing the rule module registers its `@rule` decorators into `DEFAULT_REGISTRY`, so the
rules fire inside the full `Optimizer` that `.collect()` runs; each test below therefore
exercises the *rewritten* plan and checks it against an oracle.

The oracle is DuckDB wherever the shape is expressible in SQL (explode, ASOF, unions,
joins, row numbering). Two shapes are not:

- `unpivot` — spelled here as the `UNION ALL` it is defined to be, which is an oracle DuckDB
  evaluates without going near an UNPIVOT clause.
- `sample` — its keep-set is Batcher's own seeded row hash, which no other engine can
  reproduce. The oracle is the *unrewritten* semantics instead: sample the input on its own
  (no filter → the rule cannot fire), then apply the predicate in Python. That is exactly
  what `Filter(Sample(x))` means, so it catches a commute that changed the sampled set —
  including the fixed-count case, where the rule MUST refuse to commute.

Every rule is covered with nulls, duplicates, an empty input, and — for the join rules —
an outer join, whose null-extended rows are what a careless pushdown corrupts.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.extra.pushdown_gaps
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential


def _lists(empty: bool = False):
    rows = [] if empty else [1, 2, 2, 3, 4]
    xs = [] if empty else [[1, 2], [3, 1], [3], None, []]
    g = [] if empty else ["a", "b", "a", "b", None]
    return pa.table(
        {
            "id": pa.array(rows, pa.int64()),
            "xs": pa.array(xs, pa.list_(pa.int64())),
            "g": pa.array(g, pa.string()),
        }
    )


def _wide(empty: bool = False):
    n = 0 if empty else 3
    return pa.table(
        {
            "id": pa.array([1, 2, 2][:n], pa.int64()),
            "a": pa.array([10, None, 30][:n], pa.int64()),
            "b": pa.array([40, 50, None][:n], pa.int64()),
        }
    )


def _trades(empty: bool = False):
    n = 0 if empty else 6
    return pa.table(
        {
            "sym": pa.array(["A", "A", "A", "B", "B", "C"][:n], pa.string()),
            "ts": pa.array([10, 25, 40, 10, 30, 5][:n], pa.int64()),
            "price": pa.array([100, 101, 102, 200, 201, None][:n], pa.int64()),
        }
    )


def _quotes():
    return pa.table(
        {
            "sym": pa.array(["A", "A", "B", "B", "C"], pa.string()),
            "ts": pa.array([5, 30, 12, 28, 99], pa.int64()),
            "bid": pa.array([1, 2, 3, None, 5], pa.int64()),
        }
    )


def _part(ks):
    return pa.table(
        {
            "k": pa.array(ks, pa.int64()),
            "v": pa.array([None if k is None else k * 2 for k in ks], pa.int64()),
            "z": pa.array([None if k is None else k * 3 for k in ks], pa.int64()),
        }
    )


# --- push_filter_through_unnest / prefilter_unnest_by_list_contains ------------

_EXPLODE_SQL = "SELECT id, unnest(xs) AS xs, g FROM t"


@pytest.mark.parametrize("empty", [False, True])
def test_filter_below_explode_on_passthrough(duck, empty):
    t = _lists(empty)
    duck.register("t", t)
    out = bt.from_arrow(t).explode("xs").filter(col("id") > 1).collect()
    assert_same(out, duck.sql(f"SELECT * FROM ({_EXPLODE_SQL}) WHERE id > 1"))


def test_filter_below_explode_on_null_passthrough(duck):
    t = _lists()
    duck.register("t", t)
    out = bt.from_arrow(t).explode("xs").filter(col("g").is_null()).collect()
    assert_same(out, duck.sql(f"SELECT * FROM ({_EXPLODE_SQL}) WHERE g IS NULL"))


@pytest.mark.parametrize("empty", [False, True])
def test_filter_on_exploded_column_prefilters_the_list(duck, empty):
    # `xs = 3` appears in two rows' lists (a duplicate) and in neither the null nor the
    # empty list — the pre-filter must drop exactly the rows that contribute nothing.
    t = _lists(empty)
    duck.register("t", t)
    out = bt.from_arrow(t).explode("xs").filter(col("xs") == 3).collect()
    assert_same(out, duck.sql(f"SELECT * FROM ({_EXPLODE_SQL}) WHERE xs = 3"))


def test_mixed_filter_over_explode(duck):
    t = _lists()
    duck.register("t", t)
    out = bt.from_arrow(t).explode("xs").filter((col("xs") == 3) & (col("id") == 2)).collect()
    assert_same(out, duck.sql(f"SELECT * FROM ({_EXPLODE_SQL}) WHERE xs = 3 AND id = 2"))


def test_explode_then_group_by(duck):
    t = _lists()
    duck.register("t", t)
    out = (
        bt.from_arrow(t)
        .explode("xs")
        .filter(col("id") >= 2)
        .group_by("xs")
        .agg(n=bt.count())
        .collect()
    )
    assert_same(
        out,
        duck.sql(f"SELECT xs, count(*) AS n FROM ({_EXPLODE_SQL}) WHERE id >= 2 GROUP BY xs"),
    )


# --- push_filter_through_unpivot / push_filter_into_unpivot_columns ------------

# The definition of `unpivot(index=["id"], on=["a", "b"])`, spelled as SQL.
_MELT_SQL = (
    "SELECT id, 'a' AS variable, a AS value FROM w "
    "UNION ALL SELECT id, 'b' AS variable, b AS value FROM w"
)


def _melted(table):
    return bt.from_arrow(table).unpivot(index=["id"], on=["a", "b"])


@pytest.mark.parametrize("empty", [False, True])
def test_filter_below_unpivot_on_index(duck, empty):
    w = _wide(empty)
    duck.register("w", w)
    out = _melted(w).filter(col("id") == 2).collect()  # `id = 2` is a duplicate key
    assert_same(out, duck.sql(f"SELECT * FROM ({_MELT_SQL}) WHERE id = 2"))


@pytest.mark.parametrize("empty", [False, True])
def test_variable_equality_prunes_melted_columns(duck, empty):
    w = _wide(empty)
    duck.register("w", w)
    out = _melted(w).filter(col("variable") == "b").collect()
    assert_same(out, duck.sql(f"SELECT * FROM ({_MELT_SQL}) WHERE variable = 'b'"))


def test_unpivot_filter_on_value_stays_above(duck):
    # A `value` predicate cannot descend; nulls must still be dropped by it.
    w = _wide()
    duck.register("w", w)
    out = _melted(w).filter(col("value") > 20).collect()
    assert_same(out, duck.sql(f"SELECT * FROM ({_MELT_SQL}) WHERE value > 20"))


# --- push_filter_through_sample ------------------------------------------------


def _sampled_then_filtered(table, **sample_kwargs):
    """The unrewritten meaning of `Filter(Sample(x))`: sample alone, then filter in Python."""
    rows = bt.from_arrow(table).sample(**sample_kwargs).collect().to_pylist()
    return [r for r in rows if r["id"] is not None and r["id"] > 1]


@pytest.mark.parametrize("empty", [False, True])
def test_fraction_sample_commutes_with_filter(empty):
    t = _lists(empty)
    out = bt.from_arrow(t).sample(0.6, seed=11).filter(col("id") > 1).collect()
    assert out.to_pylist() == _sampled_then_filtered(t, fraction=0.6, seed=11)


def test_fixed_count_sample_does_not_commute_with_filter():
    # The rule must REFUSE here: `sample(n=)` keeps the n smallest-hash rows of the whole
    # input, so the answer is "sample first, then filter" — never "filter, then sample".
    t = _lists()
    out = bt.from_arrow(t).sample(n=3, seed=11).filter(col("id") > 1).collect()
    assert out.to_pylist() == _sampled_then_filtered(t, n=3, seed=11)


# --- push_filter_through_asof_join / push_filter_into_asof_by_keys -------------

_ASOF_SQL = (
    "SELECT t.sym, t.ts, t.price, q.bid FROM trades t "
    "ASOF LEFT JOIN quotes q ON t.sym = q.sym AND t.ts >= q.ts"
)


def _asof(trades):
    return bt.from_arrow(trades).join_asof(bt.from_arrow(_quotes()), on="ts", by="sym")


@pytest.mark.parametrize("empty", [False, True])
def test_filter_on_left_columns_below_asof(duck, empty):
    trades = _trades(empty)
    duck.register("trades", trades)
    duck.register("quotes", _quotes())
    out = _asof(trades).filter(col("ts") > 10).collect()
    assert_same(out, duck.sql(f"SELECT * FROM ({_ASOF_SQL}) WHERE ts > 10"))


def test_filter_on_asof_by_key_mirrors_to_the_right(duck):
    # `sym = 'A'` is a `by` key: it may descend to the left AND be mirrored to the right.
    trades = _trades()
    duck.register("trades", trades)
    duck.register("quotes", _quotes())
    out = _asof(trades).filter(col("sym") == "A").collect()
    assert_same(out, duck.sql(f"SELECT * FROM ({_ASOF_SQL}) WHERE sym = 'A'"))


def test_filter_on_asof_right_column_is_not_pushed(duck):
    # 'C' matches no quote at ts <= 5, and 'B'@30's nearest quote has a NULL bid. If the
    # predicate were pushed into the right side, a FARTHER quote would become the match.
    trades = _trades()
    duck.register("trades", trades)
    duck.register("quotes", _quotes())
    out = _asof(trades).filter(col("bid") >= 2).collect()
    assert_same(out, duck.sql(f"SELECT * FROM ({_ASOF_SQL}) WHERE bid >= 2"))


def test_asof_unmatched_rows_survive_a_left_filter(duck):
    # 'C' never matches (its only quote is at ts=99, after the trade) → null-extended.
    trades = _trades()
    duck.register("trades", trades)
    duck.register("quotes", _quotes())
    out = _asof(trades).filter(col("price").is_null() | (col("ts") < 11)).collect()
    assert_same(out, duck.sql(f"SELECT * FROM ({_ASOF_SQL}) WHERE price IS NULL OR ts < 11"))


def test_asof_output_pruned_to_the_projection(duck):
    trades = _trades()
    duck.register("trades", trades)
    duck.register("quotes", _quotes())
    out = _asof(trades).select("sym", "bid").collect()
    assert_same(out, duck.sql(f"SELECT sym, bid FROM ({_ASOF_SQL})"))


# --- prune_union_columns_under_{aggregate,join,unpivot} ------------------------


def _union(a, b, *, distinct=False):
    return bt.from_arrow(a).union(bt.from_arrow(b), distinct=distinct)


_UNION_SQL = "SELECT * FROM p1 UNION ALL SELECT * FROM p2"


@pytest.mark.parametrize("empty", [False, True])
def test_union_under_aggregate(duck, empty):
    p1, p2 = _part([1, 2, 2, None]), _part([] if empty else [2, 3])
    duck.register("p1", p1)
    duck.register("p2", p2)
    out = _union(p1, p2).group_by("k").agg(total=col("v").sum()).collect()
    assert_same(out, duck.sql(f"SELECT k, sum(v) AS total FROM ({_UNION_SQL}) GROUP BY k"))


def test_distinct_union_under_aggregate_keeps_every_column(duck):
    # The branches must NOT be pruned: `UNION` dedups over k, v AND z. (p1 and p2 share
    # k=2 with identical v/z, so exactly one of those rows survives the dedup.)
    p1, p2 = _part([1, 2]), _part([2, 3])
    duck.register("p1", p1)
    duck.register("p2", p2)
    out = _union(p1, p2, distinct=True).group_by("k").agg(n=bt.count()).collect()
    assert_same(
        out,
        duck.sql(
            "SELECT k, count(*) AS n FROM (SELECT * FROM p1 UNION SELECT * FROM p2) GROUP BY k"
        ),
    )


def test_union_under_inner_join(duck):
    p1, p2, r = _part([1, 2, 2]), _part([3, None]), pa.table({"k": [1, 2, 5], "c": [9, 8, 7]})
    duck.register("p1", p1)
    duck.register("p2", p2)
    duck.register("r", r)
    out = _union(p1, p2).join(bt.from_arrow(r), on="k").select("k", "v", "c").collect()
    assert_same(
        out,
        duck.sql(f"SELECT u.k, u.v, r.c FROM ({_UNION_SQL}) u JOIN r ON u.k = r.k"),
    )


def test_union_under_left_outer_join(duck):
    # The null-producing side must stay null-producing: k=3 and k=NULL match nothing.
    p1, p2, r = _part([1, 2, 2]), _part([3, None]), pa.table({"k": [1, 2], "c": [9, 8]})
    duck.register("p1", p1)
    duck.register("p2", p2)
    duck.register("r", r)
    out = _union(p1, p2).join(bt.from_arrow(r), on="k", how="left").select("k", "v", "c").collect()
    assert_same(
        out,
        duck.sql(f"SELECT u.k, u.v, r.c FROM ({_UNION_SQL}) u LEFT JOIN r ON u.k = r.k"),
    )


def test_union_under_unpivot(duck):
    p1, p2 = _part([1, 2]), _part([3, None])
    duck.register("p1", p1)
    duck.register("p2", p2)
    out = _union(p1, p2).unpivot(index=["k"], on=["v"]).collect()
    assert_same(
        out,
        duck.sql(f"SELECT k, 'v' AS variable, v AS value FROM ({_UNION_SQL})"),
    )


# --- drop_dead_row_index (and the filter pushdown it must NOT enable) ----------


def test_row_index_survives_a_filter_above_it(duck):
    # The index numbers rows by POSITION. If a filter were pushed below the RowId, the
    # survivors would be renumbered 0..k — this is the test that catches it.
    t = pa.table({"k": pa.array([5, 6, 7, 8], pa.int64())})
    duck.register("t", t)
    out = bt.from_arrow(t).with_row_index().filter(col("k") > 6).collect()
    assert_same(
        out,
        duck.sql("SELECT * FROM (SELECT row_number() OVER () - 1 AS index, k FROM t) WHERE k > 6"),
    )


def test_dead_row_index_is_dropped(duck):
    t = pa.table({"k": pa.array([5, 6, 6, None], pa.int64())})
    duck.register("t", t)
    out = bt.from_arrow(t).with_row_index().group_by("k").agg(n=bt.count()).collect()
    assert_same(out, duck.sql("SELECT k, count(*) AS n FROM t GROUP BY k"))
