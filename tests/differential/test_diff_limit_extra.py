"""Differential tests (vs DuckDB) for the `limit_extra` LIMIT / top-N rewrites.

Each query is one the optimizer rewrites — a top-N sinks below a projection, is pushed
into a `UNION ALL`'s branches, collapses with another top-N, or folds to the empty
relation — and each must return exactly what DuckDB returns, *in the same order*
(`assert_same_ordered`), over NULL keys, ties, an empty relation, and a limit larger than
the relation.

A tie in an `ORDER BY` makes *which* rows a top-N returns engine-defined (arrow's partial
sort is not stable, and SQL does not specify it either), so the tie cases here order by a
column and project only that column: every valid answer is then the same table, and the
test still exercises the rewrite on a non-total order.

The bounded-sample rules have no DuckDB counterpart (Batcher's `sample(n=…)` is a stable
per-row hash, not SQL `USING SAMPLE`), so they are checked against the Batcher-to-Batcher
oracle instead: the rewritten plan must return exactly what the un-limited sample does.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered, assert_tables_equal
from batcher import col
from batcher.kyber.rules.extra import limit_extra as _limit_extra  # noqa: F401  (registers)

_X = [3, 1, 4, 1, 5, 9, 2, 6]
_G = [1, 1, 2, 2, 2, 3, 3, 3]  # a deliberately tie-heavy key


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "id": list(range(len(_X))),  # unique → a total order for the ordered asserts
            "x": _X,
            "g": _G,
            "n": [1, None, 3, None, 5, 6, None, 8],  # NULL sort keys
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def u(duck):
    tbl = pa.table(
        {"id": [100, 101, 102, 103], "x": [7, 0, 8, 10], "g": [4, 4, 5, 5], "n": [1, 2, 3, 4]}
    )
    duck.register("u", tbl)
    return tbl


@pytest.fixture
def empty(duck):
    tbl = pa.table(
        {
            "id": pa.array([], pa.int64()),
            "x": pa.array([], pa.int64()),
            "g": pa.array([], pa.int64()),
            "n": pa.array([], pa.int64()),
        }
    )
    duck.register("e", tbl)
    return tbl


# --- topn_through_project --------------------------------------------------------


def test_topn_through_project(duck, t):
    out = (
        bt.from_arrow(t)
        .select(col("id").alias("k"), (col("x") * 2).alias("w"))
        .sort("k")
        .limit(3)
        .collect()
    )
    assert_same_ordered(out, duck.sql("SELECT id AS k, x * 2 AS w FROM t ORDER BY k LIMIT 3"))


def test_topn_through_project_with_offset_and_descending(duck, t):
    out = (
        bt.from_arrow(t)
        .select(col("id").alias("k"), col("x"))
        .sort("k", descending=True)
        .limit(3, offset=2)
        .collect()
    )
    assert_same_ordered(out, duck.sql("SELECT id AS k, x FROM t ORDER BY k DESC LIMIT 3 OFFSET 2"))


def test_topn_through_project_with_null_keys(duck, t):
    # NULLs sort last (both engines' default for ASC) and the limit must cut the same rows.
    out = bt.from_arrow(t).select(col("n"), col("id")).sort("n", "id").limit(5).collect()
    assert_same_ordered(out, duck.sql("SELECT n, id FROM t ORDER BY n, id LIMIT 5"))


# --- push_topn_into_union --------------------------------------------------------


def test_topn_over_union_all(duck, t, u):
    out = (
        bt.from_arrow(t)
        .select("id", "x")
        .union(bt.from_arrow(u).select("id", "x"))
        .sort("id")
        .limit(4)
        .collect()
    )
    assert_same_ordered(
        out,
        duck.sql(
            "SELECT id, x FROM (SELECT id, x FROM t UNION ALL SELECT id, x FROM u) "
            "ORDER BY id LIMIT 4"
        ),
    )


def test_topn_over_union_all_with_ties(duck, t, u):
    # A tie-heavy key, projected alone: every valid top-N is the same table.
    out = (
        bt.from_arrow(t)
        .select("g")
        .union(bt.from_arrow(u).select("g"))
        .sort("g")
        .limit(5)
        .collect()
    )
    assert_same_ordered(
        out,
        duck.sql("SELECT g FROM (SELECT g FROM t UNION ALL SELECT g FROM u) ORDER BY g LIMIT 5"),
    )


def test_topn_over_union_beyond_total_rows(duck, t, u):
    out = (
        bt.from_arrow(t)
        .select("id")
        .union(bt.from_arrow(u).select("id"))
        .sort("id")
        .limit(100)
        .collect()
    )
    assert_same_ordered(
        out,
        duck.sql(
            "SELECT id FROM (SELECT id FROM t UNION ALL SELECT id FROM u) ORDER BY id LIMIT 100"
        ),
    )


def test_topn_over_distinct_union(duck, t, u):
    # Not rewritten (the dedup crosses branches) — but it must still be right.
    out = (
        bt.from_arrow(t)
        .select("g")
        .union(bt.from_arrow(u).select("g"), distinct=True)
        .sort("g")
        .limit(3)
        .collect()
    )
    assert_same_ordered(
        out, duck.sql("SELECT g FROM (SELECT g FROM t UNION SELECT g FROM u) ORDER BY g LIMIT 3")
    )


# --- collapse_topn_over_topn -----------------------------------------------------


def test_topn_of_topn_same_keys(duck, t):
    out = (
        bt.from_arrow(t)
        .select("id", "x")
        .sort("x", "id")
        .limit(5)
        .sort("x", "id")
        .limit(2)
        .collect()
    )
    assert_same_ordered(
        out,
        duck.sql(
            "SELECT id, x FROM (SELECT id, x FROM t ORDER BY x, id LIMIT 5) ORDER BY x, id LIMIT 2"
        ),
    )


def test_topn_of_topn_different_keys(duck, t):
    # Different orderings do not collapse — the inner one is a real row filter.
    out = bt.from_arrow(t).select("id", "x").sort("x", "id").limit(5).sort("id").limit(3).collect()
    assert_same_ordered(
        out,
        duck.sql(
            "SELECT id, x FROM (SELECT id, x FROM t ORDER BY x, id LIMIT 5) ORDER BY id LIMIT 3"
        ),
    )


# --- the empty marker ------------------------------------------------------------


def test_topn_limit_zero(duck, t):
    out = bt.from_arrow(t).select("id").sort("id").limit(0).collect()
    assert_same_ordered(out, duck.sql("SELECT id FROM t ORDER BY id LIMIT 0"))


def test_empty_limit_prunes_a_filter_sort_distinct_chain(duck, t):
    out = bt.from_arrow(t).select("g").filter(col("g") > 1).distinct().sort("g").limit(0).collect()
    assert_same_ordered(
        out, duck.sql("SELECT g FROM (SELECT DISTINCT g FROM t WHERE g > 1) ORDER BY g LIMIT 0")
    )


def test_topn_over_empty_relation(duck, empty):
    out = bt.from_arrow(empty).select("id").sort("id").limit(3).collect()
    assert_same_ordered(out, duck.sql("SELECT id FROM e ORDER BY id LIMIT 3"))


# --- prune_sort_keys_after_unique_key --------------------------------------------


def test_sort_by_unique_key_then_more(duck, t):
    # `id` is unique, so `x` never breaks a tie — with or without the pruning, the order
    # is the same one DuckDB produces.
    out = bt.from_arrow(t).select("id", "x").sort("id", "x").limit(4).collect()
    assert_same_ordered(out, duck.sql("SELECT id, x FROM t ORDER BY id, x LIMIT 4"))


def test_sort_by_tied_key_then_unique_key(duck, t):
    # `g` ties heavily; `id` is the real tiebreak and must survive.
    out = bt.from_arrow(t).select("g", "id").sort("g", "id").limit(6).collect()
    assert_same_ordered(out, duck.sql("SELECT g, id FROM t ORDER BY g, id LIMIT 6"))


# --- the bounded-sample rules (Batcher-to-Batcher: no SQL counterpart) -----------


def test_limit_over_bounded_sample_is_the_identity(t):
    sampled = bt.from_arrow(t).sample(n=3, seed=11)
    assert_tables_equal(sampled.limit(5).collect(), sampled.collect(), ordered=True)
    assert_tables_equal(sampled.limit(3).collect(), sampled.collect(), ordered=True)


def test_limit_inside_the_sample_bound_still_cuts(t):
    sampled = bt.from_arrow(t).sample(n=3, seed=11)
    capped = sampled.limit(2).collect()
    assert capped.num_rows == 2
    assert_tables_equal(capped, sampled.collect().slice(0, 2), ordered=True)


def test_offset_past_the_sample_bound_is_empty(t):
    sampled = bt.from_arrow(t).sample(n=3, seed=11)
    out = sampled.limit(5, offset=3).collect()
    assert out.num_rows == 0
    assert_tables_equal(out, sampled.collect().slice(0, 0), ordered=True)
