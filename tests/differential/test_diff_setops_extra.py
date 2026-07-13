"""The extended set-op rewrites must match DuckDB after optimization.

Each rule is licensed by *set* semantics, so every shape is run **twice** — once as `UNION`
and once as `UNION ALL` — against the DuckDB query that means the same thing. That pairing is
the point: a rule that leaked past its `distinct` gate would collapse the duplicates the
`UNION ALL` case is required to keep, and the ALL query would fail. The source table
deliberately carries duplicate rows and a NULL key, and every shape is repeated over an
empty input.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.extra.setops_extra  # (importing registers the rules)
from batcher import col

_ROWS = {"a": [1, 2, 2, 3, None], "b": [10, 20, 20, 30, 40]}


@pytest.fixture
def t(duck):
    tbl = pa.table(_ROWS)
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def empty(duck):
    tbl = pa.table({"a": pa.array([], type=pa.int64()), "b": pa.array([], type=pa.int64())})
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def nn(duck):
    tbl = pa.table(
        {"x": pa.array([1, 2, 2, 3], type=pa.int64())},
        schema=pa.schema([pa.field("x", pa.int64(), nullable=False)]),
    )
    duck.register("nn", tbl)
    return tbl


# --- complementary filters partition the relation -------------------------------


def test_null_partition_union(duck, t):
    from conftest import assert_same

    ds = bt.from_arrow(t)
    out = (
        ds.filter(col("a").is_null())
        .union(ds.filter(col("a").is_not_null()), distinct=True)
        .collect()
    )
    assert_same(
        out,
        duck.sql("SELECT * FROM t WHERE a IS NULL UNION SELECT * FROM t WHERE a IS NOT NULL"),
    )


def test_null_partition_union_all(duck, t):
    from conftest import assert_same

    # The rule must NOT fire: the duplicate rows of `t` survive a UNION ALL.
    ds = bt.from_arrow(t)
    out = ds.filter(col("a").is_null()).union(ds.filter(col("a").is_not_null())).collect()
    assert_same(
        out,
        duck.sql("SELECT * FROM t WHERE a IS NULL UNION ALL SELECT * FROM t WHERE a IS NOT NULL"),
    )


def test_null_partition_union_over_empty_input(duck, empty):
    from conftest import assert_same

    ds = bt.from_arrow(empty)
    out = (
        ds.filter(col("a").is_null())
        .union(ds.filter(col("a").is_not_null()), distinct=True)
        .collect()
    )
    assert_same(
        out,
        duck.sql("SELECT * FROM t WHERE a IS NULL UNION SELECT * FROM t WHERE a IS NOT NULL"),
    )


def test_nullable_comparison_partition_is_not_a_partition(duck, t):
    from conftest import assert_same

    # `a > 2` and `a <= 2` are both NULL on the NULL row, which therefore appears in NEITHER
    # branch. If the rule wrongly fired, the NULL row would come back.
    ds = bt.from_arrow(t)
    out = ds.filter(col("a") > 2).union(ds.filter(col("a") <= 2), distinct=True).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE a > 2 UNION SELECT * FROM t WHERE a <= 2"))


def test_non_nullable_comparison_partition(duck, nn):
    from conftest import assert_same

    ds = bt.from_arrow(nn)
    out = ds.filter(col("x") > 2).union(ds.filter(col("x") <= 2), distinct=True).collect()
    assert_same(out, duck.sql("SELECT * FROM nn WHERE x > 2 UNION SELECT * FROM nn WHERE x <= 2"))


def test_non_nullable_comparison_partition_union_all(duck, nn):
    from conftest import assert_same

    ds = bt.from_arrow(nn)
    out = ds.filter(col("x") > 2).union(ds.filter(col("x") <= 2)).collect()
    assert_same(
        out, duck.sql("SELECT * FROM nn WHERE x > 2 UNION ALL SELECT * FROM nn WHERE x <= 2")
    )


# --- filters on the same relation merge into one OR -----------------------------


def test_overlapping_filters_union(duck, t):
    from conftest import assert_same

    # `a = 2` and `b = 20` select the SAME (duplicated) rows: the dedup is load bearing.
    ds = bt.from_arrow(t)
    out = ds.filter(col("a") == 2).union(ds.filter(col("b") == 20), distinct=True).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE a = 2 UNION SELECT * FROM t WHERE b = 20"))


def test_overlapping_filters_union_all(duck, t):
    from conftest import assert_same

    # The rule must NOT fire: a row satisfying both predicates appears TWICE.
    ds = bt.from_arrow(t)
    out = ds.filter(col("a") == 2).union(ds.filter(col("b") == 20)).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE a = 2 UNION ALL SELECT * FROM t WHERE b = 20"))


def test_disjoint_filters_union(duck, t):
    from conftest import assert_same

    ds = bt.from_arrow(t)
    out = ds.filter(col("a") == 1).union(ds.filter(col("b") == 30), distinct=True).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE a = 1 UNION SELECT * FROM t WHERE b = 30"))


def test_three_filters_on_one_relation_union(duck, t):
    from conftest import assert_same

    ds = bt.from_arrow(t)
    out = (
        ds.filter(col("a") == 1)
        .union(ds.filter(col("a") == 2), ds.filter(col("b") == 30), distinct=True)
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT * FROM t WHERE a = 1 UNION SELECT * FROM t WHERE a = 2 "
            "UNION SELECT * FROM t WHERE b = 30"
        ),
    )


def test_filters_on_different_relations_union(duck, t):
    from conftest import assert_same

    other = pa.table({"a": [7, 8], "b": [70, 80]})
    duck.register("u", other)
    out = (
        bt.from_arrow(t)
        .filter(col("a") == 1)
        .union(bt.from_arrow(other).filter(col("b") == 70), distinct=True)
        .collect()
    )
    assert_same(out, duck.sql("SELECT * FROM t WHERE a = 1 UNION SELECT * FROM u WHERE b = 70"))


def test_filters_union_over_empty_input(duck, empty):
    from conftest import assert_same

    ds = bt.from_arrow(empty)
    out = ds.filter(col("a") == 1).union(ds.filter(col("b") == 30), distinct=True).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE a = 1 UNION SELECT * FROM t WHERE b = 30"))


# --- absorption of a subsumed branch --------------------------------------------


def test_relation_union_its_own_filtered_subset(duck, t):
    from conftest import assert_same

    ds = bt.from_arrow(t)
    out = ds.union(ds.filter(col("a") == 1), distinct=True).collect()
    assert_same(out, duck.sql("SELECT * FROM t UNION SELECT * FROM t WHERE a = 1"))


def test_relation_union_all_its_own_filtered_subset(duck, t):
    from conftest import assert_same

    # The rule must NOT fire: the subset's rows are additional copies under bag semantics.
    ds = bt.from_arrow(t)
    out = ds.union(ds.filter(col("a") == 1)).collect()
    assert_same(out, duck.sql("SELECT * FROM t UNION ALL SELECT * FROM t WHERE a = 1"))


def test_relation_union_its_own_limited_subset(duck, t):
    from conftest import assert_same

    ds = bt.from_arrow(t)
    out = ds.union(ds.sort("a").limit(2), distinct=True).collect()
    assert_same(
        out, duck.sql("SELECT * FROM t UNION SELECT * FROM (SELECT * FROM t ORDER BY a LIMIT 2)")
    )


def test_absorption_over_empty_input(duck, empty):
    from conftest import assert_same

    ds = bt.from_arrow(empty)
    out = ds.union(ds.filter(col("a") == 1), distinct=True).collect()
    assert_same(out, duck.sql("SELECT * FROM t UNION SELECT * FROM t WHERE a = 1"))


# --- the union's dedup under a consumer that dedups anyway ------------------------


def test_max_over_a_distinct_union(duck, t):
    from conftest import assert_same

    ds = bt.from_arrow(t)
    out = ds.union(ds, distinct=True).group_by("a").agg(m=col("b").max()).collect()
    assert_same(
        out,
        duck.sql("SELECT a, max(b) AS m FROM (SELECT * FROM t UNION SELECT * FROM t) GROUP BY a"),
    )


def test_sum_over_a_distinct_union_keeps_the_dedup(duck, t):
    from conftest import assert_same

    # SUM is duplicate-SENSITIVE: dropping the union's dedup would double every group.
    ds = bt.from_arrow(t)
    out = ds.union(ds, distinct=True).group_by("a").agg(s=col("b").sum()).collect()
    assert_same(
        out,
        duck.sql("SELECT a, sum(b) AS s FROM (SELECT * FROM t UNION SELECT * FROM t) GROUP BY a"),
    )


def test_count_over_a_distinct_union_keeps_the_dedup(duck, t):
    from conftest import assert_same

    ds = bt.from_arrow(t)
    out = ds.union(ds, distinct=True).group_by("a").agg(c=col("b").count()).collect()
    assert_same(
        out,
        duck.sql("SELECT a, count(b) AS c FROM (SELECT * FROM t UNION SELECT * FROM t) GROUP BY a"),
    )


def test_max_over_a_union_all_is_unchanged(duck, t):
    from conftest import assert_same

    ds = bt.from_arrow(t)
    out = ds.union(ds).group_by("a").agg(m=col("b").max()).collect()
    assert_same(
        out,
        duck.sql(
            "SELECT a, max(b) AS m FROM (SELECT * FROM t UNION ALL SELECT * FROM t) GROUP BY a"
        ),
    )


def test_max_over_a_distinct_union_of_empty(duck, empty):
    from conftest import assert_same

    ds = bt.from_arrow(empty)
    out = ds.union(ds, distinct=True).group_by("a").agg(m=col("b").max()).collect()
    assert_same(
        out,
        duck.sql("SELECT a, max(b) AS m FROM (SELECT * FROM t UNION SELECT * FROM t) GROUP BY a"),
    )


# --- the union's dedup on a semi/anti join's build side --------------------------


def test_semi_join_against_a_distinct_union(duck, t):
    from conftest import assert_same

    ds = bt.from_arrow(t)
    out = ds.join(ds.union(ds, distinct=True), on="a", how="semi").collect()
    assert_same(
        out,
        duck.sql(
            "SELECT * FROM t WHERE EXISTS "
            "(SELECT 1 FROM (SELECT * FROM t UNION SELECT * FROM t) u WHERE u.a = t.a)"
        ),
    )


def test_anti_join_against_a_distinct_union(duck, t):
    from conftest import assert_same

    ds = bt.from_arrow(t)
    other = bt.from_arrow(pa.table({"a": [1, 1, 9], "b": [10, 10, 90]}))
    duck.register("u", pa.table({"a": [1, 1, 9], "b": [10, 10, 90]}))
    out = ds.join(other.union(other, distinct=True), on="a", how="anti").collect()
    assert_same(
        out,
        duck.sql(
            "SELECT * FROM t WHERE NOT EXISTS "
            "(SELECT 1 FROM (SELECT * FROM u UNION SELECT * FROM u) v WHERE v.a = t.a)"
        ),
    )


def test_inner_join_against_a_distinct_union_keeps_the_dedup(duck, t):
    from conftest import assert_same

    # An inner join emits one row per matching right row — the dedup changes the row count.
    ds = bt.from_arrow(t)
    other = bt.from_arrow(pa.table({"a": [1, 2], "c": [100, 200]}))
    duck.register("u", pa.table({"a": [1, 2], "c": [100, 200]}))
    out = (
        ds.join(other.union(other, distinct=True), on="a", how="inner")
        .select("a", "b", "c")
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT t.a, t.b, v.c FROM t JOIN (SELECT * FROM u UNION SELECT * FROM u) v "
            "ON t.a = v.a"
        ),
    )
