"""Every cell of the null/NaN semantics page, checked against DuckDB (AP-183).

`docs/user-guide/transform/columns/null-semantics.md` states how a null and a NaN behave in
comparisons, Boolean logic, membership, aggregates, sorts and keys. Each test here is one
row of that page, run on Batcher and on DuckDB over the same data.

Each projection carries the row id `i`, so the order-independent `assert_same` still
compares cell by cell: a multiset of ``(i, answer)`` pairs is equal only if every row's
answer is. The tables go through `duck_materialize`, because DuckDB evaluates a predicate
pushed into a registered Arrow scan with IEEE semantics (NaN never equal) while its own
executor ranks NaN above every number -- the documented behaviour Batcher matches.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered, duck_materialize

pytestmark = pytest.mark.differential

NAN = math.nan

PAIRS = pa.table(
    {
        "i": pa.array([0, 1, 2, 3, 4, 5, 6], pa.int64()),
        "x": pa.array([1.0, None, NAN, NAN, None, 2.0, -0.0], pa.float64()),
        "y": pa.array([1.0, None, NAN, 1.0, 1.0, NAN, 0.0], pa.float64()),
    }
)

BOOLS = pa.table(
    {
        "i": pa.array(range(9), pa.int64()),
        "a": pa.array([True, True, True, False, False, False, None, None, None]),
        "b": pa.array([True, False, None, True, False, None, True, False, None]),
    }
)


@pytest.fixture
def pairs(duck):
    duck_materialize(duck, "pairs", PAIRS)
    return bt.from_arrow(PAIRS)


@pytest.fixture
def bools(duck):
    duck_materialize(duck, "bools", BOOLS)
    return bt.from_arrow(BOOLS)


_COMPARISONS = [
    ("eq", lambda x, y: x == y, "x = y"),
    ("ne", lambda x, y: x != y, "x <> y"),
    ("lt", lambda x, y: x < y, "x < y"),
    ("le", lambda x, y: x <= y, "x <= y"),
    ("gt", lambda x, y: x > y, "x > y"),
    ("ge", lambda x, y: x >= y, "x >= y"),
    ("eq_missing", lambda x, y: x.eq_missing(y), "x IS NOT DISTINCT FROM y"),
    ("gt_huge", lambda x, y: x > 1e308, "x > 1e308"),
    ("eq_none", lambda x, y: x == None, "x = NULL"),  # noqa: E711
    ("lt_none", lambda x, y: x < None, "x < NULL"),
    ("eq_missing_none", lambda x, y: x.eq_missing(None), "x IS NOT DISTINCT FROM NULL"),
]


@pytest.mark.parametrize(("name", "build", "sql"), _COMPARISONS, ids=[c[0] for c in _COMPARISONS])
def test_comparison_cells_match_duckdb(duck, pairs, name, build, sql):
    """Null compares to null, NaN equals itself and outranks every number."""
    out = pairs.select("i", r=build(bt.col("x"), bt.col("y"))).collect()
    assert_same(out, duck.sql(f"SELECT i, {sql} AS r FROM pairs"))


@pytest.mark.parametrize(
    ("build", "sql"),
    [
        (lambda a, b: a & b, "a AND b"),
        (lambda a, b: a | b, "a OR b"),
        (lambda a, b: ~a, "NOT a"),
    ],
    ids=["and", "or", "not"],
)
def test_kleene_logic_matches_duckdb(duck, bools, build, sql):
    """`False & null` is false, `True | null` is true, every other null combination is null."""
    out = bools.select("i", r=build(bt.col("a"), bt.col("b"))).collect()
    assert_same(out, duck.sql(f"SELECT i, {sql} AS r FROM bools"))


@pytest.mark.parametrize(
    ("build", "sql"),
    [
        (lambda x: x.is_in([1.0, None]), "x IN (1.0, NULL)"),
        (lambda x: x.is_in([1.0, NAN]), "x IN (1.0, 'NaN'::DOUBLE)"),
        (
            lambda x: x.is_in([1.0, None], nulls_equal=True),
            "coalesce(x IN (1.0, NULL), x IS NULL, false)",
        ),
    ],
    ids=["null_member", "nan_member", "nulls_equal"],
)
def test_membership_matches_duckdb(duck, pairs, build, sql):
    """A null member turns non-matches null; a NaN member matches NaN; `nulls_equal` is total.

    DuckDB has no `nulls_equal` flag, so that row is checked against its definition spelled
    in SQL: a null input matches the `None` member, and nothing else is ever null.
    """
    out = pairs.select("i", r=build(bt.col("x"))).collect()
    assert_same(out, duck.sql(f"SELECT i, {sql} AS r FROM pairs"))


@pytest.mark.parametrize(
    ("agg", "sql"),
    [
        (lambda x: x.sum(), "sum(x)"),
        (lambda x: x.mean(), "avg(x)"),
        (lambda x: x.max(), "max(x)"),
        (lambda x: x.min(), "min(x)"),
        (lambda x: x.count(), "count(x)"),
    ],
    ids=["sum", "mean", "max", "min", "count"],
)
def test_aggregates_skip_null_and_keep_nan(duck, pairs, agg, sql):
    """Nulls are skipped, NaN propagates through sum/mean and is the max."""
    out = pairs.agg(r=agg(bt.col("x"))).collect()
    assert_same(out, duck.sql(f"SELECT {sql} AS r FROM pairs"))


@pytest.mark.parametrize("descending", [False, True], ids=["asc", "desc"])
def test_sort_places_nan_as_largest_and_nulls_last(duck, pairs, descending):
    """The ordered comparison is the point here, so it uses `assert_same_ordered`."""
    out = pairs.sort("x", descending=descending).select("x").collect()
    direction = "DESC" if descending else "ASC"
    assert_same_ordered(out, duck.sql(f"SELECT x FROM pairs ORDER BY x {direction} NULLS LAST"))


def test_group_keys_collect_every_null_and_every_nan(duck, pairs):
    """One null group and one NaN group; `-0.0` joins the `0.0`-valued group."""
    out = pairs.group_by("x").agg(n=bt.count()).collect()
    assert_same(out, duck.sql("SELECT x, count(*) AS n FROM pairs GROUP BY x"))


def test_distinct_matches_duckdb(duck, pairs):
    out = pairs.select("x").distinct().collect()
    assert_same(out, duck.sql("SELECT DISTINCT x FROM pairs"))


def test_join_matches_nan_and_signed_zero_but_never_null(duck):
    """A NaN key joins NaN and `-0.0` joins `0.0`, as DuckDB's join does; null joins nothing."""
    left = pa.table({"k": pa.array([1.0, None, NAN, -0.0, 5.0])})
    right = pa.table({"k": pa.array([1.0, None, NAN, 0.0, 6.0]), "v": pa.array([1, 2, 3, 4, 5])})
    duck_materialize(duck, "lt", left)
    duck_materialize(duck, "rt", right)
    out = bt.from_arrow(left).join(bt.from_arrow(right), on="k").select("v").collect()
    assert_same(out, duck.sql("SELECT v FROM lt JOIN rt ON lt.k = rt.k"))


def test_empty_and_single_row_inputs(duck):
    """An empty relation and a one-row null still agree on every comparison shape."""
    one = pa.table({"i": pa.array([0], pa.int64()), "x": pa.array([None], pa.float64())})
    duck_materialize(duck, "one", one)
    ds = bt.from_arrow(one)
    for build, sql in [(lambda x: x == None, "x = NULL"), (lambda x: x > NAN, "x > 'NaN'::DOUBLE")]:  # noqa: E711
        assert_same(
            ds.select("i", r=build(bt.col("x"))).collect(),
            duck.sql(f"SELECT i, {sql} AS r FROM one"),
        )
        assert_same(
            ds.filter(bt.col("i") > 5).select("i", r=build(bt.col("x"))).collect(),
            duck.sql(f"SELECT i, {sql} AS r FROM one WHERE i > 5"),
        )
