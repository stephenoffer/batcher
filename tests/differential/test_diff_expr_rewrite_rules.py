"""The expression rewrite rules must be semantics-preserving — checked against DuckDB.

`kyber/rules/exprs/` holds roughly fifty rules that rewrite one expression into a cheaper one:
`x = x` to a constant, `x * 0` to zero, an anchored regex to `starts_with`, a reduction pushed
through `unique`. None of them was mentioned by any test, and this is the rule family where being
wrong is worst: an optimizer that changes an answer produces no error, and the cheaper plan is
the one that ships.

The traps are null and NaN. `x = x` is NULL for a null `x`, not TRUE. `x * 0` is NULL, not 0.
`min(unique(x))` equals `min(x)` but `sum(unique(x))` does not equal `sum(x)`. Each case below
pins the semantics DuckDB gives, so a rule that widens past its guard fails here.

All of these passed when written; the file exists so a future rule cannot quietly break them.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

_SELF_COMPARISONS = [
    ("eq", lambda c: c == c, "="),
    ("ne", lambda c: c != c, "<>"),
    ("ge", lambda c: c >= c, ">="),
    ("le", lambda c: c <= c, "<="),
    ("gt", lambda c: c > c, ">"),
    ("lt", lambda c: c < c, "<"),
]


@pytest.fixture
def nullable() -> pa.Table:
    return pa.table(
        {
            "i": pa.array([1, 2, None], pa.int64()),
            "f": pa.array([1.5, float("nan"), None], pa.float64()),
        }
    )


@pytest.mark.parametrize(("name", "build", "op"), _SELF_COMPARISONS)
@pytest.mark.parametrize("column", ["i", "f"])
def test_a_self_comparison_keeps_null_semantics(duck, nullable, name, build, op, column):
    """`x = x` folds to TRUE only where `x` is known: a null row stays null, not true."""
    got = bt.from_arrow(nullable).select(r=build(bt.col(column))).collect()
    duck.register("t", nullable)
    assert_same(got, duck.sql(f"SELECT {column} {op} {column} AS r FROM t"))


@pytest.mark.parametrize(
    ("label", "build", "sql"),
    [
        ("mul by zero", lambda: bt.col("i") * bt.lit(0), "i * 0"),
        ("mod by one", lambda: bt.col("i") % bt.lit(1), "i % 1"),
        ("div by one", lambda: bt.col("i") / bt.lit(1), "i / 1.0"),
    ],
)
def test_an_arithmetic_fold_keeps_null_semantics(duck, nullable, label, build, sql):
    """`x * 0` is NULL for a null `x`; folding it to a literal zero would invent a value."""
    got = bt.from_arrow(nullable).select(r=build()).collect()
    duck.register("t", nullable)
    assert_same(got, duck.sql(f"SELECT {sql} AS r FROM t"))


@pytest.mark.parametrize("column", ["i"])
@pytest.mark.parametrize(
    ("label", "method"), [("is_nan", "is_nan"), ("is_infinite", "is_infinite")]
)
def test_a_nan_check_on_an_integer_column_keeps_nulls(duck, nullable, column, label, method):
    """An integer is never NaN, but a *null* integer is not `False` either."""
    got = bt.from_arrow(nullable).select(r=getattr(bt.col(column), method)()).collect()
    duck.register("t", nullable)
    fn = "isnan" if method == "is_nan" else "isinf"
    assert_same(got, duck.sql(f"SELECT {fn}(CAST({column} AS DOUBLE)) AS r FROM t"))


@pytest.fixture
def strings() -> pa.Table:
    return pa.table(
        {"s": pa.array(["hello", "HELLO", "hel", "", None, "a.c", "xhellox", "hello\nworld"])}
    )


@pytest.mark.parametrize(
    "pattern",
    [
        "^hel",  # -> starts_with
        "^hello",
        "lo$",  # -> ends_with
        "hello$",
        "^hello$",  # -> equality
        "^$",  # the empty string, anchored
        "ell",  # -> contains
        "a.c",  # the dot is a metacharacter, so this must NOT become a literal
        r"a\.c",  # ...and this one must
        "^",
        "$",
        "^HELLO$",
        "hel.*",
        "(?i)^hel",
        "^h.l",
        r"\d",
        "^[a-z]+$",
    ],
)
def test_a_rewritten_regex_matches_duckdb(duck, strings, pattern):
    """The prefix/suffix/anchored rewrites must not change which rows match."""
    got = bt.from_arrow(strings).select(r=bt.col("s").str.regexp_matches(pattern)).collect()
    duck.register("t", strings)
    assert_same(got, duck.sql("SELECT regexp_matches(s, ?) AS r FROM t", params=[pattern]))


# --------------------------------------------------------------------------- #
# List rewrites. DuckDB is the oracle for the reduction semantics, including
# `list_sum` of an empty list being NULL rather than zero.
# --------------------------------------------------------------------------- #
_LISTS = [[3, 1, 2, 1, 3], [5], [], None, [1, None, 2], [7, 7, 7], [-2, -2, 5]]


@pytest.fixture
def lists() -> pa.Table:
    return pa.table({"a": pa.array(_LISTS, pa.list_(pa.int64()))})


@pytest.mark.parametrize(
    ("label", "method", "sql"),
    [
        ("min", "min", "list_min(list_distinct(a))"),
        ("max", "max", "list_max(list_distinct(a))"),
    ],
)
def test_a_set_reading_reduction_survives_being_pushed_through_unique(
    duck, lists, label, method, sql
):
    """`min(unique(x))` is `min(x)`: dedup changes multiplicity, never the set of values."""
    through_unique = (
        bt.from_arrow(lists).select(r=getattr(bt.col("a").list.unique().list, method)()).collect()
    )
    direct = bt.from_arrow(lists).select(r=getattr(bt.col("a").list, method)()).collect()
    assert through_unique.to_pydict() == direct.to_pydict(), (
        f"{label} must be unaffected by a preceding unique()"
    )
    duck.register("t", lists)
    assert_same(through_unique, duck.sql(f"SELECT {sql} AS r FROM t"))


@pytest.mark.parametrize("method", ["sum", "len", "mean"])
def test_a_multiplicity_reading_reduction_is_not_pushed_through_unique(lists, method):
    """`sum(unique(x))` is NOT `sum(x)`, so the rule must leave these alone."""
    through_unique = (
        bt.from_arrow(lists).select(r=getattr(bt.col("a").list.unique().list, method)()).to_pydict()
    )
    direct = bt.from_arrow(lists).select(r=getattr(bt.col("a").list, method)()).to_pydict()
    # The duplicate-bearing rows must differ; if they match, a reduction was pushed through.
    assert through_unique != direct, (
        f"{method}(unique(x)) equals {method}(x), so the dedup was dropped or pushed through"
    )


@pytest.mark.parametrize("method", ["sum", "min", "max", "len"])
def test_a_reduction_is_unchanged_by_a_preceding_reverse(lists, method):
    """`reverse` is a permutation, so every reduction is invariant under it."""
    reduce = getattr(bt.col("a").list.reverse().list, method)
    reversed_first = bt.from_arrow(lists).select(r=reduce()).to_pydict()
    direct = bt.from_arrow(lists).select(r=getattr(bt.col("a").list, method)()).to_pydict()
    assert reversed_first == direct


def test_reverse_is_an_involution(lists) -> None:
    got = bt.from_arrow(lists).select(r=bt.col("a").list.reverse().list.reverse()).to_pydict()["r"]
    assert got == _LISTS


def test_unique_is_idempotent_and_keeps_first_seen_order(lists) -> None:
    once = bt.from_arrow(lists).select(r=bt.col("a").list.unique()).to_pydict()["r"]
    twice = bt.from_arrow(lists).select(r=bt.col("a").list.unique().list.unique()).to_pydict()["r"]
    assert once == twice
    assert once[0] == [3, 1, 2], "first-seen order, as documented"


def test_a_full_slice_is_the_input(lists) -> None:
    assert bt.from_arrow(lists).select(r=bt.col("a").list.slice(0)).to_pydict()["r"] == _LISTS


def test_a_slice_of_a_slice_composes(lists) -> None:
    got = bt.from_arrow(lists).select(r=bt.col("a").list.slice(1, 3).list.slice(1, 2)).to_pydict()
    want = [None if v is None else v[1:4][1:3] for v in _LISTS]
    assert got["r"] == want


def test_a_get_past_the_end_is_null(lists) -> None:
    assert bt.from_arrow(lists).select(r=bt.col("a").list.get(99)).to_pydict()["r"] == [None] * 7


def test_a_field_of_a_constructed_struct_is_the_input() -> None:
    ds = bt.from_pydict({"x": [1, 2, None], "y": ["a", "b", None]})
    got = ds.select(r=bt.struct(x=bt.col("x"), y=bt.col("y")).struct.field("x")).to_pydict()
    assert got["r"] == [1, 2, None]
