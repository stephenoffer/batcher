"""The `predicate_algebra.bounds` unions must match DuckDB after the full optimizer runs.

Every case is written as the original disjunction; Batcher collapses it into one
comparison and the oracle evaluates it as written. The fixture carries NULLs (so the
three-valued behaviour of `OR` is exercised) and values inside, on, and outside each
boundary.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.predicate_algebra
from _harness import assert_same
from batcher import col, lit
from batcher.plan.expr_ir import InList


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "x": pa.array([0, 1, 2, 3, 4, 5, 7, 9, 10, None], type=pa.int64()),
            "y": pa.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5], type=pa.int64()),
            "d": pa.array(
                [dt.date(2020, m, 1) for m in range(1, 11)],
                type=pa.date32(),
            ),
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def empty(duck):
    tbl = pa.table(
        {
            "x": pa.array([], type=pa.int64()),
            "y": pa.array([], type=pa.int64()),
            "d": pa.array([], type=pa.date32()),
        }
    )
    duck.register("t", tbl)
    return tbl


_CASES = [
    ("widen_upper", lambda: (col("x") < lit(3)) | (col("x") < lit(7)), "x < 3 OR x < 7"),
    ("widen_upper_mixed", lambda: (col("x") <= lit(3)) | (col("x") < lit(7)), "x <= 3 OR x < 7"),
    (
        "widen_upper_same_bound",
        lambda: (col("x") < lit(3)) | (col("x") <= lit(3)),
        "x < 3 OR x <= 3",
    ),
    ("widen_lower", lambda: (col("x") > lit(7)) | (col("x") >= lit(3)), "x > 7 OR x >= 3"),
    (
        "widen_lower_same_bound",
        lambda: (col("x") > lit(3)) | (col("x") >= lit(3)),
        "x > 3 OR x >= 3",
    ),
    ("close_lower", lambda: (col("x") == lit(3)) | (col("x") > lit(3)), "x = 3 OR x > 3"),
    ("close_upper", lambda: (col("x") == lit(3)) | (col("x") < lit(3)), "x = 3 OR x < 3"),
    ("equality_absorbs", lambda: (col("x") == lit(5)) & (col("x") > lit(1)), "x = 5 AND x > 1"),
    (
        "equality_absorbs_neq",
        lambda: (col("x") == lit(5)) & (col("x") != lit(2)),
        "x = 5 AND x <> 2",
    ),
    (
        "in_list_absorbs_equality",
        lambda: InList(col("x"), (1, 2)) | (col("x") == lit(3)),
        "x IN (1, 2) OR x = 3",
    ),
    (
        "in_lists_merge",
        lambda: InList(col("x"), (1, 2)) | InList(col("x"), (2, 3)),
        "x IN (1, 2) OR x IN (2, 3)",
    ),
    (
        "ranges_overlap",
        lambda: (
            ((col("x") >= lit(1)) & (col("x") <= lit(5)))
            | ((col("x") >= lit(4)) & (col("x") <= lit(9)))
        ),
        "(x >= 1 AND x <= 5) OR (x >= 4 AND x <= 9)",
    ),
    (
        "ranges_touch",
        lambda: (
            ((col("x") >= lit(1)) & (col("x") <= lit(4)))
            | ((col("x") >= lit(4)) & (col("x") <= lit(9)))
        ),
        "(x >= 1 AND x <= 4) OR (x >= 4 AND x <= 9)",
    ),
    (
        "ranges_nested",
        lambda: (
            ((col("x") >= lit(1)) & (col("x") <= lit(9)))
            | ((col("x") >= lit(4)) & (col("x") <= lit(5)))
        ),
        "(x >= 1 AND x <= 9) OR (x >= 4 AND x <= 5)",
    ),
    (
        "ranges_disjoint_untouched",
        lambda: (
            ((col("x") >= lit(1)) & (col("x") <= lit(2)))
            | ((col("x") >= lit(7)) & (col("x") <= lit(9)))
        ),
        "(x >= 1 AND x <= 2) OR (x >= 7 AND x <= 9)",
    ),
    (
        "date_ranges_overlap",
        lambda: (
            ((col("d") >= lit(dt.date(2020, 1, 1))) & (col("d") <= lit(dt.date(2020, 6, 1))))
            | ((col("d") >= lit(dt.date(2020, 5, 1))) & (col("d") <= lit(dt.date(2020, 9, 1))))
        ),
        "(d >= DATE '2020-01-01' AND d <= DATE '2020-06-01') OR "
        "(d >= DATE '2020-05-01' AND d <= DATE '2020-09-01')",
    ),
    (
        "different_columns_untouched",
        lambda: (col("x") < lit(3)) | (col("y") < lit(7)),
        "x < 3 OR y < 7",
    ),
    (
        "opposite_directions_untouched",
        lambda: (col("x") < lit(3)) | (col("x") > lit(7)),
        "x < 3 OR x > 7",
    ),
]


@pytest.mark.parametrize(
    ("expr", "sql"), [(e, s) for _, e, s in _CASES], ids=[n for n, _, _ in _CASES]
)
def test_bound_union_matches_duckdb(duck, t, expr, sql):
    out = bt.from_arrow(t).select(r=expr()).collect()
    assert_same(out, duck.sql(f"SELECT {sql} AS r FROM t"))


@pytest.mark.parametrize(
    ("expr", "sql"), [(e, s) for _, e, s in _CASES], ids=[n for n, _, _ in _CASES]
)
def test_bound_union_matches_duckdb_on_empty_input(duck, empty, expr, sql):
    out = bt.from_arrow(empty).select(r=expr()).collect()
    assert_same(out, duck.sql(f"SELECT {sql} AS r FROM t"))


@pytest.mark.parametrize(
    ("expr", "sql"), [(e, s) for _, e, s in _CASES], ids=[n for n, _, _ in _CASES]
)
def test_bound_union_inside_a_filter_matches_duckdb(duck, t, expr, sql):
    out = bt.from_arrow(t).filter(expr()).select(x=col("x")).collect()
    assert_same(out, duck.sql(f"SELECT x FROM t WHERE {sql}"))


# --- collapse_degenerate_range_to_equality -----------------------------------
#
# `x >= c AND x <= c` becomes `x = c`, so the rows kept must be identical. Written here as the
# range and evaluated by DuckDB as written. The NULL row is what makes this worth checking
# against an oracle rather than by inspection: the range answers `NULL AND NULL` and the
# equality answers `NULL`, and a filter drops both — but only an oracle proves the two
# spellings really do agree on the boundary row and the null one at once.


@pytest.mark.parametrize(
    ("pred", "sql"),
    [
        (lambda: (col("x") >= lit(3)) & (col("x") <= lit(3)), "x >= 3 AND x <= 3"),
        (lambda: (col("x") <= lit(3)) & (col("x") >= lit(3)), "x <= 3 AND x >= 3"),
        # A value not present in the column: the collapse must keep zero rows, not all of them.
        (lambda: (col("x") >= lit(6)) & (col("x") <= lit(6)), "x >= 6 AND x <= 6"),
        # The extremes of the fixture.
        (lambda: (col("x") >= lit(0)) & (col("x") <= lit(0)), "x >= 0 AND x <= 0"),
        (lambda: (col("x") >= lit(10)) & (col("x") <= lit(10)), "x >= 10 AND x <= 10"),
        # A date column, where the equality is against a date literal rather than an int.
        (
            lambda: (col("d") >= lit(dt.date(2020, 5, 1))) & (col("d") <= lit(dt.date(2020, 5, 1))),
            "d >= DATE '2020-05-01' AND d <= DATE '2020-05-01'",
        ),
        # Alongside another conjunct, so the collapse happens inside a larger predicate.
        (
            lambda: (col("x") >= lit(3)) & (col("x") <= lit(3)) & (col("y") == lit(5)),
            "x >= 3 AND x <= 3 AND y = 5",
        ),
        # Inside a disjunction, where the collapsed equality is not the controlling term.
        (
            lambda: ((col("x") >= lit(3)) & (col("x") <= lit(3))) | (col("x") > lit(9)),
            "(x >= 3 AND x <= 3) OR x > 9",
        ),
        # Under a NOT, where turning the range into an equality must not change the negation.
        (lambda: ~((col("x") >= lit(3)) & (col("x") <= lit(3))), "NOT (x >= 3 AND x <= 3)"),
    ],
)
def test_degenerate_range_collapse_matches_duckdb(duck, t, pred, sql):
    out = bt.from_arrow(t).filter(pred()).collect()
    assert_same(out, duck.sql(f"SELECT * FROM t WHERE {sql}"))


def test_degenerate_range_collapse_in_a_projection_matches_duckdb(duck, t):
    # In a projection the three-valued result itself is observable, not just which rows pass —
    # so this is where a NULL turning into FALSE would show up.
    expr = (col("x") >= lit(3)) & (col("x") <= lit(3))
    out = bt.from_arrow(t).select(x=col("x"), r=expr).collect()
    assert_same(out, duck.sql("SELECT x, x >= 3 AND x <= 3 AS r FROM t"))
