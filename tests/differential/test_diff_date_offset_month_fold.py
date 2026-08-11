"""Folding `<date literal> + interval N month` must not change what the engine computes.

`kyber.rules.extra.temporal_folds.fold_date_offset` evaluates a month shift over a literal
at plan time instead of leaving it to the per-row kernel. The engine shifts months with
chrono's `checked_add_months`, which **clamps** to the last day of the target month when the
source day does not exist there — so folding is only exact where that clamp cannot fire, and
the rule refuses every other case.

These tests pin both halves of that:

- the folded plan agrees with **DuckDB**, and
- it agrees with the engine's **own unfolded kernel**, reached by putting the same offset on
  a column instead of a literal. That second oracle is the one that matters, because it is
  the thing the fold is replacing — DuckDB agreeing does not by itself prove Batcher's two
  paths agree with each other.

The refusal cases (day 29/30/31) are as important as the folded ones: they are where Python's
"keep the day" arithmetic would silently disagree with the engine, and the test asserts the
answer is still the engine's.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher.kyber.rules.extra.temporal_folds import fold_date_offset
from batcher.plan.expr_ir import Lit
from batcher.plan.expr_ir.func_nodes import DateOffset

# Sources whose day-of-month is at or below 28 (the fold applies) and above it (it must not).
_FOLDABLE_DAYS = [1, 15, 28]
_CLAMPING_DAYS = [29, 30, 31]
# Month shifts that cross a year boundary, land on February, and go backwards.
_SHIFTS = [1, 2, 12, -1, -12, 13]


@pytest.mark.differential
@pytest.mark.parametrize("day", _FOLDABLE_DAYS + _CLAMPING_DAYS)
@pytest.mark.parametrize("months", _SHIFTS)
def test_folded_month_offset_equals_the_engines_own_kernel(duck, day, months):
    """The literal (foldable) and column (never folded) paths must agree, and match DuckDB.

    A source day of 31 shifted onto a 30-day month is exactly where a naive fold would
    produce the 31st of the next month and the engine produces the 30th, so these cases are
    what would catch a fold that ignored the clamp.
    """
    base = dt.date(2024, 1, day)  # 2024 is a leap year: Feb has 29 days, not 28
    table = pa.table({"d": pa.array([base], pa.date32()), "k": pa.array([1], pa.int64())})
    duck.register("t", table)
    ds = bt.from_arrow(table)

    # Literal input: the shape `fold_date_offset` may rewrite.
    from_literal = ds.select(out=bt.lit(base).dt.offset_by(f"{months}mo")).collect()
    # Column input: the same offset the fold can never reach, so this is the engine's kernel.
    from_column = ds.select(out=bt.col("d").dt.offset_by(f"{months}mo")).collect()
    assert from_literal.to_pydict() == from_column.to_pydict(), (
        f"folded literal disagrees with the engine's own kernel for day={day} months={months}"
    )

    # DuckDB types `DATE + INTERVAL` as TIMESTAMP; cast back so the comparison is about the
    # value and not that SQL convention (Batcher's offset is type-preserving, as documented).
    expected = duck.sql(
        f"SELECT CAST(DATE '{base}' + INTERVAL '{months}' MONTH AS DATE) AS out FROM t"
    )
    assert_same(from_literal, expected)


@pytest.mark.differential
@pytest.mark.parametrize("day", _CLAMPING_DAYS)
def test_the_rule_refuses_a_day_that_could_clamp(day):
    """A unit-level guard on the refusal itself, so a future edit cannot widen it silently.

    The differential test above would also catch a wrong answer, but only where a clamp
    actually fires; this asserts the rule declines the whole class.
    """
    expr = DateOffset(Lit(dt.date(2024, 1, day)), 1, 0, 0)
    assert fold_date_offset(expr) is expr, f"day {day} can clamp and must not be folded"


@pytest.mark.differential
@pytest.mark.parametrize("day", _FOLDABLE_DAYS)
def test_the_rule_folds_a_day_that_cannot_clamp(day):
    """The fold must actually fire, or the optimization is inert and the tests prove nothing."""
    expr = DateOffset(Lit(dt.date(2024, 1, day)), 12, 0, 0)
    folded = fold_date_offset(expr)
    assert isinstance(folded, Lit), f"day {day} cannot clamp and should have folded"
    assert folded.value == dt.date(2025, 1, day)


@pytest.mark.differential
def test_a_month_offset_mixed_with_days_is_left_alone():
    """Mixing months with days would need the engine's application order; the rule declines."""
    expr = DateOffset(Lit(dt.date(2024, 1, 1)), 1, 5, 0)
    assert fold_date_offset(expr) is expr


@pytest.mark.differential
def test_the_tpch_q20_range_predicate_matches_duckdb(duck):
    """The shape this was written for: a range whose upper bound is `+ interval '1' year`.

    Rows sit either side of both bounds and exactly on them, so an off-by-one in the folded
    constant changes the row count.
    """
    days = [
        dt.date(1993, 12, 31),
        dt.date(1994, 1, 1),
        dt.date(1994, 6, 15),
        dt.date(1994, 12, 31),
        dt.date(1995, 1, 1),
        dt.date(1995, 6, 1),
    ]
    table = pa.table(
        {"l_shipdate": pa.array(days, pa.date32()), "q": pa.array(range(len(days)), pa.int64())}
    )
    duck.register("li", table)
    ds = bt.from_arrow(table)
    got = ds.filter(
        (bt.col("l_shipdate") >= bt.lit(dt.date(1994, 1, 1)))
        & (bt.col("l_shipdate") < bt.lit(dt.date(1994, 1, 1)).dt.offset_by("12mo"))
    ).collect()
    expected = duck.sql(
        "SELECT * FROM li WHERE l_shipdate >= DATE '1994-01-01' "
        "AND l_shipdate < DATE '1994-01-01' + INTERVAL '12' MONTH"
    )
    assert_same(got, expected)
    # Three of the six rows fall inside the year; a fold that dropped or widened the bound
    # would not produce exactly those.
    assert got.num_rows == 3
