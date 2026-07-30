"""Plan-shape and does-not-fire tests for the `CAST(ts AS DATE)` range family.

Six rules, one per comparison, turning `cast(ts, date) <op> DATE 'd'` into a bound on the raw
timestamp so zone maps, bloom filters, and source predicate pushdown can act on it. The
correctness of the *boundary* is the whole rule, so what these pin is which instant each
operator lands on and — the load-bearing part — that a timezone-aware column is refused,
because there the day boundary is a local midnight rather than a fixed instant.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, lit
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra.temporal_date_cast import (
    DATE_CAST_RANGE_RULES,
    rewrite_date_cast_range,
)
from batcher.plan.expr_ir import Binary, Col, Lit

_DAY = dt.date(2024, 1, 1)
_LO = dt.datetime(2024, 1, 1)
_HI = dt.datetime(2024, 1, 2)


def _plan(pred):
    """A Filter over a naive timestamp `ts`, a UTC-aware `tz`, and a date column `d`."""
    table = pa.table(
        {
            "ts": pa.array([dt.datetime(2024, 1, 1, 5)], pa.timestamp("us")),
            "tz": pa.array([dt.datetime(2024, 1, 1, 5)], pa.timestamp("us", tz="UTC")),
            "d": pa.array([dt.date(2024, 1, 1)], pa.date32()),
            "n": pa.array([1], pa.int64()),
        }
    )
    return bt.from_arrow(table).filter(pred)._plan


def _fire(op: str, pred):
    """The rewritten predicate's IR, or ``None`` when the rule declined."""
    out = rewrite_date_cast_range(_plan(pred), op)
    return None if out is None else out.predicate.to_ir()


def _ge(value):
    return Binary("ge", Col("ts"), Lit(value))


def _lt(value):
    return Binary("lt", Col("ts"), Lit(value))


# --- registration -----------------------------------------------------------


def test_one_rule_per_comparison_is_registered():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert {f"cast_date_{op}_to_range" for op in ("eq", "ne", "lt", "le", "gt", "ge")} <= names
    assert len(DATE_CAST_RANGE_RULES) == 6


def test_each_rule_declares_both_spellings_of_its_comparison():
    # A predicate written `DATE 'd' < ts::date` arrives mirrored, and `_match_date_cast`
    # normalizes it — so a rule naming only its own operator would miss half of them.
    mirror = {"eq": "eq", "ne": "ne", "lt": "gt", "le": "ge", "gt": "lt", "ge": "le"}
    for rule_obj in DATE_CAST_RANGE_RULES:
        op = rule_obj.name.split("_")[2]
        assert rule_obj.expr_ops == frozenset({op, mirror[op]}), rule_obj.name


# --- the band each comparison lands on --------------------------------------


def test_equality_becomes_the_half_open_day():
    assert (
        _fire("eq", col("ts").cast("date") == lit(_DAY))
        == Binary("and", _ge(_LO), _lt(_HI)).to_ir()
    )


def test_inequality_becomes_the_complement_of_the_day():
    assert (
        _fire("ne", col("ts").cast("date") != lit(_DAY))
        == Binary("or", _lt(_LO), Binary("ge", Col("ts"), Lit(_HI))).to_ir()
    )


@pytest.mark.parametrize(
    ("op", "pred", "want"),
    [
        # Strictly before the day means before its first instant.
        ("lt", lambda: col("ts").cast("date") < lit(_DAY), lambda: _lt(_LO)),
        # On or before the day means before the *next* day's first instant.
        ("le", lambda: col("ts").cast("date") <= lit(_DAY), lambda: _lt(_HI)),
        # Strictly after the day means at or after the next day's first instant.
        (
            "gt",
            lambda: col("ts").cast("date") > lit(_DAY),
            lambda: Binary("ge", Col("ts"), Lit(_HI)),
        ),
        # On or after the day means at or after its first instant.
        ("ge", lambda: col("ts").cast("date") >= lit(_DAY), lambda: _ge(_LO)),
    ],
)
def test_ordered_comparison_lands_on_the_right_end_of_the_band(op, pred, want):
    assert _fire(op, pred()) == want().to_ir()


def test_literal_written_first_is_mirrored():
    # `DATE 'd' < ts::date` is `ts::date > d`, which is the `gt` rule.
    assert (
        _fire("gt", lit(_DAY) < col("ts").cast("date")) == Binary("ge", Col("ts"), Lit(_HI)).to_ir()
    )


def test_date32_spelling_is_matched_too():
    # `plan.types.CAST_DTYPES` accepts both names for the same Arrow type.
    assert _fire("ge", col("ts").cast("date32") >= lit(_DAY)) == _ge(_LO).to_ir()


def test_rewrite_is_idempotent():
    once = rewrite_date_cast_range(_plan(col("ts").cast("date") >= lit(_DAY)), "ge")
    assert rewrite_date_cast_range(once, "ge") is None


def test_rewrite_reaches_a_conjunct_inside_a_larger_predicate():
    pred = (col("ts").cast("date") >= lit(_DAY)) & (col("n") > 0)
    out = rewrite_date_cast_range(_plan(pred), "ge")
    assert out is not None
    assert out.predicate.to_ir() == Binary("and", _ge(_LO), Binary("gt", Col("n"), Lit(0))).to_ir()


# --- the guards -------------------------------------------------------------


def test_timezone_aware_column_is_refused():
    # The day boundary is a *local* midnight there, which is not a fixed instant across a DST
    # transition — so no pair of naive bounds names it. This is the rule's whole soundness
    # argument and the one guard that cannot be relaxed.
    assert _fire("eq", col("tz").cast("date") == lit(_DAY)) is None


def test_date_column_is_refused():
    # The cast is a no-op on a date column; `drop_self_cast_in_filter` removes it, and turning
    # it into a timestamp band here would change the comparison's type.
    assert _fire("eq", col("d").cast("date") == lit(_DAY)) is None


def test_datetime_literal_is_refused():
    # A `datetime` compared against a date-typed cast is a different comparison (one side is
    # coerced), so this rule claims nothing about it.
    assert _fire("eq", col("ts").cast("date") == lit(dt.datetime(2024, 1, 1))) is None


def test_a_non_date_cast_is_refused():
    assert _fire("eq", col("ts").cast("timestamp") == lit(dt.datetime(2024, 1, 1))) is None


def test_a_cast_of_an_expression_is_refused():
    # The rewrite exists to expose a bare column; there is none here.
    assert _fire("eq", (col("ts").cast("date")).cast("date") == lit(_DAY)) is None


def test_each_rule_ignores_the_other_operators():
    assert _fire("eq", col("ts").cast("date") >= lit(_DAY)) is None
    assert _fire("ge", col("ts").cast("date") == lit(_DAY)) is None


def test_unrelated_predicate_is_left_alone():
    assert _fire("eq", col("n") == 1) is None
