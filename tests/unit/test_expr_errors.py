"""Actionable plan-build errors: bad cast dtype and unknown-column suggestions."""

from __future__ import annotations

import pytest

from batcher import col
from batcher._internal.errors import PlanError
from batcher.plan.schema import SchemaRef, suggest_columns

pytestmark = pytest.mark.unit


def test_cast_unknown_dtype_raises_with_suggestion():
    with pytest.raises(PlanError, match="unknown cast dtype"):
        col("x").cast("flot64")
    # The message should suggest the near-miss.
    try:
        col("x").cast("flot64")
    except PlanError as e:
        assert "float64" in str(e)


def test_cast_valid_dtype_ok():
    assert col("x").cast("int64") is not None
    assert col("x").cast("string") is not None


def test_suggest_columns_finds_near_miss():
    assert "did you mean 'salary'" in suggest_columns("salar", ["salary", "name", "id"])
    assert suggest_columns("xyz", ["salary", "name"]) == ""


def test_schema_field_error_includes_suggestion():
    import pyarrow as pa

    ref = SchemaRef.from_arrow(pa.schema([("amount", pa.int64()), ("name", pa.string())]))
    with pytest.raises(KeyError, match="did you mean 'amount'"):
        ref.field("amont")


# --- an aggregate outside `agg()` ---------------------------------------------------


@pytest.mark.parametrize("op", ["__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__"])
def test_comparing_an_aggregate_builds_an_expression_not_a_python_bool(op):
    """``col("x").sum() == 6`` must build a predicate, never evaluate to a `bool`.

    `AggExpr` is not an `Expr` subclass; it forwards the operators it supports to `Expr`
    explicitly. The comparisons were missing from that list, so Python fell back to the
    default identity comparison and ``col("x").sum() == 6`` returned the *bool* `False`.
    That is not merely a missing feature: `with_columns` accepted the bool and wrote a
    constant `False` column for a sum that really was 6 — a silently wrong answer that no
    error path could see, because by then there was no aggregate left to reject.
    """
    got = getattr(col("x").sum(), op)(6)
    assert not isinstance(got, bool), f"{op} collapsed to a Python bool: {got!r}"
    assert hasattr(got, "to_ir"), f"{op} did not build an expression: {type(got).__name__}"


def test_an_aggregate_is_unhashable_for_the_same_reason_an_expression_is():
    """Defining `__eq__` obliges `AggExpr` to refuse hashing, exactly as `Expr` does."""
    with pytest.raises(TypeError, match="not hashable"):
        hash(col("x").sum())


@pytest.mark.parametrize(
    ("build", "want"),
    [
        (lambda ds: ds.select(s=col("x").sum()), {"s": [6]}),
        (lambda ds: ds.with_columns(s=col("x").sum()), {"x": [1, 2, 3], "s": [6, 6, 6]}),
        (lambda ds: ds.filter(col("x") > col("x").mean()), {"x": [3]}),
    ],
    ids=["select", "with_columns", "filter"],
)
def test_an_aggregate_outside_agg_is_the_whole_frame_aggregate(build, want):
    """An aggregate in a row-shaped context now *means* something, and this pins which.

    It used to raise a `PlanError` pointing at `group_by().agg()`. It no longer does:
    an all-aggregate `select` is the whole-frame aggregation (one row), and anywhere
    else the aggregate broadcasts to every row — the reading Polars and pandas give it,
    and the only one under which `with_columns(share=x / x.sum())` has a row per row.

    What must still hold is the property this test was written for: the failure mode
    it guarded against was `TypeError: unsupported literal type: AggExpr` leaking from
    the optimizer, naming neither the call nor the fix. Every case below now answers
    instead, and the two that genuinely cannot be expressed keep a `PlanError` (see
    the tests that follow).
    """
    import batcher as bt

    ds = bt.from_pydict({"x": [1, 2, 3]})
    assert build(ds).to_pydict() == want


def test_an_aggregate_of_an_aggregate_is_still_refused():
    """`sum(sum(x))` has no meaning at any nesting depth, and the refusal says so."""
    import batcher as bt
    from batcher.plan.expr_ir import AggExpr

    ds = bt.from_pydict({"x": [1, 2, 3]})
    nested = AggExpr("sum", AggExpr("sum", col("x")))
    with pytest.raises(PlanError, match="aggregate"):
        ds.select(s=nested).collect()


def test_an_aggregate_comparison_computes_the_right_answer_inside_agg():
    """The predicate the comparison now builds is evaluated, and it is correct."""
    import batcher as bt

    ds = bt.from_pydict({"x": [1, 2, 3]})
    assert ds.agg(r=col("x").sum() == 6).to_pydict() == {"r": [True]}
    assert ds.agg(r=col("x").sum() == 7).to_pydict() == {"r": [False]}
