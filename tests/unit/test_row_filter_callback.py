"""`ds.ml.filter`: a Python predicate per row, without paying for it twice.

The predicate itself is the expensive part and there is nothing to be done about that. What
this pins is everything *around* it — that the surviving rows keep their exact Arrow types
rather than being rebuilt from Python values, that dropping rows is declared to the optimizer
as changing no column so a cheap vectorized predicate can still sink below it, and that a
predicate returning something that is always truthy is rejected instead of quietly keeping
every row.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher._internal.errors import PlanError
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.rules.projections import push_filter_through_map_batches
from batcher.plan.logical import Filter, MapBatches

pytestmark = pytest.mark.unit


def _ds():
    return bt.from_pydict({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})


# --- the result -------------------------------------------------------------------------


def test_it_keeps_the_rows_the_predicate_accepts():
    assert _ds().ml.filter(lambda row: row["x"] % 2 == 0).to_pydict() == {
        "x": [2, 4],
        "y": [20, 40],
    }


def test_it_keeps_every_row_or_none_without_losing_the_schema():
    kept = _ds().ml.filter(lambda row: True)
    dropped = _ds().ml.filter(lambda row: False)
    assert kept.to_pydict() == _ds().to_pydict()
    assert dropped.to_pydict() == {"x": [], "y": []}
    assert dropped.schema == kept.schema


def test_an_empty_input_survives_the_stage():
    empty = bt.from_pydict({"x": pa.array([], type=pa.int64())})
    assert empty.ml.filter(lambda row: True).to_pydict() == {"x": []}


def test_it_matches_the_equivalent_expression_filter():
    """The whole point is that it is the same relation, only slower to compute."""
    assert _ds().ml.filter(lambda row: row["x"] > 2).to_pydict() == (
        _ds().filter(col("x") > 2).to_pydict()
    )


@pytest.mark.parametrize(
    "column",
    [
        pytest.param(pa.array([Decimal("1.50")] * 3, type=pa.decimal128(5, 2)), id="decimal"),
        pytest.param(pa.array([[1], [2], [3]], type=pa.large_list(pa.int64())), id="large_list"),
        pytest.param(
            pa.array([datetime.datetime(2020, 1, 1)] * 3, type=pa.timestamp("s")), id="timestamp"
        ),
    ],
)
def test_column_types_survive_exactly_rather_than_being_re_inferred(column):
    """The `flat_map` workaround rebuilds the table from row dicts, which re-types it.

    Each of these is a column the round trip genuinely damages — the decimal loses two
    digits of declared precision, the large list narrows to a 32-bit-offset list, and the
    second-resolution timestamp becomes microseconds. A mask keeps the column it was handed,
    which is why this is its own adapter rather than a `flat_map` returning ``[row]``.
    """
    ds = bt.from_pydict({"c": column})
    masked = ds.ml.filter(lambda row: True).schema.field("c").type
    round_tripped = ds.ml.flat_map(lambda row: [row], output_columns=["c"]).schema.field("c").type
    assert masked == ds.schema.field("c").type
    assert round_tripped != masked


def test_a_null_only_column_keeps_its_declared_type():
    ds = bt.from_pydict({"n": pa.array([None, None, None], type=pa.int64())})
    out = ds.ml.filter(lambda row: True)
    assert out.schema.field("n").type == pa.int64()


# --- what the optimizer is told ---------------------------------------------------------


def test_a_vectorized_filter_still_sinks_below_the_python_one():
    """Dropping rows changes no column, so the cheap predicate can run first.

    Without the `preserves_columns` declaration the vectorized filter would be stranded
    above the Python one and every row would pay for the slow predicate.
    """
    plan = _ds().ml.filter(lambda row: row["x"] > 1).filter(col("y") < 40)._plan
    out = push_filter_through_map_batches(plan, None)
    assert isinstance(out, MapBatches)
    assert isinstance(out.input, Filter)
    optimized = Optimizer().logical_rewrite(plan)
    assert isinstance(optimized, MapBatches)
    assert isinstance(optimized.input, Filter)


def test_pushdown_does_not_change_the_answer():
    plan = _ds().ml.filter(lambda row: row["x"] > 1).filter(col("y") < 40)
    assert plan.to_pydict() == {"x": [2, 3], "y": [20, 30]}


def test_input_columns_is_carried_to_the_stage():
    plan = _ds().ml.filter(lambda row: row["x"] > 1, input_columns=["x"])._plan
    assert plan.input_columns == ("x",)


# --- a predicate that is not one --------------------------------------------------------


def test_a_dict_return_is_rejected_rather_than_keeping_every_row():
    with pytest.raises(PlanError, match="must return True or False"):
        _ds().ml.filter(lambda row: row).to_pydict()


def test_a_list_return_is_rejected_too():
    with pytest.raises(PlanError, match="always truthy"):
        _ds().ml.filter(lambda row: [row["x"]]).to_pydict()


def test_a_number_is_accepted_as_a_truth_value():
    """``lambda r: r['x'] % 2`` is terse but unambiguous, and Python agrees with it."""
    assert _ds().ml.filter(lambda row: row["x"] % 2).to_pydict() == {
        "x": [1, 3, 5],
        "y": [10, 30, 50],
    }


def test_a_missing_return_drops_every_row_rather_than_raising():
    def forgot(row):
        row["x"] > 1  # noqa: B015 - deliberately the mistake under test

    assert _ds().ml.filter(forgot).to_pydict() == {"x": [], "y": []}


def test_a_non_callable_is_rejected_at_plan_time():
    with pytest.raises(PlanError):
        _ds().ml.filter("x > 1")


# --- async ------------------------------------------------------------------------------


def test_an_async_predicate_is_awaited_per_row():
    async def keep_even(row):
        return row["x"] % 2 == 0

    assert _ds().ml.filter(keep_even).to_pydict() == {"x": [2, 4], "y": [20, 40]}


def test_a_negative_await_bound_is_rejected():
    async def keep(row):
        return True

    with pytest.raises(PlanError, match="max_concurrency"):
        _ds().ml.filter(keep, max_concurrency=-1)


# --- dirty data -------------------------------------------------------------------------


def _parse(row):
    return int(row["s"]) > 1


def test_a_raising_predicate_can_drop_rows_within_a_budget():
    """One malformed record should not end a six-hour job, and the row surface had no way
    to say so — `max_errored_rows` existed on `map_batches` and nowhere a row callback could
    reach it."""
    ds = bt.from_pydict({"s": ["1", "2", "oops", "4"]})
    assert ds.ml.filter(_parse, max_errored_rows=10).to_pydict() == {"s": ["2", "4"]}


def test_the_default_is_still_strict():
    ds = bt.from_pydict({"s": ["1", "oops"]})
    with pytest.raises(ValueError):
        ds.ml.filter(_parse).to_pydict()


def test_the_budget_is_a_bound_not_a_licence():
    """Past the allowance the error propagates, so a real bug on clean data still fails."""
    ds = bt.from_pydict({"s": ["a", "b", "c", "d"]})
    with pytest.raises(ValueError):
        ds.ml.filter(_parse, max_errored_rows=1).to_pydict()


# --- what the predicate is handed -------------------------------------------------------


def test_a_declared_predicate_receives_only_the_columns_it_declared():
    """Boxing one column of a wide batch rather than all of them is where the cost is —
    measured 12x end to end. Safe here because the output is the input masked, so narrowing
    what the predicate saw cannot drop a column from the result."""
    seen: list[list[str]] = []

    def note(row):
        seen.append(sorted(row))
        return True

    out = _ds().ml.filter(note, input_columns=["x"])
    assert out.to_pydict() == _ds().to_pydict()  # every column still comes out
    assert seen and all(cols == ["x"] for cols in seen)  # only the declared one went in


def test_an_undeclared_predicate_still_receives_every_column():
    seen: list[list[str]] = []

    def note(row):
        seen.append(sorted(row))
        return True

    _ds().ml.filter(note).collect()
    assert seen and all(cols == ["x", "y"] for cols in seen)


def test_reading_an_undeclared_column_says_what_happened():
    """A bare `KeyError` from inside the lambda points at the lambda, not at the declaration
    several lines above that removed the column."""
    with pytest.raises(PlanError, match=r"not in its declared input_columns"):
        _ds().ml.filter(lambda row: row["y"] > 1, input_columns=["x"]).collect()


def test_an_async_predicate_is_narrowed_the_same_way():
    seen: list[list[str]] = []

    async def note(row):
        seen.append(sorted(row))
        return True

    _ds().ml.filter(note, input_columns=["x"]).collect()
    assert seen and all(cols == ["x"] for cols in seen)
