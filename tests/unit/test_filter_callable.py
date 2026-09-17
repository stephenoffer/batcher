"""`ds.filter(fn)`: a batch-level Python predicate, without paying for it twice.

The predicate itself is the expensive part and there is nothing to be done about that. What
this pins is everything *around* it: that the surviving rows keep their exact Arrow types
rather than being rebuilt from the callback's format, that dropping rows is declared to the
optimizer as changing no column so a cheap vectorized predicate can still sink below it, and
that an answer which is not a boolean mask is rejected instead of quietly keeping rows.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
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


def _even(batch):
    return pc.equal(pc.bit_wise_and(batch["x"], 1), 0)


# --- the result -------------------------------------------------------------------------


def test_it_keeps_the_rows_the_predicate_accepts():
    assert _ds().filter(_even).to_pydict() == {"x": [2, 4], "y": [20, 40]}


def test_it_keeps_every_row_or_none_without_losing_the_schema():
    kept = _ds().filter(lambda batch: [True] * batch.num_rows)
    dropped = _ds().filter(lambda batch: np.zeros(batch.num_rows, dtype=bool))
    assert kept.to_pydict() == _ds().to_pydict()
    assert dropped.to_pydict() == {"x": [], "y": []}
    assert dropped.schema == kept.schema


def test_an_empty_input_survives_the_stage():
    empty = bt.from_pydict({"x": pa.array([], type=pa.int64())})
    assert empty.filter(lambda batch: [True] * batch.num_rows).to_pydict() == {"x": []}


def test_it_matches_the_equivalent_expression_filter():
    """The whole point is that it is the same relation, only slower to compute."""
    assert _ds().filter(lambda batch: pc.greater(batch["x"], 2)).to_pydict() == (
        _ds().filter(col("x") > 2).to_pydict()
    )


def test_a_null_in_the_mask_drops_the_row_as_sql_does():
    mask = pa.array([True, None, True, None, False])
    assert _ds().filter(lambda batch: mask).to_pydict() == {"x": [1, 3], "y": [10, 30]}


@pytest.mark.parametrize(
    ("batch_format", "predicate"),
    [
        ("pyarrow", lambda b: pc.greater(b["x"], 2)),
        ("numpy", lambda b: b["x"] > 2),
        ("pandas", lambda b: b["x"] > 2),
        ("polars", lambda b: b["x"] > 2),
    ],
)
def test_every_batch_format_hands_over_a_batch_and_takes_back_a_mask(batch_format, predicate):
    if batch_format == "polars":
        pytest.importorskip("polars")
    got = _ds().filter(predicate, batch_format=batch_format).to_pydict()
    assert got == {"x": [3, 4, 5], "y": [30, 40, 50]}


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

    Each of these is a column the round trip genuinely damages. A mask keeps the column it
    was handed, which is why this is its own adapter rather than a `flat_map` returning
    ``[row]``.
    """
    ds = bt.from_pydict({"c": column})
    masked = ds.filter(lambda b: [True] * len(b), batch_format="pandas").schema
    round_tripped = ds.flat_map(lambda row: [row], output_columns=["c"]).schema.field("c").type
    assert masked.field("c").type == ds.schema.field("c").type
    assert round_tripped != masked.field("c").type


def test_a_null_only_column_keeps_its_declared_type():
    ds = bt.from_pydict({"n": pa.array([None, None, None], type=pa.int64())})
    out = ds.filter(lambda b: [True] * b.num_rows)
    assert out.schema.field("n").type == pa.int64()


# --- what the optimizer is told ---------------------------------------------------------


def test_a_vectorized_filter_still_sinks_below_the_python_one():
    """Dropping rows changes no column, so the cheap predicate can run first.

    Without the `preserves_columns` declaration the vectorized filter would be stranded
    above the Python one and every row would pay for the slow predicate.
    """
    plan = _ds().filter(lambda b: pc.greater(b["x"], 1)).filter(col("y") < 40)._plan
    out = push_filter_through_map_batches(plan, None)
    assert isinstance(out, MapBatches)
    assert isinstance(out.input, Filter)
    optimized = Optimizer().logical_rewrite(plan)
    assert isinstance(optimized, MapBatches)
    assert isinstance(optimized.input, Filter)


def test_pushdown_does_not_change_the_answer():
    plan = _ds().filter(lambda b: pc.greater(b["x"], 1)).filter(col("y") < 40)
    assert plan.to_pydict() == {"x": [2, 3], "y": [20, 30]}


def test_input_columns_is_carried_to_the_stage():
    plan = _ds().filter(_even, input_columns=["x"])._plan
    assert plan.input_columns == ("x",)


# --- an answer that is not a mask -------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        pytest.param(lambda b: b, "boolean", id="the-batch"),
        pytest.param(lambda b: {"x": True}, "boolean", id="a-dict"),
        pytest.param(lambda b: None, "not a boolean column", id="a-missing-return"),
        pytest.param(lambda b: True, "not a boolean column", id="a-scalar"),
        pytest.param(lambda b: b["x"], "must return a boolean mask", id="an-int-column"),
        pytest.param(lambda b: [True], "exactly one value per row", id="too-short"),
    ],
)
def test_an_answer_that_is_not_a_boolean_mask_is_rejected(answer, message):
    """A truthy integer column would keep rows for a reason nobody wrote, so nothing is coerced."""
    with pytest.raises(PlanError, match=message):
        _ds().filter(answer).to_pydict()


def test_a_non_callable_non_expression_is_rejected_at_plan_time():
    with pytest.raises(PlanError, match="takes an expression"):
        _ds().filter(42)


def test_a_callable_cannot_be_mixed_with_another_predicate():
    with pytest.raises(PlanError, match="one callable predicate on its own"):
        _ds().filter(_even, col("x") > 1)
    with pytest.raises(PlanError, match="one callable predicate on its own"):
        _ds().filter(_even, x=1)


def test_a_callable_option_without_a_callable_is_refused():
    with pytest.raises(PlanError, match=r"\['batch_size', 'num_gpus'\]"):
        _ds().filter(col("x") > 1, batch_size=2, num_gpus=1)


# --- classes and bound arguments --------------------------------------------------------


class _Above:
    """A predicate class built once per worker, with constructor and call arguments."""

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, batch, *, scale=1):
        return pc.greater(pc.multiply(batch["x"], scale), self.threshold)


def test_a_class_predicate_takes_its_constructor_and_call_arguments():
    out = _ds().filter(_Above, fn_constructor_args=(15,), fn_kwargs={"scale": 10})
    assert out.to_pydict() == {"x": [2, 3, 4, 5], "y": [20, 30, 40, 50]}
    assert isinstance(out._plan.fn, type)  # still built once per worker


def test_constructor_arguments_need_a_class():
    with pytest.raises(PlanError, match="only apply to a class fn"):
        _ds().filter(_even, fn_constructor_args=(1,))


# --- async ------------------------------------------------------------------------------


def test_an_async_predicate_is_awaited_per_batch():
    async def keep_even(batch):
        return _even(batch)

    assert _ds().filter(keep_even).to_pydict() == {"x": [2, 4], "y": [20, 40]}


def test_a_negative_await_bound_is_rejected():
    async def keep(batch):
        return [True] * batch.num_rows

    with pytest.raises(PlanError, match="max_concurrency"):
        _ds().filter(keep, max_concurrency=-1)


# --- dirty data -------------------------------------------------------------------------


def _parse(batch):
    return pa.array([int(v) > 1 for v in batch["s"].to_pylist()])


def test_a_raising_predicate_can_drop_rows_within_a_budget():
    """One malformed record should not end a six-hour job: the failing batch is bisected down
    to the row that raises, and that row is dropped within the budget."""
    ds = bt.from_pydict({"s": ["1", "2", "oops", "4"]})
    assert ds.filter(_parse, max_errored_rows=10).to_pydict() == {"s": ["2", "4"]}


def test_the_default_is_still_strict():
    ds = bt.from_pydict({"s": ["1", "oops"]})
    with pytest.raises(ValueError):
        ds.filter(_parse).to_pydict()


def test_the_budget_is_a_bound_not_a_licence():
    """Past the allowance the error propagates, so a real bug on clean data still fails."""
    ds = bt.from_pydict({"s": ["a", "b", "c", "d"]})
    with pytest.raises(ValueError):
        ds.filter(_parse, max_errored_rows=1).to_pydict()


# --- what the predicate is handed -------------------------------------------------------


def test_a_declared_predicate_receives_only_the_columns_it_declared():
    """Converting one column of a wide batch rather than all of them is where the cost is.
    Safe here because the output is the input masked, so narrowing what the predicate saw
    cannot drop a column from the result."""
    seen: list[list[str]] = []

    def note(batch):
        seen.append(sorted(batch.schema.names))
        return [True] * batch.num_rows

    out = _ds().filter(note, input_columns=["x"])
    assert out.to_pydict() == _ds().to_pydict()  # every column still comes out
    assert seen and all(cols == ["x"] for cols in seen)  # only the declared one went in


def test_an_undeclared_predicate_still_receives_every_column():
    seen: list[list[str]] = []

    def note(batch):
        seen.append(sorted(batch.schema.names))
        return [True] * batch.num_rows

    _ds().filter(note).collect()
    assert seen and all(cols == ["x", "y"] for cols in seen)


def test_reading_an_undeclared_column_says_what_happened():
    """A bare `KeyError` from inside the lambda points at the lambda, not at the declaration
    several lines above that removed the column."""
    with pytest.raises(PlanError, match=r"not in its declared input_columns"):
        _ds().filter(lambda b: pc.greater(b["y"], 1), input_columns=["x"]).collect()


def test_zero_copy_batch_false_hands_over_a_writable_batch():
    def mutate(batch):
        batch["x"][:] = 0  # raises on a read-only zero-copy view
        return batch["x"] == 0

    got = _ds().filter(mutate, batch_format="numpy", zero_copy_batch=False).to_pydict()
    assert got == _ds().to_pydict()
