"""A column-*name* argument handed an expression says so, instead of a message about `&`.

Twenty-five public methods take a column name as a string. Every one of them tested
`value not in ds.columns`, which evaluates `Expr.__eq__` against each name, builds an
expression, and asks it for a truth value — so `ds.sum(col("v"))`, the spelling a caller
arriving from Polars reaches for, answered with *the truth value of an Expr is ambiguous;
use & | ~ to combine predicates*: a message about boolean operators, naming neither the
method nor the argument, for a call that used none.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.plan.schema import column_name

#: The old message, which must no longer reach a caller through any of these methods.
_LEAKED = "truth value of an Expr is ambiguous"


@pytest.fixture
def ds():
    return bt.from_pydict({"ts": [1, 2, 3], "v": [1.0, 2.0, 3.0], "xs": [[1], [2], [3]]})


@pytest.mark.unit
def test_column_name_passes_a_string_through():
    assert column_name("ts", arg="time_col", api="with_watermark") == "ts"


@pytest.mark.unit
def test_column_name_names_the_column_it_can_recover():
    with pytest.raises(PlanError) as excinfo:
        column_name(bt.col("ts"), arg="time_col", api="with_watermark")
    message = str(excinfo.value)
    assert "with_watermark(): time_col must be a column name" in message
    assert "pass 'ts'" in message, "the recoverable name is the actionable half"
    assert _LEAKED not in message


@pytest.mark.unit
def test_column_name_degrades_on_a_computed_expression():
    """A derived expression has no name to suggest, so it points at `with_columns`."""
    with pytest.raises(PlanError) as excinfo:
        column_name(bt.col("v") + 1, arg="column", api="explode")
    message = str(excinfo.value)
    assert "pass the column's name" in message
    assert "with_columns" in message
    assert _LEAKED not in message


@pytest.mark.unit
def test_column_name_rejects_a_non_string_non_expression():
    with pytest.raises(PlanError, match="must be a column name \\(str\\), not int"):
        column_name(5, arg="column", api="sum")


# Every public method whose first argument is a column name, and the call that exercises it.
_CALLS = {
    "sum": lambda d, c: d.sum(c),
    "mean": lambda d, c: d.mean(c),
    "median": lambda d, c: d.median(c),
    "min": lambda d, c: d.min(c),
    "max": lambda d, c: d.max(c),
    "std": lambda d, c: d.std(c),
    "var": lambda d, c: d.var(c),
    "count_distinct": lambda d, c: d.count_distinct(c),
    "approx_count_distinct": lambda d, c: d.approx_count_distinct(c),
    "has_nulls": lambda d, c: d.has_nulls(c),
    "n_null": lambda d, c: d.n_null(c),
    "mode": lambda d, c: d.mode(c),
    "product": lambda d, c: d.product(c),
    "skew": lambda d, c: d.skew(c),
    "kurtosis": lambda d, c: d.kurtosis(c),
    "mad": lambda d, c: d.mad(c),
    "corr": lambda d, c: d.corr(c, "v"),
    "cov": lambda d, c: d.cov(c, "v"),
    "explode": lambda d, c: d.explode(c),
    "with_row_index": lambda d, c: d.with_row_index(c),
    "with_watermark": lambda d, c: d.with_watermark(c, "1 second"),
    "session_window": lambda d, c: d.session_window(c, "5m", n=bt.col("v").sum()),
}


@pytest.mark.unit
@pytest.mark.parametrize("api", sorted(_CALLS))
def test_every_column_name_argument_refuses_an_expression_clearly(ds, api):
    """The error names the method and the argument, and never leaks the `&`-operator text."""
    with pytest.raises(PlanError) as excinfo:
        _CALLS[api](ds, bt.col("ts"))
    message = str(excinfo.value)
    assert _LEAKED not in message, f"{api} still leaks the boolean-operator message"
    assert "must be a column name" in message
    assert api in message, f"{api} does not name itself in its own error"


#: `with_row_index` names a *new* column, so "ts" would collide.
_VALID_NAME = {"with_row_index": "idx"}


@pytest.mark.unit
@pytest.mark.parametrize("api", sorted(_CALLS))
def test_the_string_spelling_still_works(ds, api):
    """The guard is a type check on the way in, not a behaviour change."""
    _CALLS[api](ds, _VALID_NAME.get(api, "ts"))
