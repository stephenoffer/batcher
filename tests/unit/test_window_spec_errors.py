"""`window(functions=...)` rejects a malformed spec with a typed, actionable error.

The spec vocabulary is small and irregular — a bare name for a ranking function, a
`(func, column)` pair for an aggregate, `(func, column, offset)` for `lag`/`lead`, and
`("ntile", n)` where the second element is a *count* rather than a column. That last one is
the odd member, and it was the one with no validation: `int(spec[1])` was handed whatever
arrived, and an `Expr` answered it through `__trunc__`, so the natural mistake of writing
`("ntile", col("v"))` by analogy with every other function surfaced as
``TypeError: __trunc__ returned non-Integral (type MathExpr)`` — a Python internal naming
neither the function nor the argument, from the public API.

These are unit tests rather than differential ones because there is no result to compare:
the contract is which exception, carrying which words.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_arrow(
        pa.table({"v": pa.array([1, 2, 3, 4], pa.int64()), "o": pa.array([1, 2, 3, 4], pa.int64())})
    )


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(("ntile", bt.col("v")), id="column"),
        pytest.param(("ntile", "v"), id="column-name"),
        pytest.param(("ntile", None), id="none"),
    ],
)
def test_ntile_rejects_anything_that_is_not_a_count(ds, bad):
    with pytest.raises(PlanError, match="tile"):
        ds.window(order_by=["o"], functions={"x": bad})


def test_ntile_still_accepts_a_count(ds):
    """The control. A validation that rejected everything would satisfy the tests above."""
    result = ds.window(order_by=["o"], functions={"x": ("ntile", 2)}).collect()
    assert result.column("x").to_pylist() == [1, 1, 2, 2]


def test_the_arity_error_still_names_the_spelling(ds):
    with pytest.raises(PlanError, match=r"\('ntile', n\)"):
        ds.window(order_by=["o"], functions={"x": ("ntile", 2, 3)})
