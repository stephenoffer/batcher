"""A NumPy scalar is what real data work produces, and it must survive plan lowering.

`arr.max()`, `np.percentile(...)`, a pandas `Series.max()`, the result of any integer
arithmetic on an array — all of these reach the API as filter thresholds constantly. Almost
none of them subclass the Python types the wire encoder dispatches on: `numpy.float64`
subclasses `float` and so worked by accident, while `numpy.int64`, every other integer
width, `numpy.bool_` and a 0-d array did not.

They failed in `to_ir` with a bare `TypeError: unsupported literal type: int64` — and on a
lazy API that means the traceback points at `collect()` rather than at the `filter` that
built the plan.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.unit


@pytest.fixture
def frame():
    return bt.from_arrow(
        pa.table(
            {
                "x": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
                "b": pa.array([True, False, True, False, True]),
                "d": pa.array([dt.date(2020, 1, i + 1) for i in range(5)]),
            }
        )
    )


@pytest.mark.parametrize(
    "threshold",
    [
        np.int8(3),
        np.int16(3),
        np.int32(3),
        np.int64(3),
        np.uint32(3),
        np.uint64(3),
        np.float32(3.0),
        np.float64(3.0),
        np.array(3),  # a 0-d array is a scalar
    ],
)
def test_every_numpy_scalar_width_reaches_execution(frame, threshold):
    """Not merely "does not raise" — the surviving rows must be the right ones."""
    assert frame.filter(bt.col("x") > threshold).to_pydict()["x"] == [4, 5]


def test_a_numpy_bool_compares_against_a_bool_column(frame):
    kept = frame.filter(bt.col("b") == np.bool_(True)).to_pydict()["b"]
    assert kept == [True, True, True]


def test_a_numpy_datetime_becomes_a_date(frame):
    """`.item()` lands `datetime64` on the `date`/`datetime` the encoder already handles."""
    kept = frame.filter(bt.col("d") > np.datetime64("2020-01-03")).to_pydict()["d"]
    assert kept == [dt.date(2020, 1, 4), dt.date(2020, 1, 5)]


def test_the_shapes_this_actually_arrives_as(frame):
    """The call sites that produce these in practice, rather than hand-built scalars."""
    assert frame.filter(bt.col("x") > np.array([1, 2, 3]).max()).to_pydict()["x"] == [4, 5]
    assert frame.filter(bt.col("x") > np.percentile([1, 2, 3, 4, 5], 50)).to_pydict()["x"] == [
        4,
        5,
    ]
    counts = np.array([10, 20, 30])
    assert frame.filter(bt.col("x") == counts.argmax()).to_pydict()["x"] == [2]


def test_the_literal_holds_a_plain_python_value():
    """Normalized at construction, not at lowering, so the value is plain everywhere.

    The plan signature, the plan cache key and every equality comparison read this value;
    leaving a NumPy scalar in place would make those behave by whatever that type's
    `__eq__`/`__hash__` happen to do.
    """
    from batcher.plan.expr_ir import Lit

    for scalar, expected in ((np.int64(7), 7), (np.float32(1.5), 1.5), (np.bool_(True), True)):
        value = Lit(scalar).value
        assert type(value) is type(expected), (scalar, value)
        assert value == expected


def test_a_python_scalar_is_untouched():
    """The safety property: nothing about ordinary literals may change."""
    from batcher.plan.expr_ir import Lit

    for scalar in (7, 1.5, True, "a", dt.date(2020, 1, 1), float("nan")):
        value = Lit(scalar).value
        assert type(value) is type(scalar)
        assert value is scalar or value != value or value == scalar


def test_a_real_array_is_still_rejected():
    """A 1-d array is not a scalar and must not be silently reduced to an element."""
    frame = bt.from_arrow(pa.table({"x": pa.array([1, 2, 3])}))
    with pytest.raises(Exception):  # noqa: B017 - any failure is correct; success is not
        frame.filter(bt.col("x") > np.array([1, 2, 3])).collect()
