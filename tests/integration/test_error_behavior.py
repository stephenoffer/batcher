"""Bad input must produce clean, typed errors — never crash the process.

Integer division/modulo by zero in particular used to abort the whole process
via a Cranelift JIT trap (SIGILL); these tests lock in that it raises instead.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


@pytest.fixture
def t():
    return pa.table({"v": [1, 2, 3], "d": [2, 0, 4], "s": ["a", "b", "c"]})


@pytest.mark.parametrize(
    "expr_fn",
    [
        lambda: col("v") % 0,  # integer modulo by literal zero
        lambda: col("v") / 0,  # integer division by literal zero
        lambda: col("v") % col("d"),  # divisor column contains a zero
        lambda: col("v") / col("d"),
    ],
)
def test_integer_divide_by_zero_raises_cleanly(t, expr_fn):
    with pytest.raises(RuntimeError, match="division or modulo by zero"):
        bt.from_arrow(t).select(r=expr_fn()).collect()


def test_valid_integer_division(t):
    out = bt.from_arrow(t).select(r=col("v") / 2).collect()
    assert out.column("r").to_pylist() == [0, 1, 1]  # truncating integer division


def test_float_division_by_zero_is_inf(t):
    # Float division follows IEEE (inf/nan), not an error.
    out = bt.from_arrow(t).select(r=col("v") / 0.0).collect()
    vals = out.column("r").to_pylist()
    assert all(v == float("inf") for v in vals)


def test_sum_of_string_column_raises(t):
    with pytest.raises(RuntimeError, match="sum is not supported"):
        bt.from_arrow(t).group_by().agg(x=col("s").sum()).collect()


def test_non_boolean_filter_predicate_raises(t):
    with pytest.raises(RuntimeError, match="predicate must be boolean"):
        bt.from_arrow(t).filter(col("v") + 1).collect()


def test_unknown_column_raises():
    from batcher._internal.errors import PlanError

    t = pa.table({"a": [1, 2, 3]})
    with pytest.raises(PlanError, match="unknown column"):
        bt.from_arrow(t).select("nope").collect()
