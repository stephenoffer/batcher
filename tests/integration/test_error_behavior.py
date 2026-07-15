"""Bad input must produce clean, typed errors — never crash the process.

Integer division/modulo by zero used to abort the whole process via a Cranelift JIT
trap (SIGILL). It no longer raises OR traps: `/` is true (float) division, so `x / 0`
is IEEE `±inf`, and integer `x % 0` returns NULL — both matching DuckDB (ledger B30).
The point these tests still lock in is *no process abort* on a zero divisor.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


@pytest.fixture
def t():
    return pa.table({"v": [1, 2, 3], "d": [2, 0, 4], "s": ["a", "b", "c"]})


def test_integer_modulo_by_zero_is_null(t):
    # `x % 0` returns NULL for the offending row (DuckDB), never a trap/panic.
    out = bt.from_arrow(t).select(r=col("v") % col("d")).collect()
    assert out.column("r").to_pylist() == [1, None, 3]
    # A literal zero divisor: every row NULL.
    out2 = bt.from_arrow(t).select(r=col("v") % 0).collect()
    assert out2.column("r").to_pylist() == [None, None, None]


def test_division_is_true_division_like_duckdb(t):
    # `/` is true (float) division, matching DuckDB — a zero divisor is IEEE ±inf,
    # not an error and not a trap.
    out = bt.from_arrow(t).select(r=col("v") / 2).collect()
    assert out.column("r").to_pylist() == [0.5, 1.0, 1.5]
    div0 = bt.from_arrow(t).select(r=col("v") / col("d")).collect().column("r").to_pylist()
    assert div0[0] == 0.5 and math.isinf(div0[1]) and div0[2] == 0.75


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
