"""Scalar math, NaN filling, activations, label encoding and scalers.

Trigonometric and hyperbolic functions, `fill_nan`, `label_encode` and the three scalers
have an exact DuckDB spelling and are compared against it. The activations are ML
spellings with no DuckDB builtin, so each is held to a pure-Python reference of the formula
its docstring states. Every case carries a null row, and runs on one row and on no rows.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered

pytestmark = pytest.mark.differential

_SCHEMA = pa.schema([("i", pa.int64()), ("g", pa.string()), ("x", pa.float64())])


def _table(xs: list, gs: list | None = None) -> pa.Table:
    gs = gs if gs is not None else ["a"] * len(xs)
    return pa.table({"i": list(range(len(xs))), "g": gs, "x": xs}, schema=_SCHEMA)


# Domains differ per function, so each case names its own inputs; every one holds a null.
_UNIT = [-0.9, -0.5, 0.0, 0.25, None, 0.99]
_WIDE = [-3.0, -0.5, 0.0, 1.5, None, 4.0]
_ACOSH = [1.0, 1.5, 2.0, None, 10.0]
_POS = [0.5, 1.0, 2.5, None, 100.0, 200.0]

# (batcher expression factory, DuckDB expression, input values)
_DUCK_CASES = {
    "arcsin": (lambda: bt.col("x").arcsin(), "asin(x)", _UNIT),
    "arctan": (lambda: bt.col("x").arctan(), "atan(x)", _WIDE),
    "arccosh": (lambda: bt.col("x").arccosh(), "acosh(x)", _ACOSH),
    "arctanh": (lambda: bt.col("x").arctanh(), "atanh(x)", _UNIT),
    "tan": (lambda: bt.col("x").tan(), "tan(x)", _WIDE),
    "cosh": (lambda: bt.col("x").cosh(), "cosh(x)", _WIDE),
    # 200 is past where gamma() overflows, which is the reason lgamma exists.
    "lgamma": (lambda: bt.col("x").lgamma(), "lgamma(x)", _POS),
    "fill_nan": (
        lambda: bt.col("x").fill_nan(0.0),
        "CASE WHEN isnan(x) THEN 0.0 ELSE x END",
        [1.0, float("nan"), None, -2.0, float("nan")],
    ),
    "label_encode": (
        lambda: bt.col("x").label_encode(),
        "dense_rank() OVER (ORDER BY x) - 1",
        [3.0, 1.0, None, 3.0, 2.0, -5.0],
    ),
    "maxabs_scale": (
        lambda: bt.col("x").maxabs_scale(partition_by=["g"]),
        "x / max(abs(x)) OVER (PARTITION BY g)",
        [-8.0, 2.0, None, 4.0, 1.0, -0.5],
    ),
    "minmax_scale": (
        lambda: bt.col("x").minmax_scale(partition_by=["g"]),
        "(x - min(x) OVER w) / (max(x) OVER w - min(x) OVER w)",
        [-8.0, 2.0, None, 4.0, 1.0, -0.5],
    ),
    "normalize_l1": (
        lambda: bt.col("x").normalize_l1(partition_by=["g"]),
        "x / sum(abs(x)) OVER (PARTITION BY g)",
        [-8.0, 2.0, None, 4.0, 1.0, -0.5],
    ),
}
_GROUPS = ["a", "a", "a", "b", "b", "b"]


def _run(table: pa.Table, expr) -> pa.Table:
    return bt.from_arrow(table).with_columns(r=expr).sort("i").select("i", "r").to_arrow()


def _duck_query(sql: str) -> str:
    return f"SELECT i, {sql} AS r FROM t WINDOW w AS (PARTITION BY g) ORDER BY i"


@pytest.mark.parametrize("shape", ["full", "one_row", "empty"])
@pytest.mark.parametrize("case", list(_DUCK_CASES))
def test_scalar_and_scaler_expressions_match_duckdb(duck, case, shape):
    make, sql, xs = _DUCK_CASES[case]
    gs = _GROUPS[: len(xs)] if len(xs) == len(_GROUPS) else None
    table = _table(xs, gs)
    if shape == "one_row":
        table = table.slice(0, 1)
    elif shape == "empty":
        table = table.slice(0, 0)
    duck.register("t", table)
    got = _run(table, make())
    assert got.num_rows == table.num_rows
    assert_same_ordered(got, duck.sql(_duck_query(sql)))


def test_label_encode_codes_are_dense_and_value_ordered():
    got = bt.from_pydict({"x": ["pear", "apple", "fig", "apple"]}).select(
        r=bt.col("x").label_encode()
    )
    assert got.to_pydict()["r"] == [2, 0, 1, 0]


def test_minmax_scale_of_a_constant_partition_is_nan_as_documented():
    got = _run(_table([4.0, 4.0]), bt.col("x").minmax_scale()).column("r").to_pylist()
    assert all(math.isnan(v) for v in got), got


def _softplus(x: float) -> float:
    return math.log1p(math.exp(x))


def _logit(x: float) -> float:
    if x == 0.0:
        return -math.inf
    if x == 1.0:
        return math.inf
    return math.log(x / (1.0 - x))


_ACTIVATIONS = {
    "relu": (lambda: bt.col("x").relu(), lambda x: max(x, 0.0), _WIDE),
    "softplus": (lambda: bt.col("x").softplus(), _softplus, _WIDE),
    "hardtanh": (lambda: bt.col("x").hardtanh(), lambda x: min(max(x, -1.0), 1.0), _WIDE),
    "leaky_relu_default": (
        lambda: bt.col("x").leaky_relu(),
        lambda x: x if x > 0 else 0.01 * x,
        _WIDE,
    ),
    "leaky_relu_slope": (
        lambda: bt.col("x").leaky_relu(negative_slope=0.2),
        lambda x: x if x > 0 else 0.2 * x,
        _WIDE,
    ),
    # The docstring's bounds: 0 and 1 map to -inf and +inf.
    "logit": (lambda: bt.col("x").logit(), _logit, [0.0, 0.1, 0.5, None, 0.9, 1.0]),
}


@pytest.mark.parametrize("case", list(_ACTIVATIONS))
def test_activation_matches_its_documented_formula(case):
    make, ref, xs = _ACTIVATIONS[case]
    got = _run(_table(xs), make()).column("r").to_pylist()
    want = [None if x is None else ref(x) for x in xs]
    assert len(got) == len(want)
    for x, b, w in zip(xs, got, want, strict=True):
        if w is None:
            assert b is None, f"null input {x!r} gave {b!r}"
        elif math.isinf(w):
            assert b == w, f"{case}({x}) = {b}, want {w}"
        else:
            assert b == pytest.approx(w, rel=1e-12, abs=1e-15), f"{case}({x}) = {b}, want {w}"


@pytest.mark.parametrize("case", list(_ACTIVATIONS))
def test_activation_on_empty_input_returns_no_rows(case):
    make, _, _ = _ACTIVATIONS[case]
    assert _run(_table([]), make()).num_rows == 0
