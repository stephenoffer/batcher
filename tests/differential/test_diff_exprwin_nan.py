"""NaN handling for the window-adjacent Expr methods (`clip`, `cut`).

The engine compares floats by a *total order* in which NaN ranks above every finite
value. That total order is correct for sort/group keys, but two value-transforming
methods must not inherit it: ``clip`` and ``cut`` both leave NaN untouched in
Polars/pandas (the reference), so their Python lowering carries an explicit NaN guard.
Regression cover for two wrong-answer bugs where the guard was missing:

* ``clip(lo, hi)`` pulled NaN down to ``hi`` (NaN > hi is true under the total order);
* ``cut(breaks)`` labelled NaN as the top bin (NaN > every break) instead of null.

Expected values below match Polars 1.36 and pandas.
"""

from __future__ import annotations

import math

import batcher as bt


def _nan_positions(values: list) -> list[bool]:
    return [isinstance(v, float) and math.isnan(v) for v in values]


def test_clip_leaves_nan_untouched() -> None:
    x = [1.0, 5.0, 10.0, float("nan"), None, float("inf"), float("-inf")]
    got = bt.from_pydict({"x": x}).select(r=bt.col("x").clip(2.0, 8.0)).to_pydict()["r"]
    # The finite values clamp into [2, 8]; +/-inf clamp to the bounds; NULL stays NULL;
    # NaN stays NaN (NOT pulled to the upper bound).
    assert got[0] == 2.0
    assert got[1] == 5.0
    assert got[2] == 8.0
    assert math.isnan(got[3])  # was wrongly 8.0 before the fix
    assert got[4] is None
    assert got[5] == 8.0
    assert got[6] == 2.0


def test_clip_single_bound_leaves_nan_untouched() -> None:
    x = [float("nan"), 1.0, 5.0]
    lower_only = bt.from_pydict({"x": x}).select(r=bt.col("x").clip(2.0)).to_pydict()["r"]
    upper_only = bt.from_pydict({"x": x}).select(r=bt.col("x").clip(None, 3.0)).to_pydict()["r"]
    assert math.isnan(lower_only[0]) and lower_only[1:] == [2.0, 5.0]
    assert math.isnan(upper_only[0]) and upper_only[1:] == [1.0, 3.0]


def test_clip_integer_column_unaffected() -> None:
    got = bt.from_pydict({"x": [1, 5, 10, None]}).select(r=bt.col("x").clip(2, 8)).to_pydict()["r"]
    assert got == [2, 5, 8, None]


def test_cut_maps_nan_to_null_bin() -> None:
    x = [1.0, 5.0, 10.0, float("nan"), None, float("inf"), float("-inf")]
    got = bt.from_pydict({"x": x}).select(r=bt.col("x").cut([2.0, 8.0])).to_pydict()["r"]
    # NaN and NULL both yield a null bin; +/-inf fall in the outer bins.
    assert got == [
        "(-inf, 2]",
        "(2, 8]",
        "(8, inf]",
        None,  # NaN -> null bin (was wrongly "(8, inf]" before the fix)
        None,  # NULL -> null bin
        "(8, inf]",
        "(-inf, 2]",
    ]


def test_cut_integer_column_unaffected() -> None:
    got = (
        bt.from_pydict({"x": [1, 5, 10, None]})
        .select(r=bt.col("x").cut([2.0, 8.0]))
        .to_pydict()["r"]
    )
    assert got == ["(-inf, 2]", "(2, 8]", "(8, inf]", None]
