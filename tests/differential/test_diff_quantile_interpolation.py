"""`quantile_cont` interpolation must match DuckDB **exactly**, not merely closely.

The engine interpolates between the two bracketing order statistics. There are two algebraically
identical ways to write that, and they are not numerically identical:

* subtractive, ``lo + (hi - lo) * frac``
* convex, ``lo * (1 - frac) + hi * frac``

They differ in the last ulp on roughly a quarter of ordinary inputs, and they differ
catastrophically at the top of the range: ``hi - lo`` overflows to infinity for two large
opposite-signed doubles, so the subtractive form returned ``inf`` for a finite input.

DuckDB uses the convex form. This module is the external check on that claim, because the Rust
unit test beside `quickselect_quantile` was a restatement of the implementation's own formula
and so could not have caught the disagreement.
"""

from __future__ import annotations

import math
import random

import pytest

import batcher as bt

pytestmark = pytest.mark.differential

_QUANTILES = [0.0, 0.05, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 0.95, 1.0]


def _duck_quantile(con, values: list[float], q: float) -> float:
    rows = " union all ".join(f"select {v!r}::DOUBLE as x" for v in values)
    return con.sql(f"select quantile_cont(x, {q}) from ({rows})").fetchone()[0]


def _batcher_quantile(values: list[float], q: float) -> float:
    return bt.from_pydict({"x": list(values)}).agg(r=bt.col("x").quantile(q)).to_pydict()["r"][0]


def test_continuous_quantile_is_bit_identical_to_duckdb():
    """Exact equality, not `approx`. A tolerance here would pass for either formula.

    That is the whole point: the two spellings agree to within any tolerance anyone would write
    and disagree in the last bit, so an approximate comparison reports a property nobody checked.
    """
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    rng = random.Random(20260826)
    mismatches = []
    for _ in range(120):
        n = rng.randint(2, 40)
        values = [round(rng.uniform(-1000.0, 1000.0), 1) for _ in range(n)]
        q = rng.choice(_QUANTILES)
        got = _batcher_quantile(values, q)
        want = _duck_quantile(con, values, q)
        if got != want:
            mismatches.append((q, n, got, want))
    assert not mismatches, f"{len(mismatches)} quantiles differ from DuckDB: {mismatches[:5]}"


@pytest.mark.parametrize("q", [0.25, 0.5, 0.75])
def test_a_finite_input_never_yields_an_infinite_quantile(q):
    """The two extremes of the double range bracket a finite answer.

    ``hi - lo`` is ``3.4e308`` here, which is not representable, so the subtractive form
    produced ``inf`` for every quantile of a two-row table. Nothing about the input is
    exceptional except its magnitude.
    """
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    values = [-1.7e308, 1.7e308]
    got = _batcher_quantile(values, q)
    assert math.isfinite(got), f"quantile({q}) of two finite doubles came back {got}"
    assert got == _duck_quantile(con, values, q)


def test_median_of_two_large_doubles_is_their_midpoint():
    """``(a + b) / 2`` overflows before it halves; the median must not.

    ``median`` reduces to the ``q = 0.5`` interpolation, so this is the same defect seen through
    the aggregate users actually call. DuckDB answers ``1.35e308``.
    """
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    values = [1e308, 1.7e308]
    got = bt.from_pydict({"x": values}).agg(m=bt.col("x").median()).to_pydict()["m"][0]
    assert math.isfinite(got), f"median of two finite doubles came back {got}"
    assert (
        got
        == con.sql(
            "select median(x) from (select 1e308::DOUBLE as x union all select 1.7e308::DOUBLE)"
        ).fetchone()[0]
    )


def test_window_and_list_median_agree_with_the_aggregate():
    """The same value through three code paths — aggregate, window, list — must be one number.

    They are three separate implementations of "the middle of two doubles", which is exactly how
    two of them come to be fixed and the third not.
    """
    values = [1e308, 1.7e308]
    agg = bt.from_pydict({"x": values}).agg(m=bt.col("x").median()).to_pydict()["m"][0]
    win = (
        bt.from_pydict({"g": [1, 1], "x": values})
        .with_columns(m=bt.col("x").median().over(partition_by="g"))
        .to_pydict()["m"]
    )
    lst = bt.from_pydict({"l": [values]}).select(m=bt.col("l").list.median()).to_pydict()["m"][0]
    assert win == [agg, agg]
    assert lst == agg
