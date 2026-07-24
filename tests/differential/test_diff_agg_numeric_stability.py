"""Numeric-stability differential coverage for the moment aggregates.

`var`/`stddev` moved to Welford (B9) to stop the ``Sum(x^2) - Sum(x)^2/n`` cancellation, but
`covar_pop`/`covar_samp`/`corr`/`skewness`/`kurtosis` kept a sum-of-powers state and
the same cancelling finalize. On a column with a large offset (timestamps, ids), that
subtraction of two nearly-equal large numbers catastrophically cancels:
`covar_pop([1e9+1, 1e9+2, …])` came back as exactly `0` (true `2`), and `corr` as
`NULL`. These pin the central-moment (co-moment) rewrite against DuckDB — the values
DuckDB itself returns are the oracle.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, corr, covar_pop, covar_samp

pytestmark = pytest.mark.differential

# Values at a 1e9 offset: exactly representable in f64, so the true covariance/correlation
# is exact, but the sum-of-powers formula cancelled it to 0 / NULL.
_OFF = 1e9
_X = [_OFF + 1, _OFF + 2, _OFF + 3, _OFF + 4, _OFF + 5]
_Y = [_OFF + 1, _OFF + 3, _OFF + 2, _OFF + 7, _OFF + 4]


def _tbl():
    return pa.table({"x": _X, "y": _Y})


def _approx_matches(out, duck, cols):
    # Two-pass central moments and DuckDB's algorithm agree to f64 precision but not
    # bit-for-bit, so compare with a relative tolerance. This still catches the old
    # cancelling formula, which returned 0.0 or NULL — off by ~100%, not ~1e-9.
    got = out.to_pydict()
    want = dict(zip(cols, duck.fetchone(), strict=True))
    for k in cols:
        assert got[k][0] == pytest.approx(want[k], rel=1e-7), f"{k}: {got[k][0]} vs {want[k]}"


def test_covar_corr_stable_at_large_offset(duck):
    duck.register("t", _tbl())
    out = (
        bt.from_arrow(_tbl())
        .agg(
            cp=covar_pop(col("x"), col("y")),
            cs=covar_samp(col("x"), col("y")),
            c=corr(col("x"), col("y")),
        )
        .collect()
    )
    rel = duck.sql("SELECT covar_pop(x,y) AS cp, covar_samp(x,y) AS cs, corr(x,y) AS c FROM t")
    _approx_matches(out, rel, ["cp", "cs", "c"])


def test_skewness_kurtosis_stable_at_large_offset(duck):
    # Enough points (asymmetric) that kurtosis is defined (n >= 4) and non-trivial.
    base = (1.0, 2.0, 3.0, 4.0, 10.0, 1.0, 2.0)
    xs = [_OFF + v for v in base]
    ds = bt.from_arrow(pa.table({"x": xs}))
    out = ds.agg(s=col("x").skewness(), k=col("x").kurtosis()).collect()
    # Skewness/kurtosis are translation-invariant, so the oracle is DuckDB on the
    # *un-offset* data — where DuckDB is stable. (At `_OFF` DuckDB's own sum-of-powers
    # formula catastrophically cancels and returns NaN, so it cannot be the oracle there;
    # the point of the fix is that Batcher's two-pass state stays exact at the offset.)
    duck.register("b", pa.table({"x": list(base)}))
    rel = duck.sql("SELECT skewness(x) AS s, kurtosis(x) AS k FROM b")
    _approx_matches(out, rel, ["s", "k"])


def test_covar_corr_grouped_single_node_equals_distributed():
    # Same offset, grouped — the mergeable central-moment state must give the identical
    # result single-node and across workers (the single-node == distributed invariant).
    g = {
        "g": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "x": [_OFF + v for v in (1, 2, 3, 4, 1, 5, 2, 8)],
        "y": [_OFF + v for v in (1, 3, 2, 7, 1, 6, 3, 9)],
    }
    ds = (
        bt.from_pydict(g)
        .group_by("g")
        .agg(
            c=corr(col("x"), col("y")),
            s=col("x").skewness(),
            cp=covar_pop(col("x"), col("y")),
        )
    )
    single = {
        k: (round(c, 6), round(s, 6), round(cp, 6))
        for k, c, s, cp in zip(
            *[ds.collect().to_pydict()[col_] for col_ in ("g", "c", "s", "cp")], strict=True
        )
    }
    dd = ds.collect(distributed=True, num_workers=3).to_pydict()
    multi = {
        k: (round(c, 6), round(s, 6), round(cp, 6))
        for k, c, s, cp in zip(dd["g"], dd["c"], dd["s"], dd["cp"], strict=True)
    }
    assert single == multi


# --- a non-finite value must not be clipped away ------------------------------------


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"]
)
@pytest.mark.parametrize("agg", ["var", "std"])
def test_a_non_finite_value_does_not_produce_zero_variance(agg, bad):
    """`var`/`std` over data containing NaN or +/-inf must not report `0.0`.

    The Welford finalize clipped its result with ``max(0.0)`` to absorb the tiny negative
    `M2` that cancellation can leave. `f64::max` returns the *other* operand when one is
    NaN, so that clip also turned a NaN `M2` — which any NaN or infinity in the input
    produces, since Welford's centering computes ``inf - inf`` — into a confident `0.0`.

    Zero variance is not a near-miss: it is the signal that a column is constant, and it is
    what drift detection, feature selection, and a data-quality "is this column dead?" check
    all key on. `mean` and `sum` over the same rows propagate NaN, so the aggregates also
    disagreed with each other about whether the data was finite.
    """
    ds = bt.from_pydict({"v": [1.0, bad, 2.0]})
    got = ds.agg(r=getattr(col("v"), agg)()).to_pydict()["r"][0]
    assert got != 0.0, f"{agg} over data containing {bad} reported zero variance"
    assert got != got, f"{agg} over data containing {bad} should be NaN, got {got}"


def test_the_clip_still_absorbs_the_negative_m2_it_was_written_for():
    """A constant column still yields exactly 0.0, not a tiny negative from cancellation."""
    ds = bt.from_pydict({"v": [_OFF + 1] * 6})
    got = ds.agg(v=col("v").var(), s=col("v").std()).to_pydict()
    assert got["v"][0] == 0.0, got
    assert got["s"][0] == 0.0, got


def test_every_path_agrees_about_a_non_finite_variance():
    """...and the four schedulings agree, which is where the property suite caught it."""
    ds = bt.from_pydict({"k": [0, 0, 0], "v": [1.0, float("nan"), 2.0]})
    plan = ds.group_by("k").agg(s=col("v").std())
    results = {
        "collect": plan.collect().to_pydict()["s"][0],
        "spill": plan.collect(spill=True).to_pydict()["s"][0],
        "distributed": plan.collect(distributed=True, num_workers=2).to_pydict()["s"][0],
    }
    assert all(v != v for v in results.values()), f"paths disagree about NaN: {results}"
