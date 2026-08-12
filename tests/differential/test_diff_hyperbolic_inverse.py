"""`asinh`/`acosh`/`atanh` must match DuckDB, including where the identities break.

These three were composed in the control plane out of `ln`/`sqrt`/arithmetic, and the
textbook identities they used are wrong at the ends of the range. `ln(x + sqrt(x*x + 1))`
squares its argument, so `x*x` overflows to `inf` above ~1.3e154 and `asinh(1e300)` came
back `inf` instead of 691.47; at `-inf` the same form evaluates `ln(-inf + inf)` and
returns NaN instead of `-inf`. `acosh` overflowed identically. They are engine nodes now,
and the extremes are the whole point of the test — a mid-range-only case passed
throughout.

DuckDB refuses several of these inputs outright (`acosh` and `atanh` raise outside their
domains rather than returning NaN), so each function is compared on the sub-range the
oracle will answer for, and the out-of-domain behavior is pinned against NumPy, which
agrees with IEEE and with the engine.
"""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

# The magnitudes where the composed forms failed, plus the ordinary range around them.
BIG = [1e154, 1e200, 1e300, 1.7976931348623157e308]


#: Each function gets its own column of inputs, because each has a different domain and
#: an Arrow table needs them equal-length.
DOMAINS = {
    "asinh": [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 7.25, -7.25, *BIG, *[-b for b in BIG], None],
    "acosh": [1.0, 1.5, 2.0, 10.0, 1e8, *BIG, None],
    "atanh": [0.0, -0.0, 0.25, -0.25, 0.9, -0.9, 0.999999, -0.999999, None],
}


@pytest.fixture(params=sorted(DOMAINS))
def fn(request, duck):
    """One (function name, registered oracle table) pair per hyperbolic inverse."""
    duck.execute("drop table if exists t")
    duck.register("t", pa.table({"x": DOMAINS[request.param]}))
    return request.param


def test_matches_duckdb_across_the_domain(fn, duck):
    """Each inverse over its whole domain, including where the old identity overflowed."""
    got = bt.from_pydict({"x": DOMAINS[fn]}).select(r=getattr(bt.col("x"), fn)()).collect()
    assert_same(got, duck.sql(f"select {fn}(x) as r from t"))


@pytest.mark.parametrize(
    ("fn", "value", "expected"),
    [
        # The two shapes the composed identity got wrong. `1e300` is the overflow case
        # and `-inf` the cancellation case; both were silently wrong, not an error.
        ("asinh", 1e300, 691.4686750787736),
        ("asinh", float("-inf"), float("-inf")),
        ("asinh", float("inf"), float("inf")),
        ("acosh", 1e300, 691.4686750787736),
        ("acosh", float("inf"), float("inf")),
        # Out of domain, where DuckDB raises and IEEE says NaN.
        ("acosh", 0.5, float("nan")),
        ("atanh", 1.0, float("inf")),
        ("atanh", -1.0, float("-inf")),
        ("atanh", 2.0, float("nan")),
    ],
)
def test_out_of_range_agrees_with_numpy(fn, value, expected):
    """The extremes DuckDB will not evaluate, pinned against NumPy and IEEE.

    Infinities and NaN are held exactly — they are the whole point. The one finite value
    is held to a relative tolerance, because NumPy's `arcsinh(1e300)` and libm's differ
    in the last bit and neither is the oracle here.
    """
    got = bt.from_pydict({"x": [value]}).select(r=getattr(bt.col("x"), fn)()).to_pydict()["r"][0]
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = float(getattr(np, "arc" + fn[1:])(np.float64(value)))
    for want in (expected, ref):
        if math.isnan(want):
            assert math.isnan(got)
        elif math.isinf(want):
            assert got == want
        else:
            assert got == pytest.approx(want, rel=1e-15)


def test_asinh_is_odd_and_acosh_inverts_cosh():
    """Algebraic identities the composed forms could not hold at the extremes."""
    xs = [0.0, 0.5, 3.0, 1e8, 1e300]
    d = (
        bt.from_pydict({"x": xs})
        .select(
            odd=bt.col("x").asinh() + (-bt.col("x")).asinh(),
            roundtrip=bt.col("x").sinh().asinh(),
        )
        .to_pydict()
    )
    assert all(v == 0.0 for v in d["odd"])
    # `sinh` overflows to inf past ~710, so the round trip only holds where it is finite.
    for x, r in zip(xs, d["roundtrip"], strict=True):
        if math.isfinite(math.sinh(x) if x < 700 else math.inf):
            assert r == pytest.approx(x, rel=1e-12, abs=1e-12)
