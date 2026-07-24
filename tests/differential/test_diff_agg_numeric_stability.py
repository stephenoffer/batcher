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


# --- the rolling moments carry the same cancellation risk as the whole-column ones ---


@pytest.mark.parametrize("offset", [0.0, 1e6, 1e9, 1e12, 1e15], ids=lambda o: f"offset{o:g}")
@pytest.mark.parametrize("agg", ["rolling_var", "rolling_std"])
def test_a_rolling_moment_survives_a_large_offset(agg, offset):
    """`rolling_var` must not lose the spread to the mean, as the whole-column `var` does not.

    `_rolling_var` composed the variance as ``E[x^2] - E[x]^2``, the sum-of-powers formula
    `var_state` was rewritten to escape. It subtracts two nearly equal large numbers, so it
    loses a digit for every digit by which the mean exceeds the spread. On
    ``[k+1, ..., k+6]`` with a 3-wide frame, where the true variance is exactly 1.0, it
    returned 0.999939 at ``k=1e6``, **0.0** at ``k=1e9`` (which reads as "this window is
    constant"), and **-201326592** at ``k=1e12`` -- a negative variance, which is not a
    rounding error but an impossible value, and which makes `rolling_std` take the square
    root of a negative number.

    An epoch-second timestamp is ~1.7e9 and a monetary column in cents reaches 1e12, so the
    failing range is the ordinary one. Centering on the partition mean first is exact,
    because ``Var(x) = Var(x - k)`` for any constant `k`.
    """
    values = [offset + d for d in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]
    ds = bt.from_pydict({"i": list(range(len(values))), "v": values})
    got = (
        ds.with_columns(r=getattr(col("v"), agg)(3, order_by=[col("i")]))
        .sort(col("i"))
        .to_pydict()["r"]
    )
    # The first row's frame holds one value, so the sample statistic is undefined there.
    assert got[0] != got[0], f"a one-value frame should be NaN, got {got[0]}"
    for i, value in enumerate(got[1:], start=1):
        assert value >= 0.0, f"row {i}: negative {agg} {value} is not a possible value"
    # Frames 3..6 are three consecutive integers, whose sample variance is exactly 1.
    for i in range(2, len(values)):
        assert got[i] == pytest.approx(1.0, rel=1e-12), f"row {i}: {agg} = {got[i]}"


def test_a_rolling_variance_is_never_negative_on_a_constant_window():
    """A constant frame has zero variance, and the clamp must not turn a NaN into zero."""
    ds = bt.from_pydict({"i": [0, 1, 2, 3], "v": [1e12, 1e12, 1e12, 1e12]})
    got = ds.with_columns(r=col("v").rolling_var(3, order_by=[col("i")])).sort(col("i"))
    assert got.to_pydict()["r"][1:] == [0.0, 0.0, 0.0]
    # ...while a non-finite value in the frame still propagates rather than being clipped.
    nan_ds = bt.from_pydict({"i": [0, 1, 2], "v": [1.0, float("nan"), 2.0]})
    out = nan_ds.with_columns(r=col("v").rolling_var(3, order_by=[col("i")])).to_pydict()["r"]
    assert all(v != v for v in out), f"NaN must propagate through the clamp, got {out}"


# --- an approximate quantile is still an order statistic -----------------------------


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("constant", [7.0] * 2000),
        ("constant_at_a_bucket_floor", [1.0] * 2000),
        ("two_values", [2.0, 8.0] * 1000),
        ("five_values", [1.0, 2.0, 3.0, 4.0, 5.0] * 400),
        ("all_negative", [-1.0, -2.0, -3.0, -4.0, -5.0] * 400),
    ],
)
def test_an_approximate_quantile_stays_inside_the_data(name, values):
    """`approx_quantile` may be inexact, but it may not leave ``[min, max]``.

    The DDSketch walk returned the bucket's geometric centre, which is not a value any row
    need have taken. When the data sits at the floor of its bucket -- a column of one
    repeated value is the extreme case -- every interior quantile landed outside the data:
    2,000 rows of `7.0` reported a median of `7.0288`, above the maximum. On a
    low-cardinality column the same effect pushed `q=0.9` past the largest value, so a
    "rows above the 90th percentile" filter matched nothing.
    """
    ds = bt.from_pydict({"v": values})
    lo, hi = min(values), max(values)
    qs = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    got = ds.agg(**{f"q{i}": col("v").approx_quantile(q) for i, q in enumerate(qs)}).to_pydict()
    for i, q in enumerate(qs):
        value = got[f"q{i}"][0]
        assert lo <= value <= hi, f"{name}: approx_quantile({q}) = {value} outside [{lo}, {hi}]"


def test_a_constant_column_has_that_constant_at_every_approximate_quantile():
    """Clamping makes the degenerate case exact, not merely in-range."""
    ds = bt.from_pydict({"v": [7.0] * 2000})
    got = ds.agg(m=col("v").approx_median(), q=col("v").approx_quantile(0.9)).to_pydict()
    assert got == {"m": [7.0], "q": [7.0]}


@pytest.mark.parametrize("offset", [0.0, 1e6, 1e9, 1e12, 1e15], ids=lambda o: f"offset{o:g}")
def test_zscore_survives_a_large_offset(offset):
    """`zscore` divides by a window standard deviation, which must not cancel to zero.

    `_window_mean_std` built the deviation from ``E[x^2] - E[x]^2``. At an offset of 1e9
    that difference cancelled to exactly 0 and every z-score became `inf`; at 1e12 it
    cancelled *negative* and the square root made every z-score `NaN`. The two-pass form
    ``E[(x - mean)^2]`` is available here because the window mean is already broadcast to
    every row, so it costs one more window aggregate over the same partition.
    """
    values = [offset + d for d in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]
    got = (
        bt.from_pydict({"i": list(range(len(values))), "v": values})
        .with_columns(z=col("v").zscore())
        .sort(col("i"))
        .to_pydict()["z"]
    )
    assert all(v == v and abs(v) != float("inf") for v in got), f"non-finite z-scores: {got}"
    # Symmetric data about its mean: the z-scores mirror, and the extremes are +/-1.3363.
    assert got[-1] == pytest.approx(1.3363062095621219, rel=1e-9)
    assert got[0] == pytest.approx(-1.3363062095621219, rel=1e-9)


@pytest.mark.parametrize("offset", [0.0, 1e9, 1e12, 1e15], ids=lambda o: f"offset{o:g}")
@pytest.mark.parametrize("agg", ["expanding_var", "expanding_std"])
def test_an_expanding_moment_survives_a_large_offset(agg, offset):
    """The running moments cancel exactly as the trailing-frame ones did.

    `expanding_var` accumulated ``E[x^2] - E[x]^2`` over a growing prefix: 0.0 at an offset
    of 1e9, and -161061273 -- a negative variance -- at 1e12.
    """
    values = [offset + d for d in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]
    got = (
        bt.from_pydict({"i": list(range(len(values))), "v": values})
        .with_columns(r=getattr(col("v"), agg)(order_by=[col("i")]))
        .sort(col("i"))
        .to_pydict()["r"]
    )
    assert got[0] != got[0], "a one-value prefix has no sample variance"
    for i, value in enumerate(got[1:], start=1):
        assert value >= 0.0, f"row {i}: negative {agg} {value} is not a possible value"
    want = [0.5, 1.0, 5 / 3, 2.5, 3.5]  # sample variance of 1..k, k = 2..6
    if agg == "expanding_std":
        want = [w**0.5 for w in want]
    assert got[1:] == pytest.approx(want, rel=1e-9)
