"""Exponentially weighted moving statistics, checked against pandas and Polars.

DuckDB has no EWM function and no closed form for one that survives in floating point,
so the oracle here is the other two reference implementations rather than the usual
DuckDB. That is not a weaker check: pandas and Polars are the systems whose spelling and
defaults Batcher deliberately matches, so agreeing with them *is* the contract.

Where the two references disagree, Batcher follows pandas and its own aggregates:

* a **null input row** yields a null output in Polars and the carried value in pandas.
  Batcher returns null, like Polars, because every other Batcher window function propagates
  a null input.
* the **first row of `ewm_std`/`ewm_var`** is null in pandas and ``0.0`` in Polars. Batcher
  returns null, like pandas, because a single observation has no spread — the same reason
  ``bt.col('x').std()`` over one row is null.

Both disagreements are asserted below rather than left implicit, so a future change to
either cannot drift silently.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential

pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")

_ALPHAS = [0.1, 0.5, 0.9, 1.0]


def _batcher(x, method: str, *, dtype=None, **decay):
    dtype = dtype or pa.float64()
    ds = bt.from_arrow(
        pa.table(
            {
                "t": pa.array(list(range(len(x))), type=pa.int64()),
                "x": pa.array(x, type=dtype),
            }
        )
    )
    w = getattr(bt.col("x"), method)(**decay).over(order_by=["t"])
    return ds.with_columns(e=w).sort("t").to_pydict()["e"]


def _close(got, want, what, tol=1e-9):
    assert len(got) == len(want), f"{what}: length {len(got)} != {len(want)}"
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        g_null = g is None or (isinstance(g, float) and math.isnan(g))
        w_null = w is None or (isinstance(w, float) and math.isnan(w))
        if g_null or w_null:
            assert g_null and w_null, f"{what}[{i}]: {g!r} vs {w!r}"
        else:
            assert abs(g - w) <= tol * max(1.0, abs(w)), f"{what}[{i}]: {g} vs {w}"


@pytest.mark.parametrize("alpha", _ALPHAS)
@pytest.mark.parametrize(
    "x",
    [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [5.0, -3.0, 0.0, 12.5, -1.25],
        [7.0],
        [1e9, 1e9 + 1, 1e9 + 2, 1e9 + 3],  # a large offset the naive variance form loses
    ],
    ids=["ramp", "mixed", "single", "offset"],
)
def test_ewm_mean_matches_pandas_and_polars(alpha, x):
    got = _batcher(x, "ewm_mean", alpha=alpha)
    _close(got, pd.Series(x).ewm(alpha=alpha).mean().tolist(), "ewm_mean vs pandas")
    _close(
        got,
        pl.DataFrame({"x": x})
        .select(pl.col("x").ewm_mean(alpha=alpha, adjust=True, ignore_nulls=False))
        .to_series()
        .to_list(),
        "ewm_mean vs polars",
    )


@pytest.mark.parametrize("alpha", _ALPHAS)
@pytest.mark.parametrize("method", ["ewm_std", "ewm_var"])
def test_ewm_spread_matches_pandas(alpha, method):
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]
    got = _batcher(x, method, alpha=alpha)
    want = getattr(pd.Series(x).ewm(alpha=alpha), method.removeprefix("ewm_"))().tolist()
    _close(got, want, f"{method} vs pandas")


@pytest.mark.parametrize("method", ["ewm_mean", "ewm_std", "ewm_var"])
def test_a_null_input_row_yields_null_but_still_ages_the_decay(method):
    """Polars' `ignore_nulls=False` reading: the gap costs weight, it does not pause time."""
    x = [1.0, None, 3.0, 4.0]
    got = _batcher(x, method, alpha=0.5)
    assert got[1] is None, f"{method}: a null input must not invent a value"
    want = (
        pl.DataFrame({"x": x})
        .select(getattr(pl.col("x"), method)(alpha=0.5, adjust=True, ignore_nulls=False))
        .to_series()
        .to_list()
    )
    # Polars reports 0.0 for the undefined first spread; compare from the first row where
    # both are defined, and assert the deliberate difference separately below.
    start = 1 if method != "ewm_mean" else 0
    _close(got[start:], want[start:], f"{method} with a null, vs polars")


def test_the_first_spread_row_is_null_not_zero():
    """The one deliberate departure from Polars, pinned so it cannot drift silently."""
    for method in ("ewm_std", "ewm_var"):
        assert _batcher([1.0, 2.0, 3.0], method, alpha=0.5)[0] is None
    # ...and it is the same answer the ordinary aggregate gives for a one-row window.
    ds = bt.from_pydict({"x": [1.0]})
    assert ds.select(s=bt.col("x").std()).to_pydict()["s"] == [None]


@pytest.mark.parametrize(
    ("decay", "alpha"),
    [
        ({"span": 3.0}, 2.0 / 4.0),
        ({"com": 1.0}, 1.0 / 2.0),
        ({"half_life": 1.0}, 0.5),
        ({"half_life": 2.0}, 1.0 - math.exp(-math.log(2.0) / 2.0)),
    ],
    ids=["span", "com", "half_life_1", "half_life_2"],
)
def test_every_decay_spelling_resolves_to_the_same_alpha(decay, alpha):
    x = [1.0, 4.0, 2.0, 8.0]
    _close(
        _batcher(x, "ewm_mean", **decay),
        _batcher(x, "ewm_mean", alpha=alpha),
        f"ewm_mean {decay}",
    )
    # ...and pandas resolves the same spelling identically.
    _close(
        _batcher(x, "ewm_mean", **decay),
        pd.Series(x)
        .ewm(**{("halflife" if k == "half_life" else k): v for k, v in decay.items()})
        .mean()
        .tolist(),
        f"ewm_mean {decay} vs pandas",
    )


def test_an_integer_column_widens_to_float():
    got = _batcher([1, 2, 3], "ewm_mean", dtype=pa.int64(), alpha=0.5)
    _close(got, pd.Series([1.0, 2.0, 3.0]).ewm(alpha=0.5).mean().tolist(), "int ewm")


def test_each_partition_restarts_the_recurrence():
    table = pa.table(
        {
            "g": pa.array(["a", "b", "a", "b"]),
            "t": pa.array([0, 0, 1, 1], type=pa.int64()),
            "x": pa.array([1.0, 100.0, 2.0, 200.0], type=pa.float64()),
        }
    )
    got = (
        bt.from_arrow(table)
        .with_columns(e=bt.col("x").ewm_mean(alpha=0.5).over(partition_by=["g"], order_by=["t"]))
        .sort("g", "t")
        .to_pydict()["e"]
    )
    a = pd.Series([1.0, 2.0]).ewm(alpha=0.5).mean().tolist()
    b = pd.Series([100.0, 200.0]).ewm(alpha=0.5).mean().tolist()
    _close(got, a + b, "per-partition ewm")


def test_ewm_matches_single_node_when_distributed():
    n = 300
    table = pa.table(
        {
            "g": pa.array([f"s{i % 5}" for i in range(n)]),
            "t": pa.array(list(range(n)), type=pa.int64()),
            "x": pa.array([float((i * 37) % 101) for i in range(n)], type=pa.float64()),
        }
    )
    ds = bt.from_arrow(table)
    w = bt.col("x").ewm_mean(span=10).over(partition_by=["g"], order_by=["t"])
    one = ds.with_columns(e=w).sort("g", "t").to_pydict()["e"]
    many = ds.repartition(8).with_columns(e=w).sort("g", "t").to_pydict()["e"]
    _close(one, many, "ewm repartitioned")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "exactly one"),
        ({"alpha": 0.5, "span": 3.0}, "exactly one"),
        ({"alpha": 0.0}, "outside"),
        ({"alpha": 1.5}, "outside"),
        ({"span": 0.5}, "span must be"),
        ({"com": -1.0}, "com must be"),
        ({"half_life": 0.0}, "half_life must be"),
    ],
    ids=["none", "two", "alpha_zero", "alpha_big", "span", "com", "half_life"],
)
def test_a_bad_decay_is_rejected_at_plan_time(kwargs, message):
    with pytest.raises(PlanError, match=message):
        bt.col("x").ewm_mean(**kwargs)


def test_ewm_requires_an_order():
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(PlanError, match="requires order_by"):
        ds.with_columns(e=bt.col("x").ewm_mean(alpha=0.5).over()).to_pydict()


def test_ewm_mean_by_decays_with_elapsed_time_like_polars():
    """The form an irregular feed needs: a gap of an hour must not cost a second's weight."""
    import datetime as dt

    base = dt.datetime(2024, 1, 1)
    stamps = [base + dt.timedelta(minutes=m) for m in (0, 1, 5, 6, 60)]
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    table = pa.table({"t": pa.array(stamps), "x": pa.array(x, type=pa.float64())})
    got = (
        bt.from_arrow(table)
        .with_columns(e=bt.col("x").ewm_mean_by("t", "2m"))
        .sort("t")
        .to_pydict()["e"]
    )
    want = (
        pl.DataFrame({"t": stamps, "x": x})
        .select(pl.col("x").ewm_mean_by("t", half_life="2m"))
        .to_series()
        .to_list()
    )
    _close(got, want, "ewm_mean_by vs polars")
    # ...and it is genuinely different from the per-row decay, so this is not two names for
    # one computation: the 54-minute gap should nearly reset the smoother.
    per_row = _batcher(x, "ewm_mean", alpha=0.5)
    assert abs(got[-1] - 5.0) < 1e-6, "an hour's gap should leave almost only the new value"
    assert abs(per_row[-1] - got[-1]) > 0.1, "the two decays must not coincide"


def test_ewm_mean_by_handles_a_null_value_and_a_null_key():
    """A null value leaves the anchor at the last row actually seen (Polars' rule)."""
    import datetime as dt

    base = dt.datetime(2024, 1, 1)
    stamps = [base + dt.timedelta(minutes=m) for m in (0, 1, 5, 6)]
    x = [1.0, None, 3.0, 4.0]
    table = pa.table({"t": pa.array(stamps), "x": pa.array(x, type=pa.float64())})
    got = (
        bt.from_arrow(table)
        .with_columns(e=bt.col("x").ewm_mean_by("t", "2m"))
        .sort("t")
        .to_pydict()["e"]
    )
    want = (
        pl.DataFrame({"t": stamps, "x": x})
        .select(pl.col("x").ewm_mean_by("t", half_life="2m"))
        .to_series()
        .to_list()
    )
    _close(got, want, "ewm_mean_by with a null value")


def test_ewm_mean_by_restarts_per_partition_and_survives_repartitioning():
    n = 240
    table = pa.table(
        {
            "g": pa.array([f"s{i % 6}" for i in range(n)]),
            "t": pa.array([i * 7 % 500 for i in range(n)], type=pa.int64()),
            "x": pa.array([float((i * 13) % 29) for i in range(n)], type=pa.float64()),
        }
    )
    ds = bt.from_arrow(table)
    w = bt.col("x").ewm_mean_by("t", 25, partition_by=["g"])
    one = ds.with_columns(e=w).sort("g", "t").to_pydict()["e"]
    many = ds.repartition(8).with_columns(e=w).sort("g", "t").to_pydict()["e"]
    _close(one, many, "ewm_mean_by repartitioned")


def test_a_numeric_half_life_is_in_the_keys_own_units():
    ds = bt.from_pydict({"t": [0, 10, 20], "x": [1.0, 2.0, 3.0]})
    got = ds.with_columns(e=bt.col("x").ewm_mean_by("t", 10)).to_pydict()["e"]
    # A gap of exactly one half-life halves the previous value's weight.
    _close(got, [1.0, 1.5, 2.25], "numeric half_life")


@pytest.mark.parametrize(
    ("half_life", "message"),
    [(0, "must be > 0"), (-5, "must be > 0"), ("1mo", "calendar unit"), ("junk", "cannot parse")],
    ids=["zero", "negative", "calendar", "unparseable"],
)
def test_a_bad_half_life_is_rejected_at_plan_time(half_life, message):
    with pytest.raises(PlanError, match=message):
        bt.col("x").ewm_mean_by("t", half_life)


def test_only_the_mean_decays_by_time():
    """`ewm_std`/`ewm_var` have no time-decayed form here, and say so rather than guessing."""
    assert not hasattr(bt.col("x"), "ewm_std_by")
    assert not hasattr(bt.col("x"), "ewm_var_by")
