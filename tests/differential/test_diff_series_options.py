"""EWM options, `interpolate(max_gap=, by=)` and `qcut` (AP-268, AP-291, AP-292).

DuckDB has none of these, so the oracles are the libraries whose spelling Batcher copies:
pandas and Polars for the EWM options, Polars `interpolate_by` for time-weighted
interpolation, and pandas `qcut` for quantile binning. The sequences are short enough to
check by hand, and each carries nulls.

Two differences from pandas are asserted rather than left implicit. A null EWM input row
is null here (Polars) where pandas carries the last value. And `interpolate(max_gap=n)`
leaves a gap wider than `n` null as a whole, where pandas' `limit=n` fills its first `n`.
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

_X = [1.0, None, 3.0, 4.0, None, None, 10.0, 2.0]


def _ewm(x, method: str, **kw):
    ds = bt.from_arrow(pa.table({"t": list(range(len(x))), "x": pa.array(x, pa.float64())}))
    return (
        ds.with_columns(e=getattr(bt.col("x"), method)(**kw).over(order_by=["t"]))
        .sort("t")
        .to_pydict()["e"]
    )


def _null(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _close_at(got, want, rows, what):
    for i in rows:
        assert not _null(got[i]) and not _null(want[i]), f"{what}[{i}]: {got[i]} vs {want[i]}"
        assert abs(got[i] - want[i]) <= 1e-9 * max(1.0, abs(want[i])), f"{what}[{i}]"


_OBSERVED = [i for i, v in enumerate(_X) if v is not None]


@pytest.mark.parametrize("adjust", [True, False])
@pytest.mark.parametrize("ignore_nulls", [False, True])
@pytest.mark.parametrize("method", ["ewm_mean", "ewm_var", "ewm_std"])
def test_ewm_options_match_pandas(adjust, ignore_nulls, method):
    got = _ewm(_X, method, alpha=0.4, adjust=adjust, ignore_nulls=ignore_nulls)
    stat = method.removeprefix("ewm_")
    want = getattr(
        pd.Series(_X, dtype=float).ewm(alpha=0.4, adjust=adjust, ignore_na=ignore_nulls), stat
    )().tolist()
    # The spread is undefined for one observation (null in both).
    rows = _OBSERVED if method == "ewm_mean" else _OBSERVED[1:]
    _close_at(got, want, rows, f"{method} adjust={adjust} ignore_nulls={ignore_nulls}")
    # The pinned difference: a null input row is null here, the carried value in pandas.
    assert got[4] is None and not _null(want[4])


@pytest.mark.parametrize("adjust", [True, False])
@pytest.mark.parametrize("ignore_nulls", [False, True])
def test_ewm_mean_options_match_polars_row_for_row(adjust, ignore_nulls):
    got = _ewm(_X, "ewm_mean", alpha=0.4, adjust=adjust, ignore_nulls=ignore_nulls)
    want = (
        pl.DataFrame({"x": _X})
        .select(pl.col("x").ewm_mean(alpha=0.4, adjust=adjust, ignore_nulls=ignore_nulls))
        .to_series()
        .to_list()
    )
    assert [g is None for g in got] == [w is None for w in want]
    _close_at(got, want, _OBSERVED, "ewm_mean vs polars")


def test_adjust_false_is_the_hand_checkable_recursion():
    """``y = (1 - a) y_prev + a x`` with ``a = 0.5`` over ``[1, 2, 3]``: 1, 1.5, 2.25."""
    assert _ewm([1.0, 2.0, 3.0], "ewm_mean", alpha=0.5, adjust=False) == [1.0, 1.5, 2.25]


@pytest.mark.parametrize("min_periods", [0, 1, 2, 3])
def test_min_periods_matches_pandas(min_periods):
    got = _ewm(_X, "ewm_mean", alpha=0.5, min_periods=min_periods)
    want = pd.Series(_X, dtype=float).ewm(alpha=0.5, min_periods=min_periods).mean().tolist()
    seen = 0
    for i, v in enumerate(_X):
        seen += v is not None
        if v is None or seen < max(min_periods, 1):
            assert got[i] is None, f"row {i}"
        else:
            _close_at(got, want, [i], f"min_periods={min_periods}")


def test_the_defaults_are_unchanged():
    """No option given is the historical `adjust=True, ignore_nulls=False` output."""
    explicit = _ewm(_X, "ewm_mean", alpha=0.3, adjust=True, ignore_nulls=False, min_periods=1)
    assert _ewm(_X, "ewm_mean", alpha=0.3) == explicit


def test_ewm_option_validation():
    with pytest.raises(PlanError, match="min_periods"):
        bt.col("x").ewm_mean(alpha=0.5, min_periods=-1)
    with pytest.raises(PlanError, match="adjust"):
        bt.col("x").ewm_mean(alpha=0.5, adjust="no")


# --- interpolate -----------------------------------------------------------------------


def _interp(t, x, **kw):
    ds = bt.from_arrow(pa.table({"t": t, "x": pa.array(x, pa.float64())}))
    w = bt.col("x").interpolate(**kw)
    if "by" not in kw:
        w = w.over(order_by=["t"])
    return ds.with_columns(i=w).sort("t").to_pydict()["i"]


_T = [0, 1, 5, 6, 7, 8, 20, 21]
_Y = [0.0, None, 5.0, None, None, 8.0, None, 9.0]


def test_interpolate_by_matches_polars():
    got = _interp(_T, _Y, by="t")
    want = (
        pl.DataFrame({"t": _T, "x": _Y})
        .select(pl.col("x").interpolate_by("t"))
        .to_series()
        .to_list()
    )
    assert got == pytest.approx(want)


def test_interpolate_by_a_timestamp_with_a_duration_gap():
    import datetime as dt

    base = dt.datetime(2024, 1, 1)
    at = [base + dt.timedelta(minutes=m) for m in (0, 1, 4, 30, 40)]
    ds = bt.from_pydict({"at": at, "x": [0.0, None, 4.0, None, 40.0]})
    out = ds.with_columns(i=bt.col("x").interpolate(by="at", max_gap="10m")).sort("at")
    assert out.to_pydict()["i"] == [0.0, 1.0, 4.0, None, 40.0]


@pytest.mark.parametrize("max_gap", [0, 1, 2, 3])
def test_max_gap_counts_null_rows(max_gap):
    got = _interp(_T, _Y, max_gap=max_gap)
    unlimited = _interp(_T, _Y)
    # Gap lengths in null rows: one (row 1), two (rows 3-4), one (row 6).
    gap_of = {1: 1, 3: 2, 4: 2, 6: 1}
    for i, v in enumerate(_Y):
        if v is not None:
            assert got[i] == v
        elif gap_of[i] > max_gap:
            assert got[i] is None, f"row {i}"
        else:
            assert got[i] == pytest.approx(unlimited[i])


def test_max_gap_differs_from_pandas_limit_by_design():
    """pandas' `limit=1` fills the first row of the two-row gap; `max_gap=1` fills none."""
    want = pd.Series(_Y).interpolate(limit=1, limit_area="inside").tolist()
    got = _interp(_T, _Y, max_gap=1)
    assert not _null(want[3]) and got[3] is None


def test_interpolate_defaults_and_edges_are_unchanged():
    assert _interp([1, 2, 3], [None, 2.0, None]) == [None, 2.0, None]
    assert _interp([1], [None]) == [None]
    assert _interp([], []) == []
    with pytest.raises(PlanError, match="duration"):
        bt.col("x").interpolate(max_gap="5m")


# --- qcut ------------------------------------------------------------------------------


def _qcut(x, *args, **kw):
    ds = bt.from_arrow(pa.table({"k": list(range(len(x))), "x": pa.array(x, pa.float64())}))
    return ds.select(k=bt.col("k"), b=bt.col("x").qcut(*args, **kw)).sort("k").to_pydict()["b"]


def _pandas_codes(x, q, **kw):
    codes = pd.qcut(pd.Series(x, dtype=float), q, labels=False, **kw)
    return [None if _null(c) else int(c) for c in codes]


@pytest.mark.parametrize(
    "x",
    [
        [5.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0],
        [1.0, 2.0, 3.0],
        [0.1, 0.7, None, 0.3, float("nan"), 0.9, 0.5, 0.2, 0.2, 0.8],
        [3.0, -1.0, 2.5, 2.5, 10.0, -4.0, 0.0],
    ],
    ids=["ramp", "three", "nulls-nan", "ties"],
)
@pytest.mark.parametrize("q", [1, 2, 3, 4, 10, [0.0, 0.1, 0.5, 1.0], [0.2, 0.8]])
def test_qcut_matches_pandas(x, q):
    try:
        want = _pandas_codes(x, q)
    except ValueError:
        with pytest.raises(Exception, match="not unique"):
            _qcut(x, q)
        return
    assert _qcut(x, q) == want


def test_duplicate_edges_raise_or_drop_like_pandas():
    tied = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0]
    with pytest.raises(ValueError):
        _pandas_codes(tied, 4)
    with pytest.raises(Exception, match="not unique"):
        _qcut(tied, 4)
    assert _qcut(tied, 4, duplicates="drop") == _pandas_codes(tied, 4, duplicates="drop")
    # A constant column has one distinct edge and therefore no bin.
    assert _qcut([3.0, 3.0, 3.0], 4, duplicates="drop") == [None, None, None]


def test_qcut_labels_and_partitions():
    x = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
    ds = bt.from_pydict({"g": [1, 1, 1, 1, 2, 2, 2, 2], "x": x})
    out = ds.with_columns(
        h=bt.col("x").qcut(2, ["lo", "hi"]),
        within=bt.col("x").qcut(2).over(partition_by="g"),
    ).sort("x")
    assert out.to_pydict()["h"] == ["lo", "lo", "lo", "lo", "hi", "hi", "hi", "hi"]
    assert out.to_pydict()["within"] == [0, 0, 1, 1, 0, 0, 1, 1]
    assert out.collect().schema.field("within").type == pa.int64()


def test_qcut_empty_and_one_row():
    assert _qcut([], 4) == []
    assert _qcut([7.0], 1) == _pandas_codes([7.0], 1)


def test_qcut_validation():
    with pytest.raises(PlanError, match="labels"):
        bt.col("x").qcut(4, ["a", "b"])
    with pytest.raises(PlanError, match="labels cannot"):
        bt.col("x").qcut(2, ["a", "b"], duplicates="drop")
    with pytest.raises(PlanError, match="strictly increase"):
        bt.col("x").qcut([0.5, 0.2])
    with pytest.raises(PlanError, match="duplicates"):
        bt.col("x").qcut(4, duplicates="keep")
