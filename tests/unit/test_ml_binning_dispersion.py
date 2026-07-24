"""Binning expressions and dispersion-ratio statistics.

`cut` is pinned to its structural definition — left-open intervals, one bucket more than
breaks, labels in bucket order — and to the pandas convention it follows. The dispersion
ratios are checked against their closed forms over numpy, because each exists to say something
a raw standard deviation does not, so "trust the standard deviation" is the wrong check.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


def _agg(ds: bt.Dataset, expr) -> float:
    return ds.agg(m=expr).collect().column("m")[0].as_py()


# --- cut -------------------------------------------------------------------------------


def test_cut_returns_the_bin_index() -> None:
    ds = bt.from_pydict({"x": [5, 18, 40, 70]})
    assert ds.with_columns(b=bt.cut("x", [12, 19, 65])).to_pydict()["b"] == [0, 1, 2, 3]


def test_cut_is_left_open_by_default() -> None:
    # A value equal to a break falls into the lower bucket (right=True, interval (a, b]).
    ds = bt.from_pydict({"x": [12, 19]})
    assert ds.with_columns(b=bt.cut("x", [12, 19])).to_pydict()["b"] == [0, 1]


def test_cut_right_open_puts_the_edge_in_the_upper_bucket() -> None:
    ds = bt.from_pydict({"x": [12, 19]})
    assert ds.with_columns(b=bt.cut("x", [12, 19], right=False)).to_pydict()["b"] == [1, 2]


def test_cut_maps_to_labels() -> None:
    ds = bt.from_pydict({"x": [5, 40, 70]})
    got = bt.cut("x", [12, 65], labels=["low", "mid", "high"])
    assert ds.with_columns(b=got).to_pydict()["b"] == ["low", "mid", "high"]


def test_cut_matches_numpy_digitize() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=500)
    breaks = [-1.0, 0.0, 1.0]
    ds = bt.from_pydict({"x": values.tolist()})
    got = ds.with_columns(b=bt.cut("x", breaks)).to_pydict()["b"]
    # right=True means bucket = count of breaks strictly below the value.
    expected = [int(np.sum(np.array(breaks) < v)) for v in values]
    assert got == expected


def test_cut_rejects_empty_breaks() -> None:
    with pytest.raises(PlanError, match="at least one break"):
        bt.cut("x", [])


def test_cut_rejects_wrong_label_count() -> None:
    with pytest.raises(PlanError, match="one label per bucket"):
        bt.cut("x", [1, 2], labels=["a", "b"])


def test_cut_composes_in_group_by() -> None:
    ds = bt.from_pydict({"x": [1, 5, 9]})
    counts = (
        ds.with_columns(band=bt.cut("x", [3, 7]))
        .group_by("band")
        .agg(n=bt.col("x").count())
        .sort("band")
        .to_pydict()
    )
    assert counts["n"] == [1, 1, 1]


# --- dispersion ratios -----------------------------------------------------------------


@pytest.fixture(scope="module")
def sample() -> tuple[np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(1)
    x = rng.gamma(2.0, 1.5, 400) + 1.0
    return x, bt.from_pydict({"x": x.tolist()})


def test_index_of_dispersion_matches_numpy(sample) -> None:
    x, ds = sample
    expected = x.var(ddof=1) / x.mean()
    assert _agg(ds, bt.index_of_dispersion("x")) == pytest.approx(expected)


def test_index_of_dispersion_is_one_for_a_poisson_stream() -> None:
    rng = np.random.default_rng(3)
    x = rng.poisson(5.0, 20000).astype(float)
    ds = bt.from_pydict({"x": x.tolist()})
    assert _agg(ds, bt.index_of_dispersion("x")) == pytest.approx(1.0, abs=0.1)


def test_signal_to_noise_matches_numpy(sample) -> None:
    x, ds = sample
    assert _agg(ds, bt.signal_to_noise("x")) == pytest.approx(x.mean() / x.std(ddof=1))


def test_studentized_range_matches_numpy(sample) -> None:
    x, ds = sample
    expected = (x.max() - x.min()) / x.std(ddof=1)
    assert _agg(ds, bt.studentized_range("x")) == pytest.approx(expected)


def test_relative_range_matches_numpy(sample) -> None:
    x, ds = sample
    assert _agg(ds, bt.relative_range("x")) == pytest.approx((x.max() - x.min()) / x.mean())


def test_dispersion_ratios_compose_with_group_by() -> None:
    ds = bt.from_pydict({"g": ["a", "a", "b", "b"], "x": [1.0, 3.0, 10.0, 30.0]})
    got = ds.group_by("g").agg(m=bt.signal_to_noise("x")).sort("g").to_pydict()
    assert len(got["m"]) == 2
