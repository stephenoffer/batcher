"""What `skewness` and `kurtosis` answer for a group with no variance.

Every value equal makes the second central moment zero, so both the skewness ratio
``m3/m2**1.5`` and the kurtosis ratio ``m4/m2**2`` are ``0/0``. SQL does not say what that
should be, and **DuckDB does not answer it the same way twice**: on the identical constant
column its ``kurtosis`` returns NULL and its ``skewness`` returns NaN.

Batcher returns NULL for both. That is a deliberate divergence on ``skewness``, taken for
internal consistency — matching DuckDB there would mean making Batcher's two moment
aggregates disagree with each other in order to reproduce a disagreement inside the oracle.
NULL is also the more useful answer for the callers that reach for these: a drift check or a
feature selector treats NULL as missing, where NaN is a number that silently poisons a
comparison.

This module exists because the divergence was previously unwritten and untested. It asserts
the difference rather than tolerating it, so that a future change to either engine's rule
fails here loudly instead of being absorbed. `assert_same` is deliberately not used: it is
the tolerance that would hide exactly this.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

_CONSTANT = [
    pytest.param([5.0] * 6, id="six-identical"),
    pytest.param([2.5] * 4, id="four-identical"),
    pytest.param([1.0] * 3, id="three-identical-the-skew-minimum"),
    pytest.param([0.0] * 5, id="all-zero"),
    pytest.param([-1.5] * 8, id="all-negative"),
]


def _batcher(values: list[float]) -> tuple[float | None, float | None]:
    out = (
        bt.from_arrow(pa.table({"x": pa.array(values, pa.float64())}))
        .agg(s=bt.col("x").skew(), k=bt.col("x").kurtosis())
        .to_pydict()
    )
    return out["s"][0], out["k"][0]


@pytest.mark.parametrize("values", _CONSTANT)
def test_both_moments_are_null_for_a_group_with_no_variance(values):
    """Batcher's own answer, and the property that makes it defensible: the two agree."""
    skew, kurt = _batcher(values)
    assert skew is None, f"skewness of a constant column should be NULL, got {skew!r}"
    assert kurt is None, f"kurtosis of a constant column should be NULL, got {kurt!r}"


@pytest.mark.parametrize("values", _CONSTANT)
def test_the_divergence_from_duckdb_is_exactly_skewness_and_is_duckdbs_own_inconsistency(values):
    """Pin the shape of the difference, including that it is one-sided.

    If DuckDB ever makes its two moments agree, this fails and the choice above should be
    revisited — which is the point of asserting it rather than tolerating it.
    """
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.register("t", pa.table({"x": pa.array(values, pa.float64())}))
    duck_skew, duck_kurt = con.sql("SELECT skewness(x), kurtosis(x) FROM t").fetchone()

    assert duck_skew is not None and math.isnan(duck_skew), (
        f"DuckDB's skewness is expected to be NaN here, got {duck_skew!r}"
    )
    assert duck_kurt is None, (
        f"DuckDB's kurtosis is expected to be NULL here, got {duck_kurt!r} — if this changed, "
        "DuckDB may have made its moments self-consistent and the divergence should be revisited"
    )

    batcher_skew, batcher_kurt = _batcher(values)
    assert batcher_kurt == duck_kurt, "kurtosis must still agree with the oracle"
    assert batcher_skew is None and duck_skew != duck_skew, "the divergence is skewness only"


def test_a_group_that_does_have_variance_still_matches_duckdb():
    """The negative control: without it this module would pass for an engine that always
    returned NULL, which is the failure mode a divergence test invites."""
    duckdb = pytest.importorskip("duckdb")
    values = [1.0, 2.0, 4.0, 8.0, 3.0, 5.0]
    con = duckdb.connect()
    con.register("t", pa.table({"x": pa.array(values, pa.float64())}))
    duck_skew, duck_kurt = con.sql("SELECT skewness(x), kurtosis(x) FROM t").fetchone()
    skew, kurt = _batcher(values)
    assert skew is not None and kurt is not None
    assert skew == pytest.approx(duck_skew, rel=1e-12)
    assert kurt == pytest.approx(duck_kurt, rel=1e-12)
