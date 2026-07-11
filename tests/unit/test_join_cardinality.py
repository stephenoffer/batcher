"""Join cardinality math: each join type must be estimated as its own algebra.

Every join type used to share one estimate. That is not a tuning imprecision, it produces
counts no execution can ever emit:

* `semi` and `anti` **partition** the left relation, so they cannot both be `|L|`.
* an outer join **preserves** its outer side, so `|L LEFT JOIN R| >= |L|` always.
* a cartesian product is `|L| x |R|`, not `max(|L|, |R|)`.

These are hard invariants, not heuristics, and they are what the memory budget and the
build-side choice are derived from. Each is asserted below against the *estimator*, so a
regression is caught without executing a join.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.config import active_config
from batcher.kyber.stats import StatsEstimator

pytestmark = pytest.mark.unit

_LEFT_ROWS = 1000
_RIGHT_ROWS = 10


def _sides():
    left = pa.table({"k": list(range(_LEFT_ROWS)), "a": list(range(_LEFT_ROWS))})
    right = pa.table({"k": list(range(_RIGHT_ROWS)), "b": list(range(_RIGHT_ROWS))})
    return bt.from_arrow(left), bt.from_arrow(right)


def _rows(dataset, ndv: dict[str, float] | None = None) -> float:
    learned = {"__column_ndv__": ndv} if ndv else {}
    est = StatsEstimator(dataset._sources, learned, active_config().optimizer.cardinality)
    return est.estimate(dataset._plan).rows


def _joined(how: str, ndv: dict[str, float] | None = None) -> float:
    left, right = _sides()
    return _rows(left.join(right, on="k", how=how), ndv)


# `k` is unique on the left; `_side_ndv` caps the right's ndv at its 10 rows.
_NDV = {"k": float(_LEFT_ROWS)}


def test_inner_join_uses_containment():
    # |L|x|R| / max(d_L, d_R) = 1000*10/1000 = 10
    assert _joined("inner", _NDV) == pytest.approx(10.0)


def test_left_join_never_falls_below_the_preserved_side():
    # The inner estimate is 10, but every one of the 1000 left rows is emitted
    # (null-padded when unmatched). Estimating 10 was impossible.
    assert _joined("left", _NDV) == pytest.approx(float(_LEFT_ROWS))


def test_right_join_never_falls_below_the_preserved_side():
    assert _joined("right", _NDV) >= _RIGHT_ROWS


def test_full_join_never_falls_below_either_side():
    assert _joined("full", _NDV) >= max(_LEFT_ROWS, _RIGHT_ROWS)


def test_semi_and_anti_partition_the_left_relation():
    semi = _joined("semi", _NDV)
    anti = _joined("anti", _NDV)
    # |semi| + |anti| = |L| by definition; both returning |L| was a contradiction.
    assert semi + anti == pytest.approx(float(_LEFT_ROWS))
    assert 0.0 <= semi <= _LEFT_ROWS
    assert 0.0 <= anti <= _LEFT_ROWS


def test_semi_join_matches_the_containment_fraction():
    # d_R/d_L = 10/1000 of the left keys are present in R.
    assert _joined("semi", _NDV) == pytest.approx(_LEFT_ROWS * (_RIGHT_ROWS / _LEFT_ROWS))


def test_semi_anti_fall_back_to_the_upper_bound_without_distinct_counts():
    # Unknowable match fraction -> over-budget (|L|) rather than risk an under-estimate.
    assert _joined("semi") == pytest.approx(float(_LEFT_ROWS))
    assert _joined("anti") == pytest.approx(float(_LEFT_ROWS))


def test_inner_join_never_exceeds_the_cartesian_bound():
    # A tiny known ndv on one side must not inflate the estimate past |L|x|R|.
    assert _joined("inner", {"k": 1.0}) <= _LEFT_ROWS * _RIGHT_ROWS


def test_cross_join_is_the_product_not_the_max():
    """A comma join lowers to an equi-join on a synthetic constant `__cross_key`.

    Its ndv is unmeasured, so the containment estimate used to fall through to
    `max(|L|, |R|)` — under-estimating the cartesian product by `min(|L|, |R|)`, exactly
    the operator whose size most needs to be believed.
    """
    left = pa.table({"k": list(range(_LEFT_ROWS))})
    right = pa.table({"j": list(range(_RIGHT_ROWS))})
    ds = bt.sql("select k, j from L, R", L=bt.from_arrow(left), R=bt.from_arrow(right))
    assert _rows(ds) == pytest.approx(float(_LEFT_ROWS * _RIGHT_ROWS))
    assert ds.collect().num_rows == _LEFT_ROWS * _RIGHT_ROWS  # the estimate is exact here
