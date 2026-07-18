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


# --- disjoint key ranges make an inner/semi join provably empty -------------------
#
# Bounds are valid supersets of the actual values, so two non-overlapping key ranges share
# no value and cannot match — the join analogue of an out-of-bounds equality. It stays a
# DEFAULT estimate (not an EXACT-empty proof), so it steers cost/join-order without letting
# `count()` answer 0 from a possibly-loose bound.


def _scan(source_id):
    import pyarrow as pa

    from batcher.plan.logical import Scan
    from batcher.plan.schema import SchemaRef

    return Scan(source_id=source_id, schema=SchemaRef(pa.schema([("d", pa.int64())])))


def _src(lo, hi, rows):
    from batcher.plan.source_stats import SourceStatistics
    from batcher.plan.stats import ColumnStat, Provenance

    return SourceStatistics(
        row_count=rows,
        exact_rows=True,
        columns={
            "d": ColumnStat(
                min=lo, max=hi, ndv=float(hi - lo + 1), null_count=0, provenance=Provenance.EXACT
            )
        },
    )


def _join(join_type):
    from batcher.plan.logical import Join
    from batcher.plan.logical.join import JoinOutputCol

    left, right = _scan(0), _scan(1)
    out = (JoinOutputCol(side="left", name="d", alias="d"),)
    return Join(
        left=left, right=right, left_keys=("d",), right_keys=("d",), join_type=join_type, output=out
    )


def _est(left_range, right_range):
    return StatsEstimator([None, None], source_stats=[_src(*left_range), _src(*right_range)])


def test_disjoint_keys_make_inner_join_empty():
    est = _est((1, 10, 1000), (100, 200, 1000))
    result = est.estimate(_join("inner"))
    assert result.rows == 0.0
    assert result.provenance is not None  # DEFAULT, not EXACT — a hint, not a proof


def test_disjoint_keys_make_semi_join_empty():
    est = _est((1, 10, 1000), (100, 200, 1000))
    assert est.estimate(_join("semi")).rows == 0.0


def test_overlapping_keys_are_estimated_normally():
    est = _est((1, 150, 1000), (100, 200, 1000))
    assert est.estimate(_join("inner")).rows > 0.0


def test_disjoint_keys_do_not_empty_a_left_join():
    # A LEFT join preserves its left side, so disjoint keys give |L| unmatched rows, not 0.
    est = _est((1, 10, 1000), (100, 200, 1000))
    assert est.estimate(_join("left")).rows >= 1000.0


# --- skewed join keys: a hot value's cross product floors the estimate -------------
#
# Selinger's uniform |L||R|/max(ndv) assumes every key value is equally likely. A heavy
# hitter present on both sides produces (f_L·|L|)·(f_R·|R|) rows by itself, which the uniform
# estimate misses — the single biggest under-estimate risk in the join path (a skewed build
# side OOMs). Summing the shared-MCV cross product is a firm lower bound, so it only raises.


def _skew_join(left_mcv, right_mcv, ndv=100.0, rows=1000):
    import pyarrow as pa

    from batcher.plan.logical import Scan
    from batcher.plan.logical.join import JoinOutputCol
    from batcher.plan.schema import SchemaRef
    from batcher.plan.source_stats import SourceStatistics
    from batcher.plan.stats import ColumnStat, Provenance

    sch = SchemaRef(pa.schema([("k", pa.int64())]))
    left, right = Scan(source_id=0, schema=sch), Scan(source_id=1, schema=sch)
    out = (JoinOutputCol(side="left", name="k", alias="k"),)
    from batcher.plan.logical import Join

    node = Join(
        left=left, right=right, left_keys=("k",), right_keys=("k",), join_type="inner", output=out
    )

    def src(mcv):
        return SourceStatistics(
            row_count=rows,
            exact_rows=True,
            columns={
                "k": ColumnStat(
                    min=0, max=100, ndv=ndv, null_count=0, provenance=Provenance.EXACT, mcv=mcv
                )
            },
        )

    est = StatsEstimator([None, None], source_stats=[src(left_mcv), src(right_mcv)])
    return est.estimate(node).rows


def test_uniform_join_is_the_selinger_estimate():
    # No hot value → the skew floor is 0, so the plain Selinger ratio stands: 1000²/100.
    assert _skew_join({}, {}) == pytest.approx(1000 * 1000 / 100)


def test_shared_hot_key_floors_the_estimate_above_selinger():
    # Value 7 at 50% on both sides contributes (0.5·1000)·(0.5·1000) = 250000 by itself.
    got = _skew_join({"7": 0.5}, {"7": 0.5})
    assert got == pytest.approx(250000.0)
    assert got > 1000 * 1000 / 100  # far above the uniform estimate


def test_disjoint_hot_keys_do_not_inflate():
    # Hot values that don't match across sides contribute nothing to the join.
    assert _skew_join({"7": 0.5}, {"3": 0.5}) == pytest.approx(1000 * 1000 / 100)


def test_skew_floor_is_capped_at_the_cartesian_bound():
    got = _skew_join({"7": 1.0}, {"7": 1.0}, ndv=1.0)
    assert got == pytest.approx(1000 * 1000)  # never exceeds |L|·|R|


def _semi_join(left_mcv, right_mcv, join_type="semi", ndv=100.0, rows=1000):
    import pyarrow as pa

    from batcher.plan.logical import Join, Scan
    from batcher.plan.logical.join import JoinOutputCol
    from batcher.plan.schema import SchemaRef
    from batcher.plan.source_stats import SourceStatistics
    from batcher.plan.stats import ColumnStat, Provenance

    sch = SchemaRef(pa.schema([("k", pa.int64())]))
    left, right = Scan(source_id=0, schema=sch), Scan(source_id=1, schema=sch)
    out = (JoinOutputCol(side="left", name="k", alias="k"),)
    node = Join(
        left=left, right=right, left_keys=("k",), right_keys=("k",), join_type=join_type, output=out
    )

    def src(mcv):
        return SourceStatistics(
            row_count=rows,
            exact_rows=True,
            columns={
                "k": ColumnStat(
                    min=0, max=100, ndv=ndv, null_count=0, provenance=Provenance.EXACT, mcv=mcv
                )
            },
        )

    return (
        StatsEstimator([None, None], source_stats=[src(left_mcv), src(right_mcv)])
        .estimate(node)
        .rows
    )


def test_semi_join_is_floored_by_a_shared_hot_key():
    # Value 7 is 60% of the left and present in R's MCV, so ≥0.6·1000 rows survive — above
    # the uniform d_R/d_L fraction (which here is 1.0·1000, so the floor is not binding); use
    # a case where the containment fraction is smaller than the hot mass.
    got = _semi_join({"7": 0.6}, {"7": 0.1}, ndv=1000.0)  # d_R/d_L small → base tiny
    assert got >= 0.6 * 1000  # the hot key alone guarantees this many


def test_anti_join_has_no_skew_floor():
    # Absence from R's top-value MCV does not prove a value is missing from R, so anti gets
    # no floor — it stays the containment estimate (here bounded by |L|).
    got = _semi_join({"7": 0.6}, {"7": 0.1}, join_type="anti", ndv=1000.0)
    assert got <= 1000


# --- cross joins: the product for outer types, but NOT for semi/anti ---------------
#
# A comma/cross join lowers to an equi-join on a synthetic constant key. Every left row then
# matches every right row, so inner/left/right/full all emit |L|x|R| — but a `semi` emits each
# left row once and an `anti` emits none. Dispatching per type is what stops the product being
# handed to semi/anti (an anti cross join is empty, and used to be estimated at |L|).


def _cart_rows(join_type, lrows=100.0, rrows=7.0):
    from batcher.plan.stats import Provenance, RelStats

    class _Node:
        pass

    node = _Node()
    node.join_type = join_type
    est = StatsEstimator([], {}, active_config().optimizer.cardinality)
    left = RelStats(lrows, Provenance.EXACT)
    right = RelStats(rrows, Provenance.EXACT)
    return est._cartesian_rows(node, left, right)[0]


@pytest.mark.parametrize("join_type", ["inner", "left", "right", "full"])
def test_cross_join_is_the_full_product(join_type):
    assert _cart_rows(join_type) == pytest.approx(700.0)


def test_cross_join_semi_keeps_each_left_row_once():
    assert _cart_rows("semi") == pytest.approx(100.0)


def test_cross_join_anti_keeps_nothing():
    assert _cart_rows("anti") == pytest.approx(0.0)


def test_cross_join_anti_over_empty_right_keeps_everything():
    assert _cart_rows("anti", rrows=0.0) == pytest.approx(100.0)
