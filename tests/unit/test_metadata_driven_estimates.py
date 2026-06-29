"""The metadata-driven estimator: every learned signal measurably sharpens the
estimate, and a richer-provenance estimate replaces a guess.

This is the cardinality half of the adaptive moat — Core measures, Kyber consumes.
It guards the loop end to end (learned rows, measured selectivity ratio, column
ndv, range quantiles) against regressions while the metadata stack evolves. Each
case is hub-independent: the learned state is supplied directly to the estimator.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col, count
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.signature import plan_signature
from batcher.plan.stats import Provenance


def _est(plan, sources, learned):
    return CardinalityEstimator(sources, learned).estimate(plan)


def test_learned_rows_replace_default_for_aggregate():
    ds = bt.from_arrow(pa.table({"k": [i % 10 for i in range(1000)]})).group_by("k").agg(n=count())
    plan, sources = ds._plan, ds._sources

    base = _est(plan, sources, {})  # no learning → "groups ≈ 10% of input" guess
    assert base.provenance is Provenance.DEFAULT

    learned = {plan_signature(plan): {"rows": 10.0}}  # measured: 10 groups
    after = _est(plan, sources, learned)
    assert after.provenance is Provenance.LEARNED
    assert after.rows == 10  # the measurement, not the 100-row guess


def test_inner_join_with_empty_side_estimates_exact_zero():
    # An EXACT-empty side makes an inner join EXACT-empty, so the estimate is a
    # provable zero (powering the count()/is_empty() metadata shortcut).
    a = bt.from_arrow(pa.table({"k": [1, 2, 3]})).limit(0)
    b = bt.from_arrow(pa.table({"k": [1, 2, 3]}))
    est = _est(a.join(b, on="k")._plan, a.join(b, on="k")._sources, {})
    assert est.rows == 0
    assert est.provenance is Provenance.EXACT


def test_left_join_empty_right_is_not_exact_empty():
    # A LEFT join with an empty *right* keeps every left row → not provably empty.
    a = bt.from_arrow(pa.table({"k": [1, 2, 3]}))
    b = bt.from_arrow(pa.table({"k": [1, 2, 3]})).limit(0)
    ds = a.join(b, on="k", how="left")
    est = _est(ds._plan, ds._sources, {})
    assert not (est.rows == 0 and est.provenance is Provenance.EXACT)


def test_asof_join_empty_left_estimates_exact_zero():
    # ASOF is left-style: an empty left → an EXACT-zero estimate.
    a = bt.from_arrow(pa.table({"k": [1, 2, 3]})).limit(0)
    b = bt.from_arrow(pa.table({"k": [1, 2, 3]}))
    ds = a.join_asof(b, on="k")
    est = _est(ds._plan, ds._sources, {})
    assert est.rows == 0
    assert est.provenance is Provenance.EXACT


def test_column_ndv_sharpens_equality_selectivity():
    ds = bt.from_arrow(pa.table({"k": [i % 50 for i in range(1000)]})).filter(col("k") == 7)
    plan, sources = ds._plan, ds._sources

    base = _est(plan, sources, {})  # default eq-selectivity 0.1 → ~100 rows
    after = _est(plan, sources, {"__column_ndv__": {"k": 50.0}})  # 1/50 → ~20 rows
    assert after.rows < base.rows
    assert abs(after.rows - 20) < 5


def test_measured_selectivity_ratio_generalizes_over_size():
    # Same predicate, two different input sizes; the measured ratio applies to both.
    p1, s1 = _filter_plan(1000)
    sig = plan_signature(p1)
    learned = {sig: {"selectivity": 0.25}}

    e_small = _est(p1, s1, learned)
    p2, s2 = _filter_plan(4000)
    e_big = _est(p2, s2, learned)
    assert e_small.provenance is Provenance.LEARNED and e_big.provenance is Provenance.LEARNED
    assert abs(e_small.rows - 250) < 1 and abs(e_big.rows - 1000) < 1


def test_range_quantiles_sharpen_range_selectivity():
    ds = bt.from_arrow(pa.table({"v": list(range(1000))})).filter(col("v") < 250)
    plan, sources = ds._plan, ds._sources
    probs = [i / 10 for i in range(11)]
    values = [i * 100 for i in range(11)]  # 0..1000 deciles
    learned = {"__column_quantiles__": {"v": {"probs": probs, "values": values}}}

    base = _est(plan, sources, {})  # flat 1/3 range default → ~333
    after = _est(plan, sources, learned)  # interpolated ~0.25 → ~250
    assert after.rows < base.rows
    assert abs(after.rows - 250) < 40


def test_distinct_uses_column_ndv():
    ds = bt.from_arrow(pa.table({"k": [i % 8 for i in range(1000)]})).distinct()
    plan, sources = ds._plan, ds._sources

    base = _est(plan, sources, {})  # flat 50% → ~500
    after = _est(plan, sources, {"__column_ndv__": {"k": 8.0}})  # ndv → ~8
    assert after.provenance is Provenance.LEARNED
    assert after.rows < base.rows
    assert abs(after.rows - 8) < 2


def test_join_key_ndv_capped_at_filtered_input():
    # Left is filtered to ~10 rows (via learned quantiles); the key's *source* ndv is
    # 1000, but a 10-row input can carry at most 10 distinct keys — capping corrects
    # an otherwise absurd estimate.
    left = bt.from_arrow(pa.table({"id": list(range(1000))})).filter(col("id") < 10)
    right = bt.from_arrow(pa.table({"id": list(range(50)), "w": list(range(50))}))
    j = left.join(right, on="id")
    probs = [i / 10 for i in range(11)]
    learned = {
        "__column_ndv__": {"id": 1000.0},
        "__column_quantiles__": {"id": {"probs": probs, "values": [i * 100 for i in range(11)]}},
    }
    e = CardinalityEstimator(j._sources, learned).estimate(j._plan)
    # Capped: left ndv→~10, divisor→50 → ~10 rows. Uncapped would divide by 1000 → ~0.5.
    assert e.rows > 5


def _filter_plan(rows: int):
    ds = bt.from_arrow(pa.table({"fv": list(range(rows))})).filter(col("fv") < rows // 4)
    return ds._plan, ds._sources


def test_source_ndv_seeds_cold_join_cardinality():
    # A cold join (no learned stats) must use source-provided NDV for the
    # |L||R|/max(ndv) estimate instead of the max(left, right) fallback — the metadata
    # `SourceStatistics` carries but the estimator previously ignored. With k holding 10
    # distinct values in each 1000-row side, the many-to-many join is ~1000*1000/10 =
    # 100k rows; the fallback would guess 1000 (100x low → bad join order).
    from batcher.plan.source_stats import SourceStatistics
    from batcher.plan.stats import ColumnStat

    a = bt.from_arrow(pa.table({"k": [i % 10 for i in range(1000)]}))
    b = bt.from_arrow(pa.table({"k": [i % 10 for i in range(1000)]}))
    ds = a.join(b, on="k")
    plan, sources = ds._plan, ds._sources

    cold = CardinalityEstimator(sources, {}).estimate(plan)
    assert cold.rows == 1000  # max(left, right) fallback — the NDV-blind guess

    col_ndv = {"k": ColumnStat(ndv=10.0)}
    stats = [SourceStatistics(row_count=1000, columns=col_ndv) for _ in sources]
    warm = CardinalityEstimator(sources, {}, source_stats=stats).estimate(plan)
    assert warm.rows == 100_000  # 1000 * 1000 / 10 — the NDV-based equi-join estimate
