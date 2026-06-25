"""Selinger-style predicate selectivity in the cardinality estimator."""

from __future__ import annotations

import batcher as bt
from batcher import col, lit
from batcher.kyber.cardinality import CardinalityEstimator


def _rows(ds, learned=None) -> float:
    est = CardinalityEstimator(ds._sources, learned)
    return est.estimate(ds._plan).rows


def test_equality_is_selective():
    ds = bt.from_pydict({"x": list(range(100))}).filter(col("x") == lit(5))
    assert _rows(ds) == 100 * 0.1


def test_not_equal_complements_equality():
    ds = bt.from_pydict({"x": list(range(100))}).filter(col("x") != lit(5))
    assert abs(_rows(ds) - 90.0) < 1e-9


def test_range_selectivity():
    ds = bt.from_pydict({"x": list(range(100))}).filter(col("x") > lit(5))
    assert abs(_rows(ds) - 100.0 / 3.0) < 1e-9


def test_conjunction_exponential_backoff():
    ds = bt.from_pydict({"x": list(range(100)), "y": list(range(100))}).filter(
        (col("x") == lit(5)) & (col("y") > lit(3))
    )
    # Conjuncts (0.1, 1/3) combine with exponential backoff: most selective at full
    # weight, the next dampened by a 1/2 exponent — `0.1 · (1/3)^(1/2)`. This sits
    # above the pure independence product (3.33 rows), the correlation-robust bias.
    expected = 100 * (0.1 * (1.0 / 3.0) ** 0.5)
    assert abs(_rows(ds) - expected) < 1e-9
    assert _rows(ds) > 100 * 0.1 * (1.0 / 3.0)  # backoff ≥ pure product


def test_conjunction_backoff_is_order_independent_and_damped():
    # Three conjuncts, each 0.1: pure product would be 0.001 (0.1 rows). Backoff
    # damps the tail: 0.1 · 0.1^(1/2) · 0.1^(1/4) = far less aggressive.
    base = {"x": list(range(100)), "y": list(range(100)), "z": list(range(100))}
    ds = bt.from_pydict(base).filter(
        (col("x") == lit(5)) & (col("y") == lit(5)) & (col("z") == lit(5))
    )
    expected = 100 * (0.1 * 0.1**0.5 * 0.1**0.25)
    assert abs(_rows(ds) - expected) < 1e-9
    assert _rows(ds) > 100 * 0.1**3  # strictly above the independence product


def test_disjunction_inclusion_exclusion():
    ds = bt.from_pydict({"x": list(range(100))}).filter((col("x") == lit(5)) | (col("x") == lit(6)))
    # 0.1 + 0.1 - 0.1*0.1 = 0.19
    assert abs(_rows(ds) - 19.0) < 1e-9


def test_negation_complements():
    ds = bt.from_pydict({"x": list(range(100))}).filter(~(col("x") > lit(5)))
    assert abs(_rows(ds) - 100 * (1.0 - 1.0 / 3.0)) < 1e-9


def test_learned_ndv_sharpens_equality():
    ds = bt.from_pydict({"x": list(range(100))}).filter(col("x") == lit(5))
    # With a known 50 distinct values, equality keeps ~1/50 of rows.
    assert _rows(ds, learned={"__column_ndv__": {"x": 50}}) == 100 * (1.0 / 50.0)


def test_composite_join_key_ndv_is_damped():
    # A two-column join key with learned per-column ndvs 100 and 50. The classic
    # independence product would treat the key as 100*50 = 5000 distinct, but real
    # composite keys are correlated: the damped estimate is 100 · 50^(1/2) ≈ 707,
    # between the max (100) and the product (5000). Join rows ≈ |L|·|R| / ndv.
    # Inputs larger than the damped ndv so the rows-cap doesn't mask the damping.
    left = bt.from_pydict({"a": list(range(2000)), "b": list(range(2000))})
    right = bt.from_pydict({"a": list(range(2000)), "b": list(range(2000))})
    ds = left.join(right, on=["a", "b"])
    damped_ndv = 100.0 * 50.0**0.5
    rows = _rows(ds, learned={"__column_ndv__": {"a": 100.0, "b": 50.0}})
    assert abs(rows - (2000 * 2000) / damped_ndv) < 1.0
    assert rows > (2000 * 2000) / (100.0 * 50.0)  # damped ndv < product ⇒ more rows


def test_is_null_predicates():
    ds = bt.from_pydict({"x": list(range(100))})
    assert abs(_rows(ds.filter(col("x").is_null())) - 5.0) < 1e-9
    assert abs(_rows(ds.filter(col("x").is_not_null())) - 95.0) < 1e-9


def test_window_preserves_input_cardinality():
    # A Window appends columns and never changes the row count. Before the
    # cardinality branch existed, a Window fell through to `unknown_rows` (1e12),
    # poisoning cost decisions above it.
    base = bt.from_pydict({"g": [1, 1, 2] * 10, "v": list(range(30))})
    ds = base.window(
        partition_by=["g"],
        order_by=["v"],
        functions={"rn": "row_number"},
    )
    assert _rows(ds) == 30.0


def test_filter_above_window_is_not_unknown():
    # Cost above a Window must track the real input size, not the 1e12 fallback.
    base = bt.from_pydict({"g": [1, 1, 2] * 10, "v": list(range(30))})
    ds = base.window(partition_by=["g"], order_by=["v"], functions={"rn": "row_number"}).filter(
        col("v") > lit(5)
    )
    # 30 rows, range selectivity 1/3 → 10, well below `unknown_rows`.
    assert abs(_rows(ds) - 30.0 / 3.0) < 1e-9


def test_join_uses_key_ndv_when_known():
    left = bt.from_pydict({"k": list(range(200)), "a": list(range(200))})
    right = bt.from_pydict({"k": list(range(50)), "b": list(range(50))})
    ds = left.join(right, on="k")
    # |L|·|R| / max(ndv) = 200·50 / 100 = 100
    assert _rows(ds, learned={"__column_ndv__": {"k": 100}}) == 100.0
    # Without ndv, fall back to the larger side.
    assert _rows(ds) == 200.0


def test_composite_join_key_damps_combined_ndv():
    # A two-column join key: each side's distinct-combination count combines the
    # per-key ndvs with exponential backoff (correlation-robust), not a raw product.
    # With ndv 10 and 10: 10 · 10^(1/2) ≈ 31.62 (capped at each side's rows), so the
    # denominator is 31.62 and 200·50 / 31.62 ≈ 316.2.
    left = bt.from_pydict({"k1": list(range(200)), "k2": list(range(200))})
    right = bt.from_pydict({"k1": list(range(50)), "k2": list(range(50))})
    ds = left.join(right, on=["k1", "k2"])
    learned = {"__column_ndv__": {"k1": 10, "k2": 10}}
    damped = 10.0 * 10.0**0.5
    assert abs(_rows(ds, learned=learned) - (200 * 50) / damped) < 1e-6
    # Without ndv on the keys, still falls back to the larger side.
    assert _rows(ds) == 200.0
