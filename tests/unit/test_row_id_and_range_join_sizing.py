"""Two operators the sizing layers could not see: `RowId` and `RangeJoin`.

Both were missing from a per-operator table, and a missing entry in either falls through
to a default that is wrong in the dangerous direction.

- **`RowId` had no `StatsEstimator` branch**, so `with_row_index()` fell to the
  `unknown_rows` placeholder: an EXACT 1,000-row relation became a 1e12-row guess for
  every operator above it. A join above a `with_row_index` then picked its build side,
  and admission sized the query, against a fiction. `with_row_index` is strictly 1:1 —
  it appends a counter and changes nothing else.

- **`RangeJoin` was missing from `kyber.annotate._BREAKER_KINDS`**, so Carbonite budgeted
  it at ~one morsel. It is the one operator in that set whose output is super-linear in
  its inputs, which makes the under-estimate the largest: a 50k x 50k inequality join
  estimated at 833M rows was admitted against a 393 KB envelope. `bc_interp::ops::joins`
  documents it as a breaker needing both sides whole.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.kyber.stats import StatsEstimator
from batcher.plan.stats import Provenance, SortOrder


def _ds(n: int = 1_000):
    return bt.from_arrow(pa.table({"x": list(range(n)), "y": list(range(n))}))


def _est(ds):
    return StatsEstimator(ds._sources)


# --- RowId: strictly 1:1 -------------------------------------------------------


def test_row_id_preserves_exact_row_count():
    ds = _ds()
    est = _est(ds)
    base = est.estimate(ds._plan)
    rowid = est.estimate(ds.with_row_index("i")._plan)
    assert base.rows_exact
    assert rowid.rows == base.rows == 1_000
    assert rowid.provenance is base.provenance


def test_row_id_carries_input_column_stats_through():
    """A counter is appended; no input value is added, removed, or reordered."""
    ds = _ds()
    est = _est(ds)
    base = est.estimate(ds._plan)
    rowid = est.estimate(ds.with_row_index("i")._plan)
    for name, stat in base.columns.items():
        assert rowid.column(name) == stat


def test_row_id_counter_column_is_known_when_rows_are():
    ds = _ds()
    stat = _est(ds).estimate(ds.with_row_index("i", offset=5)._plan).column("i")
    assert (stat.min, stat.max) == (5.0, 1_004.0)
    assert stat.ndv == 1_000.0
    assert stat.null_count == 0.0


def test_row_id_counter_is_not_invented_under_an_estimated_count():
    """Under a non-EXACT count the range would be a guess, and a guessed min/max on a
    synthetic key is exactly what a downstream `filter(i < k)` would size itself against."""
    ds = _ds()
    filtered = ds.filter(bt.col("x") > 10).with_row_index("i")
    rs = _est(ds).estimate(filtered._plan)
    assert not rs.rows_exact
    assert rs.column("i").min is None and rs.column("i").max is None


def test_row_id_records_its_own_ordering_only_when_the_child_has_none():
    ds = _ds()
    est = _est(ds)
    assert est.estimate(ds.with_row_index("i")._plan).sorted_by == (SortOrder("i"),)
    sorted_child = ds.sort("x").with_row_index("i")
    assert est.estimate(sorted_child._plan).sorted_by == est.estimate(ds.sort("x")._plan).sorted_by


def test_row_id_result_is_unchanged():
    """The sizing fix must not move a row."""
    ds = _ds(10).with_row_index("i")
    assert ds.collect().to_pydict()["i"] == list(range(10))


# --- RangeJoin: a breaker, budgeted like one -----------------------------------


def _range_join_plan(n: int = 50_000):
    left = bt.from_arrow(pa.table({"x": list(range(n))}))
    right = bt.from_arrow(pa.table({"y": list(range(n))}))
    return left.join(right, how="cross").filter(bt.col("x") < bt.col("y"))


def _op(plan, kind: str):
    import batcher.kyber as kyber

    physical = kyber.optimize(plan._plan, sources=plan._sources)
    matches = [op for op in physical.ops if op.kind == kind]
    assert matches, f"no {kind} in plan: {[op.kind for op in physical.ops]}"
    return matches[0]


def test_range_join_is_budgeted_as_a_breaker():
    op = _op(_range_join_plan(), "RangeJoin")
    # `rows x width`, not the one-morsel streaming footprint the missing entry gave it.
    assert op.bounds.m_max_bytes > op.properties.est_rows  # >= 1 byte/row, i.e. not a morsel
    assert op.bounds.m_max_bytes > 10**9


def test_range_join_gets_shuffle_parallelism_sizing():
    """Parallelism sizing is gated on the same breaker set, so the omission cost it both."""
    assert _op(_range_join_plan(), "RangeJoin").bounds.n_max_parallelism >= 1


def test_range_join_over_budget_verdict_is_advisory_not_a_rejection():
    """Raising the envelope must not start failing legitimate queries: the binding
    operator here is sized from a Selinger guess, and the admission contract is that a
    guess routes a plan out-of-core but never rejects it."""
    import batcher.kyber as kyber
    from batcher.carbonite.base import ResourceContext
    from batcher.carbonite.policies.admission import BudgetingAdmission
    from batcher.config import active_config

    ds = _range_join_plan()
    physical = kyber.optimize(ds._plan, sources=ds._sources)
    verdict = BudgetingAdmission(available_bytes=1 << 30).validate(
        physical, ResourceContext(config=active_config())
    )
    assert not verdict.feasible
    assert verdict.advisory


def test_range_join_result_is_unchanged():
    """The sizing fix must not move a row (the differential oracle covers the values)."""
    left = bt.from_arrow(pa.table({"x": [1, 2, 3, 4, 5]}))
    right = bt.from_arrow(pa.table({"y": [2, 3, 6]}))
    out = left.join(right, how="cross").filter(bt.col("x") < bt.col("y")).collect()
    assert sorted(zip(out.to_pydict()["x"], out.to_pydict()["y"], strict=True)) == [
        (1, 2), (1, 3), (1, 6), (2, 3), (2, 6), (3, 6), (4, 6), (5, 6),
    ]  # fmt: skip


# --- fixed-count Sample: a breaker budgeted as one --------------------------------


def _sample_op(n=None, fraction=None):
    import batcher.kyber as kyber

    table = pa.table({"v": list(range(200_000))})
    ds = (
        bt.from_arrow(table).sample(n=n, seed=1)
        if n is not None
        else bt.from_arrow(table).sample(fraction=fraction, seed=1)
    )
    physical = kyber.optimize(ds._plan, sources=ds._sources)
    return next(op for op in physical.ops if op.kind == "Sample")


@pytest.mark.unit
@pytest.mark.parametrize("n", [1_000, 100_000])
def test_fixed_count_sample_is_budgeted_for_its_heap(n):
    """`sample(n=)` keeps the n smallest-hash rows of the whole relation, holding a size-n
    heap — `bc_interp::ops::reshape` calls it "a breaker: it must see all rows". Budgeted by
    kind it got one morsel however large n was."""
    op = _sample_op(n=n)
    assert op.bounds.m_max_bytes >= n  # scales with n, not pinned at a morsel
    assert op.bounds.n_max_parallelism >= 1


@pytest.mark.unit
def test_fixed_count_sample_budget_tracks_n_not_the_input():
    """The heap is `min(n, input)` rows, so the budget must rise with n and then stop."""
    small, large = _sample_op(n=1_000), _sample_op(n=100_000)
    capped = _sample_op(n=10_000_000)  # far above the 200k-row input
    assert small.bounds.m_max_bytes < large.bounds.m_max_bytes
    assert capped.properties.est_rows == 200_000


@pytest.mark.unit
def test_fraction_sample_stays_streaming():
    """The contrast that makes this node-aware rather than kind-aware: a fraction sample is
    a per-row predicate holding nothing, and must keep its one-morsel envelope."""
    frac, fixed = _sample_op(fraction=0.5), _sample_op(n=100_000)
    assert frac.bounds.n_max_parallelism == 0  # inherits the pipeline width
    assert frac.bounds.m_max_bytes < fixed.bounds.m_max_bytes


@pytest.mark.unit
@pytest.mark.parametrize("n", [0, 1, 100, 5_000, 99_999])
def test_sample_results_are_unchanged_by_the_budgeting_fix(n):
    table = pa.table({"v": list(range(5_000))})
    assert bt.from_arrow(table).sample(n=n, seed=7).collect().num_rows == min(n, 5_000)


def test_provenance_is_not_over_claimed_by_either_fix():
    """Neither fix may hand the metadata-answer layer an EXACT it cannot prove."""
    ds = _ds()
    est = _est(ds)
    assert est.estimate(ds.with_row_index("i")._plan).provenance is Provenance.EXACT
    assert not est.estimate(_range_join_plan(100)._plan).rows_exact
