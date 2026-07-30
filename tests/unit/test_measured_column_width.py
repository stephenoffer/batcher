"""A column's byte width must be measured, and must survive the narrowing that drops bounds.

Two coupled defects. A column of 2 KB documents was priced at the flat 36-byte string prior,
56x under -- and width is not cosmetic: the memory envelope, the morsel row cap, broadcast
eligibility and the task fan-out are all derived from bytes-per-row, so under-estimating it
sizes every one of them too permissively, which is the direction that OOMs.

The second is why the first was invisible on the paths that matter. The conductor narrows
per-column statistics to `column_bounds_needed(plan)`, because computing *bounds* is an
O(rows) pass. But that narrowing gated every column fact, so a plan naming no predicate
columns at all -- a `group_by`, a plain scan -- got no statistics whatsoever and fell back to
the prior, while the identical source under a `filter` reported the truth.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.source_stats import collect_source_stats, column_bounds_needed
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.cost import CostModel
from batcher.plan.stats import Provenance

pytestmark = pytest.mark.unit

_ROWS = 2_000


def _stats(ds, plan):
    """Exactly what the conductor passes: narrowed to the plan's predicate columns."""
    return collect_source_stats(ds._sources, None, need_columns=column_bounds_needed(plan))


def _row_bytes(ds, plan) -> float:
    return CostModel(CardinalityEstimator(ds._sources, source_stats=_stats(ds, plan))).row_bytes(
        plan
    )


def _frame(**columns):
    return bt.from_arrow(pa.table({k: pa.array(v) for k, v in columns.items()}))


@pytest.mark.parametrize(
    ("label", "values", "expected"),
    [
        ("2 KB documents", ["x" * 2000] * _ROWS, 2004.0),
        ("empty strings", [""] * _ROWS, 4.0),
        ("short tokens", ["ab"] * _ROWS, 6.0),
    ],
)
def test_a_scan_is_priced_at_the_measured_width(label, values, expected):
    """The 36-byte string prior stood in for a number Arrow already tracks."""
    ds = _frame(d=values)
    assert _row_bytes(ds, ds._plan) == pytest.approx(expected, rel=0.01), label


def test_a_plan_with_no_predicate_still_gets_column_statistics():
    """The regression: `column_bounds_needed` is empty for a `group_by`, and that emptied
    the whole statistics set rather than only the bounds."""
    ds = _frame(d=["x" * 2000] * _ROWS, g=list(np.arange(_ROWS) % 10))
    grouped = ds.group_by("g").agg(n=bt.col("d").count())._plan
    stat = _stats(ds, grouped)[0].columns.get("d")
    assert stat is not None
    assert stat.avg_bytes == pytest.approx(2004.0, rel=0.01)


def test_the_two_plan_shapes_agree_on_the_width():
    """The sharpest form: the same source, the same column, under a plan that needs bounds
    and one that does not. The width may not depend on whether a predicate exists."""
    ds = _frame(d=["x" * 2000] * _ROWS)
    under_filter = _stats(ds, ds.filter(bt.col("d") != "")._plan)[0].columns["d"].avg_bytes
    under_scan = _stats(ds, ds._plan)[0].columns["d"].avg_bytes
    assert under_filter == under_scan == pytest.approx(2004.0, rel=0.01)


def test_the_cheap_facts_are_exact_and_carry_the_null_count():
    ds = _frame(d=["a"] * 900 + [None] * 100)
    stat = _stats(ds, ds._plan)[0].columns["d"]
    assert stat.null_count == 100.0
    assert stat.null_count_provenance is Provenance.EXACT
    assert stat.avg_bytes is not None


def test_a_bounded_column_keeps_its_bounds():
    """The safety property: adding cheap facts must not displace a real bounds pass."""
    ds = _frame(n=list(range(_ROWS)))
    stat = _stats(ds, ds.filter(bt.col("n") > 5)._plan)[0].columns["n"]
    assert (stat.min, stat.max) == (0, _ROWS - 1)
    assert stat.provenance is Provenance.EXACT


def test_the_width_is_memoized_per_source():
    """It is a constant of an immutable source, so a second query must not recompute it.

    Without the memo this is a fixed per-query control-plane cost that grows with column
    count -- hurting the small case to help the large one.
    """
    ds = _frame(**{f"c{i}": list(range(100)) for i in range(20)})
    source = ds._sources[0]
    first = [source.column_cheap_stat(f"c{i}") for i in range(20)]
    assert all(s is source.column_cheap_stat(f"c{i}") for i, s in enumerate(first))


def test_wide_rows_still_produce_the_right_answer():
    """A width is a sizing input, so it may never change the relation."""
    ds = _frame(d=["x" * 2000] * 100, g=list(np.arange(100) % 4))
    out = ds.group_by("g").agg(n=bt.col("d").count()).collect().to_pydict()
    assert sorted(out["n"]) == [25, 25, 25, 25]
