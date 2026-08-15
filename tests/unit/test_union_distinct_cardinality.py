"""A `UNION` is a `DISTINCT` over the concatenation, and must be counted as one.

`_estimate_union` reasoned from the branches' **row counts** — `union_ndv([n_1, n_2])`, which
models each branch as contributing `n_i` distinct values — and then floored the result at the
largest branch's row count. Both halves are wrong whenever the branches are wider than they
are deep: two 10,000-row branches over a 200-value column estimated 10,000 rows against 200
actual, a 50x over-estimate on the default spelling of `UNION` in SQL.

It was also self-contradictory. `col_prop.union_columns` already merges the branches' *column*
distinct counts correctly, so the same node reported a column with 200 distinct values inside
10,000 output rows. And the optimizer builds this node itself — it rewrites
`Distinct(Union(all))` into `Union(distinct)` — so a plan that estimated correctly *before*
optimization estimated 50x worse after it, which is the shape that makes a rewrite look like a
regression somewhere else entirely.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.config import active_config
from batcher.kyber.optimizer.facade import optimize_logical
from batcher.kyber.stats import StatsEstimator

pytestmark = pytest.mark.unit

_CFG = active_config().optimizer.cardinality
_ROWS = 4000
_KEYS = 200


def _estimate(dataset, *, optimized: bool = False) -> float:
    stats = [s.statistics() for s in dataset._sources]
    plan = dataset._plan
    if optimized:
        plan = optimize_logical(plan, sources=dataset._sources, source_stats=stats)
    return StatsEstimator(dataset._sources, {}, _CFG, source_stats=stats).estimate(plan).rows


@pytest.fixture
def narrow():
    """4,000 rows over 200 key values — far more rows than distinct values."""
    return bt.from_pydict({"k": [i % _KEYS for i in range(_ROWS)]})


def test_a_union_counts_distinct_values_not_rows(narrow):
    united = narrow.union(narrow).distinct()
    assert united.count() == _KEYS
    assert _estimate(united) == pytest.approx(_KEYS, rel=0.05)


def test_the_optimized_plan_agrees_with_the_unoptimized_one(narrow):
    """`Distinct(Union(all))` and the `Union(distinct)` it is rewritten to are one relation."""
    united = narrow.union(narrow).distinct()
    assert _estimate(united, optimized=True) == pytest.approx(_estimate(united), rel=0.05)


def test_union_all_still_concatenates(narrow):
    """The dedup path must not touch `UNION ALL`, which really does add the rows."""
    both = narrow.union(narrow)
    assert both.count() == 2 * _ROWS
    assert _estimate(both) == pytest.approx(2 * _ROWS)


def test_the_row_count_agrees_with_the_propagated_column_statistics(narrow):
    """The node cannot claim more rows than its own merged column ndv allows."""
    united = narrow.union(narrow).distinct()
    stats = [s.statistics() for s in united._sources]
    result = StatsEstimator(united._sources, {}, _CFG, source_stats=stats).estimate(united._plan)
    assert result.columns["k"].ndv is not None
    assert result.rows <= result.columns["k"].ndv * 1.05


def test_the_estimate_stays_inside_the_frechet_bounds():
    """`max_i d_i <= |A union B| <= sum_i d_i`, and never above the concatenated rows.

    Disjoint branches sit at the top of that range. The estimate inherits `union_ndv`'s
    deliberate low bias on the *column* side, so it is not required to land on the sum — only
    to stay inside the bounds and far below the row count, which is what the old row-count
    model violated.
    """
    left = bt.from_pydict({"k": list(range(100))})
    right = bt.from_pydict({"k": list(range(100, 200))})
    united = left.union(right).distinct()
    assert united.count() == 200
    estimated = _estimate(united)
    assert 100 <= estimated <= 200


def test_a_multi_column_union_is_damped_not_multiplied():
    """The same `combine_ndv` a `DISTINCT` over the same columns would use."""
    frame = bt.from_pydict(
        {"a": [i % 50 for i in range(_ROWS)], "b": [i % 20 for i in range(_ROWS)]}
    )
    united = frame.union(frame).distinct()
    direct = frame.union(frame).select("a", "b").distinct()
    assert _estimate(united) == pytest.approx(_estimate(direct), rel=0.05)
    assert _estimate(united) <= 2 * _ROWS


def test_an_empty_union_stays_provably_empty():
    """Deduplicating no rows yields no rows — and the *proof* must survive, not just the count.

    `Distinct` and a grouped `Aggregate` both floored their estimate at one row, which is not a
    rounding error: it destroys the emptiness proof, so `count()` over a pruned-to-empty
    subtree stopped answering from metadata and executed the operator instead.
    """
    empty = bt.from_pydict({"k": [], "v": []})
    stats = [s.statistics() for s in empty._sources]

    def result(dataset):
        return StatsEstimator(
            dataset._sources, {}, _CFG, source_stats=[s.statistics() for s in dataset._sources]
        ).estimate(dataset._plan)

    assert stats  # the fixture really does carry an exact zero
    for dataset in (
        empty.union(empty).distinct(),
        empty.select("k").distinct(),
        empty.group_by("k").agg(s=bt.col("v").sum()),
    ):
        estimate = result(dataset)
        assert estimate.rows == pytest.approx(0.0)
        assert estimate.rows_exact, "the emptiness proof was lost"
        assert dataset.count() == 0

    # A *global* aggregate is the opposite case: it emits its one row over an empty input.
    global_agg = result(empty.agg(s=bt.col("v").sum()))
    assert global_agg.rows == pytest.approx(1.0)
    assert empty.agg(s=bt.col("v").sum()).count() == 1
