"""The join-order search budget must scale with the query, not with a constant.

`kyber.rules.joins.order_budget` decides how hard to search for a join order. The whole
point of it is that neither axis it replaces predicts the answer: leaf count does not
predict how much *search work* a graph costs (a 14-leaf chain evaluates 455 pairs and a
14-leaf star 53,248), and no static cap predicts what a better order is *worth* (the same
star over a thousand rows planned for 25.5 s to save microseconds).

So these tests pin both directions — a cheap query is triaged down to the floor, an
expensive one is granted more — and the bound on what a small query can spend, which is the
regression the module exists to prevent.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.config import active_config
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.rules.joins import order_budget
from batcher.kyber.stats.estimator import StatsEstimator
from batcher.plan.logical import Join, JoinOutputCol, Scan
from batcher.plan.schema import SchemaRef
from batcher.plan.source_stats import SourceStatistics


def _two_way_join(rows: int):
    """A two-leaf equi-join over sources of `rows` rows each, and a context to cost it."""
    schema = SchemaRef(pa.schema([pa.field("k", pa.int64()), pa.field("v", pa.int64())]))
    left, right = Scan(0, schema), Scan(1, schema)
    plan = Join(
        left,
        right,
        ("k",),
        ("k",),
        "inner",
        (JoinOutputCol("left", "k", "k"), JoinOutputCol("right", "v", "v")),
    )
    stats = [SourceStatistics(row_count=rows), SourceStatistics(row_count=rows)]
    est = StatsEstimator([None, None], source_stats=stats)
    ctx = OptimizerContext(config=active_config(), sources=[None, None], hub=None, estimator=est)
    return plan, ctx


def test_a_cheap_query_is_triaged_down_to_the_floor():
    """A query too small to be worth searching gets the floor, not a data-blind constant."""
    plan, ctx = _two_way_join(1_000)
    assert order_budget.search_pair_budget(plan, ctx) == order_budget._MIN_PAIRS


def test_the_budget_grows_with_the_data():
    """The same join graph over more data is granted a strictly larger search.

    This is the property the module exists for, and the one no leaf-count or flat-cap rule
    can express: the graph is identical in both calls, so every static input to the decision
    is identical too, and only the estimated cost of running it differs.
    """
    small, small_ctx = _two_way_join(1_000)
    large, large_ctx = _two_way_join(400_000_000)
    small_budget = order_budget.search_pair_budget(small, small_ctx)
    large_budget = order_budget.search_pair_budget(large, large_ctx)
    assert large_budget > small_budget
    # ...and the growth is real, not a rounding artifact: the large query clears the floor
    # the small one sits on.
    assert small_budget == order_budget._MIN_PAIRS
    assert large_budget > order_budget._MIN_PAIRS


def test_the_budget_is_monotone_in_the_data_size():
    """More data never buys *less* search — the ordering the decision depends on holds."""
    budgets = [
        order_budget.search_pair_budget(*_two_way_join(rows))
        for rows in (1_000, 1_000_000, 100_000_000, 10_000_000_000)
    ]
    assert budgets == sorted(budgets)


def test_the_budget_grows_with_the_cluster():
    """The same query on a wider fleet is granted a larger search.

    The other half of "adapts without configuration": a distributed plan pays for a shuffle
    the single-node one does not, so it costs more to run and is worth more to plan. The
    budget picks that up for free because it is priced through the cost model, which already
    takes the `HardwareProfile` — nothing here knows what a worker is.
    """
    from batcher.plan.resource import HardwareProfile

    def budget_at(workers: int) -> int:
        plan, _ = _two_way_join(2_000_000_000)
        est = StatsEstimator(
            [None, None], source_stats=[SourceStatistics(row_count=2_000_000_000)] * 2
        )
        ctx = OptimizerContext(
            config=active_config(),
            sources=[None, None],
            hub=None,
            estimator=est,
            hardware=HardwareProfile(cpu_cores=96, memory_bytes=192 * 2**30, worker_count=workers),
        )
        return order_budget.search_pair_budget(plan, ctx)

    single_node = budget_at(1)
    cluster = budget_at(64)
    assert cluster > single_node
    # Both are real searches rather than the floor, so the comparison is between two budgets
    # the cost model actually derived and not between two clamps.
    assert single_node > order_budget._MIN_PAIRS


def test_an_enormous_estimate_is_still_capped():
    """A runaway cardinality cannot buy unbounded planning time."""
    plan, ctx = _two_way_join(10**15)
    assert order_budget.search_pair_budget(plan, ctx) == order_budget._MAX_PAIRS


def test_an_unpriceable_region_searches_like_a_small_query():
    """A cost the model cannot produce yields the floor, never the ceiling.

    The direction matters: a missing estimate is not evidence that a query is large, and
    treating it as such would spend the ceiling's worth of planning on a query about which
    nothing is known.
    """

    class _NoCost:
        def costs(self):
            raise RuntimeError("no cost model")

    plan, _ = _two_way_join(1_000)
    assert order_budget.search_pair_budget(plan, _NoCost()) == order_budget._MIN_PAIRS


def _plan_pairs(build_join, n: int, rows: int = 1_000) -> int:
    """Join pairs the optimizer evaluates planning an `n`-leaf join over `rows` rows."""
    from batcher.config import OptimizerConfig
    from batcher.kyber.optimizer.facade import Optimizer
    from batcher.kyber.rules.joins import order_search

    calls = [0]
    original = order_search._join_plans

    def counting(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    ds = build_join(n, rows)
    # The plan cache would otherwise answer from a previous test's memo and count nothing.
    cfg = active_config().replace(optimizer=OptimizerConfig(plan_cache_entries=0))
    order_search._join_plans = counting
    try:
        Optimizer(cfg, ds._sources).logical_rewrite(ds._plan)
    finally:
        order_search._join_plans = original
    return calls[0]


def _star(n: int, rows: int):
    """A fact table joined to `n - 1` dimensions — the dense shape, ~2**n connected subsets."""
    import batcher as bt

    ds = bt.from_pydict(
        {
            "id": list(range(rows)),
            **{f"k{i}": [j % 7 for j in range(rows)] for i in range(n - 1)},
        }
    )
    for i in range(n - 1):
        dim = bt.from_pydict({f"d{i}": list(range(7)), f"v{i}": list(range(7))})
        ds = ds.join(dim, left_on=f"k{i}", right_on=f"d{i}", how="inner")
    return ds


def _chain(n: int, rows: int):
    """`n` tables joined end to end — the sparse shape, n(n+1)/2 connected subsets."""
    import batcher as bt

    ds = bt.from_pydict({"a0": list(range(rows)), "b0": [i % 11 for i in range(rows)]})
    for i in range(1, n):
        nxt = bt.from_pydict({f"a{i}": list(range(rows)), f"b{i}": [j % 11 for j in range(rows)]})
        ds = ds.join(nxt, left_on=f"b{i - 1}", right_on=f"a{i}", how="inner")
    return ds


def test_a_dense_graph_over_small_data_spends_nothing_discovering_it_cannot_afford_it():
    """The regression this module was written for, counted rather than timed.

    A 15-leaf star over a thousand rows evaluated 114,688 join pairs — 25.5 s of planning for
    a query that executes in milliseconds. Asserting the *pair count* rather than the elapsed
    time keeps the check deterministic on a shared machine while measuring the same thing:
    pairs are what planning time is made of (~150 us each, measured).

    Zero, not merely "few", is the assertion. The density is predicted from the connected-
    subset count before any pair is built, so an unaffordable search is declined having spent
    nothing. A version that burned the budget and *then* fell back to greedy would still be
    bounded, and would still be 500x better than the flat cap — and it would also be throwing
    away every pair it paid for. That distinction is invisible to a timing assertion at this
    scale, so it is pinned here.
    """
    assert _plan_pairs(_star, 15) == 0


def test_a_sparse_graph_keeps_its_full_search():
    """A chain the budget can afford is searched in full, so its plan does not move.

    The positive control for the test above, and the property that makes the budget safe to
    ship: the shape real queries have is sparse, and a sparse graph is cheap to search. A
    budget that triaged those to greedy as well would be trading plan quality for planning
    time nobody was spending.
    """
    assert 0 < _plan_pairs(_chain, 15) <= order_budget._MIN_PAIRS
