"""Past the DP's budget, join ordering keeps the cheaper of a left-deep and a bushy greedy tree.

The left-deep greedy builder grows one tree from the smallest leaf, so in a two-fact query
whose smallest leaf hangs off the wrong fact it walks that whole fact before anything
selective joins -- TPC-DS q72 went through 16.4M `inventory` rows that way because
`warehouse`, five rows, only joins `inventory`. Greedy operator ordering (`order_goo`) keeps a
forest and builds the selective side of the other fact first.
"""

from __future__ import annotations

import pytest

import batcher as bt
import batcher.kyber.rules.joins.order as order
from batcher.config import active_config
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.rules.joins.order import reorder_joins

pytestmark = pytest.mark.unit

# Two facts sharing `item`: `inv` carries the tiny `wh` dimension, `sales` carries the
# selective `promo` and `cust` dimensions. The smallest leaf is `wh`.
_QUERY = """
SELECT count(*) AS n, sum(s_qty) AS q
FROM sales
JOIN inv ON s_item = i_item
JOIN wh ON i_wh = w_sk
JOIN promo ON s_promo = p_sk
JOIN cust ON s_cust = c_sk
WHERE p_kind = 'rare' AND c_flag = 1
"""


def _session() -> bt.Session:
    n_inv, n_sales = 40_000, 20_000
    sess = bt.Session()
    sess.register(
        "inv",
        bt.from_pydict(
            {"i_item": [i % 500 for i in range(n_inv)], "i_wh": [i % 3 for i in range(n_inv)]}
        ),
    )
    sess.register("wh", bt.from_pydict({"w_sk": [0, 1, 2], "w_name": ["a", "b", "c"]}))
    sess.register(
        "sales",
        bt.from_pydict(
            {
                "s_item": [i % 500 for i in range(n_sales)],
                "s_promo": [i % 100 for i in range(n_sales)],
                "s_cust": [i % 1_000 for i in range(n_sales)],
                "s_qty": [i % 7 for i in range(n_sales)],
            }
        ),
    )
    sess.register(
        "promo",
        bt.from_pydict(
            {"p_sk": list(range(100)), "p_kind": ["rare" if i == 3 else "x" for i in range(100)]}
        ),
    )
    sess.register(
        "cust",
        bt.from_pydict(
            {"c_sk": list(range(1_000)), "c_flag": [int(i % 50 == 3) for i in range(1_000)]}
        ),
    )
    return sess


def _ctx(ds) -> OptimizerContext:
    return OptimizerContext(
        config=active_config(),
        sources=ds._sources,
        hub=None,
        estimator=CardinalityEstimator(ds._sources, {}),
    )


@pytest.fixture
def no_dp_budget(monkeypatch):
    """Force the DP to bail, as it does on a graph its budget cannot cover."""
    monkeypatch.setattr(order, "search_pair_budget", lambda region, ctx: 0)


def test_the_fallback_is_no_costlier_than_the_left_deep_tree(no_dp_budget, monkeypatch) -> None:
    ds = _session().sql(_QUERY)
    ctx = _ctx(ds)
    both = reorder_joins(ds._plan, ctx)
    monkeypatch.setattr(order, "rebuild_goo", lambda *a, **k: None)
    left_deep = reorder_joins(ds._plan, ctx)
    cost = ctx.costs()
    assert cost.cost(both).total() <= cost.cost(left_deep).total()
    assert cost.cost(both).total() < cost.cost(left_deep).total(), (
        "control: on this shape the left-deep tree from the 3-row leaf is the costlier one"
    )


def test_the_fallback_returns_the_same_rows(no_dp_budget) -> None:
    got = _session().sql(_QUERY).collect().to_pydict()
    n_sales = 20_000
    # Brute force over the generators above: a sale qualifies when its promo is 3 and its
    # customer is 3 mod 50; each matches the 80 inventory rows of its item, each of
    # which matches exactly one warehouse.
    rows = [i for i in range(n_sales) if i % 100 == 3 and (i % 1_000) % 50 == 3]
    assert len(rows) == 200, "control: the fixture must leave rows for the join to find"
    assert got == {"n": [len(rows) * 80], "q": [sum((i % 7) * 80 for i in rows)]}
