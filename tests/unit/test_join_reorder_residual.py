"""A cross-relation filter inside a join region no longer freezes the join under it.

Predicate pushdown parks a two-relation predicate (`a.x < b.y`) on the lowest join whose
output has both columns, and that join then looked like an opaque leaf to reordering. The
rule now hoists such a filter, searches the wider graph, and re-attaches the predicate above
the first join that makes it evaluable. These tests pin both halves of that: the shape does
open up, and the relation it computes does not change.
"""

from __future__ import annotations

import batcher as bt
from batcher.config import active_config
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.rules.joins.order import reorder_joins
from batcher.kyber.rules.pushdown import rewrite_predicate
from batcher.plan.expr_ir import referenced_columns
from batcher.plan.logical import Filter, Join
from batcher.plan.visitor import walk


def _tables():
    """A two-fact star: both facts key on `item`, and a `qty` comparison spans them.

    Shaped after TPC-DS q72, where `inv_quantity_on_hand < cs_quantity` reads `inventory`
    and `catalog_sales`. `item` has few distinct values on both sides, so joining the facts
    to each other first is the expensive order and joining each to its dimension first is
    the cheap one.
    """
    sales = bt.from_pydict(
        {
            "item": [1, 2, 3, 1, 2, 3, 1, 2],
            "day": [10, 10, 11, 11, 12, 12, 13, 13],
            "qty": [5, 6, 7, 8, 9, 10, 11, 12],
        }
    )
    inv = bt.from_pydict(
        {
            "item": [1, 1, 2, 2, 3, 3, 1, 2, 3, 1],
            "on_hand": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "wh": [1, 1, 2, 2, 1, 2, 1, 2, 1, 2],
        }
    )
    days = bt.from_pydict({"day": [10, 11, 12, 13], "label": ["a", "b", "c", "d"]})
    return sales, inv, days


def _ctx(ds):
    return OptimizerContext(
        config=active_config(),
        sources=ds._sources,
        hub=None,
        estimator=CardinalityEstimator(ds._sources, {}),
    )


def _query():
    sales, inv, days = _tables()
    joined = sales.join(inv, on="item").join(days, on="day")
    return joined.filter(bt.col("on_hand") < bt.col("qty"))


def test_a_cross_relation_filter_no_longer_walls_off_the_join_below_it():
    ds = _query()
    pushed = rewrite_predicate(ds._plan)
    # Pushdown parks the two-relation predicate between the joins: that is the shape this
    # rule has to see through, so if it stops happening the test below stops testing anything.
    assert any(isinstance(n, Filter) and isinstance(n.input, Join) for n in walk(pushed)), (
        "expected pushdown to place the cross-relation filter directly on a join"
    )
    out = reorder_joins(pushed, _ctx(ds))
    assert out.to_ir() != pushed.to_ir(), "the region under the filter was never reordered"


def test_the_hoisted_predicate_is_applied_exactly_once():
    ds = _query()
    reordered = reorder_joins(rewrite_predicate(ds._plan), _ctx(ds))
    applied = [
        n
        for n in walk(reordered)
        if isinstance(n, Filter) and {"on_hand", "qty"} <= set(referenced_columns(n.predicate))
    ]
    assert len(applied) == 1, f"predicate applied {len(applied)} times, must be exactly once"


def test_the_result_is_unchanged_by_reordering():
    sales, inv, days = _tables()
    s, i, d = (t.collect().to_pydict() for t in (sales, inv, days))
    label = dict(zip(d["day"], d["label"], strict=True))
    expected = sorted(
        (item, day, qty, oh, wh, label[day])
        for item, day, qty in zip(s["item"], s["day"], s["qty"], strict=True)
        for j, oh, wh in zip(i["item"], i["on_hand"], i["wh"], strict=True)
        if j == item and oh < qty and day in label
    )
    got = _query().select("item", "day", "qty", "on_hand", "wh", "label").collect().to_pydict()
    assert (
        sorted(
            zip(
                got["item"],
                got["day"],
                got["qty"],
                got["on_hand"],
                got["wh"],
                got["label"],
                strict=True,
            )
        )
        == expected
    )
