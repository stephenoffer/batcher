"""Plan-shape unit tests for the `left_join_null_key_to_antijoin` rewrite.

The differential file next door proves the *answers* agree with DuckDB; these prove the plan
actually changed, which no result comparison can see — a rewrite that silently stops firing
would leave every differential test green.
"""

from __future__ import annotations

import batcher as bt
from batcher.kyber import optimize_logical
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.plan.logical import Join
from batcher.plan.visitor import walk


def _session():
    sess = bt.Session()
    sess.register(
        "l",
        bt.from_pydict({"tid": [1, 2, 3], "item": [10, 20, 30], "qty": [1, 2, 3]}).to_arrow(),
    )
    sess.register("r", bt.from_pydict({"rtid": [2], "ritem": [20], "amt": [9.0]}).to_arrow())
    return sess


def _join_types(sql: str) -> list[str]:
    sess = _session()
    plan = sess.sql(sql)._plan
    optimized = optimize_logical(plan, sources=sess._sources if hasattr(sess, "_sources") else None)
    return [n.join_type for n in walk(optimized) if isinstance(n, Join)]


def test_rule_registered():
    assert "left_join_null_key_to_antijoin" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_null_key_filter_becomes_an_anti_join():
    types = _join_types(
        "SELECT tid, qty FROM l LEFT JOIN r ON tid = rtid AND item = ritem WHERE rtid IS NULL"
    )
    assert types == ["anti"], types


def test_null_key_filter_under_an_aggregate_becomes_an_anti_join():
    """The SQL shape TPC-DS q78 writes: the consumer above the filter is an `Aggregate`."""
    types = _join_types(
        "SELECT tid, sum(qty) AS q FROM l LEFT JOIN r ON tid = rtid WHERE rtid IS NULL GROUP BY tid"
    )
    assert types == ["anti"], types


def test_a_payload_is_null_keeps_the_left_join():
    """`amt` is not a join key, so a null there does not mean "unmatched"."""
    types = _join_types(
        "SELECT tid FROM l LEFT JOIN r ON tid = rtid AND item = ritem WHERE amt IS NULL"
    )
    assert types == ["left"], types


def test_reading_a_right_column_keeps_the_left_join():
    """An anti join has no right-hand columns to give, so the rewrite must decline."""
    types = _join_types(
        "SELECT tid, amt FROM l LEFT JOIN r ON tid = rtid AND item = ritem WHERE rtid IS NULL"
    )
    assert types == ["left"], types


def test_an_inner_join_is_untouched():
    types = _join_types("SELECT tid FROM l JOIN r ON tid = rtid WHERE rtid IS NULL")
    assert "anti" not in types, types
