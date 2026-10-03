"""Plan-shape unit tests for semijoin/antijoin pushdown through an inner join."""

from __future__ import annotations

import batcher as bt
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY


def _find_joins(ir: dict, join_type: str) -> list[dict]:
    """All `hash_join` nodes of the given `join_type` in the IR tree."""
    out: list[dict] = []

    def rec(node: object) -> None:
        if isinstance(node, dict):
            if node.get("op") == "hash_join" and node.get("join_type") == join_type:
                out.append(node)
            for v in node.values():
                rec(v)
        elif isinstance(node, list):
            for v in node:
                rec(v)

    rec(ir)
    return out


def _session() -> bt.Session:
    # Distinct join-key names (as real schemas have), so the comma-join lowers to a
    # proper equi-join rather than tripping same-name column resolution.
    s = bt.Session()
    s.register("orders", bt.from_pydict({"o_ok": [1, 2, 3, 4], "o_ck": [10, 20, 30, 40]}))
    s.register("cust", bt.from_pydict({"c_ck": [10, 20, 30, 40], "nm": ["a", "b", "c", "d"]}))
    s.register("qualifying", bt.from_pydict({"q_ok": [1, 1, 3]}))
    return s


_QUERY = (
    "SELECT cust.nm FROM orders, cust "
    "WHERE orders.o_ck = cust.c_ck AND orders.o_ok IN (SELECT q_ok FROM qualifying)"
)


def test_rule_registered():
    assert "push_semijoin_through_join" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_semijoin_pushed_below_inner_join():
    # `orders.o_ok IN (...)` filters a column from the orders side of orders⋈cust, so the
    # semijoin must sink onto orders — below the inner join, not above it.
    plan = _session().sql(_QUERY)._plan
    ir = Optimizer().optimize(plan).ir
    semis = _find_joins(ir, "semi")
    assert semis, "expected a semi join in the plan"
    # Pushed down ⇒ the semi join's left subtree no longer contains the inner join
    # (it filters the base orders relation directly).
    assert not _find_joins(semis[0]["left"], "inner"), (
        "semi join should sit below the inner join after pushdown, not above it"
    )


def test_semijoin_pushdown_preserves_results():
    # The rewrite is semantics-preserving: orders 1 and 3 qualify → custs a, c.
    got = _session().sql(_QUERY + " ORDER BY cust.nm").collect().to_pydict()
    assert got["nm"] == ["a", "c"]


def test_antijoin_pushed_below_inner_join():
    # NOT IN → anti join; same pushdown applies (anti also only filters its left input).
    q = _QUERY.replace("IN (SELECT", "NOT IN (SELECT")
    ir = Optimizer().optimize(_session().sql(q)._plan).ir
    antis = _find_joins(ir, "anti")
    assert antis, "expected an anti join in the plan"
    assert not _find_joins(antis[0]["left"], "inner"), "anti join should sink below the inner join"


class _Rows:
    def __init__(self, rows: float) -> None:
        self.rows = rows


class _StubContext:
    """An optimizer context whose estimator answers from a fixed `plan -> rows` table."""

    def __init__(self, rows: dict[str, float]) -> None:
        self.estimator = self
        self._rows = rows

    def estimate(self, plan: str) -> _Rows:
        return _Rows(self._rows[plan])


def test_a_rounding_difference_is_a_tie_and_the_semijoin_goes_first():
    # TPC-H q18 at sf10: both filters estimated to keep all of `orders`, the inner join by
    # 7e-6 rows less. An exact comparison read that as the inner join being the more
    # selective, kept the semijoin above it, and joined all 15M orders to keep 6,398.
    from batcher.kyber.rules.joins.outer_float import semijoin_goes_first

    ctx = _StubContext({"a": 15_000_000.0, "semi": 15_000_000.0, "inner": 14_999_999.999993334})
    assert semijoin_goes_first(ctx, "a", "semi", "inner")
    # Controls: a genuinely more selective inner join still keeps the semijoin above it,
    # and a genuinely more selective semijoin still sinks.
    ctx = _StubContext({"a": 15_000_000.0, "semi": 15_000_000.0, "inner": 14_000_000.0})
    assert not semijoin_goes_first(ctx, "a", "semi", "inner")
    ctx = _StubContext({"a": 15_000_000.0, "semi": 6_398.0, "inner": 15_000_000.0})
    assert semijoin_goes_first(ctx, "a", "semi", "inner")
