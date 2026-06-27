"""Plan-shape + result tests for dropping a comma-join's redundant `__cross_key`."""

from __future__ import annotations

import batcher as bt
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY


def _join_key_counts(ir: object, out: list[int]) -> list[int]:
    if isinstance(ir, dict):
        if ir.get("op") == "hash_join":
            out.append(len(ir.get("left_keys") or []))
        for v in ir.values():
            _join_key_counts(v, out)
    elif isinstance(ir, list):
        for v in ir:
            _join_key_counts(v, out)
    return out


def _has_cross_key(ir: object) -> bool:
    if isinstance(ir, dict):
        if ir.get("op") == "hash_join" and "__cross_key" in (ir.get("left_keys") or []):
            return True
        return any(_has_cross_key(v) for v in ir.values())
    if isinstance(ir, list):
        return any(_has_cross_key(v) for v in ir)
    return False


def _session() -> bt.Session:
    s = bt.Session()
    s.register("cust", bt.from_pydict({"c_ck": [1, 2, 3], "nm": ["a", "b", "c"]}))
    s.register("orders", bt.from_pydict({"o_ck": [1, 1, 3], "o_ok": [10, 11, 12]}))
    s.register("line", bt.from_pydict({"l_ok": [10, 10, 12], "qty": [5, 7, 9]}))
    return s


def test_rule_registered():
    assert "drop_redundant_cross_key" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_cross_key_dropped_under_semijoin():
    # A 3-table comma join with a semijoin between the join-condition filter and the
    # comma join (the TPC-H Q18 shape): the redundant `__cross_key` must still be dropped.
    s = _session()
    plan = s.sql(
        "SELECT c_ck FROM cust, orders WHERE c_ck = o_ck "
        "AND o_ok IN (SELECT l_ok FROM line GROUP BY l_ok HAVING sum(qty) > 6)"
    )._plan
    ir = Optimizer().optimize(plan).ir
    assert not _has_cross_key(ir), "the redundant __cross_key should be dropped"


def test_results_preserved():
    # order 10 has qty 5+7=12 > 6 → qualifies; order 12 has qty 9 > 6 → qualifies.
    # cust 1 owns order 10, cust 3 owns order 12 → both kept.
    s = _session()
    got = s.sql(
        "SELECT nm FROM cust, orders WHERE c_ck = o_ck "
        "AND o_ok IN (SELECT l_ok FROM line GROUP BY l_ok HAVING sum(qty) > 6) ORDER BY nm"
    ).collect()
    assert got.to_pydict()["nm"] == ["a", "c"]
