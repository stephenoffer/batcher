"""Plan-shape + result tests for dropping a redundant Distinct under a semi/anti join."""

from __future__ import annotations

import batcher as bt
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY


def _has_distinct(ir: object) -> bool:
    if isinstance(ir, dict):
        return ir.get("op") == "distinct" or any(_has_distinct(v) for v in ir.values())
    if isinstance(ir, list):
        return any(_has_distinct(v) for v in ir)
    return False


def _session() -> bt.Session:
    s = bt.Session()
    s.register("cust", bt.from_pydict({"c_ck": [1, 2, 3, 4], "nm": ["a", "b", "c", "d"]}))
    # Duplicate keys on the build side — the dedup the rule proves redundant.
    s.register("orders", bt.from_pydict({"o_ck": [1, 1, 1, 3, 3]}))
    return s


def test_rule_registered():
    assert "drop_redundant_distinct_build" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_not_exists_drops_build_distinct():
    # NOT EXISTS → anti join; the decorrelated build-side Distinct is removed.
    s = _session()
    plan = s.sql(
        "SELECT nm FROM cust WHERE NOT EXISTS (SELECT * FROM orders WHERE o_ck = c_ck)"
    )._plan
    ir = Optimizer().optimize(plan).ir
    assert not _has_distinct(ir), "the redundant build-side distinct should be dropped"


def test_results_preserved_anti():
    # custs 1 and 3 have orders → NOT EXISTS keeps 2 and 4.
    s = _session()
    got = s.sql(
        "SELECT nm FROM cust WHERE NOT EXISTS (SELECT * FROM orders WHERE o_ck = c_ck) ORDER BY nm"
    ).collect()
    assert got.to_pydict()["nm"] == ["b", "d"]


def test_results_preserved_semi():
    # IN → semi join; custs 1 and 3 match.
    s = _session()
    got = s.sql("SELECT nm FROM cust WHERE c_ck IN (SELECT o_ck FROM orders) ORDER BY nm").collect()
    assert got.to_pydict()["nm"] == ["a", "c"]
