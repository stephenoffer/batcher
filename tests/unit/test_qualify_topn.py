"""Plan-shape unit tests for `qualify_to_partition_topn`."""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.fusion import _rank_bound, qualify_to_partition_topn
from batcher.plan.expr_ir import Col, Lit
from batcher.plan.logical import Filter, Window


def _ranked(fn="row_number"):
    return bt.from_pydict({"k": [1, 1, 2], "v": [3, 1, 2]}).window(
        partition_by=["k"], order_by=["v"], functions={"r": fn}
    )


def _ir_ops(ir, op, out=None):
    out = [] if out is None else out
    if isinstance(ir, dict):
        if ir.get("op") == op:
            out.append(ir)
        for v in ir.values():
            _ir_ops(v, op, out)
    elif isinstance(ir, list):
        for v in ir:
            _ir_ops(v, op, out)
    return out


def test_rule_registered():
    assert "qualify_to_partition_topn" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_fuses_and_drops_filter():
    plan = _ranked().filter(col("r") <= 2)._plan
    out = qualify_to_partition_topn(plan, None)
    assert isinstance(out, Window) and out.rank_limit == 2  # filter fully absorbed


def test_strict_lt_subtracts_one():
    plan = _ranked().filter(col("r") < 3)._plan
    out = qualify_to_partition_topn(plan, None)
    assert isinstance(out, Window) and out.rank_limit == 2


def test_extra_predicate_kept_as_filter():
    plan = _ranked().filter((col("r") <= 2) & (col("v") > 0))._plan
    out = qualify_to_partition_topn(plan, None)
    assert isinstance(out, Filter) and isinstance(out.input, Window)
    assert out.input.rank_limit == 2


def test_lower_bound_does_not_fuse():
    # `r >= 2` is not a top-N bound.
    plan = _ranked().filter(col("r") >= 2)._plan
    assert qualify_to_partition_topn(plan, None) is None


def test_non_rank_predicate_is_noop():
    plan = _ranked().filter(col("v") > 1)._plan
    assert qualify_to_partition_topn(plan, None) is None


def test_idempotent():
    plan = _ranked().filter(col("r") <= 2)._plan
    once = qualify_to_partition_topn(plan, None)
    # `once` is a bare Window now (no Filter parent) → rule no longer matches it.
    assert qualify_to_partition_topn(once, None) is None


def test_full_optimizer_fuses():
    plan = _ranked().filter(col("r") <= 2).select("k", "v")._plan
    ir = Optimizer().optimize(plan).ir
    wins = _ir_ops(ir, "window")
    assert len(wins) == 1 and wins[0]["rank_limit"] == 2
    assert _ir_ops(ir, "filter") == []  # the QUALIFY filter is gone


def test_ir_round_trip_rank_limit():
    # The fused window's rank_limit survives lowering to IR (wire contract).
    fused = qualify_to_partition_topn(_ranked().filter(col("r") <= 5)._plan, None)
    assert fused.to_ir()["rank_limit"] == 5


# --- _rank_bound helper -------------------------------------------------------


def test_rank_bound_le():
    assert _rank_bound(Col("r") <= Lit(4), "r") == 4


def test_rank_bound_lt():
    assert _rank_bound(Col("r") < Lit(4), "r") == 3


def test_rank_bound_flipped():
    # `5 >= r` ≡ `r <= 5`
    assert _rank_bound(Lit(5) >= Col("r"), "r") == 5


def test_rank_bound_eq_one_only():
    assert _rank_bound(Col("r") == Lit(1), "r") == 1
    assert _rank_bound(Col("r") == Lit(2), "r") is None  # `= 2` is not a top-N


def test_rank_bound_wrong_column():
    assert _rank_bound(Col("v") <= Lit(2), "r") is None
