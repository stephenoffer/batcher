"""Plan-shape, idempotence, and negative unit tests for the window_rules family."""

from __future__ import annotations

import dataclasses

import batcher as bt

# Importing the module registers its @rule decorators into DEFAULT_REGISTRY.
import batcher.kyber.rules.extra.window_rules as wr
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.plan.expr_ir import Lit
from batcher.plan.logical import Window


def _base():
    return bt.from_pydict({"k": [1, 1, 2], "v": [3, 1, 2]})


_RULE_NAMES = {
    "drop_dead_window",
    "dedupe_window_partition_keys",
    "dedupe_window_order_keys",
    "drop_constant_partition_key",
    "drop_constant_order_key",
}


def test_rules_registered():
    registered = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert registered >= _RULE_NAMES


# --- drop_dead_window ---------------------------------------------------------


def _windowed(**functions):
    functions = functions or {"r": "row_number"}
    return _base().window(partition_by=["k"], order_by=["v"], functions=functions)


def test_dead_window_fully_eliminated():
    win = _windowed()._plan
    plan = _windowed().select("k", "v")._plan  # never reads the "r" column
    out = wr.drop_dead_window(plan, None)
    assert out is not None and not isinstance(out.input, Window)
    assert out.input.to_ir() == win.input.to_ir()


def test_dead_window_partial_prune():
    plan = _windowed(r="row_number", s="rank").select("k", "v", "r")._plan
    out = wr.drop_dead_window(plan, None)
    assert isinstance(out.input, Window)
    assert [f.alias for f in out.input.functions] == ["r"]  # "s" dropped, "r" kept


def test_dead_window_all_used_is_noop():
    plan = _windowed().select("k", "v", "r")._plan
    assert wr.drop_dead_window(plan, None) is None


def test_dead_window_rank_limit_not_dead():
    # A rank_limit window filters rows, so it is never dead even if unused.
    from batcher.plan.expr_ir import Col
    from batcher.plan.logical import Project, Projection

    win = dataclasses.replace(_windowed()._plan, rank_limit=1)
    proj = Project(win, (Projection("k", Col("k")), Projection("v", Col("v"))))
    assert wr.drop_dead_window(proj, None) is None


def test_dead_window_idempotent():
    plan = _windowed().select("k", "v")._plan
    once = wr.drop_dead_window(plan, None)
    assert wr.drop_dead_window(once, None) is None


# --- dedupe_window_partition_keys ---------------------------------------------


def _part(*keys):
    ds = _base().window(partition_by=list(keys), order_by=["v"], functions={"r": "row_number"})
    return ds._plan


def test_dedupe_partition_keys_fires():
    out = wr.dedupe_window_partition_keys(_part("k", "k"), None)
    assert out is not None and len(out.partition_keys) == 1


def test_dedupe_partition_keys_noop_when_distinct():
    assert wr.dedupe_window_partition_keys(_part("k", "v"), None) is None


def test_dedupe_partition_keys_idempotent():
    once = wr.dedupe_window_partition_keys(_part("k", "k"), None)
    assert wr.dedupe_window_partition_keys(once, None) is None


# --- dedupe_window_order_keys -------------------------------------------------


def test_dedupe_order_keys_fires_and_keeps_first():
    plan = _base().window(
        partition_by=["k"], order_by=[("v", False), ("v", True)], functions={"r": "row_number"}
    )._plan
    out = wr.dedupe_window_order_keys(plan, None)
    assert out is not None and len(out.order_keys) == 1
    assert out.order_keys[0].descending is False  # the first occurrence wins


def test_dedupe_order_keys_noop_when_distinct():
    plan = _base().window(
        partition_by=["k"], order_by=["v", "k"], functions={"r": "row_number"}
    )._plan
    assert wr.dedupe_window_order_keys(plan, None) is None


def test_dedupe_order_keys_idempotent():
    plan = _base().window(
        partition_by=["k"], order_by=[("v", False), ("v", True)], functions={"r": "row_number"}
    )._plan
    once = wr.dedupe_window_order_keys(plan, None)
    assert wr.dedupe_window_order_keys(once, None) is None


# --- drop_constant_partition_key ----------------------------------------------


def test_drop_constant_partition_key_fires():
    plan = _base().window(
        partition_by=[bt.lit(1), "k"], order_by=["v"], functions={"r": "row_number"}
    )._plan
    out = wr.drop_constant_partition_key(plan, None)
    assert out is not None and len(out.partition_keys) == 1
    assert not isinstance(out.partition_keys[0], Lit)


def test_drop_constant_partition_key_all_constant():
    plan = _base().window(
        partition_by=[bt.lit(1)], order_by=["v"], functions={"r": "row_number"}
    )._plan
    out = wr.drop_constant_partition_key(plan, None)
    assert out is not None and out.partition_keys == ()  # one partition over all rows


def test_drop_constant_partition_key_noop():
    plan = _base().window(partition_by=["k"], order_by=["v"], functions={"r": "row_number"})._plan
    assert wr.drop_constant_partition_key(plan, None) is None


def test_drop_constant_partition_key_idempotent():
    plan = _base().window(
        partition_by=[bt.lit(1), "k"], order_by=["v"], functions={"r": "row_number"}
    )._plan
    once = wr.drop_constant_partition_key(plan, None)
    assert wr.drop_constant_partition_key(once, None) is None


# --- drop_constant_order_key --------------------------------------------------


def test_drop_constant_order_key_fires():
    plan = _base().window(
        partition_by=["k"], order_by=["v", bt.lit(1)], functions={"c": ("sum", "v")}
    )._plan
    out = wr.drop_constant_order_key(plan, None)
    assert out is not None and len(out.order_keys) == 1
    assert not isinstance(out.order_keys[0].expr, Lit)


def test_drop_constant_order_key_all_constant_is_noop():
    # No non-constant key would remain, so the rule conservatively declines.
    plan = _base().window(
        partition_by=["k"], order_by=[bt.lit(1)], functions={"r": "row_number"}
    )._plan
    assert wr.drop_constant_order_key(plan, None) is None


def test_drop_constant_order_key_noop_when_none_constant():
    plan = _base().window(partition_by=["k"], order_by=["v"], functions={"r": "row_number"})._plan
    assert wr.drop_constant_order_key(plan, None) is None


def test_drop_constant_order_key_idempotent():
    plan = _base().window(
        partition_by=["k"], order_by=["v", bt.lit(1)], functions={"c": ("sum", "v")}
    )._plan
    once = wr.drop_constant_order_key(plan, None)
    assert wr.drop_constant_order_key(once, None) is None
