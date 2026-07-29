"""Plan-shape tests for the `kyber.rules.window_algebra` family.

`nth_value(x, 1)` must become `first_value(x)` while keeping its alias, input and frame,
and `nth_value(x, n)` for any other `n` must be left exactly as written — there is no
specialized function naming the second row.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import optimize_logical
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.plan.logical import Window
from batcher.plan.visitor import walk


def _ds():
    return bt.from_pydict({"g": ["a", "a", "b"], "v": [3, 1, 2], "o": [1, 2, 1]})


def _rule(name: str):
    for r in DEFAULT_REGISTRY.rules():
        if r.name == name:
            return r
    raise AssertionError(f"rule {name!r} is not registered")


def _specs(plan):
    return [f for node in walk(plan) if isinstance(node, Window) for f in node.functions]


def _windowed(**columns):
    return _ds().with_columns(**columns)._plan


def test_nth_value_at_one_becomes_first_value():
    plan = _windowed(r=bt.nth_value(col("v"), 1).over(partition_by=["g"], order_by=["o"]))
    out = _rule("nth_value_at_one_to_first_value").apply(plan, None)
    (spec,) = _specs(out)
    assert spec.func == "first_value"
    assert spec.alias == "r"
    assert spec.input.to_ir() == col("v").to_ir()


@pytest.mark.parametrize("n", [2, 3, 10])
def test_nth_value_beyond_one_is_left_alone(n):
    plan = _windowed(r=bt.nth_value(col("v"), n).over(partition_by=["g"], order_by=["o"]))
    out = _rule("nth_value_at_one_to_first_value").apply(plan, None)
    assert out.to_ir() == plan.to_ir()


def test_only_the_matching_spec_is_rewritten():
    plan = _windowed(
        a=bt.nth_value(col("v"), 1).over(partition_by=["g"], order_by=["o"]),
        b=bt.nth_value(col("v"), 2).over(partition_by=["g"], order_by=["o"]),
    )
    out = _rule("nth_value_at_one_to_first_value").apply(plan, None)
    assert sorted(spec.func for spec in _specs(out)) == ["first_value", "nth_value"]


def test_the_partition_and_order_keys_survive():
    plan = _windowed(r=bt.nth_value(col("v"), 1).over(partition_by=["g"], order_by=["o"]))
    out = _rule("nth_value_at_one_to_first_value").apply(plan, None)
    original = next(n for n in walk(plan) if isinstance(n, Window))
    rewritten = next(n for n in walk(out) if isinstance(n, Window))
    assert [k.to_ir() for k in rewritten.partition_keys] == [
        k.to_ir() for k in original.partition_keys
    ]
    assert len(rewritten.order_keys) == len(original.order_keys)


def test_optimizer_reaches_the_specialized_form():
    plan = optimize_logical(
        _windowed(r=bt.nth_value(col("v"), 1).over(partition_by=["g"], order_by=["o"]))
    )
    assert [spec.func for spec in _specs(plan)] == ["first_value"]


def test_optimizer_is_idempotent():
    plan = _windowed(r=bt.nth_value(col("v"), 1).over(partition_by=["g"], order_by=["o"]))
    once = optimize_logical(plan)
    assert optimize_logical(once).to_ir() == once.to_ir()
