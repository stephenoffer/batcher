"""Plan-shape tests for the `kyber.rules.aggregate_algebra.extremes` family.

The quantile rules must collapse only the *extreme* percentiles, must leave an integer
column alone (where the rewrite would move the output type from Float64 to Int64), and
must leave an interior percentile alone. The `arg_min`/`arg_max` rules must fire only when
the ordering key is the measure itself.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import optimize_logical
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.plan.expr_ir import AggExpr


def _ds():
    return bt.from_pydict({"g": ["a", "a", "b"], "f": [1.0, 2.0, 3.0], "i": [1, 2, 3]})


def _rule(name: str):
    for r in DEFAULT_REGISTRY.rules():
        if r.name == name:
            return r
    raise AssertionError(f"rule {name!r} is not registered")


def _agg(expr: AggExpr):
    return _ds().group_by("g").agg(r=expr)._plan


def _fire(name: str, expr: AggExpr) -> AggExpr:
    node = _agg(expr)
    out = _rule(name).apply(node, None)
    assert out.to_ir() != node.to_ir(), f"{name} did not fire"
    return out.aggregates[0].agg


def _noop(name: str, expr: AggExpr) -> None:
    node = _agg(expr)
    assert _rule(name).apply(node, None).to_ir() == node.to_ir()


@pytest.mark.parametrize("fn", ["quantile", "approx_quantile"])
@pytest.mark.parametrize(("param", "want"), [(0.0, "min"), (1.0, "max")])
def test_extreme_quantile_becomes_an_extreme(fn, param, want):
    got = _fire("extreme_quantile_to_min_max", AggExpr(fn, col("f"), param=param))
    assert got.func == want
    assert got.param is None
    assert got.input.to_ir() == col("f").to_ir()


@pytest.mark.parametrize("param", [0.25, 0.5, 0.99])
def test_interior_quantile_is_left_alone(param):
    _noop("extreme_quantile_to_min_max", AggExpr("quantile", col("f"), param=param))


@pytest.mark.parametrize("param", [0.0, 1.0])
def test_quantile_over_an_integer_column_is_left_alone(param):
    # `quantile` answers Float64 while `min` preserves Int64; rewriting would move the
    # output column's type.
    _noop("extreme_quantile_to_min_max", AggExpr("quantile", col("i"), param=param))


@pytest.mark.parametrize(("fn", "want"), [("arg_min", "min"), ("arg_max", "max")])
def test_self_ordered_arg_extreme_collapses(fn, want):
    got = _fire("self_ordered_arg_extreme_to_min_max", AggExpr(fn, col("f"), input2=col("f")))
    assert got.func == want
    assert got.input2 is None


@pytest.mark.parametrize("fn", ["arg_min", "arg_max"])
def test_arg_extreme_ordered_by_another_column_is_left_alone(fn):
    _noop("self_ordered_arg_extreme_to_min_max", AggExpr(fn, col("f"), input2=col("i")))


def test_arg_extreme_collapses_on_an_integer_column_too():
    # No type guard is needed: arg_min and min both preserve the input type.
    got = _fire(
        "self_ordered_arg_extreme_to_min_max", AggExpr("arg_min", col("i"), input2=col("i"))
    )
    assert got.func == "min"


def test_only_the_matching_aggregate_is_rewritten():
    plan = (
        _ds()
        .group_by("g")
        .agg(a=col("f").quantile(0.0), b=col("f").quantile(0.5), c=col("f").sum())
        ._plan
    )
    out = _rule("extreme_quantile_to_min_max").apply(plan, None)
    assert [spec.agg.func for spec in out.aggregates] == ["min", "quantile", "sum"]


@pytest.mark.parametrize(
    "expr",
    [
        AggExpr("quantile", col("f"), param=0.0),
        AggExpr("approx_quantile", col("f"), param=1.0),
        AggExpr("arg_min", col("f"), input2=col("f")),
    ],
)
def test_optimizer_is_idempotent(expr):
    once = optimize_logical(_agg(expr))
    assert optimize_logical(once).to_ir() == once.to_ir()
