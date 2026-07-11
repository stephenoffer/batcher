"""Plan-shape and safety unit tests for `split_expensive_filter`.

The rule splits `Filter(cheap AND expensive)` into `Filter(expensive, Filter(cheap))` so
the expensive predicate is evaluated only on rows the cheap one kept — the engine's `and`
evaluates both operands over every row, so the fused form has no such saving.

Pinned here: it fires when (and only when) the cost model says the split pays, it puts
the cheap conjunct underneath, it leaves a conjunction of cheap comparisons fused (where
one vectorized pass genuinely wins), and it never fires on a non-conjunctive predicate.
"""

from __future__ import annotations

import dataclasses

import pytest

import batcher as bt
from batcher.config import Config, active_config
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra.filter_split import split_expensive_filter
from batcher.kyber.stats import StatsEstimator
from batcher.plan.expr_rewrite import split_conjuncts
from batcher.plan.logical import Filter, Scan


def _ctx(config: Config | None = None) -> OptimizerContext:
    cfg = config or active_config()
    estimator = StatsEstimator([], {}, cfg.optimizer.cardinality)
    return OptimizerContext(config=cfg, sources=[], hub=None, estimator=estimator)


def _config_with(**optimizer_overrides) -> Config:
    """The active config with `OptimizerConfig` fields overridden (both are frozen)."""
    cfg = active_config()
    return dataclasses.replace(
        cfg, optimizer=dataclasses.replace(cfg.optimizer, **optimizer_overrides)
    )


def _filter(pred) -> Filter:
    ds = bt.from_pydict({"x": [1, 2, 3], "y": [1, 2, 3], "s": ["a", "b", "c"]})
    return ds.filter(pred)._plan


@pytest.mark.unit
def test_registered():
    assert "split_expensive_filter" in {r.name for r in DEFAULT_REGISTRY.rules()}


@pytest.mark.unit
def test_splits_cheap_predicate_below_expensive_one():
    node = _filter((bt.col("x") > 5) & bt.col("s").str.regexp_matches("^a"))
    out = split_expensive_filter(node, _ctx())

    assert isinstance(out, Filter), "expected a split"
    assert isinstance(out.input, Filter)
    assert isinstance(out.input.input, Scan)
    # The regex ends up on top; the compiled comparison runs first, on every row.
    outer = split_conjuncts(out.predicate)
    inner = split_conjuncts(out.input.predicate)
    assert len(outer) == 1 and len(inner) == 1
    assert "regexp_matches" in str(outer[0].to_ir())
    assert inner[0].to_ir()["op"] == "gt"


@pytest.mark.unit
def test_does_not_split_two_cheap_comparisons():
    # Both compile to vector compares; one fused pass beats materializing a batch.
    node = _filter((bt.col("x") > 5) & (bt.col("y") < 9))
    assert split_expensive_filter(node, _ctx()) is None


@pytest.mark.unit
def test_does_not_split_a_single_predicate():
    assert split_expensive_filter(_filter(bt.col("s").str.regexp_matches("^a")), _ctx()) is None


@pytest.mark.unit
def test_does_not_split_a_disjunction():
    # `OR` is not a conjunction; there is nothing to stack.
    node = _filter((bt.col("x") > 5) | bt.col("s").str.regexp_matches("^a"))
    assert split_expensive_filter(node, _ctx()) is None


@pytest.mark.unit
def test_splits_two_expensive_predicates():
    # Both conjuncts are costly, but the split still wins: the fused form runs BOTH
    # regexes on every row, whereas stacking runs the second only on the ~1/3 of rows
    # the first kept. The saving is (1 - sel) x cost, which dwarfs one materialization.
    node = _filter(bt.col("s").str.regexp_matches("^a") & bt.col("s").str.regexp_matches("z$"))
    out = split_expensive_filter(node, _ctx())
    assert out is not None
    assert isinstance(out.input, Filter)


@pytest.mark.unit
def test_split_preserves_the_conjunct_set():
    pred = (bt.col("x") > 5) & bt.col("s").str.regexp_matches("^a") & (bt.col("y") < 9)
    node = _filter(pred)
    out = split_expensive_filter(node, _ctx())
    assert out is not None

    before = {str(c.to_ir()) for c in split_conjuncts(pred)}
    after = {str(c.to_ir()) for c in split_conjuncts(out.predicate)} | {
        str(c.to_ir()) for c in split_conjuncts(out.input.predicate)
    }
    assert before == after, "splitting must neither drop nor invent a conjunct"


@pytest.mark.unit
def test_min_gain_gates_the_rewrite():
    node = _filter((bt.col("x") > 5) & bt.col("s").str.regexp_matches("^a"))
    assert split_expensive_filter(node, _ctx()) is not None

    # An unreachable gain requirement must suppress the split entirely.
    strict = _config_with(filter_split_min_gain=1e9)
    assert split_expensive_filter(node, _ctx(strict)) is None


@pytest.mark.unit
def test_materialize_cost_gates_the_rewrite():
    # If materializing the intermediate batch costs more than the regex it saves,
    # the fused predicate wins.
    node = _filter((bt.col("x") > 5) & bt.col("s").str.regexp_matches("^a"))
    pricey = _config_with(filter_split_materialize_cost=1e9)
    assert split_expensive_filter(node, _ctx(pricey)) is None
