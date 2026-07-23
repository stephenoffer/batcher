"""Common-subexpression elimination binds a repeated expression to one column.

The rule must fire when the repeat is expensive enough to pay for the column it costs,
must NOT fire when it is not (a bound cheap expression is a pessimization), and must never
change what the projection computes or the schema it produces.
"""

from __future__ import annotations

import logging

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.rules.extra.cse import CSE_TEMP_PREFIX
from batcher.plan.logical import Project
from batcher.plan.visitor import walk

pytestmark = pytest.mark.unit


def _optimized(ds) -> object:
    return Optimizer(sources=ds._sources).optimize_full(ds._plan)[1]


def _bound_columns(plan) -> set[str]:
    """The distinct synthetic columns the rule bound.

    A binding is *carried forward* by each projection above the one that defines it, so the
    same name legitimately appears more than once in the plan; the set is what "how many
    subexpressions were bound" means.
    """
    return {
        item.alias
        for node in walk(plan)
        if isinstance(node, Project)
        for item in node.items
        if item.alias.startswith(CSE_TEMP_PREFIX)
    }


def _ds():
    return bt.from_arrow(pa.table({"url": ["a-b-c", "x-y-z", None], "n": [1, 2, 3]}))


def test_expensive_repeated_subexpression_is_bound_once():
    e = col("url").str.regexp_replace("-", "+")
    ds = _ds().select(a=e, b=e.str.upper(), c=e.str.len())
    bound = _bound_columns(_optimized(ds))
    assert len(bound) == 1, f"expected one binding, got {bound}"


def test_cheap_repeated_subexpression_is_left_alone():
    """Binding `n + 1` would cost a column to save a vectorized add — a pessimization."""
    cheap = col("n") + 1
    ds = _ds().select(a=cheap, b=cheap * 2, c=cheap * 3)
    assert _bound_columns(_optimized(ds)) == set()


def test_a_subexpression_used_once_is_not_bound():
    e = col("url").str.regexp_replace("-", "+")
    ds = _ds().select(a=e, b=col("n") + 1)
    assert _bound_columns(_optimized(ds)) == set()


def test_single_output_column_is_never_rewritten():
    e = col("url").str.regexp_replace("-", "+")
    ds = _ds().select(a=e)
    assert _bound_columns(_optimized(ds)) == set()


def test_rewrite_preserves_output_schema_and_values():
    e = col("url").str.regexp_replace("-", "+")
    ds = _ds().select(a=e, b=e.str.upper(), c=e.str.len())
    out = ds.collect()
    assert out.column_names == ["a", "b", "c"]  # order and names unchanged
    assert out.to_pydict() == {
        "a": ["a+b-c", "x+y-z", None],
        "b": ["A+B-C", "X+Y-Z", None],
        "c": [5, 5, None],
    }


def test_no_synthetic_column_leaks_into_the_result():
    e = col("url").str.regexp_replace("-", "+")
    out = _ds().select(a=e, b=e.str.upper()).collect()
    assert not any(c.startswith(CSE_TEMP_PREFIX) for c in out.column_names)


def test_nested_repeats_share_rather_than_recompute():
    """`f(url)` inside `f(url)||x` — the inner bind is reused by the outer, not redone."""
    inner = col("url").str.regexp_replace("-", "+")
    outer = inner.str.regexp_replace("a", "z")
    ds = _ds().select(a=inner, b=inner.str.upper(), c=outer, d=outer.str.len())
    plan = _optimized(ds)
    bound = _bound_columns(plan)
    assert len(bound) == 2, f"expected both repeats bound, got {bound}"
    assert _ds().select(a=inner, b=inner.str.upper(), c=outer, d=outer.str.len()).collect()


def test_the_rewrite_reaches_a_fixpoint(caplog):
    """The rule adds a `Project`; a non-confluent rule would spin and warn."""
    e = col("url").str.regexp_replace("-", "+")
    ds = _ds().select(a=e, b=e.str.upper(), c=e.str.len())
    with caplog.at_level(logging.WARNING, logger="batcher.kyber"):
        _optimized(ds)
    assert "fixpoint" not in caplog.text.lower()
