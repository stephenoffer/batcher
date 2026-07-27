"""Plan-shape tests for `kyber.rules.relational.windows`.

Both rules here are *structural*: they change which node sits where without changing
the relation, so a result comparison alone cannot tell whether they fired. These
assert the shape directly, and the differential suite proves the shapes agree on data.

Each rule also gets its refusal cases, because a structural rule that fires when it
should not is the failure mode that a passing result test hides: a transposition that
crosses a real dependency, and a top-N pushed under a window whose value depends on
the rows it would remove.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.config import Config
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.rules.relational.windows import (
    _order_key_ids,
    push_topn_into_unpartitioned_ranking_window,
    transpose_adjacent_windows,
)
from batcher.plan.expr_ir import Col
from batcher.plan.logical import Limit, Scan, Sort, SortKeySpec, Window, WindowFuncSpec
from batcher.plan.schema import SchemaRef


@pytest.fixture
def ctx():
    return OptimizerContext(
        config=Config(), sources=[], hub=None, estimator=CardinalityEstimator([], None)
    )


@pytest.fixture
def scan():
    schema = SchemaRef.from_arrow(
        pa.schema([pa.field("a", pa.int64()), pa.field("g", pa.string())])
    )
    return Scan(0, schema)


def _window(inp, *, partition=(), order=("a",), func="row_number", alias="r", finput=None):
    return Window(
        inp,
        tuple(Col(p) for p in partition),
        tuple(SortKeySpec(Col(o)) for o in order),
        (WindowFuncSpec(func, finput, alias),),
    )


# --- transpose_adjacent_windows -----------------------------------------------


def test_transpose_orders_independent_windows_by_spec(scan, ctx):
    """Two independent windows are swapped into canonical spec order."""
    inner = _window(scan, partition=("g",), alias="r1")
    outer = _window(inner, partition=(), alias="r2")
    out = transpose_adjacent_windows(outer, ctx)
    assert out is not None
    # The partition-free spec sorts first, so it must end up innermost.
    assert isinstance(out, Window) and out.functions[0].alias == "r1"
    assert isinstance(out.input, Window) and out.input.functions[0].alias == "r2"


def test_transpose_is_a_fixpoint_in_canonical_order(scan, ctx):
    """Already ordered, the rule declines -- otherwise the pair would swap forever."""
    inner = _window(scan, partition=(), alias="r1")
    outer = _window(inner, partition=("g",), alias="r2")
    assert transpose_adjacent_windows(outer, ctx) is None


def test_transpose_refuses_when_outer_reads_inner_output(scan, ctx):
    """A genuine dependency pins the order: the outer window sums the inner's column."""
    inner = _window(scan, partition=("g",), alias="r1")
    outer = _window(inner, partition=(), alias="r2", func="sum", finput=Col("r1"))
    assert transpose_adjacent_windows(outer, ctx) is None


# --- push_topn_into_unpartitioned_ranking_window ------------------------------


def test_topn_is_pushed_below_an_unpartitioned_ranking_window(scan, ctx):
    """A `Sort` capped at `n + offset` appears below the window, and the limit stays."""
    plan = Limit(_window(scan), n=5, offset=2)
    out = push_topn_into_unpartitioned_ranking_window(plan, ctx)
    assert out is not None
    assert isinstance(out, Limit) and out.n == 5 and out.offset == 2
    window = out.input
    assert isinstance(window, Window)
    below = window.input
    assert isinstance(below, Sort) and below.limit == 7
    assert _order_key_ids(below.keys) == _order_key_ids(window.order_keys)


def test_topn_refuses_a_partitioned_window(scan, ctx):
    """With partition keys, rank restarts per partition and a prefix is the wrong rows."""
    plan = Limit(_window(scan, partition=("g",)), n=5)
    assert push_topn_into_unpartitioned_ranking_window(plan, ctx) is None


@pytest.mark.parametrize("func", ["percent_rank", "cume_dist", "ntile"])
def test_topn_refuses_partition_size_dependent_ranking(scan, ctx, func):
    """These divide by the partition's row count, so truncating changes their value.

    They live in `WINDOW_RANKING` alongside `row_number`, which is exactly why the rule
    keeps its own narrower `_PREFIX_STABLE_RANKING` set rather than reusing that one.
    """
    plan = Limit(_window(scan, func=func), n=5)
    assert push_topn_into_unpartitioned_ranking_window(plan, ctx) is None


def test_topn_refuses_an_aggregate_window(scan, ctx):
    """A window aggregate reads the whole partition, so the input may not be truncated."""
    plan = Limit(_window(scan, func="sum", finput=Col("a")), n=5)
    assert push_topn_into_unpartitioned_ranking_window(plan, ctx) is None


def test_topn_reaches_a_fixpoint_over_an_existing_tighter_cap(scan, ctx):
    """An existing smaller cap is left alone so the rule cannot rebuild forever."""
    sorted_input = Sort(scan, (SortKeySpec(Col("a")),), limit=3)
    plan = Limit(_window(sorted_input), n=5)
    assert push_topn_into_unpartitioned_ranking_window(plan, ctx) is None
