"""Narrowing-cast pushdown — relocate a user's cast to the column's producer.

The rewrite is semantics-preserving (the cast is moved, never invented), so the
focus is: it fires only on the safe shape, it is idempotent, and the optimized query
returns identical results.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.fusion import push_down_narrowing_cast
from batcher.plan.expr_ir import Cast, Col
from batcher.plan.logical import Project, Sort


def _t():
    return bt.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6]})


def test_registered_in_default_registry():
    assert "push_down_narrowing_cast" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_cast_pushed_through_sort_to_producer():
    # select(cast(c)) over sort over with_columns(c = a + b): the cast moves down
    # into the producing projection, so int32 flows through the sort.
    plan = (
        _t()
        .with_columns(c=col("a") + col("b"))
        .sort("a")
        .select(small=col("c").cast("int32"))
        ._plan
    )
    out = push_down_narrowing_cast(plan, None)
    assert out is not None

    # Top projection now references the column directly (no cast).
    assert isinstance(out, Project)
    assert isinstance(out.items[0].expr, Col)

    # The producer below the sort now casts its definition to int32.
    sort = out.input
    assert isinstance(sort, Sort)
    producer = sort.input
    assert isinstance(producer, Project)
    c_item = next(it for it in producer.items if it.alias == "c")
    assert isinstance(c_item.expr, Cast) and c_item.expr.dtype == "int32"


def test_idempotent():
    plan = (
        _t()
        .with_columns(c=col("a") + col("b"))
        .sort("a")
        .select(small=col("c").cast("int32"))
        ._plan
    )
    once = push_down_narrowing_cast(plan, None)
    assert push_down_narrowing_cast(once, None) is None  # nothing left to push


def test_does_not_fire_when_column_used_elsewhere():
    # `c` is also passed through, so narrowing it earlier could change the kept copy.
    plan = (
        _t()
        .with_columns(c=col("a") + col("b"))
        .select(small=col("c").cast("int32"), keep=col("c"))
        ._plan
    )
    assert push_down_narrowing_cast(plan, None) is None


def test_does_not_fire_without_narrowing():
    # cast to int64 over an int64 column is not narrower → no benefit, no rewrite.
    plan = _t().with_columns(c=col("a") + col("b")).select(x=col("c").cast("int64"))._plan
    assert push_down_narrowing_cast(plan, None) is None


@pytest.mark.parametrize("with_pushdown", [True, False])
def test_result_identical_with_optimizer(with_pushdown):
    # The full optimizer (which includes the rule) must return the same values and
    # output type as the eager small-data expectation.
    ds = _t().with_columns(c=col("a") + col("b")).sort("a").select(small=col("c").cast("int32"))
    out = ds.collect()
    assert out.schema.field("small").type == pa.int32()
    assert out.to_pydict()["small"] == [5, 7, 9]


def test_optimizer_pushes_cast_below_sort():
    ds = _t().with_columns(c=col("a") + col("b")).sort("a").select(small=col("c").cast("int32"))
    ir = Optimizer().optimize(ds._plan).ir
    # The top op is a projection of a bare column; the cast now lives below the sort.
    assert ir["op"] == "project"
    assert ir["input"]["op"] == "sort"
