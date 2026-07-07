"""Plan-shape, idempotence, and negative tests for the `topn_limit` rules.

Correctness vs DuckDB (and vs Batcher's own unoptimized execution for the
order-arbitrary union case) lives in tests/differential/test_diff_topn_limit.py.
Importing the module registers its `@rule` decorators into `DEFAULT_REGISTRY`.
"""

from __future__ import annotations

import batcher as bt
from batcher.config import active_config
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra.topn_limit import (
    drop_redundant_limit,
    empty_limit_past_cardinality,
    push_limit_through_row_index,
    push_offset_limit_into_union,
)
from batcher.kyber.stats.estimator import StatsEstimator
from batcher.plan.logical import Limit, RowId, Scan, Union


def _ds(n=5):
    return bt.from_pydict({"k": list(range(n)), "v": [i * 10 for i in range(n)]})


def _ctx(ds):
    est = StatsEstimator(ds._sources)
    return OptimizerContext(config=active_config(), sources=ds._sources, hub=None, estimator=est)


def _ir(ds):
    # Pass sources so the cardinality-driven rules see exact scan row counts.
    return Optimizer(sources=ds._sources).optimize(ds._plan).ir


def _count_op(ir, op):
    n = 1 if ir.get("op") == op else 0
    for v in ir.values():
        if isinstance(v, dict) and "op" in v:
            n += _count_op(v, op)
        elif isinstance(v, list):
            n += sum(_count_op(e, op) for e in v if isinstance(e, dict) and "op" in e)
    return n


# --- registration -------------------------------------------------------------


def test_rules_registered():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert {
        "drop_redundant_limit",
        "empty_limit_past_cardinality",
        "push_limit_through_row_index",
        "push_offset_limit_into_union",
    } <= names


# --- drop_redundant_limit -----------------------------------------------------


def test_drop_redundant_limit_fires():
    ds = _ds(5).limit(10)  # exactly 5 rows, keep up to 10 → limit is a no-op
    out = drop_redundant_limit(ds._plan, _ctx(ds))
    assert isinstance(out, Scan)


def test_drop_redundant_limit_full_optimizer():
    ds = _ds(5).limit(10)
    assert _count_op(_ir(ds), "limit") == 0


def test_drop_redundant_limit_noop_when_limit_binds():
    ds = _ds(5).limit(3)  # 3 < 5 → the limit really drops rows
    assert drop_redundant_limit(ds._plan, _ctx(ds)) is None


def test_drop_redundant_limit_noop_with_offset():
    ds = _ds(5).limit(10, offset=1)  # an offset changes which rows survive
    assert drop_redundant_limit(ds._plan, _ctx(ds)) is None


def test_drop_redundant_limit_idempotent():
    ds = _ds(5).limit(10)
    ctx = _ctx(ds)
    once = drop_redundant_limit(ds._plan, ctx)
    # The result is the bare scan; the rule no longer matches a Limit there.
    assert not isinstance(once, Limit)


# --- empty_limit_past_cardinality ---------------------------------------------


def test_empty_limit_past_cardinality_fires():
    ds = _ds(5).limit(3, offset=10)  # offset 10 skips past all 5 rows → empty
    out = empty_limit_past_cardinality(ds._plan, _ctx(ds))
    assert isinstance(out, Limit) and out.n == 0


def test_empty_limit_past_cardinality_full_optimizer():
    ds = _ds(5).limit(3, offset=10)
    ir = _ir(ds)
    # The plan collapses to the empty marker (a limit with n == 0).
    assert ir["op"] == "limit" and ir["n"] == 0


def test_empty_limit_past_cardinality_noop_within_bounds():
    ds = _ds(5).limit(2, offset=1)  # offset 1 < 5 → real rows survive
    assert empty_limit_past_cardinality(ds._plan, _ctx(ds)) is None


def test_empty_limit_past_cardinality_idempotent():
    ds = _ds(5).limit(3, offset=10)
    ctx = _ctx(ds)
    once = empty_limit_past_cardinality(ds._plan, ctx)
    assert empty_limit_past_cardinality(once, ctx) is None  # n == 0 now → no refire


# --- push_limit_through_row_index ---------------------------------------------


def test_push_limit_through_row_index_fires():
    ds = _ds(5).with_row_index("idx", offset=100).limit(3, offset=1)
    out = push_limit_through_row_index(ds._plan, None)
    assert isinstance(out, RowId)
    assert isinstance(out.input, Limit)
    assert out.input.n == 3 and out.input.offset == 1
    assert out.offset == 101  # 100 (row-index base) + 1 (limit offset)


def test_push_limit_through_row_index_full_optimizer():
    ds = _ds(5).sort("k").with_row_index("idx").limit(3)
    ir = _ir(ds)
    assert ir["op"] == "row_id"  # the limit was pushed below the numbering


def test_push_limit_through_row_index_noop_over_distinct():
    # Limit above a Distinct must NOT push (Distinct changes row count); the child
    # here is a Distinct, not a RowId, so the rule leaves it alone.
    ds = _ds(5).distinct().limit(3)
    assert push_limit_through_row_index(ds._plan, None) is None


def test_push_limit_through_row_index_idempotent():
    ds = _ds(5).with_row_index("idx").limit(3, offset=1)
    once = push_limit_through_row_index(ds._plan, None)
    # After the push the inner Limit's child is the scan, not a RowId → no refire.
    assert push_limit_through_row_index(once.input, None) is None


# --- push_offset_limit_into_union ---------------------------------------------


def test_push_offset_limit_into_union_fires():
    ds = _ds(3).union(_ds(4)).limit(2, offset=1)
    out = push_offset_limit_into_union(ds._plan, None)
    assert isinstance(out, Limit) and out.n == 2 and out.offset == 1
    inner = out.input
    assert isinstance(inner, Union) and not inner.distinct
    assert all(isinstance(b, Limit) and b.n == 3 and b.offset == 0 for b in inner.inputs)


def test_push_offset_limit_into_union_noop_zero_offset():
    # offset 0 is handled by the existing push_limit_into_union rule, not this one.
    ds = _ds(3).union(_ds(4)).limit(2, offset=0)
    assert push_offset_limit_into_union(ds._plan, None) is None


def test_push_offset_limit_into_union_noop_distinct_union():
    ds = _ds(3).union(_ds(4), distinct=True).limit(2, offset=1)
    assert push_offset_limit_into_union(ds._plan, None) is None


def test_push_offset_limit_into_union_idempotent():
    ds = _ds(3).union(_ds(4)).limit(2, offset=1)
    once = push_offset_limit_into_union(ds._plan, None)
    assert push_offset_limit_into_union(once, None) is None  # branches already capped
