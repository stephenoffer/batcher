"""Plan-shape, idempotence, and negative tests for the `topn_limit` rules.

Correctness vs DuckDB (and vs Batcher's own unoptimized execution for the
order-arbitrary union case) lives in tests/differential/test_diff_topn_limit.py.
Importing the module registers its `@rule` decorators into `DEFAULT_REGISTRY`.
"""

from __future__ import annotations

from dataclasses import replace

import batcher as bt
from batcher.config import active_config
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra.topn_limit import (
    drop_redundant_limit,
    empty_limit_past_cardinality,
    fuse_limit_into_distinct,
    push_limit_through_row_index,
    push_offset_limit_into_union,
)
from batcher.kyber.stats.estimator import StatsEstimator
from batcher.plan.logical import Distinct, Limit, Project, RowId, Scan, Union
from batcher.plan.stats import Provenance


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


# --- fuse_limit_into_distinct -------------------------------------------------


def _distinct_ds(n=4000):
    """A nearly-unique key, so the fusion's cardinality gate opens."""
    return bt.from_pydict({"g": list(range(n))}).select("g").distinct()


def _find_op(ir, op):
    """The first node with tag `op`, or None."""
    if ir.get("op") == op:
        return ir
    for v in ir.values():
        if isinstance(v, dict) and "op" in v:
            found = _find_op(v, op)
            if found is not None:
                return found
        elif isinstance(v, list):
            for e in v:
                if isinstance(e, dict) and "op" in e:
                    found = _find_op(e, op)
                    if found is not None:
                        return found
    return None


def test_fuse_limit_into_distinct_registered():
    assert "fuse_limit_into_distinct" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_fuse_limit_into_distinct_fires_on_high_cardinality():
    ds = _distinct_ds().limit(5)
    out = fuse_limit_into_distinct(ds._plan, _ctx(ds))
    assert isinstance(out, Limit)
    assert isinstance(out.input, Distinct)
    assert out.input.limit == 5


def test_fuse_limit_into_distinct_reaches_the_ir():
    """The whole optimizer emits the capped operator, not just the rule in isolation."""
    ds = _distinct_ds().limit(5)
    distinct_ir = _find_op(_ir(ds), "distinct")
    assert distinct_ir is not None
    assert distinct_ir["limit"] == 5


def test_fuse_limit_into_distinct_caps_at_offset_plus_n():
    """The dedup must reach `offset + n` distinct rows for the outer window to be there."""
    ds = _distinct_ds().limit(2, offset=3)
    out = fuse_limit_into_distinct(ds._plan, _ctx(ds))
    assert out.input.limit == 5
    assert (out.n, out.offset) == (2, 3)


def test_fuse_limit_into_distinct_noop_on_measured_low_cardinality():
    """A *measured* low ndv declines: the dense direct-map already wins there.

    An unmeasured column is a different case and deliberately still fuses — the operator's
    own bounded probe decides it on measured data. Only evidence declines here.
    """
    ds = _distinct_ds().limit(5)
    ctx = _ctx(ds)
    est = ctx.estimator

    class _LowNdv:
        """An estimator reporting a measured, low distinct count."""

        def estimate(self, node):
            base = est.estimate(node)
            return replace(base, rows=10.0, provenance=Provenance.SKETCH)

    low = OptimizerContext(
        config=active_config(), sources=ds._sources, hub=None, estimator=_LowNdv()
    )
    assert fuse_limit_into_distinct(ds._plan, low) is None


def test_fuse_limit_into_distinct_fires_on_unmeasured_cardinality():
    """No ndv is not evidence of a low one, so the rule fuses and the operator probes."""
    ds = bt.from_pydict({"g": [i % 4 for i in range(1000)]}).select("g").distinct().limit(5)
    ctx = _ctx(ds)
    assert ctx.estimator.estimate(ds._plan.input).provenance is Provenance.DEFAULT
    assert fuse_limit_into_distinct(ds._plan, ctx) is not None


def test_fuse_limit_into_distinct_noop_on_keyed_distinct():
    """A keyed dedup's survivor can be replaced by a later row, so no prefix settles it."""
    ds = bt.from_pydict({"g": list(range(4000)), "v": list(range(4000))})
    keyed = ds.distinct(["g"]).limit(5)
    assert fuse_limit_into_distinct(keyed._plan, _ctx(keyed)) is None


def test_fuse_limit_into_distinct_idempotent():
    ds = _distinct_ds().limit(5)
    ctx = _ctx(ds)
    once = fuse_limit_into_distinct(ds._plan, ctx)
    assert fuse_limit_into_distinct(once, ctx) is None


def test_unfused_distinct_omits_the_limit_from_the_ir():
    """The wire shape is unchanged when nothing fused, so old plans stay byte-identical."""
    ds = bt.from_pydict({"g": [i % 4 for i in range(1000)]}).select("g").distinct()
    distinct_ir = _find_op(_ir(ds), "distinct")
    assert distinct_ir is not None
    assert "limit" not in distinct_ir


# --- the branch-cap guard must look through a projection ----------------------
#
# `push_limit_into_union` / `push_offset_limit_into_union` cap each UNION ALL branch with a
# `Limit`, and both docstrings promise the "already capped?" guard makes them fire once and
# then rest. They did not: `push_limit_through_project` runs in the same phase and rewrites
# `Limit(Project(x))` into `Project(Limit(x))`, so on the next iteration every branch is a
# bare `Project` again, the guard sees no `Limit`, and the cap is reinstalled on top of the
# one already there. The two rules then trade the plan back and forth for the whole
# `fixpoint_iterations` budget — the cycle `ml.smote`'s plan reproduced, recorded as a
# strict xfail in `test_kyber_constant_key_fixpoint.py` until it was closed.
#
# Every rewrite involved is semantics-preserving and a doubled cap is still a cap, so the
# answers were never wrong; what was wrong is that the plan depended on where the iteration
# cap landed, and that ~20 whole-plan passes were spent reaching it.


def _projected_union(offset=0, n=2):
    """`Limit(UNION ALL(Project(a), Project(b)), n, offset)` — the shape that cycled."""
    left = _ds(3).select(y=bt.col("k") + 1)
    right = _ds(4).select(y=bt.col("k") + 1)
    return left.union(right).limit(n, offset=offset)._plan


def test_push_limit_into_union_does_not_recap_a_branch_capped_below_a_project():
    from batcher.kyber.rules.algebraic.identities import (
        push_limit_into_union,
        push_limit_through_project,
    )

    once = push_limit_into_union(_projected_union(), None)
    assert once is not None, "the rule must still fire on an uncapped union"
    # What the same phase does next: the branch caps sink through their projections.
    sunk = tuple(push_limit_through_project(b, None) or b for b in once.input.inputs)
    assert all(isinstance(b, Project) for b in sunk), "the branch caps must have sunk"
    rebuilt = Limit(Union(sunk, distinct=False), once.n, once.offset)
    assert push_limit_into_union(rebuilt, None) is None, "the rule re-capped a capped branch"


def test_push_offset_limit_into_union_does_not_recap_a_branch_capped_below_a_project():
    """The `offset > 0` companion carried the identical guard, and the identical defect."""
    from batcher.kyber.rules.algebraic.identities import push_limit_through_project

    once = push_offset_limit_into_union(_projected_union(offset=3), None)
    assert once is not None
    sunk = tuple(push_limit_through_project(b, None) or b for b in once.input.inputs)
    assert all(isinstance(b, Project) for b in sunk)
    rebuilt = Limit(Union(sunk, distinct=False), once.n, once.offset)
    assert push_offset_limit_into_union(rebuilt, None) is None


def test_caps_rows_sees_through_a_project_and_nothing_else():
    """The guard's own contract: a projection is transparent, an aggregate is not."""
    from batcher.kyber.rules.algebraic.identities import _caps_rows

    capped = _ds(3).limit(2)._plan
    assert _caps_rows(capped)
    assert _caps_rows(Project(capped, _ds(3).select("k")._plan.items))
    assert not _caps_rows(_ds(3)._plan)
    # An aggregate collapses rows rather than passing them through, so a cap beneath it
    # says nothing about the relation above it — and the descent must stop.
    assert not _caps_rows(_ds(3).limit(2).group_by("k").agg(c=bt.col("v").count())._plan)


def test_the_union_limit_still_returns_the_right_rows():
    """The cycle never changed an answer, and neither does closing it."""
    left = _ds(3).select(y=bt.col("k") + 1)
    right = _ds(4).select(y=bt.col("k") + 1)
    assert left.union(right).limit(2).collect().num_rows == 2
    assert left.union(right).limit(2, offset=3).collect().num_rows == 2
    assert left.union(right).limit(100).collect().num_rows == 7
