"""Plan-shape, refusal and idempotence tests for `push_projection_through_join`.

The rule pushes a single-side, non-raising computed projection onto the join input it
reads, so the join carries the narrow result instead of its wide inputs. Result-correctness
vs DuckDB lives in `tests/differential/test_diff_join_projection.py`.
"""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.rules import join_projection as m
from batcher.plan.logical import Join, Project
from batcher.plan.visitor import walk


def _ds():
    left = bt.from_pydict({"k": [1, 2, 3], "a": [10.0, 20.0, 30.0], "b": [0.1, 0.2, 0.3]})
    right = bt.from_pydict({"k": [1, 2, 3], "grp": ["x", "y", "x"], "c": [1.0, 2.0, 3.0]})
    return left, right


def _ctx(ds):
    return Optimizer(sources=ds._sources)._context()


def _pushed_below_join(plan) -> bool:
    joins = [n for n in walk(plan) if isinstance(n, Join)]
    for j in joins:
        for side in (j.left, j.right):
            if any(isinstance(n, Project) for n in walk(side)):
                return True
    return False


def test_fires_on_single_side_derived_projection():
    left, right = _ds()
    ds = left.join(right, on="k", how="inner").with_columns(rev=col("a") * (1 - col("b")))
    out = m.push_projection_through_join(ds._plan, _ctx(ds))
    assert isinstance(out, Project)
    # The join now carries the computed `rev`; a Project sits on the left (probe) input.
    (join,) = [n for n in walk(out) if isinstance(n, Join)]
    assert any(oc.alias == "rev" for oc in join.output)


def test_refuses_mixed_side_expression():
    left, right = _ds()
    # a (left) + c (right) references both sides — cannot be computed on one input.
    ds = left.join(right, on="k", how="inner").with_columns(mix=col("a") + col("c"))
    assert m.push_projection_through_join(ds._plan, _ctx(ds)) is None


def test_refuses_non_inner_join():
    left, right = _ds()
    ds = left.join(right, on="k", how="left").with_columns(rev=col("a") * 2)
    assert m.push_projection_through_join(ds._plan, _ctx(ds)) is None


def test_refuses_division_which_can_raise():
    left, right = _ds()
    # A divide could raise on an unmatched (later-dropped) row — must stay above the join.
    ds = left.join(right, on="k", how="inner").with_columns(r=col("a") / col("b"))
    assert m.push_projection_through_join(ds._plan, _ctx(ds)) is None


def test_refuses_bare_column_passthrough():
    left, right = _ds()
    ds = left.join(right, on="k", how="inner").select("grp", "a")
    assert m.push_projection_through_join(ds._plan, _ctx(ds)) is None


def test_idempotent_and_fires_end_to_end():
    left, right = _ds()
    ds = left.join(right, on="k", how="inner").with_columns(rev=col("a") * (1 - col("b")))
    opt = Optimizer(sources=ds._sources)
    once = opt.logical_rewrite(ds._plan)
    assert _pushed_below_join(once)
    # Re-optimizing settles (the pushed expression is no longer a single-side item above
    # the join, so the rule cannot re-fire on its own output).
    settled = opt.logical_rewrite(once)
    assert opt.logical_rewrite(settled).to_ir() == settled.to_ir()
