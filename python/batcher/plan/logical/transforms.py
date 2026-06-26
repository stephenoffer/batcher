"""Plan transforms and predicates over `LogicalPlan` trees.

`remap_sources` shifts every `Scan.source_id` (used when appending a right side's
sources after the left's); `is_streamable` reports whether a plan is
partition-independent (only row-wise operators, no pipeline breaker).
"""

from __future__ import annotations

from batcher.plan.expr_ir import Col, Lit
from batcher.plan.logical.aggregate import Sort
from batcher.plan.logical.base import LogicalPlan
from batcher.plan.logical.join import Join
from batcher.plan.logical.relational import (
    Distinct,
    Filter,
    Limit,
    MapBatches,
    Project,
    Sample,
    Scan,
    Unnest,
    Unpivot,
)

__all__ = ["is_cartesian_key_pair", "is_streamable", "remap_sources"]

# Sentinel distinguishing "the column is a known constant whose value is None" from
# "the column is not a provable constant". `constant_column_value` returns this when
# the column cannot be proven constant.
_NOT_CONSTANT = object()


def is_cartesian_key_pair(
    left: LogicalPlan, left_key: str, right: LogicalPlan, right_key: str
) -> bool:
    """Whether an equi-join key pair is a cartesian pseudo-edge (same constant on both sides).

    A key pair `left_key = right_key` where both columns are provably the *same* literal
    (the `__cross_key` a comma/cross join lowers to) is always true and connects nothing
    — it expresses a cartesian product, not a real join condition. Join reordering must
    not treat it as a graph edge (or it would happily build a cross product), and key
    derivation drops it once a real key is found. Anything not provably constant-on-both
    -sides returns False (treated as a genuine join edge).
    """
    lv = constant_column_value(left, left_key)
    if lv is _NOT_CONSTANT:
        return False
    rv = constant_column_value(right, right_key)
    if rv is _NOT_CONSTANT:
        return False
    return lv == rv


def constant_column_value(plan: LogicalPlan, column: str) -> object:
    """The literal value `column` provably holds in every output row, or `_NOT_CONSTANT`.

    Traces `column` down through value-preserving operators — a `Project` that binds it
    to a `Lit` (proof) or merely renames another column, and the row-preserving
    `Filter`/`Sort`/`Limit`/`Sample`/`Distinct` and inner `Join` — to the literal that
    defines it. Used to recognize synthetic constant join keys (e.g. the `__cross_key`
    a comma/cross join lowers to): a join key that is the same constant on both sides
    carries no information, so it is a cartesian pseudo-edge, not a real join condition.
    Anything it cannot prove returns `_NOT_CONSTANT` — never a guess.
    """
    if isinstance(plan, Project):
        for item in plan.items:
            if item.alias == column:
                if isinstance(item.expr, Lit):
                    return item.expr.value
                if isinstance(item.expr, Col):  # pure rename `column ← src`
                    return constant_column_value(plan.input, item.expr.name)
                return _NOT_CONSTANT
        return _NOT_CONSTANT
    if isinstance(plan, (Filter, Sort, Limit, Sample, Distinct)):
        return constant_column_value(plan.input, column)
    if isinstance(plan, Join) and plan.join_type == "inner":
        for o in plan.output:
            if o.alias == column:
                child = plan.left if o.side == "left" else plan.right
                return constant_column_value(child, o.name)
        return _NOT_CONSTANT
    return _NOT_CONSTANT


def remap_sources(plan: LogicalPlan, offset: int) -> LogicalPlan:
    """Return a copy of `plan` with every `Scan.source_id` shifted by `offset`.

    Used when joining two datasets: the right side's sources are appended after
    the left's, so its scans must point past them.

    Only `Scan` carries a `source_id`; every other node is rebuilt generically with
    its remapped children by `transform_up`, so a new node type needs no edit here.
    The import is function-local because `plan.visitor` imports this module.
    """
    from batcher.plan.visitor import transform_up

    def shift(node: LogicalPlan) -> LogicalPlan:
        if isinstance(node, Scan):
            return Scan(node.source_id + offset, node.schema)
        return node

    return transform_up(plan, shift)


def is_streamable(plan: LogicalPlan) -> bool:
    """Whether `plan` can be executed one source batch at a time in bounded memory.

    True iff the plan contains only row-wise / per-partition operators —
    `Scan`, `Filter`, `Project`, `MapBatches`, `Unnest` — and no pipeline breaker
    (aggregate, sort, join, distinct, union, window, limit) that must see the
    whole input. Such plans are partition-independent, so running them per source
    batch yields exactly the same result as running them over the whole input.
    """
    if isinstance(plan, Scan):
        return True
    if isinstance(plan, (Filter, Project, MapBatches, Unnest, Unpivot, Sample)):
        return is_streamable(plan.input)
    return False
