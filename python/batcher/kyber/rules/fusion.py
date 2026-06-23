"""FUSION-phase rewrites — top-N fusion and per-partition top-N (`QUALIFY`).

A `Limit(Sort(...), n, offset)` only needs the first `n+offset` rows of the
sorted order. Setting the sort's `limit` lets the engine do a partial sort
(arrow `lexsort_to_indices(.., Some(limit))`) instead of fully sorting and
slicing — the difference between O(n log n) and O(n log k), which is the whole
cost of a top-N query. When `offset == 0` the `Limit` is dropped entirely.

`qualify_to_partition_topn` is the per-partition analogue: `Filter(Window([rank]),
rank <= k)` — the shape SQL `QUALIFY` lowers to — fuses the bound into the window as
`rank_limit`, so the engine keeps only the top-`k` rows per partition and the
separate filter (and the full windowed intermediate) disappears.

`fuse_topn` is registered as the `topn_fusion` `plan_rule` in `Phase.FUSION`
(see `kyber.registry.register_builtin_rules`).
"""

from __future__ import annotations

import dataclasses

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.stats.selectivity import comparison_col_side
from batcher.plan.expr_ir import Binary
from batcher.plan.expr_rewrite import combine_conjuncts, split_conjuncts
from batcher.plan.ir_tags import WINDOW_RANKING
from batcher.plan.logical import (
    Filter,
    Join,
    Limit,
    LogicalPlan,
    Scan,
    Sort,
    Union,
    Window,
)

__all__ = ["fuse_topn", "qualify_to_partition_topn"]

# Flip a comparison operator when the column is on the right (`lit <= col` ≡ `col >= lit`).
_FLIP = {"lt": "gt", "gt": "lt", "le": "ge", "ge": "le", "eq": "eq", "ne": "ne"}


def fuse_topn(plan: LogicalPlan) -> LogicalPlan:
    return _f(plan)


def _f(node: LogicalPlan) -> LogicalPlan:
    # Fuse Limit-over-Sort (only when the sort isn't already a top-N).
    if isinstance(node, Limit) and isinstance(node.input, Sort) and node.input.limit is None:
        sort = node.input
        total = node.n + node.offset
        fused_sort = Sort(_f(sort.input), sort.keys, limit=total)
        # offset 0 → the sort already yields exactly the wanted rows.
        return fused_sort if node.offset == 0 else Limit(fused_sort, node.n, node.offset)

    # Pure recursion below here — preserve object identity when nothing fused, so the
    # driver detects the fixpoint in O(1) instead of serializing the plan.
    if isinstance(node, Scan):
        return node
    if isinstance(node, Join):
        left, right = _f(node.left), _f(node.right)
        if left is node.left and right is node.right:
            return node
        return Join(left, right, node.left_keys, node.right_keys, node.join_type, node.output)
    if isinstance(node, Union):
        inputs = tuple(_f(i) for i in node.inputs)
        if all(a is b for a, b in zip(inputs, node.inputs, strict=True)):
            return node
        return Union(inputs, node.distinct)
    if hasattr(node, "input"):
        child = _f(node.input)
        return node if child is node.input else dataclasses.replace(node, input=child)
    return node


@rule(name="qualify_to_partition_topn", phase=Phase.FUSION, matches=(Filter,))
def qualify_to_partition_topn(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Filter(Window([rank], …), rank <= k)` → `Window([rank], …, rank_limit=k)`.

    Fuses a `QUALIFY <rank> <= k` (or `< k`, or `= 1`) bound into a single-ranking-
    function window, so the engine keeps only the top-`k` rows per partition instead
    of materializing every row and filtering. For `row_number` this is the top-`k`;
    for `rank`/`dense_rank` it keeps boundary ties (the bound is on the rank value).
    Any other conjuncts in the predicate stay as a filter above the fused window.
    Returns None when nothing fuses, so the rule is idempotent.
    """
    win = node.input
    if not isinstance(win, Window) or win.rank_limit is not None:
        return None
    if len(win.functions) != 1 or win.functions[0].func not in WINDOW_RANKING:
        return None
    rank_alias = win.functions[0].alias

    limit: int | None = None
    rest = []
    for conj in split_conjuncts(node.predicate):
        k = _rank_bound(conj, rank_alias) if limit is None else None
        if k is None:
            rest.append(conj)
        else:
            limit = max(0, k)
    if limit is None:
        return None

    fused = Window(
        win.input, win.partition_keys, win.order_keys, win.functions, rank_limit=limit
    )
    return fused if not rest else Filter(fused, combine_conjuncts(rest))


def _rank_bound(conj: object, rank_alias: str) -> int | None:
    """The `k` of a `rank_alias <= k` / `< k` / `= 1` conjunct, else None.

    Normalizes `lit OP col` to `col OP lit`; only upper bounds (`<=`/`<`) and the
    `= 1` special case (equivalent to `<= 1`, since ranks start at 1) yield a limit —
    a lower bound (`>=`/`>`) or a non-integer literal does not.
    """
    if not isinstance(conj, Binary):
        return None
    side = comparison_col_side(conj)
    if side is None:
        return None
    name, value, col_on_left = side
    if name != rank_alias or isinstance(value, bool) or not isinstance(value, int):
        return None
    op = conj.op if col_on_left else _FLIP[conj.op]
    if op == "le":
        return value
    if op == "lt":
        return value - 1
    if op == "eq" and value == 1:
        return 1
    return None
