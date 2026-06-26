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

import pyarrow as pa

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.stats.selectivity import comparison_col_side
from batcher.plan.expr_ir import Binary, Cast, Col, referenced_columns
from batcher.plan.expr_rewrite import combine_conjuncts, split_conjuncts
from batcher.plan.ir_tags import WINDOW_RANKING
from batcher.plan.logical import (
    Filter,
    Join,
    Limit,
    LogicalPlan,
    Project,
    Sample,
    Scan,
    Sort,
    Union,
    Window,
)
from batcher.plan.logical.aggregate import SortKeySpec
from batcher.plan.logical.relational import Projection
from batcher.plan.types import DTYPE_REGISTRY, infer_type

__all__ = [
    "collapse_adjacent_windows",
    "fuse_topn",
    "push_down_narrowing_cast",
    "qualify_to_partition_topn",
]

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

    fused = Window(win.input, win.partition_keys, win.order_keys, win.functions, rank_limit=limit)
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


def _key_sig(key: object) -> object:
    """A comparable signature for a partition expr or an order-key spec — by lowered
    IR (never `==`, which `Expr.__eq__` overloads to build a comparison expression)."""
    if isinstance(key, SortKeySpec):
        return (key.expr.to_ir(), key.descending, key.nulls_first)
    return key.to_ir()


def _keys_match(a: tuple, b: tuple) -> bool:
    """Whether two partition/order key tuples are structurally identical."""
    return len(a) == len(b) and all(_key_sig(x) == _key_sig(y) for x, y in zip(a, b, strict=True))


@rule(name="collapse_adjacent_windows", phase=Phase.FUSION, matches=(Window,))
def collapse_adjacent_windows(node: Window, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Window(Window(x, P, O, fns1), P, O, fns2)` → `Window(x, P, O, fns1 + fns2)`.

    Two windows over the *same* partitioning and ordering are one full-data pass, not
    two — a single `Window` computing every function. Safe only when (a) neither
    carries a `rank_limit` (which filters rows), (b) the partition and order keys are
    identical, (c) the outer functions don't read a column the inner window produces
    (the merged pass computes them together, so an inner output isn't yet available),
    and (d) the two function sets have no alibi alias collision. Returns None otherwise,
    so the rule is idempotent.
    """
    inner = node.input
    if not isinstance(inner, Window):
        return None
    if node.rank_limit is not None or inner.rank_limit is not None:
        return None
    if not _keys_match(node.partition_keys, inner.partition_keys):
        return None
    if not _keys_match(node.order_keys, inner.order_keys):
        return None
    inner_aliases = {f.alias for f in inner.functions}
    # Outer functions must not depend on a column the inner window introduces.
    for f in node.functions:
        if f.input is not None and referenced_columns(f.input) & inner_aliases:
            return None
    if inner_aliases & {f.alias for f in node.functions}:
        return None  # alias collision — keep them separate
    return Window(
        inner.input,
        node.partition_keys,
        node.order_keys,
        inner.functions + node.functions,
        node.rank_limit,
    )


@rule(name="push_down_narrowing_cast", phase=Phase.FUSION, matches=(Project,))
def push_down_narrowing_cast(node: Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Push a bare narrowing ``cast(col(c), T)`` down to where ``c`` is produced.

    Fires on the first projection item that is a narrowing cast of a column used
    nowhere else in the projection, when the cast can be relocated into the
    producing projection through only column-`c`-transparent operators. Rewrites the
    producer's ``c = <expr>`` to ``c = cast(<expr>, T)`` and this item to ``col(c)``.
    Returns None when nothing pushes, so the rule is idempotent (one push per pass;
    the FUSION fixpoint repeats).
    """
    for idx, item in enumerate(node.items):
        target = _bare_col_cast(item.expr)
        if target is None:
            continue
        col_name, dtype, try_cast = target
        # Every use of `col_name` in this projection must be this one cast, so
        # narrowing it earlier cannot affect any other output column.
        if sum(1 for it in node.items if col_name in referenced_columns(it.expr)) != 1:
            continue
        new_input = _push_to_producer(node.input, col_name, dtype, try_cast)
        if new_input is None:
            continue
        new_items = list(node.items)
        new_items[idx] = Projection(item.alias, Col(col_name))
        return dataclasses.replace(node, input=new_input, items=tuple(new_items))
    return None


def _bare_col_cast(expr: object) -> tuple[str, str, bool] | None:
    """``(column, dtype, try_cast)`` if `expr` is exactly ``cast(col(x), dtype)``."""
    if isinstance(expr, Cast) and isinstance(expr.input, Col):
        return expr.input.name, expr.dtype, expr.try_cast
    return None


def _push_to_producer(
    node: LogicalPlan, col_name: str, dtype: str, try_cast: bool
) -> LogicalPlan | None:
    """Relocate ``cast(col(col_name), dtype)`` into the node that produces `col_name`.

    Walks down through operators that do not read `col_name` (Filter/Sort/Limit/
    Sample) to the producing `Project`, folding the cast into its defining expression
    when that genuinely narrows the column. Returns the rewritten subtree, or None if
    the column is read on the way down, the producer isn't a plain projection, or the
    cast would not narrow.
    """
    if isinstance(node, Project):
        for i, item in enumerate(node.items):
            if item.alias != col_name:
                continue
            if isinstance(item.expr, Cast) and item.expr.dtype == dtype:
                return None  # already cast to this type — nothing to gain (idempotent)
            if not _narrows(item.expr, node.input, dtype):
                return None  # not actually narrower → no benefit, leave it
            new_items = list(node.items)
            new_items[i] = Projection(col_name, Cast(item.expr, dtype, try_cast=try_cast))
            return dataclasses.replace(node, items=tuple(new_items))
        return None  # projection doesn't produce the column (shouldn't reach here)

    # Transparent single-input operators: push through only if they don't read `col`.
    if isinstance(node, Filter):
        if col_name in referenced_columns(node.predicate):
            return None
    elif isinstance(node, Sort):
        if any(col_name in referenced_columns(k.expr) for k in node.keys):
            return None
    elif isinstance(node, (Limit, Sample)):
        pass  # neither reads column values
    else:
        return None  # Join/Aggregate/Union/Distinct/Unnest/… — stop, can't prove safe

    pushed = _push_to_producer(node.input, col_name, dtype, try_cast)
    return None if pushed is None else dataclasses.replace(node, input=pushed)


def _narrows(expr: object, input_node: LogicalPlan, dtype: str) -> bool:
    """Whether casting `expr` to `dtype` reduces its byte width (else no benefit)."""
    schema = input_node.available_schema()
    if schema is None:
        return False
    src = infer_type(expr, schema)
    target = DTYPE_REGISTRY.get(dtype)
    if src is None or target is None:
        return False
    src_w, tgt_w = _byte_width(src), _byte_width(target)
    return src_w is not None and tgt_w is not None and tgt_w < src_w


def _byte_width(t: pa.DataType) -> int | None:
    """The fixed byte width of a primitive numeric type, or None if not applicable."""
    try:
        return t.bit_width // 8
    except (ValueError, AttributeError):
        return None
