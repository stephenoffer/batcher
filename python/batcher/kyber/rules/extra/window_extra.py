"""Window rewrites — prune keys, frames and functions a window does not actually need.

`window_rules` holds the *syntactic* window simplifications (duplicate keys, literal-constant
keys, dead functions). This module adds the ones that need either a **proof from metadata**
(a column proven constant or unique, an input proven to hold ≤ 1 row) or a reading of the
frame/function specification, and it shares the metadata predicates with `agg_rules` rather
than re-spelling them.

A window is order-sensitive in a way an aggregate is not, so each rule is bounded by three
facts:

* **Changing the keys changes the values.** A partition key may only be dropped when it is
  the same value in every row (proven constant); an order key only when it cannot break a
  tie — because it is constant, because it repeats a partition key (constant *within* the
  partition), or because a *proven-unique* earlier key already totally orders the rows.
* **The frame is part of the value.** A frame is only removed where it is provably the
  engine's default (whole-partition, i.e. an unbounded frame with no ORDER BY), and a
  function is only folded when its frame is the default one — an explicit frame can be
  *empty* at a row (``2 FOLLOWING``…), where the value is NULL rather than the constant.
* **`rank_limit` filters rows.** Any rule that would drop or reorder functions stands down
  when it is set, except `rank_limit_zero_to_empty`, which is about exactly that.

Every rewrite is passed through `agg_rules._checked`: the result must have a byte-identical
output schema (names, order, types) or it is refused.
"""

from __future__ import annotations

import dataclasses
import json

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase

# The metadata predicates and the schema guard live once, in `agg_rules` (copy-paste is the
# one *wrong* way to share — python-quality.md).
from batcher.kyber.rules.extra.agg_rules import (
    _checked,
    _child_stats,
    _constant_value,
    _unique_column,
)
from batcher.plan.expr_ir import Col, Expr, Lit, when
from batcher.plan.logical import Limit, LogicalPlan, Project, Projection, Window, WindowFuncSpec

__all__ = [
    "constant_window_function_folding",
    "dedupe_window_functions",
    "drop_order_key_after_unique_key_in_window_order",
    "drop_order_key_equal_to_partition_key",
    "drop_order_key_proven_constant",
    "drop_order_keys_under_unbounded_frames",
    "drop_partition_key_proven_constant",
    "drop_redundant_unbounded_frame",
    "rank_limit_zero_to_empty",
    "simplify_window_over_single_row_partition",
]

# Ranking functions whose value over a one-row partition is exactly 1. `percent_rank`,
# `cume_dist` and `ntile` are excluded: their one-row values (0, 1.0, 1) are not simply "1",
# and their engine output type is not the Int64 a folded `Lit(1)` would give.
_UNIT_RANKING = frozenset({"row_number", "rank", "dense_rank"})
# Value functions that return the current row's input over a one-row partition (`lag`/`lead`
# are excluded — at offset ≥ 1 they are NULL, whose literal has no column type).
_ROW_VALUE_FNS = frozenset({"first_value", "last_value", "forward_fill", "backward_fill"})
# Windowed aggregates whose value over a *non-empty* frame of a constant column is that
# constant (`sum`/`avg`/`count` depend on how many rows the frame holds, so they are not).
_EXTREME_FNS = frozenset({"min", "max"})


def _sig(expr: Expr) -> object:
    """A comparable identity for a key expression — its lowered IR, never ``==``.

    `Expr.__eq__` builds a *comparison expression*, so keys are compared by `to_ir()`.
    """
    return expr.to_ir()


def _default_frame(fn: WindowFuncSpec) -> bool:
    """Whether `fn` runs over the engine's default frame (no explicit one was given)."""
    return fn.frame is None


def _whole_partition_frame(fn: WindowFuncSpec) -> bool:
    """Whether `fn` carries an explicit frame spanning the entire partition, in any units."""
    return fn.frame is not None and fn.frame.start is None and fn.frame.end is None


def _project_window(
    win: Window, kept: tuple[WindowFuncSpec, ...], folded: dict[str, Expr]
) -> LogicalPlan:
    """Rebuild `win` with only `kept`, re-deriving every original output column in a `Project`.

    The window's output is its input columns followed by one column per function, in order —
    the projection reproduces exactly that, so the schema is preserved. With no function left
    the `Window` itself disappears (it is row-preserving; callers refuse a `rank_limit`, which
    is not).
    """
    items = [Projection(c, Col(c)) for c in win.input.available_columns()]
    items += [Projection(fn.alias, folded.get(fn.alias, Col(fn.alias))) for fn in win.functions]
    inner: LogicalPlan = win.input
    if kept:
        inner = Window(win.input, win.partition_keys, win.order_keys, kept, win.rank_limit)
    return Project(inner, tuple(items))


def _prune_order_keys(node: Window, drop) -> LogicalPlan | None:
    """Drop the order keys `drop` accepts, keeping at least one (else the rule stands down).

    An empty `order_keys` is a *different operator*: ranking functions become invalid and a
    windowed aggregate switches from running to whole-partition. So a rule that prunes keys
    never prunes the last one — `drop_order_keys_under_unbounded_frames` is the one rewrite
    allowed to remove them all, and it proves no function reads the order first.
    """
    kept = tuple(key for key in node.order_keys if not drop(key))
    if len(kept) == len(node.order_keys) or not kept:
        return None
    return _checked(node, dataclasses.replace(node, order_keys=kept))


@rule(name="drop_partition_key_proven_constant", phase=Phase.REWRITE, matches=(Window,))
def drop_partition_key_proven_constant(node: Window, ctx: OptimizerContext) -> LogicalPlan | None:
    """``PARTITION BY c, …`` → drop `c` when it is proven a single non-null constant.

    A column with an EXACT ``min == max`` and no NULLs holds one value in every row, so it
    never splits a partition — every row already agrees on it. Dropping every partition key
    is fine too: that is one partition over all rows, which is what partitioning by a constant
    already means. Extends `drop_constant_partition_key`, which only sees a *literal* key.
    """
    stats = _child_stats(node, ctx)
    if stats is None or not node.partition_keys:
        return None
    kept = tuple(
        key
        for key in node.partition_keys
        if not (isinstance(key, Col) and _constant_value(stats, key.name) is not None)
    )
    if len(kept) == len(node.partition_keys):
        return None
    return _checked(node, dataclasses.replace(node, partition_keys=kept))


@rule(name="drop_order_key_proven_constant", phase=Phase.REWRITE, matches=(Window,))
def drop_order_key_proven_constant(node: Window, ctx: OptimizerContext) -> LogicalPlan | None:
    """``ORDER BY c, x`` → ``ORDER BY x`` when `c` is proven a single non-null constant.

    A constant column compares equal on every pair of rows, so it orders nothing and leaves
    every peer group (and therefore every RANGE frame) unchanged — ascending, descending, or
    nulls-first alike, since there are no NULLs. At least one key is kept, so ranking
    functions stay valid and a running aggregate stays running.
    """
    stats = _child_stats(node, ctx)
    if stats is None:
        return None
    return _prune_order_keys(
        node,
        lambda key: isinstance(key.expr, Col) and _constant_value(stats, key.expr.name) is not None,
    )


@rule(name="drop_order_key_equal_to_partition_key", phase=Phase.NORMALIZE, matches=(Window,))
def drop_order_key_equal_to_partition_key(
    node: Window, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """``PARTITION BY p ORDER BY p, x`` → ``ORDER BY x`` — a partition key cannot order.

    Every row of a partition carries the same partition-key value (NULLs group into their own
    partition, so that holds there too), so an order key that repeats a partition key is
    constant *within* the partition: it breaks no tie, moves no row and changes no peer group.
    Structural — no statistics needed. At least one order key is kept.
    """
    if not node.partition_keys or not node.order_keys:
        return None
    part = [_sig(key) for key in node.partition_keys]
    return _prune_order_keys(node, lambda key: _sig(key.expr) in part)


@rule(
    name="drop_order_key_after_unique_key_in_window_order", phase=Phase.REWRITE, matches=(Window,)
)
def drop_order_key_after_unique_key_in_window_order(
    node: Window, ctx: OptimizerContext
) -> LogicalPlan | None:
    """``ORDER BY id, x, y`` → ``ORDER BY id`` when `id` is proven unique on the input.

    A proven-unique column (EXACT ``ndv >= rows``) gives every row a different value, so it
    already totally orders them: no two rows tie on it, and the keys after it can never be
    consulted. Dropping them shrinks the sort key and leaves ranks, peer groups and RANGE
    frames identical (every row is its own peer either way).
    """
    stats = _child_stats(node, ctx)
    if stats is None or len(node.order_keys) < 2:
        return None
    for i, key in enumerate(node.order_keys[:-1]):
        if isinstance(key.expr, Col) and _unique_column(stats, key.expr.name):
            return _checked(node, dataclasses.replace(node, order_keys=node.order_keys[: i + 1]))
    return None


@rule(name="drop_redundant_unbounded_frame", phase=Phase.NORMALIZE, matches=(Window,))
def drop_redundant_unbounded_frame(node: Window, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop an ``UNBOUNDED PRECEDING .. UNBOUNDED FOLLOWING`` frame on an unordered window.

    With no ORDER BY, the engine's default frame for a windowed aggregate *is* the whole
    partition — which is precisely what an unbounded-to-unbounded frame asks for, in any
    units. Stating it explicitly only costs the engine its frame machinery, so the frame is
    dropped. Gated on ``order_keys`` being empty: with an order, the default is the running
    (RANGE) frame, and the two are genuinely different values.
    """
    if node.order_keys:
        return None
    new = tuple(
        dataclasses.replace(fn, frame=None) if _whole_partition_frame(fn) else fn
        for fn in node.functions
    )
    if all(a is b for a, b in zip(new, node.functions, strict=True)):
        return None
    return _checked(node, dataclasses.replace(node, functions=new))


@rule(name="drop_order_keys_under_unbounded_frames", phase=Phase.NORMALIZE, matches=(Window,))
def drop_order_keys_under_unbounded_frames(
    node: Window, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """Drop the ORDER BY when *every* function spans the whole partition anyway — a sort saved.

    A function whose frame is ``UNBOUNDED PRECEDING .. UNBOUNDED FOLLOWING`` reads its entire
    partition regardless of the row order, so the ordering it is computed under is unobservable
    in its output. When every function in the window is such an aggregate, the order keys (and
    with them a full sort, the window's dominant cost) are dead. The `all(...)` gate implies
    there is no ranking or fill function — those carry no frame and *do* read the order — and
    no `rank_limit`, which requires a ranking function.
    """
    if not node.order_keys or not node.functions:
        return None
    if not all(_whole_partition_frame(fn) for fn in node.functions):
        return None
    return _checked(node, dataclasses.replace(node, order_keys=()))


@rule(name="dedupe_window_functions", phase=Phase.REWRITE, matches=(Window,))
def dedupe_window_functions(node: Window, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Compute a window function that appears under two aliases once, aliasing the copies.

    Two functions with the same name, input, offset and frame produce byte-identical columns
    over the same partitioning and ordering, so the duplicate is computed once and the extra
    aliases are re-derived as projections of the representative. The `Project` reproduces the
    window's output columns in order, so the schema is unchanged. Stands down on a
    `rank_limit` window (a single ranking function — nothing to deduplicate).
    """
    if node.rank_limit is not None or len(node.functions) < 2:
        return None
    seen: dict[str, str] = {}
    kept: list[WindowFuncSpec] = []
    folded: dict[str, Expr] = {}
    for fn in node.functions:
        ident = json.dumps(dataclasses.replace(fn, alias="").to_ir(), sort_keys=True)
        rep = seen.get(ident)
        if rep is None:
            seen[ident] = fn.alias
            kept.append(fn)
        else:
            folded[fn.alias] = Col(rep)
    if not folded:
        return None
    return _checked(node, _project_window(node, tuple(kept), folded))


@rule(name="constant_window_function_folding", phase=Phase.REWRITE, matches=(Window,))
def constant_window_function_folding(node: Window, ctx: OptimizerContext) -> LogicalPlan | None:
    """Fold a window function over a proven-constant column to that constant.

    Over a column whose EXACT ``min == max`` and whose ``null_count`` is 0, every row carries
    the same value ``v`` — so ``MIN``/``MAX`` over any **non-empty** frame, and
    ``FIRST_VALUE``/``LAST_VALUE``/a fill over any partition, are all ``v``. The default frame
    always contains the current row, so it is never empty; an *explicit* frame can be (``2
    FOLLOWING``… past the partition's end, where the value is NULL), so a framed function is
    refused. ``SUM``/``AVG``/``COUNT`` are refused as well: they depend on how many rows the
    frame holds, not only on the value.
    """
    stats = _child_stats(node, ctx)
    if stats is None or node.rank_limit is not None:
        return None
    kept: list[WindowFuncSpec] = []
    folded: dict[str, Expr] = {}
    foldable = _EXTREME_FNS | _ROW_VALUE_FNS
    for fn in node.functions:
        value = None
        if isinstance(fn.input, Col) and _default_frame(fn) and fn.func in foldable:
            value = _constant_value(stats, fn.input.name)
        if value is None:
            kept.append(fn)
        else:
            folded[fn.alias] = Lit(value)
    if not folded:
        return None
    return _checked(node, _project_window(node, tuple(kept), folded))


@rule(name="simplify_window_over_single_row_partition", phase=Phase.REWRITE, matches=(Window,))
def simplify_window_over_single_row_partition(
    node: Window, ctx: OptimizerContext
) -> LogicalPlan | None:
    """Fold a window over a *proven* ≤ 1-row input into a `Project` — no partitions to build.

    With at most one row there is at most one partition of one row, so every function's value
    is decided without sorting or framing: the ranking functions are 1, ``COUNT(x)`` is 0/1 by
    nullness, and the extremes / first / last / fills are the row's own value. An empty input
    is covered by the same rewrite (the projection yields 0 rows, exactly as the window does).

    The count must be EXACT, every function must fold (a partial fold keeps the breaker), no
    function may carry an explicit frame (which can be empty even on a one-row partition), and
    a `rank_limit` window is refused — it filters rows, which a projection does not.
    """
    stats = _child_stats(node, ctx)
    if stats is None or node.rank_limit is not None:
        return None
    if not stats.rows_exact or stats.rows > 1:
        return None
    folded: dict[str, Expr] = {}
    for fn in node.functions:
        value = _single_row_window_value(fn)
        if value is None:
            return None
        folded[fn.alias] = value
    return _checked(node, _project_window(node, (), folded))


def _single_row_window_value(fn: WindowFuncSpec) -> Expr | None:
    """The value `fn` takes over a partition of exactly one row, or None if not foldable."""
    if not _default_frame(fn):
        return None  # an explicit frame may select no row at all — the value is then NULL
    if fn.func in _UNIT_RANKING:
        return Lit(1)
    if fn.input is None:
        return None
    if fn.func == "count":
        return when(fn.input.is_null()).then(0).otherwise(1)
    if fn.func in _EXTREME_FNS or fn.func in _ROW_VALUE_FNS:
        return fn.input
    return None


@rule(name="rank_limit_zero_to_empty", phase=Phase.FUSION, matches=(Window,))
def rank_limit_zero_to_empty(node: Window, _ctx: OptimizerContext) -> LogicalPlan | None:
    """A window with ``rank_limit == 0`` keeps no row — run it over the empty relation instead.

    ``QUALIFY rn < 1`` fuses (via `qualify_to_partition_topn`) into a per-partition top-0: a
    window that computes every rank and then discards every row. Feeding it the canonical empty
    marker ``Limit(x, 0)`` instead — and clearing the now-pointless `rank_limit` — produces the
    same 0 rows with the same schema, and lets `window_over_empty` / `propagate_empty_relation`
    hoist the emptiness up the plan. Idempotent: the rebuilt window has no `rank_limit`.
    """
    if node.rank_limit != 0:
        return None
    empty = Window(
        Limit(node.input, 0), node.partition_keys, node.order_keys, node.functions, rank_limit=None
    )
    return _checked(node, empty)
