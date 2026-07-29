"""Window rewrites the existing window family leaves: transposition and ranking top-N.

`extra/window_rules` and `extra/window_extra` prune and de-duplicate a single `Window`
in place -- constant partition keys, duplicate functions, redundant frames -- and
`collapse_adjacent_windows` merges two stacked windows when their specs already match.
The two rules here are the ones that *move* a window, which is what makes the existing
ones fire more often and what turns a ranking window into a bounded operation.

`transpose_adjacent_windows` is Spark's `TransposeWindow`. A query computing several
window functions over different partitionings lowers to one `Window` node per spec, in
whatever order the user wrote them. If the writing order interleaves two specs that
would collapse when adjacent, `collapse_adjacent_windows` never sees them next to each
other and both survive -- and a window is a pipeline breaker that sorts and partitions
the whole relation, so an unnecessary one is one of the most expensive nodes a plan can
carry. Swapping two independent windows into a canonical order puts equal specs
together for the collapse rule to find.

`push_topn_into_unpartitioned_ranking_window` is Spark's `LimitPushDownThroughWindow`
and DataFusion's `window_topn`. `ROW_NUMBER() OVER (ORDER BY x)` followed by a `LIMIT k`
ranks every row of the relation to keep `k` of them. With no partitioning, the rows a
limit can retain are exactly the first `k` in the window's own order, so a top-N below
the window feeds it `k` rows instead of all of them -- an O(n log n) full sort plus a
full-width materialization becomes a bounded heap.
"""

from __future__ import annotations

import dataclasses

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.expr_ir.walk import referenced_columns
from batcher.plan.expr_rewrite import expr_key
from batcher.plan.logical import Limit, LogicalPlan, Sort, Window

__all__ = ["push_topn_into_unpartitioned_ranking_window", "transpose_adjacent_windows"]


def _order_key_ids(keys) -> tuple:
    """A comparable identity for a tuple of `SortKeySpec`s.

    `SortKeySpec` holds an `Expr`, and `Expr.__eq__` *builds a comparison expression*
    rather than answering a question -- so a plain `a == b` over two of these tuples
    raises `PlanError` instead of returning a bool. Every comparison here goes through
    `expr_key`, which renders the expression to its stable IR form.
    """
    return tuple((expr_key(k.expr), k.descending, k.nulls_first) for k in keys)


def _spec_key(node: Window) -> tuple:
    """A comparable identity for a window's partition/order spec.

    Two windows sharing this key are candidates for `collapse_adjacent_windows`;
    ordering by it is what brings them together.
    """
    return (
        tuple(expr_key(e) for e in node.partition_keys),
        _order_key_ids(node.order_keys),
    )


def _window_inputs(node: Window) -> set[str]:
    """Every column the window's own spec and functions read."""
    columns: set[str] = set()
    for expr in node.partition_keys:
        columns |= referenced_columns(expr)
    for key in node.order_keys:
        columns |= referenced_columns(key.expr)
    for fn in node.functions:
        if fn.input is not None:
            columns |= referenced_columns(fn.input)
    return columns


def _outputs(node: Window) -> set[str]:
    return {fn.alias for fn in node.functions}


@rule(name="transpose_adjacent_windows", phase=Phase.REWRITE, matches=(Window,))
def transpose_adjacent_windows(node: Window, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Swap two stacked independent `Window`s into a canonical spec order.

    A window is one of the most expensive nodes in a plan: it partitions and sorts the
    whole relation and materializes an extra column per function. `collapse_adjacent_windows`
    can merge two of them into one when their partition and order specs agree, but it
    only looks at a node and its immediate child -- so when a query writes three window
    functions over specs A, B, A, the two A windows are never neighbours and both are
    computed. Sorting adjacent independent windows by spec brings equal specs together
    and lets the collapse rule delete one.

    Two conditions make the swap sound, and both are checked:

    * The outer window must not read anything the inner one produces. If it partitions
      by, orders by, or aggregates over an inner output column, it genuinely depends on
      it and the order is fixed.
    * Their output aliases must be disjoint, so neither shadows the other and the
      column set above the pair is unchanged either way.

    Beyond that a window only appends columns -- it never adds, drops, or reorders rows
    -- so two independent ones commute exactly.

    Only strictly decreasing swaps are performed, which is what makes this terminate: a
    swap happens only when the inner spec sorts after the outer, so the fixpoint cannot
    cycle a pair back and forth.
    """
    inner = node.input
    if not isinstance(inner, Window):
        return None
    if _window_inputs(node) & _outputs(inner):
        return None  # the outer window consumes the inner one's result
    if _outputs(node) & _outputs(inner):
        return None  # overlapping aliases: swapping would change which one wins
    if _spec_key(inner) <= _spec_key(node):
        return None  # already in canonical order; swapping would not terminate
    return dataclasses.replace(inner, input=dataclasses.replace(node, input=inner.input))


#: The ranking functions whose value for a row depends only on the rows *ahead* of it
#: in the window order. Truncating the input to a prefix therefore leaves their value
#: on the surviving rows unchanged.
#:
#: `WINDOW_RANKING` is deliberately not reused here. It also contains `percent_rank`,
#: `cume_dist`, and `ntile`, every one of which divides by the partition's total row
#: count -- so truncating the input silently changes the value they compute for rows
#: that were kept, which is exactly the bug this narrower set exists to prevent.
_PREFIX_STABLE_RANKING = frozenset({"row_number", "rank", "dense_rank"})


def _prefix_stable_ranking(window: Window) -> bool:
    """Whether every function in `window` keeps its value under input truncation."""
    return bool(window.order_keys) and all(
        fn.func in _PREFIX_STABLE_RANKING for fn in window.functions
    )


@rule(
    name="push_topn_into_unpartitioned_ranking_window",
    phase=Phase.PUSHDOWN,
    matches=(Limit,),
)
def push_topn_into_unpartitioned_ranking_window(
    node: Limit, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`Limit(k, Window(row_number() OVER (ORDER BY x)))` gains a top-N below the window.

    With no partition keys the window is one partition over the whole relation, so it
    sorts every row and appends a rank to every row -- to feed a limit that keeps `k`.
    Since the window's order is the only order in play, the rows a positional limit can
    retain are exactly the first `k + offset` in that order, and a `Sort` carrying that
    cap below the window produces precisely those rows. The window then ranks
    `k + offset` rows instead of the whole relation, and a full sort plus a full-width
    materialization becomes a bounded heap.

    The preconditions are each load-bearing:

    * **No partition keys.** With partitioning, rank restarts per partition, so the
      globally-first `k` rows are not the rows a limit would keep. That case is
      `qualify_to_partition_topn`'s, and it needs a rank *predicate*, not a limit.
    * **Prefix-stable ranking functions only.** `row_number`, `rank`, and `dense_rank`
      depend on the rows ahead of a row in the order, and nothing else. Aggregates and
      value functions read elsewhere in the partition, and `percent_rank`, `cume_dist`,
      and `ntile` divide by the partition's total size -- all of them would compute a
      different value once the input is truncated. See `_PREFIX_STABLE_RANKING`.
    * **Order keys present**, which the ranking functions require anyway, and which is
      what makes "the first `k` rows" a defined set rather than an arbitrary one.
    * **The offset is counted in.** A `LIMIT k OFFSET m` needs `k + m` rows below, not
      `k`.

    Declines when a `Sort` with a smaller cap already sits there, so the rule reaches a
    fixpoint instead of rebuilding the same node forever.
    """
    window = node.input
    if not isinstance(window, Window) or window.partition_keys:
        return None
    if not _prefix_stable_ranking(window):
        return None
    needed = node.n + node.offset
    below = window.input
    if isinstance(below, Sort) and _order_key_ids(below.keys) == _order_key_ids(window.order_keys):
        if below.limit is not None and below.limit <= needed:
            return None
        capped = dataclasses.replace(below, limit=needed)
    else:
        capped = Sort(below, window.order_keys, limit=needed)
    return dataclasses.replace(node, input=dataclasses.replace(window, input=capped))
