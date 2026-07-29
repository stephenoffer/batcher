"""Drop a list reordering that the operation above it cannot see.

`list_contains(list_sort(x), v)` asks whether a value is in the list, and sorting the list
first does not change the answer. Neither does reversing it, nor deduplicating it — a
value is present in `unique(x)` exactly when it is present in `x`. The sort is a per-row
`O(n log n)` that produces nothing the predicate reads.

The same argument covers `list_sort(list_reverse(x))`: sorting imposes a total order that
erases whatever order the input had, so any reordering underneath it is dead. And
`list_len(list_transform(x, f))` is `list_len(x)`, because a transform is elementwise and
length-preserving by construction.

Two shapes that look like they belong here and do not:

* `list_len(list_unique(x))` is **not** `list_len(x)` — deduplication is the one
  reordering-adjacent operation that changes the length.
* `list_get(list_sort(x), 0)` is not `list_min(x)`. It would be, were it not for where a
  sort places nulls; that is an engine detail this module deliberately does not encode.

Every rule keeps the same operand and drops a call that is null-strict, so a null list
yields a null answer on both sides.
"""

from __future__ import annotations

from collections.abc import Callable

from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, node_rule
from batcher.kyber.rules.leaf_rewrite import rewrite_node
from batcher.plan.expr_ir import Expr
from batcher.plan.expr_ir.func_nodes import ListContains, ListFunc, ListTransform
from batcher.plan.logical import Aggregate, Filter, Project, Sort, Window

__all__ = [
    "LIST_CONTAINS_THROUGH_REORDER_RULES",
    "collapse_sort_of_reordering",
    "list_len_through_list_transform",
]

_NODES = (Filter, Project, Aggregate, Sort, Window)

#: List calls that permute or deduplicate the elements without changing the *set* of
#: values present. A membership test commutes with all three.
_REORDERINGS = ("sort", "reverse", "unique")


def _contains_leaf(fn: str) -> Callable[[Expr], Expr]:
    def leaf(expr: Expr) -> Expr:
        if (
            isinstance(expr, ListContains)
            and isinstance(expr.input, ListFunc)
            and expr.input.fn == fn
        ):
            return ListContains(expr.input.input, expr.value)
        return expr

    return leaf


def _register(name: str, leaf: Callable[[Expr], Expr], expr_matches: tuple[type, ...]):
    return DEFAULT_REGISTRY.add(
        node_rule(
            name,
            Phase.NORMALIZE,
            lambda node, _ctx, _leaf=leaf: rewrite_node(node, _leaf),
            matches=_NODES,
            expr_fn=leaf,
            expr_matches=expr_matches,
        )
    )


#: `list_contains(list_sort(x), v)` / `list_reverse` / `list_unique` -> `list_contains(x, v)`.
#: Membership depends on the set of elements, which none of the three changes.
LIST_CONTAINS_THROUGH_REORDER_RULES = [
    _register(f"list_contains_through_list_{fn}", _contains_leaf(fn), (ListContains,))
    for fn in _REORDERINGS
]


def _sort_of_reordering(expr: Expr) -> Expr:
    if (
        isinstance(expr, ListFunc)
        and expr.fn == "sort"
        and isinstance(expr.input, ListFunc)
        and expr.input.fn == "reverse"
    ):
        return ListFunc("sort", expr.input.input)
    return expr


collapse_sort_of_reordering = _register(
    "collapse_list_sort_of_list_reverse", _sort_of_reordering, (ListFunc,)
)
"""`list_sort(list_reverse(x)) -> list_sort(x)`.

Sorting imposes a total order on the elements, so whatever order they arrived in is
discarded. The reversal underneath produces nothing the sort keeps. The same-function case
(`sort(sort(x))`) is `collapse_idempotent_list_sort`; this is its mixed-pair sibling."""


def _len_of_transform(expr: Expr) -> Expr:
    if isinstance(expr, ListFunc) and expr.fn == "len" and isinstance(expr.input, ListTransform):
        return ListFunc("len", expr.input.input)
    return expr


list_len_through_list_transform = _register(
    "list_len_through_list_transform", _len_of_transform, (ListFunc,)
)
"""`list_len(list_transform(x, f)) -> list_len(x)`.

`transform` is elementwise: it maps each element to one output element and preserves the
list's length by construction. So the transform is evaluated for every element only to
have its result counted and discarded — the length was already known from `x`. Unlike
`list_filter`, which is the shape this rule must not be extended to, `transform` cannot
drop an element."""
