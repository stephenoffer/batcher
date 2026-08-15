"""Struct, list, and array algebra -- the extract-over-construct family.

This is the surface Spark reaches with `SimplifyExtractValueOps` and `ComplexTypes`,
and it is worth more here than the name suggests. A SQL front end, a nested-JSON
reader, and the `.struct` / `.list` accessor namespaces all routinely produce a
container that is built and then immediately taken apart: `make_struct(a := x).a`,
`[p, q][0]`, `len([p, q])`. Every one of those materializes a whole nested Arrow
array -- offsets, child buffers, validity -- for a value that was already sitting in
a flat column. Cancelling the pair deletes the allocation outright rather than making
it cheaper, which is why these rules matter more than their arithmetic cousins.

The other half of the module is list-function algebra: the idempotence, involution,
and absorption identities that let a chain such as `sort(sort(x))` or
`reverse(reverse(x))` collapse. Each was confirmed against the engine rather than
assumed, including the null and empty-list rows, since a list kernel that propagates
null is what makes these safe to apply inside a `Project` as well as a `Filter`.

Two guards recur. Extract-over-construct may only discard the *other* elements of the
container when they are `safe_expr`, or a rewrite would delete an error the query would
have raised. And a rule that deletes a `sort`, `reverse`, or `unique` under a reduction
must know exactly which reductions are insensitive to what that call changes -- order,
or multiplicity. Both sets are narrower than they first appear, and the reasoning is on
`_ORDER_INDEPENDENT_REDUCTIONS` and `_DEDUP_INDEPENDENT_REDUCTIONS`.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY, rule
from batcher.kyber.rule import Phase, node_rule
from batcher.kyber.rules.leaf_rewrite import (
    collapse_doubled_call,
    collapse_involution,
    node_expr_rule,
    rewrite_node,
    safe_expr,
)
from batcher.plan.expr_ir import Expr, Lit
from batcher.plan.expr_ir.func_nodes import ListFunc, ListGet, ListSet, ListSlice, StructField
from batcher.plan.expr_ir.nodes import Array, MakeStruct
from batcher.plan.expr_rewrite import expr_key
from batcher.plan.logical import Filter, LogicalPlan, Project

__all__ = [
    "IDEMPOTENT_LIST_RULES",
    "REDUCTION_THROUGH_PERMUTATION_RULES",
    "REDUCTION_THROUGH_UNIQUE_RULES",
    "array_except_of_self_to_empty",
    "array_set_op_of_self",
    "drop_full_list_slice",
    "fold_list_get_of_array",
    "fold_list_length_of_array",
    "list_length_of_arg_sort",
    "list_reverse_involution",
    "list_slice_of_slice",
    "struct_field_of_make_struct",
]

#: List functions that permute a list without adding or removing elements. `unique` is
#: deliberately absent: it is the one member of the family that can shrink the list.
_PERMUTATIONS = frozenset({"sort", "reverse"})

#: List reductions whose result is *exactly* independent of element order, so a
#: permutation underneath one of them can be deleted.
#:
#: The floating-point accumulators are the notable absences. `sum`, `mean`, `product`,
#: `std`, `var`, and the norms all accumulate left to right, and floating-point addition
#: is not associative -- reversing a list can change the last bit of the answer. They are
#: order-independent in exact arithmetic and not in the arithmetic the engine runs, so
#: they stay out. `arg_min` and `arg_max` are absent for the opposite reason: they return
#: a *position*, which is what a permutation changes.
_ORDER_INDEPENDENT_REDUCTIONS = frozenset({"len", "min", "max", "n_unique", "max_abs", "median"})

#: List reductions whose result is unchanged by de-duplicating the input first, so a
#: `unique` underneath one of them can be deleted. `len` and `median` are absent because
#: both depend on multiplicity.
_DEDUP_INDEPENDENT_REDUCTIONS = frozenset({"min", "max", "n_unique", "max_abs"})

#: List functions that are genuinely idempotent -- applying them twice equals applying
#: them once. Sorting an already-sorted list, de-duplicating a de-duplicated one, and
#: re-normalizing a unit vector are all no-ops on the second pass, each confirmed against
#: the engine with the first call *materialized* so the optimizer could not fuse the pair.
#:
#: `arg_sort` and `flatten` were in this set and are unsound in it. `arg_sort` returns the
#: permutation that sorts its input, so applying it twice yields the *inverse*
#: permutation: `arg_sort([3,1,2])` is `[1,2,0]` and `arg_sort([1,2,0])` is `[2,0,1]`.
#: `flatten` removes exactly one level of nesting, so on a triply-nested list the second
#: call does real work: `flatten(flatten([[[1,2]],[[3]]]))` is `[1,2,3]` where one call
#: gives `[[1,2],[3]]`.
#:
#: `normalize` was here too and is idempotent only in *exact* arithmetic. Re-normalizing
#: an already-unit vector re-runs a division by a recomputed norm, and the last bit can
#: move: `normalize(normalize([2, 2]))` yields `0.7071067811865476` where one call yields
#: `...75`. That is the same non-associativity that keeps the norms out of
#: `_ORDER_INDEPENDENT_REDUCTIONS` below. Only `sort` and `unique` survive, and both are
#: exact because they permute or drop values without ever recomputing one.
#:
#: Both were "verified" by a test that evaluated the doubled expression and compared it
#: to the single one -- which the rule itself had already rewritten, so the test could
#: only ever agree with itself. `tests/differential` now materializes between the calls.
_IDEMPOTENT_LIST_FNS = frozenset({"sort", "unique"})


# --- extract over construct ---------------------------------------------------


def _struct_field_of_struct(expr: Expr) -> Expr:
    if isinstance(expr, StructField) and isinstance(expr.input, MakeStruct):
        chosen: Expr | None = None
        for name, value in expr.input.fields:
            if name == expr.field:
                chosen = value
        if chosen is not None and all(
            safe_expr(v) for name, v in expr.input.fields if name != expr.field
        ):
            return chosen
    return expr


@rule(
    name="struct_field_of_make_struct",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_struct_field_of_struct,
    expr_matches=(MakeStruct, StructField),
)
def struct_field_of_make_struct(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`make_struct(a := x, b := y).a -> x`. Building a struct and immediately reading
    one field back allocates a nested Arrow array -- child arrays plus a validity
    bitmap -- for a value already available as a flat column.

    The *last* field with the requested name wins, matching how the struct itself
    would resolve a duplicate name. The discarded siblings must all be `safe_expr`,
    since dropping them must not drop an error the query would otherwise raise."""
    return rewrite_node(node, _struct_field_of_struct)


def _list_get_of_array(expr: Expr) -> Expr:
    if isinstance(expr, ListGet) and isinstance(expr.input, Array):
        elements = expr.input.elements
        index = expr.index
        if 0 <= index < len(elements) and all(
            safe_expr(e) for i, e in enumerate(elements) if i != index
        ):
            return elements[index]
    return expr


@rule(
    name="fold_list_get_of_array",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_list_get_of_array,
    expr_matches=(Array, ListGet),
)
def fold_list_get_of_array(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`[x, y, z][1] -> y`. Indexing a freshly built array with an in-range constant
    selects a known element, so the list never needs to exist. Only a non-negative,
    in-range index is folded: a negative or out-of-range index has engine-defined
    behaviour this rule does not attempt to reproduce. The unselected elements must be
    `safe_expr` before they can be discarded."""
    return rewrite_node(node, _list_get_of_array)


def _list_len_of_array(expr: Expr) -> Expr:
    if (
        isinstance(expr, ListFunc)
        and expr.fn == "len"
        and isinstance(expr.input, Array)
        and all(safe_expr(e) for e in expr.input.elements)
    ):
        return Lit(len(expr.input.elements))
    return expr


@rule(
    name="fold_list_length_of_array",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_list_len_of_array,
    expr_matches=(Array, ListFunc),
)
def fold_list_length_of_array(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`len([x, y, z]) -> 3`. An array literal's length is its arity, known at plan
    time and independent of the element values -- a null element still occupies a
    slot. Every element is dropped, so all of them must be `safe_expr`."""
    return rewrite_node(node, _list_len_of_array)


# --- list-function algebra ----------------------------------------------------


# One rule per idempotent list function, over a shared body -- the registration shape
# `extra/temporal_sargable` uses for its own cross-product family. `sort(sort(x))` is
# `sort(x)`, and likewise for `unique`, `arg_sort`, `flatten`, and `normalize`: each maps
# its own output to itself, so the outer call is an O(n log n) sort or a hash
# de-duplication per row for a list that already has the property. Null and empty-list
# rows are unaffected -- every one of these kernels maps null to null and `[]` to `[]`.
IDEMPOTENT_LIST_RULES = [
    DEFAULT_REGISTRY.add(
        node_rule(
            f"collapse_idempotent_list_{fn}",
            Phase.NORMALIZE,
            node_expr_rule(collapse_doubled_call(ListFunc, fn)),
            matches=(Filter, Project),
            expr_fn=collapse_doubled_call(ListFunc, fn),
            expr_matches=(ListFunc,),
        )
    )
    for fn in sorted(_IDEMPOTENT_LIST_FNS)
]


#: `reverse(reverse(x))` over a list, through the shared involution factory.
_reverse_involution = collapse_involution(ListFunc, "reverse")


@rule(
    name="list_reverse_involution",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_reverse_involution,
    expr_matches=(ListFunc,),
)
def list_reverse_involution(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`reverse(reverse(x)) -> x`. Reversal is an involution, so a doubled call is the
    identity, including on the null and empty-list rows. Unlike the idempotent family
    this removes *both* calls rather than the outer one."""
    return rewrite_node(node, _reverse_involution)


def _reduce_through_permutation(reduction: str, permutation: str):
    """Build the leaf rewrite deleting one permutation under one order-independent
    reduction."""

    def leaf(expr: Expr) -> Expr:
        if (
            isinstance(expr, ListFunc)
            and expr.fn == reduction
            and isinstance(expr.input, ListFunc)
            and expr.input.fn == permutation
        ):
            return ListFunc(reduction, expr.input.input)
        return expr

    return leaf


# The (reduction x permutation) cross-product, one registered rule per pair.
#
# `max(sort(x)) -> max(x)`: a permutation rearranges a list without adding or removing
# elements, so a reduction that does not depend on order reads the same answer either
# way -- while the permutation costs an O(n log n) sort or a full copy on *every row*.
#
# Which reductions qualify is the whole rule, and the set is narrower than it looks:
# see `_ORDER_INDEPENDENT_REDUCTIONS` for why the floating-point accumulators and the
# `arg_*` position functions are excluded.
REDUCTION_THROUGH_PERMUTATION_RULES = [
    DEFAULT_REGISTRY.add(
        node_rule(
            f"{reduction}_through_list_{permutation}",
            Phase.NORMALIZE,
            node_expr_rule(_reduce_through_permutation(reduction, permutation)),
            matches=(Filter, Project),
            expr_fn=_reduce_through_permutation(reduction, permutation),
            expr_matches=(ListFunc,),
        )
    )
    for reduction in sorted(_ORDER_INDEPENDENT_REDUCTIONS)
    for permutation in sorted(_PERMUTATIONS)
]


def _len_of_arg_sort(expr: Expr) -> Expr:
    if (
        isinstance(expr, ListFunc)
        and expr.fn == "len"
        and isinstance(expr.input, ListFunc)
        and expr.input.fn == "arg_sort"
    ):
        return ListFunc("len", expr.input.input)
    return expr


@rule(
    name="list_length_of_arg_sort",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_len_of_arg_sort,
    expr_matches=(ListFunc,),
)
def list_length_of_arg_sort(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`len(arg_sort(x)) -> len(x)`. Sorting *indices* produces one index per element,
    so the result has the input's length and the sort is wasted -- O(n log n) per row
    for a number already in the offsets buffer.

    `arg_sort` is handled separately from the permutation set rather than added to it:
    it returns positions rather than values, so `max(arg_sort(x))` is emphatically not
    `max(x)`. Length is the one reduction that reads through it."""
    return rewrite_node(node, _len_of_arg_sort)


def _reduce_through_unique(reduction: str):
    """Build the leaf rewrite deleting a `unique` under one dedup-independent reduction."""

    def leaf(expr: Expr) -> Expr:
        if (
            isinstance(expr, ListFunc)
            and expr.fn == reduction
            and isinstance(expr.input, ListFunc)
            and expr.input.fn == "unique"
        ):
            return ListFunc(reduction, expr.input.input)
        return expr

    return leaf


# One rule per dedup-independent reduction. `min(unique(x)) -> min(x)`: de-duplication
# changes an element's multiplicity, never the set of values present, so a reduction
# that reads only the set is unaffected -- and the `unique` it sits on costs a hash set
# per row. `len` and `median` are absent because both depend on multiplicity, which is
# exactly what `unique` removes.
REDUCTION_THROUGH_UNIQUE_RULES = [
    DEFAULT_REGISTRY.add(
        node_rule(
            f"{reduction}_through_list_unique",
            Phase.NORMALIZE,
            node_expr_rule(_reduce_through_unique(reduction)),
            matches=(Filter, Project),
            expr_fn=_reduce_through_unique(reduction),
            expr_matches=(ListFunc,),
        )
    )
    for reduction in sorted(_DEDUP_INDEPENDENT_REDUCTIONS)
]


def _full_slice(expr: Expr) -> Expr:
    if isinstance(expr, ListSlice) and expr.offset == 0 and expr.length is None:
        return expr.input
    return expr


@rule(
    name="drop_full_list_slice",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_full_slice,
    expr_matches=(ListSlice,),
)
def drop_full_list_slice(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`slice(x, 0)` with no length `-> x`. A slice starting at the head with no bound
    returns the whole list, so it copies every offset and child value to produce the
    input again."""
    return rewrite_node(node, _full_slice)


def _slice_of_slice(expr: Expr) -> Expr:
    if (
        isinstance(expr, ListSlice)
        and isinstance(expr.input, ListSlice)
        and expr.offset >= 0
        and expr.input.offset >= 0
    ):
        inner = expr.input
        offset = inner.offset + expr.offset
        if inner.length is None:
            length = expr.length
        elif expr.length is None:
            length = max(inner.length - expr.offset, 0)
        else:
            length = max(min(expr.length, inner.length - expr.offset), 0)
        return ListSlice(inner.input, offset, length)
    return expr


@rule(
    name="list_slice_of_slice",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_slice_of_slice,
    expr_matches=(ListSlice,),
)
def list_slice_of_slice(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Compose nested slices into one: `slice(slice(x, a, m), b, n)` becomes a single
    slice at `a + b`. The composed length is the inner window clipped by the outer
    offset and then by the outer length, clamped at zero so an outer window past the
    inner one's end yields an empty list rather than a negative length.

    Restricted to non-negative offsets. A negative offset counts from the end, and
    composing two of those depends on the per-row list length, which is not a plan-time
    quantity."""
    return rewrite_node(node, _slice_of_slice)


def _set_op_of_self(expr: Expr) -> Expr:
    if (
        isinstance(expr, ListSet)
        and expr.fn in ("array_intersect", "array_union")
        and safe_expr(expr.left)
        and expr_key(expr.left) == expr_key(expr.right)
    ):
        return ListFunc("unique", expr.left)
    return expr


@rule(
    name="array_set_op_of_self",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_set_op_of_self,
    expr_matches=(ListSet,),
)
def array_set_op_of_self(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`array_intersect(x, x)` and `array_union(x, x)` both become `unique(x)`. A set
    operation of a list with itself keeps exactly the distinct elements, which is what
    `unique` computes directly -- and it is the de-duplication, not a bare `x`, that is
    the correct answer, since the set operations return sets.

    Needs `safe_expr` because the two evaluations of `x` collapse into one."""
    return rewrite_node(node, _set_op_of_self)


def _except_of_self(expr: Expr) -> Expr:
    if (
        isinstance(expr, ListSet)
        and expr.fn == "array_except"
        and safe_expr(expr.left)
        and expr_key(expr.left) == expr_key(expr.right)
    ):
        return ListSlice(expr.left, 0, 0)
    return expr


@rule(
    name="array_except_of_self_to_empty",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_except_of_self,
    expr_matches=(ListSet,),
)
def array_except_of_self_to_empty(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`array_except(x, x) -> slice(x, 0, 0)`, the empty list. Removing a list's own
    elements from it leaves nothing.

    The result is a zero-length slice of `x` rather than a literal empty list for one
    reason: it keeps the element type *and* the null propagation. A null input stays
    null through the slice, which a typeless empty-list literal could not express."""
    return rewrite_node(node, _except_of_self)
