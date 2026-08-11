"""Structural traversals over the expression tree.

`referenced_columns` collects the input column names an expression reads;
`remap_columns` returns a copy with column names rewritten (used to push a
predicate through a join). Both walk every node kind, so they import the node
classes from `core` and `namespaces`.
"""

from __future__ import annotations

import dataclasses

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import (
    AggExpr,
    Aliased,
    Expr,
    InList,
    Lit,
)
from batcher.plan.expr_ir.func_nodes import ListFilter, ListTransform
from batcher.plan.expr_ir.node_base import IRNode, child_fields
from batcher.plan.expr_ir.nodes import (
    Case,
    Col,
    MakeStruct,
)


def column_occurrence_counts(exprs: list[Expr]) -> dict[str, int]:
    """How many times each column name *occurs* across `exprs` — occurrences, not distinct.

    The counting sibling of `referenced_columns`: the projection rules need to know whether
    merging two projections would evaluate a column twice, which a set cannot tell them. It
    lives here because two Kyber rules had each written it out (`_col_ref_counts` and
    `_occurrence_counts` — same body, different names), and a shared walk belongs with the
    other shared walks.

    Args:
        exprs: The expressions to count column references across.

    Returns:
        Column name to the number of times it is referenced.
    """
    # Deferred: `expr_rewrite` imports this package, so a module-level import would close an
    # `expr_ir -> expr_rewrite -> expr_ir` cycle.
    from batcher.plan.expr_rewrite import transform_expr_up

    counts: dict[str, int] = {}

    def tally(expr: Expr) -> Expr:
        if isinstance(expr, Col):
            counts[expr.name] = counts.get(expr.name, 0) + 1
        return expr

    for expr in exprs:
        transform_expr_up(expr, tally)
    return counts


def referenced_columns(expr: Expr) -> frozenset[str]:
    """The set of input column names an expression reads.

    Memoized on the (immutable) node: the projection/pushdown/fusion rules call this on the
    same expressions across every fixpoint iteration, and it otherwise re-walks the whole
    subtree each time. `Expr` sets no `__slots__`, so every node has a `__dict__` to cache in.

    **The result is a `frozenset` because the cache is shared by reference.** Every caller
    gets the *same* object, so one in-place update rewrites what that expression is recorded
    as reading, for every later reader, for the rest of the process. This used to be a
    comment promising that no caller mutated it — a promise that was already false in this
    very module, where `_referenced_columns_impl` seeded its accumulator from a child's
    cached answer and then unioned the siblings into it.

    The damage was not a bad estimate. `Project.__post_init__` validates its items with this
    function, so a projection could be reported as reading a column it does not reference,
    and TPC-DS q80 failed to plan with ``projection 'id' references unknown column(s)
    ['store_id']``. Immutability makes that mechanical: `need |= referenced_columns(e)`
    still works (it rebinds `need`), `<=`, `in` and iteration are unchanged, and an actual
    in-place mutation is now an `AttributeError` at the offending line.

    Args:
        expr: The expression to inspect.

    Returns:
        The input column names it reads, as an immutable set.
    """
    if isinstance(expr, AggExpr):
        # An aggregate reached a scalar-expression context (select/with_columns/filter).
        # `group_by().agg()` splits aggregate leaves out before building those, so one
        # arriving here escaped that path — reject it clearly (it also has no `__dict__`
        # to memoize into, being `__slots__`-based).
        raise PlanError(
            "an aggregate expression (e.g. col('x').sum()) can only be used inside "
            "group_by().agg(); it cannot appear in select/with_columns/filter"
        )
    cached = expr.__dict__.get("_c_refcols")
    if cached is not None:
        return cached
    cols = frozenset(_referenced_columns_impl(expr))
    expr.__dict__["_c_refcols"] = cols
    return cols


def _referenced_columns_impl(expr: Expr) -> frozenset[str] | set[str]:
    # The set of columns an expression reads is the union of what its sub-expressions
    # read, and an `IRNode` already declares which of its fields are sub-expressions. So
    # this walks that declaration rather than enumerating node types: a per-type cascade
    # silently returns the empty set for any node nobody added an arm for, which prunes a
    # real column and fails the query with "unknown column" — a bug the old `Aliased` arm
    # was added to fix, and one that cannot recur here.
    if isinstance(expr, Col):
        return {expr.name}
    if isinstance(expr, Aliased):
        # A mid-expression alias (``(col("x").alias("y") + 1)``) is transparent: it reads
        # whatever its wrapped expression reads. Not an `IRNode`, so handled here.
        return referenced_columns(expr.inner)
    if isinstance(expr, InList):
        return referenced_columns(expr.input)  # `values` are literals, not sub-expressions
    if isinstance(expr, (ListTransform, ListFilter)):
        # A higher-order list op reads its input list column. Its body (`func`/`pred`) is
        # evaluated in a *scope of its own* over the list's flattened elements, where the
        # only free name is the `element()` placeholder — a bound variable, not a column
        # read from this operator's input. The generic field-walk below would descend into
        # that body and surface `element` as an unknown column, which is exactly what
        # failed the list-HOF differential tests. Column references stop at the input.
        return referenced_columns(expr.input)
    if isinstance(expr, Case):
        cols = referenced_columns(expr.otherwise)
        for cond, then in expr.branches:
            cols |= referenced_columns(cond) | referenced_columns(then)
        return cols
    if isinstance(expr, MakeStruct):
        cols = set()
        for _name, value in expr.fields:
            cols |= referenced_columns(value)
        return cols
    if isinstance(expr, IRNode):
        cols = set()
        for name, is_list in child_fields(expr):
            value = getattr(expr, name)
            if value is None:
                continue
            for sub in value if is_list else (value,):
                cols |= referenced_columns(sub)
        return cols
    return set()  # Lit and other leaves reference nothing


def remap_columns(expr: Expr, mapping: dict[str, str]) -> Expr:
    """Return a copy of `expr` with column names rewritten via `mapping`.

    Used to push a predicate through a join: a conjunct phrased in the join's output
    names is rewritten into one side's source names before being attached below the join.

    Node types are not enumerated here. An `IRNode` already declares which of its fields
    hold sub-expressions (`child`/`children`), so the rewrite reads that declaration and
    rebuilds the node generically. That is not only shorter than a per-type cascade: a
    cascade silently *skips* any node nobody added an arm for, and skipping means the
    pushed predicate keeps the join's output column names, which reference columns the
    source below the join does not have. Adding an `Expr` node must not be able to
    introduce that, so nothing here has to be updated when one is.

    The handful of nodes handled explicitly are the ones whose sub-expressions are not
    plain fields: `Col` is the rewrite itself, and `Case`/`MakeStruct` nest theirs inside
    tuples.
    """

    def rewrite(e: Expr) -> Expr:
        return remap_columns(e, mapping)

    if isinstance(expr, Col):
        return Col(mapping.get(expr.name, expr.name))
    if isinstance(expr, Aliased):
        return Aliased(rewrite(expr.inner), expr.name)
    if isinstance(expr, InList):
        return InList(rewrite(expr.input), expr.values)
    if isinstance(expr, ListTransform):
        # Only the input list column is a column reference (see `_referenced_columns_impl`).
        # The body binds `element()` in its own scope, so rewriting it under a join's
        # output→source mapping is at best a no-op and at worst rebinds the placeholder;
        # leave it intact, exactly as the reference walk refuses to read columns from it.
        return ListTransform(rewrite(expr.input), expr.func)
    if isinstance(expr, ListFilter):
        return ListFilter(rewrite(expr.input), expr.pred)
    if isinstance(expr, Case):
        return Case(
            [(rewrite(cond), rewrite(then)) for cond, then in expr.branches],
            rewrite(expr.otherwise),
        )
    if isinstance(expr, MakeStruct):
        return MakeStruct([(name, rewrite(value)) for name, value in expr.fields])
    if isinstance(expr, IRNode):
        updates = {}
        for name, is_list in child_fields(expr):
            value = getattr(expr, name)
            if value is None:
                continue  # an optional sub-expression that was not given
            updates[name] = [rewrite(e) for e in value] if is_list else rewrite(value)
        return dataclasses.replace(expr, **updates) if updates else expr
    return expr  # literals and other leaves reference no column


# --- Aggregate-expression splitting -----------------------------------------------
# `group_by().agg()` accepts a whole expression *over* aggregates (``sum(x)/sum(y)``).
# The engine has no operator that evaluates arithmetic inside an aggregate, so the
# control plane lowers such an expression to two operators it already runs correctly on
# one core, many cores, and many machines: an Aggregate that computes each distinct
# aggregate *leaf* into a hidden column, and a Project that recomputes the surrounding
# scalar expression over those columns. These helpers own the leaf-extraction rewrite;
# because the aggregate stays the standard mergeable primitive and the projection is a
# stateless map, the split is identical single-node and distributed.

# The prefix for a hidden aggregate column. Double-underscored and namespaced so it
# cannot collide with a user column, group key, or output alias.
_HIDDEN_AGG_PREFIX = "__batcher_agg_"


def contains_aggregate(expr: object) -> bool:
    """True if `expr` is, or transitively contains, an `AggExpr` leaf.

    A bare column or literal answers by type. That is not a micro-optimization on a rare
    shape: the projection builder asks this of *every* output column, and a `select` or
    `with_columns` over a wide relation is overwhelmingly bare `Col` pass-throughs, so
    this is the per-column inner loop of building any wide projection. Answering it with
    one type check instead of four `isinstance` tests and a field walk is what keeps that
    proportional to the columns the call actually computes.
    """
    kind = type(expr)
    if kind is Col or kind is Lit:
        return False
    if isinstance(expr, AggExpr):
        return True
    # `Case` and `MakeStruct` carry their sub-expressions in irregular fields (paired
    # branches; named fields) declared without the `child`/`children` factories, so
    # `child_fields` cannot see them — walk those shapes explicitly, or an aggregate
    # inside a CASE / struct(...) would be invisible here and wrongly rejected by
    # `group_by().agg()` as "not an expression over aggregates".
    if isinstance(expr, Case):
        return contains_aggregate(expr.otherwise) or any(
            contains_aggregate(cond) or contains_aggregate(then) for cond, then in expr.branches
        )
    if isinstance(expr, MakeStruct):
        return any(contains_aggregate(value) for _name, value in expr.fields)
    if isinstance(expr, IRNode):
        for name, is_list in child_fields(expr):
            value = getattr(expr, name)
            if is_list:
                if any(contains_aggregate(v) for v in value):
                    return True
            elif contains_aggregate(value):
                return True
    return False


class AggregateLeafRegistry:
    """Collects the distinct `AggExpr` leaves of the composite specs into hidden columns.

    Deduplicates by the leaf's source-like ``repr`` so an aggregate written twice is
    computed once. Insertion order is preserved for a stable, testable plan shape.
    """

    def __init__(self) -> None:
        self._by_key: dict[str, tuple[str, AggExpr]] = {}

    def intern(self, agg: AggExpr) -> str:
        """Register `agg`, returning the hidden column name that will hold its result."""
        key = repr(agg)
        existing = self._by_key.get(key)
        if existing is not None:
            return existing[0]
        name = f"{_HIDDEN_AGG_PREFIX}{len(self._by_key)}"
        self._by_key[key] = (name, agg)
        return name

    def leaves(self) -> list[tuple[str, AggExpr]]:
        """The ``(hidden_name, aggregate)`` pairs to add to the Aggregate, in order."""
        return list(self._by_key.values())


def _reject_nested_aggregate(agg: AggExpr) -> None:
    """An aggregate of an aggregate (``sum(x).mean()``) has no meaning — reject it early."""
    for part in (agg.input, agg.input2):
        if part is not None and contains_aggregate(part):
            raise PlanError(
                "an aggregate cannot be nested inside another aggregate "
                f"(in {agg!r}); aggregate the inner value in a prior group_by()"
            )


def split_aggregate_leaves(expr: Expr | AggExpr, registry: AggregateLeafRegistry) -> Expr:
    """Return `expr` with each `AggExpr` leaf replaced by a `Col` to its hidden column.

    Registers every leaf into `registry`; the caller adds those to an Aggregate and wraps
    it in a Project that evaluates the returned expression. A bare aggregate collapses to a
    single column reference; a scalar-only expression is returned unchanged.
    """
    if isinstance(expr, AggExpr):
        _reject_nested_aggregate(expr)
        return Col(registry.intern(expr))
    # `Case`/`MakeStruct` hide their sub-expressions from `child_fields` (see
    # `contains_aggregate`), so `dataclasses.replace` below would never reach the
    # aggregate leaves inside them — rebuild those shapes explicitly, splitting each
    # sub-expression, so a leaf inside a CASE / struct(...) is hoisted like any other.
    if isinstance(expr, Case):
        return Case(
            [
                (split_aggregate_leaves(cond, registry), split_aggregate_leaves(then, registry))
                for cond, then in expr.branches
            ],
            split_aggregate_leaves(expr.otherwise, registry),
        )
    if isinstance(expr, MakeStruct):
        return MakeStruct(
            [(name, split_aggregate_leaves(value, registry)) for name, value in expr.fields]
        )
    if isinstance(expr, IRNode):
        updates = {}
        for name, is_list in child_fields(expr):
            value = getattr(expr, name)
            if is_list:
                updates[name] = [split_aggregate_leaves(v, registry) for v in value]
            else:
                updates[name] = split_aggregate_leaves(value, registry)
        if not updates:
            return expr
        return dataclasses.replace(expr, **updates)
    # Hand-written leaf nodes (Col, Lit, ...) carry no aggregate children; an aggregate
    # hidden in an unsupported node surfaces as a clear error when `referenced_columns`
    # or `AggExpr.to_ir()` reaches it.
    return expr


def broadcast_aggregate_leaves(expr: Expr | AggExpr) -> Expr:
    """Return `expr` with each `AggExpr` leaf turned into a whole-frame window.

    The rewrite behind ``with_columns(total=col("x").sum())`` and the mixed
    ``select("g", total=col("x").sum())``: an aggregate used where a *row-shaped* value
    is expected means the aggregate over the whole frame, broadcast to every row — which
    is exactly ``agg.over()``, the relational `Window` operator with no partition. Polars
    and pandas both read it that way, and reading it any other way would need a second
    evaluation model for aggregates.

    A `select` whose items are *all* aggregates is the other reading (collapse to one
    row) and is handled by the caller before this is reached.

    Args:
        expr: An expression that may contain `AggExpr` leaves.

    Returns:
        The same expression with every aggregate leaf replaced by its `.over()` window.
    """
    if isinstance(expr, AggExpr):
        _reject_nested_aggregate(expr)
        return expr.over()
    if isinstance(expr, Case):
        return Case(
            [
                (broadcast_aggregate_leaves(cond), broadcast_aggregate_leaves(then))
                for cond, then in expr.branches
            ],
            broadcast_aggregate_leaves(expr.otherwise),
        )
    if isinstance(expr, MakeStruct):
        return MakeStruct(
            [(name, broadcast_aggregate_leaves(value)) for name, value in expr.fields]
        )
    if isinstance(expr, IRNode):
        updates = {}
        for name, is_list in child_fields(expr):
            value = getattr(expr, name)
            if is_list:
                updates[name] = [broadcast_aggregate_leaves(v) for v in value]
            else:
                updates[name] = broadcast_aggregate_leaves(value)
        if not updates:
            return expr
        return dataclasses.replace(expr, **updates)
    return expr
