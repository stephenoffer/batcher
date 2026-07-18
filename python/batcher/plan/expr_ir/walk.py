"""Structural traversals over the expression tree.

`referenced_columns` collects the input column names an expression reads;
`remap_columns` returns a copy with column names rewritten (used to push a
predicate through a join). Both walk every node kind, so they import the node
classes from `core` and `namespaces`.
"""

from __future__ import annotations

import dataclasses

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.audio import AudioFunc
from batcher.plan.expr_ir.core import (
    AggExpr,
    Aliased,
    Binary,
    Cast,
    Coalesce,
    Expr,
    InList,
    IsInf,
    IsNan,
    IsNotNull,
    IsNull,
    Math2Expr,
    MathExpr,
    Not,
)
from batcher.plan.expr_ir.func_nodes import WindowBuckets, WindowStart
from batcher.plan.expr_ir.image import ImageFunc
from batcher.plan.expr_ir.namespaces import (
    ConvertTimezone,
    DateFunc,
    DateOffset,
    DateTrunc,
    ListBinary,
    ListContains,
    ListFilter,
    ListFunc,
    ListGet,
    ListPosition,
    ListSet,
    ListSimhash,
    ListSlice,
    ListTransform,
    MapFunc,
    Strftime,
    StrFunc,
    Strptime,
    StructField,
)
from batcher.plan.expr_ir.node_base import IRNode, child_fields
from batcher.plan.expr_ir.nodes import (
    Array,
    Case,
    Col,
    Greatest,
    HashRows,
    Least,
    ListJoin,
    MakeStruct,
    NullIf,
    Sequence,
)
from batcher.plan.expr_ir.video import VideoFunc


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


def referenced_columns(expr: Expr) -> set[str]:
    """The set of input column names an expression reads.

    Memoized on the (immutable) node: the projection/pushdown/fusion rules call this
    on the same expressions across every fixpoint iteration, and it otherwise re-walks
    the whole subtree each time. The result is used only as a read-only operand
    (``need |= referenced_columns(e)``, ``<=``, ``in`` — verified: no caller mutates it),
    so sharing the cached set is safe. `Expr` sets no `__slots__`, so every node has a
    `__dict__` to cache in.
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
    cols = _referenced_columns_impl(expr)
    expr.__dict__["_c_refcols"] = cols
    return cols


def _referenced_columns_impl(expr: Expr) -> set[str]:
    if isinstance(expr, Col):
        return {expr.name}
    if isinstance(expr, Aliased):
        # A mid-expression alias (``(col("x").alias("y") + 1)``) is transparent — it
        # reads whatever its wrapped expression reads. Missing this arm pruned the
        # underlying column and failed the query with "unknown column".
        return referenced_columns(expr.inner)
    if isinstance(expr, Binary):
        return referenced_columns(expr.left) | referenced_columns(expr.right)
    if isinstance(
        expr,
        (
            Not,
            Cast,
            InList,
            IsNull,
            IsNotNull,
            IsNan,
            IsInf,
            StrFunc,
            Strftime,
            Strptime,
            ConvertTimezone,
            DateFunc,
            DateOffset,
            DateTrunc,
            ImageFunc,
            AudioFunc,
            VideoFunc,
            MathExpr,
            ListFunc,
            ListGet,
            ListSimhash,
            ListContains,
            ListPosition,
            ListTransform,
            ListFilter,
            ListSlice,
            ListJoin,
            StructField,
            MapFunc,
            WindowStart,
            WindowBuckets,
        ),
    ):
        return referenced_columns(expr.input)
    if isinstance(expr, (Coalesce, Greatest, HashRows, Least)):
        cols: set[str] = set()
        for e in expr.inputs:
            cols |= referenced_columns(e)
        return cols
    if isinstance(expr, Array):
        out: set[str] = set()
        for e in expr.elements:
            out |= referenced_columns(e)
        return out
    if isinstance(expr, MakeStruct):
        cols: set[str] = set()
        for _name, value in expr.fields:
            cols |= referenced_columns(value)
        return cols
    if isinstance(expr, Sequence):
        return (
            referenced_columns(expr.start)
            | referenced_columns(expr.stop)
            | referenced_columns(expr.step)
        )
    if isinstance(expr, (NullIf, Math2Expr, ListBinary, ListSet)):
        return referenced_columns(expr.left) | referenced_columns(expr.right)
    if isinstance(expr, Case):
        cols = referenced_columns(expr.otherwise)
        for cond, then in expr.branches:
            cols |= referenced_columns(cond) | referenced_columns(then)
        return cols
    return set()  # Lit and other leaves reference nothing


def remap_columns(expr: Expr, mapping: dict[str, str]) -> Expr:
    """Return a copy of `expr` with column names rewritten via `mapping`.

    Used to push a predicate through a join: a conjunct phrased in the join's
    output names is rewritten into one side's source names before being attached
    below the join.
    """
    if isinstance(expr, Col):
        return Col(mapping.get(expr.name, expr.name))
    if isinstance(expr, Aliased):
        return Aliased(remap_columns(expr.inner, mapping), expr.name)
    if isinstance(expr, Binary):
        return Binary(
            expr.op, remap_columns(expr.left, mapping), remap_columns(expr.right, mapping)
        )
    if isinstance(expr, Not):
        return Not(remap_columns(expr.input, mapping))
    if isinstance(expr, Cast):
        return Cast(remap_columns(expr.input, mapping), expr.dtype, try_cast=expr.try_cast)
    if isinstance(expr, InList):
        return InList(remap_columns(expr.input, mapping), expr.values)
    if isinstance(expr, IsNull):
        return IsNull(remap_columns(expr.input, mapping))
    if isinstance(expr, IsNotNull):
        return IsNotNull(remap_columns(expr.input, mapping))
    if isinstance(expr, IsNan):
        return IsNan(remap_columns(expr.input, mapping))
    if isinstance(expr, IsInf):
        return IsInf(remap_columns(expr.input, mapping))
    if isinstance(expr, StrFunc):
        return StrFunc(
            expr.fn,
            remap_columns(expr.input, mapping),
            pattern=expr.pattern,
            replacement=expr.replacement,
            start=expr.start,
            length=expr.length,
        )
    if isinstance(expr, DateFunc):
        return DateFunc(expr.fn, remap_columns(expr.input, mapping))
    if isinstance(expr, ImageFunc):
        return ImageFunc(
            expr.fn, remap_columns(expr.input, mapping), width=expr.width, height=expr.height
        )
    if isinstance(expr, AudioFunc):
        return AudioFunc(expr.fn, remap_columns(expr.input, mapping), expr.rate)
    if isinstance(expr, VideoFunc):
        return VideoFunc(expr.fn, remap_columns(expr.input, mapping))
    if isinstance(expr, DateTrunc):
        return DateTrunc(remap_columns(expr.input, mapping), expr.unit)
    if isinstance(expr, Strftime):
        return Strftime(remap_columns(expr.input, mapping), expr.format)
    if isinstance(expr, Strptime):
        return Strptime(remap_columns(expr.input, mapping), expr.format)
    if isinstance(expr, ConvertTimezone):
        return ConvertTimezone(remap_columns(expr.input, mapping), expr.from_tz, expr.to_tz)
    if isinstance(expr, DateOffset):
        return DateOffset(remap_columns(expr.input, mapping), expr.months, expr.days, expr.micros)
    if isinstance(expr, WindowStart):
        return WindowStart(
            remap_columns(expr.input, mapping), expr.width_micros, expr.origin_micros
        )
    if isinstance(expr, WindowBuckets):
        return WindowBuckets(
            remap_columns(expr.input, mapping), expr.width_micros, expr.slide_micros
        )
    if isinstance(expr, MathExpr):
        return MathExpr(expr.fn, remap_columns(expr.input, mapping))
    if isinstance(expr, ListFunc):
        return ListFunc(expr.fn, remap_columns(expr.input, mapping))
    if isinstance(expr, ListGet):
        return ListGet(remap_columns(expr.input, mapping), expr.index)
    if isinstance(expr, ListSimhash):
        return ListSimhash(remap_columns(expr.input, mapping), expr.num_bits, expr.seed)
    if isinstance(expr, ListContains):
        return ListContains(remap_columns(expr.input, mapping), expr.value)
    if isinstance(expr, ListPosition):
        return ListPosition(remap_columns(expr.input, mapping), expr.value)
    if isinstance(expr, ListTransform):
        return ListTransform(remap_columns(expr.input, mapping), expr.func)
    if isinstance(expr, ListFilter):
        return ListFilter(remap_columns(expr.input, mapping), expr.pred)
    if isinstance(expr, ListSlice):
        return ListSlice(remap_columns(expr.input, mapping), expr.offset, expr.length)
    if isinstance(expr, StructField):
        return StructField(remap_columns(expr.input, mapping), expr.field)
    if isinstance(expr, MapFunc):
        return MapFunc(expr.fn, remap_columns(expr.input, mapping), expr.key)
    if isinstance(expr, ListJoin):
        return ListJoin(remap_columns(expr.input, mapping), expr.separator)
    if isinstance(expr, ListBinary):
        return ListBinary(
            expr.fn, remap_columns(expr.left, mapping), remap_columns(expr.right, mapping)
        )
    if isinstance(expr, ListSet):
        return ListSet(
            expr.fn, remap_columns(expr.left, mapping), remap_columns(expr.right, mapping)
        )
    if isinstance(expr, Array):
        return Array([remap_columns(e, mapping) for e in expr.elements])
    if isinstance(expr, MakeStruct):
        return MakeStruct([(n, remap_columns(v, mapping)) for n, v in expr.fields])
    if isinstance(expr, Sequence):
        return Sequence(
            remap_columns(expr.start, mapping),
            remap_columns(expr.stop, mapping),
            remap_columns(expr.step, mapping),
        )
    if isinstance(expr, Coalesce):
        return Coalesce([remap_columns(e, mapping) for e in expr.inputs])
    if isinstance(expr, Greatest):
        return Greatest([remap_columns(e, mapping) for e in expr.inputs])
    if isinstance(expr, HashRows):
        return HashRows([remap_columns(e, mapping) for e in expr.inputs], expr.seed)
    if isinstance(expr, Least):
        return Least([remap_columns(e, mapping) for e in expr.inputs])
    if isinstance(expr, NullIf):
        return NullIf(remap_columns(expr.left, mapping), remap_columns(expr.right, mapping))
    if isinstance(expr, Math2Expr):
        return Math2Expr(
            expr.fn, remap_columns(expr.left, mapping), remap_columns(expr.right, mapping)
        )
    if isinstance(expr, Case):
        return Case(
            [(remap_columns(c, mapping), remap_columns(t, mapping)) for c, t in expr.branches],
            remap_columns(expr.otherwise, mapping),
        )
    return expr  # literals unchanged


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
    """True if `expr` is, or transitively contains, an `AggExpr` leaf."""
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
