"""Projection rewrites — collapse stacked projections and prune unread columns.

Chained `select` / `with_columns` / `rename` / `drop` build a stack of `Project`
nodes, each a full pass over the batch. `merge_projections` folds two adjacent
projections into one by substituting the inner projection's expressions into the
outer's — `Project(Project(x))` becomes a single `Project(x)` computing the same
columns. It guards against inlining a computation more than once, so the merge
never increases work: it removes an operator (and drops any unused inner column),
or it does nothing.

`rewrite_projection` (the whole-plan `projection_rewrite` rule) prunes columns the
plan never uses. It walks top-down tracking the *required* column set, narrowing it
at each operator, and rewrites `Join` nodes to drop output columns no downstream
operator consumes. `required_columns_per_source` is the companion analysis that
arrives at each `Scan` with exactly the columns it must produce (the source's read
projection) — consumed by the optimizer to build `PhysicalPlan.source_projections`.
"""

from __future__ import annotations

import dataclasses

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.expr_ir import AggExpr, Col, Expr, referenced_columns
from batcher.plan.expr_ir.walk import column_occurrence_counts
from batcher.plan.expr_rewrite import substitute_columns as _substitute_cols
from batcher.plan.logical import (
    Aggregate,
    AsofJoin,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    MapBatches,
    Project,
    Projection,
    RowId,
    Sample,
    Scan,
    Sort,
    Union,
    Unnest,
    Unpivot,
    WatermarkDedup,
    WatermarkStreamJoin,
    Window,
)

__all__ = [
    "eliminate_identity_project",
    "merge_projections",
    "projection_inlining_into_agg",
    "push_filter_through_project",
    "required_columns_per_source",
    "required_predicates_per_source",
    "rewrite_projection",
]


def _map_batches_need(node: MapBatches, need: set[str]) -> set[str]:
    """The columns a `map_batches` requires from its input — the one place that decides.

    A `map_batches` is a black box: the control plane cannot read the Python `fn`. So unless
    the `fn`'s inputs are *declared*, every column of the input must be kept alive — pruning to
    `need` (what the operators *above* the UDF consume) would starve an `fn` that reads a column
    it does not re-emit, silently changing its result. That is the safe default, and it is
    expensive: an embedding stage over one column of a 41-column Parquet table read all 41.

    When `input_columns` **is** declared, the UDF's true inputs are known, and the answer is
    exactly those plus whatever the plan above still needs (the UDF may pass columns through,
    and a consumer above may want them). Everything else is prunable, and the scan shrinks.

    Both pushdown walks — the plan rewrite and the per-source projection — call this, so the
    "what does a UDF need" rule exists once and the two cannot drift into disagreeing.

    Args:
        node: The `MapBatches` node.
        need: Columns the operators above this node consume.

    Returns:
        The set of input columns that must be preserved beneath `node`.
    """
    if node.input_columns is None:
        return set(node.input.available_columns())  # opaque: keep everything
    available = set(node.input.available_columns())
    # Intersect with what the input actually has: `need` may name columns the UDF *creates*.
    return (set(node.input_columns) | need) & available


@rule(name="projection_inlining_into_agg", phase=Phase.REWRITE, matches=(Aggregate,))
def projection_inlining_into_agg(node: Aggregate, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Aggregate(Project(x, renames))` → `Aggregate(x, …)` with the rename inlined.

    A pure pass-through / rename projection feeding an aggregate is one operator of
    pure overhead: the aggregate can read `x`'s columns directly. Every reference in
    the group keys and aggregate inputs to a projection alias is substituted by its
    defining column, and the projection is dropped. Restricted to projections whose
    items are all bare columns (rename/passthrough), so inlining adds zero compute
    (a computed projection might be referenced several times). Returns None otherwise.
    """
    proj = node.input
    if not isinstance(proj, Project):
        return None
    mapping: dict[str, Expr] = {}
    for item in proj.items:
        if not isinstance(item.expr, Col):
            return None  # a computed projection — leave it (avoid duplicating work)
        mapping[item.alias] = item.expr

    def subst(expr: Expr) -> Expr:
        return _substitute_cols(expr, mapping)

    # Build the new aggregate directly over `proj.input` (rewriting and re-parenting in
    # one step — a piecewise rebuild would transiently reference columns of neither side).
    new_keys = tuple(Projection(k.alias, subst(k.expr)) for k in node.group_keys)
    new_aggs = []
    for spec in node.aggregates:
        if spec.agg.input is None:
            new_aggs.append(spec)
            continue
        input2 = subst(spec.agg.input2) if spec.agg.input2 is not None else None
        agg = AggExpr(spec.agg.func, subst(spec.agg.input), param=spec.agg.param, input2=input2)
        new_aggs.append(dataclasses.replace(spec, agg=agg))
    # The watermark names an event-time column of the aggregate's *input*. Dropping the
    # projection re-parents the aggregate onto `proj.input`, where that column may be
    # known by its pre-rename name — so the watermark has to be remapped through the
    # same substitution the keys and aggregates got. Rebuilding without it at all (the
    # previous behavior) silently unbounded the streaming state.
    watermark = node.watermark
    if watermark is not None:
        source = mapping.get(watermark.time_col)
        if source is None:
            return None  # the time column is not produced by this projection — don't guess
        watermark = dataclasses.replace(watermark, time_col=source.name)
    return Aggregate(proj.input, new_keys, tuple(new_aggs), watermark)


@rule(name="eliminate_identity_project", phase=Phase.NORMALIZE, matches=(Project,))
def eliminate_identity_project(node: Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Project(x)` → `x` when the projection outputs exactly `x`'s columns, in
    order, unchanged (every item is `Col(c)` aliased to `c`). Such no-op projections
    are commonly generated by `select`-all, column pruning, or other rewrites;
    removing them drops a whole pass over the batch."""
    input_cols = node.input.available_columns()
    if len(node.items) != len(input_cols):
        return None
    for item, col_name in zip(node.items, input_cols, strict=True):
        is_passthrough = isinstance(item.expr, Col) and item.expr.name == col_name
        if item.alias != col_name or not is_passthrough:
            return None
    return node.input


@rule(name="merge_projections", phase=Phase.NORMALIZE, matches=(Project,))
def merge_projections(node: Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Project(Project(x))` → one `Project(x)` by inlining the inner expressions.

    The outer projection's expressions reference the inner projection's output
    columns; substituting each inner column with its defining expression yields
    equivalent expressions over the inner *input*, so the two passes collapse to
    one with an identical result and schema.

    Guarded to never duplicate work: if any inner output column is referenced more
    than once by the outer projection, inlining it would compute it repeatedly, so
    the merge is skipped. (Inner columns referenced zero times simply vanish —
    dead-column elimination.)
    """
    inner = node.input
    if not isinstance(inner, Project):
        return None
    counts = column_occurrence_counts([it.expr for it in node.items])
    if any(counts.get(it.alias, 0) > 1 for it in inner.items):
        return None
    inner_map = {it.alias: it.expr for it in inner.items}
    new_items = tuple(
        Projection(it.alias, _substitute_cols(it.expr, inner_map)) for it in node.items
    )
    return Project(inner.input, new_items)


@rule(name="push_filter_through_project", phase=Phase.PUSHDOWN, matches=(Filter,))
def push_filter_through_project(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Filter(Project(x), p)` → `Project(Filter(x, p'), items)` when `p` references
    only pass-through (rename) columns of the projection.

    Pushing the filter below the projection evaluates it earlier (and lets it reach
    a join/scan underneath). It is only safe for predicates over *pass-through*
    columns — aliases whose projection expression is a bare `Col`. If the predicate
    depended on a *computed* column, pushing the filter under the projection would
    reorder that computation relative to the filter and could change null/error
    behaviour (e.g. a row the filter would drop might now never hit a division that
    errors). Restricting to renames makes `p'` a pure column-renaming of `p`, so no
    computation moves across the filter and the result is identical.
    """
    inner = node.input
    if not isinstance(inner, Project):
        return None
    passthrough = {it.alias: it.expr for it in inner.items if isinstance(it.expr, Col)}
    if not referenced_columns(node.predicate) <= set(passthrough):
        return None
    new_pred = _substitute_cols(node.predicate, passthrough)
    return Project(Filter(inner.input, new_pred), inner.items)


def _window_child_need(node: Window, need: set[str]) -> set[str]:
    """Columns a Window's child must produce: the downstream-needed *input*
    columns plus every column the partition/order keys and function inputs read.
    (Function-output aliases come from the Window itself, not the child.)"""
    input_cols = set(node.input.available_columns())
    child_need = set(need) & input_cols
    for expr in node.partition_keys:
        child_need |= referenced_columns(expr)
    for key in node.order_keys:
        child_need |= referenced_columns(key.expr)
    for fn in node.functions:
        if fn.input is not None:
            child_need |= referenced_columns(fn.input)
    return child_need


def _unnest_child_need(node: Unnest, need: set[str]) -> set[str]:
    """Columns an Unnest's child must produce: the exploded list `column` always
    (it determines the output row count, even when its values aren't read), plus
    every downstream-needed pass-through column. The `alias` output maps back to
    `column` on the child."""
    child_need = {node.column}
    for n in need:
        child_need.add(node.column if n == node.alias else n)
    return child_need


def rewrite_projection(plan: LogicalPlan) -> LogicalPlan:
    """Prune columns the plan never uses.

    Currently this rewrites `Join` nodes to drop output columns no downstream
    operator consumes (a join materializes a fixed output list, so unused columns
    must be removed from the plan, not merely left unread). Other nodes only
    reference the columns they use, so pruning their sources via
    `required_columns_per_source` suffices.
    """
    return _rewrite(plan, set(plan.available_columns()))


def _surviving_items(node: Project, need: set[str]) -> tuple[Projection, ...]:
    """The `Project` items that survive pruning when only `need` is consumed downstream.

    A projection that produces nothing anyone wants still has to produce *something* —
    it is the operator that emits the rows a `count(*)` above it counts — so an empty
    result falls back to the first item.

    Shared by `_rewrite` (which builds the pruned `Project`) and `_visit` (which decides
    what the scan beneath it must read). The two MUST agree: if `_visit` assumed the
    projection kept no items while `_rewrite` kept one, the source would be read without
    the columns that item's expression references, and the scan would fail with an
    unknown column at execution time.
    """
    kept = tuple(item for item in node.items if item.alias in need)
    return kept or node.items[:1]


def _columns_read_by(items: tuple[Projection, ...]) -> set[str]:
    """The union of columns the expressions of `items` reference."""
    child_need: set[str] = set()
    for item in items:
        child_need |= referenced_columns(item.expr)
    return child_need


def _rewrite(node: LogicalPlan, need: set[str]) -> LogicalPlan:
    # Each branch returns `node` unchanged when recursion pruned nothing and rewrote no
    # child (preserving identity → O(1) fixpoint). `kept`/`new_output` are subsequences
    # of the originals, so an equal length means nothing was dropped (a value `==` would
    # invoke `Expr.__eq__` and build a comparison — never compare projection items that way).
    if isinstance(node, Scan):
        return node
    if isinstance(node, Filter):
        child = _rewrite(node.input, need | referenced_columns(node.predicate))
        return node if child is node.input else Filter(child, node.predicate)
    if isinstance(node, Project):
        # Drop output columns nothing downstream consumes (keep ≥1), then prune
        # the scan to only what the surviving expressions read.
        kept = _surviving_items(node, need)
        child = _rewrite(node.input, _columns_read_by(kept))
        if child is node.input and len(kept) == len(node.items):
            return node
        return Project(child, kept)
    if isinstance(node, Aggregate):
        child_need = set()
        for key in node.group_keys:
            child_need |= referenced_columns(key.expr)
        for spec in node.aggregates:
            if spec.agg.input is not None:
                child_need |= referenced_columns(spec.agg.input)
            # arg_min/arg_max also reference an ordering key (the second input).
            if spec.agg.input2 is not None:
                child_need |= referenced_columns(spec.agg.input2)
        # A watermark's event-time column is read by the streaming driver, not by any
        # expression in the plan, so column pruning cannot see that it is needed. Pruning
        # it away leaves the driver with no clock: the watermark never advances, closed
        # windows never evict, and the "bounded" state grows forever.
        if node.watermark is not None:
            child_need.add(node.watermark.time_col)
        child = _rewrite(node.input, child_need)
        if child is node.input:
            return node
        return Aggregate(child, node.group_keys, node.aggregates, node.watermark)
    if isinstance(node, Sort):
        child_need = set(need)
        for key in node.keys:
            child_need |= referenced_columns(key.expr)
        child = _rewrite(node.input, child_need)
        return node if child is node.input else Sort(child, node.keys, node.limit)
    if isinstance(node, Window):
        child_need = _window_child_need(node, need)
        child = _rewrite(node.input, child_need)
        if child is node.input:
            return node
        return Window(
            child,
            node.partition_keys,
            node.order_keys,
            node.functions,
            node.rank_limit,
        )
    if isinstance(node, RowId):
        # The index `alias` is synthesized here (reads no input column); pass the
        # remaining downstream needs through to the child.
        child = _rewrite(node.input, need - {node.alias})
        return node if child is node.input else RowId(child, node.alias, node.offset)
    if isinstance(node, Unnest):
        # The exploded `column` is always needed (it sets the row count); other
        # downstream needs pass through, with `alias` mapping back to `column`.
        child = _rewrite(node.input, _unnest_child_need(node, need))
        return node if child is node.input else Unnest(child, node.column, node.alias)
    if isinstance(node, Unpivot):
        # Unpivot's child must produce the index + melted `on` columns; the output's
        # variable/value columns are synthesized here, not read from the child.
        child_need = set(node.index) | set(node.on)
        child = _rewrite(node.input, child_need)
        if child is node.input:
            return node
        return Unpivot(
            child,
            node.index,
            node.on,
            node.variable_name,
            node.value_name,
        )
    if isinstance(node, Sample):
        # Sample preserves the schema, but it does NOT preserve the ROWS under pruning: a row
        # is kept iff a seeded hash of ALL its values falls under the fraction, so dropping a
        # column changes which rows survive. Its child therefore needs its full schema, not
        # just what is needed above — pruning here made `sample(0.1).select("k")` return a
        # different row count than `sample(0.1)`.
        child = _rewrite(node.input, set(node.input.available_columns()))
        return node if child is node.input else Sample(child, node.fraction, node.seed, node.n)
    if isinstance(node, Limit):
        child = _rewrite(node.input, set(need))
        return node if child is node.input else Limit(child, node.n, node.offset)
    if isinstance(node, Distinct):
        # Distinct uses every column, so its child needs them all.
        child = _rewrite(node.input, set(node.input.available_columns()))
        return node if child is node.input else Distinct(child)
    if isinstance(node, Union):
        inputs = tuple(_rewrite(i, set(i.available_columns())) for i in node.inputs)
        if all(a is b for a, b in zip(inputs, node.inputs, strict=True)):
            return node
        return Union(inputs, node.distinct)
    if isinstance(node, Join):
        # Always retain the key output columns (named after the left keys) so the
        # join keeps ≥1 column and the keys stay available.
        key_aliases = set(node.left_keys)
        new_output = tuple(
            spec for spec in node.output if spec.alias in need or spec.alias in key_aliases
        )
        left_need = set(node.left_keys) | {s.name for s in new_output if s.side == "left"}
        right_need = set(node.right_keys) | {s.name for s in new_output if s.side == "right"}
        left = _rewrite(node.left, left_need)
        right = _rewrite(node.right, right_need)
        if left is node.left and right is node.right and len(new_output) == len(node.output):
            return node
        return Join(
            left,
            right,
            node.left_keys,
            node.right_keys,
            node.join_type,
            new_output,
        )
    if isinstance(node, AsofJoin):
        # Each child needs its match keys (`on` + `by`) plus the output columns drawn
        # from that side; the output spec itself is left intact.
        left_need = {
            node.left_on,
            *node.left_by,
            *(s.name for s in node.output if s.side == "left"),
        }
        right_need = {
            node.right_on,
            *node.right_by,
            *(s.name for s in node.output if s.side == "right"),
        }
        left = _rewrite(node.left, left_need)
        right = _rewrite(node.right, right_need)
        if left is node.left and right is node.right:
            return node
        return AsofJoin(
            left,
            right,
            node.left_on,
            node.right_on,
            node.left_by,
            node.right_by,
            node.direction,
            node.output,
        )
    if isinstance(node, MapBatches):
        child = _rewrite(node.input, _map_batches_need(node, need))
        return node if child is node.input else dataclasses.replace(node, input=child)
    if isinstance(node, WatermarkDedup):
        # The dedup emits whole rows, so everything needed above must survive — plus the
        # two column sets its *state* depends on, which nothing above it references: the
        # `subset` keys it dedups on and the `event_time` it evicts by. Pruning either
        # would not fail loudly; it would silently dedup on the wrong key or stall the
        # watermark, which is a wrong answer on an unbounded input only.
        child_need = set(need) | set(node.subset) | {node.event_time}
        child = _rewrite(node.input, child_need)
        return node if child is node.input else dataclasses.replace(node, input=child)
    if isinstance(node, WatermarkStreamJoin):
        # Mirrors the `Join` arm, plus each side's event-time column: the interval
        # predicate and the state eviction both read it even though it need not appear
        # in the output.
        new_output = tuple(spec for spec in node.output if spec.alias in need)
        if not new_output:
            new_output = node.output
        left_need = {
            *node.left_keys,
            node.left_time,
            *(s.name for s in new_output if s.side == "left"),
        }
        right_need = {
            *node.right_keys,
            node.right_time,
            *(s.name for s in new_output if s.side == "right"),
        }
        left = _rewrite(node.left, left_need)
        right = _rewrite(node.right, right_need)
        if left is node.left and right is node.right and len(new_output) == len(node.output):
            return node
        return dataclasses.replace(node, left=left, right=right, output=new_output)
    raise TypeError(f"projection rewrite: unhandled node {type(node).__name__}")


def required_columns_per_source(plan: LogicalPlan) -> dict[int, list[str]]:
    """Return, per scan `source_id`, the column projection to read.

    A scan with an empty requirement still reads one column so row counts are
    preserved (e.g. ``count(*)`` needs rows but no values).
    """
    required: dict[int, list[str]] = {}
    _visit(plan, set(plan.available_columns()), required)
    return required


def required_predicates_per_source(plan: LogicalPlan) -> dict[int, dict]:
    """Return, per scan `source_id`, the predicate IR of a `Filter` directly above it.

    After predicate-pushdown rules run, a residual `Filter` typically sits just
    above each `Scan`; its predicate is the candidate for source-side pushdown. A
    pushdown-capable source translates the pushable subset; the engine keeps the
    `Filter`, so recording the predicate here never changes results — at worst the
    source ignores it. Filters separated from the scan by other operators are not
    pushed (conservative), and multiple stacked filters over one scan are AND-combined.
    """
    acc: dict[int, dict] = {}
    _collect_scan_predicates(plan, None, acc)
    return acc


def _collect_scan_predicates(node: LogicalPlan, pending: Expr | None, acc: dict[int, dict]) -> None:
    """Walk down carrying the predicate of an immediately-enclosing `Filter`."""
    if isinstance(node, Scan):
        if pending is not None:
            acc[node.source_id] = pending.to_ir()
        return
    if isinstance(node, Filter):
        combined = node.predicate if pending is None else _and(pending, node.predicate)
        _collect_scan_predicates(node.input, combined, acc)
        return
    # Any other operator breaks the Filter→Scan adjacency: descend with no pending
    # predicate (children get their own immediately-enclosing filters, if any).
    for child in _children(node):
        _collect_scan_predicates(child, None, acc)


def _and(left: Expr, right: Expr) -> Expr:
    from batcher.plan.expr_ir import Binary

    return Binary("and", left, right)


def _children(node: LogicalPlan) -> tuple[LogicalPlan, ...]:
    if isinstance(node, (Join, AsofJoin)):
        return (node.left, node.right)
    if isinstance(node, Union):
        return tuple(node.inputs)
    inp = getattr(node, "input", None)
    return (inp,) if inp is not None else ()


def _visit(node: LogicalPlan, need: set[str], acc: dict[int, list[str]]) -> None:
    if isinstance(node, Scan):
        available = node.schema.names
        keep = [c for c in available if c in need]
        if not keep and available:
            # Read one column to preserve cardinality (count(*), etc.). A schemaless
            # (0-column) scan — e.g. an empty in-memory source — keeps none.
            keep = [available[0]]
        # Accumulate in the scan's schema order (never alphabetical): the projection
        # is applied to the source as-is, so its order is the output column order.
        cols = acc.setdefault(node.source_id, [])
        cols.extend(c for c in keep if c not in cols)

    elif isinstance(node, Filter):
        _visit(node.input, need | referenced_columns(node.predicate), acc)

    elif isinstance(node, Project):
        # Every item of *this* projection is evaluated when the plan runs, whether or not
        # anything above consumes it, so the scan must supply what each one reads. Do not
        # narrow by `need`: pruning unconsumed items is `_rewrite`'s job, and it has
        # already run. A projection that still holds a dead item (because a later rule
        # removed its only consumer) would otherwise read a column short and fail with
        # `unknown column` at execution.
        _visit(node.input, _columns_read_by(node.items), acc)

    elif isinstance(node, Aggregate):
        child_need = set()
        for key in node.group_keys:
            child_need |= referenced_columns(key.expr)
        for spec in node.aggregates:
            if spec.agg.input is not None:
                child_need |= referenced_columns(spec.agg.input)
            if spec.agg.input2 is not None:
                child_need |= referenced_columns(spec.agg.input2)
        _visit(node.input, child_need, acc)

    elif isinstance(node, Sort):
        child_need = set(need)
        for key in node.keys:
            child_need |= referenced_columns(key.expr)
        _visit(node.input, child_need, acc)

    elif isinstance(node, Window):
        _visit(node.input, _window_child_need(node, need), acc)

    elif isinstance(node, Limit):
        _visit(node.input, set(need), acc)

    elif isinstance(node, Distinct):
        _visit(node.input, set(node.input.available_columns()), acc)

    elif isinstance(node, Union):
        for inp in node.inputs:
            _visit(inp, set(inp.available_columns()), acc)

    elif isinstance(node, Join):
        # Like `Project`, a join emits every column of its declared `output`, so each
        # side must supply the ones it owns. `_rewrite` is what narrows `output` to the
        # consumed columns; narrowing again by `need` here would under-read.
        left_need = set(node.left_keys)
        right_need = set(node.right_keys)
        for col in node.output:
            (left_need if col.side == "left" else right_need).add(col.name)
        _visit(node.left, left_need, acc)
        _visit(node.right, right_need, acc)

    elif isinstance(node, AsofJoin):
        left_need = {node.left_on, *node.left_by}
        right_need = {node.right_on, *node.right_by}
        for col in node.output:
            (left_need if col.side == "left" else right_need).add(col.name)
        _visit(node.left, left_need, acc)
        _visit(node.right, right_need, acc)

    elif isinstance(node, RowId):
        _visit(node.input, need - {node.alias}, acc)

    elif isinstance(node, Unnest):
        _visit(node.input, _unnest_child_need(node, need), acc)

    elif isinstance(node, Unpivot):
        _visit(node.input, set(node.index) | set(node.on), acc)

    elif isinstance(node, Sample):
        # The sample hash reads every column of its input (see `_rewrite`), so the scan below
        # must supply them all; pruning to `need` would change which rows are sampled.
        _visit(node.input, set(node.input.available_columns()), acc)

    elif isinstance(node, MapBatches):
        _visit(node.input, _map_batches_need(node, need), acc)

    else:  # pragma: no cover - defensive
        raise TypeError(f"projection pushdown: unhandled node {type(node).__name__}")
