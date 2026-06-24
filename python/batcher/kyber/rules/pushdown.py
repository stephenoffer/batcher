"""Predicate pushdown — evaluate filters as early as possible.

`rewrite_predicate` (the whole-plan `predicate_pushdown` rule) moves a `Filter`
below a `Join`: the predicate is split on `AND`, and each conjunct that references
only one side of the join is rewritten into that side's source column names and
attached beneath the join, so rows are eliminated before the (expensive) join
builds/probes. Conjuncts spanning both sides stay above the join. It is
semantics-preserving for inner joins; for outer joins it only pushes to a side that
is never null-extended (the preserved side).

`push_filter_through_aggregate` adds the node-local case of pushdown through
`Aggregate`. A predicate over an aggregate's *group-key* columns (not its aggregate
outputs) can be evaluated before grouping: every row in a group shares the group-key
values, so filtering groups by a key predicate is identical to filtering the input
rows by that predicate — but it runs on the (larger) pre-grouped input, eliminating
rows before the expensive grouping/aggregation.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.stats.selectivity import comparison_col_side
from batcher.plan.expr_ir import Binary, Col, Expr, referenced_columns, remap_columns
from batcher.plan.expr_rewrite import combine_conjuncts, split_conjuncts, substitute_columns
from batcher.plan.logical import (
    Aggregate,
    AsofJoin,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    Project,
    RowId,
    Sample,
    Scan,
    Sort,
    Union,
    Unnest,
    Unpivot,
    Window,
)

__all__ = [
    "infer_join_predicates",
    "push_filter_through_aggregate",
    "push_filter_through_sort",
    "push_filter_through_window",
    "rewrite_predicate",
]

# Comparisons whose `col OP literal` form is a constant constraint worth mirroring
# across an equi-join's key correspondence.
_INFERABLE_COMPARISONS = frozenset({"lt", "le", "gt", "ge", "eq", "ne"})


def rewrite_predicate(plan: LogicalPlan) -> LogicalPlan:
    """Push filters below joins where it is semantics-preserving."""
    return _pp(plan)


def _pp(node: LogicalPlan) -> LogicalPlan:
    # Each branch returns `node` unchanged (preserving object identity, so the driver
    # detects the fixpoint in O(1)) when recursion changed no child and no filter was
    # pushed; only a real rewrite allocates a new node.
    if isinstance(node, Scan):
        return node
    if isinstance(node, Filter):
        child = _pp(node.input)
        if isinstance(child, Join):
            pushed = _push_into_join(node.predicate, child)
            if pushed is not None:
                return pushed  # a conjunct moved below the join
        return node if child is node.input else Filter(child, node.predicate)
    if isinstance(node, Project):
        child = _pp(node.input)
        return node if child is node.input else Project(child, node.items)
    if isinstance(node, Aggregate):
        child = _pp(node.input)
        return node if child is node.input else Aggregate(child, node.group_keys, node.aggregates)
    if isinstance(node, Sort):
        child = _pp(node.input)
        return node if child is node.input else Sort(child, node.keys, node.limit)
    if isinstance(node, Window):
        child = _pp(node.input)
        if child is node.input:
            return node
        return Window(child, node.partition_keys, node.order_keys, node.functions, node.rank_limit)
    if isinstance(node, Limit):
        child = _pp(node.input)
        return node if child is node.input else Limit(child, node.n, node.offset)
    if isinstance(node, Join):
        left, right = _pp(node.left), _pp(node.right)
        if left is node.left and right is node.right:
            return node
        return Join(left, right, node.left_keys, node.right_keys, node.join_type, node.output)
    if isinstance(node, AsofJoin):
        # Recurse into both inputs; an ASOF predicate above stays above (no split).
        left, right = _pp(node.left), _pp(node.right)
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
    if isinstance(node, Distinct):
        child = _pp(node.input)
        return node if child is node.input else Distinct(child)
    if isinstance(node, Union):
        inputs = tuple(_pp(i) for i in node.inputs)
        if all(a is b for a, b in zip(inputs, node.inputs, strict=True)):
            return node
        return Union(inputs, node.distinct)
    if isinstance(node, RowId):
        # A filter must NOT push below RowId: removing rows under it would change the
        # row numbering. Stay above (recurse into the subtree only).
        child = _pp(node.input)
        return node if child is node.input else RowId(child, node.alias, node.offset)
    if isinstance(node, Unnest):
        # A filter above an Unnest stays above it: a predicate on the exploded
        # column cannot exist below the explode, so we conservatively never push
        # through (correctness over an extra optimization).
        child = _pp(node.input)
        return node if child is node.input else Unnest(child, node.column, node.alias)
    if isinstance(node, Unpivot):
        # A predicate above Unpivot references the reshaped variable/value columns,
        # which don't exist below it — stay above (no push-through).
        child = _pp(node.input)
        if child is node.input:
            return node
        return Unpivot(child, node.index, node.on, node.variable_name, node.value_name)
    if isinstance(node, Sample):
        # Pushing a filter below Sample would change which rows are eligible (and so
        # the sampled set). Keep it above — correctness over the extra pushdown.
        child = _pp(node.input)
        return node if child is node.input else Sample(child, node.fraction, node.seed, node.n)
    raise TypeError(f"predicate pushdown: unhandled node {type(node).__name__}")


def _push_into_join(predicate: Expr, join: Join) -> LogicalPlan | None:
    # Returns the rewritten plan, or `None` when no conjunct could be pushed (so the
    # caller keeps the original `Filter(join)` and preserves its identity).
    # Which sides may receive pushed predicates without changing results.
    # For an outer join, pushing to the null-supplying side is unsafe.
    can_push_left = join.join_type in {"inner", "left", "semi", "anti"}
    can_push_right = join.join_type in {"inner", "right"}

    left_map = {c.alias: c.name for c in join.output if c.side == "left"}
    right_map = {c.alias: c.name for c in join.output if c.side == "right"}
    # Join keys are always available on each side even if not in the output.
    for out_name, src in zip(join.left_keys, join.left_keys, strict=True):
        left_map.setdefault(out_name, src)
    for out_name, src in zip(join.left_keys, join.right_keys, strict=True):
        right_map.setdefault(out_name, src)

    left_aliases = set(left_map)
    right_aliases = set(right_map)

    left_push: list[Expr] = []
    right_push: list[Expr] = []
    keep: list[Expr] = []
    for conj in split_conjuncts(predicate):
        cols = referenced_columns(conj)
        if can_push_left and cols <= left_aliases:
            left_push.append(remap_columns(conj, left_map))
        elif can_push_right and cols <= right_aliases:
            right_push.append(remap_columns(conj, right_map))
        else:
            keep.append(conj)

    if not left_push and not right_push:
        return None  # nothing moved → caller keeps the original Filter(join)

    new_left = join.left
    if left_push:
        new_left = Filter(new_left, combine_conjuncts(left_push))
    new_right = join.right
    if right_push:
        new_right = Filter(new_right, combine_conjuncts(right_push))

    result: LogicalPlan = Join(
        new_left, new_right, join.left_keys, join.right_keys, join.join_type, join.output
    )
    if keep:
        result = Filter(result, combine_conjuncts(keep))
    return result


@rule(name="infer_join_predicates", phase=Phase.PUSHDOWN, matches=(Join,))
def infer_join_predicates(node: Join, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Mirror a constant key-constraint across an inner join's equi-key pairs.

    For `A ⋈ B ON a.k = b.k`, the keys are equal on every surviving (matched) row,
    so a constant constraint on one side's key holds on the other's too. If one
    input carries a `key OP literal` filter (e.g. a dimension table filtered to
    `region = 'EU'`), this rewrite adds the equivalent filter to the *other* input
    (the fact table) — which predicate pushdown then sinks into its scan and
    zone-map pruning can use to skip whole row groups. The classic star-schema
    accelerant.

    Restricted to inner joins (an outer join's preserved side must keep its
    unmatched rows, so a key constraint does not transfer). The added predicate is a
    superset of what the join already enforces, so the result is unchanged; the
    presence check makes the rule idempotent.
    """
    if node.join_type != "inner":
        return None
    new_left, new_right = node.left, node.right
    changed = False
    for lk, rk in zip(node.left_keys, node.right_keys, strict=True):
        left_cons = _column_constraints(node.left, lk)
        if left_cons:
            new_right, added = _add_inferred(new_right, rk, left_cons, lk)
            changed = changed or added
        right_cons = _column_constraints(node.right, rk)
        if right_cons:
            new_left, added = _add_inferred(new_left, lk, right_cons, rk)
            changed = changed or added
    if not changed:
        return None
    return Join(
        new_left,
        new_right,
        node.left_keys,
        node.right_keys,
        node.join_type,
        node.output,
        node.strategy,
    )


def _column_constraints(side: LogicalPlan, col: str) -> list[Expr]:
    """Constant `col OP literal` constraints provably true for column `col` of `side`'s
    output, found by tracing `col` *down* through the subtree.

    Following the column through filters (collect), through row-preserving operators
    (sort/limit/sample/distinct), through a projection that merely renames it, and into
    the originating side of an **inner** join is what makes inference *transitive*: a
    constraint deep under a chain of joins (`a.k = b.k = c.k AND a.k > 10`) reaches
    every member. Renames are followed by name, never guessed — a projection that
    *computes* `col`, or a non-inner join (whose rows may be null-extended), stops the
    trace, so a found constraint always holds for `side`'s output rows. Constraints are
    rephrased onto `col` so the caller can mirror them across the join's key pair.
    """
    if isinstance(side, Filter):
        # Conjuncts constraining `col` itself; `col` passes a filter unchanged, so
        # constraints below it apply too. (Already phrased on `col` — no remap.)
        here = [c for c in split_conjuncts(side.predicate) if _sole_constrained_column(c) == col]
        return here + _column_constraints(side.input, col)
    if isinstance(side, (Sort, Limit, Sample, Distinct)):
        return _column_constraints(side.input, col)
    if isinstance(side, Project):
        for item in side.items:
            if item.alias == col and isinstance(item.expr, Col):  # pure rename `col ← src`
                src = item.expr.name
                return [remap_columns(c, {src: col}) for c in _column_constraints(side.input, src)]
        return []
    if isinstance(side, Join) and side.join_type == "inner":
        for o in side.output:
            if o.alias == col:  # map the output alias to its source side+name
                child = side.left if o.side == "left" else side.right
                below = _column_constraints(child, o.name)
                return [remap_columns(c, {o.name: col}) for c in below]
        return []
    return []  # Scan / Aggregate / Window / Union / non-inner join: stop the trace


def _sole_constrained_column(conj: Expr) -> str | None:
    """The column name of a `col OP literal` conjunct that references only that one
    column (an inferable constant constraint), else None."""
    if not isinstance(conj, Binary) or conj.op not in _INFERABLE_COMPARISONS:
        return None
    cs = comparison_col_side(conj)
    if cs is not None and referenced_columns(conj) == {cs[0]}:
        return cs[0]
    return None


def _add_inferred(
    target: LogicalPlan, target_key: str, constraints: list[Expr], source_key: str
) -> tuple[LogicalPlan, bool]:
    """Add each `constraints` conjunct, rephrased onto `target_key`, to `target` —
    unless an identical conjunct is already present. Returns `(plan, changed)`."""
    current = split_conjuncts(target.predicate) if isinstance(target, Filter) else []
    existing = [c.to_ir() for c in current]
    fresh = [
        remapped
        for c in constraints
        if (remapped := remap_columns(c, {source_key: target_key})).to_ir() not in existing
    ]
    if not fresh:
        return target, False
    if isinstance(target, Filter):
        combined = combine_conjuncts(split_conjuncts(target.predicate) + fresh)
        return Filter(target.input, combined), True
    return Filter(target, combine_conjuncts(fresh)), True


@rule(name="push_filter_through_aggregate", phase=Phase.PUSHDOWN, matches=(Filter,))
def push_filter_through_aggregate(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Filter(Aggregate(x, keys, aggs), p)` → `Aggregate(Filter(x, p'), keys, aggs)`
    when `p` references only group-key columns.

    `p'` is `p` with each group-key output column replaced by its defining
    expression over `x`. Only safe for predicates that touch group keys alone — a
    predicate on an aggregate output (e.g. `SUM(x) > 10`, a HAVING clause) genuinely
    needs the grouped result and cannot move below the aggregation.
    """
    inner = node.input
    if not isinstance(inner, Aggregate):
        return None
    key_exprs = {k.alias: k.expr for k in inner.group_keys}
    if not referenced_columns(node.predicate) <= set(key_exprs):
        return None
    pushed = substitute_columns(node.predicate, key_exprs)
    return Aggregate(Filter(inner.input, pushed), inner.group_keys, inner.aggregates)


@rule(name="push_filter_through_sort", phase=Phase.PUSHDOWN, matches=(Filter,))
def push_filter_through_sort(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Filter(Sort(x), p)` → `Sort(Filter(x, p))`. Sorting is row-preserving and
    order-only, so filtering commutes with it — and filtering first means fewer
    rows to sort. Sort preserves the schema, so the predicate needs no rewriting.

    Skipped when the sort carries a `limit` (a top-N): there, the sort selects the
    top rows *before* the filter sees them, so filtering first would change which
    rows survive.
    """
    inner = node.input
    if isinstance(inner, Sort) and inner.limit is None:
        return Sort(Filter(inner.input, node.predicate), inner.keys, None)
    return None


@rule(name="push_filter_through_window", phase=Phase.PUSHDOWN, matches=(Filter,))
def push_filter_through_window(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Filter(Window(x, partition=P, …), p)` → `Window(Filter(x, p), …)` when `p`
    references only (simple-column) partition keys.

    A predicate on the partition key keeps or drops *whole* partitions, and a window
    is computed independently per partition — so dropping those partitions before the
    window yields the identical result for every surviving partition (true even with
    `rank_limit`, a per-partition top-k). A predicate touching a window-function
    output (a rank/sum column) genuinely needs the windowed result and cannot move.
    The partition keys are pass-through columns of `x` under the same names, so the
    predicate transfers unchanged. Conjuncts are split so a mixed predicate pushes its
    partition-only part and keeps the rest above.
    """
    win = node.input
    if not isinstance(win, Window):
        return None
    part_cols = {pk.name for pk in win.partition_keys if isinstance(pk, Col)}
    if len(part_cols) != len(win.partition_keys):
        return None  # a non-trivial partition expression — be conservative
    pushable, keep = [], []
    for conj in split_conjuncts(node.predicate):
        (pushable if referenced_columns(conj) <= part_cols else keep).append(conj)
    if not pushable:
        return None
    new_win = Window(
        Filter(win.input, combine_conjuncts(pushable)),
        win.partition_keys,
        win.order_keys,
        win.functions,
        win.rank_limit,
    )
    return new_win if not keep else Filter(new_win, combine_conjuncts(keep))
