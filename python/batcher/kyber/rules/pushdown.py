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
    Distinct,
    Filter,
    Join,
    JoinOutputCol,
    Limit,
    LogicalPlan,
    Project,
    Sample,
    Sort,
    Window,
    is_cartesian_key_pair,
)
from batcher.plan.visitor import transform_up

__all__ = [
    "derive_join_keys",
    "infer_join_predicates",
    "push_filter_through_aggregate",
    "push_filter_through_sort",
    "push_filter_through_window",
    "push_semijoin_through_join",
    "rewrite_predicate",
]

# Comparisons whose `col OP literal` form is a constant constraint worth mirroring
# across an equi-join's key correspondence.
_INFERABLE_COMPARISONS = frozenset({"lt", "le", "gt", "ge", "eq", "ne"})


def rewrite_predicate(plan: LogicalPlan) -> LogicalPlan:
    """Push filters below joins where it is semantics-preserving.

    The structural recursion — visit every node, rebuild only what changed, and
    preserve object identity so the driver detects the fixpoint in O(1) — is the
    shared `transform_up` walk. The only per-node logic is the one case that moves
    work: a `Filter` sitting directly over a `Join`. Because nothing else is
    rewritten, a filter is never sunk below a row-numbering, explode, sample, or
    aggregate-output operator — the conservative, correctness-first behavior — without
    needing a hand-written arm per node type.
    """

    def push(node: LogicalPlan) -> LogicalPlan:
        if isinstance(node, Filter) and isinstance(node.input, Join):
            pushed = _push_into_join(node.predicate, node.input)
            if pushed is not None:
                return pushed  # a conjunct moved below the join
        return node

    return transform_up(plan, push)


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


@rule(name="derive_join_keys", phase=Phase.PUSHDOWN, matches=(Filter,))
def derive_join_keys(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Absorb an equi-conjunct spanning both sides of an inner join into the join keys.

    A `WHERE a.k = b.k` over a comma join (and any cross join) is lowered as a Filter
    sitting above a cartesian join — in this engine, an inner join on a synthetic
    constant `__cross_key`. Predicate pushdown cannot move such a conjunct (it
    references *both* sides), so without this rule the equality is evaluated only
    *after* the full cartesian product is built — a catastrophic blow-up on multi-table
    queries. This rewrite turns each `eq(left_col, right_col)` conjunct into a real
    join key pair, then drops the now-redundant cartesian pseudo-keys (the `__cross_key`
    carries no information once a real key drives the join). The result is the equi-join
    the query always meant, which join reordering can then place to avoid cross products.

    Only inner joins (equi-join semantics; an outer join's null-extended rows make a
    derived key unsafe). Non-equality and single-side conjuncts are left in the filter.
    """
    join = node.input
    if not isinstance(join, Join) or join.join_type != "inner":
        return None

    left_src = {o.alias: o.name for o in join.output if o.side == "left"}
    right_src = {o.alias: o.name for o in join.output if o.side == "right"}
    left_keys = list(join.left_keys)
    right_keys = list(join.right_keys)
    existing = set(zip(left_keys, right_keys, strict=True))

    keep: list[Expr] = []
    derived = False
    for conj in split_conjuncts(node.predicate):
        pair = _equi_key_pair(conj, left_src, right_src)
        if pair is not None and pair not in existing:
            left_keys.append(pair[0])
            right_keys.append(pair[1])
            existing.add(pair)
            derived = True
        else:
            keep.append(conj)
    if not derived:
        return None

    # A real key now drives the join, so the cartesian pseudo-keys (`__cross_key`) are
    # redundant — drop them to avoid a needless constant column in the hash key.
    real = [
        (lk, rk)
        for lk, rk in zip(left_keys, right_keys, strict=True)
        if not is_cartesian_key_pair(join.left, lk, join.right, rk)
    ]
    if real:
        left_keys = [lk for lk, _ in real]
        right_keys = [rk for _, rk in real]

    new_join = Join(
        join.left,
        join.right,
        tuple(left_keys),
        tuple(right_keys),
        "inner",
        join.output,
        join.strategy,
    )
    return new_join if not keep else Filter(new_join, combine_conjuncts(keep))


def _equi_key_pair(
    conj: Expr, left_src: dict[str, str], right_src: dict[str, str]
) -> tuple[str, str] | None:
    """`(left_key, right_key)` if `conj` is `left_col = right_col` across the join, else None.

    Both operands must be bare columns, one resolving to a left output alias and the
    other to a right output alias; the returned names are the *input* (source) column
    names the join keys are phrased in.
    """
    if not isinstance(conj, Binary) or conj.op != "eq":
        return None
    lhs, rhs = conj.left, conj.right
    if not (isinstance(lhs, Col) and isinstance(rhs, Col)):
        return None
    if lhs.name in left_src and rhs.name in right_src:
        return (left_src[lhs.name], right_src[rhs.name])
    if lhs.name in right_src and rhs.name in left_src:
        return (left_src[rhs.name], right_src[lhs.name])
    return None


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


@rule(name="push_semijoin_through_join", phase=Phase.PUSHDOWN, matches=(Join,))
def push_semijoin_through_join(node: Join, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Sink a semi/anti join below an inner join, onto the child its keys come from.

    `SemiJoin(InnerJoin(A, B) ON k, S) ON A.col` == `InnerJoin(SemiJoin(A, S) ON A.col, B) ON k`
    A semi/anti join only *filters* its left input by key membership, and an inner
    join preserves every surviving row's key columns — so filtering a child before the
    join is identical to filtering the join's output (fan-out and all), but it shrinks
    the expensive join's input. This is the classic optimization behind TPC-H Q18,
    whose `o_orderkey IN (SELECT ... HAVING sum > 300)` semijoin sinks below the
    `lineitem ⋈ orders` join so only the handful of qualifying orders reach it instead
    of materializing the full 6M-row join and filtering afterward.

    Restricted to an inner join below (an outer join's null-extended side would change
    key membership), and only when the semijoin's keys all attribute to one child —
    via the inner join's output map, so a renamed key resolves to its source column.
    """
    if node.join_type not in ("semi", "anti"):
        return None
    # Peel column-only ("transparent") projects between the semijoin and an inner
    # join — projects are 1:1 on rows, so the semijoin commutes with them; remap the
    # keys to each project's source columns as we descend, and re-apply the projects
    # on top afterward (their output schema is unchanged by a filtering child).
    projects: list[Project] = []
    cur: LogicalPlan = node.left
    keys: list[str] = list(node.left_keys)
    while isinstance(cur, Project):
        passthrough = {
            item.alias: item.expr.name for item in cur.items if isinstance(item.expr, Col)
        }
        if any(k not in passthrough for k in keys):
            return None  # a key is a computed column here — cannot attribute it
        keys = [passthrough[k] for k in keys]
        projects.append(cur)
        cur = cur.input
    if not isinstance(cur, Join) or cur.join_type != "inner":
        return None
    inner = cur
    # Resolve each semijoin key (an alias in the inner join's output) to the child
    # side + source column it came from; all must land on the same side.
    out_by_alias = {o.alias: o for o in inner.output}
    sides: set[str] = set()
    src_keys: list[str] = []
    for k in keys:
        col = out_by_alias.get(k)
        if col is None:
            return None  # key is not a pass-through column — cannot attribute it
        sides.add(col.side)
        src_keys.append(col.name)
    if len(sides) != 1:
        return None  # keys span both children — cannot push to one side
    on_left = sides.pop() == "left"
    target = inner.left if on_left else inner.right
    pushed = _semijoin_onto(target, node, tuple(src_keys))
    result: LogicalPlan = Join(
        pushed if on_left else inner.left,
        inner.right if on_left else pushed,
        inner.left_keys,
        inner.right_keys,
        inner.join_type,
        inner.output,
        inner.strategy,
    )
    for project in reversed(projects):
        result = Project(result, project.items)
    return result


def _semijoin_onto(target: LogicalPlan, semi: Join, left_keys: tuple[str, ...]) -> Join:
    """Rebuild `semi` (a semi/anti join) filtering `target` on `target`'s own key names.

    The output is `target`'s columns unchanged (a semi/anti join adds none), so the
    inner join above still sees the same schema after the push.
    """
    output = tuple(JoinOutputCol("left", c, c) for c in target.available_columns())
    return Join(
        target, semi.right, left_keys, semi.right_keys, semi.join_type, output, semi.strategy
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
