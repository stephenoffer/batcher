"""Extra aggregate / GROUP BY rewrites — small, local, always-correct simplifications.

Each rule here is a node-local `@rule` over `Aggregate`: it returns a rewritten node
or `None`. The driver supplies bottom-up traversal, fixpoint iteration, and pattern
indexing on the declared `matches`. Like `algebraic.py`, every rule is unconditionally
semantics-preserving — it depends on no cardinality or cost estimate, only on the plan
shape — so it can only remove redundant work, never change a result.

The correctness reasoning threaded through these rules is SQL aggregate semantics:

* A **global** aggregate (no group keys) over 0 rows returns **1** row (COUNT = 0,
  every other aggregate NULL). A **grouped** aggregate over 0 rows returns **0** rows,
  and every *emitted* group is non-empty. Several folds below are therefore restricted
  to the *grouped* case, where "the group has ≥ 1 row" is guaranteed.
* SUM / AVG / MIN / MAX / COUNT(x) skip NULLs; COUNT(*) counts rows.
* Within one group a *group-key expression* is constant — that is what the grouping
  established — which is what makes the "aggregate of a group key" folds exact,
  including the all-NULL group (its key value is NULL and the aggregate is NULL/0 too).

Rules that move a computed output out of the aggregate wrap the aggregate in a
`Project` that re-derives exactly the original output columns, in order — so the
output schema is byte-identical and only the plan shape changes.
"""

from __future__ import annotations

import json

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.expr_ir import AggExpr, Col, Expr, Lit
from batcher.plan.expr_ir.constructors import when
from batcher.plan.logical import (
    Aggregate,
    AggregateSpec,
    Distinct,
    LogicalPlan,
    Project,
    Projection,
)

__all__ = [
    "aggregate_without_aggs_to_distinct",
    "count_constant_to_count_star",
    "count_distinct_of_group_key",
    "count_of_group_key",
    "dedupe_group_keys",
    "deduplicate_aggregate_exprs",
    "drop_constant_group_key",
    "drop_distinct_before_agg",
    "fold_constant_grouped_aggregate",
    "redundant_aggregate_of_group_key",
    "sum_constant_to_count",
]


# --- shared helpers ----------------------------------------------------------


def _ir_key(expr: Expr) -> str:
    """A hashable structural identity for an expression (its IR rendered stable)."""
    return json.dumps(expr.to_ir(), sort_keys=True)


def _agg_key(agg: AggExpr) -> str:
    """A structural identity for an aggregate, ignoring its output alias."""
    return json.dumps(
        {
            "func": agg.func,
            "input": agg.input.to_ir() if agg.input is not None else None,
            "input2": agg.input2.to_ir() if agg.input2 is not None else None,
            "param": agg.param,
        },
        sort_keys=True,
    )


def _group_key_map(node: Aggregate) -> dict[str, str]:
    """Map each group-key expression's structural identity to its output alias."""
    out: dict[str, str] = {}
    for key in node.group_keys:
        out.setdefault(_ir_key(key.expr), key.alias)
    return out


def _used_aliases(node: Aggregate) -> set[str]:
    """Every output alias the aggregate already binds (group keys + aggregates)."""
    return {k.alias for k in node.group_keys} | {s.alias for s in node.aggregates}


def _fresh_alias(used: set[str]) -> str:
    """A synthetic alias for an internal helper column not clashing with `used`."""
    base = "__agg_cnt_star"
    if base not in used:
        return base
    i = 0
    while f"{base}_{i}" in used:
        i += 1
    return f"{base}_{i}"


def _count_star_alias(node: Aggregate) -> str | None:
    """The alias of an existing ``COUNT(*)`` output, or None if there is none."""
    for spec in node.aggregates:
        if spec.agg.func == "count_star" and spec.agg.input is None:
            return spec.alias
    return None


def _project_over(
    node: Aggregate,
    inner: Aggregate,
    *,
    key_map: dict[str, Expr] | None = None,
    agg_map: dict[str, Expr] | None = None,
) -> Project:
    """Wrap `inner` in a `Project` reproducing `node`'s output columns, in order.

    `key_map` / `agg_map` override how a group-key / aggregate output alias is
    re-derived (default: a pass-through `Col(alias)` read from `inner`).
    """
    key_map = key_map or {}
    agg_map = agg_map or {}
    items = [Projection(k.alias, key_map.get(k.alias, Col(k.alias))) for k in node.group_keys]
    items += [Projection(s.alias, agg_map.get(s.alias, Col(s.alias))) for s in node.aggregates]
    return Project(inner, tuple(items))


# --- group-key rules ---------------------------------------------------------


@rule(name="dedupe_group_keys", phase=Phase.REWRITE, matches=(Aggregate,))
def dedupe_group_keys(node: Aggregate, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop a group key that repeats an earlier key's expression, re-deriving it above.

    Grouping by ``(a = e, b = e)`` forms exactly the same groups as grouping by
    ``(a = e)`` — the second copy adds no grouping — and each output ``b`` equals
    ``a`` row for row, so it is re-derived by a projection ``b = col("a")``. Correct
    for any values including NULL (grouping treats all-NULL keys as one group), and for
    the empty input (still a grouped aggregate → 0 rows either way). Fires only when a
    later key is structurally identical to an earlier one.
    """
    seen: dict[str, str] = {}
    kept: list[Projection] = []
    dropped: dict[str, str] = {}
    for key in node.group_keys:
        ident = _ir_key(key.expr)
        rep = seen.get(ident)
        if rep is not None and rep != key.alias:
            dropped[key.alias] = rep
        else:
            seen.setdefault(ident, key.alias)
            kept.append(key)
    if not dropped:
        return None
    inner = Aggregate(node.input, tuple(kept), node.aggregates)
    return _project_over(node, inner, key_map={a: Col(r) for a, r in dropped.items()})


@rule(name="drop_constant_group_key", phase=Phase.REWRITE, matches=(Aggregate,))
def drop_constant_group_key(node: Aggregate, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop a literal-constant group key, re-deriving it as a projected constant.

    A group key that is a (non-null) literal is the same value in every row, so it
    never splits a group — dropping it leaves the grouping unchanged, and the constant
    column is re-added by a projection. Restricted to the case where at least one
    *non-constant* key remains: dropping the sole key would turn a grouped aggregate
    (0 rows over empty input) into a global one (1 row over empty input), changing the
    empty-input result.
    """
    const: dict[str, Expr] = {
        k.alias: k.expr
        for k in node.group_keys
        if isinstance(k.expr, Lit) and k.expr.value is not None
    }
    if not const:
        return None
    kept = [k for k in node.group_keys if k.alias not in const]
    if not kept:
        return None  # would become a global aggregate — empty-input semantics differ
    inner = Aggregate(node.input, tuple(kept), node.aggregates)
    return _project_over(node, inner, key_map=dict(const))


# --- aggregate-of-a-group-key rules -----------------------------------------

# Aggregates that return the group key's own value when applied to a group-key
# expression (constant within the group): the extremes and boolean reductions.
_KEY_VALUE_AGGS = frozenset({"min", "max", "mode", "bool_and", "bool_or"})


@rule(name="redundant_aggregate_of_group_key", phase=Phase.REWRITE, matches=(Aggregate,))
def redundant_aggregate_of_group_key(node: Aggregate, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Replace ``min/max/mode/bool_and/bool_or`` of a group-key expression by the key.

    A group-key expression is constant within its group, so its min, max, mode, and
    boolean AND/OR all equal that constant — the group key's own output value. The
    aggregate is dropped and the column is re-derived as a projection of the key. The
    all-NULL group is handled too: its key value is NULL and each of these aggregates
    over all-NULL is NULL as well. Type-preserving (all return the input type).
    """
    if not node.group_keys:
        return None
    keymap = _group_key_map(node)
    kept: list[AggregateSpec] = []
    folded: dict[str, Expr] = {}
    for spec in node.aggregates:
        agg = spec.agg
        if (
            agg.func in _KEY_VALUE_AGGS
            and agg.input is not None
            and agg.input2 is None
            and agg.param is None
            and _ir_key(agg.input) in keymap
        ):
            folded[spec.alias] = Col(keymap[_ir_key(agg.input)])
        else:
            kept.append(spec)
    if not folded:
        return None
    inner = Aggregate(node.input, node.group_keys, tuple(kept))
    return _project_over(node, inner, agg_map=folded)


@rule(name="count_distinct_of_group_key", phase=Phase.REWRITE, matches=(Aggregate,))
def count_distinct_of_group_key(node: Aggregate, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Replace ``COUNT(DISTINCT group_key)`` by a cheap NULL check on the key.

    A group-key expression is constant within its group, so it has exactly one distinct
    value there — and ``COUNT(DISTINCT)`` skips NULLs, so the count is 1 for a non-null
    group and 0 for the all-NULL group. That is ``CASE WHEN key IS NULL THEN 0 ELSE 1``,
    which removes an expensive distinct aggregation. Exact ``count_distinct`` only (the
    approximate sketch is left untouched); grouped only (a group key implies grouping).
    """
    if not node.group_keys:
        return None
    keymap = _group_key_map(node)
    kept: list[AggregateSpec] = []
    folded: dict[str, Expr] = {}
    for spec in node.aggregates:
        agg = spec.agg
        if (
            agg.func == "count_distinct"
            and agg.input is not None
            and agg.input2 is None
            and _ir_key(agg.input) in keymap
        ):
            key_alias = keymap[_ir_key(agg.input)]
            folded[spec.alias] = when(Col(key_alias).is_null()).then(0).otherwise(1)
        else:
            kept.append(spec)
    if not folded:
        return None
    inner = Aggregate(node.input, node.group_keys, tuple(kept))
    return _project_over(node, inner, agg_map=folded)


@rule(name="count_of_group_key", phase=Phase.REWRITE, matches=(Aggregate,))
def count_of_group_key(node: Aggregate, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Replace ``COUNT(group_key)`` by ``CASE WHEN key IS NULL THEN 0 ELSE COUNT(*)``.

    ``COUNT(x)`` counts non-null values; a group-key expression is constant within its
    group, so it is either non-null in every row (count = the group's row count =
    ``COUNT(*)``) or NULL in every row (the all-NULL group, count = 0). ``COUNT(*)`` is
    reused from an existing output or added as an internal column. Grouped only.
    """
    if not node.group_keys:
        return None
    keymap = _group_key_map(node)
    targets = [
        spec
        for spec in node.aggregates
        if spec.agg.func == "count"
        and spec.agg.input is not None
        and spec.agg.input2 is None
        and _ir_key(spec.agg.input) in keymap
    ]
    if not targets:
        return None
    cnt_alias = _count_star_alias(node)
    extra: tuple[AggregateSpec, ...] = ()
    if cnt_alias is None:
        cnt_alias = _fresh_alias(_used_aliases(node))
        extra = (AggregateSpec(cnt_alias, AggExpr("count_star", None)),)
    target_aliases = {s.alias for s in targets}
    kept = [s for s in node.aggregates if s.alias not in target_aliases]
    inner = Aggregate(node.input, node.group_keys, tuple(kept) + extra)
    folded = {
        s.alias: when(Col(keymap[_ir_key(s.agg.input)]).is_null()).then(0).otherwise(Col(cnt_alias))
        for s in targets
    }
    return _project_over(node, inner, agg_map=folded)


# --- aggregate-of-a-constant rules ------------------------------------------


def _nonnull_lit(agg: AggExpr) -> Lit | None:
    """The aggregate's input if it is a non-null literal with no second input/param."""
    if (
        isinstance(agg.input, Lit)
        and agg.input.value is not None
        and agg.input2 is None
        and agg.param is None
    ):
        return agg.input
    return None


@rule(name="count_constant_to_count_star", phase=Phase.REWRITE, matches=(Aggregate,))
def count_constant_to_count_star(node: Aggregate, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Rewrite ``COUNT(<non-null literal>)`` to ``COUNT(*)``.

    A non-null constant is non-null in every row, so counting it counts the rows —
    exactly ``COUNT(*)``. Holds for both global and grouped aggregates (both count 0
    over empty input) and needs no projection, since one aggregate simply replaces
    another. A null literal is excluded (``COUNT(NULL)`` is 0, not the row count).
    """
    new: list[AggregateSpec] = []
    changed = False
    for spec in node.aggregates:
        if spec.agg.func == "count" and _nonnull_lit(spec.agg) is not None:
            new.append(AggregateSpec(spec.alias, AggExpr("count_star", None)))
            changed = True
        else:
            new.append(spec)
    if not changed:
        return None
    return Aggregate(node.input, node.group_keys, tuple(new))


@rule(name="sum_constant_to_count", phase=Phase.REWRITE, matches=(Aggregate,))
def sum_constant_to_count(node: Aggregate, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Rewrite grouped ``SUM(<integer literal c>)`` to ``c * COUNT(*)``.

    Summing a constant ``c`` over a group of ``n`` rows is ``c * n``; ``COUNT(*)`` is
    ``n``. Integer literals only — a float sum accumulates ``c`` repeatedly and would
    not be bit-identical to a single ``c * n`` multiply (rounding), while integer
    ``+``/``*`` agree under the engine's wrapping arithmetic. Grouped only: over empty
    input a grouped SUM emits no row, but a *global* ``SUM`` is NULL while ``c * 0`` is
    0. ``COUNT(*)`` is reused or added internally.
    """
    if not node.group_keys:
        return None
    targets = [
        spec
        for spec in node.aggregates
        if spec.agg.func == "sum"
        and _nonnull_lit(spec.agg) is not None
        and isinstance(spec.agg.input.value, int)  # type: ignore[union-attr]
        and not isinstance(spec.agg.input.value, bool)  # type: ignore[union-attr]
    ]
    if not targets:
        return None
    cnt_alias = _count_star_alias(node)
    extra: tuple[AggregateSpec, ...] = ()
    if cnt_alias is None:
        cnt_alias = _fresh_alias(_used_aliases(node))
        extra = (AggregateSpec(cnt_alias, AggExpr("count_star", None)),)
    target_aliases = {s.alias for s in targets}
    kept = [s for s in node.aggregates if s.alias not in target_aliases]
    inner = Aggregate(node.input, node.group_keys, tuple(kept) + extra)
    folded = {s.alias: Lit(s.agg.input.value) * Col(cnt_alias) for s in targets}  # type: ignore[union-attr]
    return _project_over(node, inner, agg_map=folded)


@rule(name="fold_constant_grouped_aggregate", phase=Phase.REWRITE, matches=(Aggregate,))
def fold_constant_grouped_aggregate(node: Aggregate, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Fold ``min/max/mode/bool_and/bool_or/count_distinct`` of a non-null constant.

    Over a non-empty group a constant ``c`` has extremes/mode/boolean-reduction all
    equal to ``c``, and exactly one distinct value (``COUNT(DISTINCT c) = 1``). Every
    emitted group of a *grouped* aggregate is non-empty, so these fold to projected
    constants. Grouped only: a *global* aggregate over empty input must yield NULL / 0,
    not the constant. The folded column carries the same type the aggregate would.
    """
    if not node.group_keys:
        return None
    kept: list[AggregateSpec] = []
    folded: dict[str, Expr] = {}
    for spec in node.aggregates:
        lit = _nonnull_lit(spec.agg)
        if lit is None:
            kept.append(spec)
        elif spec.agg.func in _KEY_VALUE_AGGS:
            folded[spec.alias] = Lit(lit.value)
        elif spec.agg.func == "count_distinct":
            folded[spec.alias] = Lit(1)
        else:
            kept.append(spec)
    if not folded:
        return None
    inner = Aggregate(node.input, node.group_keys, tuple(kept))
    return _project_over(node, inner, agg_map=folded)


# --- Distinct / group-only rules --------------------------------------------

# Aggregates whose result is unchanged when their input carries duplicate rows: the
# extremes, boolean reductions, and distinct counts. SUM/COUNT(*)/COUNT/AVG are NOT
# here (they depend on multiplicity), nor is mode (dedup changes frequencies), nor
# arg_min/arg_max (dedup can change the tie-broken row).
_DUP_INSENSITIVE = frozenset(
    {"min", "max", "bool_and", "bool_or", "count_distinct", "approx_count_distinct"}
)


@rule(name="drop_distinct_before_agg", phase=Phase.REWRITE, matches=(Aggregate,))
def drop_distinct_before_agg(node: Aggregate, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop a `Distinct` feeding an aggregate whose functions are duplicate-insensitive.

    A whole-row `Distinct` only removes rows identical in every column. The set of
    values reaching each group is therefore unchanged (a removed row had an identical
    twin), so min/max, boolean AND/OR and the distinct counts are identical with or
    without the dedup — while the dedup itself is a wasted pipeline breaker. Fires only
    when every aggregate is duplicate-insensitive and unary (a group-only aggregate,
    which already dedups, qualifies vacuously).
    """
    inner = node.input
    # `not inner.keys` is what makes the argument above hold: a removed row had an identical
    # twin only under a WHOLE-ROW dedup. A keyed dedup removes rows that differ outside its
    # key, so the values reaching each group genuinely change and dropping it is a wrong
    # answer rather than a faster plan.
    if not isinstance(inner, Distinct) or inner.keys:
        return None
    if any(
        spec.agg.func not in _DUP_INSENSITIVE or spec.agg.input2 is not None
        for spec in node.aggregates
    ):
        return None
    return Aggregate(inner.input, node.group_keys, node.aggregates)


@rule(name="aggregate_without_aggs_to_distinct", phase=Phase.REWRITE, matches=(Aggregate,))
def aggregate_without_aggs_to_distinct(
    node: Aggregate, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """Rewrite a group-only aggregate (no aggregate functions) to a `Distinct`.

    Grouping by a set of key expressions with no aggregates emits one row per distinct
    key tuple — exactly ``Distinct`` of those projected keys — and reuses the simpler
    dedup kernel. Identical for NULLs (grouped as one) and empty input (0 rows either
    way). Requires at least one group key, so a keyless no-aggregate aggregate (the
    1-row global corner case) is left untouched.
    """
    if node.aggregates or not node.group_keys:
        return None
    return Distinct(Project(node.input, node.group_keys))


# --- CSE across aggregates ---------------------------------------------------


@rule(name="deduplicate_aggregate_exprs", phase=Phase.REWRITE, matches=(Aggregate,))
def deduplicate_aggregate_exprs(node: Aggregate, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Compute an aggregate that appears under two aliases once, aliasing the copies.

    Two aggregate outputs with the same function, inputs and parameter produce
    byte-identical columns, so the duplicate is computed once and the extra aliases are
    re-derived as projections of the representative. No dependence on values, so it is
    always correct (including the empty-input single global row). Fires only when at
    least one aggregate is a structural duplicate of an earlier one.
    """
    seen: dict[str, str] = {}
    kept: list[AggregateSpec] = []
    dup: dict[str, str] = {}
    for spec in node.aggregates:
        ident = _agg_key(spec.agg)
        rep = seen.get(ident)
        if rep is not None:
            dup[spec.alias] = rep
        else:
            seen[ident] = spec.alias
            kept.append(spec)
    if not dup:
        return None
    inner = Aggregate(node.input, node.group_keys, tuple(kept))
    return _project_over(node, inner, agg_map={a: Col(r) for a, r in dup.items()})
