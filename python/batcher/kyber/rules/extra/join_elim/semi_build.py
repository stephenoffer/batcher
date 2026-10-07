"""Rewrites a semi join's build side admits because it is read as a set of keys.

A semi join asks one question of its right side: does *some* row carry this key? So the
right side's multiplicity and every column but its join key are invisible in the result,
and two shapes that are expensive as written become cheap under that reading:

* an **inner join** in the build side whose projection reads only one of its sides is an
  existence test on the other side -- a semi join (`semi_build_inner_to_semi`). The inner
  join fans each row out once per match; the semi join keeps each row at most once. Under
  set semantics the two are the same answer. This is `rules.joins.join_to_semijoin`'s
  argument, with the semi join playing the `DISTINCT`.
* a **self-join on a key with a `<>` on another column**, read only for its key, is the set
  of keys holding two different values of that column (`semi_build_self_neq_to_minmax`).
  TPC-DS q95's `ws_wh` is `web_sales ws1 JOIN web_sales ws2 ON order WHERE ws1.wh <>
  ws2.wh`, read only through `IN (SELECT order FROM ws_wh)`: at sf10 the join emits ~118M
  rows to answer a question a `GROUP BY order` over 7.2M rows answers with `min(wh) <>
  max(wh)`. `_sql.parser.subquery.neq` makes the same reduction for a correlated `EXISTS`;
  this is the uncorrelated, CTE-shaped form of it, which reaches the planner as a join.

Both are sound only where the build side really is read as a set, so both fire only on the
right side of a `semi` join, never under an inner join or a bare `DISTINCT` -- an `anti`
join reads its build side as a set too, but a `NOT IN` over a NULL key does not, and nothing
here needs it.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.extra.join_elim.evidence import _same_relation
from batcher.plan.expr_ir import AggExpr, Binary, Col, IsNotNull
from batcher.plan.logical import (
    Aggregate,
    AggregateSpec,
    Filter,
    Join,
    JoinOutputCol,
    LogicalPlan,
    Project,
    Projection,
)

__all__ = ["semi_build_inner_to_semi", "semi_build_self_neq_to_minmax"]


def _renames(plan: LogicalPlan) -> tuple[LogicalPlan, dict[str, str]]:
    """`plan` with its column-renaming `Project`s peeled, and each output name's source name.

    Only a `Project` whose every item is a bare column reference is peeled; a computed item
    ends the descent, because a value derived from a column is not that column. The mapping
    covers the names `plan` outputs.
    """
    mapping: dict[str, str] | None = None
    node = plan
    while isinstance(node, Project) and all(isinstance(p.expr, Col) for p in node.items):
        step = {p.alias: p.expr.name for p in node.items}
        mapping = step if mapping is None else {a: step[s] for a, s in mapping.items() if s in step}
        node = node.input
    if mapping is None:
        mapping = {c: c for c in plan.available_columns()}
    return node, mapping


def _join_source(join: Join, alias: str) -> JoinOutputCol | None:
    """The join output column named `alias`, or None."""
    return next((o for o in join.output if o.alias == alias), None)


@rule(name="semi_build_inner_to_semi", phase=Phase.PUSHDOWN, matches=(Join,))
def semi_build_inner_to_semi(node: Join, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Semi(L, Project_A(Inner(A, B)))` → `Semi(L, Project_A(Semi(A, B)))`.

    The projection under the semi join reads only one side of the inner join (either side;
    a right-only read swaps the operands), so the other side only tests existence. Each
    projected row of the inner join is a row of that side with at least one match, repeated
    once per match; the semi join keeps the same rows once each, and the outer semi join
    cannot tell the difference. Returns None for any other shape.
    """
    if node.join_type != "semi" or not isinstance(node.right, Project):
        return None
    proj = node.right
    inner = proj.input
    if not isinstance(inner, Join) or inner.join_type != "inner":
        return None
    if not all(isinstance(p.expr, Col) for p in proj.items):
        return None
    sides = set()
    for item in proj.items:
        src = _join_source(inner, item.expr.name)
        if src is None:
            return None
        sides.add(src.side)
    if len(sides) != 1:
        return None
    kept = sides.pop()
    output = tuple(JoinOutputCol("left", o.name, o.alias) for o in inner.output if o.side == kept)
    if kept == "left":
        semi = Join(inner.left, inner.right, inner.left_keys, inner.right_keys, "semi", output)
    else:
        semi = Join(inner.right, inner.left, inner.right_keys, inner.left_keys, "semi", output)
    return Join(
        node.left,
        Project(semi, proj.items),
        node.left_keys,
        node.right_keys,
        "semi",
        node.output,
        node.strategy,
    )


def _neq_self_join(plan: LogicalPlan, key_alias: str, ctx: OptimizerContext):
    """Match `[Project]* Filter(a <> b, Inner(X, X') on k = k')` read only for its key.

    Returns `(base, key, value)` -- the relation both sides read, and the names in it of
    the join key and the compared column -- when the two sides are the same relation
    (`evidence._same_relation`, after peeling renames), the join is on one key that is the
    same column on both sides, `a` and `b` are that relation's same non-float column taken
    one from each side, and `key_alias` resolves to the join key. None otherwise.
    """
    node, names = _renames(plan)
    if key_alias not in names or not isinstance(node, Filter):
        return None
    pred, join = node.predicate, node.input
    if not (isinstance(pred, Binary) and pred.op == "ne"):
        return None
    if not (isinstance(pred.left, Col) and isinstance(pred.right, Col)):
        return None
    if not isinstance(join, Join) or join.join_type != "inner" or len(join.left_keys) != 1:
        return None
    a, b = _join_source(join, pred.left.name), _join_source(join, pred.right.name)
    k = _join_source(join, names[key_alias])
    if a is None or b is None or k is None or {a.side, b.side} != {"left", "right"}:
        return None
    lbase, lnames = _renames(join.left)
    rbase, rnames = _renames(join.right)
    if not _same_relation(lbase, rbase, ctx):
        return None
    lval, rval = (a, b) if a.side == "left" else (b, a)
    lkey, rkey = join.left_keys[0], join.right_keys[0]
    key = lnames.get(lkey)
    value = lnames.get(lval.name)
    if key is None or value is None or rnames.get(rkey) != key or rnames.get(rval.name) != value:
        return None
    # The join side the key alias was read from must name the join key itself.
    side_key = lkey if k.side == "left" else rkey
    if k.name != side_key:
        return None
    schema = lbase.available_schema()
    if schema is None or not schema.has(value) or pa.types.is_floating(schema.field(value).type):
        # A float's `<>` and the engine's canonicalized min/max disagree on NaN and -0.0.
        return None
    return lbase, key, value


@rule(name="semi_build_self_neq_to_minmax", phase=Phase.PUSHDOWN, matches=(Join,))
def semi_build_self_neq_to_minmax(node: Join, ctx: OptimizerContext) -> LogicalPlan | None:
    """`Semi(L, keys of Filter(x <> x', X JOIN X ON k = k))` → a `GROUP BY k` with min <> max.

    Read for its key alone, the self-join is the set of `k` for which two rows of `X` share
    `k` and differ in `x`. `x <> x'` is true exactly when both are non-null and unequal, so
    that set is the keys whose non-null `x` values are not all one value: `min(x) <> max(x)`,
    both of which skip nulls (a group with no non-null `x` has a null min and is dropped, as
    the join drops it). A null `k` never joins, so its group is excluded explicitly. Floats
    are refused (see `_neq_self_join`). Needs the sides proven to be the same relation, the
    same proof the self-join eliminations use.
    """
    if ctx is None or node.join_type != "semi" or len(node.right_keys) != 1:
        return None
    found = _neq_self_join(node.right, node.right_keys[0], ctx)
    if found is None:
        return None
    base, key, value = found
    k, lo, hi = "__semi_key", "__semi_min", "__semi_max"
    grouped = Aggregate(
        Project(base, (Projection(k, Col(key)), Projection("__semi_val", Col(value)))),
        (Projection(k, Col(k)),),
        (
            AggregateSpec(lo, AggExpr("min", Col("__semi_val"))),
            AggregateSpec(hi, AggExpr("max", Col("__semi_val"))),
        ),
    )
    differs = Filter(grouped, Binary("and", Binary("ne", Col(lo), Col(hi)), IsNotNull(Col(k))))
    build = Project(differs, (Projection(node.right_keys[0], Col(k)),))
    return Join(
        node.left, build, node.left_keys, node.right_keys, "semi", node.output, node.strategy
    )
