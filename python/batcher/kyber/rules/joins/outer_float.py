"""Float a left/semi/anti join above the inner joins that only read its preserved side.

`(A LEFT JOIN B) JOIN C ON a = c` computes the outer join over every row of `A` and only then
lets `C` discard most of them. When the inner join's keys reference `A` alone, the same
relation is `(A JOIN C ON a = c) LEFT JOIN B`: each output row pairs one `a` with its `c`s and
with its `b`s (or a null-extended `b` when none matches), and which `b`s match an `a` does not
depend on `C`. The outer join then runs over the rows `C` kept rather than all of `A`.

The shape is the TPC-DS channel query: a fact table left-joined to its returns, then joined to
a date range, an item filter and a promotion filter. TPC-DS q80 left-joins all 2,880,404
`store_sales` rows to 287,867 `store_returns` rows before a 30-day `date_dim` join keeps ~1.7%
of them, and the same subtree appears once per channel.

The second effect is the larger one. `order.reorder_joins` treats a non-inner join as an opaque
leaf, so while the outer join sits beneath them the dimension joins can only be ordered
*around* it. Floated above them, the fact table and its dimensions form one inner-join graph the
reorderer can order as a whole.

`semi` and `anti` float by the same argument: whether an `a` has a match in `B` is a property of
`a` alone, so filtering by it before or after pairing `a` with `C` keeps the same pairs.

Refused, each for a reason:

- **A key on the null-supplying side.** A predicate on `b` sees NULLs for unmatched rows
  above the outer join and never sees them below it, so moving it changes the answer.
- **A computed projection between the joins.** Only a bare column rename composes through the
  rewrite; an expression over a null-extended column is exactly the case above.
- **An inner join that is estimated to multiply `A`.** Floating is a win when `C` reduces or
  preserves `A`; a fanning-out join would hand the outer join *more* rows than it had.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase, RuleCategory
from batcher.plan.expr_ir import Col
from batcher.plan.logical import Join, JoinOutputCol, LogicalPlan, Project, Projection

__all__ = ["float_outer_join_above_inner"]

# The join types whose left input is preserved row-for-row (or filtered by membership alone).
_FLOATABLE = frozenset({"left", "semi", "anti"})

# How much the inner join may grow `A`, by estimate, and still be worth floating past. A little
# over 1 so a key-preserving join to a dimension estimated at exactly |A| is not refused on noise.
_MAX_FANOUT = 1.05


@rule(
    name="float_outer_join_above_inner",
    phase=Phase.PUSHDOWN,
    matches=(Join,),
    category=RuleCategory.REWRITE,
)
def float_outer_join_above_inner(node: Join, ctx: OptimizerContext) -> LogicalPlan | None:
    """`Inner(X(A ⟕ B), C)` → `Inner(A, C) ⟕ B` when the inner keys read only `A`.

    `X` is an optional projection that only selects and renames columns. The inner join may
    hold the outer join on either side. Returns None when the shape does not match or any of
    the refusals in the module note applies.
    """
    if node.join_type != "inner":
        return None
    for outer_side in ("left", "right"):
        rewritten = _float(node, outer_side, ctx)
        if rewritten is not None:
            return rewritten
    return None


def _float(node: Join, outer_side: str, ctx: OptimizerContext) -> LogicalPlan | None:
    child = node.left if outer_side == "left" else node.right
    other = node.right if outer_side == "left" else node.left
    child_keys = node.left_keys if outer_side == "left" else node.right_keys
    other_keys = node.right_keys if outer_side == "left" else node.left_keys
    other_side = "right" if outer_side == "left" else "left"

    outer, origin = _outer_join_origin(child)
    if outer is None or origin is None:
        return None
    # Every inner key on this side must name a column of the preserved input `A`.
    a_keys: list[str] = []
    for k in child_keys:
        side_col = origin.get(k)
        if side_col is None or side_col[0] != "left":
            return None
        a_keys.append(side_col[1])
    # Every output column read from this side must trace to `A` or `B` through `origin`.
    for o in node.output:
        if o.side == outer_side and o.name not in origin:
            return None

    a = outer.left
    a_needed = list(
        dict.fromkeys([*outer.left_keys, *a_keys, *_a_columns(node, outer_side, origin)])
    )
    c_needed = list(dict.fromkeys(o.name for o in node.output if o.side == other_side))
    a_alias, c_alias = _disjoint_aliases(a_needed, c_needed)

    inner = Join(
        a,
        other,
        tuple(a_keys),
        tuple(other_keys),
        "inner",
        tuple(JoinOutputCol("left", c, a_alias[c]) for c in a_needed)
        + tuple(JoinOutputCol("right", c, c_alias[c]) for c in c_needed),
        node.strategy,
    )
    if not _does_not_fan_out(inner, a, ctx):
        return None

    # The new outer join renames nothing: each column keeps its name from the input it comes
    # from, and a bare-column projection above restores the aliases the inner join exposed. A
    # renaming semi/anti join's output did not survive the rules downstream of this one —
    # `examples/graph/analytics.py` lost its `node AS a` that way — while a projection of bare
    # columns is a shape every rule already handles.
    sources: list[tuple[str, str]] = []
    for o in node.output:
        if o.side == other_side:
            sources.append(("left", c_alias[o.name]))
            continue
        side, name = origin[o.name]
        sources.append(("left", a_alias[name]) if side == "left" else ("right", name))
    exposed: dict[tuple[str, str], str] = {}
    taken = set(inner.available_columns())
    for side, name in dict.fromkeys(sources):
        alias = name
        if side == "right":
            n = 0
            while alias in taken:
                n += 1
                alias = f"{name}__fj{n}"
            taken.add(alias)
        exposed[(side, name)] = alias
    floated = Join(
        inner,
        outer.right,
        tuple(a_alias[k] for k in outer.left_keys),
        outer.right_keys,
        outer.join_type,
        tuple(JoinOutputCol(side, name, alias) for (side, name), alias in exposed.items()),
        outer.strategy,
    )
    items = tuple(
        Projection(o.alias, Col(exposed[src])) for o, src in zip(node.output, sources, strict=True)
    )
    if [i.alias for i in items] == floated.available_columns() and all(
        i.alias == i.expr.name for i in items
    ):
        return floated
    return Project(floated, items)


def _outer_join_origin(
    child: LogicalPlan,
) -> tuple[Join | None, dict[str, tuple[str, str]] | None]:
    """The floatable join under `child` and, for each of `child`'s columns, `(side, column)`.

    `child` is the join itself or a projection of only bare columns over it. Anything else
    returns `(None, None)`.
    """
    project: Project | None = None
    if isinstance(child, Project):
        project, child = child, child.input
    if not isinstance(child, Join) or child.join_type not in _FLOATABLE:
        return None, None
    by_alias = {o.alias: (o.side, o.name) for o in child.output}
    if project is None:
        return child, by_alias
    origin: dict[str, tuple[str, str]] = {}
    for item in project.items:
        if not isinstance(item.expr, Col) or item.expr.name not in by_alias:
            return None, None
        origin[item.alias] = by_alias[item.expr.name]
    return child, origin


def _a_columns(node: Join, outer_side: str, origin: dict[str, tuple[str, str]]) -> list[str]:
    """The preserved input's columns that the inner join's output reads."""
    return [
        origin[o.name][1]
        for o in node.output
        if o.side == outer_side and origin[o.name][0] == "left"
    ]


def _disjoint_aliases(
    a_cols: list[str], c_cols: list[str]
) -> tuple[dict[str, str], dict[str, str]]:
    """Output names for `A`'s and `C`'s columns in the new inner join that cannot collide."""
    taken = set(a_cols)
    a_alias = {c: c for c in a_cols}
    c_alias: dict[str, str] = {}
    for c in c_cols:
        alias, n = c, 0
        while alias in taken:
            n += 1
            alias = f"{c}__fj{n}"
        taken.add(alias)
        c_alias[c] = alias
    return a_alias, c_alias


def _does_not_fan_out(inner: Join, a: LogicalPlan, ctx: OptimizerContext) -> bool:
    """Whether the inner join is estimated to keep at most (about) the rows of `A`."""
    a_rows = ctx.estimator.estimate(a).rows
    if a_rows <= 0:
        return False
    return ctx.estimator.estimate(inner).rows <= a_rows * _MAX_FANOUT
