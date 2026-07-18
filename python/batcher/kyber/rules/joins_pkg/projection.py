"""Push a derived projection through a join onto the side it reads, so the join
carries the narrow computed column instead of its wide inputs.

`Project(revenue = l_extendedprice * (1 - l_discount))` over `lineitem ⋈ orders` reads
only lineitem columns, so `revenue` can be computed on lineitem *before* the join. The
join then gathers one `revenue` column instead of `l_extendedprice` + `l_discount`, and —
this is where it pays most — a **distributed** join shuffles the one narrow column instead
of the two it is derived from. Column pruning (`rewrite_projection`) then drops the now-
unused source columns from the join output.

Correctness is why this is conservative. It only fires on an **inner** join (both sides
keep exactly their matched rows, so computing the expression on a side's pre-join rows and
then joining is identical to computing it on the join output — the extra unmatched rows are
dropped either way), only for expressions that reference a **single** side, and only for
expressions built from operations that **cannot raise** on the extra unmatched rows
(arithmetic +/-/*, comparisons, boolean logic, `CASE`/`COALESCE`/`CAST`, `col`, literals).
A division, a UDF, or a mixed-side expression is left in place.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.expr_ir import (
    Binary,
    Case,
    Cast,
    Coalesce,
    Col,
    Expr,
    Greatest,
    InList,
    IsNan,
    IsNotNull,
    IsNull,
    Least,
    Lit,
    referenced_columns,
)
from batcher.plan.expr_rewrite import substitute_columns
from batcher.plan.logical import Join, JoinOutputCol, LogicalPlan, Project, Projection

__all__ = ["push_projection_through_join"]

# Binary operators that cannot raise on an unmatched row (no integer div/mod-by-zero).
_SAFE_BINARY_OPS = frozenset({"add", "sub", "mul", "eq", "ne", "lt", "le", "gt", "ge", "and", "or"})
# Expression node types that cannot raise, so evaluating them on the extra (later-dropped)
# unmatched rows of an inner join is harmless. Anything else keeps the projection above.
_SAFE_LEAF_TYPES = (
    Col,
    Lit,
    IsNull,
    IsNotNull,
    IsNan,
    Coalesce,
    Greatest,
    Least,
    InList,
    Cast,
    Case,
)


def _is_push_safe(expr: Expr) -> bool:
    """Whether `expr` is built entirely from non-raising operations (so it may run on the
    inner join's unmatched rows without introducing a spurious error)."""
    if isinstance(expr, Binary):
        return (
            expr.op in _SAFE_BINARY_OPS and _is_push_safe(expr.left) and _is_push_safe(expr.right)
        )
    if isinstance(expr, _SAFE_LEAF_TYPES):
        return all(_is_push_safe(c) for c in _child_exprs(expr))
    return False


def _child_exprs(expr: Expr) -> list[Expr]:
    """Immediate sub-expressions of `expr` (best-effort over the IR node's fields)."""
    kids: list[Expr] = []
    for attr in ("left", "right", "value", "arg", "input", "condition", "otherwise"):
        v = getattr(expr, attr, None)
        if isinstance(v, Expr):
            kids.append(v)
    for attr in ("args", "values", "operands", "branches", "items"):
        v = getattr(expr, attr, None)
        if isinstance(v, (list, tuple)):
            kids.extend(x for x in v if isinstance(x, Expr))
            for x in v:
                if isinstance(x, (list, tuple)):
                    kids.extend(y for y in x if isinstance(y, Expr))
    return kids


@rule(name="push_projection_through_join", phase=Phase.PUSHDOWN, matches=(Project,))
def push_projection_through_join(node: Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Move each single-side, non-raising computed projection item onto the join input it
    reads, adding a join output column for the result. Fires only on inner joins."""
    join = node.input
    if not isinstance(join, Join) or join.join_type != "inner":
        return None

    # Where each join-output alias comes from: alias -> (side, source_name).
    out_src: dict[str, tuple[str, str]] = {oc.alias: (oc.side, oc.name) for oc in join.output}
    left_avail = set(join.left.available_columns())
    right_avail = set(join.right.available_columns())
    taken_out = {oc.alias for oc in join.output}
    taken_left = set(left_avail)
    taken_right = set(right_avail)

    new_left_items: list[Projection] = []
    new_right_items: list[Projection] = []
    extra_out: list[JoinOutputCol] = []
    new_proj_items: list[Projection] = []
    changed = False

    for item in node.items:
        pushed = _try_push(
            item, out_src, taken_out, taken_left, taken_right, left_avail, right_avail
        )
        if pushed is None:
            new_proj_items.append(item)
            continue
        side, comp_item, out_col = pushed
        (new_left_items if side == "left" else new_right_items).append(comp_item)
        extra_out.append(out_col)
        # Read the join's newly-added column (its alias may differ from the item's when the
        # item overwrites an existing join-output name), still emitting the item's own alias.
        new_proj_items.append(Projection(item.alias, Col(out_col.alias)))
        changed = True

    if not changed:
        return None

    new_left = _with_extra(join.left, new_left_items)
    new_right = _with_extra(join.right, new_right_items)
    new_join = Join(
        new_left,
        new_right,
        join.left_keys,
        join.right_keys,
        join.join_type,
        (*join.output, *extra_out),
        join.strategy,
    )
    return Project(new_join, tuple(new_proj_items))


def _try_push(item, out_src, taken_out, taken_left, taken_right, left_avail, right_avail):
    """Return `(side, computed_item, join_output_col)` if `item` is a pushable single-side
    computed expression, else None."""
    if isinstance(item.expr, Col):
        return None  # a bare column passthrough gains nothing
    refs = referenced_columns(item.expr)
    if not refs or any(r not in out_src for r in refs):
        return None
    sides = {out_src[r][0] for r in refs}
    if len(sides) != 1:
        return None
    side = next(iter(sides))
    if not _is_push_safe(item.expr):
        return None
    # Rewrite the expression from join-output aliases to the side's own source names.
    mapping = {r: Col(out_src[r][1]) for r in refs}
    rewritten = substitute_columns(item.expr, mapping)
    avail = left_avail if side == "left" else right_avail
    taken_side = taken_left if side == "left" else taken_right
    # A fresh name for the computed column on the side, and a fresh output alias.
    comp_name = _fresh(item.alias, taken_side | avail)
    taken_side.add(comp_name)
    out_alias = item.alias if item.alias not in taken_out else _fresh(item.alias, taken_out)
    taken_out.add(out_alias)
    return (
        side,
        Projection(comp_name, rewritten),
        JoinOutputCol(side=side, name=comp_name, alias=out_alias),
    )


def _fresh(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    i = 0
    while f"__pj_{base}_{i}" in taken:
        i += 1
    return f"__pj_{base}_{i}"


def _with_extra(side_plan: LogicalPlan, extra: list[Projection]) -> LogicalPlan:
    """Wrap `side_plan` in a Project that keeps all its columns and adds `extra`."""
    if not extra:
        return side_plan
    passthrough = [Projection(c, Col(c)) for c in side_plan.available_columns()]
    return Project(side_plan, (*passthrough, *extra))
