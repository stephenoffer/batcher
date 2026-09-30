"""Re-associate a star join so its measures can be pre-aggregated by the dimension key.

Join ordering builds TPC-H q10 as `((customer JOIN nation) JOIN orders) JOIN lineitem` and
groups the result by seven customer columns, five of them strings. Every join in that tree
has customer on one side and at most one of the two fact tables on the other, so
`pre_aggregate_join_measures` never sees `orders JOIN lineitem` whole and can only
pre-aggregate `lineitem` by `l_orderkey`, which reduces nothing.

Inner joins associate: when the upper join's keys on the lower join's side come only from
its fact input, `(A JOIN B) JOIN C` is `A JOIN (B JOIN C)`. Re-associated, the measures
live entirely in `B JOIN C`, which is grouped by its key to `A` (and by any group key it
holds) into partial aggregates, and the aggregate above merges the partials per group.

That is correct for any fan-out on an inner join, so no uniqueness proof is needed: a row of
`A` matches every row of `B JOIN C` with its key, before the rewrite and after it, so each
group sums the same rows' partial sums (count merges as a sum, min and max as themselves).
It is gated like `pre_aggregation_through_join`: a measured reduction, or group keys that are
strings from `A`, which the rewrite stops dragging through the fact join row by row
(`gates._moves_string_keys` has the measurement).
"""

from __future__ import annotations

import dataclasses
import itertools

import pyarrow as pa

from batcher.kyber.rules.agg_pushdown.rules import _PREAGG_MERGE
from batcher.plan.expr_ir import AggExpr, Col, referenced_columns, remap_columns
from batcher.plan.logical import (
    Aggregate,
    AggregateSpec,
    Join,
    JoinOutputCol,
    LogicalPlan,
    Projection,
)

__all__ = ["pre_aggregate_facts"]

_FRESH = itertools.count()
#: Prefix of the partial-aggregate columns this rule creates, which is also how it recognizes
#: its own output.
_PARTIAL = "__rp"


def _fresh(prefix: str) -> str:
    return f"__{prefix}{next(_FRESH)}"


def _sides(join: Join) -> dict[str, tuple[str, str]]:
    return {o.alias: (o.side, o.name) for o in join.output}


def _split(
    upper: Join,
) -> tuple[str, Join, LogicalPlan, tuple[str, ...], tuple[str, ...]] | None:
    """`(lower_side, lower, c, lower_keys, c_keys)` when one input of `upper` is an inner join."""
    for side in ("left", "right"):
        lower = upper.left if side == "left" else upper.right
        c = upper.right if side == "left" else upper.left
        if isinstance(lower, Join) and lower.join_type == "inner":
            keys = (upper.left_keys, upper.right_keys)
            return side, lower, c, keys[side == "right"], keys[side == "left"]
    return None


def _already_pushed(plan: LogicalPlan) -> bool:
    """Whether `plan` holds a partial aggregate this rule built."""
    from batcher.plan.visitor import walk

    return any(
        isinstance(n, Aggregate) and any(a.alias.startswith(_PARTIAL) for a in n.aggregates)
        for n in walk(plan)
    )


def _string_key_from(node: Aggregate, a_aliases: set[str]) -> bool:
    """Whether some group key is a string column supplied by `A`."""
    schema = node.input.available_schema()
    if schema is None:
        return False
    for key in node.group_keys:
        name = key.expr.name if isinstance(key.expr, Col) else None
        if name in a_aliases and name in schema.names:
            dtype = schema.field(name).type
            if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
                return True
    return False


def pre_aggregate_facts(plan: LogicalPlan) -> LogicalPlan:
    """`plan` with each star aggregate's facts pre-aggregated beneath its dimensions.

    Not an optimizer rule, on measurement: on one node the rewrite lost to the plan join
    ordering already chose -- 64 cores at SF100, TPC-H q10 went from 1.46x DuckDB to 4.00x,
    q18 from 1.16x to 2.71x, and q12, whose "dimension" is `lineitem`, from 2.01x to 3.67x.
    What it buys is distributed: a dimension no node can hold whole (q10's and q18's
    `customer`, 25 GB at SF1000) leaves the facts' side, so they align without it. So the
    aligned planner asks for it as an alternative and keeps it only when it aligns better.

    Args:
        plan: An optimized logical plan.

    Returns:
        The rewritten plan, or `plan` itself when no aggregate has the shape.
    """
    from batcher.plan.visitor import transform_up

    return transform_up(
        plan,
        lambda n: (_rewrite_aggregate(n) or n) if isinstance(n, Aggregate) else n,
    )


def _rewrite_aggregate(node: Aggregate) -> LogicalPlan | None:
    """`Agg((A JOIN B) JOIN C)` -> `Agg'(A JOIN Agg_partial(B JOIN C))` when the measures
    read only `B` and `C` and the upper join reaches the lower one through `B` alone."""
    upper = node.input
    if not isinstance(upper, Join) or upper.join_type != "inner" or not node.aggregates:
        return None
    # Two ways to put the facts under the aggregate: push beneath the top join as it is, or
    # first group the dimensions together. The better is the one that pre-aggregates more of
    # the facts: q18's plain push takes `lineitem JOIN orders`, q10's takes `lineitem` alone
    # where grouping `customer` with `nation` lets it take `orders JOIN lineitem`.
    options = [_push_under(node, upper)]
    together = _dimensions_together(upper)
    if together is not None and not _already_pushed(upper):
        options.append(_push_under(dataclasses.replace(node, input=together), together))
    options = [o for o in options if o is not None]
    if options:
        return max(options, key=_facts_pushed)
    split = _split(upper)
    if split is None:
        return None
    lower_side, lower, c, lower_keys, c_keys = split
    # Measures already pre-aggregated below by this rule: a second push only re-groups
    # partials that are one row per key already (q10's re-grouped per-customer sums by
    # `nation`). Recognized by the partials' own names, wherever join order left them.
    if _already_pushed(upper):
        return None
    up, low = _sides(upper), _sides(lower)
    # Name every upper-join column by where it finally comes from: A, B or C.
    origin: dict[str, tuple[str, str]] = {}
    for alias, (side, name) in up.items():
        if side != lower_side:
            origin[alias] = ("c", name)
        elif name in low:
            origin[alias] = ("ab"[low[name][0] == "right"], low[name][1])
    for a_side in ("left", "right"):
        b_side = "right" if a_side == "left" else "left"
        a_letter = "ab"[a_side == "right"]
        # The upper join must reach the lower one through B only.
        if not all(k in low and low[k][0] == b_side for k in lower_keys):
            continue
        a_aliases = {al for al, (who, _) in origin.items() if who == a_letter}
        # Measures read only B and C, through decomposable functions.
        specs = []
        for spec in node.aggregates:
            agg = spec.agg
            if agg.func not in _PREAGG_MERGE or agg.input2 is not None:
                break
            cols = referenced_columns(agg.input) if agg.input is not None else set()
            if any(col not in origin or origin[col][0] == a_letter for col in cols):
                break
            specs.append(spec)
        else:
            if not _string_key_from(node, a_aliases):
                continue
            rewritten = _rewrite(node, lower, c, lower_keys, c_keys, a_side, origin, a_letter)
            if rewritten is not None:
                return rewritten
    return None


def _facts_pushed(rewritten: Aggregate) -> int:
    """How many scans the partial aggregate of a `_push_under` rewrite reads."""
    from batcher.plan.logical import Scan
    from batcher.plan.visitor import walk

    partial = next(
        n
        for n in walk(rewritten.input)
        if isinstance(n, Aggregate) and any(a.alias.startswith(_PARTIAL) for a in n.aggregates)
    )
    return sum(isinstance(n, Scan) for n in walk(partial.input))


def _dimensions_together(upper: Join) -> Join | None:
    """`(F JOIN B) JOIN C` as `F JOIN (B JOIN C)` when `C` joins through `B` alone.

    TPC-H q10 in DataFrame form is `((lineitem JOIN orders) JOIN customer) JOIN nation`, and
    grouped by customer and nation strings no join in it has the facts on one side and every
    dimension on the other. `nation` joins `customer` only, so the two associate into one
    dimension side and `_push_under` can pre-aggregate the facts beneath it. The result
    exposes the same columns under the same names as `upper`.
    """
    split = _split(upper)
    if split is None:
        return None
    lower_side, lower, c, lower_keys, c_keys = split
    low = _sides(lower)
    for b_side in ("left", "right"):
        if all(k in low and low[k][0] == b_side for k in lower_keys):
            return _regroup(upper, lower_side, lower, c, lower_keys, c_keys, b_side)
    return None


def _regroup(
    upper: Join,
    lower_side: str,
    lower: Join,
    c: LogicalPlan,
    lower_keys: tuple[str, ...],
    c_keys: tuple[str, ...],
    b_side: str,
) -> Join:
    """`upper` rebuilt as `F JOIN (B JOIN C)`, where `B` is `lower`'s `b_side` input."""
    low = _sides(lower)
    f = lower.left if b_side == "right" else lower.right
    b = lower.left if b_side == "left" else lower.right
    f_keys = lower.right_keys if b_side == "left" else lower.left_keys
    b_keys = lower.left_keys if b_side == "left" else lower.right_keys
    # B JOIN C, carrying every column of either that `upper` exposes, and B's keys to F.
    bc_out: list[JoinOutputCol] = []
    where: dict[tuple[str, str], str] = {}

    def carry(side: str, name: str) -> str:
        if (side, name) not in where:
            where[(side, name)] = _fresh("rd")
            bc_out.append(JoinOutputCol(side, name, where[(side, name)]))
        return where[(side, name)]

    b_key_names = [carry("left", k) for k in b_keys]
    out: list[JoinOutputCol] = []
    for alias, (side, name) in _sides(upper).items():
        if side != lower_side:  # a column of C
            out.append(JoinOutputCol("right", carry("right", name), alias))
        elif low[name][0] == b_side:  # a column of B
            out.append(JoinOutputCol("right", carry("left", low[name][1]), alias))
        else:  # a column of F
            out.append(JoinOutputCol("left", low[name][1], alias))
    bc = Join(
        left=b,
        right=c,
        left_keys=tuple(low[k][1] for k in lower_keys),
        right_keys=c_keys,
        join_type="inner",
        output=tuple(bc_out),
    )
    return Join(
        left=f,
        right=bc,
        left_keys=f_keys,
        right_keys=tuple(b_key_names),
        join_type="inner",
        output=tuple(out),
    )


def _push_under(node: Aggregate, upper: Join) -> Aggregate | None:
    """`Agg(F JOIN D)` -> `Agg'(Agg_partial(F) JOIN D)` when the measures read `F` only.

    The case that needs no re-association: join order already put the dimension `D` on one
    side of the top join and every fact on the other. TPC-H q18 is the shape --
    `(lineitem JOIN orders) JOIN customer` grouped by five customer and order columns -- and
    pre-aggregated by `o_orderkey` it leaves the few thousand large orders, not the lineitems,
    to meet `customer`. `pre_aggregate_join_measures` makes the same move on a *measured*
    reduction; this one takes the string-key licence the re-association does.
    """
    if _already_pushed(upper):
        return None
    sides = _sides(upper)
    for f_side in ("left", "right"):
        d_side = "right" if f_side == "left" else "left"
        d_aliases = {a for a, (side, _) in sides.items() if side == d_side}
        f_aliases = {a for a, (side, _) in sides.items() if side == f_side}
        # Strings on the fact side too mean a dimension is still in it: pushing under this
        # join would group the facts by that dimension's strings and move nothing off them.
        if not _string_key_from(node, d_aliases) or _string_key_from(node, f_aliases):
            continue
        measures_in_f = all(
            spec.agg.func in _PREAGG_MERGE
            and spec.agg.input2 is None
            and all(
                sides.get(col, (d_side, ""))[0] == f_side
                for col in (
                    referenced_columns(spec.agg.input) if spec.agg.input is not None else ()
                )
            )
            for spec in node.aggregates
        )
        group_cols = [k.expr.name for k in node.group_keys if isinstance(k.expr, Col)]
        if not measures_in_f or len(group_cols) != len(node.group_keys):
            continue
        if any(g not in sides for g in group_cols):
            continue
        f = upper.left if f_side == "left" else upper.right
        d = upper.right if f_side == "left" else upper.left
        f_keys = upper.left_keys if f_side == "left" else upper.right_keys
        d_keys = upper.right_keys if f_side == "left" else upper.left_keys
        f_groups = [sides[g][1] for g in group_cols if sides[g][0] == f_side]
        keys = tuple(Projection(n, Col(n)) for n in dict.fromkeys([*f_keys, *f_groups]))
        partials = tuple(
            AggregateSpec(
                _fresh(_PARTIAL[2:]),
                AggExpr(
                    spec.agg.func,
                    remap_columns(spec.agg.input, {a: n for a, (_, n) in sides.items()})
                    if spec.agg.input is not None
                    else None,
                ),
            )
            for spec in node.aggregates
        )
        pushed = Aggregate(f, keys, partials)
        out = [JoinOutputCol(sides[g][0], sides[g][1], g) for g in group_cols]
        out += [JoinOutputCol(f_side, p.alias, p.alias) for p in partials]
        left, right = (pushed, d) if f_side == "left" else (d, pushed)
        lk, rk = (f_keys, d_keys) if f_side == "left" else (d_keys, f_keys)
        top = Join(
            left=left,
            right=right,
            left_keys=lk,
            right_keys=rk,
            join_type="inner",
            output=tuple(out),
            strategy=upper.strategy,
        )
        merged = tuple(
            AggregateSpec(spec.alias, AggExpr(_PREAGG_MERGE[spec.agg.func], Col(p.alias)))
            for spec, p in zip(node.aggregates, partials, strict=True)
        )
        return dataclasses.replace(node, input=top, aggregates=merged)
    return None


def _rewrite(
    node: Aggregate,
    lower: Join,
    c: LogicalPlan,
    lower_keys: tuple[str, ...],
    c_keys: tuple[str, ...],
    a_side: str,
    origin: dict[str, tuple[str, str]],
    a_letter: str,
) -> Aggregate | None:
    b_side = "right" if a_side == "left" else "left"
    a = lower.left if a_side == "left" else lower.right
    b = lower.left if b_side == "left" else lower.right
    low = _sides(lower)
    a_keys = lower.left_keys if a_side == "left" else lower.right_keys
    b_keys = lower.left_keys if b_side == "left" else lower.right_keys
    # B JOIN C, carrying B's keys to A and every B/C column the aggregate reads, each under a
    # fresh name: B's and C's own names may collide, which the old upper join resolved.
    bc_out: list[JoinOutputCol] = []
    bc_name: dict[tuple[str, str], str] = {}

    def carry(who: str, name: str) -> str:
        if (who, name) not in bc_name:
            bc_name[(who, name)] = _fresh("rs")
            side = "left" if who == "b" else "right"
            bc_out.append(JoinOutputCol(side, name, bc_name[(who, name)]))
        return bc_name[(who, name)]

    b_letter = "ab"[b_side == "right"]
    b_key_names = [carry("b", k) for k in b_keys]
    mapping: dict[str, str] = {}
    for alias, (who, name) in origin.items():
        if who != a_letter:
            mapping[alias] = carry("b" if who == b_letter else "c", name)
    bc = Join(
        left=b,
        right=c,
        left_keys=tuple(low[k][1] for k in lower_keys),
        right_keys=c_keys,
        join_type="inner",
        output=tuple(bc_out),
    )
    # Partials of B JOIN C by its keys to A and by the group keys it holds.
    group_cols = [k.expr.name for k in node.group_keys if isinstance(k.expr, Col)]
    if len(group_cols) != len(node.group_keys) or any(g not in origin for g in group_cols):
        return None
    names = dict.fromkeys([*b_key_names, *(mapping[g] for g in group_cols if g in mapping)])
    keys = tuple(Projection(n, Col(n)) for n in names)
    partials = tuple(
        AggregateSpec(
            _fresh(_PARTIAL[2:]),
            AggExpr(spec.agg.func, remap_columns(spec.agg.input, mapping))
            if spec.agg.input is not None
            else AggExpr(spec.agg.func, None),
        )
        for spec in node.aggregates
    )
    pushed = Aggregate(bc, keys, partials)
    # A JOIN partials, exposing each group key under its old name and each partial.
    out = [
        JoinOutputCol(a_side, origin[g][1], g)
        for g in group_cols
        if origin.get(g, ("", ""))[0] == a_letter
    ]
    p_side = "right" if a_side == "left" else "left"
    out += [JoinOutputCol(p_side, mapping[g], g) for g in group_cols if g in mapping]
    out += [JoinOutputCol(p_side, p.alias, p.alias) for p in partials]
    left, right = (a, pushed) if a_side == "left" else (pushed, a)
    lk, rk = (a_keys, tuple(b_key_names)) if a_side == "left" else (tuple(b_key_names), a_keys)
    top = Join(
        left=left, right=right, left_keys=lk, right_keys=rk, join_type="inner", output=tuple(out)
    )
    merged = tuple(
        AggregateSpec(spec.alias, AggExpr(_PREAGG_MERGE[spec.agg.func], Col(p.alias)))
        for spec, p in zip(node.aggregates, partials, strict=True)
    )
    return dataclasses.replace(node, input=top, aggregates=merged)
