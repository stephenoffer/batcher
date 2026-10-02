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

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.agg_pushdown.gates import _OUTER_GROUP_SHARE
from batcher.kyber.rules.agg_pushdown.rules import _MIN_PREAGG_REDUCTION, _PREAGG_MERGE
from batcher.plan.expr_ir import AggExpr, Col, referenced_columns, remap_columns
from batcher.plan.logical import (
    Aggregate,
    AggregateSpec,
    Join,
    JoinOutputCol,
    LogicalPlan,
    Projection,
)

__all__ = ["pre_aggregate_beneath_dimension", "pre_aggregate_facts"]

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


#: The row reduction the partial aggregate must promise, from its keys' distinct counts.
_MIN_KEY_REDUCTION = 4.0


@rule(name="pre_aggregate_beneath_dimension", phase=Phase.FUSION, matches=(Aggregate,))
def pre_aggregate_beneath_dimension(node: Aggregate, ctx: OptimizerContext) -> LogicalPlan | None:
    """Pre-aggregate the facts with the dimension that only groups them, beneath the one that
    supplies string keys.

    TPC-DS q4/q11/q74's `year_total` is `(store_sales JOIN customer) JOIN date_dim`, grouped by
    seven `customer` strings and `d_year`, summing a `store_sales` measure. No join in that
    tree has the facts on one side and every grouping dimension on the other, so neither
    `pre_aggregation_through_join` nor `_push_under` can move the strings off the facts: every
    one of 27M rows at sf10 is joined to `customer` and row-encoded by seven strings into the
    group hash, and that aggregate is 3 of q11's 3.5 s.

    Re-associated, `customer JOIN Agg(store_sales JOIN date_dim by ss_customer_sk, d_year)`,
    the strings meet 1.5M partial groups instead. When join order already put the facts and
    the plain dimension together, `(store_sales JOIN date_dim) JOIN customer`, the partial
    goes beneath the top join as it is (`_push_under`). Both are correct for any fan-out
    (see `_rewrite`), so neither needs the uniqueness proof `pre_aggregation_through_join`
    waits for -- which on preloaded tables is a measured exact ndv that is rarely there.

    This is the one shape `pre_aggregate_facts` takes that is a rule rather than an
    alternative the aligned planner may keep, so it is narrowed to where that function's
    measured single-node losses (TPC-H q10, q12, q18 at sf100) do not reach. Re-associated,
    the measures must read the facts (`B`) alone, which excludes q10, whose measures are on
    the far side of the upper join. Either way the string keys must come from the dimension
    the partial leaves out, and the keys the partial groups by must promise a
    `_MIN_KEY_REDUCTION` from the product of their distinct counts -- which q18's
    `o_orderkey`, one group per order, cannot.

    Args:
        node: The aggregate.
        ctx: The optimizer context, for the estimator.

    Returns:
        The re-associated aggregate, or None.
    """
    upper = node.input
    if not isinstance(upper, Join) or upper.join_type != "inner" or not node.aggregates:
        return None
    for rewritten in (_push_under(node, upper), _reassociated(node, upper, facts_only=True)):
        if rewritten is not None and _partial_reduces(rewritten, node, ctx):
            return rewritten
    return None


def _partial_reduces(rewritten: Aggregate, node: Aggregate, ctx: OptimizerContext) -> bool:
    """Whether the partial pays: it shrinks its input, and the aggregate above keeps the grain.

    The partial's keys must be able to form at most a `_MIN_KEY_REDUCTION`th of its input,
    by the *product* of their distinct counts -- the bound independent keys can reach --
    rather than the estimator's group count, which damps that product on an assumption of
    correlation and so can invent a reduction (`gates._moves_string_keys` has the case).

    And a partial that only just clears that bar must also keep the outer aggregate's grain,
    by the `_OUTER_GROUP_SHARE` `_moves_string_keys` applies: its saving is the string work
    the outer aggregate stops doing per fact row, which is small when the outer aggregate
    groups far more coarsely. `lineitem JOIN orders GROUP BY o_orderpriority`, five groups,
    would otherwise take a 1.5M-group partial -- a 4x reduction -- it does not need
    (26 -> 95 ms). A partial the estimator expects to reduce by `_MIN_PREAGG_REDUCTION` or
    more pays at any grain, because its hash table is small: TPC-DS q70's per-state ROLLUP
    level groups 5.8M fact rows into 51 stores, and q3's `(ss_item_sk, d_year)` over one
    manufacturer's items and one month is a few hundred groups, which the key product --
    every item, every year -- overstates by four orders of magnitude. The product stays the
    test for the first bar, where an overstatement can only refuse. A side the plan has
    already grouped (another rule's partial) is never grouped again.
    """
    from batcher.plan.visitor import walk

    partial = next(
        n
        for n in walk(rewritten.input)
        if isinstance(n, Aggregate) and any(a.alias.startswith(_PARTIAL) for a in n.aggregates)
    )
    if any(isinstance(n, Aggregate) for n in walk(partial.input)):
        return False
    try:
        side = ctx.estimator.estimate(partial.input)
        groups = 1.0
        for key in partial.group_keys:
            col = side.columns.get(key.alias)
            if col is None or col.ndv is None:
                return False
            groups *= max(1.0, col.ndv)
        if groups * _MIN_KEY_REDUCTION > side.rows:
            return False
        if _group_bound(node, ctx) * _OUTER_GROUP_SHARE >= groups:
            return True
        expected = ctx.estimator.estimate(partial).rows
    except Exception:  # an unsizable side is no evidence of a reduction
        return False
    return expected > 0 and expected * _MIN_PREAGG_REDUCTION <= side.rows


def _group_bound(node: Aggregate, ctx: OptimizerContext) -> float:
    """The most groups `node` can form: the product of its keys' distinct counts.

    Measured the way the partial is, so the two sides of the comparison are the same kind of
    number. The estimator's own group count damps that product on an assumption that the keys
    are correlated, and seven customer strings are correlated -- with the customer -- so it
    read TPC-DS q4's `year_total` as coarser than its own partial and refused a push worth
    3.8 s -> 1.1 s at sf10. Falls back to that estimate when a key's count is unknown.
    """
    stats = ctx.estimator.estimate(node.input)
    bound = 1.0
    for key in node.group_keys:
        col = stats.columns.get(key.expr.name) if isinstance(key.expr, Col) else None
        if col is None or col.ndv is None:
            return ctx.estimator.estimate(node).rows
        bound *= max(1.0, col.ndv)
    return min(bound, stats.rows)


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
    return _reassociated(node, upper)


def _reassociated(node: Aggregate, upper: Join, *, facts_only: bool = False) -> Aggregate | None:
    """`Agg((A JOIN B) JOIN C)` -> `Agg'(A JOIN Agg_partial(B JOIN C))`, or None.

    The measures must read only `B` and `C` (`B` alone when `facts_only`), and the upper join
    must reach the lower one through `B`.
    """
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
            barred = {a_letter, "c"} if facts_only else {a_letter}
            if any(col not in origin or origin[col][0] in barred for col in cols):
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
