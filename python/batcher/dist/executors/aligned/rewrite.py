"""Push each join against a broadcast input down to the broadcast input it keys on.

A fact-first join order, `(lineitem JOIN (customer JOIN orders)) JOIN nation`, is the right
shape on one node: every dimension becomes one hash build and the fact table streams past
them. Per key-range unit it is the wrong one. Each unit rebuilds every broadcast build, and
`customer` is 150M rows at TPC-H SF1000 until `nation`, filtered to one region, cuts it to
30M -- a filter that never meets `customer` until after the joins every unit pays for.

When the outer join's other side `B2` reads no aligned source and every one of its keys is
a column of one aligned-free subtree `B1` below it, the join can be moved down to `B1`:
`B1 JOIN B2` is formed first, and `B2`'s columns ride up to where the join used to be.
Inner joins associate and commute, and a filter or projection below the old join cannot
reference `B2`'s columns, so the relation is unchanged. `B1 JOIN B2` is then one
aligned-free subtree, which `hoist.hoist_broadcasts` evaluates once, before any unit runs.

The path followed is restricted to inner joins, filters and plain-column projections, and
each column the moved join adds travels under a fresh private alias, so nothing above can
see a column it did not see before; a final projection restores the old join's exact output.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable

from batcher.plan.expr_ir import Col, col
from batcher.plan.logical import Filter, Join, JoinOutputCol, LogicalPlan, Project, Projection

__all__ = ["distinct_membership_sides", "group_broadcast_joins"]

_FRESH = itertools.count()


def _fresh(name: str) -> str:
    return f"__ab{next(_FRESH)}_{name}"


def _source_free(node: LogicalPlan, aligned: frozenset[int]) -> bool:
    from batcher.plan.visitor import scanned_source_ids

    scanned = scanned_source_ids(node)
    return bool(scanned) and not (scanned & aligned)


def _names_on(node: Join, origins: list[tuple[str, str]], side: str) -> list[str] | None:
    """Each key, given as `(side, name)` in `node`'s inputs, as a column of `side`'s input.

    A key already on `side` is itself; one on the other side stands for `side`'s half of
    the key pair it belongs to, since an inner join emits only rows where the two are
    equal. TPC-H q9 joins `lineitem` to `partsupp` on `lineitem`'s part key and then to
    green parts on the same column; through the pair, the part join moves onto `partsupp`,
    and `partsupp JOIN part` becomes one broadcast of ~40M rows rather than 800M.
    """
    own, other = (
        (node.left_keys, node.right_keys) if side == "left" else (node.right_keys, node.left_keys)
    )
    names = []
    for key_side, name in origins:
        if key_side == side:
            names.append(name)
        elif name in other:
            names.append(own[other.index(name)])
        else:
            return None
    return names


def _push(
    node: LogicalPlan,
    b2: LogicalPlan,
    keys: list[str],
    b2_keys: tuple[str, ...],
    aligned,
    carry: list[str],
) -> tuple[LogicalPlan, dict[str, str]] | None:
    """`node` with `b2` joined in on `keys` (names in `node`'s output) at the broadcast leaf.

    Returns the rewritten node, whose output is `node`'s plus each `carry` column of `b2`,
    and the map from each of those names to its alias there; None when no such leaf exists
    on a path of inner joins, filters and plain projections.
    """
    if _source_free(node, aligned):
        b2_out = {c: _fresh(c) for c in carry}
        output = [JoinOutputCol("left", c, c) for c in node.available_columns()]
        output += [JoinOutputCol("right", c, alias) for c, alias in b2_out.items()]
        joined = Join(
            left=node,
            right=b2,
            left_keys=tuple(keys),
            right_keys=b2_keys,
            join_type="inner",
            output=tuple(output),
        )
        return joined, b2_out
    if isinstance(node, Filter):
        pushed = _push(node.input, b2, keys, b2_keys, aligned, carry)
        if pushed is None:
            return None
        return Filter(input=pushed[0], predicate=node.predicate), pushed[1]
    if isinstance(node, Project):
        by_alias = {item.alias: item.expr for item in node.items}
        if not all(isinstance(by_alias.get(k), Col) for k in keys):
            return None
        pushed = _push(node.input, b2, [by_alias[k].name for k in keys], b2_keys, aligned, carry)
        if pushed is None:
            return None
        inner, colmap = pushed
        up = {c: _fresh(c) for c in colmap}
        items = (*node.items, *(Projection(up[c], col(colmap[c])) for c in colmap))
        return Project(input=inner, items=items), up
    if isinstance(node, Join) and node.join_type == "inner":
        origin = {o.alias: (o.side, o.name) for o in node.output}
        if not all(k in origin for k in keys):
            return None
        first = origin[keys[0]][0]
        pushed = None
        for side in (first, "right" if first == "left" else "left"):
            names = _names_on(node, [origin[k] for k in keys], side)
            child = node.left if side == "left" else node.right
            if names is not None:
                pushed = _push(child, b2, names, b2_keys, aligned, carry)
            if pushed is not None:
                break
        if pushed is None:
            return None
        new_child, colmap = pushed
        up = {c: _fresh(c) for c in colmap}
        output = (*node.output, *(JoinOutputCol(side, colmap[c], up[c]) for c in colmap))
        left, right = (new_child, node.right) if side == "left" else (node.left, new_child)
        return (
            Join(
                left=left,
                right=right,
                left_keys=node.left_keys,
                right_keys=node.right_keys,
                join_type="inner",
                output=output,
                strategy=node.strategy,
            ),
            up,
        )
    return None


def _move_down(node: Join, aligned: frozenset[int]) -> LogicalPlan | None:
    """`node` with its broadcast side joined in further down the other side, or None."""
    if node.join_type != "inner":
        return None
    if _source_free(node.right, aligned) and not _source_free(node.left, aligned):
        spine, b2, spine_keys, b2_keys, spine_side = (
            node.left,
            node.right,
            node.left_keys,
            node.right_keys,
            "left",
        )
    elif _source_free(node.left, aligned) and not _source_free(node.right, aligned):
        spine, b2, spine_keys, b2_keys, spine_side = (
            node.right,
            node.left,
            node.right_keys,
            node.left_keys,
            "right",
        )
    else:
        return None
    if (
        isinstance(spine, Join)
        and _source_free(spine.left, aligned) is False
        and (spine.right is b2 or spine.left is b2)
    ):
        return None
    # Only what the old join emitted from `b2` rides up: carrying every column of a wide
    # dimension widens the broadcast each unit holds, for columns nothing above reads.
    carry = list(dict.fromkeys(o.name for o in node.output if o.side != spine_side))
    pushed = _push(spine, b2, list(spine_keys), b2_keys, aligned, carry)
    if pushed is None:
        return None
    moved, colmap = pushed
    if _source_free(moved, aligned):
        return None
    items = tuple(
        Projection(o.alias, col(o.name if o.side == spine_side else colmap[o.name]))
        for o in node.output
    )
    return Project(input=moved, items=items)


def group_broadcast_joins(body: LogicalPlan, aligned: frozenset[int]) -> LogicalPlan:
    """`body` with every movable broadcast join pushed down to its broadcast leaf."""
    from batcher.plan.visitor import transform_up

    def step(node: LogicalPlan) -> LogicalPlan:
        if isinstance(node, Join):
            moved = _move_down(node, aligned)
            if moved is not None:
                return moved
        return node

    return transform_up(body, step)


def distinct_membership_sides(
    plan: LogicalPlan, worth: Callable[[LogicalPlan], bool] | None = None
) -> LogicalPlan:
    """`plan` with each semi or anti join's right side cut to its distinct join keys.

    `worth`, given, picks the right sides to cut; the rest are left as they are.

    Such a join asks only whether a key occurs on its right side, never how often, so the
    right side's distinct keys answer it exactly. What that buys the aligned planner is an
    input it can hold: `NOT EXISTS (SELECT * FROM orders WHERE o_custkey = c_custkey)`
    becomes the set of customer keys with an order, a grouped aggregate the fleet computes
    file by file, where `orders` itself (1.5B rows at TPC-H SF1000) could be neither held on
    every node nor aligned with `customer`, which is stored in a different key's order.
    """
    from batcher.plan.logical import Aggregate
    from batcher.plan.visitor import transform_up

    def step(node: LogicalPlan) -> LogicalPlan:
        if (
            isinstance(node, Join)
            and node.join_type in ("semi", "anti")
            and not isinstance(node.right, Aggregate)
            and (worth is None or worth(node.right))
        ):
            keys = tuple(Projection(k, col(k)) for k in dict.fromkeys(node.right_keys))
            return dataclasses.replace(node, right=Aggregate(node.right, keys, ()))
        return node

    return transform_up(plan, step)
