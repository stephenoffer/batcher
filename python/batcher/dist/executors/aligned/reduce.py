"""Shrink what every unit joins: hash joins, and broadcasts cut to the keys that can match.

Both run on the cut body *after* its broadcast subtrees are hoisted, so each broadcast is a
scan of a result already held on the driver, and both are about the same cost: every unit
repeats every join build, so a build that is larger than it needs to be is paid once per
unit, 84 times on TPC-H q5 at SF1000.

`prefer_hash_joins` undoes a choice made for the whole table. Kyber picks a sort-merge join
when both inputs are too large to hash, which is true of `lineitem` and `orders` at SF1000
and false of one unit's key range of them; a unit's inputs are bounded by `_UNIT_BYTES`
and the broadcast budget, so its joins hash. Measured on one q5 unit: 3.3 s to 2.5 s.

`reduce_broadcasts` is a semi-join reduction between broadcasts. q5 joins `supplier` on
`(l_suppkey, c_nationkey) = (s_suppkey, s_nationkey)`, and `c_nationkey` is a column of the
held `customer JOIN nation JOIN region` broadcast, filtered to one region. Every row that
reaches the join therefore carries one of that region's five nation keys, so a `supplier`
row with any other can never match: the 10M-row build every unit made is 2M. Measured on
the same unit, with the hash joins: 2.5 s to 1.6 s, the same 87,273 rows.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa

from batcher.dist.executors.aligned.analysis import AlignedPlan
from batcher.io.source import InMemorySource
from batcher.plan.expr_ir import Col, col
from batcher.plan.logical import (
    Filter,
    Join,
    JoinOutputCol,
    LogicalPlan,
    Project,
    Projection,
    Scan,
)
from batcher.plan.schema import SchemaRef

__all__ = ["prefer_hash_joins", "push_top_n", "reduce_broadcasts", "sliceable_broadcasts"]

#: A reduction is kept only when it removes at least this share of the broadcast's rows.
_MIN_CUT = 0.1


def prefer_hash_joins(body: LogicalPlan) -> LogicalPlan:
    """`body` with each sort-merge join planned as a hash join."""
    from batcher.plan.visitor import transform_up

    def step(node: LogicalPlan) -> LogicalPlan:
        if isinstance(node, Join) and node.strategy == "sort_merge":
            return dataclasses.replace(node, strategy="hash")
        return node

    return transform_up(body, step)


def _resolve(node: LogicalPlan, key: str, held: dict[int, pa.Table]) -> tuple[int, str] | None:
    """The held broadcast, and its column, whose values every row of `node` carries in `key`.

    Followed through filters, plain-column projections and inner joins, and at an inner join
    across its key pair too: each row it emits has equal values in the two key columns.
    """
    if isinstance(node, Scan):
        return (node.source_id, key) if node.source_id in held else None
    if isinstance(node, Filter):
        return _resolve(node.input, key, held)
    if isinstance(node, Project):
        expr = {item.alias: item.expr for item in node.items}.get(key)
        return _resolve(node.input, expr.name, held) if isinstance(expr, Col) else None
    if isinstance(node, Join) and node.join_type == "inner":
        origin = {o.alias: (o.side, o.name) for o in node.output}.get(key)
        if origin is None:
            return None
        side, name = origin
        own, other = (
            (node.left_keys, node.right_keys)
            if side == "left"
            else (node.right_keys, node.left_keys)
        )
        child, twin = (node.left, node.right) if side == "left" else (node.right, node.left)
        found = _resolve(child, name, held)
        if found is None and name in own:
            found = _resolve(twin, other[own.index(name)], held)
        return found
    return None


def _semi_join(target: pa.Table, target_key: str, keys: pa.Table, key: str) -> pa.Table:
    """The rows of `target` whose `target_key` appears in `keys[key]`, by the engine."""
    from batcher.dist.executors.ray_runtime import _single_node

    plan = _semi_plan(Scan(0, SchemaRef.from_arrow(target.schema)), target_key, 1, key, keys.schema)
    return _single_node(plan, [_source(target), _source(keys)])


def _semi_plan(
    target: LogicalPlan, target_key: str, keys_sid: int, key: str, schema: pa.Schema
) -> LogicalPlan:
    """`target` restricted to the rows whose `target_key` appears in held `keys_sid`."""
    right = Project(Scan(keys_sid, SchemaRef.from_arrow(schema)), (Projection(key, col(key)),))
    return Join(
        left=target,
        right=right,
        left_keys=(target_key,),
        right_keys=(key,),
        join_type="semi",
        output=tuple(JoinOutputCol("left", c, c) for c in target.available_columns()),
    )


def _source(table: pa.Table) -> InMemorySource:
    batches = table.to_batches() or [pa.RecordBatch.from_pylist([], schema=table.schema)]
    return InMemorySource(batches, zone_maps=False, ephemeral=True)


def reduce_broadcasts(
    body: LogicalPlan,
    held: dict[int, pa.Table],
    local: dict[int, LogicalPlan] | None = None,
) -> None:
    """Replace each held broadcast by the rows of it that its inner join can match.

    Each hoisted subtree is one `Scan` of a held result, appearing once in `body`, so a
    join with such a scan as one side is the only place those rows are read. A broadcast
    each node evaluates for itself (`local`, by placeholder id) has no rows here yet, so
    its plan is semi-joined to the reducing broadcast instead, which every node holds:
    TPC-H q9 at SF100 then builds each unit's join over the 4.7M `partsupp` rows of green
    parts rather than all 80M.
    """
    from batcher.plan.visitor import walk

    local = {} if local is None else local
    for node in walk(body):
        if not isinstance(node, Join) or node.join_type != "inner":
            continue
        for side in ("left", "right"):
            b2 = node.left if side == "left" else node.right
            spine = node.right if side == "left" else node.left
            if not isinstance(b2, Scan) or (b2.source_id not in held and b2.source_id not in local):
                continue
            b2_keys, spine_keys = (
                (node.left_keys, node.right_keys)
                if side == "left"
                else (node.right_keys, node.left_keys)
            )
            for b2_key, spine_key in zip(b2_keys, spine_keys, strict=True):
                found = _resolve(spine, spine_key, held)
                if found is None or found[0] == b2.source_id:
                    continue
                if b2.source_id in local:
                    local[b2.source_id] = _semi_plan(
                        local[b2.source_id], b2_key, found[0], found[1], held[found[0]].schema
                    )
                    continue
                table = held[b2.source_id]
                reduced = _semi_join(table, b2_key, held[found[0]], found[1])
                if reduced.num_rows <= (1 - _MIN_CUT) * table.num_rows:
                    held[b2.source_id] = reduced


#: Join types under which a broadcast side's rows that match nothing are dropped, by side.
_DROPS_UNMATCHED = {
    "inner": ("left", "right"),
    "semi": ("right",),
    "anti": ("right",),
    "left": ("right",),
}


def _aligned_origin(node: LogicalPlan, key: str, aligned: frozenset[int]) -> tuple[int, str] | None:
    """The aligned scan, and its column, that `key` of `node` is read from, or None."""
    if isinstance(node, Scan):
        return (node.source_id, key) if node.source_id in aligned else None
    if isinstance(node, Filter):
        return _aligned_origin(node.input, key, aligned)
    if isinstance(node, Project):
        expr = {item.alias: item.expr for item in node.items}.get(key)
        return _aligned_origin(node.input, expr.name, aligned) if isinstance(expr, Col) else None
    if isinstance(node, Join) and node.join_type == "inner":
        origin = {o.alias: (o.side, o.name) for o in node.output}.get(key)
        if origin is None:
            return None
        side, name = origin
        return _aligned_origin(node.left if side == "left" else node.right, name, aligned)
    return None


def sliceable_broadcasts(body: LogicalPlan, cut, held: dict[int, pa.Table]) -> dict[int, str]:
    """Held broadcasts a unit may read restricted to its own key range, with their key column.

    One is joined to the aligned side on the alignment key, by a join that drops its rows
    that match nothing. A unit's aligned rows all carry keys in its range, so a broadcast
    row outside that range can match none of them: TPC-H q22's anti join of `customer` to
    the 100M customer keys that have an order reads, per unit, only the keys of its own
    `customer` range, and builds a hash table over those instead of all 100M.
    """
    from batcher.plan.visitor import walk

    if cut.key.keyless:
        return {}
    sliced: dict[int, str] = {}
    for node in walk(body):
        if not isinstance(node, Join) or len(node.left_keys) != 1:
            continue
        for side in _DROPS_UNMATCHED.get(node.join_type, ()):
            b = node.left if side == "left" else node.right
            other = node.right if side == "left" else node.left
            if not isinstance(b, Scan) or b.source_id not in held:
                continue
            b_key = (node.left_keys if side == "left" else node.right_keys)[0]
            other_key = (node.right_keys if side == "left" else node.left_keys)[0]
            found = _aligned_origin(other, other_key, cut.aligned)
            if found is not None and found[1] == cut.key.column_of(found[0]):
                sliced[b.source_id] = b_key
    return sliced


def push_top_n(plan: AlignedPlan) -> AlignedPlan:
    """`plan` with each row-returning cut a residual top-N reads cut to its own top N per unit.

    A cut without an aggregate returns its units' rows concatenated: a disjoint union, so the
    first N of the whole are among the first N of each unit. TPC-H q10 at SF1000 ends in
    `ORDER BY revenue DESC LIMIT 20` over 44.5M per-customer groups, which the residual
    aligned on `custkey` computes whole per unit; every one of them crossed to the driver to
    be sorted there, and the driver (a 32 GB head node) was OOM-killed on the second run.
    Each unit now keeps its own 20. The residual's sort still decides: this only drops rows
    no unit could contribute to its answer.

    Taken only for a placeholder the residual reads once, through plain-column projections,
    under a sort with a limit; the sort keys must be columns of the cut's output.
    """
    from batcher.plan.logical import Sort
    from batcher.plan.visitor import walk

    scans = [n.source_id for n in walk(plan.residual) if isinstance(n, Scan)]
    cuts = list(plan.cuts)
    for node in walk(plan.residual):
        if not (isinstance(node, Sort) and node.limit is not None):
            continue
        found = _placeholder_under(node.input, plan.placeholders)
        if found is None or scans.count(found[0]) != 1:
            continue
        sid, names = found
        i = plan.placeholders.index(sid)
        if cuts[i].aggregate is not None:
            continue
        keys = []
        for key in node.keys:
            if not (isinstance(key.expr, Col) and key.expr.name in names):
                break
            keys.append(dataclasses.replace(key, expr=col(names[key.expr.name])))
        else:
            body = Sort(cuts[i].body, tuple(keys), limit=node.limit)
            cuts[i] = dataclasses.replace(cuts[i], body=body)
    return dataclasses.replace(plan, cuts=tuple(cuts))


def _placeholder_under(node: LogicalPlan, placeholders) -> tuple[int, dict[str, str]] | None:
    """The placeholder scan `node` reads through plain-column projections, and each of
    `node`'s columns as that scan's column."""
    if isinstance(node, Scan):
        if node.source_id not in placeholders:
            return None
        return node.source_id, {c: c for c in node.available_columns()}
    if isinstance(node, Project):
        found = _placeholder_under(node.input, placeholders)
        if found is None:
            return None
        sid, names = found
        mapped = {
            item.alias: names[item.expr.name]
            for item in node.items
            if isinstance(item.expr, Col) and item.expr.name in names
        }
        return sid, mapped
    return None
