"""Stream a decorrelated subquery's aggregate restricted to the keys its outer query produces.

A correlated subquery decorrelates to `Join(outer, Aggregate(inner, group_keys=[k]))`, and the
aggregate reads the whole inner relation although only the outer side's keys are ever read
back. Both executors that hold their inputs restrict it once the outer side exists
(`bc_interp::join_par::sideways`, and the streaming executor's prepass in `stream::builds`). The
chunked path cannot: it streams one relation, and when that relation is under the *outer* side,
the aggregate is a build side prepared, whole, before a single outer row exists. TPC-H q21 at
sf100 is the shape: `lineitem` is scanned three times, the chunks drive the outer spine, and the
`GROUP BY l_orderkey` over all 600M rows builds 150M groups for the ~7M orders the spine keeps.

So the query runs in two stages, each on the chunked path it would have taken:

1. The join's outer (probe) input, alone. It is a spine over the driving scan, so the chunks
   drive it exactly as before; its result is the only relation this path holds in full.
2. The rest of the plan, with the outer input replaced by that result and the aggregate's scan
   replaced by `Semi(scan, keys(result))`. The aggregate's own relation is now the largest one
   scanned, so it drives: the engine reads it row group by row group, and the semi join's key
   set becomes a runtime filter at that read (`bc_interp::stream::runtime_filter`).

**Why it is sound.** The join discards right rows no left row matches (`inner`, `left`,
`semi`, `anti`; `bc_ir::JoinType`), and the key is traced to the scan only through filters,
column-forwarding projections and an aggregate grouping on it, so a scan row whose key the outer
side lacks can only form a group the join discards whole — the argument
`join_par::sideways` makes, applied to the same plans. The outer input's rows are computed once
and read back unchanged, so the query's other operators see exactly what they saw before.

Layer: api (orchestration). It decides nothing Kyber has not: it acts only on the plan's
`prefer_sideways` verdict, and declines, returning `None`, on any shape it cannot prove.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Callable

    from batcher.io.source import Source
    from batcher.plan.physical import PhysicalPlan

__all__ = ["run_staged_sideways"]

#: Join types whose unmatched right rows are never emitted, so the right side may be restricted.
_RESTRICTABLE = frozenset({"inner", "left", "semi", "anti"})

#: The column the second stage's semi join reads the outer keys from.
_KEY = "__sideways_key"


def run_staged_sideways(
    sources: list[Source],
    opt: PhysicalPlan,
    input_bytes_of: Callable[[int], int],
    run_stage: Callable[[list[Source], PhysicalPlan], list[pa.RecordBatch] | None],
) -> list[pa.RecordBatch] | None:
    """The plan's result computed in the two stages the module describes, or `None`.

    Args:
        sources: The plan's bound sources.
        opt: The optimized physical plan; its `prefer_sideways` verdict is what licenses this.
        input_bytes_of: Maps a source id to its projected input bytes; picks the driving scan.
        run_stage: Runs one stage on the chunked path (`chunked.execute_chunked` without the
            sideways verdict, so a stage never re-enters here).

    Returns:
        The result batches, or `None` when the plan has no join this can restrict, or a stage
        declined; the caller then takes the path it would have taken anyway.
    """
    found = _sideways_join(opt.ir, input_bytes_of)
    if found is None:
        return None
    path, scan_id, column, crossed = found
    join = _at(opt.ir, path)
    outer = run_stage(sources, _stage(opt, join["left"]))
    if not outer:
        return None
    left_key = join["left_keys"][0]
    keys = [b.select([left_key]).rename_columns([_KEY]) for b in outer]
    from batcher.io.source import InMemorySource

    outer_id, keys_id = len(sources), len(sources) + 1
    staged = [
        *sources,
        InMemorySource(outer, zone_maps=False, ephemeral=True),
        InMemorySource(keys, zone_maps=False, ephemeral=True),
    ]
    columns = _scan_columns(sources[scan_id], opt.source_projections.get(scan_id))
    if columns is None:
        return None
    ir = copy.deepcopy(opt.ir)
    rejoined = _at(ir, path)
    rejoined["left"] = {"op": "scan", "source_id": outer_id}
    if not _restrict_scan(rejoined["right"], scan_id, column, keys_id, columns):
        return None
    if not crossed:
        # A semi or anti join straight over the scan (TPC-H q4's `orders SEMI lineitem`). It
        # reads only whether a key exists, so its right side may be replaced by that side's key
        # set, and the key set is an aggregate the chunks can drive: the plan as written was not
        # chunkable at all, because `lineitem` is its build side.
        key = join["right_keys"][0]
        rejoined["right"] = {
            "op": "aggregate",
            "input": rejoined["right"],
            "group_keys": [{"expr": {"e": "col", "name": key}, "alias": key}],
            "aggregates": [],
        }
    return run_stage(staged, _stage(opt, ir))


def _stage(opt: PhysicalPlan, ir: dict[str, Any]) -> PhysicalPlan:
    """`opt` running `ir` instead: the same pushed reads, no sideways verdict, no op budgets.

    The per-operator budgets are keyed by pre-order position in the plan they were priced for,
    so they would be read against the wrong operators of a stage and are dropped rather than
    misapplied.
    """
    return dataclasses.replace(opt, ir=ir, ops=(), prefer_sideways=False)


def _sideways_join(
    ir: dict[str, Any], input_bytes_of: Callable[[int], int]
) -> tuple[tuple[str, ...], int, str, bool] | None:
    """`(path to the join, restricted scan id, key column there, crossed an aggregate)`, or `None`.

    A candidate is a restrictable join on one key whose right input traces the key to a scan
    read once there (crossing an aggregate that groups on it, unless the join is semi or anti),
    whose left input does not read the restricted scan, and where the plan's largest scan is
    either under the left input or the restricted scan itself. Anywhere else, the chunks drive
    a relation this does not restrict, and staging only adds a pass.

    The second case is TPC-H q4 at sf100: `orders SEMI lineitem`, where the semi join built its
    key set from all 304M qualifying `lineitem` rows to answer for 5.7M orders. Staged, the
    orders run first and `lineitem` is read restricted to their keys, still driving the chunks.
    """
    scanned = _scans(ir)
    if not scanned:
        return None
    largest = max(set(scanned), key=input_bytes_of)
    best = None
    for path, node in _walk(ir):
        if node.get("op") != "hash_join" or node.get("join_type") not in _RESTRICTABLE:
            continue
        right_keys = node.get("right_keys") or []
        if len(right_keys) != 1 or len(node.get("left_keys") or []) != 1:
            continue
        traced = _trace(node["right"], right_keys[0], node["join_type"] not in ("semi", "anti"))
        if traced is None:
            continue
        scan_id, column, crossed = traced
        left_scans = _scans(node["left"])
        if scan_id in left_scans or (largest not in left_scans and largest != scan_id):
            continue
        if _scans(node["right"]).count(scan_id) != 1:
            continue
        if best is None or len(path) < len(best[0]):
            best = (path, scan_id, column, crossed)
    return best


def _trace(node: dict[str, Any], key: str, need_aggregate: bool) -> tuple[int, str, bool] | None:
    """The scan `key` comes from, its name there, and whether the trace crossed an aggregate.

    Through row-wise nodes and a group key.
    """
    crossed = False
    while True:
        op = node.get("op")
        if op == "scan":
            return (node["source_id"], key, crossed) if crossed or not need_aggregate else None
        if op == "filter":
            node = node["input"]
        elif op == "project":
            item = next((p for p in node["exprs"] if p["alias"] == key), None)
            if item is None or item["expr"].get("e") != "col":
                return None
            key, node = item["expr"]["name"], node["input"]
        elif op == "aggregate":
            item = next((g for g in node["group_keys"] if g["alias"] == key), None)
            if item is None or item["expr"].get("e") != "col":
                return None
            key, node, crossed = item["expr"]["name"], node["input"], True
        else:
            return None


def _restrict_scan(
    node: dict[str, Any], scan_id: int, column: str, keys_id: int, columns: list[str]
) -> bool:
    """Replace the one `scan_id` scan under `node` with `Semi(scan, keys)`, in place."""
    for path, child in _walk(node):
        if path and child.get("op") == "scan" and child.get("source_id") == scan_id:
            parent = _at(node, path[:-1])
            parent[path[-1]] = {
                "op": "hash_join",
                "left": child,
                "right": {"op": "scan", "source_id": keys_id},
                "left_keys": [column],
                "right_keys": [_KEY],
                "join_type": "semi",
                "output": [{"side": "left", "name": c, "alias": c} for c in columns],
                "strategy": "hash",
            }
            return True
    return False


def _scan_columns(source: Source, projection: list[str] | None) -> list[str] | None:
    """The columns a scan of `source` produces: its pushed projection, else its whole schema."""
    if projection is not None:
        return list(projection)
    try:
        return list(source.schema().names)
    except Exception:
        return None


def _walk(node: dict[str, Any], path: tuple = ()):
    """Every IR node under `node` with its path, pre-order.

    Generic over the node's values rather than a list of known child keys, so that a relational
    child under a key this module does not name (a union's `inputs`, a range join's sides) is
    still visited: a scan missed here could let the restricted scan be read unrestricted on the
    outer side as well, or the largest scan be misjudged. Expressions are dicts too, but carry
    `e` rather than `op`, and are never mistaken for a relation.
    """
    if "op" in node:
        yield path, node
    for key, value in node.items():
        if isinstance(value, dict):
            yield from _walk(value, (*path, key))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    yield from _walk(item, (*path, key, i))


def _at(node: Any, path: tuple) -> dict[str, Any]:
    """The node `path` names under `node`."""
    for key in path:
        node = node[key]
    return node


def _scans(node: dict[str, Any]) -> list[int]:
    """Every scan's source id under `node`, with repeats."""
    return [n["source_id"] for _, n in _walk(node) if n.get("op") == "scan"]
