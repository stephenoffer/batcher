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

## The second staged form: an aggregate over a held input

The chunked path streams one relation and reads every other one whole, so a plan that scans a
fact table twice holds the second copy. TPC-H q18 at sf1000 is the shape: one `lineitem` scan
streams into the final join, and the other -- 6 billion rows, under the `HAVING sum > 300`
aggregate -- is read into memory before the engine starts, and the query is SIGKILLed on a
247 GiB node. `run_staged_held` runs that aggregate's subtree first, on the chunked path, where
its scan is the largest and therefore the one that streams, and the rest of the plan reads its
result: a few thousand rows where it held billions. It acts only when the held input is large
against the memory the chunked path may hold, and declines when another subtree over the same
source computes the same thing -- TPC-H q15 compares a view's `sum` for equality with a `max` of
the same sums, and two executors summing in different orders disagree in the last bit.

Layer: api (orchestration). It decides nothing Kyber has not: the sideways form acts only on the
plan's `prefer_sideways` verdict, the held form only on measured input sizes, and both decline,
returning `None`, on any shape they cannot prove.
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

__all__ = ["run_partitioned_build", "run_staged_held", "run_staged_sideways"]

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


def run_staged_held(
    sources: list[Source],
    opt: PhysicalPlan,
    input_bytes_of: Callable[[int], int],
    run_stage: Callable[[list[Source], PhysicalPlan], list[pa.RecordBatch] | None],
    held_limit: int,
) -> list[pa.RecordBatch] | None:
    """The plan's result with each oversized held input's aggregate run first, or `None`.

    Args:
        sources: The plan's bound sources.
        opt: The optimized physical plan.
        input_bytes_of: Maps a source id to its projected input bytes; picks the driving scan.
        run_stage: Runs one stage on the chunked path (`chunked.execute_chunked`).
        held_limit: Projected bytes above which a scan the chunks do not drive is staged.

    Returns:
        The result batches, or `None` when no held input qualifies or a stage declined; the
        caller then takes the path it would have taken anyway.
    """
    scanned = _scans(opt.ir)
    if len(set(scanned)) < 2:
        return None
    driving = max(set(scanned), key=input_bytes_of)
    ir = copy.deepcopy(opt.ir)
    staged = list(sources)
    for scan_id in sorted(set(scanned) - {driving}):
        if input_bytes_of(scan_id) <= held_limit:
            continue
        path = _held_subtree(ir, scan_id)
        if path is None or _computed_twice(ir, path, staged):
            continue
        out = run_stage(staged, _stage(opt, copy.deepcopy(_at(ir, path))))
        if not out:
            return None
        from batcher.io.source import InMemorySource

        staged.append(InMemorySource(out, zone_maps=False, ephemeral=True))
        _at(ir, path[:-1])[path[-1]] = {"op": "scan", "source_id": len(staged) - 1}
    if len(staged) == len(sources):
        return None
    return run_stage(staged, _stage(opt, ir))


#: Operators a held input's staged subtree may contain: row-wise, plus the breakers whose
#: output does not grow with their input the way a join's can.
_HELD_OPS = frozenset({"filter", "project", "aggregate", "sort", "limit", "distinct"})


def _held_subtree(ir: dict[str, Any], scan_id: int) -> tuple | None:
    """The path to the topmost join-free subtree over only `scan_id` holding an aggregate."""
    for path, node in _walk(ir):
        if not path or "e" in node or node.get("op") == "scan":
            continue
        # Relations only: a binary expression carries an `op` too (`gt`), but also an `e`.
        nodes = [n for _, n in _walk(node) if "e" not in n]
        if any(n.get("op") not in _HELD_OPS | {"scan"} for n in nodes):
            continue
        if [n["source_id"] for n in nodes if n.get("op") == "scan"] != [scan_id]:
            continue
        if any(n.get("op") in ("aggregate", "distinct") for n in nodes):
            return path
    return None


def _computed_twice(ir: dict[str, Any], path: tuple, sources: list[Source]) -> bool:
    """Whether another subtree computes what `path` does, over a scan of the same source."""
    target = _normalized(_at(ir, path), sources)
    for other_path, node in _walk(ir):
        if "e" in node or other_path[: len(path)] == path or path[: len(other_path)] == other_path:
            continue
        if _normalized(node, sources) == target:
            return True
    return False


def _normalized(node: dict[str, Any], sources: list[Source]) -> str:
    """`node`'s IR with each scan naming its source *object*, not its binding id."""
    import json

    def norm(v):
        if isinstance(v, dict):
            if v.get("op") == "scan":
                return {"op": "scan", "source": id(sources[v["source_id"]])}
            return {k: norm(x) for k, x in v.items()}
        if isinstance(v, list):
            return [norm(x) for x in v]
        return v

    return json.dumps(norm(node), sort_keys=True, default=str)


#: How each decomposable aggregate's per-pass results combine (`run_partitioned_build`).
_COMBINE = {
    "count_star": "sum",
    "count": "sum",
    "sum": "sum",
    "min": "min",
    "max": "max",
    "bool_and": "bool_and",
    "bool_or": "bool_or",
}

#: The most passes a partitioned build takes before declining to the out-of-core route.
_MAX_PASSES = 32


def run_partitioned_build(
    sources: list[Source],
    opt: PhysicalPlan,
    input_bytes_of: Callable[[int], int],
    run_stage: Callable[[list[Source], PhysicalPlan], list[pa.RecordBatch] | None],
    needed: int,
    budget: int,
) -> list[pa.RecordBatch] | None:
    """The plan's result in `P` passes, each holding 1/P of its largest build side, or `None`.

    The chunked path's build sides do not spill, so a plan whose builds outgrow the envelope
    left it for the out-of-core route, which spills every join, nested, and joined TPC-H q9's
    bucket pairs one at a time: 370 s at sf100 on a 32 GiB node, where DuckDB took 25. Here the
    largest build is restricted to the keys with `hash(key) mod P == p`, one pass per `p`, and
    the passes' aggregates are combined by the same algebra the distributed path merges with.

    **Why it is sound.** The partitioned join is inner, or semi with the build as its right
    side, and every operator between it and the plan's top aggregate is a filter, projection,
    inner join, or the probe side of a semi join. So each row reaching the aggregate joined
    exactly one build row, whose key fixes the one pass it belongs to: the passes partition
    the aggregate's input, and an aggregate of sums, counts, minima and maxima over a partition
    combines exactly. Probe rows whose key is in another partition match nothing, and the
    runtime key filter the restricted build implies skips most of them during the decode.

    Args:
        sources: The plan's bound sources.
        opt: The optimized physical plan.
        input_bytes_of: Maps a source id to its projected input bytes; sizes build sides.
        run_stage: Runs one stage on the chunked path.
        needed: The build bytes the engine reported when it gave up.
        budget: The budget it gave up against.

    Returns:
        The result batches, or `None` when the plan's shape does not qualify or a pass still
        does not fit; the caller then takes the route it would have taken anyway.
    """
    found = _top_aggregate(opt.ir)
    if found is None or budget <= 0:
        return None
    agg_path, agg = found
    scanned = _scans(opt.ir)
    driving = max(set(scanned), key=input_bytes_of) if scanned else None
    target = _partition_join(agg["input"], input_bytes_of, driving, _agg_reads(agg))
    if target is None:
        return None
    side_path, keys, side_bytes = target
    passes = _pass_count(needed, budget, side_bytes)
    if passes is None:
        return None
    partials: list[pa.RecordBatch] = []
    for p in range(passes):
        stage = copy.deepcopy(agg)
        parent, side = _at(stage["input"], side_path[:-1]), side_path[-1]
        restrict = _in_part(keys, passes, p)
        parent[side] = {"op": "filter", "input": parent[side], "predicate": restrict}
        out = run_stage(sources, _stage(opt, stage))
        if out is None:
            return None
        partials.extend(b for b in out if b.num_rows)
    if not partials:
        return None
    combine = {
        "op": "aggregate",
        "input": {"op": "scan", "source_id": len(sources)},
        "group_keys": [
            {"expr": {"e": "col", "name": k["alias"]}, "alias": k["alias"]}
            for k in agg["group_keys"]
        ],
        "aggregates": [
            {
                "func": _COMBINE[a["func"]],
                "alias": a["alias"],
                "input": {"e": "col", "name": a["alias"]},
            }
            for a in agg["aggregates"]
        ],
    }
    ir = copy.deepcopy(opt.ir)
    if agg_path:
        _at(ir, agg_path[:-1])[agg_path[-1]] = combine
    else:
        ir = combine
    # The combine reads only the passes' partial groups, a resident relation the chunked path
    # cannot stream (it declines one), so it runs on the resident engine. The plan's other
    # sources are not read above the aggregate and are bound empty.
    from batcher import core

    resident: list[list[pa.RecordBatch]] = [[] for _ in sources]
    return core.execute_local(_stage(opt, ir), [*resident, partials])


def _top_aggregate(ir: dict[str, Any]) -> tuple[tuple, dict[str, Any]] | None:
    """The plan's top aggregate below any sort/limit/project, if every aggregate decomposes."""
    path: tuple = ()
    node = ir
    while node.get("op") in ("sort", "limit", "project", "filter"):
        path, node = (*path, "input"), node["input"]
    if node.get("op") != "aggregate" or not _decomposes(node):
        return None
    return path, node


def _decomposes(agg: dict[str, Any]) -> bool:
    """Whether every aggregate of `agg` combines across partitions of its input (`_COMBINE`)."""
    return all(
        spec.get("func") in _COMBINE and not set(spec) - {"func", "alias", "input"}
        for spec in agg.get("aggregates") or []
    )


def _renames(project: dict[str, Any]) -> bool:
    """Whether `project` only selects columns, under their own names, computing nothing."""
    return all(
        item["expr"].get("e") == "col" and item["expr"].get("name") == item["alias"]
        for item in project.get("exprs") or []
    )


#: Marks a column read by something other than an aggregate's plain input (a key, a predicate,
#: an expression): it must not be a pre-aggregation's partial value. Equals no `_COMBINE` func.
_OPAQUE = ""


def _agg_reads(agg: dict[str, Any]) -> dict[str, str] | None:
    """Column -> how `agg` folds it, or `None` when `agg` must not see a group split in two.

    Counting a pre-aggregation's rows counts its groups, so a count over partial groups is
    wrong however it is combined; anything else read as more than a plain input is opaque.
    """
    reads: dict[str, str] = {}
    for spec in agg.get("aggregates") or []:
        if spec["func"] in ("count", "count_star"):
            return None
        inp = spec.get("input") or {}
        if inp.get("e") == "col":
            reads[inp["name"]] = reads.get(inp["name"], spec["func"])
            if reads[inp["name"]] != spec["func"]:
                reads[inp["name"]] = _OPAQUE
        else:
            reads.update(dict.fromkeys(_ir_columns(inp), _OPAQUE))
    for key in agg.get("group_keys") or []:
        reads.update(dict.fromkeys(_ir_columns(key), _OPAQUE))
    return reads


def _ir_columns(node: Any) -> set[str]:
    """Every column an IR expression (or list or dict of them) names."""
    if isinstance(node, dict):
        found = {node["name"]} if node.get("e") == "col" and "name" in node else set()
        for v in node.values():
            found |= _ir_columns(v)
        return found
    if isinstance(node, list):
        return set().union(*(_ir_columns(v) for v in node)) if node else set()
    return set()


def _opaque(reads: dict[str, str] | None, cols: set[str] | list[str]) -> dict[str, str] | None:
    return None if reads is None else {**reads, **dict.fromkeys(cols, _OPAQUE)}


def _side_reads(
    join: dict[str, Any], side: str, reads: dict[str, str] | None
) -> dict[str, str] | None:
    """`reads` in one join input's own column names: the join's `output` renames, its keys read."""
    if reads is None:
        return None
    if join.get("output") is None:
        return None  # no explicit output mapping to follow: do not cross a pre-aggregation
    mine = {o["alias"]: o["name"] for o in join["output"] if o["side"] == side}
    below = {mine[c]: f for c, f in reads.items() if c in mine}
    return _opaque(below, join.get(f"{side}_keys") or [])


def _partition_join(
    node: dict[str, Any], input_bytes_of, driving: int | None, reads: dict[str, str] | None
) -> tuple[tuple, list[str], int] | None:
    """`(path, keys, scanned bytes)` of the largest build side that may be partitioned, or `None`.

    Reached only through filters, projections, inner joins and semi joins' probe sides (see
    `run_partitioned_build`); a candidate is an inner or semi equi-join whose build side does
    not read the driving scan -- that relation streams, it is not a build.

    It also crosses a pre-aggregation (an eager aggregate Kyber pushed below a join), which a
    pass then emits as *partial* groups: a group whose rows span passes arrives as several
    rows. That is sound only when nothing above it can tell. `reads` is what the path above
    does with each column (`_agg_reads`): every partial value must be folded by the aggregate
    above with the function that combines it -- a sum of partial sums, a minimum of partial
    minima -- and read nowhere else, and only joins, filters and plain selections may lie
    between. `None` means a pre-aggregation may not be crossed at all.
    """
    best: tuple[int, tuple, list[str]] | None = None

    def visit(n: dict[str, Any], path: tuple, reads: dict[str, str] | None) -> None:
        nonlocal best
        op = n.get("op")
        if op == "filter":
            visit(n["input"], (*path, "input"), _opaque(reads, _ir_columns(n.get("predicate"))))
        elif op == "project":
            visit(n["input"], (*path, "input"), reads if _renames(n) else None)
        elif op == "aggregate" and reads is not None and _decomposes(n):
            if all(
                reads.get(spec["alias"], _COMBINE[spec["func"]]) == _COMBINE[spec["func"]]
                for spec in n.get("aggregates") or []
            ):
                visit(n["input"], (*path, "input"), _agg_reads(n))
        elif op == "hash_join" and n.get("join_type") in ("inner", "semi"):
            # The build is the side that does not read the driving scan -- for a semi join only
            # the right side may be restricted, for an inner join either may be the build.
            sides = ("right",) if n["join_type"] == "semi" else ("right", "left")
            for side in sides:
                keys = n.get(f"{side}_keys")
                if not keys or driving in _scans(n[side]):
                    continue
                size = sum(input_bytes_of(i) for i in _scans(n[side]))
                if best is None or size > best[0]:
                    best = (size, (*path, side), list(keys))
            visit(n["left"], (*path, "left"), _side_reads(n, "left", reads))
            if n["join_type"] == "inner":
                visit(n["right"], (*path, "right"), _side_reads(n, "right", reads))

    visit(node, (), reads)
    return None if best is None else (best[1], best[2], best[0])


def _pass_count(needed: int, budget: int, side_bytes: int) -> int | None:
    """How many passes fit the builds under `budget` with one side split, or `None`.

    Only the partitioned side shrinks, so the passes are sized from its share: the other
    builds keep `needed - side` and the side contributes `side / passes`, against eight tenths
    of the budget for headroom. Its scanned bytes stand in for its build bytes, capped at what
    the engine reported. When the other builds alone exceed that, no pass count can fit and
    the plan declines rather than paying for passes that will each be refused.
    """
    side = min(side_bytes, needed)
    room = int(budget * 0.8) - (needed - side)
    if side <= 0 or room <= 0:
        return None
    passes = max(2, -(-side // room))
    return passes if passes <= _MAX_PASSES else None


def _in_part(keys: list[str], parts: int, p: int) -> dict[str, Any]:
    """`((hash(keys) mod parts) + parts) mod parts == p`: the row hash is signed."""

    def lit(v: int) -> dict[str, Any]:
        return {"e": "lit", "value": {"int": v}}

    hashed = {"e": "hash", "inputs": [{"e": "col", "name": k} for k in keys]}
    rem = {"e": "binary", "op": "mod", "left": hashed, "right": lit(parts)}
    shifted = {"e": "binary", "op": "add", "left": rem, "right": lit(parts)}
    part = {"e": "binary", "op": "mod", "left": shifted, "right": lit(parts)}
    return {"e": "binary", "op": "eq", "left": part, "right": lit(p)}


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
