"""Evaluate the parts of an aligned cut that read no aligned source, once, before its units.

A cut's body often hangs dimensions off its facts -- a filtered `customer`, `nation JOIN
region` -- that are the same on every unit. `hoist_broadcasts` finds each maximal such
subtree, evaluates it (`evaluate_broadcast`: on the driver, or aligned itself when it is too
large for that), and puts a scan of its result in its place, so every node fetches one
object per dimension rather than every unit re-reading it. A result past `BROADCAST_BYTES`
declines the cut and names the sources it read, so the plan can be chosen again with that
join left to the residual.
"""

from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa

from batcher._internal.logging import get_logger
from batcher.config.env import env_int
from batcher.dist.executors.aligned.analysis import AlignedCut
from batcher.dist.executors.aligned.local import on_node_schema
from batcher.dist.executors.aligned.units import projected_bytes
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan, Scan
from batcher.plan.schema import SchemaRef

__all__ = ["BROADCAST_BYTES", "DRIVER_READ_BYTES", "evaluate_broadcast", "hoist_broadcasts"]

#: Bytes a broadcast result may reach: every worker holds it and every unit joins over it.
BROADCAST_BYTES = env_int("BATCHER_ALIGNED_BROADCAST_BYTES", 3 << 30, floor=1 << 20)
# Decoded input bytes past which a broadcast subtree is read by the fleet, not the driver.
_NESTED_BROADCAST_BYTES = env_int("BATCHER_ALIGNED_NESTED_BYTES", 1 << 30, floor=1 << 20)
#: Decoded bytes the driver may read for a broadcast subtree or a residual source.
DRIVER_READ_BYTES = env_int("BATCHER_ALIGNED_DRIVER_READ_BYTES", 16 << 30, floor=1 << 20)


def evaluate_broadcast(
    node: LogicalPlan, scanned: set[int], sources: list[Source], workers: int
) -> pa.Table | None:
    """The result of one aligned-free subtree, computed on the driver, or None to decline.

    On the driver, over only the sources it scans: the dimension chains a star query hangs
    off its facts (`customer JOIN nation JOIN region`) are multi-join shapes the one-shot
    dispatcher refuses, and they are small once filtered. Not through the distributed
    dispatcher even for a large read, because that spawns the shuffle fleet, which holds
    nearly every core of every node, and the units then queued behind it: TPC-H q7 at
    SF1000 ran its 77 units in 48 s of wall time at 30% CPU for ~17 s of work.
    """
    from batcher.dist.executors.aligned.run import run_plan
    from batcher.dist.executors.partition_io import source_pushdown
    from batcher.dist.executors.ray_runtime import _single_node
    from batcher.plan.visitor import transform_up

    inputs = sum(projected_bytes(sources[s], source_pushdown(node, s)[0]) or 0 for s in scanned)
    if inputs > _NESTED_BROADCAST_BYTES:
        # A subtree too large for the driver may be aligned itself, on another key: TPC-H q9
        # joins its facts on `orderkey` and a green-part filter of `part JOIN partsupp` on
        # `partkey`, 800M `partsupp` rows reduced to ~40M. Run it the same way, keyless of
        # the outer cut, and broadcast what it returns. One that aligns on nothing is still
        # read by the fleet, split by file (`scan_plan`).
        from batcher.dist.executors.aligned.route import choose_plan, scan_plan

        inner = choose_plan(node, sources, strict=False, single_table=True) or scan_plan(
            node, sources
        )
        if inner is not None:
            found = run_plan(inner, sources, workers)
            if found is not None:
                return found
    if inputs > DRIVER_READ_BYTES:
        return None
    # Only the sources the subtree scans, renumbered from zero: the single-node runner
    # materializes every source it is handed.
    order = sorted(scanned)

    def renumber(n: LogicalPlan) -> LogicalPlan:
        if isinstance(n, Scan):
            return dataclasses.replace(n, source_id=order.index(n.source_id))
        return n

    return _single_node(transform_up(node, renumber), [sources[s] for s in order])


def hoist_broadcasts(
    node: LogicalPlan,
    cut: AlignedCut,
    sources: list[Source],
    held: dict[int, pa.Table],
    workers: int,
    local: dict[int, LogicalPlan],
    oversized: set[int] | None = None,
) -> LogicalPlan | None:
    """`node` with each maximal aligned-free subtree replaced by a scan of its result.

    Results are added to `held` under fresh source ids past the query's own. A large
    subtree that nothing filters is instead left for each node to evaluate (`local.py`),
    added to `local` under its id. Returns None when a result outgrows `BROADCAST_BYTES`,
    which declines the whole cut; the query sources that result read are added to
    `oversized`, so the plan can be chosen again without broadcasting them.

    The subtrees are independent of one another, so they are evaluated concurrently: TPC-H
    q8 at SF1000 has three (`customer` by region, `part` by type, `supplier` by nation),
    and in sequence their reads added up before any unit could start.
    """
    from batcher.plan.visitor import children, scanned_source_ids, with_children

    found: list[tuple[LogicalPlan, set[int]]] = []

    def collect(n: LogicalPlan) -> None:
        scanned = scanned_source_ids(n)
        if scanned and not (scanned & cut.aligned):
            # A plan may reference one subtree object twice (TPC-H q21's `supplier JOIN
            # nation` under both EXISTS arms). `swap` keys on identity, so hoisting it twice
            # would leave one placeholder id that no scan reads.
            if all(n is not f for f, _ in found):
                found.append((n, scanned))
            return
        for child in children(n):
            collect(child)

    collect(node)
    if not found:
        return node
    on_nodes = [on_node_schema(n, scanned, sources, BROADCAST_BYTES) for n, scanned in found]
    with ThreadPoolExecutor(max_workers=len(found)) as pool:
        tables = list(
            pool.map(
                lambda f: f[2] or evaluate_broadcast(f[0], f[1], sources, workers),
                [(*f, schema) for f, schema in zip(found, on_nodes, strict=True)],
            )
        )
    too_big = [
        scanned
        for (_, scanned), t in zip(found, tables, strict=True)
        if isinstance(t, pa.Table) and t.nbytes > BROADCAST_BYTES
    ]
    if oversized is not None:
        oversized.update(sid for scanned in too_big for sid in scanned if sid < len(sources))
    if too_big or any(t is None for t in tables):
        get_logger("dist").info(
            "aligned: a broadcast is unreadable or past %d GiB (%s); declined",
            BROADCAST_BYTES >> 30,
            ", ".join("none" if t is None else f"{t.nbytes / 2**30:.1f} GiB" for t in tables),
        )
        return None
    scans: dict[int, LogicalPlan] = {}
    for (subtree, _), table in zip(found, tables, strict=True):
        sid = len(sources) + len(held) + len(local)
        if isinstance(table, pa.Schema):
            local[sid] = subtree
        else:
            held[sid] = table
        schema = table if isinstance(table, pa.Schema) else table.schema
        scans[id(subtree)] = Scan(sid, SchemaRef.from_arrow(schema))

    def swap(n: LogicalPlan) -> LogicalPlan:
        if id(n) in scans:
            return scans[id(n)]
        kids = children(n)
        return with_children(n, [swap(k) for k in kids]) if kids else n

    return swap(node)
