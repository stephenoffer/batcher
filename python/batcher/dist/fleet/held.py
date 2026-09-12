"""What the session fleet already holds, per node, so sizing does not fight its own fleet.

The fleet planner sizes each worker against the cores that are **free** on its node, which is
right when the busy cores belong to another tenant and wrong when they belong to us. A warm
session fleet holds most of the cluster by design -- that is what makes the next `collect()`
start warm -- so on the second query of a session the planner reads a nearly-full cluster and
cuts a fleet of one-core workers out of it.

Measured on the 28-node / 384-core mixed cluster this was written for, with
`reuse_session_fleet` at its default:

    idle cluster (query 1)            32 slots, 357 cores  (5x24, 3x23, 8x15, 16x3)
    fleet warm, 27 CPU free (query 2) 32 slots,  32 cores  (32x1)

An eleven-fold collapse in planned capacity, caused entirely by the engine counting its own
reservation as somebody else's load. Every consumer of the per-worker grants degrades with it:
the rayon width shipped to each worker (`ray_runtime.lifecycle.engine_config_json`) and the
capacity weighting the split assigner uses (`ray_runtime.capacity.fleet_worker_cpus`).

Attribution is per node rather than a total because the planner's question is per node: a
scalar of "we hold 350 cores" cannot say whether the 96-core machine is free. The placement
group knows exactly, since it is what reserved them -- `placement_group_table` maps each bundle
to the node Ray placed it on, and the bundle specs say what each bundle took.

Best-effort throughout: an unreadable table returns `{}` and every caller then sees the free
cores it saw before this existed, which is the pre-existing behaviour rather than a new
failure mode.
"""

from __future__ import annotations

from batcher._internal.logging import note_suppressed

__all__ = ["session_fleet_held_cores"]


def session_fleet_held_cores() -> dict[str, float]:
    """CPU held per node id by the reusable session fleet, empty when there is none.

    Only the **session** fleet is counted. A query-scoped fleet belongs to a query that is
    still running, so its cores genuinely are unavailable to a concurrent sizing decision;
    the session fleet is the one this query is about to be handed.

    Returns:
        Node id to CPU count, empty when no session fleet is held or the topology is
        unreadable.
    """
    from batcher.dist.fleet import _fleet as session

    with session._SESSION_LOCK:
        fleet = session._SESSION
        pg = getattr(fleet, "pg", None) if fleet is not None else None
        if fleet is None or pg is None or not fleet.actors:
            return {}

    try:
        import ray
        from ray.util.placement_group import placement_group_table

        table = placement_group_table(pg)
        # `bundles` maps bundle index -> its resource dict; `bundles_to_node_id` says where
        # Ray actually placed each one. Both are needed: the bundle carries the grant and
        # only the table knows the machine it landed on.
        specs = table.get("bundles") or {}
        placed = table.get("bundles_to_node_id") or {}
        held: dict[str, float] = {}
        for index, resources in specs.items():
            node = placed.get(index)
            if not node:
                continue
            cpu = float((resources or {}).get("CPU", 0.0) or 0.0)
            if cpu > 0:
                held[node] = held.get(node, 0.0) + cpu
        del ray
        return held
    except Exception as exc:  # pragma: no cover - a sizing hint never fails a query
        note_suppressed("dist", "read the session fleet's held cores", exc)
        return {}
