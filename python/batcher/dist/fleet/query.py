"""The query-lifetime fleet: one placement group held for the whole adaptive query.

Split from `_fleet` on the seam between the two fleets it was holding at once. `_fleet` owns
the *session* fleet — the warm cache reused across separate `collect()` calls — and the ambient
handle every Flight operator borrows through. This module owns the other one: the fleet the
adaptive loop spawns for a single staged query, so an intermediate can stay partitioned on the
workers between stages instead of collecting through the driver.

The public import path is unchanged; `batcher.dist.fleet` re-exports this as it always did.
"""

from __future__ import annotations

from batcher._internal.hardware import available_cpu_count
from batcher.dist.fleet._fleet import ShuffleFleet, release_session_fleet
from batcher.dist.fleet.plan_id import active_query_scopes

__all__ = ["maybe_spawn_query_fleet"]


def maybe_spawn_query_fleet(num_workers: int | None, transport: str) -> ShuffleFleet | None:
    """Spawn a query-lifetime fleet when the adaptive Flight path warrants one.

    Returns a `ShuffleFleet` only when `distributed.persistent_fleet` is enabled and
    the resolved transport is Flight on a genuine multi-worker cluster; otherwise
    `None`, so the caller leaves each operator to spawn its own fleet (the default,
    bit-identical path). The worker count is fixed here for the whole query so every
    stage shuffles over the same fleet.
    """
    from batcher.config import active_config

    if not active_config().distributed.persistent_fleet:
        return None

    import math

    from batcher.dist.executors.ray_runtime import (
        _ensure_ray,
        clamp_workers,
        current_envelope,
        engine_config_json,
        release_autoscale,
        request_autoscale,
        resolve_transport,
    )

    workers = num_workers or available_cpu_count()
    _ensure_ray(workers)
    # Size the ask and the clamp against the grant the fleet's actors will actually
    # request. `_spawn_fleet_with_addrs` builds both its placement-group bundles and its
    # actor options from the ambient envelope, so a fleet sized as though each worker
    # needed a single core asks the autoscaler for a fraction of the cores it will demand
    # and clamps to a fan-out the cluster cannot place — the placement group then goes
    # unsatisfiable and the fleet degrades to the one or two actors that fit, which is the
    # 0.6 s -> 16 s collapse the comment below describes from the other direction. With no
    # envelope this is `1.0` and `0`, exactly the previous behavior.
    env = current_envelope()
    per_cpu = env.num_cpus if env is not None else 1.0
    per_mem = int(env.memory_bytes) if env is not None else 0
    # Ask the autoscaler for the fleet's cores and wait (bounded) for them while sizing
    # it; release the request once sized — the spawned actors keep the nodes busy, so the
    # autoscaler never reclaims them under the fleet, and the floor needn't stay pinned.
    request_autoscale(math.ceil(workers * per_cpu))
    try:
        workers = clamp_workers(workers, per_cpu, memory_bytes=per_mem)
        if workers <= 1 or resolve_transport(transport, workers) != "flight":
            return None
        from batcher.dist.flight_aggregate import _shuffle_credits

        # A fleet reserves the cluster's whole CPU capacity, so two cannot both be placed,
        # and the release below is a NO-OP while another pipeline holds the warm fleet —
        # spawning anyway contends with the reservation that pipeline is running on and
        # degrades to the 1-2 workers it can place (measured: a join 0.6 s -> 16 s). Share
        # instead; `None` is the documented borrow path, and this query's lease keeps the
        # fleet alive across its stages. The test is the query count, not whether the
        # release took: the caller holds this query's own lease already, so the release
        # always no-ops and cannot see a concurrent holder. >1 means someone else does.
        release_session_fleet()
        if active_query_scopes() > 1:
            return None
        return ShuffleFleet.spawn(workers, _shuffle_credits(), engine_config_json())
    finally:
        release_autoscale()
