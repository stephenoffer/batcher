"""What a gang-scheduled fleet's placement-group bundles select on.

`create_worker_placement` reserves one bundle per worker and the bundles decide where the
fleet lands. Two properties of a node are worth stating there rather than leaving to the
scheduler, and both are stated the same way — a `bundle_label_selector` — which is why they
live together:

* **The availability zone.** A shuffle moves nearly all its bytes worker to worker, and every
  cloud prices and delays those bytes by whether the two workers share a zone. The bundles are
  interchangeable, so a fleet that fits in one zone belongs in one.
* **The capacity market.** A shuffle fleet holds accumulated partial state and the mapped
  output its peers have yet to fetch, none of which a descriptor can rebuild, so a reclamation
  costs the stage rather than a task.

Both belong on the bundles rather than on the actors. An actor pinned to a bundle inherits the
bundle's node, so a selector on the actor could only contradict a decision already made — and
a contradicted actor never schedules, which is a hang rather than a fallback.

Both degrade to `{}`, and every caller reads `{}` as "state nothing". A selector that fires on
missing data moves a fleet for a reason that is not there, and an unsatisfiable one is not
refused: the group is waited on for the whole placement budget and then abandoned.
"""

from __future__ import annotations

from batcher._internal.logging import note_suppressed
from batcher.config import active_config
from batcher.plan.resource import SchedulingEnvelope

__all__ = ["fleet_market_selector", "fleet_zone_selector"]


def fleet_zone_selector(workers: int, env: SchedulingEnvelope | None) -> dict[str, str]:
    """The one-zone bundle label selector for this fleet, or `{}`.

    Gated on `distributed.zone_aware_placement` and on the fleet being one whose traffic
    crosses the zone boundary at all. A GPU collective is excluded: it is already STRICT_PACK
    onto a single node, so it is inside one zone by construction, and adding a selector to it
    could only narrow which node that is.

    Args:
        workers: Bundles in the fleet.
        env: The fleet's scheduling grant, or `None` when there is none.

    Returns:
        A `{label_key: zone}` mapping, or `{}` when no zone should be stated.
    """
    if env is not None and env.gpu_collective:
        return {}
    if not active_config().distributed.zone_aware_placement:
        return {}
    try:
        from batcher.dist.executors.ray_runtime.capacity import Demand, preferred_fleet_zone

        return preferred_fleet_zone(workers, Demand.from_envelope(env, count=workers))
    except Exception as exc:  # pragma: no cover - a cost hint never fails a placement
        note_suppressed("dist", "choose an availability zone for the fleet", exc)
        return {}


def fleet_market_selector(workers: int, env: SchedulingEnvelope | None) -> dict[str, str]:
    """The market-type bundle label selector for this fleet, or `{}`.

    A GPU collective is excluded for the same reason the zone selector excludes it: it is
    already STRICT_PACK onto one node, so narrowing which node that is can only make the gang
    harder to form, for a preference it does not benefit from.

    Args:
        workers: Bundles in the fleet.
        env: The fleet's scheduling grant, whose `capacity_preference` this reads.

    Returns:
        A `{label_key: market_type}` mapping, or `{}` when no market should be stated.
    """
    if env is None or env.gpu_collective:
        return {}
    try:
        from batcher.dist.executors.ray_runtime.fabric.market import capacity_bundle_selector

        return capacity_bundle_selector(
            env.capacity_preference, workers=workers, num_cpus=env.num_cpus
        )
    except Exception as exc:  # pragma: no cover - a placement hint never fails a placement
        note_suppressed("dist", "choose a capacity market for the fleet", exc)
        return {}
