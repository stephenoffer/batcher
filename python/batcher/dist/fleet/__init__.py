"""The query-lifetime shuffle fleet and the partitioned intermediate it produces.

`_fleet` holds the `ShuffleFleet` (one placement group + worker fleet reused across an
adaptive query's breaker stages) and the borrow/spawn helpers every Flight operator
uses; `plan_id` holds the per-query shuffle fence that keeps concurrent pipelines sharing
one fleet from reading each other's buckets; `source` holds the `FlightMaterializedSource`
a stage leaves partitioned on the fleet for the next stage to scan in place. Kept as a
small package so no file grows unbounded and the flight operators import one cohesive home.
"""

from __future__ import annotations

from batcher.dist.fleet._fleet import (
    ShuffleFleet,
    acquire_fleet,
    current_fleet,
    maybe_spawn_query_fleet,
    release_fleet,
    release_session_fleet,
    reset_fleet,
    session_fleet_lease,
    set_fleet,
)
from batcher.dist.fleet.source import FlightFetchSplit, FlightMaterializedSource

__all__ = [
    "FlightFetchSplit",
    "FlightMaterializedSource",
    "ShuffleFleet",
    "acquire_fleet",
    "current_fleet",
    "maybe_spawn_query_fleet",
    "release_fleet",
    "release_session_fleet",
    "reset_fleet",
    "session_fleet_lease",
    "set_fleet",
]
