"""The query-lifetime shuffle fleet and the partitioned intermediate it produces.

`_fleet` holds the `ShuffleFleet` (one placement group + worker fleet reused across an
adaptive query's breaker stages) and the borrow/spawn helpers every Flight operator
uses; `source` holds the `FlightMaterializedSource` a stage leaves partitioned on the
fleet for the next stage to scan in place. Kept as a small package so neither file
grows unbounded and the flight operators import one cohesive home.
"""

from __future__ import annotations

from batcher.dist.fleet._fleet import (
    ShuffleFleet,
    acquire_fleet,
    current_fleet,
    maybe_spawn_query_fleet,
    reset_fleet,
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
    "reset_fleet",
    "set_fleet",
]
