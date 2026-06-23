"""Shuffle lineage — how to recompute an output a lost worker produced.

Carbonite's fault tolerance is Spark-style *recompute from lineage*: a shuffle
output isn't replicated, it's regenerated from the deterministic map task that
produced it. `ShuffleLineage` records the coordinate of that work (which source
partition of which stage) and the epoch that distinguishes a fresh recompute from
the stale output a dead worker left behind. The recompute *action* is a caller
thunk — the distributed layer owns the map IR and the workers — so this carries
only what Carbonite needs to coordinate: the identity and the epoch.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ShuffleLineage"]


@dataclass(frozen=True, slots=True)
class ShuffleLineage:
    """The coordinate + epoch of one recomputable map output.

    `stage`/`src_partition` identify the map task; `epoch` increments each time the
    output is regenerated, so a reducer never confuses a recomputed partition with
    the stale one a lost worker had published under the previous epoch.
    """

    stage: int
    src_partition: int
    epoch: int = 0

    def reincarnate(self) -> ShuffleLineage:
        """The lineage for a fresh recompute attempt (the next epoch)."""
        return ShuffleLineage(self.stage, self.src_partition, self.epoch + 1)
