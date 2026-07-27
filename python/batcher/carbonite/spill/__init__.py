"""Carbonite out-of-core spilling: the two-tier scratch store for oversized state.

Groups the pieces of the spill path — what a spilled partition *is* (`SpillHandle` /
`SpillTier`), the streaming `BucketWriter` that produces one, the `TieredSpillStore`
that routes buckets between local disk and object storage, and the `disk` module that
measures the volume underneath. Re-exports only; the logic lives in the sibling modules.
"""

from __future__ import annotations

from batcher.carbonite.spill.handle import SpillHandle, SpillTier
from batcher.carbonite.spill.store import TieredSpillStore
from batcher.carbonite.spill.writer import BucketWriter

__all__ = ["BucketWriter", "SpillHandle", "SpillTier", "TieredSpillStore"]
