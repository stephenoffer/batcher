"""Resource contracts between Kyber (optimizer), Carbonite (resource manager), and `dist`.

Kyber annotates each physical operator with the resources it expects to need
(`ResourceBounds`); Carbonite validates the plan against the machine or cluster and returns a
`FeasibilityVerdict`, whose counter-offer Kyber can re-plan around — closing the
optimizer/resource loop without either layer importing the other. `SchedulingEnvelope` is the
per-task grant that falls out, and `HardwareProfile` (with the fleet's `ClusterShape`) is the
hardware all of it is sized against.

A façade: every name below is defined in a sibling module and re-exported here, so
`batcher.plan.resource` remains the one import path.
"""

from __future__ import annotations

from batcher.plan.resource.bounds import (
    FeasibilityVerdict,
    ResourceBounds,
    SchedulingEnvelope,
)
from batcher.plan.resource.cluster import ClusterShape, NodeShape
from batcher.plan.resource.hardware import HardwareProfile
from batcher.plan.resource.locality import LocalityShares

__all__ = [
    "ClusterShape",
    "FeasibilityVerdict",
    "HardwareProfile",
    "LocalityShares",
    "NodeShape",
    "ResourceBounds",
    "SchedulingEnvelope",
]
