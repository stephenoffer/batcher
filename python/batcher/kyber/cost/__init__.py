"""Cost model — what will this plan *cost* to run?

Cardinality answers "how many rows"; cost turns rows into a comparable estimate of work
along four axes: **cpu** (compute), **mem** (peak working set, the spill signal), **io**
(bytes scanned/written/spilled), **net** (bytes shuffled across the cluster). Passes that
choose between alternatives — join order, join strategy, whether to spill — pick the
lower-cost plan; WS9 SLA targets reweight the axes into one objective.

The package keeps the four-axis model in one place while separating the questions:

* `model` — `Cost`, `CostModel`, and the per-operator closed forms.
* `terms` — the machine-shaped multipliers those forms fold in (cache residency, spill
  volume, external-merge passes, sort comparison counts).
* `shuffle` — the `net` axis: what a plan costs to move across a cluster, which is zero
  by construction on a single node and dominant at ten thousand of them.

The public surface is unchanged from when this was one module.
"""

from __future__ import annotations

from batcher.kyber.cost.model import Cost, CostCoefficients, CostModel, CostWeights

__all__ = ["Cost", "CostCoefficients", "CostModel", "CostWeights"]
