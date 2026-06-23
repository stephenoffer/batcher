"""The optimizer context — shared analysis threaded through every rule.

Kyber runs an ordered list of phased `Rule`s (see `kyber.rule`); every rule, from
constant folding to the hundredth join rule, receives this context. It carries the
read-only inputs a rule may *consume* (config, bound sources, the MetadataHub, and a
shared cardinality estimator) plus a `notes` bag where a rule records decisions for
explain/telemetry without widening its signature.

Rules never make execution happen and never collect runtime metadata — Core does
that. Rules *consume* metadata (estimates carry provenance) and *decide*. This keeps
the layering that makes the feedback loop work: Core measures, Kyber decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from batcher.config import Config
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.metadata import MetadataHub

if TYPE_CHECKING:
    from batcher.kyber.cost import CostModel

__all__ = ["OptimizerContext"]


@dataclass(slots=True)
class OptimizerContext:
    """Shared state threaded through the rule pipeline.

    Read-only inputs (config, sources, hub, estimator, cost model) plus a `notes`
    bag where rules record decisions for explain/telemetry (e.g. join build-side
    choices) without widening the rule signature. The estimator is shared so
    cardinality is computed against one consistent view of the metadata; the cost
    model (calibrated from measured `op_stats` when available) lets selection/
    join-order rules rank alternatives by cost rather than raw row counts.
    """

    config: Config
    sources: list
    hub: MetadataHub | None
    estimator: CardinalityEstimator
    cost_model: CostModel | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def costs(self) -> CostModel:
        """The cost model for this run, building a default-coefficient one over the
        shared estimator if none was supplied (so cost is always available)."""
        if self.cost_model is None:
            from batcher.kyber.cost import CostModel

            self.cost_model = CostModel(self.estimator, self.config.optimizer.cost_coeffs)
        return self.cost_model
