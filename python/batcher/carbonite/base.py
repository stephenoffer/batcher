"""Policy seams for the Carbonite resource manager.

`ResourceManager` is a thin orchestrator that composes one policy of each kind below.
These `Protocol`s name the four resource decisions it delegates:

- **admission** — is this plan feasible, and if not, what is the counter-offer?
- **flow control** — how many in-flight batch slots may a shuffle channel hold?
- **memory estimation** — what envelope does this plan need?
- **scheduling** — what per-task grant should a distributed worker be given?

An alternate policy drops in by being constructed in place of the default, without
touching the manager's public surface or `api`'s call site. The concrete
implementations live in `carbonite.policies`.

**Spilling is deliberately not a `Protocol` here.** It reads as a fifth policy and is
not one: `policies.spill_advice.SpillAdvisor` is a concrete collaborator, because there
is exactly one implementation and no second in sight, and a `Protocol` with one
implementation is an empty frame around a class. The seam to widen when a second spill
strategy actually arrives is that class — this docstring named a `SpillPolicy` that never
existed, which is the kind of thing a reader trusts and then goes looking for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from batcher.carbonite.memory.learned import LearnedMemoryModel
    from batcher.config import Config
    from batcher.plan.physical import PhysicalPlan
    from batcher.plan.resource import FeasibilityVerdict, ResourceBounds, SchedulingEnvelope

__all__ = [
    "AdmissionPolicy",
    "FlowControlPolicy",
    "MemoryEstimator",
    "ResourceContext",
    "SchedulingPolicy",
]


@dataclass(frozen=True, slots=True)
class ResourceContext:
    """Read-only inputs a policy reads when making a resource decision.

    Carries the engine `Config` and the query's memory envelope. `envelope_bytes`
    is the raw memory a query may draw on (the configured hard cap, else the RAM
    available at query start), sampled **once** by the `ResourceManager` so every
    decision — admission, spill, reserve — reasons about the *same* number rather
    than re-sampling live free RAM at each point (which drifts and made the three
    decisions disagree). `None` for a standalone policy with no manager; the policy
    then samples its own.

    `memory_model` is the learned per-family memory model fit from the hub's measured
    `m_peak_bytes` (see `carbonite.memory.learned`): the read-only handle by which a
    policy blends its plan estimate toward measured reality. `None` when no hub is
    wired (a standalone policy, or api not yet threading one) — every policy then falls
    back to the plan estimate, so sizing is byte-for-byte the plan-only behavior.
    """

    config: Config
    envelope_bytes: int | None = None
    memory_model: LearnedMemoryModel | None = None


class AdmissionPolicy(Protocol):
    """Decides whether a plan may run within available resources."""

    def validate(self, plan: PhysicalPlan, ctx: ResourceContext) -> FeasibilityVerdict:
        """Return a feasibility verdict (optionally a counter-offer) for `plan`."""
        ...


class FlowControlPolicy(Protocol):
    """Grants in-flight batch credits (AIMD flow control / backpressure)."""

    def grant(self, requested: int, ctx: ResourceContext) -> int:
        """Return how many of `requested` credits to grant."""
        ...


class MemoryEstimator(Protocol):
    """Estimates the memory/credit/parallelism envelope a plan needs."""

    def envelope(self, plan: PhysicalPlan, ctx: ResourceContext) -> ResourceBounds:
        """Return the resource bounds the plan is expected to need."""
        ...


class SchedulingPolicy(Protocol):
    """Turns a plan's per-operator bounds into a per-Ray-task scheduling grant.

    Carbonite protects: it clamps Kyber's *desired* parallelism/memory against the
    live machine (cpu budget, available RAM) so a task request can never exceed what
    a node can honor. The data plane reads the resulting `SchedulingEnvelope` as the
    Ray `.options(num_cpus=, memory=, num_gpus=)` hints and the worker fan-out.
    """

    def envelope(
        self,
        plan: PhysicalPlan,
        ctx: ResourceContext,
        *,
        requested_workers: int | None,
        available_bytes: int,
    ) -> SchedulingEnvelope:
        """Return the per-task scheduling grant for `plan`."""
        ...
