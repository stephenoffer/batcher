"""Resource contracts between Kyber (optimizer) and Carbonite (resource manager).

Kyber annotates each physical operator with the resources it expects to need
(`ResourceBounds`); Carbonite validates the plan against the cluster/machine and
returns a `FeasibilityVerdict`. If infeasible, the verdict carries a counter-offer
that Kyber can re-plan around (e.g. force a spill-friendly join) — closing the
optimizer↔resource loop without either layer importing the other.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["FeasibilityVerdict", "ResourceBounds", "SchedulingEnvelope"]


@dataclass(frozen=True, slots=True)
class ResourceBounds:
    """R = (M_max, C_max, N_max) for one physical operator.

    * `m_max_bytes`     — peak memory envelope the operator may use.
    * `c_max_credits`   — max in-flight RecordBatch credits (flow-control bound).
    * `n_max_parallelism` — max concurrent morsels/workers for the operator.
    """

    m_max_bytes: int
    c_max_credits: int
    n_max_parallelism: int


@dataclass(frozen=True, slots=True)
class FeasibilityVerdict:
    """Carbonite's answer to "can this plan run within these bounds?"."""

    feasible: bool
    binding_constraint: str | None = None  # "memory" | "credits" | "parallelism" | None
    suggested_bounds: ResourceBounds | None = None


@dataclass(frozen=True, slots=True)
class SchedulingEnvelope:
    """Per-task scheduling grant Carbonite derives from a plan's `ResourceBounds`.

    The plain-int payload the distributed executor turns into Ray scheduling hints
    (`.options(num_cpus=, memory=, num_gpus=)`) plus the worker/reducer fan-out and
    the shuffle credit window. It lives in the neutral `plan` layer so Kyber and
    Carbonite can both name it and `dist` can receive it without any layer importing
    another — and nothing live (a policy, a pool) ever crosses the Ray boundary.

    * `num_cpus`     — CPU shares requested per task (Ray default is an implicit 1).
    * `memory_bytes` — heap bytes requested per task (a soft Ray scheduling hint).
    * `num_gpus`     — GPUs requested per task; `0.0` for the CPU relational path,
                       `>0` (incl. fractional) for GPU-tagged map/inference tasks.
    * `n_tasks`      — worker/reducer fan-out, derived from estimated rows (replaces
                       a blind `os.cpu_count()`), clamped to the machine's budget.
    * `credits`      — initial shuffle credit window (flow-control bound).
    """

    num_cpus: float = 1.0
    memory_bytes: int = 0
    num_gpus: float = 0.0
    n_tasks: int = 1
    # A conservative default window (matches the engine's `DEFAULT_CREDITS`) so a
    # default-constructed envelope never starts a shuffle at a 1-batch serialized
    # window. The scheduling policy overrides this from `FlowControlConfig`.
    credits: int = 4
    # Optional GPU model to pin tasks/actors to (a `ray.util.accelerators` name such
    # as `"NVIDIA_A100"`); `None` lets Ray pick any GPU. Passed straight to
    # `.options(accelerator_type=...)` for GPU map/inference stages.
    accelerator_type: str | None = None
