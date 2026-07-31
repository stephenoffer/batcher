"""Resource contracts between Kyber (optimizer) and Carbonite (resource manager).

Kyber annotates each physical operator with the resources it expects to need
(`ResourceBounds`); Carbonite validates the plan against the cluster/machine and
returns a `FeasibilityVerdict`. If infeasible, the verdict carries a counter-offer
that Kyber can re-plan around (e.g. force a spill-friendly join) — closing the
optimizer↔resource loop without either layer importing the other.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "FeasibilityVerdict",
    "HardwareProfile",
    "ResourceBounds",
    "SchedulingEnvelope",
]


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """The hardware Kyber is planning *for* — detected, never assumed.

    Kyber otherwise plans against fixed constants tuned on one machine (a 4 MiB broadcast
    threshold, a 12 GB GPU, `target_rows_per_task` blind to core count), so the same plan is
    produced on a 4-core laptop and a 128-core server and is wrong on both. This is the neutral
    contract that carries the real numbers into the optimizer: the conductor resolves it once —
    from this machine single-node, from the cluster's topology when distributed — and threads it
    through `OptimizerContext`. It lives in `plan` so Kyber can read it and `api`/`dist` can
    populate it without any layer importing another.

    On a **heterogeneous cluster** the fields describe the *binding* node for each resource, not
    an average: `gpu_memory_bytes` is the **smallest** GPU (a working set sized to the largest
    would OOM every other one), and `memory_bytes` the representative worker. Sizing to the
    weakest node is what keeps a plan valid on every node it might land on.

    All fields default to `0` meaning "unknown", so a partial profile (a CPU-only driver that
    cannot see remote GPUs) degrades to the caller's own default rather than to a wrong number.

    * `cpu_cores`         — usable cores per worker (cgroup-quota aware, not host count).
    * `memory_bytes`      — usable RAM per worker (host RAM ∧ cgroup limit).
    * `l3_cache_bytes`    — last-level cache per cache domain; the broadcast-residency bound.
    * `gpu_count`         — GPU **devices** reachable by the plan (`0` on a CPU-only host):
                            this machine's devices single-node, the cluster's device total
                            distributed. Devices, never GPU-bearing *nodes* — it is consumed
                            as a multiplier for the whole-fleet VRAM budget
                            (`one_gpu_bytes * gpu_count`), which counting nodes would
                            under-state eightfold on an 8-GPU box.
    * `gpu_memory_bytes`  — usable VRAM of the *smallest* visible GPU; `0` when unknown.
    * `worker_count`      — workers the plan will run across (`1` single-node); lets Kyber
                            reason about total vs per-node budgets on a cluster.
    * `accelerator_type`  — the *model* of the binding GPU (`"NVIDIA_H100"`), `""` when
                            unknown or when the fleet is mixed. VRAM alone cannot answer
                            "how fast is the host link", "can this device be partitioned",
                            or "what does it draw" — every one of which changes a plan, and
                            none of which is derivable from a byte count. `""` is the
                            pre-existing behavior: every model-specific decision then reports
                            no opinion and the plan is sized exactly as it was before.
    """

    cpu_cores: int = 0
    memory_bytes: int = 0
    l3_cache_bytes: int = 0
    gpu_count: int = 0
    gpu_memory_bytes: int = 0
    worker_count: int = 1
    accelerator_type: str = ""

    @classmethod
    def local(cls) -> HardwareProfile:
        """Detect the profile of *this* machine — the single-node and driver default.

        Reads the neutral hardware layer only, so it is safe to call from anywhere and needs
        no cluster. A distributed run replaces this with a cluster-derived profile via
        [`for_cluster`]; every field a probe cannot answer stays `0` ("unknown").
        """
        from batcher._internal.device_specs import resolve_device_name
        from batcher._internal.hardware import (
            available_cpu_count,
            gpu_inventory,
            l3_cache_bytes,
            machine_memory_bytes,
        )

        gpus = gpu_inventory()
        vram = min((int(g.get("memory_bytes") or 0) for g in gpus), default=0)
        # The device *model*, resolved from whatever the driver called it. Only when every
        # local device is the same model: a mixed host has no single binding model, and
        # naming one of them would attach one device's power and host link to another's plan.
        names = {resolve_device_name(str(g.get("name") or "")) or "" for g in gpus}
        model = names.pop() if len(names) == 1 else ""
        return cls(
            cpu_cores=available_cpu_count(),
            memory_bytes=machine_memory_bytes(),
            l3_cache_bytes=l3_cache_bytes(),
            gpu_count=len(gpus),
            gpu_memory_bytes=vram,
            worker_count=1,
            accelerator_type=model,
        )

    @classmethod
    def for_cluster(
        cls,
        *,
        cpu_cores: int,
        memory_bytes: int,
        worker_count: int,
        gpu_count: int = 0,
        gpu_memory_bytes: int = 0,
        l3_cache_bytes: int = 0,
        accelerator_type: str = "",
    ) -> HardwareProfile:
        """A profile for a distributed run, built by the conductor from live cluster topology.

        The caller passes the *binding* node's figures (smallest GPU VRAM, representative
        worker RAM/cores) so a plan sized against this profile is valid on every node it may
        land on. `l3_cache_bytes` is the binding worker's cache when the caller could probe the
        workers for it (Ray's topology omits cache), and `0` when it couldn't — which keeps a
        cache-sized threshold at its default rather than guessing from the driver's machine.

        `accelerator_type` is the model every GPU node shares, or `""` on a mixed fleet — the
        same rule `cluster_accelerator_type()` follows, because there is no honest single
        answer when the models differ.
        """
        return cls(
            cpu_cores=max(0, cpu_cores),
            memory_bytes=max(0, memory_bytes),
            l3_cache_bytes=max(0, l3_cache_bytes),
            gpu_count=max(0, gpu_count),
            gpu_memory_bytes=max(0, gpu_memory_bytes),
            worker_count=max(1, worker_count),
            accelerator_type=accelerator_type,
        )


@dataclass(frozen=True, slots=True)
class ResourceBounds:
    """R = (M_max, C_max, N_max, CPU) for one physical operator.

    * `m_max_bytes`     — peak memory envelope the operator may use.
    * `c_max_credits`   — max in-flight RecordBatch credits (flow-control bound).
    * `n_max_parallelism` — max concurrent morsels/workers for the operator.
    * `c_cpu_shares`    — CPU shares one task running this operator needs. A
      CPU-heavy breaker (hash/sort) saturates a core (`1.0`); a CPU-light,
      IO/decode-bound streaming op asks for a fraction so more tasks pack per node.
    * `prefers_locality` — whether the operator's shuffle is small enough that
      co-locating its workers (PACK) beats spreading them (SPREAD). Set by Kyber from
      the estimated shuffle volume; consumed by Carbonite to pick a placement strategy
      preference. A pure plan property — the live cluster decides the final strategy.
    """

    m_max_bytes: int
    c_max_credits: int
    n_max_parallelism: int
    c_cpu_shares: float = 1.0
    prefers_locality: bool = False


@dataclass(frozen=True, slots=True)
class FeasibilityVerdict:
    """Carbonite's answer to "can this plan run within these bounds?"."""

    feasible: bool
    binding_constraint: str | None = None  # "memory" | "credits" | "parallelism" | None
    suggested_bounds: ResourceBounds | None = None
    # The operator whose demand binds the constraint, as a plain ``"kind#id"`` string (e.g.
    # ``"Aggregate#3"``), or `None` when nothing was sizable. `binding_constraint` says
    # *which resource* ran out; this says *who ran it out*, which is the actionable half —
    # a user told "this query will spill" can do nothing, while one told the plan's third
    # aggregate is the breaker knows which step to reshape. A string, not an op reference,
    # so the verdict stays a plain data contract that can cross the Ray boundary.
    binding_op: str | None = None
    # Whether this verdict rests on a *guess* rather than a measurement or a proof — set
    # when the operator that binds the constraint carries `Provenance.DEFAULT`. An
    # advisory infeasibility should still steer the plan toward its out-of-core path, but
    # it must never *fail* a query: a plan Kyber could not size may well fit. The
    # conductor honours the routing and suppresses the error.
    advisory: bool = False


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
    * `placement_strategy` — preferred Ray placement-group strategy for the worker
                       fleet (`SPREAD | PACK | STRICT_PACK | STRICT_SPREAD`). A
                       *preference* derived from the plan; the distributed executor
                       resolves it against the live cluster (e.g. downgrades SPREAD to
                       PACK on a tiny cluster where spreading buys nothing).
    * `prefer_cpu_only_nodes` — keep this (relational) fleet off GPU nodes when CPU-only
                       nodes can host it, so a CPU shuffle never steals an inference
                       stage's GPU-node cores. `dist` turns it into a node-label selector
                       against the live topology; a no-op on a homogeneous cluster.
    * `gpu_collective` — the GPU stage's UDF runs its own multi-GPU collective (NCCL/etc.)
                       internally, so `dist` gang-schedules its actors co-located
                       (STRICT_PACK). Batcher never touches a tensor — the Arrow contract
                       at operator boundaries is unchanged; only placement is affected.
    * `inflight_depth` — per-actor submit-ahead depth for a GPU/inference actor pool: how
                       many partitions one actor may have in flight at once. `1` is the
                       one-at-a-time default; `>1` keeps a GPU fed across the
                       dispatch/gather round-trip. Set by the conductor from measured GPU
                       utilization; consumed only by the `dist` actor-pool driver.
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
    # Custom Ray resources requested per task, as an immutable `((name, amount), ...)`.
    #
    # `num_gpus` covers only what Ray reports as the `GPU` resource (NVIDIA, AMD, Intel,
    # MetaX). Every other accelerator is a *custom resource*: `TPU`, `neuron_cores`
    # (Trainium/Inferentia), `HPU` (Gaudi), `NPU`. Without this field they are unreachable,
    # so a TPU stage requests `num_gpus` on a node that has none and pends forever. Kept
    # generic rather than one field per vendor so it equally carries a resource an operator
    # defined on their own on-prem cluster.
    #
    # A tuple, not a dict, because this dataclass is frozen *and hashable* — a dict field
    # would make `hash(envelope)` raise. `dist` converts it back at the Ray boundary.
    resources: tuple[tuple[str, float], ...] = ()
    # Scheduling hints resolved entirely in the `dist` layer (never serialized to the
    # JSON IR / FFI). Defaults preserve today's behavior: SPREAD, no node-class
    # preference, no collective co-location.
    placement_strategy: str = "SPREAD"
    prefer_cpu_only_nodes: bool = False
    gpu_collective: bool = False
    inflight_depth: int = 1
