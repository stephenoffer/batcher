"""GPU-vs-CPU backend policy — Kyber's cost-based decision of *where* a plan runs.

Choosing GPU vs CPU (and, on GPU, single-device vs sharded across the cluster) is an
optimization decision, so it lives in Kyber, not in the executor: **Kyber decides, Core
executes.** The decision is a pure function of the plan's estimated cardinality and width
(via Kyber's `CardinalityEstimator`), the live GPU count, and the per-GPU memory budget — no
execution and no I/O beyond the footer stats the estimator already reads.

It exists because a GPU is not free: host<->device transfer, kernel launch, and the first-touch
cuDF import are a fixed overhead that a small query never amortizes, and one GPU's memory is
finite. A single `backend="gpu"` flag that ships *any* matching shape to *one* GPU therefore
mispredicts two whole regimes — tiny inputs (CPU wins) and inputs larger than one GPU (a
single-dispatch OOMs). `decide_gpu_backend` covers those: below `gpu_min_rows` → CPU; fits one
GPU → single-dispatch; exceeds one GPU but fits the cluster's GPUs (and is shardable) →
distributed; exceeds them all → the spillable CPU engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from batcher.config import active_config

if TYPE_CHECKING:
    from batcher.io.source import Source
    from batcher.metadata import MetadataHub
    from batcher.plan.logical import LogicalPlan

__all__ = ["GpuDecision", "GpuMapParams", "decide_gpu_backend", "decide_gpu_map_params"]

# The GPU-packing quanta a model's memory fraction is rounded UP to, so Ray can co-locate
# several light inference stages on one GPU (a 3 GB model on a 12 GB GPU → 0.25 → 4 per GPU)
# instead of each wasting a whole device. A model larger than one GPU reserves whole GPUs.
_PACK_QUANTA = (0.25, 0.5, 1.0)


@dataclass(frozen=True, slots=True)
class GpuDecision:
    """Kyber's verdict for one plan: whether to use the GPU, and if so whether to shard it.

    `distributed` is only meaningful when `use_gpu` is True: True means the working set exceeds
    one GPU's memory so the run must fan out across GPUs (the mergeable distributed aggregate);
    False means a single-device dispatch fits. `est_rows` is the estimated input cardinality the
    decision used (`-1` when unknown) — the x-coordinate the adaptive crossover records against.
    `reason` is a short human string for the decision log / `explain()`.

    `desired_gpus` is how many devices would let the working set run in ONE wave — the number
    the autoscaler should be asked for, which is not the number the cluster currently has. A
    fan-out that asks for its current device count pins the floor against scale-down and can
    never scale *up*, so a query that could use thirty-two devices runs on the four it happened
    to find. `0` means the plan does not want devices at all."""

    use_gpu: bool
    distributed: bool
    reason: str
    est_rows: int = -1
    desired_gpus: int = 0


def _estimate(plan: LogicalPlan, sources: list[Source], hub: MetadataHub | None):
    """`(rows, working_set_gb)` for the volume the GPU actually processes, or `(None, None)` when
    the size is unknown (an estimator failure or an unbounded source).

    For a *reducing* operator the plan's OUTPUT cardinality massively understates the work and
    the memory: the GPU reads and reduces the whole INPUT. So the estimate descends past every
    reducing node to the first one whose output is what it processes.

    Descending past a **run** of them, rather than only the top node, is what makes the common
    analytical shape estimable at all: `group_by().agg().sort().limit(10)` has a `Limit` on top,
    whose output cardinality is ten. Estimating that put every such query below the small-input
    threshold and refused it the GPU on the grounds that ten rows do not amortize a kernel
    launch — while the scan underneath it was a billion rows. A map-shaped plan (filter/project)
    already has output ~ processed, so it estimates directly."""
    from batcher.kyber import load_learned_stats
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.plan.logical import Aggregate, Distinct, Limit, Sort

    target = plan
    while isinstance(target, (Aggregate, Distinct, Limit, Sort)):
        below = getattr(target, "input", None)
        if below is None:
            break
        target = below
    try:
        learned = load_learned_stats(hub) if hub is not None else None
        est = CardinalityEstimator(sources=sources, learned=learned)
        rows = int(est.estimate(target).rows)
        width = est.row_width(target, active_config().optimizer.row_bytes)
    except Exception:
        return None, None
    if rows <= 0:
        return 0, 0.0
    return rows, rows * max(width, 1) / 1e9


def decide_gpu_backend(
    plan: LogicalPlan,
    sources: list[Source],
    hub: MetadataHub | None = None,
    *,
    gpu_count: int,
    force: bool = False,
    gpu_memory_gb: float | None = None,
    accelerator_type: str | None = None,
) -> GpuDecision:
    """Decide whether `plan` should run on the GPU, and single-device vs sharded.

    `gpu_count` is the live cluster's GPU count (0 → always CPU). `force=True` (an explicit
    `backend="gpu"`) honors the user past the small-input threshold but still routes by memory —
    so even a forced request avoids a single-dispatch OOM on data larger than one GPU. The
    memory budget per GPU is `gpu_memory_gb` when the caller could read it from the live cluster,
    else `distributed.gpu_memory_gb`; the whole-cluster budget is that times `gpu_count`.

    The distinction matters on a mixed fleet. The config fallback probes the **driver's** devices,
    so a CPU-only head node scheduling eight A100s planned against a 12 GB T4 constant and sharded
    (or refused outright as "exceeds all GPUs") a working set one device would have held."""
    if gpu_count < 1:
        return GpuDecision(False, False, "no GPU on the cluster")

    dc = active_config().distributed
    rows, ws_gb = _estimate(plan, sources, hub)

    # Unknown size: honor a forced request (single-dispatch, with its own CPU fallback on OOM);
    # otherwise stay on the CPU engine rather than gamble the GPU overhead on an unknown input.
    if rows is None:
        return (
            GpuDecision(True, False, "forced; size unknown")
            if force
            else GpuDecision(False, False, "size unknown; GPU overhead not justified")
        )

    # The row threshold below which the GPU overhead isn't amortized: the measured crossover
    # learned from this hub's own GPU/CPU runs when available (Core measures, Kyber consumes),
    # else the config default. This is what makes the backend choice adaptive to the hardware.
    from batcher.kyber.gpu.adaptive import learned_gpu_min_rows

    learned_min = learned_gpu_min_rows(hub, accelerator_type)
    # `is None`, not truthiness: `learned_gpu_min_rows` clamps to `[default/8, default*8]`, so a
    # legitimately-configured small `gpu_min_rows` (the config invites retuning) can learn a 0 —
    # which `or` discarded, silently reverting to the default *and* dropping the "learned "
    # prefix so `explain()` misreported which threshold was actually used.
    min_rows = dc.gpu_min_rows if learned_min is None else learned_min
    learned = "learned " if learned_min is not None else ""
    # ...and the same floor in bytes, because a row count assumes a row width.
    #
    # The threshold above exists because the GPU's fixed overhead — host<->device transfer,
    # kernel launch, the first-touch cuDF import — is only amortized by *work*, and rows
    # proxy for work only while a row is the ~64 bytes `optimizer.row_bytes` assumes. Across
    # the modality range that proxy inverts: at the shipped 10M rows a narrow relation clears
    # the gate at 0.64 GB of input, while a decoded 224x224x3 image column needs **1,505 GB**.
    # So a 100 GB image query — unambiguously GPU-worthy, and the workload this path exists
    # for — was refused the GPU for being "too small to amortize the overhead".
    #
    # The byte figure was already in hand: `_estimate` computes `ws_gb` for the memory
    # routing below and the size gate simply did not read it. Derived from the same knob
    # (`min_rows x row_bytes`) rather than added as a third, and combined with OR, so a
    # narrow query clears exactly the floor it always did.
    min_gb = min_rows * active_config().optimizer.row_bytes / 1e9
    if not force and rows < min_rows and ws_gb < min_gb:
        return GpuDecision(
            False,
            False,
            f"{rows} rows / ~{ws_gb:.2f}GB < {learned}min_rows={min_rows} "
            f"and min ~{min_gb:.2f}GB: CPU wins on overhead",
            rows,
        )

    # Size is necessary but not sufficient. A relational stage's bytes cross the host link
    # before a kernel sees them, and on PCIe that link is slower than a server's own memory:
    # a big enough scan can clear every threshold above and still finish sooner on the CPU.
    # The verdict is only consulted when the device model is known and only ever *refuses* —
    # a forced request is still honored, and an unrecognized device has no opinion.
    if not force and accelerator_type and rows > 0 and ws_gb > 0:
        veto = _transfer_veto(accelerator_type, ws_gb, rows)
        if veto is not None:
            return GpuDecision(False, False, veto, rows)

    one_gpu_gb = max(
        gpu_memory_gb if gpu_memory_gb and gpu_memory_gb > 0 else dc.resolved_gpu_memory_gb(),
        1e-9,
    )
    if ws_gb <= one_gpu_gb:
        return GpuDecision(
            True, False, f"~{ws_gb:.1f}GB fits one GPU ({one_gpu_gb:.0f}GB)", rows, 1
        )
    shardable = _is_shardable(plan)
    # How many devices would hold the working set in one wave, which is what the autoscaler is
    # asked for. Capped so a badly-estimated query cannot ask a cluster to grow without bound.
    wanted = min(math.ceil(ws_gb / one_gpu_gb), max(1, int(dc.gpu_max_autoscale_devices)))
    if ws_gb <= one_gpu_gb * gpu_count and shardable:
        return GpuDecision(
            True,
            True,
            f"~{ws_gb:.1f}GB exceeds one GPU: shard across {gpu_count} GPUs",
            rows,
            wanted,
        )
    # Beyond the cluster's *aggregate* VRAM the question is no longer how much memory the
    # cluster has at once, but how small a shard can be made. A plan with a mergeable reducer
    # oversubscribes shards past the device count and pipelines them, so what has to fit a
    # device is one shard, not the working set — and each shard reduces to one row per group
    # before anything is folded. Refusing those outright meant the fan-out built for exactly
    # this case could never be reached: the rule turned "too big for one pass" into "too big
    # for the GPU at all".
    #
    # A plan with NO mergeable reducer genuinely does need the whole set resident, so it still
    # goes to the (spillable) CPU engine.
    shards = gpu_count * max(1, int(dc.gpu_shard_oversubscribe))
    if shardable and ws_gb / shards <= one_gpu_gb:
        return GpuDecision(
            True,
            True,
            f"~{ws_gb:.1f}GB exceeds all {gpu_count} GPUs, but shards to "
            f"~{ws_gb / shards:.2f}GB across {shards}",
            rows,
            wanted,
        )
    # Not shardable, and larger than one device. A single dispatch is the only accelerated form
    # available and it does not fit, so it would OOM and fall back anyway; the CPU engine spills
    # and is the honest destination. Reported as such rather than attempted and abandoned.
    scope = f"exceeds all {gpu_count} GPUs" if ws_gb > one_gpu_gb * gpu_count else "exceeds one GPU"
    why = "nothing to shard on" if not shardable else "CPU engine (spillable)"
    return GpuDecision(False, False, f"~{ws_gb:.1f}GB {scope}: {why}", rows)


def _is_shardable(plan: LogicalPlan) -> bool:
    """Whether `plan` divides across devices, so its per-device memory is one shard's.

    Two shapes do: one with a mergeable reducer, whose shards fold, and a row-local one, whose
    shards concatenate. Answered from the plan's own IR through the shared algebra in
    `plan.distribution` rather than re-derived here — the optimizer routing a plan to the
    fan-out and the backend building it must agree about which plans divide, and two statements
    of that rule are the one way they could ever disagree.

    Never raises: a plan that cannot be lowered (a `map_batches` UDF) simply is not shardable.
    """
    from batcher._internal.logging import note_suppressed
    from batcher.plan.distribution import flatten_ops, shard_plan

    try:
        ops = flatten_ops(plan.to_ir())
        return ops is not None and shard_plan(ops) is not None
    except Exception as exc:  # pragma: no cover - routing must never break a plan
        note_suppressed("kyber", "test the plan for a mergeable reducer", exc)
        return False


@dataclass(frozen=True, slots=True)
class GpuMapParams:
    """Kyber's resource sizing for one GPU `map_batches` (inference) stage: how much of a GPU to
    reserve per worker and the initial batch size to seed the online throughput controller."""

    num_gpus: float
    batch_size: int | None
    reason: str


#: Floating-point work a relational row costs on average: a few comparisons and an
#: accumulate. Relational operators are not compute-bound by any margin, which is the whole
#: reason the host copy decides the verdict for them and not for inference.
_RELATIONAL_FLOPS_PER_ROW = 4.0


def _transfer_veto(accelerator_type: str, working_set_gb: float, rows: int) -> str | None:
    """A reason to stay on the CPU when the host copy would cost more than the device saves.

    `None` when the device is worth using, when its model is unrecognized, or when the
    arithmetic cannot be formed — so this only ever removes a GPU choice that the transfer
    model says loses, and never adds one.
    """
    from batcher.kyber.gpu.energy import device_energy_advice

    bytes_per_row = working_set_gb * 1e9 / max(1, rows)
    advice = device_energy_advice(
        accelerator_type,
        bytes_per_row=bytes_per_row,
        flops_per_row=_RELATIONAL_FLOPS_PER_ROW,
    )
    if advice.speedup <= 0 or advice.speedup >= 1.0:
        return None
    return (
        f"{accelerator_type} would run this at {advice.speedup:.2f}x the CPU once the host "
        f"copy is charged ({advice.transfer_share:.0%} of device time is transfer): CPU wins"
    )


def _mig_fraction(model_memory_gb: float, accelerator_type: str) -> float | None:
    """The device fraction a MIG instance would give this model, or `None` to use the quanta.

    Preferred over the coarse packing quanta wherever it applies, because it is both finer (a
    seventh of a device rather than a quarter) and stronger: a partition isolates memory and
    faults, while a fractional request only shares a scheduler. `None` whenever partitioning
    does not apply — no device model, the switch off, a device that cannot partition, or a
    model that needs the whole device — and the caller then packs exactly as it did before.
    """
    if not accelerator_type or not active_config().accelerator.prefer_mig:
        return None
    from batcher._internal.hardware.mig import smallest_profile_for

    profile = smallest_profile_for(model_memory_gb, accelerator_type)
    return profile.gpu_fraction if profile is not None else None


def decide_gpu_map_params(
    model_memory_gb: float,
    num_gpus: float,
    batch_size: int | None,
    gpu_memory_gb: float | None = None,
    *,
    assign_num_gpus: bool = True,
    input_row_bytes: float = 0.0,
    accelerator_type: str = "",
) -> GpuMapParams:
    """Size a GPU inference stage from the model's memory footprint vs one GPU's memory.

    A user-set `num_gpus` (>0) or `batch_size` is always honored — this only *fills* what the
    user left unset, and only when `model_memory_gb` is known (the signal that the stage is a GPU
    model, since an unset `num_gpus` defaults to 0/CPU). The GPU-fraction packs light models
    (fraction rounded up to a packing quantum, so several share a device → higher utilization)
    and reserves whole GPUs for a model larger than one. The batch-size seed spends the VRAM left
    after the model on activations at `gpu_activation_bytes_per_row`, clamped — the online
    `ThroughputController` refines it from measured throughput/VRAM, but a memory-aware start
    beats a fixed 256 (too small for a light model, an instant OOM for a heavy one).

    `gpu_memory_gb` is the *cluster's* binding (smallest) device, supplied by the caller when
    the topology could report it. Without it this falls back to
    `DistributedConfig.resolved_gpu_memory_gb()`, which probes the **local** process — and on
    the usual topology (a CPU-only head node scheduling GPU workers) that sees no device and
    returns a 12 GB T4 constant. Packing a fleet of A100s against a T4 wastes ~85% of each
    device; packing a fleet of T4s against a driver's A100 OOMs every worker.

    `assign_num_gpus=False` is the non-GPU accelerator path (a TPU / Trainium / Inferentia stage,
    which carries `num_gpus == 0` plus a custom resource). There `num_gpus` is left untouched — a
    fractional GPU request on such a stage asks for a device the cluster hasn't got, and on a
    GPU-less accelerator fleet that pends forever — while the batch size is still seeded from the
    device's memory (`gpu_memory_gb`, here the accelerator's HBM). Cross-chip packing is the
    user's resource count, not a `num_gpus` fraction, so the seed budgets against one device.

    `input_row_bytes` is the estimated Arrow width of one input row, which the batch seed
    charges alongside the activation prior because both are resident on the device at once.
    `0.0` — the default, and what a caller with no estimator passes — reproduces the previous
    activation-only budget exactly.

    `accelerator_type` is the binding device's model, when the topology could name it. Given
    one, and with `accelerator.prefer_mig` on, the packing fraction comes from the device's
    *own* MIG profiles instead of the coarse quanta: a model that fits a `1g` instance asks for
    a seventh of an H100 rather than a quarter, and gets memory and fault isolation the
    fractional request does not provide. `""` — an unlabelled or mixed fleet — keeps the
    quanta, which is exactly the behavior before this."""
    dc = active_config().distributed
    gpu_gb = max(
        gpu_memory_gb if gpu_memory_gb and gpu_memory_gb > 0 else dc.resolved_gpu_memory_gb(),
        1e-9,
    )
    cap = 0.85  # leave VRAM headroom for activations + fragmentation (the guides' ~80-85%)

    if model_memory_gb <= 0.0:
        return GpuMapParams(num_gpus, batch_size, "model memory unknown; left as given")

    out_gpus = num_gpus
    if assign_num_gpus and num_gpus <= 0.0:  # user left it unset → decide the packing fraction
        frac = model_memory_gb / (gpu_gb * cap)
        if frac <= 1.0:
            # No `next()` default: this branch is guarded by `frac <= 1.0` and `_PACK_QUANTA`
            # ends at 1.0, so a quantum always matches. A default here would disguise that.
            out_gpus = _mig_fraction(model_memory_gb, accelerator_type) or next(
                q for q in _PACK_QUANTA if q >= frac
            )
        else:
            out_gpus = float(math.ceil(frac))

    # The device-memory budget the batch seed spends: the packed GPU fraction, or one whole
    # device for a non-GPU accelerator (its cross-chip packing is the user's resource count).
    budget_devices = out_gpus if assign_num_gpus else 1.0

    out_bs = batch_size
    if batch_size is None:  # seed the throughput controller from the VRAM headroom
        # Budget against this actor's *share*, not the whole device. A packed fraction does
        # see one device, but `_PACK_QUANTA` exists precisely so several actors co-locate on
        # it (0.25 → four per GPU), and each one sizing its activations against the full VRAM
        # means they all claim it at once: at the shipped defaults a 3 GB model packs two per
        # 12 GB device and each seeds 65,536 rows, demanding 2 x (3 + 4.3) = 14.6 GB — a
        # guaranteed OOM at exactly the packing factor the fraction was chosen for. Scaling by
        # `out_gpus` gives 2 x (3 + 2.1) = 10.2 GB, which fits.
        #
        # No fabricated floor either: `gpu_gb * 0.05` invented 5% of a *whole* device (0.6 GB,
        # ~9k rows of activations) for a stage that by construction has no room for it. The
        # `max(..., 1)` clamp below already guarantees a legal batch size, which is all the
        # floor was there for.
        headroom_gb = max(gpu_gb * budget_devices * cap - model_memory_gb, 0.0)
        # A batch occupies the device twice over: the **input rows** themselves, and the
        # activations the forward pass derives from them. Only the second was charged, at a
        # flat `gpu_activation_bytes_per_row` (64 KiB) described as suiting "typical
        # vision/embedding activations" — and the input was treated as free.
        #
        # That is a rounding error on a numeric feature row and the whole budget on the data
        # this rule exists for. A decoded 224x224x3 `uint8` image is 147 KiB per row before a
        # single activation, so the input tensor alone is more than twice what the seed
        # budgets; one 1080p RGB frame is 5.9 MiB, 95x it. The seeded batch then asks the
        # device for several times the VRAM it has, which is an OOM on the first dispatch
        # rather than a slow start the `ThroughputController` could recover from.
        #
        # Charged at the **Arrow** width, which is what Batcher can actually know. A model
        # that upcasts `uint8` pixels to `float32` on device occupies four times this; that is
        # a property of the user's model rather than of the plan, and inventing a multiplier
        # for it would put a fabricated number inside a memory bound. The controller's
        # measured VRAM feedback is what closes the rest.
        act = max(dc.gpu_activation_bytes_per_row, 1)
        per_row = act + max(0.0, input_row_bytes)
        out_bs = int(min(max(headroom_gb * 1e9 / per_row, 1), 65_536))
    return GpuMapParams(
        out_gpus,
        out_bs,
        f"model {model_memory_gb:.1f}GB → num_gpus={out_gpus}, batch_size={out_bs}",
    )
