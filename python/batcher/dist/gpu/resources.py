"""What a GPU task asks Ray for — the fractional half of the relational fan-out.

Every relational GPU task in this package asked for `num_gpus=1`, and the fan-out is built to
make that expensive. `gpu_shard_oversubscribe` deliberately cuts four times as many shards as
there are devices, precisely so each one is small; Ray then runs exactly one of them per device
and queues the rest. A fleet whose own shard count says each piece is a quarter of a device
runs at a quarter of its capacity, and every counter reports full utilization of the thing it
is measuring — because one task really is on each device, doing a quarter of what it could.

This module closes that gap. `shard_task_share` estimates one shard's device working set from
the descriptors the driver already built, hands it to Carbonite's packing decision, and returns
the fraction to request; `gpu_shard_options` turns that into the Ray options the tasks are
submitted with. Nothing here decides *whether* a device is the right place for the work — that
is Kyber's `decide_gpu_backend` — only how much of one a shard should hold.

**Over-packing degrades, it does not fail.** A shard granted a share it turns out not to fit is
caught by the subdivision ladder that already exists (`shards.run_subdivided`): it is divided
and rerun on the device, exactly as an under-estimated shard has always been. That is what
makes packing safe to have on by default. Under-packing, by contrast, has no ladder — the idle
device simply stays idle and nothing reports it.

Every unknown resolves to a whole device, which is the behavior this package had before.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from batcher._internal.device_share import MAX_COTENANTS
from batcher._internal.logging import note_suppressed
from batcher.config import active_config

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.carbonite.accel.fractional import TaskPacking

__all__ = [
    "binding_device_bytes",
    "descriptor_bytes",
    "fleet_derate",
    "gpu_shard_options",
    "largest_shard_bytes",
    "shard_task_share",
    "share_for_bytes",
    "task_device_tenants",
]


def task_device_tenants() -> int:
    """How many tasks this worker's device is packed with, read from Ray's own grant.

    The worker-side inverse of everything above. The driver decides a fraction, Ray enforces it
    as a *scheduling* constraint, and the worker then has to size its **memory** to match —
    because Ray does not. `num_gpus=0.25` means four tasks may run on one board; it does not
    stop any of them allocating the whole of it, and until this existed none of them knew not
    to. Every co-tenant read the device's full capacity, planned an RMM pool against it, and
    with the default `pool_initial_fraction` of 0.5 four of them reserved 200% of the device
    between them. On a single-GPU box, where the fan-out never packs, the figure is always one
    and the bug cannot occur.

    Read from the assigned resources rather than passed down from the driver so the two cannot
    drift: what Ray actually granted this task is the only figure that is true after a retry, a
    rescheduling onto a different node, or a packing the driver revised.

    Returns:
        Co-tenants per device, at least `1`. `1` whenever Ray is not the scheduler, the grant
        cannot be read, or a whole device was granted — every one of which is the unpacked
        behavior this path had before.
    """
    try:
        import ray

        # Guarded rather than merely wrapped: `get_runtime_context()` will *initialize* Ray on a
        # process that has not connected, so an unguarded call turns a question about this
        # task's share into a cluster connection — on the head node, during a single-node run,
        # from inside a memory-sizing path that has no business starting a cluster at all.
        if not ray.is_initialized():
            return 1
        granted = float(ray.get_runtime_context().get_assigned_resources().get("GPU", 0.0))
    except Exception as exc:
        note_suppressed("dist", "read this task's device share", exc)
        return 1
    if granted <= 0.0 or granted >= 1.0:
        return 1
    # Round rather than ceil: Ray reports the fraction back as a float, so a share of one third
    # arrives as 0.33 and `1 / 0.33` is 3.03. Ceiling that would claim a fourth co-tenant that
    # cannot be scheduled and shrink every pool by a quarter for a rounding artifact.
    return max(1, min(MAX_COTENANTS, round(1.0 / granted)))


def descriptor_bytes(descriptor: dict, row_bytes: float) -> int:
    """One shard's estimated input size, in bytes.

    Derived from the footer-captured row count the descriptor already carries — no I/O, and no
    second estimate that could disagree with the one the fan-out was sized against. An in-memory
    descriptor sums its batches' own reported sizes instead, which is a measurement rather than
    an estimate and is preferred whenever it is available.

    Args:
        descriptor: A descriptor from `partition_descriptors`.
        row_bytes: Estimated Arrow width of one row, from `plan.types.schema_row_bytes`.

    Returns:
        Bytes, `0` when neither the row count nor the batches could be read — which every
        caller reads as "unknown" and resolves to a whole device.
    """
    batches = descriptor.get("batches")
    if batches:
        total = 0
        for batch in batches:
            size = getattr(batch, "nbytes", None)
            total += int(size) if size else int(batch.num_rows * max(row_bytes, 1.0))
        return total
    from batcher.dist.executors.partition_io import descriptor_rows

    return int(descriptor_rows(descriptor) * max(row_bytes, 1.0))


def largest_shard_bytes(descriptors: Sequence[dict], row_bytes: float) -> int:
    """The biggest shard's estimated input size — the one the share has to fit.

    The *largest*, not the mean, and the distinction is the whole point. One fraction is chosen
    for the entire fan-out, so sizing it to the average guarantees that the shard which most
    needed room is the one that does not get it. On a skewed key the two figures differ by the
    skew factor, which is exactly the case a fan-out already struggles with.

    Args:
        descriptors: Every shard's descriptor.
        row_bytes: Estimated Arrow width of one row.

    Returns:
        Bytes, `0` for an empty fan-out or one whose sizes could not be read.
    """
    return max((descriptor_bytes(d, row_bytes) for d in descriptors), default=0)


def shard_task_share(
    descriptors: Sequence[dict],
    schema: pa.Schema | None = None,
    *,
    gpu_count: int = 0,
    resident_bytes: float = 0.0,
) -> TaskPacking:
    """How much of a device one shard task should hold.

    Args:
        descriptors: Every shard's descriptor, as the fan-out built them.
        schema: The schema the shards are read with, for the per-row width. `None` — an
            unreadable or unprojected source — declines to pack rather than guessing a width.
        gpu_count: The cluster's live device count, which bounds the reported device demand.
            `0` leaves the demand unbounded, which is only ever a reporting figure.
        resident_bytes: Bytes *each* task holds in addition to its own shard — a broadcast
            join's build side is the case this exists for. Every co-tenant on a device carries
            its own copy, so the replicated side is charged per tenant rather than per device;
            charging it once would pack four tasks around a build side that four of them are
            each about to materialize.

    Returns:
        A `TaskPacking`. `fraction == 1.0` is the unpacked answer and is what every unknown,
        every disabled config, and every shard too large to share resolves to.
    """
    concurrency = max(1, len(descriptors))
    if schema is None:
        from batcher.carbonite.accel.fractional import whole_device_packing

        return whole_device_packing(concurrency, "shard row width unknown; whole devices")

    from batcher.plan.types import schema_row_bytes

    return share_for_bytes(
        largest_shard_bytes(descriptors, schema_row_bytes(schema)),
        concurrency,
        gpu_count=gpu_count,
        resident_bytes=resident_bytes,
    )


def binding_device_bytes() -> float:
    """The memory of the device the packing has to fit — the *cluster's* smallest, not the
    driver's.

    The distinction decides whether this feature does anything at all. A fan-out is scheduled
    from a head node that usually has no GPU, and `DistributedConfig.resolved_gpu_memory_gb`
    probes the **local process**: on that topology it finds no device and returns a 12 GB T4
    constant. Packing an 80 GB H100 fleet against 12 GB grants every shard a whole device it
    needed an eighth of, which is silently the behavior the packing was written to replace. The
    same figure is wrong the other way on a fat-GPU driver scheduling small workers, where it
    would over-commit every one of them.

    The *smallest* device in the fleet, because one fraction is granted to the whole fan-out and
    a share sized against the largest is an over-commitment on every other device it lands on.

    Returns:
        Bytes for one device. Falls back to the configured/local figure when the cluster cannot
        be read — a single-node run, an uninitialized Ray — which is exactly where that figure
        is the right one.
    """
    from batcher.dist.executors.ray_runtime.accelerators import cluster_gpu_memory_gb

    dc = active_config().distributed
    return (cluster_gpu_memory_gb() or dc.resolved_gpu_memory_gb()) * 1e9


def fleet_derate() -> float:
    """How much of the fleet a shard may assume is healthy, as a factor on the co-tenancy.

    Packing multiplies a sick device's blast radius. Four tasks sharing a board clamped to half
    its clocks take four times as long to release the memory they hold, and a fan-out submits
    one set of options for every shard — so the share has to be safe on the *worst* device a
    shard might land on, for exactly the reason it is sized against the smallest device's
    memory rather than the largest.

    Proportionate rather than binary. A large fleet always has a sick device somewhere, and
    disabling packing fleet-wide for one of five hundred would make the feature evaporate on
    the clusters it exists for. One degraded device in eight cuts the co-tenancy by an eighth;
    one in five hundred does not move it.

    Quarantined devices are excluded from both sides: they are not schedulable, so a shard
    cannot land on one, and counting them would penalize a fleet for correctly taking a broken
    board out of rotation.

    Returns:
        A factor in `[1 / MAX_COTENANTS, 1.0]`. `1.0` on a healthy fleet **and** on one that
        could not be probed, which is what keeps a cluster with no health telemetry packing
        exactly as it did. The floor is the reciprocal of the maximum co-tenancy, so the worst
        this can do is stop packing — never refuse to place work.
    """
    from batcher.dist.executors.ray_runtime.hardware_probe import cluster_device_health

    try:
        records = cluster_device_health()
    except Exception as exc:  # a health hint must never fail a fan-out
        note_suppressed("dist", "read the fleet's device health for gpu shard packing", exc)
        return 1.0
    total = sum(int(r.get("devices") or 0) for r in records)
    quarantined = sum(len(r.get("quarantined") or ()) for r in records)
    degraded = sum(len(r.get("degraded") or ()) for r in records)
    schedulable = total - quarantined
    if schedulable <= 0:
        return 1.0
    healthy = max(0, schedulable - degraded)
    return max(1.0 / MAX_COTENANTS, healthy / schedulable)


def share_for_bytes(
    shard_bytes: float,
    concurrency: int,
    *,
    gpu_count: int = 0,
    resident_bytes: float = 0.0,
) -> TaskPacking:
    """The packing for a fan-out whose shard size the caller has already measured.

    `shard_task_share` prices descriptors against one schema, which is the whole answer for a
    chain or a join. A union has several inputs with several schemas and one shared task body,
    so its largest shard has to be priced per input and reduced to a single figure before the
    packing question can even be asked. That figure is what this takes.

    Args:
        shard_bytes: The largest shard's estimated input size, across every input.
        concurrency: How many shard tasks the fan-out submits.
        gpu_count: The cluster's live device count, which bounds the reported device demand.
        resident_bytes: Bytes each task holds beyond its own shard.

    Returns:
        A `TaskPacking` whose `fraction` never exceeds `1.0`. A shard task runs on **one**
        device — cuDF binds to the device Ray gave it and uses no other — so a shard too large
        for a device does not get two, it gets a whole one and then the subdivision ladder.
        Requesting two would hold a second device for the duration and compute on none of it.
    """
    from batcher.carbonite.accel.fractional import plan_task_packing, whole_device_packing

    dc = active_config().distributed
    want = max(1, concurrency)
    if not dc.gpu_pack_shards:
        return whole_device_packing(want, "gpu_pack_shards is off")
    if dc.gpu_task_fraction > 0.0:
        return _pinned_share(min(1.0, dc.gpu_task_fraction), want)
    if 0 < want <= gpu_count:
        # Packing exists to fit *more* shards than devices onto the fleet. With no more shards
        # than devices there is nothing to gain and a fleet to lose: a fractional request lets
        # Ray put several shards on one board, and it does — measured on four T4s, a four-shard
        # fan-out placed three shards on one node and one on another, leaving two devices idle
        # for the whole query. A whole device per shard is what makes Ray spread them.
        return whole_device_packing(want, "one shard per device; nothing to pack")
    if shard_bytes <= 0:
        return whole_device_packing(want, "shard size unknown; whole devices")

    # The device holds the shard's columns *and* whatever the operator derives from them: at the
    # moment a partial aggregate emits its last group, both the input batch it is reading and
    # the hash table it has built are resident. A factor below two would be an assertion that
    # one of the two is free, which is not true of any operator this path runs.
    need = (shard_bytes + max(0.0, resident_bytes)) * max(1.0, dc.gpu_shard_expansion)
    packing = plan_task_packing(
        need,
        device_bytes=binding_device_bytes(),
        # A device can only hold `MAX_COTENANTS` tasks, so a fan-out of a thousand shards over
        # eight devices is still bounded by the fleet rather than by its own shard count — the
        # surplus queues, as it always did, and reporting it as device demand would ask an
        # autoscaler for a cluster the packing never needed.
        concurrency=min(want, max(1, gpu_count) * MAX_COTENANTS) if gpu_count > 0 else want,
        used_bytes=0.0,
        # A shard's options are chosen once for the whole fan-out, so the co-tenancy has to be
        # safe on the worst device a shard might land on — the same argument that sizes it
        # against the smallest device's memory.
        derate=fleet_derate(),
        # A relational shard is not a resident model: it is read, reduced, and released inside
        # one task. Partitioning the device for it would fix the instance count for the whole
        # job at the size of one shard, which is the figure the fan-out most often gets wrong.
        prefer_isolation=False,
    )
    if packing.fraction > 1.0:
        # A shard larger than a device: hold one and let the subdivision ladder make it fit.
        # Holding two would idle the second for the whole task.
        return whole_device_packing(
            want, f"{need / 1e9:.1f}GB exceeds one device; whole device, then subdivide"
        )
    return _capped(packing, dc.gpu_max_tasks_per_device, want)


def _pinned_share(fraction: float, concurrency: int) -> TaskPacking:
    """The packing a configured `gpu_task_fraction` asks for, taken at face value.

    An explicit fraction is an operator statement about a fleet the estimator cannot see — a
    device shared with a service, a part whose memory the driver misreports — so it is applied
    without re-deriving it. It is still routed through the same record so a pinned run and a
    decided one report identically.
    """
    from batcher._internal.device_share import cotenants_per_device, devices_for, share_bytes
    from batcher.carbonite.accel.fractional import TaskPacking

    device_bytes = binding_device_bytes()
    per_device = min(cotenants_per_device(fraction), concurrency)
    return TaskPacking(
        fraction=fraction,
        per_device=max(1, per_device),
        devices=devices_for(fraction, concurrency),
        share_bytes_=share_bytes(device_bytes, fraction),
        reason=f"gpu_task_fraction pinned to {fraction}",
    )


def _capped(packing: TaskPacking, max_per_device: int, concurrency: int) -> TaskPacking:
    """Re-round a packing down to at most `max_per_device` co-tenants.

    The memory arithmetic can allow more tenants than a device should actually run: every one
    of them is a CUDA context, a share of one copy engine, and a process whose allocation spike
    the others feel. The ceiling is a deployment property rather than a derivable one, so it is
    applied here rather than folded into the byte figure — which would make the decision log
    blame the shard size for a limit the operator set.
    """
    from batcher._internal.device_share import balanced_fraction, devices_for
    from batcher.carbonite.accel.fractional import TaskPacking, whole_device_packing

    ceiling = max(1, min(int(max_per_device), MAX_COTENANTS))
    if packing.per_device <= ceiling:
        return packing
    if ceiling == 1:
        return whole_device_packing(concurrency, f"gpu_max_tasks_per_device={max_per_device}")
    fraction = balanced_fraction(ceiling)
    return TaskPacking(
        fraction=fraction,
        per_device=ceiling,
        devices=devices_for(fraction, concurrency),
        share_bytes_=packing.share_bytes_,
        reason=f"{packing.reason}; capped at {ceiling} per device",
    )


def gpu_shard_options(
    descriptors: Sequence[dict],
    schema: pa.Schema | None = None,
    *,
    gpu_count: int = 0,
    resident_bytes: float = 0.0,
) -> tuple[dict, TaskPacking]:
    """Ray remote options for this fan-out's shard tasks, and the packing that produced them.

    The packing comes back alongside the options rather than being folded into them, because a
    caller has to be able to *report* what was decided: a run that packed four shards per device
    and one that ran one per device are the same options dict to Ray and entirely different
    events to whoever is watching the fleet.

    Args:
        descriptors: Every shard's descriptor.
        schema: The schema the shards are read with.
        gpu_count: The cluster's live device count.
        resident_bytes: Bytes each task holds beyond its own shard, such as a broadcast join's
            replicated build side.

    Returns:
        `(options, packing)`. The options are the ones `tasks.gpu_task_options` produces with
        the packed fraction substituted for the whole device it otherwise requests.
    """
    from batcher.dist.gpu.tasks import gpu_task_options

    packing = shard_task_share(
        descriptors, schema, gpu_count=gpu_count, resident_bytes=resident_bytes
    )
    return gpu_task_options(num_gpus=packing.fraction), packing
