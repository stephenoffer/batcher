"""Energy-aware accelerator choices — which device, how many, and is it worth the watts.

`policy` answers "GPU or CPU" on time. In a GPU datacenter that is half the question, because
the binding constraint on a full hall is not slots or seconds but power: a rack's busway caps
what its devices may draw, and a stage that finishes 20% faster while drawing twice the power
has made the fleet slower, not faster, for everyone queued behind it.

Three decisions live here, all of them Kyber's (they choose; they never execute):

* **Which device class.** On a mixed fleet, the smallest device that fits wastes the least
  VRAM — the rule `recommend_accelerator_type` already implements — but the *most efficient*
  device that fits wastes the least power, and those are different devices. Which one to
  prefer is a configured policy, not a constant.
* **How many devices.** A power envelope bounds fan-out independently of how many devices are
  idle. Exceeding it does not fail; it clamps every device in the zone, which reads as the
  whole rack mysteriously slowing down.
* **Whether a device is worth it at all.** A scan-shaped stage is bandwidth-bound, so it gains
  a factor of the memory-bandwidth ratio and pays a factor of the power ratio. Below the
  device's roofline ridge those two can cancel, and the honest answer is to stay on the CPU.

Every function reports "no opinion" (`None`, or `-1` for a count) when the inputs are unknown,
so an unrecognized device or an unconfigured envelope leaves the existing decision untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed
from batcher.config import active_config
from batcher.metadata.hardware_scope import scoped

if TYPE_CHECKING:
    from batcher.metadata import MetadataHub

__all__ = [
    "EnergyAdvice",
    "device_energy_advice",
    "learned_work_per_joule",
    "power_bounded_devices",
    "record_measured_efficiency",
    "select_device_class",
    "stage_joules",
]

#: Hub namespace for measured device efficiency, and the sample floor below which a bucket is
#: not trusted. Efficiency swings with the *workload* as much as the device — a starved stage
#: measures badly on the fastest part — so a handful of runs is not evidence about the hardware.
_EFFICIENCY_NS = "gpu_work_per_joule"
_MIN_SAMPLES = 8


def select_device_class(
    candidates: list[str] | tuple[str, ...],
    model_gib: float,
    *,
    prefer_efficiency: bool | None = None,
    headroom: float = 0.15,
    hub: MetadataHub | None = None,
) -> str | None:
    """Choose the device class a stage should be pinned to, or `None` to leave it unpinned.

    Two orderings over the devices that fit, and the choice between them is a real trade:
    smallest-that-fits leaves the large devices free for work that needs them, which minimizes
    queueing; most-efficient-that-fits minimizes watts per unit of work, which is what matters
    when the fleet is power-bound rather than slot-bound.

    Args:
        candidates: Device model names available on the fleet.
        model_gib: The stage's resident footprint per worker.
        prefer_efficiency: Order by throughput per watt rather than by size. `None` reads
            `accelerator.efficiency_first_placement` from the active config.
        headroom: Fraction of a device's memory left free when deciding what fits.
        hub: The metadata hub, consulted for *measured* efficiency when ordering by it. A
            device this fleet has actually run beats one the datasheet merely rates highly,
            because the datasheet ratio is peak-against-peak and a real stage rarely is. Only
            used when every fitting candidate has been measured — a partial ordering would
            rank the measured against the unmeasured, which compares two different things.

    Returns:
        A device model name to pin to, or `None` when nothing fits, nothing is known, or every
        candidate fits (in which case a pin would only constrain placement).
    """
    from batcher._internal.device_specs import device_spec, rank_devices_by_efficiency

    if model_gib <= 0 or not candidates:
        return None
    need = model_gib / (1.0 - min(0.9, max(0.0, headroom)))
    sized = [(name, device_spec(name)) for name in candidates]
    known = {name: spec.memory_gib for name, spec in sized if spec is not None}
    if len(known) < 2:
        return None  # homogeneous or unknowable: nothing to choose between
    fitting = {name: gib for name, gib in known.items() if gib >= need}
    if not fitting or len(fitting) == len(known):
        return None  # nothing fits (shard instead), or everything fits (do not constrain)
    if prefer_efficiency is None:
        prefer_efficiency = active_config().accelerator.efficiency_first_placement
    if prefer_efficiency:
        measured = {n: learned_work_per_joule(hub, n) for n in fitting}
        if all(v is not None for v in measured.values()) and measured:
            return max(measured, key=lambda n: (measured[n], n))
        # The one efficiency ordering, rather than a second sort that could disagree with it.
        # Devices with no published power figure are unrankable and drop out; if that leaves
        # nothing, the size ordering below is the honest fallback.
        ranked = rank_devices_by_efficiency(sorted(fitting))
        if ranked:
            return ranked[0]
    return min(fitting, key=lambda n: (fitting[n], n))


def power_bounded_devices(
    requested: int,
    accelerator_type: str | None,
    *,
    utilization: float = 1.0,
) -> int:
    """Clamp a requested device count to what the configured power envelope allows.

    Args:
        requested: Devices the sizing path asked for.
        accelerator_type: Device model those devices are.
        utilization: Utilization the stage is expected to drive them at.

    Returns:
        The device count to use. Equal to `requested` when no budget is configured or the
        device model is unrecognized, and at least 1 otherwise: a budget too small for a
        single device is a misconfiguration to surface at admission, not a silent zero-device
        plan here.
    """
    from batcher.plan.energy.power import configured_power_envelope

    # The same clamp Carbonite admits against, computed once in the neutral layer: the two
    # subsystems cannot import each other, and a second copy of this arithmetic is how a plan
    # comes to be sized for one fan-out and granted another.
    return configured_power_envelope().clamp_devices(requested, accelerator_type, utilization)


def stage_joules(
    seconds: float,
    accelerator_type: str | None,
    device_count: int,
    utilization: float = 1.0,
) -> float:
    """Energy a stage is expected to draw, in joules.

    Args:
        seconds: Expected wall-clock duration the devices are held.
        accelerator_type: Device model.
        device_count: Devices held.
        utilization: Expected mean utilization.

    Returns:
        Joules including each device's host share, `0.0` when the device is unrecognized.
    """
    from batcher.plan.energy.power import device_power_watts, energy_joules

    watts = device_power_watts(accelerator_type, utilization, include_host=True)
    return energy_joules(watts * max(0, device_count), seconds)


@dataclass(frozen=True, slots=True)
class EnergyAdvice:
    """Whether moving a stage to a device is worth its power, and why.

    Attributes:
        worth_it: True when the device is expected to use less energy for the same work.
        speedup: Expected throughput ratio against the CPU path, `0.0` when unknown.
        power_ratio: Device draw against the CPU path's draw, `0.0` when unknown.
        energy_ratio: Expected device energy against CPU energy for the same work; below
            `1.0` means the device is the cheaper machine to run. The CPU side of that ratio
            is a *constant* — one server's draw and vector throughput — not a measurement of
            the machine in hand, so treat this as a comparison between candidate devices
            rather than as a figure to bill against. The measured counterpart is
            `learned_work_per_joule`, and it takes precedence wherever it exists.
        transfer_share: Fraction of the device's time spent moving bytes across the host link
            rather than computing. Above roughly a half the stage is a copy with a kernel
            attached, and no faster device fixes it — the fix is to keep the data resident or
            to leave the stage on the CPU.
        reason: One line for the decision log.
    """

    worth_it: bool
    speedup: float = 0.0
    power_ratio: float = 0.0
    energy_ratio: float = 0.0
    transfer_share: float = 0.0
    reason: str = ""


#: Draw of the CPU path a stage would otherwise run on, in watts: one server's worth of
#: sockets and memory at load. A rough figure, and it only ever appears as a *ratio* against a
#: device's draw, so the comparison is far less sensitive to it than an absolute joule count.
_CPU_PATH_WATTS = 400.0

#: Peak dense half-precision throughput of that same CPU path, in TFLOP/s — roughly what a
#: two-socket server reaches with AVX-512. Used only as the denominator of a ratio.
_CPU_PATH_TFLOPS = 2.0

#: Default fraction of the input a stage returns, used to charge the result's trip back across
#: the host link. 0.1 describes the *reducing* shapes this decision actually routes — group-by
#: aggregates, distinct, top-N — where the output is a group count rather than a row count.
#: It is wrong by an order of magnitude for a projection, which returns what it reads, so a
#: caller with a non-reducing shape passes its own figure rather than inheriting this one.
_RESULT_FRACTION = 0.1


def device_energy_advice(
    accelerator_type: str | None,
    *,
    bytes_per_row: float,
    flops_per_row: float,
    cpu_gbps: float = 20.0,
    achieved_fraction: float = 1.0,
    resident: bool = False,
    precision: str = "half",
    link_efficiency: float = 1.0,
    result_fraction: float = _RESULT_FRACTION,
) -> EnergyAdvice:
    """Judge a stage's device move on time *and* energy, with the host copy charged for.

    Three terms, per row, so the verdict is independent of how many rows there are:

    * **the copy** — every byte crosses the host link before a kernel sees it, and on PCIe that
      link is slower than a server's own memory bandwidth. This is the term a data engine
      forgets and then cannot explain, and it is why a device wins on inference and loses on a
      projection. A coherent CPU-GPU package (`nvlink-c2c`) moves it by an order of magnitude,
      which changes the answer rather than shading it;
    * **the kernel** — the roofline: bytes over device bandwidth against FLOPs over device
      throughput, whichever binds;
    * **the return** — the result's trip back, charged at a fraction of the input because the
      relational shapes worth offloading reduce.

    The CPU path is scored on the same roofline, against its own memory bandwidth and its own
    vector throughput, and it pays no copy. Comparing those totals is what makes "a scan is not
    worth a GPU" and "decode is" fall out of one calculation instead of two heuristics.

    Args:
        accelerator_type: The candidate device model.
        bytes_per_row: Bytes of input the stage reads per row.
        flops_per_row: Floating-point work the stage does per row.
        cpu_gbps: Effective memory bandwidth of the CPU path, in GB/s.
        achieved_fraction: Fraction of nameplate a real kernel reaches, in (0, 1]. The device
            figures are peak; a caller with a measured number passes it and gets an honest
            verdict instead of an optimistic one.
        resident: The data is already in device memory — a stage fed by another GPU stage, or
            a model already loaded. Skips the copy, which is usually the whole argument.
        precision: `"half"` or `"fp8"`. A quantized stage runs on the FP8 unit, which is twice
            the rate on the generations that have one and absent on those that do not — so
            asking for it where it does not exist falls back to the half-precision figure
            rather than inventing a rate.
        result_fraction: Bytes returned per byte read. The default describes the reducing
            shapes this decision routes; a projection returns roughly what it reads and should
            pass `1.0`, which doubles the copy and moves the verdict accordingly.
        link_efficiency: Fraction of its nameplate host link the device's *measured* PCIe link
            actually negotiated, in (0, 1]. A card that came up at x8 gen4 against a x16 gen5
            slot passes `0.25`, and the copy term grows fourfold — which is the honest figure
            and is exactly the shape of node where a device decision goes wrong unnoticed.
            `1.0`, the default, is the nameplate assumption every caller made before.

    Returns:
        The advice. With an unrecognized device every ratio is `0.0` and `worth_it` is True,
        preserving whatever decision the caller would have made without an energy opinion.
    """
    from batcher._internal.device_specs import (
        device_fp8_tflops,
        device_half_tflops,
        device_memory_bandwidth_gbps,
        device_tdp_watts,
        host_transfer_seconds,
    )

    bandwidth = device_memory_bandwidth_gbps(accelerator_type)
    watts = device_tdp_watts(accelerator_type)
    if bandwidth <= 0 or watts <= 0 or bytes_per_row <= 0:
        return EnergyAdvice(worth_it=True, reason="device unknown: no energy opinion")

    reach = min(1.0, max(1e-3, achieved_fraction))
    # The CPU path is a roofline too. Charging it only memory bandwidth made a
    # compute-heavy row look free on the CPU and cost the device its whole advantage —
    # the verdict said an inference stage was not worth a GPU, which is exactly backwards.
    cpu_seconds = max(
        bytes_per_row / (max(1.0, cpu_gbps) * 1e9),
        flops_per_row / (_CPU_PATH_TFLOPS * 1e12),
    )

    memory_seconds = bytes_per_row / (bandwidth * 1e9 * reach)
    tflops = device_half_tflops(accelerator_type)
    if precision == "fp8":
        tflops = device_fp8_tflops(accelerator_type) or tflops
    compute_seconds = flops_per_row / (tflops * 1e12 * reach) if tflops > 0 else 0.0
    kernel_seconds = max(memory_seconds, compute_seconds)
    shape = "compute-bound" if compute_seconds > memory_seconds else "bandwidth-bound"

    transfer_seconds = 0.0
    if not resident:
        # A measured link below nameplate stretches the copy by exactly its reciprocal: the
        # bytes are unchanged and the wire is narrower. Clamped into (0, 1] so a nonsensical
        # ratio cannot divide by zero or make the copy free.
        link = min(1.0, max(1e-3, link_efficiency))
        transfer_seconds = host_transfer_seconds(bytes_per_row, accelerator_type) / link
        returned = bytes_per_row * max(0.0, result_fraction)
        transfer_seconds += host_transfer_seconds(returned, accelerator_type) / link
    device_seconds = kernel_seconds + transfer_seconds

    speedup = cpu_seconds / device_seconds if device_seconds > 0 else 0.0
    power_ratio = (watts + watts * 0.25) / _CPU_PATH_WATTS
    energy_ratio = power_ratio / speedup if speedup > 0 else 0.0
    transfer_share = transfer_seconds / device_seconds if device_seconds > 0 else 0.0
    dominated = " (host copy dominates)" if transfer_share > 0.5 else ""
    return EnergyAdvice(
        worth_it=energy_ratio <= 1.0 and speedup >= 1.0,
        speedup=speedup,
        power_ratio=power_ratio,
        energy_ratio=energy_ratio,
        transfer_share=transfer_share,
        reason=(
            f"{shape}: {speedup:.2f}x throughput for {power_ratio:.1f}x power "
            f"({energy_ratio:.2f}x energy){dominated}"
        ),
    )


def record_measured_efficiency(
    hub: MetadataHub | None,
    accelerator_type: str | None,
    joules: float,
    work: int,
    *,
    kind: str = "rows",
) -> None:
    """Fold one stage's measured work-per-joule into what this fleet has learned.

    **Core measures, Kyber consumes.** The datasheet says an H100 does 3.2x an A100's dense
    FLOPS; it does not say what *this* workload gets on either, and for a bandwidth-bound
    stage the answer is nowhere near the FLOPS ratio. A fleet that runs the same shape daily
    can measure the difference, and that measurement is worth more than any ratio derived from
    a specification.

    Stored as running sums per device model, so folding is O(1) and order-independent — the
    same mergeable shape everything else here uses.

    Args:
        hub: The metadata hub, or `None` to skip recording.
        accelerator_type: Device model the stage ran on; an unresolvable name is skipped
            rather than pooled, because pooling unlike devices converges on an average right
            for neither.
        joules: Energy the stage drew.
        work: Rows emitted or tokens generated.
        kind: `"rows"` or `"tokens"`; the two are not comparable and never share a bucket.
    """
    if hub is None or not accelerator_type or joules <= 0 or work <= 0:
        return
    from batcher._internal.device_specs import device_generation

    generation = device_generation(accelerator_type)
    keys = [f"{accelerator_type}:{kind}"]
    if generation:
        # Also folded into the generation's bucket, so a part this fleet has not run yet
        # inherits what its siblings measured rather than starting cold.
        keys.append(f"gen:{generation}:{kind}")
    try:
        for key in keys:
            bucket = hub.get_keyed_param(scoped(_EFFICIENCY_NS), key) or {}
            hub.put_keyed_param(
                scoped(_EFFICIENCY_NS),
                key,
                {
                    "joules": float(bucket.get("joules", 0.0)) + float(joules),
                    "work": float(bucket.get("work", 0.0)) + float(work),
                    "n": int(bucket.get("n", 0)) + 1,
                },
            )
    except Exception as exc:  # pragma: no cover - learning must never break a query
        note_suppressed("kyber", "record measured efficiency", exc)


def learned_work_per_joule(
    hub: MetadataHub | None,
    accelerator_type: str | None,
    *,
    kind: str = "rows",
) -> float | None:
    """What this fleet has measured a device to deliver per joule, or `None` when it hasn't.

    `None` until the bucket holds enough samples, and `None` for a device nothing has run on.
    A caller must fall back to the datasheet ratio rather than treating an absent measurement
    as a bad one — that is the difference between "we have not measured this device" and
    "this device is slow".

    Args:
        hub: The metadata hub, or `None`.
        accelerator_type: Device model to look up.
        kind: `"rows"` or `"tokens"`, matching what was recorded.

    Returns:
        Measured work per joule, or `None` when unknown or under-sampled.
    """
    if hub is None or not accelerator_type:
        return None
    from batcher._internal.device_specs import device_generation

    # The device's own samples first, then its generation's. An H100 and an H200 differ in
    # memory and bandwidth but share an instruction set and an FP8 unit, so a measurement from
    # one transfers to the other in a way an Ampere measurement does not — which is what makes
    # a newly added part usable immediately instead of cold for its first thousand runs.
    keys = [f"{accelerator_type}:{kind}"]
    generation = device_generation(accelerator_type)
    if generation:
        keys.append(f"gen:{generation}:{kind}")
    for key in keys:
        try:
            bucket = hub.get_keyed_param(scoped(_EFFICIENCY_NS), key) or {}
        except Exception as exc:  # pragma: no cover
            note_suppressed("kyber", "read measured efficiency", exc)
            return None
        joules = float(bucket.get("joules", 0.0))
        work = float(bucket.get("work", 0.0))
        if int(bucket.get("n", 0)) >= _MIN_SAMPLES and joules > 0 and work > 0:
            return work / joules
    return None
