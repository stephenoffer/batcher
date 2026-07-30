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

from batcher.config import active_config

__all__ = [
    "EnergyAdvice",
    "device_energy_advice",
    "power_bounded_devices",
    "select_device_class",
    "stage_joules",
]


def select_device_class(
    candidates: list[str] | tuple[str, ...],
    model_gib: float,
    *,
    prefer_efficiency: bool | None = None,
    headroom: float = 0.15,
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

    Returns:
        A device model name to pin to, or `None` when nothing fits, nothing is known, or every
        candidate fits (in which case a pin would only constrain placement).
    """
    from batcher._internal.device_specs import device_spec, device_tflops_per_watt

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
        ranked = sorted(fitting, key=lambda n: (-device_tflops_per_watt(n), n))
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
    from batcher.plan.energy.power import max_concurrent_devices

    energy = active_config().accelerator.energy
    if energy.power_budget_watts <= 0 or requested <= 0:
        return requested
    usable = energy.power_budget_watts * (1.0 - min(0.9, max(0.0, energy.power_headroom)))
    allowed = max_concurrent_devices(usable, accelerator_type, utilization)
    if allowed < 0:
        return requested  # unknown device: no opinion
    return max(1, min(requested, allowed))


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
            `1.0` means the device is the cheaper machine to run.
        reason: One line for the decision log.
    """

    worth_it: bool
    speedup: float = 0.0
    power_ratio: float = 0.0
    energy_ratio: float = 0.0
    reason: str = ""


#: Draw of the CPU path a stage would otherwise run on, in watts: one server's worth of
#: sockets and memory at load. A rough figure, and it only ever appears as a *ratio* against a
#: device's draw, so the comparison is far less sensitive to it than an absolute joule count
#: would be.
_CPU_PATH_WATTS = 400.0

#: Peak dense half-precision throughput of that same CPU path, in TFLOP/s — roughly what a
#: two-socket server reaches with AVX-512. Used only as the denominator of a ratio, for the
#: same reason.
_CPU_PATH_TFLOPS = 2.0


def device_energy_advice(
    accelerator_type: str | None,
    *,
    bytes_per_row: float,
    flops_per_row: float,
    cpu_gbps: float = 20.0,
    achieved_fraction: float = 1.0,
) -> EnergyAdvice:
    """Judge a stage's device move on energy rather than on time alone.

    The roofline argument, made explicit. A stage with `flops_per_row / bytes_per_row` below a
    device's ridge point is bandwidth-bound: its speedup is the memory-bandwidth ratio, not the
    FLOPS ratio, and a device that draws five times the power for three times the bandwidth is
    burning energy to finish sooner. Above the ridge the FLOPS ratio applies and the device
    wins on both axes, which is why inference and decode belong on one and a projection does
    not.

    Args:
        accelerator_type: The candidate device model.
        bytes_per_row: Bytes of input the stage reads per row.
        flops_per_row: Floating-point work the stage does per row.
        cpu_gbps: Effective memory bandwidth of the CPU path, in GB/s.
        achieved_fraction: Fraction of the nameplate ratio a real kernel is expected to
            reach, in (0, 1]. The ratios below are peak-against-peak and so are an upper
            bound; a caller with a measured figure passes it and gets an honest verdict
            instead of an optimistic one.

    Returns:
        The advice. With an unrecognized device every ratio is `0.0` and `worth_it` is True,
        preserving whatever decision the caller would have made without an energy opinion.
    """
    from batcher._internal.device_specs import (
        device_arithmetic_intensity,
        device_half_tflops,
        device_memory_bandwidth_gbps,
        device_tdp_watts,
    )

    bandwidth = device_memory_bandwidth_gbps(accelerator_type)
    watts = device_tdp_watts(accelerator_type)
    if bandwidth <= 0 or watts <= 0 or bytes_per_row <= 0:
        return EnergyAdvice(worth_it=True, reason="device unknown: no energy opinion")

    ridge = device_arithmetic_intensity(accelerator_type)
    intensity = flops_per_row / bytes_per_row
    if intensity >= ridge > 0:
        # Above the ridge the device is compute-bound, so its advantage is the FLOPS ratio.
        speedup = max(1.0, device_half_tflops(accelerator_type) / _CPU_PATH_TFLOPS)
        shape = "compute-bound"
    else:
        # Below it the tensor cores are idle waiting on HBM, and the only ratio that applies
        # is memory bandwidth — which is why a scan gains far less from a device than its
        # FLOPS figure suggests.
        speedup = max(1.0, bandwidth / max(1.0, cpu_gbps))
        shape = "bandwidth-bound"

    speedup *= min(1.0, max(1e-3, achieved_fraction))
    power_ratio = (watts + watts * 0.25) / _CPU_PATH_WATTS
    energy_ratio = power_ratio / max(1e-9, speedup)
    return EnergyAdvice(
        worth_it=energy_ratio <= 1.0,
        speedup=speedup,
        power_ratio=power_ratio,
        energy_ratio=energy_ratio,
        reason=(
            f"{shape}: {speedup:.1f}x throughput for {power_ratio:.1f}x power "
            f"({energy_ratio:.2f}x energy)"
        ),
    )
