"""Device power draw — the neutral model every power-aware decision reads.

A GPU datacenter is provisioned in watts before it is provisioned in slots. A rack has a
breaker, a row has a busway, and a hall has a substation; none of them care how many devices
are physically installed, only what those devices draw at once. That makes power a *plannable
resource* in exactly the way memory already is: Kyber can prefer the placement that fits the
envelope, Carbonite can refuse to admit work that would exceed it, and the executor can report
what a stage actually cost.

The model here is deliberately simple and stated rather than hidden:

    P(u) = idle + (tdp - idle) * u

for device utilization `u` in [0, 1], plus a per-device host share for the CPU, memory, fans,
and PSU losses attributable to one accelerator. A real board's curve is not linear — it is
convex at low utilization and clamps at the power limit — but the linear form needs only the
two figures a datasheet publishes, and it is used to *compare* placements and to size a budget
with headroom, both of which are ratios. A curve fitted per device would be more precise and
would need per-device telemetry that most fleets do not export.

Everything reports `0.0` for an unknown device rather than a guess, so a fleet whose model
names this build does not recognize keeps whatever behavior it had instead of being scheduled
against fabricated watts.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PowerEnvelope",
    "configured_power_envelope",
    "device_power_watts",
    "energy_joules",
    "fleet_power_watts",
    "host_overhead_watts",
    "max_concurrent_devices",
]

#: Host-side power attributable to one accelerator: CPU share, system memory, fans, and PSU
#: conversion loss. Expressed as a fraction of the device's own TDP rather than a fixed number
#: of watts, because a 72 W L4 sits in a chassis proportionally lighter than a 700 W H100 —
#: a flat constant would swamp the small device and vanish against the large one.
_HOST_OVERHEAD_FRACTION = 0.25


def host_overhead_watts(accelerator_type: str | None) -> float:
    """Host-side watts attributable to one device — CPU share, RAM, fans, and PSU loss.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        Watts to add to the device's own draw when budgeting a whole server, `0.0` when the
        device is unrecognized.
    """
    from batcher._internal.device_specs import device_tdp_watts

    return device_tdp_watts(accelerator_type) * _HOST_OVERHEAD_FRACTION


def device_power_watts(
    accelerator_type: str | None,
    utilization: float = 1.0,
    *,
    include_host: bool = False,
) -> float:
    """Power draw of one device at a given utilization, in watts.

    Args:
        accelerator_type: A Ray accelerator-type name.
        utilization: Fraction of the device's compute in use, clamped to [0, 1]. `0.0`
            reports the idle draw — a reserved-but-unused device is not free.
        include_host: Add the host share (`host_overhead_watts`) so the figure describes a
            whole server slot rather than a bare board.

    Returns:
        Watts, or `0.0` when the device's power figures are unknown.
    """
    from batcher._internal.device_specs import device_idle_watts, device_tdp_watts

    tdp = device_tdp_watts(accelerator_type)
    if tdp <= 0:
        return 0.0
    idle = device_idle_watts(accelerator_type)
    u = min(1.0, max(0.0, utilization))
    watts = idle + (tdp - idle) * u
    return watts + host_overhead_watts(accelerator_type) if include_host else watts


def fleet_power_watts(
    devices: dict[str, int] | list[tuple[str, int]],
    utilization: float = 1.0,
    *,
    include_host: bool = False,
) -> float:
    """Total draw of a mixed fleet at a uniform utilization, in watts.

    Unrecognized device models contribute `0.0`, so the result is a *lower bound* on a fleet
    with unknown hardware in it. That is the right direction for a budget check: it can fail
    to catch an overrun, but it can never invent one.

    Args:
        devices: Device model name to device count, either as a mapping or as pairs.
        utilization: Fraction of compute in use across the fleet, clamped to [0, 1].
        include_host: Include the per-device host share.

    Returns:
        Total watts across the fleet.
    """
    pairs = devices.items() if isinstance(devices, dict) else devices
    return sum(
        device_power_watts(name, utilization, include_host=include_host) * max(0, count)
        for name, count in pairs
    )


def energy_joules(watts: float, seconds: float) -> float:
    """Energy for a constant draw over a duration, in joules.

    Args:
        watts: Average power draw.
        seconds: Duration the draw was held.

    Returns:
        Joules (watt-seconds); `0.0` when either input is non-positive.
    """
    if watts <= 0 or seconds <= 0:
        return 0.0
    return watts * seconds


def max_concurrent_devices(
    budget_watts: float,
    accelerator_type: str | None,
    utilization: float = 1.0,
    *,
    include_host: bool = True,
) -> int:
    """How many devices of one model fit inside a power budget, at a given utilization.

    The concurrency bound a power-capped rack imposes, which is frequently tighter than its
    slot count: sixteen 700 W devices need 11.2 kW of device power alone, more than a
    single 208 V/60 A rack circuit delivers.

    Args:
        budget_watts: Watts available to the fleet.
        accelerator_type: A Ray accelerator-type name.
        utilization: Utilization the devices will be driven at, clamped to [0, 1].
        include_host: Charge each device its host share too — the default, because a budget
            drawn at the rack covers whole servers.

    Returns:
        Devices that fit, `0` when even one does not. Returns `-1` for "unbounded" when the
        device's power is unknown, so a caller can tell "no devices fit" from "no opinion".
    """
    per_device = device_power_watts(accelerator_type, utilization, include_host=include_host)
    if per_device <= 0:
        return -1  # unknown device → no opinion, rather than a bound of zero
    if budget_watts <= 0:
        return 0
    return int(budget_watts // per_device)


@dataclass(frozen=True, slots=True)
class PowerEnvelope:
    """The power a workload is allowed to draw, and what it is expected to draw.

    The power analogue of `ResourceBounds`: Kyber estimates `expected_watts` for a plan from
    the devices it selected, Carbonite checks it against `budget_watts` before admitting the
    work, and the executor reports the measured figure back. All three speak this one contract
    rather than each carrying its own notion of "too much power".

    A `budget_watts` of `0.0` means unbounded — no power cap is configured — which is the
    default everywhere so that a fleet without a configured envelope behaves exactly as it did
    before power became a plannable resource.

    Attributes:
        budget_watts: Cap the workload may not exceed; `0.0` for unbounded.
        expected_watts: Estimated draw of the planned work.
        headroom_fraction: Fraction of the budget deliberately left unused, absorbing the gap
            between the linear power model and a real board's curve.
    """

    budget_watts: float = 0.0
    expected_watts: float = 0.0
    headroom_fraction: float = 0.1

    @property
    def unbounded(self) -> bool:
        """Whether no power cap is configured, in which case every check passes."""
        return self.budget_watts <= 0

    @property
    def usable_watts(self) -> float:
        """The budget minus its headroom — the figure a feasibility check compares against."""
        if self.unbounded:
            return 0.0
        return self.budget_watts * (1.0 - min(0.9, max(0.0, self.headroom_fraction)))

    def fits(self) -> bool:
        """Whether the expected draw fits inside the usable budget.

        Returns:
            True when unbounded, when the expected draw is unknown (`0.0`), or when it fits.
        """
        if self.unbounded or self.expected_watts <= 0:
            return True
        return self.expected_watts <= self.usable_watts

    def devices_that_fit(self, accelerator_type: str | None, utilization: float = 1.0) -> int:
        """Devices of one model the usable budget can power.

        Args:
            accelerator_type: A Ray accelerator-type name.
            utilization: Utilization the devices will be driven at.

        Returns:
            The device count that fits, or `-1` for "no opinion" when the envelope is
            unbounded or the device model is unrecognized.
        """
        if self.unbounded:
            return -1
        return max_concurrent_devices(self.usable_watts, accelerator_type, utilization)

    def clamp_devices(
        self,
        requested: int,
        accelerator_type: str | None,
        utilization: float = 1.0,
    ) -> int:
        """Clamp a requested device count to what this envelope can power.

        The one place the clamp is computed, so a plan is never refused for one fan-out and
        then scheduled at another: Kyber's sizing and Carbonite's grant both call this.

        Args:
            requested: Devices the caller wants.
            accelerator_type: The device model those devices are.
            utilization: Utilization they are expected to run at.

        Returns:
            `requested` unchanged when the envelope is unbounded or the device is
            unrecognized, and at least 1 otherwise — a budget too small for a single device is
            a misconfiguration to surface with a verdict, not a silent zero-device plan.
        """
        if requested <= 0:
            return requested
        allowed = self.devices_that_fit(accelerator_type, utilization)
        return requested if allowed < 0 else max(1, min(requested, allowed))

    def scale_to_fit(self, device_watts: float) -> int:
        """Devices of a given draw that fit in the usable budget.

        Args:
            device_watts: Draw of one device, as reported by `device_power_watts`.

        Returns:
            The device count that fits, or `-1` when unbounded or the draw is unknown.
        """
        if self.unbounded or device_watts <= 0:
            return -1
        return int(self.usable_watts // device_watts)


def configured_power_envelope(expected_watts: float = 0.0) -> PowerEnvelope:
    """The deployment's power envelope, read from the active configuration.

    Neutral on purpose: Kyber sizes against this and Carbonite admits against it, and the two
    subsystems cannot import each other, so the alternative to one accessor here is the same
    arithmetic pasted into both — which is exactly how a grant and a verdict come to disagree.

    Args:
        expected_watts: A draw the caller has already estimated, `0.0` when unknown.

    Returns:
        A `PowerEnvelope`; `unbounded` when no budget is configured, which is the default.
    """
    from batcher.config import active_config

    energy = active_config().accelerator.energy
    return PowerEnvelope(
        budget_watts=max(0.0, energy.power_budget_watts),
        expected_watts=max(0.0, expected_watts),
        headroom_fraction=energy.power_headroom,
    )
