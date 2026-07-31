"""The power envelope as an admission decision — Carbonite protecting a rack's breaker.

Memory admission asks "will this plan fit in RAM". On a GPU fleet there is a second question of
the same shape and the same consequence: will this stage fit in the watts the deployment is
allowed to draw. A rack of sixteen 700 W devices needs more than eleven kilowatts of device
power alone, more than one 208 V circuit delivers, and exceeding the budget does not fail
cleanly — the driver clamps every device in the zone, and the whole rack reads as mysteriously
slower.

So power is admitted the way memory is: a configured envelope, a check before the work starts,
and a counter-offer rather than a refusal. The counter-offer is a device count, carried in
`ResourceBounds.n_max_parallelism` because that is exactly what it bounds — how many workers
may hold a device at once.

The envelope itself is read in the neutral layer (`plan.energy.configured_power_envelope`), so
the scheduling grant Kyber sizes against and the verdict Carbonite returns cannot disagree
about what the budget allows. Two rules keep this safe to leave enabled:
an unconfigured budget admits everything, and an unrecognized device model admits everything,
because clamping a fleet against fabricated watts is worse than not clamping it at all.
"""

from __future__ import annotations

from batcher._internal.logging import note_suppressed
from batcher.plan.energy.power import configured_power_envelope, device_power_watts
from batcher.plan.resource import FeasibilityVerdict, ResourceBounds

__all__ = ["devices_within_budget", "enforced_limit_watts", "validate_fleet_power"]


def devices_within_budget(
    accelerator_type: str | None,
    requested: int,
    *,
    utilization: float = 1.0,
) -> int:
    """Clamp a device count to what the configured budget can power.

    Args:
        accelerator_type: The device model those devices are.
        requested: Devices the caller wants.
        utilization: Utilization the devices are expected to run at.

    Computed *through* `validate_fleet_power` rather than beside it, so the number a stage is
    granted and the counter-offer a refusal reports are the same number by construction. Two
    code paths to the same figure is how a plan comes to be refused for one fan-out and then
    scheduled at another.

    Returns:
        The device count to use: `requested` unchanged when no budget is configured or the
        device model is unrecognized, and at least 1 otherwise — a budget too small for a
        single device is a misconfiguration to surface with a verdict, not a silent
        zero-device plan.
    """
    verdict = validate_fleet_power(accelerator_type, requested, utilization=utilization)
    if verdict.feasible or verdict.suggested_bounds is None:
        return requested
    return max(1, min(requested, verdict.suggested_bounds.n_max_parallelism))


def validate_fleet_power(
    accelerator_type: str | None,
    devices: int,
    *,
    utilization: float = 1.0,
) -> FeasibilityVerdict:
    """Whether a fleet of `devices` fits the configured power envelope.

    Args:
        accelerator_type: The device model.
        devices: Devices the stage wants to hold.
        utilization: Utilization they are expected to run at.

    Returns:
        A verdict. Feasible when no budget is configured, when the device model is
        unrecognized, or when the fleet fits. Otherwise infeasible with
        `binding_constraint="power"` and a counter-offer whose `n_max_parallelism` is the
        device count the budget can actually run. The verdict is `advisory` when the draw
        was modelled from a datasheet rather than measured, which it always is at planning
        time — power admission should steer a plan, never fail a query.
    """
    envelope = configured_power_envelope()
    if envelope.unbounded or devices <= 0:
        return FeasibilityVerdict(feasible=True)
    per_device = device_power_watts(
        accelerator_type,
        utilization,
        include_host=True,
        enforced_limit_watts=enforced_limit_watts(accelerator_type),
    )
    if per_device <= 0:
        return FeasibilityVerdict(feasible=True)  # unknown device: no opinion
    expected = per_device * devices
    if expected <= envelope.usable_watts:
        return FeasibilityVerdict(feasible=True)
    allowed = max(1, int(envelope.usable_watts // per_device))
    return FeasibilityVerdict(
        feasible=False,
        binding_constraint="power",
        suggested_bounds=ResourceBounds(
            m_max_bytes=0,
            c_max_credits=0,
            n_max_parallelism=allowed,
        ),
        binding_op=accelerator_type or "gpu",
        advisory=True,
    )


def enforced_limit_watts(accelerator_type: str | None = None) -> float:
    """The power cap *this* node's devices run under, when they are the ones being priced.

    A power-constrained hall runs its parts below nameplate — a 700 W device capped at 500 is
    ordinary on rented capacity — and an admission check using the datasheet over-states the
    draw by the difference, refusing fan-outs the rack can power. The driver knows the enforced
    limit; the datasheet does not.

    **Only when the local devices are the model being asked about.** This runs where the plan
    is built, and on the usual topology that is a head node whose hardware is not the fleet's.
    Applying its cap to a fleet-wide budget would under-state what the fleet draws, and
    under-stating a breaker's load is the one error here with a physical consequence — so a
    model mismatch reports nothing and the datasheet stands.

    Args:
        accelerator_type: The model being priced. `None` accepts any local device, which is
            correct only for a caller asking about *this* node.

    Returns:
        The highest limit across the matching local devices — the maximum rather than the
        mean, since a mean under-states a node whose devices are capped unevenly — or `0.0`
        when nothing matched or nothing could be read.
    """
    from batcher._internal.device_specs import resolve_device_name
    from batcher._internal.hardware.nvml import device_telemetry

    try:
        wanted = resolve_device_name(accelerator_type) if accelerator_type else ""
        limits = [
            d.power_limit_watts
            for d in device_telemetry()
            if d.power_limit_watts > 0 and (not wanted or resolve_device_name(d.name) == wanted)
        ]
    except Exception as exc:  # pragma: no cover - a probe must never break admission
        note_suppressed("carbonite", "read the enforced device power limit", exc)
        return 0.0
    return max(limits) if limits else 0.0
