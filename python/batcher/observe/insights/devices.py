"""Findings about the accelerators themselves, as opposed to the plan that ran on them.

Every other rule family here reads the query profile: what each operator cost, how many rows it
moved, whether it spilled. That is the right source for a finding about the *plan*, and it is
structurally blind to the failures that make a GPU node slow, because none of them changes what
the plan did. A device clamped for a third of the run, one whose slot trained at half width, and
one another tenant is using all produce a profile identical to a healthy run's — just with
larger numbers in it.

These two rules read the device sampling window instead, and they only speak when there is one.
Sampling is opt-in (`accelerator.telemetry_sampling`, or the dashboard), so on a run without it
they return nothing rather than guessing from a single instantaneous reading — which, taken when
a query finishes, describes an idle fleet.

**A verdict, not a number.** The rules do not report utilization; the panels already do. They
report what limited the device and what to change, because a findings list exists to change what
somebody does. The classification is `hardware.telemetry.bottleneck`'s, shared with the terminal
report, so the dashboard and the console cannot disagree about what a device was doing.
"""

from __future__ import annotations

from typing import Any

from batcher.observe.insights.kinds import Insight

__all__ = ["derated_host_link", "device_bottleneck"]

#: Confidence below which a verdict is reported as informational rather than as a warning. A
#: low-confidence verdict means a second limit sits close behind the named one, so acting on it
#: buys less than the headline suggests — worth saying, not worth alarming about.
_CONFIDENT = 0.4


def device_bottleneck(
    _profile: dict[str, Any], _ops: list[dict[str, Any]], _total_ms: float
) -> list[Insight]:
    """What limited the accelerators across the sampled window, and what to change.

    One finding, not one per device. A fleet where seven devices are compute bound and one is
    throttled has one thing worth doing, and listing eight findings buries it — so this reports
    the highest-confidence *actionable* verdict and leaves the per-device table to the panels.

    Silent on a healthy fleet by construction: `compute_bound` is not actionable, because the
    device is already doing its job and the only remedy is more hardware.
    """
    from batcher.observe.accelerators.diagnosis import device_verdicts
    from batcher.observe.accelerators.series import device_window

    window = device_window()
    if window is None:
        return []
    verdicts = device_verdicts(window)
    if not verdicts:
        return []

    from batcher._internal.hardware.telemetry.bottleneck import fleet_verdict

    lead = fleet_verdict(verdicts)
    if lead is None:
        return []
    sharing = [v for v in verdicts if v.verdict == lead.verdict]
    return [
        Insight(
            severity="warning" if lead.confidence >= _CONFIDENT else "info",
            rule="device-bottleneck",
            title=f"GPU {lead.index} was {lead.verdict.replace('_', ' ')}",
            evidence=(
                f"Across the sampled window, {lead.detail}. "
                f"{len(sharing)} of {len(verdicts)} device(s) showed the same limit. "
                "This is read from the devices during the run, not from the plan, which is why "
                "it is invisible in the per-operator timings above."
            ),
            action=lead.advice,
            detail={
                "device": lead.index,
                "verdict": lead.verdict,
                "confidence": round(lead.confidence, 2),
                "devices_affected": len(sharing),
            },
        )
    ]


def derated_host_link(
    _profile: dict[str, Any], _ops: list[dict[str, Any]], _total_ms: float
) -> list[Insight]:
    """A device whose PCIe link trained below what both ends support.

    Not a tuning finding — a node fault, and the reason it is worth a rule of its own is that no
    amount of pipeline work recovers it. A card at x8 on a x16 part moves half the bytes per
    second it should, forever, and every upstream lever a starved-device finding would suggest
    is wasted effort against a reseated riser.

    Read live rather than from the window, because link geometry does not move within a run and
    the finding is worth reporting even on a run nobody sampled.
    """
    try:
        from batcher._internal.hardware.telemetry.throughput import device_throughput

        readings = device_throughput()
    except Exception:
        return []
    derated = [r for r in readings if r.readable and r.link_derated]
    if not derated:
        return []
    worst = min(derated, key=lambda r: r.pcie_gen * max(1, r.pcie_width))
    names = ", ".join(str(r.index) for r in derated)
    return [
        Insight(
            severity="critical",
            rule="host-link-derated",
            title=f"{len(derated)} device(s) on a host link below their capability",
            evidence=(
                f"GPU {worst.index} negotiated gen{worst.pcie_gen} x{worst.pcie_width} against "
                f"gen{worst.pcie_gen_max} x{worst.pcie_width_max} that both ends support "
                f"(devices: {names}). Every byte into these devices crosses that link, so the "
                "ceiling applies to the whole node and nothing in the pipeline can raise it."
            ),
            action=(
                "Treat this as a node to drain rather than a job to tune: reseat the card or "
                "riser, and check the slot's lane allocation in firmware. Until then, place "
                "transfer-heavy stages on the unaffected devices."
            ),
            detail={
                "devices": [r.index for r in derated],
                "gen": worst.pcie_gen,
                "width": worst.pcie_width,
                "gen_max": worst.pcie_gen_max,
                "width_max": worst.pcie_width_max,
            },
        )
    ]
