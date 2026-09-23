"""Turning a sampled window into the one sentence a reader needed.

`energy.format_device_table` shows what each device is doing right now. This shows what limited
each device across a run, which is a different question and the one people open a report to
answer. The difference is the sampling window: a table of instantaneous readings taken after a
query finished describes an idle fleet, and describes it accurately.

The report is deliberately short. A device gets one line, one verdict, and one thing to change,
because the failure mode of a diagnostic page is that it lists twenty numbers and leaves the
reader to work out which two matter. The numbers are still there — every one of them is a
Prometheus series exported from `gauges` — and this is the layer that says what they mean.

**Verdicts come from `hardware.telemetry.bottleneck`, not from here.** Two surfaces render
them: `format_bottleneck_report` below, which the GPU-diagnosis guide has readers import
directly, and `observe.insights.devices`, which folds the fleet verdict into a run's findings.
A classification living in either would say something subtly different in the other, so it is
one function and these two modules only format it.

This paragraph named the *terminal accelerator report* as the second surface until 2026-09-22.
It renders no verdict and never has: `api/session/accelerators/` carries a `throttled` row read
straight from NVML's throttle reasons, which is a reading rather than a diagnosis. Worth
correcting rather than deleting, because the sentence is the argument for where the
classification lives, and an argument resting on a surface that does not exist is one nobody
can check.

**Nothing is invented when nothing was sampled.** A run with no window says so in one line, and
a device with too few samples reports `unknown` rather than a verdict derived from two readings.
That distinction is the entire value of the page: a confident wrong diagnosis costs more than no
diagnosis, because someone acts on it.
"""

from __future__ import annotations

from batcher._internal.hardware.telemetry.bottleneck import Bottleneck, classify_device
from batcher._internal.hardware.telemetry.processes import device_shared_with_others
from batcher._internal.hardware.telemetry.sampler import TelemetrySampler, saturation_shape

__all__ = [
    "device_verdicts",
    "format_bottleneck_report",
    "format_saturation_line",
    "window_snapshot",
]


def device_verdicts(window: TelemetrySampler | None = None) -> tuple[Bottleneck, ...]:
    """One verdict per device sampled in the window, in device order.

    Args:
        window: The accumulator to read, or `None` to take the process's running one from
            `series.device_window`.

    Returns:
        The verdicts, empty when nothing was sampled. Every device that got samples gets a
        verdict, including `"unknown"` ones — a device the sampler watched and could not
        classify is a different report line from a device it never saw.
    """
    if window is None:
        from batcher.observe.accelerators.series import device_window

        window = device_window()
    if window is None:
        return ()
    out: list[Bottleneck] = []
    for index in window.devices():
        sm = window.summary(index, "sm")
        if sm.samples == 0:
            continue
        occupancy = window.summary(index, "sm_occupancy")
        out.append(
            classify_device(
                index,
                sm=sm,
                memory=window.summary(index, "memory"),
                pcie=window.summary(index, "pcie_utilization"),
                throttled=window.summary(index, "throttled"),
                codec=window.summary(index, "codec"),
                # Whether a neighbour was computing on the board. Nothing supplied this, so
                # `classify_device` took its `None` default — documented as "treated as not
                # shared" — and the `contended` verdict could not fire at all. Every other
                # input here comes from the sampled window; contention is the one signal the
                # window cannot hold, because it is a fact about *processes* rather than about
                # the device, so it is read here at classification time.
                shared=device_shared_with_others(index),
                # Passed only when DCGM actually contributed. An all-zero occupancy summary
                # from a host without DCGM would otherwise classify every busy device as
                # occupancy limited, which is the most expensive wrong answer available here.
                occupancy=occupancy if occupancy.samples else None,
            )
        )
    return tuple(out)


def window_snapshot(window: TelemetrySampler | None = None) -> dict:
    """The sampled window as plain numbers, for the metrics snapshot and the dashboard.

    Distinct from the Prometheus gauges, which are instantaneous by design: a scrape wants the
    reading now, and a snapshot document wants the shape of the run so far. Both exist because
    they answer different questions, and a dashboard rendering a sparkline from repeated scrapes
    still cannot tell a steady device from a swinging one — the shape is computed here, from
    every sample, rather than from the handful a scraper happened to catch.

    Args:
        window: The accumulator to read, or `None` to take the process's running one.

    Returns:
        `{"sampled": bool, "devices": {index: {...}}}`, with `sampled` False and no devices
        when nothing has been sampled. That flag is what stops a consumer rendering an
        all-zeros panel as a quiet fleet.
    """
    if window is None:
        from batcher.observe.accelerators.series import device_window

        window = device_window()
    if window is None:
        return {"sampled": False, "devices": {}}
    verdicts = {v.index: v for v in device_verdicts(window)}
    devices: dict[str, dict] = {}
    for index in window.devices():
        sm = window.summary(index, "sm")
        if sm.samples == 0:
            continue
        verdict = verdicts.get(index)
        devices[str(index)] = {
            "samples": sm.samples,
            "sm_mean": round(sm.mean, 4),
            "sm_peak": round(sm.peak, 4),
            "sm_trough": round(sm.trough, 4),
            "shape": saturation_shape(sm),
            "power_watts_mean": round(window.summary(index, "power_watts").mean, 1),
            "pcie_utilization_mean": round(window.summary(index, "pcie_utilization").mean, 4),
            "throttled_fraction": round(window.summary(index, "throttled").mean, 4),
            "verdict": verdict.verdict if verdict else "",
            "advice": verdict.advice if verdict else "",
        }
    return {"sampled": bool(devices), "devices": devices}


def format_saturation_line(index: int, window: TelemetrySampler) -> str:
    """One device's load shape and headline numbers, as a single aligned line.

    Args:
        index: Device index.
        window: The accumulator holding the samples.

    Returns:
        A fixed-column line, or `""` when the device has no SM samples.
    """
    sm = window.summary(index, "sm")
    if sm.samples == 0:
        return ""
    power = window.summary(index, "power_watts")
    pcie = window.summary(index, "pcie_utilization")
    shape = saturation_shape(sm) or "-"
    return (
        f"{index:<3}  {shape:<10}  sm {sm.mean:>4.0%} "
        f"(peak {sm.peak:>4.0%}, low {sm.trough:>4.0%})  "
        f"bus {pcie.mean:>4.0%}  {power.mean:>5.0f} W  {sm.samples:>5} samples"
    )


def format_bottleneck_report(window: TelemetrySampler | None = None) -> str:
    """What limited each device across the sampled window, and what to change.

    The page to read when a run was correct and slower than it should have been, and the
    answer was not in the plan.

    Args:
        window: The accumulator to read, or `None` to take the process's running one.

    Returns:
        A plain-text report, or one line explaining that nothing was sampled — which is the
        honest answer on a host with no devices, and on a run where sampling was never started.
    """
    if window is None:
        from batcher.observe.accelerators.series import device_window

        window = device_window()
    if window is None:
        return "devices: no sampling window (start one with accelerator.telemetry_sampling)"
    devices = window.devices()
    if not devices:
        return "devices: sampled, but no device reported a reading"

    lines = ["gpu  shape       load                              bus     power   window"]
    for index in devices:
        line = format_saturation_line(index, window)
        if line:
            lines.append(line)

    verdicts = device_verdicts(window)
    if verdicts:
        lines.append("")
        lines.append("gpu  verdict            what to change")
        for verdict in verdicts:
            lines.append(f"{verdict.index:<3}  {verdict.verdict:<17}  {verdict.advice}")
            lines.append(f"     {'':17}  ({verdict.detail})")

    from batcher._internal.hardware.telemetry.bottleneck import fleet_verdict

    lead = fleet_verdict(verdicts)
    if lead is not None:
        lines.append("")
        lines.append(f"lead finding: gpu {lead.index} is {lead.verdict} — {lead.advice}")
    return "\n".join(lines)
