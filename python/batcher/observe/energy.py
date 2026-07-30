"""Reporting what a run cost in watts — the terminal view and the metrics rows.

A GPU-hour is what a datacenter bills; joules are what it buys; and the gap between the two is
where every efficiency conversation happens. `plan.energy` produces the numbers, and this
renders them: a compact per-stage table for a human at a terminal, and a flat row set for a
metrics sink.

Two reporting rules the rest of `observe` also follows, restated because they matter more here
than elsewhere. **An unknown figure is omitted, never zero** — a run whose device model this
build does not recognize should report no energy rather than 0 J, because a zero in a cost
report is a claim and an omission is not. For the same reason the total says whether it was
measured from device readings or modelled from datasheets, since an estimate presented as a
measurement is the one error a cost report cannot recover from. And **efficiency is work per
joule**, so a stage that emitted nothing has no efficiency figure rather than an infinite one.

Neutral, like the rest of `observe`: it reads the ledger and the device telemetry, and imports
no subsystem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher._internal.hardware.nvml import DeviceTelemetry
    from batcher.plan.energy import EnergyLedger, GridProfile

__all__ = [
    "energy_metrics",
    "format_device_table",
    "format_energy_report",
    "format_fleet_efficiency",
]


def _si(joules: float) -> str:
    """Joules at a human scale: J, kJ, MJ, or GJ, three significant figures."""
    for unit, scale in (("GJ", 1e9), ("MJ", 1e6), ("kJ", 1e3)):
        if joules >= scale:
            return f"{joules / scale:.3g} {unit}"
    return f"{joules:.3g} J"


def format_energy_report(ledger: EnergyLedger, grid: GridProfile | None = None) -> str:
    """A per-stage energy table with the run's roll-up beneath it.

    Args:
        ledger: The run's energy ledger.
        grid: The site's grid profile, for cost and carbon lines. `None` reads the configured
            one, so a deployment that set its price and intensity gets those lines without
            passing anything. Lines are omitted rather than zeroed when it is unconfigured.

    Returns:
        A plain-text block, or a single line saying nothing was recorded when the ledger is
        empty.
    """
    if grid is None:
        from batcher.plan.energy.carbon import configured_grid

        grid = configured_grid()
    if not ledger.stages:
        return "energy: nothing recorded (no accelerator stage ran, or accounting is off)"

    stages = ledger.by_stage()
    rows = ["stage                     device            energy    util   idle    work/J"]
    for stage in stages:
        per_joule = stage.tokens_per_joule or stage.rows_per_joule
        work = f"{per_joule:,.1f}" if per_joule is not None else "-"
        idle = stage.idle_joules / stage.joules if stage.joules > 0 else 0.0
        rows.append(
            f"{stage.stage[:24]:<24}  {(stage.accelerator_type or 'cpu')[:16]:<16}  "
            f"{_si(stage.joules):>8}  {stage.utilization:>5.0%}  {idle:>4.0%}  {work:>8}"
        )

    total = ledger.total_joules
    rows.append("")
    measured = ledger.measured_joules / total if total > 0 else 0.0
    basis = "measured" if measured >= 0.999 else f"{measured:.0%} measured, rest modelled"
    rows.append(f"total {_si(total)} across {len(stages)} stage(s) ({basis})")
    if total > 0:
        idle_share = f"{ledger.idle_fraction():.0%}"
        rows.append(f"idle  {_si(ledger.total_idle_joules)} ({idle_share} of total)")
    tpj, rpj = ledger.tokens_per_joule(), ledger.rows_per_joule()
    if tpj is not None:
        rows.append(f"efficiency {tpj:,.1f} tokens/J")
    if rpj is not None:
        rows.append(f"efficiency {rpj:,.1f} rows/J")
    if grid is not None and grid.configured and total > 0:
        if grid.price_per_kwh > 0:
            rows.append(f"cost   {grid.cost(total):,.4f} per run at {grid.price_per_kwh}/kWh")
        if grid.gco2e_per_kwh > 0:
            rows.append(
                f"carbon {grid.carbon_grams(total):,.1f} g CO2e "
                f"at {grid.gco2e_per_kwh:g} g/kWh, PUE {grid.pue:g}"
            )
    hottest = max(stages, key=lambda s: s.joules, default=None)
    if hottest is not None and len(stages) > 1:
        share = hottest.joules / total if total > 0 else 0.0
        rows.append(f"hottest {hottest.stage} at {share:.0%} of the run's energy")
    return "\n".join(rows)


def energy_metrics(ledger: EnergyLedger, grid: GridProfile | None = None) -> dict[str, float]:
    """The run's energy figures as flat metric rows.

    Keys are prefixed `energy.` so they group in a sink alongside the existing `query.` and
    `shuffle.` families. Undefined figures are absent rather than zero.

    Args:
        ledger: The run's energy ledger.
        grid: The site's grid profile, for the cost and carbon rows. `None` reads the
            configured one.

    Returns:
        Metric name to value.
    """
    if grid is None:
        from batcher.plan.energy.carbon import configured_grid

        grid = configured_grid()
    out = {f"energy.{k}": v for k, v in ledger.summary().items()}
    total = ledger.total_joules
    if grid is not None and grid.configured and total > 0:
        if grid.price_per_kwh > 0:
            out["energy.cost"] = grid.cost(total)
        if grid.gco2e_per_kwh > 0:
            out["energy.carbon_grams"] = grid.carbon_grams(total)
        out["energy.facility_joules"] = grid.facility_joules(total)
    for device, joules in ledger.by_device().items():
        if device:
            out[f"energy.device.{device}"] = joules
    return out


def format_device_table(readings: Sequence[DeviceTelemetry] | None = None) -> str:
    """A live per-device view: draw, utilization, memory, temperature, and any clamp.

    This is the table that answers "why is this run slow" when the answer is not in the plan:
    a device pinned at its power limit, one clamped thermally, or one whose memory another
    tenant has filled.

    Args:
        readings: Telemetry records, or `None` to read them live.

    Returns:
        A plain-text table, or one line saying telemetry is unavailable.
    """
    if readings is None:
        from batcher._internal.hardware.nvml import device_telemetry

        readings = device_telemetry()
    if not readings:
        return "devices: no telemetry (NVML unavailable on this host)"
    faults, links = _device_conditions()
    rows = ["gpu  name                        power      sm    memory        temp  state"]
    for d in readings:
        power = f"{d.power_watts:.0f}/{d.power_limit_watts:.0f} W" if d.power_watts else "-"
        if d.memory_total_bytes:
            memory = f"{d.memory_used_bytes >> 30}/{d.memory_total_bytes >> 30} GiB"
        else:
            memory = "-"
        rows.append(
            f"{d.index:<3}  {d.name[:26]:<26}  {power:>10}  {d.sm_utilization:>4.0%}  "
            f"{memory:>12}  {d.temperature_c:>4.0f}C  {_device_state(d, faults, links)}"
        )
    return "\n".join(rows)


def _device_conditions():
    """`(faults_by_index, links_by_index)`, both empty where nothing could be read.

    Read once per table rather than per row, and never allowed to fail: this is a status
    display, and a display that raises is worse than one missing a column.
    """
    try:
        from batcher._internal.hardware.fabric import device_pcie_links
        from batcher._internal.hardware.faults import device_faults

        return (
            {f.index: f for f in device_faults() if f.readable},
            dict(enumerate(device_pcie_links())),
        )
    except Exception:  # pragma: no cover - a status table must never fail a run
        return ({}, {})


def _device_state(reading, faults: dict, links: dict) -> str:
    """The `state` cell for one device: the worst condition it is in, most serious first.

    Ordered by what an operator should do about it, not by severity of the underlying fault.
    A device needing replacement outranks one returning wrong data, which outranks one merely
    running slow — and a degraded *link* comes last precisely because it is the one that never
    announces itself: it belongs in the table, below anything more urgent.
    """
    fault = faults.get(reading.index)
    if fault is not None and fault.remap_failure:
        return "rma:row-remap-failed"
    if reading.ecc_uncorrected:
        return f"ecc:{reading.ecc_uncorrected}"
    if fault is not None and fault.needs_reset:
        return "reset-pending"
    if reading.throttled:
        return ",".join(reading.throttle_reasons)
    link = links.get(reading.index)
    if link is not None and link.degraded:
        return f"link:{link.degradation_ratio:.0%}"
    return "ok"


def format_fleet_efficiency(ledger: EnergyLedger) -> str:
    """Work per joule by device model, so a mixed fleet can be compared against itself.

    The question a heterogeneous fleet asks and a total cannot answer: which of these devices
    is actually the cheaper machine to run for this workload. A newer part drawing 40% more
    power while doing 2.5x the work wins, and only a per-device ratio says so.

    Args:
        ledger: The run's energy ledger.

    Returns:
        One line per device model, ordered most efficient first, or a note when the ledger
        holds a single device model (nothing to compare) or none at all.
    """
    rows: dict[str, tuple[float, int, int]] = {}
    for stage in ledger.stages:
        if not stage.accelerator_type:
            continue
        joules, rows_, tokens = rows.get(stage.accelerator_type, (0.0, 0, 0))
        rows[stage.accelerator_type] = (
            joules + stage.joules,
            rows_ + stage.rows,
            tokens + stage.tokens,
        )
    if len(rows) < 2:
        return "fleet efficiency: one device model (nothing to compare)"

    def rate(entry: tuple[float, int, int]) -> float:
        joules, rows_, tokens = entry
        work = tokens or rows_
        return work / joules if joules > 0 and work > 0 else 0.0

    lines = []
    for device, entry in sorted(rows.items(), key=lambda kv: (-rate(kv[1]), kv[0])):
        joules, rows_, tokens = entry
        unit = "tokens" if tokens else "rows"
        value = rate(entry)
        measure = f"{value:,.1f} {unit}/J" if value > 0 else "no work recorded"
        lines.append(f"{device:<24}  {_si(joules):>10}  {measure}")
    return "\n".join(lines)
