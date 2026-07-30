"""Energy as a first-class plan quantity: power draw, grid conversion, and per-stage accounting.

A GPU datacenter provisions watts before it provisions slots, so power belongs beside memory
and CPU in the neutral contract layer rather than inside any one subsystem. Kyber reads these
to prefer the placement that fits an envelope, Carbonite to refuse work that would exceed it,
Core to record what a stage actually drew, and `observe` to report it — none of them importing
another.

Three modules, one responsibility each:

* `power` — the device power model (`P(u) = idle + (tdp - idle) * u`) and `PowerEnvelope`,
  the budget/expected-draw pair a feasibility check compares.
* `carbon` — joules to money and grams CO2e via a `GridProfile`. There is deliberately no
  built-in table of regional carbon intensities; an unconfigured profile reports zero.
* `accounting` — `StageEnergy` records and the `EnergyLedger` that rolls them up into
  tokens-per-joule, rows-per-joule, and the idle fraction.
"""

from __future__ import annotations

from batcher.plan.energy.accounting import EnergyLedger, StageEnergy
from batcher.plan.energy.carbon import (
    GridProfile,
    carbon_grams,
    energy_cost,
    joules_to_kwh,
    kwh_to_joules,
)
from batcher.plan.energy.power import (
    PowerEnvelope,
    device_power_watts,
    energy_joules,
    fleet_power_watts,
    host_overhead_watts,
    max_concurrent_devices,
)

__all__ = [
    "EnergyLedger",
    "GridProfile",
    "PowerEnvelope",
    "StageEnergy",
    "carbon_grams",
    "device_power_watts",
    "energy_cost",
    "energy_joules",
    "fleet_power_watts",
    "host_overhead_watts",
    "joules_to_kwh",
    "kwh_to_joules",
    "max_concurrent_devices",
]
