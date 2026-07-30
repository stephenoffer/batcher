# Accelerator options

This page documents the `accelerator` configuration section: the facts about a GPU fleet that
Batcher cannot discover for itself. What a rack's power budget is, what a kilowatt-hour costs
here, how carbon-intense this grid is, when a device stops being worth scheduling on, and how
inference stages are sized against their KV cache.

Every default is inert. A deployment that sets none of these places work exactly as it did
before, which is what makes each control safe to turn on one at a time.
{doc}`../user-guide/gpu-fleets` is the task-oriented walkthrough; this page is the field
reference.

## Placement and sizing

| Field | Default | Meaning |
|-------|---------|---------|
| `vram_headroom` | `0.15` | Fraction of each device held back for the CUDA context, fragmentation, and activation peaks. |
| `fabric_aware_placement` | `True` | Report when a gang-scheduled collective is wider than any NVLink domain the fleet has. |
| `prefer_mig` | `True` | Prefer a hardware partition over a whole device when a model fits one. |
| `efficiency_first_placement` | `False` | On a mixed fleet, prefer the most throughput-per-watt device that fits rather than the smallest. |
| `kv_cache_dtype` | `"fp16"` | KV-cache element type used to size inference concurrency. `"fp8"` halves the cache. |
| `kv_cache_headroom` | `0.1` | Fraction of a device left free when sizing an inference stage. |
| `max_context_tokens` | `0` | Context length to size the cache for. 0 means the model's own maximum. |

These fields are the {py:class}`AcceleratorConfig <batcher.config.AcceleratorConfig>`
dataclass, with the two nested sections below.

## Energy

| Field | Default | Meaning |
|-------|---------|---------|
| `power_budget_watts` | `0.0` | Watts the job may draw. 0 is unbounded; a budget clamps GPU fan-out below the slot count. |
| `power_headroom` | `0.1` | Fraction of the budget left unused, absorbing the gap between the linear power model and a real board. |
| `carbon_intensity` | `0.0` | Grid intensity in grams CO2e per kWh. 0 means unconfigured and reports no emissions. |
| `price_per_kwh` | `0.0` | Energy price. 0 means unpriced. |
| `pue` | `1.0` | Facility power usage effectiveness. 1.0 accounts for the servers only. |
| `region` | `""` | Site identifier carried into energy reports. |
| `renewable_fraction` | `0.0` | Carbon-free share of supply, for reporting only. |
| `accounting` | `True` | Record per-stage energy into the run's ledger. |
| `telemetry_interval_s` | `5.0` | Seconds between device telemetry samples during a stage. |

These fields are the {py:class}`EnergyConfig <batcher.config.EnergyConfig>` dataclass. There
is deliberately no built-in table of regional carbon intensities: a grid's intensity varies by
hour, season, and supply contract, so a shipped default would be authoritative-looking and
wrong for your site.

```python
from batcher import Config
from batcher.config import AcceleratorConfig, EnergyConfig

energy = EnergyConfig(power_budget_watts=10_000.0, pue=1.15)
cfg = Config().replace(accelerator=AcceleratorConfig(energy=energy))
print(cfg.accelerator.energy.power_budget_watts)
# 10000.0
```

## Device health

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `False` | Consult live device telemetry before placing accelerator work. Needs `pynvml` on every worker. |
| `quarantine_on_ecc` | `True` | Take a device out of rotation when it reports an uncorrectable ECC error. |
| `max_temperature_c` | `87.0` | Temperature above which a device is treated as degraded. |
| `quarantine_below_derate` | `0.3` | Derate at or below which a degraded device stops being scheduled. |
| `max_memory_fraction` | `0.95` | Resident memory fraction above which a device is treated as full. |

These fields are the {py:class}`DeviceHealthConfig <batcher.config.DeviceHealthConfig>`
dataclass. A clamped device is derated rather than removed; one reporting uncorrectable ECC
errors is quarantined outright. Absent telemetry never quarantines anything.

## See also

- {doc}`../user-guide/gpu-fleets`: the same controls in the order you would adopt them.
- {doc}`options`: every other configuration section.
- {doc}`environment`: the `BATCHER_ACCELERATOR_*` spelling of these fields.
- {doc}`../api/governance`: data residency, the placement constraint that pairs with these.
- {doc}`../ml/gpu`: choosing devices and batch sizes from the pipeline side.
