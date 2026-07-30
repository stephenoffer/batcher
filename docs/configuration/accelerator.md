# Accelerator options

This page documents the `accelerator` configuration section: the facts about a GPU fleet that
Batcher cannot discover for itself. What a rack's power budget is, what a kilowatt-hour costs
here, how carbon-intense this grid is, when a device stops being worth scheduling on, and how
inference stages are sized against their KV cache.

Every default is inert. A deployment that sets none of these places work exactly as it did
before, which is what makes each control safe to turn on one at a time.
{doc}`/user-guide/operate/gpu-fleets` is the task-oriented walkthrough; this page is the field
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

## Device memory

| Field | Default | Meaning |
|-------|---------|---------|
| `allocator` | `"default"` | Allocator strategy: `default`, `pool`, `async`, or `managed`. |
| `pool_initial_fraction` | `0.5` | Fraction of reservable device memory the pool takes at startup. |
| `pool_max_fraction` | `1.0` | Fraction of reservable device memory the pool may grow to. |
| `spill_to_host` | `False` | Move columns to host memory instead of failing when the device fills. |
| `statistics` | `False` | Track allocation counts and the device high-water mark. |

These fields are the {py:class}`DeviceMemoryConfig <batcher.config.DeviceMemoryConfig>`
dataclass. The default allocator pays a synchronizing device allocation for every intermediate
column, so a chain of many small operators spends most of its time in the driver. A pool pays
that cost once and suballocates, which is the largest constant-factor lever on this page.
Reserving more up front trades a longer first allocation for no growth pauses later.

`spill_to_host` turns a class of hard out-of-memory failure into a slowdown, which is what
makes a shard that misjudged its size survivable. It is off by default so an over-large query
is loud rather than quietly slow.

```python
from batcher import Config
from batcher.config import AcceleratorConfig, DeviceMemoryConfig

memory = DeviceMemoryConfig(allocator="pool", pool_initial_fraction=0.7, spill_to_host=True)
cfg = Config().replace(accelerator=AcceleratorConfig(memory=memory))
print(cfg.accelerator.memory.allocator)
# pool
```

## Device memory

These fields decide how a GPU worker's device memory is allocated, and what happens when it
runs out. They are separate from `vram_headroom` above, which decides how much of the device
Batcher considers its own: this section decides how the allocations inside that budget are
served.

| Field | Default | Meaning |
|-------|---------|---------|
| `allocator` | `"default"` | Allocator strategy: `default`, `pool`, `async`, or `managed`. |
| `pool_initial_fraction` | `0.5` | Fraction of the device's reservable memory the pool reserves at startup. |
| `pool_max_fraction` | `1.0` | Fraction the pool may grow to. Below `1.0` leaves the remainder to a co-tenant. |
| `spill_to_host` | `False` | Let cuDF move columns to host memory rather than fail when the device fills. |
| `statistics` | `False` | Track allocation counts and the device high-water mark. |

These fields are the {py:class}`DeviceMemoryConfig <batcher.config.DeviceMemoryConfig>`
dataclass.

Unconfigured, RAPIDS asks the CUDA driver for every intermediate column a query produces, and
a driver allocation is a synchronizing call. A translated chain of a dozen operators over a
hundred shards makes thousands of them, so `allocator="pool"` is the setting with the largest
constant-factor effect on GPU query time. It is off by default because a pool reserves memory
that a co-tenant on the same device can then no longer see.

The pool is sized from what Carbonite says is reservable, which is the device's capacity less
`vram_headroom` and less whatever another process already holds. A device that cannot report
its memory gets no pool at all rather than one sized from a guess.

`managed` backs the pool with unified memory, so a working set larger than the device migrates
over the bus instead of failing. `async` uses the driver's own stream-ordered pool where the
installed RMM offers one.

Turn on `spill_to_host` where a shard occasionally misjudges its size: it turns a class of hard
out-of-memory into a slowdown. With `statistics` on, a shard that does overflow is subdivided
by the factor its own high-water mark says will clear it, rather than being halved repeatedly.

```python
from batcher import Config
from batcher.config import AcceleratorConfig, DeviceMemoryConfig

memory = DeviceMemoryConfig(allocator="pool", pool_max_fraction=0.8, spill_to_host=True)
cfg = Config().replace(accelerator=AcceleratorConfig(memory=memory))
print(cfg.accelerator.memory.allocator)
# pool
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

- {doc}`/user-guide/operate/gpu-fleets`: the same controls in the order you would adopt them.
- {doc}`options`: every other configuration section.
- {doc}`environment`: the `BATCHER_ACCELERATOR_*` spelling of these fields.
- {doc}`/api/operations/governance`: data residency, the placement constraint that pairs with these.
- {doc}`/ml/inference/gpu`: choosing devices and batch sizes from the pipeline side.
