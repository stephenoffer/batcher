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
| `fabric_gbps` | `0.0` | The node's aggregate RDMA rate, for a container that cannot see `/sys/class/infiniband`. 0 measures it. |
| `bind_host_to_device_numa` | `True` | Pin a GPU worker's host-side threads to the CPUs on its device's NUMA node. |
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
| `torch_expandable_segments` | `True` | Back PyTorch's allocator segments with growable virtual reservations, so a workload with varying tensor sizes stops fragmenting. |
| `torch_memory_fraction` | `True` | Cap each process at its share of its device through PyTorch's own allocator, so one stage's overrun cannot take down its co-tenants. |
| `torch_gc_threshold` | `0.0` | Share of the per-process cap past which PyTorch reclaims cached blocks proactively. `0.0` keeps the allocator's reactive default. |

These fields are the {py:class}`DeviceMemoryConfig <batcher.config.DeviceMemoryConfig>`
dataclass.

### Two allocators, two failure modes

The first five fields configure RAPIDS/RMM, which the relational GPU kernels allocate through.
The three `torch_` fields configure PyTorch's caching allocator, which is what every inference
stage allocates through. A worker routinely uses both, and they fail differently, so both are
configured before the first tensor is allocated.

PyTorch's failure mode is fragmentation. The allocator carves the device into fixed segments
and splits blocks out of them, so a workload whose tensor sizes vary, meaning mixed image
resolutions or mixed sequence lengths and therefore every real batch, leaves each segment
holding a live block too small to reuse. The job then dies at 60% VRAM with a message saying
plenty is free, and the free bytes are real and unusable at the size being asked for.
`torch_expandable_segments` is the fix, and PyTorch ships it off.

`torch_memory_fraction` matters most when several actors share a device. Without a cap, a stage
that misjudges its footprint exhausts the device and every co-tenant fails with it. With one,
the process that overran fails its own allocation and the retry recovers it. The share is
derived from `vram_headroom` and how many actors the stage packs onto the device, so it agrees
with the budget admission already reserved.

Both are skipped when `PYTORCH_CUDA_ALLOC_CONF` is already set: an operator who tuned the
allocator by hand outranks these defaults, and the settings interact, so merging would be worse
than either.

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
| `quarantine_on_remap_failure` | `True` | Take a device out of rotation when its memory row remapping has failed. |
| `drain_on_reset_pending` | `False` | Stop scheduling onto a device holding a repair that only its next reset applies. |

These fields are the {py:class}`DeviceHealthConfig <batcher.config.DeviceHealthConfig>`
dataclass. A clamped device is derated rather than removed; one reporting uncorrectable ECC
errors is quarantined outright. Absent telemetry never quarantines anything.

The same thresholds apply to AMD accelerators, which are read from the `amdgpu` driver's own
sysfs tree rather than from ROCm, so an Instinct node is judged with no ROCm install and no
`pynvml`. Two conditions are AMD's own. An unrepairable error in the memory controller is the
counterpart of an uncorrectable ECC error and quarantines the board. The same class of error
in a compute block derates it instead, because that one can come from a single bad command and
clears on a reset.

`max_temperature_c` is a ceiling rather than the whole rule. Where the driver publishes the
part's own slowdown point, the lower of the two applies, because parts clamp themselves tens
of degrees apart and one fleet-wide figure is simultaneously too strict on some and too lax on
others.

The two remapping fields cover a failure mode ECC does not. HBM repairs itself by retiring a
faulty row from a fixed pool of spares, so a device reports a *pending* repair (it applies at
the next reset, and until then the faulty row is still mapped in) or a *failed* one (the
spares are gone). A remap failure is quarantined by default because, unlike every other
condition here, it does not recover: no reset repairs it. A pending repair only degrades,
since the device is still returning correct results; turn on `drain_on_reset_pending` where a
run is long enough that "the next boundary" is hours away.

## What the fleet reports about itself

None of the settings above helps unless the conditions they describe are visible.
{py:func}`bt.accelerators() <batcher.accelerators>` reports them, and
{py:func}`bt.show_accelerators() <batcher.show_accelerators>` prints the same thing with the
silent conditions called out by device:

- a host link that renegotiated below what the slot and the card both support, with the
  fraction of nameplate bandwidth it is left with;
- memory that has repaired itself as far as it can, or is holding a repair for the next reset;
- ECC disabled, an exclusive compute mode, a power limit at the part's floor, or persistence
  mode off, each of which costs throughput or correctness without raising anything;
- how many MIG instances the device is partitioned into, or the AMD compute and memory
  partition it is in, which changes what every other figure on the row is about;
- the RDMA ports that are up, what they have carried, and what they got wrong carrying it, or
  where there is no RDMA, the Ethernet links and the rate a shuffle is priced against;
- the container limits that cost the job something: a `/dev/shm` too small to stage a worker's
  input through, a memlock ceiling that stops host memory being pinned, or a descriptor limit
  a partitioned scan will exhaust. Each names the flag that raises it.

On a Ray cluster the report also probes every accelerator node, because NVML answers only
about the host it runs on: `fleet.health` carries the nodes with a device that should not be
scheduled on.

### Checking a node before it takes work

A report is something an operator reads after a job came back slow. A deployment check runs
before the fleet takes work at all, and wants a list rather than a page:

```python
import batcher as bt

for problem in bt.accelerator_problems():
    print(problem)
```

Each entry is a complete sentence naming the device and the condition, so a failing check can
be pasted into an alert without a lookup table. Run it once on each node shape you rent.

An empty list means the node is healthy *or* that nothing could be read, and those are not the
same. {py:func}`bt.accelerators() <batcher.accelerators>` is where they are told apart, and a
check that treated an unreadable node as a broken one would fail a fleet the day a base image
stopped shipping `pynvml`.

```{eval-rst}
.. autofunction:: batcher.accelerator_problems
```

## See also

- {doc}`/user-guide/operate/gpu-fleets`: the same controls in the order you would adopt them.
- {doc}`options`: every other configuration section.
- {doc}`environment`: the `BATCHER_ACCELERATOR_*` spelling of these fields.
- {doc}`/api/operations/governance`: data residency, the placement constraint that pairs with these.
- {doc}`/ml/inference/gpu`: choosing devices and batch sizes from the pipeline side.
