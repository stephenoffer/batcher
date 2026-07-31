# GPU fleets

This page covers running Batcher inside a GPU datacenter: sizing work against what a device
actually is, budgeting the power a job may draw, keeping a multi-device stage on the fast
interconnect, taking a sick device out of rotation, and constraining where regulated data may
be computed.

Everything here is off or unbounded by default. A deployment that configures none of it
schedules exactly as it did before, which is what makes each control safe to turn on one at a
time.

## What Batcher knows about a device

A cluster manager reports how many accelerators a node has and which model they are. It does
not report how much memory that model has, how much power it draws, how wide its coherent
interconnect is, or whether it can be partitioned. Batcher keeps those figures in a table keyed
by the same model name, and every decision below reads from it.

The figures are vendor nameplate numbers for the dense tensor path, without the structured
sparsity multiplier. They are used as ratios, so consistency of basis matters more than
absolute accuracy. An unrecognized model reports unknown rather than a default, and each
decision then falls back to whatever it did before.

```python
from batcher._internal.device_specs import device_spec

spec = device_spec("NVIDIA_H100")
print(spec.memory_gib, spec.tdp_watts, spec.nvlink_domain, spec.mig_slices)
# 80 700.0 8 7
```

## See what the fleet looks like

`bt.accelerators()` reports what this process and its cluster can see: the local devices with
their nameplate figures and any live readings, the cluster's shape, and the power envelope.
`bt.show_accelerators()` prints the same thing for a human. Keys are present only when their
source could answer, so a CPU-only host reports a backend and an empty device list rather than
a page of zeros.

```python
import batcher as bt

report = bt.accelerators()
print(report["backend"] in {"cuda", "rocm", "xpu", "mps", "tpu", "neuron", "hpu", "cpu"})
# True
print(isinstance(report["devices"], list))
# True
```

On a GPU node the device rows carry the live draw, SM utilization, temperature, and any clamp
the driver has applied. A `fleet` key appears when Ray is up and the cluster has accelerator
nodes, with the widest coherent NVLink domain and how many racks and power zones the fleet
spans.

## Whether a device is worth using at all

The question most engines answer with a heuristic. Batcher answers it with a time model, and
the term that decides it is the one a data engine is most tempted to leave out: every byte of a
relational stage crosses the host link before a kernel sees it, and on PCIe that link is slower
than a server's own memory.

```python
from batcher.kyber.gpu import device_energy_advice

scan = device_energy_advice("NVIDIA_H100", bytes_per_row=64.0, flops_per_row=4.0)
print(scan.transfer_share > 0.5)
# True
resident = device_energy_advice(
    "NVIDIA_H100", bytes_per_row=64.0, flops_per_row=4.0, resident=True
)
print(resident.speedup > scan.speedup)
# True
```

The same scan is roughly 2x on an H100 and 0.5x on a T4 — slower than the CPU outright, because
a PCIe 3.0 copy costs more than scanning the data. On a Grace-Blackwell package the coherent
host link is an order of magnitude faster and the same stage is worth offloading.

Batcher declines to route a stage to a device this model says loses, and says so in the plan's
reason. Two conditions bound that, both deliberate: the fleet's device model must be known
(an unlabelled cluster gets no opinion), and the fleet must not already have *measured* a
GPU/CPU crossover for itself. A measurement from this hardware outranks a model whose
CPU-bandwidth constant may not describe it, so once Batcher has timed both backends here it
decides on those timings instead.

Two things change the answer, and both are worth reaching for before a faster device: keeping
the data resident (a stage fed by another GPU stage pays no copy at all), and giving the stage
more work per byte. A stage below its device's roofline ridge is a copy with a kernel attached.

## Budget the power a job may draw

A rack is provisioned in watts before it is provisioned in slots. Sixteen 700 W devices need
more than eleven kilowatts of device power alone, which is more than a single rack circuit
delivers, so the breaker binds long before the slot count does. Exceeding the budget rarely
trips anything: the driver clamps every device in the zone instead, and the whole rack reads as
mysteriously slower.

Set the budget your fleet actually has, and Batcher clamps GPU fan-out to fit it:

```python
import dataclasses

from batcher import Config
from batcher.config import AcceleratorConfig, EnergyConfig

energy = EnergyConfig(power_budget_watts=10_000.0, power_headroom=0.1)
config = Config().replace(accelerator=AcceleratorConfig(energy=energy))
print(config.accelerator.energy.power_budget_watts)
# 10000.0
```

`power_headroom` is the fraction deliberately left unused. Batcher models a board's draw as a
straight line between its idle and its limit, which is close enough to compare placements and
optimistic at the top of the range, so the headroom absorbs the difference.

A budget of `0.0`, the default, means unbounded. An unrecognized device model is never clamped,
because clamping it would mean inventing the watts it draws.

## Report what a run cost

Energy is recorded per stage, and the report is where a GPU-hour becomes actionable. Wrap the
work in `bt.measure_energy()` and every accelerator stage inside records what it drew:

```python
import batcher as bt

with bt.measure_energy() as energy:
    out = bt.from_pydict({"g": [1, 1, 2], "v": [10, 20, 30]}).group_by("g").agg(
        total=bt.col("v").sum()
    ).to_pydict()

print(sorted(out))
# ['g', 'total']
print(energy.total_joules >= 0.0)
# True
```

A CPU-only run records nothing, which is why the ledger above is empty: only accelerator
stages draw device power. The figures a datacenter cares about are efficiency ratios rather
than totals: tokens per joule for a generative stage, rows per joule for a relational one, and
the share of energy spent holding devices that were not computing.

```python
from batcher.observe import format_energy_report
from batcher.plan.energy import EnergyLedger, StageEnergy

ledger = EnergyLedger()
ledger.record(
    StageEnergy("Decode#1", "NVIDIA_H100", 8, 120.0, 0.35, joules=1_900_000.0, rows=4_000_000)
)
ledger.record(
    StageEnergy(
        "Generate#2", "NVIDIA_H100", 8, 400.0, 0.92, joules=2_400_000.0, tokens=88_000_000
    )
)
print(ledger.idle_fraction() > 0)
# True
print("hottest Generate#2" in format_energy_report(ledger))
# True
```

Every stage measured from a real device reading is also folded into Batcher's learned
statistics on the way out of the block. The next run's device choice is then made against what
this fleet delivers rather than against a datasheet ratio, which matters because a datasheet
compares peak to peak and a real stage rarely is: a starved H100 can do less work per joule
than a fed A100, and no specification says so. Modelled figures are deliberately not learned
from, since folding them would teach the optimizer its own assumptions back.

The idle share is the number to act on. Above roughly a third, the pipeline is starving its
devices, and the fix is upstream: more prefetch, larger batches, or fewer devices. A faster
kernel changes nothing.

To turn joules into money and emissions, give Batcher the two facts it cannot discover, your
energy price and your grid's carbon intensity, along with the facility PUE:

```python
from batcher.plan.energy import GridProfile

grid = GridProfile(region="nordic", gco2e_per_kwh=20.0, price_per_kwh=0.05, pue=1.15)
print(round(grid.carbon_grams(3.6e6), 1))
# 23.0
```

There is deliberately no built-in table of regional carbon intensities. A grid's intensity
varies by hour, by season, and by supply contract, so a shipped default would look
authoritative and be wrong for your site. Leave `gco2e_per_kwh` at zero and Batcher reports no
emissions rather than an invented figure.

### Comparing devices, and exporting the numbers

On a mixed fleet the useful question is not what the run cost but which hardware was cheaper to
run it on, which a total cannot answer:

```python
from batcher.observe import energy_metrics, format_fleet_efficiency
from batcher.plan.energy import EnergyLedger, StageEnergy

mixed = EnergyLedger()
mixed.record(StageEnergy("A#1", "NVIDIA_H100", 8, 10.0, 0.9, joules=1000.0, tokens=90_000))
mixed.record(StageEnergy("A#2", "NVIDIA_TESLA_V100", 8, 10.0, 0.9, joules=1000.0, tokens=20_000))
print(format_fleet_efficiency(mixed).splitlines()[0].startswith("NVIDIA_H100"))
# True
```

`energy_metrics` renders the same ledger as flat `energy.*` rows for a metrics sink, alongside
whatever you already scrape. Undefined figures are absent rather than zero, so a scrape never
records an efficiency nobody measured:

```python
rows = energy_metrics(mixed)
print(rows["energy.device.NVIDIA_H100"], "tokens_per_joule" in "".join(rows))
# 1000.0 True
```

## Keep a multi-device stage on the fast interconnect

Eight devices inside one NVLink domain exchange at hundreds of gigabytes per second. The same
eight split across two hosts exchange over the network, and a collective that was an
on-package copy becomes the stage's entire runtime. Nothing fails, and nothing in the job's own
timings says why it got slower.

Batcher reads the fleet's shape from node labels and gang-schedules a collective inside one
domain when the fleet has one wide enough. Label your nodes so it can:

| Label | Meaning |
| --- | --- |
| `ray.io/accelerator-type` | Device model, which yields memory, power, and domain width |
| `batcher.io/rack` | Physical enclosure, bounding a rack-scale domain and a shared busway |
| `batcher.io/fabric` | RDMA partition; two nodes in different partitions cannot reach each other on the fast path |
| `batcher.io/power-zone` | The breaker or busway whose budget the node's draw counts against |
| `topology.kubernetes.io/zone` | Availability zone, read before Ray's own label |

An unlabeled fleet reports no rack, fabric, or power zone, and placement degrades to the
node-level decision it made before. That is deliberate: a placement hint that fires on missing
data moves work for a reason that is not there.

When a stage needs more devices than any single domain holds, Batcher still places it, and says
so in the decision log rather than pretending the collective stayed local.

`plan_collective` is where those constraints compose, and it is worth calling directly when you
are sizing a fleet rather than running on one. It applies residency, the per-zone power budget,
and the efficiency order before the fabric preference, because each of them can remove a node
the fabric would otherwise have chosen:

```python
from batcher.dist.executors.ray_runtime.fabric import GpuNodeTopology, plan_collective

fleet = (
    GpuNodeTopology("a", 8, "NVIDIA_H100", rack="r1", fabric="ib0"),
    GpuNodeTopology("b", 8, "NVIDIA_H100", rack="r1", fabric="ib0"),
)
print(plan_collective(8, fleet).strategy, plan_collective(16, fleet).spans_fabric)
# STRICT_PACK True
```

A fleet that no node survives is reported as such rather than as an unreadable topology: those
are different failures, and only one of them is a labelling problem.

## Partition a device instead of holding it

A three-gigabyte embedding model on an eighty-gigabyte device uses four percent of the memory
and the whole schedulable unit. Fractional GPU scheduling fixes the scheduling half but not the
isolation half: co-tenants share one memory space, so one task's allocation spike is every
other task's failure.

Batcher prefers a hardware partition whenever a model fits one, which gives each worker its own
memory and its own fault domain:

```python
from batcher.carbonite.accel import mig_plan

plan = mig_plan(model_gib=6.0, accelerator_type="NVIDIA_H100", concurrency=14)
print(plan.profile.name, plan.instances_per_device, plan.devices_needed)
# 1g.10gb 7 2
```

Fourteen small workers land on two devices rather than fourteen. Set `prefer_mig=False` on
`AcceleratorConfig` to keep whole devices instead. Creating the instances themselves is a
privileged driver operation your platform performs at provisioning time; Batcher plans against
them and never reconfigures a device.

## Size an inference stage by its KV cache

A language model's weights are a fixed cost paid once. The variable cost is the key/value
cache, which grows with every token of every sequence in flight, and it is what actually runs a
device out of memory. A stage told to run 256 concurrent sequences on a device that holds 40
does not run slowly: it either fails on the first full batch, or the serving engine preempts
and recomputes, which reads as a throughput regression with no error anywhere.

```python
from batcher.carbonite.accel import KvCacheBudget, kv_bytes_per_token

per_token = kv_bytes_per_token(layers=80, kv_heads=8, head_dim=128, dtype="fp16")
budget = KvCacheBudget(
    device_bytes=80 << 30,
    weight_bytes=40 << 30,
    bytes_per_token=per_token,
    context_tokens=8192,
)
print(budget.fits, budget.max_sequences > budget.sequences_at(16384))
# True True
```

Two levers move that number more than anything else. Halving the context you size for roughly
doubles concurrency, and most workloads have a long tail of short prompts that never needed the
maximum. Switching the cache to FP8 halves it again: set `kv_cache_dtype` on
`AcceleratorConfig`.

The `kv_heads` argument is the grouped count under grouped-query attention, not the attention
head count. A model with 8 KV heads against 64 attention heads has an eighth of the cache, and
that is frequently the difference between one device and four.

## Take a sick device out of rotation

At fleet scale a device rarely fails by disappearing. It stays present and gets slower or
wronger: the driver clamps its clocks because an inlet temperature rose, its power limit sits
below what the workload needs, or its memory reports an uncorrectable error, which means a
tensor that read back is not the tensor that was written.

Turn on health checking and Batcher reads live telemetry before placing accelerator work:

```python
from batcher.config import AcceleratorConfig, DeviceHealthConfig

health = DeviceHealthConfig(enabled=True, max_temperature_c=85.0)
print(AcceleratorConfig(health=health).health.quarantine_on_ecc)
# True
```

A clamped device is derated rather than removed, because the clamp is often your own power cap
working as intended and a power-bound fleet cannot afford to drop the slot. A device reporting
uncorrectable ECC errors is quarantined outright, whatever it costs in throughput.

Health checking needs `pynvml` on every worker, which is why it is off by default. Without it,
every device is assumed healthy, exactly as before.

```{important}
Absent telemetry never quarantines anything. A fleet that loses its telemetry keeps
scheduling, because the alternative is a cluster that goes offline the day a dependency stops
being installed.
```

## Diagnose a slow GPU stage

Before tuning a kernel, check whether the devices were ever busy. The most common GPU pipeline
problem is a device waiting on the stage in front of it, and the fix for that is the opposite
of the fix for a saturated one.

```python
from batcher.ml.devices import device_feed_advice

print(isinstance(device_feed_advice(), str))
# True
```

On a GPU host the sentence names the mean utilization and what it implies: below roughly 40
percent the pipeline is starving the devices and the lever is upstream, above 85 percent they
are saturated and the lever is more devices or a cheaper model. A clamped device is called out
separately, because that ceiling is the hardware rather than the feed.

`batcher.carbonite.accel.schedulable_device_count()` gives the matching count for pool sizing:
devices that passed the health verdicts, or `None` when there is no telemetry to judge from.
Size to `None` by keeping whatever count you had, since an absent probe is not evidence that a
fleet is unhealthy.

## Constrain where regulated data is computed

Storage residency is the half most systems cover: the bytes live in a bucket in a named region.
The half that breaks an obligation is compute. A job reads an EU dataset, the scheduler finds
spare capacity elsewhere because that is where the queue is shortest, and rows cross a border
inside a shuffle.

A residency catalog states which regions a dataset may be processed in, and the check runs
before placement:

```python
from batcher.governance import DataResidency, ResidencyCatalog

catalog = ResidencyCatalog(mode="advisory")
catalog.register(
    DataResidency("s3://eu-customers/", frozenset({"eu-north-1"}), "GDPR Art. 44")
)

verdict = catalog.check("s3://eu-customers/orders", "us-east-1")
print(verdict.allowed, verdict.enforced)
# False False
```

The mode is one of `RESIDENCY_MODES`. Start at `advisory`, which returns a failing
`ResidencyVerdict` your logs record while the job proceeds, so you can find every violating
placement in a real workload before anything blocks. Move to `strict` once the log is clean;
refusals then raise `AccessDeniedError` with the obligation named.

A job reading several datasets may run only where all of them may:

```python
catalog.register(DataResidency("s3://uk-customers/", frozenset({"eu-west-2"}), "UK GDPR"))
print(sorted(catalog.permitted_regions(["s3://eu-customers/a", "s3://uk-customers/b"])))
# []
```

An empty intersection is a real answer rather than an error. Those two inputs cannot be joined
in any single region, so the job has to be split.

An unregistered dataset is unrestricted. Batcher never infers a region from a bucket name or an
endpoint, because guessing a legal fact is wrong in whichever direction it errs.

Install the catalog once, at startup, with `bt.governance.set_residency`. The scheduler reads
it through `active_residency`, so a rule applies to every stage rather than only the ones that
remembered to pass it:

```python
from batcher.governance import ResidencyCatalog, active_residency, set_residency

previous = set_residency(ResidencyCatalog(mode="advisory"))
print(active_residency().mode)
# advisory
_ = set_residency(previous)
```

On a cluster, the catalog reaches the scheduler through
`batcher.dist.executors.ray_runtime.fabric.permitted_nodes`, which keeps only the accelerator
nodes whose region every input permits. A node with no region label is never filtered out, so
labelling your fleet is part of enabling the control. `residency_report` gives the before and
after device counts, which is what distinguishes a fleet narrowed by a compliance rule from one
that is merely busy.

## Requirements and limitations

- Device power, bandwidth, and interconnect figures cover the datacenter accelerators Batcher
  recognizes by model name. An unrecognized model reports unknown, and every decision falls
  back to its prior behavior rather than to a substituted figure.
- Live telemetry requires `pynvml` and a mounted driver. Without it, power reporting falls back
  to the modelled draw and health checking is inert.
- Fabric-aware placement needs node labels. Batcher cannot discover a rack or an RDMA partition
  on its own.
- MIG instances must already exist. Batcher plans against the profiles a device supports and
  never reconfigures one.
- Residency applies to the regions Batcher can see on node labels. A worker whose region is
  unlabeled is never refused, so labeling is part of enabling the control.

## See also

- {doc}`/ml/inference/gpu`: choosing devices and batch sizes for a model, from the pipeline side.
- {doc}`/user-guide/trust/governance`: the row and column half of the same policy layer.
- {doc}`/configuration/options`: every accelerator field with its default and unit.
- {doc}`/api/operations/governance`: the residency reference.
- {doc}`/deep-dives/distribution/distributed-scheduling`: how a stage becomes tasks on a cluster.
- {doc}`/user-guide/operate/observability`: the metrics sink the `energy.*` rows land in.
