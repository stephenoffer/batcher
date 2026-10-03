# Hardware awareness

This page describes what the optimizer knows about the machine it is planning for, where that knowledge comes from, and which parts of it are measured rather than assumed.

Two plans identical on paper can differ tenfold because of the hardware underneath. A spill to local flash is cheap and a spill to a network volume isn't. Kyber never samples hardware itself: it reads static facts from the layer below it and consumes *measurements* Core recorded on earlier runs. Core measures, Kyber decides, Carbonite protects.

You can see the machine Batcher planned for at the bottom of an analyzed plan:

```python
import batcher as bt

ds = bt.from_pydict({"g": [i % 7 for i in range(4000)], "x": [float(i) for i in range(4000)]})
report = ds.group_by("g").agg(s=bt.sum("x")).explain(analyze=True)
print([line for line in report.splitlines() if line.startswith("machine:")])
# e.g. ['machine: GenuineIntel/8c/30GiB/l3=35MiB/nvme [3f80fb99c73a]']
```

That line is the CPU vendor, the cores this process may use, its memory ceiling, the L3 size, the storage class, and the hardware fingerprint in brackets.

## The facts describe the process, not the host

Every CPU, memory, and cache figure is what *this process* may use. Inside a container that is the whole problem: `os.cpu_count()` reports the host's cores, `SC_PHYS_PAGES` the host's RAM, and `/sys` every core and NUMA node on the box. Each probe reads the host source and then narrows it by whatever binds:

| Fact | What narrows it |
|---|---|
| Logical CPUs | the affinity mask (a cpuset pin), the cgroup CFS bandwidth quota, and a batch scheduler's core grant, whichever is tightest |
| Physical cores | SMT siblings collapsed within the affinity mask, then capped by the logical-CPU budget above, since a bandwidth quota narrows what the mask cannot |
| Memory ceiling | host RAM less any reserved hugepage pool, then `memory.max`, `memory.high`, a scheduler's memory grant, and `RLIMIT_AS` |
| Swap availability | the tightest `memory.swap.max` anywhere in the cgroup ancestry, since v2 enforces it at every level |
| Cache sizes and NUMA nodes | restricted to the CPUs in the affinity mask, and reported as the binding domain rather than an average |

The bandwidth quota is the one to know about. It's what Kubernetes sets for a `cpu` limit, it appears in neither the affinity mask nor `/sys`, and exceeding it throttles the whole cgroup for the rest of the period.

![How the machine's real shape reaches a plan decision, as a fold and then a fan-out. Three bounds narrow what this process really gets rather than what the host reports: the affinity mask, which is the cpuset pin; the cgroup CFS quota, which is the Kubernetes cpu limit; and a batch scheduler's grant from Slurm, PBS, LSF or SGE. The tightest bound wins, never below 1, giving the effective machine in cores, memory and devices. available_parallelism honours the affinity mask but not the bandwidth quota, so a pod capped at 15 cores on a 16-core node reports 16 and sizes every pool one thread too wide; the count is re-read every 100 ms, so a worker pinned after process start is picked up. Three unrelated decisions then fan out. The core count sets shard and pool width, at every physical core plus a third of the SMT siblings. The memory ceiling sets the memory budget and spill, from the cgroup ceiling rather than the node's advertised RAM. The device inventory, with MIG profiles and NVLink and PCIe islands, sets device placement. The same record hashes to a 12-character fingerprint, and that fingerprint scopes every learned value measured in machine units, nanoseconds, bytes and batch sizes, so unlike machines never blend their coefficients. A statement about the data is never scoped: a column has the same distinct count whatever machine reads it.](/_static/diagrams/hardware_awareness.svg)

When you want a fixed width rather than the detected one, set it:

```python
import batcher as bt

print(bt.active_config().execution.parallelism)  # 0, meaning auto-detect
four = bt.Config().replace(execution=bt.ExecutionConfig(parallelism=4))
with bt.config_context(four):
    print(bt.from_pydict({"x": [1, 2, 3]}).agg(s=bt.sum("x")).to_pydict())  # {'s': [6]}
```

## The hardware fingerprint

Every learned parameter in machine units is scoped to a *class of machine*. The key is a short digest of the facts that change performance and stay put across reboots: CPU vendor and model, logical CPUs, physical cores, NUMA nodes, vector width, page size, bucketed memory, L2 and L3 sizes, storage class, accelerators, and the operating system, plus a fabric class where one exists.

Three things are left out on purpose. The full CPU flag list changes with a microcode update. Exact memory bytes differ between two nodes of one instance type by whatever firmware reserved. Load, temperature, and clock speed vary minute to minute. Bucketing memory and omitting those lets a fleet of one instance type learn once rather than a hundred times.

## What the optimizer reads

Static facts come from `_internal/hardware`, at layer 0, where both Kyber and Carbonite can reach them:

| Fact | Where it lands |
|---|---|
| L3 cache size | the cache-miss multiplier on probe-heavy operators |
| Storage device class | what a spilled byte costs |
| NIC and fabric link rate | what a shuffled byte costs against a local one |
| NVLink, PCIe links, peer islands | where a multi-GPU exchange is placed |
| Device model, generation, MIG profiles | which accelerator a stage should use |

The storage class is read off the block device, because LVM over a network volume and LVM over local NVMe present the same device prefix and are thirty times apart. The accelerator inventory is available from Python:

```python
import batcher as bt

print(sorted(bt.accelerators()))  # ['backend', 'devices', 'power', 'site']
```

## What the optimizer measures

Four loops carry hardware behavior from one run into the next, all keyed by the fingerprint:

- **Cost coefficients.** `calibration` fits the per-row coefficients from measured operator times, shrunk toward the shipped defaults in proportion to how little evidence there is.
- **CPU utilization.** Each operator family's measured utilization overrides its static per-task CPU share, so a CPU-bound family asks for a whole core and an IO-bound one packs several per core.
- **Read throughput.** Each source's measured read rate becomes a multiplier on its byte cost, relative to the plan's median source. It can change a ranking between sources and never re-scales IO against CPU.
- **The spill device.** Spilled bytes over operator wall time is a lower bound on the device's throughput, so a high reading can prove a structurally pessimistic storage class wrong and bring its factor down. A low reading proves nothing and changes nothing. The corrected factor never falls below the local-flash floor.

:::{dropdown} Telling a contended CPU from an idle one
A family whose cores sat idle asks for less of a core. That is right for a family that never wanted the cores and backwards for one whose cores were taken away, since a smaller reservation packs more tasks onto contended cores. Three independent signals break the tie, and any one is sufficient:

| Signal | What it catches | What it misses |
|---|---|---|
| Involuntary context switches | Another runnable thread taking the core | A clamp that preempts nothing |
| Major page faults | The box paging against the query | Anything that is not memory |
| CPU clamping | The quota or the silicon stopping the work | Nothing the other two catch |

CPU clamping covers CFS quota throttling, which dequeues a thread only at the end of a period and so barely registers as a context switch, and thermal throttling, read as a delta from the CPU's own counters on bare metal. All three are compared as a median over the family's history, and firing suppresses the learned share in favor of the static prior.
:::

## What stays out of planning

- **Accelerator temperature** reaches Carbonite as a *health* signal, marking a device degraded or blocklisted. It isn't a cost term.
- **Measured power** feeds the accelerator choice only inside an explicit measurement block. Elsewhere the choice uses datasheet ratios.
- **Instantaneous load** belongs to Carbonite, which reads it live to size fan-out and gate admission. Planning stays reproducible.
- **Vector width** is in the fingerprint rather than a cost term. The compiled-versus-interpreted ratio is fitted per machine class, so a host whose silicon favours the compiled tier learns that on its own.

## See also

- {doc}`The cost model </architecture/deep-dives/adaptive/cost-model>`: where these factors are spent.
- {doc}`Learned metadata </architecture/deep-dives/adaptive/learned-metadata>`: the Core-measures and Kyber-consumes loop in general.
- {doc}`Physical properties </architecture/deep-dives/query/physical-properties>`: the other thing a plan carries besides row counts.
