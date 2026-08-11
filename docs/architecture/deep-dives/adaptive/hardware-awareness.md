# Hardware awareness

This page describes what the optimizer knows about the machine it is planning for, where that
knowledge comes from, and which parts of it are measured rather than assumed.

Two plans that are identical on paper can differ tenfold in practice because of the hardware
underneath them. A spill to local flash is cheap and a spill to a network volume is not; a byte
read from the page cache and a byte read from a cold object store are the same byte to a row
count and two orders of magnitude apart in reality. An optimizer that ignores this is
confidently right on one class of machine and confidently wrong on the rest.

Kyber never samples hardware itself. It reads static facts from the layer below it, and it
consumes *measurements* that Core recorded on earlier runs. That split is the architecture's
rule: Core measures, Kyber decides, Carbonite protects.

## The hardware fingerprint

Every learned parameter is scoped to a *class of machine* rather than to the fleet. The
scoping key is a short digest built from the facts that change performance materially and stay
put across reboots: CPU vendor and model, logical CPUs, physical cores, NUMA nodes, vector
width, page size, bucketed memory, L2 and L3 sizes, storage class, accelerators, and the
operating system. A node with a fabric appends its fabric class.

Three things are deliberately left out, and the reasoning generalizes:

- **The full CPU flag list.** A microcode update changes it without changing anything the
  engine can exploit, which would discard every coefficient learned on the host.
- **Exact memory bytes.** Two nodes of one instance type differ by whatever the kubelet and
  firmware reserved, so raw bytes would give every node a key of its own.
- **Load, temperature and clock speed.** Real and important, but they vary minute to minute. A
  fingerprint that changed under load would re-learn from scratch every time the box got busy,
  which is exactly when the learned values matter most.

Bucketing memory and omitting the flag list are what let near-identical nodes share a key, so a
fleet of one instance type learns once rather than a hundred times.

## What the optimizer reads

Static facts come from `_internal/hardware`, at layer 0, where both Kyber and Carbonite can
reach them without importing each other:

| Fact | Where it lands |
|---|---|
| L3 cache size | the cache-miss multiplier on probe-heavy operators |
| Storage device class | what a spilled byte costs |
| NIC and fabric link rate | what a shuffled byte costs against a local one |
| NVLink, PCIe links, peer islands | where a multi-GPU exchange is placed |
| Device model, generation, MIG profiles | which accelerator a stage should use |

The storage class is read off the block device rather than assumed from the instance type,
because it cannot be inferred: LVM over a network volume and LVM over local NVMe present the
same device prefix and are thirty times apart.

## What the optimizer measures

Three loops carry hardware behavior from one run into the next. All three are keyed by the
fingerprint, so a driver planning for workers of a different class reads the workers' history
rather than its own.

**Cost coefficients.** Core records each operator's wall time and row counts. `calibration` fits
the per-row coefficients from that history, shrunk toward the shipped defaults in proportion to
how little evidence there is, so a cold store keeps the defaults and a well-exercised one
converges on the measurement.

**CPU utilization.** Each operator family's measured CPU utilization overrides its static
per-task CPU share, so a CPU-bound family asks for a whole core and an IO-bound one packs
several per core. A family whose history shows preemption is suppressed, because low
utilization has two causes with opposite fixes and shrinking the share under contention only
deepens the contention.

**Read throughput.** Each source's measured read rate becomes a multiplier on what its bytes
cost, relative to the plan's median source. The relative form is deliberate: it can change a
*ranking* between two sources and can never re-scale the IO axis against CPU.

**CPU clamping.** Whether the machine was holding the query back, in the two ways it can. See
below, because this one exists to stop a specific wrong inference rather than to price
anything.

### The spill device, falsified

The storage class is a structural reading, and it is biased in one direction by construction: a
composite device resolves to the *slowest* class beneath it, and an NVMe namespace reached over
a fabric transport is called network storage on the strength of a transport string. Both are
right when they are right, and neither is falsifiable without a measurement, so a local RAID0
of four NVMe reported as rotational prices a spilled byte thirty times too high and every
out-of-core plan on that machine is contorted to avoid a spill the device would have absorbed.

Core already records the bytes each operator spilled and how long it ran, so the fleet's own
history can settle it. The correction runs one way only, and the reason is worth stating
because it looks like a limitation:

The only clock on the record is the operator's **whole** wall time, which includes its compute.
So spilled bytes over elapsed time is not the device's throughput, it is a *lower bound* on it.
A lower bound supports one inference and refuses the other. A high reading proves the device
moved that many bytes per millisecond, so a class claiming it is slower is wrong and the factor
may come down. A low reading proves nothing, because the operator may simply have been
compute-bound, so the factor is left alone.

That asymmetry happens to line up with the bias it corrects. The structural reading errs toward
pessimism, and pessimism is the direction a lower bound can disprove.

The corrected factor is shrunk toward the class by how much evidence exists and never falls
below the local-flash floor, so a handful of samples or one anomalous run cannot move a plan.

### When the box is holding the query back

Every measurement above is spent on making a plan cheaper. This one exists to stop a *wrong
inference*, which is a different job and worth separating.

The per-task CPU share is sized from measured utilization: a family whose cores sat idle asks
for less of a core, so several of its tasks pack onto one. That is right for a family that
never wanted the cores and exactly backwards for one whose cores were taken away, and the two
are indistinguishable from utilization alone. Getting it backwards starts a loop that feeds
itself, because a smaller reservation packs more tasks onto the contended cores, which lowers
utilization further.

Three independent measurements break the tie, and any one is sufficient:

| Signal | What it catches | What it misses |
|---|---|---|
| Involuntary context switches | Another runnable thread taking the core | A clamp that preempts nothing |
| Major page faults | The box paging against the query | Anything that is not memory |
| CPU clamping | The quota or the silicon stopping the work | Nothing the other two catch |

The third is the one added last, and it covers the case the first two structurally cannot.
**Quota throttling** dequeues a thread at the *end of a CFS period*, so at the default 100 ms
period it yields on the order of ten involuntary switches per core-second against a threshold
of two hundred, and it faults not at all. A container clamped to a third of its quota therefore
measured as a perfectly quiet box while every core it was owed sat idle by decree. Under a
container orchestrator this is the *usual* way a CPU gets clamped. **Thermal throttling** is
the other way: the silicon slowing itself because it is too hot, which preempts nothing and
faults nothing either. It is read from the CPU's own counters, as a delta since the previous
query rather than a count since boot, because a machine that throttled during last month's
heatwave is not throttling now. Those counters exist only on bare metal, so in a cloud guest
this signal reads zero and the quota one carries.

Both are compared as a *median* over the family's history, for the same reason the preemption
signal is: one clamped run inside an otherwise clear history is not a regime, and the action
gated here is conservative. It suppresses a learned CPU share and keeps the static prior, so
firing early costs one family's tuning while firing late costs the spiral.

## What the optimizer does not see

Being explicit about this matters more than the list is long, because each of these is
sometimes assumed to be in play.

- **Accelerator temperature.** GPU thermal telemetry is collected and is rich, but it reaches
  Carbonite only, where it marks a device degraded or blocklists it. It is a *health* signal
  there, never a *cost* one, so a thermally clamped accelerator is still costed at its
  nameplate rate. The CPU equivalent *is* read, but for a different purpose: see the section
  above.
- **Measured power.** Energy accounting exists and a measured work-per-joule figure feeds the
  accelerator choice, but the meter only runs inside an explicit measurement block. On the
  default path nothing records energy, so the device choice falls back to datasheet ratios.
- **Instantaneous load.** Nothing in *planning* reads current utilization, queue depth or
  available memory. Those belong to Carbonite, which reads them live to size fan-out and gate
  admission. The division is deliberate: a plan should be reproducible, and a plan that changed
  with the minute could not be compared against the one before it.
- **Vector width, as an explicit term.** No cost term reads the vector width. Two hosts with
  different vector units share a cost *shape* and differ only through what they learned. That
  is closer to right than it sounds, and the obvious fix would be wrong. The width is in the
  fingerprint and the compiled-versus-interpreted parameter is fitted per machine class, so a
  host whose silicon favours the compiled tier learns that on its own. Scaling the *cold*
  prior by vector width, which looks like the missing piece, is not: that parameter is a
  ratio between two already-vectorized paths, so a wider unit speeds up both sides of it and
  largely cancels. What a cold machine cannot know is the weighting of the CPU axis against
  the IO and network axes, which are hardware-scaled while it is not, and correcting that
  needs benchmark evidence across machine classes, not a constant.

## See also

- {doc}`The cost model </architecture/deep-dives/adaptive/cost-model>`: where these factors are
  spent.
- {doc}`Learned metadata </architecture/deep-dives/adaptive/learned-metadata>`: the Core-measures
  and Kyber-consumes loop in general.
- {doc}`Physical properties </architecture/deep-dives/query/physical-properties>`: the other
  thing a plan carries besides row counts.
