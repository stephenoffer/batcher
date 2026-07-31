# Teaching Kyber the shape of its cluster

A working record of the work that let the optimizer reason about *where* its workers are, not
just how many there are. Kept in-tree for contributors and not a published page: it names gaps
and unverified areas a user-facing page should not carry.

Companion to {doc}`gpu_datacenter_program`, which made a device's own properties visible to the
control plane. This one is about the fleet those devices sit in.

## The gap this closed

Kyber planned against `HardwareProfile`, whose fields all describe the **binding node**: the
smallest VRAM, the weakest worker's cores, one aggregate device count. That is the right
contract for sizing an operator so it is valid wherever it lands, and it cannot express
anything about how work is placed relative to other work.

The consequence was concrete. These two fleets are identical on every field the optimizer could
read:

| | Fleet A | Fleet B |
|---|---|---|
| Devices | 32 | 32 |
| `gpu_count` | 32 | 32 |
| `gpu_memory_bytes` | 80 GiB | 80 GiB |
| Nodes | 4 x 8 GPUs | 32 x 1 GPU |
| Share of a hash exchange that stays on-host | 25% | 3% |
| Share that crosses NVLink | 22% | 0% |

Kyber produced the same plan for both, ranked against a network cost that was up to four times
too high on Fleet A. The topology that distinguishes them lives in `dist`, which the optimizer
may not import.

## What landed

| Piece | Where | What it does |
|---|---|---|
| `ClusterShape` / `NodeShape` | `plan/resource/cluster.py` | One record per node: cores, RAM, devices, device model, coherent-fabric width, rack, zone, power zone, egress rate, unhealthy device count. Carries no policy. |
| `LocalityShares` | `plan/resource/locality.py` | What fraction of an exchange across `W` workers stays in a worker, a NVLink domain, a host, a rack, and what crosses the network. |
| Topology projection | `dist/.../fabric/shape.py` | Reads live Ray nodes into the neutral shape. The one-way bridge that lets `kyber` see the fleet without importing `dist`. |
| Tier pricing | `kyber/cost/locality.py` | Prices each tier from measured link rates and collapses the shares into one multiplier on the `net` axis. |
| Placement preference | `kyber/cost/placement.py` | PACK vs SPREAD as a comparison against the tier saving, replacing a single absolute byte threshold. |
| Window straggler | `kyber/cost/imbalance.py` | Charges a partitioned window for a measured hot partition, which it can neither pre-reduce nor salt away. |
| Plan-cache key | `kyber/plan_cache.py` | Keyed on the fleet's *structure*, so an autoscaler swapping a node keeps its memoized plans and a genuinely different fleet gets its own. |
| Report section | `api/session/accelerators/planning.py` | `bt.accelerators()["planning"]`: the shape the optimizer saw and what it did to the cost of a shuffle. |

## The rules this follows

The same five {doc}`gpu_datacenter_program` states, with two that carried most of the weight
here:

1. **Unknown stays unknown.** Every tier whose bandwidth cannot be read is charged the network
   rate. An unlabelled fleet, a container without the host's `/sys`, a device model nothing
   recognizes, and a single-node run all produce a factor of exactly `1.0` — the flat model,
   bit for bit. The discount is only ever taken against a measured or declared rate.
2. **The discount only ever subtracts.** The tier factor is clamped to `(0, 1]` and the
   placement rule can only *add* a PACK. Neither can make an exchange look dearer than the flat
   model already charged, so the failure mode is a pessimistic plan rather than a shuffle
   nobody budgeted for.

## Two defects found while building this, and what they teach

**`worker_count` is the node count.** It is documented as "workers the plan will run across",
and `cluster_hardware_profile` sets it to `len(node_classes())`. The volume term tolerates that
— it only needs the `1/W` local discount to be roughly right — but the tier shares do not: at
one worker per node an exchange has *no* intra-node tier, so every non-local byte is charged to
the network and the entire distinction vanishes on exactly the dense fleets the model is for.
The tier shares are now priced against `ClusterShape.exchange_width()`, the fleet's schedulable
capacity. The volume term was deliberately left on `worker_count`, because changing it moves
every existing distributed ranking and that is a separate, benchmarkable change.

**A CPU worker does not use NVLink.** The first version of the share arithmetic folded every
worker on a host into that host's coherent-fabric group. For a device fan-out that is right;
for a relational shuffle it discounted intra-node traffic by ~90x on a rate that traffic never
touches. Caught by a unit test asserting the two units differ on one fleet.

## What is deliberately not priced

Each of these is a real effect that was left alone rather than modelled on a figure nobody
measured. Modelling one badly is worse than leaving the cost where it was.

- **The rack tier.** Charged at the full network rate. A rack's own fabric genuinely beats a
  spine crossing on most builds, but by how much is a deployment's oversubscription ratio,
  which this process cannot read.
- **The cost of concentrating a fleet.** Packing workers onto fewer nodes gives up their read
  bandwidth, their page cache, and their failure domain. There is no per-node read-bandwidth
  figure available here, so the trade is bounded (a gang may occupy at most a handful of nodes)
  rather than priced.
- **Skew on anything but a window.** An aggregate pre-reduces its hot key and a join is salted.
  Charging either would penalize the mechanism that removes the problem.
- **The `cache_factor` per-thread cache share.** Investigated and rejected. The parallel
  aggregate builds a hash table per morsel and the parallel join one per bucket, so the resident
  table and the available cache both divide by the worker count and the ratio cancels. The
  whole-L3 comparison is right for the partitioned paths.

## Not done

Ranked by expected value. Each is a *specification*, not a claim about the code.

1. **Heterogeneous-fleet VRAM budgeting.** `decide_gpu_backend` sizes the whole-cluster budget
   as `smallest_device x device_count`. On a fleet of 4 x 80 GiB and 8 x 40 GiB that reports
   480 GiB against a true 640 GiB, and refuses work the cluster can hold.
   `ClusterShape.aggregate_gpu_memory_bytes` already computes the honest figure; the routing
   decision has not been moved onto it.
2. **Domain-aligned device fan-out.** A shard count that is not a multiple of the coherent
   fabric width leaves a collective straddling the fabric boundary. `ClusterShape` carries the
   width; nothing rounds a fan-out to it.
3. **Per-node fabric measurement.** `NodeShape.fabric_gbps` is populated from the operator's
   declared `accelerator.fabric_gbps` only, because the projection runs on the driver and the
   driver's NIC is not the workers'. A real per-node rate needs a worker-side probe reported
   back through Core.
4. **Rail counts.** `NodeShape.rails` is always `0`. The per-node rail map exists
   (`_internal.hardware.fabric.rails`) but is a worker-side reading, same shape of gap as (3).
5. **Health-derived device counts.** `NodeShape.unhealthy_gpus` is carried and never populated;
   the fleet-health probe that could fill it lives in the report path, not the planning path.
6. **The volume term's worker count.** See the defect note above. Correcting `shuffle_bytes` to
   the true fan-out is a real improvement and moves every existing distributed ranking, so it
   needs a benchmark run rather than a unit test.
7. **A skew term for range-partitioned sorts.** A global sort range-partitions, which is as
   unsaltable as a window, and is not charged for it.

## Verification

`tests/unit/test_cluster_shape.py`, `test_cost_locality_tiers.py`,
`test_cost_placement_preference.py`, `test_cost_window_skew.py`, and
`test_accelerator_report_planning.py`. The property that matters most is asserted directly: the
shares partition the exchange (they sum to one, none negative) for every fleet shape and worker
count tested, because a set that does not silently rescales every `net` cost that reads it.

**Not benchmarked.** No multi-node cluster was available in the session that wrote this, so the
tier factors are verified for shape, monotonicity, and their degradation to the flat model, and
not against measured wall-clock on a real fleet. Item (6) above is blocked on exactly that.

## See also

- {doc}`kyber` — where the cost model's network axis is described for readers.
- {doc}`gpu_datacenter_program` — the device-level work this builds on.
