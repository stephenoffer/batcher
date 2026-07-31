# The inter-GPU fabric program

A working record of the work that made Batcher schedule against the *wires* of a multi-GPU
cluster rather than against its device count. It is kept in-tree for contributors and is not a
published page: it names what is decided against what is merely reported, which a user-facing
page should not carry.

It continues `gpu_datacenter_program.md`, whose closing register named the gap this closes
first: **no Rust data-plane work**, and a fabric the control plane could describe but not
schedule against.

## The gap this closed

The datacenter program made the fleet legible: how much a device draws, whether it is healthy,
how wide its coherent domain is. What stayed invisible was the *relationship* between devices,
and every question below decides throughput while leaving correctness untouched.

| Question | Before | Now |
|---|---|---|
| Can these two devices copy to each other directly? | the bus distance only | `fabric.p2p.peer_matrix`, fabric overlaid on the bus |
| Which devices form a coherent group? | `nvlink_domain` (a model's nameplate) | `fabric.p2p.peer_islands`, from the live links |
| Which NIC does each device leave through? | nearest, asked per device | `fabric.rails.assign_rails`, node-wide and balanced |
| Is the node using all of its rails? | unavailable | `rail_imbalance`, `carbonite.transfer.rail_usage` |
| What does a byte off a *device* cost? | the node's summed port rate | `kyber.gpu.exchange.device_net_gbps` (rail share ∧ host link) |
| How should a redistribution be ordered? | one pair at a time, through the host | `carbonite.transfer.device_exchange` |
| How big should a host-side staging ring be? | unasked | `carbonite.transfer.staging` |
| What is the collective library told? | nothing; it probed | `dist.gpu.fabric.collective_env` |
| Which peer was the slow one? | unavailable | `bc-transport::peers`, `carbonite.transfer.straggler_peer` |
| How fast is each device model here? | unavailable | `kyber.gpu.adaptive.learned_device_throughput` |

## The rules this followed

The five rules from the datacenter program carry over unchanged. Two are worth restating
because every module here turns on them:

1. **Unknown stays unknown.** An unreadable rail map, an unpriced link, and an unmeasured
   device each yield "no opinion", and every consumer keeps the behavior it had. The device
   exchange refuses on an unpriced node rather than proceeding optimistically, because the
   optimism would be spent on the newer of two code paths.
2. **The lanes hold.** Kyber decides the width and the price, Carbonite plans the movement and
   the buffers, `dist` schedules and configures, `_internal` measures the wires, and `api`
   reports. The one figure two subsystems both needed (`fabric_fraction`) sits at layer 0
   where each reads the same object, rather than being pasted into both.

## What landed

| Layer | Module | What it answers |
|---|---|---|
| 0 | `_internal/hardware/fabric/p2p.py` | the fabric-over-bus class matrix, islands, bandwidth, bisection, the tightest group |
| 0 | `_internal/hardware/fabric/rails.py` | device-to-NIC assignment node-wide, imbalance, per-device rail share, stream dealing |
| 3 | `carbonite/transfer/device_exchange.py` | congestion-free exchange rounds, a fabric-ordered ring, the cost of both paths, the refusals |
| 3 | `carbonite/transfer/staging.py` | chunk at the bandwidth-delay product, ring depth, pinning, the host budget |
| 3 | `carbonite/transfer/peers.py` | per-peer throughput and the straggler diagnosis |
| 3 | `carbonite/transfer/codec.py` | the wire codec, decided against the link it will cross |
| 3 | `carbonite/transfer/locality.py` | `DEVICE_LOCAL` / `DEVICE_P2P` modes and `select_device_mode` |
| 3 | `carbonite/transfer/fabric_usage.py` | `rail_usage`: the same measurement per rail |
| 3 | `kyber/gpu/exchange.py` | what a device byte costs, and how wide a stage may fan out before it |
| 3 | `kyber/gpu/adaptive.py` | the crossover keyed by workload shape; per-device-model throughput |
| 4 | `dist/gpu/fabric/collective_env.py` | the collective library's environment, from measured wires |
| 4 | `dist/gpu/fabric/placement.py` | the device group, throughput-weighted shard dealing, the adaptive shard factor |
| 4 | `dist/executors/ray_runtime/scheduling.py` | `plan_collective`'s bundle layout, finally wired |
| 2 | `observe/fabric.py` | the node's wiring as `fabric.*` metric rows for a fleet dashboard |
| 5 | `api/session/accelerators/wires.py` | the rail, peer, and device-cost sections of the report |
| — | `crates/bc-transport/src/peers.rs` | per-peer bytes and nanoseconds on the consumer, and the FFI to read them |

### The first Rust data-plane work in this area

`bc-transport/src/peers.rs` is the answer to the previous program's "no Rust data-plane work".
It is deliberately the smallest useful thing: two counters per peer on the consumer side, folded
at the point a fetch completes, plus the fetch and retry counts that say whether a slow peer was
slow or merely redialed. The retry's failed attempt is *not* timed, because charging a dead
connection's timeout to a peer's bandwidth makes a healthy node read as the slowest wire in the
fleet.

The striped path records per shard rather than per bucket. Each shard is its own TCP flow, so
its bytes over its own duration is the per-stream rate the striping exists to multiply; timing
the joined future instead would divide the bucket by the slowest flow's wall time and report a
rate no flow achieved.

## The defects this found

**A rail map cannot be built from `throughput_delta`.** `rail_usage` first derived its rail set
from the per-port delta, which omits a port that moved nothing. On the node the function exists
to catch — one rail carrying everything, seven idle — the delta holds a single entry, and the
function reported a perfectly balanced one-rail node instead of a seven-rail imbalance. The set
now comes from the baseline, which lists every active port.

**The tightest group was chosen on the wrong axis.** `device_links.tightest_device_group` ranks
on PCIe distance alone, so on a fabric node it prefers a switch-local pair over a pair joined by
NVLink. That is a twentyfold error in the direction that looks plausible: the placement is
defensible on the topology it read, and it is not the topology the traffic uses.
`p2p.tightest_peer_group` is the fabric-aware form, and the two agree exactly where there is no
fabric.

**A per-device probe is not a node-wide answer.** `nearest_rdma_device` is correct for one
device and wrong for a node: asked independently, eight devices can all name the same NIC.

**The learned crossover pooled unlike workloads.** Named in the previous program's register as
the obvious next step, and it is worse than it sounds: the pooled value still *overrides* the
config default, so a transfer-bound projection and a narrow group-by average toward a threshold
right for neither. Both lines of a crossover now come from the same rung of the ladder — a
shaped GPU fit against a pooled CPU fit is two workloads' regressions crossed against each other.

**`plan_collective` had no caller.** The scheduler reserved `W` identical bundles and discovered
the fleet's residency rules, its power-zone budgets, and its fabric domains by the placement
failing or by the stage running slowly. It is wired now, with the refusals that keep it from
reshaping a gang it does not understand: a short plan falls back rather than reserving a partial
gang, which would succeed and then hang the stage on a world size it never receives.

## What is decided, and what is only reported

The distinction that matters most in this area, kept explicit because a reader will otherwise
assume the wrong half:

**Decided** (a call site consults it, and behavior changes):

- the GPU task's `runtime_env` carries the node's rail-aligned collective environment;
- a gang-scheduled collective reserves `plan_collective`'s bundle layout;
- a fan-out's shard factor rises with the fleet's *measured* device spread;
- the GPU/CPU crossover is read per workload shape;
- `flight_compression="auto"` picks the codec from the measured fabric, because past roughly
  25 Gb/s a compressor is the ceiling rather than the wire;
- the per-peer connection count is floored at the node's rail count, since a striped fetch can
  only use as many paths as it has flows.

**Reported** (correct, tested, taught, and consulted by the user rather than by the scheduler):

- the rail layout, the imbalance, and the peer topology in `bt.accelerators()`;
- the two wiring problems in `bt.accelerator_problems()`;
- per-rail and per-peer throughput in `ShuffleSession.stats()`;
- what a device byte costs against a host byte, and the chunk, depth, and pinning its link
  wants.

One attempt at a further consumer was **reverted, and the reason is worth recording**:
`ml.devices.device_feed_advice` was extended to name the chunk size and ring depth a starved
pipeline should feed with, which reads as an obvious improvement and is an
independence-contract violation. `ml` may import Carbonite by the layer matrix, but `core/udf`
imports `ml.inference` (the documented upward-edge debt), and `ml.inference.pipelines` imports
`ml.devices` — so an `ml.devices -> carbonite` edge makes `core -> carbonite` transitively.
`lint-imports` is what caught it. The advice keeps its previous wording, and the sizing is
reported through `api` instead, which nothing in `core` can reach.

**Available and not yet consulted by a scheduler**: `plan_exchange` / `worth_device_exchange`
and the staging planner beyond that report. Both are decisions about a *device-to-device copy*, and Batcher
does not perform one: the Arrow contract at every operator boundary is unchanged and the
framework doing the copying would carry the plan out. They are wired into the advice and the
report, and the honest statement is that the schedule they describe is not yet executed by
anything in this tree.

## What this program did **not** do

- **No measurement on a GPU.** Written and tested on a CPU-only host, so every bandwidth figure
  the tests exercise is described rather than measured, and no throughput claim appears
  anywhere in the tree. `benchmarks/gpu_backend/` is where those numbers belong once a fleet
  runs them.
- **No device-to-device copy.** See above. The exchange schedule is a plan, not an execution.
- **No GPUDirect Storage or RDMA data path.** `io.splits.gds` still only answers whether a path
  *could* be served by the DMA path. Nothing uses cuFile, and the Flight shuffle still moves
  host-resident Arrow.
- **The rail map does not steer the transport.** The connection count now respects the rail
  count, so there are at least as many flows as paths for the routing to spread over, but the
  Flight client does not bind its outbound connections to a local interface: which rail a flow
  lands on is still the kernel's decision. Binding a source address per connection is the next
  Rust change, and it is the one that turns the rail map from advice into a data path. An
  earlier `stream_rails` helper that named a rail per stream was **deleted** rather than kept:
  an entry point with no caller is documentation that looks like a feature.
- **AMD's XGMI fabric is still unread**, for the reason the previous program gave: the sysfs
  names have moved between kernel releases, and a fabric figure that is wrong is worse than one
  that is absent.
- **The collective environment is not validated against a real NCCL run.** Every variable is a
  documented knob and the values come from measured topology, but no test in this tree starts a
  collective. A deployment adopting it should compare `NCCL_DEBUG=INFO` output against the
  rail map before trusting it.

## Coverage

Twelve test modules and 208 unit tests, none requiring a GPU, a cluster, or the compiled
engine, plus eight Rust unit tests in `bc-transport`. The properties they pin are the
conservative directions — an unreadable topology, an unpriced link, an unmeasured device, a
fleet too small to have a straggler — because those are the paths that fail silently.
