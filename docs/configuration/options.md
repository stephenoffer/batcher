# Configuration options

This page is the field-by-field reference for `Config`. Each section below corresponds to one attribute of `Config`, such as `config.execution` or `config.memory`. Defaults are the engine's tuned constants. To change a field, derive a new section with `dataclasses.replace`.

Sections appear in the order they're declared on `Config`. Within a section, the fields you're most likely to reach for come first, then the power-user thresholds.

```python
import dataclasses
from batcher import Config

base = Config()
cfg = base.replace(
    execution=dataclasses.replace(base.execution, morsel_rows=8192),
)
print(cfg.execution.morsel_rows)
# 8192
```

## execution

How work is sized and parallelized.

| Field | Default | Meaning |
|-------|---------|---------|
| `parallelism` | `0` | Worker threads. `0` means use all available cores. |
| `morsel_rows` | `16384` | Rows per morsel (the unit of vectorized, scheduled work). Shipped to the Rust data plane. |
| `morsel_bytes` | `1048576` (1 MiB) | Byte budget per morsel. A morsel splits at whichever bound (rows or bytes) trips first, so wide/variable-width data stays memory-bounded. Shipped to the data plane. |
| `split_bytes` | `134217728` (128 MiB) | Target byte size of one file split, so source readers never materialize a whole large file at once. |
| `cpus_per_task` | `1.0` | CPU shares requested per distributed Ray task. A heavy native op can ask for more. |
| `cpu_share_io` | `0.5` | CPU shares a CPU-light / IO-bound distributed stage (scan, filter, project, write) requests. It sits below `1.0` so such tasks pack more than one per core. This is a cold-start prior only. Once a query runs, Kyber overrides it with each operator's measured CPU utilization. Distributed path only. |
| `cpu_share_min` | `0.25` | Floor for the adaptive per-task CPU share, so an IO-bound stage never asks for an unschedulable sliver of a core. |
| `adaptive_morsel_sizing` | `True` | Shrink the per-morsel (rows, bytes) target under memory pressure so the streaming working set stays bounded. Result-invariant; the static target is used unchanged until the pressure monitor reports elevated. Set `False` to pin the static target. |
| `fuse_linear` | `True` | Fuse chains of linear streaming operators (filter/project) into one pass over the input morsels instead of a dispatch and buffer per operator. Result-invariant; engages only on a chain of two or more fusable ops. |
| `max_concurrent_queries` | `0` | Queries admitted at once; further arrivals queue. `0` is unbounded and is a true bypass, not a large limit. Above `0`, each admitted query also requests a narrower worker pool (`cores // running`), so N concurrent queries don't each ask for the whole machine. See {doc}`/user-guide/trust/hardening`. |
| `admission_queue_depth` | `1000` | Queries allowed to wait for a slot. A further arrival raises `AdmissionTimeout` rather than joining an unbounded queue, because a queue nobody drains is an outage that presents as slowness. |
| `admission_timeout_s` | `0.0` | Seconds a query waits for a slot before raising `AdmissionTimeout`. `0` waits indefinitely. |
| `shrink_output_dtypes` | `False` | Re-narrow a pass-through output column back to its source numeric width, such as `Int32` ids widened on input, halving its footprint. It's lossless but data-dependent, so it's off by default. With it off, output types match `Dataset.schema` exactly. |

The remaining execution fields are **power-user performance thresholds**: they tune *how* the parallel executor runs an operator and are result-invariant (a query produces the identical result at any setting). Each default equals the Rust constant it replaced, so leaving them untouched is bit-identical to the engine's tuned baseline. Reach for them only to tune a known hot path.

| Field | Default | Meaning |
|-------|---------|---------|
| `bloom_fp_rate` | `0.01` | False-positive rate for the hash-join probe-side bloom pre-filter. |
| `bloom_min_build_rows` | `65536` | Build-row floor above which the probe bloom pays for itself. |
| `window_parallel_row_threshold` | `32768` | Window row count above which per-partition sorts run across cores. |
| `radix_parallel_threshold` | `0` | Partial-row count above which aggregate `combine` regroups via parallel hash-radix partitioning. `0` derives it from the machine (partitions × 256), so the crossover scales with the core count; a positive value pins it. Performance only, never a different result. |
| `sort_merge_fanin` | `16` | Maximum runs merged per pass in the external (spilling) sort's k-way merge. |
| `skew_bucket_factor` | `4` | A join bucket is "hot" when it exceeds this multiple of the average bucket. |
| `skew_min_bucket_rows` | `65536` | Absolute row floor below which a join bucket is never treated as skewed. |
| `skew_min_bucket_bytes` | `4194304` (4 MiB) | Absolute byte floor below which a join bucket is never treated as skewed. |

## memory

Buffer-pool envelope and the out-of-core spill story.

Setting `max_memory_bytes` is what opts the in-memory engine into spilling. The data plane receives a per-operator spill budget of `max_memory_bytes` times `hard_limit`, and the Rust runtime memory pool spills any stateful operator that exceeds it rather than letting the process run out of memory. That covers aggregate, distinct, sort, join, and windowed-by-partition. Leave it `None`, the default, to run fully in memory at the lowest overhead. See the bounded-memory recipe in {doc}`profiles`.

| Field | Default | Meaning |
|-------|---------|---------|
| `soft_limit` | `0.85` | Throttle new allocations at this fraction of the envelope. Must satisfy `0 < soft_limit <= hard_limit <= 1`. |
| `hard_limit` | `0.90` | Spill to disk at this fraction; also scales the data-plane spill budget derived from `max_memory_bytes`. |
| `max_memory_bytes` | `None` | Hard memory cap in bytes. `None` runs fully in memory (no spill); set it to bound memory (honoring a container/cgroup limit) **and enable spilling**. |
| `default_total_bytes` | `8589934592` (8 GiB) | Fallback total RAM assumed when `max_memory_bytes` is unset and the OS reports no usable figure. |
| `spill_dir` | `None` | Scratch directory for spill files. `None` uses a per-query temp dir. |
| `spill_remote_uri` | `None` | fsspec URL, such as `s3://` or `gs://`, that the local spill tier overflows to, so a PB-scale spill doesn't die when local disk fills. |
| `spill_local_budget_bytes` | `None` | Local spill-tier capacity before overflowing to `spill_remote_uri`. |
| `spill_compression` | `"auto"` | Arrow-IPC codec for spilled batches. `"auto"` lets the engine choose. `"lz4"`, `"zstd"`, or `None` force it. |
| `spill_bucket_max_bytes` | `134217728` (128 MiB) | A spilled aggregate bucket larger than this is re-partitioned (grace recursion) so a skewed key set degrades gracefully instead of OOMing the reduce. |
| `unbounded_memory` | `False` | Opt out of the auto-sensed spill budget and keep the fully in-memory fast path, with no out-of-core spilling. Set it when you'd rather a query fail fast than spill to disk. |
| `result_cache_max_bytes` | `268435456` (256 MiB) | Byte budget for the process-wide result cache backing `Dataset.cache()`, described in {doc}`/user-guide/operate/performance`. Cached Arrow results are held LRU and evicted to stay within this, so caching never grows the process without bound. |
| `streaming_state_max_bytes` | `0` | Cap on one streaming operator's in-memory state (window partials, dedup keys, join buffers). Exceeding it raises a clear `ResourceError` (a stalled-watermark signal) instead of OOMing. `0` derives the cap from the hard memory budget. |

## flow_control

Credit-based backpressure for the shuffle, the Carbonite flow-control model.

| Field | Default | Meaning |
|-------|---------|---------|
| `default_credits` | `16` | In-flight batch slots when an operator has no estimate. One credit is one buffered batch. |
| `credit_ceiling_factor` | `4` | Maximum credit window is `default_credits * credit_ceiling_factor`. |
| `credit_byte_budget` | `268435456` (256 MiB) | Byte ceiling for one shuffle channel's credit window, so wide rows can't buffer GBs even within the count ceiling. |
| `shuffle_fan_in` | `8` | Maximum inbound streams a shuffle node fans in before the reduce becomes a tree of combiner stages. |
| `aimd_alpha` | `1` | Additive increase: credits added per round trip. |
| `aimd_beta` | `0.5` | Multiplicative decrease applied on congestion. |
| `backpressure_high` | `0.70` | Buffer occupancy at which the producer is throttled. |
| `backpressure_low` | `0.40` | Buffer occupancy at which the producer resumes. |

These fields are the {py:class}`FlowControlConfig <batcher.FlowControlConfig>`
dataclass (the API reference lists them all). Construct one and swap it onto
`Config` to retune the shuffle:

```python
from batcher import Config, FlowControlConfig

cfg = Config().replace(flow_control=FlowControlConfig(default_credits=8))
print(cfg.flow_control.default_credits)
# 8
```

## optimizer

Kyber's planning thresholds, cost model, and learned-stats behavior. This section
nests three sub-sections: `cardinality`, `cost_coeffs`, and `cost_weights`.

| Field | Default | Meaning |
|-------|---------|---------|
| `join_dp_max_tables` | `12` | At or below this many joined tables, use exact DP join ordering. |
| `greedy_max_tables` | `25` | Above `join_dp_max_tables` and up to this, use the greedy heuristic. |
| `reoptimize_error` | `2.0` | Re-optimize when `abs(actual - estimate) / estimate` exceeds this. |
| `target_rows_per_task` | `4000000` | Target rows per distributed task; worker fan-out tracks data size, not CPU count. |
| `fixpoint_iterations` | `8` | Maximum rewrite-phase iterations before bailing. |
| `row_bytes` | `64` | Per-row footprint used by the memory-budgeting estimate. |
| `learning_smoothing_alpha` | `0.5` | Exponential-smoothing factor toward new observations. |
| `cost_calibration_min_samples` | `20` | Minimum measured samples before a cost coefficient is calibrated from runtime. |
| `cost_calibration_clamp` | `10.0` | A calibrated coefficient stays within this factor of its default, so noise cannot produce a degenerate model. |
| `quantile_probs` | `(0.0, 0.25, 0.5, 0.75, 1.0)` | Quantile grid collected for histogram-based selectivity. |
| `build_bloom_index` | `False` | On write, build a per-column membership bloom index so a later read can data-skip an equality / `IN` predicate whose value is absent. Opt-in (~1.2 MB per million rows per column). |
| `target_bytes_per_task` | `268435456` (256 MiB) | Target bytes per distributed task; partition counts take the max of the row- and byte-derived fan-out, so a few wide rows (videos, embeddings) still shard finely enough to fit memory. |
| `broadcast_max_bytes` | `0` (auto) | Build-side byte threshold below which a join is broadcast, meaning replicated to every worker, rather than shuffled. This is the analog of Spark's `autoBroadcastJoinThreshold`, but it's sized to cache rather than to memory, because a broadcast join wins only while its one hash table stays cache-resident. `0` detects the threshold from the last-level cache. A positive value pins it. Read the effective value from `resolved_broadcast_max_bytes`. The runtime guard falls back to a shuffle if the materialized build side exceeds the threshold. |
| `cardinality` | `CardinalityConfig()` | Selinger-style fallback selectivities (sub-section below). |
| `cost_coeffs` | `CostCoefficients()` | Per-unit operator costs (sub-section below). |
| `cost_weights` | `CostWeights()` | Relative weight of CPU, IO, and network when collapsing cost to a scalar (sub-section below). |

This section is the {py:class}`OptimizerConfig <batcher.OptimizerConfig>` dataclass
(its full field list, including the three nested sub-sections, is in the API
reference). Construct one and swap it onto `Config`:

```python
from batcher import Config, OptimizerConfig

cfg = Config().replace(optimizer=OptimizerConfig(join_dp_max_tables=8))
print(cfg.optimizer.join_dp_max_tables)
# 8
```

### optimizer.cardinality

Fallback selectivities used before anything is learned, superseded by learned and
sketch-based values when present.

| Field | Default | Meaning |
|-------|---------|---------|
| `unknown_rows` | `1e12` | Row count assumed for a source of unknown size, so an unknown side is never chosen as the smaller build side. |
| `default_filter_selectivity` | `0.5` | Fraction of rows assumed to pass an unmodeled filter. |
| `eq_selectivity` | `0.1` | Selectivity of `col = literal`. |
| `range_selectivity` | `0.3333...` (1/3) | Selectivity of `col < / <= / > / >= literal`. |
| `null_selectivity` | `0.05` | Selectivity of `col IS NULL`. |

### optimizer.cost_coeffs

Per-row and per-byte work units, comparable across operators.

| Field | Default | Meaning |
|-------|---------|---------|
| `scan_row` | `1.0` | Cost per scanned row. |
| `filter_row` | `0.5` | Cost per filtered row. |
| `project_row` | `0.3` | Cost per projected row. |
| `hash_build_row` | `2.0` | Cost to insert a row into a hash table. |
| `hash_probe_row` | `1.0` | Cost to probe a hash table per row. |
| `output_row` | `0.5` | Cost per emitted row. |
| `sort_row` | `1.0` | Per-row sort cost, multiplied by `log2(n)`. |
| `distinct_row` | `2.0` | Cost per row for distinct. |
| `union_row` | `0.2` | Cost per row for union. |
| `map_row` | `5.0` | Cost per row for an opaque UDF, assumed expensive. |
| `bytes_per_row` | `64.0` | Rough row width used for the IO and network axes. |

### optimizer.cost_weights

How the cost axes combine into one scalar.

| Field | Default | Meaning |
|-------|---------|---------|
| `cpu` | `1.0` | Weight of the CPU axis. |
| `io` | `1.0` | Weight of the IO axis. |
| `net` | `2.0` | Weight of the network axis; shuffle bytes hurt more than local bytes. |

## pid

Gains for the adaptive batch-size controller, a PID loop over relative latency error
that grows or shrinks the per-batch row count toward a target latency. Shipped to
the Rust data plane so the Python and Rust controllers never drift.

| Field | Default | Meaning |
|-------|---------|---------|
| `kp` | `0.4` | Proportional gain. |
| `ki` | `0.05` | Integral gain. |
| `kd` | `0.1` | Derivative gain. |
| `integral_clamp` | `5.0` | Anti-windup bound on the integral term. |
| `max_step_fraction` | `0.5` | Cap on a single step's size change (plus or minus 50%). |

These gains are the {py:class}`PIDConfig <batcher.PIDConfig>` dataclass (the full
field list is in the API reference). Construct one and swap it onto `Config`:

```python
from batcher import Config, PIDConfig

cfg = Config().replace(pid=PIDConfig(kp=0.6))
print(cfg.pid.kp)
# 0.6
```

## metadata

Where learned statistics (the MetadataHub) live and how fast confidence decays.

| Field | Default | Meaning |
|-------|---------|---------|
| `backend` | `"in_process"` | Storage backend: `"in_process"`, `"sqlite"`, `"redis"`, or `"object_storage"`. |
| `uri` | `None` | Connection or path for a non-in-process backend. |
| `decay_per_day` | `0.1` | Daily confidence decay for learned stats (roughly a one-week half-life). |

These fields are the {py:class}`MetadataConfig <batcher.MetadataConfig>` dataclass
(the full field list is in the API reference). Construct one and swap it onto
`Config`. For example, to persist learned stats across restarts:

```python
from batcher import Config, MetadataConfig

cfg = Config().replace(metadata=MetadataConfig(backend="sqlite"))
print(cfg.metadata.backend)
# sqlite
```

## governance

Whether the row filters and column masks in a `SecurityCatalog` are advisory or mandatory.

| Field | Default | Meaning |
|-------|---------|---------|
| `mode` | `"off"` | `"off"`, `"advisory"`, or `"strict"`. |
| `default_deny` | `False` | Deny a table no grant mentions, rather than leaving it ungoverned. |
| `audit_path` | `None` | Append every governance decision to this JSONL file. |
| `require_verified_principal` | `False` | Refuse a principal that was asserted rather than established by a verifier. |

These fields are the {py:class}`GovernanceConfig <batcher.GovernanceConfig>` dataclass.
Reach for `"advisory"` before `"strict"`: it warns instead of refusing, which is how you
find the ungoverned reads in a live workload. {doc}`/user-guide/trust/hardening` walks through
the migration.

## tenant

Which tenant a scope's work belongs to, so two workloads in one process do not share the
process-global caches and learned statistics by accident.

| Field | Default | Meaning |
|-------|---------|---------|
| `tenant_id` | `""` | Names the tenant. Empty means no tenancy: everything behaves as before. |
| `cache_share` | `0.0` | Share of the result-cache budget this tenant may hold. 0 is unbounded. |
| `max_concurrent_queries` | `0` | Concurrency cap for this tenant. 0 is unbounded. |

These fields are the {py:class}`TenantConfig <batcher.TenantConfig>` dataclass. Set them
with the {py:func}`bt.tenant <batcher.tenant>` scope, not by replacing the section:

```python
import batcher as bt

with bt.tenant("analytics", max_concurrent_queries=4):
    print(bt.active_config().tenant.tenant_id)
# analytics
```

A tenant is a cooperating workload, not an adversary: two in one process share an address
space. See {doc}`/user-guide/trust/hardening`.

## distributed

How the engine attaches to a Ray cluster, shuffles across it, and stays correct
through node and task failures. Ray is used for scheduling only; bulk shuffle data
moves over Arrow Flight, bypassing the Ray object store. See the **fault-tolerant
cluster recipe in {doc}`profiles`.

| Field | Default | Meaning |
|-------|---------|---------|
| `ray_address` | `None` | Ray cluster address. `None` attaches to a running cluster when `RAY_ADDRESS` is set, or when Batcher detects a managed control plane such as Anyscale. A distributed query on a managed workspace therefore fans out across the cluster with no configuration instead of stranding on a local single-node Ray. It falls back to a local start only when no cluster is reachable. Set an explicit address to override. |
| `namespace` | `"batcher"` | Ray namespace for batcher's shuffle actors, so they are isolatable. |
| `runtime_env` | `None` | `runtime_env` dict shipped to workers so `batcher` + its native extension are present cluster-wide. |
| `transport` | `"auto"` | Shuffle transport. `"auto"` picks Flight on a multi-node cluster, disk on a single node / shared filesystem; `"flight"`/`"disk"` force it. |
| `shared_filesystem` | `False` | True when every worker shares a filesystem at the same path, so the disk shuffle is safe cluster-wide. |
| `dashboard` | `False` | Show the Ray dashboard. |
| `tls` | `ShuffleTlsConfig()` (off) | TLS/mTLS for the inter-node Arrow Flight shuffle. Sub-section below. |
| `adaptive_credits` | `True` | AIMD shuffle credits: the window grows and shrinks per remote fetch from observed memory backpressure instead of holding the static grant. Flow control only, so the merged output is unchanged. `False` pins the static `default_credits` window. |
| `runtime_bloom_join` | `"auto"` | Build a bloom from the join build side and push it to the probe side to drop non-matching rows before they shuffle, cutting network volume for a selective fact-to-dimension join. `"auto"` engages only when Kyber estimates the probe is much larger than the build. `True` always engages it and `False` never does. Inner and semi joins only. |
| `shared_memory_transfer` | `True` | Same-node shared-memory shuffle: a mapper mirrors each bucket to a memory-mapped Arrow-IPC file, using Linux `/dev/shm` when available, that a same-node reducer reads via mmap with no gRPC. It's pressure-gated, so it's skipped when the node is tight on memory, and best-effort. A miss falls back to Flight, which is bit-identical. |
| `locality_aware_scheduling` | `True` | Host a reducer whose bucket concentrates on one node on that node, turning the bulk of its fetches into same-node hits. Result-preserving; pays off on a multi-node cluster with a skewed / co-partitioned shuffle. A single-node fleet resolves to "nothing to place" from the worker addresses alone, with no remote call. |
| `persistent_fleet` | `False` | Reserve one placement group and worker fleet for a whole adaptive multi-stage query, keeping each stage's intermediate partitioned on the workers instead of collecting to the driver. Removes per-stage placement churn and the driver funnel. |
| `resilience` | `"default"` | Named fault-tolerance profile. `"default"` keeps the conservative budgets below; `"spot"` hardens them as a bundle (more restarts / recompute, keepalive on, one speculative backup) for a churning spot-node cluster. Explicit knobs override the profile. See {doc}`../architecture/fault-tolerance`. |

This section is the {py:class}`DistributedConfig <batcher.config.config.DistributedConfig>`
dataclass (the API reference lists every field). Construct one and swap it onto `Config`. For example, to isolate a job's shuffle actors in their own Ray namespace:

```python
from batcher import Config
from batcher.config import DistributedConfig

cfg = Config().replace(distributed=DistributedConfig(namespace="nightly-etl"))
print(cfg.distributed.namespace)
# nightly-etl
```

### Fault tolerance

The first line of defense is Ray-level retries; beneath it, a lost shuffle worker's
output is recomputed from its (durable) source partition and re-fetched.

| Field | Default | Meaning |
|-------|---------|---------|
| `task_max_retries` | `2` | Ray reruns a failed shuffle task this many times. Shuffle tasks are deterministic and recomputed from a durable source, so a rerun is safe. |
| `retry_on_transient` | `True` | Extend task retries to application exceptions (not just worker death). |
| `actor_max_restarts` | `1` | Respawn a crashed compute actor (the map/inference pool) this many times. |
| `actor_max_task_retries` | `1` | Rerun an in-flight actor call on the respawned actor this many times. |
| `recovery_max_attempts` | `3` | Recompute→retry rounds before a still-broken shuffle fails loudly. A larger/flakier cluster may want more. |
| `recovery_backoff_base_s` | `0.5` | Base of the exponential backoff slept between recovery rounds (`0` disables the sleep). |
| `flight_idle_timeout_s` | `60.0` | Max gap between batches in a shuffle fetch before the peer is treated as dead. Generous so a long GC pause is not misread as death; bounded so a truly dead peer is detected and recomputed. |
| `flight_keepalive_s` | `None` | HTTP/2 keepalive ping interval. `None`/`0` disables it; set it to detect a silently-dropped connection faster than the idle timeout. |
| `placement_timeout_s` | `60.0` | How long gang-scheduling waits for a worker placement group before falling back to default scheduling (a real cluster may need to autoscale up). |
| `speculation_max_backups` | `1` | Max concurrent speculative backup tasks at a shuffle barrier. One backup catches the single worst straggler without letting a uniformly slow stage spawn a backup per task. `0` disables straggler speculation, making the barrier a plain wait. |
| `speculation_straggler_factor` | `1.5` | Back up a task whose elapsed time exceeds this multiple of the median finished task's time. Batcher also learns this factor per operator family from measured task-time variance, so a stage that finishes uniformly raises its own bar. |
| `speculation_min_finished_frac` | `0.75` | Fraction of tasks that must finish before speculation starts. |
| `skew_join_salt` | `0` | Spread a hot join key's rows across this many reducers (`0` disables skew-aware salting). |
| `skew_join_fraction` | `0.10` | A value is "hot" when it exceeds this fraction of a side's rows. |
| `shuffle_token` | `None` | Shared secret authenticating Flight shuffle fetches. Also read from `BATCHER_SHUFFLE_TOKEN`. |
| `shuffle_port_range` | `None` | `(min, max)` the Flight shuffle listener may bind, instead of an OS-ephemeral port. Also read from `BATCHER_SHUFFLE_PORT_RANGE` (`"40000-40100"`). |

### Restricted networks

The defaults assume nodes can reach each other freely, which is true on a normal cloud VPC and often false on-premises. Two knobs cover firewalls, multi-homed hosts, and NAT.

`shuffle_port_range` confines the Flight listener to a range you can open in a firewall rule. By default each worker takes an ephemeral port, which never collides but obliges you to open the entire ephemeral range node-to-node. Make the range at least as wide as the number of workers that share a node. A worker that can't find a free port fails with an error naming the range rather than binding somewhere unreachable.

```bash
export BATCHER_SHUFFLE_PORT_RANGE=40000-40100
```

`BATCHER_ADVERTISE_HOST` overrides the address a worker advertises to its peers. Batcher uses the node IP Ray reports, which is correct almost everywhere. Set this when the address peers must dial differs, such as on a multi-homed host whose shuffle belongs on a second network interface, or on a NAT'd or VPC-peered network. Set it per node, in the pod spec or the node environment, because the right value differs on each one.

### Cluster saturation and autoscaling

Out of the box the distributed engine fills the whole cluster with no tuning. It attaches to the running cluster, even on a managed workspace that exports no `RAY_ADDRESS`. It fans out to one worker per node, gives each worker an even share of that node's cores so morsel parallelism saturates every core, and scales the shuffle reducer count with the worker count. On an autoscaling-capable cluster it also asks the autoscaler for the cores a query wants, then waits a bounded time for the new nodes to arrive before sizing the fan-out. A big query therefore runs on the grown cluster instead of clamping to the pre-scale size and leaving the new capacity for the next job.

The autoscale wait auto-enables when Batcher detects an autoscaling cluster, meaning Anyscale, a spot node, or `BATCHER_AUTOSCALE=1`. It stays off on a fixed or single-node cluster, so the default needs no configuration. Override any of it explicitly with the fields below.

| Field | Default | Meaning |
|-------|---------|---------|
| `autoscale_wait_s` | `-1.0` (auto) | Seconds to wait for autoscaler-launched nodes before sizing the fan-out. `-1` resolves to a bounded wait on an autoscaling cluster and to `0`, meaning off, on a fixed one. `0` disables it even on an autoscaling cluster. A positive value caps the budget. |
| `autoscale_poll_s` | `5.0` | Poll interval while waiting for capacity to arrive. |
| `autoscale_stall_s` | `90.0` | Grace window. Give up the wait once capacity has been flat this long, which happens on a fixed cluster or when spot capacity can't be had, so the wait never blocks the whole budget on nodes that won't arrive. Any capacity gain resets it. Sized longer than a node's boot time. |
| `max_shuffle_partitions` | `2048` | Cap on shuffle reducers, so the reducer count scales with the cluster but an all-to-all exchange stays bounded (not O(nodes²)) at thousands of nodes. `0` disables the cap. |

The worker fan-out itself isn't a `Config` field. Pass `num_workers=` to the terminal call, such as `ds.collect(num_workers=16)`, to pin it and skip the wait. Leave it unset and Batcher auto-sizes to one worker per node on a multi-node cluster, or to all cores on a single node.

The `BATCHER_AUTOSCALE` environment variable is authoritative in both directions. `1` forces the wait on and `0` forces it off, even on a managed cluster. The wait is pure scheduling, so the result is identical whether it waits or not.

### distributed.tls

The shuffle carries query data straight between worker processes, including columns a governance policy has already decrypted or masked. On a network you don't fully control, encrypt it. `config.distributed.tls` is a
{py:class}`ShuffleTlsConfig <batcher.config.config.ShuffleTlsConfig>`: the fields are **paths**
to PEM material your platform already mounts on every worker (a Kubernetes secret volume,
cert-manager, a cloud private CA). Batcher reads them at worker start and issues no
certificates itself.

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `False` | Turn on TLS for the Flight shuffle. Off means a plaintext shuffle, which is the right default only on a trusted network. |
| `ca_cert_path` | `""` | The CA a peer's certificate must chain to. It's the trust root in both directions, because one cluster CA usually signs both server and client certificates. |
| `server_cert_path` | `""` | This node's server certificate, presented on its Flight port. |
| `server_key_path` | `""` | The private key for `server_cert_path`. |
| `client_cert_path` | `""` | This node's client certificate, presented under mTLS when fetching from a peer. Empty means outbound connections are server-auth only. |
| `client_key_path` | `""` | The private key for `client_cert_path`. Set together with the certificate or not at all. |
| `require_client_auth` | `False` | mTLS: verify a client certificate on every incoming fetch, so a process that can merely reach the port cannot pull shuffle data. |
| `server_name` | `"batcher-shuffle"` | The name checked against a peer certificate's SAN. Peers are dialed by address, so the certificate rarely matches the literal host. Set this to the name your certificates actually carry. |

A half-configured TLS setup fails at config time, not at the first fetch: enabling TLS
without `ca_cert_path`, without the server certificate/key pair, or with only one half of
the client pair raises `ConfigError`. Each field also has a
`BATCHER_DISTRIBUTED_TLS_<FIELD>` env override, which is how a deployment injects paths
without shipping a config file.

```python
# docs: skip
from batcher import Config, set_config
from batcher.config import DistributedConfig, ShuffleTlsConfig

set_config(
    Config().replace(
        distributed=DistributedConfig(
            transport="flight",
            tls=ShuffleTlsConfig(
                enabled=True,
                ca_cert_path="/etc/batcher/ca.pem",
                server_cert_path="/etc/batcher/server.pem",
                server_key_path="/etc/batcher/server.key",
                client_cert_path="/etc/batcher/client.pem",
                client_key_path="/etc/batcher/client.key",
                require_client_auth=True,
            ),
        )
    )
)
```

Pair it with `shuffle_token` above: TLS proves *who* the peer is, the token proves it is
allowed to fetch this partition.

## observability

What the engine tells you about what it did: the `batcher.*` logger hierarchy and the structured per-query event log. The event log is one JSON document per query carrying the plan, the Kyber and Carbonite decisions, and the measured per-operator profile.

| Field | Default | Meaning |
|-------|---------|---------|
| `verbosity` | `"normal"` | The one dial most users need. Takes `silent`, `quiet`, `normal`, `verbose`, `debug`, or `trace`, or the equivalent integer `0` to `5`. It sets `log_level` and `progress` together. |
| `log_level` | `None` | Explicit threshold for the `batcher.*` loggers: `CRITICAL`/`ERROR`/`WARNING`/`INFO`/`DEBUG`. `None` derives it from `verbosity`; setting it overrides just this one. Read the effective value from `resolved_log_level`. |
| `console` | `True` | Emit records to stderr. `False` for a file-only setup. |
| `log_file` | `None` | Path to a rotating log file, or `None` for no file handler. |
| `log_file_max_bytes` | `10000000` (10 MB) | Bytes per log file before it rotates. |
| `log_file_backups` | `3` | How many rotated files to keep. |
| `log_format` | `"human"` | `"human"` is a readable one-line layout; `"json"` writes one JSON object per record, for a log shipper. |
| `event_log` | `True` | Write the structured per-query event log, the analog of Spark's event log. It attaches a profile collector for the whole query, then assembles, encodes, and writes one JSON document per query. Set it to `False` when nothing reads the log and you want the control-plane overhead back. `explain(analyze=True)` and `stats()` give you the same profile on demand without it. |
| `otel_traces` | `False` | Emit an OpenTelemetry span per query, with a child span per operator, into the tracer your app already configured. It needs `opentelemetry` installed and a provider, because Batcher owns no exporter. It reuses the profile the event log already measures, so turning it on adds the emit, not the measurement. |
| `event_log_dir` | `""` | Directory for event-log documents. Empty resolves to `$BATCHER_HOME/logs` (or `~/.batcher/logs`) at write time. |
| `event_log_max_files` | `200` | Keep at most this many event-log files; the oldest are pruned on write. `0` is unbounded. |
| `progress` | `None` | Explicit progress-bar mode: `"auto"` renders only into a real TTY that has not set `NO_COLOR`/`TERM=dumb`, `"on"` forces it, `"off"` disables it. `None` derives it from `verbosity`. Read the effective value from `resolved_progress`. See {doc}`/user-guide/operate/observability`. |
| `ui` | `False` | Start the web dashboard automatically on the first query. It's off by default, because binding a port should be asked for. `bt.start_ui()` is the explicit spelling. Use this field for a long-running service that always wants the dashboard. |
| `ui_host` | `"127.0.0.1"` | Interface the dashboard binds. Loopback by default: it exposes query text, plans, and logs, and Batcher ships no authentication. Set it to a routable address only deliberately. |
| `ui_port` | `4040` | Dashboard port. `0` asks the OS for any free port. |

These fields are the
{py:class}`ObservabilityConfig <batcher.config.config.ObservabilityConfig>` dataclass. Construct
one and swap it onto `Config`. For example, to ship JSON logs and see every decision the engine makes:

```python
from batcher import Config
from batcher.config import ObservabilityConfig

cfg = Config().replace(
    observability=ObservabilityConfig(log_level="INFO", log_format="json")
)
print((cfg.observability.resolved_log_level, cfg.observability.event_log))
# ('INFO', True)
```

Every field takes a `BATCHER_OBSERVABILITY_*` environment override, so a deployment can raise the log level or turn the event log off with `BATCHER_OBSERVABILITY_EVENT_LOG=0` without touching code.

## Inspecting and editing

`Config()` is the defaults. To read the config a query would actually run under, call `active_config()`. It resolves to the innermost of the layered sources: the config file, the `BATCHER_*` environment variables, `set_config`, and any enclosing `config_context` block. That makes it what you check when a tunable doesn't seem to be taking effect.

```python
from batcher.config import Config, ExecutionConfig, active_config, config_context

print(active_config().execution.morsel_rows)
# 16384

with config_context(Config().replace(execution=ExecutionConfig(morsel_rows=4096))):
    print(active_config().execution.morsel_rows)
    # 4096

print(active_config().execution.morsel_rows)
# 16384
```

The returned `Config` is frozen, like every section: derive a new one rather than trying
to mutate it.

Read any field through its section, and derive a new config to change one.

```python
import dataclasses
from batcher import Config

base = Config()
print(base.optimizer.cardinality.eq_selectivity)
# 0.1

cfg = base.replace(
    optimizer=dataclasses.replace(
        base.optimizer,
        cardinality=dataclasses.replace(base.optimizer.cardinality, eq_selectivity=0.05),
    )
)
print(cfg.optimizer.cardinality.eq_selectivity)
# 0.05
```

## Requirements and limitations

Batcher validates a config where you install it, not where it's used. Invalid values raise `ConfigError` at the config entry point, meaning `set_config`, `config_context`, `from_env`, or `from_file`, rather than failing confusingly at runtime. That covers a negative retry count, a `soft_limit` above `hard_limit`, and a non-positive timeout. A half-configured TLS block fails the same way.

`Config` and every section are frozen. Derive a new one with `Config.replace` or `dataclasses.replace` rather than assigning to a field.

The worker fan-out is a terminal-call parameter, `ds.collect(num_workers=...)`, not a `Config` field. The `distributed` section tunes how the fan-out behaves once chosen, not how wide it is.

This page documents the fields most deployments reach for. The dataclasses carry further power-user thresholds that the API reference enumerates in full.

## See also

- {doc}`environment` for the `BATCHER_*` variables and the JSON file format.
- {doc}`profiles` for worked configurations built from these fields.
- {doc}`../architecture/fault-tolerance` for what the `distributed` retry and recovery fields protect against.
