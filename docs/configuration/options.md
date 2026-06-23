# Configuration options

The full reference for every field on `Config`. Each section below corresponds to
one attribute of `Config` (`config.execution`, `config.memory`, and so on). Defaults
are the engine's tuned constants; change a field by deriving a new section with
`dataclasses.replace`.

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

## memory

Buffer-pool envelope and the out-of-core spill story.

Setting **`max_memory_bytes`** is what opts the in-memory engine into spilling: the
data plane receives a per-operator spill budget of `max_memory_bytes × hard_limit`,
and the Rust runtime memory pool spills any stateful operator (aggregate, distinct,
sort, join, windowed-by-partition) that exceeds it — instead of letting the process
run out of memory. Leave it `None` to run fully in memory (the default, lowest
overhead). See the **OOM-resilient** profile in [profiles](profiles.md).

| Field | Default | Meaning |
|-------|---------|---------|
| `soft_limit` | `0.85` | Throttle new allocations at this fraction of the envelope. Must satisfy `0 < soft_limit <= hard_limit <= 1`. |
| `hard_limit` | `0.90` | Spill to disk at this fraction; also scales the data-plane spill budget derived from `max_memory_bytes`. |
| `max_memory_bytes` | `None` | Hard memory cap in bytes. `None` runs fully in memory (no spill); set it to bound memory (honoring a container/cgroup limit) **and enable spilling**. |
| `default_total_bytes` | `8589934592` (8 GiB) | Fallback total RAM assumed when `max_memory_bytes` is unset and the OS reports no usable figure. |
| `spill_dir` | `None` | Scratch directory for spill files. `None` uses a per-query temp dir. |
| `spill_remote_uri` | `None` | fsspec URL (`s3://`, `gs://`, …) the local spill tier overflows to, so a PB-scale spill does not die when local disk fills. |
| `spill_local_budget_bytes` | `None` | Local spill-tier capacity before overflowing to `spill_remote_uri`. |
| `spill_compression` | `"lz4"` | Arrow-IPC codec for spilled batches (`"lz4"`/`"zstd"`/`None`). |
| `spill_bucket_max_bytes` | `134217728` (128 MiB) | A spilled aggregate bucket larger than this is re-partitioned (grace recursion) so a skewed key set degrades gracefully instead of OOMing the reduce. |

## flow_control

Credit-based backpressure for the shuffle, the Carbonite flow-control model.

| Field | Default | Meaning |
|-------|---------|---------|
| `default_credits` | `4` | In-flight batch slots when an operator has no estimate. One credit is one buffered batch. |
| `credit_ceiling_factor` | `16` | Maximum credit window is `default_credits * credit_ceiling_factor`. |
| `credit_byte_budget` | `268435456` (256 MiB) | Byte ceiling for one shuffle channel's credit window, so wide rows can't buffer GBs even within the count ceiling. |
| `shuffle_fan_in` | `8` | Maximum inbound streams a shuffle node fans in before the reduce becomes a tree of combiner stages. |
| `aimd_alpha` | `1` | Additive increase: credits added per round trip. |
| `aimd_beta` | `0.5` | Multiplicative decrease applied on congestion. |
| `backpressure_high` | `0.70` | Buffer occupancy at which the producer is throttled. |
| `backpressure_low` | `0.40` | Buffer occupancy at which the producer resumes. |

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
| `cardinality` | `CardinalityConfig()` | Selinger-style fallback selectivities (sub-section below). |
| `cost_coeffs` | `CostCoefficients()` | Per-unit operator costs (sub-section below). |
| `cost_weights` | `CostWeights()` | Relative weight of CPU, IO, and network when collapsing cost to a scalar (sub-section below). |

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

## metadata

Where learned statistics (the MetadataHub) live and how fast confidence decays.

| Field | Default | Meaning |
|-------|---------|---------|
| `backend` | `"in_process"` | Storage backend: `"in_process"`, `"sqlite"`, `"redis"`, or `"object_storage"`. |
| `uri` | `None` | Connection or path for a non-in-process backend. |
| `decay_per_day` | `0.1` | Daily confidence decay for learned stats (roughly a one-week half-life). |

## distributed

How the engine attaches to a Ray cluster, shuffles across it, and stays correct
through node and task failures. Ray is used for scheduling only; bulk shuffle data
moves over Arrow Flight, bypassing the Ray object store. See the **fault-tolerant
cluster** profile in [profiles](profiles.md).

| Field | Default | Meaning |
|-------|---------|---------|
| `ray_address` | `None` | Ray cluster address. `None` attaches to a running cluster when `RAY_ADDRESS` is set, else starts a local one. |
| `namespace` | `"batcher"` | Ray namespace for batcher's shuffle actors, so they are isolatable. |
| `runtime_env` | `None` | `runtime_env` dict shipped to workers so `batcher` + its native extension are present cluster-wide. |
| `transport` | `"auto"` | Shuffle transport. `"auto"` picks Flight on a multi-node cluster, disk on a single node / shared filesystem; `"flight"`/`"disk"` force it. |
| `shared_filesystem` | `False` | True when every worker shares a filesystem at the same path, so the disk shuffle is safe cluster-wide. |
| `dashboard` | `False` | Show the Ray dashboard. |
| `adaptive_credits` | `False` | Opt-in AIMD shuffle credits: the window adapts to observed memory backpressure instead of the static grant. |

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
| `speculation_max_backups` | `0` | Max concurrent speculative backup tasks at a shuffle barrier. `0` disables straggler speculation (the barrier is a plain wait). |
| `speculation_straggler_factor` | `1.5` | Back up a task whose elapsed time exceeds this multiple of the median finished task's time. |
| `speculation_min_finished_frac` | `0.75` | Fraction of tasks that must finish before speculation starts. |
| `skew_join_salt` | `0` | Spread a hot join key's rows across this many reducers (`0` disables skew-aware salting). |
| `skew_join_fraction` | `0.10` | A value is "hot" when it exceeds this fraction of a side's rows. |
| `shuffle_token` | `None` | Shared secret authenticating Flight shuffle fetches. Also read from `BATCHER_SHUFFLE_TOKEN`. |

Invalid values (a negative retry count, `soft_limit` above `hard_limit`, a
non-positive timeout) raise `ConfigError` at the config entry point
(`set_config`, `config_context`, `from_env`, `from_file`) rather than failing
confusingly at runtime.

## Inspecting and editing

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
