# Configuration options

This page is the field-by-field reference for {py:class}`Config <batcher.Config>`. Each section below corresponds to one attribute of {py:class}`Config <batcher.Config>`, such as `config.execution` or `config.memory`. Defaults are the engine's tuned constants. To change a field, derive a new section with `dataclasses.replace`.

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
| `shrink_output_dtypes` | `False` | Re-narrow a pass-through output column back to its source numeric width, such as `Int32` ids widened on input, halving its footprint. It's lossless but data-dependent, so it's off by default. With it off, output types match {py:obj}`Dataset.schema <batcher.Dataset.schema>` exactly. |

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

Buffer-pool envelope and the out-of-core spill story. Setting `max_memory_bytes` is what opts the in-memory engine into spilling. The data plane receives a per-operator spill budget of `max_memory_bytes` times `hard_limit`, and the Rust runtime memory pool spills any stateful operator that exceeds it rather than letting the process run out of memory. That covers aggregate, distinct, sort, join, and windowed-by-partition. Leave it `None`, the default, to run fully in memory at the lowest overhead. See the bounded-memory recipe in {doc}`profiles`.

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
| `result_cache_max_bytes` | `268435456` (256 MiB) | Byte budget for the process-wide result cache backing {py:meth}`Dataset.cache() <batcher.Dataset.cache>`, described in {doc}`/user-guide/operate/tuning/performance`. Cached Arrow results are held LRU and evicted to stay within this, so caching never grows the process without bound. |
| `streaming_state_max_bytes` | `0` | Cap on one streaming operator's in-memory state (window partials, dedup keys, join buffers). Exceeding it raises a clear {py:exc}`ResourceError <batcher.ResourceError>` (a stalled-watermark signal) instead of OOMing. `0` derives the cap from the hard memory budget. |
| `respect_cgroup_high` | `True` | Budget against the cgroup v2 `memory.high` throttle threshold, not just the `memory.max` kill threshold. Inert wherever `memory.high` is unset. |
| `stall_aware_pressure` | `True` | Let the kernel's memory PSI raise the pressure level, so the engine spills while reclaim is still coping. It can only raise a level, never lower one. |
| `oom_kill_backoff` | `0.8` | Fraction of the auto-sensed envelope kept when this container's `memory.events` shows it has already been OOM-killed. `1.0` disables the backoff. An explicit `max_memory_bytes` is never scaled. See {doc}`/architecture/deep-dives/memory/buffer-pool` for what these three kernel signals measure. |

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

## streaming

The micro-batch loop's cadence and bookkeeping. These govern how a long-running
streaming query behaves between batches, not what it computes. The memory a streaming
operator's state may hold is `memory.streaming_state_max_bytes`, in the `memory` section.

| Field | Default | Meaning |
|-------|---------|---------|
| `idle_poll_seconds` | `0.2` | How long a runner waits before asking an idle unbounded source for data again. Only an empty pass pays it. |
| `progress_history` | `100` | Micro-batch progress records a query handle keeps for `recent_progress`. |

Lower `idle_poll_seconds` when first-row latency after a quiet stretch matters more than
the cost of re-listing a directory or re-asking a broker for its partitions. Raise it when
a stream is idle most of the time and the listing is expensive, such as a large cloud
prefix. The single-node and distributed runners read the same value, so an idle stream
behaves the same on one machine and on a cluster.

```python
from batcher import Config, StreamingConfig

cfg = Config().replace(streaming=StreamingConfig(idle_poll_seconds=0.05))
print(cfg.streaming.idle_poll_seconds)
# 0.05
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
| `cardinality` | {py:class}`CardinalityConfig() <batcher.config.config.CardinalityConfig>` | Selinger-style fallback selectivities (sub-section below). |
| `cost_coeffs` | {py:class}`CostCoefficients() <batcher.config.config.CostCoefficients>` | Per-unit operator costs (sub-section below). |
| `cost_weights` | {py:class}`CostWeights() <batcher.config.config.CostWeights>` | Relative weight of CPU, IO, and network when collapsing cost to a scalar (sub-section below). |

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

Whether the row filters and column masks in a {py:class}`SecurityCatalog <batcher.SecurityCatalog>` are advisory or mandatory.

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

Ray attachment, the shuffle transport, fault tolerance, restricted networks, autoscaling,
and TLS. Large enough to be its own page: see {doc}`distributed-options`.

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
| `progress` | `None` | Explicit progress-bar mode: `"auto"` renders only into a real TTY that has not set `NO_COLOR`/`TERM=dumb`, `"on"` forces it, `"off"` disables it. `None` derives it from `verbosity`. Read the effective value from `resolved_progress`. See {doc}`/user-guide/operate/running/observability`. |
| `ui` | `False` | Start the web dashboard automatically on the first query. It's off by default, because binding a port should be asked for. {py:func}`bt.start_ui() <batcher.start_ui>` is the explicit spelling. Use this field for a long-running service that always wants the dashboard. |
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

`Config()` is the defaults. To read the config a query would actually run under, call `active_config()`. It resolves to the innermost of the layered sources: the config file, the `BATCHER_*` environment variables, {py:func}`set_config <batcher.set_config>`, and any enclosing {py:func}`config_context <batcher.config_context>` block. That makes it what you check when a tunable doesn't seem to be taking effect.

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

Batcher validates a config where you install it, not where it's used. Invalid values raise {py:exc}`ConfigError <batcher.ConfigError>` at the config entry point, meaning `set_config`, `config_context`, `from_env`, or `from_file`, rather than failing confusingly at runtime. That covers a negative retry count, a `soft_limit` above `hard_limit`, and a non-positive timeout. A half-configured TLS block fails the same way.

`Config` and every section are frozen. Derive a new one with {py:meth}`Config.replace <batcher.Config.replace>` or `dataclasses.replace` rather than assigning to a field.

The worker fan-out is a terminal-call parameter, {py:meth}`ds.collect(num_workers=...) <batcher.Dataset.collect>`, not a `Config` field. The `distributed` section tunes how the fan-out behaves once chosen, not how wide it is.

This page documents the fields most deployments reach for. The dataclasses carry further power-user thresholds that the API reference enumerates in full.

## See also

- {doc}`environment` for the `BATCHER_*` variables and the JSON file format.
- {doc}`profiles` for worked configurations built from these fields.
- {doc}`../architecture/fault-tolerance` for what the `distributed` retry and recovery fields protect against.
