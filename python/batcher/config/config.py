"""The single frozen `Config` and its typed sections.

Defaults encode the engine's tuned constants (morsel size, memory envelopes,
selectivity/cost coefficients, PID gains) in one place rather than scattered magic
numbers — this module is the single source of truth for every tunable, and the
Rust-relevant subset is shipped to the data plane as part of the execution config
(see `core` / `bc_ir::EngineConfig`).

Precedence, highest first: ``config_context`` > programmatic ``set_config`` >
``BATCHER_*`` env vars > a JSON file at ``BATCHER_CONFIG_FILE`` > defaults. The env
and file layers are evaluated once when this module is imported (see
`_initial_config`); ``set_config`` / ``config_context`` override them at runtime.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

__all__ = [
    "CardinalityConfig",
    "Config",
    "CostCoefficients",
    "CostWeights",
    "DistributedConfig",
    "ExecutionConfig",
    "FlowControlConfig",
    "MemoryConfig",
    "MetadataConfig",
    "ObservabilityConfig",
    "OptimizerConfig",
    "PIDConfig",
    "active_config",
    "config_context",
    "set_config",
]

# Wire-contract key order for the Rust engine-config payload (`bc_ir::EngineConfig`).
# Single source of truth: both the dict builder and the memoized serializer key off
# this tuple, so the JSON shape can never drift between the two code paths.
_ENGINE_CONFIG_FIELDS = (
    "morsel_rows",
    "morsel_bytes",
    "parallelism",
    "memory_budget_bytes",
    "spill_dir",
    "spill_compression",
    "fuse_linear",
    "shrink_output_dtypes",
    # Performance-threshold knobs (mirror `bc_arrow::RuntimeTuning`).
    "bloom_fp_rate",
    "bloom_min_build_rows",
    "window_parallel_row_threshold",
    "radix_parallel_threshold",
    "sort_merge_fanin",
    "skew_bucket_factor",
    "skew_min_bucket_rows",
    "skew_min_bucket_bytes",
)


@functools.lru_cache(maxsize=128)
def _engine_config_json(values: tuple[object, ...]) -> str:
    """Memoized engine-config serialization, keyed by its (hashable) value tuple.

    The base payload depends only on a handful of frozen knobs, but it is serialized
    on every native call and every streaming micro-batch. Caching by value collapses
    that to one `json.dumps` per distinct configuration.
    """
    return json.dumps(dict(zip(_ENGINE_CONFIG_FIELDS, values, strict=True)))


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """How work is sized and scheduled: thread count and morsel dimensions.

    The engine's unit of work is a morsel — a small `RecordBatch` sized to fit cache
    so scheduling stays granular and cache-friendly. These defaults saturate every
    core without tuning; the per-field reference is in the configuration guide.

    Examples:
        .. doctest::

            >>> from batcher.config import ExecutionConfig
            >>> ExecutionConfig().morsel_rows
            16384
    """

    # 0 means "use all available cores".
    parallelism: int = 0
    # Default morsel size in rows (§1.4): fits L2/L3, amortizes scheduling. This is
    # the value shipped to Rust as `EngineConfig.morsel_rows`; the Rust
    # `bc_arrow::DEFAULT_MORSEL_ROWS` const is only the standalone-test default.
    morsel_rows: int = 16_384
    # Byte budget per morsel, shipped to Rust as `EngineConfig.morsel_bytes`. A
    # morsel is split at whichever bound (rows or bytes) trips first, so
    # wide/variable-width data (large strings, embeddings, blob handles) stays
    # cache- and memory-bounded. ~16_384 rows × 64 B, so narrow data is unaffected.
    morsel_bytes: int = 1 << 20  # 1 MiB
    # Target byte size of a single file split (source readers chunk large files into
    # splits so the driver never materializes a whole file at once).
    split_bytes: int = 128 * 1024 * 1024
    # CPU shares requested per distributed Ray task. Makes Ray's implicit default of
    # 1 explicit and tunable (a heavy native op can ask for more); the scheduler
    # places tasks against this. This is the CPU-heavy default (a breaker that
    # saturates a core); CPU-light stages use `cpu_share_io` below.
    cpus_per_task: float = 1.0
    # CPU shares a CPU-light / IO-bound distributed stage requests (scan, filter,
    # project, write, CPU-only preprocessing). Below 1.0 so such tasks pack more than
    # one per core — they wait on IO/decode rather than saturating a core. Affects the
    # distributed (Ray) path only; single-node uses the rayon pool (`parallelism`).
    # This is the per-operator-kind *cold-start* prior; once a query has run, Kyber
    # overrides it with the measured CPU utilization of each operator family.
    cpu_share_io: float = 0.5
    # Floor for the adaptive per-task CPU share. A learned utilization below this
    # still requests this many cores, so an IO-bound stage never asks for an
    # unschedulable sliver of a CPU (mirrors the GPU fraction's 0.25 floor).
    cpu_share_min: float = 0.25
    # Adaptive morsel sizing: shrink the per-morsel (rows, bytes) target under memory
    # pressure so the streaming working set stays bounded when memory is tight — the
    # "size blocks to memory" lever. Result-invariant (a morsel only batches data, it
    # never changes the output), so a query's result is identical whether this is on or
    # off. On by default: when memory is NOT under pressure the configured
    # `morsel_rows`/`morsel_bytes` target is used unchanged (so the small-query fast path
    # and single-node==distributed equivalence are byte-identical); the target only
    # shrinks once the live `PressureMonitor` reports ELEVATED or worse. Set to False to
    # pin the static target regardless of pressure.
    adaptive_morsel_sizing: bool = True
    # Fuse runs of linear, per-morsel streaming operators (Filter/Project) into a single
    # pass over the input's morsels in the parallel executor, instead of one rayon
    # dispatch + intermediate buffer per operator. Result-invariant (same rows, same
    # order — verified against the sequential oracle and the full DuckDB differential
    # suite); shipped to Rust as `EngineConfig.fuse_linear`. On by default — it only
    # engages on a chain of ≥2 fusable ops (so single-op and breaker-only plans are
    # untouched) and is a measured win on linear pipelines with no regression elsewhere.
    # Set to False to pin the staged operator-at-a-time path.
    fuse_linear: bool = True
    # Re-narrow output columns to their source numeric width. The FFI widens narrow
    # numerics (Int8/16/32, Float16/32) to Int64/Float64 once on input so every kernel
    # stays on two well-tested paths; with this on, an output column that is a
    # pass-through of a narrow *source* column (same name, type == the widened image)
    # and whose values all fit is cast back to the source width — halving the footprint
    # of Int32-id / Float32-feature columns that ride through unchanged. Lossless (a
    # value that would overflow keeps the wide type) but data-dependent, so it is
    # **off by default**: with it off, output types and the pre-execution
    # `Dataset.schema` agree exactly. Shipped to Rust as
    # `EngineConfig.shrink_output_dtypes`.
    shrink_output_dtypes: bool = False
    # Automatically offload a large-payload (`large_binary`) column out of line around
    # a Sort, so the payload rides through the breaker as a tiny content-addressed URI
    # handle instead of filling its buffers/spill files (read back right after). The
    # explicit `Dataset.offload_blobs`/`materialize_blobs`, placed automatically and
    # result-identically. Off by default: it trades blob bytes crossing the breaker for
    # a content-store round-trip, a win only for genuinely large payloads. Control-plane
    # only (rewrites the plan in `api`), so it is NOT part of the Rust engine config.
    auto_offload_blobs: bool = False
    # --- Performance-threshold knobs (power-user perf tuning) --------------------
    # These mirror `bc_arrow::RuntimeTuning` / `bc_ir::EngineConfig` and tune *how*
    # the parallel executor runs an operator (parallel-vs-serial thresholds, the
    # probe bloom, merge fan-in, skew detection). They are performance-only: a query
    # produces the identical result at any setting. Each default equals the Rust
    # const it replaced, so leaving them untouched is bit-identical to the old engine.
    # Reach for these only to tune a known hot path; most users never set them.
    #
    # False-positive rate for the hash-join probe-side bloom pre-filter.
    bloom_fp_rate: float = 0.01
    # Build-row floor above which the probe bloom pays for itself.
    bloom_min_build_rows: int = 1 << 16
    # Window row count above which per-partition sorts run across cores.
    window_parallel_row_threshold: int = 1 << 15
    # Concatenated-input row count above which aggregate `combine` regroups via
    # parallel hash-radix partitioning.
    radix_parallel_threshold: int = 200_000
    # Maximum runs merged per pass in the external (spilling) sort's k-way merge.
    sort_merge_fanin: int = 16
    # A join bucket is "hot" when it exceeds this multiple of the average bucket.
    skew_bucket_factor: int = 4
    # Absolute row floor below which a join bucket is never treated as skewed.
    skew_min_bucket_rows: int = 4 * 16_384
    # Absolute byte floor below which a join bucket is never treated as skewed.
    skew_min_bucket_bytes: int = 4 * (1 << 20)


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """The memory envelope and when to spill to disk.

    Carbonite keeps per-node memory bounded against these limits: it throttles at the
    soft limit and spills aggregating, joining, and sorting operators to disk before
    the hard limit, so a large query stays alive instead of running out of memory.

    Examples:
        .. doctest::

            >>> from batcher.config import MemoryConfig
            >>> MemoryConfig().soft_limit
            0.85
    """

    soft_limit: float = 0.85  # throttle at 85% of the envelope
    hard_limit: float = 0.90  # spill at 90%
    # Hard memory cap in bytes for the buffer pool / spill decision. `None` (the
    # default) is *auto-sensed*: the `api` layer fills it once at the terminal-op
    # boundary from the live memory envelope (host RAM, honoring a container/cgroup
    # limit) and freezes it for the query — so a zero-config query spills stateful
    # operators out of core (budget = cap × `hard_limit`, shipped to the data plane)
    # instead of OOMing. Set it explicitly to pin a cap the OS won't report; set
    # `unbounded_memory` to opt out of auto-sensing and stay fully in-memory.
    max_memory_bytes: int | None = None
    # Opt out of the auto-sensed spill budget: keep the in-memory fast path with no
    # out-of-core spilling in the data-plane engine (the pre-auto-tuning behavior).
    # The data-plane spill budget is then 0 (unbounded) regardless of `max_memory_bytes`;
    # a `max_memory_bytes` still set continues to bound the control-plane admission
    # envelope. For power users who would rather a query fail fast than spill to disk.
    unbounded_memory: bool = False
    # Fallback total RAM (bytes) assumed when neither `max_memory_bytes` is set nor
    # the OS reports a usable figure. One home for what was a copy-pasted literal.
    default_total_bytes: int = 8 << 30  # 8 GiB
    # Out-of-core spill tiers. The local tier (NVMe) is fast and capacity-bounded;
    # once `spill_local_budget_bytes` is exhausted, new buckets overflow to
    # `spill_remote_uri` (any fsspec URL: s3://, gs://, …) so a PB-scale spill does
    # not die when local disk fills. `spill_dir` overrides the local scratch dir
    # (default: a per-query tempdir). `spill_compression` is the Arrow-IPC codec for
    # spilled batches: "auto" (the default) picks per spill by the batch's dominant
    # column type — ZSTD for blob/large-text payloads, none for all-numeric schemas,
    # LZ4 for strings/mixed; "lz4"/"zstd"/None force one codec. Spilled data is
    # transient, so this only trades CPU for disk I/O and footprint at scale
    # (result-invariant — IPC self-describes its compression).
    spill_dir: str | None = None
    spill_remote_uri: str | None = None
    spill_local_budget_bytes: int | None = None
    spill_compression: str | None = "auto"
    # Grace recursion trigger: when a single spilled aggregate bucket's on-disk size
    # exceeds this, it is re-partitioned (by a secondary hash of the group key) into
    # sub-buckets and reduced one at a time — so a *skewed* key set that overflows one
    # bucket degrades gracefully out-of-core instead of OOMing the reduce.
    spill_bucket_max_bytes: int = 128 << 20  # 128 MiB (compressed)
    # Byte budget for the process-wide result cache (`Dataset.cache()`): the *storage*
    # half of the memory envelope. The cache holds materialized Arrow results LRU and
    # evicts to stay within this, yielding the RAM back to execution under pressure, so
    # caching never grows the process without bound. Opt-in per dataset, so this only
    # bounds what an explicitly-cached plan may retain.
    result_cache_max_bytes: int = 256 << 20  # 256 MiB
    # Local-SSD read-through cache for remote (S3/GCS/Azure) file bytes — the engine's
    # Disk-Cache analog. `None` (default) disables it; set a directory to cache fetched
    # remote files there, byte-bounded to `file_cache_max_bytes` with LRU eviction. It
    # only accelerates re-reads of the same remote file — transparent, ephemeral, and
    # result-invariant (a cache miss just re-fetches). Local paths are never cached.
    file_cache_dir: str | None = None
    file_cache_max_bytes: int = 8 << 30  # 8 GiB budget (used only when enabled)
    # Cap on one streaming operator's in-memory state (windowed-aggregate partials,
    # watermark-dedup keys, stream-join buffers). That state is bounded by the
    # watermark *advancing*; a stalled watermark (an event-time gap, or one stream
    # going quiet) lets it grow without bound. Exceeding this raises a clear
    # `ResourceError` — a stalled-watermark / huge-key-space signal — instead of a
    # silent OOM. `0` derives it from the hard memory budget (see
    # `streaming_state_budget_bytes`); a positive value overrides.
    streaming_state_max_bytes: int = 0

    def streaming_state_budget_bytes(self) -> int:
        """The effective per-operator streaming-state cap in bytes.

        The explicit `streaming_state_max_bytes` when set, else the hard memory budget
        (`max_memory_bytes` or `default_total_bytes`, scaled by `hard_limit`) so the
        cap scales with the configured envelope rather than a fixed magic number.
        """
        if self.streaming_state_max_bytes > 0:
            return self.streaming_state_max_bytes
        base = (
            self.max_memory_bytes if self.max_memory_bytes is not None else self.default_total_bytes
        )
        return int(base * self.hard_limit)


@dataclass(frozen=True, slots=True)
class FlowControlConfig:
    """Credit-based backpressure for the shuffle and data transport.

    A credit is one in-flight batch slot; a producer blocks when its peer runs out,
    so a fast stage cannot flood a slow one and blow up memory. The credit window
    adapts with an AIMD loop (like TCP). Tune these only for unusual cluster shapes;
    the per-field reference is in the configuration guide.

    Examples:
        .. doctest::

            >>> from batcher.config import FlowControlConfig
            >>> FlowControlConfig().default_credits
            4
    """

    # Credit window (in-flight RecordBatch slots) when the operator has no estimate.
    # One credit = one buffered batch, so this bounds a shuffle channel's memory.
    # Carbonite is the authority that supplies it, and clamps any per-operator
    # request to `default_credits x ceiling`. Shipped to Rust as `EngineConfig`.
    default_credits: int = 4
    credit_ceiling_factor: int = 16  # max window = default_credits x this
    # Byte ceiling for one shuffle channel's credit window (C53). A credit ≈ one
    # `morsel_bytes` batch, so a count-only ceiling can buffer GBs for wide rows
    # (embeddings, blobs). The granted window is also clamped to
    # `credit_byte_budget // morsel_bytes`, so a channel's buffered memory is bounded
    # regardless of row width. With the default 1 MiB morsel this is a no-op for
    # narrow data (256 ≥ the count ceiling of 64).
    credit_byte_budget: int = 256 << 20  # 256 MiB per channel
    # Max inbound streams a shuffle node fans in. Above this many upstreams the
    # reduce becomes a tree of combiner stages (depth log_fan_in(workers)), so
    # per-node fan-in stays bounded as the cluster grows to many thousands.
    shuffle_fan_in: int = 8
    aimd_alpha: int = 1  # additive increase: +1 credit / RTT
    aimd_beta: float = 0.5  # multiplicative decrease on congestion
    backpressure_high: float = 0.70
    backpressure_low: float = 0.40


@dataclass(frozen=True, slots=True)
class CardinalityConfig:
    """Selinger-style defaults the cardinality estimator falls back to before
    anything is learned. Superseded by learned/sketch values when present.

    Examples:
        .. doctest::

            >>> from batcher.config import CardinalityConfig
            >>> CardinalityConfig().eq_selectivity
            0.1
    """

    # Used when a source's size is unknown (e.g. CSV): large enough that an unknown
    # side is never preferred as the (smaller) build side.
    unknown_rows: float = 1e12
    default_filter_selectivity: float = 0.5
    eq_selectivity: float = 0.1  # col = literal
    range_selectivity: float = 1.0 / 3.0  # col <|<=|>|>= literal
    null_selectivity: float = 0.05  # col IS NULL
    # A value appearing in at least this fraction of a column's rows is recorded as a
    # most-common-value (MCV), so `col = <that value>` uses its measured frequency
    # instead of the uniform `1/ndv` — the skew case where `1/ndv` is most wrong.
    mcv_min_fraction: float = 0.05


@dataclass(frozen=True, slots=True)
class CostWeights:
    """Relative importance of each axis when collapsing `Cost` to a scalar.
    WS9 swaps these per query to honor latency/cost/throughput targets."""

    cpu: float = 1.0
    io: float = 1.0
    net: float = 2.0  # shuffle bytes hurt more than local bytes


@dataclass(frozen=True, slots=True)
class CostCoefficients:
    """Per-unit costs. Constants today; calibrated from measured `op_stats` later.
    All values are in abstract, mutually-comparable "work units" per row/byte."""

    scan_row: float = 1.0
    filter_row: float = 0.5
    project_row: float = 0.3
    hash_build_row: float = 2.0  # insert into a hash table
    hash_probe_row: float = 1.0
    output_row: float = 0.5
    sort_row: float = 1.0  # multiplied by log2(n)
    distinct_row: float = 2.0
    union_row: float = 0.2
    map_row: float = 5.0  # opaque UDF: assume expensive
    bytes_per_row: float = 64.0  # rough row width for io/net axes


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """Knobs for the Kyber optimizer: join planning, cost, and cardinality.

    Controls how hard the optimizer works (exact dynamic-programming join ordering up
    to a table count, greedy beyond it), how it estimates cost and row counts from
    learned statistics and sketches, and when a measured estimate is wrong enough to
    trigger re-optimization mid-query. Defaults suit most workloads.

    Examples:
        .. doctest::

            >>> from batcher.config import OptimizerConfig
            >>> OptimizerConfig().join_dp_max_tables
            12
    """

    join_dp_max_tables: int = 12  # DP-CCP exact threshold
    greedy_max_tables: int = 25  # greedy heuristic threshold
    # Build a per-column membership bloom index when persisting a written source's
    # stats, so a later read can data-skip an equality/`IN` predicate whose value is
    # absent (a point lookup inside [min, max] that zone-map bounds can't prune).
    # Opt-in (default off): the index is built over every int/text column on write
    # (~1.2 MB per million rows per column) and stored in the source's metadata.
    build_bloom_index: bool = False
    reoptimize_error: float = 2.0  # re-optimize when |actual-est|/est exceeds this
    # Target rows handled by one distributed task; a breaker's estimated parallelism
    # is ceil(rows / this), so worker fan-out tracks data size instead of cpu_count.
    target_rows_per_task: int = 4_000_000
    # Target *bytes* handled by one distributed task. Spill/shuffle partition counts
    # take the max of the row- and byte-derived fan-out, so a few wide rows (GB
    # videos, embeddings) still shard finely enough to fit memory. ~target_rows × 64.
    target_bytes_per_task: int = 256 * 1024 * 1024  # 256 MiB
    fixpoint_iterations: int = 8  # max rewrite-phase iterations before bailing
    row_bytes: int = 64  # per-row footprint for the memory-budgeting estimate
    # Build-side byte threshold below which a join is broadcast (the right side is
    # replicated to every worker) rather than shuffled — Spark's
    # autoBroadcastJoinThreshold. Both the planner's *estimate*-based decision and the
    # distributed executor's *runtime* guard read this one value: if the materialized
    # build side actually exceeds it (the estimate was wrong), the executor falls back
    # to a shuffle join instead of OOMing the driver by replicating an over-large side.
    broadcast_max_bytes: int = 10 * 1024 * 1024  # 10 MiB
    learning_smoothing_alpha: float = 0.5  # exp-smoothing toward new observations
    # Cost-model calibration from measured op_stats: a kind needs at least this many
    # samples before its coefficient is calibrated (else the default constant stands),
    # and each calibrated coefficient is clamped to within this factor of its default
    # so timing noise can never produce a degenerate cost model.
    cost_calibration_min_samples: int = 20
    cost_calibration_clamp: float = 10.0
    # Quantile grid Core collects for histogram-based selectivity.
    quantile_probs: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    cardinality: CardinalityConfig = CardinalityConfig()
    cost_coeffs: CostCoefficients = CostCoefficients()
    cost_weights: CostWeights = CostWeights()


@dataclass(frozen=True, slots=True)
class PIDConfig:
    """Gains for the adaptive batch-size controller — a PID loop over relative
    latency error that grows/shrinks the per-batch row count toward a target
    latency. Implemented identically in `bc-udf::BatchSizeController` (data plane)
    and `ml.inference._LatencyController` (Python); shipped to Rust as `EngineConfig`
    so the two never drift.

    Examples:
        .. doctest::

            >>> from batcher.config import PIDConfig
            >>> PIDConfig().kp
            0.4
    """

    kp: float = 0.4
    ki: float = 0.05
    kd: float = 0.1
    integral_clamp: float = 5.0  # anti-windup bound on the integral term
    max_step_fraction: float = 0.5  # cap per-step size change to +/-50%


@dataclass(frozen=True, slots=True)
class MetadataConfig:
    """Where learned statistics live and how fast they age.

    Core measures real cardinalities and operator costs each run and records them in
    the `MetadataHub`; Kyber reads them back to plan better next time. This selects
    the backend (in-process, SQLite, Redis, object storage) and how quickly old
    observations decay, so plans keep improving as a query is re-run.

    The default `in_process` backend keeps learned stats for the life of the process
    (plans improve within a session) but discards them on exit. To carry learning
    across restarts, set `backend="sqlite"` — with no `uri` it persists to a per-user
    file (``$BATCHER_HOME`` or ``~/.batcher/metadata.db``), so cross-run learning is a
    single line with no path to manage. On a spot/autoscaling cluster the ``"spot"``
    resilience profile auto-upgrades a still-default in-process store to shared object
    storage when a location is discoverable (``BATCHER_METADATA_URI``, any fsspec URL, or a
    managed cluster's ``ANYSCALE_ARTIFACT_STORAGE``), so learning survives a preempted driver
    moving nodes (see `batcher.config.profiles`).

    Examples:
        .. doctest::

            >>> from batcher.config import MetadataConfig
            >>> MetadataConfig().backend
            'in_process'
    """

    backend: str = "in_process"  # "in_process" | "sqlite" | "redis" | "object_storage"
    # Backend location. None means the backend's default — for `sqlite`, a persistent
    # per-user file (see `metadata.backends.default_sqlite_uri`); pass `":memory:"` for
    # an ephemeral SQLite store.
    uri: str | None = None
    decay_per_day: float = 0.1  # confidence half-life ~ a week


# Sentinel `autoscale_wait_s` meaning "auto": the config layer resolves it to a bounded
# wait on an autoscaling-capable cluster and to `0` (off) on a fixed one
# (`profiles.resolve_autoscale_wait`). It keeps `0` a first-class explicit "off" that a
# power user can set and have honored — distinct from "never configured". Resolved before
# the value reaches the runtime, which only ever sees a concrete `>= 0`.
AUTOSCALE_WAIT_AUTO: float = -1.0


@dataclass(frozen=True, slots=True)
class DistributedConfig:
    """How the engine attaches to and shuffles across a Ray cluster.

    Ray is scheduling only; the data plane shuffles via Carbonite/Arrow Flight or
    (single-node / shared filesystem) Arrow-IPC files. These knobs decide which.
    """

    # Ray cluster address. None → attach to an existing cluster when ``RAY_ADDRESS``
    # is set in the environment, else start a local one. Set explicitly (e.g.
    # ``"ray://head:10001"`` or ``"auto"``) to force attaching to a running cluster.
    ray_address: str | None = None
    # Ray namespace for batcher's shuffle actors, so they're isolatable.
    namespace: str = "batcher"
    # ``runtime_env`` dict shipped to workers (e.g. ``{"working_dir": ...}`` or
    # ``{"py_modules": [...]}``) so ``batcher`` + its native extension are present
    # cluster-wide. None when batcher is already installed on every node.
    runtime_env: dict[str, object] | None = None
    # Trust that every worker node's image already carries a *compatible* batcher,
    # so the driver should not upload its own package. Default False: when attaching
    # to a remote cluster with no explicit ``runtime_env``, the driver self-ships its
    # exact batcher package (py_modules, cached by Ray) so worker code matches the
    # driver's — correctness over a one-time ~10MB upload. A pip-installed driver
    # cannot assume an arbitrary cluster carries a matching batcher, and the old
    # "skip shipping for site-packages installs" heuristic produced silent
    # ModuleNotFoundError on workers for the common local-install→remote-cluster case.
    # Set True for a production image that bakes batcher in (skips the upload).
    trust_cluster_image: bool = False
    # Shuffle transport: ``"auto"`` picks Flight on a genuine multi-node cluster
    # (the disk shuffle's work_dir is driver-local and unreachable cross-node) and
    # disk on a single node / shared filesystem. ``"flight"`` / ``"disk"`` force it.
    transport: str = "auto"
    # True when every worker shares a filesystem (NFS / mounted object store) at the
    # same path, so the disk shuffle is safe cluster-wide and ``"auto"`` keeps disk.
    shared_filesystem: bool = False
    # Show the Ray dashboard. Off by default (and for local/test runs); a real
    # multi-node cluster benefits from it for the task/actor timeline.
    dashboard: bool = False
    # Object store (plasma) size in bytes for a *locally started* Ray (None → Ray's
    # default, ~30% of RAM). Applied only when batcher starts a local cluster — Ray
    # rejects `object_store_memory` when attaching to an existing cluster (which owns
    # its own store). The data plane bypasses the object store (Arrow Flight), so this
    # only bounds the small control-plane metadata; set it for an object-store-heavy
    # mixed workload or a memory-constrained box.
    object_store_memory_bytes: int | None = None
    # AIMD adaptive shuffle credits: the credit window grows/shrinks per remote fetch
    # from observed memory backpressure (TCP-like) instead of the static grant. On by
    # default — it is result-preserving (flow control only, never affects the merged
    # output) and lets the shuffle back off under memory pressure instead of holding a
    # fixed window, which is the safer behavior at scale (the distributed arm of OOM
    # survival). Set False to pin the static `default_credits` window.
    adaptive_credits: bool = True
    # Straggler mitigation: max concurrent speculative *backup* tasks at a shuffle
    # barrier. 0 (default) disables speculation — the barrier behaves exactly like
    # `ray.get`. Positive values let one slow survivor get a backup copy (the barrier
    # takes whichever finishes first); shuffle tasks are deterministic so the result
    # is identical. Bounded so speculation never oversubscribes the cluster.
    speculation_max_backups: int = 0
    # Back up a still-running task whose elapsed time exceeds this multiple of the
    # median finished task's time, once `speculation_min_finished_frac` have finished.
    speculation_straggler_factor: float = 1.5
    speculation_min_finished_frac: float = 0.75
    # Shuffle recompute-on-worker-loss recovery: how many recompute→retry rounds
    # before a still-broken shuffle fails loudly, and the exponential backoff base
    # between rounds (so a flaky network is not retried in a tight loop). A larger
    # cluster with a higher background failure rate may want more attempts.
    recovery_max_attempts: int = 3
    recovery_backoff_base_s: float = 0.5
    # Broken-record tolerance for the distributed scan. A single corrupt file /
    # unreadable row-group otherwise raises out of a worker's read and — because the
    # error is *deterministic* (a rerun fails identically) — the recompute loop retries
    # it `recovery_max_attempts` times and then fails the whole cluster job. Real
    # data-lake tables routinely carry a few bad files, so a fatal read is the wrong
    # default for them but the right one for a small, trusted input. The policy travels
    # with each partition manifest, so it reaches every worker without shipping config:
    #   "error" (default) — any read failure fails the query (today's fail-fast behavior).
    #   "skip" — a split (file / row-group group) that fails to read is skipped and the
    #       scan continues; the count of skipped splits is recorded on the worker
    #       (`skipped_splits()`), so a silent data loss is observable. Skipping isolates
    #       failures per split (the bulk coalesced dataset scan, whose mid-stream decode
    #       error can't be attributed to one split, is bypassed for the per-split reader),
    #       so one bad file never discards its healthy siblings in the same partition.
    on_read_error: str = "error"
    # Ray-level task/actor fault tolerance — the *first* line of defense, beneath the
    # shuffle recompute loop above. A transient task failure (a flaky node, a dropped
    # connection) is retried by Ray itself before the heavier app-level recompute
    # engages. `task_max_retries` reruns a failed shuffle task (deterministic +
    # recomputed from a durable source, so a rerun is safe); `retry_on_transient`
    # extends those retries to application exceptions (not just worker death), gated to
    # transport-classified transient errors once that classification lands.
    # `actor_max_restarts` lets a crashed compute actor (the map/inference pool)
    # respawn, and `actor_max_task_retries` reruns its in-flight call on the respawned
    # actor. 0 anywhere keeps Ray's no-retry/no-restart default. These do not touch the
    # Flight shuffle-server actors, whose loss is handled by the recompute loop.
    task_max_retries: int = 2
    retry_on_transient: bool = True
    actor_max_restarts: int = 1
    actor_max_task_retries: int = 1
    # Timeouts (seconds). `flight_idle_timeout` bounds the gap *between* batches in a
    # shuffle fetch before the peer is treated as dead — generous so a long GC pause
    # isn't misread as death, but bounded so a truly dead peer is detected and its
    # partition recomputed. `flight_keepalive` is the HTTP/2 keepalive ping interval
    # (None/0 = off) that detects a silently-dropped connection faster than the idle
    # timeout alone. `placement_timeout` bounds how long gang-scheduling waits for a
    # worker placement group before falling back to default scheduling (a real
    # cluster may need to autoscale up).
    flight_idle_timeout_s: float = 60.0
    flight_keepalive_s: float | None = None
    # Max TCP connections a reducer stripes one peer's shuffle fetches across. One gRPC
    # channel is one HTTP/2 connection is one TCP flow, and a cloud NIC caps a *single*
    # flow well below line rate (e.g. AWS ~5 Gbps of a 10 Gbps NIC), so funneling a
    # peer's whole shuffle through one connection halves cross-node throughput. The
    # consumer pool grows to this bound only under concurrent fetches to the same peer,
    # so a cold peer still costs one connection. Default 4 saturates a 10–25 Gbps NIC;
    # 1 restores the single-connection behavior.
    flight_connections_per_peer: int = 4
    # Wire compression for shuffle batches: "none", "lz4", or "zstd". A cross-node fetch
    # is NIC-bound, so compressing the Arrow buffers before they cross the wire is the
    # only way past line rate — and real shuffle data (sorted runs, repeated group keys,
    # dictionary strings, nulls) compresses several-fold, unlike the object store's
    # uncompressed blocks. "lz4" (~GB/s/core, gives up fast on incompressible data) is a
    # near-free default; "zstd" trades CPU for a higher ratio; "none" disables it.
    flight_compression: str = "lz4"
    placement_timeout_s: float = 60.0
    # Bounded wait for the Ray autoscaler to grow the cluster before clamping a query's
    # worker fan-out. When a query wants more workers than the cluster can schedule now,
    # `clamp_workers` asks the autoscaler to scale up (`request_resources`) and then
    # waits up to this many seconds — polling every `autoscale_poll_s` — for the new
    # nodes to arrive, so a big job actually *uses* the scaled-up cluster instead of
    # running under-provisioned and only the next job benefiting. Stops early the moment
    # capacity covers the request (and the `autoscale_stall_s` grace window bails fast when
    # capacity goes flat). The default is `AUTOSCALE_WAIT_AUTO` (-1) = "auto": the config
    # layer resolves it to a bounded wait on an autoscaling-capable cluster (detected via
    # env — Anyscale / spot / `BATCHER_AUTOSCALE=1`) and to `0` (off) on a fixed one, so
    # saturation needs no tuning. Set it explicitly to override — `0` disables the wait
    # (honored even on an autoscaling cluster), a positive value caps the budget.
    autoscale_wait_s: float = AUTOSCALE_WAIT_AUTO
    autoscale_poll_s: float = 5.0
    # Grace window for the autoscale wait: while waiting (`autoscale_wait_s > 0`), stop
    # early once cluster capacity has stayed flat for this many seconds — the autoscaler
    # has nothing more to add (a fixed cluster) or cannot satisfy the request (e.g. spot
    # `InsufficientInstanceCapacity`), so blocking the rest of the budget for nodes that
    # will not arrive only delays the query. Any capacity gain resets the window (more may
    # follow). Sized longer than a node's boot time so a genuinely-launching node is not
    # abandoned before it joins.
    autoscale_stall_s: float = 90.0
    # Skew-aware join salting for a huge x huge hot key. When a single join key is
    # dominated by a few "hot" values, those rows otherwise co-partition onto one
    # reducer and overload it (memory + the output explosion + a straggler). With
    # salting on, a pre-pass detects the hot values (Misra-Gries) and the shuffle
    # spreads each hot key's probe rows across `skew_join_salt` reducers while
    # replicating its build rows to all of them — so the hot key's work fans across
    # the cluster instead of one node. 0 (default) disables it: the shuffle is the
    # plain co-partition and single-node==distributed is bit-identical. Single-key,
    # left-driven (inner/left/semi/anti) joins only; other shapes fall back to plain.
    # Opt-in because the detection pre-pass re-scans both inputs — worth it only for a
    # known-skewed huge join, where it prevents a reducer OOM / straggler.
    skew_join_salt: int = 0
    # A value is "hot" when it exceeds this fraction of a side's rows. Lower → more
    # keys salted. Only consulted when `skew_join_salt > 0`.
    skew_join_fraction: float = 0.10
    # Runtime bloom-filter join reduction (sideways information passing). A shuffle
    # join builds a bloom over the small (build/right) side's keys and uses it to drop
    # provably-non-matching rows of the large (probe/left) side *before* they are
    # shuffled — cutting network volume for selective fact⋈dimension joins. Always
    # correct (the bloom has no false negatives, so only non-matching rows are dropped);
    # inner/semi joins only. It serializes the build side's map ahead of the probe's to
    # ready the bloom — a win when the probe is much larger and the join selective, an
    # overhead on balanced joins — so the choice is cardinality-driven:
    #   "auto" (default) — engage only when Kyber estimates the probe is much larger
    #       than the build (see `dist.executors.join._bloom_beneficial`); zero config,
    #       the metadata-driven default.
    #   True  — always engage (for a known-selective workload).
    #   False — never engage (pin the plain shuffle).
    runtime_bloom_join: bool | str = "auto"
    # Shared-secret token authenticating Flight shuffle fetches (N5). When set, a
    # peer must present it to fetch a partition, so a process that can merely reach
    # the port cannot exfiltrate shuffle data. None (default) disables the check —
    # appropriate on a trusted/isolated cluster network. Also read from the
    # `BATCHER_SHUFFLE_TOKEN` env var so it can be injected without a config file.
    shuffle_token: str | None = None
    # Same-node shared-memory shuffle transfer. When on, a mapper mirrors each bucket to
    # a memory-mapped Arrow IPC file (Linux tmpfs `/dev/shm` when available, else a temp
    # dir) and a same-node reducer in another process reads it via mmap ZERO-COPY — no
    # gRPC, no loopback TCP (the plasma-class same-node fast path). Measured 23x over
    # loopback Flight on a real cluster (1.2 -> 27 GB/s cross-process, same node). On by
    # default: the common GPU-cluster shape packs several actors per node, so many shuffle
    # fetches are same-node. It is safe to default on because it is (a) *pressure-gated* —
    # the mapper skips the second tmpfs copy when the node is under memory pressure, so it
    # cannot OOM a tight/churning spot node — and (b) best-effort and result-preserving: a
    # shm miss (bucket not mirrored, another node, or shm unavailable) falls back to
    # Flight, which is bit-identical, so single-node==distributed holds either way.
    shared_memory_transfer: bool = True
    # Locality-aware reducer placement. When on, a reducer whose bucket is concentrated
    # on one node is hosted on an actor on that node, so the bulk of its fetches become
    # same-node (shared-memory/direct) hits instead of network transfers. Result-
    # preserving (placement never changes the output, only where bytes travel), so it is
    # safe; off by default keeps the plain round-robin placement. Pays off on a
    # multi-node cluster with a skewed/co-partitioned shuffle; a no-op for an evenly
    # spread bucket (no node dominates) and on a single node (everything is same-node).
    locality_aware_scheduling: bool = False
    # Persistent shuffle-actor fleet for the adaptive Flight path. When on, an adaptive
    # multi-stage query reserves ONE placement group + worker fleet for the whole query
    # and reuses it across breaker stages: a stage's intermediate stays partitioned on
    # the workers (a `FlightMaterializedSource`) instead of being collected to the
    # driver, and the next stage's mappers read their bucket in place. This removes the
    # per-stage placement-group churn (which would otherwise deadlock — a new stage's
    # gang reservation contending with the prior stage's still-held bundles) and the
    # driver funnel. Off by default: with it off the Flight path collects between stages
    # exactly as before, so single-node==distributed stays bit-identical. Result is
    # unchanged either way (the mergeable algebra guarantees it); this only changes
    # where the bytes live between stages.
    persistent_fleet: bool = False
    # Reuse one shuffle-actor fleet across *separate* distributed queries in a session,
    # so a second `collect(distributed=True)` skips the ~1-2s actor + placement-group +
    # Flight-server spawn that otherwise dominates a short query (measured: a warm sf10
    # group-by is ~1.5s shuffle/compute but pays another ~1.5s spawning the fleet each
    # call). The cached fleet is health-checked before reuse and respawned if a worker
    # died, and auto-released after `session_fleet_idle_s` of no use so an idle session
    # never pins the cluster. Result-identical (same mergeable shuffle, just warm
    # actors). On by default — it is the in-memory-warm-workers win Ray Data gets from a
    # long-lived streaming executor. Disabled automatically while a `persistent_fleet`
    # adaptive query owns a fleet (that one wins, so there is never a second placement
    # group to deadlock against).
    reuse_session_fleet: bool = True
    # Seconds an idle reused session fleet lives before it is torn down and its cluster
    # cores released. Short enough that a finished session frees the cluster promptly,
    # long enough to span the gaps between queries in an interactive/iterative session.
    session_fleet_idle_s: float = 30.0
    # How many times to (re-)attempt a `persistent_fleet` adaptive query on a fresh fleet
    # when a worker dies holding an *already-materialized* cross-stage intermediate (which
    # has no fine-grained recompute, unlike an in-stage loss). The whole deterministic
    # query re-runs on a fresh fleet of survivors, so the result is unchanged; more
    # attempts ride out more preemptions before surfacing a persistent failure. The spot
    # profile raises it for a churning cluster. Only consulted when `persistent_fleet` is on.
    fleet_max_attempts: int = 2
    # Streaming heterogeneous inference pipeline. When on, a linear `map_batches` chain
    # that crosses a resource-class boundary (a CPU preprocess stage feeding a GPU /
    # load-once inference stage) is split into per-stage actor pools that stream
    # partitions stage→stage over Arrow Flight, so the CPU and GPU stages OVERLAP (the
    # GPU runs partition k while the CPU prepares k+1) instead of one actor running the
    # whole chain per partition. Off by default: with it off the chain runs
    # embarrassingly parallel exactly as before, so single-node==distributed stays
    # bit-identical. Result is unchanged either way — only the scheduling overlaps.
    stream_inference: bool = False
    # Keep GPU / load-once inference actor pools WARM across `collect()` calls in a session,
    # keyed by pipeline identity (the model's `map_batches` fn). The model then loads ONCE
    # per session and is reused, instead of every distributed-map call paying the actor spawn
    # + CUDA-context init + model load again (~5 s on 8 GPUs, and the guides' "actors are
    # ~20x slower on the first batch"). The analog of `reuse_session_fleet` for the inference
    # plane — the long-lived-actor win Ray Data leaves on the table (it respawns the pool per
    # execution). Pools are torn down at process exit or by `release_inference_pools()`; a
    # pool whose actors died (preemption) is transparently respawned on next use. On by
    # default; result-identical (same model, same per-batch contract).
    warm_inference_pools: bool = True
    # `channels_last` + `torch.compile` a **vision (CNN)** model in the managed `ds.ml.infer`
    # path — kernel fusion + graph capture for ~2x GPU inference at inference-identical results
    # (measured: 1.9x on ResNet-50, predicted labels unchanged). Applied ONLY to convolutional
    # models: text transformers have dynamic sequence lengths that trigger per-shape recompiles
    # and are tokenization-bound, where compile measured ~0.9x (a regression), so they are left
    # eager. The one-time compile is amortized by the warm pool over a batch-inference job.
    # GPU-gated and fallback-safe (eager on CPU / non-CNN / compile failure / off). On by
    # default; a tiny vision job that can't amortize the warm-up can set this False.
    torch_compile: bool = True
    # Auto mixed-precision for a GPU inference stage: run the model's forward under
    # `torch.autocast` in the accelerator's fast half type (`recommend_inference_dtype` —
    # BF16 on Ampere+, FP16 on Turing/Volta/MPS) so the matmuls/convs hit the tensor cores at
    # ~1.5-2x throughput, while autocast keeps reductions/softmax in FP32 for stability
    # (measured: label agreement ~0.9999 vs FP32). Unlike `torch_compile` this needs no model
    # object — it wraps the opaque per-batch call — so it also optimizes a plain
    # `map_batches(model, num_gpus=...)`, the raw-model path that otherwise runs FP32. Ray Data
    # leaves this to the user (`torch_dtype`); Batcher applies it by default. GPU-gated and
    # fallback-safe (no-op on CPU / a GPU without fast half / when torch is absent / on any
    # failure). Set False to force FP32 (bit-exact repro, or a model that needs full precision).
    autocast_inference: bool = True
    # Ship cuDF (RAPIDS) to the `backend="gpu"` worker tasks so the GPU group-by uses cuDF's
    # mature kernels (~3x the hand-rolled torch fallback, and the engine behind Polars-GPU).
    # cuDF's pip install is cached per node after the first task; numpy stays pinned so returned
    # arrays unpickle on the driver. Off → the torch fallback (no install, slower). On by default.
    gpu_backend_cudf: bool = True
    # Shards per GPU for the distributed GPU aggregate. Fanning out one shard per GPU puts an
    # unbounded slice on each device — a big source then OOMs a single GPU and the whole query
    # collapses to the CPU fallback. Oversubscribing (this many shards per GPU) bounds each
    # shard, and because Ray runs at most `#GPUs` `num_gpus=1` tasks at once the extra shards
    # queue and pipeline: finer granularity load-balances across a heterogeneous / churning spot
    # cluster and makes a spot-preempted shard's retry cheap (1/N the work). The mergeable
    # combine is correct for any shard count. 1 = one shard per GPU (the old behavior).
    gpu_shard_oversubscribe: int = 4
    # Kyber's cost-based GPU-backend policy (`backend="auto"`, and the memory routing under an
    # explicit `backend="gpu"`). Below `gpu_min_rows` estimated rows the fixed GPU overhead —
    # host<->device transfer, kernel launch, first-touch cuDF import — is not amortized, so
    # `auto` stays on the CPU engine (the GPU analog of `distribute_min_rows`). `gpu_memory_gb`
    # is the usable memory budget of ONE GPU: a plan whose estimated working set exceeds it is
    # routed to the distributed GPU aggregate (shard across GPUs) instead of a single-dispatch
    # that would OOM; a set exceeding the whole cluster's GPU memory stays on the (spillable) CPU
    # engine. Defaults target a T4 (16 GB, ~12 usable).
    #
    # `gpu_min_rows` is set from the measured crossover of the distributed GPU group-by vs the
    # native (fast, multi-core Rust) CPU engine on the 8xT4 cluster: at ~4M rows the GPU loses
    # ~5x (host<->device transfer, cuDF import, task dispatch dominate); by ~100M it wins ~2-7x.
    # Break-even sits near ~10M rows, so below that `auto` stays on the CPU engine. Retune if the
    # CPU engine or the GPU data plane gets materially faster.
    gpu_min_rows: int = 10_000_000
    gpu_memory_gb: float = 12.0
    # Estimated GPU activation bytes per row, used to seed a GPU inference stage's initial batch
    # size from the VRAM left after the model (headroom_bytes / this). A coarse per-row estimate
    # the online `ThroughputController` then corrects from measured VRAM/throughput — it only has
    # to put the *starting* batch in the right order of magnitude (small model → big first batch,
    # heavy model → cautious). ~64 KB/row suits typical vision/embedding activations.
    gpu_activation_bytes_per_row: int = 65_536
    # `distributed="auto"` distributes only when it PAYS. On a multi-node cluster the Ray
    # fan-out (SPREAD placement + task dispatch + result gather) is a ~2 s fixed cost, so a
    # small query runs far faster single-node (measured: an 80k-row filter is ~55 ms
    # single-node vs ~2.1 s distributed). Below this estimated input-row count, `auto` stays
    # single-node even on a big cluster — the "sub-second small queries, low fixed overhead"
    # mandate — while a GPU stage always distributes (it must reach the cluster's GPUs) and an
    # unknown/large size distributes as before. Result-identical either way; an explicit
    # `distributed=True/False` always overrides. Set to 0 to always distribute on a cluster.
    distribute_min_rows: int = 1_000_000
    # Cap on the number of shuffle partitions (reducers / hash buckets) an all-to-all
    # exchange creates — aggregate, join, sort, window, distinct. Without a cap the count
    # equals the worker fan-out (one per node), so the exchange is O(nodes²): at 10k nodes a
    # keyed group-by would open ~100M mapper→reducer streams and collapse. The reducer count
    # only needs to be enough to balance keys and keep each reducer's state in memory — past a
    # few thousand it adds shuffle overhead, not parallelism (Spark's 200-default lineage). So
    # `n_reducers = min(workers, this)`: regular clusters (≤ this many nodes) are UNCHANGED,
    # while very large clusters stay bounded. Mergeable algebra makes any reducer count
    # correct (result-identical), so this is purely a scaling knob. 0 disables the cap.
    max_shuffle_partitions: int = 2048
    # Submit-ahead cap on concurrent map/inference partition tasks (`gather_map_results`).
    # Seeding Ray with every partition at once floods the scheduler / object store at high
    # fan-out (the "too many pending tasks" anti-pattern). `0` (default) derives the window
    # from live capacity — `pending_window_factor x` schedulable cores — so ordinary
    # fan-outs still submit everything up front (window >= n is byte-identical to the old
    # behavior) while a 100k-partition job stays bounded. A positive value pins the cap
    # explicitly. Result-identical either way (assembly is index-addressed; only the number
    # of in-flight tasks changes). Set very large to force the old submit-all behavior.
    max_pending_tasks: int = 0
    pending_window_factor: int = 4
    # Stateless map-task placement strategy. `"auto"` (default) resolves SPREAD vs Ray's
    # locality-aware DEFAULT against the live cluster: it keeps SPREAD only where packing
    # is a real risk (many sub-core tasks that DEFAULT would stack onto one node, idling
    # the rest) and prefers DEFAULT otherwise — restoring argument-locality on small
    # clusters and avoiding SPREAD's O(nodes) scheduler cost past `map_spread_node_cap`
    # nodes. `"always"` forces SPREAD (the previous unconditional behavior); `"never"`
    # forces DEFAULT. Placement never changes which rows a partition holds, so the result
    # is identical for any choice — this is a scheduling knob only.
    map_spread: str = "auto"
    # Above this many alive nodes, `"auto"` map placement prefers DEFAULT: SPREAD evaluates
    # every node per task and becomes the scheduler bottleneck at scale, where DEFAULT's
    # utilization balancing already spreads load.
    map_spread_node_cap: int = 100
    # Per-task CPU share below which `"auto"` map placement treats packing as a real risk
    # and keeps SPREAD (sub-half-core tasks would otherwise stack many-per-node under
    # DEFAULT). At or above it, near-whole-core tasks fill nodes naturally, so DEFAULT wins.
    map_spread_pack_share: float = 0.5
    # Keep a relational (CPU) fleet off GPU nodes with a HARD node-class selector. Ray's
    # built-in GPU-node avoidance (`RAY_scheduler_avoid_gpu_nodes`) is best-effort — a CPU
    # shuffle task can still land on an idle GPU node and hold its cores from an inference
    # stage. When this is on AND the live cluster's CPU-only nodes can host the fleet
    # (`cpu_only_can_host`), `dist` requires a tiny amount of the `cpu_node_resource` custom
    # resource so the fleet is *hard*-restricted to nodes that advertise it (a node property
    # re-advertised on respawn, so it survives spot/autoscaler node replacement, unlike
    # ephemeral node IDs). Off by default: the selector is emitted only on a cluster that
    # advertises `cpu_node`, and never when it would make the fleet unschedulable — so a
    # cluster that doesn't opt in is unchanged (Ray's soft avoidance). Result-identical
    # (placement only). Deploy-time: label CPU-only nodes with `resources={"cpu_node": N}`.
    heterogeneous_node_isolation: bool = False
    # The custom resource CPU-only nodes advertise for `heterogeneous_node_isolation`.
    cpu_node_resource: str = "cpu_node"
    # Submit-ahead depth per GPU/inference actor — how many partitions an actor may have in
    # flight at once. `1` (default) is the old one-at-a-time behavior. A depth > 1 keeps a
    # GPU fed across the dispatch/gather round-trip (Ray Data's `max_tasks_in_flight`
    # guidance: shallow pipelines leave GPUs idle, depth 8-16 can multiply utilization).
    # Bounded by `_MAP_INFLIGHT_MAX` in the executor. Result-identical (assembly is
    # index-addressed; only pipeline depth changes).
    map_inflight_depth: int = 1
    # Let measured GPU utilization RAISE the per-actor depth above `map_inflight_depth`
    # (bounded): a stage whose prior runs recorded low utilization submits deeper to fill
    # the GPU; a near-saturated stage keeps the shallow default. On by default; only ever
    # increases depth from a *prior* measurement, so a first run is unchanged.
    map_inflight_adaptive: bool = True
    # Named fault-tolerance profile. ``"default"`` keeps the conservative budgets above
    # (tuned for a stable on-demand cluster — minimal retries, no keepalive, no
    # straggler speculation). ``"spot"`` hardens them as a bundle for a churning
    # spot-node cluster where preemption is continuous: more actor restarts and
    # recompute attempts to ride out repeated loss, HTTP/2 keepalive on to notice a
    # dropped peer fast, and one speculative backup so a degraded-but-alive node cannot
    # stall a barrier. The profile is applied *below* any value set explicitly, so
    # precedence is `explicit override > profile > default` — pin an individual knob to
    # override just that one while keeping the rest of the profile. Resolved once at
    # every config entry point (see `batcher.config.profiles`).
    resilience: str = "default"


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    """Logging and per-query event-log settings — how the engine reports what it did.

    Controls the `batcher.*` logger hierarchy (console + optional rotating file) and the
    structured per-query event log (one JSON document per query: the plan, the
    Kyber/Carbonite decisions, and the measured per-operator profile). Env overrides use
    the ``BATCHER_OBSERVABILITY_*`` prefix (e.g. ``BATCHER_OBSERVABILITY_LOG_LEVEL=DEBUG``,
    ``BATCHER_OBSERVABILITY_EVENT_LOG=0`` to disable the event log).
    """

    # Threshold for the `batcher.*` loggers and the Rust data-plane tracing bridge:
    # one of CRITICAL/ERROR/WARNING/INFO/DEBUG. WARNING by default — quiet unless asked.
    log_level: str = "WARNING"
    # Emit log records to stderr. On by default (at `log_level`); set False for a
    # file-only setup.
    console: bool = True
    # Path to a rotating log file, or None for no file handler.
    log_file: str | None = None
    # Maximum bytes per log file before rotation, and how many rotated files to keep.
    log_file_max_bytes: int = 10_000_000
    log_file_backups: int = 3
    # Record format: "human" (a readable one-line layout) or "json" (one JSON object
    # per record, for log shippers).
    log_format: str = "human"
    # Write a structured per-query event log (the Spark event-log analog). On by default.
    event_log: bool = True
    # Directory for event-log documents. Empty → ``$BATCHER_HOME/logs`` (or
    # ``~/.batcher/logs``), resolved at write time so `config` stays free of filesystem I/O.
    event_log_dir: str = ""
    # Keep at most this many event-log files (oldest pruned on write). 0 → unbounded.
    event_log_max_files: int = 200


@dataclass(frozen=True, slots=True)
class Config:
    """The complete engine configuration — every tunable in one frozen object.

    The single source of truth for engine tunables, grouped into typed sections:
    `execution` (parallelism, morsel size, file splits), `memory` (the memory
    envelope and spill tiers), `flow_control` (credit-based shuffle backpressure),
    `optimizer` (Kyber join planning, cost, and cardinality), `pid` (the adaptive
    batch-size controller gains), `metadata` (where learned statistics live and how
    fast they age), and `distributed` (Ray attachment and shuffle transport).

    Immutable: derive a variant with `replace` (whole-section swap) rather than
    mutating, and read the one in effect via `active_config`. The Rust-relevant
    subset is shipped to the data plane by `engine_config_json`.

    Precedence, highest first: `config_context` > `set_config` > ``BATCHER_*`` env
    vars > a JSON file at ``BATCHER_CONFIG_FILE`` > the defaults below. The env and
    file layers are read once at import; `set_config` / `config_context` override
    them at runtime.

    Examples:
        .. doctest::

            >>> from batcher.config import Config
            >>> Config().execution.morsel_rows
            16384
    """

    execution: ExecutionConfig = ExecutionConfig()
    memory: MemoryConfig = MemoryConfig()
    flow_control: FlowControlConfig = FlowControlConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    pid: PIDConfig = PIDConfig()
    metadata: MetadataConfig = MetadataConfig()
    distributed: DistributedConfig = DistributedConfig()
    observability: ObservabilityConfig = ObservabilityConfig()

    def replace(self, **section_overrides: object) -> Config:
        """Return a new Config with whole sections replaced."""
        return replace(self, **section_overrides)  # type: ignore[arg-type]

    def engine_config_json(self) -> str:
        """Serialize the Rust-relevant execution knobs for the data plane.

        These keys are the wire contract with `bc_ir::EngineConfig` — keep them in
        lockstep with that struct (a Python↔Rust default-parity test guards drift).
        Core ships this string alongside the plan IR on every native execution.

        `memory_budget_bytes` is the soft cap that makes the in-memory engine spill
        stateful operators out of core: `memory.max_memory_bytes` scaled by
        `memory.hard_limit`. `max_memory_bytes` is auto-sensed by the `api` resolver
        for a zero-config query (so the default path *does* get a budget and spills),
        and is `0` (unbounded — stay fully in-memory) only when the user set
        `memory.unbounded_memory` or a caller bypassed the resolver.

        The result is memoized by the knob values: a frozen `Config` re-serializes
        the same string on every native call (and every streaming micro-batch), so
        the `json.dumps` runs once per distinct value tuple rather than per call.
        """
        return _engine_config_json(self._engine_config_values())

    def engine_config_json_with(self, op_budgets: dict[int, int]) -> str:
        """`engine_config_json` plus Kyber's per-operator spill budgets.

        `op_budgets` maps a pre-order `op_id` to its byte envelope
        (`PhysicalPlan.op_budgets()`). The engine budgets each stateful operator
        against *its* entry instead of the single global `memory_budget_bytes`, so a
        small operator no longer spills while a large neighbour assumes the whole
        budget. JSON object keys must be strings; the Rust side parses them back to
        the operator id. An empty map reproduces `engine_config_json` exactly, so
        callers with no `PhysicalOp` DAG (streaming, UDFs, distributed workers) are
        unaffected.
        """
        if not op_budgets:
            return self.engine_config_json()
        cfg = self._engine_config_dict()
        cfg["op_budgets"] = {str(op_id): budget for op_id, budget in op_budgets.items()}
        return json.dumps(cfg)

    def _engine_config_values(self) -> tuple[object, ...]:
        """The Rust-relevant execution knobs as a hashable tuple (the cache key)."""
        return (
            self.execution.morsel_rows,
            self.execution.morsel_bytes,
            self.execution.parallelism,
            self._rust_memory_budget_bytes(),
            self.memory.spill_dir,
            self.memory.spill_compression,
            self.execution.fuse_linear,
            self.execution.shrink_output_dtypes,
            self.execution.bloom_fp_rate,
            self.execution.bloom_min_build_rows,
            self.execution.window_parallel_row_threshold,
            self.execution.radix_parallel_threshold,
            self.execution.sort_merge_fanin,
            self.execution.skew_bucket_factor,
            self.execution.skew_min_bucket_rows,
            self.execution.skew_min_bucket_bytes,
        )

    def _engine_config_dict(self) -> dict[str, object]:
        """The Rust-relevant execution knobs as a plain dict (shared serialization)."""
        return dict(zip(_ENGINE_CONFIG_FIELDS, self._engine_config_values(), strict=True))

    def validate(self) -> Config:
        """Validate the configuration, raising `ConfigError` on a bad value.

        Catches out-of-range and inconsistent tunables (a negative retry count, a
        soft limit above the hard limit, a non-positive timeout) at the config entry
        points so they fail early and clearly instead of surfacing as a confusing
        runtime failure. Returns `self` so it can be chained. Pure (no side effects).
        The checks live in `config.validation` to keep this module focused.
        """
        from batcher.config.validation import validate_config

        validate_config(self)
        return self

    def _rust_memory_budget_bytes(self) -> int:
        """The per-operator spill budget shipped to the data plane (bytes).

        Derived statically from `MemoryConfig` so `config` stays neutral — it never
        senses (the `api` auto-tuning resolver fills `max_memory_bytes` from the live
        envelope before execution; see `api._autotune`). `0` means unbounded (stay
        fully in-memory): returned when the user opted out via `unbounded_memory`, or
        when `max_memory_bytes` is still unset because a caller bypassed the resolver
        (an ad-hoc `Config`), in which case the safe pre-auto-tuning behavior holds.
        """
        mem = self.memory
        if mem.unbounded_memory:
            return 0
        cap = mem.max_memory_bytes
        if cap is None or cap <= 0:
            return 0
        return int(cap * mem.hard_limit)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None, base: Config | None = None) -> Config:
        """Overlay ``BATCHER_<SECTION>_<FIELD>`` env vars onto `base` (defaults).

        Nested sections compose by path, e.g.
        ``BATCHER_OPTIMIZER_CARDINALITY_EQ_SELECTIVITY``.
        """
        env = os.environ if environ is None else environ
        return _resolved(_overlay_env(base if base is not None else cls(), "BATCHER", env))

    @classmethod
    def from_file(cls, path: str | os.PathLike[str], base: Config | None = None) -> Config:
        """Overlay a JSON document of nested section overrides onto `base`.

        The JSON mirrors the section structure, e.g.
        ``{"execution": {"morsel_rows": 4096}, "optimizer": {"cardinality": {...}}}``.
        """
        data = json.loads(Path(path).read_text())
        return _resolved(_overlay_dict(base if base is not None else cls(), data))


def _coerce(raw: str, to: type) -> object:
    if to is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if to is int:
        return int(raw)
    if to is float:
        return float(raw)
    return raw


def _overlay_env(obj: Config, prefix: str, env: dict[str, str]) -> Config:
    """Recursively overlay env vars onto a (possibly nested) frozen config object."""
    updates: dict[str, object] = {}
    for field in dataclasses.fields(obj):
        current = getattr(obj, field.name)
        key = f"{prefix}_{field.name.upper()}"
        if dataclasses.is_dataclass(current):
            replaced = _overlay_env(current, key, env)  # type: ignore[arg-type]
            if replaced is not current:
                updates[field.name] = replaced
        elif key in env:
            updates[field.name] = _coerce(env[key], type(current))
    return replace(obj, **updates) if updates else obj


def _overlay_dict(obj: Config, data: dict[str, object]) -> Config:
    """Recursively overlay a nested dict of overrides onto a frozen config object."""
    fields = {f.name for f in dataclasses.fields(obj)}
    updates: dict[str, object] = {}
    for name, value in data.items():
        if name not in fields:
            continue
        current = getattr(obj, name)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            updates[name] = _overlay_dict(current, value)  # type: ignore[arg-type]
        else:
            updates[name] = value
    return replace(obj, **updates) if updates else obj


def _resolved(cfg: Config) -> Config:
    """Auto-select the spot profile on a preemptible node, apply the resilience profile,
    auto-enable the bounded autoscale wait on an autoscaling cluster, then validate — the
    single resolution chokepoint every config entry point shares so auto-detection, the
    profile, and range checks run in lockstep regardless of how the config was built. A
    user-chosen `resilience` / `autoscale_wait_s` is never overridden (explicit wins)."""
    from batcher.config.profiles import (
        apply_resilience_profile,
        detect_spot_environment,
        resolve_autoscale_wait,
    )

    if cfg.distributed.resilience == "default" and detect_spot_environment():
        cfg = cfg.replace(distributed=replace(cfg.distributed, resilience="spot"))
    cfg = resolve_autoscale_wait(apply_resilience_profile(cfg))
    return cfg.validate()


# Active-config plumbing -------------------------------------------------------


def _initial_config() -> Config:
    """Layer the static config sources once at import: defaults < file < env."""
    base = Config()
    path = os.environ.get("BATCHER_CONFIG_FILE")
    if path:
        base = Config.from_file(path, base=base)
    return Config.from_env(base=base)


_active: contextvars.ContextVar[Config] = contextvars.ContextVar(
    # Config is a frozen, immutable dataclass, so sharing one default instance is
    # safe — B039's mutable-shared-default hazard does not apply here.
    "batcher_active_config",
    default=_initial_config(),  # noqa: B039
)


def active_config() -> Config:
    """The Config in effect for the current context."""
    return _active.get()


def set_config(config: Config) -> None:
    """Set the process-wide active Config (above env/file, below `config_context`).

    Validates `config` first, so a bad tunable raises `ConfigError` here rather than
    surfacing later as a confusing runtime failure.

    Examples:
        .. doctest::

            >>> from batcher.config import Config, ExecutionConfig, active_config, set_config
            >>> set_config(Config().replace(execution=ExecutionConfig(morsel_rows=4096)))
            >>> active_config().execution.morsel_rows
            4096
            >>> set_config(Config())  # restore defaults
    """
    _active.set(_resolved(config))


@contextlib.contextmanager
def config_context(config: Config) -> Iterator[Config]:
    """Temporarily activate `config` for the duration of the `with` block.

    Validates `config` on entry (raises `ConfigError` on a bad value).

    Examples:
        .. doctest::

            >>> from batcher.config import Config, ExecutionConfig, active_config, config_context
            >>> cfg = Config().replace(execution=ExecutionConfig(morsel_rows=4096))
            >>> with config_context(cfg):
            ...     active_config().execution.morsel_rows
            4096
            >>> active_config().execution.morsel_rows
            16384
    """
    resolved = _resolved(config)
    token = _active.set(resolved)
    try:
        yield resolved
    finally:
        _active.reset(token)
