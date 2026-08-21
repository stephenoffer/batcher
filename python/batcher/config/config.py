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
import typing
from collections.abc import Iterator
from dataclasses import dataclass, replace

from batcher.config.accelerator import AcceleratorConfig
from batcher.config.env import falsy, truthy
from batcher.config.fault_tolerance import FaultToleranceConfig

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
    "streaming",
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


@functools.lru_cache(maxsize=128)
def _engine_config_json_budgeted(
    values: tuple[object, ...],
    budgets: tuple[tuple[int, int], ...],
    prefer_materializing_aggregate: bool = False,
) -> str:
    """`_engine_config_json` plus per-operator spill budgets, memoized on both parts.

    The budgeted payload is rebuilt for the *same* plan on every run of a repeated query
    and on every micro-batch of a streaming query — where the operator DAG, and so the
    budget map, is fixed for the life of the stream. Keying the cache on the sorted budget
    items collapses that to one `json.dumps` per distinct (config, budget) pair.
    """
    payload = dict(zip(_ENGINE_CONFIG_FIELDS, values, strict=True))
    payload["op_budgets"] = {str(op_id): budget for op_id, budget in budgets}
    if prefer_materializing_aggregate:
        payload["prefer_materializing_aggregate"] = True
    return json.dumps(payload)


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
    #: Skip the conductor's per-query orchestration for plans that provably do not need it:
    #: run Kyber (through its plan cache) and the engine, and nothing else.
    #:
    #: On a small query the orchestration, not the engine, is the cost. Measured at 10,000
    #: rows with the event log off: `collect()` 1.935 ms, of which `execute_plan` plus the
    #: Arrow table build is **0.360 ms** — the other 82% is admission, adaptive morsel
    #: sizing, pressure classification, the resource decision, profile assembly, the event
    #: bus, and the learned-stats close-out. DuckDB answers the same query in 0.780 ms, so
    #: the skipped work is the whole of the gap and then some.
    #:
    #: **Result-invariant**: the fast path runs the same optimized plan through the same
    #: `core.execute_local` the ordinary path calls, so it returns the same rows, names and
    #: types. What it gives up is *adaptivity and observability*, which is why it is off by
    #: default and narrowly gated (`api/orchestration/fast_path.py::eligible`) to plans that
    #: cannot need them: small, in-memory, single-node, no UDF, no spill.
    #:
    #: The real cost is the **cross-query learned-stats loop** — a query answered on this
    #: path teaches Kyber nothing, so a repeatedly-issued small query never sharpens its own
    #: estimates. Leave this off if you rely on learned selectivity or cardinality; turn it
    #: on for a latency-sensitive serving path where the plan shape is already known good.
    #: (The *intra*-query adaptive loop is already off below 20M rows, so only the
    #: cross-query half is at stake.)
    #:
    #: This flag also gates the **prepared-execution cache**
    #: (`api/orchestration/prepared.py`), which memoizes a query's whole derivation so a
    #: *re-issued* one is a dict lookup plus the engine call. That is where most of the
    #: measured win now is: over five small shapes at 1,000 rows, per-query control-plane
    #: overhead falls **25x** (5.30 ms to 0.21 ms in total) and end-to-end latency **7.3x**.
    #: The cache inherits this gate and this trade wholesale — it never widens either.
    fast_path: bool = False
    # Fuse runs of linear, per-morsel streaming operators (Filter/Project) into a single
    # pass over the input's morsels in the parallel executor, instead of one rayon
    # dispatch + intermediate buffer per operator. Result-invariant (same rows, same
    # order — verified against the sequential oracle and the full DuckDB differential
    # suite); shipped to Rust as `EngineConfig.fuse_linear`. On by default — it only
    # engages on a chain of ≥2 fusable ops (so single-op and breaker-only plans are
    # untouched) and is a measured win on linear pipelines with no regression elsewhere.
    # Set to False to pin the staged operator-at-a-time path.
    fuse_linear: bool = True
    #: Run plans on the **streaming** executor: pull morsels through the linear runs and
    #: materialize only at breakers, instead of collecting every operator's full output.
    #: Peak memory becomes a constant (the breakers' state plus one morsel per worker) rather
    #: than the sum of every intermediate — which is why TPC-H sf100's deep join trees peaked at
    #: 133 GB and were OOM-killed — and on those shapes it is also *faster*, because the copies
    #: it stops making were not free. Shipped to Rust as `EngineConfig.streaming`.
    #:
    #: The streaming breakers fold in memory rather than spilling, so one whose state exceeds
    #: `memory.max_memory_bytes` hands the query back and it is re-run on the materializing
    #: executor, which spills. Set False to force that executor for every query — a bisecting
    #: escape hatch, not a tuning knob.
    streaming: bool = True
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
    # Partial-row count above which aggregate `combine` regroups via parallel hash-radix
    # partitioning. `0` (the default) derives it from the machine: the crossover is a fixed
    # number of rows *per partition*, so it scales with the core count rather than being one
    # constant that is too high on a big box and too low on a small one. A positive value pins
    # it — performance only, never a different result.
    radix_parallel_threshold: int = 0
    # Maximum runs merged per pass in the external (spilling) sort's k-way merge.
    sort_merge_fanin: int = 16
    # A join bucket is "hot" when it exceeds this multiple of the average bucket.
    skew_bucket_factor: int = 4
    # Absolute row floor below which a join bucket is never treated as skewed.
    skew_min_bucket_rows: int = 4 * 16_384
    # Absolute byte floor below which a join bucket is never treated as skewed.
    skew_min_bucket_bytes: int = 4 * (1 << 20)

    # --- UDF process isolation ----------------------------------------------------
    # What a `map_batches` UDF child process inherits from the engine:
    #
    #   "none"   - the child inherits everything (pre-hardening behaviour). For an
    #              embedder whose UDFs are as trusted as its own code, and who needs a
    #              variable this engine has no way to know about.
    #   "env"    - the child's environment is rebuilt from `udf_env_allowlist`, so it
    #              cannot read `env:` secret material or `BATCHER_SECRET_COMMAND`.
    #              The default: it closes a real exposure and costs one dict rebuild
    #              per child, once, at pool startup.
    #   "strict" - "env" plus the resource ceilings below and the UDF timeout.
    #
    # This is defense in depth, NOT a sandbox: a UDF is arbitrary Python in a process
    # that can reach any syscall through `ctypes`. Untrusted code belongs in a
    # container. See `core/udf/isolation.py` for why an import allowlist would be a
    # false claim rather than a defence.
    udf_isolation: str = "env"
    # Extra environment variables a UDF child keeps, on top of the built-in allowlist
    # (PATH/HOME/TMPDIR/locale/thread+device pinning). Name what your UDFs need.
    udf_env_allowlist: tuple[str, ...] = ()
    # Address-space ceiling per UDF child, 0 to inherit. Under "strict" this is what
    # actually stops a runaway allocation: it raises `MemoryError` inside the guilty
    # child instead of letting the kernel's OOM killer pick a victim, which on a
    # shared box is frequently some other process.
    udf_memory_limit_bytes: int = 0
    # CPU-seconds per UDF child, 0 to inherit. Bounds an infinite loop without the
    # driver having to be watching.
    udf_cpu_limit_seconds: int = 0
    # Wall-clock seconds a single UDF pool map may take, 0 for no limit. Without it a
    # wedged child hangs the query forever with no error and no signal; with it the
    # query raises and names the UDF.
    udf_timeout_s: float = 0.0

    # --- Concurrency admission --------------------------------------------------------
    # How many queries may execute at once in this process. 0 (the default) is unbounded,
    # which is today's behavior; a positive value bounds them and divides the cores among
    # the ones running.
    #
    # This exists because of a measurement, not a theory: going from 1 concurrent client
    # to 16 made throughput FALL (124 -> 88 QPS) while p50 rose 7.6 ms -> 178 ms. Each
    # query asks the executor for a pool sized to every core, so sixteen of them put ~240
    # runnable threads on 15 cores. See `carbonite/policies/concurrency.py`.
    max_concurrent_queries: int = 0
    # Queries allowed to wait for a slot before an arrival is refused. An unbounded queue
    # is an outage that presents as slowness; the number matches the queue depth the
    # Databricks comparator documents, so it is defensible rather than invented.
    admission_queue_depth: int = 1000
    # Seconds a query waits for a slot before raising `AdmissionTimeout`. 0 waits forever.
    admission_timeout_s: float = 0.0


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
    # Treat the cgroup v2 `memory.high` throttle threshold — not just the `memory.max`
    # kill threshold — as the ceiling the engine budgets against. Past `memory.high` the
    # kernel does not fail an allocation; it puts every allocating task to sleep in direct
    # reclaim, so a query planned to sit between `high` and `max` runs at a fraction of its
    # rate for its whole duration while every counter reports success. Kubernetes memory QoS
    # sets `memory.high` from the pod's *request* and `memory.max` from its *limit*, which is
    # exactly that gap. On by default and inert wherever `memory.high` is unset (bare metal,
    # cgroup v1, most non-K8s containers), which is the behavior the engine already had.
    respect_cgroup_high: bool = True
    # Let the kernel's memory PSI `full` share raise the pressure level. `full` is the share
    # of a window in which *every* runnable task in the cgroup was stalled on memory: it
    # climbs for seconds before an OOM kill while `memory.current` sits pinned at the limit
    # the reclaim is defending, so it is the only warning early enough to spill on. It can
    # only ever raise the level (never lower one the byte accounting reported), so with it on
    # the engine spills sooner and never later.
    stall_aware_pressure: bool = True
    # Fraction of the memory envelope kept when this cgroup's `memory.events` shows it has
    # already been OOM-killed. A kill is proof — not a prediction — that the workload does not
    # fit at the size it last ran, so a restarted worker that re-derives the same envelope
    # walks into the same kill. `1.0` disables the backoff.
    oom_kill_backoff: float = 0.8
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
    #
    # `spill_local_budget_bytes` is `None` (auto: derived from measured free disk),
    # a positive byte cap, or `0` — no local scratch, so every bucket overflows
    # straight to `spill_remote_uri` (the "local disk already full" tier, e.g. a
    # node with no usable NVMe). `0` without a remote URI keeps everything local.
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
    #
    # `"auto"` puts it on whatever fast local disk each node has, which is the only way to
    # enable it once for a fleet: the right directory is a per-node fact (`/ephemeral` on one
    # provider, `/mnt/local_disk` on the next), so a literal path in a shared config is the
    # wrong one everywhere but the machine it was written for. A node with no fast local disk
    # resolves `"auto"` to no cache at all rather than competing for the container overlay
    # that the read it is caching would otherwise never touch.
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

        Examples:
            .. doctest::

                >>> from batcher.config import MemoryConfig
                >>> cfg = MemoryConfig(streaming_state_max_bytes=256 << 20)
                >>> cfg.streaming_state_budget_bytes()
                268435456

        Returns:
            The per-operator streaming-state cap, in bytes.
        """
        if self.streaming_state_max_bytes > 0:
            return self.streaming_state_max_bytes
        base = (
            self.max_memory_bytes if self.max_memory_bytes is not None else self.default_total_bytes
        )
        return int(base * self.hard_limit)


@dataclass(frozen=True, slots=True)
class StreamingConfig:
    """Cadence and bookkeeping for a long-running streaming query.

    These govern the micro-batch *loop*, not what it computes: how long an idle stream
    waits before looking for data again, how much progress history a query handle keeps,
    and how long a silent partition holds the event-time watermark back. The memory a
    streaming operator's state may hold is `MemoryConfig`'s `streaming_state_max_bytes`,
    not here.

    Examples:
        .. doctest::

            >>> from batcher.config import StreamingConfig
            >>> StreamingConfig().idle_poll_seconds
            0.2
    """

    # How long a runner waits before asking an *idle* unbounded source for data again.
    # Only the empty path pays it: an epoch with rows is staged immediately. Too small
    # burns a core re-listing a directory or re-asking a broker for its partitions; too
    # large adds latency to the first row after a quiet stretch. The single-node and
    # distributed runners share this value, so an idle stream behaves the same on one
    # machine and on a cluster.
    idle_poll_seconds: float = 0.2
    # Longest a `map_batches` streaming window may spend filling before it is flushed
    # anyway, in seconds. Applies ONLY to an unbounded (streaming) source; a bounded one
    # keeps the pure size-based window, so batch throughput is unchanged.
    #
    # `stream_windowed` batches source batches into a window before applying the UDF,
    # because a `map_batches` pipeline only fans across the worker pool when it is handed
    # several batches at once. It closed that window on rows (`target_rows_per_task`,
    # 4,000,000) or bytes (128 MiB) — both *size* bounds, and a stream is not bounded in
    # size but in rate. The result was that first output waited for four million rows to
    # arrive: at 2,000 rows/s that is 33 minutes, and on a 10 rows/s device topic about
    # 4.6 days. The pipeline was not hung and not leaking; it was buffering, and nothing
    # said so.
    #
    # A time bound is what every streaming engine uses for the same reason (Spark's
    # `Trigger.ProcessingTime`, Flink's `bufferTimeout`): flush on size *or* age, whichever
    # comes first, so throughput still governs a fast stream while a slow one stays
    # responsive. The window is only ever cut *earlier*, and `map_batches` makes no
    # guarantee about how rows are grouped into calls (the existing window already varies
    # with row width and worker count), so this cannot change a result.
    #
    # One second is Spark's own default trigger cadence, and it is checked when a batch
    # arrives rather than on a timer — a source that blocks for a minute between batches
    # flushes on the next arrival, not mid-read.
    max_window_latency_seconds: float = 1.0
    # Micro-batch progress records a `StreamingQuery` handle retains for
    # `recent_progress`. Bounded because a query that runs for a week at a 200ms
    # cadence produces three million of them; Spark's `recentProgress` is bounded the
    # same way (`spark.sql.streaming.numRecentProgressUpdates`, default 100).
    progress_history: int = 100
    # Processing-time seconds a stream partition may deliver nothing before it stops
    # holding the event-time watermark back (Flink's `WatermarkStrategy.withIdleness`).
    #
    # The watermark is the *minimum* over per-partition event-time maxima, because that is
    # the strongest claim a multi-partition stream can actually support. A minimum has one
    # failure mode: a partition that goes silent — an empty Kafka partition, a shard with no
    # writers — pins the watermark forever, so no window ever closes and the retained state
    # grows until the memory cap fires. Idleness is the release valve, and it is a real
    # trade rather than a free fix: a partition idle for longer than this finds the rows it
    # eventually delivers ruled late. Raise it when a partition is legitimately bursty;
    # set it to zero (or below) to disable idleness entirely and get the fully conservative
    # watermark that never advances past a silent partition.
    watermark_idle_timeout_seconds: float = 60.0
    # Derive each trigger's admission cap from the query's own measured throughput, instead
    # of holding it at whatever `max_offsets_per_trigger` was hand-set to (Spark's
    # `spark.streaming.backpressure.enabled`).
    #
    # A micro-batch that overruns its interval leaves the next one starting late against a
    # larger backlog, which overruns by more; the divergence compounds and ends not in a slow
    # query but in the epoch that no longer fits in memory. A static cap bounds that, but it
    # has to be set for the worst trigger the query will ever see, so it throttles every other
    # one and goes stale as soon as the cluster, the data, or the plan changes.
    #
    # Off by default, as it is in Spark, and it only ever *lowers* a source's configured
    # limit. An admission cap changes how much of a stream a trigger reads, never what the
    # query computes from it, so it cannot change a result. See
    # `carbonite.policies.rate_control`.
    backpressure_enabled: bool = False
    # PID weights, with Spark's names and Spark's defaults so an operator's existing tuning
    # advice carries over verbatim (`spark.streaming.backpressure.pid.*`).
    #
    # The integral term is the one worth understanding before changing: it is what removes
    # *steady-state* error. A purely proportional controller settles at a rate slightly above
    # what the query can sustain and then stays permanently a little behind — which is the
    # compounding case this exists to prevent, arrived at more slowly.
    backpressure_pid_proportional: float = 1.0
    backpressure_pid_integral: float = 0.2
    backpressure_pid_derivative: float = 0.0
    # Floor on the derived rate, in rows per second (Spark's `pid.minRate`). A controller that
    # can reach zero can stall a query permanently: a trigger admitting nothing publishes no
    # progress record, and a controller with no progress never revises the cap that stalled it.
    backpressure_min_rate: float = 100.0
    # An operator's own ceiling on the derived cap, in rows per trigger. `0` leaves the
    # estimator unbounded above, which is the useful default because the source's configured
    # limit already bounds it — this is for pinning a hard maximum independently of the source.
    backpressure_max_rows_per_trigger: int = 0
    # How many changelog deltas a stateful checkpoint may record before it writes a whole
    # snapshot again. Zero disables incremental checkpointing entirely.
    #
    # Rewriting the running state every micro-batch makes the checkpoint's cost grow with
    # the state it protects: an aggregate with no watermark never evicts, so a query that
    # accumulates ten million groups pays a ten-million-row write on every trigger, forever.
    # A delta costs the *batch's* distinct group count instead, and recovery combines the
    # newest snapshot with the deltas after it (sound because `combine` is associative and
    # commutative, invariant #7).
    #
    # The interval is what bounds recovery: a longer chain writes less and replays more. Ten
    # keeps the replay to at most ten combines while cutting the steady-state write cost by
    # roughly the same factor, and the size rule in
    # `core.streaming_query.state_policy::write_state` refuses a delta that is not actually
    # smaller — so a stream whose every batch touches every group falls back to whole
    # snapshots and can never be worse off than before.
    checkpoint_delta_interval: int = 10

    def __post_init__(self) -> None:
        """Reject a cadence or history that cannot mean anything."""
        if self.checkpoint_delta_interval < 0:
            raise ValueError(
                "streaming.checkpoint_delta_interval must be >= 0 (0 = always snapshot "
                f"whole state), got {self.checkpoint_delta_interval}"
            )
        if self.idle_poll_seconds <= 0:
            raise ValueError(
                f"streaming.idle_poll_seconds must be > 0, got {self.idle_poll_seconds}: "
                "a zero wait spins a core on every idle stream"
            )
        if self.progress_history < 1:
            raise ValueError(
                f"streaming.progress_history must be >= 1, got {self.progress_history}"
            )
        # A floor at or below zero is the one setting that can stall a stream permanently: a
        # trigger admitting nothing publishes no progress record, and a controller with no
        # progress never revises the cap that stalled it. Rejected rather than clamped,
        # because unlike a transient tunable this one cannot recover on its own.
        if self.backpressure_min_rate <= 0:
            raise ValueError(
                f"streaming.backpressure_min_rate must be > 0, got "
                f"{self.backpressure_min_rate}: a floor of zero lets the rate controller "
                "throttle a stream to a standstill it cannot measure its way out of"
            )
        if self.backpressure_max_rows_per_trigger < 0:
            raise ValueError(
                "streaming.backpressure_max_rows_per_trigger must be >= 0 (0 = unbounded), "
                f"got {self.backpressure_max_rows_per_trigger}"
            )
        # Negative weights inject positive feedback: the controller would answer an overrun
        # by admitting *more*, which is the runaway the whole mechanism exists to prevent.
        for name, weight in (
            ("proportional", self.backpressure_pid_proportional),
            ("integral", self.backpressure_pid_integral),
            ("derivative", self.backpressure_pid_derivative),
        ):
            if weight < 0:
                raise ValueError(
                    f"streaming.backpressure_pid_{name} must be >= 0, got {weight}: a "
                    "negative weight makes the controller speed up when it falls behind"
                )


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
            16
    """

    # Credit window (in-flight RecordBatch slots) a shuffle channel *starts* at when the
    # operator has no learned estimate. One credit = one buffered batch, so this bounds a
    # channel's memory; the AIMD controller then slow-starts it up to the bandwidth-delay
    # product and holds it there. A cross-node fetch's throughput is `window x batch / RTT`,
    # so the start matters most for the first/short fetches AIMD can't yet have ramped:
    # measured on a 50 ms-RTT link, a single 18 MiB partition transfers at 2.4 MiB/s at 4
    # credits vs 7.7 MiB/s at 16 (3.2x) — the old default of 4 throttled every cross-node
    # shuffle's opening rounds. 16 credits is ~16 MiB/channel of narrow-row buffering (the
    # byte ceiling below re-shrinks it for wide rows), well within the per-channel budget.
    # Carbonite is the authority that supplies it, and clamps any per-operator request to
    # `default_credits x ceiling`. Shipped to Rust as `EngineConfig`.
    default_credits: int = 16
    # Max window = default_credits x this. Kept so the ceiling (16 x 4 = 64 credits ≈ 64
    # MiB/channel of narrow rows) is unchanged from the historical `4 x 16`; only the
    # *starting* window rose, not the memory ceiling.
    credit_ceiling_factor: int = 4
    # Byte ceiling for one shuffle channel's credit window. A credit ≈ one
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
    # Max mapper buckets a *flat* gather (join / sort / window reduce) streams at once.
    # Distinct from `shuffle_fan_in` above, which bounds the aggregate's combiner-TREE
    # depth so a node never *folds* more than that many partials. A flat gather has no
    # tree: the reducer concatenates every mapper's bucket and therefore holds all of it
    # regardless, so throttling the fetch to the tree's fan-in buys no memory — it only
    # serializes the network. At the old shared value of 8, a 16-worker cluster pulled its
    # buckets in two half-idle waves and a reducer's fetch ran at ~39 MB/s. In-flight
    # memory stays bounded by the per-channel credit window (`credit_byte_budget`), which
    # is the actual buffering governor; this only caps how many peers are dialed at once,
    # so a thousand-mapper shuffle still can't open a thousand sockets.
    shuffle_fetch_fan_in: int = 32
    aimd_alpha: int = 1  # additive increase: +1 credit / RTT
    aimd_beta: float = 0.5  # multiplicative decrease on congestion
    backpressure_high: float = 0.70
    backpressure_low: float = 0.40


@dataclass(frozen=True, slots=True)
class CardinalityConfig:
    """Selinger-style defaults the cardinality estimator falls back to.

    Cold-start values only: superseded by learned/sketch statistics once a query has run
    and the measured selectivities are available.

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
    # String-pattern predicates (`LIKE`, `contains`, `starts_with`, a regex match).
    # Without a string histogram their true selectivity is unknowable, but they are
    # near-universally *selective* — a substring search that matched half the table
    # would not be worth writing. Falling back to `default_filter_selectivity` (0.5)
    # made Kyber believe TPC-H Q9's `p_name LIKE '%green%'` kept 100k of 200k parts
    # (it keeps 10.7k), hiding the most selective join in the query and steering the
    # order into gigabyte intermediates. These are the conventional optimizer defaults
    # (Postgres/Spark use the same order of magnitude) and are cold-start values only:
    # the learning loop replaces them with the measured selectivity on re-execution.
    substring_selectivity: float = 0.05  # col LIKE '%x%' / contains / regex match
    prefix_selectivity: float = 0.10  # col LIKE 'x%' / starts_with / ends_with
    # A value appearing in at least this fraction of a column's rows is recorded as a
    # most-common-value (MCV), so `col = <that value>` uses its measured frequency
    # instead of the uniform `1/ndv` — the skew case where `1/ndv` is most wrong.
    mcv_min_fraction: float = 0.05


@dataclass(frozen=True, slots=True)
class CostWeights:
    """Relative importance of each axis when collapsing `Cost` to a scalar.

    Swapped per query to honor a latency / cost / throughput target.

    Examples:
        .. doctest::

            >>> from batcher.config import CostWeights
            >>> CostWeights().net  # a shuffled byte costs twice a local one
            2.0
    """

    cpu: float = 1.0
    io: float = 1.0
    net: float = 2.0  # shuffle bytes hurt more than local bytes


@dataclass(frozen=True, slots=True)
class CostCoefficients:
    """Per-unit costs, in abstract, mutually-comparable "work units" per row or byte.

    Constants today; calibrated from measured `op_stats` later.

    Examples:
        .. doctest::

            >>> from batcher.config import CostCoefficients
            >>> CostCoefficients().hash_build_row  # inserting costs twice a probe
            2.0
    """

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
    # The divisor `expr_cost` applies to an expression the Cranelift tier compiles — the
    # single parameter separating compiled from interpreted pricing. A prior until
    # `calibration` fits it: the engine tags each operator with the tier that ran it
    # (`op_stats.backend`), and the ratio of the two tiers' expression-normalized per-row
    # times says how much the model misprices one against the other. It is a fitted model
    # parameter, not a hardware benchmark — it also absorbs systematic error in the
    # hand-written interpreted-expression cost table.
    jit_speedup: float = 4.0


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
    # Floor on rewrite-phase iterations. The effective bound is `max(this, plan_depth + 4)`:
    # pushdown descends one level per iteration, so the distance to a fixpoint scales with plan
    # depth (`kyber.optimizer._fixpoint_bound`). A convergent phase still exits early.
    fixpoint_iterations: int = 8
    row_bytes: int = 64  # per-row footprint for the memory-budgeting estimate
    # `split_expensive_filter`: the engine's `and` evaluates BOTH operands over every
    # row (no short-circuit, no selection vector), so a conjunction pairing a cheap
    # selective predicate with an expensive one pays the expensive one on rows the
    # cheap one would have dropped. Splitting into stacked `Filter`s makes the second
    # see only survivors, at the price of materializing one extra compacted batch.
    # `filter_split_materialize_cost` is that price, in the same per-row work-units as
    # `CostCoefficients` (~one `filter_row`); `filter_split_min_gain` is the cost ratio
    # a split must beat before it is taken, so marginal rewrites are left alone.
    filter_split_materialize_cost: float = 1.0
    filter_split_min_gain: float = 1.25
    # Plan-level common-subplan elimination (`kyber.common_subplan`): the largest
    # materialized result worth holding so a subplan appearing more than once in a query
    # is executed once. A repeated subtree costs a full extra execution — a `GROUP BY`
    # feeding both operands of a join runs twice — and reuse trades that for holding the
    # result until the query ends, so the budget is the whole gate. Sized like
    # `memory.result_cache_max_bytes` and for the same reason; `0` turns the rewrite off.
    common_subplan_max_bytes: int = 256 * 1024 * 1024  # 256 MiB
    # Build-side byte threshold below which a join is broadcast (the right side is
    # replicated to every worker) rather than shuffled — Spark's
    # autoBroadcastJoinThreshold. Both the planner's *estimate*-based decision and the
    # distributed executor's *runtime* guard read this one value: if the materialized
    # build side actually exceeds it (the estimate was wrong), the executor falls back
    # to a shuffle join instead of OOMing the driver by replicating an over-large side.
    #
    # Sized to *cache*, not to memory. A broadcast join builds ONE hash table and probes
    # it from every core, so each probe row is a random access into it: the strategy wins
    # only while that table stays cache-resident. Past it, the partitioned (shuffle) join
    # wins, because each of its buckets probes a small, L2-resident table instead. TPC-H
    # sf1 measures the crossover between 4 and 10 MiB (q3's 4.4 MB build over a 3.2M-row
    # probe: 52 ms partitioned vs 83 ms broadcast), so the table — not the machine's RAM —
    # is what this bounds.
    #
    # NOTE: this is a *true* byte size. It was previously read against a flat 64 B/row
    # width estimate that over-sized narrow relations ~4x (a two-`int64` key costed as
    # 64 B/row, not 16), so the effective threshold was ~4x smaller than its nominal
    # 10 MiB. `plan.types.widths` now makes the width type-exact; this value is the
    # recalibrated equivalent.
    # `0` (the default) means **detect it from the last-level cache** — see
    # `resolved_broadcast_max_bytes`. A positive value pins the threshold, for a machine whose
    # cache the probe cannot read (a non-Linux host) or to deliberately force a strategy.
    broadcast_max_bytes: int = 0
    # The shuffle volume below which co-locating (PACK) a small shuffle's workers beats
    # spreading them — a *network* decision, split out from `broadcast_max_bytes` which is a
    # *cache* decision. The two shared one knob, so an L3-sized broadcast threshold would have
    # silently moved a placement choice with it; they answer different questions and now have
    # different homes. Kept at the historical 4 MiB so placement behavior is unchanged.
    locality_max_bytes: int = 4 * 1024 * 1024  # 4 MiB
    # Static blend weight, NOT an EWMA rate: how far a *single* decision moves from its
    # plan estimate toward a measured value (`LearnedMemoryModel.blend_peak`, the pressure
    # EWMA, GPU/stream autobatch). Per-signature scalars that accumulate observations use
    # `learned_scalar_alpha_floor` instead — the two were one knob, with one value serving
    # two incompatible meanings.
    learning_smoothing_alpha: float = 0.5
    # Floor on the step of a per-signature scalar's exponential moving average
    # (`kyber.learning._smooth`, `kyber.learned_tuning.priors._smooth`). The step is
    # `max(floor, 1/(n_obs+1))`: a running mean while evidence is thin, then an EWMA with a
    # ~`1/floor`-observation memory. A floor of 0.5 — the old shared value — meant the
    # newest run always carried half the weight, so a learned join size or partition count
    # never converged and one anomalous run swung it by 50%.
    learned_scalar_alpha_floor: float = 0.1
    # Learned cardinality correction: Core reports, per operator, the rows it actually
    # produced against the rows Kyber estimated *before* correction. The geometric mean
    # of that q-error, per operator signature, multiplies the next structural estimate —
    # so a join Kyber has consistently under-estimated 8x is next planned for at 8x. A
    # signature needs at least `min_samples` observations before its factor is trusted
    # (one anomalous run must not steer a plan), and every factor is clamped to
    # `[1/max_factor, max_factor]` so a pathological measurement cannot produce a
    # degenerate estimate. Set `max_factor <= 1.0` to disable the loop entirely.
    cardinality_correction_min_samples: int = 2
    cardinality_correction_max_factor: float = 32.0
    # Only the most recent `window` observations of a signature are averaged. The
    # structural estimator sharpens as the column-stat loop learns NDVs/quantiles, and
    # data drifts, so an all-history mean would keep applying a correction the estimator
    # has already outgrown. Set to 0 to disable the loop entirely.
    cardinality_correction_window: int = 8
    # Cost-model calibration from measured op_stats: a kind needs at least this many
    # samples before its coefficient is calibrated (else the default constant stands),
    # and each calibrated coefficient is clamped to within this factor of its default
    # so timing noise can never produce a degenerate cost model.
    cost_calibration_min_samples: int = 20
    cost_calibration_clamp: float = 10.0
    # Quantile grid Core collects for histogram-based selectivity.
    quantile_probs: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    # Ceiling on the `rows x columns` an in-memory source may be sketched for cold-start
    # distinct counts before the optimizer runs (`api.terminal._metadata.seed_column_ndv`).
    # Only the estimator's join/group/equality columns are sketched, so the cell count is
    # rows x a handful of columns, not the whole relation. HLL runs at ~0.4 ns/cell across
    # cores, so this default admits sf100 `lineitem`'s three join keys (~1.8G cells, ~0.7 s
    # once per source) — a cost the plan it fixes repays immediately, since the blind plan
    # peaks at 23 GB on TPC-H Q8 at sf10 alone. A source past the ceiling keeps learning
    # its ndv from the post-run pass. Result-invariant either way: ndv only steers choice.
    ndv_sketch_max_cells: int = 1 << 31
    # How many optimized plans to memoize (`kyber.plan_cache`); 0 disables the cache.
    # Optimization is a pure function of the plan, its sources, this config, and the learned
    # statistics, so re-issuing an identical query need not re-derive an identical plan —
    # and on a join-heavy query that derivation costs more than the engine's execution.
    # Bounded LRU: a cached entry pins its in-memory sources alive, so the cap also bounds
    # what the cache can keep from being collected.
    plan_cache_entries: int = 256
    cardinality: CardinalityConfig = CardinalityConfig()
    cost_coeffs: CostCoefficients = CostCoefficients()
    cost_weights: CostWeights = CostWeights()

    def resolved_broadcast_max_bytes(self, l3_cache_bytes: int = 0, workers: int = 1) -> int:
        """The build-side broadcast threshold in bytes, sized to the cache or the cluster.

        On **one node** (`workers <= 1`) this is a cache question. A broadcast join builds one
        hash table and probes it from every core, so it wins only while that table stays
        L3-resident; past that the partitioned (shuffle) join wins. The right threshold is
        therefore a share of the L3 the probing cores share — not a fixed byte count, which is
        wrong by the cache ratio across the fleet (≈1 MiB on a small ARM
        core to 32+ MiB per CCX on an EPYC). Given the detected `l3_cache_bytes`, this returns
        `_BROADCAST_L3_FRACTION` of it.

        Across a **cluster** (`workers > 1`) it is a network question instead, and the answer
        is a different order of magnitude. Replicating `B` bytes to `W` workers costs `B x W`
        on the wire; co-partitioning costs the probe side plus the build side, and the probe
        side is the large one. A dimension table ten times too big for L3 still beats shuffling
        a fact table fifty times its size, so applying the cache figure to a cluster declines
        broadcasts that are overwhelmingly worth taking — a 4 MiB threshold that is really a
        per-core cache share, against Spark's 10 MB `autoBroadcastJoinThreshold`, on the
        transport (`resolve_transport` picks Flight for every genuine multi-node cluster) where
        the shuffle it is avoiding is a network round-trip rather than a cache miss.

        The distributed ceiling is what a worker must *hold*, not what its cache likes, so it
        is a fixed multiple of the cache figure (`_BROADCAST_DISTRIBUTED_FACTOR`) floored at
        `_BROADCAST_DISTRIBUTED_FLOOR` — the floor so a small-cache host does not decline every
        broadcast a large-cache host takes. It stays an estimate either way: the executor
        re-checks the *measured* build side against this same number before replicating it, so
        a planner under-estimate costs a fallback to the shuffle rather than a cluster-wide OOM.

        A pinned `broadcast_max_bytes` (any positive value) always wins, at any worker count.
        When it is `0` (auto) and the cache is unknown (`l3_cache_bytes <= 0`, e.g. a non-Linux
        host), it falls back to `_BROADCAST_FALLBACK_BYTES` — the historical 4 MiB — so behavior
        only ever *improves* where the cache is readable and is unchanged where it is not.

        Examples:
            .. doctest::

                >>> from batcher.config import OptimizerConfig
                >>> OptimizerConfig().resolved_broadcast_max_bytes(16 * 1024 * 1024)
                4194304

                >>> OptimizerConfig().resolved_broadcast_max_bytes(16 * 1024 * 1024, workers=8)
                67108864

                >>> OptimizerConfig(broadcast_max_bytes=10 << 20).resolved_broadcast_max_bytes(0)
                10485760

        Args:
            l3_cache_bytes: Detected last-level cache size; `0` when undetectable.
            workers: The worker fan-out the plan targets; `1` (the default) is single-node.

        Returns:
            The broadcast-eligibility threshold in bytes.
        """
        if self.broadcast_max_bytes > 0:
            return self.broadcast_max_bytes
        cache = (
            _BROADCAST_FALLBACK_BYTES
            if l3_cache_bytes <= 0
            else max(1, int(l3_cache_bytes * _BROADCAST_L3_FRACTION))
        )
        if workers <= 1:
            return cache
        return max(_BROADCAST_DISTRIBUTED_FLOOR, cache * _BROADCAST_DISTRIBUTED_FACTOR)


#: Share of the last-level cache a broadcast hash table may occupy before the partitioned join
#: wins. Chosen so a 16 MiB L3 — the machine the 4 MiB default was tuned on — resolves to that
#: same 4 MiB, making the switch to detection a no-op there and an adaptation everywhere else.
#: A policy ratio, not a hardware fact, so it stays a named constant rather than being detected.
_BROADCAST_L3_FRACTION = 0.25
#: Broadcast threshold when the cache cannot be read (non-Linux) and none was pinned — the
#: historical default, so an unreadable machine behaves exactly as before.
_BROADCAST_FALLBACK_BYTES = 4 * 1024 * 1024
#: How much larger a distributed broadcast budget is than the single-node cache one. The
#: replicated side is held once per worker, so this is bounded by a worker's memory envelope
#: rather than its cache: 16x a quarter-L3 is 64 MiB on a 16 MiB-L3 host, comfortably inside
#: that envelope and comfortably above the dimension tables the widening exists to admit.
_BROADCAST_DISTRIBUTED_FACTOR = 16
#: Floor on the distributed budget, so a host whose cache the probe cannot read — or one with
#: a genuinely small cache — still broadcasts an ordinary dimension table.
_BROADCAST_DISTRIBUTED_FLOOR = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class PIDConfig:
    """Gains for the adaptive batch-size PID controller over batch-latency error.

    The loop grows/shrinks the per-batch row count toward a target latency. It is
    implemented identically in `bc-udf::BatchSizeController` (data plane) and
    `ml.inference._LatencyController` (Python); shipped to Rust as `EngineConfig` so
    the two never drift.

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
class TenantConfig:
    """Which tenant this scope's work belongs to, and what it may consume.

    Batcher is a library inside one process, so a tenant here is **a cooperating
    workload, not an adversary**. Two tenants in one process share an address space; one
    can read the other's memory directly and no Python-level control changes that. What
    this section does is stop them sharing by *accident* — through the process-global
    caches, pools, and learned-statistics store they would otherwise both land in — and
    bound what each may consume.

    That distinction is the whole design. Use it to keep a nightly ETL from evicting an
    interactive team's cached results, or to keep one team's learned column statistics out
    of another's optimizer. Do not use it to isolate mutually untrusting parties: run one
    process per trust domain, as {doc}`../user-guide/hardening` says.

    Examples:
        .. doctest::

            >>> from batcher.config import TenantConfig
            >>> TenantConfig().tenant_id
            ''
    """

    #: Names the tenant. Empty (the default) means "no tenancy" — every process-global
    #: structure behaves exactly as it did before this existed, which is what keeps this
    #: from changing anything for a single-workload deployment.
    tenant_id: str = ""
    #: Share of the process result-cache budget this tenant may hold, 0.0-1.0. 0 means
    #: unbounded (the historical behavior).
    cache_share: float = 0.0
    #: Maximum queries this tenant may run concurrently. 0 means unbounded.
    max_concurrent_queries: int = 0


@dataclass(frozen=True, slots=True)
class GovernanceConfig:
    """Whether row/column policy is advisory or mandatory, and where denials are recorded.

    Governance is enforced as a plan rewrite inside a ``bt.security(...)`` block. That is
    the right default for a library — a `Dataset` built outside one is exactly what it was
    before a catalog existed — and the wrong one for a deployment, where "the developer
    forgot the `with` block" must not be the difference between a masked column and a
    plain one. This section is that switch.

    ``mode`` is the one that matters:

    - ``"off"`` (default) leaves every read exactly as it is today.
    - ``"advisory"`` logs a warning for each read that a strict deployment would refuse,
      and proceeds. This exists so an operator can find every ungoverned read in a real
      workload *before* flipping the switch; without it, strict mode is unadoptable and
      therefore shelfware.
    - ``"strict"`` refuses a read that no `security()` block covers, and refuses a source
      that cannot be governed at all — an in-memory table or a live stream has no durable
      name to write a policy about, so it is rejected rather than silently exempted.

    Examples:
        .. doctest::

            >>> from batcher.config import GovernanceConfig
            >>> GovernanceConfig().mode
            'off'
    """

    #: ``off`` | ``advisory`` | ``strict`` — see the class docstring.
    mode: str = "off"
    #: Deny a table no grant mentions, instead of leaving it ungoverned. Turns the catalog
    #: from a list of restrictions into a list of permissions.
    default_deny: bool = False
    #: Append every governance decision to this JSONL file. An audit trail a caller can
    #: switch off by not passing a sink is not an audit trail.
    audit_path: str | None = None
    #: Refuse a `security()` block whose principal was *asserted* rather than established
    #: by a `CredentialVerifier` (`bt.authenticate`). Closes the "anyone can claim to be
    #: admin" gap for code paths a deployment controls; it cannot close it against code
    #: running inside the engine's own process, which is why the trust boundary stays the
    #: process. See `governance.authn`.
    require_verified_principal: bool = False


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
class ShuffleTlsConfig:
    """TLS for the inter-node Arrow Flight shuffle — encrypt the wire, authenticate peers.

    The shuffle moves query data (including already-decrypted or masked columns) directly
    between worker processes. On any network the operator does not fully control, that
    traffic must be encrypted and the peers mutually authenticated. These are **file
    paths** to PEM material the platform mounts on every worker (a Kubernetes secret
    volume, cert-manager, a cloud private-CA); Batcher reads them at worker start and
    issues no certificates itself — minting and rotating them is a platform concern.

    Off by default (`enabled=False`) for a plaintext shuffle on a trusted network. One
    cluster CA typically signs both the server and the client certificates, so
    `ca_cert_path` is the trust root for both directions; set `require_client_auth` to
    turn on mTLS (a connecting peer must present a certificate that CA signed).
    Overridable via ``BATCHER_DISTRIBUTED_TLS_<FIELD>`` env vars.

    Examples:
        .. doctest::

            >>> from batcher.config import ShuffleTlsConfig
            >>> ShuffleTlsConfig().enabled  # plaintext shuffle by default
            False
            >>> tls = ShuffleTlsConfig(
            ...     enabled=True,
            ...     ca_cert_path="/etc/batcher/ca.pem",
            ...     server_cert_path="/etc/batcher/server.pem",
            ...     server_key_path="/etc/batcher/server.key",
            ...     require_client_auth=True,
            ... )
            >>> tls.server_name  # the SAN a peer's certificate must carry
            'batcher-shuffle'
    """

    enabled: bool = False
    # The CA a peer's certificate must chain to (trust root for both server and client).
    ca_cert_path: str = ""
    # This node's server certificate + private key (what it presents on its Flight port).
    server_cert_path: str = ""
    server_key_path: str = ""
    # This node's client certificate + key, presented under mTLS when fetching from a
    # peer. Empty → outbound connections are server-auth only.
    client_cert_path: str = ""
    client_key_path: str = ""
    # Server side: require and verify a client certificate on every incoming fetch (mTLS).
    require_client_auth: bool = False
    # The name verified against a peer certificate's SAN. Peers are dialed by address, so
    # the certificate rarely matches the literal host; set this to the name the cluster's
    # certificates actually carry.
    server_name: str = "batcher-shuffle"


@dataclass(frozen=True, slots=True)
class DistributedConfig:
    """How the engine attaches to and shuffles across a Ray cluster.

    Ray is scheduling only; the data plane shuffles via Carbonite/Arrow Flight or
    (single-node / shared filesystem) Arrow-IPC files. These knobs decide which.

    Examples:
        .. doctest::

            >>> from batcher.config import DistributedConfig
            >>> DistributedConfig().ray_address is None  # attach locally, or $RAY_ADDRESS
            True
            >>> DistributedConfig().namespace  # the shuffle actors are isolated here
            'batcher'
    """

    # Ray cluster address. None → attach to an existing cluster when ``RAY_ADDRESS``
    # is set in the environment, else start a local one. Set explicitly (e.g.
    # ``"ray://head:10001"`` or ``"auto"``) to force attaching to a running cluster.
    ray_address: str | None = None
    # Ray namespace for batcher's shuffle actors, so they're isolatable.
    namespace: str = "batcher"
    # TLS/mTLS for the inter-node shuffle wire (off by default). See `ShuffleTlsConfig`.
    tls: ShuffleTlsConfig = ShuffleTlsConfig()
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
    # barrier. One slow survivor gets a backup copy and the barrier takes whichever
    # finishes first; shuffle tasks are deterministic so the result is identical.
    # Bounded so speculation never oversubscribes the cluster. 0 disables it entirely
    # (the barrier becomes a plain `ray.get`).
    #
    # **1 by default.** A barrier is only as fast as its slowest task, so one straggler
    # — a hot partition, a throttled disk, a noisy neighbour — stalls the whole stage.
    # Ray Data has no straggler mitigation of any kind (no speculative execution, no
    # task re-launch); its guides can only tell users to detect skew by hand and
    # re-partition. Spark ships speculation, and this is Batcher's equivalent.
    #
    # The default is 1, not higher, because the cost of a wrong guess is a duplicated
    # task: one backup catches the single worst straggler, which is the bulk of the win,
    # without letting a uniformly-slow stage spawn a backup per task. Two gates keep it
    # from firing spuriously — a task must exceed `speculation_straggler_factor` × the
    # median finished time, AND `speculation_min_finished_frac` of tasks must already be
    # done, so nothing is backed up before there is a meaningful median to compare to.
    # The factor is additionally *learned* per operator family from measured task-time
    # variance (`_learned_straggler_factor`), so a stage that finishes uniformly raises
    # its own bar and effectively opts out.
    speculation_max_backups: int = 1
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
    # Shuffle-output replication factor: how many workers hold each mapper's published
    # buckets. 1 (default) = today's single copy, so a lost worker's output must be
    # *recomputed* — re-reading its source partition from object storage and re-running
    # the map, which is usually the longest part of the query. 2+ replicates each bucket
    # onto peers on *other nodes*, so a reducer whose mapper is gone transparently fetches
    # a byte-identical copy from a survivor: worker loss costs one re-fetch instead of a
    # full recompute round, and the recompute loop stays as the backstop for when every
    # copy is gone.
    #
    # This is affordable precisely because of the mergeable algebra: what a mapper
    # publishes is *pre-aggregated partial state*, typically far smaller than the source
    # that produced it, so copying it is much cheaper than regenerating it. (Spark cannot
    # make the same trade — its shuffle carries raw rows, so it recomputes the map stage
    # when a node takes its shuffle files with it.)
    #
    # The cost is one extra network copy of the shuffle output per additional replica, so
    # it stays at 1 for a stable on-demand cluster and rises to 2 under the `spot`
    # resilience profile, where preemption is expected rather than exceptional.
    #
    # **Scope: every Flight shuffle** — aggregate (both the flat reduce and the combiner
    # tree a wide shuffle takes), join, sort, and window. The driver placement lives in
    # `dist/shuffle_replication.py::replicate_shuffle_output`, which assigns off-node hosts
    # (`carbonite/resilience/replication.py::assign_replica_hosts`), calls
    # `replicate_buckets`, and passes the acked addresses into the reducers' `replicas=`.
    # A join publishes two shuffle stages (left on 0, right on 1) under one address, so its
    # replica is all-or-nothing across both — a copy holding one side would under-join.
    # One residue: inside the combiner tree only the *leaf* partials carry replicas. An
    # interior combiner's output lives on a single node and is not copied, so losing a
    # worker mid-tree still costs a recompute round.
    #
    # A replica is advertised only once its `replicate_buckets` call has acked, and a
    # source's replicas are retired when it is recomputed: a replica holds the *old*
    # epoch's ticket, and an unregistered ticket reads back as an EMPTY bucket rather than
    # an error, so a stale fallback would silently drop that mapper's rows rather than
    # failing. That invariant is what makes the fallback safe; do not relax it.
    shuffle_replication: int = 1
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
    # Advertise each worker's *fabric* address for the shuffle instead of the address Ray
    # knows it by. Off by default, and worth turning on for exactly one shape of cluster: a
    # GPU node whose Ray IP is its management Ethernet while its InfiniBand ports — two orders
    # of magnitude faster — carry nothing, because nothing addresses them. With it on, each
    # worker resolves its own active fabric interface's IPv4 address and advertises that, which
    # is the same fix `BATCHER_ADVERTISE_HOST` performs by hand except that it needs setting
    # once for the cluster rather than once per node.
    #
    # Off by default because it can only be verified per deployment: the fabric address has to
    # be routable *between workers*, and a fleet where some nodes have IPoIB configured and
    # others do not would advertise addresses half its peers cannot dial. A worker that finds
    # no fabric address keeps its Ray address, so a partially-configured fleet degrades one
    # node at a time rather than failing the shuffle; `BATCHER_ADVERTISE_HOST` still wins over
    # both, because a node that names its own address has already settled the question.
    prefer_fabric_interface: bool = False
    # Ray-level task/actor fault tolerance — the *first* line of defense, beneath the
    # shuffle recompute loop above. A transient task failure (a flaky node, a dropped
    # connection) is retried by Ray itself before the heavier app-level recompute
    # engages. `task_max_retries` reruns a failed shuffle task (deterministic +
    # recomputed from a durable source, so a rerun is safe); `retry_on_transient`
    # extends those retries to application exceptions (not just worker death), gated to
    # transport-classified transient errors once that classification lands.
    # `actor_max_restarts` lets a crashed compute actor (the map/inference pool)
    # respawn, and `actor_max_task_retries` reruns its in-flight call on the respawned
    # actor. These do not touch the Flight shuffle-server actors, whose loss is handled by
    # the recompute loop.
    #
    # **`task_max_retries` is not a count of preemption retries, and `0` is not "one fewer
    # retry".** Ray's rule is a step function, not a dial: with a *non-zero* value, system
    # errors — spot preemption, worker crash, node loss — are retried **indefinitely** and do
    # not decrement the count, which bounds only *application* errors (and only when
    # `retry_on_transient` turns `retry_exceptions` on). With **`0`, the task is not retryable
    # at all** and dies permanently on the first preemption. So `2` here means "unlimited
    # preemption retries, two application retries", and the one edit that silently removes
    # every spot protection on this path is setting it to zero. Ray's own default is 3; the
    # `spot` resilience profile raises this to 4 alongside `actor_max_restarts`, which is what
    # the field guidance asks for (`../optimization-guides`,
    # `foundations/core/scheduling-and-resources.md`, "Spot Instance Preemption and Task Retry
    # Semantics").
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
    # Wire compression for shuffle batches: "none", "lz4", "zstd", or "auto". A cross-node
    # fetch is NIC-bound, so compressing the Arrow buffers before they cross the wire is the
    # only way past line rate — and real shuffle data (sorted runs, repeated group keys,
    # dictionary strings, nulls) compresses several-fold, unlike the object store's
    # uncompressed blocks. "lz4" (~GB/s/core, gives up fast on incompressible data) is a
    # near-free default; "zstd" trades CPU for a higher ratio; "none" disables it. "auto"
    # decides against the node's measured fabric: past roughly 25 Gb/s a compressor becomes
    # the ceiling rather than the wire, and on a 400 Gb/s port compressing costs throughput
    # rather than buying it (`carbonite.transfer.codec`).
    flight_compression: str = "lz4"
    placement_timeout_s: float = 60.0
    # Bounded retry window for *attaching* to a Ray cluster whose head is not answering yet.
    #
    # The driver and the head come up concurrently in every orchestrated environment: a
    # KubeRay driver pod is admitted before the head pod passes its readiness probe, and a
    # Slurm job's `ray start --head` on the first node races the step that runs the query.
    # A single failed attach there is not "there is no cluster", it is "not yet" — and
    # treating the two alike is expensive, because the fallback is to start a *local*
    # single-node Ray and run what the user asked to distribute on one machine, silently.
    #
    # Retried with exponential backoff for this many seconds before giving up. Set to 0 to
    # restore the old single-attempt behavior (a local dev run inside a workspace whose
    # cluster is deliberately down falls back immediately rather than pausing here).
    #
    # Exhausting the window is only a *fallback* when the address was merely detected. An
    # address the user configured explicitly (`ray_address`, or `RAY_ADDRESS`) raises
    # instead: they named a cluster, and running single-node in its place is a wrong answer
    # to a question they asked precisely.
    cluster_connect_timeout_s: float = 30.0
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
    # Grace window for the FIRST sign of growth. Distinct from `autoscale_stall_s`, which
    # governs a stall *after* the cluster has started growing: this bounds how long to wait
    # before concluding the autoscaler is not going to act at all. A request the autoscaler
    # cannot satisfy — a fixed cluster already at max, or infeasible spot capacity — produces
    # zero growth from the start, and the query is already runnable on the capacity it has
    # (the fan-out clamps to it), so the extra nodes are an optimization, not a prerequisite.
    # Waiting the full `autoscale_stall_s` for them taxed EVERY cold query that asked for more
    # cores than the cluster has (a large aggregate's data-parallel fan-out routinely exceeds
    # the node count) — 90 s of dead time before a 9 s query. Kept above a couple of poll
    # intervals so a cluster whose nodes register as pending within a few seconds still gets
    # its full `autoscale_stall_s`; the moment any growth appears, that longer window governs.
    autoscale_startup_grace_s: float = 12.0
    # Skew-aware join salting for a huge x huge hot key. When a single join key is
    # dominated by a few "hot" values, those rows otherwise co-partition onto one
    # reducer and overload it (memory + the output explosion + a straggler). Salting
    # spreads each hot key's probe rows across several reducers while replicating its
    # build rows to all of them, so the hot key's work fans across the cluster instead
    # of landing on one node. Single-key, left-driven (inner/left/semi/anti) joins whose
    # reducer does not finalize a fused aggregate; every other shape falls back to the
    # plain co-partition (`dist/skew.py::salting_preserves_result` is the guard, and it
    # is a correctness guard, not a preference — see its docstring).
    #
    # **This is the fan-out, not an on/off switch, and 0 does not mean off.** Salting is
    # result-preserving, so it engages on *measured* skew whatever this says:
    # `dist/skew.py::resolve_hot_keys` takes the hot values from the set learned for this
    # join shape on a previous run, else from the column statistics Kyber already holds,
    # else from a Misra-Gries pre-pass — which it runs on its own above
    # `_DETECT_MIN_INPUT_ROWS`, because past that size one pass costs ~4% on a join that
    # turns out uniform against 5.8x for an undetected 40% hot key. The fan-out is then
    # sized from the key's measured share (`salt_factor`).
    #
    # So the values are: a positive number FORCES the pre-pass and pins the fan-out to it
    # (for a join you already know is skewed and do not want to pay discovery on); `0`
    # (default) leaves both to the measurement; and a NEGATIVE number is the actual off
    # switch, for pinning the plain co-partition shuffle.
    skew_join_salt: int = 0
    # A value is "hot" when it exceeds this fraction of a side's rows. Lower → more
    # keys salted. Also the conservative floor for sizing the fan-out when the hottest
    # value's true share was never measured.
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
    # Shared-secret token authenticating Flight shuffle fetches. When set, a
    # peer must present it to fetch a partition, so a process that can merely reach
    # the port cannot exfiltrate shuffle data. None (default) disables the check —
    # appropriate on a trusted/isolated cluster network. Also read from the
    # `BATCHER_SHUFFLE_TOKEN` env var so it can be injected without a config file.
    shuffle_token: str | None = None
    # Closed port range the Flight shuffle listener may bind, as (min, max) inclusive.
    # None (default) takes an OS-ephemeral port, which never collides and needs no
    # configuration — the right default on a flat cluster network. A firewalled network
    # (most on-prem, and locked-down cloud VPCs) cannot open the whole ephemeral range
    # node-to-node, so setting a range lets the operator open exactly it. Make it wide
    # enough for every worker that shares a node: the bind takes the first free port and
    # fails with a message naming the range if none is left, rather than falling back to a
    # port nothing can reach. Also read from `BATCHER_SHUFFLE_PORT_RANGE` ("40000-40100")
    # so a deployment can inject it without a config file.
    shuffle_port_range: tuple[int, int] | None = None
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
    # safe. Pays off on a multi-node cluster with a skewed/co-partitioned shuffle; a
    # no-op for an evenly spread bucket (no node dominates).
    #
    # On by default. It was off because deciding placement meant a `node_id` round-trip
    # per worker, charged to every shuffle — including the single-node case, where the
    # answer is always "nothing to place". Node identity now comes from the workers'
    # advertised shuffle addresses, which the driver already holds, so a single-node
    # fleet resolves to "no placement" with no remote call at all and a multi-node one
    # pays a single fan-out for the per-mapper byte sizes.
    locality_aware_scheduling: bool = True
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
    # is split at EVERY resource-class boundary into per-stage actor pools that stream
    # partitions stage→stage over Arrow Flight, so the stages OVERLAP (a model runs
    # partition k while the stage below prepares k+1) instead of one actor running the
    # whole chain per partition. A chain with two models gets a pool each rather than
    # both in one actor sharing a device, and a postprocess after inference gets its own
    # CPU pool rather than spending device time on host work. **On by default**: the
    # non-overlapped path leaves the GPU
    # idle for the whole CPU decode/preprocess of every partition — the single largest
    # avoidable cost in a batch-inference pipeline, and the pathology the Ray guides spend
    # their GPU-utilization chapter telling users to hand-tune around. Overlapping it is a
    # pure scheduling win, so it should not need opting into.
    # Result is unchanged either way — only the scheduling overlaps: every stage runs the
    # identical sub-plan through `core.execute_with_udfs`, which is what
    # `test_stream_inference.py::test_streaming_pipeline_equals_single_node` (and the
    # three-stage and non-overlapped-map variants) pin. A chain with no resource-class
    # boundary to split at (`split_into_resource_stages` → None) falls back to the
    # embarrassingly-parallel map, so a homogeneous pipeline is unaffected.
    # Set False to pin the old non-overlapped scheduling.
    stream_inference: bool = True
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
    # Re-run every GPU result on the CPU engine and report where they disagree
    # (`api/terminal/gpu_backend/verify.py`). Off by default: it doubles the work and walks the
    # result in Python, so it is a diagnostic, not a production setting.
    #
    # It exists because the device tier is the one tier that cannot share the Rust `Expr` —
    # cuDF has no maintained Rust binding, so it is a *translator*, and translators drift. Its
    # test suite runs on pandas rather than cuDF (CI has no GPU) and is therefore structurally
    # unable to see a device-only divergence; two have already shipped, both column-type bugs
    # with correct values. Until there is a GPU CI lane this is the only thing that observes
    # the device. Turn it on for benchmark and staging runs, and for any change to
    # `core/gpu_plan/`.
    gpu_shadow_verify: bool = False
    # The most devices a single query may ask the autoscaler to grow to. The request is sized
    # from the working set (how many devices would hold it in one wave), so a badly-estimated
    # query would otherwise be able to ask a cluster to grow without bound. Reaching the cap is
    # not a failure: the query simply runs in more waves on fewer devices.
    gpu_max_autoscale_devices: int = 64
    # A shard that did not FIT the device is divided into this many pieces and rerun on the
    # device, for up to `rounds` further halvings. The shard count is fixed before the query
    # runs, from an estimate, and estimates are wrong exactly where it matters — a skewed key, a
    # wider row than the footer suggested, a neighbouring tenant on the device — so one shard
    # can miss while every other one fits. Subdividing is exact, because the stage is mergeable:
    # a shard's partial and the concatenation of its pieces' partials are the same value. `1`
    # disables it, sending an over-large shard straight to the host.
    gpu_shard_subdivide: int = 4
    gpu_shard_subdivide_rounds: int = 3
    # Let several shards of one fan-out share a device, by requesting a FRACTION of a GPU per
    # shard instead of a whole one. `gpu_shard_oversubscribe` already cuts four times as many
    # shards as there are devices, precisely so each is small — and then every shard asked for a
    # whole device, so Ray ran one per device and queued the rest. A fleet whose own shard count
    # says each piece is a quarter of a device was running at a quarter of its capacity while
    # every utilization counter read full. The share is derived from the largest shard's
    # estimated working set against one device's memory and rounded UP to a packing quantum
    # (`_internal.device_share`), so a shard that needs a whole device still gets one. Over-
    # packing degrades rather than fails: a shard that does not fit its share is caught by the
    # subdivision ladder above and rerun in pieces, exactly as an under-estimated shard always
    # was. Result-identical either way — the mergeable combine is correct for any placement.
    # Off → the previous one-whole-device-per-shard behavior.
    gpu_pack_shards: bool = True
    # Pin the per-shard device share instead of deriving it. `0.0` (the default) derives it.
    # A positive value is an operator statement about a fleet the estimator cannot see — a
    # device shared with a long-running service, a part whose memory the driver misreports —
    # and is applied as given. Use `1.0` to force whole devices for one job without turning
    # `gpu_pack_shards` off fleet-wide.
    gpu_task_fraction: float = 0.0
    # Ceiling on shards resident on one device, which floors the derived share. The memory
    # arithmetic can allow more tenants than a device should actually run: each is a CUDA
    # context, a share of one copy engine, and a process whose allocation spike the others
    # feel. A deployment property rather than a derivable one, so it is a knob.
    gpu_max_tasks_per_device: int = 4
    # Multiple of a shard's INPUT bytes the device must hold for it. At the moment a partial
    # aggregate emits its last group, both the input batch it is reading and the hash table it
    # has built are resident; a factor below 2 asserts one of the two is free, which is not
    # true of any operator this path runs. Raise it for a chain that materializes more than one
    # intermediate (a join followed by a wide projection); the subdivision ladder is what
    # catches the cases it still under-states.
    gpu_shard_expansion: float = 2.0
    # How many shard partials the driver folds together at once. The fan-out bounds *device*
    # memory by dividing the input, and then concatenated every shard's output on the driver
    # before combining a single row: a group-by over a million groups fanned across a thousand
    # shards materialized a billion rows in one process — the same failure the sharding exists
    # to prevent, moved to the host, and arriving precisely on the large multi-GPU clusters the
    # fan-out is for, because the shard count grows with the fleet. Folding in waves keeps one
    # accumulator and drops each wave, so peak driver memory tracks the wave size and the group
    # count rather than the shard count. Exact at any wave size: the fold is associative and
    # commutative over its own output (`plan.distribution.recombine`). `0` or `1` folds
    # everything at once, which is the previous behavior; a fan-out with fewer partials than
    # one wave is unchanged either way.
    gpu_merge_wave: int = 32
    # Fraction of one device's memory the REPLICATED leaves of a multi-way plan tree may
    # occupy. A tree fan-out splits one leaf and gives every worker the whole of the others, so
    # those others are resident on every device simultaneously, beside that device's own shard
    # and whatever the joins above them build. This is the bound that turns a star-schema query
    # (small dimensions, huge fact table) into a fan-out and a big-to-big join into a declined
    # one — and declining is the point: the alternative is an out-of-memory on every device at
    # once, which is the single failure mode a GPU query has no way to recover from cheaply.
    # Raise it on a fleet whose devices are large relative to the dimensions; lower it for a
    # plan that materializes wide intermediates above its joins.
    gpu_tree_broadcast_fraction: float = 0.35
    # How long a GPU fan-out waits for a device to actually be free before giving the query to
    # the CPU engine. A GPU task asks Ray for a device *and* a core, and on a busy cluster the
    # core is the one it does not get: a placement group holding every CPU (a shuffle, another
    # tenant's stage) leaves the fan-out's tasks PENDING with devices sitting idle, and
    # `ray.get` on a pending task waits forever. That is the one failure a GPU backend
    # documented as "always safe to request" must not have — the answer was available on the
    # host the whole time. Measured here on a 16-device fleet: every GPU query blocked
    # indefinitely behind a leaked placement group until the driver was killed.
    # `0` restores the unbounded wait.
    gpu_admission_wait_s: float = 30.0
    # Smallest shard a GPU fan-out will cut. `gpu_shard_oversubscribe` says how many shards a
    # device *may* pipeline; this says when cutting another one stops paying for itself.
    #
    # Shard count used to be `#devices x oversubscribe` regardless of how much data there was,
    # so a 6M-row scan on a 16-device fleet was cut into 64 shards — each a Ray task, each
    # paying a worker dispatch, a cuDF first touch and a device-allocator setup to process
    # about a hundred thousand rows. Measured here: TPC-H q6 at sf1 took **196 seconds** that
    # way against 0.12 s on the CPU engine, and essentially all of it was per-task fixed cost.
    # Sizing from bytes instead keeps the fan-out wide where the data is wide and collapses it
    # to a single shard where it is not.
    #
    # 128 MiB is roughly where one T4 shard's compute (a few tens of milliseconds) overtakes
    # the dispatch that delivered it. Raise it on a fleet with slower task dispatch; lower it
    # where per-shard work is unusually expensive per byte.
    gpu_min_shard_bytes: int = 128 << 20
    # Let a GPU worker process serve more than one shard (`max_calls=0`).
    #
    # Ray tears a worker down after every GPU task by default, to guarantee the device memory
    # is released. For this path that guarantee is bought at a price nothing else pays: each
    # shard starts a new Python process, imports cuDF and rebuilds the RMM pool before it can
    # touch a row. Measured on a T4 against one 7.3M-row shard of TPC-H `lineitem` — 1.07 s to
    # import cuDF, 0.98 s to configure the allocator, 0.26 s to read the shard onto the device,
    # and 0.15 s to run the kernels. Two seconds of set-up per sixth of a second of work, on
    # every shard of every query.
    #
    # Reuse is safe here because the leak the default guards against is not one this path has:
    # a shard builds cuDF frames and drops them, and the async allocator returns freed blocks
    # to the driver. Set False for a fleet running a UDF that does hold device memory, and get
    # process-per-task isolation back at that cost.
    gpu_worker_reuse: bool = True
    # A GPU shard that fails for any other reason — a lost worker, an untranslatable expression —
    # is recomputed by the native CPU engine on a CPU worker, instead of the whole query
    # abandoning the accelerated path. Both compute the same mergeable partial, so the combined
    # answer is identical either way; only that shard is slower. Off → a lost shard fails the GPU
    # attempt and the caller re-runs everything on the host, which is the older, coarser behavior.
    gpu_shard_cpu_fallback: bool = True
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
    # `0.0` (the default) means **detect it** — see `resolved_gpu_memory_gb`. A positive value
    # pins the budget, for a device the probe cannot see (a remote GPU worker sized from a
    # CPU-only driver) or to deliberately under-commit a shared device.
    gpu_memory_gb: float = 0.0
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
    # `n_reducers = min(workers x shuffle_partition_multiplier, this)`, so very large
    # clusters stay bounded. Mergeable algebra makes any reducer count correct
    # (result-identical), so this is purely a scaling knob. 0 disables the cap.
    max_shuffle_partitions: int = 2048
    # CEILING on shuffle buckets per worker, as a multiple of the worker count — not a
    # target. One bucket per worker is the floor and the cold-start shape; this bounds how
    # far a *measured* shuffle volume may raise it above that. What buckets above the floor
    # buy is lower per-reducer memory and finer work units.
    #
    # They do **not** buy skew tolerance, and it is worth being precise about that because
    # the Spark-style "many partitions per executor" argument is usually stated as though
    # they do. A hash bucket is the unit a key cannot be split below, so a single dominant
    # key is indivisible however many buckets there are: measured on 12.5M rows with 40% of
    # them on one key, max/mean bucket load is 3.8 at 8 buckets and **51.8 at 128** — the
    # hot bucket does not shrink, only the mean does. Splitting one key's work needs salting
    # (`dist/skew.py`), not a finer hash. Buckets do flatten a *wide* hot band, but hashing
    # already flattens that: the same measurement over 50k moderately hot keys is 1.01.
    #
    # They are not free either: the exchange opens `mappers x reducers` streams, so every
    # extra multiple is another round of Flight fetches per mapper. 1 pins the count at one
    # bucket per worker.
    shuffle_partition_multiplier: int = 4
    # CEILING on map partitions per worker — the shuffle's **task granularity**, as a
    # multiple of the worker count. The reduce-side multiplier above sizes hash buckets;
    # this sizes the units the *input* is cut into, which is what decides how much work a
    # straggler holds and how much a preemption loses.
    #
    # At 1 the task unit is a node's share of the input, and both costs are charged whole: a
    # worker running at half speed still holds a full partition and the map barrier waits on
    # it, and a worker that dies loses a full partition that one survivor then replays end to
    # end. Above 1, `map_barrier` deals partitions out as actors go idle, so a slow worker
    # takes fewer of them and a dead worker's remaining partitions spread across every
    # survivor. This is the Spark "many tasks per executor" argument, and it is the half of
    # it that actually holds — the other half, that fine partitions dilute skew, does not:
    # skew lives in the hash buckets, not in the input split (see `shuffle_partition_multiplier`).
    #
    # A ceiling rather than a target. A source that cannot yield this many splits produces
    # fewer partitions instead of empty tasks, so a small input is unaffected, and an
    # in-memory (driver-resident) source stays at one partition per worker because shipping
    # it in more pieces buys no recovery — the driver still holds it. The cost above the
    # floor is that the exchange opens `mappers x reducers` streams and this multiplies the
    # first factor, so it is bounded by `max_shuffle_partitions` like the second. 1 pins the
    # old one-partition-per-worker unit.
    map_partition_multiplier: int = 4
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
    # Reserve a shuffle fleet's placement group inside ONE availability zone when the cluster
    # spans several and one of them can host the whole fleet.
    #
    # A shuffle moves nearly all of its bytes worker to worker, and every cloud prices and
    # delays those bytes by whether the two workers share a zone. Anyscale's field engineering
    # puts cross-AZ transfer above 40% of total AWS spend on distributed workloads and at
    # 20-40% added latency on synchronous ones, which makes this a first-order cost item rather
    # than a rounding one. A fleet spread evenly over three zones sends about two thirds of its
    # shuffle across that boundary, for nothing — the bundles are interchangeable, so a fleet
    # that fits in one zone can simply be placed in one.
    #
    # A single-AZ compute config (`STRICT_ZONAL_PACK` on Anyscale) is the better fix where the
    # operator controls provisioning: it removes the traffic instead of routing around it, and
    # covers the head node too. This is the runtime answer for the fleet that cannot do that —
    # an accelerator cluster whose instance types are scarce enough that cross-zone autoscaling
    # is the only way to get capacity, and which therefore spans zones regardless.
    #
    # On by default because it is placement-only (result-identical), it is applied to the
    # bundles rather than the tasks — so a group that cannot form is abandoned at the existing
    # timeout and the stage falls back to default scheduling, where a pin on the tasks would
    # leave them pending forever — and it is a no-op on every cluster that cannot benefit: one
    # zone, unlabelled nodes, or no single zone with room for the fleet. The zone chosen is the
    # one with the most free capacity that can host it (`capacity.preferred_fleet_zone`).
    #
    # Set False to let the fleet spread across zones, which is the right call when zone
    # diversity is being bought deliberately for availability rather than paid for by accident.
    zone_aware_placement: bool = True
    # Submit-ahead depth per GPU/inference actor — how many partitions an actor may have in
    # flight at once. `2` (default) double-buffers: one partition executes on the device while
    # the next is dispatched/gathered, so the GPU stays fed across that round-trip instead of
    # idling one-at-a-time (`1`) — the out-of-the-box lever toward >90% GPU utilization on the
    # *first* run, before any measurement. It is deliberately not deeper by default: with no
    # learned peak VRAM yet, a VRAM-heavy model at depth 8-16 could OOM; the adaptive loop
    # (`map_inflight_adaptive`) raises it further only once a run has *measured* low utilization
    # and headroom. Ray Data's `max_tasks_in_flight` guidance (depth 8-16 can multiply
    # utilization) is reached that way, safely. Bounded by `_MAP_INFLIGHT_MAX` in the executor.
    # Result-identical (assembly is index-addressed; only pipeline depth changes).
    map_inflight_depth: int = 2
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
    # How long before a known termination deadline a worker starts draining — enough time
    # to migrate its published shuffle output to a survivor before the kill lands. Only
    # consulted when a deadline is discoverable (`SLURM_JOB_END_TIME`, or an explicit
    # `BATCHER_DEADLINE_EPOCH_S` from any launcher that knows when its lease expires); a
    # cluster with no lease never reaches this. The default matches both AWS's ~2 minute
    # spot notice and the `--signal=B:USR1@120` an HPC job is conventionally submitted
    # with, so the deadline path and the signal path drain on the same budget. Too short
    # and the kill lands mid-migration, which costs the same recompute as not draining;
    # too long and the fleet stops taking work while it still had useful time. Draining
    # only moves *where* a partial lives, never what it holds, so this never changes a
    # result. See `carbonite/resilience/deadline.py`.
    drain_lead_s: float = 120.0

    def resolved_gpu_memory_gb(self) -> float:
        """The usable memory budget of one GPU, detected when `gpu_memory_gb` is `0.0`.

        Kyber routes on this: a working set that fits one device is dispatched to it, one that
        does not is sharded across the cluster, and a GPU inference stage seeds its batch size
        from the VRAM left after the model. All three are wrong by the ratio of the real device
        to the assumed one, and the assumed one used to be a hardcoded 12.0 — a T4. On an 80 GB
        A100 that shards a working set six times over that one device would have held, and
        seeds inference batches ~6x too small, which is exactly the "leaves the GPU idle"
        failure this is supposed to prevent.

        Reports the device's **total** memory in decimal GB. It is a *capacity*, and every
        caller subtracts `accelerator.vram_headroom` from it once, itself — which is the
        contract that had drifted. Detection used to fold in a private `0.75`, so a stage
        packed against it applied its own `0.85` on top and budgeted 64% of the board, while
        the same stage on a Ray cluster took `cluster_gpu_memory_gb()`, which folded in
        nothing, and budgeted 85%. One decision, two answers, chosen by whether Ray was up.

        The unit is decimal GB rather than GiB for the same single-meaning reason: Kyber sizes
        a working set as `rows x width / 1e9`, and dividing a device by `1 << 30` to compare
        against that over-stated an 80 GiB board by 7.4%.

        Falls back to a T4-shaped constant when no device is visible, which keeps a CPU-only
        driver planning for a remote GPU worker as it did before.

        Examples:
            .. doctest::

                >>> from batcher.config import DistributedConfig
                >>> DistributedConfig(gpu_memory_gb=40.0).resolved_gpu_memory_gb()
                40.0

        Returns:
            Total GPU memory in decimal GB for a single device.
        """
        if self.gpu_memory_gb > 0.0:
            return self.gpu_memory_gb
        from batcher._internal.hardware import gpu_inventory

        devices = [int(d.get("memory_bytes") or 0) for d in gpu_inventory()]
        smallest = min((b for b in devices if b > 0), default=0)
        if smallest <= 0:
            return _GPU_MEMORY_GB_FALLBACK
        return smallest / 1e9


#: Total GB assumed when no device is visible: a T4's nameplate, which is the device the
#: historical default described. Stated as a capacity like every other value this resolves to,
#: so the caller's own headroom subtraction lands on it once and the CPU-only driver planning
#: for a remote GPU worker keeps a budget within a gigabyte of the one it always had.
_GPU_MEMORY_GB_FALLBACK = 16.0


@dataclass(frozen=True, slots=True)
class VerbosityLevel:
    """One rung of the verbosity ladder: what it shows and what it costs."""

    name: str
    log_level: str
    progress: str
    #: The Rust `tracing` threshold, which can reach TRACE where Python's logging cannot.
    native_level: str
    summary: str


# The ladder, index == integer verbosity, so `verbosity=2` and `verbosity="normal"` are the
# same rung and a `-vv` style counter maps straight through. Ordered least → most output,
# which is the order every `-v` convention uses. `normal` is the default and reproduces the
# engine's historical behavior exactly (WARNING + auto progress), so adding this dial changed
# nothing for anyone who does not touch it.
VERBOSITY_LEVELS: tuple[VerbosityLevel, ...] = (
    VerbosityLevel("silent", "CRITICAL", "off", "off", "nothing but unrecoverable failures"),
    VerbosityLevel("quiet", "ERROR", "off", "error", "errors only; no progress bar"),
    VerbosityLevel("normal", "WARNING", "auto", "warn", "warnings + progress bar (default)"),
    VerbosityLevel("verbose", "INFO", "auto", "info", "+ optimizer and resource decisions"),
    VerbosityLevel("debug", "DEBUG", "auto", "debug", "+ per-phase timings and plan detail"),
    VerbosityLevel("trace", "DEBUG", "on", "trace", "+ Rust per-morsel spans; bar forced on"),
)
_VERBOSITY_BY_NAME = {level.name: i for i, level in enumerate(VERBOSITY_LEVELS)}
DEFAULT_VERBOSITY = "normal"


def _verbosity_rank(value: str | int) -> int:
    """The ladder index for a name or number, clamped into range; unknown names → default.

    Tolerant on purpose. This resolves a value that can arrive from an env var or a JSON
    config file, and an unreadable verbosity must not be the thing that stops a query from
    running — `validation` is where a bad value is *reported*, loudly and separately.
    """
    if isinstance(value, bool):  # `bool` is an `int`; treat True/False as unset, not 1/0.
        return _VERBOSITY_BY_NAME[DEFAULT_VERBOSITY]
    if isinstance(value, int):
        return max(0, min(len(VERBOSITY_LEVELS) - 1, value))
    text = str(value).strip().lower()
    if text.isdigit():
        return max(0, min(len(VERBOSITY_LEVELS) - 1, int(text)))
    return _VERBOSITY_BY_NAME.get(text, _VERBOSITY_BY_NAME[DEFAULT_VERBOSITY])


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    """Logging and per-query event-log settings — how the engine reports what it did.

    Controls the `batcher.*` logger hierarchy (console + optional rotating file) and the
    structured per-query event log (one JSON document per query: the plan, the
    Kyber/Carbonite decisions, and the measured per-operator profile). Env overrides use
    the ``BATCHER_OBSERVABILITY_*`` prefix (e.g. ``BATCHER_OBSERVABILITY_VERBOSITY=debug``,
    ``BATCHER_OBSERVABILITY_EVENT_LOG=0`` to disable the event log).

    **`verbosity` is the one dial most users need.** It is a named preset over the
    individual knobs — the ``-v``/``-vv`` ladder every CLI has, spelled out. `log_level`
    and `progress` are its two components, and each defaults to `None` meaning "derive me
    from `verbosity`"; set either explicitly to override just that one. That `None` is what
    makes the precedence unambiguous — with a concrete default there is no way to tell "the
    user asked for WARNING" from "nobody said anything", so a preset could not know whether
    it was allowed to act.

    Examples:
        .. doctest::

            >>> from batcher.config import ObservabilityConfig
            >>> ObservabilityConfig().resolved_log_level  # quiet unless asked
            'WARNING'
            >>> ObservabilityConfig(verbosity="debug").resolved_log_level
            'DEBUG'
            >>> ObservabilityConfig(verbosity="debug", log_level="ERROR").resolved_log_level
            'ERROR'
            >>> ObservabilityConfig(verbosity="quiet").resolved_progress
            'off'
    """

    # The single user-facing dial. One of silent | quiet | normal | verbose | debug | trace,
    # or the equivalent integer 0-5 (so `-vv` maps straight through). Sets `log_level` and
    # `progress` together; see `resolved_log_level` / `resolved_progress` for the table.
    verbosity: str | int = "normal"
    # Threshold for the `batcher.*` loggers and the Rust data-plane tracing bridge:
    # one of CRITICAL/ERROR/WARNING/INFO/DEBUG. `None` derives it from `verbosity`.
    log_level: str | None = None
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
    # Emit an OpenTelemetry span per query (with a child span per operator) into the
    # host's globally-configured tracer, so the engine's work appears in the enterprise's
    # existing traces. Off by default; requires `opentelemetry` installed *and* a provider
    # the host app configured (Batcher owns no exporter). Uses the same measured profile
    # as the event log, so turning it on adds only the span emit, not extra measurement.
    otel_traces: bool = False
    # Directory for event-log documents. Empty → ``$BATCHER_HOME/logs`` (or
    # ``~/.batcher/logs``), resolved at write time so `config` stays free of filesystem I/O.
    event_log_dir: str = ""
    # Keep at most this many event-log files (oldest pruned on write). 0 → unbounded.
    event_log_max_files: int = 200
    # Live terminal progress bar: "auto" renders only into a real TTY that has not set
    # NO_COLOR/TERM=dumb, "on" forces it, "off" disables it. `None` derives it from
    # `verbosity`. "auto" is the only safe resolved default — escape codes written into a
    # redirected log file are worse than no bar.
    progress: str | None = None
    # Start the web dashboard automatically on the first query. Off by default: binding a
    # port is not something a library should do without being asked. `bt.start_ui()` is
    # the explicit spelling; this is for a service that wants it always on.
    ui: bool = False
    # Where the dashboard binds. Loopback by default — the UI shows query text, plans, and
    # logs, so exposing it on a routable address must be a deliberate act.
    ui_host: str = "127.0.0.1"
    ui_port: int = 4040

    @property
    def resolved_log_level(self) -> str:
        """The effective logger threshold — `log_level` if set, else derived from `verbosity`.

        Read this rather than `log_level`, which is `None` whenever the user is driving with
        the `verbosity` dial.

        Examples:
            .. doctest::

                >>> from batcher.config import ObservabilityConfig
                >>> ObservabilityConfig().resolved_log_level
                'WARNING'
                >>> ObservabilityConfig(verbosity="verbose").resolved_log_level
                'INFO'
                >>> ObservabilityConfig(verbosity="verbose", log_level="ERROR").resolved_log_level
                'ERROR'

        Returns:
            One of CRITICAL/ERROR/WARNING/INFO/DEBUG.
        """
        return self.log_level or VERBOSITY_LEVELS[_verbosity_rank(self.verbosity)].log_level

    @property
    def resolved_progress(self) -> str:
        """The effective progress mode — `progress` if set, else derived from `verbosity`.

        Examples:
            .. doctest::

                >>> from batcher.config import ObservabilityConfig
                >>> ObservabilityConfig().resolved_progress
                'auto'
                >>> ObservabilityConfig(verbosity="quiet").resolved_progress
                'off'
                >>> ObservabilityConfig(verbosity="trace").resolved_progress
                'on'

        Returns:
            One of ``"auto"``, ``"on"``, ``"off"``.
        """
        return self.progress or VERBOSITY_LEVELS[_verbosity_rank(self.verbosity)].progress

    @property
    def resolved_native_log_level(self) -> str:
        """The Rust data plane's tracing threshold, which alone can reach ``TRACE``.

        Python's `logging` has no level below DEBUG, but the engine's `tracing` spans do, and
        they are where per-morsel work is visible. So ``verbosity="trace"`` means DEBUG in
        Python and TRACE in Rust — the one place the two ladders legitimately differ, and the
        reason this is a separate property rather than a reuse of `resolved_log_level`.

        Examples:
            .. doctest::

                >>> from batcher.config import ObservabilityConfig
                >>> ObservabilityConfig(verbosity="trace").resolved_native_log_level
                'trace'
                >>> ObservabilityConfig(verbosity="debug").resolved_native_log_level
                'debug'

        Returns:
            A `tracing` level name.
        """
        if self.log_level:
            return self.log_level
        return VERBOSITY_LEVELS[_verbosity_rank(self.verbosity)].native_level


def _check_overrides(caller: str, kind: type, overrides: dict) -> None:
    """Reject an override that is not a field of `kind`, naming the caller.

    `dataclasses.replace` already refuses, but as ``TypeError: TenantConfig.__init__() got an
    unexpected keyword argument`` — a class the user did not mention, from a function they did
    not call. A misspelled tunable is the whole reason anyone reads this message.
    """
    unknown = sorted(set(overrides) - {f.name for f in dataclasses.fields(kind)})
    if not unknown:
        return
    from batcher._internal.errors import ConfigError

    raise ConfigError(
        f"{caller}(): {unknown} " + ("is not a" if len(unknown) == 1 else "are not") + f" "
        f"{kind.__name__} field{'' if len(unknown) == 1 else 's'}.",
        available=tuple(sorted(f.name for f in dataclasses.fields(kind))),
        available_label="Available settings",
    )


@dataclass(frozen=True, slots=True)
class Config:
    """The complete engine configuration — every tunable in one frozen object.

    The single source of truth for engine tunables, grouped into typed sections:
    `execution` (parallelism, morsel size, file splits), `memory` (the memory
    envelope and spill tiers), `flow_control` (credit-based shuffle backpressure),
    `optimizer` (Kyber join planning, cost, and cardinality), `pid` (the adaptive
    batch-size controller gains), `metadata` (where learned statistics live and how
    fast they age), `distributed` (Ray attachment and shuffle transport), and
    `accelerator` (a GPU fleet's power envelope, health thresholds, and placement).

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
    streaming: StreamingConfig = StreamingConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    pid: PIDConfig = PIDConfig()
    metadata: MetadataConfig = MetadataConfig()
    distributed: DistributedConfig = DistributedConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    governance: GovernanceConfig = GovernanceConfig()
    tenant: TenantConfig = TenantConfig()
    # `default_factory`, unlike the sections above it: those are defined in this module, so
    # ruff can prove they are frozen and allows the call; `AcceleratorConfig` is imported, so
    # it cannot, and RUF009 fires. The value is identical either way.
    accelerator: AcceleratorConfig = dataclasses.field(default_factory=AcceleratorConfig)
    fault_tolerance: FaultToleranceConfig = dataclasses.field(default_factory=FaultToleranceConfig)

    def replace(self, **section_overrides: object) -> Config:
        """Return a new Config with whole sections replaced.

        Examples:
            .. doctest::

                >>> from batcher.config import Config, ExecutionConfig
                >>> cfg = Config().replace(execution=ExecutionConfig(morsel_rows=4096))
                >>> cfg.execution.morsel_rows
                4096

        Args:
            **section_overrides: Whole config sections to swap in, keyed by section
                name (``execution``, ``memory``, ``optimizer``, ...).

        Returns:
            A new Config with the given sections replaced; the original is unchanged.

        Raises:
            ConfigError: If a keyword does not name a config section.
        """
        _check_overrides("Config.replace", Config, section_overrides)
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

        Examples:
            .. doctest::

                >>> import json
                >>> from batcher.config import Config
                >>> knobs = json.loads(Config().engine_config_json())
                >>> knobs["morsel_rows"]
                16384

        Returns:
            A JSON string of the Rust-relevant execution knobs.
        """
        return _engine_config_json(self._engine_config_values())

    def engine_config_json_with(
        self, op_budgets: dict[int, int], *, prefer_materializing_aggregate: bool = False
    ) -> str:
        """`engine_config_json` plus Kyber's per-operator spill budgets.

        `op_budgets` maps a pre-order `op_id` to its byte envelope
        (`PhysicalPlan.op_budgets()`). The engine budgets each stateful operator
        against *its* entry instead of the single global `memory_budget_bytes`, so a
        small operator no longer spills while a large neighbour assumes the whole
        budget. JSON object keys must be strings; the Rust side parses them back to
        the operator id. An empty map reproduces `engine_config_json` exactly, so
        callers with no `PhysicalOp` DAG (streaming, UDFs, distributed workers) are
        unaffected.

        Examples:
            .. doctest::

                >>> import json
                >>> from batcher.config import Config
                >>> knobs = json.loads(Config().engine_config_json_with({0: 1 << 20}))
                >>> knobs["op_budgets"]
                {'0': 1048576}

        Args:
            op_budgets: Per-operator byte envelopes keyed by pre-order ``op_id``.
            prefer_materializing_aggregate: Kyber's verdict that this plan's grouped
                aggregate is cheaper materialized than streamed, taken from the estimated
                group count only the control plane has. A hint the engine may decline: it
                re-checks the plan shape and ANDs in its own memory-affordability test.

        Returns:
            A JSON string extending `engine_config_json` with the per-operator
            budgets; an empty map reproduces `engine_config_json` exactly.
        """
        if not op_budgets and not prefer_materializing_aggregate:
            return self.engine_config_json()
        return _engine_config_json_budgeted(
            self._engine_config_values(),
            tuple(sorted(op_budgets.items())),
            prefer_materializing_aggregate,
        )

    def _engine_config_values(self) -> tuple[object, ...]:
        """The Rust-relevant execution knobs as a hashable tuple (the cache key)."""
        return (
            self.execution.morsel_rows,
            self.execution.morsel_bytes,
            self.execution.parallelism,
            self.spill_budget_bytes(),
            self.memory.spill_dir,
            self.memory.spill_compression,
            self.execution.fuse_linear,
            self.execution.shrink_output_dtypes,
            self.execution.streaming,
            self.execution.bloom_fp_rate,
            self.execution.bloom_min_build_rows,
            self.execution.window_parallel_row_threshold,
            self.execution.radix_parallel_threshold,
            self.execution.sort_merge_fanin,
            self.execution.skew_bucket_factor,
            self.execution.skew_min_bucket_rows,
            self.execution.skew_min_bucket_bytes,
        )

    def validate(self) -> Config:
        """Validate the configuration, raising `ConfigError` on a bad value.

        Catches out-of-range and inconsistent tunables (a negative retry count, a
        soft limit above the hard limit, a non-positive timeout) at the config entry
        points so they fail early and clearly instead of surfacing as a confusing
        runtime failure. Returns `self` so it can be chained. Pure (no side effects).
        The checks live in `config.validation` to keep this module focused.

        Examples:
            .. doctest::

                >>> from batcher.config import Config
                >>> cfg = Config()
                >>> cfg.validate() is cfg
                True

        Returns:
            This Config, unchanged, so the call can be chained.

        Raises:
            ConfigError: If a tunable is out of range or inconsistent.
        """
        from batcher.config.validation import validate_config

        validate_config(self)
        return self

    def spill_budget_bytes(self) -> int:
        """The per-operator spill budget, in bytes — the point at which an operator spills.

        Shipped to the data plane as the engine config's `memory_budget_bytes`, and read by
        the cost model so that "this operator will spill" means the same thing to the planner
        and to the engine that will run it.

        Derived statically from `MemoryConfig` so `config` stays neutral — it never
        senses (the `api` auto-tuning resolver fills `max_memory_bytes` from the live
        envelope before execution; see `api._autotune`). `0` means unbounded (stay
        fully in-memory) and is returned **only** when the user explicitly opted out
        via `unbounded_memory`.

        When `max_memory_bytes` is unset because a caller bypassed the resolver — an
        ad-hoc `Config`, an embedded/library use, or the streaming aggregate path,
        none of which run `api._autotune` — this falls back to the *static*
        `default_total_bytes` envelope rather than to `0`. That distinction is the
        difference between the spill machinery being real and being decorative: a `0`
        budget disarms it everywhere at once (no `bc-resource` pool in `bc-py`, no
        `agg_spill` in `bc-interp::par`, and `check_budget` a no-op in the streaming
        breakers), so every stateful operator accumulates without bound and the
        process OOMs instead of spilling. A budget that is merely *wrong* only costs
        time: it is a spill threshold, and spilling is result-invariant, so
        over-estimating it on a small box spills later and under-estimating it on a
        large one spills sooner — either beats being killed.

        Examples:
            .. doctest::

                >>> import dataclasses
                >>> from batcher.config import Config
                >>> cfg = Config()
                >>> capped = dataclasses.replace(
                ...     cfg, memory=dataclasses.replace(cfg.memory, max_memory_bytes=1_000_000_000)
                ... )
                >>> capped.spill_budget_bytes()
                900000000

                >>> opted_out = dataclasses.replace(
                ...     cfg, memory=dataclasses.replace(cfg.memory, unbounded_memory=True)
                ... )
                >>> opted_out.spill_budget_bytes()
                0

        Returns:
            The spill threshold in bytes, or 0 when the user opted out of any bound.
        """
        mem = self.memory
        if mem.unbounded_memory:
            return 0
        cap = mem.max_memory_bytes
        if cap is None or cap <= 0:
            cap = mem.default_total_bytes
        return int(cap * mem.hard_limit)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None, base: Config | None = None) -> Config:
        """Overlay ``BATCHER_<SECTION>_<FIELD>`` env vars onto `base` (defaults).

        Nested sections compose by path, e.g.
        ``BATCHER_OPTIMIZER_CARDINALITY_EQ_SELECTIVITY``.

        Examples:
            .. doctest::

                >>> from batcher.config import Config
                >>> cfg = Config.from_env({"BATCHER_EXECUTION_MORSEL_ROWS": "4096"})
                >>> cfg.execution.morsel_rows
                4096

        Args:
            environ: The environment mapping to read, or None for ``os.environ``.
            base: The config to overlay onto, or None for the defaults.

        Returns:
            A new Config with the matching env vars applied over `base`.
        """
        env = os.environ if environ is None else environ
        return _resolved(_overlay_env(base if base is not None else cls(), "BATCHER", env))

    @classmethod
    def from_file(cls, path: str | os.PathLike[str], base: Config | None = None) -> Config:
        """Overlay a JSON document of nested section overrides onto `base`.

        The JSON mirrors the section structure, e.g.
        ``{"execution": {"morsel_rows": 4096}, "optimizer": {"cardinality": {...}}}``.

        Examples:
            .. doctest::

                >>> import json, tempfile
                >>> from pathlib import Path
                >>> from batcher.config import Config
                >>> path = Path(tempfile.mkdtemp()) / "cfg.json"
                >>> _ = path.write_text(json.dumps({"execution": {"morsel_rows": 4096}}))
                >>> Config.from_file(path).execution.morsel_rows
                4096

        Args:
            path: The JSON document to read.
            base: The config to overlay onto, or None for the defaults.

        Returns:
            A new Config with the document's overrides applied over `base`.
        """
        from batcher.config.serde import read_document

        data = read_document(path)
        return _resolved(_overlay_dict(base if base is not None else cls(), data))

    @classmethod
    def from_dict(cls, data: dict[str, object], base: Config | None = None) -> Config:
        """Build a Config from a nested dict of section overrides.

        The inverse of `to_dict`, and the shared implementation behind `from_file`. Keys
        not naming a real section or field are ignored rather than raising, so a document
        written for a newer Batcher still loads. Use `set_option` when you want an unknown
        name to be an error.

        Examples:
            .. doctest::

                >>> from batcher.config import Config
                >>> Config.from_dict({"execution": {"morsel_rows": 4096}}).execution.morsel_rows
                4096

        Args:
            data: A nested dict mirroring the section structure.
            base: The config to overlay onto, or None for the defaults.

        Returns:
            A new validated Config with the overrides applied over `base`.
        """
        return _resolved(_overlay_dict(base if base is not None else cls(), data))

    @classmethod
    def from_toml(cls, path: str | os.PathLike[str], base: Config | None = None) -> Config:
        """Load a Config from a TOML document whose tables mirror the config sections.

        Parsed with the standard library, so no extra dependency is needed. A section is a
        TOML table: ``[execution]`` with ``morsel_rows = 4096`` under it.

        Examples:
            .. doctest::

                >>> import tempfile
                >>> from pathlib import Path
                >>> from batcher.config import Config
                >>> p = Path(tempfile.mkdtemp()) / "batcher.toml"
                >>> _ = p.write_text("[execution]\\nmorsel_rows = 4096\\n")
                >>> Config.from_toml(p).execution.morsel_rows
                4096

        Args:
            path: The TOML document to read.
            base: The config to overlay onto, or None for the defaults.

        Returns:
            A new validated Config with the document's overrides applied over `base`.
        """
        from batcher.config.serde import read_document

        return cls.from_dict(read_document(path, fmt="toml"), base=base)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str], base: Config | None = None) -> Config:
        """Load a Config from a YAML document whose top-level keys mirror the sections.

        Requires `pyyaml`; the error says so if it is missing rather than surfacing an
        ImportError from inside the parser.

        Examples:
            .. doctest::

                >>> import tempfile
                >>> from pathlib import Path
                >>> from batcher.config import Config
                >>> p = Path(tempfile.mkdtemp()) / "batcher.yaml"
                >>> _ = p.write_text("execution:\\n  morsel_rows: 4096\\n")
                >>> Config.from_yaml(p).execution.morsel_rows  # doctest: +SKIP
                4096

        Args:
            path: The YAML document to read.
            base: The config to overlay onto, or None for the defaults.

        Returns:
            A new validated Config with the document's overrides applied over `base`.
        """
        from batcher.config.serde import read_document

        return cls.from_dict(read_document(path, fmt="yaml"), base=base)

    def to_dict(self, *, only_non_default: bool = False) -> dict[str, object]:
        """Convert to a nested plain-dict, round-tripping through `from_dict`.

        The round-trip is closed over *resolved* configs. `from_dict` runs the same
        resolution step every entry point does, which auto-detects the environment (a spot
        node, an autoscaling cluster), so reloading a config captured on one machine can
        legitimately differ from raw defaults on another. Reloading an already-resolved
        config is idempotent, which is the property to rely on.

        Examples:
            .. doctest::

                >>> from batcher.config import Config
                >>> resolved = Config.from_dict(Config().to_dict())
                >>> Config.from_dict(resolved.to_dict()) == resolved
                True

        Args:
            only_non_default: Emit only the values differing from the built-in defaults,
                producing the smallest document that reproduces this config.

        Returns:
            A nested dict mirroring the section structure.
        """
        from batcher.config.serde import config_to_dict

        return config_to_dict(self, only_non_default=only_non_default)

    def non_defaults(self) -> dict[str, object]:
        """The options that differ from the built-in defaults, as a flat dotted-key dict.

        The answer to "what is actually set here?" — the first thing worth printing when a
        run behaves differently on one machine than another, because environment variables
        and config files both land here.

        Examples:
            .. doctest::

                >>> from batcher.config import Config, ExecutionConfig
                >>> Config().replace(execution=ExecutionConfig(morsel_rows=4096)).non_defaults()
                {'execution.morsel_rows': 4096}

        Returns:
            A dict mapping dotted option path to its current value, for changed options only.
        """
        return self.diff(Config())

    def diff(self, other: Config) -> dict[str, object]:
        """The options where this config differs from `other`, as a flat dotted-key dict.

        Examples:
            .. doctest::

                >>> from batcher.config import Config, ExecutionConfig
                >>> a = Config().replace(execution=ExecutionConfig(morsel_rows=4096))
                >>> a.diff(Config())
                {'execution.morsel_rows': 4096}
                >>> Config().diff(Config())
                {}

        Args:
            other: The config to compare against.

        Returns:
            A dict mapping dotted option path to *this* config's value, for each option
            whose value differs.
        """
        from batcher.config.options import _leaves

        theirs = dict(_leaves(other))
        return {path: value for path, value in _leaves(self) if theirs.get(path) != value}

    def __repr__(self) -> str:
        """A one-line summary naming only the options that differ from the defaults.

        The generated dataclass repr is 180 fields and roughly 4,500 characters, which is
        unreadable in a traceback and useless in a notebook. This shows what was changed,
        which is the only part that carries information; `describe_options` prints the full
        table when you want it.
        """
        changed = self.non_defaults()
        if not changed:
            return "Config(<all defaults>)"
        shown = ", ".join(f"{k}={v!r}" for k, v in list(changed.items())[:8])
        more = f", +{len(changed) - 8} more" if len(changed) > 8 else ""
        return f"Config({shown}{more})"


def _coerce(raw: str, to: object) -> object:
    if to is bool:
        return truthy(raw)
    if to is int:
        return int(raw)
    if to is float:
        return float(raw)
    # A `bool | str` field (e.g. `runtime_bloom_join = "auto"`): a recognized boolean
    # token coerces to a real bool, everything else stays the string. Without this the
    # whole union was returned uncoerced, so `BATCHER_..._RUNTIME_BLOOM_JOIN=true` shipped
    # the *string* "true" — which then failed validation ("must be True, False, or 'auto'")
    # while the string literal "auto" happened to pass. Enabling/disabling the feature via
    # env raised `ConfigError`; a string-valued sentinel like "auto" still passes through.
    members = [a for a in typing.get_args(to) if a is not type(None)]
    if bool in members and (truthy(raw) or falsy(raw)):
        return truthy(raw)
    if str in members:
        return raw
    return raw


def _scalar_type(annotation: object) -> object:
    """The scalar member of an `Optional`/`X | None` annotation, else the annotation itself.

    So an env override of an optional-typed field (`int | None`, `float | None`) coerces to
    `int`/`float` rather than to the *default value's* runtime type — which is `NoneType` when
    the default is `None`, and would leave the raw string uncoerced (a wrong-typed config that
    then fails validation or ships a string across the Rust wire contract)."""
    args = typing.get_args(annotation)
    if args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


@functools.cache
def _field_scalar_types(cls: type) -> dict[str, object]:
    """Each field's coercion target type, resolved from the class annotations (cached)."""
    return {name: _scalar_type(ann) for name, ann in typing.get_type_hints(cls).items()}


def _overlay_env(obj: Config, prefix: str, env: dict[str, str]) -> Config:
    """Recursively overlay env vars onto a (possibly nested) frozen config object."""
    updates: dict[str, object] = {}
    field_types = _field_scalar_types(type(obj))
    for field in dataclasses.fields(obj):
        current = getattr(obj, field.name)
        key = f"{prefix}_{field.name.upper()}"
        if dataclasses.is_dataclass(current):
            replaced = _overlay_env(current, key, env)  # type: ignore[arg-type]
            if replaced is not current:
                updates[field.name] = replaced
        elif key in env:
            # Coerce against the *declared* field type, not `type(current)`: an optional
            # field defaulting to None would otherwise resolve to NoneType and skip coercion.
            updates[field.name] = _coerce(env[key], field_types.get(field.name, type(current)))
    return replace(obj, **updates) if updates else obj


def _overlay_dict(obj: Config, data: dict[str, object]) -> Config:
    """Recursively overlay a nested dict of overrides onto a frozen config object.

    A sequence value is restored to the declared field's type. Every serialization format
    the config is read from — JSON, TOML, YAML — has arrays but not tuples, so a sequence
    always arrives as a list; storing it as one would leave the section unequal to its
    source and unhashable, since the sections are frozen and hashable by design.
    """
    fields = {f.name for f in dataclasses.fields(obj)}
    updates: dict[str, object] = {}
    for name, value in data.items():
        if name not in fields:
            continue
        current = getattr(obj, name)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            updates[name] = _overlay_dict(current, value)  # type: ignore[arg-type]
        elif isinstance(current, tuple) and isinstance(value, list):
            updates[name] = tuple(value)
        else:
            updates[name] = value
    return replace(obj, **updates) if updates else obj


#: The last `(input, output)` pair `_resolved` produced, matched by *identity*. One slot,
#: because the access pattern it exists for is one config object resolved over and over:
#: `api.orchestration.with_auto_config` wraps every terminal op in a `config_context` over
#: the object `resolve_auto_config` memoizes, so a `collect()` loop resolves the *same*
#: object on every query. Re-deriving it costs 8.4 us, 88% of which is
#: `detect_spot_environment` reading ten environment variables to re-answer a question
#: about the machine.
#:
#: Keyed on identity, never on value: a `Config` is frozen, so the same object cannot have
#: changed, and any edit to a config builds a new object and misses. What the memo does
#: assume is that the *environment* behind `detect_spot_environment` has not moved while a
#: single config object is being reused — a node does not become preemptible mid-process,
#: and a test that exports a spot variable builds a fresh `Config` (or calls
#: `reset_resolution_memo`) and so re-detects.
_RESOLUTION_MEMO: tuple[Config, Config] | None = None


def reset_resolution_memo() -> None:
    """Forget the memoized resolution, so the next `_resolved` re-reads the environment."""
    global _RESOLUTION_MEMO
    _RESOLUTION_MEMO = None


def _resolved(cfg: Config) -> Config:
    """Auto-select the spot profile on a preemptible node, apply the resilience profile,
    auto-enable the bounded autoscale wait on an autoscaling cluster, then validate — the
    single resolution chokepoint every config entry point shares so auto-detection, the
    profile, and range checks run in lockstep regardless of how the config was built. A
    user-chosen `resilience` / `autoscale_wait_s` is never overridden (explicit wins).

    Memoized on the input's identity; see `_RESOLUTION_MEMO` for why that is sound and for
    the one assumption it makes.
    """
    global _RESOLUTION_MEMO
    from batcher.config.profiles import (
        apply_resilience_profile,
        detect_spot_environment,
        resolve_autoscale_wait,
    )

    memo = _RESOLUTION_MEMO
    if memo is not None and memo[0] is cfg:
        return memo[1]
    original = cfg
    if cfg.distributed.resilience == "default" and detect_spot_environment():
        cfg = cfg.replace(distributed=replace(cfg.distributed, resilience="spot"))
    cfg = resolve_autoscale_wait(apply_resilience_profile(cfg))
    out = cfg.validate()
    _RESOLUTION_MEMO = (original, out)
    return out


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
    """The Config in effect for the current context.

    Resolves to the innermost of: an enclosing `config_context` block, the process-wide
    `set_config`, then the static defaults layered with the config file and env vars.

    Examples:
        .. doctest::

            >>> from batcher.config import active_config
            >>> active_config().execution.morsel_rows
            16384

    Returns:
        The active Config. It is frozen — derive a new one with `Config.replace` rather
        than mutating it.
    """
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

    Args:
        config: The Config to make active. It is validated before being installed.
    """
    _active.set(_resolved(config))


@contextlib.contextmanager
def tenant(tenant_id: str, **overrides: object) -> Iterator[Config]:
    """Run this block's work as `tenant_id`, isolated from other tenants' shared state.

    Built on `config_context`, and that is the load-bearing design decision rather than an
    implementation detail: the active config is a `ContextVar`, so a tenant scope is
    correct under threads and asyncio, nests properly, and cannot leak past its `with`
    block. A module-global "current tenant" would get all three wrong.

    Inside the block, the process-global structures that would otherwise be shared are
    keyed by tenant: the result cache, the plan cache, and the learned-statistics
    namespace. Without it — the default — nothing is keyed and behavior is unchanged.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> with bt.tenant("analytics"):
            ...     bt.active_config().tenant.tenant_id
            'analytics'
            >>> bt.active_config().tenant.tenant_id
            ''

    Args:
        tenant_id: Names the tenant. Empty restores un-tenanted behavior.
        **overrides: Other `TenantConfig` fields, e.g. ``max_concurrent_queries=4``.

    Yields:
        The `Config` in effect inside the block.

    Raises:
        ConfigError: If an override is not a `TenantConfig` field.
    """
    current = active_config()
    _check_overrides("tenant", TenantConfig, overrides)
    scoped = current.replace(
        tenant=dataclasses.replace(current.tenant, tenant_id=tenant_id, **overrides)
    )
    with config_context(scoped) as cfg:
        yield cfg


@contextlib.contextmanager
def config_context(config: Config) -> Iterator[Config]:
    """Temporarily activate `config` for the duration of the `with` block.

    Validates `config` on entry (raises `ConfigError` on a bad value).

    Examples:
        .. doctest::

            >>> from batcher.config import Config, ExecutionConfig
            >>> from batcher.config import active_config, config_context
            >>> cfg = Config().replace(execution=ExecutionConfig(morsel_rows=4096))
            >>> with config_context(cfg):
            ...     active_config().execution.morsel_rows
            4096
            >>> active_config().execution.morsel_rows
            16384

    Args:
        config: The Config to activate for the block. It is validated on entry.

    Yields:
        The resolved Config that is active inside the block.
    """
    resolved = _resolved(config)
    token = _active.set(resolved)
    try:
        yield resolved
    finally:
        _active.reset(token)
