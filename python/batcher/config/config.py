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
    "OptimizerConfig",
    "PIDConfig",
    "active_config",
    "config_context",
    "set_config",
]


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
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
    # places tasks against this.
    cpus_per_task: float = 1.0


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    soft_limit: float = 0.85  # throttle at 85% of the envelope
    hard_limit: float = 0.90  # spill at 90%
    # Hard memory cap in bytes for the buffer pool / spill decision. None derives it
    # from system RAM; set it to honor a container/cgroup limit the OS won't report.
    # Setting it also opts the in-memory engine into out-of-core spilling: the budget
    # shipped to the data plane is this cap × `hard_limit` (see `engine_config_json`),
    # and the Rust runtime memory pool spills stateful operators that exceed it
    # instead of letting the process OOM.
    max_memory_bytes: int | None = None
    # Fallback total RAM (bytes) assumed when neither `max_memory_bytes` is set nor
    # the OS reports a usable figure. One home for what was a copy-pasted literal.
    default_total_bytes: int = 8 << 30  # 8 GiB
    # Out-of-core spill tiers. The local tier (NVMe) is fast and capacity-bounded;
    # once `spill_local_budget_bytes` is exhausted, new buckets overflow to
    # `spill_remote_uri` (any fsspec URL: s3://, gs://, …) so a PB-scale spill does
    # not die when local disk fills. `spill_dir` overrides the local scratch dir
    # (default: a per-query tempdir). `spill_compression` is the Arrow-IPC codec for
    # spilled batches ("lz4"/"zstd"/None); spilled data is transient, so a cheap-fast
    # codec trades CPU for disk I/O and footprint at scale.
    spill_dir: str | None = None
    spill_remote_uri: str | None = None
    spill_local_budget_bytes: int | None = None
    spill_compression: str | None = "lz4"
    # Grace recursion trigger: when a single spilled aggregate bucket's on-disk size
    # exceeds this, it is re-partitioned (by a secondary hash of the group key) into
    # sub-buckets and reduced one at a time — so a *skewed* key set that overflows one
    # bucket degrades gracefully out-of-core instead of OOMing the reduce.
    spill_bucket_max_bytes: int = 128 << 20  # 128 MiB (compressed)


@dataclass(frozen=True, slots=True)
class FlowControlConfig:
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
    anything is learned. Superseded by learned/sketch values when present."""

    # Used when a source's size is unknown (e.g. CSV): large enough that an unknown
    # side is never preferred as the (smaller) build side.
    unknown_rows: float = 1e12
    default_filter_selectivity: float = 0.5
    eq_selectivity: float = 0.1  # col = literal
    range_selectivity: float = 1.0 / 3.0  # col <|<=|>|>= literal
    null_selectivity: float = 0.05  # col IS NULL


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
    so the two never drift."""

    kp: float = 0.4
    ki: float = 0.05
    kd: float = 0.1
    integral_clamp: float = 5.0  # anti-windup bound on the integral term
    max_step_fraction: float = 0.5  # cap per-step size change to +/-50%


@dataclass(frozen=True, slots=True)
class MetadataConfig:
    backend: str = "in_process"  # "in_process" | "sqlite" | "redis" | "object_storage"
    uri: str | None = None
    decay_per_day: float = 0.1  # confidence half-life ~ a week


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
    placement_timeout_s: float = 60.0
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
    # Runtime bloom-filter join reduction (sideways information passing). When on, a
    # shuffle join builds a bloom over the small (build/right) side's keys and uses it
    # to drop provably-non-matching rows of the large (probe/left) side *before* they
    # are shuffled — cutting network volume for selective fact⋈dimension joins.
    # Always correct (the bloom has no false negatives, so only non-matching rows are
    # dropped). Opt-in (default off) because it serializes the build side's map ahead
    # of the probe's to ready the bloom — a win when the probe is much larger and the
    # join selective, an overhead on balanced joins. Inner/semi single-key joins only.
    runtime_bloom_join: bool = False
    # Shared-secret token authenticating Flight shuffle fetches (N5). When set, a
    # peer must present it to fetch a partition, so a process that can merely reach
    # the port cannot exfiltrate shuffle data. None (default) disables the check —
    # appropriate on a trusted/isolated cluster network. Also read from the
    # `BATCHER_SHUFFLE_TOKEN` env var so it can be injected without a config file.
    shuffle_token: str | None = None


@dataclass(frozen=True, slots=True)
class Config:
    """The complete engine configuration. Immutable; derive with `replace()`."""

    execution: ExecutionConfig = ExecutionConfig()
    memory: MemoryConfig = MemoryConfig()
    flow_control: FlowControlConfig = FlowControlConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    pid: PIDConfig = PIDConfig()
    metadata: MetadataConfig = MetadataConfig()
    distributed: DistributedConfig = DistributedConfig()

    def replace(self, **section_overrides: object) -> Config:
        """Return a new Config with whole sections replaced."""
        return replace(self, **section_overrides)  # type: ignore[arg-type]

    def engine_config_json(self) -> str:
        """Serialize the Rust-relevant execution knobs for the data plane.

        These keys are the wire contract with `bc_ir::EngineConfig` — keep them in
        lockstep with that struct (a Python↔Rust default-parity test guards drift).
        Core ships this string alongside the plan IR on every native execution.

        `memory_budget_bytes` is the soft cap that makes the in-memory engine spill
        stateful operators out of core. It is positive only when the user has set
        `memory.max_memory_bytes` (the explicit spill-decision cap), scaled by
        `memory.hard_limit`; otherwise it is `0` (unbounded — the engine stays fully
        in-memory, so the default fast path is unchanged).
        """
        return json.dumps(
            {
                "morsel_rows": self.execution.morsel_rows,
                "morsel_bytes": self.execution.morsel_bytes,
                "parallelism": self.execution.parallelism,
                "memory_budget_bytes": self._rust_memory_budget_bytes(),
                "spill_dir": self.memory.spill_dir,
            }
        )

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

        Derived statically from `MemoryConfig` so `config` stays neutral (no live
        sensing here — the Rust runtime pool is the adaptive backstop). `0` means
        unbounded: opt into spilling by setting `memory.max_memory_bytes`.
        """
        cap = self.memory.max_memory_bytes
        if cap is None or cap <= 0:
            return 0
        return int(cap * self.memory.hard_limit)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None, base: Config | None = None) -> Config:
        """Overlay ``BATCHER_<SECTION>_<FIELD>`` env vars onto `base` (defaults).

        Nested sections compose by path, e.g.
        ``BATCHER_OPTIMIZER_CARDINALITY_EQ_SELECTIVITY``.
        """
        env = os.environ if environ is None else environ
        return _overlay_env(base if base is not None else cls(), "BATCHER", env).validate()

    @classmethod
    def from_file(cls, path: str | os.PathLike[str], base: Config | None = None) -> Config:
        """Overlay a JSON document of nested section overrides onto `base`.

        The JSON mirrors the section structure, e.g.
        ``{"execution": {"morsel_rows": 4096}, "optimizer": {"cardinality": {...}}}``.
        """
        data = json.loads(Path(path).read_text())
        return _overlay_dict(base if base is not None else cls(), data).validate()


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
    """
    _active.set(config.validate())


@contextlib.contextmanager
def config_context(config: Config) -> Iterator[Config]:
    """Temporarily activate `config` for the duration of the `with` block.

    Validates `config` on entry (raises `ConfigError` on a bad value).
    """
    token = _active.set(config.validate())
    try:
        yield config
    finally:
        _active.reset(token)
