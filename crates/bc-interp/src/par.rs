//! The multi-core executor.
//!
//! Same operator semantics as the sequential reference (it calls the same `ops`
//! primitives), but it schedules work across a rayon thread pool:
//!
//! * **filter / project** — run per morsel in parallel (embarrassingly parallel).
//! * **aggregate / distinct** — partial-aggregate each morsel in parallel, then
//!   `combine` + `finalize` (the mergeable path from `bc-runtime::agg`).
//! * **join** — materialize both sides, hash-**shuffle** each into one bucket per
//!   worker, and join the buckets in parallel. Equal keys land in the same
//!   bucket, so the per-bucket joins are independent and their union is the full
//!   join — the identical strategy the distributed layer uses across actors.
//!
//! Result order for hash-based operators (aggregate/distinct/join) depends on the
//! worker count and so is not stable across machines; callers compare these
//! results as multisets (their outputs are unordered relations).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use arrow::array::{Array, RecordBatch};
use bc_ir::{AggFunc, AggregateItem, EngineConfig, ProjectionItem, RelOp};
use bc_resource::{CancelToken, MemoryPool, MemoryReservation};
use bc_runtime::agg::spill::{combine_finalize_spilling, DiskSpillStore, SpillCodec, SpillStore};
use bc_runtime::{agg, shuffle};
use rayon::prelude::*;

use crate::agg_par;
use crate::error::InterpError;
use crate::join_par::{
    broadcast_join, broadcast_join_streaming, build_side_swap_pays, flip_output, is_skewed_bucket,
    is_skewed_bucket_bytes, skew_salting_eligible, spilling_asof_join,
    spilling_hash_join_streaming,
};
use crate::metrics::{ExecMetrics, IdGen, OpMetric, Stopwatch};
use crate::ops;
use crate::{batch_bytes, count_rows};

/// Default rows per morsel — the unit of parallel work (§1.4). The control plane
/// can override it per execution via `EngineConfig.morsel_rows`.
const DEFAULT_TARGET_MORSEL: usize = bc_arrow::DEFAULT_MORSEL_ROWS;
/// Default byte budget per morsel — the byte-aware companion to the row target.
const DEFAULT_TARGET_MORSEL_BYTES: usize = bc_arrow::DEFAULT_MORSEL_BYTES;

/// Per-execution resource policy supplied by the control plane (Carbonite +
/// `EngineConfig`).
///
/// Defaults to all-in-memory at the engine's default morsel size and all cores
/// (fastest when the working set fits). `agg_spill` bounds peak memory by spilling
/// stateful operators to disk; `morsel_rows` / `parallelism` come from the control
/// plane's `EngineConfig`. The executor only obeys the envelope it is handed — the
/// decisions of *whether*, *how much*, and *how wide* are the control plane's.
#[derive(Clone)]
pub struct ExecOptions {
    pub agg_spill: Option<SpillOptions>,
    /// Process-wide memory accounting pool — the **runtime** backstop. A stateful
    /// breaker reserves its footprint against the pool before it builds/merges
    /// state; a reservation the pool can't grant — because other live reservations
    /// have filled the envelope — forces the operator to spill instead of pushing
    /// the process toward OOM. The pool is the live-memory ceiling the static
    /// per-operator estimate can't enforce on its own. `None` (the default) means no
    /// accounting — the fast path pays nothing. Present only when the control plane
    /// shipped a positive `memory_budget_bytes`, alongside `agg_spill`, so a forced
    /// spill always has a configured spill path.
    ///
    /// First cut: reservations are released when the operator finishes (RAII), so
    /// the budget tracks concurrently-live operator state. Holding a reservation
    /// across operator boundaries — so a downstream breaker sees an upstream
    /// breaker's *retained output* — is a follow-up (it needs the reservation
    /// threaded through `exec`'s return value).
    pub pool: Option<Arc<MemoryPool>>,
    /// Cooperative cancellation flag, polled at operator and morsel boundaries.
    ///
    /// `None` (the default) means the query cannot be cancelled and nothing is polled, so
    /// every path that does not opt in is byte-for-byte what it was. When present it is a
    /// `Relaxed` load of an `AtomicBool` — see `bc_resource::cancel` for why that ordering
    /// is both sufficient and free.
    pub cancel: Option<CancelToken>,
    /// Per-operator spill budget (bytes), keyed by the pre-order `op_id`. When an
    /// operator has an entry, [`ExecOptions::op_budget`] returns *its* envelope
    /// instead of the one global `agg_spill.memory_budget_bytes`, so each stateful
    /// operator is budgeted byte-true rather than every operator assuming it owns
    /// the whole budget. Shared (`Arc`) so the recursive `exec` clones it for free;
    /// empty (the default) ⇒ every operator falls back to the global budget.
    pub op_budgets: Arc<HashMap<u32, usize>>,
    /// Rows per morsel for parallel scheduling.
    pub morsel_rows: usize,
    /// Byte budget per morsel. A morsel is split at whichever bound (rows or
    /// bytes) trips first, so wide/variable-width data stays cache- and
    /// memory-bounded. For narrow data the row bound dominates, leaving behavior
    /// unchanged.
    pub morsel_bytes: usize,
    /// Worker threads for the parallel executor; 0 = all available cores.
    pub parallelism: usize,
    /// Fuse a maximal run of linear, per-morsel streaming operators (Filter/Project)
    /// into a *single* pass over the input's morsels, instead of one `par_iter` +
    /// intermediate `Vec` per operator. Same rows in the same order (a relation-level
    /// no-op verified against the sequential oracle); only morsel boundaries and the
    /// number of rayon dispatches change. Off by default — opt-in until it has cleared
    /// a full differential + seq==par==JIT + benchmark cycle as the default.
    pub fuse_linear: bool,
    /// Performance-threshold knobs (bloom, radix/window parallel thresholds, sort
    /// fan-in, skew) the control plane may tune per query. Default equals
    /// `RuntimeTuning::default()`, i.e. the historical consts — so absent any
    /// override the parallel executor behaves exactly as before. Threaded into the
    /// `bc-runtime` `_with` overloads on the hot path only; the sequential oracle
    /// keeps the default tuning.
    pub tuning: bc_arrow::RuntimeTuning,
}

impl Default for ExecOptions {
    fn default() -> Self {
        Self {
            agg_spill: None,
            pool: None,
            cancel: None,
            op_budgets: Arc::new(HashMap::new()),
            morsel_rows: DEFAULT_TARGET_MORSEL,
            morsel_bytes: DEFAULT_TARGET_MORSEL_BYTES,
            parallelism: 0,
            fuse_linear: false,
            tuning: bc_arrow::RuntimeTuning::default(),
        }
    }
}

impl ExecOptions {
    /// Apply the control plane's execution config (morsel size + parallelism +
    /// spill envelope). A zero `morsel_rows`/`morsel_bytes` (unset) keeps the engine
    /// default so the executor never morselizes to nothing. A positive
    /// `memory_budget_bytes` populates `agg_spill` so the main `execute_plan` path
    /// can spill stateful operators out of core; a zero budget leaves `agg_spill`
    /// `None` (fully in-memory), so a small query pays no spill cost.
    /// `Err(InterpError::Cancelled)` if cancellation has been requested, else `Ok(())`.
    ///
    /// Call this only where unwinding is already safe — between morsels, between
    /// operators, between spill runs — because returning here drops whatever the executor
    /// is holding. Mid-operator it would leak a partially-built hash table's reservation.
    #[inline]
    pub fn check_cancelled(&self) -> Result<(), InterpError> {
        match &self.cancel {
            Some(token) if token.is_cancelled() => Err(InterpError::Cancelled),
            _ => Ok(()),
        }
    }

    pub fn with_engine_config(mut self, cfg: &EngineConfig) -> Self {
        self.morsel_rows = if cfg.morsel_rows == 0 {
            DEFAULT_TARGET_MORSEL
        } else {
            cfg.morsel_rows
        };
        self.morsel_bytes = if cfg.morsel_bytes == 0 {
            DEFAULT_TARGET_MORSEL_BYTES
        } else {
            cfg.morsel_bytes
        };
        self.parallelism = cfg.parallelism;
        self.fuse_linear = cfg.fuse_linear;
        self.tuning = cfg.runtime_tuning();
        if cfg.memory_budget_bytes > 0 {
            self.agg_spill = Some(SpillOptions {
                memory_budget_bytes: cfg.memory_budget_bytes,
                dir: cfg
                    .spill_dir
                    .as_ref()
                    .map(PathBuf::from)
                    .unwrap_or_else(std::env::temp_dir),
                codec: SpillCodec::from_config_str(cfg.spill_compression.as_deref()),
            });
        }
        if !cfg.op_budgets.is_empty() {
            self.op_budgets = Arc::new(cfg.op_budgets.clone());
        }
        self
    }

    /// The combined row+byte morsel target driving [`ops::morselize`].
    pub(crate) fn morsel_target(&self) -> bc_arrow::MorselTarget {
        bc_arrow::MorselTarget::new(self.morsel_rows, self.morsel_bytes)
    }

    /// The spill budget for one operator: its Kyber-assigned per-operator bound
    /// (`op_budgets`) when present and positive, else the global
    /// `agg_spill.memory_budget_bytes`. `None` means there is no spill envelope at
    /// all (the in-memory fast path) — `op_budgets` is meaningless without a
    /// configured spill path, so an entry is only honored when spilling is enabled.
    fn op_budget(&self, op_id: u32) -> Option<usize> {
        let global = self.agg_spill.as_ref()?.memory_budget_bytes;
        Some(
            self.op_budgets
                .get(&op_id)
                .copied()
                .filter(|&b| b > 0)
                .unwrap_or(global),
        )
    }
}

/// Memory envelope + scratch location for spilling stateful operators.
#[derive(Clone, Default)]
pub struct SpillOptions {
    /// Soft cap on bytes of in-memory operator state before grace partitioning.
    pub memory_budget_bytes: usize,
    /// Directory for spill files (one IPC file per hash partition).
    pub dir: PathBuf,
    /// Compression codec for the spilled IPC streams. Perf-only and
    /// result-invariant (IPC self-describes its compression). Default `None`
    /// (uncompressed) keeps the historical bytes; set from
    /// `EngineConfig.spill_compression`.
    pub codec: SpillCodec,
}

impl SpillOptions {
    /// This envelope re-scoped to one operator's resolved budget (same spill dir),
    /// so the grace fan-out (`grace_partitions`, `spilling_hash_join`,
    /// `window_spilling`, …) partitions against the *same* per-operator budget the
    /// admission decision used — otherwise a per-op budget smaller than the global
    /// would admit-to-spill but then under-partition against the larger global.
    fn with_budget(&self, budget: usize) -> Self {
        Self {
            memory_budget_bytes: budget,
            dir: self.dir.clone(),
            codec: self.codec,
        }
    }
}

/// The in-memory-vs-spill decision for a stateful breaker, produced by [`admit`].
enum Admit {
    /// Proceed in memory. Hold the (optional) reservation until the operator's
    /// state is freed — its `Drop` returns the bytes to the pool. `None` means
    /// there is no pool to account against (the default fast path).
    InMemory(Option<MemoryReservation>),
    /// Spill out of core (a configured `agg_spill` path always exists when this is
    /// returned).
    Spill,
}

/// Decide whether a stateful operator runs in memory or spills, accounting its
/// footprint against the shared pool when it proceeds.
///
/// Spills when either the operator's own estimate already exceeds *its* budget — the
/// per-operator [`ExecOptions::op_budget`], byte-true from Kyber when present, else
/// the global envelope — **or** the process-wide pool cannot admit `estimate_bytes`
/// against its live reservations. The latter is the runtime backstop a static
/// estimate cannot enforce on its own. With no envelope (the default) `op_budget`
/// is `None`, so it always admits with no accounting and the fast path is unchanged.
impl ExecOptions {
    /// Worker threads this query may use — `parallelism`, or the machine's available cores when
    /// it is `0` ("all available cores"). The streaming executor shards its driving scan by this.
    ///
    /// Reads `available_parallelism` rather than `rayon::current_num_threads` deliberately. The
    /// latter reports the *global* pool's width, and on a Ray worker that pool is built before
    /// the actor's CPU affinity is applied and so is stuck at one thread (the throttle
    /// [`execute_parallel_with_metrics`] documents). Sizing the shard count from it would inherit
    /// that mistake and split a whole partition into a single shard; `available_parallelism`
    /// reads the affinity that has since landed.
    pub fn workers(&self) -> usize {
        if self.parallelism > 0 {
            self.parallelism
        } else {
            bc_arrow::usable_cores()
        }
    }
}

fn admit(opts: &ExecOptions, op_id: u32, estimate_bytes: usize) -> Admit {
    match opts.pool.as_ref() {
        // The pool accounts *actual* bytes, so it is the spill authority: reserve the
        // footprint cooperatively and spill only when the pool cannot admit it. Deciding
        // on actual bytes, not the per-operator *estimate*, is what stops a spurious
        // out-of-core pass when transient state exceeds a small estimate but still fits
        // RAM — e.g. a low-cardinality / global aggregate's pre-combine partials, whose
        // `op_budget` is the (tiny) combined-output size. The per-op budget still sizes
        // the grace fan-out once a spill is chosen.
        //
        // NOTE on "cooperatively". `try_reserve_cooperative` asks the largest *other*
        // registered `Spillable` to give memory back before failing the requester — Spark's
        // `MemoryConsumer` model, and the thing that stops a small operator dying while a
        // large neighbour sits on the budget.
        //
        // Exactly one consumer registers today, and it is the one that matters most on the
        // path where this is reached: `bc_py::flight::ShuffleSpiller` puts the **published
        // shuffle store** in the registry when a Flight server binds. That memory is
        // finished work waiting to be collected, so writing it out stalls nobody and costs
        // one re-read, and it is the memory the pool could not otherwise see at all.
        //
        // So the behaviour splits by deployment, and it is worth being exact about which
        // you are reading. On a distributed worker the registry is non-empty and a breaker
        // that cannot reserve makes the shuffle store yield first. On a **single-node**
        // query no Flight server exists, the registry is empty, and this is precisely
        // `try_reserve`: the requester is always the one that spills, however little it
        // holds and however much a neighbour does. Closing that half needs a `Spillable`
        // impl on the operators that own in-progress state (the aggregate's hash table, the
        // sort's runs), which is harder than it looks — the pool may call `spill` from
        // another thread, mid-`par_iter`, on state the owning operator is actively reading.
        Some(pool) => match pool.try_reserve_cooperative(estimate_bytes) {
            Ok(reservation) => Admit::InMemory(Some(reservation)),
            // Pool full (and, once consumers register, still full after they spilled):
            // spill if there is a path to spill to, else best-effort in memory (a pool
            // without an envelope can't strand the operator).
            Err(_) if opts.agg_spill.is_some() => Admit::Spill,
            Err(_) => Admit::InMemory(None),
        },
        // No pool (a standalone `agg_spill` envelope, e.g. a cargo test): fall back to
        // the per-operator estimate as the trigger so that path is unchanged.
        None if opts.op_budget(op_id).is_some_and(|b| estimate_bytes > b) => Admit::Spill,
        None => Admit::InMemory(None),
    }
}

/// Execute a plan across all available cores (all-in-memory policy).
pub fn execute_parallel(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
) -> Result<Vec<RecordBatch>, InterpError> {
    execute_parallel_with(plan, sources, &ExecOptions::default())
}

/// Execute a plan across all available cores under an explicit resource policy.
/// Identical results to [`execute_parallel`]; only peak memory differs when
/// spilling engages.
pub fn execute_parallel_with(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    opts: &ExecOptions,
) -> Result<Vec<RecordBatch>, InterpError> {
    let (out, _metrics) = execute_parallel_with_metrics(plan, sources, opts)?;
    Ok(out)
}

/// Execute across all cores and also return per-operator [`ExecMetrics`]. Result
/// batches are identical to [`execute_parallel_with`]; metrics are a side-channel.
pub fn execute_parallel_with_metrics(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    opts: &ExecOptions,
) -> Result<(Vec<RecordBatch>, ExecMetrics), InterpError> {
    let mut m = ExecMetrics::default();
    let mut ids = IdGen::new();
    tracing::debug!(
        target: "batcher.engine",
        sources = sources.len(),
        parallelism = opts.parallelism,
        "executing plan (metered)"
    );
    // `parallelism == 0` uses rayon's global pool (all cores); a positive value
    // runs the whole plan inside a scoped pool of that width, so the control
    // plane's `EngineConfig.parallelism` bounds the worker count (and the
    // hash-shuffle bucket count, which keys off `current_num_threads`).
    //
    // The plan walk that records metrics is itself single-threaded — only the
    // per-operator work fans out across rayon and is fully joined before each
    // `OpMetric` is recorded — so a plain `&mut ExecMetrics` is race-free.
    // Always run inside a width-sized scoped pool — never rayon's *global* pool. On a Ray
    // worker the global pool is built (lazily, at first use) before Ray pins the actor's
    // CPU affinity, so it is fixed at ONE thread and every `par_iter` on it runs
    // single-threaded — the silent throttle that made a distributed map ~Ncores× too slow.
    // `parallelism == 0` ("all cores") therefore resolves to `available_parallelism`, which
    // reads the now-applied affinity; a positive value pins the requested width. On a
    // single node this is the same width as the global pool, so the result is unchanged.
    let width = auto_width(opts, sources, plan);
    let out = pool_for(width)?.install(|| exec(plan, sources, opts, &mut m, &mut ids))?;
    Ok((out, m))
}

/// The worker count this execution runs on.
///
/// An explicit `EngineConfig.parallelism` is honored verbatim — the control plane asked
/// for that width and the hash-shuffle bucket count keys off it. Otherwise the width is
/// "all cores", **capped by the number of morsels the inputs can actually produce**.
///
/// A worker with no morsel to take does no work, but it is not free: rayon still wakes
/// it, it contends for the pool's job queue, and — because a scoped pool is cached per
/// width — a one-row query would otherwise install and spin a 96-thread pool. Batcher's
/// stated goal of low fixed overhead on sub-second queries is exactly this case. The cap
/// is an upper bound on useful parallelism at the leaves, so it can never remove
/// parallelism a plan could have used, and it never changes a result (scheduling only).
///
/// **Exception — media decode.** The morsel cap assumes per-morsel work is O(morsel)
/// cheap columnar work, so one morsel needs at most one core. A `.image`/`.audio`/`.video`
/// decode breaks that assumption: it does heavy, embarrassingly-parallel per-row work
/// *inside* the morsel (its own rayon fan-out — see `bc_expr::eval::media`), and its input
/// is tiny *encoded* bytes (a 5 KB JPEG) that the byte-aware count still sees as one
/// morsel. A whole corpus of images would then decode on a single core. When the plan
/// carries a media decode we lift the cap to all cores; the intra-kernel fan-out shares
/// this same pool (rayon work-stealing, no oversubscription), so it is right whether the
/// input is one morsel or many. Still scheduling only — the result is unchanged.
fn auto_width(opts: &ExecOptions, sources: &[Vec<RecordBatch>], plan: &RelOp) -> usize {
    if opts.parallelism > 0 {
        return opts.parallelism;
    }
    let cores = bc_arrow::usable_cores();
    if plan.contains_media_decode() {
        return cores.max(1);
    }
    cores.min(max_useful_workers(opts, sources)).max(1)
}

/// An upper bound on workers that could have a morsel to process: the largest number of
/// morsels any single source yields. Operators fan out over one input's morsels at a
/// time, so the widest leaf bounds the widest `par_iter`.
///
/// The count is **byte-aware**, matching how the scan actually splits: `morselize` bounds a
/// morsel by rows *and* by the byte budget, so a byte-heavy source (few rows, large blobs —
/// decoded audio/video/images, embeddings, wide strings) yields far more morsels than
/// `rows / target_rows` suggests. Counting by rows alone capped those workloads to a single
/// worker — a 176 MB audio batch of 2,000 rows morselizes into ~176 pieces but was scheduled
/// on one core. Taking the max of the row- and byte-derived counts restores the parallelism
/// the plan can genuinely use, while still collapsing to 1 for a truly tiny query.
fn max_useful_workers(opts: &ExecOptions, sources: &[Vec<RecordBatch>]) -> usize {
    let target = opts.morsel_target();
    let target_rows = target.rows.max(1);
    let target_bytes = target.bytes.max(1);
    sources
        .iter()
        .map(|batches| {
            let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            let by_rows = rows.div_ceil(target_rows);
            if target.byte_bounded() {
                let bytes: usize = batches.iter().map(ops::sliced_batch_bytes).sum();
                // A source cannot yield more morsels than it has rows: `morselize` keeps a
                // single over-budget row as its own one-row morsel, never splits within a
                // row. Capping the byte estimate by the row count keeps a few-row/huge-blob
                // source from over-provisioning workers that would have no morsel to take.
                by_rows.max(bytes.div_ceil(target_bytes).min(rows))
            } else {
                by_rows
            }
        })
        .max()
        .unwrap_or(1)
        .max(1)
}

/// Process-wide cache of fixed-width rayon thread pools, keyed by worker count.
///
/// `EngineConfig.parallelism > 0` pins a query to a scoped pool of that width.
/// Building a fresh `ThreadPool` (and spawning its worker threads) per execution
/// is a real cost on the small/streaming path — under streaming that is a new pool
/// *per micro-batch*. We instead build one pool per distinct width once and reuse
/// it across executions. Sharing a single pool per width is also *more* correct
/// than a fresh pool each time: it bounds the total worker-thread count instead of
/// letting concurrent queries each spawn `parallelism` threads. Width is the cache
/// key because `current_num_threads()` drives the hash-shuffle bucket count, so a
/// query must run on a pool of exactly the width it asked for.
/// Stack reserved per worker thread.
///
/// The partner of [`bc_ir::MAX_PLAN_DEPTH`], and the two must be read together. That guard
/// rejects a plan IR nesting past 512 levels so deserialization cannot walk off the stack —
/// but it was calibrated against **parsing** (~3.2 KiB per level), and parsing is the cheap
/// pass. `Expr` is a recursive enum, and `eval`, the analyses, and the compiler-generated
/// `Drop` each descend it carrying Arrow arrays and match temporaries: measured on the debug
/// profile, **~20 KiB per level, about six times parsing**.
///
/// So on rayon's 2 MiB default a worker *evaluated* only to ~84 levels and aborted by 104,
/// while the guard happily admitted anything under 512. Everything in that window died on a
/// **SIGSEGV** — which Rust turns into an uncatchable `SIGABRT`, and which a `_FlightWorker`
/// actor reports as an opaque `ActorDiedError`. It was not a hypothetical window:
/// `is_in` over 100 values (318 for `TfidfVectorizer(stop_words="english")`), an
/// `IsotonicCalibrator` at its default 100 bins, and a 100-term arithmetic chain all landed
/// in it.
///
/// 32 MiB carries evaluation to a measured 509 levels — past `MAX_PLAN_DEPTH`, so the guard
/// is the binding constraint again and a too-deep plan raises `PlanTooDeepError` instead of
/// killing the process. It is reserved address space, not resident memory: pages are touched
/// only to the depth actually used, so a 96-worker pool costs ~3 GiB of virtual mapping and
/// essentially no RSS.
///
/// **Raising `MAX_PLAN_DEPTH` without raising this re-opens the window.** The control plane
/// also keeps its own trees shallow rather than relying on the headroom — `Expr.is_in` folds
/// to `InList` instead of an n-deep `OR` chain, and the indicator sums fold balanced.
const WORKER_STACK_BYTES: usize = 32 * 1024 * 1024;

pub(crate) fn pool_for(width: usize) -> Result<Arc<rayon::ThreadPool>, InterpError> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let pools = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = pools
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(pool) = guard.get(&width) {
        return Ok(Arc::clone(pool));
    }
    let mut builder = rayon::ThreadPoolBuilder::new()
        .num_threads(width)
        .stack_size(WORKER_STACK_BYTES);
    // Experimental, opt-in CPU pinning (`BATCHER_PIN_THREADS=1`): pin each worker to a
    // distinct core for cache/NUMA locality on a big exclusive box. Result-invariant
    // (scheduling only), so the differential suite proves it changes nothing; off by
    // default because pinning a process-shared pool can hurt concurrent queries.
    if pin_threads_enabled() {
        builder = builder.start_handler(pin_current_thread);
    }
    let pool = Arc::new(
        builder
            .build()
            .map_err(|_| InterpError::ThreadPool(width))?,
    );
    guard.insert(width, Arc::clone(&pool));
    Ok(pool)
}

/// Whether worker-thread CPU pinning is enabled — read once from the
/// `BATCHER_PIN_THREADS` environment variable (`1`/`true`). Experimental and
/// perf-only: pinning helps cache/NUMA locality on a dedicated many-core box but can
/// hurt when concurrent queries share the process-wide pool, so it stays opt-in
/// pending benchmark validation on target hardware. It never changes results.
fn pin_threads_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("BATCHER_PIN_THREADS").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// Pin the calling rayon worker (logical index `idx`) to a CPU, following the topology
/// order from `bc_arrow::placement`. Linux-only — hard affinity via `sched_setaffinity`;
/// a no-op on other platforms (macOS exposes only advisory affinity hints), so pinning is
/// best-effort and never errors out of execution.
///
/// The CPU id comes from [`bc_arrow::pinning_order`] rather than `idx % usable_cores()`,
/// which was wrong in two ways that both showed up as "pinning enabled, nothing happened".
/// It named ids `0..n` even when a cgroup or `taskset` had narrowed the process to, say,
/// `48-95` — `sched_setaffinity` then refused every call and the error was (correctly)
/// ignored, so every worker stayed unpinned. And where it did land, adjacent indices fell on
/// SMT siblings of one core on the parts that enumerate siblings adjacently, running the pool
/// at half throughput on half the machine. The order fixes both: it only names CPUs in the
/// mask, fills distinct physical cores before any sibling, and strides across NUMA nodes.
#[cfg(target_os = "linux")]
fn pin_current_thread(idx: usize) {
    // Computed once for the process, not per worker: the order is a directory walk over
    // /sys, and a 96-CPU host would otherwise repeat it for every pool thread started.
    static ORDER: OnceLock<Vec<usize>> = OnceLock::new();
    let order = ORDER.get_or_init(bc_arrow::pinning_order);
    // An unreadable topology means "do not pin". Falling back to a modulo over the core
    // count is what produced the silent no-op above; an unpinned thread is strictly better
    // than one pinned by guesswork.
    if order.is_empty() {
        return;
    }
    let cpu = order[idx % order.len()];
    // SAFETY: a zeroed `cpu_set_t` is a valid empty set; `CPU_SET` sets one CPU id taken
    // from this process's own affinity mask, and `sched_setaffinity(0, ...)` targets the
    // current thread with a correctly-sized set. A failure (e.g. the mask narrowed between
    // detection and this call) is ignored — pinning is best-effort and never affects
    // correctness.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        let _ = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

/// No-op on non-Linux platforms (hard CPU affinity is unavailable / advisory-only).
#[cfg(not(target_os = "linux"))]
fn pin_current_thread(_idx: usize) {}

/// Backend tag for an expression operator from its compiled-JIT outcomes: `"jit"`
/// when every sub-expression compiled, `"interp"` when none did, `"interp+jit"`
/// for a mix (some fell back to the interpreter).
fn backend_tag(jits: &[bool]) -> &'static str {
    let compiled = jits.iter().filter(|c| **c).count();
    match (compiled, jits.len()) {
        (0, _) => "interp",
        (c, n) if c == n => "jit",
        _ => "interp+jit",
    }
}

fn exec(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    opts: &ExecOptions,
    m: &mut ExecMetrics,
    ids: &mut IdGen,
) -> Result<Vec<RecordBatch>, InterpError> {
    // Between operators is the cheapest safe unwind point: nothing is half-built here.
    opts.check_cancelled()?;
    // Pre-order id (parents before children) — same numbering the sequential
    // executor and the Python control plane use.
    let op_id = ids.next();
    // Fuse a run of ≥2 linear streaming ops (Filter/Project) into one per-morsel pass.
    // Off unless the control plane opted in; the result is a relation-level no-op
    // (same rows, same order — verified against the sequential oracle).
    if opts.fuse_linear && is_fusable(plan) && fusable_input(plan).is_some_and(is_fusable) {
        return exec_fused(plan, op_id, sources, opts, m, ids);
    }
    // Fuse a left-deep run of ≥2 inner hash joins into ONE pass over the probe's morsels:
    // build every table once, then thread each morsel through every probe. Falls back
    // (`None`) to the operator-at-a-time path for any chain it cannot stream.
    if opts.fuse_linear {
        if let Some(out) = exec_join_pipeline(plan, op_id, sources, opts, m, ids)? {
            return Ok(out);
        }
    }
    match plan {
        RelOp::Scan { source_id } => {
            let t0 = Stopwatch::start();
            let batches = sources.get(*source_id).ok_or(InterpError::UnknownSource {
                source_id: *source_id,
                available: sources.len(),
            })?;
            let out = ops::morselize_par(batches, opts.morsel_target());
            let rows = count_rows(&out);
            push_metric(m, op_id, "scan", rows, &out, t0, false, "interp");
            Ok(out)
        }

        RelOp::Filter { input, predicate } => {
            let parts = exec(input, sources, opts, m, ids)?;
            let rows_in = count_rows(&parts);
            let t0 = Stopwatch::start();
            // Compile the predicate once (using the first morsel as a sample),
            // then reuse the fused JIT function across all morsels.
            let jit = parts.first().and_then(|b| ops::try_compile(predicate, b));
            let backend = backend_tag(&[jit.is_some()]);
            // One conjunct-order per operator, shared across every worker: the morsels of
            // one Filter see the same data distribution, so what the first few measure is
            // what the rest should be ordered by. Lock-free and safe to share because the
            // conjuncts of an `AND` commute — see `bc_expr::ConjunctOrder`.
            let order = bc_expr::ConjunctOrder::new(predicate);
            let out: Vec<RecordBatch> = parts
                .par_iter()
                .map(|b| {
                    opts.check_cancelled()?;
                    ops::filter_batch_jit(b, predicate, &jit, order.as_ref())
                })
                .collect::<Result<_, InterpError>>()?;
            push_metric(m, op_id, "filter", rows_in, &out, t0, false, backend);
            Ok(out)
        }

        RelOp::Project { input, exprs } => {
            let parts = exec(input, sources, opts, m, ids)?;
            let rows_in = count_rows(&parts);
            let t0 = Stopwatch::start();
            let jits: Vec<ops::Jit> = match parts.first() {
                Some(sample) => exprs
                    .iter()
                    .map(|e| ops::try_compile_computed(&e.expr, sample))
                    .collect(),
                None => exprs.iter().map(|_| None).collect(),
            };
            let backend = backend_tag(&jits.iter().map(|j| j.is_some()).collect::<Vec<_>>());
            let out: Vec<RecordBatch> = parts
                .par_iter()
                .map(|b| {
                    opts.check_cancelled()?;
                    ops::project_batch_jit(b, exprs, &jits)
                })
                .collect::<Result<_, InterpError>>()?;
            // A projection can add a wide column (a large string, an embedding, a
            // decoded image), so re-bound the output to the byte budget.
            let out = ops::remorselize(out, opts.morsel_target());
            push_metric(m, op_id, "project", rows_in, &out, t0, false, backend);
            Ok(out)
        }

        RelOp::Unnest {
            input,
            column,
            alias,
            outer,
            index_alias,
        } => {
            let parts = exec(input, sources, opts, m, ids)?;
            let rows_in = count_rows(&parts);
            let t0 = Stopwatch::start();
            let out: Vec<RecordBatch> = parts
                .par_iter()
                .map(|b| {
                    opts.check_cancelled()?;
                    ops::unnest_batch(b, column, alias, *outer, index_alias.as_deref())
                })
                .collect::<Result<_, InterpError>>()?;
            // Unnest multiplies rows (a list of N explodes one row into N), so a
            // within-budget input morsel can produce an over-budget output morsel.
            let out = ops::remorselize(out, opts.morsel_target());
            push_metric(m, op_id, "unnest", rows_in, &out, t0, false, "interp");
            Ok(out)
        }

        RelOp::RowId {
            input,
            alias,
            offset,
        } => {
            // A global sequential counter, so the id pass is single-threaded over the
            // ordered upstream morsels — identical to the sequential path. (The
            // upstream still runs in parallel; only the cheap id fill is serial.)
            let parts = exec(input, sources, opts, m, ids)?;
            let rows_in = count_rows(&parts);
            let t0 = Stopwatch::start();
            let out = ops::add_row_ids(&parts, alias, *offset)?;
            push_metric(m, op_id, "row_id", rows_in, &out, t0, false, "interp");
            Ok(out)
        }

        RelOp::Unpivot {
            input,
            index,
            on,
            variable_name,
            value_name,
        } => {
            let parts = exec(input, sources, opts, m, ids)?;
            let rows_in = count_rows(&parts);
            let t0 = Stopwatch::start();
            let out: Vec<RecordBatch> = parts
                .par_iter()
                .map(|b| {
                    opts.check_cancelled()?;
                    ops::unpivot_batch(b, index, on, variable_name, value_name)
                })
                .collect::<Result<_, InterpError>>()?;
            // Unpivot stacks `on` columns into rows, multiplying row count, so
            // re-bound the output to the byte budget.
            let out = ops::remorselize(out, opts.morsel_target());
            push_metric(m, op_id, "unpivot", rows_in, &out, t0, false, "interp");
            Ok(out)
        }

        RelOp::Sample {
            input,
            fraction,
            seed,
            n,
        } => {
            let parts = exec(input, sources, opts, m, ids)?;
            let rows_in = count_rows(&parts);
            // Captured before the match, live through the fixed-count breaker's scan.
            let in_bytes = batch_bytes(&parts);
            let t0 = Stopwatch::start();
            match n {
                // Fixed-count: a breaker over all morsels (a global n-smallest-hash pass
                // holds the whole input to pick the k winners), so record its input peak —
                // a top-k over a huge input is not the ~0-peak streaming op `push_metric`
                // would have logged.
                Some(k) => {
                    let out = ops::sample_n_batches(&parts, *k, *seed)?;
                    push_breaker(
                        m, op_id, "sample", rows_in, 0, in_bytes, &out, t0, false, "interp",
                    );
                    Ok(out)
                }
                // Fractional: streaming per-morsel; its peak is the sampled result alone.
                None => {
                    let out: Vec<RecordBatch> = parts
                        .par_iter()
                        .map(|b| {
                            opts.check_cancelled()?;
                            ops::sample_batch(b, *fraction, *seed)
                        })
                        .collect::<Result<_, InterpError>>()?;
                    push_metric(m, op_id, "sample", rows_in, &out, t0, false, "interp");
                    Ok(out)
                }
            }
        }

        RelOp::Aggregate {
            input,
            group_keys,
            aggregates,
        } => {
            // Streaming Filter/Project → Aggregate fusion: fold each source morsel through
            // the fusable linear chain and directly into its partial state, never
            // materializing the full (often multi-GB) filtered/projected relation. Gated on
            // the control plane opting into linear fusion, a fusable input, and every
            // aggregate being constant-state — its out-of-core spill folds from *partials*
            // via grace partitioning, whereas the value-list aggregates (median / quantile /
            // distinct / mode / histogram) spill from the raw morsels and keep the
            // materializing path. The partials are exactly those the unfused parallel path
            // builds (same per-morsel partial-then-combine), so the result is identical
            // within the float tolerance the parallel path already carries.
            // Fuse a join directly beneath the aggregate (optionally through a Filter/Project
            // chain) so the join output is folded into partials in one pass, never materialized.
            // Declines (`None`) to the operator-at-a-time path for any shape it cannot stream.
            if opts.fuse_linear {
                if let Some(out) = try_fused_join_aggregate(
                    input, op_id, group_keys, aggregates, sources, opts, m, ids,
                )? {
                    return Ok(out);
                }
            }
            if opts.fuse_linear && is_fusable(input) && !needs_parts_for_spill(aggregates) {
                return exec_agg_fused(input, op_id, group_keys, aggregates, sources, opts, m, ids);
            }
            let parts = exec(input, sources, opts, m, ids)?;
            if parts.is_empty() {
                return Err(InterpError::EmptyAggregateInput);
            }
            let rows_in = count_rows(&parts);
            // Captured before the input vector is consumed below.
            let in_bytes = batch_bytes(&parts);
            let t0 = Stopwatch::start();
            let funcs = ops::agg_funcs(aggregates);
            // Compile computed group keys / aggregate inputs once (e.g.
            // `SUM(price * qty)`); reused across every morsel's partial, amortizing
            // the compile cost exactly like Filter/Project. `parts` is non-empty here.
            let agg_jit = ops::compile_agg(group_keys, aggregates, &parts[0]);
            // Measure what this group-by actually reduces before committing to a shape.
            // Pre-aggregating a near-unique key builds a hash table per morsel only to throw
            // it away in the merge; `agg_par` partitions the input instead and aggregates it
            // once. When grouping does reduce, the sample's partials are the first slice of
            // the work below, and are reused rather than recomputed.
            let partition_keys = agg_par::partitionable(group_keys, &parts);
            let (partials, est_groups) = match agg_par::decide(
                &parts,
                group_keys,
                aggregates,
                &agg_jit,
                partition_keys.as_deref(),
            )? {
                // The partition path holds the gathered relation where the reducing path can
                // spill its partials, so the pool decides. Declining costs the sample only.
                agg_par::AggPlan::Partition {
                    keys,
                    width,
                    groups,
                } => {
                    match admit(opts, op_id, agg_par::partition_footprint(in_bytes)) {
                        Admit::InMemory(_reservation) => {
                            let out = agg_par::partitioned_aggregate(
                                &parts, &keys, group_keys, aggregates, &agg_jit, &funcs, width,
                            )?;
                            // The partition path holds the gathered relation *and* the source
                            // morsels at once (~2×), the same footprint it was admitted on —
                            // recording 1× would under-count the non-reducing group-by.
                            push_breaker(
                                m,
                                op_id,
                                "aggregate",
                                rows_in,
                                0,
                                agg_par::partition_footprint(in_bytes) as u64,
                                &out,
                                t0,
                                false,
                                "par-agg-partitioned",
                            );
                            return Ok(out);
                        }
                        // The pool declined the gathered relation, so the bounded reducing
                        // path runs instead — over a group-by the sample says does NOT
                        // reduce, which is exactly where the merge wants a wide regroup.
                        Admit::Spill => (
                            agg_par::partials(&parts, group_keys, aggregates, &agg_jit)?,
                            groups,
                        ),
                    }
                }
                agg_par::AggPlan::Partials { partials, groups } => (partials, groups),
            };

            // Spill once the partial state exceeds the per-operator budget *or* the
            // shared pool can't admit it (cross-operator pressure); otherwise merge
            // in memory, holding a reservation for the merged state. Both branches
            // yield the same relation. An empty input (0 rows) has no working set to
            // bound and the spill primitives assume at least one row to sort/partition,
            // so it always takes the in-memory oracle path (correct empty/degenerate
            // result, e.g. a global aggregate's single all-null row or a group-by's
            // zero groups).
            let state_bytes = partial_state_bytes(&partials);
            let mut spilled = false;
            let mut spill_vol = 0u64;
            let decision = if rows_in > 0 {
                admit(opts, op_id, state_bytes)
            } else {
                Admit::InMemory(None)
            };
            let (group_columns, agg_cols) = match decision {
                Admit::Spill => {
                    let global = opts.agg_spill.as_ref().expect("spill implies an envelope");
                    // Re-scope the envelope to this operator's resolved budget so the
                    // grace fan-out partitions against the same budget admission used.
                    let sp = &global
                        .with_budget(opts.op_budget(op_id).unwrap_or(global.memory_budget_bytes));
                    spilled = true;
                    tracing::info!(
                        target: "batcher.engine",
                        op_id,
                        budget_bytes = sp.memory_budget_bytes,
                        "aggregate spilling to disk (out-of-core, bounded memory)"
                    );
                    // A lone median/quantile, n_unique, or mode spills out-of-core with
                    // bounded memory (their per-group value list can exceed memory on a
                    // hot key); a *mix* of such a value-list aggregate with constant-state
                    // ones (`median(x), sum(y)`) is bounded compositionally by
                    // `try_bounded_mixed_spill`; every other shape uses the in-memory
                    // grace path. At most one dispatch does work — the rest return `None`.
                    let bounded = ops::try_bounded_quantile_spill(
                        &parts, group_keys, aggregates, &sp.dir, sp.codec,
                    )?
                    .or(ops::try_bounded_distinct_spill(
                        &parts, group_keys, aggregates, &sp.dir, sp.codec,
                    )?)
                    .or(ops::try_bounded_mode_spill(
                        &parts, group_keys, aggregates, &sp.dir, sp.codec,
                    )?)
                    .or(ops::try_bounded_histogram_spill(
                        &parts, group_keys, aggregates, &sp.dir, sp.codec,
                    )?)
                    .or(ops::try_bounded_mixed_spill(
                        &parts,
                        group_keys,
                        aggregates,
                        &sp.dir,
                        sp.memory_budget_bytes,
                        sp.codec,
                    )?);
                    match bounded {
                        Some((gc, ac)) => (gc, ac),
                        None => {
                            let p = grace_partitions(&partials, sp.memory_budget_bytes);
                            let mut store = DiskSpillStore::with_codec(
                                sp.dir.join(format!("agg-{p}p")),
                                p,
                                sp.codec,
                            )?;
                            let res = combine_finalize_spilling(
                                partials,
                                &funcs,
                                &mut store,
                                sp.memory_budget_bytes,
                            )?;
                            spill_vol = store.spilled_bytes(); // measured grace-spill volume
                            warn_if_skewed(op_id, "aggregate", &store);
                            (res.group_columns, res.agg_columns)
                        }
                    }
                }
                Admit::InMemory(_reservation) => {
                    let merged = agg::combine_sized(
                        &partials,
                        &funcs,
                        opts.tuning.radix_parallel_threshold,
                        est_groups,
                    )?;
                    let agg_cols = agg::finalize(&funcs, &merged)?;
                    (merged.group_columns, agg_cols)
                }
            };
            // Re-morselize the grouped output (zero-copy slices) so a downstream breaker — a
            // post-aggregate projection, a sort, or a join on the grouped rows — fans back out
            // across cores rather than running on the single combined batch. A high-cardinality
            // GROUP BY (tens/hundreds of thousands of groups) otherwise emits one big batch whose
            // consumers run single-threaded. Low-cardinality output is one small batch, so this
            // is a no-op there. (The partitioned path already emits per-partition batches.)
            let out = ops::remorselize(
                vec![ops::build_agg_batch(
                    group_keys,
                    aggregates,
                    &group_columns,
                    &agg_cols,
                )?],
                opts.morsel_target(),
            );
            push_breaker_spilled(
                m,
                op_id,
                "aggregate",
                rows_in,
                0,
                in_bytes,
                &out,
                t0,
                spilled,
                spill_vol,
                "interp",
            );
            Ok(out)
        }

        RelOp::Sort { input, keys, limit } => {
            // Fused late-materialized join + top-N: when a `LIMIT k` top-N sits directly on an
            // inner hash join, gather only the sort-key columns + a `(bucket, row, row)` locator,
            // top-k that narrow relation, then gather the wide payload for just the `k` survivors
            // — never materializing the (up to multi-GB) full join output. Declines (falls through
            // to the ordinary join-then-top-N) for a spilling build or non-bare sort keys.
            if let Some(k) = limit {
                if let RelOp::HashJoin {
                    left,
                    right,
                    left_keys,
                    right_keys,
                    join_type: bc_ir::JoinType::Inner,
                    output,
                    strategy: bc_ir::JoinStrategy::Hash,
                } = input.as_ref()
                {
                    let jt0 = Stopwatch::start();
                    // Whether this fusion applies can only be known after the build side is
                    // materialized, so execute the join's children against a SCRATCH
                    // `ExecMetrics`/`IdGen` and commit to the real ones only on success — a
                    // bail must leave operator numbering and metrics exactly as the ordinary
                    // fallback (which re-executes `input`) would produce them. The scratch ids
                    // mirror a plain pre-order walk of the fused HashJoin: it consumes `op_id+1`
                    // (no metric — it is folded into the sort), then its left/right subtrees.
                    let mut sm = ExecMetrics::default();
                    let mut sid = IdGen::at(op_id + 1);
                    let _join_id = sid.next(); // the HashJoin's own pre-order id
                    let lbs = exec(left, sources, opts, &mut sm, &mut sid)?;
                    let rbs = exec(right, sources, opts, &mut sm, &mut sid)?;
                    let build_rows = count_rows(&rbs) as usize;
                    let build_bytes = batch_bytes(&rbs) as usize
                        + bc_runtime::join::estimate_build_bytes(build_rows);
                    if let Admit::InMemory(_reservation) = admit(opts, op_id, build_bytes) {
                        let p = rayon::current_num_threads().max(1);
                        if let Some(out) = ops::join_top_n(
                            &lbs,
                            &rbs,
                            left_keys,
                            right_keys,
                            output,
                            keys,
                            *k,
                            p,
                            &opts.tuning,
                        )? {
                            // Commit the speculative numbering + metrics: `sid` now sits exactly
                            // where a plain walk of `input` would leave it (`op_id + 1 +
                            // input.node_count()`), so everything after the sort still aligns
                            // with the control plane's annotation.
                            *ids = sid;
                            m.ops.extend(std::mem::take(&mut sm.ops));
                            // Commit the speculative numbering + metrics: `sid` now sits exactly
                            // where a plain walk of `input` would leave it (`op_id + 1 +
                            // input.node_count()`), so everything after the sort still aligns
                            // with the control plane's annotation.
                            let rows_in = count_rows(&lbs) + count_rows(&rbs);
                            push_metric(
                                m,
                                op_id,
                                "sort",
                                rows_in,
                                &out,
                                jt0,
                                false,
                                "interp-jointopn",
                            );
                            return Ok(out);
                        }
                    }
                    // Declined: drop the scratch metrics/ids (leaving `ids` untouched at
                    // `op_id + 1`) and fall through. The ordinary path below re-executes
                    // `input` (join included) and numbers it exactly as a plain walk would —
                    // rare (spill / computed sort key), so the re-execution is acceptable.
                }
            }
            let parts = exec(input, sources, opts, m, ids)?;
            let rows_in = count_rows(&parts);
            // Captured before the input vector is consumed below.
            let in_bytes = batch_bytes(&parts);
            let t0 = Stopwatch::start();
            if parts.is_empty() {
                push_metric(m, op_id, "sort", rows_in, &[], t0, false, "interp");
                return Ok(Vec::new());
            }
            let mut spilled = false;
            let mut sort_spill_vol = 0u64;
            // Scratch the sort allocates on top of its materialized input, which
            // `get_array_memory_size` (in `batch_bytes`) cannot see: a full in-memory sort
            // builds a permutation index (`lexsort_to_indices`, ~one u32 per row) and, for
            // the parallel sample-sort, transient range partitions. A top-N keeps only small
            // per-morsel heaps, so it adds nothing. Folding this in stops the peak
            // under-count that pushes a sort near the spill boundary the wrong way.
            let mut sort_scratch = 0u64;
            let out = match limit {
                // Top-N: each morsel computes its local top-k in parallel (cheap), and only
                // the sort-key values + a (morsel, row) locator of the P×k survivors are
                // merged — the wide payload is gathered once, for just the final k rows
                // (`parallel_top_n`, result-identical to gathering every candidate eagerly).
                // No full-input materialization; the same mergeable shape the distributed
                // top-N uses.
                Some(k) => vec![ops::parallel_top_n(&parts, keys, *k)?],
                // Full sort: out-of-core (spill sorted runs + k-way merge) when the
                // input exceeds the budget or the pool can't admit it; else
                // in-memory.
                None => {
                    let bytes = batch_bytes(&parts);
                    match admit(opts, op_id, bytes as usize) {
                        Admit::Spill => {
                            spilled = true;
                            let sp = opts.agg_spill.as_ref().expect("spill implies an envelope");
                            // Bound each sorted run to one morsel before spilling: an
                            // oversized upstream batch (a join/aggregate output that was
                            // never re-morselized) would otherwise become a single run
                            // larger than the working-set budget. The merge phase is
                            // already fan-in bounded, so this caps peak sort memory.
                            let parts = ops::remorselize(parts, opts.morsel_target());
                            // Grow each pass-0 run to a quarter of the operator's envelope
                            // (bounded by the default) rather than one run per morsel: the
                            // merge rewrites the dataset once per pass, so fewer, larger
                            // runs is directly fewer passes. A quarter leaves room for the
                            // run's own concat + sort scratch inside the envelope; with no
                            // envelope, fall back to the module default.
                            let run_target = opts
                                .op_budget(op_id)
                                .map(|b| (b as u64 / 4).max(1 << 20))
                                .unwrap_or(ops::DEFAULT_RUN_TARGET_BYTES)
                                .min(ops::DEFAULT_RUN_TARGET_BYTES);
                            let (sorted, vol) = ops::external_merge_sort(
                                parts,
                                keys,
                                &sp.dir.join("sort"),
                                opts.tuning.sort_merge_fanin,
                                run_target,
                                sp.codec,
                                opts.cancel.as_ref(),
                            )?;
                            sort_spill_vol = vol; // measured pass-0 spill volume
                            sorted
                        }
                        Admit::InMemory(_reservation) => {
                            // Permutation index (~4 B/row) plus sample-sort range scratch.
                            sort_scratch = rows_in.saturating_mul(8);
                            let combined = ops::materialize(&parts)?;
                            // Parallel sample-sort for a large single float-key full sort
                            // (range-partition + per-range parallel sort); falls back to
                            // the serial sort where it doesn't apply.
                            match ops::parallel_sort_batch(&combined, keys, None)? {
                                // Ranges come back already in key order: the sorted
                                // relation is their concatenation, so hand them to the
                                // caller as-is rather than copying them into one batch.
                                Some(sorted) => sorted,
                                None => vec![ops::sort_batch(&combined, keys, None)?],
                            }
                        }
                    }
                }
            };
            // A spilled sort holds only ~a sorted run plus the merge fan-in resident, not the
            // whole input, so the input no longer bounds its peak; cap it at the operator
            // budget so Carbonite's memory model isn't taught the in-core footprint. An
            // in-memory sort adds its permutation/range scratch to the materialized input.
            let peak_in = if spilled {
                in_bytes.min(opts.op_budget(op_id).map(|b| b as u64).unwrap_or(in_bytes))
            } else {
                in_bytes.saturating_add(sort_scratch)
            };
            push_breaker_spilled(
                m,
                op_id,
                "sort",
                rows_in,
                0,
                peak_in,
                &out,
                t0,
                spilled,
                sort_spill_vol,
                "interp",
            );
            Ok(out)
        }

        RelOp::Window {
            input,
            partition_keys,
            order_keys,
            functions,
            rank_limit,
        } => {
            // A breaker: partitioning/ordering needs the whole input. Under a memory
            // envelope with real PARTITION BY keys, grace-partition by those keys and
            // run the kernel one bucket at a time (bounded memory); otherwise
            // materialize and run the single-pass kernel.
            let parts = exec(input, sources, opts, m, ids)?;
            let rows_in = count_rows(&parts);
            // Captured before the input vector is consumed below.
            let in_bytes = batch_bytes(&parts);
            let t0 = Stopwatch::start();
            let bytes = batch_bytes(&parts);
            let has_keys = !partition_keys.is_empty();
            let (out, spill, spill_vol) = match admit(opts, op_id, bytes as usize) {
                // Grace-partition by PARTITION BY keys and run the kernel one bucket
                // at a time (bounded memory).
                Admit::Spill if has_keys => {
                    let global = opts.agg_spill.as_ref().expect("spill implies an envelope");
                    let budget = opts.op_budget(op_id).unwrap_or(global.memory_budget_bytes);
                    let (out, vol) = crate::window_spill::window_spilling(
                        &parts,
                        partition_keys,
                        order_keys,
                        functions,
                        *rank_limit,
                        budget,
                        &global.dir,
                        global.codec,
                    )?;
                    (out, true, vol)
                }
                // No PARTITION BY: the kernel needs the whole relation at once and
                // cannot grace-partition, so spilling can't bound it. Fail with a
                // typed, catchable error rather than letting the process OOM.
                Admit::Spill => {
                    return Err(InterpError::MemoryBudgetExceeded {
                        needed: bytes as usize,
                        budget: opts.op_budget(op_id).unwrap_or(0),
                        reason: "window without PARTITION BY cannot spill",
                    });
                }
                Admit::InMemory(_reservation) => {
                    let out = match ops::materialize(&parts) {
                        Ok(combined) => {
                            vec![ops::window_batch_with(
                                &combined,
                                partition_keys,
                                order_keys,
                                functions,
                                *rank_limit,
                                opts.tuning.window_parallel_row_threshold,
                            )?]
                        }
                        Err(_) => Vec::new(),
                    };
                    (out, false, 0)
                }
            };
            // The window kernel emits its whole result as one (up to full-input-sized)
            // batch. Left as-is, every downstream operator processes that lone batch on a
            // single core (a col-ref Project over a 6M-row window output measured ~50 ms
            // single-threaded). Re-morselize so the pipeline below the breaker fans back
            // out across cores; the split is zero-copy Arrow slices. Same rows, same order.
            let out = ops::remorselize(out, opts.morsel_target());
            push_breaker_spilled(
                m, op_id, "window", rows_in, 0, in_bytes, &out, t0, spill, spill_vol, "interp",
            );
            Ok(out)
        }

        RelOp::Limit { input, n, offset } => {
            let parts = exec(input, sources, opts, m, ids)?;
            let rows_in = count_rows(&parts);
            let t0 = Stopwatch::start();
            let out = ops::limit(parts, *n, *offset);
            push_metric(m, op_id, "limit", rows_in, &out, t0, false, "interp");
            Ok(out)
        }

        RelOp::AsofJoin {
            left,
            right,
            left_on,
            right_on,
            left_by,
            right_by,
            direction,
            tolerance,
            allow_exact_matches,
            output,
        } => {
            // ASOF is a sorted nearest-match within each `by` group. The inputs are
            // computed in parallel, then joined: with `by` keys, equal `by` values
            // co-partition to the same bucket on both sides (the nearest-`on` match
            // never crosses a `by` group), so the buckets are independent ASOF joins
            // run in parallel and their union is the full result. A keyless ASOF has
            // no key to partition on → one sequential pass, matching the oracle.
            let left_batches = exec(left, sources, opts, m, ids)?;
            let right_batches = exec(right, sources, opts, m, ids)?;
            // The probe side (left) drives the per-row probe cost; the build side (right)
            // drives the hash table's memory. Their sum made both meaningless.
            let rows_in = count_rows(&left_batches);
            let rows_build = count_rows(&right_batches);
            let in_bytes = batch_bytes(&left_batches) + batch_bytes(&right_batches);
            let t0 = Stopwatch::start();
            let left = ops::materialize(&left_batches)?;
            let right = ops::materialize(&right_batches)?;
            let mut spilled = false;
            let out = if left_by.is_empty() {
                // A keyless ASOF needs one global order on `on`, so it cannot
                // grace-partition. If a memory envelope is configured and the inputs
                // exceed it, fail loudly with a typed error rather than risk an OOM
                // (mirrors the no-PARTITION-BY window). With no envelope (the default)
                // it runs in memory exactly as before.
                let bytes = left.get_array_memory_size() + right.get_array_memory_size();
                if let Some(budget) = opts.op_budget(op_id) {
                    if bytes > budget {
                        return Err(InterpError::MemoryBudgetExceeded {
                            needed: bytes,
                            budget,
                            reason: "keyless ASOF join needs one global order and cannot spill",
                        });
                    }
                }
                vec![ops::asof_join_batches(
                    &left,
                    &right,
                    left_on,
                    right_on,
                    left_by,
                    right_by,
                    *direction,
                    *tolerance,
                    *allow_exact_matches,
                    output,
                )?]
            } else {
                // Spill to a grace ASOF join when the larger side exceeds the budget
                // or the shared pool can't admit it; otherwise join each co-partitioned
                // bucket in memory. Both yield the same relation.
                let bytes = left
                    .get_array_memory_size()
                    .max(right.get_array_memory_size());
                match admit(opts, op_id, bytes) {
                    Admit::Spill => {
                        let global = opts.agg_spill.as_ref().expect("spill implies an envelope");
                        let sp = &global.with_budget(
                            opts.op_budget(op_id).unwrap_or(global.memory_budget_bytes),
                        );
                        spilled = true;
                        spilling_asof_join(
                            &left,
                            &right,
                            left_on,
                            right_on,
                            left_by,
                            right_by,
                            *direction,
                            *tolerance,
                            *allow_exact_matches,
                            output,
                            sp,
                        )?
                    }
                    Admit::InMemory(_reservation) => {
                        let p = rayon::current_num_threads().max(1);
                        let li = ops::key_indices(&left, left_by)?;
                        let ri = ops::key_indices(&right, right_by)?;
                        let lb = shuffle::partition_by_keys(&left, &li, p)?;
                        let rb = shuffle::partition_by_keys(&right, &ri, p)?;
                        (0..p)
                            .into_par_iter()
                            .map(|i| {
                                ops::asof_join_batches(
                                    &lb[i],
                                    &rb[i],
                                    left_on,
                                    right_on,
                                    left_by,
                                    right_by,
                                    *direction,
                                    *tolerance,
                                    *allow_exact_matches,
                                    output,
                                )
                            })
                            .collect::<Result<Vec<_>, InterpError>>()?
                    }
                }
            };
            push_breaker(
                m,
                op_id,
                "asof_join",
                rows_in,
                rows_build,
                in_bytes,
                &out,
                t0,
                spilled,
                "interp",
            );
            Ok(out)
        }

        RelOp::RangeJoin {
            left,
            right,
            conditions,
            join_type,
            output,
        } => {
            // The inputs are computed in parallel; the join itself is one sweep. A range
            // join *is* decomposable — a left row's matches depend on the whole right side
            // and on nothing else about the left — so chunking the left side would
            // parallelize the sweep, at the cost of re-sorting the right side per chunk.
            // That trade only pays when the left side dominates, so it is left for the
            // block decomposition that would also make this distributable, rather than
            // guessed at here. Sequential O(n log n + k) already replaces a quadratic plan.
            let left_batches = exec(left, sources, opts, m, ids)?;
            let right_batches = exec(right, sources, opts, m, ids)?;
            let rows_in = count_rows(&left_batches);
            let rows_build = count_rows(&right_batches);
            let in_bytes = batch_bytes(&left_batches) + batch_bytes(&right_batches);
            let t0 = Stopwatch::start();
            let left = ops::materialize(&left_batches)?;
            let right = ops::materialize(&right_batches)?;
            let out = vec![ops::range_join_batches(
                &left, &right, conditions, *join_type, output,
            )?];
            push_breaker(
                m,
                op_id,
                "range_join",
                rows_in,
                rows_build,
                in_bytes,
                &out,
                t0,
                false,
                "interp",
            );
            Ok(out)
        }

        RelOp::HashJoin {
            left,
            right,
            left_keys,
            right_keys,
            join_type,
            output,
            strategy,
        } => {
            let left_batches = exec(left, sources, opts, m, ids)?;
            let right_batches = exec(right, sources, opts, m, ids)?;

            // ── Runtime build-side correction ────────────────────────────────────────
            // The planner chose which side to build from *estimated* cardinalities. Both
            // relations are now materialized, so their sizes are facts, and every decision
            // below this line — spill vs in-memory, streaming probe vs shuffle, hash-table
            // cache residency — is made against the build side. Correcting the orientation
            // here costs two slice rebindings and an output re-label; leaving it wrong
            // costs a grace hash join that did not need to happen. See
            // `join_par::build_side_swap_pays` for why this is restricted to `Inner`.
            let swap = build_side_swap_pays(
                *join_type,
                count_rows(&left_batches) as usize,
                count_rows(&right_batches) as usize,
            );
            let (left_batches, right_batches) = if swap {
                (right_batches, left_batches)
            } else {
                (left_batches, right_batches)
            };
            let (left_keys, right_keys) = if swap {
                (right_keys, left_keys)
            } else {
                (left_keys, right_keys)
            };
            let flipped_output: Vec<bc_ir::JoinOutputCol>;
            let output: &[bc_ir::JoinOutputCol] = if swap {
                flipped_output = flip_output(output);
                &flipped_output
            } else {
                output
            };

            // The probe side (left) drives the per-row probe cost; the build side (right)
            // drives the hash table's memory. Their sum made both meaningless. Both are
            // read *after* the correction above, so what Carbonite learns as `n_build` is
            // the table the join actually built, not the one the planner nominated.
            let rows_in = count_rows(&left_batches);
            let rows_build = count_rows(&right_batches);
            // The hash table / chain / null mask built over the build side is the join's
            // largest allocation and is live alongside both materialized inputs at the
            // probe's peak, yet `get_array_memory_size` (in `batch_bytes`) sees only the raw
            // build columns. Add the structural overhead `estimate_build_bytes` accounts for
            // (2–10× on narrow keys) — the same figure admission reserves against below — so
            // the peak Carbonite learns from matches what the join actually holds.
            let in_bytes = batch_bytes(&left_batches)
                + batch_bytes(&right_batches)
                + bc_runtime::join::estimate_build_bytes(rows_build as usize) as u64;
            let t0 = Stopwatch::start();

            // Byte-true build size computed from the build *batches* — WITHOUT
            // concatenating them. The old `materialize(&right_batches)` here built one
            // giant batch before the spill check, so a build too big for memory OOMed
            // before it could spill. The size is the columns plus the hash table /
            // chain / null mask `get_array_memory_size` omits (2–10× on narrow keys).
            let build_rows = count_rows(&right_batches) as usize;
            let build_bytes = batch_bytes(&right_batches) as usize
                + bc_runtime::join::estimate_build_bytes(build_rows);
            // Hold the reservation for the whole in-memory join (build + shuffle +
            // probe), so the build side is accounted in the shared pool while it is
            // live — otherwise a concurrent query sees free budget that isn't and
            // over-commits. Dropped when this arm returns.
            let _build_guard = match admit(opts, op_id, build_bytes) {
                Admit::Spill => {
                    let global = opts.agg_spill.as_ref().expect("spill implies an envelope");
                    let sp = &global
                        .with_budget(opts.op_budget(op_id).unwrap_or(global.memory_budget_bytes));
                    // Stream both sides to disk batch-by-batch (never materializing
                    // the full build side), then join one bucket at a time.
                    let (out, spill_vol) = spilling_hash_join_streaming(
                        &left_batches,
                        &right_batches,
                        left_keys,
                        right_keys,
                        *join_type,
                        output,
                        sp,
                    )?;
                    push_breaker_spilled(
                        m,
                        op_id,
                        "hash_join",
                        rows_in,
                        rows_build,
                        in_bytes,
                        &out,
                        t0,
                        true,
                        spill_vol,
                        "interp",
                    );
                    return Ok(out);
                }
                Admit::InMemory(reservation) => reservation,
            };

            // Broadcast: the planner found the right side small enough to replicate.
            // Probe the large left side without shuffling it (no key partitioning).
            if *strategy == bc_ir::JoinStrategy::Broadcast {
                // The build side is concatenated here (it is the small one, and one hash
                // table is built over the whole of it). The probe side is NOT — a broadcast
                // join streams it morsel by morsel, and that copy is the query's largest.
                // The shuffle path below never concatenates either side.
                let right = ops::materialize(&right_batches)?;
                // Stream the probe wherever it is provably safe (a probe-driven join type
                // over integer keys); otherwise fall back to concatenating it. The relation
                // is the same either way — see `broadcast_join_streaming`.
                if let Some(out) = broadcast_join_streaming(
                    &left_batches,
                    &right,
                    left_keys,
                    right_keys,
                    *join_type,
                    output,
                )? {
                    // The streaming broadcast never materializes the probe (the query's
                    // largest relation) — it flows one morsel at a time — so its true peak is
                    // the build side (columns + hash table) plus a single probe morsel and the
                    // result, not the whole probe `in_bytes` includes. Recording `in_bytes`
                    // here over-provisioned the broadcast strategy and biased the planner away
                    // from it (its whole point is to avoid holding the probe).
                    let probe_morsel = left_batches
                        .iter()
                        .map(|b| batch_bytes(std::slice::from_ref(b)))
                        .max()
                        .unwrap_or(0);
                    let build_live = batch_bytes(&right_batches)
                        + bc_runtime::join::estimate_build_bytes(rows_build as usize) as u64;
                    push_breaker(
                        m,
                        op_id,
                        "hash_join",
                        rows_in,
                        rows_build,
                        build_live.saturating_add(probe_morsel),
                        &out,
                        t0,
                        false,
                        "interp",
                    );
                    return Ok(out);
                }
                let left = ops::materialize(&left_batches)?;
                let out = broadcast_join(&left, &right, left_keys, right_keys, *join_type, output)?;
                // The broadcast probe emits only ~`probe_rows/build_rows` chunks (as few as
                // a handful when the build side is large), which would run every downstream
                // operator on those few big batches single-threaded. Re-morselize (zero-copy
                // slices) so the pipeline below fans back out across cores.
                let out = ops::remorselize(out, opts.morsel_target());
                push_breaker(
                    m,
                    op_id,
                    "hash_join",
                    rows_in,
                    rows_build,
                    in_bytes,
                    &out,
                    t0,
                    false,
                    "interp",
                );
                return Ok(out);
            }

            // A `Hash` join whose build side turns out to be small enough to hold as ONE table
            // is better served by the streaming probe than by the shuffle below — and here, at
            // execution, that is not an estimate: `right_batches` is in hand, so `build_rows`
            // is the *fact* the planner could only guess at.
            //
            // Why it wins: the shuffle path scatters BOTH sides into buckets, and the probe is
            // the query's largest relation — measured on TPC-H q5, partitioning the 1.2M-row
            // probe cost 20 ms against the 6 ms of the join it was preparing. The streaming path
            // copies no probe at all. What used to make it lose anyway was its *serial* build
            // (10.7 ms on one core); `bc_runtime::join::build` now shards that across every
            // core, so the trade that justified the shuffle no longer holds.
            //
            // `streaming_supported` answers from the build's schema + row count alone, so a join
            // it cannot serve (a `Right`/`Full` join, a non-integer key, a build past the
            // cache-radix cliff) falls through to the shuffle below having copied nothing. The
            // planner's `Broadcast`/`SortMerge` choices are still honored above; this only
            // reconsiders `Hash`, and only downward, where the real size says it is safe.
            if *strategy == bc_ir::JoinStrategy::Hash && !right_batches.is_empty() {
                let schema = right_batches[0].schema();
                let key_types: Option<Vec<_>> = right_keys
                    .iter()
                    .map(|k| schema.field_with_name(k).ok().map(|f| f.data_type()))
                    .collect();
                let eligible = key_types.as_ref().is_some_and(|ts| {
                    bc_runtime::join::streaming_supported(
                        ops::map_join_type(*join_type),
                        ts,
                        build_rows,
                    )
                });
                if eligible {
                    let right = ops::materialize(&right_batches)?;
                    // NOT morselized, unlike the planner-chosen broadcast above. Measured:
                    // doing so here is a 4% geomean REGRESSION across TPC-H (10 queries worse,
                    // 4 better). This branch is reached after the planner picked `Hash`, so the
                    // probe has come through a filter/project that already re-morselized it —
                    // the extra pass buys no parallelism and costs a copy. The other site is
                    // fed straight from a scan, which is why it needs the split and this
                    // does not.
                    if let Some(out) = broadcast_join_streaming(
                        &left_batches,
                        &right,
                        left_keys,
                        right_keys,
                        *join_type,
                        output,
                    )? {
                        // The probe never materializes, so the true peak is the build side
                        // (columns + table) plus one probe morsel and the result — the same
                        // accounting the planner-chosen broadcast above records.
                        let probe_morsel = left_batches
                            .iter()
                            .map(|b| batch_bytes(std::slice::from_ref(b)))
                            .max()
                            .unwrap_or(0);
                        let build_live = batch_bytes(&right_batches)
                            + bc_runtime::join::estimate_build_bytes(rows_build as usize) as u64;
                        push_breaker(
                            m,
                            op_id,
                            "hash_join",
                            rows_in,
                            rows_build,
                            build_live.saturating_add(probe_morsel),
                            &out,
                            t0,
                            false,
                            "interp-shared",
                        );
                        return Ok(out);
                    }
                }
            }

            let (out, skewed_any) = join_partitioned(
                &left_batches,
                &right_batches,
                left_keys,
                right_keys,
                *join_type,
                output,
                *strategy,
                opts,
            )?;
            let out = ops::remorselize(out, opts.morsel_target());
            let backend = match (skewed_any, *strategy == bc_ir::JoinStrategy::SortMerge) {
                (true, _) => "interp-skew",
                (false, true) => "interp-smj",
                (false, false) => "interp",
            };
            push_breaker(
                m,
                op_id,
                "hash_join",
                rows_in,
                rows_build,
                in_bytes,
                &out,
                t0,
                false,
                backend,
            );
            Ok(out)
        }

        RelOp::Distinct {
            input,
            keys,
            order,
            limit,
        } => {
            // A `DISTINCT ON` carrying a limit goes to the sequential oracle. This path's
            // `distinct_on` is the spilling dedup and does not promise first-seen order, so
            // truncating it could keep a different `k` than the oracle keeps — two tiers
            // disagreeing on the answer, which is invariant #6. Kyber only fuses a limit into a
            // whole-row `DISTINCT`, so this is unreachable in practice and exists so a
            // hand-written plan cannot diverge.
            if limit.is_some() && !keys.is_empty() {
                return crate::execute(plan, sources);
            }
            let parts = exec(input, sources, opts, m, ids)?;
            let rows_in = count_rows(&parts);
            // Captured before the input vector is consumed below.
            let in_bytes = batch_bytes(&parts);
            let t0 = Stopwatch::start();
            // The limited whole-row case keeps the first `k` distinct rows in input order.
            // `parts` is already materialized here, so unlike the streaming breaker this saves
            // no input reads — it is here to agree with the oracle, not to be fast. The
            // spilling `distinct` below emits bucket order, which truncation would scramble.
            if let Some(k) = limit {
                let out: Vec<RecordBatch> = bc_runtime::agg::distinct_prefix(&parts, *k)?
                    .into_iter()
                    .collect();
                push_breaker_spilled(
                    m, op_id, "distinct", rows_in, 0, in_bytes, &out, t0, false, 0, "interp",
                );
                return Ok(out);
            }
            let (batches, spilled, spill_vol) = match keys.is_empty() {
                true => distinct(&parts, opts, op_id)?,
                false => distinct_on(&parts, keys, order, opts, op_id)?,
            };
            // Re-morselize (zero-copy slices) so a downstream breaker — a COUNT(DISTINCT)'s
            // outer GROUP BY, or a join — fans back out across cores instead of processing the
            // whole relation on one thread. A single large distinct batch feeding an aggregate
            // ran that aggregate at ~1% CPU (TPC-H Q16: the outer group-by over the deduped
            // rows was 62% of the query, serial).
            let out = ops::remorselize(batches, opts.morsel_target());
            push_breaker_spilled(
                m, op_id, "distinct", rows_in, 0, in_bytes, &out, t0, spilled, spill_vol, "interp",
            );
            Ok(out)
        }

        RelOp::Union {
            inputs,
            distinct: dedup,
        } => {
            // Branches run **across the pool**, not one after another. They are independent
            // plans — own scans, own joins, own aggregates — and a serial loop here was the
            // materializing twin of the defect fixed in `stream::parallel`: measured on TPC-DS
            // q22's five grouping levels, the union ran at 5.8 cores where one level alone runs
            // at 63.6, and cost 8x the sum of its parts.
            //
            // The obstacle was never the data, it was `m`/`ids`: both are `&mut` and cannot
            // cross threads. `IdGen::at` + `RelOp::node_count` already exist for exactly this
            // (the fused join pipeline runs its build before its probe), so each branch gets the
            // id range a pre-order walk would have handed it, and its metrics land in a scratch
            // `ExecMetrics` that is merged back **in branch order**. Numbering and metrics are
            // therefore identical to the serial loop's, which is what keeps them aligned with
            // the control plane's `annotate_ops`.
            let mut all = Vec::new();
            if inputs.len() > 1 && rayon::current_num_threads() > 1 {
                // Pre-order: this `Union` took `op_id`, so branch k starts after every node of
                // branches 0..k.
                let mut starts = Vec::with_capacity(inputs.len());
                let mut next = ids.peek();
                for inp in inputs {
                    starts.push(next);
                    next += inp.node_count();
                }
                let per: Vec<(Vec<RecordBatch>, ExecMetrics)> = inputs
                    .par_iter()
                    .zip(starts)
                    .map(|(inp, start)| {
                        let mut sm = ExecMetrics::default();
                        let mut sid = IdGen::at(start);
                        exec(inp, sources, opts, &mut sm, &mut sid).map(|b| (b, sm))
                    })
                    .collect::<Result<_, _>>()?;
                for (batches, sm) in per {
                    all.extend(batches);
                    for op in sm.ops {
                        m.record(op);
                    }
                }
                *ids = IdGen::at(next);
            } else {
                for inp in inputs {
                    all.extend(exec(inp, sources, opts, m, ids)?);
                }
            }
            // Promotable-but-different branch types (`int64 ∪ float64`) are coerced to the
            // union's advertised supertype before concat/dedup, matching DuckDB.
            let all = crate::coerce_union_branches(all)?;
            let rows_in = count_rows(&all);
            // Captured before `all` is consumed: the dedup path materializes and hashes it.
            let in_bytes = batch_bytes(&all);
            let t0 = Stopwatch::start();
            if *dedup {
                // A deduplicating UNION runs the full grace-capable `distinct` (materialize
                // + hash + possible spill) — a breaker holding its input plus the deduped
                // result — not the ~0-peak streaming op `push_metric` recorded.
                let (out, spilled, spill_vol) = distinct(&all, opts, op_id)?;
                push_breaker_spilled(
                    m, op_id, "union", rows_in, 0, in_bytes, &out, t0, spilled, spill_vol, "interp",
                );
                Ok(out)
            } else {
                // UNION ALL streams: it concatenates handles and holds only the result.
                push_metric(m, op_id, "union", rows_in, &all, t0, false, "interp");
                Ok(all)
            }
        }
    }
}

/// Record one parallel-executor operator metric from its result batches.
#[allow(clippy::too_many_arguments)]
/// A linear, per-morsel, row-wise streaming operator that can be fused into a single
/// pass over its input's morsels. Filter and Project qualify (pure per-batch, no global
/// state, no row multiplication that would need re-morselizing mid-chain). Unnest /
/// Unpivot (row multiplication), Sample, and RowId (global counter) are left out of the
/// first cut.
fn is_fusable(op: &RelOp) -> bool {
    matches!(op, RelOp::Filter { .. } | RelOp::Project { .. })
}

/// The single input of a fusable op (its child in the linear chain).
fn fusable_input(op: &RelOp) -> Option<&RelOp> {
    match op {
        RelOp::Filter { input, .. } | RelOp::Project { input, .. } => Some(input),
        _ => None,
    }
}

/// Join two materialized relations with the **partitioned** algorithm: bucket both sides by
/// key, then join each co-partitioned pair across cores, spreading a hot bucket's probe over
/// worker chunks. Returns the joined batches and whether any bucket was skewed.
///
/// This is the algorithm to use when the build side is *not* small: it builds `p` tables of
/// `build/p` rows across cores, where a broadcast/streamed join builds one table over the whole
/// build side, serially. Shared by the ordinary hash-join arm and the fused join pipeline, so a
/// pipeline stage whose build is too large to broadcast runs exactly what the unfused path runs.
#[allow(clippy::too_many_arguments)]
fn join_partitioned(
    left_batches: &[RecordBatch],
    right_batches: &[RecordBatch],
    left_keys: &[String],
    right_keys: &[String],
    join_type: bc_ir::JoinType,
    output: &[bc_ir::JoinOutputCol],
    strategy: bc_ir::JoinStrategy,
    opts: &ExecOptions,
) -> Result<(Vec<RecordBatch>, bool), InterpError> {
    let p = rayon::current_num_threads().max(1);

    // Bucket both sides straight from their morsels, gathering each row **once**.
    // `materialize(&batches)` here used to concatenate a relation only for
    // `partition_by_keys` to gather every row again a moment later — two full copies
    // where one does. The buckets, their contents, and the row order within each are
    // identical (`ops::repartition`), so the per-bucket join and the `seq == par`
    // oracle see exactly what they saw before.
    let rb = ops::partition_morsels(right_batches, right_keys, p)?;
    let lb = ops::partition_morsels(left_batches, left_keys, p)?;

    // Skew handling: a hot key sends all its rows to one bucket, making that
    // per-bucket join a straggler. Detect it for free from the partition
    // sizes (no extra pass) and spread the over-large bucket's *driving*
    // (probe) side across worker chunks against its (co-partitioned) build
    // bucket — the chunked join `broadcast_join` uses. The driving side is
    // the right for a `Right` join, the left otherwise; `Full` is ineligible.
    // Every bucket still computes the same relation.
    let salt = skew_salting_eligible(join_type);
    let driving_is_right = matches!(join_type, bc_ir::JoinType::Right);
    let driving_bucket = |i: usize| if driving_is_right { &rb[i] } else { &lb[i] };
    let (driving_rows, driving_bytes) = if driving_is_right {
        (
            count_rows(right_batches) as usize,
            batch_bytes(right_batches) as usize,
        )
    } else {
        (
            count_rows(left_batches) as usize,
            batch_bytes(left_batches) as usize,
        )
    };
    let avg = driving_rows / p.max(1);
    let avg_bytes = driving_bytes / p.max(1);
    // Hot by rows OR by bytes: a hot key of wide rows concentrates work even
    // at a modest row count, which the row-only test cannot see. Salting is
    // result-invisible, so widening the trigger never changes the output.
    let is_hot = |i: usize| {
        let b = driving_bucket(i);
        is_skewed_bucket(
            b.num_rows(),
            avg,
            opts.tuning.skew_bucket_factor,
            opts.tuning.skew_min_bucket_rows,
        ) || is_skewed_bucket_bytes(
            b.get_array_memory_size(),
            avg_bytes,
            opts.tuning.skew_bucket_factor,
            opts.tuning.skew_min_bucket_bytes,
        )
    };
    let skewed_any = salt && (0..p).any(is_hot);

    // Per-bucket join honors the planner's strategy (hash or sort-merge);
    // equal keys share a bucket, so the union of per-bucket joins is the
    // full join for either algorithm.
    let out: Vec<RecordBatch> = (0..p)
        .into_par_iter()
        .map(|i| -> Result<Vec<RecordBatch>, InterpError> {
            if salt && is_hot(i) {
                broadcast_join(&lb[i], &rb[i], left_keys, right_keys, join_type, output)
            } else {
                Ok(vec![ops::join_batches_with(
                    &lb[i],
                    &rb[i],
                    left_keys,
                    right_keys,
                    join_type,
                    output,
                    strategy,
                    &opts.tuning,
                )?])
            }
        })
        .collect::<Result<Vec<_>, InterpError>>()?
        .into_iter()
        .flatten()
        .collect();
    // Even out the per-bucket output batches (a hot key makes one bucket far larger
    // than the rest) so the downstream operators see balanced, core-sized morsels.
    let out = ops::remorselize(out, opts.morsel_target());
    Ok((out, skewed_any))
}

/// One stage of a fused join pipeline: a build side hashed **once**, plus everything needed
/// to probe a morsel against it and emit the joined batch.
struct JoinStage<'a> {
    op_id: u32,
    table: bc_runtime::join::BroadcastProbe,
    build: RecordBatch,
    probe_keys: &'a [String],
    output: &'a [bc_ir::JoinOutputCol],
    schema: arrow::datatypes::SchemaRef,
    rows_build: u64,
}

/// Execute a left-deep run of inner hash joins as ONE pass over the probe side's morsels.
///
/// The operator-at-a-time path runs a join chain by fully materializing each join's output
/// and handing it to the next: TPC-H q5 writes a 94 MB intermediate, reads it back to join it
/// again into a 17 MB one, and reads *that* back. Two costs follow. The obvious one is the
/// memory traffic. The one that actually dominates is parallelism: each join fans out over
/// *its own input's* morsels, so a chain that funnels 6 M rows down to 182 k and then to 7 k
/// ends up running its last joins over a handful of morsels — a couple of busy cores while
/// the rest idle (measured: ~50 % CPU on q5, with no single operator above 25 % of wall).
///
/// Fused, the chain is driven by the *base* probe's morsels — 366 of them for `lineitem` —
/// and every stage runs inside that same `par_iter`. Each morsel is threaded through every
/// probe in turn and only the final, joined batch is ever materialized. Parallelism is the
/// base's morsel count for the whole chain, and the intermediates never exist.
///
/// **The relation is unchanged.** Each stage applies exactly the join the unfused path
/// applies — same build table (`BroadcastProbe` is the same build/probe the streaming
/// broadcast uses), same output columns, same row order (morsels in order, rows in order
/// within a morsel). Only the boundary at which rows are materialized moves.
///
/// Returns `None` — having consumed nothing — for any chain it cannot stream: fewer than two
/// joins, a non-inner join type, or a build side `BroadcastProbe` declines (too large for its
/// table to stay cache-resident, or keys outside the streamable set). The caller then runs the
/// ordinary path.
///
/// Whether it applies can only be known *after* the build sides are materialized, so the
/// build/base execution runs against a **scratch** `ExecMetrics`/`IdGen` and is committed to
/// the real ones only on success — a bail leaves operator numbering and metrics exactly as the
/// fallback path will produce them.
fn exec_join_pipeline(
    plan: &RelOp,
    op_id: u32,
    sources: &[Vec<RecordBatch>],
    opts: &ExecOptions,
    m: &mut ExecMetrics,
    ids: &mut IdGen,
) -> Result<Option<Vec<RecordBatch>>, InterpError> {
    // The left spine of inner hash joins, outermost first. `node` ends as the base probe.
    let mut spine: Vec<&RelOp> = Vec::new();
    let mut node = plan;
    while let RelOp::HashJoin {
        left, join_type, ..
    } = node
    {
        if !matches!(join_type, bc_ir::JoinType::Inner) {
            break;
        }
        spine.push(node);
        node = left;
    }
    if spine.len() < 2 {
        return Ok(None);
    }
    let base_plan = node;
    let n = spine.len();

    // Ids, without running anything: the spine takes the ids the recursion hands out on the way
    // down, then the base probe's subtree, then each build side from the innermost out.
    // `node_count` gives each subtree's span, so the builds can be executed after the base and
    // still receive exactly the ids a plain pre-order walk would.
    let join_ids: Vec<u32> = (0..n as u32).map(|i| op_id + i).collect();
    let base_first_id = op_id + n as u32;
    let mut build_first_ids: Vec<u32> = Vec::with_capacity(n);
    let mut next_id = base_first_id + base_plan.node_count();
    for join in spine.iter().rev() {
        let RelOp::HashJoin { right, .. } = join else {
            unreachable!("the spine holds only hash joins")
        };
        build_first_ids.push(next_id);
        next_id += right.node_count();
    }

    // Probe side, then every build (innermost out) — the recursion's own order.
    let mut sm = ExecMetrics::default();
    let mut pid = IdGen::at(base_first_id);
    let mut cur = exec(base_plan, sources, opts, &mut sm, &mut pid)?;
    debug_assert_eq!(pid.peek(), build_first_ids[0]);
    if cur.is_empty() {
        return Ok(None); // empty probe: let the ordinary path produce the empty relation
    }
    let mut builds: Vec<Vec<RecordBatch>> = Vec::with_capacity(n);
    for (i, join) in spine.iter().rev().enumerate() {
        let RelOp::HashJoin { right, .. } = join else {
            unreachable!("the spine holds only hash joins")
        };
        let mut bid = IdGen::at(build_first_ids[i]);
        builds.push(exec(right, sources, opts, &mut sm, &mut bid)?);
    }
    // Committed. Everything below only chooses *how* to join what is already in hand, so the
    // chain never hands work back after paying for it.
    *ids = IdGen::at(next_id);
    m.ops.extend(std::mem::take(&mut sm.ops));

    // Walk the chain innermost out. Consecutive stages that should stream are fused into ONE
    // pass over the morsels in flight; a stage that should not is joined the ordinary
    // partitioned way, materializing at that point and no other.
    // The *live* tuning, not the compiled-in default: these values are shipped from Python's
    // `ExecutionConfig`, and a fused stage must make the same bloom decision the unfused join
    // would — otherwise fusing a chain would silently change a user's configured behaviour.
    let tuning = &opts.tuning;
    let mut i = 0usize;
    while i < n {
        let cur_rows = count_rows(&cur);
        let cur_schema = cur
            .first()
            .map(|b| b.schema())
            .expect("cur is non-empty: the base was, and every stage emits at least a schema");

        // How many consecutive stages from `i` can stream? A stage streams when its build is
        // hashable into one table AND is the *smaller* side. The second condition is the one
        // that matters: a streamed stage builds ONE table over the whole build side, serially,
        // where the partitioned join builds `p` tables of `build/p` rows across cores — so
        // streaming only pays once the probe is large enough to amortize that serial build.
        // (TPC-H q2 is the counter-example: a 1.5 k-row probe against an 800 k-row build.)
        let mut run: Vec<JoinStage> = Vec::new();
        let mut schema = cur_schema;
        let mut j = i;
        while j < n {
            let RelOp::HashJoin {
                left_keys,
                right_keys,
                join_type,
                output,
                ..
            } = spine[n - 1 - j]
            else {
                unreachable!("the spine holds only hash joins")
            };
            let rows_build = count_rows(&builds[j]);
            // Judged against the rows entering this *run*. Inside a run the relation only
            // changes by the joins we are fusing, and a chain is built to funnel down — so this
            // is an upper bound on what a later stage in the run actually probes with.
            if rows_build > cur_rows {
                break;
            }
            let build = ops::materialize(&builds[j])?;
            let Some(table) = bc_runtime::join::BroadcastProbe::new(
                &ops::columns_by_name(&build, right_keys)?,
                ops::map_join_type(*join_type),
                cur_rows as usize,
                tuning.bloom_fp_rate,
                tuning.bloom_min_build_rows,
            ) else {
                break;
            };
            // One shape check per stage: every morsel reaching it carries `schema`, so a probe
            // that accepts this sample's keys accepts all of them.
            let sample = RecordBatch::new_empty(Arc::clone(&schema));
            if !table.accepts(&ops::columns_by_name(&sample, left_keys)?) {
                break;
            }
            let out_schema = ops::join_output_schema(&sample, &build, output)?;
            run.push(JoinStage {
                op_id: join_ids[n - 1 - j],
                table,
                build,
                probe_keys: left_keys,
                output,
                schema: Arc::clone(&out_schema),
                rows_build,
            });
            schema = out_schema;
            j += 1;
        }

        if run.is_empty() {
            // This stage's build is the larger side (or is unhashable): run the algorithm the
            // unfused path would, on the relation in flight.
            let RelOp::HashJoin {
                left_keys,
                right_keys,
                join_type,
                output,
                strategy,
                ..
            } = spine[n - 1 - i]
            else {
                unreachable!("the spine holds only hash joins")
            };
            let t0 = Stopwatch::start();
            let rows_in = count_rows(&cur);
            let rows_build = count_rows(&builds[i]);
            let (out, skewed) = join_partitioned(
                &cur, &builds[i], left_keys, right_keys, *join_type, output, *strategy, opts,
            )?;
            let elapsed_ns = t0.elapsed_ns();
            let (cpu_ns, peak_rss_bytes, hw) = t0.measure();
            m.record(OpMetric {
                op_id: join_ids[n - 1 - i],
                kind: "hash_join",
                rows_in,
                rows_build,
                rows_out: count_rows(&out),
                elapsed_ns,
                wall_span_ns: 0,
                cpu_ns,
                threads: rayon::current_num_threads().max(1) as u32,
                peak_bytes: batch_bytes(&out),
                result_bytes: batch_bytes(&out),
                spilled: false,
                spill_bytes: 0,
                peak_rss_bytes,
                backend: if skewed { "interp-skew" } else { "interp" },
                hw,
            });
            cur = out;
            i += 1;
            continue;
        }

        // One pass over the morsels in flight, through every stage of the run.
        let t0 = Stopwatch::start();
        let rows_in = count_rows(&cur);
        let out: Vec<RecordBatch> =
            cur.par_iter()
                .map(|morsel| {
                    let mut b = morsel.clone();
                    for st in &run {
                        let keys = ops::columns_by_name(&b, st.probe_keys)?;
                        let idx = st.table.probe(&keys).ok_or_else(|| {
                            InterpError::UnknownJoinColumn(st.probe_keys.join(", "))
                        })?;
                        b = ops::gather_join_output_with(
                            &b,
                            &st.build,
                            &idx,
                            st.output,
                            Arc::clone(&st.schema),
                        )?;
                    }
                    Ok(b)
                })
                .collect::<Result<Vec<_>, InterpError>>()?;
        // A join concatenates both sides' columns, so re-bound the output to the byte budget
        // exactly as the unfused join path does.
        let out = ops::remorselize(out, opts.morsel_target());

        // One metric per fused stage. Only the run's last stage has a materialized relation to
        // measure and only its first sees the rows entering the run — an interior stage of a
        // fused pass has no relation of its own, and inventing one would put a fiction into the
        // cardinalities Kyber learns from.
        let k = run.len() as u64;
        let elapsed = t0.elapsed_ns().max(1) / k;
        let (run_cpu, run_rss, run_hw) = t0.measure();
        let cpu = run_cpu / k;
        let hw = run_hw.split(k);
        let rows_out = count_rows(&out);
        let out_bytes = batch_bytes(&out);
        for (s, st) in run.iter().enumerate() {
            let last = s + 1 == run.len();
            m.record(OpMetric {
                op_id: st.op_id,
                kind: "hash_join",
                rows_in: if s == 0 { rows_in } else { 0 },
                rows_build: st.rows_build,
                rows_out: if last { rows_out } else { 0 },
                elapsed_ns: elapsed,
                wall_span_ns: 0,
                cpu_ns: cpu,
                threads: rayon::current_num_threads().max(1) as u32,
                peak_bytes: if last { out_bytes } else { 0 },
                result_bytes: if last { out_bytes } else { 0 },
                spilled: false,
                spill_bytes: 0,
                // RSS growth is a high-water delta, not an additive quantity, so it is
                // attributed whole to the stage that owns the materialized relation rather
                // than split like the additive counters.
                peak_rss_bytes: if last { run_rss } else { 0 },
                backend: "interp-pipelined",
                hw,
            });
        }
        cur = out;
        i = j;
    }
    Ok(Some(cur))
}

/// One compiled stage of a fused linear pipeline: a per-morsel operator with its
/// expression(s) compiled once (against a representative sample) and reused across
/// every morsel — the same compile-once-per-operator discipline as the unfused path.
enum FusedStage<'a> {
    Filter {
        op_id: u32,
        predicate: &'a bc_expr::Expr,
        jit: ops::Jit,
        /// Measured conjunct order for this operator, shared across its morsels. Built
        /// here rather than per morsel for the same reason `jit` is: it is per-operator
        /// state, and rebuilding it per morsel would throw away the measurement.
        order: Option<bc_expr::ConjunctOrder>,
        backend: &'static str,
    },
    Project {
        op_id: u32,
        exprs: &'a [bc_ir::ProjectionItem],
        jits: Vec<ops::Jit>,
        backend: &'static str,
    },
}

impl FusedStage<'_> {
    fn apply(&self, b: &RecordBatch) -> Result<RecordBatch, InterpError> {
        match self {
            FusedStage::Filter {
                predicate,
                jit,
                order,
                ..
            } => ops::filter_batch_jit(b, predicate, jit, order.as_ref()),
            FusedStage::Project { exprs, jits, .. } => ops::project_batch_jit(b, exprs, jits),
        }
    }
    fn op_id(&self) -> u32 {
        match self {
            FusedStage::Filter { op_id, .. } | FusedStage::Project { op_id, .. } => *op_id,
        }
    }
    fn kind(&self) -> &'static str {
        match self {
            FusedStage::Filter { .. } => "filter",
            FusedStage::Project { .. } => "project",
        }
    }
    fn backend(&self) -> &'static str {
        match self {
            FusedStage::Filter { backend, .. } | FusedStage::Project { backend, .. } => backend,
        }
    }
}

/// Compile one fusable op into a [`FusedStage`], using `sample` (its input's first
/// morsel) for the JIT — mirroring the per-op compile in the unfused arms exactly, so
/// the compiled fast path / interpreter-fallback choice is identical.
fn compile_stage<'a>(op_id: u32, op: &'a RelOp, sample: Option<&RecordBatch>) -> FusedStage<'a> {
    match op {
        RelOp::Filter { predicate, .. } => {
            let jit = sample.and_then(|s| ops::try_compile(predicate, s));
            let backend = backend_tag(&[jit.is_some()]);
            FusedStage::Filter {
                op_id,
                predicate,
                jit,
                order: bc_expr::ConjunctOrder::new(predicate),
                backend,
            }
        }
        RelOp::Project { exprs, .. } => {
            let jits: Vec<ops::Jit> = match sample {
                Some(s) => exprs
                    .iter()
                    .map(|e| ops::try_compile_computed(&e.expr, s))
                    .collect(),
                None => exprs.iter().map(|_| None).collect(),
            };
            let backend = backend_tag(&jits.iter().map(|j| j.is_some()).collect::<Vec<_>>());
            FusedStage::Project {
                op_id,
                exprs,
                jits,
                backend,
            }
        }
        _ => unreachable!("compile_stage is only called on fusable ops"),
    }
}

/// Execute a maximal run of fusable ops in one pass over the input's morsels.
///
/// Produces exactly the rows the unfused path does, in the same order — filter∘project
/// applied per morsel is identical to filter-all-morsels then project-all-morsels, and
/// the morsels are concatenated in order. Only the morsel boundaries and the number of
/// rayon dispatches differ. Per-operator metrics are emitted with **exact** row counts
/// (selectivity, the cardinality-critical signal, is preserved); the segment's
/// wall-time is split evenly across the fused stages (per-op timing is an attribution
/// once fused — the documented trade).
fn exec_fused(
    plan: &RelOp,
    op_id: u32,
    sources: &[Vec<RecordBatch>],
    opts: &ExecOptions,
    m: &mut ExecMetrics,
    ids: &mut IdGen,
) -> Result<Vec<RecordBatch>, InterpError> {
    // Collect the chain outermost→innermost, assigning pre-order ids (the outermost
    // already holds `op_id`); `base` ends as the first non-fusable input. This is the
    // exact pre-order numbering the recursive `exec` would assign.
    let mut chain: Vec<(u32, &RelOp)> = vec![(op_id, plan)];
    let mut base = fusable_input(plan).expect("a fusable op has an input");
    while is_fusable(base) {
        chain.push((ids.next(), base));
        base = fusable_input(base).expect("a fusable op has an input");
    }

    // Execute the non-fusable base (consumes its own pre-order id and subtree, and
    // pushes its own metric — so the scan/base metric precedes the fused-op metrics,
    // just as in the recursive path).
    let base_morsels = exec(base, sources, opts, m, ids)?;
    let base_rows = count_rows(&base_morsels);

    let t0 = Stopwatch::start();
    // Compile each stage bottom-up (innermost first), advancing a sample batch through
    // the chain so each op compiles against the schema it actually sees — identical to
    // the unfused path compiling each op against its input's first morsel.
    let mut sample = base_morsels.first().cloned();
    let mut stages: Vec<FusedStage> = Vec::with_capacity(chain.len());
    for (id, op) in chain.iter().rev() {
        let stage = compile_stage(*id, op, sample.as_ref());
        if let Some(s) = &sample {
            sample = Some(stage.apply(s)?);
        }
        stages.push(stage);
    }

    // One pass per morsel through every stage, tracking the row count after each stage
    // (`stages` is in apply order, innermost→outermost) so the per-op metrics are exact.
    let n = stages.len();
    let results: Vec<(RecordBatch, Vec<u64>)> = base_morsels
        .par_iter()
        .map(|b| {
            opts.check_cancelled()?;
            let mut cur = b.clone();
            let mut stage_rows = Vec::with_capacity(n);
            for stage in &stages {
                cur = stage.apply(&cur)?;
                stage_rows.push(cur.num_rows() as u64);
            }
            Ok((cur, stage_rows))
        })
        .collect::<Result<Vec<_>, InterpError>>()?;

    // Single-threaded reduce after the join (keeps the `&mut ExecMetrics` race-free):
    // sum per-stage rows and gather the final morsels in order.
    let mut totals = vec![0u64; n];
    let mut out: Vec<RecordBatch> = Vec::with_capacity(results.len());
    for (batch, rows) in results {
        for (i, r) in rows.iter().enumerate() {
            totals[i] += r;
        }
        out.push(batch);
    }
    // A projection can widen a column, so re-bound the fused output to the byte budget
    // (matches the unfused Project path; relation-preserving — rows and order unchanged).
    let out = ops::remorselize(out, opts.morsel_target());

    // Emit one metric per fused op in apply order (children before parents, as the
    // recursion does). Rows exact; wall-time split evenly; peak bytes to the outermost
    // op (the one whose output is `out`).
    let elapsed = t0.elapsed_ns().max(1) / n as u64;
    let (fused_cpu, fused_rss, fused_hw) = t0.measure();
    let cpu = fused_cpu / n as u64;
    let hw = fused_hw.split(n as u64);
    let threads = rayon::current_num_threads().max(1) as u32;
    let out_bytes = batch_bytes(&out);
    for (i, stage) in stages.iter().enumerate() {
        let rows_in = if i == 0 { base_rows } else { totals[i - 1] };
        let peak_bytes = if stage.op_id() == op_id { out_bytes } else { 0 };
        m.record(OpMetric {
            op_id: stage.op_id(),
            kind: stage.kind(),
            rows_in,
            rows_out: totals[i],
            elapsed_ns: elapsed,
            wall_span_ns: 0,
            cpu_ns: cpu,
            threads,
            peak_bytes,
            result_bytes: peak_bytes,
            rows_build: 0,
            spilled: false,
            spill_bytes: 0,
            // The high-water delta belongs to the outermost op, the one whose output is the
            // materialized relation; splitting a high-water mark would be meaningless.
            peak_rss_bytes: if stage.op_id() == op_id { fused_rss } else { 0 },
            backend: stage.backend(),
            hw,
        });
    }
    Ok(out)
}

/// Whether any aggregate needs the raw input morsels (not just partials) for its
/// out-of-core spill — the value-list aggregates whose per-group state is unbounded.
/// Those keep the materializing path; everything else spills from partials via grace.
fn needs_parts_for_spill(aggregates: &[AggregateItem]) -> bool {
    aggregates.iter().any(|a| {
        matches!(
            a.func,
            AggFunc::Median
                | AggFunc::Quantile
                | AggFunc::CountDistinct
                | AggFunc::Mode
                // The contiguity statistics hold a per-group value list exactly as `Median`
                // does, so they need the same partitioning to stay bounded. Omitting them
                // here compiles and passes every small test, and lets a grouped `n50` over a
                // hot key grow its list until the process dies.
                | AggFunc::NLength
                | AggFunc::LCount
                | AggFunc::AuN
                | AggFunc::Histogram
                | AggFunc::ListAgg
                | AggFunc::ApproxCountDistinct
                | AggFunc::ApproxQuantile
                | AggFunc::Entropy
                | AggFunc::Mad
                | AggFunc::QuantileDisc
                | AggFunc::ApproxTopK
        )
    })
}

/// Fused HashJoin → (Filter/Project)* → Aggregate: build the broadcast probe table once, then
/// thread each probe morsel through the join, the linear chain, and straight into a partial
/// aggregate — the join's output (on TPC-H `lineitem ⋈ orders` it is 6M rows / ~200 MB) is
/// never materialized, and the separate group-by pass over it is gone. DuckDB and Polars win
/// this shape precisely because they fuse it; this is the same fusion, on the same mergeable
/// partials the unfused path builds (same probe, same `gather_join_output_with`, same
/// per-morsel `partial`-then-`combine`), so the `seq == par` oracle sees an identical relation.
///
/// Returns `None` — fall through to the operator-at-a-time path, which is correct for every
/// shape — when the base is not an inner hash join, the probe is empty, or the build side is
/// too large / unhashable for a broadcast probe (the partitioned shuffle join is right there).
/// A value-list aggregate (median/quantile/mode/…) also declines: it needs the raw rows for its
/// out-of-core spill. Gated on `fuse_linear`, like the sibling fusions.
#[allow(clippy::too_many_arguments)]
fn try_fused_join_aggregate(
    input: &RelOp,
    op_id: u32,
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    sources: &[Vec<RecordBatch>],
    opts: &ExecOptions,
    m: &mut ExecMetrics,
    ids: &mut IdGen,
) -> Result<Option<Vec<RecordBatch>>, InterpError> {
    if needs_parts_for_spill(aggregates) {
        return Ok(None);
    }
    // Peel the linear Filter/Project chain (outermost first); the base must be an inner join.
    let mut chain: Vec<&RelOp> = Vec::new();
    let mut node = input;
    while is_fusable(node) {
        chain.push(node);
        node = fusable_input(node).expect("a fusable op has an input");
    }
    let RelOp::HashJoin {
        left,
        right,
        left_keys,
        right_keys,
        join_type,
        output,
        ..
    } = node
    else {
        return Ok(None);
    };
    if !matches!(join_type, bc_ir::JoinType::Inner) {
        return Ok(None);
    }

    // Pre-order ids (matching `exec`): chain ops, the join, then the probe & build subtrees.
    let chain_ids: Vec<u32> = chain.iter().map(|_| ids.next()).collect();
    let join_id = ids.next();
    let mut sm = ExecMetrics::default();
    let mut pid = IdGen::at(ids.peek());
    let probe_batches = exec(left, sources, opts, &mut sm, &mut pid)?;
    if probe_batches.is_empty() {
        return Ok(None); // empty probe: the ordinary path builds the empty relation. ids untouched.
    }
    let mut bid = IdGen::at(pid.peek());
    let build_batches = exec(right, sources, opts, &mut sm, &mut bid)?;
    let next_id = bid.peek();

    // Broadcast the build side — at **any** size, unlike the un-fused join beside it.
    //
    // `BroadcastProbe::new` refuses a build past L3 because a flat probe would lose to the
    // partitioned join. That is the wrong comparison here: declining does not send this query to
    // a partitioned probe, it sends it to materializing the join's whole output and grouping it
    // in a second pass. At sf10 `lineitem ⋈ orders` that is a 2.0 GB intermediate and a separate
    // 60M-row pass, against a probe whose cache misses are bounded by the (much smaller) build.
    // See `BroadcastProbe::over_any_build`.
    let build = ops::materialize(&build_batches)?;
    let tuning = &opts.tuning;
    let probe_rows: usize = probe_batches.iter().map(|b| b.num_rows()).sum();
    let Some(table) = bc_runtime::join::BroadcastProbe::over_any_build(
        &ops::columns_by_name(&build, right_keys)?,
        ops::map_join_type(*join_type),
        probe_rows,
        tuning.bloom_fp_rate,
        tuning.bloom_min_build_rows,
    ) else {
        return Ok(None);
    };
    let first_keys = ops::columns_by_name(&probe_batches[0], left_keys)?;
    if !table.accepts(&first_keys) {
        return Ok(None);
    }

    // Committed: advance ids and fold `sm`'s subtree metrics in exactly once.
    *ids = IdGen::at(next_id);
    m.ops.extend(std::mem::take(&mut sm.ops));

    // Compile the chain stages and the aggregate against the (empty) post-join / post-chain
    // schema — identical to the shape every morsel produces below.
    let join_schema = ops::join_output_schema(&probe_batches[0], &build, output)?;
    let mut sample = RecordBatch::new_empty(Arc::clone(&join_schema));
    let mut stages: Vec<FusedStage> = Vec::with_capacity(chain.len());
    for (id, op) in chain_ids.iter().zip(chain.iter()).rev() {
        let stage = compile_stage(*id, op, Some(&sample));
        sample = stage.apply(&sample)?;
        stages.push(stage);
    }
    let agg_jit = ops::compile_agg(group_keys, aggregates, &sample);

    let t0 = Stopwatch::start();
    // One pass: probe → gather join output → linear chain → fold into a partial. The join
    // output for a morsel lives only until it is folded, so peak memory is the build side
    // (columns + hash table) plus one morsel's join output and its partial — never the whole
    // join. Runs across cores; the partials merge associatively, so this equals the serial fold.
    let partials: Vec<agg::Partial> = probe_batches
        .par_iter()
        .map(|morsel| {
            let keys = ops::columns_by_name(morsel, left_keys)?;
            let idx = table
                .probe(&keys)
                .ok_or_else(|| InterpError::UnknownJoinColumn(left_keys.join(", ")))?;
            let mut cur = ops::gather_join_output_with(
                morsel,
                &build,
                &idx,
                output,
                Arc::clone(&join_schema),
            )?;
            for stage in &stages {
                cur = stage.apply(&cur)?;
            }
            ops::eval_partial_jit(&cur, group_keys, aggregates, &agg_jit)
        })
        .collect::<Result<Vec<_>, InterpError>>()?;

    // Combine the partials — the same in-memory / grace-spill path as the unfused aggregate.
    let funcs = ops::agg_funcs(aggregates);
    let state_bytes = partial_state_bytes(&partials);
    let (group_columns, agg_cols) = match admit(opts, op_id, state_bytes) {
        Admit::Spill => {
            let global = opts.agg_spill.as_ref().expect("spill implies an envelope");
            let sp =
                &global.with_budget(opts.op_budget(op_id).unwrap_or(global.memory_budget_bytes));
            let p = grace_partitions(&partials, sp.memory_budget_bytes);
            let mut store =
                DiskSpillStore::with_codec(sp.dir.join(format!("agg-{p}p")), p, sp.codec)?;
            let res =
                combine_finalize_spilling(partials, &funcs, &mut store, sp.memory_budget_bytes)?;
            (res.group_columns, res.agg_columns)
        }
        Admit::InMemory(_reservation) => {
            let merged =
                agg::combine_with(&partials, &funcs, opts.tuning.radix_parallel_threshold)?;
            let agg_cols = agg::finalize(&funcs, &merged)?;
            (merged.group_columns, agg_cols)
        }
    };
    let out = vec![ops::build_agg_batch(
        group_keys,
        aggregates,
        &group_columns,
        &agg_cols,
    )?];
    push_breaker(
        m,
        op_id,
        "aggregate",
        probe_rows as u64,
        count_rows(&build_batches),
        batch_bytes(&build_batches)
            + bc_runtime::join::estimate_build_bytes(build.num_rows()) as u64,
        &out,
        t0,
        false,
        "fused-join-agg",
    );
    let _ = (join_id, join_type);
    Ok(Some(out))
}

/// Run the fused linear chain over one morsel, returning the chained batch and the row count
/// after each stage (which is what gives the fused ops exact selectivity metrics).
fn run_chain(
    stages: &[FusedStage],
    b: &RecordBatch,
    opts: &ExecOptions,
) -> Result<(RecordBatch, Vec<u64>), InterpError> {
    opts.check_cancelled()?;
    let mut cur = b.clone();
    let mut stage_rows = Vec::with_capacity(stages.len());
    for stage in stages {
        cur = stage.apply(&cur)?;
        stage_rows.push(cur.num_rows() as u64);
    }
    Ok((cur, stage_rows))
}

/// Emit one metric per fused linear op (children before parents), as `exec_fused` does.
///
/// The chain's wall/CPU time is split evenly across its stages: they ran interleaved inside
/// one parallel map, so there is no per-stage measurement to report and an even split is the
/// only honest attribution.
fn emit_stage_metrics(
    m: &mut ExecMetrics,
    stages: &[FusedStage],
    totals: &[u64],
    base_rows: u64,
    stage_t0: Stopwatch,
) {
    let n = stages.len().max(1) as u64;
    let stage_elapsed = stage_t0.elapsed_ns().max(1) / n;
    let (fused_cpu, _fused_rss, fused_hw) = stage_t0.measure();
    let stage_cpu = fused_cpu / n;
    let stage_hw = fused_hw.split(n);
    let threads = rayon::current_num_threads().max(1) as u32;
    for (i, stage) in stages.iter().enumerate() {
        let rows_in = if i == 0 { base_rows } else { totals[i - 1] };
        m.record(OpMetric {
            op_id: stage.op_id(),
            kind: stage.kind(),
            rows_in,
            rows_out: totals[i],
            elapsed_ns: stage_elapsed,
            wall_span_ns: 0,
            cpu_ns: stage_cpu,
            threads,
            peak_bytes: 0,
            result_bytes: 0,
            rows_build: 0,
            spilled: false,
            spill_bytes: 0,
            // These linear stages hold no relation of their own (the aggregate downstream
            // does), so there is no working set to attribute a high-water mark to.
            peak_rss_bytes: 0,
            backend: stage.backend(),
            hw: stage_hw,
        });
    }
}

/// Fused Filter/Project → Aggregate: build each morsel's partial state directly from the
/// linear chain's per-morsel output, without ever collecting the transformed relation.
/// The chain is numbered and metered exactly as the recursive `exec` would (so adaptive
/// metadata and the metric tree are unchanged), and the partials feed the same
/// `combine`/grace-spill path as the unfused aggregate. Caller has checked the input is a
/// fusable chain and no aggregate needs raw morsels for spill.
#[allow(clippy::too_many_arguments)]
fn exec_agg_fused(
    input: &RelOp,
    op_id: u32,
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    sources: &[Vec<RecordBatch>],
    opts: &ExecOptions,
    m: &mut ExecMetrics,
    ids: &mut IdGen,
) -> Result<Vec<RecordBatch>, InterpError> {
    // Number the fusable chain (Filter/Project) below the aggregate, then execute the
    // first non-fusable input — the exact pre-order numbering the recursive `exec` uses.
    let mut chain: Vec<(u32, &RelOp)> = vec![(ids.next(), input)];
    let mut base = fusable_input(input).expect("a fusable op has an input");
    while is_fusable(base) {
        chain.push((ids.next(), base));
        base = fusable_input(base).expect("a fusable op has an input");
    }
    let base_morsels = exec(base, sources, opts, m, ids)?;
    if base_morsels.is_empty() {
        return Err(InterpError::EmptyAggregateInput);
    }
    let base_rows = count_rows(&base_morsels);
    // The materialized input relation stays live through the whole fold (the `par_iter`
    // below only borrows it), so it is part of the aggregate's peak working set — exactly
    // as the unfused parallel path records via `push_breaker`. Recording only the tiny
    // grouped output (the former `push_metric`) reintroduced the peak under-count the
    // metrics contract exists to prevent: a 60M-row group-by over 4 groups reported ~0.
    let base_bytes = batch_bytes(&base_morsels);

    let stage_t0 = Stopwatch::start();
    // Compile each linear stage innermost→outermost against the schema it sees, then the
    // aggregate's expressions against the post-chain sample (mirrors the unfused compiles).
    let mut sample = base_morsels.first().cloned();
    let mut stages: Vec<FusedStage> = Vec::with_capacity(chain.len());
    for (id, op) in chain.iter().rev() {
        let stage = compile_stage(*id, op, sample.as_ref());
        if let Some(s) = &sample {
            sample = Some(stage.apply(s)?);
        }
        stages.push(stage);
    }
    let agg_sample = sample.as_ref().expect("non-empty base has a sample");
    let agg_jit = ops::compile_agg(group_keys, aggregates, agg_sample);

    // Per morsel: run the chain, then fold the result straight into a partial. Track the
    // row count after each stage so the fused ops keep exact selectivity metrics.
    //
    // A *sample* of the morsels goes first, and its chained batches are kept rather than
    // dropped. Fusing pre-aggregates every morsel, which is the wrong shape for exactly the
    // same reason it is wrong unfused: when the group-by does not reduce, the per-morsel
    // hash build is thrown away and the merge inherits the whole relation. The sample says
    // which regime this is, and keeping its chained batches means choosing "partition" costs
    // no re-run of the chain over the rows already filtered.
    let n = stages.len();
    let threads = rayon::current_num_threads().max(1);
    let sample_n = agg_par::sample_size(threads, base_morsels.len());
    let sample_chained: Vec<(RecordBatch, Vec<u64>)> = base_morsels[..sample_n]
        .par_iter()
        .map(|b| run_chain(&stages, b, opts))
        .collect::<Result<Vec<_>, InterpError>>()?;
    let sample_partials: Vec<agg::Partial> = sample_chained
        .par_iter()
        .map(|(b, _)| ops::eval_partial_jit(b, group_keys, aggregates, &agg_jit))
        .collect::<Result<Vec<_>, InterpError>>()?;
    let sample_rows: usize = sample_chained.iter().map(|(b, _)| b.num_rows()).sum();

    let mut totals = vec![0u64; n];
    let add_rows = |rows: &[u64], totals: &mut Vec<u64>| {
        for (i, r) in rows.iter().enumerate() {
            totals[i] += r;
        }
    };

    // The chain has only run over the sample, so the size of the post-chain relation is not
    // yet known and `base_rows` — the chain's *input* — stands in for it. It is exact when
    // nothing filters, and an over-estimate when something does; both only widen the
    // partitioning slightly, which the flat region of `agg_par::GROUPS_PER_PARTITION`
    // absorbs. Running the chain over everything first to get the true figure would cost the
    // whole materialization this branch may then decline.
    let partition_width = match agg_par::plain_key_columns(group_keys) {
        Some(keys) if base_morsels.len() >= 2 => {
            agg_par::width_from_sample(&sample_partials, sample_rows, sample_n, base_rows as usize)
                .map(|w| (keys, w))
        }
        _ => None,
    };
    if let Some((keys, width)) = partition_width {
        if let Admit::InMemory(_reservation) =
            admit(opts, op_id, agg_par::partition_footprint(base_bytes))
        {
            let mut chained: Vec<RecordBatch> = Vec::with_capacity(base_morsels.len());
            for (b, rows) in sample_chained {
                add_rows(&rows, &mut totals);
                chained.push(b);
            }
            let rest: Vec<(RecordBatch, Vec<u64>)> = base_morsels[sample_n..]
                .par_iter()
                .map(|b| run_chain(&stages, b, opts))
                .collect::<Result<Vec<_>, InterpError>>()?;
            for (b, rows) in rest {
                add_rows(&rows, &mut totals);
                chained.push(b);
            }
            emit_stage_metrics(m, &stages, &totals, base_rows, stage_t0);
            let agg_t0 = Stopwatch::start();
            let funcs = ops::agg_funcs(aggregates);
            let chained_bytes = batch_bytes(&chained);
            let out = agg_par::partitioned_aggregate(
                &chained, &keys, group_keys, aggregates, &agg_jit, &funcs, width,
            )?;
            push_breaker(
                m,
                op_id,
                "aggregate",
                *totals.last().unwrap_or(&base_rows),
                0,
                agg_par::partition_footprint(chained_bytes) as u64,
                &out,
                agg_t0,
                false,
                "par-agg-partitioned",
            );
            return Ok(out);
        }
    }

    // The merge below cannot measure its own output size; the sample already taken can.
    let est_groups =
        agg_par::groups_from_sample(&sample_partials, sample_rows, sample_n, base_rows as usize);

    // Reducing (or the pool declined the partitioned shape): fold every remaining morsel
    // straight into a partial, dropping its chained batch as the fusion always has. The
    // sample's partials are the first slice of that work and are reused, not recomputed.
    let mut partials = Vec::with_capacity(base_morsels.len());
    for ((_, rows), partial) in sample_chained.into_iter().zip(sample_partials) {
        add_rows(&rows, &mut totals);
        partials.push(partial);
    }
    let rest: Vec<(agg::Partial, Vec<u64>)> = base_morsels[sample_n..]
        .par_iter()
        .map(|b| {
            let (cur, stage_rows) = run_chain(&stages, b, opts)?;
            let partial = ops::eval_partial_jit(&cur, group_keys, aggregates, &agg_jit)?;
            Ok((partial, stage_rows))
        })
        .collect::<Result<Vec<_>, InterpError>>()?;
    for (partial, rows) in rest {
        add_rows(&rows, &mut totals);
        partials.push(partial);
    }
    emit_stage_metrics(m, &stages, &totals, base_rows, stage_t0);

    // Combine the partials — the same in-memory / grace-spill path as the unfused
    // aggregate (no value-list aggregate reaches here, so no raw-morsel spill is needed).
    let agg_t0 = Stopwatch::start();
    let funcs = ops::agg_funcs(aggregates);
    let rows_in = *totals.last().unwrap_or(&base_rows);
    let state_bytes = partial_state_bytes(&partials);
    let mut spilled = false;
    let mut spill_vol = 0u64;
    let decision = if rows_in > 0 {
        admit(opts, op_id, state_bytes)
    } else {
        Admit::InMemory(None)
    };
    let (group_columns, agg_cols) = match decision {
        Admit::Spill => {
            let global = opts.agg_spill.as_ref().expect("spill implies an envelope");
            let sp =
                &global.with_budget(opts.op_budget(op_id).unwrap_or(global.memory_budget_bytes));
            spilled = true;
            let p = grace_partitions(&partials, sp.memory_budget_bytes);
            let mut store =
                DiskSpillStore::with_codec(sp.dir.join(format!("agg-{p}p")), p, sp.codec)?;
            let res =
                combine_finalize_spilling(partials, &funcs, &mut store, sp.memory_budget_bytes)?;
            spill_vol = store.spilled_bytes(); // measured volume routed to disk
            warn_if_skewed(op_id, "aggregate", &store);
            (res.group_columns, res.agg_columns)
        }
        Admit::InMemory(_reservation) => {
            let merged = agg::combine_sized(
                &partials,
                &funcs,
                opts.tuning.radix_parallel_threshold,
                est_groups,
            )?;
            let agg_cols = agg::finalize(&funcs, &merged)?;
            (merged.group_columns, agg_cols)
        }
    };
    let out = vec![ops::build_agg_batch(
        group_keys,
        aggregates,
        &group_columns,
        &agg_cols,
    )?];
    // A spilled fold holds only ~the budget resident, not the whole input, so the input
    // relation no longer bounds its peak; cap it at the operator's resolved budget so
    // Carbonite doesn't learn that a bounded-memory spill "needed" the full input.
    let peak_in = if spilled {
        base_bytes.min(
            opts.op_budget(op_id)
                .map(|b| b as u64)
                .unwrap_or(base_bytes),
        )
    } else {
        base_bytes
    };
    push_breaker_spilled(
        m,
        op_id,
        "aggregate",
        rows_in,
        0,
        peak_in,
        &out,
        agg_t0,
        spilled,
        spill_vol,
        "interp",
    );
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn push_metric(
    m: &mut ExecMetrics,
    op_id: u32,
    kind: &'static str,
    rows_in: u64,
    out: &[RecordBatch],
    t0: Stopwatch,
    spilled: bool,
    backend: &'static str,
) {
    let bytes = batch_bytes(out);
    let (cpu_ns, peak_rss_bytes, hw) = t0.measure();
    m.record(OpMetric {
        op_id,
        kind,
        rows_in,
        rows_build: 0,
        rows_out: count_rows(out),
        elapsed_ns: t0.elapsed_ns(),
        wall_span_ns: 0,
        cpu_ns,
        threads: rayon::current_num_threads().max(1) as u32,
        peak_bytes: bytes,
        result_bytes: bytes,
        spilled,
        spill_bytes: 0,
        peak_rss_bytes,
        backend,
        hw,
    });
}

/// Record a **pipeline breaker**: it materializes `in_bytes` of input and builds its
/// result at the same time, so both are live at its peak. `rows_build` is a join's
/// build-side rows (0 elsewhere). Streaming operators use `push_metric`, whose peak is
/// its result alone.
#[allow(clippy::too_many_arguments)]
fn push_breaker(
    m: &mut ExecMetrics,
    op_id: u32,
    kind: &'static str,
    rows_in: u64,
    rows_build: u64,
    in_bytes: u64,
    out: &[RecordBatch],
    t0: Stopwatch,
    spilled: bool,
    backend: &'static str,
) {
    push_breaker_spilled(
        m, op_id, kind, rows_in, rows_build, in_bytes, out, t0, spilled, 0, backend,
    );
}

/// Above this spill skew (largest partition / mean), one partition dominates badly enough
/// that the grace merge thrashes on it — worth telling the operator so they can salt the key.
const SPILL_SKEW_WARN: f32 = 3.0;

/// Emit an operator-visible diagnostic when a spill's partitions are badly imbalanced (a hot
/// key piled one partition). Surfaced through the tracing bridge; salting the shuffle key or
/// raising the partition count is the fix. Pure side effect — never changes a result.
fn warn_if_skewed(op_id: u32, kind: &str, store: &DiskSpillStore) {
    let skew = store.spill_skew();
    if skew > SPILL_SKEW_WARN {
        tracing::warn!(
            target: "batcher.engine",
            op_id,
            kind,
            skew,
            "skewed spill: one partition dominates — salt the key or raise the partition count"
        );
    }
}

/// A breaker that also reports its measured **spill volume** (`spill_bytes`). The spill
/// sites that own their `SpillStore` call this with `store.spilled_bytes()`; every other
/// breaker uses [`push_breaker`], which passes `0` (did not spill / unmeasured).
#[allow(clippy::too_many_arguments)]
fn push_breaker_spilled(
    m: &mut ExecMetrics,
    op_id: u32,
    kind: &'static str,
    rows_in: u64,
    rows_build: u64,
    in_bytes: u64,
    out: &[RecordBatch],
    t0: Stopwatch,
    spilled: bool,
    spill_bytes: u64,
    backend: &'static str,
) {
    let result_bytes = batch_bytes(out);
    let (cpu_ns, peak_rss_bytes, hw) = t0.measure();
    m.record(OpMetric {
        op_id,
        kind,
        rows_in,
        rows_build,
        rows_out: count_rows(out),
        elapsed_ns: t0.elapsed_ns(),
        wall_span_ns: 0,
        cpu_ns,
        threads: rayon::current_num_threads().max(1) as u32,
        peak_bytes: in_bytes.saturating_add(result_bytes),
        result_bytes,
        spilled,
        spill_bytes,
        peak_rss_bytes,
        backend,
        hw,
    });
}

/// Estimated bytes of the per-morsel partial aggregate state — the memory the
/// in-place `combine` would have to hold at once. Used to decide whether to spill.
fn partial_state_bytes(partials: &[agg::Partial]) -> usize {
    partials
        .iter()
        .map(|p| {
            let groups: usize = p
                .group_columns
                .iter()
                .map(|c| c.get_array_memory_size())
                .sum();
            let states: usize = p
                .states
                .iter()
                .flat_map(|s| s.iter())
                .map(|c| c.get_array_memory_size())
                .sum();
            groups + states
        })
        .sum()
}

/// Grace fan-out: enough hash partitions that each holds roughly one budget's
/// worth of state. At least 2 (spilling with 1 partition saves no memory).
fn grace_partitions(partials: &[agg::Partial], budget_bytes: usize) -> usize {
    // Capped like every other grace operator's fan-out. Uncapped, an aggregate whose state
    // is three orders of magnitude over the envelope asked for thousands of spill files,
    // each receiving shards too small to write efficiently. A bucket that is still too large
    // is not left that way: `combine_finalize_spilling` measures it before reading it and
    // re-partitions it out of core, which is the bounded path anyway.
    crate::spill_split::grace_bucket_count(partial_state_bytes(partials), budget_bytes)
}

/// Grace hash join: partition both sides by join key into `P` disk-backed
/// buckets, then join one bucket at a time so only a single partition's build
/// table is resident. Equal keys share a bucket, so the union of per-bucket joins
/// is the full join (every type — unmatched-right tracking is per bucket, which is
/// correct because a right row's matches all live in its bucket). Result is the
/// same multiset the in-memory path produces; only peak memory differs.
#[allow(clippy::too_many_arguments)]
/// Parallel distinct: partial-dedup each morsel, then combine.
/// Parallel DISTINCT (also the dedup half of UNION): partial-dedup each morsel,
/// then combine. DISTINCT is an all-columns group-by with no aggregates, so it
/// spills through the *same* grace path as aggregation when the partial state
/// exceeds the memory envelope — high-cardinality DISTINCT/UNION stays bounded
/// instead of OOMing. Returns the deduplicated batch and whether it spilled.
/// Returns `(deduped batch, spilled, spill_bytes)` — the measured spill volume is `0`
/// unless the grace path engaged.
fn distinct(
    parts: &[RecordBatch],
    opts: &ExecOptions,
    op_id: u32,
) -> Result<(Vec<RecordBatch>, bool, u64), InterpError> {
    if parts.is_empty() {
        return Err(InterpError::EmptyAggregateInput);
    }
    let schema = parts[0].schema();

    // Single-pass fast path for a HIGH-cardinality, in-memory DISTINCT: hash each row once
    // (partition by all columns, dedup per bucket) instead of the per-morsel `partial` +
    // `combine` double-hash. Gated so it never regresses the other cases:
    //   * high cardinality only — probed on the first morsel's distinct ratio, which is
    //     exactly what `partial` buys: below a 2x local collapse the per-morsel pass is a
    //     whole extra hash of every row for nothing, and above it `combine` is then trivial.
    //     Nothing here needs, or infers, the GLOBAL key cardinality.
    //   * in-memory — a spilling DISTINCT still streams through the grace `combine`.
    // A single dense integer column needs neither hash nor gather: `DISTINCT` over it is a
    // presence bitmap indexed by `value - min`, two linear passes and no partial state.
    // Declines (returning `None`) for anything wider, nullable, non-integer, or sparse.
    if let Some(out) = agg::distinct_dense(parts)? {
        return Ok((vec![out], false, 0));
    }

    let probe = ops::distinct_partial(&parts[0])?;
    let sample_rows = parts[0].num_rows();
    let sample_distinct = probe.group_columns.first().map_or(0, |a| a.len());
    let high_card = sample_rows > 0 && (sample_distinct as f64) >= 0.5 * sample_rows as f64;
    if high_card {
        let bytes = batch_bytes(parts) as usize;
        if matches!(admit(opts, op_id, bytes), Admit::InMemory(_)) {
            let p = rayon::current_num_threads().max(1);
            let out = agg::distinct_parts(parts, p)?;
            return Ok((out, false, 0));
        }
    }

    let partials: Vec<agg::Partial> = parts
        .par_iter()
        .map(ops::distinct_partial)
        .collect::<Result<_, InterpError>>()?;
    let state_bytes = partial_state_bytes(&partials);
    let (group_columns, spilled, spill_vol) = match admit(opts, op_id, state_bytes) {
        Admit::Spill => {
            let global = opts.agg_spill.as_ref().expect("spill implies an envelope");
            let budget = opts.op_budget(op_id).unwrap_or(global.memory_budget_bytes);
            let sp = &global.with_budget(budget);
            let p = grace_partitions(&partials, sp.memory_budget_bytes);
            let dir = sp.dir.join(format!("distinct-{p}p"));
            let mut store = DiskSpillStore::with_codec(dir, p, sp.codec)?;
            // No aggregates: `&[]` makes this a pure dedup over the group columns.
            let res = combine_finalize_spilling(partials, &[], &mut store, sp.memory_budget_bytes)?;
            warn_if_skewed(op_id, "distinct", &store);
            (res.group_columns, true, store.spilled_bytes())
        }
        Admit::InMemory(_reservation) => (
            agg::combine_with(&partials, &[], opts.tuning.radix_parallel_threshold)?.group_columns,
            false,
            0,
        ),
    };
    Ok((
        vec![RecordBatch::try_new(schema, group_columns)?],
        spilled,
        spill_vol,
    ))
}

/// Parallel `DISTINCT ON`: keep one whole row per dedup key, across cores.
///
/// Scheduling only — the reduction is `bc_runtime::agg::distinct_on`, the same function the
/// sequential oracle calls and the same one the distributed map and reduce sides call. What is
/// decided here is where it runs: over key-disjoint partitions in memory, or bucket by bucket
/// through disk when the input does not fit the operator's envelope.
///
/// Returns `(reduced batches, spilled, spill_bytes)`, the batches narrowed back to the input's
/// columns (an ordering the plan computed is compared on and then dropped).
fn distinct_on(
    parts: &[RecordBatch],
    keys: &[String],
    order: &[bc_ir::SortKey],
    opts: &ExecOptions,
    op_id: u32,
) -> Result<(Vec<RecordBatch>, bool, u64), InterpError> {
    let Some(first) = parts.first() else {
        return Err(InterpError::EmptyAggregateInput);
    };
    let bytes = batch_bytes(parts) as usize;
    if let Admit::InMemory(_reservation) = admit(opts, op_id, bytes) {
        return Ok((ops::parallel_distinct_on(parts, keys, order)?, false, 0));
    }
    // Out of core. Widen here rather than inside the spill path so the ordering columns are
    // already in the batches that go to disk, and a bucket read back needs no re-evaluation.
    let ncols = first.num_columns();
    let wide: Vec<RecordBatch> = parts
        .par_iter()
        .map(|b| ops::distinct_on_widen(b, keys, order).map(|(b, _, _)| b))
        .collect::<Result<_, InterpError>>()?;
    let (_, key_idx, ord) = ops::distinct_on_widen(first, keys, order)?;
    let global = opts.agg_spill.as_ref().expect("spill implies an envelope");
    let budget = opts.op_budget(op_id).unwrap_or(global.memory_budget_bytes);
    let (out, spill_vol) = crate::distinct_on_spill::distinct_on_spilling(
        &wide,
        &key_idx,
        &ord,
        budget,
        &global.dir,
        global.codec,
    )?;
    if wide[0].num_columns() == ncols {
        return Ok((out, true, spill_vol));
    }
    let keep: Vec<usize> = (0..ncols).collect();
    let narrowed = out
        .iter()
        .map(|b| b.project(&keep))
        .collect::<Result<_, _>>()?;
    Ok((narrowed, true, spill_vol))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execute;
    use crate::join_par::SKEW_MIN_BUCKET_ROWS;
    use arrow::array::{Array, ArrayRef, Int64Array, StringArray};
    use std::sync::Arc;

    /// One join fixture as `(left keys, left vals, right keys, right vals)`, nulls included.
    type JoinCase = (
        Vec<Option<i64>>,
        Vec<Option<i64>>,
        Vec<Option<i64>>,
        Vec<Option<i64>>,
    );

    fn batch(keys: &[i64], vals: &[i64]) -> RecordBatch {
        RecordBatch::try_from_iter(vec![
            ("k", Arc::new(Int64Array::from(keys.to_vec())) as ArrayRef),
            ("v", Arc::new(Int64Array::from(vals.to_vec())) as ArrayRef),
        ])
        .unwrap()
    }

    /// A plan with no media decode — the common case for the `auto_width` cap tests.
    fn no_media_plan() -> RelOp {
        RelOp::Scan { source_id: 0 }
    }

    /// A join whose planner-nominated build side turns out to be the *larger* relation is
    /// re-oriented at execution, and the re-oriented join is the same relation.
    ///
    /// Both halves matter. The multiset equality against the sequential oracle is the
    /// correctness proof: the oracle never swaps (`lib.rs` ignores the orientation
    /// entirely), so agreeing with it is exactly the statement that a swap changes nothing
    /// observable but the row order. The `rows_build` assertion is what stops the test from
    /// being true by construction — without it a rule that never fired would pass, and
    /// `rows_build` is reported *after* the correction, so it names the table the join
    /// actually built. 40 k build against 5 k probe clears both the 2× ratio and the
    /// one-morsel floor; after the swap the engine must report having built the 5 k side.
    #[test]
    fn oversized_build_side_is_re_oriented_and_still_matches_the_oracle() {
        let probe_keys: Vec<i64> = (0..5_000).collect();
        let probe_vals: Vec<i64> = (0..5_000).map(|k| k * 10).collect();
        let build_keys: Vec<i64> = (0..40_000).collect();
        let build_vals: Vec<i64> = (0..40_000).map(|k| k * 100).collect();

        let plan = RelOp::HashJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(RelOp::Scan { source_id: 1 }),
            left_keys: vec!["k".into()],
            right_keys: vec!["k".into()],
            join_type: bc_ir::JoinType::Inner,
            output: vec![
                bc_ir::JoinOutputCol {
                    side: bc_ir::JoinSide::Left,
                    name: "v".into(),
                    alias: "lv".into(),
                },
                bc_ir::JoinOutputCol {
                    side: bc_ir::JoinSide::Right,
                    name: "v".into(),
                    alias: "rv".into(),
                },
            ],
            strategy: bc_ir::JoinStrategy::Hash,
        };
        let sources = vec![
            vec![batch(&probe_keys, &probe_vals)],
            vec![batch(&build_keys, &build_vals)],
        ];

        // Sorted so the comparison is a multiset one: the swap deliberately changes the
        // emitted order (output follows the probe), which is the one thing it may change.
        let rows = |bs: &[RecordBatch]| -> Vec<(i64, i64)> {
            let mut out = Vec::new();
            for b in bs {
                let lv = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
                let rv = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
                for i in 0..b.num_rows() {
                    out.push((lv.value(i), rv.value(i)));
                }
            }
            out.sort();
            out
        };

        let (par, metrics) =
            execute_parallel_with_metrics(&plan, &sources, &ExecOptions::default()).unwrap();
        let oracle = execute(&plan, &sources).unwrap();

        // Column identity survives the re-label: `lv` still comes from the left input.
        assert_eq!(rows(&par), rows(&oracle));
        assert_eq!(rows(&par).len(), 5_000);
        assert!(rows(&par).iter().all(|(lv, rv)| *rv == *lv * 10));

        let join = metrics
            .ops
            .iter()
            .find(|o| o.kind == "hash_join")
            .expect("the plan has a hash join");
        assert_eq!(
            join.rows_build, 5_000,
            "the executor built the planner's oversized side instead of correcting it"
        );
        assert_eq!(join.rows_in, 40_000, "the larger side should be the probe");
    }

    /// The fused join→aggregate must produce exactly what the sequential oracle produces —
    /// over unique keys, a 1:N build (duplicate keys), null keys (never match), and an all-miss
    /// join (empty output → the degenerate aggregate) — and must actually *fire* (not silently
    /// fall back), else the test asserts nothing.
    #[test]
    fn fused_join_aggregate_matches_the_oracle() {
        use bc_expr::Expr;
        fn b2(c0: &str, v0: Vec<Option<i64>>, c1: &str, v1: Vec<Option<i64>>) -> RecordBatch {
            RecordBatch::try_from_iter(vec![
                (c0, Arc::new(Int64Array::from(v0)) as ArrayRef),
                (c1, Arc::new(Int64Array::from(v1)) as ArrayRef),
            ])
            .unwrap()
        }
        // Aggregate(group by g, SUM(v)) over HashJoin(probe(pk,v) ⋈ build(bk,g)).
        let plan = RelOp::Aggregate {
            input: Box::new(RelOp::HashJoin {
                left: Box::new(RelOp::Scan { source_id: 0 }),
                right: Box::new(RelOp::Scan { source_id: 1 }),
                left_keys: vec!["pk".into()],
                right_keys: vec!["bk".into()],
                join_type: bc_ir::JoinType::Inner,
                output: vec![
                    bc_ir::JoinOutputCol {
                        side: bc_ir::JoinSide::Left,
                        name: "v".into(),
                        alias: "v".into(),
                    },
                    bc_ir::JoinOutputCol {
                        side: bc_ir::JoinSide::Right,
                        name: "g".into(),
                        alias: "g".into(),
                    },
                ],
                strategy: bc_ir::JoinStrategy::Hash,
            }),
            group_keys: vec![ProjectionItem {
                expr: Expr::Col { name: "g".into() },
                alias: "g".into(),
            }],
            aggregates: vec![AggregateItem {
                func: AggFunc::Sum,
                input: Some(Expr::Col { name: "v".into() }),
                input2: None,
                alias: "s".into(),
                param: None,
            }],
        };
        let norm = |bs: &[RecordBatch]| -> Vec<(Option<i64>, Option<i64>)> {
            let mut rows = Vec::new();
            for b in bs {
                let g = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
                let s = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
                for i in 0..b.num_rows() {
                    rows.push((
                        (!g.is_null(i)).then(|| g.value(i)),
                        (!s.is_null(i)).then(|| s.value(i)),
                    ));
                }
            }
            rows.sort();
            rows
        };
        let cases: Vec<JoinCase> = vec![
            (
                (0..200).map(|i| Some(i % 10)).collect(),
                (0..200).map(Some).collect(),
                (0..10).map(Some).collect(),
                (0..10).map(|i| Some(i % 3)).collect(),
            ),
            (
                vec![Some(1), Some(2), Some(1), Some(2)],
                vec![Some(10), Some(20), Some(30), Some(40)],
                vec![Some(1), Some(1), Some(2)], // 1:N — key 1 appears twice on the build
                vec![Some(5), Some(6), Some(7)],
            ),
            (
                vec![None, Some(1), Some(2), None],
                vec![Some(10), Some(11), Some(12), Some(13)],
                vec![None, Some(1)], // a null build key never matches
                vec![Some(0), Some(1)],
            ),
            (
                vec![Some(99), Some(98)],
                vec![Some(1), Some(2)],
                vec![Some(1), Some(2)],
                vec![Some(0), Some(1)], // all miss → empty join → degenerate aggregate
            ),
        ];
        let mut fired = false;
        for (pk, v, bk, g) in cases {
            let sources = vec![vec![b2("pk", pk, "v", v)], vec![b2("bk", bk, "g", g)]];
            let opts = ExecOptions {
                fuse_linear: true,
                morsel_rows: 8,
                ..ExecOptions::default()
            };
            let (fused, mt) = execute_parallel_with_metrics(&plan, &sources, &opts).unwrap();
            let oracle = execute(&plan, &sources).unwrap();
            assert_eq!(
                norm(&fused),
                norm(&oracle),
                "fused join-agg diverged from the oracle"
            );
            fired |= mt.ops.iter().any(|o| o.backend == "fused-join-agg");
        }
        assert!(
            fired,
            "the fused join-aggregate path never fired — the test proved nothing"
        );
    }

    /// A byte-heavy source (few rows, large blobs) must lift the worker cap: counting
    /// morsels by rows alone would collapse it to one worker even though the scan splits
    /// it into many byte-bounded morsels. Regression for the media/embedding throttle.
    #[test]
    fn byte_heavy_source_lifts_worker_cap() {
        use arrow::array::BinaryArray;

        // 100 rows × 64 KiB ≈ 6.4 MiB of binary — far under the 16,384-row target but well
        // over the 1 MiB byte budget, so the scan yields several morsels, not one.
        let blob = vec![0u8; 64 * 1024];
        let arr = BinaryArray::from_iter_values((0..100).map(|_| blob.as_slice()));
        let rb = RecordBatch::try_from_iter(vec![("b", Arc::new(arr) as ArrayRef)]).unwrap();
        let sources = vec![vec![rb]];
        let opts = ExecOptions::default();

        // Row-only counting would have been 100 / 16,384 = 1 worker.
        assert_eq!(100usize.div_ceil(opts.morsel_target().rows), 1);
        // Byte-aware counting sees the ~6 morsels the byte budget produces.
        let workers = max_useful_workers(&opts, &sources);
        assert!(
            workers >= 6,
            "byte-heavy source should allow >=6 workers, got {workers}"
        );
    }

    /// The fused Sort-top-N-over-inner-hash-join path must number the join's children with
    /// the SAME pre-order op-ids a plain walk assigns, so the runtime feedback keyed by op-id
    /// still lines up with the operator the control plane annotated. The join is fused (folded
    /// into the sort), so it emits no `hash_join` metric and its own id (1) is unused — but its
    /// left/right subtrees must keep ids 2 and 3, not slide down to 1 and 2. Before the fix the
    /// children were executed with the live `IdGen`, skipping the join's id, so every descendant
    /// op-id was shifted by one (and on decline the subtree was numbered twice).
    #[test]
    fn fused_join_top_n_keeps_child_op_ids_aligned() {
        use bc_expr::Expr;
        use bc_ir::{JoinOutputCol, JoinSide, JoinStrategy, JoinType, SortKey};

        // Sort{limit=2}( HashJoin.Inner.Hash( Scan0, Scan1 ) ), ORDER BY the left key.
        let plan = RelOp::Sort {
            input: Box::new(RelOp::HashJoin {
                left: Box::new(RelOp::Scan { source_id: 0 }),
                right: Box::new(RelOp::Scan { source_id: 1 }),
                left_keys: vec!["k".into()],
                right_keys: vec!["k".into()],
                join_type: JoinType::Inner,
                output: vec![
                    JoinOutputCol {
                        side: JoinSide::Left,
                        name: "v".into(),
                        alias: "lv".into(),
                    },
                    JoinOutputCol {
                        side: JoinSide::Right,
                        name: "v".into(),
                        alias: "rv".into(),
                    },
                ],
                strategy: JoinStrategy::Hash,
            }),
            keys: vec![SortKey {
                expr: Expr::Col { name: "lv".into() },
                descending: false,
                nulls_first: false,
            }],
            limit: Some(2),
        };
        let sources = vec![
            vec![batch(&[1, 2, 3, 4], &[10, 20, 30, 40])],
            vec![batch(&[1, 2, 3, 4], &[100, 200, 300, 400])],
        ];
        let (_out, m) = execute_parallel_with_metrics(&plan, &sources, &ExecOptions::default())
            .expect("fused join top-n runs");

        // The fused sort is op 0; the join (op 1) is folded in and emits no metric; the two
        // scans are the join's children and MUST carry the pre-order ids 2 and 3.
        let sort = m
            .ops
            .iter()
            .find(|o| o.kind == "sort")
            .expect("a sort metric");
        assert_eq!(sort.op_id, 0);
        assert_eq!(
            sort.backend, "interp-jointopn",
            "the fused path must engage"
        );
        assert!(
            m.ops.iter().all(|o| o.kind != "hash_join"),
            "the join is fused, so it records no separate metric"
        );
        let mut scan_ids: Vec<u32> = m
            .ops
            .iter()
            .filter(|o| o.kind == "scan")
            .map(|o| o.op_id)
            .collect();
        scan_ids.sort_unstable();
        assert_eq!(
            scan_ids,
            vec![2, 3],
            "the join's children must keep the pre-order ids a plain walk assigns"
        );
    }

    /// Peak working-set measurement contract for the *parallel* breakers (the metrics
    /// contract integration test covers only the sequential oracle). These guard the
    /// under-counts Carbonite's per-family memory model would otherwise learn.
    mod peak_bytes_contract {
        use super::*;
        use bc_expr::{BinaryOp, Expr, Literal};
        use bc_ir::{AggFunc, AggregateItem, ProjectionItem};

        fn peak(m: &ExecMetrics, kind: &str) -> (u64, u64) {
            let op = m
                .ops
                .iter()
                .find(|o| o.kind == kind)
                .unwrap_or_else(|| panic!("no {kind} metric in {:?}", m.ops));
            (op.peak_bytes, op.result_bytes)
        }

        /// A fused Filter→Aggregate must report the input it holds, not its tiny grouped
        /// output. This is the exact 60M-rows/4-groups under-count the contract exists to
        /// stop — the fused path used to record `push_metric` (peak == output).
        #[test]
        fn fused_aggregate_reports_input_peak_not_tiny_output() {
            let plan = RelOp::Aggregate {
                input: Box::new(RelOp::Filter {
                    input: Box::new(RelOp::Scan { source_id: 0 }),
                    predicate: Expr::Binary {
                        op: BinaryOp::Ge,
                        left: Box::new(Expr::Col { name: "v".into() }),
                        right: Box::new(Expr::Lit {
                            value: Literal::Int(0),
                        }),
                    },
                }),
                group_keys: vec![ProjectionItem {
                    expr: Expr::Col { name: "k".into() },
                    alias: "k".into(),
                }],
                aggregates: vec![AggregateItem {
                    func: AggFunc::Sum,
                    input: Some(Expr::Col { name: "v".into() }),
                    input2: None,
                    alias: "s".into(),
                    param: None,
                }],
            };
            // 4 groups; the filter keeps every row so the fold materializes the whole input.
            let keys: Vec<i64> = (0..4_000).map(|i| i % 4).collect();
            let vals: Vec<i64> = (0..4_000).collect();
            let opts = ExecOptions {
                morsel_rows: 128,
                fuse_linear: true,
                ..ExecOptions::default()
            };
            let (out, m) =
                execute_parallel_with_metrics(&plan, &[vec![batch(&keys, &vals)]], &opts).unwrap();
            assert_eq!(out.iter().map(|b| b.num_rows()).sum::<usize>(), 4);
            let (peak, result) = peak(&m, "aggregate");
            assert!(
                peak >= 4_000 * 16,
                "fused aggregate peak {peak} must account for the 4000-row (16 B/row) input"
            );
            assert!(
                result * 50 < peak,
                "grouped output {result} must be a tiny fraction of peak {peak}"
            );
        }

        /// A hash join's peak must include the build-side hash table / chain / null mask —
        /// the join's largest allocation, invisible to `get_array_memory_size`.
        #[test]
        fn hash_join_peak_includes_build_structure() {
            let plan = RelOp::HashJoin {
                left: Box::new(RelOp::Scan { source_id: 0 }),
                right: Box::new(RelOp::Scan { source_id: 1 }),
                left_keys: vec!["k".into()],
                right_keys: vec!["k".into()],
                join_type: bc_ir::JoinType::Inner,
                output: vec![
                    bc_ir::JoinOutputCol {
                        side: bc_ir::JoinSide::Left,
                        name: "v".into(),
                        alias: "lv".into(),
                    },
                    bc_ir::JoinOutputCol {
                        side: bc_ir::JoinSide::Right,
                        name: "v".into(),
                        alias: "rv".into(),
                    },
                ],
                strategy: bc_ir::JoinStrategy::Hash,
            };
            let probe: Vec<i64> = (0..2_000).map(|i| i % 500).collect();
            let build: Vec<i64> = (0..500).collect();
            let (_out, m) = execute_parallel_with_metrics(
                &plan,
                &[vec![batch(&probe, &probe)], vec![batch(&build, &build)]],
                &ExecOptions::default(),
            )
            .unwrap();
            let (peak, _result) = peak(&m, "hash_join");
            let raw_inputs = (2_000 + 500) * 16;
            assert!(
                peak > raw_inputs,
                "join peak {peak} must exceed raw input bytes {raw_inputs} by the hash structure"
            );
        }

        /// A spilling aggregate reports its measured spill *volume*, not just a bool — the
        /// magnitude Carbonite sizes spill scratch and disk bandwidth from.
        #[test]
        fn a_spilling_aggregate_reports_its_spill_volume() {
            let plan = RelOp::Aggregate {
                input: Box::new(RelOp::Scan { source_id: 0 }),
                group_keys: vec![ProjectionItem {
                    expr: Expr::Col { name: "k".into() },
                    alias: "k".into(),
                }],
                aggregates: vec![AggregateItem {
                    func: AggFunc::Sum,
                    input: Some(Expr::Col { name: "v".into() }),
                    input2: None,
                    alias: "s".into(),
                    param: None,
                }],
            };
            // 20k distinct keys → a large hash state; a 1 KiB budget forces the grace spill.
            let keys: Vec<i64> = (0..20_000).collect();
            let vals: Vec<i64> = (0..20_000).collect();
            let opts = ExecOptions {
                morsel_rows: 512,
                agg_spill: Some(super::SpillOptions {
                    memory_budget_bytes: 1024,
                    dir: std::env::temp_dir(),
                    codec: super::SpillCodec::None,
                }),
                ..ExecOptions::default()
            };
            let (out, m) =
                execute_parallel_with_metrics(&plan, &[vec![batch(&keys, &vals)]], &opts).unwrap();
            assert_eq!(out.iter().map(|b| b.num_rows()).sum::<usize>(), 20_000);
            let agg = m
                .ops
                .iter()
                .find(|o| o.kind == "aggregate")
                .expect("aggregate metric");
            assert!(agg.spilled, "a 1 KiB budget over 20k groups must spill");
            assert!(
                agg.spill_bytes > 0,
                "a spilling aggregate must report a measured spill volume, got {}",
                agg.spill_bytes
            );
        }

        /// A spilling sort reports its measured pass-0 spill volume, not just `spilled`.
        #[test]
        fn a_spilling_sort_reports_its_spill_volume() {
            let plan = RelOp::Sort {
                input: Box::new(RelOp::Scan { source_id: 0 }),
                keys: vec![bc_ir::SortKey {
                    expr: Expr::Col { name: "v".into() },
                    descending: false,
                    nulls_first: false,
                }],
                limit: None,
            };
            let vals: Vec<i64> = (0..20_000).rev().collect();
            let opts = ExecOptions {
                morsel_rows: 512,
                agg_spill: Some(super::SpillOptions {
                    memory_budget_bytes: 1024,
                    dir: std::env::temp_dir(),
                    codec: super::SpillCodec::None,
                }),
                ..ExecOptions::default()
            };
            let (_out, m) =
                execute_parallel_with_metrics(&plan, &[vec![batch(&vals, &vals)]], &opts).unwrap();
            let sort = m
                .ops
                .iter()
                .find(|o| o.kind == "sort")
                .expect("sort metric");
            assert!(sort.spilled, "a 1 KiB budget over 20k rows must spill");
            assert!(
                sort.spill_bytes > 0,
                "a spilling sort must report a measured spill volume, got {}",
                sort.spill_bytes
            );
        }

        /// A spilling window reports its measured spill volume too (all spilling operators —
        /// aggregate, distinct, union, hash_join, sort, window — now do).
        #[test]
        fn a_spilling_window_reports_its_spill_volume() {
            use bc_ir::{SortKey, WindowFn, WindowFunc};

            let plan = RelOp::Window {
                input: Box::new(RelOp::Scan { source_id: 0 }),
                partition_keys: vec![Expr::Col { name: "k".into() }],
                order_keys: vec![SortKey {
                    expr: Expr::Col { name: "v".into() },
                    descending: false,
                    nulls_first: false,
                }],
                functions: vec![WindowFunc {
                    func: WindowFn::Sum,
                    input: Some(Expr::Col { name: "v".into() }),
                    offset: 1,
                    frame: None,
                    alpha: None,
                    half_life: None,
                    alias: "s".into(),
                }],
                rank_limit: None,
            };
            let keys: Vec<i64> = (0..8_000).map(|i| i % 50).collect();
            let vals: Vec<i64> = (0..8_000).collect();
            let opts = ExecOptions {
                agg_spill: Some(super::SpillOptions {
                    memory_budget_bytes: 1024,
                    dir: std::env::temp_dir(),
                    codec: super::SpillCodec::None,
                }),
                ..ExecOptions::default()
            };
            let (_out, m) =
                execute_parallel_with_metrics(&plan, &[vec![batch(&keys, &vals)]], &opts).unwrap();
            let w = m
                .ops
                .iter()
                .find(|o| o.kind == "window")
                .expect("window metric");
            assert!(
                w.spilled,
                "a 1 KiB budget must force the window grace spill"
            );
            assert!(
                w.spill_bytes > 0,
                "a spilling window must report a measured spill volume, got {}",
                w.spill_bytes
            );
        }

        /// A full in-memory sort allocates a permutation index on top of its materialized
        /// input, so its peak must exceed the sorted result it emits (equal row count).
        #[test]
        fn in_memory_sort_peak_exceeds_result_by_scratch() {
            let plan = RelOp::Sort {
                input: Box::new(RelOp::Scan { source_id: 0 }),
                keys: vec![bc_ir::SortKey {
                    expr: Expr::Col { name: "v".into() },
                    descending: false,
                    nulls_first: false,
                }],
                limit: None,
            };
            let vals: Vec<i64> = (0..8_000).rev().collect();
            let (_out, m) = execute_parallel_with_metrics(
                &plan,
                &[vec![batch(&vals, &vals)]],
                &ExecOptions::default(),
            )
            .unwrap();
            let (peak, result) = peak(&m, "sort");
            assert!(
                peak > result,
                "sort peak {peak} must exceed its result {result} by the permutation scratch"
            );
        }
    }

    /// An explicit `parallelism` is the control plane's decision and is never overridden —
    /// the hash-shuffle bucket count keys off exactly this width.
    #[test]
    fn explicit_parallelism_is_honored_verbatim() {
        let opts = ExecOptions {
            parallelism: 7,
            ..ExecOptions::default()
        };
        // One tiny source: the morsel cap would say 1, but the explicit width wins.
        assert_eq!(
            auto_width(&opts, &[vec![batch(&[1], &[1])]], &no_media_plan()),
            7
        );
    }

    /// A worker with no morsel to take still costs a wake-up and queue contention, so the
    /// automatic width never exceeds the morsels the inputs can yield. A one-row query
    /// must not install a 96-thread pool.
    #[test]
    fn auto_width_never_exceeds_available_morsels() {
        let opts = ExecOptions {
            parallelism: 0,
            morsel_rows: 2,
            ..ExecOptions::default()
        };
        // 1 row → 1 morsel → 1 worker.
        assert_eq!(
            auto_width(&opts, &[vec![batch(&[1], &[1])]], &no_media_plan()),
            1
        );
        // 5 rows at 2 rows/morsel → 3 morsels; capped further by the machine's cores.
        let cores = std::thread::available_parallelism().map_or(1, |v| v.get());
        let five = vec![batch(&[1, 2, 3, 4, 5], &[1, 2, 3, 4, 5])];
        assert_eq!(auto_width(&opts, &[five], &no_media_plan()), 3.min(cores));
    }

    /// The bound is the *widest* source, not their sum: an operator fans out over one
    /// input's morsels at a time, so a large probe side must not be throttled by a tiny
    /// build side.
    #[test]
    fn auto_width_uses_the_widest_source() {
        let opts = ExecOptions {
            parallelism: 0,
            morsel_rows: 1,
            ..ExecOptions::default()
        };
        let small = vec![batch(&[1], &[1])];
        let large = vec![batch(&[1, 2, 3, 4], &[1, 2, 3, 4])];
        let cores = std::thread::available_parallelism().map_or(1, |v| v.get());
        assert_eq!(
            auto_width(&opts, &[small, large], &no_media_plan()),
            4.min(cores)
        );
    }

    /// No sources (a literal-only plan) still needs one worker, never zero.
    #[test]
    fn auto_width_is_at_least_one() {
        let opts = ExecOptions {
            parallelism: 0,
            ..ExecOptions::default()
        };
        assert_eq!(auto_width(&opts, &[], &no_media_plan()), 1);
        assert_eq!(auto_width(&opts, &[vec![]], &no_media_plan()), 1);
    }

    /// A media-decode plan lifts the morsel-count cap: its per-row decode is heavy and
    /// parallelizes *inside* the morsel, so a corpus that is a single (tiny-encoded-bytes)
    /// morsel must still get every core — not one. Without this, `read.images(decode=True)`
    /// over a sub-morsel corpus decodes single-threaded.
    #[test]
    fn media_decode_plan_uses_all_cores_despite_single_morsel() {
        use bc_expr::{Expr, ImageFunc};

        let opts = ExecOptions {
            parallelism: 0,
            ..ExecOptions::default()
        };
        // One tiny source → one morsel → the cap would say 1 worker.
        let one_morsel = &[vec![batch(&[1], &[1])]][..];
        assert_eq!(auto_width(&opts, one_morsel, &no_media_plan()), 1);

        // The same input under an image-decode projection: width jumps to all cores.
        let decode = RelOp::Project {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            exprs: vec![ProjectionItem {
                expr: Expr::Image {
                    func: ImageFunc::ToTensor,
                    input: Box::new(Expr::Col { name: "k".into() }),
                    width: Some(224),
                    height: Some(224),
                    mean: None,
                    std: None,
                    channels_first: false,
                    format: None,
                    fill: None,
                },
                alias: "img".into(),
            }],
        };
        // "All cores" means the cores this process may actually *use*: `usable_cores` caps
        // `available_parallelism` by the cgroup CPU quota. Asserting against the raw
        // `available_parallelism` would pin the oversubscription this lift used to cause on a
        // quota-throttled node — the media path is exactly where it hurts, since it lifts the
        // morsel cap specifically to take every core.
        assert_eq!(
            auto_width(&opts, one_morsel, &decode),
            bc_arrow::usable_cores()
        );
        // An explicit parallelism still wins over the media lift (control-plane decision).
        let pinned = ExecOptions {
            parallelism: 3,
            ..ExecOptions::default()
        };
        assert_eq!(auto_width(&pinned, one_morsel, &decode), 3);
    }

    fn str_batch(vals: &[&str]) -> RecordBatch {
        RecordBatch::try_from_iter(vec![(
            "s",
            Arc::new(StringArray::from(vals.to_vec())) as ArrayRef,
        )])
        .unwrap()
    }

    /// A zero budget (the default) leaves the spill envelope off, so the in-memory
    /// fast path is unchanged. A positive budget populates `agg_spill` from the
    /// control plane's config — this is what makes the main `execute_plan` path
    /// able to spill out of core.
    #[test]
    fn with_engine_config_gates_spill_on_budget() {
        let unbounded = ExecOptions::default().with_engine_config(&EngineConfig::default());
        assert!(unbounded.agg_spill.is_none());

        let budgeted = ExecOptions::default().with_engine_config(&EngineConfig {
            memory_budget_bytes: 4096,
            spill_dir: Some("/scratch/spill".into()),
            ..EngineConfig::default()
        });
        let sp = budgeted.agg_spill.expect("positive budget enables spill");
        assert_eq!(sp.memory_budget_bytes, 4096);
        assert_eq!(sp.dir, PathBuf::from("/scratch/spill"));

        // No spill_dir → falls back to the OS temp dir rather than failing.
        let no_dir = ExecOptions::default().with_engine_config(&EngineConfig {
            memory_budget_bytes: 4096,
            spill_dir: None,
            ..EngineConfig::default()
        });
        assert_eq!(no_dir.agg_spill.unwrap().dir, std::env::temp_dir());
    }

    /// `pool_for` returns the *same* cached pool for a repeated width (so streaming
    /// micro-batches reuse threads instead of spawning a fresh pool each call) and a
    /// pool of exactly the requested width (the hash-shuffle bucket count keys off
    /// it).
    #[test]
    fn pool_for_reuses_pool_per_width() {
        let a = pool_for(3).unwrap();
        let b = pool_for(3).unwrap();
        assert!(
            Arc::ptr_eq(&a, &b),
            "same width must return the cached pool"
        );
        assert_eq!(a.current_num_threads(), 3);

        let c = pool_for(2).unwrap();
        assert!(!Arc::ptr_eq(&a, &c), "a different width gets its own pool");
        assert_eq!(c.current_num_threads(), 2);
    }

    /// The fused linear pipeline (Scan→Filter→Project) is bit-identical to both the
    /// unfused parallel path and the sequential oracle — same rows in the same order.
    /// Exercises a JIT-eligible arithmetic projection across multiple morsels.
    #[test]
    fn fused_linear_chain_matches_unfused_and_oracle() {
        use bc_expr::{BinaryOp, Expr, Literal};
        use bc_ir::ProjectionItem;

        // Scan → Filter(k > 2) → Project(k, v, k + v AS sum).
        let plan = RelOp::Project {
            input: Box::new(RelOp::Filter {
                input: Box::new(RelOp::Scan { source_id: 0 }),
                predicate: Expr::Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(Expr::Col { name: "k".into() }),
                    right: Box::new(Expr::Lit {
                        value: Literal::Int(2),
                    }),
                },
            }),
            exprs: vec![
                ProjectionItem {
                    expr: Expr::Col { name: "k".into() },
                    alias: "k".into(),
                },
                ProjectionItem {
                    expr: Expr::Col { name: "v".into() },
                    alias: "v".into(),
                },
                ProjectionItem {
                    expr: Expr::Binary {
                        op: BinaryOp::Add,
                        left: Box::new(Expr::Col { name: "k".into() }),
                        right: Box::new(Expr::Col { name: "v".into() }),
                    },
                    alias: "sum".into(),
                },
            ],
        };

        let sources = vec![vec![
            batch(&[1, 5, 3, 2], &[10, 20, 30, 40]),
            batch(&[7, 0, 4], &[1, 2, 3]),
        ]];

        let rows = |out: &[RecordBatch]| -> Vec<(i64, i64, i64)> {
            let mut v = Vec::new();
            for b in out {
                let k = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
                let val = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
                let s = b.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
                for i in 0..b.num_rows() {
                    v.push((k.value(i), val.value(i), s.value(i)));
                }
            }
            v
        };

        let oracle = rows(&execute(&plan, &sources).unwrap());
        // Small morsels so the fused pass runs over several morsels, not one.
        let unfused_opts = ExecOptions {
            morsel_rows: 2,
            ..ExecOptions::default()
        };
        let fused_opts = ExecOptions {
            morsel_rows: 2,
            fuse_linear: true,
            ..ExecOptions::default()
        };
        let unfused = rows(&execute_parallel_with(&plan, &sources, &unfused_opts).unwrap());
        let fused = rows(&execute_parallel_with(&plan, &sources, &fused_opts).unwrap());

        // Linear chain preserves order, so equality is exact (not just multiset).
        assert_eq!(oracle, vec![(5, 20, 25), (3, 30, 33), (7, 1, 8), (4, 3, 7)]);
        assert_eq!(unfused, oracle);
        assert_eq!(fused, oracle);
    }

    /// Fusion emits a metric per fused op with the SAME op_ids, kinds, and (exact)
    /// row counts the unfused path records — so the learning/calibration loop is
    /// unaffected by the fused flag.
    #[test]
    fn fused_chain_records_per_op_metrics() {
        use bc_expr::{BinaryOp, Expr, Literal};
        use bc_ir::ProjectionItem;

        let plan = RelOp::Project {
            input: Box::new(RelOp::Filter {
                input: Box::new(RelOp::Scan { source_id: 0 }),
                predicate: Expr::Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(Expr::Col { name: "k".into() }),
                    right: Box::new(Expr::Lit {
                        value: Literal::Int(2),
                    }),
                },
            }),
            exprs: vec![ProjectionItem {
                expr: Expr::Col { name: "v".into() },
                alias: "v".into(),
            }],
        };
        let sources = vec![vec![batch(&[1, 5, 3, 2], &[10, 20, 30, 40])]];

        let metric = |opts: &ExecOptions, kind: &str| -> (u32, u64, u64) {
            let (_out, m) = execute_parallel_with_metrics(&plan, &sources, opts).unwrap();
            let op = m
                .ops
                .iter()
                .find(|o| o.kind == kind)
                .unwrap_or_else(|| panic!("no {kind} metric"));
            (op.op_id, op.rows_in, op.rows_out)
        };

        let base = ExecOptions {
            morsel_rows: 2,
            ..ExecOptions::default()
        };
        let fused = ExecOptions {
            fuse_linear: true,
            ..base.clone()
        };
        // Same op_id + exact row counts for filter (4 in → 2 out) and project (2 → 2).
        assert_eq!(metric(&fused, "filter"), metric(&base, "filter"));
        assert_eq!(metric(&fused, "project"), metric(&base, "project"));
        assert_eq!(metric(&fused, "scan"), metric(&base, "scan"));
    }

    /// A tiny byte budget splits a wide-string morsel into many morsels even when
    /// the row count is far under `morsel_rows`, and the row multiset is preserved
    /// — the byte-aware path neither loses nor reorders rows.
    #[test]
    fn byte_budget_splits_wide_morsels_but_preserves_rows() {
        let wide: Vec<String> = (0..64).map(|i| format!("{i:0256}")).collect();
        let refs: Vec<&str> = wide.iter().map(|s| s.as_str()).collect();
        let data = vec![str_batch(&refs)];

        let plan = RelOp::Scan { source_id: 0 };
        let seq = execute(&plan, std::slice::from_ref(&data)).unwrap();

        let opts = ExecOptions {
            morsel_rows: 16_384, // row bound will not trip
            morsel_bytes: 512,   // byte bound forces fine splitting
            ..ExecOptions::default()
        };
        let (par, _m) = execute_parallel_with_metrics(&plan, &[data], &opts).unwrap();
        assert_eq!(
            rows(&seq),
            rows(&par),
            "byte-aware morselize lost/changed rows"
        );
        assert!(
            par.len() > 1,
            "tiny byte budget should split into multiple morsels, got {}",
            par.len()
        );
    }

    /// Row-only behavior is unchanged: with the byte bound effectively off, a
    /// narrow batch under `morsel_rows` is not split.
    #[test]
    fn narrow_data_is_not_byte_split() {
        let data = vec![batch(&[1, 2, 3, 4], &[10, 20, 30, 40])];
        let plan = RelOp::Scan { source_id: 0 };
        let opts = ExecOptions {
            morsel_rows: 16_384,
            morsel_bytes: 1 << 20, // 1 MiB; 4 i64 rows are ~tens of bytes
            ..ExecOptions::default()
        };
        let (par, _m) = execute_parallel_with_metrics(&plan, &[data], &opts).unwrap();
        assert_eq!(par.len(), 1, "narrow batch must stay a single morsel");
    }

    fn rows(batches: &[RecordBatch]) -> std::collections::BTreeSet<String> {
        let mut out = std::collections::BTreeSet::new();
        for b in batches {
            for i in 0..b.num_rows() {
                let cells: Vec<String> = (0..b.num_columns())
                    .map(|c| scalar(b.column(c), i))
                    .collect();
                out.insert(cells.join("|"));
            }
        }
        out
    }

    fn scalar(a: &ArrayRef, i: usize) -> String {
        if let Some(x) = a.as_any().downcast_ref::<Int64Array>() {
            return if x.is_null(i) {
                "∅".into()
            } else {
                x.value(i).to_string()
            };
        }
        if let Some(x) = a.as_any().downcast_ref::<StringArray>() {
            return if x.is_null(i) {
                "∅".into()
            } else {
                x.value(i).to_string()
            };
        }
        "?".into()
    }

    /// The parallel executor must produce the same multiset of rows as the
    /// sequential reference, regardless of how the input is split.
    #[test]
    fn parallel_matches_sequential_aggregate() {
        // group_by k sum(v)
        use bc_expr::Expr;
        use bc_ir::{AggFunc, AggregateItem, ProjectionItem};

        let plan = RelOp::Aggregate {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            group_keys: vec![ProjectionItem {
                expr: Expr::Col { name: "k".into() },
                alias: "k".into(),
            }],
            aggregates: vec![AggregateItem {
                func: AggFunc::Sum,
                input: Some(Expr::Col { name: "v".into() }),
                input2: None,
                alias: "s".into(),
                param: None,
            }],
        };
        // Same data, split two different ways.
        let one = vec![batch(&[1, 2, 1, 3, 2, 1], &[10, 20, 30, 40, 50, 60])];
        let many = vec![
            batch(&[1, 2], &[10, 20]),
            batch(&[1, 3], &[30, 40]),
            batch(&[2, 1], &[50, 60]),
        ];
        let seq = execute(&plan, std::slice::from_ref(&one)).unwrap();
        let par1 = execute_parallel(&plan, &[one]).unwrap();
        let par2 = execute_parallel(&plan, &[many]).unwrap();
        assert_eq!(rows(&seq), rows(&par1));
        assert_eq!(rows(&seq), rows(&par2));
    }

    /// `{k -> median*1000 (rounded), or a sentinel for a null result}` from an
    /// aggregate result of `[k:i64, m:f64]`, for order-independent comparison.
    fn quantile_by_key(batches: &[RecordBatch]) -> std::collections::BTreeMap<i64, i64> {
        use arrow::array::Float64Array;
        let mut map = std::collections::BTreeMap::new();
        for b in batches {
            let k = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let m = b.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
            for i in 0..b.num_rows() {
                let v = if m.is_null(i) {
                    i64::MIN // sentinel: null quantile (all-null group)
                } else {
                    (m.value(i) * 1000.0).round() as i64
                };
                map.insert(k.value(i), v);
            }
        }
        map
    }

    fn median_plan(q_param: Option<f64>) -> RelOp {
        use bc_expr::Expr;
        use bc_ir::{AggFunc, AggregateItem, ProjectionItem};
        let (func, param) = match q_param {
            None => (AggFunc::Median, None),
            Some(q) => (AggFunc::Quantile, Some(q)),
        };
        RelOp::Aggregate {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            group_keys: vec![ProjectionItem {
                expr: Expr::Col { name: "k".into() },
                alias: "k".into(),
            }],
            aggregates: vec![AggregateItem {
                func,
                input: Some(Expr::Col { name: "v".into() }),
                input2: None,
                param,
                alias: "m".into(),
            }],
        }
    }

    /// A `[k:i64 (non-null), v:i64 (nullable)]` batch.
    fn nbatch(ks: &[i64], vs: &[Option<i64>]) -> RecordBatch {
        use arrow::datatypes::{DataType, Field, Schema};
        let schema = Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, true),
        ]);
        RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![
                std::sync::Arc::new(Int64Array::from(ks.to_vec())),
                std::sync::Arc::new(Int64Array::from(vs.to_vec())),
            ],
        )
        .unwrap()
    }

    /// The bounded out-of-core median (forced via a tiny spill budget) must equal the
    /// in-memory median exactly — including a hot key, nulls, and an all-null group.
    #[test]
    fn spilling_median_matches_in_memory() {
        let plan = median_plan(None);
        // Hot key 0 dominates (200 values, each of 0..49 four times → median 24.5);
        // cold keys; a key with a null value (ignored); an all-null group (→ null).
        let mut ks: Vec<i64> = Vec::new();
        let mut vs: Vec<Option<i64>> = Vec::new();
        for i in 0..200i64 {
            ks.push(0);
            vs.push(Some(i % 50));
        }
        for v in [Some(3), Some(1), Some(2)] {
            ks.push(1);
            vs.push(v);
        }
        for v in [Some(10), None, Some(30)] {
            // non-null {10,30} → median 20
            ks.push(2);
            vs.push(v);
        }
        for _ in 0..2 {
            ks.push(3);
            vs.push(None); // all-null group → null median
        }
        let data = vec![nbatch(&ks, &vs)];

        let seq = execute(&plan, std::slice::from_ref(&data)).unwrap();
        let dir = std::env::temp_dir().join(format!("bc_med_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1, // force the bounded out-of-core path
                dir,
                codec: SpillCodec::None,
            }),
            morsel_rows: 8, // many morsels → many spilled runs
            ..ExecOptions::default()
        };
        let spilled = execute_parallel_with(&plan, &[data], &opts).unwrap();
        assert_eq!(quantile_by_key(&seq), quantile_by_key(&spilled));
    }

    /// Canonical sorted multiset of result rows, each cell formatted with nulls and
    /// f64 rounded to milli — so a result relation can be compared regardless of row
    /// order or which spill path produced it (works for any column types).
    fn canonical_rows(batches: &[RecordBatch]) -> Vec<Vec<String>> {
        use arrow::array::{Float64Array, Int64Array};
        let mut rows: Vec<Vec<String>> = Vec::new();
        for b in batches {
            for i in 0..b.num_rows() {
                let cells: Vec<String> = (0..b.num_columns())
                    .map(|c| {
                        let col = b.column(c);
                        if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
                            if a.is_null(i) {
                                "null".to_string()
                            } else {
                                a.value(i).to_string()
                            }
                        } else if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
                            if a.is_null(i) {
                                "null".to_string()
                            } else {
                                ((a.value(i) * 1000.0).round() as i64).to_string()
                            }
                        } else {
                            format!("{col:?}")
                        }
                    })
                    .collect();
                rows.push(cells);
            }
        }
        rows.sort();
        rows
    }

    /// A grouped aggregate mixing a value-list aggregate (`median`, whose per-group
    /// state can blow memory on a hot key) with constant-state aggregates
    /// (`sum`/`max`) — the shape that today falls to the in-memory grace path. The
    /// spilled result MUST equal the sequential oracle exactly: this oracle guards the
    /// mixed-aggregate spill contract (and would catch any future bounded-path rewrite
    /// that misaligns a value-list column with its group).
    #[test]
    fn spilling_mixed_aggregate_matches_in_memory() {
        use bc_expr::Expr;
        use bc_ir::{AggFunc, AggregateItem, ProjectionItem};

        let item = |func, alias: &str| AggregateItem {
            func,
            input: Some(Expr::Col { name: "v".into() }),
            input2: None,
            param: None,
            alias: alias.into(),
        };
        let plan = RelOp::Aggregate {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            group_keys: vec![ProjectionItem {
                expr: Expr::Col { name: "k".into() },
                alias: "k".into(),
            }],
            aggregates: vec![
                item(AggFunc::Median, "m"),
                item(AggFunc::Sum, "s"),
                item(AggFunc::Max, "mx"),
            ],
        };
        // Hot key 0 (200 values), cold keys, a key with a null value, an all-null group.
        let mut ks: Vec<i64> = Vec::new();
        let mut vs: Vec<Option<i64>> = Vec::new();
        for i in 0..200i64 {
            ks.push(0);
            vs.push(Some(i % 50));
        }
        for v in [Some(3), Some(1), Some(2)] {
            ks.push(1);
            vs.push(v);
        }
        for v in [Some(10), None, Some(30)] {
            ks.push(2);
            vs.push(v);
        }
        for _ in 0..2 {
            ks.push(3);
            vs.push(None);
        }
        let data = vec![nbatch(&ks, &vs)];

        let seq = execute(&plan, std::slice::from_ref(&data)).unwrap();
        let dir = std::env::temp_dir().join(format!("bc_mixed_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1, // force the out-of-core path
                dir,
                codec: SpillCodec::None,
            }),
            morsel_rows: 8, // many morsels → many spilled runs
            ..ExecOptions::default()
        };
        let spilled = execute_parallel_with(&plan, &[data], &opts).unwrap();
        assert_eq!(canonical_rows(&seq), canonical_rows(&spilled));
    }

    /// Build a grouped aggregate over `funcs` (each on column `v`, grouped by `k`).
    fn mixed_plan(funcs: &[bc_ir::AggFunc]) -> RelOp {
        use bc_expr::Expr;
        use bc_ir::{AggregateItem, ProjectionItem};
        let aggregates = funcs
            .iter()
            .enumerate()
            .map(|(i, &func)| AggregateItem {
                func,
                input: Some(Expr::Col { name: "v".into() }),
                input2: None,
                param: None,
                alias: format!("a{i}"),
            })
            .collect();
        RelOp::Aggregate {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            group_keys: vec![ProjectionItem {
                expr: Expr::Col { name: "k".into() },
                alias: "k".into(),
            }],
            aggregates,
        }
    }

    /// Property/fuzz net for the mixed-aggregate spill path: across many random
    /// datasets (hot keys, nulls, varied cardinality) and aggregate combinations that
    /// mix a value-list aggregate (median/n_unique/mode — per-group list state) with
    /// constant-state aggregates (sum/min/max/count/mean), the spilled result MUST
    /// equal the in-memory oracle exactly. This guards the contract that any bounded
    /// mixed-aggregate rewrite must preserve — a single misaligned value-list column
    /// would surface here.
    #[test]
    fn spilling_mixed_aggregate_fuzz_matches_in_memory() {
        use bc_ir::AggFunc::{Count, CountDistinct, Max, Mean, Median, Min, Mode, Sum};

        // Deterministic xorshift64 (no Math.random in tests; seed per case).
        fn xs(s: &mut u64) -> u64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        }

        let combos: [&[bc_ir::AggFunc]; 7] = [
            &[Median, Sum],
            &[CountDistinct, Sum, Max],
            &[Median, CountDistinct],
            &[Mode, Count],
            &[Median, Sum, Max, Min],
            &[Sum, Median, Mean], // value-list aggregate not first
            &[Mode, CountDistinct, Sum],
        ];

        for (ci, combo) in combos.iter().enumerate() {
            for case in 0..10u64 {
                let mut s =
                    0x9E37_79B9_7F4A_7C15u64 ^ ((ci as u64) << 40) ^ (case.wrapping_mul(0x100));
                let n = 40 + (xs(&mut s) % 360) as usize;
                let kmod = 1 + (xs(&mut s) % 6) as i64; // 1..6 distinct keys → hot keys
                let vmod = 1 + (xs(&mut s) % 40) as i64;
                let mut ks: Vec<i64> = Vec::with_capacity(n);
                let mut vs: Vec<Option<i64>> = Vec::with_capacity(n);
                for _ in 0..n {
                    ks.push((xs(&mut s) % kmod as u64) as i64);
                    vs.push(if xs(&mut s) % 10 == 0 {
                        None // ~10% nulls (incl. occasional all-null groups)
                    } else {
                        Some((xs(&mut s) % vmod as u64) as i64)
                    });
                }
                let plan = mixed_plan(combo);
                let data = vec![nbatch(&ks, &vs)];
                let seq = execute(&plan, std::slice::from_ref(&data)).unwrap();
                let dir = std::env::temp_dir()
                    .join(format!("bc_mixfuzz_{}_{ci}_{case}", std::process::id()));
                let opts = ExecOptions {
                    agg_spill: Some(SpillOptions {
                        memory_budget_bytes: 1, // force the out-of-core path
                        dir,
                        codec: SpillCodec::None,
                    }),
                    morsel_rows: 8,
                    ..ExecOptions::default()
                };
                let spilled = execute_parallel_with(&plan, &[data], &opts).unwrap();
                assert_eq!(
                    canonical_rows(&seq),
                    canonical_rows(&spilled),
                    "mismatch: combo {ci} case {case}"
                );
            }
        }
    }

    /// The mixed-aggregate bounded path must actually *engage* for a value-list +
    /// constant-state mix (so the oracle/fuzz tests above exercise it, not grace), and
    /// *abstain* for shapes that are already bounded — a lone aggregate (the
    /// single-aggregate paths own it) or an all-constant-state set (grace bounds it).
    #[test]
    fn mixed_spill_gate_engages_only_for_mixed_value_list() {
        use bc_expr::Expr;
        use bc_ir::{AggFunc, AggregateItem, ProjectionItem};

        let gk = vec![ProjectionItem {
            expr: Expr::Col { name: "k".into() },
            alias: "k".into(),
        }];
        let parts = vec![nbatch(&[0, 0, 1, 1], &[Some(1), Some(3), Some(2), Some(4)])];
        let dir = std::env::temp_dir().join(format!("bc_mixgate_{}", std::process::id()));
        let agg = |f, a: &str| AggregateItem {
            func: f,
            input: Some(Expr::Col { name: "v".into() }),
            input2: None,
            param: None,
            alias: a.into(),
        };

        // median + sum → engages the bounded mixed path.
        let mixed = [agg(AggFunc::Median, "m"), agg(AggFunc::Sum, "s")];
        assert!(
            ops::try_bounded_mixed_spill(&parts, &gk, &mixed, &dir, 1, SpillCodec::None)
                .unwrap()
                .is_some(),
            "median+sum should engage the bounded mixed path"
        );

        // all constant-state → abstains (grace already bounds per-group accumulators).
        let cs_only = [agg(AggFunc::Sum, "s"), agg(AggFunc::Max, "mx")];
        assert!(
            ops::try_bounded_mixed_spill(&parts, &gk, &cs_only, &dir, 1, SpillCodec::None)
                .unwrap()
                .is_none(),
            "all-constant-state should fall through to grace"
        );

        // a lone aggregate → abstains (the single-aggregate paths own it).
        let lone = [agg(AggFunc::Median, "m")];
        assert!(
            ops::try_bounded_mixed_spill(&parts, &gk, &lone, &dir, 1, SpillCodec::None)
                .unwrap()
                .is_none(),
            "a lone aggregate is not the mixed path's job"
        );
    }

    /// The bounded out-of-core n_unique (COUNT DISTINCT, forced via a tiny spill
    /// budget) must equal the in-memory n_unique exactly — including a hot key with
    /// many duplicates, nulls (excluded), and an all-null group (→ 0).
    #[test]
    fn spilling_n_unique_matches_in_memory() {
        use bc_expr::Expr;
        use bc_ir::{AggFunc, AggregateItem, ProjectionItem};

        let plan = RelOp::Aggregate {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            group_keys: vec![ProjectionItem {
                expr: Expr::Col { name: "k".into() },
                alias: "k".into(),
            }],
            aggregates: vec![AggregateItem {
                func: AggFunc::CountDistinct,
                input: Some(Expr::Col { name: "v".into() }),
                input2: None,
                param: None,
                alias: "nd".into(),
            }],
        };
        // Hot key 0: values 0..49 repeated 4× → 50 distinct. Cold key 1: {1,1,2} → 2.
        // Key 2: {10, null, 30} → 2 (null excluded). Key 3: all null → 0.
        let mut ks: Vec<i64> = Vec::new();
        let mut vs: Vec<Option<i64>> = Vec::new();
        for i in 0..200i64 {
            ks.push(0);
            vs.push(Some(i % 50));
        }
        for v in [Some(1), Some(1), Some(2)] {
            ks.push(1);
            vs.push(v);
        }
        for v in [Some(10), None, Some(30)] {
            ks.push(2);
            vs.push(v);
        }
        for _ in 0..2 {
            ks.push(3);
            vs.push(None);
        }
        let data = vec![nbatch(&ks, &vs)];

        let seq = execute(&plan, std::slice::from_ref(&data)).unwrap();
        let dir = std::env::temp_dir().join(format!("bc_ndistinct_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1, // force the bounded out-of-core path
                dir,
                codec: SpillCodec::None,
            }),
            morsel_rows: 8, // many morsels → many spilled runs
            ..ExecOptions::default()
        };
        let spilled = execute_parallel_with(&plan, &[data], &opts).unwrap();
        assert_eq!(count_by_key(&seq), count_by_key(&spilled));
        // Sanity: the expected distinct counts.
        let expected: std::collections::BTreeMap<i64, i64> =
            [(0, 50), (1, 2), (2, 2), (3, 0)].into_iter().collect();
        assert_eq!(count_by_key(&seq), expected);
    }

    fn count_by_key(batches: &[RecordBatch]) -> std::collections::BTreeMap<i64, i64> {
        let mut map = std::collections::BTreeMap::new();
        for b in batches {
            let k = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let c = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..b.num_rows() {
                map.insert(k.value(i), c.value(i));
            }
        }
        map
    }

    /// The bounded out-of-core mode (forced via a tiny spill budget) must equal the
    /// in-memory mode exactly — most frequent value, ties → smallest, nulls excluded,
    /// all-null group → null.
    #[test]
    fn spilling_mode_matches_in_memory() {
        use bc_expr::Expr;
        use bc_ir::{AggFunc, AggregateItem, ProjectionItem};

        let plan = RelOp::Aggregate {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            group_keys: vec![ProjectionItem {
                expr: Expr::Col { name: "k".into() },
                alias: "k".into(),
            }],
            aggregates: vec![AggregateItem {
                func: AggFunc::Mode,
                input: Some(Expr::Col { name: "v".into() }),
                input2: None,
                param: None,
                alias: "mo".into(),
            }],
        };
        // key 0: 5 appears most (→ 5). key 1: 1,1,2,2 tie → smallest (1).
        // key 2: {10, null, 30} each once → tie → smallest non-null (10).
        // key 3: all null → null.
        let mut ks: Vec<i64> = Vec::new();
        let mut vs: Vec<Option<i64>> = Vec::new();
        for v in [Some(5), Some(5), Some(5), Some(1), Some(1), Some(2)] {
            ks.push(0);
            vs.push(v);
        }
        for v in [Some(1), Some(1), Some(2), Some(2)] {
            ks.push(1);
            vs.push(v);
        }
        for v in [Some(10), None, Some(30)] {
            ks.push(2);
            vs.push(v);
        }
        for _ in 0..2 {
            ks.push(3);
            vs.push(None);
        }
        let data = vec![nbatch(&ks, &vs)];

        let seq = execute(&plan, std::slice::from_ref(&data)).unwrap();
        let dir = std::env::temp_dir().join(format!("bc_mode_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1, // force the bounded out-of-core path
                dir,
                codec: SpillCodec::None,
            }),
            morsel_rows: 4, // many morsels → many spilled runs (runs span batches)
            ..ExecOptions::default()
        };
        let spilled = execute_parallel_with(&plan, &[data], &opts).unwrap();
        assert_eq!(mode_by_key(&seq), mode_by_key(&spilled));
        let expected: std::collections::BTreeMap<i64, Option<i64>> =
            [(0, Some(5)), (1, Some(1)), (2, Some(10)), (3, None)]
                .into_iter()
                .collect();
        assert_eq!(mode_by_key(&seq), expected);
    }

    fn mode_by_key(batches: &[RecordBatch]) -> std::collections::BTreeMap<i64, Option<i64>> {
        let mut map = std::collections::BTreeMap::new();
        for b in batches {
            let k = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let mo = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..b.num_rows() {
                let v = if mo.is_null(i) {
                    None
                } else {
                    Some(mo.value(i))
                };
                map.insert(k.value(i), v);
            }
        }
        map
    }

    /// The bounded out-of-core histogram (forced via a tiny spill budget) must equal
    /// the in-memory histogram exactly — distinct value→count maps, nulls excluded,
    /// all-null group → null map.
    #[test]
    fn spilling_histogram_matches_in_memory() {
        use bc_expr::Expr;
        use bc_ir::{AggFunc, AggregateItem, ProjectionItem};

        let plan = RelOp::Aggregate {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            group_keys: vec![ProjectionItem {
                expr: Expr::Col { name: "k".into() },
                alias: "k".into(),
            }],
            aggregates: vec![AggregateItem {
                func: AggFunc::Histogram,
                input: Some(Expr::Col { name: "v".into() }),
                input2: None,
                param: None,
                alias: "h".into(),
            }],
        };
        // key 0: {5:3, 1:2, 2:1}. key 1: {1:2, 2:2}. key 2: {10:1, 30:1} (null skipped).
        // key 3: all null → null map.
        let mut ks: Vec<i64> = Vec::new();
        let mut vs: Vec<Option<i64>> = Vec::new();
        for v in [Some(5), Some(5), Some(5), Some(1), Some(1), Some(2)] {
            ks.push(0);
            vs.push(v);
        }
        for v in [Some(1), Some(1), Some(2), Some(2)] {
            ks.push(1);
            vs.push(v);
        }
        for v in [Some(10), None, Some(30)] {
            ks.push(2);
            vs.push(v);
        }
        for _ in 0..2 {
            ks.push(3);
            vs.push(None);
        }
        let data = vec![nbatch(&ks, &vs)];

        let seq = execute(&plan, std::slice::from_ref(&data)).unwrap();
        let dir = std::env::temp_dir().join(format!("bc_hist_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1,
                dir,
                codec: SpillCodec::None,
            }),
            morsel_rows: 4,
            ..ExecOptions::default()
        };
        let spilled = execute_parallel_with(&plan, &[data], &opts).unwrap();
        assert_eq!(histogram_by_key(&seq), histogram_by_key(&spilled));
        let mut expected: std::collections::BTreeMap<i64, Option<Vec<(i64, i64)>>> =
            std::collections::BTreeMap::new();
        expected.insert(0, Some(vec![(1, 2), (2, 1), (5, 3)]));
        expected.insert(1, Some(vec![(1, 2), (2, 2)]));
        expected.insert(2, Some(vec![(10, 1), (30, 1)]));
        expected.insert(3, None);
        assert_eq!(histogram_by_key(&seq), expected);
    }

    #[allow(clippy::type_complexity)]
    fn histogram_by_key(
        batches: &[RecordBatch],
    ) -> std::collections::BTreeMap<i64, Option<Vec<(i64, i64)>>> {
        use arrow::array::MapArray;
        let mut out = std::collections::BTreeMap::new();
        for b in batches {
            let k = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let m = b.column(1).as_any().downcast_ref::<MapArray>().unwrap();
            for i in 0..b.num_rows() {
                let entry = if m.is_null(i) {
                    None
                } else {
                    let s = m.value(i);
                    let keys = s.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
                    let vals = s.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
                    let mut pairs: Vec<(i64, i64)> = (0..keys.len())
                        .map(|j| (keys.value(j), vals.value(j)))
                        .collect();
                    pairs.sort();
                    Some(pairs)
                };
                out.insert(k.value(i), entry);
            }
        }
        out
    }

    /// Same, for a non-median continuous quantile (q = 0.25).
    #[test]
    fn spilling_quantile_matches_in_memory() {
        let plan = median_plan(Some(0.25));
        let mut ks: Vec<i64> = Vec::new();
        let mut vs: Vec<Option<i64>> = Vec::new();
        for i in 0..120i64 {
            ks.push(0);
            vs.push(Some(i));
        }
        for v in [Some(5), Some(15), Some(25), Some(35)] {
            ks.push(1);
            vs.push(v);
        }
        let data = vec![nbatch(&ks, &vs)];
        let seq = execute(&plan, std::slice::from_ref(&data)).unwrap();
        let dir = std::env::temp_dir().join(format!("bc_q_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1,
                dir,
                codec: SpillCodec::None,
            }),
            morsel_rows: 7,
            ..ExecOptions::default()
        };
        let spilled = execute_parallel_with(&plan, &[data], &opts).unwrap();
        assert_eq!(quantile_by_key(&seq), quantile_by_key(&spilled));
    }

    /// Unnest explodes a list column into one row per element (dropping null/empty
    /// lists), and the parallel path matches the sequential oracle.
    #[test]
    fn unnest_explodes_list_and_matches_sequential() {
        use arrow::array::ListArray;
        use arrow::datatypes::Int64Type;

        fn list_batch() -> RecordBatch {
            let ids = Int64Array::from(vec![10, 20, 30, 40]);
            // [1,2] | [] (empty → no rows) | null (→ no rows) | [3]
            let xs = ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
                Some(vec![Some(1), Some(2)]),
                Some(vec![]),
                None,
                Some(vec![Some(3)]),
            ]);
            RecordBatch::try_from_iter(vec![
                ("id", Arc::new(ids) as ArrayRef),
                ("xs", Arc::new(xs) as ArrayRef),
            ])
            .unwrap()
        }

        let plan = RelOp::Unnest {
            outer: false,
            index_alias: None,
            input: Box::new(RelOp::Scan { source_id: 0 }),
            column: "xs".into(),
            alias: "x".into(),
        };
        let data = vec![list_batch()];
        let seq = execute(&plan, std::slice::from_ref(&data)).unwrap();
        let par = execute_parallel(&plan, &[data]).unwrap();
        assert_eq!(rows(&seq), rows(&par));

        // Columns are (id, x); null/empty lists drop their row entirely.
        let expected: std::collections::BTreeSet<String> = ["10|1", "10|2", "40|3"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(rows(&seq), expected);
    }

    /// Unpivot reshapes wide → long (one row per `on` column), and the parallel path
    /// matches the sequential oracle.
    #[test]
    fn unpivot_reshapes_wide_to_long_and_matches_sequential() {
        fn wide_batch() -> RecordBatch {
            let ids = Int64Array::from(vec![1, 2]);
            let q1 = Int64Array::from(vec![10, 40]);
            let q2 = Int64Array::from(vec![20, 50]);
            RecordBatch::try_from_iter(vec![
                ("id", Arc::new(ids) as ArrayRef),
                ("q1", Arc::new(q1) as ArrayRef),
                ("q2", Arc::new(q2) as ArrayRef),
            ])
            .unwrap()
        }

        let plan = RelOp::Unpivot {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            index: vec!["id".into()],
            on: vec!["q1".into(), "q2".into()],
            variable_name: "variable".into(),
            value_name: "value".into(),
        };
        let data = vec![wide_batch()];
        let seq = execute(&plan, std::slice::from_ref(&data)).unwrap();
        let par = execute_parallel(&plan, &[data]).unwrap();
        assert_eq!(rows(&seq), rows(&par));

        // 2 rows × 2 melted columns → 4 rows (id|variable|value).
        let expected: std::collections::BTreeSet<String> =
            ["1|q1|10", "2|q1|40", "1|q2|20", "2|q2|50"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        assert_eq!(rows(&seq), expected);
    }

    /// Sample is deterministic and partition-independent: the parallel path (many
    /// ASOF join matches each left row to the nearest-≤ right row within its `by`
    /// group; the parallel path matches the sequential oracle.
    #[test]
    fn asof_join_backward_matches_sequential() {
        use bc_ir::{JoinOutputCol, JoinSide};
        fn left_batch() -> RecordBatch {
            // sym, ts
            let sym = arrow::array::StringArray::from(vec!["A", "A", "B"]);
            let ts = Int64Array::from(vec![10, 25, 10]);
            RecordBatch::try_from_iter(vec![
                ("sym", Arc::new(sym) as ArrayRef),
                ("ts", Arc::new(ts) as ArrayRef),
            ])
            .unwrap()
        }
        fn right_batch() -> RecordBatch {
            let sym = arrow::array::StringArray::from(vec!["A", "A", "B"]);
            let ts = Int64Array::from(vec![5, 20, 8]);
            let bid = Int64Array::from(vec![1, 2, 3]);
            RecordBatch::try_from_iter(vec![
                ("sym", Arc::new(sym) as ArrayRef),
                ("ts", Arc::new(ts) as ArrayRef),
                ("bid", Arc::new(bid) as ArrayRef),
            ])
            .unwrap()
        }
        let plan = RelOp::AsofJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(RelOp::Scan { source_id: 1 }),
            left_on: "ts".into(),
            right_on: "ts".into(),
            left_by: vec!["sym".into()],
            right_by: vec!["sym".into()],
            direction: bc_ir::AsofDirection::Backward,
            tolerance: None,
            allow_exact_matches: true,
            output: vec![
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "sym".into(),
                    alias: "sym".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "ts".into(),
                    alias: "ts".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Right,
                    name: "bid".into(),
                    alias: "bid".into(),
                },
            ],
        };
        let src = vec![vec![left_batch()], vec![right_batch()]];
        let seq = execute(&plan, &src).unwrap();
        let par = execute_parallel(&plan, &src).unwrap();
        assert_eq!(rows(&seq), rows(&par));
        // (A,10)->bid 1 (ts5≤10); (A,25)->bid 2 (ts20≤25); (B,10)->bid 3 (ts8≤10).
        let expected: std::collections::BTreeSet<String> = ["A|10|1", "A|25|2", "B|10|3"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(rows(&seq), expected);
    }

    /// morsels) keeps exactly the same rows as the sequential oracle (one batch).
    #[test]
    fn sample_is_deterministic_and_partition_independent() {
        fn rows_batch(lo: i64, hi: i64) -> RecordBatch {
            let v = Int64Array::from((lo..hi).collect::<Vec<_>>());
            RecordBatch::try_from_iter(vec![("v", Arc::new(v) as ArrayRef)]).unwrap()
        }
        let plan = RelOp::Sample {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            fraction: 0.5,
            seed: 1234,
            n: None,
        };
        // One big batch vs three smaller ones covering the same 0..300 values.
        let one = vec![rows_batch(0, 300)];
        let many = vec![
            rows_batch(0, 100),
            rows_batch(100, 200),
            rows_batch(200, 300),
        ];
        let seq = execute(&plan, std::slice::from_ref(&one)).unwrap();
        let par_one = execute_parallel(&plan, &[one]).unwrap();
        let par_many = execute_parallel(&plan, &[many]).unwrap();
        // Same rows regardless of batching or seq/par (content-hash sampling).
        assert_eq!(rows(&seq), rows(&par_one));
        assert_eq!(rows(&seq), rows(&par_many));
        // And it actually sampled (not all, not none) at ~50%.
        let kept = rows(&seq).len();
        assert!(kept > 50 && kept < 250, "kept {kept} of 300");
    }

    /// Fixed-count sample keeps exactly `n` rows, identical regardless of chunking
    /// (the global n-smallest hashes), and identical seq vs par.
    #[test]
    fn sample_n_is_exact_and_partition_independent() {
        fn rows_batch(lo: i64, hi: i64) -> RecordBatch {
            let v = Int64Array::from((lo..hi).collect::<Vec<_>>());
            RecordBatch::try_from_iter(vec![("v", Arc::new(v) as ArrayRef)]).unwrap()
        }
        let plan = RelOp::Sample {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            fraction: 1.0,
            seed: 99,
            n: Some(40),
        };
        let one = vec![rows_batch(0, 300)];
        let many = vec![
            rows_batch(0, 100),
            rows_batch(100, 200),
            rows_batch(200, 300),
        ];
        let seq = execute(&plan, std::slice::from_ref(&one)).unwrap();
        let par_one = execute_parallel(&plan, &[one]).unwrap();
        let par_many = execute_parallel(&plan, &[many]).unwrap();
        assert_eq!(count_rows(&seq), 40); // exactly n
        assert_eq!(rows(&seq), rows(&par_one));
        assert_eq!(rows(&seq), rows(&par_many)); // chunking-independent
    }

    #[test]
    fn window_spilling_matches_in_memory() {
        use bc_expr::Expr;
        use bc_ir::{SortKey, WindowFn, WindowFunc};

        // PARTITION BY k ORDER BY v: row_number + running sum(v).
        let plan = RelOp::Window {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            partition_keys: vec![Expr::Col { name: "k".into() }],
            order_keys: vec![SortKey {
                expr: Expr::Col { name: "v".into() },
                descending: false,
                nulls_first: false,
            }],
            functions: vec![
                WindowFunc {
                    func: WindowFn::RowNumber,
                    input: None,
                    offset: 1,
                    frame: None,
                    alpha: None,
                    half_life: None,
                    alias: "rn".into(),
                },
                WindowFunc {
                    func: WindowFn::Sum,
                    input: Some(Expr::Col { name: "v".into() }),
                    offset: 1,
                    frame: None,
                    alpha: None,
                    half_life: None,
                    alias: "s".into(),
                },
            ],
            rank_limit: None,
        };
        let data = vec![
            batch(&[1, 2, 1, 3, 2, 1], &[10, 20, 30, 40, 50, 60]),
            batch(&[4, 2, 5, 1, 3, 6], &[1, 2, 3, 4, 5, 6]),
            batch(&[1, 7, 2, 8, 3, 9], &[7, 8, 9, 10, 11, 12]),
        ];
        let seq = execute(&plan, std::slice::from_ref(&data)).unwrap();

        // memory_budget_bytes = 1 forces the grace-partitioned spill path.
        let dir = std::env::temp_dir().join(format!("bc_par_winspill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1,
                dir,
                codec: SpillCodec::None,
            }),
            ..ExecOptions::default()
        };
        let spilled = execute_parallel_with(&plan, &[data], &opts).unwrap();
        assert_eq!(rows(&seq), rows(&spilled));
    }

    #[test]
    fn window_appends_row_number_and_partition_sum() {
        use bc_expr::Expr;
        use bc_ir::{SortKey, WindowFn, WindowFunc};

        // PARTITION BY k ORDER BY v: row_number, and sum(v) over the partition.
        let plan = RelOp::Window {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            partition_keys: vec![Expr::Col { name: "k".into() }],
            order_keys: vec![SortKey {
                expr: Expr::Col { name: "v".into() },
                descending: false,
                nulls_first: false,
            }],
            functions: vec![
                WindowFunc {
                    func: WindowFn::RowNumber,
                    input: None,
                    offset: 1,
                    frame: None,
                    alpha: None,
                    half_life: None,
                    alias: "rn".into(),
                },
                WindowFunc {
                    func: WindowFn::Sum,
                    input: Some(Expr::Col { name: "v".into() }),
                    offset: 1,
                    frame: None,
                    alpha: None,
                    half_life: None,
                    alias: "s".into(),
                },
            ],
            rank_limit: None,
        };
        // k: [1,2,1,2,1], v: [30,5,10,15,20]
        let data = vec![batch(&[1, 2, 1, 2, 1], &[30, 5, 10, 15, 20])];
        let seq = execute(&plan, std::slice::from_ref(&data)).unwrap();
        let par = execute_parallel(&plan, &[data]).unwrap();
        // Both must agree (window is deterministic; rows compared as a multiset).
        assert_eq!(rows(&seq), rows(&par));

        // Verify concrete values from the sequential reference (input order kept).
        let b = &seq[0];
        assert_eq!(b.num_columns(), 4); // k, v, rn, s
        let col = |name: &str| {
            let i = b.schema().index_of(name).unwrap();
            b.column(i)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .clone()
        };
        let rn = col("rn");
        let s = col("s");
        // The window has an ORDER BY (for row_number), so SUM is a *running*
        // (cumulative) aggregate in sorted order — matching SQL semantics.
        // k=1 sorted by v asc: 10(rn1)→10, 20(rn2)→30, 30(rn3)→60.
        // k=2 sorted by v asc: 5(rn1)→5, 15(rn2)→20.
        // Original row order: 0:k1 v30, 1:k2 v5, 2:k1 v10, 3:k2 v15, 4:k1 v20.
        assert_eq!((rn.value(0), s.value(0)), (3, 60));
        assert_eq!((rn.value(1), s.value(1)), (1, 5));
        assert_eq!((rn.value(2), s.value(2)), (1, 10));
        assert_eq!((rn.value(3), s.value(3)), (2, 20));
        assert_eq!((rn.value(4), s.value(4)), (2, 30));
    }

    /// Fused `QUALIFY row_number() <= k`: the window keeps only the top-k rows per
    /// partition, and the parallel path agrees with the sequential oracle.
    #[test]
    fn window_rank_limit_keeps_top_k_per_partition() {
        use bc_expr::Expr;
        use bc_ir::{SortKey, WindowFn, WindowFunc};

        let plan = RelOp::Window {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            partition_keys: vec![Expr::Col { name: "k".into() }],
            order_keys: vec![SortKey {
                expr: Expr::Col { name: "v".into() },
                descending: false,
                nulls_first: false,
            }],
            functions: vec![WindowFunc {
                func: WindowFn::RowNumber,
                input: None,
                offset: 1,
                frame: None,
                alpha: None,
                half_life: None,
                alias: "rn".into(),
            }],
            rank_limit: Some(2),
        };
        // k=1: v=[30,10,20] → keep v=10(rn1),20(rn2); k=2: v=[5,15] → keep both.
        let data = vec![batch(&[1, 2, 1, 2, 1], &[30, 5, 10, 15, 20])];
        let seq = execute(&plan, std::slice::from_ref(&data)).unwrap();
        let par = execute_parallel(&plan, &[data]).unwrap();
        assert_eq!(rows(&seq), rows(&par)); // parity with the oracle

        let total: usize = seq.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 4); // 2 partitions × top-2
        for b in &seq {
            let i = b.schema().index_of("rn").unwrap();
            let rn = b.column(i).as_any().downcast_ref::<Int64Array>().unwrap();
            assert!(
                (0..rn.len()).all(|j| rn.value(j) <= 2),
                "every kept rn must be <= 2"
            );
        }
    }

    /// Spilling aggregation (tiny budget → forced disk grace partitioning) must
    /// produce exactly the sequential oracle's relation — only memory differs.
    #[test]
    fn spilling_aggregate_matches_sequential() {
        use bc_expr::Expr;
        use bc_ir::{AggFunc, AggregateItem, ProjectionItem};

        let plan = RelOp::Aggregate {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            group_keys: vec![ProjectionItem {
                expr: Expr::Col { name: "k".into() },
                alias: "k".into(),
            }],
            aggregates: vec![AggregateItem {
                func: AggFunc::Sum,
                input: Some(Expr::Col { name: "v".into() }),
                input2: None,
                alias: "s".into(),
                param: None,
            }],
        };
        let data = vec![
            batch(&[1, 2, 1, 3, 2, 1], &[10, 20, 30, 40, 50, 60]),
            batch(&[4, 2, 5, 1, 3, 6], &[1, 2, 3, 4, 5, 6]),
            batch(&[1, 7, 2, 8, 3, 9], &[7, 8, 9, 10, 11, 12]),
        ];
        let seq = execute(&plan, std::slice::from_ref(&data)).unwrap();

        // memory_budget_bytes = 1 forces the spill branch and many partitions.
        let dir = std::env::temp_dir().join(format!("bc_par_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1,
                dir,
                codec: SpillCodec::None,
            }),
            ..ExecOptions::default()
        };
        let spilled = execute_parallel_with(&plan, &[data], &opts).unwrap();
        assert_eq!(rows(&seq), rows(&spilled));
    }

    #[test]
    fn spilling_distinct_matches_sequential() {
        // DISTINCT must spill through the grace path (not OOM) and still equal the
        // sequential oracle. memory_budget_bytes = 1 forces the spill branch.
        let plan = RelOp::Distinct {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            keys: Vec::new(),
            order: Vec::new(),
            limit: None,
        };
        let data = vec![
            batch(&[1, 2, 1, 3, 2, 1], &[10, 20, 10, 40, 20, 10]),
            batch(&[4, 2, 5, 1, 3, 6], &[1, 20, 3, 10, 40, 6]),
            batch(&[1, 7, 2, 8, 3, 9], &[10, 8, 20, 10, 40, 12]),
        ];
        let seq = execute(&plan, std::slice::from_ref(&data)).unwrap();

        let dir =
            std::env::temp_dir().join(format!("bc_par_distinct_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1,
                dir,
                codec: SpillCodec::None,
            }),
            ..ExecOptions::default()
        };
        let spilled = execute_parallel_with(&plan, &[data], &opts).unwrap();
        assert_eq!(rows(&seq), rows(&spilled));
        // And the in-memory parallel path (no envelope) also matches.
        let mem = execute_parallel_with(
            &plan,
            &[vec![batch(&[1, 1, 2], &[1, 1, 2])]],
            &ExecOptions::default(),
        )
        .unwrap();
        assert_eq!(rows(&mem).len(), 2);
    }

    #[test]
    fn parallel_matches_sequential_join() {
        use bc_ir::{JoinOutputCol, JoinSide, JoinStrategy, JoinType};

        let plan = RelOp::HashJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(RelOp::Scan { source_id: 1 }),
            left_keys: vec!["k".into()],
            right_keys: vec!["k".into()],
            join_type: JoinType::Inner,
            output: vec![
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "k".into(),
                    alias: "k".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "v".into(),
                    alias: "lv".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Right,
                    name: "v".into(),
                    alias: "rv".into(),
                },
            ],
            strategy: JoinStrategy::Hash,
        };
        let left = vec![batch(&[1, 2, 3, 2], &[10, 20, 30, 40])];
        let right = vec![batch(&[2, 3, 3], &[1, 2, 3])];
        let seq = execute(&plan, &[left.clone(), right.clone()]).unwrap();
        let par = execute_parallel(&plan, &[left, right]).unwrap();
        assert_eq!(rows(&seq), rows(&par));
    }

    /// The parallel ASOF join partitions both sides by the `by` keys and joins each
    /// bucket independently; the union must equal the sequential single-pass oracle.
    /// `by` = "k", `on` = "v" (backward). Group 3 exists only on the left → its rows
    /// must survive with null right columns (left-style), exercising empty-right
    /// buckets.
    #[test]
    fn parallel_matches_sequential_asof_join() {
        use bc_ir::{JoinOutputCol, JoinSide};

        let plan = RelOp::AsofJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(RelOp::Scan { source_id: 1 }),
            left_on: "v".into(),
            right_on: "v".into(),
            left_by: vec!["k".into()],
            right_by: vec!["k".into()],
            direction: bc_ir::AsofDirection::Backward,
            tolerance: None,
            allow_exact_matches: true,
            output: vec![
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "k".into(),
                    alias: "k".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "v".into(),
                    alias: "lv".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Right,
                    name: "v".into(),
                    alias: "rv".into(),
                },
            ],
        };
        let left = vec![batch(&[1, 1, 2, 3], &[10, 25, 40, 5])];
        let right = vec![batch(&[1, 1, 2], &[5, 20, 30])];
        let seq = execute(&plan, &[left.clone(), right.clone()]).unwrap();
        let par = execute_parallel(&plan, &[left, right]).unwrap();
        assert_eq!(rows(&seq), rows(&par));
    }

    /// Grace ASOF join (forced by a tiny budget → both sides partitioned to disk and
    /// joined one bucket pair at a time) must equal the in-memory ASOF — the
    /// mergeable-spill invariant for the new bounded-memory ASOF path.
    /// A spilling ASOF join whose `by` groups are wildly uneven must still equal the
    /// in-memory result.
    ///
    /// The fan-out is sized from the *larger side's total*, which says nothing about how any
    /// one `by` value is distributed — so a hot group leaves a bucket far over the envelope,
    /// and both sides were materialized whole before joining. Re-splitting is legal here for
    /// the hash join's reason plus one more: a nearest-`on` match never crosses a `by` group,
    /// so re-partitioning by the `by` keys keeps every group whole in one sub-bucket and each
    /// sub-pair stays an independent ASOF join.
    ///
    /// The budget is small but not degenerate, so the split is chosen from a measured
    /// partition size rather than by everything trivially exceeding it.
    #[test]
    fn spilling_asof_join_with_skewed_by_groups_matches_in_memory() {
        use bc_ir::{JoinOutputCol, JoinSide};

        let plan = RelOp::AsofJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(RelOp::Scan { source_id: 1 }),
            left_on: "v".into(),
            right_on: "v".into(),
            left_by: vec!["k".into()],
            right_by: vec!["k".into()],
            direction: bc_ir::AsofDirection::Backward,
            tolerance: None,
            allow_exact_matches: true,
            output: vec![
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "k".into(),
                    alias: "k".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "v".into(),
                    alias: "lv".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Right,
                    name: "v".into(),
                    alias: "rv".into(),
                },
            ],
        };

        // One hot `by` group carrying the bulk of both sides, plus cold ones.
        let mut lk = Vec::new();
        let mut lv = Vec::new();
        for i in 0..400 {
            lk.push(7);
            lv.push(i * 2);
        }
        for k in 0..30 {
            lk.push(k);
            lv.push(1000 + k);
        }
        let mut rk = Vec::new();
        let mut rv = Vec::new();
        for i in 0..200 {
            rk.push(7);
            rv.push(i * 3);
        }
        for k in 10..40 {
            rk.push(k);
            rv.push(900 + k);
        }
        let left = vec![batch(&lk, &lv)];
        let right = vec![batch(&rk, &rv)];
        let in_mem = execute(&plan, &[left.clone(), right.clone()]).unwrap();

        let dir = std::env::temp_dir().join(format!("bc_asof_skew_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 2048,
                dir,
                codec: SpillCodec::None,
            }),
            morsel_rows: 64,
            ..ExecOptions::default()
        };
        let spilled = execute_parallel_with(&plan, &[left, right], &opts).unwrap();
        assert_eq!(
            rows(&in_mem),
            rows(&spilled),
            "a re-split ASOF bucket diverged from the in-memory join"
        );
    }

    #[test]
    fn spilling_asof_join_matches_in_memory() {
        use bc_ir::{JoinOutputCol, JoinSide};

        let plan = RelOp::AsofJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(RelOp::Scan { source_id: 1 }),
            left_on: "v".into(),
            right_on: "v".into(),
            left_by: vec!["k".into()],
            right_by: vec!["k".into()],
            direction: bc_ir::AsofDirection::Backward,
            tolerance: None,
            allow_exact_matches: true,
            output: vec![
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "k".into(),
                    alias: "k".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "v".into(),
                    alias: "lv".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Right,
                    name: "v".into(),
                    alias: "rv".into(),
                },
            ],
        };
        // Several `by` groups so partitioning spreads them across buckets.
        let left = vec![batch(&[1, 1, 2, 3, 4, 5], &[10, 25, 40, 5, 7, 9])];
        let right = vec![batch(&[1, 1, 2, 3, 4], &[5, 20, 30, 1, 8])];
        let in_mem = execute(&plan, &[left.clone(), right.clone()]).unwrap();

        let dir = std::env::temp_dir().join(format!("bc_asof_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1, // force grace partitioning
                dir,
                codec: SpillCodec::None,
            }),
            ..ExecOptions::default()
        };
        let spilled = execute_parallel_with(&plan, &[left, right], &opts).unwrap();
        assert_eq!(rows(&in_mem), rows(&spilled), "spilled ASOF mismatch");
    }

    /// A keyless ASOF over a configured envelope it exceeds fails loudly with a typed
    /// error (it cannot grace-partition), instead of risking an OOM.
    #[test]
    fn keyless_asof_over_budget_errors() {
        use bc_ir::{JoinOutputCol, JoinSide};

        let plan = RelOp::AsofJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(RelOp::Scan { source_id: 1 }),
            left_on: "v".into(),
            right_on: "v".into(),
            left_by: vec![],
            right_by: vec![],
            direction: bc_ir::AsofDirection::Backward,
            tolerance: None,
            allow_exact_matches: true,
            output: vec![JoinOutputCol {
                side: JoinSide::Left,
                name: "v".into(),
                alias: "lv".into(),
            }],
        };
        let left = vec![batch(&[1, 2, 3], &[10, 20, 30])];
        let right = vec![batch(&[1, 2, 3], &[5, 15, 25])];
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1, // any real input exceeds this
                dir: std::env::temp_dir(),
                codec: SpillCodec::None,
            }),
            ..ExecOptions::default()
        };
        let err = execute_parallel_with(&plan, &[left, right], &opts).unwrap_err();
        assert!(
            matches!(err, InterpError::MemoryBudgetExceeded { .. }),
            "expected MemoryBudgetExceeded, got {err:?}"
        );
    }

    /// External merge sort (tiny budget + tiny morsels → many spilled runs, then a
    /// k-way merge) must produce the exact same ordering as the in-memory sort.
    #[test]
    fn external_sort_matches_sequential() {
        use bc_expr::Expr;
        use bc_ir::SortKey;

        let plan = RelOp::Sort {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            keys: vec![SortKey {
                expr: Expr::Col { name: "v".into() },
                descending: false,
                nulls_first: false,
            }],
            limit: None,
        };
        // Unique values so the total order is unambiguous (lexsort ties aside).
        let data = vec![
            batch(&[1, 2, 3], &[50, 10, 80]),
            batch(&[4, 5, 6], &[30, 90, 20]),
            batch(&[7, 8, 9], &[70, 40, 60]),
        ];
        let seq = execute(&plan, std::slice::from_ref(&data)).unwrap();

        let dir = std::env::temp_dir().join(format!("bc_sort_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1, // force the external-sort branch
                dir,
                codec: SpillCodec::None,
            }),
            morsel_rows: 2, // tiny morsels → multiple sorted runs to merge
            ..ExecOptions::default()
        };
        let spilled = execute_parallel_with(&plan, &[data], &opts).unwrap();

        // Sort output is ordered: compare the exact value sequence, not a multiset.
        let seq_v = ordered_col(&seq, "v");
        let spill_v = ordered_col(&spilled, "v");
        assert_eq!(seq_v, vec![10, 20, 30, 40, 50, 60, 70, 80, 90]);
        assert_eq!(seq_v, spill_v);
    }

    /// More runs than the merge fan-in forces *multiple* merge passes; the bounded
    /// streaming k-way merge must still equal the in-memory sort exactly (here under
    /// a descending key, exercising the row-encoded order across passes).
    #[test]
    fn external_sort_multipass_matches_sequential() {
        use bc_expr::Expr;
        use bc_ir::SortKey;

        let plan = RelOp::Sort {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            keys: vec![SortKey {
                expr: Expr::Col { name: "v".into() },
                descending: true,
                nulls_first: false,
            }],
            limit: None,
        };
        // 60 unique values in a scrambled order → with morsel_rows=1 that is 60 runs,
        // well above the fan-in (16), so the merge runs several passes.
        let ids: Vec<i64> = (0..60).collect();
        let vals: Vec<i64> = (0..60).map(|i| (i * 37 + 11) % 60).collect();
        let data = vec![batch(&ids, &vals)];
        let seq = execute(&plan, std::slice::from_ref(&data)).unwrap();

        let dir = std::env::temp_dir().join(format!("bc_sort_multipass_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1, // force the external-sort branch
                dir,
                codec: SpillCodec::None,
            }),
            morsel_rows: 1, // one row per morsel → 60 runs → multi-pass merge
            ..ExecOptions::default()
        };
        let spilled = execute_parallel_with(&plan, &[data], &opts).unwrap();

        let seq_v = ordered_col(&seq, "v");
        let spill_v = ordered_col(&spilled, "v");
        let mut expected: Vec<i64> = (0..60).collect();
        expected.reverse(); // descending
        assert_eq!(seq_v, expected);
        assert_eq!(seq_v, spill_v);
    }

    fn ordered_col(batches: &[RecordBatch], name: &str) -> Vec<i64> {
        let mut out = Vec::new();
        for b in batches {
            let i = b.schema().index_of(name).unwrap();
            let a = b.column(i).as_any().downcast_ref::<Int64Array>().unwrap();
            out.extend((0..a.len()).map(|j| a.value(j)));
        }
        out
    }

    /// Grace hash join (tiny budget → forced disk partitioning) must equal the
    /// sequential oracle for every join type, including the outer types whose
    /// unmatched-row emission is the subtle part.
    #[test]
    fn spilling_join_matches_sequential() {
        use bc_ir::{JoinOutputCol, JoinSide, JoinStrategy, JoinType};

        let join_plan = |jt: JoinType| RelOp::HashJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(RelOp::Scan { source_id: 1 }),
            left_keys: vec!["k".into()],
            right_keys: vec!["k".into()],
            join_type: jt,
            output: vec![
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "k".into(),
                    alias: "lk".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "v".into(),
                    alias: "lv".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Right,
                    name: "v".into(),
                    alias: "rv".into(),
                },
            ],
            strategy: JoinStrategy::Hash,
        };
        // Keys overlap partially so inner/left/right/full/semi/anti all differ.
        let left = vec![batch(&[1, 2, 3, 2, 5], &[10, 20, 30, 40, 50])];
        let right = vec![batch(&[2, 3, 3, 4], &[1, 2, 3, 4])];

        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            let plan = join_plan(jt);
            let seq = execute(&plan, &[left.clone(), right.clone()]).unwrap();

            let dir =
                std::env::temp_dir().join(format!("bc_join_spill_{}_{:?}", std::process::id(), jt));
            let opts = ExecOptions {
                agg_spill: Some(SpillOptions {
                    memory_budget_bytes: 1, // force grace partitioning
                    dir,
                    codec: SpillCodec::None,
                }),
                ..ExecOptions::default()
            };
            let spilled =
                execute_parallel_with(&plan, &[left.clone(), right.clone()], &opts).unwrap();
            assert_eq!(rows(&seq), rows(&spilled), "join type {jt:?} mismatch");
        }
    }

    /// A grace join whose buckets are wildly uneven must still equal the sequential oracle.
    ///
    /// The bucket count is sized from the build side's *average* bytes per bucket, so under
    /// key skew one bucket holds far more than its share. Both sides used to be materialized
    /// whole before being joined, so that bucket OOMs at exactly the point spilling was
    /// meant to have prevented it — the standard reason a skewed Spark join dies. The fix
    /// re-splits an over-large bucket with a depth-derived salt, on *both* sides, and this
    /// pins the property that makes that legal: the union of the sub-bucket joins is the
    /// same relation, for every join type.
    ///
    /// The budget is small but not degenerate, so the split path is chosen by the measured
    /// partition size rather than by everything trivially exceeding it, and the skew is
    /// severe enough (one key carrying most rows on both sides) that a re-split cannot
    /// separate the hot key and the depth limit is reached.
    #[test]
    fn spilling_join_with_skewed_buckets_matches_sequential() {
        use bc_ir::{JoinOutputCol, JoinSide, JoinStrategy, JoinType};

        let join_plan = |jt: JoinType| RelOp::HashJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(RelOp::Scan { source_id: 1 }),
            left_keys: vec!["k".into()],
            right_keys: vec!["k".into()],
            join_type: jt,
            output: vec![
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "k".into(),
                    alias: "lk".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "v".into(),
                    alias: "lv".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Right,
                    name: "v".into(),
                    alias: "rv".into(),
                },
            ],
            strategy: JoinStrategy::Hash,
        };

        // One hot key (7) carries the bulk of both sides; a scattering of cold keys, some
        // matching and some not, so the outer/semi/anti emissions all differ.
        let mut lk = Vec::new();
        let mut lv = Vec::new();
        for i in 0..600 {
            lk.push(7);
            lv.push(i);
        }
        for k in 0..40 {
            lk.push(k);
            lv.push(1000 + k);
        }
        let mut rk = Vec::new();
        let mut rv = Vec::new();
        for i in 0..300 {
            rk.push(7);
            rv.push(i);
        }
        for k in 20..60 {
            rk.push(k);
            rv.push(2000 + k);
        }
        let left = vec![batch(&lk, &lv)];
        let right = vec![batch(&rk, &rv)];

        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            let plan = join_plan(jt);
            let seq = execute(&plan, &[left.clone(), right.clone()]).unwrap();
            let dir =
                std::env::temp_dir().join(format!("bc_join_skew_{}_{:?}", std::process::id(), jt));
            let opts = ExecOptions {
                agg_spill: Some(SpillOptions {
                    // Small enough to spill and to leave the hot bucket over budget, large
                    // enough that the cold buckets fit and the split is a real decision.
                    memory_budget_bytes: 4096,
                    dir,
                    codec: SpillCodec::None,
                }),
                morsel_rows: 64,
                ..ExecOptions::default()
            };
            let spilled =
                execute_parallel_with(&plan, &[left.clone(), right.clone()], &opts).unwrap();
            assert_eq!(
                rows(&seq),
                rows(&spilled),
                "skewed grace join diverged from the oracle for join type {jt:?}"
            );
        }
    }

    /// The join honors the runtime pool the same way the aggregate does: a large
    /// static budget (so the build side fits on its own) but a pool already consumed
    /// by another reservation forces the grace hash join — and the result is still
    /// the oracle's. Guards the join's `admit` wiring (it dropped the reservation in
    /// an earlier cut, so a concurrent query couldn't see the build side).
    #[test]
    fn pool_pressure_triggers_join_spill() {
        use bc_ir::{JoinOutputCol, JoinSide, JoinStrategy, JoinType};

        let plan = RelOp::HashJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(RelOp::Scan { source_id: 1 }),
            left_keys: vec!["k".into()],
            right_keys: vec!["k".into()],
            join_type: JoinType::Inner,
            output: vec![
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "k".into(),
                    alias: "lk".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Right,
                    name: "v".into(),
                    alias: "rv".into(),
                },
            ],
            strategy: JoinStrategy::Hash,
        };
        let left = vec![batch(&[1, 2, 3, 2, 5], &[10, 20, 30, 40, 50])];
        let right = vec![batch(&[2, 3, 3, 4], &[1, 2, 3, 4])];
        let seq = execute(&plan, &[left.clone(), right.clone()]).unwrap();

        let pool = MemoryPool::new(64);
        let _held = pool.try_reserve(63).unwrap(); // leave < the build side
        let dir = std::env::temp_dir().join(format!("bc_pool_join_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1 << 30,
                dir,
                codec: SpillCodec::None,
            }),
            pool: Some(Arc::clone(&pool)),
            ..ExecOptions::default()
        };
        let (out, m) = execute_parallel_with_metrics(&plan, &[left, right], &opts).unwrap();
        let join = m.ops.iter().find(|o| o.kind == "hash_join").unwrap();
        assert!(join.spilled, "pool pressure must force the grace hash join");
        assert_eq!(rows(&seq), rows(&out));
    }

    /// Metrics are a pure side-channel: the metered executor returns batches
    /// identical to the plain one, and records exactly one `OpMetric` per plan
    /// node with the expected pre-order ids, kinds, and row counts.
    #[test]
    fn metered_matches_unmetered_and_records_ops() {
        use bc_expr::{BinaryOp, Expr, Literal};

        // Filter(Scan): pre-order ids 0 (filter), 1 (scan).
        let plan = RelOp::Filter {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            predicate: Expr::Binary {
                op: BinaryOp::Gt,
                left: Box::new(Expr::Col { name: "v".into() }),
                right: Box::new(Expr::Lit {
                    value: Literal::Int(25),
                }),
            },
        };
        let data = vec![batch(&[1, 2, 3, 4], &[10, 20, 30, 40])];

        let plain = execute_parallel(&plan, std::slice::from_ref(&data)).unwrap();
        let (metered, m) =
            execute_parallel_with_metrics(&plan, &[data], &ExecOptions::default()).unwrap();
        assert_eq!(
            rows(&plain),
            rows(&metered),
            "metrics must not change results"
        );

        assert_eq!(m.ops.len(), 2, "one metric per node");
        let filter = m.ops.iter().find(|o| o.kind == "filter").unwrap();
        let scan = m.ops.iter().find(|o| o.kind == "scan").unwrap();
        assert_eq!(filter.op_id, 0, "filter is pre-order root");
        assert_eq!(scan.op_id, 1, "scan follows its parent");
        assert_eq!(scan.rows_out, 4);
        assert_eq!(filter.rows_in, 4);
        assert_eq!(filter.rows_out, 2, "v > 25 keeps 30,40");
    }

    /// The spill decision — silent in the result — is now observable in metrics:
    /// a tiny budget forces grace partitioning and the aggregate's metric flags it.
    #[test]
    fn spill_flag_observable() {
        use bc_expr::Expr;
        use bc_ir::{AggFunc, AggregateItem, ProjectionItem};

        let plan = RelOp::Aggregate {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            group_keys: vec![ProjectionItem {
                expr: Expr::Col { name: "k".into() },
                alias: "k".into(),
            }],
            aggregates: vec![AggregateItem {
                func: AggFunc::Sum,
                input: Some(Expr::Col { name: "v".into() }),
                input2: None,
                alias: "s".into(),
                param: None,
            }],
        };
        let data = vec![batch(&[1, 2, 1, 3, 2, 1], &[10, 20, 30, 40, 50, 60])];

        // No budget → no spill.
        let (_, in_mem) = execute_parallel_with_metrics(
            &plan,
            std::slice::from_ref(&data),
            &ExecOptions::default(),
        )
        .unwrap();
        let agg = in_mem.ops.iter().find(|o| o.kind == "aggregate").unwrap();
        assert!(!agg.spilled, "no envelope means no spill");

        // Tiny budget → forced grace partitioning.
        let dir = std::env::temp_dir().join(format!("bc_metric_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1,
                dir,
                codec: SpillCodec::None,
            }),
            ..ExecOptions::default()
        };
        let (_, spilled) = execute_parallel_with_metrics(&plan, &[data], &opts).unwrap();
        let agg = spilled.ops.iter().find(|o| o.kind == "aggregate").unwrap();
        assert!(agg.spilled, "tiny budget must trip the spill flag");
    }

    /// The broadcast strategy must produce the same relation as the default hash
    /// strategy (= the oracle) for every join type — it only changes data movement.
    #[test]
    fn broadcast_join_matches_oracle() {
        use bc_ir::{JoinOutputCol, JoinSide, JoinStrategy, JoinType};

        let join_plan = |jt: JoinType, strategy: JoinStrategy| RelOp::HashJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(RelOp::Scan { source_id: 1 }),
            left_keys: vec!["k".into()],
            right_keys: vec!["k".into()],
            join_type: jt,
            output: vec![
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "k".into(),
                    alias: "lk".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "v".into(),
                    alias: "lv".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Right,
                    name: "v".into(),
                    alias: "rv".into(),
                },
            ],
            strategy,
        };
        // Large-ish left (the probe side), small right (the broadcast side), with
        // duplicate keys on both sides so every join type is exercised non-trivially.
        let left = vec![batch(
            &[1, 2, 3, 2, 5, 3, 7, 2, 4, 6],
            &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        )];
        let right = vec![batch(&[2, 3, 3, 4], &[1, 2, 3, 4])];

        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            let oracle = execute(
                &join_plan(jt, JoinStrategy::Hash),
                &[left.clone(), right.clone()],
            )
            .unwrap();
            let bcast = execute_parallel_with(
                &join_plan(jt, JoinStrategy::Broadcast),
                &[left.clone(), right.clone()],
                &ExecOptions::default(),
            )
            .unwrap();
            assert_eq!(
                rows(&oracle),
                rows(&bcast),
                "broadcast join type {jt:?} mismatch"
            );
        }
    }

    /// A broadcast join whose probe side is empty must still emit **one zero-row batch**,
    /// not zero batches.
    ///
    /// A batch is the only carrier of a schema. The chunked probe path produces one batch
    /// per probe row-range, so an empty probe yields no chunks and therefore no batches —
    /// and every downstream pipeline breaker (join / aggregate / distinct) materializes
    /// its input and fails with `EmptyJoinInput` when handed none. Empty intermediates are
    /// routine (an incremental batch with no changes, a filter that matches nothing), so
    /// the invariant is: an operator whose input carried a schema returns a batch.
    #[test]
    fn broadcast_join_over_an_empty_probe_still_carries_a_schema() {
        use bc_ir::{JoinOutputCol, JoinSide, JoinStrategy, JoinType};

        let join_plan = |jt: JoinType| RelOp::HashJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(RelOp::Scan { source_id: 1 }),
            left_keys: vec!["k".into()],
            right_keys: vec!["k".into()],
            join_type: jt,
            output: vec![JoinOutputCol {
                side: JoinSide::Left,
                name: "k".into(),
                alias: "lk".into(),
            }],
            strategy: JoinStrategy::Broadcast,
        };
        let empty = vec![batch(&[], &[])];
        let full = vec![batch(&[2, 3], &[1, 2])];

        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            // `Right` drives from the right, so give each join type an empty probe side.
            let (l, r) = if matches!(jt, JoinType::Right) {
                (full.clone(), empty.clone())
            } else {
                (empty.clone(), full.clone())
            };
            let out = execute_parallel_with(
                &join_plan(jt),
                &[l.clone(), r.clone()],
                &ExecOptions::default(),
            )
            .unwrap();
            assert!(
                !out.is_empty(),
                "{jt:?}: empty probe produced no batch, losing the schema"
            );
            assert_eq!(out[0].schema().field(0).name(), "lk", "{jt:?}");
            // The relation itself is still whatever the sequential oracle says — a `Full`
            // join over an empty probe legitimately emits the other side's unmatched rows.
            let oracle = execute(&join_plan(jt), &[l, r]).unwrap();
            assert_eq!(rows(&oracle), rows(&out), "{jt:?}: relation changed");
        }
    }

    /// The sort-merge strategy must produce the same relation as the default hash
    /// strategy (= the oracle) for every join type, through the parallel executor
    /// (so it exercises the per-bucket sort-merge after the hash shuffle).
    #[test]
    fn sort_merge_join_matches_oracle() {
        use bc_ir::{JoinOutputCol, JoinSide, JoinStrategy, JoinType};

        let join_plan = |jt: JoinType, strategy: JoinStrategy| RelOp::HashJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(RelOp::Scan { source_id: 1 }),
            left_keys: vec!["k".into()],
            right_keys: vec!["k".into()],
            join_type: jt,
            output: vec![
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "k".into(),
                    alias: "lk".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "v".into(),
                    alias: "lv".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Right,
                    name: "v".into(),
                    alias: "rv".into(),
                },
            ],
            strategy,
        };
        // Duplicate keys on both sides exercise the equal-key cross product.
        let left = vec![batch(&[2, 1, 2, 5, 3, 2], &[10, 20, 30, 40, 50, 60])];
        let right = vec![batch(&[2, 3, 3, 4, 2], &[1, 2, 3, 4, 5])];

        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            let oracle = execute(
                &join_plan(jt, JoinStrategy::Hash),
                &[left.clone(), right.clone()],
            )
            .unwrap();
            let smj = execute_parallel_with(
                &join_plan(jt, JoinStrategy::SortMerge),
                &[left.clone(), right.clone()],
                &ExecOptions::default(),
            )
            .unwrap();
            assert_eq!(
                rows(&oracle),
                rows(&smj),
                "sort-merge join type {jt:?} mismatch"
            );
        }
    }

    /// A heavily skewed join over a *tiny* build side must produce the sequential oracle's
    /// relation — and it now does so on the **shared-build** path, not the salted shuffle.
    ///
    /// That is the point, not a regression. Salting exists to rescue the shuffle: a hot key
    /// sends all its rows to one bucket, and that bucket's join becomes a straggler. The
    /// shared-build path never buckets anything — it holds one table and streams the probe
    /// morsel by morsel, and morsels are independent — so a hot *probe* key cannot make a
    /// straggler in the first place. Skew-immunity by construction beats skew-repair.
    /// (Spreading a 3-row dimension table across 8 buckets was never going to help anyway.)
    ///
    /// Salting is still exercised where the shuffle still runs:
    /// `skewed_join_with_string_keys_still_salts` (below) and
    /// `skewed_right_join_matches_oracle_and_salts` (a `Right` join, which the streaming path
    /// cannot serve — it must reconcile unmatched build rows across every morsel).
    #[test]
    fn skewed_join_takes_the_shared_build_path_and_matches_the_oracle() {
        use bc_ir::{JoinOutputCol, JoinSide, JoinType};

        // Left: ~80k rows of the hot key (1) with unique values, plus a little cold
        // data. Right: a one-row-per-key dimension.
        let hot = SKEW_MIN_BUCKET_ROWS + 5_000;
        let mut keys = vec![1i64; hot];
        keys.extend([2, 3, 2, 3]);
        let vals: Vec<i64> = (0..keys.len() as i64).collect(); // unique → rows are distinct
        let left = vec![batch(&keys, &vals)];
        let right = vec![batch(&[1, 2, 3], &[1000, 2000, 3000])];

        let plan = RelOp::HashJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(RelOp::Scan { source_id: 1 }),
            left_keys: vec!["k".into()],
            right_keys: vec!["k".into()],
            join_type: JoinType::Inner,
            output: vec![
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "v".into(),
                    alias: "lv".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Right,
                    name: "v".into(),
                    alias: "rv".into(),
                },
            ],
            strategy: bc_ir::JoinStrategy::Hash,
        };

        let oracle = execute(&plan, &[left.clone(), right.clone()]).unwrap();
        // Force 8 workers so bucket sizes (and the skew threshold) are deterministic.
        let opts = ExecOptions {
            parallelism: 8,
            ..ExecOptions::default()
        };
        let (out, metrics) = execute_parallel_with_metrics(&plan, &[left, right], &opts).unwrap();

        assert_eq!(rows(&oracle), rows(&out), "skewed join result mismatch");
        let join_backend = metrics
            .ops
            .iter()
            .find(|m| m.kind == "hash_join")
            .map(|m| m.backend);
        assert_eq!(
            join_backend,
            Some("interp-shared"),
            "a tiny int64 build should be held as one table, not shuffled"
        );
    }

    /// The salting machinery itself, still on an *inner* join: a `Utf8` key is a shape the
    /// streaming path cannot serve (it fast-paths integer keys only), so this falls through to
    /// the shuffle — where a hot key really does concentrate one bucket, and the salt is what
    /// spreads it. Same relation as the sequential oracle, and the skew path actually taken.
    #[test]
    fn skewed_join_with_string_keys_still_salts() {
        use arrow::array::StringArray;
        use bc_ir::{JoinOutputCol, JoinSide, JoinType};

        fn str_batch(keys: &[&str], vals: &[i64]) -> RecordBatch {
            RecordBatch::try_from_iter(vec![
                ("k", Arc::new(StringArray::from(keys.to_vec())) as ArrayRef),
                ("v", Arc::new(Int64Array::from(vals.to_vec())) as ArrayRef),
            ])
            .unwrap()
        }

        let hot = SKEW_MIN_BUCKET_ROWS + 5_000;
        let mut keys: Vec<&str> = vec!["hot"; hot];
        keys.extend(["a", "b", "a", "b"]);
        let vals: Vec<i64> = (0..keys.len() as i64).collect();
        let left = vec![str_batch(&keys, &vals)];
        let right = vec![str_batch(&["hot", "a", "b"], &[1000, 2000, 3000])];

        let plan = RelOp::HashJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(RelOp::Scan { source_id: 1 }),
            left_keys: vec!["k".into()],
            right_keys: vec!["k".into()],
            join_type: JoinType::Inner,
            output: vec![
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "v".into(),
                    alias: "lv".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Right,
                    name: "v".into(),
                    alias: "rv".into(),
                },
            ],
            strategy: bc_ir::JoinStrategy::Hash,
        };

        let oracle = execute(&plan, &[left.clone(), right.clone()]).unwrap();
        let opts = ExecOptions {
            parallelism: 8,
            ..ExecOptions::default()
        };
        let (out, metrics) = execute_parallel_with_metrics(&plan, &[left, right], &opts).unwrap();

        assert_eq!(
            rows(&oracle),
            rows(&out),
            "skewed string join result mismatch"
        );
        let join_backend = metrics
            .ops
            .iter()
            .find(|m| m.kind == "hash_join")
            .map(|m| m.backend);
        assert_eq!(join_backend, Some("interp-skew"), "skew path was not taken");
    }

    /// A skewed RIGHT join (hot key on the driving *right* side) salts via the
    /// flip-to-left path and matches the sequential oracle.
    #[test]
    fn skewed_right_join_matches_oracle_and_salts() {
        use bc_ir::{JoinOutputCol, JoinSide, JoinType};

        // Right side is the hot/driving side; left is a one-row-per-key dimension.
        let hot = SKEW_MIN_BUCKET_ROWS + 5_000;
        let mut rkeys = vec![1i64; hot];
        rkeys.extend([2, 3, 4]); // key 4 has no left match → null-left in the result
        let rvals: Vec<i64> = (0..rkeys.len() as i64).collect();
        let right = vec![batch(&rkeys, &rvals)];
        let left = vec![batch(&[1, 2, 3], &[1000, 2000, 3000])];

        let plan = RelOp::HashJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(RelOp::Scan { source_id: 1 }),
            left_keys: vec!["k".into()],
            right_keys: vec!["k".into()],
            join_type: JoinType::Right,
            output: vec![
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "v".into(),
                    alias: "lv".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Right,
                    name: "v".into(),
                    alias: "rv".into(),
                },
            ],
            strategy: bc_ir::JoinStrategy::Hash,
        };

        let oracle = execute(&plan, &[left.clone(), right.clone()]).unwrap();
        let opts = ExecOptions {
            parallelism: 8,
            ..ExecOptions::default()
        };
        let (out, metrics) = execute_parallel_with_metrics(&plan, &[left, right], &opts).unwrap();
        assert_eq!(
            rows(&oracle),
            rows(&out),
            "skewed right join result mismatch"
        );
        let backend = metrics
            .ops
            .iter()
            .find(|m| m.kind == "hash_join")
            .map(|m| m.backend);
        assert_eq!(
            backend,
            Some("interp-skew"),
            "right-join skew path not taken"
        );
    }

    // --- Phase 2: runtime memory backstop (the shared pool) -------------------

    fn sum_by_k_plan() -> RelOp {
        use bc_expr::Expr;
        use bc_ir::{AggFunc, AggregateItem, ProjectionItem};
        RelOp::Aggregate {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            group_keys: vec![ProjectionItem {
                expr: Expr::Col { name: "k".into() },
                alias: "k".into(),
            }],
            aggregates: vec![AggregateItem {
                func: AggFunc::Sum,
                input: Some(Expr::Col { name: "v".into() }),
                input2: None,
                alias: "s".into(),
                param: None,
            }],
        }
    }

    /// The runtime backstop: with a *large* static budget (so the per-operator
    /// estimate alone would run in memory) but a pool whose headroom is already
    /// consumed by another live reservation, the aggregate must spill — and still
    /// equal the sequential oracle. This is what a static pre-execution estimate
    /// cannot do.
    #[test]
    fn pool_pressure_triggers_aggregate_spill() {
        let plan = sum_by_k_plan();
        let data = vec![batch(&[1, 2, 1, 3, 2, 1], &[10, 20, 30, 40, 50, 60])];
        let seq = execute(&plan, std::slice::from_ref(&data)).unwrap();

        let pool = MemoryPool::new(64);
        // Another operator already holds all but 1 byte, so this aggregate's
        // footprint can't be admitted even though the static budget is huge.
        let _held = pool.try_reserve(63).unwrap();
        let dir = std::env::temp_dir().join(format!("bc_pool_pressure_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1 << 30, // static check would say "in memory"
                dir,
                codec: SpillCodec::None,
            }),
            pool: Some(Arc::clone(&pool)),
            ..ExecOptions::default()
        };
        let (out, m) = execute_parallel_with_metrics(&plan, &[data], &opts).unwrap();
        let agg = m.ops.iter().find(|o| o.kind == "aggregate").unwrap();
        assert!(
            agg.spilled,
            "pool pressure (not the static estimate) must force the spill"
        );
        assert_eq!(rows(&seq), rows(&out));
    }

    /// With pool headroom, the same operator runs in memory (no spurious spill) and
    /// every reservation is released afterward — RAII returns the budget, so a
    /// multi-operator plan never leaks the envelope.
    #[test]
    fn pool_with_headroom_runs_in_memory_and_releases() {
        let plan = sum_by_k_plan();
        let data = vec![batch(&[1, 2, 1, 3, 2, 1], &[10, 20, 30, 40, 50, 60])];
        let seq = execute(&plan, std::slice::from_ref(&data)).unwrap();

        let pool = MemoryPool::new(1 << 30);
        let dir = std::env::temp_dir().join(format!("bc_pool_headroom_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1 << 30,
                dir,
                codec: SpillCodec::None,
            }),
            pool: Some(Arc::clone(&pool)),
            ..ExecOptions::default()
        };
        let (out, m) = execute_parallel_with_metrics(&plan, &[data], &opts).unwrap();
        let agg = m.ops.iter().find(|o| o.kind == "aggregate").unwrap();
        assert!(!agg.spilled, "ample headroom means no spill");
        assert_eq!(rows(&seq), rows(&out));
        assert_eq!(pool.used(), 0, "all reservations released after execution");
    }

    /// A window with no PARTITION BY cannot grace-partition, so when it can't fit
    /// it fails with a typed, catchable error rather than OOM-killing the process.
    #[test]
    fn global_window_over_budget_errors() {
        use bc_expr::Expr;
        use bc_ir::{WindowFn, WindowFunc};
        let plan = RelOp::Window {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            partition_keys: vec![], // global window — unspillable
            order_keys: vec![],
            functions: vec![WindowFunc {
                func: WindowFn::Sum,
                input: Some(Expr::Col { name: "v".into() }),
                offset: 1,
                frame: None,
                alpha: None,
                half_life: None,
                alias: "s".into(),
            }],
            rank_limit: None,
        };
        let data = vec![batch(&[1, 2, 3], &[10, 20, 30])];

        let pool = MemoryPool::new(8);
        let _held = pool.try_reserve(8).unwrap(); // pool full
        let dir = std::env::temp_dir().join(format!("bc_global_win_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1 << 30,
                dir,
                codec: SpillCodec::None,
            }),
            pool: Some(Arc::clone(&pool)),
            ..ExecOptions::default()
        };
        let err = execute_parallel_with(&plan, &[data], &opts).unwrap_err();
        assert!(
            matches!(err, InterpError::MemoryBudgetExceeded { .. }),
            "global window over budget must raise MemoryBudgetExceeded, got {err:?}"
        );

        // The same global window runs fine with no envelope (default fast path).
        let data = vec![batch(&[1, 2, 3], &[10, 20, 30])];
        execute_parallel_with(&plan, &[data], &ExecOptions::default()).unwrap();
    }
}
