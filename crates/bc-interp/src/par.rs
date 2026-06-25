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
use bc_ir::{EngineConfig, RelOp};
use bc_resource::{MemoryPool, MemoryReservation};
use bc_runtime::agg::spill::{combine_finalize_spilling, DiskSpillStore};
use bc_runtime::{agg, shuffle};
use rayon::prelude::*;

use crate::error::InterpError;
use crate::join_par::{
    broadcast_join, is_skewed_bucket, is_skewed_bucket_bytes, skew_salting_eligible,
    spilling_asof_join, spilling_hash_join_streaming,
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
#[derive(Clone)]
pub struct SpillOptions {
    /// Soft cap on bytes of in-memory operator state before grace partitioning.
    pub memory_budget_bytes: usize,
    /// Directory for spill files (one IPC file per hash partition).
    pub dir: PathBuf,
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
fn admit(opts: &ExecOptions, op_id: u32, estimate_bytes: usize) -> Admit {
    match opts.pool.as_ref() {
        // The pool accounts *actual* bytes, so it is the spill authority: reserve the
        // footprint cooperatively (a full pool first asks the largest *other* consumer
        // — operator or concurrent query — to spill, stranding this one only if that
        // still isn't enough), and spill only when even that can't admit it. Deciding
        // on actual bytes, not the per-operator *estimate*, is what stops a spurious
        // out-of-core pass when transient state exceeds a small estimate but still fits
        // RAM — e.g. a low-cardinality / global aggregate's pre-combine partials, whose
        // `op_budget` is the (tiny) combined-output size. The per-op budget still sizes
        // the grace fan-out once a spill is chosen.
        Some(pool) => match pool.try_reserve_cooperative(estimate_bytes) {
            Ok(reservation) => Admit::InMemory(Some(reservation)),
            // Pool full even after cooperative spilling: spill if there is a path to
            // spill to, else best-effort in memory (a pool without an envelope can't
            // strand the operator).
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
    // `parallelism == 0` uses rayon's global pool (all cores); a positive value
    // runs the whole plan inside a scoped pool of that width, so the control
    // plane's `EngineConfig.parallelism` bounds the worker count (and the
    // hash-shuffle bucket count, which keys off `current_num_threads`).
    //
    // The plan walk that records metrics is itself single-threaded — only the
    // per-operator work fans out across rayon and is fully joined before each
    // `OpMetric` is recorded — so a plain `&mut ExecMetrics` is race-free.
    let out = if opts.parallelism > 0 {
        let pool = pool_for(opts.parallelism)?;
        pool.install(|| exec(plan, sources, opts, &mut m, &mut ids))
    } else {
        exec(plan, sources, opts, &mut m, &mut ids)
    }?;
    Ok((out, m))
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
fn pool_for(width: usize) -> Result<Arc<rayon::ThreadPool>, InterpError> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let pools = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = pools
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(pool) = guard.get(&width) {
        return Ok(Arc::clone(pool));
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(width)
            .build()
            .map_err(|_| InterpError::ThreadPool(width))?,
    );
    guard.insert(width, Arc::clone(&pool));
    Ok(pool)
}

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
    // Pre-order id (parents before children) — same numbering the sequential
    // executor and the Python control plane use.
    let op_id = ids.next();
    // Fuse a run of ≥2 linear streaming ops (Filter/Project) into one per-morsel pass.
    // Off unless the control plane opted in; the result is a relation-level no-op
    // (same rows, same order — verified against the sequential oracle).
    if opts.fuse_linear && is_fusable(plan) && fusable_input(plan).is_some_and(is_fusable) {
        return exec_fused(plan, op_id, sources, opts, m, ids);
    }
    match plan {
        RelOp::Scan { source_id } => {
            let t0 = Stopwatch::start();
            let batches = sources.get(*source_id).ok_or(InterpError::UnknownSource {
                source_id: *source_id,
                available: sources.len(),
            })?;
            let out = ops::morselize(batches, opts.morsel_target());
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
            let out: Vec<RecordBatch> = parts
                .par_iter()
                .map(|b| ops::filter_batch_jit(b, predicate, &jit))
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
                    .map(|e| ops::try_compile(&e.expr, sample))
                    .collect(),
                None => exprs.iter().map(|_| None).collect(),
            };
            let backend = backend_tag(&jits.iter().map(|j| j.is_some()).collect::<Vec<_>>());
            let out: Vec<RecordBatch> = parts
                .par_iter()
                .map(|b| ops::project_batch_jit(b, exprs, &jits))
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
        } => {
            let parts = exec(input, sources, opts, m, ids)?;
            let rows_in = count_rows(&parts);
            let t0 = Stopwatch::start();
            let out: Vec<RecordBatch> = parts
                .par_iter()
                .map(|b| ops::unnest_batch(b, column, alias))
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
                .map(|b| ops::unpivot_batch(b, index, on, variable_name, value_name))
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
            let t0 = Stopwatch::start();
            let out: Vec<RecordBatch> = match n {
                // Fixed-count: a breaker over all morsels (global n-smallest hashes).
                Some(k) => ops::sample_n_batches(&parts, *k, *seed)?,
                None => parts
                    .par_iter()
                    .map(|b| ops::sample_batch(b, *fraction, *seed))
                    .collect::<Result<_, InterpError>>()?,
            };
            push_metric(m, op_id, "sample", rows_in, &out, t0, false, "interp");
            Ok(out)
        }

        RelOp::Aggregate {
            input,
            group_keys,
            aggregates,
        } => {
            let parts = exec(input, sources, opts, m, ids)?;
            if parts.is_empty() {
                return Err(InterpError::EmptyAggregateInput);
            }
            let rows_in = count_rows(&parts);
            let t0 = Stopwatch::start();
            let funcs = ops::agg_funcs(aggregates);
            let partials: Vec<agg::Partial> = parts
                .par_iter()
                .map(|b| ops::eval_partial(b, group_keys, aggregates))
                .collect::<Result<_, InterpError>>()?;

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
                    // A lone median/quantile, n_unique, or mode spills out-of-core with
                    // bounded memory (their per-group value list can exceed memory on a
                    // hot key); a *mix* of such a value-list aggregate with constant-state
                    // ones (`median(x), sum(y)`) is bounded compositionally by
                    // `try_bounded_mixed_spill`; every other shape uses the in-memory
                    // grace path. At most one dispatch does work — the rest return `None`.
                    let bounded =
                        ops::try_bounded_quantile_spill(&parts, group_keys, aggregates, &sp.dir)?
                            .or(ops::try_bounded_distinct_spill(
                                &parts, group_keys, aggregates, &sp.dir,
                            )?)
                            .or(ops::try_bounded_mode_spill(
                                &parts, group_keys, aggregates, &sp.dir,
                            )?)
                            .or(ops::try_bounded_histogram_spill(
                                &parts, group_keys, aggregates, &sp.dir,
                            )?)
                            .or(ops::try_bounded_mixed_spill(
                                &parts,
                                group_keys,
                                aggregates,
                                &sp.dir,
                                sp.memory_budget_bytes,
                            )?);
                    match bounded {
                        Some((gc, ac)) => (gc, ac),
                        None => {
                            let p = grace_partitions(&partials, sp.memory_budget_bytes);
                            let mut store =
                                DiskSpillStore::new(sp.dir.join(format!("agg-{p}p")), p)?;
                            let res = combine_finalize_spilling(partials, &funcs, &mut store)?;
                            (res.group_columns, res.agg_columns)
                        }
                    }
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
            push_metric(m, op_id, "aggregate", rows_in, &out, t0, spilled, "interp");
            Ok(out)
        }

        RelOp::Sort { input, keys, limit } => {
            let parts = exec(input, sources, opts, m, ids)?;
            let rows_in = count_rows(&parts);
            let t0 = Stopwatch::start();
            if parts.is_empty() {
                push_metric(m, op_id, "sort", rows_in, &[], t0, false, "interp");
                return Ok(Vec::new());
            }
            let out = match limit {
                // Top-N: each morsel computes its local top-k in parallel (cheap),
                // then we merge only the P×k survivors and take the global top-k —
                // no full-input materialization. This is also mergeable, so it is
                // the same shape the distributed top-N uses.
                Some(k) => {
                    let locals: Vec<RecordBatch> = parts
                        .par_iter()
                        .map(|b| ops::sort_batch(b, keys, Some(*k)))
                        .collect::<Result<_, InterpError>>()?;
                    let merged = ops::materialize(&locals)?;
                    vec![ops::sort_batch(&merged, keys, Some(*k))?]
                }
                // Full sort: out-of-core (spill sorted runs + k-way merge) when the
                // input exceeds the budget or the pool can't admit it; else
                // in-memory.
                None => {
                    let bytes = batch_bytes(&parts);
                    match admit(opts, op_id, bytes as usize) {
                        Admit::Spill => {
                            let sp = opts.agg_spill.as_ref().expect("spill implies an envelope");
                            // Bound each sorted run to one morsel before spilling: an
                            // oversized upstream batch (a join/aggregate output that was
                            // never re-morselized) would otherwise become a single run
                            // larger than the working-set budget. The merge phase is
                            // already fan-in bounded, so this caps peak sort memory.
                            let parts = ops::remorselize(parts, opts.morsel_target());
                            ops::external_merge_sort(
                                parts,
                                keys,
                                &sp.dir.join("sort"),
                                opts.tuning.sort_merge_fanin,
                            )?
                        }
                        Admit::InMemory(_reservation) => {
                            let combined = ops::materialize(&parts)?;
                            vec![ops::sort_batch(&combined, keys, None)?]
                        }
                    }
                }
            };
            push_metric(m, op_id, "sort", rows_in, &out, t0, false, "interp");
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
            let t0 = Stopwatch::start();
            let bytes = batch_bytes(&parts);
            let has_keys = !partition_keys.is_empty();
            let (out, spill) = match admit(opts, op_id, bytes as usize) {
                // Grace-partition by PARTITION BY keys and run the kernel one bucket
                // at a time (bounded memory).
                Admit::Spill if has_keys => {
                    let global = opts.agg_spill.as_ref().expect("spill implies an envelope");
                    let budget = opts.op_budget(op_id).unwrap_or(global.memory_budget_bytes);
                    let out = crate::window_spill::window_spilling(
                        &parts,
                        partition_keys,
                        order_keys,
                        functions,
                        *rank_limit,
                        budget,
                        &global.dir,
                    )?;
                    (out, true)
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
                    (out, false)
                }
            };
            push_metric(m, op_id, "window", rows_in, &out, t0, spill, "interp");
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
            backward,
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
            let rows_in = count_rows(&left_batches) + count_rows(&right_batches);
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
                    &left, &right, left_on, right_on, left_by, right_by, *backward, output,
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
                            &left, &right, left_on, right_on, left_by, right_by, *backward, output,
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
                                    &lb[i], &rb[i], left_on, right_on, left_by, right_by,
                                    *backward, output,
                                )
                            })
                            .collect::<Result<Vec<_>, InterpError>>()?
                    }
                }
            };
            push_metric(m, op_id, "asof_join", rows_in, &out, t0, spilled, "interp");
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
            let rows_in = count_rows(&left_batches) + count_rows(&right_batches);
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
                    let out = spilling_hash_join_streaming(
                        &left_batches,
                        &right_batches,
                        left_keys,
                        right_keys,
                        *join_type,
                        output,
                        sp,
                    )?;
                    push_metric(m, op_id, "hash_join", rows_in, &out, t0, true, "interp");
                    return Ok(out);
                }
                Admit::InMemory(reservation) => reservation,
            };

            // Fits in memory: now materialize both sides for the in-memory path.
            let left = ops::materialize(&left_batches)?;
            let right = ops::materialize(&right_batches)?;

            // Broadcast: the planner found the right side small enough to replicate.
            // Probe the large left side without shuffling it (no key partitioning).
            if *strategy == bc_ir::JoinStrategy::Broadcast {
                let out = broadcast_join(&left, &right, left_keys, right_keys, *join_type, output)?;
                push_metric(m, op_id, "hash_join", rows_in, &out, t0, false, "interp");
                return Ok(out);
            }

            let p = rayon::current_num_threads().max(1);
            let li = ops::key_indices(&left, left_keys)?;
            let ri = ops::key_indices(&right, right_keys)?;
            let lb = shuffle::partition_by_keys(&left, &li, p)?;
            let rb = shuffle::partition_by_keys(&right, &ri, p)?;

            // Skew handling: a hot key sends all its rows to one bucket, making that
            // per-bucket join a straggler. Detect it for free from the partition
            // sizes (no extra pass) and spread the over-large bucket's *driving*
            // (probe) side across worker chunks against its (co-partitioned) build
            // bucket — the chunked join `broadcast_join` uses. The driving side is
            // the right for a `Right` join, the left otherwise; `Full` is ineligible.
            // Every bucket still computes the same relation.
            let salt = skew_salting_eligible(*join_type);
            let driving_is_right = matches!(*join_type, bc_ir::JoinType::Right);
            let driving_bucket = |i: usize| if driving_is_right { &rb[i] } else { &lb[i] };
            let driving_side = if driving_is_right { &right } else { &left };
            let avg = driving_side.num_rows() / p.max(1);
            let avg_bytes = driving_side.get_array_memory_size() / p.max(1);
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
                        broadcast_join(&lb[i], &rb[i], left_keys, right_keys, *join_type, output)
                    } else {
                        Ok(vec![ops::join_batches_with(
                            &lb[i],
                            &rb[i],
                            left_keys,
                            right_keys,
                            *join_type,
                            output,
                            *strategy,
                            &opts.tuning,
                        )?])
                    }
                })
                .collect::<Result<Vec<_>, InterpError>>()?
                .into_iter()
                .flatten()
                .collect();
            let backend = match (skewed_any, *strategy == bc_ir::JoinStrategy::SortMerge) {
                (true, _) => "interp-skew",
                (false, true) => "interp-smj",
                (false, false) => "interp",
            };
            push_metric(m, op_id, "hash_join", rows_in, &out, t0, false, backend);
            Ok(out)
        }

        RelOp::Distinct { input } => {
            let parts = exec(input, sources, opts, m, ids)?;
            let rows_in = count_rows(&parts);
            let t0 = Stopwatch::start();
            let (batch, spilled) = distinct(&parts, opts, op_id)?;
            let out = vec![batch];
            push_metric(m, op_id, "distinct", rows_in, &out, t0, spilled, "interp");
            Ok(out)
        }

        RelOp::Union {
            inputs,
            distinct: dedup,
        } => {
            let mut all = Vec::new();
            for inp in inputs {
                all.extend(exec(inp, sources, opts, m, ids)?);
            }
            let rows_in = count_rows(&all);
            let t0 = Stopwatch::start();
            let (out, spilled) = if *dedup {
                let (batch, sp) = distinct(&all, opts, op_id)?;
                (vec![batch], sp)
            } else {
                (all, false)
            };
            push_metric(m, op_id, "union", rows_in, &out, t0, spilled, "interp");
            Ok(out)
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

/// One compiled stage of a fused linear pipeline: a per-morsel operator with its
/// expression(s) compiled once (against a representative sample) and reused across
/// every morsel — the same compile-once-per-operator discipline as the unfused path.
enum FusedStage<'a> {
    Filter {
        op_id: u32,
        predicate: &'a bc_expr::Expr,
        jit: ops::Jit,
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
            FusedStage::Filter { predicate, jit, .. } => ops::filter_batch_jit(b, predicate, jit),
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
                backend,
            }
        }
        RelOp::Project { exprs, .. } => {
            let jits: Vec<ops::Jit> = match sample {
                Some(s) => exprs.iter().map(|e| ops::try_compile(&e.expr, s)).collect(),
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
    let cpu = t0.cpu_ns() / n as u64;
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
            cpu_ns: cpu,
            threads,
            peak_bytes,
            spilled: false,
            backend: stage.backend(),
        });
    }
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
    m.record(OpMetric {
        op_id,
        kind,
        rows_in,
        rows_out: count_rows(out),
        elapsed_ns: t0.elapsed_ns(),
        cpu_ns: t0.cpu_ns(),
        threads: rayon::current_num_threads().max(1) as u32,
        peak_bytes: batch_bytes(out),
        spilled,
        backend,
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
    let total = partial_state_bytes(partials);
    let budget = budget_bytes.max(1);
    total.div_ceil(budget).max(2)
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
fn distinct(
    parts: &[RecordBatch],
    opts: &ExecOptions,
    op_id: u32,
) -> Result<(RecordBatch, bool), InterpError> {
    if parts.is_empty() {
        return Err(InterpError::EmptyAggregateInput);
    }
    let schema = parts[0].schema();
    let partials: Vec<agg::Partial> = parts
        .par_iter()
        .map(ops::distinct_partial)
        .collect::<Result<_, InterpError>>()?;
    let state_bytes = partial_state_bytes(&partials);
    let (group_columns, spilled) = match admit(opts, op_id, state_bytes) {
        Admit::Spill => {
            let global = opts.agg_spill.as_ref().expect("spill implies an envelope");
            let budget = opts.op_budget(op_id).unwrap_or(global.memory_budget_bytes);
            let sp = &global.with_budget(budget);
            let p = grace_partitions(&partials, sp.memory_budget_bytes);
            let dir = sp.dir.join(format!("distinct-{p}p"));
            let mut store = DiskSpillStore::new(dir, p)?;
            // No aggregates: `&[]` makes this a pure dedup over the group columns.
            let res = combine_finalize_spilling(partials, &[], &mut store)?;
            (res.group_columns, true)
        }
        Admit::InMemory(_reservation) => (
            agg::combine_with(&partials, &[], opts.tuning.radix_parallel_threshold)?.group_columns,
            false,
        ),
    };
    Ok((RecordBatch::try_new(schema, group_columns)?, spilled))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execute;
    use crate::join_par::SKEW_MIN_BUCKET_ROWS;
    use arrow::array::{Array, ArrayRef, Int64Array, StringArray};
    use std::sync::Arc;

    fn batch(keys: &[i64], vals: &[i64]) -> RecordBatch {
        RecordBatch::try_from_iter(vec![
            ("k", Arc::new(Int64Array::from(keys.to_vec())) as ArrayRef),
            ("v", Arc::new(Int64Array::from(vals.to_vec())) as ArrayRef),
        ])
        .unwrap()
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
        let seq = execute(&plan, &[data.clone()]).unwrap();

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
        let seq = execute(&plan, &[one.clone()]).unwrap();
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

        let seq = execute(&plan, &[data.clone()]).unwrap();
        let dir = std::env::temp_dir().join(format!("bc_med_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1, // force the bounded out-of-core path
                dir,
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

        let seq = execute(&plan, &[data.clone()]).unwrap();
        let dir = std::env::temp_dir().join(format!("bc_mixed_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1, // force the out-of-core path
                dir,
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
            ops::try_bounded_mixed_spill(&parts, &gk, &mixed, &dir, 1)
                .unwrap()
                .is_some(),
            "median+sum should engage the bounded mixed path"
        );

        // all constant-state → abstains (grace already bounds per-group accumulators).
        let cs_only = [agg(AggFunc::Sum, "s"), agg(AggFunc::Max, "mx")];
        assert!(
            ops::try_bounded_mixed_spill(&parts, &gk, &cs_only, &dir, 1)
                .unwrap()
                .is_none(),
            "all-constant-state should fall through to grace"
        );

        // a lone aggregate → abstains (the single-aggregate paths own it).
        let lone = [agg(AggFunc::Median, "m")];
        assert!(
            ops::try_bounded_mixed_spill(&parts, &gk, &lone, &dir, 1)
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

        let seq = execute(&plan, &[data.clone()]).unwrap();
        let dir = std::env::temp_dir().join(format!("bc_ndistinct_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1, // force the bounded out-of-core path
                dir,
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

        let seq = execute(&plan, &[data.clone()]).unwrap();
        let dir = std::env::temp_dir().join(format!("bc_mode_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1, // force the bounded out-of-core path
                dir,
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

        let seq = execute(&plan, &[data.clone()]).unwrap();
        let dir = std::env::temp_dir().join(format!("bc_hist_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1,
                dir,
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
        let seq = execute(&plan, &[data.clone()]).unwrap();
        let dir = std::env::temp_dir().join(format!("bc_q_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1,
                dir,
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
            input: Box::new(RelOp::Scan { source_id: 0 }),
            column: "xs".into(),
            alias: "x".into(),
        };
        let data = vec![list_batch()];
        let seq = execute(&plan, &[data.clone()]).unwrap();
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
        let seq = execute(&plan, &[data.clone()]).unwrap();
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
            backward: true,
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
        let seq = execute(&plan, &[one.clone()]).unwrap();
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
        let seq = execute(&plan, &[one.clone()]).unwrap();
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
                    alias: "rn".into(),
                },
                WindowFunc {
                    func: WindowFn::Sum,
                    input: Some(Expr::Col { name: "v".into() }),
                    offset: 1,
                    frame: None,
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
        let seq = execute(&plan, &[data.clone()]).unwrap();

        // memory_budget_bytes = 1 forces the grace-partitioned spill path.
        let dir = std::env::temp_dir().join(format!("bc_par_winspill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1,
                dir,
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
                    alias: "rn".into(),
                },
                WindowFunc {
                    func: WindowFn::Sum,
                    input: Some(Expr::Col { name: "v".into() }),
                    offset: 1,
                    frame: None,
                    alias: "s".into(),
                },
            ],
            rank_limit: None,
        };
        // k: [1,2,1,2,1], v: [30,5,10,15,20]
        let data = vec![batch(&[1, 2, 1, 2, 1], &[30, 5, 10, 15, 20])];
        let seq = execute(&plan, &[data.clone()]).unwrap();
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
                alias: "rn".into(),
            }],
            rank_limit: Some(2),
        };
        // k=1: v=[30,10,20] → keep v=10(rn1),20(rn2); k=2: v=[5,15] → keep both.
        let data = vec![batch(&[1, 2, 1, 2, 1], &[30, 5, 10, 15, 20])];
        let seq = execute(&plan, &[data.clone()]).unwrap();
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
        let seq = execute(&plan, &[data.clone()]).unwrap();

        // memory_budget_bytes = 1 forces the spill branch and many partitions.
        let dir = std::env::temp_dir().join(format!("bc_par_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1,
                dir,
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
        };
        let data = vec![
            batch(&[1, 2, 1, 3, 2, 1], &[10, 20, 10, 40, 20, 10]),
            batch(&[4, 2, 5, 1, 3, 6], &[1, 20, 3, 10, 40, 6]),
            batch(&[1, 7, 2, 8, 3, 9], &[10, 8, 20, 10, 40, 12]),
        ];
        let seq = execute(&plan, &[data.clone()]).unwrap();

        let dir =
            std::env::temp_dir().join(format!("bc_par_distinct_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1,
                dir,
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
            backward: true,
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
            backward: true,
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
            backward: true,
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
        let seq = execute(&plan, &[data.clone()]).unwrap();

        let dir = std::env::temp_dir().join(format!("bc_sort_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1, // force the external-sort branch
                dir,
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
        let seq = execute(&plan, &[data.clone()]).unwrap();

        let dir = std::env::temp_dir().join(format!("bc_sort_multipass_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1, // force the external-sort branch
                dir,
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
                }),
                ..ExecOptions::default()
            };
            let spilled =
                execute_parallel_with(&plan, &[left.clone(), right.clone()], &opts).unwrap();
            assert_eq!(rows(&seq), rows(&spilled), "join type {jt:?} mismatch");
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

        let plain = execute_parallel(&plan, &[data.clone()]).unwrap();
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
        let (_, in_mem) =
            execute_parallel_with_metrics(&plan, &[data.clone()], &ExecOptions::default()).unwrap();
        let agg = in_mem.ops.iter().find(|o| o.kind == "aggregate").unwrap();
        assert!(!agg.spilled, "no envelope means no spill");

        // Tiny budget → forced grace partitioning.
        let dir = std::env::temp_dir().join(format!("bc_metric_spill_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1,
                dir,
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

    /// A heavily skewed join (one hot key dominating one bucket) must (a) produce
    /// the same relation as the sequential oracle, and (b) actually take the skew
    /// path (the over-large bucket spread across worker chunks).
    #[test]
    fn skewed_join_matches_oracle_and_salts() {
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
        let seq = execute(&plan, &[data.clone()]).unwrap();

        let pool = MemoryPool::new(64);
        // Another operator already holds all but 1 byte, so this aggregate's
        // footprint can't be admitted even though the static budget is huge.
        let _held = pool.try_reserve(63).unwrap();
        let dir = std::env::temp_dir().join(format!("bc_pool_pressure_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1 << 30, // static check would say "in memory"
                dir,
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
        let seq = execute(&plan, &[data.clone()]).unwrap();

        let pool = MemoryPool::new(1 << 30);
        let dir = std::env::temp_dir().join(format!("bc_pool_headroom_{}", std::process::id()));
        let opts = ExecOptions {
            agg_spill: Some(SpillOptions {
                memory_budget_bytes: 1 << 30,
                dir,
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
