//! `bc-py` — the PyO3 boundary that assembles the Rust engine into the
//! `batcher._native` extension module.
//!
//! This crate is the *only* one that links PyO3; everything else is a pure-Rust
//! library that is `cargo test`/fuzz-testable without a Python interpreter. The
//! boundary is deliberately thin: the Python control plane builds a plan,
//! lowers it to the JSON IR, and ships it here alongside input relations as
//! pyarrow batches. Conversion is zero-copy via the Arrow C Data Interface, so a
//! `RecordBatch` crosses the boundary without serialization.

use arrow::array::RecordBatch;
use arrow::datatypes::DataType;
use arrow_pyarrow::PyArrowType;
use bc_ir::{EngineConfig, RelOp};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// The engine's allocator, installed here because this crate is the cdylib every
/// `bc-*` crate is linked into — one `#[global_allocator]` covers the whole data plane.
///
/// Every morsel-parallel operator allocates its output buffers per morsel, and glibc's
/// malloc serves buffers of that size (~64 KB and up) through `mmap`/`munmap`. Each
/// `munmap` must invalidate the mapping on every core, so it broadcasts a TLB-shootdown
/// IPI; with 96 workers freeing a buffer per morsel, that interrupt storm is a
/// serialization point in the middle of an embarrassingly parallel scan. Measured on a
/// 6M-row filter: 21.4 ms sequential, and parallel wall time bottoming out at 4.0 ms
/// (a 5.3x speedup on 96 cores) before *regressing* past 32 workers. mimalloc's
/// per-thread heaps recycle the pages instead of returning them, and the same filter
/// scales to 1.46 ms (15x) with no regression.
///
/// This changes no result — only where the bytes come from. It is invisible to
/// `cargo test` on the pure crates (which link no allocator and keep the system one).
#[global_allocator]
static GLOBAL_ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod bloom;
mod errors;
mod flight;
mod hardware;
mod normalize;
mod pool;
mod process;
mod shuffle;
mod sketches;
mod tracing_init;
use normalize::{
    narrow_output, normalize_batch, original_narrow_types, parse_aggregates, parse_group_keys,
    resolve_cast_dtype, supported_cast_dtypes, unwrap_batches,
};
use process::shared_memory_pool;

/// Execute a plan against in-memory input relations, returning the result morsels.
///
/// * `plan_json` — the relational IR document produced by the control plane.
/// * `sources` — `sources[i]` is the relation bound to `Scan { source_id: i }`,
///   each a list of pyarrow `RecordBatch`es (morsels).
/// * `engine_config` — JSON-serialized `EngineConfig` (morsel size, parallelism)
///   from the live Python `Config`; `""` falls back to the engine defaults.
///
/// Returns only the result morsels (zero-copy via the Arrow C Data Interface).
/// Callers that want the per-operator metrics side-channel use
/// [`execute_plan_metered`] instead — the single-node executor (`core`) does, so
/// it can feed measured runtime facts back to Kyber.
///
/// Runs on the Tier-0 interpreter today; tier selection becomes transparent to
/// this entry point once the JIT lands.
#[pyfunction]
#[pyo3(signature = (plan_json, sources, engine_config="", query_id=None))]
fn execute_plan(
    py: Python<'_>,
    plan_json: &str,
    sources: Vec<Vec<PyArrowType<RecordBatch>>>,
    engine_config: &str,
    query_id: Option<&str>,
) -> PyResult<Vec<PyArrowType<RecordBatch>>> {
    let ExecSetup {
        plan,
        sources,
        opts,
        narrow,
        streaming,
        budget,
        materialize_fits,
    } = prepare_exec(plan_json, sources, engine_config, query_id)?;
    let out = py
        .allow_threads(|| {
            if streaming {
                match bc_interp::execute_streaming_parallel_or_hand_off(
                    &plan,
                    &sources,
                    opts.workers(),
                    budget,
                    materialize_fits,
                    opts.cancel.as_ref(),
                ) {
                    // Either a breaker would have blown the envelope (the materializing executor
                    // spills it; re-run there rather than OOM) or the streaming executor found it
                    // could not spread this plan and handed it back. The work already done is
                    // lost, which is the price of the fast path — and far cheaper than the crash,
                    // or the single-pipeline join, it replaces.
                    Err(e) if wants_materializing(&e) => {
                        bc_interp::execute_parallel_with(&plan, &sources, &opts)
                    }
                    other => other,
                }
            } else {
                bc_interp::execute_parallel_with(&plan, &sources, &opts)
            }
        })
        .map_err(errors::interp_to_pyerr)?;
    let out = narrow_output(out, &narrow);
    Ok(out.into_iter().map(PyArrowType).collect())
}

/// Execute a plan and also return a per-operator metrics document.
///
/// Identical results to [`execute_plan`], plus a JSON `ExecMetrics` string carrying
/// per-operator row counts, timings, peak bytes, spill flags, and backend tags.
/// The metrics ride a side-channel string, never interleaved with the columnar data
/// — Core transcribes them into `OperatorFeedback` so Kyber can calibrate its cost
/// model on the next run. Returns `(batches, metrics_json)`.
#[pyfunction]
#[pyo3(signature = (plan_json, sources, engine_config="", query_id=None))]
fn execute_plan_metered(
    py: Python<'_>,
    plan_json: &str,
    sources: Vec<Vec<PyArrowType<RecordBatch>>>,
    engine_config: &str,
    query_id: Option<&str>,
) -> PyResult<(Vec<PyArrowType<RecordBatch>>, String)> {
    let ExecSetup {
        plan,
        sources,
        opts,
        narrow,
        streaming,
        budget,
        materialize_fits,
    } = prepare_exec(plan_json, sources, engine_config, query_id)?;
    // Started outside `allow_threads` so it brackets *everything* the executor does,
    // including a streaming-to-materializing hand-off. This is the one boundary that sees
    // every tier, which is why the whole-execution resource measurement is taken here rather
    // than inside an executor: the streaming tier cannot attribute OS counters to an operator
    // (its operators interleave), so without this the default tier reported no CPU, memory,
    // or disk consumption at all.
    let query_watch = bc_interp::QueryStopwatch::start();
    let (out, metrics) = py
        .allow_threads(|| {
            if streaming {
                match bc_interp::execute_streaming_parallel_metered_or_hand_off(
                    &plan,
                    &sources,
                    opts.workers(),
                    budget,
                    materialize_fits,
                    opts.cancel.as_ref(),
                ) {
                    Err(e) if wants_materializing(&e) => {
                        bc_interp::execute_parallel_with_metrics(&plan, &sources, &opts)
                    }
                    other => other,
                }
            } else {
                bc_interp::execute_parallel_with_metrics(&plan, &sources, &opts)
            }
        })
        .map_err(errors::interp_to_pyerr)?;
    let metrics = metrics.with_query(query_watch);
    let out = narrow_output(out, &narrow);
    Ok((
        out.into_iter().map(PyArrowType).collect(),
        metrics.to_json(),
    ))
}

/// Shared setup for the execute entry points: parse the plan + engine config and
/// normalize the input morsels (narrow numeric types → Int64/Float64) once.
struct ExecSetup {
    plan: RelOp,
    sources: Vec<Vec<RecordBatch>>,
    opts: bc_interp::ExecOptions,
    /// Pre-widening source column types, for re-narrowing the output. Empty unless asked for.
    narrow: std::collections::HashMap<String, DataType>,
    /// Run on the streaming executor.
    streaming: bool,
    /// The streaming breakers' memory envelope (`0` = unbounded).
    budget: usize,
    /// Whether the materializing executor's footprint fits this query's envelope — the
    /// permission the streaming executor needs before it may decline a plan it cannot shard.
    materialize_fits: bool,
}

/// Whether this query runs on the streaming executor. `true` by default.
///
/// Streaming pulls morsels through the linear runs and materializes only at breakers, so its peak
/// memory is a constant rather than the sum of every operator's output — and on the shapes where
/// that matters it is also *faster*, because the copies it stops making were not free.
///
/// **The two executors do not dominate one another, and that is why this is not a simple swap.**
/// Streaming bounds the *intermediates* but its breakers fold in memory; the materializing
/// executor has unbounded intermediates but breakers that spill out of core. A plan whose
/// aggregate state exceeds the envelope is one the materializing executor survives and this one
/// would OOM on. So the streaming breakers check their state against `memory_budget_bytes` and
/// return `MemoryBudgetExceeded` instead of dying — and `execute_plan` catches exactly that and
/// re-runs on the executor that can spill. Streaming takes the queries it fits (the
/// overwhelming majority, and every one whose intermediates were the problem) and gives way on
/// the ones it does not, rather than quietly turning a spill into a crash.
///
/// Set `streaming = false` to force the materializing executor — a bisecting escape hatch, not a
/// tuning knob.
fn use_streaming(cfg: &EngineConfig) -> bool {
    cfg.streaming
}

/// The memory envelope the streaming breakers must stay inside — `0` means unbounded.
fn stream_budget(cfg: &EngineConfig) -> usize {
    cfg.memory_budget_bytes
}

/// True for the two signals that mean "re-run this on the materializing executor".
///
/// Neither is a failure. `MemoryBudgetExceeded` says a streaming breaker would have blown the
/// envelope and cannot spill; `PreferMaterializing` says the streaming executor found, from the
/// build sides it prepared, that it cannot spread this plan across cores. Both are answered the
/// same way, and answering them is always sound: the two executors are checked against the same
/// sequential oracle, so this changes only speed and peak memory.
fn wants_materializing(e: &bc_interp::InterpError) -> bool {
    matches!(
        e,
        bc_interp::InterpError::MemoryBudgetExceeded { .. }
            | bc_interp::InterpError::PreferMaterializing { .. }
    )
}

fn prepare_exec(
    plan_json: &str,
    sources: Vec<Vec<PyArrowType<RecordBatch>>>,
    engine_config: &str,
    query_id: Option<&str>,
) -> PyResult<ExecSetup> {
    let plan = RelOp::from_json(plan_json).map_err(errors::ir_to_pyerr)?;
    let cfg = EngineConfig::from_json(engine_config).map_err(to_pyerr)?;
    let mut opts = bc_interp::ExecOptions::default().with_engine_config(&cfg);
    // A positive budget activates the runtime memory backstop via the *process-wide*
    // pool (per-query pools would let N concurrent queries each hold `budget` and OOM).
    // Zero budget ⇒ no pool ⇒ the fast path pays nothing.
    if cfg.memory_budget_bytes > 0 {
        opts.pool = Some(shared_memory_pool(cfg.memory_budget_bytes));
    }
    // Look the token up rather than creating one: the control plane registered the id before
    // it started optimizing, so a cancel that arrived during planning is already recorded on
    // that token. A query with no id — or one whose scope has already closed — polls nothing,
    // so an uncancellable path costs exactly what it did before.
    opts.cancel = query_id.and_then(bc_resource::cancel::token_for);
    let budget = stream_budget(&cfg);
    let sources: Vec<Vec<RecordBatch>> = sources
        .into_iter()
        .map(|relation| relation.into_iter().map(|b| b.0).collect())
        .collect();
    // Use the streaming executor when configured — except for a plan it cannot spread across
    // cores whose input is small enough that the faster materializing executor is safe.
    //
    // A self-join / correlated subquery (a source scanned more than once) makes the streaming
    // executor fall to its single-threaded sequential pipeline, where the joins probe one morsel
    // at a time; the materializing executor partitions the probe across every core instead
    // (TPC-H q21 measured 251 ms streaming vs 92 ms materializing at sf1). Streaming's own value
    // on such a plan is bounded *intermediate* memory — the join outputs it never holds in full —
    // so it can only be given up when those intermediates cannot approach the envelope. The
    // guard is the input footprint: with the input at well under the budget, a few× blow-up
    // still fits, and the materializing executor's breakers spill on top of that; a large input
    // keeps the bounded streaming path. Correctness is identical either way (both executors are
    // checked against the sequential oracle) — this trades only memory headroom for speed.
    // **The sources this plan scans, not every source bound to the session**
    // (`RelOp::scanned_source_ids`). Judged by the catalog, a session holding the 24 TPC-DS
    // tables made every query look like 1.76 GB, which the `x8` below turns into 14.1 GB
    // against a 7.73 GB envelope — so `materialize_fits` was false for *every* TPC-DS query at
    // every size and what it gates, mostly Kyber's grouped-aggregate verdict, could not fire at
    // all. Suite geomeans against the three rounds before it (0.840/0.859/0.879 and
    // 1.184/1.201/1.189): **TPC-H 0.826, TPC-DS 1.157**, each below every one of them.
    let scanned = plan.scanned_source_ids();
    let src_bytes: usize = sources
        .iter()
        .enumerate()
        .filter(|(i, _)| scanned.contains(i))
        .flat_map(|(_, relation)| relation.iter())
        .map(|b| b.get_array_memory_size())
        .sum();
    //
    // The same guard also decides whether the streaming executor may *decline* a plan it turns
    // out it cannot shard. It answers "is the materializing executor affordable here", which is
    // a question about this query's footprint and the envelope, and nothing the executor itself
    // is told. Whether the plan is one it cannot shard is the executor's half of the answer, and
    // it can only be read off the build sides once they exist — so the two halves meet by the
    // executor reporting `PreferMaterializing` and this side honoring it (see `run_materializing`).
    let materialize_fits = budget > 0 && src_bytes.saturating_mul(8) < budget;
    // Two independent reasons the materializing executor is the better answer, sharing one
    // envelope guard. The first is structural and the engine can see it: a plan streaming
    // cannot shard (a repeated source). The second is a *cardinality* question the engine
    // cannot answer before it has grouped — a join-free grouped aggregate is cheaper
    // materialized once the group count is high, and dearer below it — so Kyber decides it
    // and sends the verdict in the config. The shape check stays here as the engine's own
    // guard, so a stale or over-eager flag cannot reroute a plan this was never measured on.
    let materialize_is_safe_and_faster = (!bc_interp::streaming_parallelizes(&plan)
        || (cfg.prefer_materializing_aggregate
            && bc_interp::materializing_aggregate_is_faster(&plan)))
        && materialize_fits;
    let streaming = use_streaming(&cfg) && !materialize_is_safe_and_faster;
    // Record pre-widening source widths *before* normalization (which widens them
    // away), and only when output re-narrowing is requested; an empty map makes
    // `narrow_output` a no-op (the default fast path).
    let narrow = if cfg.shrink_output_dtypes {
        original_narrow_types(&sources)
    } else {
        std::collections::HashMap::new()
    };
    let sources: Vec<Vec<RecordBatch>> = sources
        .into_iter()
        .map(|relation| {
            relation
                .iter()
                .map(normalize_batch)
                .collect::<PyResult<Vec<_>>>()
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(ExecSetup {
        plan,
        sources,
        opts,
        narrow,
        streaming,
        budget,
        materialize_fits,
    })
}

/// Ask a running query to stop at its next morsel boundary.
///
/// Returns whether a query with that id was running. `false` means it already finished,
/// which is not an error: the race between a cancel and a completion has no correct loser.
#[pyfunction]
fn cancel_query(query_id: &str) -> bool {
    bc_resource::cancel::cancel(query_id)
}

/// Open a cancellation registration for `query_id`. The control plane owns its lifetime.
#[pyfunction]
fn register_query(query_id: &str) {
    let _ = bc_resource::cancel::register(query_id);
}

/// Close the registration for `query_id`. Idempotent.
#[pyfunction]
fn unregister_query(query_id: &str) {
    bc_resource::cancel::unregister(query_id);
}

/// The engine's default shuffle credit window (in-flight `RecordBatch` slots).
///
/// The window in force whenever the control plane has *not* handed one down: a session
/// built without an explicit grant, or a producer that receives a malformed seed. Exposed
/// so Carbonite can assert the two sides agree — when they drift, it is exactly the
/// un-granted paths that silently run at the wrong window.
#[pyfunction]
fn default_credits() -> u32 {
    bc_transport::DEFAULT_CREDITS
}

/// Ids of the queries currently executing in this process, sorted.
#[pyfunction]
fn running_queries() -> Vec<String> {
    bc_resource::cancel::running()
}

/// Ask every running query in this process to stop. For interpreter shutdown.
///
/// Returns how many were asked, so a shutdown path can report whether it interrupted
/// anything instead of guessing.
#[pyfunction]
fn cancel_all_queries() -> usize {
    bc_resource::cancel::cancel_all()
}

/// Map any engine error into a Python exception. The error hierarchy mapping
/// (PlanError/ExecutionError/...) is refined once the Python error types exist;
/// for now everything surfaces as a `RuntimeError` carrying the engine message.
pub(crate) fn to_pyerr<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Distributed map step: aggregate one partition into partial state.
#[pyfunction]
fn partial_aggregate(
    group_keys_json: &str,
    aggregates_json: &str,
    batches: Vec<PyArrowType<RecordBatch>>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let group_keys = parse_group_keys(group_keys_json)?;
    let aggregates = parse_aggregates(aggregates_json)?;
    let batches = unwrap_batches(batches)?;
    let out =
        bc_interp::dist::partial_aggregate(&group_keys, &aggregates, &batches).map_err(to_pyerr)?;
    Ok(PyArrowType(out))
}

/// Execute `plan_json` and fold its output straight into partial-aggregate state,
/// without ever handing the intermediate rows back to Python.
///
/// The fused shuffle-reduce step. A distributed `GROUP BY` over a join has its reducer
/// run the per-bucket join and then aggregate the result — and doing that as two FFI
/// calls (`execute_plan` → Python list → `partial_aggregate`) materializes the *whole*
/// join output as Python `RecordBatch` objects on the way through. On TPC-H sf10 that is
/// 3.75M rows / ~106 MB per reducer, built and re-imported for no reason: the only thing
/// the caller wants is the handful of partial-state rows at the end.
///
/// Folding the two inside Rust keeps the intermediate in the engine, so the reducer's
/// peak memory is the join's own working set rather than the join's output *plus* a
/// Python mirror of it. Semantics are unchanged — the same plan, the same
/// `partial_aggregate`/`combine_finalize` on the same batches, so the mergeable algebra
/// (and therefore the result) is bit-identical to the two-call path.
///
/// `finalize` mirrors the caller's contract: `true` when the group keys are a superset of
/// the join key (every group is whole in this bucket, so it can be finalized here),
/// `false` to emit partial state for a cross-bucket `combine_finalize` on the driver.
#[pyfunction]
#[pyo3(signature = (plan_json, sources, group_keys_json, aggregates_json, engine_config="", finalize=false))]
fn execute_plan_aggregated(
    py: Python<'_>,
    plan_json: &str,
    sources: Vec<Vec<PyArrowType<RecordBatch>>>,
    group_keys_json: &str,
    aggregates_json: &str,
    engine_config: &str,
    finalize: bool,
) -> PyResult<PyArrowType<RecordBatch>> {
    let ExecSetup {
        plan,
        sources,
        opts,
        narrow,
        ..
    } = prepare_exec(plan_json, sources, engine_config, None)?;
    let group_keys = parse_group_keys(group_keys_json)?;
    let aggregates = parse_aggregates(aggregates_json)?;
    let out = py.allow_threads(|| {
        let rows = bc_interp::execute_parallel_with(&plan, &sources, &opts)?;
        // Narrow exactly where `execute_plan` would have, so the aggregate sees the same
        // dtypes it saw when this ran as two calls (a no-op unless `shrink_output_dtypes`).
        let rows = narrow_output(rows, &narrow);
        let partial = bc_interp::dist::partial_aggregate(&group_keys, &aggregates, &rows)?;
        if finalize {
            bc_interp::dist::combine_finalize(&group_keys, &aggregates, &[partial])
        } else {
            Ok(partial)
        }
    });
    Ok(PyArrowType(out.map_err(to_pyerr)?))
}

/// Distributed reduce step: merge partial-state batches and finalize.
#[pyfunction]
fn combine_finalize(
    group_keys_json: &str,
    aggregates_json: &str,
    partials: Vec<PyArrowType<RecordBatch>>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let group_keys = parse_group_keys(group_keys_json)?;
    let aggregates = parse_aggregates(aggregates_json)?;
    let partials = unwrap_batches(partials)?;
    let out =
        bc_interp::dist::combine_finalize(&group_keys, &aggregates, &partials).map_err(to_pyerr)?;
    Ok(PyArrowType(out))
}

/// Combine step WITHOUT finalize: merge partial-state batches into a single partial
/// batch (same wire format), so a streaming driver can keep one running state across
/// micro-batches, bounded by the number of groups, and `combine_finalize` once.
#[pyfunction]
fn combine(
    group_keys_json: &str,
    aggregates_json: &str,
    partials: Vec<PyArrowType<RecordBatch>>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let group_keys = parse_group_keys(group_keys_json)?;
    let aggregates = parse_aggregates(aggregates_json)?;
    let partials = unwrap_batches(partials)?;
    let out = bc_interp::dist::combine(&group_keys, &aggregates, &partials).map_err(to_pyerr)?;
    Ok(PyArrowType(out))
}

/// Native Parquet read of one object's selected row-groups into pyarrow batches.
///
/// `bc_io` decodes Parquet in Rust and fetches the projected column chunks of the
/// requested row-groups concurrently from object storage (S3/GCS/Azure/HTTP/local),
/// returning zero-copy Arrow `RecordBatch`es — the throughput path the distributed
/// scan uses instead of PyArrow's per-chunk reads. `row_groups` empty = all;
/// `columns` `None` = all (else a name projection pushed into the decode).
#[pyfunction]
#[pyo3(signature = (uri, row_groups, columns, batch_size))]
fn read_parquet(
    py: Python<'_>,
    uri: &str,
    row_groups: Vec<usize>,
    columns: Option<Vec<String>>,
    batch_size: usize,
) -> PyResult<Vec<PyArrowType<RecordBatch>>> {
    // Release the GIL across the (object-store-I/O-bound) read so other Python threads
    // on the worker — the engine's fold, concurrent split reads — run during the S3
    // fetch instead of serializing behind it. Holding the GIL here made the native read
    // ~3x slower than PyArrow (which releases it) in the distributed path.
    let batches = py
        .allow_threads(|| bc_io::read_parquet(uri, &row_groups, columns.as_deref(), batch_size))
        .map_err(to_pyerr)?;
    Ok(batches.into_iter().map(PyArrowType).collect())
}

/// Native Parquet read with a pushed predicate applied as footer-statistics row-group
/// pruning before decode.
///
/// Same as [`read_parquet`] plus `predicate`: the compact JSON `to_native_predicate`
/// emits. Row-groups whose statistics prove no row can match are skipped (their column
/// chunks are never fetched or decoded). Pruning is superset-safe — the engine keeps the
/// `Filter` operator, so a non-pushable or unparseable predicate simply reads every
/// requested row-group and the result is identical, just with more rows read.
#[pyfunction]
#[pyo3(signature = (uri, row_groups, columns, batch_size, predicate))]
fn read_parquet_filtered(
    py: Python<'_>,
    uri: &str,
    row_groups: Vec<usize>,
    columns: Option<Vec<String>>,
    batch_size: usize,
    predicate: &str,
) -> PyResult<Vec<PyArrowType<RecordBatch>>> {
    let batches = py
        .allow_threads(|| {
            bc_io::read_parquet_filtered(
                uri,
                &row_groups,
                columns.as_deref(),
                batch_size,
                predicate,
            )
        })
        .map_err(to_pyerr)?;
    Ok(batches.into_iter().map(PyArrowType).collect())
}

/// Native read of MANY whole Parquet objects in one pass, returning per-file batch lists.
///
/// The many-small-files throughput path: one GIL release and one runtime pass overlap every
/// file's footer + column-chunk GETs under a global concurrency budget, instead of a
/// `read_parquet` call (and FFI round trip) per file. `columns` `None` = all.
#[pyfunction]
#[pyo3(signature = (uris, columns, batch_size))]
fn read_parquet_many(
    py: Python<'_>,
    uris: Vec<String>,
    columns: Option<Vec<String>>,
    batch_size: usize,
) -> PyResult<Vec<Vec<PyArrowType<RecordBatch>>>> {
    let per_file = py
        .allow_threads(|| bc_io::read_parquet_many(&uris, columns.as_deref(), batch_size))
        .map_err(to_pyerr)?;
    Ok(per_file
        .into_iter()
        .map(|batches| batches.into_iter().map(PyArrowType).collect())
        .collect())
}

/// Per-column footer facts as they cross to Python:
/// `(name, has_stats, null_count, null_known, nan_seen, distinct)`.
type FooterColumn = (String, bool, u64, bool, bool, Option<i64>);

/// Aggregated Parquet footer statistics for many files, computed natively.
///
/// The planning-path counterpart to [`read_parquet_many`]. Walking footers from Python
/// costs a pybind11 object per *column chunk*, which is O(files x row_groups x columns)
/// interpreter work on the driver before a single data page is read; on 200 files x 20
/// row-groups x 30 columns that measured ~750 ms of overhead over ~95 ms of real footer
/// I/O. Here the walk is native and the footers come from the reader's validated cache, so
/// a file already touched this process contributes with no I/O at all.
///
/// Returns `(bounds, columns, total_rows, total_bytes, row_group_count, files_read,
/// sort_declared)`.
/// `bounds` is a 2-row `RecordBatch` — **row 0 = min, row 1 = max** — with one field per
/// entry of `columns`, typed as that column is typed, so every bound keeps its exact Arrow
/// type instead of being converted per value. A null bound means *unknown* (never *no
/// value*), and `files_read` below `uris.len()` means some footer was unreadable, so the
/// caller must not report the row count as exact.
///
/// `sort_declared` is only the *precondition* for a global-sortedness claim, never the
/// claim: the proof that deletes a `Sort` stays in `io.stats.sortedness`, and this merely
/// tells the caller whether running it could succeed.
#[pyfunction]
#[pyo3(signature = (uris))]
#[allow(clippy::type_complexity)]
fn parquet_footer_stats(
    py: Python<'_>,
    uris: Vec<String>,
) -> PyResult<(
    PyArrowType<RecordBatch>,
    Vec<FooterColumn>,
    i64,
    i64,
    i64,
    usize,
    bool,
)> {
    // Released across the whole pass: the footer GETs are object-store I/O and the walk
    // itself is native, so nothing here needs the interpreter.
    let stats = py
        .allow_threads(|| bc_io::parquet_footer_stats(&uris))
        .map_err(to_pyerr)?;
    let columns = stats
        .columns
        .into_iter()
        .map(|c| {
            (
                c.name,
                c.has_stats,
                c.null_count,
                c.null_known,
                c.nan_seen,
                c.distinct,
            )
        })
        .collect();
    Ok((
        PyArrowType(stats.bounds),
        columns,
        stats.total_rows,
        stats.total_bytes,
        stats.row_group_count,
        stats.files_read,
        stats.sort_declared,
    ))
}

/// Per-file Parquet bounds for `columns`, in the add-action layout, built natively.
///
/// The sibling of [`parquet_footer_stats`]: one row per file rather than one per dataset,
/// laid out `path | num_records | min.<col> | max.<col> | null_count.<col>` — the same shape
/// a lakehouse log publishes, so a copy-on-write `MERGE` can skip files on a plain directory
/// with no transaction log. Footers come from the cache the statistics pass already filled,
/// so a query doing both reads each footer once.
///
/// A file whose footer is unreadable keeps its row with NULL bounds, which every consumer
/// must read as *unknown* (keep the file), never as *no match*.
#[pyfunction]
#[pyo3(signature = (uris, columns))]
fn parquet_file_manifest(
    py: Python<'_>,
    uris: Vec<String>,
    columns: Vec<String>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let batch = py
        .allow_threads(|| bc_io::parquet_file_manifest(&uris, &columns))
        .map_err(to_pyerr)?;
    Ok(PyArrowType(batch))
}

/// Native Avro (OCF) decode to Arrow — the columnar replacement for the row-by-row
/// `fastavro` Python path (measured ~33x faster on 3 M rows). `data` is one whole Avro
/// file's bytes (the Python `AvroSource` already holds them from the split's handle);
/// `batch_size` is the output `RecordBatch` row count. Errors surface to Python, where
/// `AvroSource` falls back to `fastavro`, so the result is identical either way.
#[pyfunction]
#[pyo3(signature = (data, batch_size))]
fn read_avro(
    py: Python<'_>,
    data: &[u8],
    batch_size: usize,
) -> PyResult<Vec<PyArrowType<RecordBatch>>> {
    let batches = py
        .allow_threads(|| bc_io::read_avro_bytes(data, batch_size))
        .map_err(to_pyerr)?;
    Ok(batches.into_iter().map(PyArrowType).collect())
}

/// What a payload's leading bytes say it is, or `None` when nothing recognizes them.
///
/// The IO layer's `sniff_mime` calls this so the magic-number table has exactly one home:
/// the same `bc-expr` function `.str.mime_type()` evaluates. A Python-side copy of the
/// table would be a second answer to "what is this file", and the two would drift the
/// first time a format was added to one of them — silently, on a column whose whole job
/// is to route rows to different branches.
///
/// Scalar rather than batched because the caller is already opening a file per row; the
/// FFI hop is nothing next to that, and a batched form would need the readers restructured
/// around it for no measurable gain.
#[pyfunction]
fn sniff_mime(data: &[u8]) -> Option<&'static str> {
    bc_interp::sniff_mime(data)
}

/// The optional cargo features compiled into this engine, sorted.
///
/// Kept as an explicit list rather than derived, because a feature only belongs here once
/// it changes what the *control plane* may plan: `video` does (it decides whether
/// `.video.frames` is a native kernel or a Python fallback), and a feature that only
/// changes how something is done internally does not.
fn engine_features() -> Vec<&'static str> {
    let mut features: Vec<&'static str> = Vec::new();
    if cfg!(feature = "video") {
        features.push("video");
    }
    features.sort_unstable();
    features
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__engine_version__", env!("CARGO_PKG_VERSION"))?;
    // The build profile, so the benchmark harness can refuse to report a timing taken
    // against an unoptimized engine. `just build` (the dev profile) sets no `opt-level`
    // and leaves `debug_assertions` on, while every comparator is an installed release
    // wheel — timing one against the other measures the profile, not the engine. Nothing
    // on the Python side can see this, which is why the engine has to say so.
    m.add(
        "__engine_profile__",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;
    // The optional cargo features this engine was compiled with. Optional decoders link
    // system C libraries, so whether one is present is a property of the *binary*, not of
    // the Python environment — and nothing on the Python side can otherwise see it. The
    // control plane needs it at plan-build time to choose between a native kernel and a
    // Python fallback: guessing wrong means either a run-time failure on a path the plan
    // promised, or a slow per-row fallback on a build that never needed one.
    m.add("__engine_features__", engine_features())?;
    m.add_function(wrap_pyfunction!(sniff_mime, m)?)?;
    m.add_function(wrap_pyfunction!(tracing_init::init_tracing, m)?)?;
    m.add_function(wrap_pyfunction!(execute_plan, m)?)?;
    m.add_function(wrap_pyfunction!(execute_plan_metered, m)?)?;
    m.add_function(wrap_pyfunction!(cancel_query, m)?)?;
    m.add_function(wrap_pyfunction!(register_query, m)?)?;
    m.add_function(wrap_pyfunction!(unregister_query, m)?)?;
    m.add_function(wrap_pyfunction!(running_queries, m)?)?;
    m.add_function(wrap_pyfunction!(cancel_all_queries, m)?)?;
    m.add_function(wrap_pyfunction!(default_credits, m)?)?;
    m.add_function(wrap_pyfunction!(read_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(read_avro, m)?)?;
    m.add_function(wrap_pyfunction!(read_parquet_filtered, m)?)?;
    m.add_function(wrap_pyfunction!(read_parquet_many, m)?)?;
    m.add_function(wrap_pyfunction!(parquet_footer_stats, m)?)?;
    m.add_function(wrap_pyfunction!(parquet_file_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(partial_aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(execute_plan_aggregated, m)?)?;
    m.add_function(wrap_pyfunction!(combine, m)?)?;
    m.add_function(wrap_pyfunction!(combine_finalize, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle::combine_finalize_spilling, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle::partition_batches, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle::partition_batches_salted, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle::range_partition_batches, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle::range_partition_batches_str, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle::column_string_quantiles, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle::salted_partition_batches, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle::gather_combine, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle::gather_concat, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle::gather_to_files, m)?)?;
    m.add_function(wrap_pyfunction!(bloom::build_key_bloom, m)?)?;
    m.add_function(wrap_pyfunction!(bloom::merge_blooms, m)?)?;
    m.add_function(wrap_pyfunction!(bloom::bloom_filter_batches, m)?)?;
    m.add_function(wrap_pyfunction!(bloom::build_column_bloom, m)?)?;
    m.add_function(wrap_pyfunction!(sketches::estimate_distinct, m)?)?;
    m.add_function(wrap_pyfunction!(sketches::column_ndv, m)?)?;
    m.add_function(wrap_pyfunction!(sketches::column_stats, m)?)?;
    m.add_function(wrap_pyfunction!(sketches::column_quantiles, m)?)?;
    m.add_function(wrap_pyfunction!(sketches::column_stats_full, m)?)?;
    m.add_function(wrap_pyfunction!(sketches::tail_quantiles, m)?)?;
    m.add_function(wrap_pyfunction!(sketches::tdigest_partial, m)?)?;
    m.add_function(wrap_pyfunction!(sketches::tdigest_quantile, m)?)?;
    m.add_function(wrap_pyfunction!(sketches::heavy_hitters, m)?)?;
    m.add_function(wrap_pyfunction!(sketches::reservoir_sample, m)?)?;
    m.add_class::<flight::FlightShuffleServer>()?;
    m.add_function(wrap_pyfunction!(flight::flight_fetch, m)?)?;
    m.add_function(wrap_pyfunction!(flight::set_flight_transport_config, m)?)?;
    m.add_function(wrap_pyfunction!(flight::set_flight_client_tls, m)?)?;
    m.add_function(wrap_pyfunction!(flight::shm_available, m)?)?;
    m.add_function(wrap_pyfunction!(flight::shuffle_peer_stats, m)?)?;
    m.add_function(wrap_pyfunction!(flight::shuffle_flow_totals, m)?)?;
    m.add_function(wrap_pyfunction!(flight::shuffle_bdp_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(flight::reset_shuffle_peer_stats, m)?)?;
    m.add_function(wrap_pyfunction!(supported_cast_dtypes, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_cast_dtype, m)?)?;
    m.add_class::<flight::ShuffleClient>()?;
    m.add_class::<pool::MemoryPool>()?;
    m.add_function(wrap_pyfunction!(pool::engine_pool_stats, m)?)?;
    // What the engine process itself detected about its CPU, and what its allocator is
    // holding — neither is observable from the Python side (see `hardware.rs`).
    hardware::register(m)?;
    // Classified shuffle-fetch exceptions: the control plane catches `Retryable` as
    // worker loss (recompute + retry) and lets `Fatal` propagate (fail fast).
    errors::register(m)?;
    Ok(())
}
