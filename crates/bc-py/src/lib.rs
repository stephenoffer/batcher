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
mod normalize;
mod process;
mod shuffle;
mod sketches;
mod tracing_init;
use errors::transport_to_pyerr;
use normalize::{
    narrow_output, normalize_batch, original_narrow_types, parse_aggregates, parse_group_keys,
    supported_cast_dtypes, unwrap_batches,
};
use process::{shared_memory_pool, shared_runtime};

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
#[pyo3(signature = (plan_json, sources, engine_config=""))]
fn execute_plan(
    py: Python<'_>,
    plan_json: &str,
    sources: Vec<Vec<PyArrowType<RecordBatch>>>,
    engine_config: &str,
) -> PyResult<Vec<PyArrowType<RecordBatch>>> {
    let (plan, sources, opts, narrow, streaming, budget) =
        prepare_exec(plan_json, sources, engine_config)?;
    let out = py
        .allow_threads(|| {
            if streaming {
                match bc_interp::execute_streaming_parallel(&plan, &sources, opts.workers(), budget)
                {
                    // A breaker would have blown the envelope. The materializing executor spills
                    // it; re-run there rather than OOM. The work already done is lost, which is
                    // the price of the fast path — and far cheaper than the crash it replaces.
                    Err(e) if needs_spill(&e) => {
                        bc_interp::execute_parallel_with(&plan, &sources, &opts)
                    }
                    other => other,
                }
            } else {
                bc_interp::execute_parallel_with(&plan, &sources, &opts)
            }
        })
        .map_err(to_pyerr)?;
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
#[pyo3(signature = (plan_json, sources, engine_config=""))]
fn execute_plan_metered(
    py: Python<'_>,
    plan_json: &str,
    sources: Vec<Vec<PyArrowType<RecordBatch>>>,
    engine_config: &str,
) -> PyResult<(Vec<PyArrowType<RecordBatch>>, String)> {
    let (plan, sources, opts, narrow, streaming, budget) =
        prepare_exec(plan_json, sources, engine_config)?;
    let (out, metrics) = py
        .allow_threads(|| {
            if streaming {
                match bc_interp::execute_streaming_parallel_metered(
                    &plan,
                    &sources,
                    opts.workers(),
                    budget,
                ) {
                    Err(e) if needs_spill(&e) => {
                        bc_interp::execute_parallel_with_metrics(&plan, &sources, &opts)
                    }
                    other => other,
                }
            } else {
                bc_interp::execute_parallel_with_metrics(&plan, &sources, &opts)
            }
        })
        .map_err(to_pyerr)?;
    let out = narrow_output(out, &narrow);
    Ok((
        out.into_iter().map(PyArrowType).collect(),
        metrics.to_json(),
    ))
}

/// Shared setup for the execute entry points: parse the plan + engine config and
/// normalize the input morsels (narrow numeric types → Int64/Float64) once.
type ExecSetup = (
    RelOp,
    Vec<Vec<RecordBatch>>,
    bc_interp::ExecOptions,
    std::collections::HashMap<String, DataType>,
    bool,
    usize,
);

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

/// True for the one error that means "this plan needs to spill, and I cannot".
fn needs_spill(e: &bc_interp::InterpError) -> bool {
    matches!(e, bc_interp::InterpError::MemoryBudgetExceeded { .. })
}

fn prepare_exec(
    plan_json: &str,
    sources: Vec<Vec<PyArrowType<RecordBatch>>>,
    engine_config: &str,
) -> PyResult<ExecSetup> {
    let plan = RelOp::from_json(plan_json).map_err(to_pyerr)?;
    let cfg = EngineConfig::from_json(engine_config).map_err(to_pyerr)?;
    let mut opts = bc_interp::ExecOptions::default().with_engine_config(&cfg);
    // A positive budget activates the runtime memory backstop via the *process-wide*
    // pool (per-query pools would let N concurrent queries each hold `budget` and OOM).
    // Zero budget ⇒ no pool ⇒ the fast path pays nothing.
    if cfg.memory_budget_bytes > 0 {
        opts.pool = Some(shared_memory_pool(cfg.memory_budget_bytes));
    }
    let streaming = use_streaming(&cfg);
    let budget = stream_budget(&cfg);
    let sources: Vec<Vec<RecordBatch>> = sources
        .into_iter()
        .map(|relation| relation.into_iter().map(|b| b.0).collect())
        .collect();
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
    Ok((plan, sources, opts, narrow, streaming, budget))
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
    let (plan, sources, opts, narrow, _streaming, _budget) =
        prepare_exec(plan_json, sources, engine_config)?;
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

/// A process-wide memory accounting pool (Carbonite's reserve-before-allocate
/// enforcement primitive, from `bc-resource`). Carbonite sets the limit from its
/// memory envelope and reserves/releases against it so the engine spills instead
/// of OOMing. Accounts bytes; it does not allocate them.
#[pyclass]
struct MemoryPool {
    inner: std::sync::Arc<bc_resource::MemoryPool>,
}

#[pymethods]
impl MemoryPool {
    /// Create a pool admitting up to `limit_bytes` reserved at once.
    #[new]
    fn new(limit_bytes: u64) -> Self {
        Self {
            inner: bc_resource::MemoryPool::new(limit_bytes as usize),
        }
    }

    /// Try to reserve `bytes`; returns `True` on success, `False` if the pool is
    /// full (the caller should then spill / back-pressure). Never partially
    /// reserves — a `False` leaves the pool untouched.
    fn try_reserve(&self, bytes: u64) -> bool {
        self.inner.try_reserve_bytes(bytes as usize).is_ok()
    }

    /// Release `bytes` back to the pool (clamped so a double-release can't underflow).
    fn release(&self, bytes: u64) {
        self.inner.release_bytes(bytes as usize);
    }

    /// Resize the envelope. Live reservations are untouched; only what future
    /// reservations admit against changes (an autoscaler grew/shrank the budget).
    fn set_limit(&self, limit_bytes: u64) {
        self.inner.set_limit(limit_bytes as usize);
    }

    /// Bytes currently reserved.
    #[getter]
    fn used(&self) -> u64 {
        self.inner.used() as u64
    }

    /// Bytes currently free (`limit - used`).
    #[getter]
    fn available(&self) -> u64 {
        self.inner.available() as u64
    }

    /// The pool's hard limit in bytes.
    #[getter]
    fn limit(&self) -> u64 {
        self.inner.limit() as u64
    }
}

/// A node-local Arrow Flight shuffle server. Each distributed worker holds one;
/// mappers `publish` their output partitions on it and advertise `addr`, and
/// reducers `flight_fetch` those partitions over the network with credit-bounded
/// streaming — moving shuffle data node→node **without ever touching the Ray
/// object store** (only the tiny address/ticket strings transit Ray).
///
/// Background serving runs on the process-wide [`shared_runtime`]; the exchange's
/// own `ServerHandle` keeps this server's serve task alive for the object's life.
#[pyclass]
struct FlightShuffleServer {
    pub(crate) exchange: bc_transport::ShuffleExchange,
    addr: String,
}

#[pymethods]
impl FlightShuffleServer {
    /// Create a node-local Flight shuffle server.
    ///
    /// `advertise_host` is the node's **routable** address (the Ray node IP). When
    /// given, the server binds all interfaces (`0.0.0.0:0`) and advertises
    /// `{advertise_host}:{port}` so reducers on *other* nodes can reach it — the
    /// fix for a cross-node cluster, where a loopback `127.0.0.1` advertise is
    /// unreachable. Omitted/empty keeps the single-host loopback behavior.
    /// `tls_cert_pem` / `tls_key_pem` (when both given) make the server present a TLS
    /// certificate so the inter-node shuffle is encrypted; `tls_client_ca_pem` in
    /// addition turns on mTLS — a connecting peer must present a certificate that CA
    /// signed. All PEM is minted by the operator's own PKI; Batcher issues nothing.
    #[new]
    #[pyo3(signature = (advertise_host=None, token=None, tls_cert_pem=None, tls_key_pem=None, tls_client_ca_pem=None))]
    fn new(
        advertise_host: Option<String>,
        token: Option<String>,
        tls_cert_pem: Option<String>,
        tls_key_pem: Option<String>,
        tls_client_ca_pem: Option<String>,
    ) -> PyResult<Self> {
        let host = advertise_host.filter(|h| !h.is_empty());
        let token = token.filter(|t| !t.is_empty());
        let tls = match (tls_cert_pem, tls_key_pem) {
            (Some(cert), Some(key)) => {
                let mut cfg = bc_transport::TlsServerConfig::new(
                    bc_transport::TlsIdentity::from_pem(cert, key),
                );
                if let Some(ca) = tls_client_ca_pem.filter(|c| !c.is_empty()) {
                    cfg = cfg.with_client_ca(ca);
                }
                Some(cfg)
            }
            (None, None) => None,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "shuffle TLS requires both a certificate and a private key",
                ))
            }
        };
        // A loopback bind for the single-host case; all-interfaces when a routable
        // advertise host is given so cross-node reducers can dial it.
        let bind = if host.is_some() {
            "0.0.0.0:0"
        } else {
            "127.0.0.1:0"
        };
        let exchange = shared_runtime()
            .block_on(bc_transport::ShuffleExchange::bind_tls(
                bind,
                host.as_deref(),
                token,
                tls,
            ))
            .map_err(to_pyerr)?;
        let addr = exchange.advertised_addr().to_string();
        Ok(Self { exchange, addr })
    }

    /// The routable `host:port` to advertise to reducers.
    #[getter]
    fn addr(&self) -> String {
        self.addr.clone()
    }

    /// Expose `batches` under `ticket` (canonical `plan/stage/src/dst/epoch`).
    fn publish(
        &self,
        py: Python<'_>,
        ticket: &str,
        batches: Vec<PyArrowType<RecordBatch>>,
    ) -> PyResult<()> {
        let t = bc_transport::ShuffleTicket::from_string(ticket).map_err(to_pyerr)?;
        let batches: Vec<RecordBatch> = batches
            .iter()
            .map(|b| normalize_batch(&b.0))
            .collect::<PyResult<_>>()?;
        py.allow_threads(|| shared_runtime().block_on(self.exchange.publish(&t, batches)));
        Ok(())
    }

    /// High-water mark of in-flight batches for `ticket` (peak the producer ever
    /// had un-acked), or `None` if the ticket was never published. Lets a test
    /// assert the credit bound was honored: this never exceeds the granted window.
    fn max_inflight(&self, py: Python<'_>, ticket: &str) -> PyResult<Option<i64>> {
        let t = bc_transport::ShuffleTicket::from_string(ticket).map_err(to_pyerr)?;
        Ok(py.allow_threads(|| shared_runtime().block_on(self.exchange.max_inflight(&t))))
    }

    /// Read a partition this server itself published, without a network hop — the
    /// `DIRECT_MEMORY` fast path for a same-process reducer. `None` if `ticket`
    /// was never published here, so the caller falls back to a network fetch.
    fn local_fetch(
        &self,
        py: Python<'_>,
        ticket: &str,
    ) -> PyResult<Option<Vec<PyArrowType<RecordBatch>>>> {
        let t = bc_transport::ShuffleTicket::from_string(ticket).map_err(to_pyerr)?;
        let batches =
            py.allow_threads(|| shared_runtime().block_on(self.exchange.local_partition(&t)));
        Ok(batches.map(|bs| bs.into_iter().map(PyArrowType).collect()))
    }

    /// Mirror `ticket`'s `batches` to a same-node shared-memory file (Arrow IPC over a
    /// memory map) under this server's advertised address, so a reducer in *another*
    /// process on the same host can read them with no gRPC/loopback hop. Best-effort:
    /// a write error is swallowed (the reducer falls back to Flight). A batch the boundary
    /// cannot normalize (a `UInt64` above `i64::MAX`) surfaces as an error rather than being
    /// mirrored in a corrupted form.
    fn publish_shared(
        &self,
        py: Python<'_>,
        ticket: &str,
        batches: Vec<PyArrowType<RecordBatch>>,
    ) -> PyResult<()> {
        let batches: Vec<RecordBatch> = batches
            .iter()
            .map(|b| normalize_batch(&b.0))
            .collect::<PyResult<_>>()?;
        let addr = self.addr.clone();
        py.allow_threads(|| {
            let _ = bc_transport::publish_shared(&addr, ticket, &batches);
        });
        Ok(())
    }

    /// Read a partition a same-node peer published under `(source_addr, ticket)` from
    /// shared memory (mmap), or `None` if absent (an empty bucket, an un-shm'd peer, or
    /// shm off) so the caller falls back to Flight.
    fn shm_fetch(
        &self,
        py: Python<'_>,
        source_addr: &str,
        ticket: &str,
    ) -> PyResult<Option<Vec<PyArrowType<RecordBatch>>>> {
        let batches = py
            .allow_threads(|| bc_transport::fetch_shared(source_addr, ticket))
            .map_err(to_pyerr)?;
        Ok(batches.map(|bs| bs.into_iter().map(PyArrowType).collect()))
    }

    /// Remove every shared-memory file this server published (plan teardown).
    fn clear_shared(&self, py: Python<'_>) {
        let addr = self.addr.clone();
        py.allow_threads(|| bc_transport::clear_shared(&addr));
    }

    /// Evict one published partition (its reducers have fetched it), freeing it.
    fn release(&self, py: Python<'_>, ticket: &str) -> PyResult<()> {
        let t = bc_transport::ShuffleTicket::from_string(ticket).map_err(to_pyerr)?;
        py.allow_threads(|| shared_runtime().block_on(self.exchange.release(&t)));
        Ok(())
    }

    /// Evict every partition for plan `plan_id` (call at plan teardown so a reused
    /// worker doesn't accumulate finished plans' shuffle outputs).
    fn clear_plan(&self, py: Python<'_>, plan_id: u64) {
        py.allow_threads(|| shared_runtime().block_on(self.exchange.clear_plan(plan_id)));
    }

    /// Evict every published partition on this server.
    fn clear(&self, py: Python<'_>) {
        py.allow_threads(|| shared_runtime().block_on(self.exchange.clear()));
    }

    /// Number of partitions currently retained (telemetry / leak tests).
    #[getter]
    fn partition_count(&self, py: Python<'_>) -> usize {
        py.allow_threads(|| shared_runtime().block_on(self.exchange.partition_count()))
    }
}

impl Drop for FlightShuffleServer {
    /// Retire published partitions when the Python object is collected.
    ///
    /// A published batch is a zero-copy view of a pyarrow array whose release callback
    /// needs the GIL. The store is shared (via `Arc`) with the background serve task, so
    /// after this object is gone the task still holds the batches and only frees them when
    /// the runtime reaps the aborted task — on a tokio worker thread. If that happens after
    /// the interpreter has finalized, acquiring the GIL turns into a thread-exit that
    /// unwinds through Rust and aborts the process (`std::terminate`). pyo3 runs this drop
    /// under the GIL, so clearing the store here releases the pyarrow buffers now; the
    /// serve task's leftover `Arc` then drops an empty store, touching no Python state.
    fn drop(&mut self) {
        // Guard against the (not-expected) case of being dropped on a runtime thread, where
        // `block_on` would panic; skipping is no worse than the pre-fix behavior.
        if tokio::runtime::Handle::try_current().is_err() {
            shared_runtime().block_on(self.exchange.clear());
        }
    }
}

/// Fetch a shuffle partition from a remote `FlightShuffleServer` over a
/// credit-bounded `DoExchange` stream (bypassing any object store).
///
/// `credits` is the flow-control window — the producer never buffers more than
/// `credits` `RecordBatch`es ahead of the reducer (clamped to >= 1). Carbonite's
/// `FlowControlPolicy` supplies this from the operator's `ResourceBounds`; the
/// default keeps the conservative window when callers don't override it.
#[pyfunction]
#[pyo3(signature = (addr, ticket, credits=bc_transport::DEFAULT_CREDITS, token=None))]
fn flight_fetch(
    py: Python<'_>,
    addr: &str,
    ticket: &str,
    credits: u32,
    token: Option<&str>,
) -> PyResult<Vec<PyArrowType<RecordBatch>>> {
    let batches = py
        .allow_threads(|| bc_transport::fetch_blocking_with_credits(addr, ticket, credits, token))
        .map_err(transport_to_pyerr)?;
    Ok(batches.into_iter().map(PyArrowType).collect())
}

/// Set the process-wide Flight transport tunables from the control plane.
///
/// `idle_timeout_ms` bounds the gap between batches before a peer is treated as
/// dead (`0` keeps the current value); `keepalive_ms` is the HTTP/2 keepalive ping
/// interval (`0` off); `connections_per_peer` bounds how many TCP connections a
/// consumer stripes a peer's fetches across (`0` keeps the current value) — raising it
/// past a single flow is what lets a cross-node fetch reach a cloud NIC's line rate
/// instead of the per-flow cap; `compression` is the shuffle wire codec (0 none / 1 lz4
/// / 2 zstd, `None` keeps the current value) — moving fewer bytes over a NIC-bound link
/// lifts effective throughput past line rate for the compressible data a real shuffle
/// carries. Called once per worker process when its Flight server starts.
#[pyfunction]
#[pyo3(signature = (idle_timeout_ms, keepalive_ms=0, connections_per_peer=0, compression=None))]
fn set_flight_transport_config(
    idle_timeout_ms: u64,
    keepalive_ms: u64,
    connections_per_peer: u64,
    compression: Option<u64>,
) {
    bc_transport::set_transport_timeouts(idle_timeout_ms, keepalive_ms);
    bc_transport::set_connections_per_peer(connections_per_peer);
    if let Some(code) = compression {
        bc_transport::set_compression(code);
    }
}

/// Install (or clear) the process-wide client TLS for outbound shuffle fetches.
///
/// `ca_pem` is the CA a peer's server certificate must chain to and `server_name` the
/// name verified against it; `client_cert_pem`/`client_key_pem` (when both given) present
/// this node's certificate under mTLS. Passing an empty `ca_pem` clears it (plaintext).
/// Called once per worker from the control plane, alongside `set_flight_transport_config`.
#[pyfunction]
#[pyo3(signature = (ca_pem, server_name, client_cert_pem=None, client_key_pem=None))]
fn set_flight_client_tls(
    ca_pem: &str,
    server_name: &str,
    client_cert_pem: Option<String>,
    client_key_pem: Option<String>,
) -> PyResult<()> {
    if ca_pem.is_empty() {
        bc_transport::set_client_tls(None);
        return Ok(());
    }
    let mut cfg = bc_transport::TlsClientConfig::new(ca_pem, server_name);
    match (client_cert_pem, client_key_pem) {
        (Some(cert), Some(key)) => {
            cfg = cfg.with_identity(bc_transport::TlsIdentity::from_pem(cert, key));
        }
        (None, None) => {}
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "client mTLS requires both a certificate and a private key",
            ))
        }
    }
    bc_transport::set_client_tls(Some(cfg));
    Ok(())
}

/// Whether a same-node shared-memory transfer directory is usable on this host (so the
/// control plane can avoid selecting SHARED_MEMORY where it would never work).
#[pyfunction]
fn shm_available() -> bool {
    bc_transport::shm_available()
}

/// A pooled, persistent shuffle consumer.
///
/// Holds a `ClientPool` for its lifetime, so a reducer's many `fetch`es reuse gRPC
/// channels (one per peer) instead of rebuilding them on every call as the free
/// `flight_fetch` does. This is the consumer-side scaling primitive: connection
/// setup is paid once per peer, not once per partition, so an all-to-all shuffle
/// costs O(peers) connections. Driven by the process-wide [`shared_runtime`].
#[pyclass]
struct ShuffleClient {
    pub(crate) pool: std::sync::Arc<bc_transport::ClientPool>,
}

#[pymethods]
impl ShuffleClient {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            pool: std::sync::Arc::new(bc_transport::ClientPool::new()),
        })
    }

    /// Fetch `ticket` from `addr` over a credit-gated stream on a pooled channel.
    #[pyo3(signature = (addr, ticket, credits=bc_transport::DEFAULT_CREDITS, token=None))]
    fn fetch(
        &self,
        py: Python<'_>,
        addr: &str,
        ticket: &str,
        credits: u32,
        token: Option<&str>,
    ) -> PyResult<Vec<PyArrowType<RecordBatch>>> {
        let t = bc_transport::ShuffleTicket::from_string(ticket).map_err(to_pyerr)?;
        let batches = py
            .allow_threads(|| {
                shared_runtime().block_on(self.pool.fetch_secured(addr, &t, credits, token))
            })
            .map_err(transport_to_pyerr)?;
        Ok(batches.into_iter().map(PyArrowType).collect())
    }

    /// Number of peers with a live cached channel (telemetry/tests).
    #[getter]
    fn connection_count(&self) -> usize {
        self.pool.connection_count()
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__engine_version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(tracing_init::init_tracing, m)?)?;
    m.add_function(wrap_pyfunction!(execute_plan, m)?)?;
    m.add_function(wrap_pyfunction!(execute_plan_metered, m)?)?;
    m.add_function(wrap_pyfunction!(read_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(read_avro, m)?)?;
    m.add_function(wrap_pyfunction!(read_parquet_filtered, m)?)?;
    m.add_function(wrap_pyfunction!(read_parquet_many, m)?)?;
    m.add_function(wrap_pyfunction!(partial_aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(execute_plan_aggregated, m)?)?;
    m.add_function(wrap_pyfunction!(combine, m)?)?;
    m.add_function(wrap_pyfunction!(combine_finalize, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle::combine_finalize_spilling, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle::partition_batches, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle::range_partition_batches, m)?)?;
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
    m.add_class::<FlightShuffleServer>()?;
    m.add_function(wrap_pyfunction!(flight_fetch, m)?)?;
    m.add_function(wrap_pyfunction!(set_flight_transport_config, m)?)?;
    m.add_function(wrap_pyfunction!(set_flight_client_tls, m)?)?;
    m.add_function(wrap_pyfunction!(shm_available, m)?)?;
    m.add_function(wrap_pyfunction!(supported_cast_dtypes, m)?)?;
    m.add_class::<ShuffleClient>()?;
    m.add_class::<MemoryPool>()?;
    // Classified shuffle-fetch exceptions: the control plane catches `Retryable` as
    // worker loss (recompute + retry) and lets `Fatal` propagate (fail fast).
    errors::register(m)?;
    Ok(())
}
