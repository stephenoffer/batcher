//! `bc-py` — the PyO3 boundary that assembles the Rust engine into the
//! `batcher._native` extension module.
//!
//! This crate is the *only* one that links PyO3; everything else is a pure-Rust
//! library that is `cargo test`/fuzz-testable without a Python interpreter. The
//! boundary is deliberately thin: the Python control plane builds a plan,
//! lowers it to the JSON IR, and ships it here alongside input relations as
//! pyarrow batches. Conversion is zero-copy via the Arrow C Data Interface, so a
//! `RecordBatch` crosses the boundary without serialization.

use std::sync::Arc;

use arrow::array::{Array, RecordBatch};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema};
use arrow_pyarrow::PyArrowType;
use bc_ir::{AggregateItem, EngineConfig, ProjectionItem, RelOp};
use bc_sketches::Mergeable; // brings `.merge()` into scope for ColumnStats
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

mod bloom;
mod errors;
mod process;
mod shuffle;
use errors::transport_to_pyerr;
use process::{shared_memory_pool, shared_runtime};

/// The widened type the engine's Int64/Float64 kernels operate on, or `None` to
/// leave a column as-is. Real-world data is full of narrow numerics (Int32 ids,
/// Float32 features, unsigned counts); normalizing them once at the boundary lets
/// every operator stay on the two well-tested numeric paths.
fn widen_to(dt: &DataType) -> Option<DataType> {
    use DataType::*;
    match dt {
        Int8 | Int16 | Int32 | UInt8 | UInt16 | UInt32 | UInt64 => Some(Int64),
        Float16 | Float32 => Some(Float64),
        _ => None,
    }
}

/// Upcast narrow numeric columns of one batch to Int64/Float64. Non-numeric and
/// already-wide columns are passed through untouched (a cheap `Arc` clone).
fn normalize_batch(batch: &RecordBatch) -> RecordBatch {
    let schema = batch.schema();
    let mut changed = false;
    let mut fields: Vec<Field> = Vec::with_capacity(schema.fields().len());
    let mut columns = Vec::with_capacity(batch.num_columns());
    for (i, field) in schema.fields().iter().enumerate() {
        let col = batch.column(i);
        match widen_to(col.data_type()) {
            Some(target) => match cast(col, &target) {
                Ok(arr) => {
                    changed = true;
                    fields.push(Field::new(field.name(), target, field.is_nullable()));
                    columns.push(arr);
                }
                Err(_) => {
                    fields.push(field.as_ref().clone());
                    columns.push(col.clone());
                }
            },
            None => {
                fields.push(field.as_ref().clone());
                columns.push(col.clone());
            }
        }
    }
    if !changed {
        return batch.clone();
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap_or_else(|_| batch.clone())
}

/// Unwrap a Python list of pyarrow batches into normalized Arrow record batches.
pub(crate) fn unwrap_batches(batches: Vec<PyArrowType<RecordBatch>>) -> Vec<RecordBatch> {
    batches.into_iter().map(|b| normalize_batch(&b.0)).collect()
}

pub(crate) fn parse_group_keys(json: &str) -> PyResult<Vec<ProjectionItem>> {
    serde_json::from_str(json).map_err(to_pyerr)
}

pub(crate) fn parse_aggregates(json: &str) -> PyResult<Vec<AggregateItem>> {
    serde_json::from_str(json).map_err(to_pyerr)
}

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
    let (plan, sources, opts) = prepare_exec(plan_json, sources, engine_config)?;
    let out = py
        .allow_threads(|| bc_interp::execute_parallel_with(&plan, &sources, &opts))
        .map_err(to_pyerr)?;
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
    let (plan, sources, opts) = prepare_exec(plan_json, sources, engine_config)?;
    let (out, metrics) = py
        .allow_threads(|| bc_interp::execute_parallel_with_metrics(&plan, &sources, &opts))
        .map_err(to_pyerr)?;
    Ok((
        out.into_iter().map(PyArrowType).collect(),
        metrics.to_json(),
    ))
}

/// Shared setup for the execute entry points: parse the plan + engine config and
/// normalize the input morsels (narrow numeric types → Int64/Float64) once.
fn prepare_exec(
    plan_json: &str,
    sources: Vec<Vec<PyArrowType<RecordBatch>>>,
    engine_config: &str,
) -> PyResult<(RelOp, Vec<Vec<RecordBatch>>, bc_interp::ExecOptions)> {
    let plan = RelOp::from_json(plan_json).map_err(to_pyerr)?;
    let cfg = EngineConfig::from_json(engine_config).map_err(to_pyerr)?;
    let mut opts = bc_interp::ExecOptions::default().with_engine_config(&cfg);
    // A positive budget activates the runtime memory backstop via the *process-wide*
    // pool (per-query pools would let N concurrent queries each hold `budget` and OOM).
    // Zero budget ⇒ no pool ⇒ the fast path pays nothing.
    if cfg.memory_budget_bytes > 0 {
        opts.pool = Some(shared_memory_pool(cfg.memory_budget_bytes));
    }
    let sources: Vec<Vec<RecordBatch>> = sources
        .into_iter()
        .map(|relation| {
            relation
                .into_iter()
                .map(|b| normalize_batch(&b.0))
                .collect()
        })
        .collect();
    Ok((plan, sources, opts))
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
    let out =
        bc_interp::dist::partial_aggregate(&group_keys, &aggregates, &unwrap_batches(batches))
            .map_err(to_pyerr)?;
    Ok(PyArrowType(out))
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
    let out =
        bc_interp::dist::combine_finalize(&group_keys, &aggregates, &unwrap_batches(partials))
            .map_err(to_pyerr)?;
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
    let out = bc_interp::dist::combine(&group_keys, &aggregates, &unwrap_batches(partials))
        .map_err(to_pyerr)?;
    Ok(PyArrowType(out))
}

/// Estimate the number of distinct (non-null) values in a column across batches,
/// using HyperLogLog++. Mergeable, so it can be computed per partition.
#[pyfunction]
fn estimate_distinct(column: &str, batches: Vec<PyArrowType<RecordBatch>>) -> PyResult<f64> {
    let mut sketch: Option<bc_sketches::ColumnStats> = None;
    for batch in batches {
        let b = batch.0;
        let col = b
            .column_by_name(column)
            .ok_or_else(|| PyRuntimeError::new_err(format!("no column {column:?}")))?;
        let stats = bc_sketches::ColumnStats::from_array(col);
        match &mut sketch {
            Some(s) => s.merge(&stats),
            None => sketch = Some(stats),
        }
    }
    Ok(sketch.map_or(0.0, |s| s.distinct_estimate()))
}

/// Per-column statistics for the optimizer (the W2 metadata FFI seam): for each
/// requested column, merge `ColumnStats` (HLL distinct + KLL quantiles) across all
/// batches and return a dict of scalar summaries. Keys per column:
/// `ndv` (distinct estimate), `count`, `null_count`, `null_fraction`, `avg_bytes`
/// (measured per-row byte width), and `min`/`max` (`None` for non-numeric columns).
/// Mergeable, so it composes across partitions — Core can collect this during
/// execution and persist it to the MetadataHub for Kyber's `__column_ndv__` /
/// `__column_avg_bytes__` / range-selectivity to consume.
#[pyfunction]
fn column_stats(
    columns: Vec<String>,
    batches: Vec<PyArrowType<RecordBatch>>,
) -> PyResult<std::collections::HashMap<String, std::collections::HashMap<String, Option<f64>>>> {
    let mut merged: std::collections::HashMap<String, bc_sketches::ColumnStats> =
        std::collections::HashMap::new();
    for batch in &batches {
        let b = &batch.0;
        for name in &columns {
            if let Some(col) = b.column_by_name(name) {
                let stats = bc_sketches::ColumnStats::from_array(col);
                merged
                    .entry(name.clone())
                    .and_modify(|s| s.merge(&stats))
                    .or_insert(stats);
            }
        }
    }
    let mut out = std::collections::HashMap::new();
    for (name, s) in merged {
        let mut d = std::collections::HashMap::new();
        d.insert("ndv".to_string(), Some(s.distinct_estimate()));
        d.insert("count".to_string(), Some(s.count as f64));
        d.insert("null_count".to_string(), Some(s.null_count as f64));
        d.insert("null_fraction".to_string(), Some(s.null_fraction()));
        d.insert("avg_bytes".to_string(), Some(s.avg_byte_width()));
        d.insert("min".to_string(), s.min());
        d.insert("max".to_string(), s.max());
        out.insert(name, d);
    }
    Ok(out)
}

/// Per-column quantile boundaries (the KLL sketch) for histogram-based range
/// selectivity in the optimizer. For each numeric column, return the value at each
/// requested probability in `probs` (so Kyber can interpolate `fraction <= literal`);
/// non-numeric columns return an empty list. Mergeable across batches, so Core can
/// collect it online and persist it to the MetadataHub alongside `column_stats`.
#[pyfunction]
fn column_quantiles(
    columns: Vec<String>,
    batches: Vec<PyArrowType<RecordBatch>>,
    probs: Vec<f64>,
) -> PyResult<std::collections::HashMap<String, Vec<f64>>> {
    let mut merged: std::collections::HashMap<String, bc_sketches::ColumnStats> =
        std::collections::HashMap::new();
    for batch in &batches {
        let b = &batch.0;
        for name in &columns {
            if let Some(col) = b.column_by_name(name) {
                let stats = bc_sketches::ColumnStats::from_array(col);
                merged
                    .entry(name.clone())
                    .and_modify(|s| s.merge(&stats))
                    .or_insert(stats);
            }
        }
    }
    let mut out = std::collections::HashMap::new();
    for (name, s) in merged {
        // A full set of boundaries only exists for numeric columns (KLL present);
        // if any probability has no quantile, return an empty list for that column.
        let vals: Vec<f64> = probs.iter().filter_map(|&q| s.quantile(q)).collect();
        out.insert(
            name,
            if vals.len() == probs.len() {
                vals
            } else {
                Vec::new()
            },
        );
    }
    Ok(out)
}

/// Tail-accurate quantiles (the TDigest sketch) for numeric columns. Where the
/// coarse KLL grid in `column_quantiles` is built for range selectivity, TDigest
/// is accurate in the tails (p99/p999) — what an `approx_quantile` answer wants.
/// For each numeric column, returns the value at each requested probability;
/// non-numeric or empty columns return an empty list. Mergeable across batches.
#[pyfunction]
fn tail_quantiles(
    columns: Vec<String>,
    batches: Vec<PyArrowType<RecordBatch>>,
    probs: Vec<f64>,
) -> PyResult<std::collections::HashMap<String, Vec<f64>>> {
    let mut digests: std::collections::HashMap<String, bc_sketches::TDigest> =
        std::collections::HashMap::new();
    for batch in &batches {
        let b = &batch.0;
        for name in &columns {
            if let Some(col) = b.column_by_name(name) {
                let Ok(f) = cast(col, &DataType::Float64) else {
                    continue;
                };
                let Some(arr) = f.as_any().downcast_ref::<arrow::array::Float64Array>() else {
                    continue;
                };
                let d = digests.entry(name.clone()).or_default();
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        d.add(arr.value(i));
                    }
                }
            }
        }
    }
    let mut out = std::collections::HashMap::new();
    for (name, mut d) in digests {
        let vals: Vec<f64> = probs.iter().filter_map(|&q| d.quantile(q)).collect();
        out.insert(
            name,
            if vals.len() == probs.len() {
                vals
            } else {
                Vec::new()
            },
        );
    }
    Ok(out)
}

/// Heavy hitters (the Misra-Gries `FrequentItems` sketch) per column: the values
/// whose frequency exceeds `fraction` of the rows, with their estimated counts.
/// Kyber consumes this for skew detection (a hot join key → salting). Values are
/// rendered to strings (cast to Utf8) so any column type can be labelled; columns
/// that cannot cast are skipped. Mergeable in spirit — built across all batches.
#[pyfunction]
fn heavy_hitters(
    columns: Vec<String>,
    batches: Vec<PyArrowType<RecordBatch>>,
    fraction: f64,
) -> PyResult<std::collections::HashMap<String, Vec<(String, u64)>>> {
    // Misra-Gries capacity: 1/fraction guarantees all keys above `fraction` survive.
    let capacity = ((1.0 / fraction).ceil() as usize).max(1);
    let mut items: std::collections::HashMap<String, bc_sketches::FrequentItems<String>> =
        std::collections::HashMap::new();
    for batch in &batches {
        let b = &batch.0;
        for name in &columns {
            if let Some(col) = b.column_by_name(name) {
                let Ok(s) = cast(col, &DataType::Utf8) else {
                    continue;
                };
                let Some(arr) = s.as_any().downcast_ref::<arrow::array::StringArray>() else {
                    continue;
                };
                let fi = items
                    .entry(name.clone())
                    .or_insert_with(|| bc_sketches::FrequentItems::new(capacity));
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        fi.add(arr.value(i).to_string());
                    }
                }
            }
        }
    }
    let mut out = std::collections::HashMap::new();
    for (name, fi) in items {
        out.insert(name, fi.heavy_hitters(fraction));
    }
    Ok(out)
}

/// A uniform random row sample (the reservoir sketch, Algorithm R) of size `k`
/// across all batches, returned as one `RecordBatch`. Used for sampling-based
/// estimation / `TABLESAMPLE` without materializing the whole input. When the
/// input has at most `k` rows, returns them all.
#[pyfunction]
fn reservoir_sample(
    batches: Vec<PyArrowType<RecordBatch>>,
    k: usize,
) -> PyResult<PyArrowType<RecordBatch>> {
    use arrow::array::UInt64Array;

    if batches.is_empty() {
        return Err(PyRuntimeError::new_err(
            "reservoir_sample: no input batches",
        ));
    }
    let schema = batches[0].0.schema();
    let refs: Vec<&RecordBatch> = batches.iter().map(|b| &b.0).collect();
    let combined = arrow::compute::concat_batches(&schema, refs)
        .map_err(|e| PyRuntimeError::new_err(format!("concat failed: {e}")))?;
    let total = combined.num_rows();
    if total <= k {
        return Ok(PyArrowType(combined));
    }
    // Reservoir of global row indices; deterministic seed keeps it reproducible.
    let mut reservoir = bc_sketches::ReservoirSample::new(k);
    for idx in 0..total {
        reservoir.add(idx as u64);
    }
    let indices = UInt64Array::from(reservoir.sample().to_vec());
    let mut cols = Vec::with_capacity(combined.num_columns());
    for col in combined.columns() {
        let taken = arrow::compute::take(col, &indices, None)
            .map_err(|e| PyRuntimeError::new_err(format!("take failed: {e}")))?;
        cols.push(taken);
    }
    let sampled = RecordBatch::try_new(schema, cols)
        .map_err(|e| PyRuntimeError::new_err(format!("rebatch failed: {e}")))?;
    Ok(PyArrowType(sampled))
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
    #[new]
    #[pyo3(signature = (advertise_host=None, token=None))]
    fn new(advertise_host: Option<String>, token: Option<String>) -> PyResult<Self> {
        let host = advertise_host.filter(|h| !h.is_empty());
        let token = token.filter(|t| !t.is_empty());
        let exchange = shared_runtime()
            .block_on(async {
                match (host.as_deref(), token) {
                    (Some(h), tok) => {
                        bc_transport::ShuffleExchange::bind_secured("0.0.0.0:0", Some(h), tok).await
                    }
                    (None, None) => bc_transport::ShuffleExchange::bind_ephemeral().await,
                    (None, tok @ Some(_)) => {
                        // Auth on a single-host loopback server (e.g. tests).
                        bc_transport::ShuffleExchange::bind_secured("127.0.0.1:0", None, tok).await
                    }
                }
            })
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
        let batches: Vec<RecordBatch> = batches.iter().map(|b| normalize_batch(&b.0)).collect();
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
    /// a write error is swallowed (the reducer falls back to Flight).
    fn publish_shared(&self, py: Python<'_>, ticket: &str, batches: Vec<PyArrowType<RecordBatch>>) {
        let batches: Vec<RecordBatch> = batches.iter().map(|b| normalize_batch(&b.0)).collect();
        let addr = self.addr.clone();
        py.allow_threads(|| {
            let _ = bc_transport::publish_shared(&addr, ticket, &batches);
        });
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

/// Set the process-wide Flight transport timeouts from the control plane.
///
/// `idle_timeout_ms` bounds the gap between batches before a peer is treated as
/// dead (`0` keeps the current value); `keepalive_ms` is the HTTP/2 keepalive ping
/// interval (`0` off). Called once per worker process when its Flight server starts.
#[pyfunction]
#[pyo3(signature = (idle_timeout_ms, keepalive_ms=0))]
fn set_flight_transport_config(idle_timeout_ms: u64, keepalive_ms: u64) {
    bc_transport::set_transport_timeouts(idle_timeout_ms, keepalive_ms);
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
    m.add_function(wrap_pyfunction!(execute_plan, m)?)?;
    m.add_function(wrap_pyfunction!(execute_plan_metered, m)?)?;
    m.add_function(wrap_pyfunction!(partial_aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(combine, m)?)?;
    m.add_function(wrap_pyfunction!(combine_finalize, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle::partition_batches, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle::range_partition_batches, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle::salted_partition_batches, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle::gather_combine, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle::gather_concat, m)?)?;
    m.add_function(wrap_pyfunction!(bloom::build_key_bloom, m)?)?;
    m.add_function(wrap_pyfunction!(bloom::merge_blooms, m)?)?;
    m.add_function(wrap_pyfunction!(bloom::bloom_filter_batches, m)?)?;
    m.add_function(wrap_pyfunction!(bloom::build_column_bloom, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_distinct, m)?)?;
    m.add_function(wrap_pyfunction!(column_stats, m)?)?;
    m.add_function(wrap_pyfunction!(column_quantiles, m)?)?;
    m.add_function(wrap_pyfunction!(tail_quantiles, m)?)?;
    m.add_function(wrap_pyfunction!(heavy_hitters, m)?)?;
    m.add_function(wrap_pyfunction!(reservoir_sample, m)?)?;
    m.add_class::<FlightShuffleServer>()?;
    m.add_function(wrap_pyfunction!(flight_fetch, m)?)?;
    m.add_function(wrap_pyfunction!(set_flight_transport_config, m)?)?;
    m.add_function(wrap_pyfunction!(shm_available, m)?)?;
    m.add_class::<ShuffleClient>()?;
    m.add_class::<MemoryPool>()?;
    // Classified shuffle-fetch exceptions: the control plane catches `Retryable` as
    // worker loss (recompute + retry) and lets `Fatal` propagate (fail fast).
    errors::register(m)?;
    Ok(())
}
