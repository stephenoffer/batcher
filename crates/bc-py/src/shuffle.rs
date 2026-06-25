//! Shuffle FFI: partitioners and the concurrent reducer gather.
//!
//! The partitioners (`partition_batches` / `range_partition_batches` /
//! `salted_partition_batches`) are thin wrappers over the mergeable `bc-interp`
//! kernels — a mapper splits its output into one bucket per reducer.
//!
//! The **gather** primitives are the reducer's other half. A reducer must pull its
//! bucket from *every* mapper; doing that one blocking `fetch` at a time costs `W`
//! sequential network round-trips. `gather_combine`/`gather_concat` instead fetch all
//! mappers **concurrently** on the shared tokio runtime (bounded by a `fan_in`
//! semaphore so peak memory stays independent of `W`) and fold the results in Rust:
//!
//! * `gather_combine` incrementally `combine`s aggregate partials into one running
//!   state — `combine` is associative+commutative, so a completion-order fold is
//!   bit-identical to the serial mapper-order fold. Memory is one running partial
//!   (sized by the group count) plus `fan_in` in-flight fetches.
//! * `gather_concat` collects raw rows (window / sort / join reducers, which need the
//!   whole bucket and re-establish order downstream).
//!
//! Co-located sources (the reducer fetching its own published bucket) take the
//! `local_partition` no-socket path. A *retryable* fetch fault (a lost/idle peer) is
//! reported as that source's index so the driver recomputes it and retries — exactly
//! the existing `("retry", srcs)` contract; a *fatal* fault propagates and fails the
//! query fast. Ticket minting and epoch/plan fencing stay in Python: Rust only sees
//! opaque ticket strings.

use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow_pyarrow::PyArrowType;
use bc_interp::InterpError;
use bc_transport::{classify, FetchFault, ShuffleTicket, TransportError};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::errors::transport_to_pyerr;
use crate::process::shared_runtime;
use crate::{parse_aggregates, parse_group_keys, to_pyerr, unwrap_batches};
use crate::{FlightShuffleServer, ShuffleClient};

/// Hash-shuffle batches into `num_partitions` buckets by the given key columns.
#[pyfunction]
pub(crate) fn partition_batches(
    batches: Vec<PyArrowType<RecordBatch>>,
    key_indices: Vec<usize>,
    num_partitions: usize,
) -> PyResult<Vec<Vec<PyArrowType<RecordBatch>>>> {
    let parts =
        bc_interp::dist::partition_batches(&unwrap_batches(batches), &key_indices, num_partitions)
            .map_err(to_pyerr)?;
    Ok(wrap_buckets(parts))
}

/// Range-shuffle batches into `n_buckets` globally-ordered buckets by the leading
/// sort key at `key_index` and the ascending `boundaries` — the distributed-sort
/// counterpart of `partition_batches`. Nulls route to the front/back bucket per
/// `nulls_first`/`descending` to match single-node null ordering. The key must be a
/// numeric column (compared as f64, matching the prior NumPy `searchsorted` path).
#[pyfunction]
pub(crate) fn range_partition_batches(
    batches: Vec<PyArrowType<RecordBatch>>,
    key_index: usize,
    boundaries: Vec<f64>,
    n_buckets: usize,
    nulls_first: bool,
    descending: bool,
) -> PyResult<Vec<Vec<PyArrowType<RecordBatch>>>> {
    let parts = bc_interp::dist::range_partition_batches(
        &unwrap_batches(batches),
        key_index,
        &boundaries,
        n_buckets,
        nulls_first,
        descending,
    )
    .map_err(to_pyerr)?;
    Ok(wrap_buckets(parts))
}

/// Skew-aware shuffle for a single-key distributed join: a hot key's rows are
/// salted across reducers instead of overloading one. `hot_keys` are the hot values
/// rendered as strings (matching `heavy_hitters`); `replicate=false` is the probe
/// side (one salted bucket per hot row), `replicate=true` the build side (every hot
/// row to all salted buckets). Cold keys hash identically to `partition_batches`, so
/// the joined relation is unchanged — only the hot key's work fans across reducers.
#[pyfunction]
pub(crate) fn salted_partition_batches(
    batches: Vec<PyArrowType<RecordBatch>>,
    key_indices: Vec<usize>,
    num_partitions: usize,
    hot_keys: Vec<String>,
    salt_count: u32,
    replicate: bool,
) -> PyResult<Vec<Vec<PyArrowType<RecordBatch>>>> {
    let hot: std::collections::HashSet<String> = hot_keys.into_iter().collect();
    let parts = bc_interp::dist::salted_partition_batches(
        &unwrap_batches(batches),
        &key_indices,
        num_partitions,
        &hot,
        salt_count,
        replicate,
    )
    .map_err(to_pyerr)?;
    Ok(wrap_buckets(parts))
}

fn wrap_buckets(parts: Vec<Vec<RecordBatch>>) -> Vec<Vec<PyArrowType<RecordBatch>>> {
    parts
        .into_iter()
        .map(|bucket| bucket.into_iter().map(PyArrowType).collect())
        .collect()
}

/// A reducer fetch failure that must surface as a Python exception once the GIL is
/// re-acquired (a `PyErr` cannot be built while the GIL is released inside the runtime).
enum GatherErr {
    /// A fatal transport fault (decode/protocol/auth) — fail the query fast.
    Fatal(TransportError),
    /// A combine/finalize error over the fetched partials.
    Combine(InterpError),
    /// A fetch task panicked or was cancelled.
    Join(String),
}

impl GatherErr {
    fn into_pyerr(self) -> PyErr {
        match self {
            GatherErr::Fatal(e) => transport_to_pyerr(e),
            GatherErr::Combine(e) => to_pyerr(e),
            GatherErr::Join(m) => PyRuntimeError::new_err(m),
        }
    }
}

/// Fetch every source concurrently, invoking `on_batches` for each non-empty result
/// as it arrives; returns the indices of sources that hit a *retryable* fault.
///
/// Co-located sources (`addr == own_addr`) read the local store with no socket. Remote
/// fetches run on the shared runtime, bounded by a `fan_in` semaphore so no more than
/// `fan_in` are in flight at once. A fatal fault aborts; a retryable one is collected.
async fn drive(
    own: &FlightShuffleServer,
    pool: Arc<bc_transport::ClientPool>,
    sources: &[(String, ShuffleTicket)],
    credits: u32,
    fan_in: usize,
    token: Option<String>,
    mut on_batches: impl FnMut(Vec<RecordBatch>) -> Result<(), InterpError>,
) -> Result<Vec<usize>, GatherErr> {
    let mut unreachable = Vec::new();

    // Co-located buckets first — a cheap in-process read, no network, no permit.
    let own_addr = own.exchange.advertised_addr();
    let mut set: JoinSet<(usize, Result<Vec<RecordBatch>, TransportError>)> = JoinSet::new();
    let sem = Arc::new(Semaphore::new(fan_in.max(1)));
    for (idx, (addr, ticket)) in sources.iter().enumerate() {
        if addr.as_str() == own_addr {
            let batches = own
                .exchange
                .local_partition(ticket)
                .await
                .unwrap_or_default();
            if !batches.is_empty() {
                on_batches(batches).map_err(GatherErr::Combine)?;
            }
            continue;
        }
        let (pool, sem, addr, ticket, token) = (
            pool.clone(),
            sem.clone(),
            addr.clone(),
            *ticket,
            token.clone(),
        );
        set.spawn(async move {
            // Hold a permit for the whole fetch so at most `fan_in` stream concurrently.
            let _permit = sem.acquire_owned().await;
            let res = pool
                .fetch_secured(&addr, &ticket, credits, token.as_deref())
                .await;
            (idx, res)
        });
    }

    while let Some(joined) = set.join_next().await {
        let (idx, res) = joined.map_err(|e| GatherErr::Join(e.to_string()))?;
        match res {
            Ok(batches) if batches.is_empty() => {}
            Ok(batches) => on_batches(batches).map_err(GatherErr::Combine)?,
            Err(e) => match classify(&e) {
                FetchFault::Retryable => unreachable.push(idx),
                FetchFault::Fatal => return Err(GatherErr::Fatal(e)),
            },
        }
    }
    unreachable.sort_unstable();
    Ok(unreachable)
}

/// Concurrently gather aggregate partials from every `(addr, ticket)` source and fold
/// them into one merged partial (or, when `finalize`, the finalized output rows).
///
/// Returns `(payload, unreachable)`. If `unreachable` is non-empty the payload is
/// `None` (the state is incomplete — the driver recomputes those sources and retries);
/// otherwise the payload is the single combined/finalized batch, or `None` when every
/// bucket was empty. This is the concurrent replacement for the serial per-mapper
/// fetch+combine loop, with peak memory bounded by `fan_in` in-flight fetches plus the
/// one running state.
#[pyfunction]
#[pyo3(signature = (server, client, group_keys_json, aggregates_json, sources, fan_in, finalize, credits=bc_transport::DEFAULT_CREDITS, token=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_combine(
    py: Python<'_>,
    server: &FlightShuffleServer,
    client: &ShuffleClient,
    group_keys_json: &str,
    aggregates_json: &str,
    sources: Vec<(String, String)>,
    fan_in: usize,
    finalize: bool,
    credits: u32,
    token: Option<String>,
) -> PyResult<(Option<PyArrowType<RecordBatch>>, Vec<usize>)> {
    let group_keys = parse_group_keys(group_keys_json)?;
    let aggregates = parse_aggregates(aggregates_json)?;
    let sources = parse_sources(sources)?;
    let pool = client.pool.clone();

    let out: Result<(Option<RecordBatch>, Vec<usize>), GatherErr> = py.allow_threads(|| {
        shared_runtime().block_on(async {
            let mut running: Option<RecordBatch> = None;
            let fold = |batches: Vec<RecordBatch>| -> Result<(), InterpError> {
                let merged: Vec<RecordBatch> = match running.take() {
                    Some(r) => std::iter::once(r).chain(batches).collect(),
                    None => batches,
                };
                running = Some(bc_interp::dist::combine(&group_keys, &aggregates, &merged)?);
                Ok(())
            };
            let unreachable = drive(server, pool, &sources, credits, fan_in, token, fold).await?;
            if !unreachable.is_empty() {
                return Ok((None, unreachable)); // incomplete → driver recomputes + retries
            }
            let payload = match running {
                Some(state) if finalize => Some(
                    bc_interp::dist::combine_finalize(&group_keys, &aggregates, &[state])
                        .map_err(GatherErr::Combine)?,
                ),
                other => other,
            };
            Ok((payload, Vec::new()))
        })
    });

    let (payload, unreachable) = out.map_err(GatherErr::into_pyerr)?;
    Ok((payload.map(PyArrowType), unreachable))
}

/// Concurrently gather raw batches from every `(addr, ticket)` source into one list —
/// the window/sort/join reducer pattern, which needs the whole bucket and re-orders it
/// downstream. Returns `(batches, unreachable)`; a non-empty `unreachable` leaves the
/// batches partial (the driver recomputes and retries), matching `gather_combine`.
#[pyfunction]
#[pyo3(signature = (server, client, sources, fan_in, credits=bc_transport::DEFAULT_CREDITS, token=None))]
pub(crate) fn gather_concat(
    py: Python<'_>,
    server: &FlightShuffleServer,
    client: &ShuffleClient,
    sources: Vec<(String, String)>,
    fan_in: usize,
    credits: u32,
    token: Option<String>,
) -> PyResult<(Vec<PyArrowType<RecordBatch>>, Vec<usize>)> {
    let sources = parse_sources(sources)?;
    let pool = client.pool.clone();

    let out: Result<(Vec<RecordBatch>, Vec<usize>), GatherErr> = py.allow_threads(|| {
        shared_runtime().block_on(async {
            let mut rows: Vec<RecordBatch> = Vec::new();
            let collect = |batches: Vec<RecordBatch>| -> Result<(), InterpError> {
                rows.extend(batches);
                Ok(())
            };
            let unreachable =
                drive(server, pool, &sources, credits, fan_in, token, collect).await?;
            Ok((rows, unreachable))
        })
    });

    let (rows, unreachable) = out.map_err(GatherErr::into_pyerr)?;
    Ok((rows.into_iter().map(PyArrowType).collect(), unreachable))
}

/// Parse the `(addr, ticket_string)` sources into `(addr, ShuffleTicket)`.
fn parse_sources(sources: Vec<(String, String)>) -> PyResult<Vec<(String, ShuffleTicket)>> {
    sources
        .into_iter()
        .map(|(addr, ticket)| Ok((addr, ShuffleTicket::from_string(&ticket).map_err(to_pyerr)?)))
        .collect()
}
