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

/// Validate partition-key inputs at the FFI boundary before they reach the engine.
///
/// An out-of-range key index or a zero partition count would otherwise index a column
/// out of bounds / trip a `debug_assert` deep in the runtime and **panic through the
/// FFI** — a `PanicException`, which derives from `BaseException` and so slips past a
/// caller's `except Exception`. Reject them here with a clean, catchable `ValueError`.
fn validate_partition_args(
    batches: &[RecordBatch],
    key_indices: &[usize],
    num_partitions: usize,
) -> PyResult<()> {
    if num_partitions == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_partitions must be >= 1",
        ));
    }
    for batch in batches {
        let ncols = batch.num_columns();
        if let Some(&bad) = key_indices.iter().find(|&&i| i >= ncols) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "key index {bad} out of range for a batch with {ncols} columns"
            )));
        }
    }
    Ok(())
}

/// Hash-shuffle batches into `num_partitions` buckets by the given key columns.
#[pyfunction]
pub(crate) fn partition_batches(
    batches: Vec<PyArrowType<RecordBatch>>,
    key_indices: Vec<usize>,
    num_partitions: usize,
) -> PyResult<Vec<Vec<PyArrowType<RecordBatch>>>> {
    let batches = unwrap_batches(batches)?;
    validate_partition_args(&batches, &key_indices, num_partitions)?;
    let parts = bc_interp::dist::partition_batches(&batches, &key_indices, num_partitions)
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
    let batches = unwrap_batches(batches)?;
    validate_partition_args(&batches, std::slice::from_ref(&key_index), n_buckets)?;
    let parts = bc_interp::dist::range_partition_batches(
        &batches,
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
    let batches = unwrap_batches(batches)?;
    validate_partition_args(&batches, &key_indices, num_partitions)?;
    let parts = bc_interp::dist::salted_partition_batches(
        &batches,
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

/// The node identity of a shuffle address — its host, dropping the `:port`. Advertised
/// addresses are `{node_ip}:{port}`, so equal hosts ⇒ same node (⇒ shm is reachable).
fn host_of(addr: &str) -> &str {
    addr.rsplit_once(':').map(|(h, _)| h).unwrap_or(addr)
}

/// Fetch every source concurrently, invoking `on_batches` for each non-empty result
/// as it arrives; returns the indices of sources that hit a *retryable* fault.
///
/// Co-located sources (`addr == own_addr`) read the local store with no socket. Remote
/// fetches run on the shared runtime, bounded by a `fan_in` semaphore so no more than
/// `fan_in` are in flight at once. A fatal fault aborts; a retryable one is collected.
///
/// `replicas[i]` holds the *fallback* addresses for source `i` — peers carrying a
/// byte-identical copy of that bucket under the same ticket (see the replication factor
/// in `DistributedConfig`). A retryable fault against one address transparently falls
/// over to the next, so losing a worker costs a re-fetch from a survivor rather than the
/// lineage recompute (re-read the source, re-run the map) it would otherwise force. A
/// source is reported unreachable only once *every* copy is gone, which is when the
/// driver's recompute loop is genuinely the right answer. Empty (the default) ⇒ the
/// single-address behavior, unchanged.
#[allow(clippy::too_many_arguments)]
async fn drive(
    own: &FlightShuffleServer,
    pool: Arc<bc_transport::ClientPool>,
    sources: &[(String, ShuffleTicket)],
    replicas: &[Vec<String>],
    credits: u32,
    fan_in: usize,
    token: Option<String>,
    shm: bool,
    mut on_batches: impl FnMut(Vec<RecordBatch>) -> Result<(), InterpError>,
) -> Result<Vec<usize>, GatherErr> {
    let mut unreachable = Vec::new();

    // Co-located buckets first — a cheap in-process read, no network, no permit.
    let own_addr = own.exchange.advertised_addr();
    let own_host = host_of(own_addr);
    let mut set: JoinSet<(usize, Result<Vec<RecordBatch>, TransportError>)> = JoinSet::new();
    let sem = Arc::new(Semaphore::new(fan_in.max(1)));

    // How many TCP flows to split EACH peer's bucket across. Batcher runs one Flight
    // endpoint per node, so a reducer pulls each node's whole bucket over a single
    // stream — one TCP flow, capped below the NIC's line rate. When there are few
    // distinct remote peers (the small-cluster / autoscaling-ramp / skew case), split
    // each bucket across several connections to use the whole link; when there are many
    // peers the gather is already flow-parallel across them, so don't over-connect.
    // `ceil(fan_in / distinct_peers)` keeps total concurrent streams ~`fan_in`, clamped
    // to the per-peer connection bound.
    let distinct_remote = sources
        .iter()
        .filter(|(a, _)| a.as_str() != own_addr)
        .map(|(a, _)| a.as_str())
        .collect::<std::collections::HashSet<_>>()
        .len();
    let stripe = if distinct_remote == 0 {
        1
    } else {
        (bc_transport::connections_per_peer() as u32)
            .min((fan_in.max(1)).div_ceil(distinct_remote) as u32)
            .max(1)
    };
    for (idx, (addr, ticket)) in sources.iter().enumerate() {
        // Every address carrying this bucket: the primary, then its replicas. They hold
        // byte-identical batches under the same ticket, so which one answers is invisible
        // to the result — only to how long it takes.
        let mut candidates: Vec<&str> =
            Vec::with_capacity(1 + replicas.get(idx).map_or(0, Vec::len));
        candidates.push(addr.as_str());
        candidates.extend(replicas.get(idx).into_iter().flatten().map(String::as_str));

        // A copy on this very worker is free (local store, no socket) wherever it sits in
        // the candidate list — so a replica that landed here also skips the network.
        if candidates.contains(&own_addr) {
            if let Some(batches) = own.exchange.local_partition(ticket).await {
                if !batches.is_empty() {
                    on_batches(batches).map_err(GatherErr::Combine)?;
                }
                continue;
            }
            // Not actually registered here — fall through to a remote copy.
        }
        let remote: Vec<String> = candidates
            .iter()
            .filter(|c| **c != own_addr)
            .map(|c| c.to_string())
            .collect();
        if remote.is_empty() {
            continue; // only copy is a local one that read back empty (unchanged behavior)
        }
        let (pool, sem, ticket, token) = (pool.clone(), sem.clone(), *ticket, token.clone());
        // Owned: the task outlives `own`'s borrow, so the co-location test needs its own copy.
        let own_host = own_host.to_string();
        set.spawn(async move {
            // Hold a permit for the whole fetch so at most `fan_in` stream concurrently.
            let _permit = sem.acquire_owned().await;
            let mut last: Option<TransportError> = None;
            // Try each copy in turn; a retryable fault (a lost/idle peer) falls over to the
            // next replica instead of failing the source. Only when every copy is gone does
            // this report the fault the driver recomputes from.
            for addr in &remote {
                // Same node, different process: a zero-copy shared-memory mmap read beats a
                // loopback Flight hop by ~20x. Try it inside the concurrent set (so cross-node
                // fetches still fan out in parallel) and fall back to Flight on a miss — the
                // producer may not have mirrored this bucket (shm off, or skipped under memory
                // pressure), which is a benign, result-preserving fallback.
                if shm && host_of(addr) == own_host.as_str() {
                    let (a, t) = (addr.clone(), ticket.to_string());
                    // shm read is blocking file I/O + decode → off the async reactor.
                    if let Ok(Ok(Some(batches))) =
                        tokio::task::spawn_blocking(move || bc_transport::fetch_shared(&a, &t))
                            .await
                    {
                        return (idx, Ok(batches));
                    }
                }
                match pool
                    .fetch_secured_striped(addr, &ticket, credits, token.as_deref(), stripe)
                    .await
                {
                    Ok(batches) => return (idx, Ok(batches)),
                    // A fatal fault (decode/protocol/auth) is not a lost peer — every replica
                    // would fail it identically, so fail fast instead of retrying the same bug.
                    Err(e) if matches!(classify(&e), FetchFault::Fatal) => return (idx, Err(e)),
                    Err(e) => last = Some(e),
                }
            }
            (idx, Err(last.expect("remote is non-empty")))
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
///
/// `replicas[i]` lists the fallback addresses holding a copy of source `i`'s bucket, so a
/// lost mapper is served from a survivor instead of recomputed (see `drive`).
#[pyfunction]
#[pyo3(signature = (server, client, group_keys_json, aggregates_json, sources, fan_in, finalize, credits=bc_transport::DEFAULT_CREDITS, token=None, shm=false, replicas=Vec::new()))]
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
    shm: bool,
    replicas: Vec<Vec<String>>,
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
            let unreachable = drive(
                server, pool, &sources, &replicas, credits, fan_in, token, shm, fold,
            )
            .await?;
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
///
/// `replicas[i]` lists the fallback addresses holding a copy of source `i`'s bucket, so a
/// lost mapper is served from a survivor instead of recomputed (see `drive`).
#[pyfunction]
#[pyo3(signature = (server, client, sources, fan_in, credits=bc_transport::DEFAULT_CREDITS, token=None, shm=false, replicas=Vec::new()))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_concat(
    py: Python<'_>,
    server: &FlightShuffleServer,
    client: &ShuffleClient,
    sources: Vec<(String, String)>,
    fan_in: usize,
    credits: u32,
    token: Option<String>,
    shm: bool,
    replicas: Vec<Vec<String>>,
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
            let unreachable = drive(
                server, pool, &sources, &replicas, credits, fan_in, token, shm, collect,
            )
            .await?;
            Ok((rows, unreachable))
        })
    });

    let (rows, unreachable) = out.map_err(GatherErr::into_pyerr)?;
    Ok((rows.into_iter().map(PyArrowType).collect(), unreachable))
}

/// Spilling distributed reduce: merge the partials the shuffle wrote to `input_paths`
/// (Arrow-IPC stream files, one per mapper for this reducer) and finalize, out of core.
///
/// The reducer's other half from [`gather_combine`]: where that folds every mapper's partial
/// into one running state in RAM, this reads the reducer's shuffle files one at a time and
/// grace-partitions them to disk under `memory_budget_bytes`, so a high-cardinality
/// `GROUP BY` / `DISTINCT` / `COUNT(DISTINCT)` whose merged group state exceeds one worker's
/// RAM completes instead of OOMing — the distributed arm of Batcher's out-of-core guarantee.
/// Result-identical to the in-memory reduce over the same partials (group order differs).
/// `spill_dir` is scratch for the grace partitions (defaults to the OS temp dir);
/// `spill_compression` selects the spill IPC codec (`"lz4"`/`"zstd"`/`"auto"`/None). The GIL
/// is released for the fold.
#[pyfunction]
#[pyo3(signature = (group_keys_json, aggregates_json, input_paths, memory_budget_bytes, spill_dir=None, spill_compression=None))]
pub(crate) fn combine_finalize_spilling(
    py: Python<'_>,
    group_keys_json: &str,
    aggregates_json: &str,
    input_paths: Vec<String>,
    memory_budget_bytes: usize,
    spill_dir: Option<String>,
    spill_compression: Option<String>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let group_keys = parse_group_keys(group_keys_json)?;
    let aggregates = parse_aggregates(aggregates_json)?;
    let paths: Vec<std::path::PathBuf> = input_paths.into_iter().map(Into::into).collect();
    let dir = spill_dir
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let out = py
        .allow_threads(|| {
            bc_interp::dist::combine_finalize_spilling(
                &group_keys,
                &aggregates,
                &paths,
                memory_budget_bytes,
                &dir,
                spill_compression.as_deref(),
            )
        })
        .map_err(to_pyerr)?;
    Ok(PyArrowType(out))
}

/// Parse the `(addr, ticket_string)` sources into `(addr, ShuffleTicket)`.
fn parse_sources(sources: Vec<(String, String)>) -> PyResult<Vec<(String, ShuffleTicket)>> {
    sources
        .into_iter()
        .map(|(addr, ticket)| Ok((addr, ShuffleTicket::from_string(&ticket).map_err(to_pyerr)?)))
        .collect()
}
