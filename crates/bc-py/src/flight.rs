//! Flight FFI: the Arrow Flight shuffle transport surface exposed to Python.
//!
//! This module owns everything the control plane touches to move shuffle data
//! node→node: the node-local [`FlightShuffleServer`] mappers publish onto, the
//! pooled [`ShuffleClient`] and free [`flight_fetch`] reducers pull with, and the
//! process-wide transport tunables (timeouts, connection striping, compression,
//! TLS/mTLS). Bulk batches ride `bc-transport`'s credit-bounded `DoExchange`
//! streams — **never the Ray object store**; only address and ticket strings
//! transit Ray. Every blocking call runs on the process-wide
//! [`shared_runtime`](crate::process::shared_runtime) with the GIL released.

use arrow::array::RecordBatch;
use arrow_pyarrow::PyArrowType;
use pyo3::prelude::*;

use crate::errors::transport_to_pyerr;
use crate::normalize::normalize_batch;
use crate::process::shared_runtime;
use crate::to_pyerr;

/// A node-local Arrow Flight shuffle server. Each distributed worker holds one;
/// mappers `publish` their output partitions on it and advertise `addr`, and
/// reducers `flight_fetch` those partitions over the network with credit-bounded
/// streaming — moving shuffle data node→node **without ever touching the Ray
/// object store** (only the tiny address/ticket strings transit Ray).
///
/// Background serving runs on the process-wide [`shared_runtime`]; the exchange's
/// own `ServerHandle` keeps this server's serve task alive for the object's life.
#[pyclass]
pub(crate) struct FlightShuffleServer {
    pub(crate) exchange: std::sync::Arc<bc_transport::ShuffleExchange>,
    addr: String,
    /// Keeps this server's [`ShuffleSpiller`] registration alive.
    ///
    /// The pool holds only a `Weak`, so it is this field that decides how long the shuffle
    /// store is a spill candidate: exactly as long as the server serving it exists. Drop
    /// the server and the entry goes dead, to be swept on the pool's next slow path.
    _spiller: std::sync::Arc<dyn bc_resource::Spillable>,
}

/// The published shuffle store, offered to the memory pool as something it may spill.
///
/// **The first real occupant of a seam that had none.** `MemoryPool::try_reserve_cooperative`
/// is written to ask registered consumers to yield memory before it refuses a reservation,
/// but nothing in the workspace implemented [`bc_resource::Spillable`], so it behaved
/// exactly as the plain `try_reserve` and every over-budget operator spilled *itself*
/// regardless of what else on the node was holding more.
///
/// Published shuffle output is the right first answer to that. It is finished work sitting
/// idle waiting to be collected, so writing it to disk stalls nobody and costs one re-read
/// — where spilling the natural alternatives (an aggregate's half-built hash table, a
/// sort's in-progress runs) interrupts an operator that is actively using them, mid-
/// `par_iter`, and is a much larger piece of work. It is also the memory Carbonite's pool
/// cannot otherwise see at all, which makes it exactly the memory a reservation is most
/// likely to be losing to.
struct ShuffleSpiller {
    exchange: std::sync::Arc<bc_transport::ShuffleExchange>,
}

impl bc_resource::Spillable for ShuffleSpiller {
    /// Free at least `target` bytes of published buckets, returning what was freed.
    ///
    /// Never re-enters the pool (it only writes files and drops `Arc`s), and never blocks:
    /// the pool may call this from a tokio worker thread, where waiting on the store's lock
    /// would deadlock the runtime serving the fetches that would otherwise drain it. A busy
    /// store returns `0`, which the pool reads as "cannot help right now" and moves on to
    /// the next consumer.
    fn spill(&self, target: usize) -> usize {
        self.exchange.try_spill_at_least(target)
    }

    /// Resident published bytes — the pool spills the largest consumer first.
    fn spillable_bytes(&self) -> usize {
        self.exchange.retained_bytes()
    }
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
    ///
    /// `port_min`/`port_max` (both or neither) confine the listener to a closed port
    /// range instead of taking an OS-ephemeral port. An ephemeral port is the right
    /// default — it never collides — but it forces a firewalled cluster to open the whole
    /// ephemeral range node-to-node, which many on-prem and locked-down cloud networks
    /// will not do. Given a range, the server takes the first free port in it, so the
    /// operator can open exactly that range. The range must be wide enough for every
    /// worker sharing a node; too narrow and the bind fails with a message saying so
    /// rather than silently falling back to an unreachable port.
    #[new]
    #[pyo3(signature = (advertise_host=None, token=None, tls_cert_pem=None, tls_key_pem=None, tls_client_ca_pem=None, port_min=None, port_max=None))]
    fn new(
        advertise_host: Option<String>,
        token: Option<String>,
        tls_cert_pem: Option<String>,
        tls_key_pem: Option<String>,
        tls_client_ca_pem: Option<String>,
        port_min: Option<u16>,
        port_max: Option<u16>,
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
        let iface = if host.is_some() {
            "0.0.0.0"
        } else {
            "127.0.0.1"
        };
        let ports: Vec<u16> = match (port_min, port_max) {
            (Some(lo), Some(hi)) => {
                if lo > hi {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "shuffle port range is empty: port_min ({lo}) > port_max ({hi})"
                    )));
                }
                (lo..=hi).collect()
            }
            (None, None) => vec![0], // OS-ephemeral: one attempt, never collides.
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "shuffle port range needs both port_min and port_max, or neither",
                ))
            }
        };
        // Try each port in turn. Every failure is retried rather than only "address in
        // use": several workers racing for the same node's range fail here in more ways
        // than one, and the meaningful signal is "no port in the range worked", which the
        // error below reports with the range the operator actually configured.
        let mut last_err = None;
        for port in &ports {
            let bind = format!("{iface}:{port}");
            match shared_runtime().block_on(bc_transport::ShuffleExchange::bind_tls(
                &bind,
                host.as_deref(),
                token.clone(),
                tls.clone(),
            )) {
                Ok(exchange) => {
                    let addr = exchange.advertised_addr().to_string();
                    let exchange = std::sync::Arc::new(exchange);
                    let spiller: std::sync::Arc<dyn bc_resource::Spillable> =
                        std::sync::Arc::new(ShuffleSpiller {
                            exchange: std::sync::Arc::clone(&exchange),
                        });
                    // `0` gets-or-creates the process pool without disturbing its limit,
                    // which only ever grows: a server built before the first `execute_plan`
                    // registers against a pool the first query then sizes. Nothing reserves
                    // in between, because reservations happen inside `execute_plan`, after
                    // it has set the budget.
                    crate::process::shared_memory_pool(0).register_consumer(&spiller);
                    return Ok(Self {
                        exchange,
                        addr,
                        _spiller: spiller,
                    });
                }
                Err(e) => last_err = Some(e),
            }
        }
        let err = last_err.map(|e| e.to_string()).unwrap_or_default();
        Err(pyo3::exceptions::PyOSError::new_err(format!(
            "could not bind a shuffle port in {}-{} on {iface}; widen the range so every \
             worker on a node gets one (last error: {err})",
            ports.first().copied().unwrap_or(0),
            ports.last().copied().unwrap_or(0),
        )))
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
        let addr = self.addr.clone();
        py.allow_threads(|| {
            shared_runtime().block_on(self.exchange.clear_plan(plan_id));
            // The shm half, which nothing freed before. `/dev/shm` is RAM-backed, so a
            // stale bucket file is a second memory leak on the same node — and unlike the
            // in-memory store it has no eviction of any kind. `clear`/`clear_shared` are
            // already paired this way; this pairs the plan-scoped form to match.
            bc_transport::clear_plan_shared(&addr, plan_id);
        });
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

    /// Bytes this server's published partitions currently hold in memory.
    ///
    /// The shuffle's resident footprint, which Carbonite's buffer pool does not account
    /// for: a published partition is never *reserved*, it is simply held until a reducer
    /// fetches it. `PressureMonitor` names this store as the reason it falls back to
    /// reading process RSS, and RSS cannot say which part of the footprint is the shuffle.
    /// Lock-free, so it is safe to poll.
    #[getter]
    fn retained_bytes(&self) -> u64 {
        self.exchange.retained_bytes() as u64
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
        // Also drop this server's shared-memory buckets, which live in tmpfs (RAM) and are
        // *not* part of the in-memory store cleared above.
        //
        // The startup reaper deliberately spares directories owned by our own pid — it
        // cannot tell a dead session's from a live one's, and reaping a live peer's buckets
        // would be far worse than leaking. So a long-lived worker that creates and drops
        // many sessions (the session-fleet shape) accumulated its *own* dead sessions'
        // shm until the process exited. Clearing here is the same-process half of that:
        // the reaper covers processes that died without reaching any teardown, and this
        // covers sessions that ended inside a process still running.
        bc_transport::clear_shared(&self.addr);
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
pub(crate) fn flight_fetch(
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
/// carries.
///
/// `shuffle_store_cap_bytes` bounds the *in-memory* shuffle-output store: above it a
/// worker spills its largest published buckets to local disk and reads them back on
/// fetch. This is the one large footprint Carbonite's buffer pool cannot see — a published
/// bucket is never *reserved*, it is simply held until a reducer fetches it, so with
/// `workers` mappers each producing `workers` buckets a node holds its whole share of the
/// shuffle in memory no reservation covers. Spilling is result-preserving (the same
/// batches return through an Arrow IPC round trip), so it trades a re-read for a memory
/// bound. `0` (the default) is unbounded, which is the historical behaviour.
///
/// Called once per worker process when its Flight server starts. The cap is captured by
/// each store at construction, so it must be set before the server is created.
#[pyfunction]
#[pyo3(signature = (idle_timeout_ms, keepalive_ms=0, connections_per_peer=0, compression=None, shuffle_store_cap_bytes=0))]
pub(crate) fn set_flight_transport_config(
    idle_timeout_ms: u64,
    keepalive_ms: u64,
    connections_per_peer: u64,
    compression: Option<u64>,
    shuffle_store_cap_bytes: u64,
) {
    bc_transport::set_transport_timeouts(idle_timeout_ms, keepalive_ms);
    bc_transport::set_connections_per_peer(connections_per_peer);
    if let Some(code) = compression {
        bc_transport::set_compression(code);
    }
    bc_transport::set_shuffle_store_cap(shuffle_store_cap_bytes);
}

/// Install (or clear) the process-wide client TLS for outbound shuffle fetches.
///
/// `ca_pem` is the CA a peer's server certificate must chain to and `server_name` the
/// name verified against it; `client_cert_pem`/`client_key_pem` (when both given) present
/// this node's certificate under mTLS. Passing an empty `ca_pem` clears it (plaintext).
/// Called once per worker from the control plane, alongside `set_flight_transport_config`.
#[pyfunction]
#[pyo3(signature = (ca_pem, server_name, client_cert_pem=None, client_key_pem=None))]
pub(crate) fn set_flight_client_tls(
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
pub(crate) fn shm_available() -> bool {
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
pub(crate) struct ShuffleClient {
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
