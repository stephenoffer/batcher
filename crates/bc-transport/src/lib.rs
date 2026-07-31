//! Arrow Flight inter-node transport for Batcher's distributed shuffle.
//!
//! # Why this exists
//!
//! Batcher's shuffle currently materializes each partition as an Arrow-IPC file
//! on local disk and re-reads it during the reduce phase. That works on a single
//! machine but cannot move data between processes/hosts. This crate is the
//! foundation for a *true* multi-node shuffle: it moves Arrow [`RecordBatch`]es
//! directly between processes over [Arrow Flight] (gRPC), bypassing any object
//! store or shared filesystem.
//!
//! [Arrow Flight]: https://arrow.apache.org/docs/format/Flight.html
//!
//! # How the distributed layer uses it
//!
//! * **One Flight server per node.** Every worker process starts a single
//!   [`FlightServer`] bound to an ephemeral port and advertises that address to
//!   the scheduler. The server hosts the node's local shuffle output.
//!
//! * **Ticket = the shuffle coordinate.** A map task, after partitioning its
//!   output, registers each output partition under a string ticket that encodes
//!   the full shuffle coordinate, conceptually `(plan, stage, src_part, dst_part)`
//!   (e.g. `"p7/s3/12/45"`). The ticket is the routing key; the bytes are an
//!   opaque Flight ticket on the wire.
//!
//! * **Reducers `DoGet` from each upstream node.** A reduce task for partition
//!   `dst_part` knows, from the scheduler, which nodes produced map output for it.
//!   It opens a [`FlightClient`] to each upstream node and `fetch`es the ticket
//!   for its `dst_part`, streaming the upstream's `RecordBatch`es over gRPC. The
//!   reducer concatenates the streams from all upstreams and runs the reduce.
//!   This replaces the disk write + object-store round-trip entirely.
//!
//! The schema is preserved across the wire so the reducer reconstructs batches
//! that are byte-for-byte equivalent (same schema, same column values) to what
//! the mapper registered.
//!
//! # Credit-based flow control (Carbonite credit model)
//!
//! A naive `DoGet` lets a fast producer encode and push its entire partition
//! onto the wire regardless of how fast the reducer drains it; gRPC's transport
//! window provides *some* back-pressure, but it is byte-oriented and opaque, so
//! the engine cannot reason about how many `RecordBatch`es are buffered ahead of
//! the consumer. Batcher's architecture instead uses an explicit, batch-grained
//! **credit** scheme: **1 credit = 1 `RecordBatch` slot**. The consumer grants
//! credits; the producer may only have as many batches in flight as it holds
//! credits, and **blocks at 0 credits** until the consumer grants more.
//!
//! This crate implements that with Flight's bidirectional [`DoExchange`] stream
//! (see [`ShuffleExchange::fetch`] / the server's [`FlightService::do_exchange`]):
//!
//! * **Consumer → producer (control):** the consumer opens the exchange and
//!   sends *credit-grant* messages — bare [`FlightData`] frames carrying no Arrow
//!   payload, with the ticket in `flight_descriptor.path[0]` (first message only)
//!   and the granted credit count as a little-endian `u32` in `app_metadata`. It
//!   seeds an initial window of `credits` and tops up by one each time it
//!   consumes a batch, keeping roughly `credits` batches in flight.
//! * **Producer → consumer (data):** the producer streams Arrow-encoded batches
//!   but acquires one credit (a [`tokio::sync::Semaphore`] permit fed by incoming
//!   grants) *before encoding/sending each batch*. With zero credits it parks on
//!   the semaphore. **The producer therefore never buffers more than `credits`
//!   batches ahead of the consumer** — the key property.
//!
//! [`DoExchange`]: https://arrow.apache.org/docs/format/Flight.html
//!
//! # How the distributed layer uses it
//!
//! * **One [`ShuffleExchange`] per node.** Each worker process owns a single
//!   exchange (wrapping one [`FlightServer`]) bound to an ephemeral port and
//!   advertises its address to the scheduler.
//! * **Mappers `publish`.** After partitioning, a map task calls
//!   [`ShuffleExchange::publish`] once per output partition under a
//!   [`ShuffleTicket`] encoding `(plan, stage, src_part, dst_part, epoch)`.
//! * **Reducers `fetch` from every node.** A reduce task for `dst_part` calls
//!   [`ShuffleExchange::fetch`] against each upstream node's `(addr, ticket)`,
//!   streaming that node's batches with a bounded credit window, and concatenates
//!   the per-node streams. This replaces the disk write + object-store round-trip.
//!
//! # Public API
//!
//! * [`ShuffleExchange`] — node-level handle: `publish` partitions, `fetch`
//!   remote partitions with credit-bounded streaming.
//! * [`ShuffleTicket`] — structured shuffle coordinate with `to_string`/
//!   `from_string`.
//! * [`FlightServer`] — host named partitions and serve them via `DoGet`.
//! * [`FlightClient`] — connect to a peer and `fetch` a ticket's batches.
//! * [`TransportError`] — error type returned by the client/server helpers.

use std::net::SocketAddr;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow_flight::flight_service_server::FlightServiceServer;
use arrow_flight::{FlightData, Ticket};
use futures::stream::{StreamExt, TryStreamExt};
use tonic::transport::{Channel, Server};

use crate::handler::FlightHandler;
use crate::store::PartitionStore;

mod exchange;
mod handler;
mod peers;
mod shared;
mod store;
mod ticket;
mod tls;
#[cfg(test)]
mod tls_test_certs;

pub use exchange::{classify, ClientPool, FetchFault, ShuffleExchange};
pub use peers::{
    peer_transfers, record_fetch, record_retry, reset_peer_transfers, slowest_peer, PeerTransfer,
};
pub use shared::{clear_plan_shared, clear_shared, fetch_shared, publish_shared, shm_available};
pub use ticket::ShuffleTicket;
pub use tls::{TlsClientConfig, TlsIdentity, TlsServerConfig};

/// Default number of in-flight `RecordBatch` credits for a credit-bounded
/// exchange when the caller does not specify one.
///
/// **Kept equal to `FlowControlConfig.default_credits` on the control-plane side.** This
/// is the value in force whenever Carbonite does not hand one down: a `ShuffleSession`
/// built without an explicit grant passes `credits=None`, and a producer that receives a
/// missing or malformed seed falls back here too. It was 4 while the control plane's
/// authority said 16, so exactly the paths where Carbonite had *not* spoken ran at the
/// throttled window — measured on a 50 ms-RTT link, one 18 MiB partition moves at
/// 2.4 MiB/s at 4 credits and 7.7 MiB/s at 16 (3.2x), because a cross-node fetch's
/// throughput ceiling is `window x batch / RTT` and 4 batches do not fill the
/// bandwidth-delay product.
///
/// This is a *starting* window, not a ceiling: the AIMD controller and the byte-budgeted
/// `credit_ceiling` still govern what a channel may grow to, so raising the floor changes
/// how fast a short fetch reaches its operating point, not how much memory it may hold.
/// At the default 1 MiB morsel it is ~16 MiB per channel, well inside the 256 MiB
/// per-channel budget.
pub const DEFAULT_CREDITS: u32 = 16;

/// HTTP/2 per-stream receive window for a bulk Arrow transfer.
///
/// tonic/hyper default this to **64 KiB**, which is the classic gRPC bulk-throughput
/// cliff: a single stream can only have `window` bytes outstanding per round-trip, so
/// a 1 MiB shuffle morsel needs ~16 flow-control round-trips just to cross the wire —
/// on a cross-node link that collapses throughput to `window / RTT`. Sizing the stream
/// window well above the morsel (and the credit window's bytes) lets a whole batch — and
/// several batches ahead of it — stream without stalling on window updates. The app-level
/// **credit** governor (not this window) is what bounds resident batches, so enlarging the
/// transport window changes throughput, not the memory bound.
pub const H2_STREAM_WINDOW: u32 = 16 * 1024 * 1024;

/// HTTP/2 whole-connection receive window.
///
/// A pooled [`FlightClient`] multiplexes every concurrent fetch to a peer over a single
/// HTTP/2 connection, so the *connection* window is shared across all in-flight streams;
/// at the 64 KiB default it throttles the entire fan-out regardless of per-stream tuning.
/// Sized to hold several full-window streams at once (`>= fan_in x` [`H2_STREAM_WINDOW`]
/// in the common case) so the aggregate reduce-side gather stays wire-bound, not
/// window-bound.
pub const H2_CONNECTION_WINDOW: u32 = 128 * 1024 * 1024;

/// Errors surfaced by the transport's client/server helpers.
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    /// The gRPC transport failed to connect or bind.
    #[error("transport error: {0}")]
    Transport(#[from] tonic::transport::Error),
    /// A Flight RPC returned a non-OK status (e.g. unknown ticket -> NotFound).
    /// Boxed: `tonic::Status` is large, so keeping it inline would bloat every
    /// `Result<_, TransportError>` (clippy `result_large_err`).
    #[error("flight status: {0}")]
    Status(Box<tonic::Status>),
    /// An Arrow error occurred while encoding/decoding batches.
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    /// A Flight-level error (encode/decode/protocol) from arrow-flight.
    #[error("flight error: {0}")]
    Flight(#[from] arrow_flight::error::FlightError),
    /// The background server task could not be joined.
    #[error("join error: {0}")]
    Join(String),
    /// A fetch saw no batch from the peer within the idle window — a hung/dead
    /// peer. Distinct from `Io` so it classifies as *retryable* (the partition can
    /// be recomputed and re-fetched) rather than a fatal protocol error.
    #[error("fetch idle timeout after {0:?} waiting on peer")]
    IdleTimeout(std::time::Duration),
    /// Address parsing / IO error.
    #[error("io error: {0}")]
    Io(String),
}

/// Process-wide transport tunables, settable once per worker process from the
/// control plane (Carbonite). Globals (not threaded through every fetch signature)
/// because they are uniform for a process's lifetime and the fetch path is deep;
/// the default reproduces the historical hardcoded behavior.
mod tunables {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    /// Idle gap between batches before a fetch fails (ms). Default 60_000 = the old
    /// hardcoded 60 s.
    static FETCH_IDLE_TIMEOUT_MS: AtomicU64 = AtomicU64::new(60_000);
    /// HTTP/2 keepalive ping interval (ms); `0` = off (tonic default). Detects a
    /// silently-dropped peer connection faster than the idle timeout alone.
    static KEEPALIVE_MS: AtomicU64 = AtomicU64::new(0);
    /// Max TCP connections a pooled consumer opens to one peer. A single HTTP/2
    /// connection is one TCP flow, and a cloud NIC caps a *single* flow well below
    /// line rate (e.g. AWS ~5 Gbps of a 10 Gbps NIC), so funneling every fetch to a
    /// peer through one connection leaves half the link idle. Striping concurrent
    /// fetches across a few connections lifts aggregate throughput to line rate. The
    /// pool grows lazily to this bound only when a peer actually sees concurrent
    /// fetches, so a cold/rarely-hit peer still costs one connection. Default 4 (two
    /// saturate a 10 Gbps NIC; four gives headroom for 25 Gbps instances).
    static CONNECTIONS_PER_PEER: AtomicU64 = AtomicU64::new(4);

    /// Set the process-wide transport timeouts. `idle_ms == 0` keeps the current
    /// idle timeout; `keepalive_ms == 0` disables keepalive.
    pub fn set_transport_timeouts(idle_ms: u64, keepalive_ms: u64) {
        if idle_ms > 0 {
            FETCH_IDLE_TIMEOUT_MS.store(idle_ms, Ordering::Relaxed);
        }
        KEEPALIVE_MS.store(keepalive_ms, Ordering::Relaxed);
    }

    /// Set the per-peer connection bound (see [`CONNECTIONS_PER_PEER`]). `0` keeps the
    /// current value; clamped to at least 1 on read. Settable once per worker process
    /// from the control plane (Carbonite), like the timeouts.
    pub fn set_connections_per_peer(n: u64) {
        if n > 0 {
            CONNECTIONS_PER_PEER.store(n, Ordering::Relaxed);
        }
    }

    /// The current per-peer connection bound (always >= 1).
    pub fn connections_per_peer() -> usize {
        CONNECTIONS_PER_PEER.load(Ordering::Relaxed).max(1) as usize
    }

    /// Wire compression codec for shuffle batches, as a code: 0 = none, 1 = LZ4-frame,
    /// 2 = Zstd. The producer compresses each batch's Arrow buffers before they cross
    /// the wire; the consumer auto-decompresses (the codec is carried in the IPC message
    /// metadata). A cross-node fetch is NIC-bound at ~line rate, so moving *fewer* bytes
    /// is the only way past that ceiling — and real shuffle data (sorted runs, repeated
    /// group keys, dictionary strings, nulls) compresses several-fold, unlike the object
    /// store's uncompressed blocks. Default LZ4-frame: ~GB/s/core, so it keeps up with the
    /// link and gives up fast on incompressible data (near-free there).
    static COMPRESSION: AtomicU64 = AtomicU64::new(1);

    /// Byte cap on the in-memory shuffle-output store, `0` = unbounded (the default, and
    /// the historical behaviour). Above it the store spills its largest published buckets
    /// to local disk and reads them back on fetch — result-preserving, so this trades a
    /// re-read for a memory bound. Set per worker from Carbonite, which knows the envelope.
    static SHUFFLE_STORE_CAP: AtomicU64 = AtomicU64::new(0);

    /// Set the in-memory shuffle-store cap in bytes (`0` disables spilling).
    pub fn set_shuffle_store_cap(bytes: u64) {
        SHUFFLE_STORE_CAP.store(bytes, Ordering::Relaxed);
    }

    /// The current in-memory shuffle-store cap in bytes (`0` = unbounded).
    pub fn shuffle_store_cap() -> usize {
        SHUFFLE_STORE_CAP.load(Ordering::Relaxed) as usize
    }

    /// Set the shuffle wire-compression codec (0 none / 1 lz4 / 2 zstd). Values outside
    /// that range are ignored (keep current). Settable per worker from Carbonite.
    pub fn set_compression(code: u64) {
        if code <= 2 {
            COMPRESSION.store(code, Ordering::Relaxed);
        }
    }

    /// The current compression codec for the Flight encoder, or `None` (no compression).
    pub fn compression() -> Option<arrow::ipc::CompressionType> {
        match COMPRESSION.load(Ordering::Relaxed) {
            1 => Some(arrow::ipc::CompressionType::LZ4_FRAME),
            2 => Some(arrow::ipc::CompressionType::ZSTD),
            _ => None,
        }
    }

    /// The current fetch idle timeout.
    pub fn fetch_idle_timeout() -> Duration {
        Duration::from_millis(FETCH_IDLE_TIMEOUT_MS.load(Ordering::Relaxed))
    }

    /// The current keepalive interval, or `None` when disabled.
    pub fn keepalive() -> Option<Duration> {
        match KEEPALIVE_MS.load(Ordering::Relaxed) {
            0 => None,
            ms => Some(Duration::from_millis(ms)),
        }
    }

    /// Process-wide client TLS for outbound fetches. A reducer dials every peer in the
    /// same cluster with the same trust settings, so — like the other transport
    /// tunables — this is set once per worker rather than threaded through every
    /// `fetch`. `None` (the default) keeps outbound connections plaintext.
    static CLIENT_TLS: std::sync::RwLock<Option<std::sync::Arc<crate::TlsClientConfig>>> =
        std::sync::RwLock::new(None);

    /// Install (or clear, with `None`) the process-wide client TLS config.
    pub fn set_client_tls(cfg: Option<crate::TlsClientConfig>) {
        *CLIENT_TLS.write().unwrap_or_else(|e| e.into_inner()) = cfg.map(std::sync::Arc::new);
    }

    /// The active client TLS config, if any.
    pub fn client_tls() -> Option<std::sync::Arc<crate::TlsClientConfig>> {
        CLIENT_TLS.read().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

pub use tunables::{
    client_tls, compression, connections_per_peer, fetch_idle_timeout, keepalive, set_client_tls,
    set_compression, set_connections_per_peer, set_shuffle_store_cap, set_transport_timeouts,
    shuffle_store_cap,
};

impl From<tonic::Status> for TransportError {
    fn from(status: tonic::Status) -> Self {
        TransportError::Status(Box::new(status))
    }
}

pub(crate) type TransportResult<T> = Result<T, TransportError>;

/// A running Flight server hosting named partitions for one node.
///
/// Register partitions with [`FlightServer::register`] (before or after start —
/// the store is shared), then start the gRPC server with
/// [`FlightServer::serve`] / [`FlightServer::serve_ephemeral`].
pub struct FlightServer {
    store: Arc<PartitionStore>,
    token: Option<String>,
    tls: Option<TlsServerConfig>,
}

impl Default for FlightServer {
    fn default() -> Self {
        Self::new()
    }
}

impl FlightServer {
    /// Create a new, empty server.
    pub fn new() -> Self {
        Self {
            store: Arc::new(PartitionStore::default()),
            token: None,
            tls: None,
        }
    }

    /// Build a server over a shared store with both the shuffle `token` and TLS. `tls`
    /// encrypts the connection (and, when it carries a client CA, mutually authenticates
    /// peers); `None` keeps the server plaintext.
    pub(crate) fn with_store_token_tls(
        store: Arc<PartitionStore>,
        token: Option<String>,
        tls: Option<TlsServerConfig>,
    ) -> Self {
        Self { store, token, tls }
    }

    /// Apply this server's TLS config to a tonic builder, or leave it plaintext.
    fn tls_builder(&self) -> TransportResult<Server> {
        let builder = tuned_server();
        match &self.tls {
            None => Ok(builder),
            Some(cfg) => {
                tls::check_pem("server certificate", &cfg.identity.cert_pem)?;
                tls::check_pem("server private key", &cfg.identity.key_pem)?;
                if let Some(ca) = &cfg.client_ca_pem {
                    tls::check_pem("client CA", ca)?;
                }
                builder
                    .tls_config(cfg.to_tonic())
                    .map_err(|e| tls::tls_error("server tls config", e))
            }
        }
    }

    /// Register a named partition. The `ticket` is the routing key reducers use
    /// in [`FlightClient::fetch`]; `batches` are served verbatim over `DoGet`.
    pub async fn register(&self, ticket: impl Into<String>, batches: Vec<RecordBatch>) {
        self.store.register(ticket.into(), batches).await;
    }

    /// Build the tonic [`Server`] future bound to `addr`.
    ///
    /// Returns the future driving the server; await it (typically in a spawned
    /// task) to run until the process exits. Prefer [`Self::serve_ephemeral`]
    /// when you need to learn the bound port.
    pub async fn serve(self, addr: SocketAddr) -> TransportResult<()> {
        let mut builder = self.tls_builder()?;
        let svc = FlightServiceServer::new(FlightHandler {
            store: self.store,
            token: self.token,
        });
        builder
            .add_service(svc)
            .serve(addr)
            .await
            .map_err(TransportError::Transport)
    }

    /// Bind to `127.0.0.1:0` (or the given host with port 0), learn the OS-chosen
    /// port, and start serving in the background.
    ///
    /// Returns the bound [`SocketAddr`] together with a [`ServerHandle`] keeping
    /// the background task alive. Dropping the handle aborts the server.
    pub async fn serve_ephemeral(self) -> TransportResult<(SocketAddr, ServerHandle)> {
        self.serve_on("127.0.0.1:0").await
    }

    /// Like [`Self::serve_ephemeral`] but lets the caller pick the bind string
    /// (host + optional `:0` for an ephemeral port).
    pub async fn serve_on(self, bind: &str) -> TransportResult<(SocketAddr, ServerHandle)> {
        // Bind a std listener first so we can read back the OS-assigned port
        // before tonic takes ownership of the socket.
        let std_listener = std::net::TcpListener::bind(bind)
            .map_err(|e| TransportError::Io(format!("bind {bind}: {e}")))?;
        std_listener
            .set_nonblocking(true)
            .map_err(|e| TransportError::Io(format!("set_nonblocking: {e}")))?;
        let local_addr = std_listener
            .local_addr()
            .map_err(|e| TransportError::Io(format!("local_addr: {e}")))?;

        let listener = tokio::net::TcpListener::from_std(std_listener)
            .map_err(|e| TransportError::Io(format!("from_std: {e}")))?;
        // `TCP_NODELAY` on every accepted connection. [`tuned_server`] asks tonic for it,
        // but that setting only reaches sockets tonic accepts through its *own* listener;
        // with a caller-supplied `incoming` (which is how the port is learned before tonic
        // takes the socket) the accepted streams keep the kernel default, Nagle on.
        //
        // Nagle on the producer's side against the consumer's delayed-ACK timer is a
        // textbook 40 ms stall, and it lands on precisely the fetches that cannot hide it:
        // a *serial* one — a next-stage worker reading one intermediate bucket, a GPU
        // consumer pulling one morsel — pays the full stall per fetch, measured here at
        // 41.75 ms to move a single one-row batch over loopback. A concurrent gather
        // pipelines it away, which is why the wide shuffle never showed it and the
        // one-at-a-time paths quietly did.
        let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener).map(|conn| {
            if let Ok(stream) = &conn {
                // Best-effort: a socket that refuses the option still works, just slower.
                let _ = stream.set_nodelay(true);
            }
            conn
        });

        let mut builder = self.tls_builder()?;
        let svc = FlightServiceServer::new(FlightHandler {
            store: self.store,
            token: self.token,
        });
        let handle =
            tokio::spawn(
                async move { builder.add_service(svc).serve_with_incoming(incoming).await },
            );

        Ok((local_addr, ServerHandle { task: handle }))
    }
}

/// A tonic [`Server`] builder with the bulk-transfer HTTP/2 windows applied.
///
/// The producer streams shuffle data client-bound, so the *consumer's* receive window
/// (set on the channel in [`FlightClient::build_channel`]) governs the data direction;
/// these server-side windows govern the consumer→producer control stream and any
/// server-received data, and keep both peers off the 64 KiB default so no direction is
/// silently window-throttled. `tcp_nodelay` avoids Nagle-delaying credit acks.
fn tuned_server() -> Server {
    Server::builder()
        .initial_stream_window_size(Some(H2_STREAM_WINDOW))
        .initial_connection_window_size(Some(H2_CONNECTION_WINDOW))
        .tcp_nodelay(true)
}

/// Keeps a background Flight server alive; dropping it aborts the server task.
pub struct ServerHandle {
    task: tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
}

impl ServerHandle {
    /// Abort the background server task immediately.
    pub fn abort(&self) {
        self.task.abort();
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// A client to one peer node's [`FlightServer`].
pub struct FlightClient {
    inner: arrow_flight::FlightClient,
}

impl FlightClient {
    /// Connect to a peer Flight server at `addr` (e.g. `"http://127.0.0.1:50051"`
    /// or a bare `"127.0.0.1:50051"`).
    pub async fn connect(addr: impl AsRef<str>) -> TransportResult<Self> {
        Ok(Self::from_channel(
            Self::build_channel(addr.as_ref()).await?,
        ))
    }

    /// Connect with an explicit TLS config rather than the process-wide default — the
    /// peer's certificate is verified against `tls.ca_pem`, and (under mTLS) this node's
    /// `tls.identity` is presented. The connection dials `https`.
    pub async fn connect_tls(
        addr: impl AsRef<str>,
        tls: &crate::TlsClientConfig,
    ) -> TransportResult<Self> {
        Ok(Self::from_channel(
            Self::build_channel_tls(addr.as_ref(), Some(tls)).await?,
        ))
    }

    /// Establish a tonic [`Channel`] to `addr`, accepting a bare `host:port` or a
    /// full URI. Exposed so a [`ClientPool`] can cache and reuse the channel across
    /// fetches instead of reconnecting per partition.
    ///
    /// [`ClientPool`]: crate::exchange::ClientPool
    pub async fn build_channel(addr: &str) -> TransportResult<Channel> {
        // The production reducer path reads the process-wide client TLS (set once per
        // worker); `build_channel_tls` is the shared implementation an explicit
        // `connect_tls` also uses.
        Self::build_channel_tls(addr, crate::client_tls().as_deref()).await
    }

    /// Build a channel to `addr`, applying `client_tls` when given (else plaintext).
    pub(crate) async fn build_channel_tls(
        addr: &str,
        client_tls: Option<&crate::TlsClientConfig>,
    ) -> TransportResult<Channel> {
        // With TLS the channel must dial `https` and carry the tonic TLS config;
        // otherwise it stays plaintext `http` as before.
        let uri = match client_tls {
            Some(_) => tls::https_uri(addr),
            None if addr.contains("://") => addr.to_string(),
            None => format!("http://{addr}"),
        };
        let mut endpoint = Channel::from_shared(uri.into_bytes())
            .map_err(|e| TransportError::Io(format!("invalid uri: {e}")))?
            // Bulk Arrow transfer: enlarge the HTTP/2 receive windows off their 64 KiB
            // default so a 1 MiB morsel streams without ~16 flow-control round-trips, and a
            // pooled channel's multiplexed fan-out isn't throttled by the connection window
            // (the primary cross-node throughput fix). `tcp_nodelay` keeps the tiny credit
            // grants from being Nagle-delayed. Resident batches stay bounded by credits.
            .initial_stream_window_size(Some(crate::H2_STREAM_WINDOW))
            .initial_connection_window_size(Some(crate::H2_CONNECTION_WINDOW))
            .tcp_nodelay(true);
        if let Some(cfg) = client_tls {
            tls::check_pem("peer CA", &cfg.ca_pem)?;
            endpoint = endpoint
                .tls_config(cfg.to_tonic())
                .map_err(|e| tls::tls_error("client tls config", e))?;
        }
        // Keepalive pings detect a silently-dropped peer connection (a crashed node
        // whose TCP never RSTs) faster than the between-batch idle timeout, so the
        // fetch surfaces a retryable fault promptly instead of hanging a full window.
        if let Some(interval) = crate::keepalive() {
            endpoint = endpoint
                .keep_alive_while_idle(true)
                .http2_keep_alive_interval(interval);
        }
        Ok(endpoint.connect().await?)
    }

    /// Connect and present `token` (if any) on every `DoGet`, as an `authorization:
    /// Bearer <token>` header — the credential a token-secured [`FlightServer`] requires.
    pub async fn connect_with_token(
        addr: impl AsRef<str>,
        token: Option<&str>,
    ) -> TransportResult<Self> {
        let mut client = Self::connect(addr).await?;
        if let Some(token) = token.filter(|t| !t.is_empty()) {
            let value = format!("{}{token}", crate::handler::AUTH_SCHEME)
                .parse()
                .map_err(|e| TransportError::Io(format!("invalid shuffle token: {e}")))?;
            client
                .inner
                .metadata_mut()
                .insert(crate::handler::AUTH_HEADER, value);
        }
        Ok(client)
    }

    /// Wrap an already-established [`Channel`] (cheap; channels are clonable and
    /// multiplex over HTTP/2, so one channel backs many `FlightClient`s).
    pub fn from_channel(channel: Channel) -> Self {
        Self {
            inner: arrow_flight::FlightClient::new(channel),
        }
    }

    /// `DoGet` the named ticket and collect all returned batches.
    ///
    /// The schema is reconstructed from the stream, so returned batches match
    /// what the server registered. An unknown ticket surfaces as a
    /// [`TransportError::Status`] with `NotFound`.
    pub async fn fetch(&mut self, ticket: impl Into<String>) -> TransportResult<Vec<RecordBatch>> {
        let ticket = Ticket {
            ticket: ticket.into().into_bytes().into(),
        };
        // The high-level client hands back a FlightRecordBatchStream that
        // reconstructs the schema and decodes each FlightData into a RecordBatch.
        let mut record_stream = self.inner.do_get(ticket).await?;

        let mut batches = Vec::new();
        while let Some(batch) = record_stream.try_next().await? {
            batches.push(batch);
        }
        Ok(batches)
    }

    /// Open a bidirectional `DoExchange` stream, sending the consumer's request
    /// (credit-grant) stream and returning the producer's decoded
    /// [`RecordBatch`] stream. Used by [`ShuffleExchange`] for credit-bounded
    /// fetches.
    pub(crate) async fn do_exchange<S>(
        &mut self,
        request: S,
    ) -> TransportResult<arrow_flight::decode::FlightRecordBatchStream>
    where
        S: futures::Stream<Item = Result<FlightData, arrow_flight::error::FlightError>>
            + Send
            + 'static,
    {
        Ok(self.inner.do_exchange(request).await?)
    }
}

/// Convenience blocking wrapper: connect + fetch on a fresh single-threaded
/// runtime. Handy from non-async call sites (e.g. the current disk-shuffle
/// reducer) while the engine is being made async end-to-end.
pub fn fetch_blocking(addr: &str, ticket: &str) -> TransportResult<Vec<RecordBatch>> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| TransportError::Io(format!("runtime: {e}")))?;
    rt.block_on(async {
        let mut client = FlightClient::connect(addr).await?;
        client.fetch(ticket).await
    })
}

/// Credit-bounded blocking fetch: connect + credit-gated `DoExchange` on a fresh
/// single-threaded runtime, keeping at most `credits` `RecordBatch`es in flight.
///
/// This is the flow-controlled counterpart to [`fetch_blocking`] (which uses an
/// un-credited `DoGet` and lets a fast producer race ahead). The distributed
/// reducer calls this so a Carbonite-granted window bounds producer memory —
/// `credits` is clamped to at least 1 by [`ShuffleExchange::fetch_with_credits`].
pub fn fetch_blocking_with_credits(
    addr: &str,
    ticket: &str,
    credits: u32,
    token: Option<&str>,
) -> TransportResult<Vec<RecordBatch>> {
    let ticket = ShuffleTicket::from_string(ticket)?;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| TransportError::Io(format!("runtime: {e}")))?;
    rt.block_on(ShuffleExchange::fetch_secured(
        addr, &ticket, credits, token,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};

    fn batch_a() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
            ],
        )
        .unwrap()
    }

    fn batch_a2() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![4, 5])),
                Arc::new(StringArray::from(vec![Some("d"), Some("e")])),
            ],
        )
        .unwrap()
    }

    fn batch_b() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            false,
        )]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![1.5, 2.5, 3.5]))],
        )
        .unwrap()
    }

    async fn start_server() -> (SocketAddr, ServerHandle) {
        let server = FlightServer::new();
        server
            .register("p1/s0/0/0", vec![batch_a(), batch_a2()])
            .await;
        server.register("p1/s0/0/1", vec![batch_b()]).await;
        server.register("empty", vec![]).await;
        server.serve_ephemeral().await.unwrap()
    }

    /// A *serial* fetch must not sit in a Nagle/delayed-ACK stall.
    ///
    /// The server's accepted sockets need `TCP_NODELAY` set explicitly: tonic's builder
    /// option does not reach a caller-supplied `incoming`, and without it the producer's
    /// small writes wait on the consumer's delayed-ACK timer. Measured at 41.75 ms per
    /// one-row fetch before the fix and 1.02 ms after, so the bound below sits an order of
    /// magnitude clear of both — it cannot fire on a slow machine without the stall, and
    /// cannot pass with it. A *concurrent* gather pipelines the stall away, which is why
    /// only the one-at-a-time paths (a cross-stage bucket read, a streaming morsel) ever
    /// paid it, and why no throughput test caught it.
    #[tokio::test(flavor = "multi_thread")]
    async fn serial_fetches_do_not_stall_on_nagle() {
        let server = FlightServer::new();
        // A canonical 5-field ticket, so the credit-gated `fetch_secured` path is the one
        // measured (the plain-string keys `start_server` registers are `DoGet` only).
        server.register("7/0/0/0/0", vec![batch_b()]).await;
        let (addr, _handle) = server.serve_ephemeral().await.unwrap();
        let pool = ClientPool::new();
        let addr = addr.to_string();
        let ticket = ShuffleTicket::from_string("7/0/0/0/0").unwrap();

        pool.fetch_secured(&addr, &ticket, 4, None).await.unwrap(); // warm the channel

        const N: u32 = 10;
        let started = std::time::Instant::now();
        for _ in 0..N {
            let got = pool.fetch_secured(&addr, &ticket, 4, None).await.unwrap();
            assert_eq!(got.len(), 1, "the fetch must still return the partition");
        }
        let per_fetch_ms = started.elapsed().as_secs_f64() * 1000.0 / f64::from(N);
        assert!(
            per_fetch_ms < 15.0,
            "serial fetch took {per_fetch_ms:.1} ms; a ~40 ms figure is the delayed-ACK \
             stall returning (TCP_NODELAY lost on the server's accepted sockets)"
        );
    }

    #[tokio::test]
    async fn fetch_roundtrips_multiple_partitions() {
        let (addr, _handle) = start_server().await;

        let mut client = FlightClient::connect(addr.to_string()).await.unwrap();

        // Partition 1: two batches, multiple columns incl. nullable Utf8.
        let got = client.fetch("p1/s0/0/0").await.unwrap();
        let expected = [batch_a(), batch_a2()];
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_eq!(g.schema(), e.schema(), "schema preserved");
            assert_eq!(g, e, "values preserved");
        }

        // Partition 2: a different schema entirely.
        let got = client.fetch("p1/s0/0/1").await.unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].schema(), batch_b().schema());
        assert_eq!(got[0], batch_b());
    }

    #[tokio::test]
    async fn unknown_ticket_errors() {
        let (addr, _handle) = start_server().await;
        let mut client = FlightClient::connect(addr.to_string()).await.unwrap();

        let err = client.fetch("does/not/exist").await.unwrap_err();
        // The high-level client surfaces a NotFound as a Flight(Tonic(..)) error.
        let status = match &err {
            TransportError::Status(s) => (**s).clone(),
            TransportError::Flight(arrow_flight::error::FlightError::Tonic(s)) => (**s).clone(),
            other => panic!("expected NotFound status, got: {other:?}"),
        };
        assert_eq!(status.code(), tonic::Code::NotFound, "got: {status:?}");
    }

    #[tokio::test]
    async fn empty_partition_returns_no_batches() {
        let (addr, _handle) = start_server().await;
        let mut client = FlightClient::connect(addr.to_string()).await.unwrap();
        let got = client.fetch("empty").await.unwrap();
        assert!(got.is_empty(), "empty partition yields zero batches");
    }

    // --- ShuffleTicket --------------------------------------------------------

    #[test]
    fn shuffle_ticket_roundtrips() {
        let t = ShuffleTicket::new(7, 3, 12, 45, 2);
        let s = t.to_string();
        assert_eq!(s, "7/3/12/45/2");
        assert_eq!(ShuffleTicket::from_string(&s).unwrap(), t);
        // Display matches to_string.
        assert_eq!(format!("{t}"), s);
        // A few representative values incl. zero epoch and large plan id.
        for t in [
            ShuffleTicket::new(0, 0, 0, 0, 0),
            ShuffleTicket::new(u64::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX),
            ShuffleTicket::new(1, 2, 3, 4, 0),
        ] {
            assert_eq!(ShuffleTicket::from_string(&t.to_string()).unwrap(), t);
        }
    }

    #[test]
    fn shuffle_ticket_rejects_malformed() {
        assert!(
            ShuffleTicket::from_string("1/2/3/4").is_err(),
            "too few fields"
        );
        assert!(
            ShuffleTicket::from_string("1/2/3/4/5/6").is_err(),
            "too many fields"
        );
        assert!(
            ShuffleTicket::from_string("a/2/3/4/5").is_err(),
            "non-numeric"
        );
        assert!(
            ShuffleTicket::from_string("-1/2/3/4/5").is_err(),
            "negative"
        );
    }

    // --- ShuffleExchange round-trip ------------------------------------------

    /// Build `n` single-row Int64 batches with values `start..start+n`, so the
    /// receiver can assert both count and order.
    fn seq_batches(start: i64, n: i64) -> Vec<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        (start..start + n)
            .map(|v| {
                RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(vec![v]))])
                    .unwrap()
            })
            .collect()
    }

    #[tokio::test]
    async fn shuffle_exchange_roundtrips_multiple_partitions() {
        // One node hosting two destination partitions of one map task.
        let producer = ShuffleExchange::bind_ephemeral().await.unwrap();
        let addr = producer.addr().to_string();

        let t0 = ShuffleTicket::new(1, 0, 0, 0, 0);
        let t1 = ShuffleTicket::new(1, 0, 0, 1, 0);
        producer.publish(&t0, vec![batch_a(), batch_a2()]).await;
        producer.publish(&t1, vec![batch_b()]).await;

        // A second node acting purely as a reducer fetches both.
        let reducer = ShuffleExchange::bind_ephemeral().await.unwrap();

        let got0 = reducer.fetch(&addr, &t0).await.unwrap();
        assert_eq!(got0, vec![batch_a(), batch_a2()]);

        let got1 = reducer.fetch(&addr, &t1).await.unwrap();
        assert_eq!(got1, vec![batch_b()]);
    }

    #[tokio::test]
    async fn shuffle_exchange_unknown_ticket_is_empty_not_error() {
        // An unpublished ticket is the *expected* empty-bucket case in a shuffle (a
        // mapper that produced no rows for a reducer never publishes it), so the
        // transport resolves NotFound to an empty partition rather than an error.
        // A real fault (unreachable peer) still propagates — see the Python-level
        // test_gather_unreachable_peer_raises_not_silent_empty.
        let producer = ShuffleExchange::bind_ephemeral().await.unwrap();
        let addr = producer.addr().to_string();
        let missing = ShuffleTicket::new(9, 9, 9, 9, 9);
        let got = ShuffleExchange::fetch_with_credits(&addr, &missing, 2)
            .await
            .unwrap();
        assert!(got.is_empty());
    }

    // --- Credit-based flow control -------------------------------------------

    #[tokio::test]
    async fn credit_window_bounds_producer_and_preserves_order() {
        // A producer with MANY batches and a SMALL credit window. The consumer
        // must receive all batches in order, and the producer must never have
        // had more than `WINDOW` batches in flight (verified via the gauge).
        const N: i64 = 50;
        const WINDOW: u32 = 3;

        let producer = ShuffleExchange::bind_ephemeral().await.unwrap();
        let addr = producer.addr().to_string();
        let ticket = ShuffleTicket::new(2, 1, 0, 0, 0);
        producer.publish(&ticket, seq_batches(0, N)).await;

        let got = ShuffleExchange::fetch_with_credits(&addr, &ticket, WINDOW)
            .await
            .unwrap();

        // Correctness: all batches, in order.
        assert_eq!(got.len() as i64, N, "received every batch");
        for (i, b) in got.iter().enumerate() {
            let col = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            assert_eq!(col.value(0), i as i64, "batch {i} out of order");
        }

        // Flow control: the producer never ran more than WINDOW batches ahead.
        let max_inflight = producer.max_inflight(&ticket).await.unwrap();
        assert!(
            max_inflight >= 1 && max_inflight <= WINDOW as i64,
            "in-flight high-water mark {max_inflight} must be within (0, {WINDOW}]",
        );
    }

    /// A consumer that over-grants must not be able to make the producer buffer past the
    /// window it seeded.
    ///
    /// The credit bound used to be the *consumer's* arithmetic: the producer added
    /// whatever permits arrived. A reducer with a buggy top-up — or one that simply
    /// claimed more — made a healthy mapper encode and hold its whole partition, and a
    /// large enough claim panicked the serve task outright on `add_permits`. This drives
    /// the exchange by hand so it can grant dishonestly, then checks the producer's own
    /// in-flight gauge.
    #[tokio::test]
    async fn an_over_granting_consumer_cannot_widen_the_producers_window() {
        const N: i64 = 60;
        const WINDOW: u32 = 4;

        let producer = ShuffleExchange::bind_ephemeral().await.unwrap();
        let addr = producer.addr().to_string();
        let ticket = ShuffleTicket::new(9, 0, 0, 0, 0);
        producer.publish(&ticket, seq_batches(0, N)).await;

        let mut client = FlightClient::connect(&addr).await.unwrap();
        let (grant_tx, grant_rx) = tokio::sync::mpsc::channel::<FlightData>(64);
        let ticket_str = ticket.to_string();
        let first = FlightData {
            flight_descriptor: Some(arrow_flight::FlightDescriptor {
                r#type: arrow_flight::flight_descriptor::DescriptorType::Path as i32,
                path: vec![ticket_str, String::new()],
                ..Default::default()
            }),
            app_metadata: handler::encode_credits(WINDOW).into(),
            ..Default::default()
        };
        grant_tx.send(first).await.unwrap();

        let request = futures::StreamExt::map(
            tokio_stream::wrappers::ReceiverStream::new(grant_rx),
            Ok::<_, arrow_flight::error::FlightError>,
        );
        let mut response = client.do_exchange(request).await.unwrap();

        // Drain, granting back a wildly inflated top-up after every batch.
        let mut seen = 0i64;
        while let Ok(Some(_batch)) = response.try_next().await {
            seen += 1;
            let _ = grant_tx
                .send(FlightData {
                    app_metadata: handler::encode_credits(100_000).into(),
                    ..Default::default()
                })
                .await;
        }
        drop(grant_tx);

        assert_eq!(seen, N, "the transfer must still complete");
        let max_inflight = producer.max_inflight(&ticket).await.unwrap();
        // `WINDOW + 1`, and the `+ 1` is real rather than slack. The clamp reads
        // `available_permits()` to decide how much room a grant may fill, while the
        // producer decrements that same count at `acquire()` and only marks the batch
        // in-flight at `on_send()`. A grant landing between those two points sees one more
        // slot free than the gauge will shortly show, so in-flight can transiently reach
        // one past the window. Closing it would need the permit count and the gauge to move
        // under one lock, on the per-batch path, to buy back a single batch slot.
        //
        // What matters is that the bound is *a* bound: without the clamp this same test
        // observes 56 in flight against a seeded window of 4 — the producer encoding its
        // whole partition because the consumer said it could.
        assert!(
            max_inflight <= WINDOW as i64 + 1,
            "a dishonest consumer widened the producer's window to {max_inflight} (seeded {WINDOW})",
        );
    }

    /// A `ClientPool` is process-lifetime and keyed by advertised address, and a worker
    /// advertises an *ephemeral* port — so every peer restart, autoscaling replacement, and
    /// actor recycle mints a new key. Nothing ever removed one, so each dead peer kept its
    /// entry and up to `connections_per_peer` live HTTP/2 connections: an unbounded leak of
    /// memory and file descriptors in the one process that outlives every query.
    #[tokio::test]
    async fn a_peer_that_proves_unreachable_is_dropped_from_the_pool() {
        let pool = ClientPool::new();
        let ticket = ShuffleTicket::new(3, 0, 0, 0, 0);

        // A port nothing is listening on: connect fails, the redial fails, the peer goes.
        let dead = "127.0.0.1:1";
        assert!(pool.fetch_with_credits(dead, &ticket, 4).await.is_err());
        assert_eq!(
            pool.connection_count(),
            0,
            "an unreachable peer kept its pool entry and its connections",
        );
    }

    /// The eviction must be tied to proven unreachability, not to idleness: a live peer
    /// that is merely quiet between queries should keep its warm connections, since
    /// re-establishing them is exactly the cost this pool exists to avoid.
    #[tokio::test]
    async fn a_live_but_idle_peer_keeps_its_pooled_connections() {
        let producer = ShuffleExchange::bind_ephemeral().await.unwrap();
        let addr = producer.addr().to_string();
        let ticket = ShuffleTicket::new(4, 0, 0, 0, 0);
        producer.publish(&ticket, seq_batches(0, 3)).await;

        let pool = ClientPool::new();
        assert_eq!(
            pool.fetch_with_credits(&addr, &ticket, 4)
                .await
                .unwrap()
                .len(),
            3
        );
        assert_eq!(pool.connection_count(), 1);

        // A second fetch after the first finished must reuse the entry, not rebuild it.
        assert_eq!(
            pool.fetch_with_credits(&addr, &ticket, 4)
                .await
                .unwrap()
                .len(),
            3
        );
        assert_eq!(
            pool.connection_count(),
            1,
            "a live peer's pool entry was dropped"
        );
    }

    /// A **slow** consumer, which is the case every other flow-control test misses.
    ///
    /// The existing tests drain as fast as the producer can send, so the semaphore is
    /// almost never empty when the producer reaches it — the blocking path that *is* the
    /// flow control is barely exercised. Pausing between reads forces the producer to park
    /// at zero credits repeatedly, which is the condition the bound exists for: without it
    /// a fast mapper encodes its whole partition into the transport while a busy reducer
    /// works through the first few batches.
    #[tokio::test]
    async fn a_slow_consumer_parks_the_producer_at_its_window() {
        const N: i64 = 40;
        const WINDOW: u32 = 3;

        let producer = ShuffleExchange::bind_ephemeral().await.unwrap();
        let addr = producer.addr().to_string();
        let ticket = ShuffleTicket::new(11, 0, 0, 0, 0);
        producer.publish(&ticket, seq_batches(0, N)).await;

        let mut client = FlightClient::connect(&addr).await.unwrap();
        let (grant_tx, grant_rx) = tokio::sync::mpsc::channel::<FlightData>(64);
        grant_tx
            .send(FlightData {
                flight_descriptor: Some(arrow_flight::FlightDescriptor {
                    r#type: arrow_flight::flight_descriptor::DescriptorType::Path as i32,
                    path: vec![ticket.to_string(), String::new()],
                    ..Default::default()
                }),
                app_metadata: handler::encode_credits(WINDOW).into(),
                ..Default::default()
            })
            .await
            .unwrap();
        let request = futures::StreamExt::map(
            tokio_stream::wrappers::ReceiverStream::new(grant_rx),
            Ok::<_, arrow_flight::error::FlightError>,
        );
        let mut response = client.do_exchange(request).await.unwrap();

        let mut seen = 0i64;
        while let Ok(Some(_)) = response.try_next().await {
            seen += 1;
            // Work between reads: the producer must wait rather than run ahead.
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            let _ = grant_tx
                .send(FlightData {
                    app_metadata: handler::encode_credits(1).into(),
                    ..Default::default()
                })
                .await;
        }
        drop(grant_tx);

        assert_eq!(seen, N, "a slow consumer must still receive every batch");
        let max_inflight = producer.max_inflight(&ticket).await.unwrap();
        assert!(
            max_inflight <= WINDOW as i64 + 1,
            "the producer ran {max_inflight} ahead of a slow consumer (window {WINDOW})",
        );
        assert!(
            max_inflight >= WINDOW as i64,
            "the producer filled only {max_inflight} of {WINDOW} credits, so it never \
             parked — this test is not exercising the blocking path it exists for"
        );
    }

    /// The bound must hold across window sizes, not just the two the other tests pick.
    #[tokio::test]
    async fn the_credit_bound_holds_across_window_sizes() {
        for (n, window) in [(1i64, 1u32), (7, 1), (25, 2), (25, 8), (60, 5), (13, 64)] {
            let producer = ShuffleExchange::bind_ephemeral().await.unwrap();
            let addr = producer.addr().to_string();
            let ticket = ShuffleTicket::new(12, 0, 0, 0, 0);
            producer.publish(&ticket, seq_batches(0, n)).await;

            let got = ShuffleExchange::fetch_with_credits(&addr, &ticket, window)
                .await
                .unwrap();
            assert_eq!(got.len() as i64, n, "n={n} window={window}: lost batches");
            for (i, b) in got.iter().enumerate() {
                let col = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
                assert_eq!(
                    col.value(0),
                    i as i64,
                    "n={n} window={window}: out of order"
                );
            }
            let max_inflight = producer.max_inflight(&ticket).await.unwrap();
            assert!(
                max_inflight >= 1 && max_inflight <= window as i64,
                "n={n} window={window}: in-flight high-water {max_inflight} outside (0, {window}]",
            );
        }
    }

    #[tokio::test]
    async fn credit_window_of_one_still_transfers_all() {
        // Tightest possible window: strict lock-step. Still correct.
        const N: i64 = 12;
        let producer = ShuffleExchange::bind_ephemeral().await.unwrap();
        let addr = producer.addr().to_string();
        let ticket = ShuffleTicket::new(3, 0, 0, 0, 0);
        producer.publish(&ticket, seq_batches(100, N)).await;

        let got = ShuffleExchange::fetch_with_credits(&addr, &ticket, 1)
            .await
            .unwrap();
        assert_eq!(got.len() as i64, N);
        for (i, b) in got.iter().enumerate() {
            let col = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            assert_eq!(col.value(0), 100 + i as i64);
        }
        let max_inflight = producer.max_inflight(&ticket).await.unwrap();
        assert!(
            max_inflight <= 1,
            "window=1 must keep <=1 in flight, got {max_inflight}"
        );
    }

    #[tokio::test]
    async fn wide_window_batches_credit_refills_within_the_bound() {
        // A wide window is where low-watermark refill engages (refill_at = WINDOW/2):
        // the consumer replenishes in bulk instead of one grant per batch. The transfer
        // must still deliver every batch in order AND never let the producer run more
        // than WINDOW batches ahead — batching refills can only tighten the effective
        // window, never loosen it, so the bound is preserved exactly.
        const N: i64 = 500;
        const WINDOW: u32 = 32;

        let producer = ShuffleExchange::bind_ephemeral().await.unwrap();
        let addr = producer.addr().to_string();
        let ticket = ShuffleTicket::new(9, 1, 0, 0, 0);
        producer.publish(&ticket, seq_batches(0, N)).await;

        let got = ShuffleExchange::fetch_with_credits(&addr, &ticket, WINDOW)
            .await
            .unwrap();

        assert_eq!(got.len() as i64, N, "received every batch");
        for (i, b) in got.iter().enumerate() {
            let col = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            assert_eq!(col.value(0), i as i64, "batch {i} out of order");
        }
        let max_inflight = producer.max_inflight(&ticket).await.unwrap();
        assert!(
            max_inflight >= 1 && max_inflight <= WINDOW as i64,
            "in-flight high-water mark {max_inflight} must stay within (0, {WINDOW}]",
        );

        // The point of the batched refill: control-message traffic collapses from one
        // grant per batch (~N) to ~2N/window. Assert it is far below the batch count —
        // a per-batch regression would push this back up to ~N.
        let grants = producer.grant_messages(&ticket).await.unwrap();
        let per_batch_ceiling = N / 4; // generously below N, far above ~2N/window (~31)
        assert!(
            grants > 0 && grants < per_batch_ceiling,
            "batched refill must send far fewer than one grant per batch: \
             {grants} grants for {N} batches at window {WINDOW}",
        );
    }

    #[tokio::test]
    async fn client_pool_pools_per_peer_and_stripes_bounded() {
        // Fetches to one peer share a single per-peer pool (one entry, so O(edges)
        // reconnects collapse to O(peers) at scale), and the pool stripes across at
        // most `connections_per_peer` TCP connections to use the whole NIC — never
        // an unbounded connection per partition.
        crate::set_connections_per_peer(4);
        let producer = ShuffleExchange::bind_ephemeral().await.unwrap();
        let addr = producer.addr().to_string();
        let tickets: Vec<_> = (0..6).map(|d| ShuffleTicket::new(8, 0, 0, d, 0)).collect();
        for (d, t) in tickets.iter().enumerate() {
            producer.publish(t, seq_batches(d as i64 * 100, 4)).await;
        }

        let pool = ClientPool::new();
        for (d, t) in tickets.iter().enumerate() {
            // Data integrity across striped connections: whichever connection carries a
            // fetch, the bytes must be exactly what was published for that ticket.
            let got = pool.fetch_with_credits(&addr, t, 2).await.unwrap();
            assert_eq!(got.len(), 4, "every batch delivered");
            for (i, b) in got.iter().enumerate() {
                let col = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
                assert_eq!(col.value(0), d as i64 * 100 + i as i64, "value misrouted");
            }
        }
        assert_eq!(pool.connection_count(), 1, "one peer => one pool entry");
        assert_eq!(
            pool.channel_count().await,
            4,
            "six fetches stripe across at most connections_per_peer (4) connections",
        );
    }

    #[tokio::test]
    async fn striped_fetch_reconstructs_whole_bucket_over_many_connections() {
        // One big bucket fetched over `stripe` shards (one TCP connection each) must
        // return EVERY batch exactly once — the union of interleaved shards is the whole
        // bucket, regardless of order. This is what makes striping help Batcher's
        // one-endpoint-per-node reduce: a single per-peer bucket streams as several flows.
        const N: i64 = 21; // not a multiple of the stripe, so shards are uneven
        const STRIPE: u32 = 4;
        let producer = ShuffleExchange::bind_ephemeral().await.unwrap();
        let addr = producer.addr().to_string();
        let ticket = ShuffleTicket::new(9, 0, 0, 0, 0);
        producer.publish(&ticket, seq_batches(0, N)).await;

        let pool = ClientPool::new();
        let got = pool
            .fetch_secured_striped(&addr, &ticket, 8, None, STRIPE)
            .await
            .unwrap();

        // Every value 0..N present exactly once (order not guaranteed across shards).
        let mut vals: Vec<i64> = got
            .iter()
            .map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .value(0)
            })
            .collect();
        vals.sort_unstable();
        assert_eq!(
            vals,
            (0..N).collect::<Vec<_>>(),
            "union of shards == whole bucket"
        );
        // The shards opened `STRIPE` parallel connections to the one peer.
        assert_eq!(
            pool.channel_count().await,
            STRIPE as usize,
            "one flow per shard"
        );

        // stripe <= 1 is exactly the un-sharded fetch (whole bucket, in order).
        let whole = pool
            .fetch_secured_striped(&addr, &ticket, 8, None, 1)
            .await
            .unwrap();
        assert_eq!(whole.len() as i64, N);
    }

    #[tokio::test]
    async fn every_compression_codec_roundtrips_exactly() {
        // Each wire codec (none / lz4 / zstd) must deliver batches byte-for-byte equal to
        // what was published — the consumer auto-decompresses from the IPC metadata.
        const N: i64 = 30;
        for code in [0u64, 1, 2] {
            crate::set_compression(code);
            let producer = ShuffleExchange::bind_ephemeral().await.unwrap();
            let addr = producer.addr().to_string();
            let ticket = ShuffleTicket::new(11, 0, 0, code as u32, 0);
            producer.publish(&ticket, seq_batches(0, N)).await;

            let got = ShuffleExchange::fetch_with_credits(&addr, &ticket, 4)
                .await
                .unwrap();
            assert_eq!(got.len() as i64, N, "codec {code}: batch count");
            for (i, b) in got.iter().enumerate() {
                let col = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
                assert_eq!(col.value(0), i as i64, "codec {code}: value at {i}");
            }
        }
        crate::set_compression(1); // restore the default for other tests
    }

    #[tokio::test]
    async fn local_partition_returns_published_without_network() {
        // A partition published on this exchange is readable directly (DIRECT_MEMORY),
        // byte-for-byte equal to what a network fetch would return, and `None` for an
        // unknown ticket.
        let exchange = ShuffleExchange::bind_ephemeral().await.unwrap();
        let ticket = ShuffleTicket::new(4, 0, 0, 0, 0);
        exchange.publish(&ticket, seq_batches(0, 5)).await;

        let local = exchange.local_partition(&ticket).await.unwrap();
        assert_eq!(local.len(), 5);
        for (i, b) in local.iter().enumerate() {
            let col = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            assert_eq!(col.value(0), i as i64);
        }

        let missing = ShuffleTicket::new(4, 0, 0, 9, 0);
        assert!(exchange.local_partition(&missing).await.is_none());
    }

    #[tokio::test]
    async fn blocking_credit_fetch_honors_window() {
        // The FFI-facing wrapper must use the credit-gated DoExchange (not the
        // un-credited DoGet `fetch_blocking` uses): a small window must bound the
        // producer's in-flight high-water mark. Run the blocking fetch (which
        // builds its own runtime) on a blocking thread so it doesn't nest runtimes.
        const N: i64 = 40;
        const WINDOW: u32 = 2;
        let producer = ShuffleExchange::bind_ephemeral().await.unwrap();
        let addr = producer.addr().to_string();
        let ticket = ShuffleTicket::new(5, 2, 0, 0, 0);
        producer.publish(&ticket, seq_batches(0, N)).await;

        let ticket_str = ticket.to_string();
        let got = tokio::task::spawn_blocking(move || {
            fetch_blocking_with_credits(&addr, &ticket_str, WINDOW, None)
        })
        .await
        .unwrap()
        .unwrap();

        assert_eq!(got.len() as i64, N, "received every batch");
        for (i, b) in got.iter().enumerate() {
            let col = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            assert_eq!(col.value(0), i as i64, "batch {i} out of order");
        }
        let max_inflight = producer.max_inflight(&ticket).await.unwrap();
        assert!(
            max_inflight >= 1 && max_inflight <= WINDOW as i64,
            "blocking credit fetch must honor window {WINDOW}, got {max_inflight}",
        );
    }

    /// A token-protected exchange must not be readable through the un-credited `DoGet`.
    ///
    /// `do_exchange` — the production reducer path — rejects a wrong or missing token, but
    /// `do_get` is registered on the *same* gRPC service and its ticket space is a handful
    /// of small integers. If it did not check the token, enabling `shuffle_token` would buy
    /// nothing: anyone who can reach the port could enumerate tickets and read every
    /// shuffle partition of every query, including columns a masking policy hid.
    #[tokio::test]
    async fn do_get_requires_the_shuffle_token_too() {
        let producer = ShuffleExchange::bind_secured("127.0.0.1:0", None, Some("s3cret".into()))
            .await
            .unwrap();
        let addr = producer.addr().to_string();
        let ticket = ShuffleTicket::new(1, 0, 0, 0, 0);
        producer.publish(&ticket, vec![batch_a()]).await;

        let mut anonymous = FlightClient::connect(&addr).await.unwrap();
        let err = anonymous
            .fetch(ticket.to_string())
            .await
            .expect_err("an unauthenticated do_get must not return partition data");
        assert!(
            format!("{err}").contains("nauthenticated"),
            "expected Unauthenticated, got: {err}"
        );
    }

    /// The same fetch succeeds once the caller presents the token.
    #[tokio::test]
    async fn do_get_succeeds_with_the_shuffle_token() {
        let producer = ShuffleExchange::bind_secured("127.0.0.1:0", None, Some("s3cret".into()))
            .await
            .unwrap();
        let addr = producer.addr().to_string();
        let ticket = ShuffleTicket::new(1, 0, 0, 0, 0);
        producer.publish(&ticket, vec![batch_a()]).await;

        let mut client = FlightClient::connect_with_token(&addr, Some("s3cret"))
            .await
            .unwrap();
        assert_eq!(
            client.fetch(ticket.to_string()).await.unwrap(),
            vec![batch_a()]
        );
    }

    /// With no token configured the server is open, as before — the single-host default.
    #[tokio::test]
    async fn do_get_is_open_when_no_token_is_configured() {
        let producer = ShuffleExchange::bind_ephemeral().await.unwrap();
        let addr = producer.addr().to_string();
        let ticket = ShuffleTicket::new(1, 0, 0, 0, 0);
        producer.publish(&ticket, vec![batch_a()]).await;

        let mut client = FlightClient::connect(&addr).await.unwrap();
        assert_eq!(
            client.fetch(ticket.to_string()).await.unwrap(),
            vec![batch_a()]
        );
    }

    // --- TLS / mTLS ----------------------------------------------------------
    // These use `connect_tls` (explicit per-connection TLS) rather than the process-wide
    // `set_client_tls`, so they never race the plaintext tests on the global.

    use crate::tls_test_certs as certs;

    fn server_identity() -> crate::TlsIdentity {
        crate::TlsIdentity::from_pem(certs::SERVER_CRT, certs::SERVER_KEY)
    }

    /// A trusted client over server-auth TLS reads its partition; the bytes are encrypted
    /// on the wire, and the client has verified the server's certificate against the CA.
    #[tokio::test]
    async fn tls_client_reads_over_an_encrypted_connection() {
        let server_tls = crate::TlsServerConfig::new(server_identity());
        let producer = ShuffleExchange::bind_tls("127.0.0.1:0", None, None, Some(server_tls))
            .await
            .unwrap();
        let addr = producer.addr().to_string();
        let ticket = ShuffleTicket::new(1, 0, 0, 0, 0);
        producer.publish(&ticket, vec![batch_a()]).await;

        let client_tls = crate::TlsClientConfig::new(certs::CA_CRT, "localhost");
        let mut client = FlightClient::connect_tls(&addr, &client_tls).await.unwrap();
        assert_eq!(
            client.fetch(ticket.to_string()).await.unwrap(),
            vec![batch_a()]
        );
    }

    /// A plaintext client cannot talk to a TLS server — the handshake fails, so an
    /// eavesdropper who never completes TLS gets nothing.
    #[tokio::test]
    async fn plaintext_client_cannot_reach_a_tls_server() {
        let server_tls = crate::TlsServerConfig::new(server_identity());
        let producer = ShuffleExchange::bind_tls("127.0.0.1:0", None, None, Some(server_tls))
            .await
            .unwrap();
        let addr = producer.addr().to_string();
        let ticket = ShuffleTicket::new(1, 0, 0, 0, 0);
        producer.publish(&ticket, vec![batch_a()]).await;

        // A plaintext client either fails to connect or fails the fetch; either way it
        // never receives partition data.
        let outcome = async {
            let mut c = FlightClient::connect(&addr).await?;
            c.fetch(ticket.to_string()).await
        }
        .await;
        assert!(
            outcome.is_err(),
            "a plaintext client must not read from a TLS server"
        );
    }

    /// A client trusting the wrong CA rejects the server's certificate.
    #[tokio::test]
    async fn client_rejects_a_server_signed_by_an_untrusted_ca() {
        let server_tls = crate::TlsServerConfig::new(server_identity());
        let producer = ShuffleExchange::bind_tls("127.0.0.1:0", None, None, Some(server_tls))
            .await
            .unwrap();
        let addr = producer.addr().to_string();

        let wrong = crate::TlsClientConfig::new(certs::ROGUE_CA_CRT, "localhost");
        let outcome = FlightClient::connect_tls(&addr, &wrong).await;
        assert!(
            outcome.is_err(),
            "a server whose cert does not chain to the trusted CA must be rejected"
        );
    }

    /// Under mTLS a client presenting a certificate the server's CA signed is accepted.
    #[tokio::test]
    async fn mtls_accepts_a_client_signed_by_the_trusted_ca() {
        let server_tls =
            crate::TlsServerConfig::new(server_identity()).with_client_ca(certs::CA_CRT);
        let producer = ShuffleExchange::bind_tls("127.0.0.1:0", None, None, Some(server_tls))
            .await
            .unwrap();
        let addr = producer.addr().to_string();
        let ticket = ShuffleTicket::new(1, 0, 0, 0, 0);
        producer.publish(&ticket, vec![batch_a()]).await;

        let client_tls = crate::TlsClientConfig::new(certs::CA_CRT, "localhost").with_identity(
            crate::TlsIdentity::from_pem(certs::CLIENT_CRT, certs::CLIENT_KEY),
        );
        let mut client = FlightClient::connect_tls(&addr, &client_tls).await.unwrap();
        assert_eq!(
            client.fetch(ticket.to_string()).await.unwrap(),
            vec![batch_a()]
        );
    }

    /// Under mTLS a client whose certificate the server's CA did not sign is rejected —
    /// the network-level analogue of a wrong shuffle token, enforced at the handshake.
    #[tokio::test]
    async fn mtls_rejects_a_client_with_an_untrusted_certificate() {
        let server_tls =
            crate::TlsServerConfig::new(server_identity()).with_client_ca(certs::CA_CRT);
        let producer = ShuffleExchange::bind_tls("127.0.0.1:0", None, None, Some(server_tls))
            .await
            .unwrap();
        let addr = producer.addr().to_string();
        let ticket = ShuffleTicket::new(1, 0, 0, 0, 0);
        producer.publish(&ticket, vec![batch_a()]).await;

        // A rogue client cert signed by an unrelated CA.
        let rogue = crate::TlsClientConfig::new(certs::CA_CRT, "localhost").with_identity(
            crate::TlsIdentity::from_pem(certs::ROGUE_CRT, certs::ROGUE_KEY),
        );
        let outcome = async {
            let mut c = FlightClient::connect_tls(&addr, &rogue).await?;
            c.fetch(ticket.to_string()).await
        }
        .await;
        assert!(
            outcome.is_err(),
            "an mTLS client with an untrusted certificate must be rejected"
        );
    }

    /// A client with no certificate is rejected by an mTLS server (client auth required).
    #[tokio::test]
    async fn mtls_rejects_a_client_with_no_certificate() {
        let server_tls =
            crate::TlsServerConfig::new(server_identity()).with_client_ca(certs::CA_CRT);
        let producer = ShuffleExchange::bind_tls("127.0.0.1:0", None, None, Some(server_tls))
            .await
            .unwrap();
        let addr = producer.addr().to_string();
        let ticket = ShuffleTicket::new(1, 0, 0, 0, 0);
        producer.publish(&ticket, vec![batch_a()]).await;

        // Server-auth only (no client identity) against a server that requires one.
        let no_cert = crate::TlsClientConfig::new(certs::CA_CRT, "localhost");
        let outcome = async {
            let mut c = FlightClient::connect_tls(&addr, &no_cert).await?;
            c.fetch(ticket.to_string()).await
        }
        .await;
        assert!(
            outcome.is_err(),
            "an mTLS server must reject a client that presents no certificate"
        );
    }

    /// TLS and the shuffle token compose: the wire is encrypted *and* the token is
    /// still required. Both layers hold at once.
    #[tokio::test]
    async fn tls_and_token_compose() {
        let server_tls = crate::TlsServerConfig::new(server_identity());
        let producer =
            ShuffleExchange::bind_tls("127.0.0.1:0", None, Some("s3cret".into()), Some(server_tls))
                .await
                .unwrap();
        let addr = producer.addr().to_string();
        let ticket = ShuffleTicket::new(1, 0, 0, 0, 0);
        producer.publish(&ticket, vec![batch_a()]).await;

        let client_tls = crate::TlsClientConfig::new(certs::CA_CRT, "localhost");
        // Right cert, wrong (absent) token → rejected over the encrypted connection.
        let mut anon = FlightClient::connect_tls(&addr, &client_tls).await.unwrap();
        assert!(anon.fetch(ticket.to_string()).await.is_err());
    }

    fn zero_row_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(Vec::<i64>::new()))]).unwrap()
    }

    #[tokio::test]
    async fn all_zero_row_bucket_resolves_to_empty_without_hanging() {
        // A bucket made only of zero-row batches must resolve to "no rows" promptly. The
        // Flight encoder emits no message for a zero-row batch, so without the producer's
        // filter the consumer would wait the whole idle window for a batch that never
        // comes and then report this healthy worker as unreachable. The outer timeout
        // (well below the default idle window) turns any such regression into a failure
        // rather than a hang.
        let producer = ShuffleExchange::bind_ephemeral().await.unwrap();
        let addr = producer.addr().to_string();
        let ticket = ShuffleTicket::new(77, 0, 0, 0, 0);
        producer
            .publish(
                &ticket,
                vec![zero_row_batch(), zero_row_batch(), zero_row_batch()],
            )
            .await;
        let out = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            ShuffleExchange::fetch_with_credits(&addr, &ticket, 2),
        )
        .await
        .expect("must not hang on an all-zero-row bucket")
        .unwrap();
        assert!(out.is_empty(), "all-zero-row bucket resolves to no rows");
    }

    #[tokio::test]
    async fn local_partition_matches_network_fetch_for_zero_row_bucket() {
        // The `local_partition` doc claims DIRECT_MEMORY is "byte-for-byte equal to what a
        // network fetch would return". A bucket with an interior zero-row batch exercises
        // that claim: the network path filters zero-row batches; local must agree.
        let producer = ShuffleExchange::bind_ephemeral().await.unwrap();
        let addr = producer.addr().to_string();
        let ticket = ShuffleTicket::new(99, 0, 0, 0, 0);
        producer
            .publish(&ticket, vec![one_row(1), zero_row_batch(), one_row(2)])
            .await;
        let net = ShuffleExchange::fetch_with_credits(&addr, &ticket, 4)
            .await
            .unwrap();
        let local = producer.local_partition(&ticket).await.unwrap();
        assert_eq!(
            local, net,
            "DIRECT_MEMORY must return the same batches as a network fetch"
        );
    }

    fn one_row(v: i64) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![v]))]).unwrap()
    }

    #[tokio::test]
    async fn credited_path_preserves_nulls_and_strings_under_every_codec() {
        // The credited do_exchange path with nulls + variable-length strings under each
        // wire codec must be byte-identical. Existing codec tests only cover Int64.
        for code in [0u64, 1, 2] {
            crate::set_compression(code);
            let producer = ShuffleExchange::bind_ephemeral().await.unwrap();
            let addr = producer.addr().to_string();
            let ticket = ShuffleTicket::new(88, code as u32, 0, 0, 0);
            producer.publish(&ticket, vec![batch_a(), batch_a2()]).await;
            let got = ShuffleExchange::fetch_with_credits(&addr, &ticket, 2)
                .await
                .unwrap();
            assert_eq!(
                got,
                vec![batch_a(), batch_a2()],
                "codec {code}: exact bytes"
            );
        }
        crate::set_compression(1);
    }

    #[test]
    fn https_uri_upgrades_bare_and_http_addresses() {
        assert_eq!(tls::https_uri("1.2.3.4:50"), "https://1.2.3.4:50");
        assert_eq!(tls::https_uri("http://1.2.3.4:50"), "https://1.2.3.4:50");
        assert_eq!(tls::https_uri("https://1.2.3.4:50"), "https://1.2.3.4:50");
    }

    #[test]
    fn malformed_pem_is_rejected_early_with_a_clear_error() {
        let err = tls::check_pem("server certificate", "not a pem").unwrap_err();
        assert!(format!("{err}").contains("not PEM-encoded"));
    }
}
