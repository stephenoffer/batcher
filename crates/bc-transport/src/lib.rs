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
use futures::stream::TryStreamExt;
use tonic::transport::{Channel, Server};

use crate::handler::FlightHandler;
use crate::store::PartitionStore;

mod exchange;
mod handler;
mod store;
mod ticket;

pub use exchange::{classify, ClientPool, FetchFault, ShuffleExchange};
pub use ticket::ShuffleTicket;

/// Default number of in-flight `RecordBatch` credits for a credit-bounded
/// exchange when the caller does not specify one.
pub const DEFAULT_CREDITS: u32 = 4;

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

    /// Set the process-wide transport timeouts. `idle_ms == 0` keeps the current
    /// idle timeout; `keepalive_ms == 0` disables keepalive.
    pub fn set_transport_timeouts(idle_ms: u64, keepalive_ms: u64) {
        if idle_ms > 0 {
            FETCH_IDLE_TIMEOUT_MS.store(idle_ms, Ordering::Relaxed);
        }
        KEEPALIVE_MS.store(keepalive_ms, Ordering::Relaxed);
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
}

pub use tunables::{fetch_idle_timeout, keepalive, set_transport_timeouts};

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
        }
    }

    /// Build a server over a shared store (so a [`ShuffleExchange`] keeps its own
    /// handle to register partitions after start), optionally requiring `token` for
    /// `do_exchange` (N5). `None` disables the auth check.
    pub(crate) fn with_store_and_token(store: Arc<PartitionStore>, token: Option<String>) -> Self {
        Self { store, token }
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
        let svc = FlightServiceServer::new(FlightHandler {
            store: self.store,
            token: self.token,
        });
        Server::builder()
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
        let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);

        let svc = FlightServiceServer::new(FlightHandler {
            store: self.store,
            token: self.token,
        });
        let handle = tokio::spawn(async move {
            Server::builder()
                .add_service(svc)
                .serve_with_incoming(incoming)
                .await
        });

        Ok((local_addr, ServerHandle { task: handle }))
    }
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

    /// Establish a tonic [`Channel`] to `addr`, accepting a bare `host:port` or a
    /// full URI. Exposed so a [`ClientPool`] can cache and reuse the channel across
    /// fetches instead of reconnecting per partition.
    ///
    /// [`ClientPool`]: crate::exchange::ClientPool
    pub async fn build_channel(addr: &str) -> TransportResult<Channel> {
        let uri = if addr.contains("://") {
            addr.to_string()
        } else {
            format!("http://{addr}")
        };
        let mut endpoint = Channel::from_shared(uri.into_bytes())
            .map_err(|e| TransportError::Io(format!("invalid uri: {e}")))?;
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

    #[tokio::test]
    async fn fetch_roundtrips_multiple_partitions() {
        let (addr, _handle) = start_server().await;

        let mut client = FlightClient::connect(addr.to_string()).await.unwrap();

        // Partition 1: two batches, multiple columns incl. nullable Utf8.
        let got = client.fetch("p1/s0/0/0").await.unwrap();
        let expected = vec![batch_a(), batch_a2()];
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
    async fn client_pool_reuses_one_channel_per_peer() {
        // Two fetches to the same peer must share a single cached gRPC channel —
        // the property that turns O(edges) reconnects into O(peers) at scale.
        let producer = ShuffleExchange::bind_ephemeral().await.unwrap();
        let addr = producer.addr().to_string();
        let t1 = ShuffleTicket::new(8, 0, 0, 0, 0);
        let t2 = ShuffleTicket::new(8, 0, 0, 1, 0);
        producer.publish(&t1, seq_batches(0, 4)).await;
        producer.publish(&t2, seq_batches(100, 4)).await;

        let pool = ClientPool::new();
        let b1 = pool.fetch_with_credits(&addr, &t1, 2).await.unwrap();
        let b2 = pool.fetch_with_credits(&addr, &t2, 2).await.unwrap();
        assert_eq!(b1.len(), 4);
        assert_eq!(b2.len(), 4);
        assert_eq!(
            pool.connection_count(),
            1,
            "both fetches reused one channel"
        );
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
}
