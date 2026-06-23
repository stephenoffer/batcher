//! The node-level [`ShuffleExchange`]: the ergonomic API the distributed layer
//! calls to publish and fetch shuffle partitions between nodes with
//! credit-bounded streaming.

use std::net::SocketAddr;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow_flight::{FlightData, FlightDescriptor};
use futures::stream::{StreamExt, TryStreamExt};

use crate::handler::encode_credits;
use crate::store::PartitionStore;
use crate::ticket::ShuffleTicket;
use crate::{
    FlightClient, FlightServer, ServerHandle, TransportError, TransportResult, DEFAULT_CREDITS,
};

/// How a failed fetch should be handled by the recovery layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchFault {
    /// A transient fault — a dropped connection, an idle/hung peer, a server that's
    /// unavailable or cancelled. The partition can be recomputed and re-fetched, so
    /// the caller should retry rather than fail the query.
    Retryable,
    /// A definite, non-transient failure (a decode error, a protocol violation, an
    /// auth rejection). Retrying will not help — surface it.
    Fatal,
}

/// Classify a transport error so the recovery layer retries only what a retry can
/// fix. A dead/idle peer or an unavailable server is `Retryable` (recompute +
/// re-fetch); everything else is `Fatal`.
pub fn classify(err: &TransportError) -> FetchFault {
    match err {
        TransportError::Transport(_) | TransportError::IdleTimeout(_) => FetchFault::Retryable,
        TransportError::Status(s) if is_retryable_code(s.code()) => FetchFault::Retryable,
        TransportError::Flight(arrow_flight::error::FlightError::Tonic(s))
            if is_retryable_code(s.code()) =>
        {
            FetchFault::Retryable
        }
        _ => FetchFault::Fatal,
    }
}

/// gRPC status codes that mean "the peer/connection is transiently gone".
fn is_retryable_code(code: tonic::Code) -> bool {
    matches!(
        code,
        tonic::Code::Unavailable | tonic::Code::Aborted | tonic::Code::Cancelled
    )
}

/// Node-level handle for the shuffle exchange: the ergonomic API the
/// distributed layer calls to move shuffle partitions between nodes.
///
/// One [`ShuffleExchange`] per worker process. Mappers [`publish`] their output
/// partitions on the local server; reducers [`fetch`] remote partitions from
/// every upstream node with credit-bounded streaming. See the crate-level docs
/// for the full picture.
///
/// [`publish`]: ShuffleExchange::publish
/// [`fetch`]: ShuffleExchange::fetch
pub struct ShuffleExchange {
    store: Arc<PartitionStore>,
    addr: SocketAddr,
    advertised: String,
    _handle: ServerHandle,
}

impl ShuffleExchange {
    /// Start a shuffle exchange on an ephemeral loopback port and begin serving.
    ///
    /// Single-host / test path: binds `127.0.0.1:0` and advertises that loopback
    /// address. For a cross-node cluster use [`Self::bind_advertised`] so reducers
    /// on other hosts get a routable address rather than this node's loopback.
    pub async fn bind_ephemeral() -> TransportResult<Self> {
        Self::bind("127.0.0.1:0").await
    }

    /// Start a shuffle exchange bound to `bind`, advertising the bound address.
    pub async fn bind(bind: &str) -> TransportResult<Self> {
        Self::bind_advertised(bind, None).await
    }

    /// Start a shuffle exchange bound to `bind`, advertising `advertise_host`.
    ///
    /// Cluster path: bind to all interfaces (`0.0.0.0:0`) so peers on other nodes
    /// can connect, but advertise `{advertise_host}:{bound_port}` — the node's
    /// **routable** address (the Ray node IP) — since the bound address is either
    /// loopback (unreachable cross-node) or `0.0.0.0` (not dialable). With
    /// `advertise_host = None` the bound address is advertised verbatim (the
    /// single-host case).
    pub async fn bind_advertised(
        bind: &str,
        advertise_host: Option<&str>,
    ) -> TransportResult<Self> {
        Self::bind_secured(bind, advertise_host, None).await
    }

    /// Like [`Self::bind_advertised`] but requires `token` for incoming fetches (N5).
    pub async fn bind_secured(
        bind: &str,
        advertise_host: Option<&str>,
        token: Option<String>,
    ) -> TransportResult<Self> {
        let store = Arc::new(PartitionStore::default());
        // Reuse FlightServer's binding logic but keep our own handle on the
        // store so publish() can register after the server is running.
        let server = FlightServer::with_store_and_token(store.clone(), token);
        let (addr, handle) = server.serve_on(bind).await?;
        let advertised = match advertise_host {
            Some(host) if !host.is_empty() => format!("{host}:{}", addr.port()),
            _ => addr.to_string(),
        };
        Ok(Self {
            store,
            addr,
            advertised,
            _handle: handle,
        })
    }

    /// The local socket address this node's exchange is bound to.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The address reducers should dial to reach this exchange.
    ///
    /// Equals [`Self::addr`] in the single-host case; on a cluster it is the
    /// routable `{node_ip}:{port}` set via [`Self::bind_advertised`].
    pub fn advertised_addr(&self) -> &str {
        &self.advertised
    }

    /// Publish (expose) a shuffle output partition under `ticket`.
    ///
    /// Mappers call this once per output partition. The batches are served to
    /// any reducer that [`fetch`]es the ticket. Re-publishing the same ticket
    /// replaces the previous batches.
    ///
    /// [`fetch`]: ShuffleExchange::fetch
    pub async fn publish(&self, ticket: &ShuffleTicket, batches: Vec<RecordBatch>) {
        self.store.register(ticket.to_string(), batches).await;
    }

    /// Read a partition this exchange itself published, without any network hop.
    ///
    /// When a reducer's bucket was produced by a mapper in the *same* process, the
    /// data already sits in this exchange's store; returning it directly avoids
    /// serializing it through a loopback Flight stream. This is the `DIRECT_MEMORY`
    /// transfer mode — the cheapest way to "move" a co-located partition, and the
    /// concrete win over routing every partition through an object store. `None`
    /// if nothing was published under `ticket` here. RecordBatch clones are shallow
    /// (Arc buffer bumps), so this does not copy the underlying data.
    pub async fn local_partition(&self, ticket: &ShuffleTicket) -> Option<Vec<RecordBatch>> {
        self.store
            .get(&ticket.to_string())
            .await
            .map(|b| (*b).clone())
    }

    /// High-water mark of how many batches the producer had in flight (sent but
    /// not yet acked by the consumer) for `ticket`, or `None` if the ticket was
    /// never published. Used to verify the credit bound; also useful telemetry.
    pub async fn max_inflight(&self, ticket: &ShuffleTicket) -> Option<i64> {
        self.store.gauge(&ticket.to_string()).await.map(|g| g.max())
    }

    /// Evict one published partition (its reducers have fetched it), freeing it.
    pub async fn release(&self, ticket: &ShuffleTicket) {
        self.store.remove(&ticket.to_string()).await;
    }

    /// Evict every partition for plan `plan_id` — call at plan teardown so a
    /// reused worker doesn't accumulate finished plans' shuffle outputs.
    pub async fn clear_plan(&self, plan_id: u64) {
        self.store.remove_prefix(&format!("{plan_id}/")).await;
    }

    /// Evict every published partition on this exchange.
    pub async fn clear(&self) {
        self.store.clear().await;
    }

    /// Number of partitions currently retained (telemetry / leak tests).
    pub async fn partition_count(&self) -> usize {
        self.store.len().await
    }

    /// Fetch a remote partition from `(addr, ticket)` with a credit-bounded
    /// stream using the default window ([`DEFAULT_CREDITS`]).
    ///
    /// See [`Self::fetch_with_credits`] for the flow-control details.
    pub async fn fetch(
        &self,
        addr: &str,
        ticket: &ShuffleTicket,
    ) -> TransportResult<Vec<RecordBatch>> {
        Self::fetch_with_credits(addr, ticket, DEFAULT_CREDITS).await
    }

    /// As [`Self::fetch_with_credits`], presenting `token` to an auth-gated peer (N5).
    pub async fn fetch_secured(
        addr: &str,
        ticket: &ShuffleTicket,
        credits: u32,
        token: Option<&str>,
    ) -> TransportResult<Vec<RecordBatch>> {
        let mut client = FlightClient::connect(addr).await?;
        credit_exchange(&mut client, ticket, credits, token).await
    }

    /// Fetch a remote partition from `(addr, ticket)` over a credit-gated
    /// `DoExchange`, keeping at most `credits` `RecordBatch`es in flight.
    ///
    /// The consumer seeds `credits` and tops up by one for each batch it
    /// consumes, so the producer never buffers more than `credits` batches
    /// ahead. `credits` is clamped to at least 1. The returned batches are
    /// byte-for-byte equivalent (schema + values) to what the producer
    /// published.
    pub async fn fetch_with_credits(
        addr: &str,
        ticket: &ShuffleTicket,
        credits: u32,
    ) -> TransportResult<Vec<RecordBatch>> {
        let mut client = FlightClient::connect(addr).await?;
        credit_exchange(&mut client, ticket, credits, None).await
    }
}

/// Whether a transport error is a Flight `NotFound` — i.e. the ticket was never
/// published. In a shuffle this is the *expected* empty-bucket case (a mapper that
/// produced no rows for a reducer never publishes the ticket), so callers map it to
/// an empty partition rather than an error. Every *other* error (a dead/unreachable
/// peer, a decode failure) is a real fault and must propagate — the reducer must not
/// silently treat it as an empty bucket.
fn is_ticket_not_found(err: &TransportError) -> bool {
    let code = match err {
        TransportError::Status(s) => Some(s.code()),
        TransportError::Flight(arrow_flight::error::FlightError::Tonic(s)) => Some(s.code()),
        _ => None,
    };
    code == Some(tonic::Code::NotFound)
}

/// Run the credit-gated `DoExchange` over an already-connected client.
///
/// The consumer seeds `credits` and tops up by one per consumed batch, so the
/// producer never buffers more than `credits` `RecordBatch`es ahead. Factored out
/// so both the connect-fresh [`ShuffleExchange::fetch_with_credits`] and the
/// channel-pooling [`ClientPool`] share one implementation.
///
/// A `NotFound` (unpublished ticket) resolves to an empty partition — the expected
/// empty-bucket case — so only genuine faults surface as errors.
pub(crate) async fn credit_exchange(
    client: &mut FlightClient,
    ticket: &ShuffleTicket,
    credits: u32,
    token: Option<&str>,
) -> TransportResult<Vec<RecordBatch>> {
    match credit_exchange_inner(client, ticket, credits, token).await {
        Err(ref e) if is_ticket_not_found(e) => Ok(Vec::new()),
        other => other,
    }
}

async fn credit_exchange_inner(
    client: &mut FlightClient,
    ticket: &ShuffleTicket,
    credits: u32,
    token: Option<&str>,
) -> TransportResult<Vec<RecordBatch>> {
    let credits = credits.max(1);
    let ticket_str = ticket.to_string();

    // A channel carries credit-grant control messages from the consumer loop below
    // into the outbound request stream. Bound it generously; grants are tiny and
    // the producer's appetite is bounded by `credits` anyway.
    let (grant_tx, grant_rx) = tokio::sync::mpsc::channel::<FlightData>(64);

    // First message: name the ticket (path[0]) + the auth token (path[1], empty when
    // unused) and seed the initial credit window.
    let first = FlightData {
        flight_descriptor: Some(FlightDescriptor {
            r#type: arrow_flight::flight_descriptor::DescriptorType::Path as i32,
            path: vec![ticket_str.clone(), token.unwrap_or("").to_string()],
            ..Default::default()
        }),
        app_metadata: encode_credits(credits).into(),
        ..Default::default()
    };
    grant_tx
        .send(first)
        .await
        .map_err(|e| TransportError::Io(format!("send initial grant: {e}")))?;

    // Outbound request stream the producer reads its grants from.
    let request_stream = tokio_stream::wrappers::ReceiverStream::new(grant_rx)
        .map(Ok::<_, arrow_flight::error::FlightError>);

    let mut response = client.do_exchange(request_stream).await?;

    // Consume the producer's data stream, topping up one credit per consumed batch.
    // The window stays ~`credits` deep: the producer is allowed to be at most
    // `credits` batches ahead of what we've pulled.
    let mut batches = Vec::new();
    loop {
        // Idle timeout (C24): a hung/dead peer must not block the reducer forever.
        // The bound is between batches, not on the whole transfer, so a large but
        // healthy partition is never cut off mid-stream. Configurable per process
        // (Carbonite); a timeout is a *retryable* fault (recompute + re-fetch).
        let idle = crate::fetch_idle_timeout();
        let next = tokio::time::timeout(idle, response.try_next()).await;
        let batch = match next {
            Ok(res) => res?,
            Err(_) => return Err(TransportError::IdleTimeout(idle)),
        };
        let Some(batch) = batch else { break };
        batches.push(batch);
        // Grant one more credit now that a slot has freed up. Use `send().await`
        // rather than `try_send` (C28) so a momentarily-full control channel never
        // *drops* a grant — a dropped grant would stall the credit-gated producer.
        // A closed channel means the producer already finished; that is benign.
        if grant_tx
            .send(FlightData {
                app_metadata: encode_credits(1).into(),
                ..Default::default()
            })
            .await
            .is_err()
        {
            break;
        }
    }
    Ok(batches)
}

/// A consumer-side pool that reuses one gRPC channel per peer address across
/// fetches, instead of reconnecting every time.
///
/// tonic channels multiplex many streams over one HTTP/2 connection, so a cached
/// channel serves a peer's whole shuffle. Reconnect cost is paid once per peer,
/// not once per partition — the difference between O(edges) and O(nodes)
/// connection setups, which is what lets the shuffle scale to a large cluster.
#[derive(Default)]
pub struct ClientPool {
    channels: dashmap::DashMap<String, tonic::transport::Channel>,
}

impl ClientPool {
    /// An empty pool. Connections are established lazily on first fetch per peer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of peer addresses with a live cached channel (telemetry/tests).
    pub fn connection_count(&self) -> usize {
        self.channels.len()
    }

    async fn channel(&self, addr: &str) -> TransportResult<tonic::transport::Channel> {
        if let Some(existing) = self.channels.get(addr) {
            return Ok(existing.clone());
        }
        // Build outside the map lock (connect is async; a DashMap entry guard can't
        // be held across an await). A concurrent first-fetch to the same peer may
        // also build one; `or_insert` keeps whichever lands first and drops the
        // loser, so a peer never ends up with two retained channels (N20).
        let channel = FlightClient::build_channel(addr).await?;
        let entry = self.channels.entry(addr.to_string()).or_insert(channel);
        Ok(entry.clone())
    }

    /// Fetch `ticket` from `addr` over a credit-gated stream on a *pooled* channel.
    ///
    /// If the cached channel is stale (the peer restarted, so the connection is
    /// dead), the first attempt fails with a transport/connect error; the channel is
    /// then evicted and the fetch is retried once on a fresh connection (C49). A
    /// `NotFound` (empty bucket) is not a connection fault and is returned as-is.
    pub async fn fetch_with_credits(
        &self,
        addr: &str,
        ticket: &ShuffleTicket,
        credits: u32,
    ) -> TransportResult<Vec<RecordBatch>> {
        self.fetch_secured(addr, ticket, credits, None).await
    }

    /// As [`Self::fetch_with_credits`], presenting `token` to an auth-gated peer (N5).
    pub async fn fetch_secured(
        &self,
        addr: &str,
        ticket: &ShuffleTicket,
        credits: u32,
        token: Option<&str>,
    ) -> TransportResult<Vec<RecordBatch>> {
        let channel = self.channel(addr).await?;
        let mut client = FlightClient::from_channel(channel);
        match credit_exchange(&mut client, ticket, credits, token).await {
            Err(e) if is_connection_error(&e) => {
                // Drop the dead channel and redial once.
                self.channels.remove(addr);
                let channel = self.channel(addr).await?;
                let mut client = FlightClient::from_channel(channel);
                credit_exchange(&mut client, ticket, credits, token).await
            }
            other => other,
        }
    }
}

/// Whether an error is a transport/connection failure (a dead peer / stale
/// channel) worth redialing — i.e. the retryable class (see [`classify`]).
fn is_connection_error(err: &TransportError) -> bool {
    classify(err) == FetchFault::Retryable
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_separates_retryable_from_fatal() {
        // A hung/idle peer and an unavailable/cancelled server are retryable: the
        // partition can be recomputed and re-fetched.
        assert_eq!(
            classify(&TransportError::IdleTimeout(
                std::time::Duration::from_secs(1)
            )),
            FetchFault::Retryable
        );
        assert_eq!(
            classify(&TransportError::from(tonic::Status::unavailable("down"))),
            FetchFault::Retryable
        );
        assert_eq!(
            classify(&TransportError::from(tonic::Status::cancelled("gone"))),
            FetchFault::Retryable
        );

        // A decode/protocol/auth failure is fatal — a retry won't help.
        assert_eq!(
            classify(&TransportError::Io("bad uri".into())),
            FetchFault::Fatal
        );
        assert_eq!(
            classify(&TransportError::from(tonic::Status::permission_denied(
                "no token"
            ))),
            FetchFault::Fatal
        );
        // NotFound (an unpublished empty bucket) is not a fault to retry here.
        assert_eq!(
            classify(&TransportError::from(tonic::Status::not_found("no ticket"))),
            FetchFault::Fatal
        );
    }

    #[test]
    fn configurable_idle_timeout_round_trips() {
        crate::set_transport_timeouts(5_000, 2_000);
        assert_eq!(
            crate::fetch_idle_timeout(),
            std::time::Duration::from_millis(5_000)
        );
        assert_eq!(
            crate::keepalive(),
            Some(std::time::Duration::from_millis(2_000))
        );
        // idle_ms == 0 keeps the current value; keepalive 0 disables.
        crate::set_transport_timeouts(0, 0);
        assert_eq!(
            crate::fetch_idle_timeout(),
            std::time::Duration::from_millis(5_000)
        );
        assert_eq!(crate::keepalive(), None);
        // Restore the default so other tests see a clean global.
        crate::set_transport_timeouts(60_000, 0);
    }
}
