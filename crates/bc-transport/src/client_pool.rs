//! The consumer-side connection pool: how a reducer dials a peer and how many streams
//! it runs against it.
//!
//! One tonic channel is one HTTP/2 connection is one TCP flow, and a cloud NIC caps a
//! *single* flow well below line rate — so a peer that is fetched from concurrently is
//! served over several connections. Separate from `exchange` because the two answer
//! different questions: `exchange` is what one stream *is* (credit-gated, ticket-named),
//! and this is how many of them a peer gets and over what connections.

use std::sync::Arc;
use std::time::Duration;

use arrow::array::RecordBatch;

use crate::exchange::{classify, credit_exchange_group_shard, group_name, FetchFault};
use crate::ticket::ShuffleTicket;
use crate::{FlightClient, TransportError, TransportResult};

/// A lazily-grown set of gRPC channels to one peer, striping concurrent fetches
/// across up to `max` TCP connections.
///
/// One tonic channel is one HTTP/2 connection is one TCP flow. A single flow is
/// capped below a cloud NIC's line rate, so a peer that a reducer fetches from
/// concurrently is served over several connections to reach line rate. The pool
/// grows only under concurrency (each fetch that finds fewer than `max` channels
/// builds one more), so a cold peer keeps a single connection and a hot peer stripes
/// — connection cost stays `O(peers)` in the common case, `O(peers x max)` at worst.
struct PeerChannels {
    /// The peer's connections; grown lazily up to `max`, then round-robined. A tokio
    /// mutex (not std) because a channel is *built* under the lock (an `.await`).
    channels: tokio::sync::Mutex<Vec<tonic::transport::Channel>>,
    /// Round-robin cursor over the built channels.
    next: std::sync::atomic::AtomicUsize,
    max: usize,
    addr: String,
}

impl PeerChannels {
    fn new(addr: String, max: usize) -> Self {
        Self {
            channels: tokio::sync::Mutex::new(Vec::new()),
            next: std::sync::atomic::AtomicUsize::new(0),
            max: max.max(1),
            addr,
        }
    }

    /// A channel to the peer: build a fresh connection while the pool is below `max`
    /// (so concurrent fetches each open their own flow), otherwise round-robin an
    /// existing one. Building holds the async lock, so the pool never overshoots `max`.
    async fn acquire(&self) -> TransportResult<tonic::transport::Channel> {
        let mut chans = self.channels.lock().await;
        if chans.len() < self.max {
            let channel = FlightClient::build_channel(&self.addr).await?;
            chans.push(channel.clone());
            return Ok(channel);
        }
        let i = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % chans.len();
        Ok(chans[i].clone())
    }

    /// Drop every connection (the peer restarted, so they are all stale); the next
    /// `acquire` rebuilds from scratch.
    async fn reset(&self) {
        self.channels.lock().await.clear();
    }

    /// Whether this peer currently holds no connections.
    async fn is_empty(&self) -> bool {
        self.channels.lock().await.is_empty()
    }

    async fn len(&self) -> usize {
        self.channels.lock().await.len()
    }
}

/// A consumer-side pool that stripes a peer's fetches across a small set of gRPC
/// connections, reused across fetches instead of reconnecting every time.
///
/// tonic channels multiplex many streams over one HTTP/2 connection; a *single* such
/// connection is one TCP flow and a cloud NIC caps a single flow below line rate, so
/// each peer gets a lazily-grown [`PeerChannels`] pool (up to
/// [`connections_per_peer`]) to use the whole link under concurrency. Reconnect cost
/// is still paid per peer, not per partition — `O(peers)` connections for a cold
/// shuffle, `O(peers x connections_per_peer)` for a throughput-bound one — which is
/// what lets the shuffle both scale to a large cluster and saturate the NIC.
///
/// [`connections_per_peer`]: crate::connections_per_peer
#[derive(Default)]
pub struct ClientPool {
    peers: dashmap::DashMap<String, Arc<PeerChannels>>,
}

impl ClientPool {
    /// An empty pool. Connections are established lazily on first fetch per peer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of peer addresses with a live connection pool (telemetry/tests). Counts
    /// *peers*, not connections — striping to one peer stays a single entry here.
    #[must_use]
    pub fn connection_count(&self) -> usize {
        self.peers.len()
    }

    /// Total open connections across all peers (telemetry/tests) — reflects striping.
    pub async fn channel_count(&self) -> usize {
        let peers: Vec<_> = self.peers.iter().map(|e| e.value().clone()).collect();
        let mut total = 0;
        for p in peers {
            total += p.len().await;
        }
        total
    }

    fn peer(&self, addr: &str) -> Arc<PeerChannels> {
        // `entry` briefly holds a shard lock but does no await, so it is safe. A
        // concurrent first-fetch to the same peer resolves to one shared pool.
        self.peers
            .entry(addr.to_string())
            .or_insert_with(|| {
                Arc::new(PeerChannels::new(
                    addr.to_string(),
                    crate::connections_per_peer(),
                ))
            })
            .clone()
    }

    async fn channel(&self, addr: &str) -> TransportResult<tonic::transport::Channel> {
        self.peer(addr).acquire().await
    }

    /// Fetch `ticket` from `addr` over a credit-gated stream on a *pooled* channel.
    ///
    /// If the cached connections are stale (the peer restarted), the first attempt
    /// fails with a transport/connect error; the peer's pool is then reset and the
    /// fetch is retried once on a fresh connection. A `NotFound` (empty bucket)
    /// is not a connection fault and is returned as-is.
    pub async fn fetch_with_credits(
        &self,
        addr: &str,
        ticket: &ShuffleTicket,
        credits: u32,
    ) -> TransportResult<Vec<RecordBatch>> {
        self.fetch_secured(addr, ticket, credits, None).await
    }

    /// As [`Self::fetch_with_credits`], presenting `token` to an auth-gated peer.
    pub async fn fetch_secured(
        &self,
        addr: &str,
        ticket: &ShuffleTicket,
        credits: u32,
        token: Option<&str>,
    ) -> TransportResult<Vec<RecordBatch>> {
        self.fetch_group(addr, &ticket.to_string(), credits, token)
            .await
    }

    /// One pooled, credit-gated fetch of `name` — a single ticket or a [`group_name`] of
    /// several buckets served over the one stream.
    async fn fetch_group(
        &self,
        addr: &str,
        name: &str,
        credits: u32,
        token: Option<&str>,
    ) -> TransportResult<Vec<RecordBatch>> {
        // Timed here rather than inside the retry, so a redial's dead-connection timeout is
        // never charged to the peer's bandwidth: a stale channel would otherwise make a
        // healthy node read as the slowest wire in the fleet.
        let started = std::time::Instant::now();
        let out = self.fetch_once_with_retry(addr, name, credits, token).await;
        if let Ok((batches, starved, first)) = out.as_ref() {
            let bytes: usize = batches.iter().map(|b| b.get_array_memory_size()).sum();
            crate::record_fetch(addr, bytes as u64, started.elapsed(), *starved, *first);
        }
        if out.is_err() {
            // Every failure path lands here, including a first `acquire` that could not
            // connect at all — which is the common shape for a peer that has gone away, and
            // the one that still leaves an entry behind. See `forget_unreachable`.
            self.forget_unreachable(addr).await;
        }
        out.map(|(batches, ..)| batches)
    }

    /// One pooled fetch, redialing once if the cached connections turn out to be stale.
    async fn fetch_once_with_retry(
        &self,
        addr: &str,
        name: &str,
        credits: u32,
        token: Option<&str>,
    ) -> TransportResult<(Vec<RecordBatch>, Duration, Duration)> {
        let channel = self.channel(addr).await?;
        let mut client = FlightClient::from_channel(channel);
        match credit_exchange_group_shard(&mut client, name, credits, token, 0, 1).await {
            Err(e) if is_connection_error(&e) => {
                // Drop the dead connections and redial once.
                crate::record_retry(addr);
                self.peer(addr).reset().await;
                let channel = self.channel(addr).await?;
                let mut client = FlightClient::from_channel(channel);
                credit_exchange_group_shard(&mut client, name, credits, token, 0, 1).await
            }
            other => other,
        }
    }

    /// Drop a peer's entry once it has proved unreachable, freeing its connections.
    ///
    /// The map is keyed by advertised address and nothing ever removed from it. A
    /// `ClientPool` is process-lifetime (one pooled consumer per worker, shared by every
    /// session), and a worker advertises an *ephemeral* port — so every peer restart, every
    /// autoscaling replacement, and every actor recycle mints a new key. The old entry
    /// stayed forever holding up to `connections_per_peer` tonic channels, each a live
    /// HTTP/2 connection with its own buffers and file descriptor. On a churning cluster
    /// that is an unbounded leak of both memory and fds in the one process that must
    /// outlive every query.
    ///
    /// Eviction is deliberately tied to *proven* unreachability — a connection error whose
    /// redial also failed — rather than to idleness. A peer that is merely quiet between
    /// queries should keep its warm connections: re-establishing them is the cost this pool
    /// exists to avoid, and dropping them would trade a bounded leak for a per-query
    /// handshake. `reset` alone was not enough because it clears the channel vector and
    /// leaves the key.
    async fn forget_unreachable(&self, addr: &str) {
        // Only if it holds no connections: a concurrent fetch to the same peer may already
        // have rebuilt it, and removing a live entry would drop channels those fetches are
        // streaming on. An entry whose `acquire` never managed to connect is empty by
        // construction, which is exactly the case that used to be left behind.
        //
        // The `Arc` is cloned out and the map guard dropped *before* the await. A dashmap
        // `get` holds that shard's lock, which is a synchronous one, and awaiting under it
        // parks the task while every other thread that wants the same shard — every
        // concurrent `peer()` for this address — blocks on it without yielding. With enough
        // concurrent fetches to one dead peer that consumes the whole runtime and the gather
        // never finishes: an unreachable peer is exactly when many fetches fail at once, so
        // the deadlock is likeliest in the case this function exists for.
        let peer = match self.peers.get(addr) {
            Some(entry) => entry.value().clone(),
            None => return,
        };
        if peer.is_empty().await {
            self.peers.remove(addr);
        }
    }

    /// Fetch one bucket from `addr` split across `stripe` interleaved shards fetched
    /// concurrently, each on its own pooled connection — so a *single* large per-peer
    /// transfer runs as `stripe` parallel TCP flows and clears the per-flow NIC cap.
    ///
    /// This is what makes the striping reach Batcher's real reduce path: it runs one
    /// Flight endpoint per node, so a reducer pulls each node's whole bucket over one
    /// stream; sharding turns that one stream into `stripe` flows. `stripe <= 1` is
    /// exactly [`Self::fetch_secured`]. Each shard runs its **own full `credits` window**,
    /// not `credits/stripe`: a shard is an independent TCP flow with its own congestion
    /// window, and on a high-bandwidth-delay-product (cross-region) link that per-flow
    /// cwnd — not the app credit window — is the throughput ceiling. Splitting one window
    /// across the flows starves each to ~1 credit and forfeits the parallelism entirely
    /// (measured: N full-window flows scale aggregate throughput ~N x over a 100 ms-RTT
    /// link — 1 -> 16 flows = 1.9 -> 22 MiB/s — versus flat with a split window). Total
    /// in-flight is therefore `stripe x credits` batches; `stripe` is bounded upstream by
    /// `connections_per_peer` (and the fan-in), and `credits` by Carbonite's byte budget,
    /// so the product stays within the channel envelope. The shards' union is the whole
    /// bucket; within-bucket order is not preserved (the reducer re-orders or commutatively
    /// combines downstream, as it already must across sources). A shard fault propagates so
    /// the gather's recovery layer recomputes the source.
    pub async fn fetch_secured_striped(
        &self,
        addr: &str,
        ticket: &ShuffleTicket,
        credits: u32,
        token: Option<&str>,
        stripe: u32,
    ) -> TransportResult<Vec<RecordBatch>> {
        self.fetch_secured_group_striped(addr, std::slice::from_ref(ticket), credits, token, stripe)
            .await
    }

    /// Fetch a **group** of buckets from `addr` over `stripe` concurrent streams.
    ///
    /// The two knobs the shuffle actually has are the number of buckets a stream carries and
    /// the number of streams a peer is pulled over, and they are the same knob seen from
    /// either end: grouping packs many small buckets into one stream, striping splits one
    /// large bucket across several. Both exist because a stream's throughput is neither the
    /// link's nor the bucket's — it is what one encode pipeline sustains, so a peer wants a
    /// roughly *constant* number of streams whatever the shuffle's shape. Measured at a fixed
    /// 1.4 GiB across one link: 4 buckets moved at 2,928 MiB/s and 4,096 at 3,115, against
    /// 7,176 in the middle where the stream count happened to land right.
    ///
    /// `tickets` must belong to one shuffle stage, so they share a schema; the server rejects
    /// a group that does not rather than reinterpreting one bucket under another's schema.
    /// The union of the shards is the whole group, and order within it is not preserved (the
    /// reducer re-orders or commutatively combines downstream, as it already must across
    /// sources).
    pub async fn fetch_secured_group_striped(
        &self,
        addr: &str,
        tickets: &[ShuffleTicket],
        credits: u32,
        token: Option<&str>,
        stripe: u32,
    ) -> TransportResult<Vec<RecordBatch>> {
        if tickets.is_empty() {
            return Ok(Vec::new());
        }
        let name = group_name(tickets);
        if stripe <= 1 {
            return self.fetch_group(addr, &name, credits, token).await;
        }
        let per_shard = credits.max(1);
        let name = &name;
        let fetches = (0..stripe).map(|shard| async move {
            let channel = self.channel(addr).await?;
            let mut client = FlightClient::from_channel(channel);
            // Per shard, not per bucket. Each shard is its own TCP flow, so its bytes over
            // its own duration is the per-stream rate the striping exists to multiply; timing
            // the whole `try_join_all` instead would divide the bucket by the slowest flow's
            // wall time and report a rate no flow achieved.
            let started = std::time::Instant::now();
            let out =
                credit_exchange_group_shard(&mut client, name, per_shard, token, shard, stripe)
                    .await;
            if let Ok((batches, starved, first)) = out.as_ref() {
                let bytes: usize = batches.iter().map(|b| b.get_array_memory_size()).sum();
                crate::record_fetch(addr, bytes as u64, started.elapsed(), *starved, *first);
            }
            out.map(|(batches, ..)| batches)
        });
        let shards = futures::future::try_join_all(fetches).await?;
        Ok(shards.into_iter().flatten().collect())
    }
}

/// Whether an error is a transport/connection failure (a dead peer / stale
/// channel) worth redialing — i.e. the retryable class (see [`classify`]).
fn is_connection_error(err: &TransportError) -> bool {
    classify(err) == FetchFault::Retryable
}
