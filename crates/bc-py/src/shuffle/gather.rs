//! The reducer's gather: how a worker pulls its bucket from every mapper.
//!
//! A reducer must fetch its share from *every* mapper, and how those fetches are packed
//! onto Flight streams is what decides the rate. A hash shuffle cuts one bucket per
//! reducer out of each mapper's output, so a cluster of `W` workers makes `W^2` buckets
//! and every one of them shrinks as the cluster grows. Fetching a bucket per stream made
//! the transfer's shape an accident of the cluster's width and cost most of the link, so
//! the stream count is held fixed here and buckets are packed into streams by *bytes* —
//! see [`drive`], which also carries the measurements.
//!
//! Separate from `shuffle`, which is the mapper's half (the partitioners): the two share
//! only the ticket vocabulary, and each is a full responsibility on its own.

use std::sync::Arc;

use arrow::array::RecordBatch;
use bc_interp::InterpError;
use bc_transport::{classify, FetchFault, ShuffleTicket, TransportError};
use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;
use tokio::task::JoinSet;

use crate::errors::transport_to_pyerr;
use crate::flight::FlightShuffleServer;
use crate::to_pyerr;

/// A reducer fetch failure that must surface as a Python exception once the GIL is
/// re-acquired (a `PyErr` cannot be built while the GIL is released inside the runtime).
pub(crate) enum GatherErr {
    /// A fatal transport fault (decode/protocol/auth) — fail the query fast.
    Fatal(TransportError),
    /// A combine/finalize error over the fetched partials.
    Combine(InterpError),
    /// A fetch task panicked or was cancelled.
    Join(String),
}

impl GatherErr {
    pub(crate) fn into_pyerr(self) -> PyErr {
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

/// One remote source as a fetch task sees it: which gather slot it answers, which bucket
/// to ask for, and every address that holds a copy.
struct RemoteSource {
    idx: usize,
    ticket: ShuffleTicket,
    /// The primary first, then its replicas. Non-empty by construction.
    candidates: Vec<String>,
}

/// What a fetch task hands back to the driver, which owns `on_batches`.
enum Fetched {
    Batches(Vec<RecordBatch>),
    Unreachable(usize, String),
    Fatal(TransportError),
}

/// Running bytes-per-bucket for one peer, so the grouping adapts to the shuffle's shape
/// instead of guessing it.
#[derive(Default)]
struct BucketSize {
    bytes: std::sync::atomic::AtomicU64,
    buckets: std::sync::atomic::AtomicU64,
}

impl BucketSize {
    fn observe(&self, bytes: u64, buckets: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.bytes.fetch_add(bytes, Relaxed);
        self.buckets.fetch_add(buckets, Relaxed);
    }

    /// How many buckets to ask for on one stream, or `1` until something has been measured.
    ///
    /// A stream's cost is amortized over what it carries, so the group is chosen in *bytes*:
    /// this is the one figure that makes the transfer's rate independent of how finely the
    /// shuffle happens to be cut. It is also the gather's memory bound — in-flight bytes are
    /// `streams x target` — which the bucket-count bound it replaces stopped being once
    /// buckets shrank below a megabyte.
    fn group_of(&self, target_bytes: u64) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let buckets = self.buckets.load(Relaxed);
        if buckets == 0 {
            return 1;
        }
        let avg = (self.bytes.load(Relaxed) / buckets).max(1);
        (target_bytes / avg).clamp(1, MAX_GROUP as u64) as usize
    }
}

/// Most buckets one stream will ever carry, however small they are.
///
/// A bound on the *name* as much as on the memory: the group travels as a comma-joined
/// ticket list in the request's descriptor path, and a gRPC header has a size limit.
const MAX_GROUP: usize = 256;

/// Fetch every source concurrently, invoking `on_batches` for each non-empty result
/// as it arrives; returns the sources that hit a *retryable* fault, each with the
/// message of the fault that made it retryable.
///
/// The message travels because the index alone is a lie by omission. A reducer that
/// cannot reach a mapper reports "unreachable worker", the driver recomputes, and after
/// `recovery_max_attempts` the query fails with a worker-loss error — on a cluster where
/// every worker is alive. The cause is then three frames and one wrong noun away from
/// whatever actually broke, which is how a ticket collision here spent hours looking like
/// a fleet problem.
///
/// Co-located sources (`addr == own_addr`) read the local store with no socket.
///
/// # How the remote fetches are shaped, and why it is not one per bucket
///
/// A hash shuffle cuts every mapper's output into one bucket per reducer, so a cluster of
/// `W` workers produces `W^2` buckets and each one shrinks as the cluster grows. Fetching
/// one bucket per stream therefore made the transfer's shape an accident of the cluster's
/// width, and it cost most of the link. Measured across one 25 Gbps link, 1.4 GiB moved at
/// the shipped defaults with only the bucket count varying: 4 buckets 2,854 MiB/s, 16
/// 5,342, 64 4,785, 256 3,888, 1,024 3,118, 4,096 **1,608** — against 7,470 MiB/s at the
/// same total when the stream count happened to land right. The right-hand fall is the one
/// a real cluster slides down as it scales.
///
/// So the streams are the thing held fixed. `bc_transport::gather_streams` of them run
/// across every peer, split evenly between the peers this gather actually has, whatever
/// they are serving: when a peer holds more buckets than its share of streams, each stream
/// carries a *group* of them, sized in bytes from what the first fetches measured; when it
/// holds fewer, one bucket is split across streams by the existing shard selector. The
/// budget in `bc_transport::gather_inflight_bytes`, divided by the streams that will run,
/// is what sizes those groups — so the gather's footprint is a byte figure rather than the
/// bucket count `fan_in` used to bound, which meant nothing once a bucket was a fraction of
/// a megabyte. `fan_in` still raises the stream count when a caller asks for more
/// concurrency than the target gives.
///
/// # Faults
///
/// `replicas[i]` holds the *fallback* addresses for source `i` — peers carrying a
/// byte-identical copy of that bucket under the same ticket (see the replication factor
/// in `DistributedConfig`). A retryable fault against a group is retried **bucket by
/// bucket**, so a lost peer costs a re-fetch from a survivor rather than the lineage
/// recompute (re-read the source, re-run the map) it would otherwise force, and a source is
/// still reported unreachable only once *every* copy of it is gone. Grouping therefore
/// changes how buckets are packed onto streams and nothing about which sources the driver
/// is told to recompute. A fatal fault (decode/protocol/auth) aborts.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn drive(
    own: &FlightShuffleServer,
    pool: Arc<bc_transport::ClientPool>,
    sources: &[(String, ShuffleTicket)],
    replicas: &[Vec<String>],
    credits: u32,
    fan_in: usize,
    token: Option<String>,
    shm: bool,
    mut on_batches: impl FnMut(Vec<RecordBatch>) -> Result<(), InterpError>,
) -> Result<Vec<(usize, String)>, GatherErr> {
    let mut unreachable: Vec<(usize, String)> = Vec::new();

    // Co-located buckets first — a cheap in-process read, no network, no stream.
    let own_addr = own.exchange.advertised_addr();
    let own_host = host_of(own_addr).to_string();
    let mut by_peer: std::collections::HashMap<String, Vec<RemoteSource>> =
        std::collections::HashMap::new();
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
            .map(|c| (*c).to_string())
            .collect();
        if remote.is_empty() {
            continue; // only copy is a local one that read back empty (unchanged behavior)
        }
        by_peer
            .entry(remote[0].clone())
            .or_default()
            .push(RemoteSource {
                idx,
                ticket: *ticket,
                candidates: remote,
            });
    }
    if by_peer.is_empty() {
        return Ok(unreachable);
    }

    // The gather's stream count is a cluster total, split across the peers it actually has:
    // a single link gets many streams and a wide fan-in one each, which is right in both
    // cases because a wide fan-in is already stream-parallel across its peers. `fan_in`
    // still raises it when a caller asks for more concurrency than the target gives.
    let peers = by_peer.len();
    let streams_total = bc_transport::gather_streams().max(fan_in.max(1));
    let target = streams_total.div_ceil(peers).max(1);
    // Bytes per stream come from the in-flight budget divided by the streams that will run,
    // so the gather's footprint is the budget whatever the cluster's width.
    let group_bytes = bc_transport::stream_bytes_for(target * peers);

    // Results come back through a channel because `on_batches` folds into one running state
    // and must stay on this task; the fetches themselves run concurrently.
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Fetched>(peers * target + 1);
    let mut set: JoinSet<()> = JoinSet::new();
    for (addr, mut work) in by_peer {
        let streams = target.min(work.len()).max(1);
        // Fewer buckets than streams ⇒ split each bucket across `stripe` shards, which is
        // the same knob seen from the other end (see `fetch_secured_group_striped`).
        let stripe = (target / work.len().max(1)).clamp(1, target) as u32;
        let sizes = Arc::new(BucketSize::default());
        // One contiguous slice of the peer's buckets per stream. Contiguous rather than
        // round-robin so a group's tickets stay adjacent, which keeps the joined name short.
        let per = work.len().div_ceil(streams);
        for _ in 0..streams {
            let rest = work.split_off(work.len().min(per));
            let mine = std::mem::replace(&mut work, rest);
            if mine.is_empty() {
                continue;
            }
            let (pool, token, addr, sizes, tx) = (
                pool.clone(),
                token.clone(),
                addr.clone(),
                sizes.clone(),
                tx.clone(),
            );
            let own_host = own_host.clone();
            set.spawn(async move {
                let mut queue = mine.into_iter().collect::<std::collections::VecDeque<_>>();
                while !queue.is_empty() {
                    let take = sizes.group_of(group_bytes).min(queue.len());
                    let group: Vec<RemoteSource> = queue.drain(..take).collect();
                    let out = fetch_group(
                        &pool,
                        &addr,
                        &group,
                        credits,
                        token.as_deref(),
                        stripe,
                        shm,
                        &own_host,
                    )
                    .await;
                    let got = match out {
                        Ok(got) => got,
                        Err(e) => {
                            let _ = tx.send(Fetched::Fatal(e)).await;
                            return;
                        }
                    };
                    let bytes: u64 = got
                        .batches
                        .iter()
                        .map(|b| b.get_array_memory_size() as u64)
                        .sum();
                    let delivered = (group.len() - got.lost.len()) as u64;
                    if delivered > 0 {
                        sizes.observe(bytes, delivered);
                    }
                    if !got.batches.is_empty()
                        && tx.send(Fetched::Batches(got.batches)).await.is_err()
                    {
                        return;
                    }
                    for (idx, msg) in got.lost {
                        if tx.send(Fetched::Unreachable(idx, msg)).await.is_err() {
                            return;
                        }
                    }
                }
            });
        }
    }
    drop(tx); // the loop below ends when every task has finished and dropped its sender

    while let Some(msg) = rx.recv().await {
        match msg {
            Fetched::Batches(batches) => on_batches(batches).map_err(GatherErr::Combine)?,
            Fetched::Unreachable(idx, m) => unreachable.push((idx, m)),
            Fetched::Fatal(e) => return Err(GatherErr::Fatal(e)),
        }
    }
    while let Some(joined) = set.join_next().await {
        joined.map_err(|e| GatherErr::Join(e.to_string()))?;
    }
    unreachable.sort_unstable();
    Ok(unreachable)
}

/// What one grouped fetch produced: the buckets that arrived, and the sources that did not.
///
/// Both halves, because a partial failure is the ordinary case for a group and dropping
/// either one is a wrong answer. Discarding the batches that *did* arrive without naming
/// their sources loses them silently — the driver recomputes only what `lost` names — and a
/// per-source fetch gave that pairing for free, so a grouped one has to reproduce it.
struct GroupOutcome {
    batches: Vec<RecordBatch>,
    lost: Vec<(usize, String)>,
}

/// How a single bucket's fetch failed once every copy of it had been tried.
enum OneFault {
    /// Decode/protocol/auth: every retry would hit it identically, so fail the query.
    Fatal(TransportError),
    /// Every copy was unreachable — the source the driver must recompute.
    Lost(String),
}

/// Pull one group of buckets from `addr`, falling back to one bucket at a time (and to
/// each bucket's replicas) if the group cannot be served.
///
/// The group is the fast path and the per-bucket walk is the recovery path, so a fault costs
/// at most one wasted grouped attempt and never changes which sources the driver is told to
/// recompute. A group of one is exactly the un-grouped fetch. The `Err` arm is reserved for a
/// fatal fault; a retryable one is reported per source in [`GroupOutcome::lost`].
#[allow(clippy::too_many_arguments)]
async fn fetch_group(
    pool: &bc_transport::ClientPool,
    addr: &str,
    group: &[RemoteSource],
    credits: u32,
    token: Option<&str>,
    stripe: u32,
    shm: bool,
    own_host: &str,
) -> Result<GroupOutcome, TransportError> {
    let mut out = GroupOutcome {
        batches: Vec::new(),
        lost: Vec::new(),
    };
    // Same node, different process: a zero-copy shared-memory mmap read beats a loopback
    // Flight hop by ~20x, and it is per bucket rather than per stream — so a same-host peer
    // reads bucket by bucket and never groups. Falling back to Flight on a miss is benign and
    // result-preserving: the producer may not have mirrored this bucket (shm off, or skipped
    // under memory pressure).
    let same_host = shm && host_of(addr) == own_host;
    if !same_host && group.len() > 1 {
        let tickets: Vec<ShuffleTicket> = group.iter().map(|s| s.ticket).collect();
        match pool
            .fetch_secured_group_striped(addr, &tickets, credits, token, stripe)
            .await
        {
            Ok(batches) => {
                out.batches = batches;
                return Ok(out);
            }
            Err(e) if matches!(classify(&e), FetchFault::Fatal) => return Err(e),
            // Retryable: fall through to the per-bucket walk, which is what knows about
            // replicas and about which *source* to report.
            Err(_) => {}
        }
    }
    for src in group {
        if same_host {
            let (a, t) = (addr.to_string(), src.ticket.to_string());
            // An shm read is blocking file I/O + decode → off the async reactor.
            if let Ok(Ok(Some(batches))) =
                tokio::task::spawn_blocking(move || bc_transport::fetch_shared(&a, &t)).await
            {
                out.batches.extend(batches);
                continue;
            }
        }
        match fetch_one(pool, src, credits, token, stripe).await {
            Ok(batches) => out.batches.extend(batches),
            Err(OneFault::Fatal(e)) => return Err(e),
            Err(OneFault::Lost(msg)) => out.lost.push((src.idx, msg)),
        }
    }
    Ok(out)
}

/// Pull one bucket, trying each address that holds a copy in turn.
async fn fetch_one(
    pool: &bc_transport::ClientPool,
    src: &RemoteSource,
    credits: u32,
    token: Option<&str>,
    stripe: u32,
) -> Result<Vec<RecordBatch>, OneFault> {
    let mut last: Option<TransportError> = None;
    for addr in &src.candidates {
        match pool
            .fetch_secured_striped(addr, &src.ticket, credits, token, stripe)
            .await
        {
            Ok(batches) => return Ok(batches),
            Err(e) if matches!(classify(&e), FetchFault::Fatal) => return Err(OneFault::Fatal(e)),
            Err(e) => last = Some(e),
        }
    }
    Err(OneFault::Lost(
        last.expect("candidates is non-empty").to_string(),
    ))
}
