//! What each peer actually carried, so a slow shuffle can name the wire it was slow on.
//!
//! A shuffle's duration is the sum of its fetches, and the fetches are not alike. One peer on
//! a renegotiated link, one reducer pulling across a rack, one node whose NIC is shared with
//! a storage daemon: each turns into a stage that takes twice as long, and every figure the
//! shuffle already reports reads the same in all three cases. Locality ratios count *where*
//! the bytes came from and credit windows describe how the producer was paced; neither says
//! how fast any of it moved.
//!
//! The client already knows. It has the bytes it decoded and the time it spent doing it, per
//! peer, and the only reason that was not a measurement is that nobody added the two counters.
//! This is the two counters: bytes and nanoseconds per peer address, plus the fetch and retry
//! counts that say whether a slow peer was slow or merely retried.
//!
//! **Recorded on the consumer, per peer, and never sampled.** The alternative — the node's
//! own port counters — measures the whole machine, including every other tenant, and cannot
//! attribute a rate to the peer it came from. The cost here is two atomic adds per fetch,
//! against a fetch that moved megabytes.
//!
//! # Starvation, and why a throughput counter is not enough
//!
//! One more counter is here for a different reader: not an operator diagnosing a slow node,
//! but the *credit controller* deciding how wide a channel's window should be. A credit
//! window's only job is to cover the channel's bandwidth-delay product — enough batches in
//! flight that the consumer never waits on the wire, and not one more, because every credit
//! past that point is buffered memory bought for no throughput.
//!
//! Nothing measured whether the consumer was waiting. The controller's sole congestion signal
//! was the node's memory pressure, so on a healthy node every round read "uncongested" and the
//! window grew to its ceiling whether or not the extra credits moved a single additional byte.
//! That is not a control loop, it is a ramp, and it is why the memory-pressure signal it backs
//! off on is so often self-inflicted.
//!
//! [`PeerTransfer::starved_nanos`] is the missing measurement: the time the consumer spent
//! blocked awaiting the *next* batch. Near zero means the producer is always ahead and the
//! window already covers the BDP (grow it and you buy buffering, nothing else); a large share
//! of the fetch means the consumer is waiting on the wire and a wider window would fill it.
//! It costs one `Instant::now()` pair per received batch, against a batch that carries a
//! morsel of Arrow data.
//!
//! The registry is process-wide because the pool it accompanies is, and it is reset explicitly
//! rather than per query: a caller that wants one stage's figures takes a baseline, exactly as
//! it does with the fabric counters.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// What one peer carried, as running totals.
///
/// Counters rather than a rate, because a rate cannot be merged: two stages' figures add, and
/// a caller comparing a baseline against a later reading needs the difference of the totals.
#[derive(Debug)]
pub struct PeerTransfer {
    bytes: AtomicU64,
    nanos: AtomicU64,
    fetches: AtomicU64,
    retries: AtomicU64,
    starved: AtomicU64,
    /// Min-filter over observed request-to-first-response times: the path's propagation
    /// delay, BBR's `RTprop`. `u64::MAX` until the first sample.
    rt_prop: AtomicU64,
    /// Max-filter over observed delivery rates in bytes per second: the path's bottleneck
    /// bandwidth, BBR's `BtlBw`. `0` until the first sample.
    btl_bw: AtomicU64,
}

impl PeerTransfer {
    /// Bytes decoded from this peer.
    pub fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }

    /// Nanoseconds spent fetching from this peer, summed across fetches.
    ///
    /// Summed, not elapsed: concurrent fetches to one peer each contribute their own duration,
    /// so this exceeds wall time when a reducer stripes. That is the figure a *per-stream*
    /// rate needs, and [`Self::gbps`] is documented as being one.
    pub fn nanos(&self) -> u64 {
        self.nanos.load(Ordering::Relaxed)
    }

    /// Fetches completed against this peer.
    pub fn fetches(&self) -> u64 {
        self.fetches.load(Ordering::Relaxed)
    }

    /// Fetches that had to be redialed because the cached connection was stale.
    pub fn retries(&self) -> u64 {
        self.retries.load(Ordering::Relaxed)
    }

    /// Nanoseconds the consumer spent blocked awaiting the *next* batch from this peer.
    ///
    /// Summed over the same fetches [`Self::nanos`] covers, so the two divide directly. This
    /// is the credit window's own feedback: a channel that never waits is already wide enough,
    /// and one that waits for most of its fetch is being throttled by the window rather than
    /// by the peer.
    pub fn starved_nanos(&self) -> u64 {
        self.starved.load(Ordering::Relaxed)
    }

    /// The path's propagation delay in nanoseconds — BBR's `RTprop` — or `None` if unsampled.
    ///
    /// A **min**-filter, and that is the whole reason it is trustworthy. Every source of error
    /// in an observed round trip is non-negative: queueing at the switch, a busy tokio worker,
    /// a scheduler slice lost on either end. The true propagation delay is therefore the
    /// smallest thing ever observed, and the minimum is the maximum-likelihood estimate of it
    /// under any non-negative noise distribution — no averaging required, and no assumption
    /// about the noise's shape.
    pub fn rt_prop_nanos(&self) -> Option<u64> {
        match self.rt_prop.load(Ordering::Relaxed) {
            u64::MAX => None,
            n => Some(n),
        }
    }

    /// The path's bottleneck bandwidth in bytes per second — BBR's `BtlBw` — or `None`.
    ///
    /// A **max**-filter, by the mirror of the argument above: a fetch can only ever deliver
    /// *slower* than the bottleneck allows. A short bucket that finished before the window
    /// opened, a consumer that was busy, a producer still assembling — each under-reports and
    /// none can over-report, so the largest rate ever seen is the estimate of the ceiling.
    /// Averaging instead would drag the estimate down by every application-limited fetch,
    /// which on a wide shuffle is most of them.
    pub fn btl_bw_bytes_per_second(&self) -> Option<u64> {
        match self.btl_bw.load(Ordering::Relaxed) {
            0 => None,
            n => Some(n),
        }
    }

    /// Bytes in flight needed to keep this path busy — the bandwidth-delay product.
    ///
    /// `BtlBw x RTprop`, the classical result: a pipe of capacity `BtlBw` and length `RTprop`
    /// holds exactly this much data. Below it the sender idles waiting for permission; above
    /// it the surplus does not move a single byte sooner, it only sits in a buffer somewhere.
    /// That is the quantity a credit window exists to match, and the quantity a loss-based
    /// controller has to discover by overshooting it.
    ///
    /// `None` until both terms are sampled. A caller must read that as "no estimate", never
    /// as a zero-width pipe.
    pub fn bdp_bytes(&self) -> Option<u64> {
        let rate = self.btl_bw_bytes_per_second()?;
        let rtt = self.rt_prop_nanos()?;
        Some(
            ((u128::from(rate) * u128::from(rtt)) / 1_000_000_000).min(u128::from(u64::MAX)) as u64,
        )
    }

    /// Share of this peer's fetch time the consumer spent waiting for data, in `[0, 1]`.
    ///
    /// `None` when nothing has been fetched, which reads as "no opinion": a controller must
    /// not treat an unmeasured channel as saturated and stop growing a window that has never
    /// been tested. Clamped, because the two counters are read separately and a fetch
    /// completing between the loads can otherwise put the ratio marginally above one.
    pub fn starved_ratio(&self) -> Option<f64> {
        let nanos = self.nanos() as f64;
        if nanos <= 0.0 {
            return None;
        }
        Some((self.starved_nanos() as f64 / nanos).clamp(0.0, 1.0))
    }

    /// Per-stream throughput from this peer, in gigabits per second.
    ///
    /// Zero when nothing was fetched or no time was measured, which reads as "no opinion"
    /// rather than as a stalled peer: a caller ranking peers must not put an unmeasured one at
    /// the bottom of the list and send an operator to inspect a healthy node.
    pub fn gbps(&self) -> f64 {
        let nanos = self.nanos() as f64;
        if nanos <= 0.0 {
            return 0.0;
        }
        self.bytes() as f64 * 8.0 / nanos
    }

    /// A flat snapshot: `(bytes, nanos, fetches, retries, starved_nanos)`.
    ///
    /// Read as five separate atomics, so a fetch completing mid-read can land in some of them
    /// and not others. That skew is one fetch's worth on a running total and cannot invert an
    /// ordering; taking a lock per fetch to remove it would cost more than the measurement.
    pub fn snapshot(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.bytes(),
            self.nanos(),
            self.fetches(),
            self.retries(),
            self.starved_nanos(),
        )
    }
}

impl Default for PeerTransfer {
    /// The min-filter starts at `u64::MAX` and the max-filter at `0`, so the first sample of
    /// each wins outright. A derived `Default` would start `rt_prop` at zero, which reads as a
    /// path of no length and would never be replaced by a real measurement.
    fn default() -> Self {
        Self {
            bytes: AtomicU64::new(0),
            nanos: AtomicU64::new(0),
            fetches: AtomicU64::new(0),
            retries: AtomicU64::new(0),
            starved: AtomicU64::new(0),
            rt_prop: AtomicU64::new(u64::MAX),
            btl_bw: AtomicU64::new(0),
        }
    }
}

/// Every peer this process has fetched from, by address.
static PEERS: RwLock<Option<HashMap<String, Arc<PeerTransfer>>>> = RwLock::new(None);

fn entry(addr: &str) -> Arc<PeerTransfer> {
    if let Some(map) = PEERS.read().unwrap_or_else(|e| e.into_inner()).as_ref() {
        if let Some(found) = map.get(addr) {
            return found.clone();
        }
    }
    let mut guard = PEERS.write().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    map.entry(addr.to_string()).or_default().clone()
}

/// Fold one completed fetch into its peer's totals.
///
/// `bytes` is what the batches occupy in memory, which is what the consumer decoded rather
/// than what crossed the wire: with compression on, the wire carried fewer. The decoded figure
/// is the one a caller wants, because it is the volume the stage actually moved and it is
/// comparable across a run whose codec changes.
///
/// `starved` is the part of `elapsed` the consumer spent blocked awaiting the next batch. It
/// is a *component* of `elapsed`, not an addition to it, which is what lets the two divide
/// into the starvation ratio the credit controller reads.
/// `first_response` is the time from issuing the request to the first frame coming back — one
/// full round trip, and the sample the `RTprop` min-filter is built from.
pub fn record_fetch(
    addr: &str,
    bytes: u64,
    elapsed: Duration,
    starved: Duration,
    first_response: Duration,
) {
    let peer = entry(addr);
    peer.bytes.fetch_add(bytes, Ordering::Relaxed);
    peer.nanos.fetch_add(as_nanos(elapsed), Ordering::Relaxed);
    // Never above the fetch it is part of: the two clocks are read at different points, and a
    // ratio above one would read as a channel starved for longer than it existed.
    peer.starved
        .fetch_add(as_nanos(starved).min(as_nanos(elapsed)), Ordering::Relaxed);
    peer.fetches.fetch_add(1, Ordering::Relaxed);
    fold_min(&peer.rt_prop, as_nanos(first_response));
    // Delivery rate for *this* fetch. Only the transfer proper counts: charging the opening
    // round trip to the rate would make every small bucket look like a slow link, and the
    // max-filter would then never see the true ceiling on a shuffle made of small buckets.
    let moving = as_nanos(elapsed).saturating_sub(as_nanos(first_response));
    if moving > 0 && bytes > 0 {
        let rate = (u128::from(bytes) * 1_000_000_000) / u128::from(moving);
        fold_max(&peer.btl_bw, rate.min(u128::from(u64::MAX)) as u64);
    }
}

/// Fold `sample` into a min-filter held in an atomic, without a lock.
///
/// A compare-and-swap loop rather than `fetch_min` so the code reads the same on every
/// toolchain that has ever built this crate. Contention is a fetch's worth, not a batch's.
fn fold_min(cell: &AtomicU64, sample: u64) {
    let mut current = cell.load(Ordering::Relaxed);
    while sample < current {
        match cell.compare_exchange_weak(current, sample, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(seen) => current = seen,
        }
    }
}

/// Fold `sample` into a max-filter held in an atomic. The mirror of [`fold_min`].
fn fold_max(cell: &AtomicU64, sample: u64) {
    let mut current = cell.load(Ordering::Relaxed);
    while sample > current {
        match cell.compare_exchange_weak(current, sample, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(seen) => current = seen,
        }
    }
}

/// A duration in nanoseconds, saturating rather than wrapping. A `Duration` holds more
/// nanoseconds than a `u64` can, and a counter that wraps is worse than one that pins.
fn as_nanos(d: Duration) -> u64 {
    d.as_nanos().min(u128::from(u64::MAX)) as u64
}

/// Note that a fetch to `addr` had to be redialed.
///
/// Kept apart from the fetch counters because a retried fetch measures the *second* attempt:
/// folding the failed one's duration in would attribute a dead connection's timeout to the
/// peer's bandwidth and make a healthy node look like a slow wire.
pub fn record_retry(addr: &str) {
    entry(addr).retries.fetch_add(1, Ordering::Relaxed);
}

/// Every peer's totals, as `(addr, bytes, nanos, fetches, retries, starved_nanos)`, sorted by
/// address.
///
/// Sorted so two readings of an unchanged process compare equal, which is what lets a test
/// assert on the whole list rather than on one entry.
pub fn peer_transfers() -> Vec<(String, u64, u64, u64, u64, u64)> {
    let guard = PEERS.read().unwrap_or_else(|e| e.into_inner());
    let Some(map) = guard.as_ref() else {
        return Vec::new();
    };
    let mut out: Vec<_> = map
        .iter()
        .map(|(addr, stats)| {
            let (bytes, nanos, fetches, retries, starved) = stats.snapshot();
            (addr.clone(), bytes, nanos, fetches, retries, starved)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// This process's running `(starved_nanos, total_nanos)` across every shuffle peer.
///
/// **Totals, not a ratio, and that is the whole point.** A credit controller acts once per
/// round, so what it needs is how the channel behaved *during that round* — and a cumulative
/// ratio cannot say. After a few seconds of a long shuffle the denominator is large enough
/// that a round of pure starvation barely moves it, so a controller reading the lifetime ratio
/// converges to a number and then stops responding to the link entirely. Two counters
/// differenced against the previous reading give the interval, which is the measurement the
/// control law is actually defined over.
///
/// Byte-weighted by construction — a peer that carried more of the shuffle contributed more of
/// both clocks — so a single trivial fetch cannot swing a verdict the way averaging per-peer
/// ratios would.
///
/// `(0, 0)` on a process that has fetched nothing. Callers must read a zero denominator as
/// "no opinion" rather than as a saturated link: a controller that treats an unmeasured
/// channel as saturated stops growing a window it has never tested.
pub fn fleet_flow_totals() -> (u64, u64) {
    let guard = PEERS.read().unwrap_or_else(|e| e.into_inner());
    let Some(map) = guard.as_ref() else {
        return (0, 0);
    };
    let mut nanos = 0u128;
    let mut starved = 0u128;
    for stats in map.values() {
        nanos += u128::from(stats.nanos());
        starved += u128::from(stats.starved_nanos());
    }
    let cap = u128::from(u64::MAX);
    (starved.min(cap) as u64, nanos.min(cap) as u64)
}

/// The largest bandwidth-delay product across every peer, in bytes, or `None` if unsampled.
///
/// **The maximum, because one credit window serves every channel this session fetches on.**
/// A window sized to the median path starves the longest one, and starving a path is the
/// failure that costs throughput; over-sizing merely costs buffer, and the byte ceiling is
/// already there to bound that. So the widest pipe sets the target and Carbonite's envelope
/// clamps it — the two constraints meeting is exactly the design, rather than one guessing at
/// the other.
pub fn fleet_bdp_bytes() -> Option<u64> {
    let guard = PEERS.read().unwrap_or_else(|e| e.into_inner());
    let map = guard.as_ref()?;
    map.values().filter_map(|s| s.bdp_bytes()).max()
}

/// The share of *all* fetch time this process has spent waiting for data, in `[0, 1]`.
///
/// The lifetime reading, for a diagnosis rather than for a control loop — see
/// [`fleet_flow_totals`] for why a controller must difference the totals instead.
///
/// `None` when nothing has been fetched.
pub fn fleet_starved_ratio() -> Option<f64> {
    let (starved, nanos) = fleet_flow_totals();
    if nanos == 0 {
        return None;
    }
    Some((starved as f64 / nanos as f64).clamp(0.0, 1.0))
}

/// The address of the slowest peer that moved a measurable amount, and its rate in Gb/s.
///
/// The straggler question, answered directly. `None` when no peer has both bytes and time
/// recorded — an unmeasured fleet has no slowest member, and naming one from a single fetch's
/// noise is how a healthy node gets drained.
pub fn slowest_peer(min_bytes: u64) -> Option<(String, f64)> {
    let guard = PEERS.read().unwrap_or_else(|e| e.into_inner());
    let map = guard.as_ref()?;
    map.iter()
        .filter(|(_, stats)| stats.bytes() >= min_bytes && stats.nanos() > 0)
        .map(|(addr, stats)| (addr.clone(), stats.gbps()))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}

/// Forget every peer's totals.
///
/// For a caller that measures one stage rather than the process. Resetting is coarser than a
/// baseline and is offered anyway, because the common case is one shuffle per process and a
/// baseline of zeros is what it wants.
pub fn reset_peer_transfers() {
    *PEERS.write().unwrap_or_else(|e| e.into_inner()) = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The registry is process-wide and the test harness runs these on threads of one
    /// process, so without a lock each test observes the others' fetches. Held for the body
    /// of every case, which is what makes the assertions about *totals* mean anything.
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn fresh() -> std::sync::MutexGuard<'static, ()> {
        let guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_peer_transfers();
        guard
    }

    #[test]
    fn a_fetch_is_folded_into_its_peers_totals() {
        let _guard = fresh();
        record_fetch(
            "h:1",
            1_000,
            Duration::from_millis(1),
            Duration::ZERO,
            Duration::ZERO,
        );
        record_fetch(
            "h:1",
            2_000,
            Duration::from_millis(1),
            Duration::ZERO,
            Duration::ZERO,
        );
        let stats = peer_transfers();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].0, "h:1");
        assert_eq!(stats[0].1, 3_000);
        assert_eq!(stats[0].3, 2);
    }

    #[test]
    fn peers_are_kept_apart() {
        let _guard = fresh();
        record_fetch(
            "h:1",
            1_000,
            Duration::from_millis(1),
            Duration::ZERO,
            Duration::ZERO,
        );
        record_fetch(
            "h:2",
            5_000,
            Duration::from_millis(1),
            Duration::ZERO,
            Duration::ZERO,
        );
        let stats = peer_transfers();
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].0, "h:1");
        assert_eq!(stats[1].1, 5_000);
    }

    #[test]
    fn a_retry_does_not_count_as_a_fetch() {
        let _guard = fresh();
        record_retry("h:1");
        let stats = peer_transfers();
        assert_eq!(stats[0].3, 0, "no fetch completed");
        assert_eq!(stats[0].4, 1, "one redial");
    }

    #[test]
    fn an_unmeasured_peer_reports_no_rate_rather_than_zero_bandwidth() {
        let _guard = fresh();
        record_retry("h:1");
        assert!(slowest_peer(0).is_none());
    }

    #[test]
    fn the_slowest_peer_is_the_one_worth_naming() {
        let _guard = fresh();
        record_fetch(
            "fast:1",
            8_000_000,
            Duration::from_millis(1),
            Duration::ZERO,
            Duration::ZERO,
        );
        record_fetch(
            "slow:1",
            8_000,
            Duration::from_millis(1),
            Duration::ZERO,
            Duration::ZERO,
        );
        let (addr, gbps) = slowest_peer(1_000).expect("both peers measured");
        assert_eq!(addr, "slow:1");
        assert!(gbps > 0.0);
    }

    #[test]
    fn a_peer_below_the_byte_floor_is_not_named() {
        let _guard = fresh();
        record_fetch(
            "tiny:1",
            8,
            Duration::from_millis(50),
            Duration::ZERO,
            Duration::ZERO,
        );
        record_fetch(
            "real:1",
            8_000_000,
            Duration::from_millis(1),
            Duration::ZERO,
            Duration::ZERO,
        );
        let (addr, _) = slowest_peer(1_000).expect("one peer clears the floor");
        assert_eq!(
            addr, "real:1",
            "a single tiny fetch is noise, not a straggler"
        );
    }

    #[test]
    fn starvation_is_the_share_of_the_fetch_spent_waiting() {
        let _guard = fresh();
        record_fetch(
            "h:1",
            1_000,
            Duration::from_millis(10),
            Duration::from_millis(4),
            Duration::ZERO,
        );
        let ratio = entry("h:1").starved_ratio().expect("measured");
        assert!((ratio - 0.4).abs() < 1e-6);
    }

    #[test]
    fn an_unmeasured_channel_has_no_starvation_opinion() {
        let _guard = fresh();
        record_retry("h:1");
        assert!(
            entry("h:1").starved_ratio().is_none(),
            "a controller must not read an untested channel as saturated"
        );
        assert!(fleet_starved_ratio().is_none());
    }

    #[test]
    fn starvation_can_never_exceed_the_fetch_it_is_part_of() {
        let _guard = fresh();
        // The two clocks are read at different points, so a pathological pair must clamp
        // rather than report a channel starved for longer than it existed.
        record_fetch(
            "h:1",
            1_000,
            Duration::from_millis(1),
            Duration::from_secs(10),
            Duration::ZERO,
        );
        assert_eq!(entry("h:1").starved_ratio(), Some(1.0));
    }

    #[test]
    fn the_fleet_ratio_is_byte_weighted_by_construction() {
        let _guard = fresh();
        // A long, fully starved fetch and a trivial saturated one. Averaging the per-peer
        // ratios would call the fleet half-starved; summing the clocks calls it starved,
        // which is what the wire actually did.
        record_fetch(
            "slow:1",
            1_000,
            Duration::from_millis(99),
            Duration::from_millis(99),
            Duration::ZERO,
        );
        record_fetch(
            "fast:1",
            1_000,
            Duration::from_millis(1),
            Duration::ZERO,
            Duration::ZERO,
        );
        let ratio = fleet_starved_ratio().expect("measured");
        assert!(ratio > 0.9, "byte-weighted, not per-peer averaged: {ratio}");
    }

    #[test]
    fn the_delay_estimate_is_the_minimum_ever_seen() {
        let _guard = fresh();
        // Every source of error in an observed round trip is non-negative -- queueing, a busy
        // worker, a lost scheduler slice -- so the truth is the smallest sample, and an
        // average would be biased upward by every one of them.
        for rtt in [50u64, 12, 200, 31] {
            record_fetch(
                "h:1",
                1_000,
                Duration::from_millis(500),
                Duration::ZERO,
                Duration::from_millis(rtt),
            );
        }
        assert_eq!(
            entry("h:1").rt_prop_nanos(),
            Some(Duration::from_millis(12).as_nanos() as u64)
        );
    }

    #[test]
    fn the_bandwidth_estimate_is_the_maximum_ever_seen() {
        let _guard = fresh();
        // A fetch can only ever deliver slower than the bottleneck allows, never faster, so
        // the ceiling is the largest rate observed. Here: 1 MB in 1 s, then 8 MB in 1 s.
        record_fetch(
            "h:1",
            1_000_000,
            Duration::from_secs(1),
            Duration::ZERO,
            Duration::ZERO,
        );
        record_fetch(
            "h:1",
            8_000_000,
            Duration::from_secs(1),
            Duration::ZERO,
            Duration::ZERO,
        );
        let bw = entry("h:1").btl_bw_bytes_per_second().expect("measured");
        assert!(
            (7_900_000..=8_100_000).contains(&bw),
            "the max-filter must hold the fast fetch, got {bw}"
        );
    }

    #[test]
    fn an_application_limited_fetch_cannot_drag_the_bandwidth_estimate_down() {
        let _guard = fresh();
        record_fetch(
            "h:1",
            8_000_000,
            Duration::from_secs(1),
            Duration::ZERO,
            Duration::ZERO,
        );
        // A tiny bucket that finished before the window ever opened. On a wide shuffle most
        // fetches look like this, so averaging would put the estimate near zero.
        for _ in 0..50 {
            record_fetch(
                "h:1",
                8,
                Duration::from_secs(1),
                Duration::ZERO,
                Duration::ZERO,
            );
        }
        let bw = entry("h:1").btl_bw_bytes_per_second().expect("measured");
        assert!(
            bw > 7_000_000,
            "fifty small fetches eroded the ceiling: {bw}"
        );
    }

    #[test]
    fn the_bandwidth_delay_product_is_the_pipe_it_describes() {
        let _guard = fresh();
        // 100 MB/s over a 20 ms path holds 2 MB in flight.
        record_fetch(
            "h:1",
            100_000_000,
            Duration::from_millis(1020),
            Duration::ZERO,
            Duration::from_millis(20),
        );
        let bdp = entry("h:1").bdp_bytes().expect("both terms sampled");
        assert!(
            (1_900_000..=2_100_000).contains(&bdp),
            "BtlBw x RTprop should be ~2 MB, got {bdp}"
        );
        assert_eq!(fleet_bdp_bytes(), Some(bdp));
    }

    #[test]
    fn one_missing_term_leaves_no_bandwidth_delay_product() {
        let _guard = fresh();
        // A retry samples neither, and a controller must read that as "no estimate" rather
        // than as a pipe of zero width, which would pin every window to its floor.
        record_retry("h:1");
        assert!(entry("h:1").bdp_bytes().is_none());
        assert!(fleet_bdp_bytes().is_none());
    }

    #[test]
    fn the_widest_path_sets_the_fleet_estimate() {
        let _guard = fresh();
        // One credit window serves every channel, so the target is the pipe that would
        // otherwise starve -- the largest -- not the typical one.
        record_fetch(
            "near:1",
            10_000_000,
            Duration::from_millis(1001),
            Duration::ZERO,
            Duration::from_millis(1),
        );
        record_fetch(
            "far:1",
            10_000_000,
            Duration::from_millis(1100),
            Duration::ZERO,
            Duration::from_millis(100),
        );
        let far = entry("far:1").bdp_bytes().expect("measured");
        assert_eq!(fleet_bdp_bytes(), Some(far));
    }

    #[test]
    fn a_reset_forgets_everything() {
        let _guard = fresh();
        record_fetch(
            "h:1",
            1_000,
            Duration::from_millis(1),
            Duration::ZERO,
            Duration::ZERO,
        );
        reset_peer_transfers();
        assert!(peer_transfers().is_empty());
    }

    #[test]
    fn the_rate_is_bits_over_seconds() {
        let _guard = fresh();
        // 1 GB in 1 second is 8 Gb/s.
        record_fetch(
            "h:1",
            1_000_000_000,
            Duration::from_secs(1),
            Duration::ZERO,
            Duration::ZERO,
        );
        let stats = peer_transfers();
        let (_, bytes, nanos, _, _, _) = &stats[0];
        assert_eq!(*bytes, 1_000_000_000);
        assert_eq!(*nanos, 1_000_000_000);
        assert!((entry("h:1").gbps() - 8.0).abs() < 1e-6);
    }
}
