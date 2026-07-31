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
#[derive(Debug, Default)]
pub struct PeerTransfer {
    bytes: AtomicU64,
    nanos: AtomicU64,
    fetches: AtomicU64,
    retries: AtomicU64,
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

    /// A flat snapshot: `(bytes, nanos, fetches, retries)`.
    ///
    /// Read as four separate atomics, so a fetch completing mid-read can land in some of them
    /// and not others. That skew is one fetch's worth on a running total and cannot invert an
    /// ordering; taking a lock per fetch to remove it would cost more than the measurement.
    pub fn snapshot(&self) -> (u64, u64, u64, u64) {
        (self.bytes(), self.nanos(), self.fetches(), self.retries())
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
pub fn record_fetch(addr: &str, bytes: u64, elapsed: Duration) {
    let peer = entry(addr);
    peer.bytes.fetch_add(bytes, Ordering::Relaxed);
    peer.nanos.fetch_add(
        elapsed.as_nanos().min(u128::from(u64::MAX)) as u64,
        Ordering::Relaxed,
    );
    peer.fetches.fetch_add(1, Ordering::Relaxed);
}

/// Note that a fetch to `addr` had to be redialed.
///
/// Kept apart from the fetch counters because a retried fetch measures the *second* attempt:
/// folding the failed one's duration in would attribute a dead connection's timeout to the
/// peer's bandwidth and make a healthy node look like a slow wire.
pub fn record_retry(addr: &str) {
    entry(addr).retries.fetch_add(1, Ordering::Relaxed);
}

/// Every peer's totals, as `(addr, bytes, nanos, fetches, retries)`, sorted by address.
///
/// Sorted so two readings of an unchanged process compare equal, which is what lets a test
/// assert on the whole list rather than on one entry.
pub fn peer_transfers() -> Vec<(String, u64, u64, u64, u64)> {
    let guard = PEERS.read().unwrap_or_else(|e| e.into_inner());
    let Some(map) = guard.as_ref() else {
        return Vec::new();
    };
    let mut out: Vec<_> = map
        .iter()
        .map(|(addr, stats)| {
            let (bytes, nanos, fetches, retries) = stats.snapshot();
            (addr.clone(), bytes, nanos, fetches, retries)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
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
        record_fetch("h:1", 1_000, Duration::from_millis(1));
        record_fetch("h:1", 2_000, Duration::from_millis(1));
        let stats = peer_transfers();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].0, "h:1");
        assert_eq!(stats[0].1, 3_000);
        assert_eq!(stats[0].3, 2);
    }

    #[test]
    fn peers_are_kept_apart() {
        let _guard = fresh();
        record_fetch("h:1", 1_000, Duration::from_millis(1));
        record_fetch("h:2", 5_000, Duration::from_millis(1));
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
        record_fetch("fast:1", 8_000_000, Duration::from_millis(1));
        record_fetch("slow:1", 8_000, Duration::from_millis(1));
        let (addr, gbps) = slowest_peer(1_000).expect("both peers measured");
        assert_eq!(addr, "slow:1");
        assert!(gbps > 0.0);
    }

    #[test]
    fn a_peer_below_the_byte_floor_is_not_named() {
        let _guard = fresh();
        record_fetch("tiny:1", 8, Duration::from_millis(50));
        record_fetch("real:1", 8_000_000, Duration::from_millis(1));
        let (addr, _) = slowest_peer(1_000).expect("one peer clears the floor");
        assert_eq!(
            addr, "real:1",
            "a single tiny fetch is noise, not a straggler"
        );
    }

    #[test]
    fn a_reset_forgets_everything() {
        let _guard = fresh();
        record_fetch("h:1", 1_000, Duration::from_millis(1));
        reset_peer_transfers();
        assert!(peer_transfers().is_empty());
    }

    #[test]
    fn the_rate_is_bits_over_seconds() {
        let _guard = fresh();
        // 1 GB in 1 second is 8 Gb/s.
        record_fetch("h:1", 1_000_000_000, Duration::from_secs(1));
        let stats = peer_transfers();
        let (_, bytes, nanos, _, _) = &stats[0];
        assert_eq!(*bytes, 1_000_000_000);
        assert_eq!(*nanos, 1_000_000_000);
        assert!((entry("h:1").gbps() - 8.0).abs() < 1e-6);
    }
}
