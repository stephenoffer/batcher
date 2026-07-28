//! Process-wide memory accounting for reserve-before-allocate.
//!
//! This is Carbonite's enforcement primitive in the data plane: a single shared
//! [`MemoryPool`] that bounds how much memory the engine's stateful operators and
//! its shuffle transfer may hold at once. Callers reserve *before* they allocate;
//! a reservation that would push the pool past its limit fails, so the caller
//! spills (or back-pressures) instead of OOMing. One pool serves the whole
//! process so single-node operators and the transfer layer draw on one envelope.
//!
//! The design follows Apache DataFusion's `MemoryPool` / `MemoryReservation`
//! (a greedy pool with RAII reservations) — adopted rather than re-derived — but
//! kept dependency-light (std + `thiserror` only) so it can sit at the bottom of
//! the crate DAG (`bc-arrow`-level, depended on by `bc-runtime` / `bc-transport`).
//!
//! Carbonite (the Python control plane) sets the limit from its memory envelope
//! (soft 85% / hard 90% of the budget) and drives the pool through `bc-py`; the
//! pool itself is policy-free — it only accounts and admits.

pub mod cancel;

pub use cancel::CancelToken;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use thiserror::Error;

/// Default soft-pressure line as basis points of the limit (8000 = 80%). Above it
/// the pool reports [`Pressure::Elevated`] so the executor can spill *proactively*
/// instead of waiting for the hard cap to stall the process. Carbonite overrides it
/// from its memory envelope via [`MemoryPool::set_soft_fraction`].
const DEFAULT_SOFT_BPS: usize = 8000;

/// Hard cap on cooperative-spill rounds inside one reservation attempt.
///
/// The loop's termination argument is "every round must free something", which holds only
/// if a consumer's reported freed bytes actually come back to *this* pool. A consumer that
/// frees memory it never charged here — a bug, or an operator whose spill releases an
/// off-pool buffer — reports progress forever while `available()` never moves, and the
/// reserving thread spins inside the pool with no timeout and no way out. Bounding the
/// rounds turns that from a hang into a failed reservation, which every caller already
/// handles (it is the signal to spill itself).
const MAX_SPILL_ROUNDS: usize = 32;

/// Coarse memory-pressure level derived from `used / limit`. The one signal the
/// executor's backpressure mechanisms (proactive spill, the morsel-admission gate,
/// the distributed credit window) all read, so single-node and distributed throttle
/// off the same envelope rather than each inventing a threshold.
///
/// Ordered, so a caller can write `pressure >= Pressure::Elevated` the way the control
/// plane's `PressureLevel` is compared. Without it every consumer of this enum had to
/// spell out a match over all three variants to ask "is it at least this bad", which is
/// the kind of thing that is written correctly the first time and wrongly the fourth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Pressure {
    /// Below the soft line — the fast path, no throttling.
    Nominal,
    /// At/above the soft line but below the hard cap — spill proactively / narrow
    /// the in-flight window before the cap forces a stall.
    Elevated,
    /// At/above the hard cap — no headroom; a new reservation can only succeed after
    /// something spills.
    Critical,
}

/// A registered memory consumer that can spill some in-memory state to disk on
/// demand, returning the bytes it freed.
///
/// This is the cooperative-spilling counterpart to reserve-before-allocate
/// (Spark's `TaskMemoryManager` / `MemoryConsumer` model): when a reservation can't
/// be granted, the pool asks the **largest *other* consumer** to spill rather than
/// failing the requester — so a small operator no longer dies while a large
/// neighbour (or a concurrent query) sits on the budget. A consumer that cannot
/// spill simply does not register and is never asked.
///
/// **One occupant so far, and it frees a different currency than it is asked for.** The
/// published shuffle store registers (`bc-py`'s `ShuffleSpiller`, over
/// `bc-transport`'s `PartitionStore`), which makes the cooperative path do real work: it
/// writes idle finished buckets to disk and hands their memory back to the process.
///
/// What it does *not* do is make the pending reservation fit. Those buckets were never
/// reserved here — a mapper hands one over and it stays resident until a reducer collects
/// it, which is precisely why they are the memory a reservation is most often losing to —
/// so spilling them does not move `available()`. The progress check below sees that, stops
/// after one round, and returns the failure. The caller is then refused, as it would have
/// been, but into a process with room for its own spill path to run. Treat that as the
/// honest reading of what the cooperative path buys today; the reservation only starts to
/// *succeed* once a consumer's memory is charged to this pool in the first place.
///
/// The operators that would close that gap are the ones already owning spillable state
/// they reserve for: the aggregate's hash table (`bc-runtime::agg::spill`) and the external
/// sort's runs. Both are mid-`par_iter` state, which is why neither is here yet.
pub trait Spillable: Send + Sync {
    /// Spill at least `target` bytes of in-memory state to disk if possible,
    /// returning the bytes actually freed (`0` if it cannot spill right now).
    ///
    /// MUST only *release* memory, never reserve — the pool may call this while
    /// resolving another reservation, so re-entering the pool from here would
    /// deadlock or recurse. It is invoked outside the pool's registry lock.
    fn spill(&self, target: usize) -> usize;

    /// This consumer's current spillable footprint in bytes — the pool spills the
    /// largest first, so this orders the victims.
    fn spillable_bytes(&self) -> usize;
}

/// Error raised when a reservation cannot be satisfied within the pool's limit.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ResourceError {
    /// The pool is full: `requested` bytes would exceed `limit` given `available`.
    #[error(
        "memory pool exhausted: requested {requested} bytes, only {available} of {limit} free"
    )]
    Exhausted {
        /// Bytes the caller tried to reserve.
        requested: usize,
        /// Bytes currently free (`limit - used`) at the moment of failure.
        available: usize,
        /// The pool's hard limit.
        limit: usize,
    },
}

/// Result alias for fallible reservations.
pub type ResourceResult<T> = Result<T, ResourceError>;

/// A snapshot of a [`MemoryPool`]'s accounting and its lifetime counters.
///
/// The four figures are only useful together. `peak_used / limit` says how close the
/// workload ran to its envelope; `denied` says whether it went over; `spill_requests`
/// says whether other operators had to give memory back to keep it running. A query that
/// spilled with a low peak was starved by something outside this pool, and one that
/// spilled with `peak == limit` and no spill requests was simply too big for the box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolStats {
    /// The admission cap in bytes.
    pub limit: usize,
    /// Bytes reserved right now.
    pub used: usize,
    /// High-water mark of `used` over the pool's life.
    pub peak_used: usize,
    /// Reservations refused for lack of headroom.
    pub denied: usize,
    /// Times a registered consumer was asked to spill.
    pub spill_requests: usize,
}

impl PoolStats {
    /// `used / limit` in `[0, 1]`; `1.0` for a zero-limit (unconfigured) pool, which is
    /// full by definition rather than empty.
    pub fn utilization(&self) -> f64 {
        if self.limit == 0 {
            return 1.0;
        }
        (self.used as f64 / self.limit as f64).min(1.0)
    }

    /// `peak_used / limit` in `[0, 1]` — the pressure the workload actually reached.
    pub fn peak_utilization(&self) -> f64 {
        if self.limit == 0 {
            return 1.0;
        }
        (self.peak_used as f64 / self.limit as f64).min(1.0)
    }
}

/// A greedy, thread-safe memory accounting pool.
///
/// Tracks `used` bytes against a fixed `limit`. Growth is admitted only when it
/// fits (no over-commit); release is always allowed. The pool accounts bytes — it
/// does not allocate them — so it is correct regardless of the underlying
/// allocator. Construct with [`MemoryPool::new`] (returns an `Arc` so reservations
/// can hold a cheap handle).
pub struct MemoryPool {
    /// The admission cap. Atomic so an autoscaler / a differently-configured query
    /// can resize the envelope at runtime ([`MemoryPool::set_limit`]) without
    /// rebuilding the pool or disturbing the live `used` accounting.
    limit: AtomicUsize,
    used: AtomicUsize,
    /// Soft-pressure line as basis points of `limit` (see [`Pressure`]).
    soft_bps: AtomicUsize,
    /// High-water mark of `used` over the pool's life. `peak / limit` is the memory
    /// pressure a workload *actually* reached, which a spot reading of `used` cannot
    /// recover after the fact — and after the fact is when anyone asks.
    peak: AtomicUsize,
    /// Reservations refused for lack of headroom. The direct evidence that the envelope
    /// is the binding constraint, where a high `peak` is only circumstantial.
    denied: AtomicUsize,
    /// Times the cooperative path had to ask a consumer to spill. Distinguishes "the
    /// budget was tight" from "the budget was tight and operators paid for it".
    spill_requests: AtomicUsize,
    /// Registered [`Spillable`] consumers (held by `Weak` so a finished operator's
    /// entry is harmless dead weight, swept lazily on the next slow-path entry). The
    /// `Mutex` is taken only on the cooperative slow path / registration — never per
    /// reservation — so the hot path stays lock-free.
    consumers: Mutex<Vec<Weak<dyn Spillable>>>,
}

impl std::fmt::Debug for MemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `Weak<dyn Spillable>` isn't `Debug`; surface only the accounting state.
        f.debug_struct("MemoryPool")
            .field("limit", &self.limit())
            .field("used", &self.used())
            .field("pressure", &self.pressure())
            .finish()
    }
}

impl MemoryPool {
    /// Create a pool admitting up to `limit` bytes in flight.
    pub fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit: AtomicUsize::new(limit),
            used: AtomicUsize::new(0),
            soft_bps: AtomicUsize::new(DEFAULT_SOFT_BPS),
            peak: AtomicUsize::new(0),
            denied: AtomicUsize::new(0),
            spill_requests: AtomicUsize::new(0),
            consumers: Mutex::new(Vec::new()),
        })
    }

    /// High-water mark of concurrently-reserved bytes over this pool's life.
    pub fn peak_used(&self) -> usize {
        self.peak.load(Ordering::Acquire)
    }

    /// Reservations this pool has refused for lack of headroom.
    pub fn denied(&self) -> usize {
        self.denied.load(Ordering::Acquire)
    }

    /// Times the cooperative path asked a registered consumer to spill.
    pub fn spill_requests(&self) -> usize {
        self.spill_requests.load(Ordering::Acquire)
    }

    /// A snapshot of everything this pool measured. Read together, the four figures say
    /// *why* a query spilled: a high `peak` with no `denied` means it ran close and made
    /// it, and `denied` with `spill_requests` means operators had to give memory back.
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            limit: self.limit(),
            used: self.used(),
            peak_used: self.peak_used(),
            denied: self.denied(),
            spill_requests: self.spill_requests(),
        }
    }

    /// The pool's current limit in bytes.
    pub fn limit(&self) -> usize {
        self.limit.load(Ordering::Acquire)
    }

    /// Resize the admission cap. Existing reservations are untouched; only what
    /// *future* reservations admit against changes. A shrink below `used` simply
    /// blocks new growth until releases bring `used` back under the new cap.
    pub fn set_limit(&self, limit: usize) {
        self.limit.store(limit, Ordering::Release);
    }

    /// Bytes currently reserved.
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }

    /// Bytes currently free (`limit - used`, saturating at 0).
    pub fn available(&self) -> usize {
        self.limit().saturating_sub(self.used())
    }

    /// Atomically grow `used` by `bytes` if it fits within the limit.
    ///
    /// Returns [`ResourceError::Exhausted`] without mutating the pool when the
    /// growth would exceed the limit. This is the low-level admission primitive;
    /// prefer [`MemoryPool::try_reserve`] for an RAII guard that releases on drop.
    pub fn try_reserve_bytes(&self, bytes: usize) -> ResourceResult<()> {
        let mut cur = self.used.load(Ordering::Acquire);
        loop {
            let limit = self.limit();
            // Saturating add guards the (pathological) overflow case as an exhaustion.
            let new = cur.saturating_add(bytes);
            if new > limit {
                self.denied.fetch_add(1, Ordering::Relaxed);
                return Err(ResourceError::Exhausted {
                    requested: bytes,
                    available: limit.saturating_sub(cur),
                    limit,
                });
            }
            match self
                .used
                .compare_exchange_weak(cur, new, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    self.observe_peak(new);
                    return Ok(());
                }
                Err(actual) => cur = actual, // raced with another reserver; retry
            }
        }
    }

    /// Raise the lifetime high-water mark to `used` if it is a new maximum.
    ///
    /// `fetch_max` rather than a load-compare-store: the reservation path is lock-free and
    /// concurrent, so a read-then-write would let two reservers racing past the previous
    /// peak each see the old value and record the smaller of their two — losing exactly
    /// the sample that mattered.
    fn observe_peak(&self, used: usize) {
        self.peak.fetch_max(used, Ordering::Relaxed);
    }

    /// Release `bytes` back to the pool. Clamped to the current `used` so a
    /// double-release can never underflow the counter.
    pub fn release_bytes(&self, bytes: usize) {
        let mut cur = self.used.load(Ordering::Acquire);
        loop {
            let new = cur - cur.min(bytes);
            match self
                .used
                .compare_exchange_weak(cur, new, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return,
                Err(actual) => cur = actual,
            }
        }
    }

    /// Open an empty RAII reservation against this pool. Grow it with
    /// [`MemoryReservation::try_grow`]; all held bytes return to the pool on drop.
    pub fn reserve(self: &Arc<Self>) -> MemoryReservation {
        MemoryReservation {
            pool: Arc::clone(self),
            size: 0,
        }
    }

    /// Reserve `bytes` and return the RAII guard, or fail if the pool is full.
    pub fn try_reserve(self: &Arc<Self>, bytes: usize) -> ResourceResult<MemoryReservation> {
        let mut r = self.reserve();
        r.try_grow(bytes)?;
        Ok(r)
    }

    /// Reserve `bytes`, asking registered consumers to spill when the pool is full.
    ///
    /// The cooperative path (Spark's model): on a plain-reserve miss, ask the
    /// **largest** registered [`Spillable`] consumers to spill — largest first — and
    /// retry, until the reservation fits or no consumer can free more. The
    /// requesting operator is *not* in the registry yet (it reserves *before*
    /// building the state it would later register), so every victim is a different
    /// operator or a concurrent query — exactly what should yield first. A final
    /// failure is returned so the caller can spill *itself* as the last resort.
    ///
    /// With no registered consumers this is exactly [`MemoryPool::try_reserve`], so
    /// it is a safe drop-in everywhere the plain reserve was used.
    pub fn try_reserve_cooperative(
        self: &Arc<Self>,
        bytes: usize,
    ) -> ResourceResult<MemoryReservation> {
        if let Ok(r) = self.try_reserve(bytes) {
            return Ok(r);
        }
        // Spill registered consumers, largest first, until it fits, a full pass frees
        // nothing, or the round budget runs out. Two independent termination conditions,
        // because "each round must free something" is only sound if the freed bytes come
        // back to *this* pool — see `MAX_SPILL_ROUNDS`.
        for _ in 0..MAX_SPILL_ROUNDS {
            let before = self.available();
            if bytes <= before {
                break;
            }
            let mut victims = self.live_consumers();
            victims.sort_by_key(|c| std::cmp::Reverse(c.spillable_bytes()));
            for v in victims {
                let need = bytes.saturating_sub(self.available());
                if need == 0 {
                    break;
                }
                self.spill_requests.fetch_add(1, Ordering::Relaxed);
                // spill() runs outside the registry lock and only releases memory.
                v.spill(need);
            }
            // Progress is measured on the pool, not on what the consumers *claimed* to
            // free. A consumer that reports bytes it never charged here would otherwise
            // keep the loop alive forever while the reservation never becomes satisfiable.
            if self.available() <= before {
                break;
            }
        }
        self.try_reserve(bytes)
    }

    /// Register a [`Spillable`] consumer so the cooperative path can ask it to spill.
    /// Held weakly: when the operator's `Arc` drops, its slot becomes dead weight and
    /// is swept on the next cooperative attempt — no explicit unregister needed.
    pub fn register_consumer(&self, consumer: &Arc<dyn Spillable>) {
        let mut guard = self.consumers.lock().unwrap_or_else(|e| e.into_inner());
        guard.retain(|w| w.strong_count() > 0); // sweep dead entries
        guard.push(Arc::downgrade(consumer));
    }

    /// Live registered consumers (upgraded from `Weak`); sweeps dead entries. Briefly
    /// locks the registry, then releases it before any `spill` runs.
    fn live_consumers(&self) -> Vec<Arc<dyn Spillable>> {
        let mut guard = self.consumers.lock().unwrap_or_else(|e| e.into_inner());
        guard.retain(|w| w.strong_count() > 0);
        guard.iter().filter_map(Weak::upgrade).collect()
    }

    /// Set the soft-pressure line as a fraction of the limit (clamped to `[0, 1]`).
    /// Carbonite drives this from its memory envelope so [`Pressure::Elevated`]
    /// fires at the same soft line the control plane uses.
    pub fn set_soft_fraction(&self, fraction: f64) {
        let bps = (fraction.clamp(0.0, 1.0) * 10_000.0).round() as usize;
        self.soft_bps.store(bps, Ordering::Release);
    }

    /// Current memory-pressure level (see [`Pressure`]). `Critical` at/above the hard
    /// cap, `Elevated` at/above the soft line, else `Nominal`. A zero limit (unbounded
    /// / unconfigured) is always `Nominal`.
    pub fn pressure(&self) -> Pressure {
        let limit = self.limit();
        if limit == 0 {
            return Pressure::Nominal;
        }
        let used = self.used();
        if used >= limit {
            return Pressure::Critical;
        }
        let soft =
            (limit as u128 * self.soft_bps.load(Ordering::Acquire) as u128 / 10_000) as usize;
        if used >= soft {
            Pressure::Elevated
        } else {
            Pressure::Nominal
        }
    }
}

/// An RAII handle to bytes held in a [`MemoryPool`].
///
/// Grows and shrinks an operator's accounted footprint; whatever remains is
/// released to the pool when the reservation is dropped, so a panicking or
/// early-returning operator never leaks budget.
#[derive(Debug)]
pub struct MemoryReservation {
    pool: Arc<MemoryPool>,
    size: usize,
}

impl MemoryReservation {
    /// Bytes currently held by this reservation.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Grow this reservation by `extra` bytes, or fail (leaving it unchanged) if
    /// the pool cannot admit the growth.
    pub fn try_grow(&mut self, extra: usize) -> ResourceResult<()> {
        self.pool.try_reserve_bytes(extra)?;
        // Saturating, matching the pool's own accounting. The pool refuses a growth that
        // would overflow `used`, so this cannot be reached in practice — but a plain `+=`
        // makes the reservation the one place in the crate where the arithmetic could
        // wrap, and a wrapped size releases the wrong number of bytes on drop.
        self.size = self.size.saturating_add(extra);
        Ok(())
    }

    /// Resize this reservation to exactly `bytes`, growing or shrinking as needed.
    ///
    /// The primitive an operator wants when it *recomputes* its footprint rather than
    /// accumulating it — a hash table that rehashed, a sort run that compacted. Expressing
    /// that as a grow/shrink pair at the call site means every such site re-derives the
    /// delta and its sign, which is how a shrink gets released twice.
    ///
    /// A failed growth leaves the reservation exactly as it was, so a caller that cannot
    /// get the larger size still holds (and still owes) the smaller one.
    ///
    /// # Errors
    /// [`ResourceError::Exhausted`] if growing to `bytes` does not fit the pool.
    pub fn try_resize(&mut self, bytes: usize) -> ResourceResult<()> {
        if bytes > self.size {
            self.try_grow(bytes - self.size)
        } else {
            self.shrink(self.size - bytes);
            Ok(())
        }
    }

    /// Shrink this reservation by `bytes` (clamped to its current size),
    /// returning that budget to the pool.
    pub fn shrink(&mut self, bytes: usize) {
        let freed = bytes.min(self.size);
        self.pool.release_bytes(freed);
        self.size -= freed;
    }

    /// Release everything this reservation holds back to the pool.
    pub fn free(&mut self) {
        self.shrink(self.size);
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        if self.size > 0 {
            self.pool.release_bytes(self.size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_and_release_track_usage() {
        let pool = MemoryPool::new(1000);
        assert_eq!(pool.used(), 0);
        assert_eq!(pool.available(), 1000);

        pool.try_reserve_bytes(400).unwrap();
        assert_eq!(pool.used(), 400);
        assert_eq!(pool.available(), 600);

        pool.release_bytes(400);
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn over_limit_reservation_fails_without_mutating() {
        let pool = MemoryPool::new(1000);
        pool.try_reserve_bytes(800).unwrap();
        let err = pool.try_reserve_bytes(300).unwrap_err();
        assert_eq!(
            err,
            ResourceError::Exhausted {
                requested: 300,
                available: 200,
                limit: 1000,
            }
        );
        // The failed reservation left `used` untouched, so the 200 free bytes remain.
        assert_eq!(pool.used(), 800);
        pool.try_reserve_bytes(200).unwrap();
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn reservation_releases_on_drop() {
        let pool = MemoryPool::new(1000);
        {
            let mut r = pool.try_reserve(500).unwrap();
            assert_eq!(pool.used(), 500);
            r.try_grow(200).unwrap();
            assert_eq!(pool.used(), 700);
            assert_eq!(r.size(), 700);
        }
        // Dropping the reservation returns all 700 bytes.
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn shrink_returns_partial_budget() {
        let pool = MemoryPool::new(1000);
        let mut r = pool.try_reserve(600).unwrap();
        r.shrink(250);
        assert_eq!(r.size(), 350);
        assert_eq!(pool.used(), 350);
    }

    #[test]
    fn double_release_cannot_underflow() {
        let pool = MemoryPool::new(1000);
        pool.try_reserve_bytes(100).unwrap();
        pool.release_bytes(100);
        pool.release_bytes(100); // extra release is clamped, not an underflow
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn try_grow_failure_leaves_reservation_intact() {
        let pool = MemoryPool::new(1000);
        let mut r = pool.try_reserve(900).unwrap();
        assert!(r.try_grow(200).is_err());
        assert_eq!(r.size(), 900);
        assert_eq!(pool.used(), 900);
    }

    /// A test consumer holding `held` pool bytes that releases them on `spill`.
    struct MockConsumer {
        reservation: std::sync::Mutex<MemoryReservation>,
        spill_calls: AtomicUsize,
    }

    impl MockConsumer {
        fn new(pool: &Arc<MemoryPool>, held: usize) -> Arc<Self> {
            // The reservation holds its own Arc to the pool, so the mock needs no
            // separate pool handle.
            let reservation = pool.try_reserve(held).unwrap();
            Arc::new(Self {
                reservation: std::sync::Mutex::new(reservation),
                spill_calls: AtomicUsize::new(0),
            })
        }
        fn spill_calls(&self) -> usize {
            self.spill_calls.load(Ordering::Acquire)
        }
    }

    impl Spillable for MockConsumer {
        fn spill(&self, target: usize) -> usize {
            self.spill_calls.fetch_add(1, Ordering::AcqRel);
            let mut r = self.reservation.lock().unwrap();
            let freed = target.min(r.size());
            r.shrink(freed); // releases to the pool — never reserves (the contract)
            freed
        }
        fn spillable_bytes(&self) -> usize {
            self.reservation.lock().unwrap().size()
        }
    }

    /// A consumer whose memory was never charged here — the real first occupant's shape.
    ///
    /// The published shuffle store frees genuine process memory on `spill`, but those bytes
    /// were never reserved against this pool, so `available()` does not move. Two things
    /// must hold, and they pull in opposite directions: the consumer must actually be
    /// *asked* (that is the whole benefit — real RAM comes back for the caller's own spill
    /// path), and the loop must still terminate and return the failure rather than asking
    /// forever for bytes that will never arrive.
    ///
    /// The progress check reads the pool, not the consumer's claim, which is exactly what
    /// makes both true. This test exists so that check is never "simplified" into trusting
    /// the returned figure.
    struct OffPoolConsumer {
        spill_calls: AtomicUsize,
        held: AtomicUsize,
    }

    impl Spillable for OffPoolConsumer {
        fn spill(&self, target: usize) -> usize {
            self.spill_calls.fetch_add(1, Ordering::AcqRel);
            // Frees memory somewhere else in the process; the pool's `used` is untouched.
            let freed = target.min(self.held.load(Ordering::Acquire));
            self.held.fetch_sub(freed, Ordering::AcqRel);
            freed
        }
        fn spillable_bytes(&self) -> usize {
            self.held.load(Ordering::Acquire)
        }
    }

    #[test]
    fn a_consumer_that_frees_untracked_memory_is_asked_once_and_then_gives_up() {
        let pool = MemoryPool::new(1000);
        let _hog = pool.try_reserve(900).unwrap();
        let off_pool = Arc::new(OffPoolConsumer {
            spill_calls: AtomicUsize::new(0),
            held: AtomicUsize::new(1 << 30), // "holding" a gigabyte the pool cannot see
        });
        pool.register_consumer(&(Arc::clone(&off_pool) as Arc<dyn Spillable>));

        assert!(
            pool.try_reserve_cooperative(500).is_err(),
            "the reservation cannot fit: freeing untracked memory does not free pool credit"
        );
        assert_eq!(
            off_pool.spill_calls.load(Ordering::Acquire),
            1,
            "it must be asked exactly once — never zero (no benefit) and never repeatedly \
             (an unterminating loop over bytes that never reach this pool)"
        );
        assert!(
            off_pool.held.load(Ordering::Acquire) < 1 << 30,
            "the ask did not reach the consumer"
        );
    }

    #[test]
    fn cooperative_reserve_spills_the_largest_other_consumer_first() {
        let pool = MemoryPool::new(1000);
        let small = MockConsumer::new(&pool, 200);
        let large = MockConsumer::new(&pool, 700);
        pool.register_consumer(&(Arc::clone(&small) as Arc<dyn Spillable>));
        pool.register_consumer(&(Arc::clone(&large) as Arc<dyn Spillable>));
        assert_eq!(pool.used(), 900); // only 100 free

        // Need 400: the pool must spill the *larger* consumer to make room, and the
        // small one should not be touched (its 200 + the 100 free isn't needed).
        let r = pool.try_reserve_cooperative(400).unwrap();
        assert_eq!(r.size(), 400);
        assert!(
            large.spill_calls() >= 1,
            "largest consumer should have spilled"
        );
        assert_eq!(
            small.spill_calls(),
            0,
            "smaller consumer should be left alone"
        );
    }

    #[test]
    fn cooperative_reserve_with_no_consumers_is_plain_reserve() {
        let pool = MemoryPool::new(1000);
        pool.try_reserve_bytes(800).unwrap();
        // Nobody to spill → behaves exactly like try_reserve (fails over budget).
        assert!(pool.try_reserve_cooperative(300).is_err());
        assert_eq!(pool.used(), 800);
        assert!(pool.try_reserve_cooperative(200).is_ok());
    }

    #[test]
    fn cooperative_reserve_fails_when_consumers_cannot_free_enough() {
        let pool = MemoryPool::new(1000);
        let c = MockConsumer::new(&pool, 300);
        pool.register_consumer(&(Arc::clone(&c) as Arc<dyn Spillable>));
        pool.try_reserve_bytes(600).unwrap(); // 900 used, 100 free; 300 spillable
                                              // Need 500: even after spilling all 300, only 400 is free → still fails, and
                                              // the loop terminates (no infinite spin) rather than hanging.
        assert!(pool.try_reserve_cooperative(500).is_err());
        assert!(c.spill_calls() >= 1);
    }

    #[test]
    fn dead_consumers_are_swept_and_never_asked() {
        let pool = MemoryPool::new(1000);
        {
            let c = MockConsumer::new(&pool, 500);
            pool.register_consumer(&(Arc::clone(&c) as Arc<dyn Spillable>));
        } // c dropped here → its reservation released, its Weak entry now dead
        assert_eq!(pool.used(), 0);
        // The dead entry must not be upgraded/asked; this just reserves cleanly.
        let r = pool.try_reserve_cooperative(800).unwrap();
        assert_eq!(r.size(), 800);
    }

    #[test]
    fn pressure_tracks_soft_and_hard_lines() {
        let pool = MemoryPool::new(1000);
        pool.set_soft_fraction(0.8);
        assert_eq!(pool.pressure(), Pressure::Nominal);
        pool.try_reserve_bytes(700).unwrap();
        assert_eq!(pool.pressure(), Pressure::Nominal); // below 80%
        pool.try_reserve_bytes(150).unwrap(); // 850 → at/above soft line
        assert_eq!(pool.pressure(), Pressure::Elevated);
        pool.try_reserve_bytes(150).unwrap(); // 1000 → at hard cap
        assert_eq!(pool.pressure(), Pressure::Critical);
    }

    /// A consumer that reports freed bytes it never charged to this pool. Its `spill`
    /// always claims progress, so a termination rule based on what consumers *say* spins
    /// forever with the pool's `available()` never moving.
    struct LyingConsumer;

    impl Spillable for LyingConsumer {
        fn spill(&self, target: usize) -> usize {
            target // claims to have freed it; releases nothing to the pool
        }
        fn spillable_bytes(&self) -> usize {
            usize::MAX
        }
    }

    #[test]
    fn cooperative_reserve_terminates_against_a_consumer_that_frees_nothing() {
        let pool = MemoryPool::new(1000);
        let liar: Arc<dyn Spillable> = Arc::new(LyingConsumer);
        pool.register_consumer(&liar);
        let _held = pool.try_reserve(900).unwrap();

        // Must return rather than spin: progress is measured on the pool, not on the claim.
        assert!(pool.try_reserve_cooperative(500).is_err());
        assert!(pool.spill_requests() > 0);
        assert!(pool.spill_requests() <= MAX_SPILL_ROUNDS * 8);
    }

    #[test]
    fn peak_and_denied_record_what_the_workload_actually_did() {
        let pool = MemoryPool::new(1000);
        {
            let _a = pool.try_reserve(400).unwrap();
            let _b = pool.try_reserve(300).unwrap();
            assert_eq!(pool.peak_used(), 700);
        }
        assert_eq!(pool.used(), 0);
        assert_eq!(
            pool.peak_used(),
            700,
            "the peak is a lifetime mark, not a live read"
        );

        assert!(pool.try_reserve(2000).is_err());
        assert_eq!(pool.denied(), 1);

        let stats = pool.stats();
        assert_eq!(stats.limit, 1000);
        assert_eq!(stats.peak_used, 700);
        assert_eq!(stats.denied, 1);
        assert!((stats.peak_utilization() - 0.7).abs() < 1e-9);
        assert_eq!(stats.utilization(), 0.0);
    }

    #[test]
    fn zero_limit_pool_stats_read_as_full_not_empty() {
        let stats = MemoryPool::new(0).stats();
        assert_eq!(stats.utilization(), 1.0);
        assert_eq!(stats.peak_utilization(), 1.0);
    }

    #[test]
    fn try_resize_grows_and_shrinks_in_one_step() {
        let pool = MemoryPool::new(1000);
        let mut r = pool.try_reserve(100).unwrap();

        r.try_resize(600).unwrap();
        assert_eq!(r.size(), 600);
        assert_eq!(pool.used(), 600);

        r.try_resize(200).unwrap();
        assert_eq!(r.size(), 200);
        assert_eq!(pool.used(), 200);

        // A refused growth leaves the reservation exactly as it was.
        assert!(r.try_resize(5000).is_err());
        assert_eq!(r.size(), 200);
        assert_eq!(pool.used(), 200);

        r.try_resize(0).unwrap();
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn pressure_levels_are_ordered() {
        assert!(Pressure::Critical > Pressure::Elevated);
        assert!(Pressure::Elevated > Pressure::Nominal);
        let pool = MemoryPool::new(1000);
        let _held = pool.try_reserve(850).unwrap();
        assert!(pool.pressure() >= Pressure::Elevated);
    }

    #[test]
    fn zero_limit_pool_is_never_under_pressure() {
        let pool = MemoryPool::new(0);
        assert_eq!(pool.pressure(), Pressure::Nominal);
    }
}
