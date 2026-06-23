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

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use thiserror::Error;

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

/// A greedy, thread-safe memory accounting pool.
///
/// Tracks `used` bytes against a fixed `limit`. Growth is admitted only when it
/// fits (no over-commit); release is always allowed. The pool accounts bytes — it
/// does not allocate them — so it is correct regardless of the underlying
/// allocator. Construct with [`MemoryPool::new`] (returns an `Arc` so reservations
/// can hold a cheap handle).
#[derive(Debug)]
pub struct MemoryPool {
    /// The admission cap. Atomic so an autoscaler / a differently-configured query
    /// can resize the envelope at runtime ([`MemoryPool::set_limit`]) without
    /// rebuilding the pool or disturbing the live `used` accounting.
    limit: AtomicUsize,
    used: AtomicUsize,
}

impl MemoryPool {
    /// Create a pool admitting up to `limit` bytes in flight.
    pub fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit: AtomicUsize::new(limit),
            used: AtomicUsize::new(0),
        })
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
                Ok(_) => return Ok(()),
                Err(actual) => cur = actual, // raced with another reserver; retry
            }
        }
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
        self.size += extra;
        Ok(())
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
}
