//! The `MemoryPool` FFI surface — Carbonite's reserve-before-allocate primitive.
//!
//! Split out of `lib.rs` so the assembly point stays a module list plus the execute
//! entry points, and so the pool's Python-visible contract (the accounting getters and
//! the lifetime counters the control plane reads back) lives in one place next to the
//! `bc_resource::MemoryPool` it wraps.

use pyo3::prelude::*;

/// A process-wide memory accounting pool (Carbonite's reserve-before-allocate
/// enforcement primitive, from `bc-resource`). Carbonite sets the limit from its
/// memory envelope and reserves/releases against it so the engine spills instead
/// of OOMing. Accounts bytes; it does not allocate them.
#[pyclass]
pub(crate) struct MemoryPool {
    inner: std::sync::Arc<bc_resource::MemoryPool>,
}

#[pymethods]
impl MemoryPool {
    /// Create a pool admitting up to `limit_bytes` reserved at once.
    #[new]
    fn new(limit_bytes: u64) -> Self {
        Self {
            inner: bc_resource::MemoryPool::new(limit_bytes as usize),
        }
    }

    /// Try to reserve `bytes`; returns `True` on success, `False` if the pool is
    /// full (the caller should then spill / back-pressure). Never partially
    /// reserves — a `False` leaves the pool untouched.
    fn try_reserve(&self, bytes: u64) -> bool {
        self.inner.try_reserve_bytes(bytes as usize).is_ok()
    }

    /// Release `bytes` back to the pool (clamped so a double-release can't underflow).
    fn release(&self, bytes: u64) {
        self.inner.release_bytes(bytes as usize);
    }

    /// Resize the envelope. Live reservations are untouched; only what future
    /// reservations admit against changes (an autoscaler grew/shrank the budget).
    fn set_limit(&self, limit_bytes: u64) {
        self.inner.set_limit(limit_bytes as usize);
    }

    /// Bytes currently reserved.
    #[getter]
    fn used(&self) -> u64 {
        self.inner.used() as u64
    }

    /// Bytes currently free (`limit - used`).
    #[getter]
    fn available(&self) -> u64 {
        self.inner.available() as u64
    }

    /// The pool's hard limit in bytes.
    #[getter]
    fn limit(&self) -> u64 {
        self.inner.limit() as u64
    }

    /// High-water mark of concurrently-reserved bytes over this pool's life.
    ///
    /// A live `used` reading cannot be recovered after the fact, and after the fact is
    /// when anyone asks how close a query ran to its envelope. Measured in the data plane
    /// so it also counts reservations the control plane never made.
    #[getter]
    fn peak_used(&self) -> u64 {
        self.inner.peak_used() as u64
    }

    /// Reservations this pool refused for lack of headroom.
    ///
    /// The direct evidence that the envelope is what bound the workload, where a peak near
    /// the limit is only circumstantial.
    #[getter]
    fn denied(&self) -> u64 {
        self.inner.denied() as u64
    }

    /// Times the cooperative path asked a registered operator to spill so a reservation
    /// could be granted — how often other operators had to pay for this one.
    #[getter]
    fn spill_requests(&self) -> u64 {
        self.inner.spill_requests() as u64
    }
}
