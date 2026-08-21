//! The `MemoryPool` FFI surface — Carbonite's reserve-before-allocate primitive.
//!
//! Split out of `lib.rs` so the assembly point stays a module list plus the execute
//! entry points, and so the pool's Python-visible contract (the accounting getters and
//! the lifetime counters the control plane reads back) lives in one place next to the
//! `bc_resource::MemoryPool` it wraps.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// What the **data plane's** process-wide pool is holding, or `None` if none exists yet.
///
/// The engine and the control plane account against two different `MemoryPool` objects.
/// `execute_plan` reserves operator state against the process-wide pool (and the Flight
/// shuffle store registers there as a spillable consumer); Carbonite's Python `BufferPool`
/// wraps a pool it constructs itself and charges its own coarse per-query reservations to
/// it. Neither can see the other, which left the control plane's pressure reading blind to
/// the reservations that dominate a real query — it could only infer them from process RSS.
///
/// This closes that in the safe direction, by *reading*. Merging the two counters would
/// double-count every query: Carbonite reserves the plan's estimated peak for the duration
/// of execution and the engine then reserves the same operator's actual bytes, so a plan
/// sized at 60% of the budget would leave the engine 40% and spill at half its envelope.
///
/// Returns:
///     A dict of `limit_bytes`, `used_bytes`, `available_bytes`, `peak_used_bytes`,
///     `denied`, `spill_requests`, `utilization`, `soft_limit_bytes` and `pressure`, or
///     `None` when no query has run under a memory budget in this process. `None` is
///     deliberately distinct from a dict of zeros, which would assert something about a pool
///     that has never existed.
///
///     The last two are what let a reader tell "the data plane is filling its envelope" from
///     "the data plane is idle and the box is full elsewhere" — opposite problems with
///     opposite fixes, and until they were carried out the control plane could only see
///     `used` and had to assume the line between them.
#[pyfunction]
pub(crate) fn engine_pool_stats(py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
    let Some(pool) = crate::process::shared_memory_pool_if_created() else {
        return Ok(None);
    };
    let stats = pool.stats();
    let out = PyDict::new(py);
    out.set_item("limit_bytes", stats.limit as u64)?;
    out.set_item("used_bytes", stats.used as u64)?;
    out.set_item(
        "available_bytes",
        stats.limit.saturating_sub(stats.used) as u64,
    )?;
    out.set_item("peak_used_bytes", stats.peak_used as u64)?;
    out.set_item("denied", stats.denied as u64)?;
    out.set_item("spill_requests", stats.spill_requests as u64)?;
    out.set_item("utilization", stats.utilization())?;
    out.set_item("soft_limit_bytes", stats.soft_limit as u64)?;
    // The level as a name rather than an ordinal: it is read by a person in a diagnostic and
    // by `PressureLevel`-shaped code in the control plane, and an integer would have to be
    // kept in step with an enum on the far side of the boundary.
    out.set_item(
        "pressure",
        match stats.pressure() {
            bc_resource::Pressure::Nominal => "NOMINAL",
            bc_resource::Pressure::Elevated => "ELEVATED",
            bc_resource::Pressure::Critical => "CRITICAL",
        },
    )?;
    Ok(Some(out.unbind()))
}

/// A memory accounting pool (Carbonite's reserve-before-allocate enforcement primitive,
/// from `bc-resource`). Carbonite sets the limit from its memory envelope and
/// reserves/releases against it so the engine spills instead of OOMing. Accounts bytes; it
/// does not allocate them.
///
/// **Constructing one makes a new, independent pool** — not a handle to the process-wide
/// pool `execute_plan` charges operator state to. The two are separate budgets on purpose
/// (see [`engine_pool_stats`], which is how the control plane reads the other one), so
/// every counter below describes only the reservations made through *this* object.
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
    /// when anyone asks how close a query ran to its envelope. Operator state and the
    /// Flight transit buffers are charged to the *engine's* pool, so their peak is in
    /// [`engine_pool_stats`] rather than here.
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
