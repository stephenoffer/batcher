//! Classified shuffle-fetch exceptions at the PyO3 boundary.
//!
//! The Rust transport classifies a fetch failure as `Retryable` (an
//! unreachable/idle/cancelled peer — worker loss) or `Fatal` (decode/protocol/auth
//! — a rerun cannot help). Preserving that verdict across the FFI lets the
//! control-plane reduce loop recompute+retry a transient loss but fail fast on a
//! deterministic fault, instead of treating every failure as worker loss and
//! recomputing up to `recovery_max_attempts` times.

use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

create_exception!(
    _native,
    RetryableShuffleError,
    PyRuntimeError,
    "A shuffle fetch failed transiently (unreachable/idle/cancelled peer); recompute + retry."
);
create_exception!(
    _native,
    FatalShuffleError,
    PyRuntimeError,
    "A shuffle fetch failed fatally (decode/protocol/auth); retrying cannot help."
);
create_exception!(
    _native,
    QueryCancelledError,
    PyRuntimeError,
    "The query was cancelled; it stopped at the next morsel boundary."
);
create_exception!(
    _native,
    PlanTooDeepError,
    PyRuntimeError,
    "The plan IR nests deeper than the native stack can deserialize."
);
create_exception!(
    _native,
    MemoryBudgetExceededError,
    PyRuntimeError,
    "An operator that cannot spill needs more than the configured memory budget."
);

/// Map a plan-IR error to a Python exception, giving depth overflow its own type.
///
/// Everything else here is a parse failure and reads fine as a generic runtime error.
/// Depth is different: it is the one `IrError` a user can act on (stop building the plan
/// in a loop), and it is the one that used to be an uncatchable `SIGABRT`, so making it
/// catchable is the point of the whole change.
pub(crate) fn ir_to_pyerr(e: bc_ir::IrError) -> PyErr {
    let msg = e.to_string();
    match e {
        bc_ir::IrError::PlanTooDeep { .. } => PlanTooDeepError::new_err(msg),
        _ => PyRuntimeError::new_err(msg),
    }
}

/// Map an interpreter error to a Python exception, giving cancellation and the memory
/// envelope their own types.
///
/// A cancelled query must not read as a generic runtime failure: the caller asked for it,
/// and the code that asked needs to distinguish "I stopped this" from "this broke".
///
/// `MemoryBudgetExceeded` is typed for the mirror-image reason. It is the one execution
/// failure with an obvious programmatic response — raise the envelope, or re-plan so the
/// non-spillable operator is not on the path — and it is only ever raised to a caller who
/// asked for a memory ceiling in the first place. As a bare `RuntimeError` the only way to
/// recognize it was to match on the message text, which is not a contract.
pub(crate) fn interp_to_pyerr(e: bc_interp::InterpError) -> PyErr {
    let msg = e.to_string();
    match e {
        bc_interp::InterpError::Cancelled => QueryCancelledError::new_err(msg),
        bc_interp::InterpError::MemoryBudgetExceeded { .. } => {
            MemoryBudgetExceededError::new_err(msg)
        }
        _ => PyRuntimeError::new_err(msg),
    }
}

/// Map a transport error to a Python exception, preserving the retryable/fatal
/// classification (`bc_transport::classify`).
pub(crate) fn transport_to_pyerr(e: bc_transport::TransportError) -> PyErr {
    let msg = e.to_string();
    match bc_transport::classify(&e) {
        bc_transport::FetchFault::Retryable => RetryableShuffleError::new_err(msg),
        bc_transport::FetchFault::Fatal => FatalShuffleError::new_err(msg),
    }
}

/// Register the classified exceptions in the `_native` module so the control plane
/// can catch them by name (`batcher._native.RetryableShuffleError`, re-exported from
/// `batcher._internal.errors`).
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "RetryableShuffleError",
        m.py().get_type::<RetryableShuffleError>(),
    )?;
    m.add("FatalShuffleError", m.py().get_type::<FatalShuffleError>())?;
    m.add("PlanTooDeepError", m.py().get_type::<PlanTooDeepError>())?;
    m.add(
        "QueryCancelledError",
        m.py().get_type::<QueryCancelledError>(),
    )?;
    m.add(
        "MemoryBudgetExceededError",
        m.py().get_type::<MemoryBudgetExceededError>(),
    )?;
    Ok(())
}
