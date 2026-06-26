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
    Ok(())
}
