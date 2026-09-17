//! Classified engine exceptions at the PyO3 boundary.
//!
//! The Rust transport classifies a fetch failure as `Retryable` (an
//! unreachable/idle/cancelled peer — worker loss) or `Fatal` (decode/protocol/auth
//! — a rerun cannot help). Preserving that verdict across the FFI lets the
//! control-plane reduce loop recompute+retry a transient loss but fail fast on a
//! deterministic fault, instead of treating every failure as worker loss and
//! recomputing up to `recovery_max_attempts` times. Cancellation, plan depth and the
//! memory envelope are typed for the same reason: each has a programmatic response, and
//! matching on message text is not a contract.
//!
//! **The classes these raise are defined in Python, not here.** `create_exception!` builds
//! a type whose base is `RuntimeError`, and a type built that way cannot be re-parented
//! afterwards (`__bases__` assignment refuses on a layout mismatch). So while these were
//! declared here, all five were `RuntimeError` subclasses in every built install and *none*
//! of them was a `BatcherError` — `except bt.BatcherError` did not catch a cancelled query,
//! a memory-budget refusal, a shuffle failure or an over-deep plan, though the exception
//! hierarchy's own docstring and the published documentation both said it did. The pure
//! Python fallbacks in `_internal.errors` had the right bases, so the contract held exactly
//! when the engine was *absent*, which is why no test saw it.
//!
//! Looking the class up on the error path instead costs an import of an already-imported
//! module, and cannot go stale: `_internal.errors` is the one definition, and a name that
//! stops existing there degrades to `RuntimeError` rather than failing the raise.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyString;

/// The Python module that owns Batcher's exception hierarchy.
const ERRORS_MODULE: &str = "batcher._internal.errors";

/// Raise `class` from `ERRORS_MODULE` with `msg`, or a plain `RuntimeError` if it is absent.
///
/// The fallback is not a formality: `bc-py` is importable on its own, and a caller that has
/// the extension without the Python package must still get a usable error rather than an
/// import failure masquerading as one.
fn typed(class: &str, msg: String) -> PyErr {
    Python::attach(|py| {
        let built = py
            .import(ERRORS_MODULE)
            .and_then(|module| module.getattr(class))
            .and_then(|cls| cls.call1((PyString::new(py, &msg),)));
        match built {
            Ok(value) => PyErr::from_value(value),
            Err(_) => PyRuntimeError::new_err(msg),
        }
    })
}

/// The engine's general-purpose failure, as `batcher.ExecutionError`.
///
/// Exposed so `lib::to_pyerr` -- the funnel every non-interpreter engine call goes through
/// -- names the same class the interpreter's unclassified errors do.
///
/// # Arguments
///
/// * `msg` - The engine's own message.
pub(crate) fn execution_error(msg: String) -> PyErr {
    typed("ExecutionError", msg)
}

/// Map a plan-IR error to a Python exception, giving depth overflow its own type.
///
/// Everything else here is a parse failure and reads fine as a generic runtime error.
/// Depth is different: it is the one `IrError` a user can act on (stop building the plan
/// in a loop), and it is the one that used to be an uncatchable `SIGABRT`, so making it
/// catchable is the point of the whole change.
pub(crate) fn ir_to_pyerr(e: bc_ir::IrError) -> PyErr {
    let msg = e.to_string();
    match e {
        bc_ir::IrError::PlanTooDeep { .. } => typed("PlanTooDeepError", msg),
        // A parse failure here means Python's `to_ir()` and Rust's `serde` tags have
        // drifted, which is the wire-contract break invariant #8 exists to prevent. It is a
        // *plan* problem and reads as one; as a bare `RuntimeError` it was the one class of
        // failure that could not be caught by the root the documentation names.
        _ => typed("PlanError", msg),
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
        bc_interp::InterpError::Cancelled => typed("QueryCancelledError", msg),
        bc_interp::InterpError::MemoryBudgetExceeded { .. } => {
            typed("MemoryBudgetExceededError", msg)
        }
        // Everything else is an operator failing while it ran, which is what
        // `ExecutionError` is for and what it was never used for. A `RuntimeError` here
        // escaped `except bt.BatcherError` -- and the commonest one by far is a value that
        // will not cast, so the single most likely runtime failure in the engine was the
        // one a caller could not catch by the documented root.
        _ => typed("ExecutionError", msg),
    }
}

/// Map a transport error to a Python exception, preserving the retryable/fatal
/// classification (`bc_transport::classify`).
pub(crate) fn transport_to_pyerr(e: bc_transport::TransportError) -> PyErr {
    let msg = e.to_string();
    match bc_transport::classify(&e) {
        bc_transport::FetchFault::Retryable => typed("RetryableShuffleError", msg),
        bc_transport::FetchFault::Fatal => typed("FatalShuffleError", msg),
    }
}

/// Register the classified exceptions in the `_native` module.
///
/// The types are Python's (see the module docstring), so this re-exports them under
/// `batcher._native.<Name>` for the callers that have always looked them up there. A
/// missing name is skipped rather than failing module import: the extension must stay
/// importable without the Python package around it.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let Ok(errors) = py.import(ERRORS_MODULE) else {
        return Ok(());
    };
    for name in [
        "RetryableShuffleError",
        "FatalShuffleError",
        "PlanTooDeepError",
        "QueryCancelledError",
        "MemoryBudgetExceededError",
    ] {
        if let Ok(cls) = errors.getattr(name) {
            m.add(name, cls)?;
        }
    }
    Ok(())
}
