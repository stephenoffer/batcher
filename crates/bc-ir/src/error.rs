//! The crate's error type: how a malformed plan IR is rejected at the wire boundary.

use thiserror::Error;

/// Errors raised while parsing or validating the plan IR.
#[derive(Debug, Error)]
pub enum IrError {
    #[error("malformed plan IR: {0}")]
    Parse(#[from] serde_json::Error),

    #[error("plan references source #{source_id}, but only {available} inputs were supplied")]
    UnknownSource { source_id: usize, available: usize },

    /// The plan nests deeper than the native stack can deserialize.
    ///
    /// Raised *before* deserialization rather than during it, because during it the
    /// failure is a stack overflow — which Rust turns into an uncatchable `SIGABRT`, not
    /// an error anyone can report. Returning this instead is the whole point of the check.
    #[error(
        "plan IR nests {depth} levels deep, past the {limit}-level limit. Deserializing it \
         would overflow the native stack. This usually means a plan was built in a loop \
         (`for _ in range(n): ds = ds.filter(...)`); collect intermediate results, or \
         combine the predicates into one expression."
    )]
    PlanTooDeep { depth: usize, limit: usize },
}
