//! The crate's error type: how a malformed plan IR is rejected at the wire boundary.

use thiserror::Error;

/// Errors raised while parsing or validating the plan IR.
#[derive(Debug, Error)]
pub enum IrError {
    #[error("malformed plan IR: {0}")]
    Parse(#[from] serde_json::Error),

    #[error("plan references source #{source_id}, but only {available} inputs were supplied")]
    UnknownSource { source_id: usize, available: usize },
}
