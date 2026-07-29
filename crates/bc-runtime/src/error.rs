//! The crate's error type: how the stateful runtime structures report failure.

use arrow::error::ArrowError;
use thiserror::Error;

/// Errors raised by runtime-library structures.
#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("aggregate {func} is not supported for column type {dtype}")]
    UnsupportedAggregate { func: String, dtype: String },

    #[error("aggregate {func} requires an input column")]
    MissingAggregateInput { func: String },

    #[error("integer SUM overflowed i64; cast the column to a wider type first")]
    SumOverflow,

    #[error("window function {func} is not supported for column type {dtype}")]
    UnsupportedWindow { func: String, dtype: String },

    #[error("window function {func} requires an input column")]
    MissingWindowInput { func: String },

    #[error("window function {func} requires order keys")]
    WindowRequiresOrder { func: String },

    #[error("malformed spilled partial: expected {expected} columns, got {got}")]
    MalformedPartial { expected: usize, got: usize },

    #[error("range-partition key must be a numeric column, got {dtype}")]
    NonNumericRangeKey { dtype: String },

    #[error("range join cannot run on this condition: {reason}")]
    UnsupportedRangeJoin { reason: String },

    /// The spill filesystem ran out of room (or the writer's quota did).
    ///
    /// Separated from the generic [`RuntimeError::Io`] because it is the one spill failure a
    /// user can act on, and because bare "No space left on device" says nothing about
    /// *which* device: spill scratch defaults to the system temp directory, which on a
    /// container is often a small overlay or a tmpfs sized well below the query's spill
    /// volume, while the large volume the user assumed was being used sits elsewhere. The
    /// path and the volume written so far are what turn that into a decision.
    #[error(
        "spill ran out of disk space after writing {written_bytes} bytes to {dir} \
         (the query needs more spill scratch than that filesystem has). Point \
         memory.spill_dir at a larger filesystem, set memory.spill_remote_uri to overflow \
         to object storage, or lower memory.max_memory_bytes so less of the query \
         materializes at once. Underlying error: {source}"
    )]
    SpillOutOfSpace {
        dir: String,
        written_bytes: u64,
        source: std::io::Error,
    },

    #[error("spill i/o error: {0}")]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Arrow(#[from] ArrowError),
}
