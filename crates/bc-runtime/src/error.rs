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

    #[error("spill i/o error: {0}")]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Arrow(#[from] ArrowError),
}
