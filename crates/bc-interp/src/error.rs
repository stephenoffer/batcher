use arrow::error::ArrowError;
use bc_expr::ExprError;
use bc_runtime::RuntimeError;
use thiserror::Error;

/// Errors raised while interpreting a plan.
#[derive(Debug, Error)]
pub enum InterpError {
    #[error("plan references source #{source_id}, but only {available} inputs were supplied")]
    UnknownSource { source_id: usize, available: usize },

    #[error("filter predicate must be boolean, got {got}")]
    NonBooleanPredicate { got: String },

    #[error("aggregation over empty input is not yet supported (no input schema)")]
    EmptyAggregateInput,

    #[error("join over an empty input side is not yet supported (no input schema)")]
    EmptyJoinInput,

    #[error("join output references unknown column: {0}")]
    UnknownJoinColumn(String),

    #[error("unnest references unknown column: {0}")]
    UnnestUnknownColumn(String),

    #[error("unnest column {column} must be a list/array, got {got}")]
    UnnestNotList { column: String, got: String },

    #[error("unpivot references unknown column: {0}")]
    UnpivotUnknownColumn(String),

    #[error("failed to build a thread pool with {0} workers")]
    ThreadPool(usize),

    #[error(
        "operator state ({needed} bytes) exceeds the memory budget ({budget} bytes) \
         and cannot spill: {reason}"
    )]
    MemoryBudgetExceeded {
        /// Estimated bytes the operator's in-memory state needs.
        needed: usize,
        /// The configured per-operator budget it exceeded.
        budget: usize,
        /// Why this operator cannot spill out of core (a `&'static` reason).
        reason: &'static str,
    },

    #[error(transparent)]
    Expr(#[from] ExprError),

    #[error(transparent)]
    Runtime(#[from] RuntimeError),

    #[error(transparent)]
    Arrow(#[from] ArrowError),
}
