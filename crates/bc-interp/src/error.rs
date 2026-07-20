//! The crate's error type: plan-interpretation failures, plus the expression and
//! runtime errors it wraps from the crates below it.

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

    #[error(
        "mixed-aggregate spill: sub-aggregate group sets disagree ({expected} vs {found} groups)"
    )]
    MixedAggregateGroupMismatch { expected: usize, found: usize },

    #[error("join over an empty input side is not yet supported (no input schema)")]
    EmptyJoinInput,

    #[error(
        "value-based RANGE window frames (`RANGE <n> PRECEDING/FOLLOWING`) are not yet \
         supported; they need typed order-key arithmetic. Use ROWS units for a \
         physical-row frame, or a peer RANGE bound (CURRENT ROW / UNBOUNDED)"
    )]
    ValueBasedRangeFrame,

    #[error(
        "pipeline breaker cannot materialize column {column:?}: its {bytes} bytes of \
         variable-width data exceed the {limit}-byte limit of a 32-bit-offset Arrow array"
    )]
    MaterializeOffsetOverflow {
        column: String,
        bytes: usize,
        limit: usize,
    },

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
        "set operation (UNION/INTERSECT/EXCEPT) column {col} has incompatible branch \
         types {left} and {right} with no common type"
    )]
    IncompatibleSetOpTypes {
        /// The 0-based output column whose branch types cannot be unified.
        col: usize,
        /// The type accumulated from the earlier branch(es).
        left: String,
        /// The conflicting later-branch type.
        right: String,
    },

    #[error(
        "malformed partial-state batch: expected {expected} columns \
         ({n_keys} group keys + {state} state), got {got}"
    )]
    MalformedPartial {
        /// Total columns the wire format requires (`n_keys + Σ widths`).
        expected: usize,
        /// Group-key column count.
        n_keys: usize,
        /// Aggregate-state column count (`Σ widths`).
        state: usize,
        /// Columns actually present on the received batch.
        got: usize,
    },

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
