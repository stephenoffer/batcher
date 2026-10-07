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

    #[error("distinct key references unknown column: {0}")]
    DistinctUnknownColumn(String),

    #[error("unnest references unknown column: {0}")]
    UnnestUnknownColumn(String),

    #[error("unnest column {column} must be a list/array, got {got}")]
    UnnestNotList { column: String, got: String },

    #[error("unpivot references unknown column: {0}")]
    UnpivotUnknownColumn(String),

    #[error("failed to build a thread pool with {0} workers")]
    ThreadPool(usize),

    /// A set operation's branches disagree on a column's type with no common supertype.
    /// Boxed: the conflict carries the path and both types, and an unboxed variant that
    /// size would widen every `Result<_, InterpError>` in the crate.
    #[error("{0}")]
    IncompatibleSetOpTypes(Box<SetOpConflict>),

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

    /// Not a failure: the streaming executor has found, from the build sides it just
    /// prepared, that it cannot spread this plan across cores, and is asking the caller to
    /// run it on the materializing executor instead.
    ///
    /// It is raised only when the caller opted in (it is the caller that knows whether the
    /// materializing executor's memory profile is affordable), and only *after* the build
    /// sides are prepared — which is the first moment the answer is exact rather than
    /// guessed from the plan's shape. The work discarded is that preparation; the work
    /// avoided is the whole probe-and-aggregate, which on this shape runs at a fraction of
    /// the machine (measured at sf10: a 60M x 15M semi join at 5.7x parallelism streaming
    /// against 62x materializing).
    ///
    /// Every executor answers the same rows, so honoring or ignoring this changes only
    /// speed and peak memory — never the result.
    #[error("this plan cannot be sharded by the streaming executor: {reason}")]
    PreferMaterializing {
        /// What on the probe spine blocked sharding (a `&'static` reason).
        reason: &'static str,
    },

    /// The query was cancelled between morsels.
    ///
    /// Not a failure of the plan: something asked for the query to stop, and the executor
    /// noticed at the next point where unwinding was safe. It is an error rather than an
    /// empty result because an empty result is indistinguishable from a query that
    /// legitimately matched nothing, and a caller that cannot tell those apart will
    /// eventually treat a cancellation as data.
    #[error("query cancelled")]
    Cancelled,

    /// Not a failure: the plan cannot stream its driving source in chunks
    /// (`stream::chunked::chunkable` is false), so the caller should run it with every source
    /// resident instead. Returned before any work is done.
    #[error("this plan cannot stream its driving source in chunks")]
    NotChunkable,

    /// The producer feeding a chunked source failed (a read error, a malformed chunk).
    #[error("reading the chunked driving source failed: {0}")]
    ChunkSource(String),

    #[error(transparent)]
    Expr(#[from] ExprError),

    #[error(transparent)]
    Runtime(#[from] RuntimeError),

    #[error(transparent)]
    Arrow(#[from] ArrowError),
}

/// Where the branches of a set operation stop agreeing on one column's type.
#[derive(Debug)]
pub struct SetOpConflict {
    /// The output column whose branch types cannot be unified, by name.
    pub column: String,
    /// The 0-based branch whose type first conflicted with the earlier branches'.
    pub branch: usize,
    /// Where inside the column the types first disagree: the column name itself for a
    /// flat type, `s.x.y` for a struct field, `s.items[]` for a list's elements.
    pub path: String,
    /// What the earlier branch(es) hold at `path`.
    pub left: String,
    /// What the conflicting branch holds at `path`.
    pub right: String,
    /// The whole column type accumulated from the earlier branch(es).
    pub left_type: String,
    /// The conflicting branch's whole column type.
    pub right_type: String,
}

impl std::fmt::Display for SetOpConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "set operation (UNION/INTERSECT/EXCEPT) column `{}` has no common type across its \
             branches: at `{}`, {} (earlier branches) vs {} (branch {}); whole column types: \
             {} and {}",
            self.column,
            self.path,
            self.left,
            self.right,
            self.branch,
            self.left_type,
            self.right_type
        )
    }
}

/// The error a memory-budgeted executor gives way with when the machine's available memory
/// has fallen below the guard's floor (`bc_resource::headroom`): the same signal a breaker
/// over its budget sends, so every caller already routes it to an executor that spills.
pub(crate) fn low_memory(h: bc_resource::headroom::Headroom) -> InterpError {
    InterpError::MemoryBudgetExceeded {
        needed: usize::try_from(h.floor).unwrap_or(usize::MAX),
        budget: usize::try_from(h.available).unwrap_or(usize::MAX),
        reason: "the machine's available memory fell below the guard's floor",
    }
}
