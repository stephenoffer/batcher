//! The opaque-operator boundary + dynamic-batching machinery for the UDF / ML
//! inference plane.
//!
//! This crate is the pure-Rust foundation of workstream W8/W3: running arbitrary
//! per-batch transforms (Python `map_batches`, model inference, GPU kernels) as
//! first-class operators over Arrow [`RecordBatch`](bc_arrow::RecordBatch)es.
//!
//! It deliberately links **no PyO3** — only `bc-py` does. The Python/GPU call is
//! injected as a Rust closure through [`FnOperator`], so the engine schedules,
//! rebatches, and governs an opaque operator without knowing whether the work
//! behind it is native Rust, a Python UDF, or a model forward pass.
//!
//! Three pieces compose the plane:
//! - [`OpaqueOperator`] — the trait every UDF/inference operator implements; it is
//!   always a pipeline breaker at the scheduling level.
//! - [`Rebatcher`] — coalesces/splits incoming batches to a target row count, so the
//!   expensive operator runs at an efficient batch size regardless of upstream
//!   morsel sizes.
//! - [`BatchSizeController`] — a PID governor that retunes the rebatch target toward
//!   a latency set-point (the adaptive dynamic-batching loop).

mod batch_size;
mod operator;
mod rebatch;

pub use batch_size::BatchSizeController;
pub use operator::{FnOperator, OpaqueOperator};
pub use rebatch::Rebatcher;

/// Errors raised while planning or running an opaque operator.
#[derive(Debug, thiserror::Error)]
pub enum UdfError {
    /// An operator produced (or was fed) a batch whose schema does not match the
    /// schema it declared via [`OpaqueOperator::schema_out`], or batches with
    /// inconsistent schemas were pushed to a [`Rebatcher`].
    #[error("schema mismatch: {0}")]
    SchemaMismatch(String),

    /// An underlying Arrow kernel (concat, slice, …) failed.
    #[error(transparent)]
    Arrow(#[from] arrow::error::ArrowError),

    /// The user-supplied operator body failed.
    #[error("operator failed: {0}")]
    Operator(String),
}

/// Crate result alias.
pub type Result<T> = std::result::Result<T, UdfError>;
