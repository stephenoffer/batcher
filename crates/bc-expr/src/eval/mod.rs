//! Evaluation bodies for the scalar `Expr` variants.
//!
//! These are the private per-variant evaluators that `Expr::eval` (in `lib.rs`)
//! dispatches into. They were split out of `lib.rs` purely for file size; the one
//! `Expr` enum and its wire-contract `serde` tags stay in `lib.rs`. Behavior is
//! unchanged — each function moved here verbatim.

pub(crate) mod binary;
pub(crate) mod date;
pub(crate) mod image;
pub(crate) mod list;
pub(crate) mod math;
pub(crate) mod str;
