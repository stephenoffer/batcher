//! Evaluation bodies for the scalar `Expr` variants.
//!
//! These are the private per-variant evaluators that `Expr::eval` (in `lib.rs`)
//! dispatches into. They were split out of `lib.rs` purely for file size; the one
//! `Expr` enum and its wire-contract `serde` tags stay in `lib.rs`. Behavior is
//! unchanged — each function moved here verbatim.

/// The set type for any evaluator that builds a membership or dedup set.
///
/// `std::collections::HashSet` defaults to SipHash, which is ~20-30 cycles per probe and was
/// the whole cost of a large `IN` over a wide column: a 204-member `l_partkey IN (…)` over
/// TPC-H `lineitem` measured 13.2 ms against 4.4 ms for an 8-member (linear-scan) list on the
/// same 6M rows, and set size cannot explain that for an O(1) probe — the hash can. `ahash` is
/// already the workspace's hasher for exactly this reason (`bc-runtime`'s join tables use it).
///
/// Every use here is **hasher-independent by construction**: these sets answer membership or
/// drive a first-occurrence dedup whose output order comes from the input scan, never from
/// iterating the set. Swapping the hasher therefore cannot change a result — which is what
/// makes it a free win rather than a trade.
///
/// It lives at the module root because three evaluators want it (`in_list`, `list_ops::set`,
/// `list::unique`) and the alternative was writing it out three times.
pub(crate) type FastSet<T> = std::collections::HashSet<T, ahash::RandomState>;

pub(crate) mod binary;
pub(crate) mod cast;
mod dispatch;
pub(crate) mod generate;
pub(crate) mod geo;
pub(crate) mod hash;
pub(crate) mod in_list;
pub(crate) mod list;
pub(crate) mod list_ops;
pub(crate) mod map;
pub(crate) mod math;
pub(crate) mod media;
pub(crate) mod security;
pub(crate) mod str;
pub(crate) mod temporal;
