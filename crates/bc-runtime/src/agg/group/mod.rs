//! Group-key assignment and the parallel `combine` regroup.
//!
//! Two responsibilities that share the hash/`RowConverter` key machinery but sit on
//! opposite sides of the aggregate:
//!
//! - [`assign`] — the per-morsel hot path (and correctness reference) that maps each row
//!   to a dense group id.
//! - [`runs`] — the same assignment for a key that arrives sorted, done by scanning runs of
//!   equal adjacent values instead of hashing. It *verifies* the ordering rather than being
//!   told about it, so it is safe to attempt on any input.
//! - [`combine`] — the high-cardinality `combine` fast path the executor reaches once the
//!   concatenated partials cross the radix-parallel threshold. It hash-radix partitions
//!   by key so every row of a group lands in one partition, then groups *and* merges each
//!   partition independently across threads with no cross-partition merge.

mod assign;
mod combine;
mod hash;
mod runs;

pub(crate) use assign::{assign_groups, dense_budget};
pub use combine::concat_disjoint;
pub(super) use combine::{
    combine_radix, combine_radix_parts, merge_state, radix_parallel_default, radix_partitions,
};

// Same seed both halves use — bucketing is independent of the seed, but sharing it keeps
// the paths consistent when one is checked against the other.
const SEED: ahash::RandomState = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);

// The null-key hash is shared with every other keying path — see `crate::keys`.
pub(crate) use crate::keys::NULL_HASH;
