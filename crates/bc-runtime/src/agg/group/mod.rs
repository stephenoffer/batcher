//! Group-key assignment and the parallel `combine` regroup.
//!
//! Two responsibilities that share the hash/`RowConverter` key machinery but sit on
//! opposite sides of the aggregate:
//!
//! - [`assign`] — the per-morsel hot path (and correctness reference) that maps each row
//!   to a dense group id.
//! - [`combine`] — the high-cardinality `combine` fast path the executor reaches once the
//!   concatenated partials cross the radix-parallel threshold. It hash-radix partitions
//!   by key so every row of a group lands in one partition, then groups *and* merges each
//!   partition independently across threads with no cross-partition merge.

mod assign;
mod combine;

pub(crate) use assign::{assign_groups, dense_budget};
pub(super) use combine::{combine_radix, merge_state};

// Same seed both halves use — bucketing is independent of the seed, but sharing it keeps
// the paths consistent when one is checked against the other.
const SEED: ahash::RandomState = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);

// A fixed hash for null keys so every null row lands in one partition (and thus one
// group). Grouping inside the partition still compares keys, so a non-null value that
// collides here is never conflated with null — only co-location depends on this value.
const NULL_HASH: u64 = 0xa5a5_5a5a_dead_beef;
