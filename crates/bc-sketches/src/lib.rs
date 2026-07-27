//! `bc-sketches` — mergeable probabilistic sketches for the optimizer.
//!
//! Sketches trade a little accuracy for a lot of space: they answer "how many
//! distinct values?", "what's the p95?", "how often does this key occur?" in
//! kilobytes instead of gigabytes. Cheap-but-good answers are exactly what the
//! optimizer needs to pick join sides, size hash tables, place histograms, and
//! detect skew — without scanning the data twice.
//!
//! **Every sketch here is [`Mergeable`].** A sketch built on one partition combines
//! with another into the sketch of the *union*, so sketches compose across
//! partitions and nodes exactly like the engine's operators (partial → combine).
//! This single contract is what makes them usable in distributed planning.
//!
//! **Adding a sketch** = one new module whose type implements [`Mergeable`] (and,
//! by convention, `Clone` + a deterministic constructor). Nothing else in the
//! crate needs to change; `lib.rs` only declares the module and re-exports the
//! type. Pure Rust, no PyO3 — `cargo test`/fuzz directly.

use std::hash::{BuildHasher, Hash};

mod bloom;
mod countmin;
mod ddsketch;
mod frequent;
mod hll;
mod kll;
mod reservoir;
mod stats;
mod tdigest;

pub use bloom::BloomFilter;
pub use countmin::CountMinSketch;
pub use ddsketch::DDSketch;
pub use frequent::FrequentItems;
pub use hll::HyperLogLog;
pub use kll::KllSketch;
pub use reservoir::ReservoirSample;
pub use stats::ColumnStats;
pub use tdigest::TDigest;

/// A summary that can be combined with another of its own type to yield the
/// summary of the combined input.
///
/// Implementations require the two sides to share their construction parameters
/// (HLL precision, KLL `k`, Count-Min dimensions); a mismatch is a programming
/// error and panics rather than silently producing a meaningless merge. Because
/// the contract is uniform, generic code can merge a whole `Vec<S: Mergeable>`
/// from many partitions with a single fold.
pub trait Mergeable {
    /// Fold `other` into `self` in place.
    fn merge(&mut self, other: &Self);
}

/// Merge an iterator of sketches into one, or `None` if empty. The generic
/// partial→combine reducer every distributed caller can share.
pub fn merge_all<S: Mergeable>(mut sketches: impl Iterator<Item = S>) -> Option<S> {
    let mut acc = sketches.next()?;
    for s in sketches {
        acc.merge(&s);
    }
    Some(acc)
}

// Deterministic results within *and across* processes, so sketches built independently on
// different partitions agree when merged. Hash quality, not cryptographic security, is
// what matters here.
//
// The "and across" half of that sentence was false until this became
// `PortableBuildHasher`. It sat over an `ahash::RandomState`, which selects an AES-NI
// backend from the compile-time `target_feature` — so two workers built with different
// `-C target-cpu` produced different registers for the same value, and `Mergeable::merge`
// combined them without complaint into a wrong estimate. That is the quietest failure in
// this crate: a cardinality estimate has no oracle at runtime, so nothing notices.
pub(crate) const SEED: bc_arrow::PortableBuildHasher =
    bc_arrow::PortableBuildHasher::with_seed(0x534B_4554_4348_4553);

/// Hash one value with the shared deterministic seed.
pub(crate) fn hash_one<T: Hash + ?Sized>(value: &T) -> u64 {
    SEED.hash_one(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_all_folds_partitions() {
        let parts = (0..4).map(|p| {
            let mut hll = HyperLogLog::new(12);
            for i in 0..25_000u64 {
                if i % 4 == p {
                    hll.add(&i);
                }
            }
            hll
        });
        let merged = merge_all(parts).unwrap();
        // Union of the four disjoint quarters ≈ 25k distinct.
        let err = (merged.estimate() - 25_000.0).abs() / 25_000.0;
        assert!(err < 0.03, "merge_all error {err}");
    }

    /// The crate's hash is a **cross-process** value, and this pins it.
    ///
    /// `SEED` used to be an `ahash::RandomState` under a comment promising that its fixed
    /// seed made partition-built sketches merge identically across processes. That was
    /// false: `ahash` selects an AES-NI backend from the compile-time `target_feature`, so
    /// two workers built with different `-C target-cpu` hashed the same value differently
    /// and `merge` combined their registers without complaint.
    ///
    /// That failure is quieter than the shuffle's. A mis-routed shuffle key splits a
    /// `GROUP BY` group, which a differential test can catch; a cardinality estimate has no
    /// oracle at runtime, so a wrong `approx_count_distinct` just looks like the sketch's
    /// documented error bar. Nothing anywhere would have flagged it.
    ///
    /// So the hash is pinned by value. A change here means sketches built by two different
    /// engine versions can no longer be merged, which is a wire-format break and has to be
    /// a deliberate, announced one — not a silent re-baseline.
    #[test]
    fn value_hashing_is_pinned_across_builds() {
        assert_eq!(hash_one(&0_u64), 2_274_247_470_533_401_384);
        assert_eq!(hash_one(&1_u64), hash_one(&1_u64));
        assert_ne!(hash_one(&0_u64), hash_one(&1_u64));
    }

    /// Two sketches built independently — as two workers would — must merge to the same
    /// registers as one built over the whole input. This is the property the seed exists
    /// for, stated as a test rather than as a comment.
    #[test]
    fn independently_built_sketches_agree_with_a_single_pass() {
        let mut whole = HyperLogLog::new(12);
        for i in 0..2_000_u64 {
            whole.add(&i);
        }

        let mut left = HyperLogLog::new(12);
        let mut right = HyperLogLog::new(12);
        for i in 0..2_000_u64 {
            if i % 2 == 0 {
                left.add(&i);
            } else {
                right.add(&i);
            }
        }
        left.merge(&right);

        assert_eq!(
            left.to_bytes(),
            whole.to_bytes(),
            "a sketch merged from two partitions differs from one built in a single pass — \
             the registers disagree, which is exactly what a non-portable hash produces"
        );
    }
}
