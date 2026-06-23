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

use std::hash::Hash;

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

// Fixed seed → deterministic results within and across processes, so sketches
// built independently on different partitions agree when merged. Hash quality,
// not cryptographic security, is what matters here.
pub(crate) const SEED: ahash::RandomState =
    ahash::RandomState::with_seeds(0xC0FF_EE01, 0xDEAD_BEEF, 0x1234_5678, 0xABCD_EF01);

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
}
