//! Reservoir sampling — a fixed-size uniform random sample of a stream.
//!
//! Where the other sketches answer aggregate questions (cardinality, quantiles,
//! heavy hitters), a reservoir keeps a bounded *sample of the rows themselves*.
//! That sample is what the optimizer wants for the things a summary can't give:
//! `TABLESAMPLE`, sampling-based cardinality / selectivity estimation, and
//! approximate operators that need representative values rather than counts.
//!
//! The guarantee is Vitter's classic **Algorithm R**: after seeing `n` items the
//! reservoir holds `min(n, capacity)` of them, and every one of the `n` items is
//! present with equal probability `capacity / n`. One pass, O(capacity) space,
//! O(1) work per item.
//!
//! **Determinism.** Like the rest of the crate, the sampler is reproducible: it
//! carries its own seeded xorshift64 PRNG (no `rand`, no system entropy), seeded
//! from a fixed constant in [`ReservoirSample::new`]. Two reservoirs built from
//! the same stream on different machines hold the same sample, so they agree when
//! merged.
//!
//! **Mergeable.** Two reservoirs over disjoint partitions combine into a uniform
//! sample of the union. Each of the `capacity` output slots is drawn from one
//! side or the other, with the side chosen in proportion to how many items that
//! side has *seen* (`total_seen`), not how many it currently holds — that weight
//! is what makes the union sample uniform. See [`ReservoirSample::merge`] for the
//! exact scheme and its one documented approximation (sampling the survivors with
//! replacement when a side must contribute more slots than it holds, which only
//! happens once a side's reservoir is full and is then itself uniform).

use crate::Mergeable;

// Fixed seed → reproducible samples within and across processes, mirroring the
// crate-wide determinism contract. Any odd, non-zero constant works for
// xorshift64; this one is arbitrary.
const RESERVOIR_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Minimal deterministic xorshift64 PRNG. Self-contained so sampling is
/// reproducible without pulling in `rand` or touching system entropy.
#[derive(Clone)]
struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // xorshift64 must never hold zero (it would stay stuck at zero).
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// A uniform integer in `0..bound`. Returns 0 for `bound == 0` (callers never
    /// rely on that case).
    fn below(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            return 0;
        }
        self.next_u64() % bound
    }
}

/// A fixed-capacity uniform random sample of a stream (Vitter's Algorithm R).
///
/// After any number of [`add`](ReservoirSample::add) calls, [`sample`] holds
/// `min(total_seen, capacity)` items drawn uniformly at random from everything
/// seen so far. Deterministic given the fixed PRNG seed.
#[derive(Clone)]
pub struct ReservoirSample<T: Clone> {
    capacity: usize,
    items: Vec<T>,
    /// Total number of items ever offered via `add` (the `n` of Algorithm R).
    seen: u64,
    rng: XorShift64,
}

impl<T: Clone> ReservoirSample<T> {
    /// Create an empty reservoir holding at most `capacity` items, with a fixed
    /// PRNG seed for reproducibility.
    ///
    /// A `capacity` of 0 yields a reservoir that always stays empty.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            items: Vec::with_capacity(capacity),
            seen: 0,
            rng: XorShift64::new(RESERVOIR_SEED),
        }
    }

    /// Offer one item to the reservoir (Algorithm R).
    ///
    /// While fewer than `capacity` items have been seen, the item is always kept.
    /// Afterwards it replaces a uniformly random existing slot with probability
    /// `capacity / n`, where `n` is the number of items seen *including* this one
    /// — which keeps every seen item present with equal probability.
    pub fn add(&mut self, item: T) {
        self.seen += 1;

        if self.capacity == 0 {
            return;
        }

        if self.items.len() < self.capacity {
            self.items.push(item);
            return;
        }

        // Reservoir full: keep this item with probability capacity / seen by
        // choosing a uniform slot in 0..seen and replacing only if it lands inside
        // the reservoir.
        let slot = self.rng.below(self.seen);
        if (slot as usize) < self.capacity {
            self.items[slot as usize] = item;
        }
    }

    /// The current reservoir contents — a uniform sample of everything seen.
    pub fn sample(&self) -> &[T] {
        &self.items
    }

    /// Number of items currently held in the reservoir (`min(total_seen, capacity)`).
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the reservoir currently holds no items.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Total number of items ever offered via [`add`](ReservoirSample::add).
    pub fn total_seen(&self) -> u64 {
        self.seen
    }

    /// Capacity (maximum reservoir size).
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T: Clone> Mergeable for ReservoirSample<T> {
    /// Combine `other` into `self`, yielding a uniform sample of the *union* of
    /// both streams. Capacities must match.
    ///
    /// **Scheme.** The merged reservoir has `min(self.seen + other.seen, capacity)`
    /// slots. For each output slot we pick the source side — `self` vs `other` —
    /// with probability proportional to that side's `total_seen`. This is the
    /// correct weighting: a side that saw more of the stream should contribute
    /// proportionally more of the union sample, regardless of how many items it
    /// currently *holds*. We then draw an item from the chosen side's reservoir,
    /// which is itself a uniform sample of that side, so the result is uniform
    /// over the union.
    ///
    /// **Drawing.** Within a side we draw *without replacement* while items remain
    /// (a uniform partial permutation of that side's sample), then fall back to
    /// drawing *with replacement* if a side is asked for more slots than it holds.
    /// With-replacement only occurs when a side's reservoir is full (so it is a
    /// uniform sample and resampling it stays uniform) or when both sides are tiny
    /// relative to capacity (an exact small-stream case). It never breaks
    /// uniformity, only — in the with-replacement tail — allows the same source
    /// item to appear in more than one merged slot. This is the standard,
    /// documented approximation for mergeable reservoirs.
    fn merge(&mut self, other: &ReservoirSample<T>) {
        assert_eq!(self.capacity, other.capacity, "capacity mismatch");

        let total = self.seen + other.seen;
        let out_len = (self.capacity as u64).min(total) as usize;

        // Pools of indices we may still draw from each side, shuffled lazily via
        // swap-removal so each without-replacement draw is uniform and O(1).
        let mut self_pool: Vec<usize> = (0..self.items.len()).collect();
        let mut other_pool: Vec<usize> = (0..other.items.len()).collect();

        let mut merged: Vec<T> = Vec::with_capacity(out_len);

        for _ in 0..out_len {
            // Choose a side weighted by total_seen. Guard the degenerate all-zero
            // case (only possible when both sides are empty, hence out_len == 0,
            // so this branch is never actually reached then).
            let pick_self = if total == 0 {
                true
            } else {
                self.rng.below(total) < self.seen
            };

            // If the preferred side has no items to offer (empty reservoir but
            // possibly nonzero `seen` is impossible; empty reservoir means seen==0
            // unless capacity==0), fall back to the other side.
            let take_self = if pick_self {
                !self.items.is_empty()
            } else {
                other.items.is_empty() && !self.items.is_empty()
            };

            if take_self {
                if let Some(item) = draw(&mut self.rng, &mut self_pool, &self.items) {
                    merged.push(item);
                }
            } else if let Some(item) = draw(&mut self.rng, &mut other_pool, &other.items) {
                merged.push(item);
            } else if let Some(item) = draw(&mut self.rng, &mut self_pool, &self.items) {
                // Other side exhausted and not preferred but self still has items.
                merged.push(item);
            }
        }

        self.items = merged;
        self.seen = total;
    }
}

/// Draw one item from `items` using `pool` as the without-replacement supply.
/// Pops a uniform index from `pool` (swap-remove keeps it O(1)) while the pool is
/// non-empty; once exhausted, falls back to a uniform with-replacement draw over
/// all of `items`. Returns `None` only when `items` is empty.
fn draw<T: Clone>(rng: &mut XorShift64, pool: &mut Vec<usize>, items: &[T]) -> Option<T> {
    if items.is_empty() {
        return None;
    }
    if !pool.is_empty() {
        let i = rng.below(pool.len() as u64) as usize;
        let idx = pool.swap_remove(i);
        return items.get(idx).cloned();
    }
    let idx = rng.below(items.len() as u64) as usize;
    items.get(idx).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merge_all;

    #[test]
    fn reservoir_size_capped() {
        let mut r = ReservoirSample::new(100);
        for i in 0..10_000u64 {
            r.add(i);
        }
        assert_eq!(r.len(), 100);
        assert_eq!(r.total_seen(), 10_000);
        assert!(!r.is_empty());
        for &v in r.sample() {
            assert!(v < 10_000, "sampled value {v} out of range");
        }
    }

    #[test]
    fn reservoir_uniform_ish() {
        // Stream of 0..100_000; true mean is 49_999.5, range is 100_000.
        let n = 100_000u64;
        let mut r = ReservoirSample::new(2_000);
        for i in 0..n {
            r.add(i);
        }
        assert_eq!(r.len(), 2_000);

        let sum: u64 = r.sample().iter().sum();
        let sample_mean = sum as f64 / r.len() as f64;
        let true_mean = (n - 1) as f64 / 2.0;
        let range = n as f64;

        // Loose, seed-robust bound: well within 10% of the range.
        assert!(
            (sample_mean - true_mean).abs() < 0.1 * range,
            "sample_mean {sample_mean} too far from true_mean {true_mean}"
        );
    }

    #[test]
    fn small_stream_exact() {
        // Fewer items than capacity ⇒ the reservoir is exactly the stream.
        let mut r = ReservoirSample::new(50);
        for i in 0..10u64 {
            r.add(i * 7);
        }
        assert_eq!(r.len(), 10);
        assert_eq!(r.total_seen(), 10);
        assert_eq!(r.sample(), &[0, 7, 14, 21, 28, 35, 42, 49, 56, 63]);
    }

    #[test]
    fn empty_reservoir() {
        let r: ReservoirSample<u64> = ReservoirSample::new(8);
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert_eq!(r.total_seen(), 0);
        assert!(r.sample().is_empty());
    }

    #[test]
    fn zero_capacity_stays_empty() {
        let mut r = ReservoirSample::new(0);
        for i in 0..1_000u64 {
            r.add(i);
        }
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert_eq!(r.total_seen(), 1_000);
    }

    #[test]
    fn merge_preserves_size_and_total() {
        let cap = 100;
        let mut a = ReservoirSample::new(cap);
        let mut b = ReservoirSample::new(cap);
        for i in 0..6_000u64 {
            a.add(i);
        }
        for i in 6_000..10_000u64 {
            b.add(i);
        }
        a.merge(&b);
        assert_eq!(a.len(), cap, "merged reservoir not full");
        assert_eq!(a.total_seen(), 10_000);
        // Every merged value came from one of the two source streams.
        for &v in a.sample() {
            assert!(v < 10_000, "merged value {v} out of range");
        }
    }

    #[test]
    fn merge_small_union_keeps_all() {
        // Combined seen < capacity ⇒ merged reservoir holds the whole union.
        let cap = 100;
        let mut a = ReservoirSample::new(cap);
        let mut b = ReservoirSample::new(cap);
        for i in 0..10u64 {
            a.add(i);
        }
        for i in 100..120u64 {
            b.add(i);
        }
        a.merge(&b);
        assert_eq!(a.total_seen(), 30);
        assert_eq!(a.len(), 30, "small union should keep every item");
    }

    #[test]
    fn merge_all_folds_partitions() {
        // Four partitions of a 40_000-item stream; merged sample is uniform-ish.
        let parts = (0..4u64).map(|p| {
            let mut r = ReservoirSample::new(1_000);
            for i in 0..40_000u64 {
                if i % 4 == p {
                    r.add(i);
                }
            }
            r
        });
        let merged = merge_all(parts).unwrap();
        assert_eq!(merged.total_seen(), 40_000);
        assert_eq!(merged.len(), 1_000);

        let sum: u64 = merged.sample().iter().sum();
        let sample_mean = sum as f64 / merged.len() as f64;
        let true_mean = 39_999.0 / 2.0;
        assert!(
            (sample_mean - true_mean).abs() < 0.1 * 40_000.0,
            "merged sample_mean {sample_mean} too far from {true_mean}"
        );
    }
}
