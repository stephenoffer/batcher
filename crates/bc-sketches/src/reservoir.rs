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
//! sample of the union. The `capacity` output slots are split between the sides in
//! proportion to how many items each has *seen* (`total_seen`), not how many it
//! currently holds — that weight is what makes the union sample uniform — and each
//! side then contributes that many of its items without replacement, so no source
//! row is ever duplicated or silently dropped. The split is a closed-form,
//! symmetric function of the two states and each side's draw is seeded from that
//! side alone, which makes the merge **commutative**: partials combine in any
//! order. See [`ReservoirSample::merge`] for the scheme and for the one place
//! order still matters (associativity past capacity, which is lossy by
//! construction rather than by choice).

use crate::Mergeable;

// Fixed seed → reproducible samples within and across processes, mirroring the
// crate-wide determinism contract. Any odd, non-zero constant works for
// xorshift64; this one is arbitrary.
const RESERVOIR_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64 finalizer — avalanches a low-entropy seed (a partition index, a
/// pair of small counts) into a well-distributed 64-bit value. xorshift64 warms up
/// poorly from such seeds, so every seed this module derives goes through here.
#[inline]
fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

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
        Self::with_seed(capacity, RESERVOIR_SEED)
    }

    /// Create an empty reservoir with an explicit PRNG seed.
    ///
    /// Every reservoir built by [`new`](Self::new) shares one constant seed, so
    /// two partitions of the *same length* make identical keep/replace decisions
    /// and retain the same stream *positions*. That is harmless for a single
    /// stream, but when partitions are formed positionally (round-robin, or equal
    /// row-group splits) it correlates the partitions' samples. Pass a distinct
    /// seed per partition — a partition index is enough — to decorrelate them
    /// while keeping each partition individually reproducible.
    pub fn with_seed(capacity: usize, seed: u64) -> Self {
        Self {
            capacity,
            items: Vec::with_capacity(capacity),
            seen: 0,
            rng: XorShift64::new(mix64(seed)),
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

/// How many of the `out_len` merged slots each side contributes.
///
/// The weighting is `seen`-proportional — a side that saw more of the stream owns
/// proportionally more of the union sample, regardless of how many items it
/// currently *holds*. We take the **expected** split (`out_len · seenₛ / total`)
/// rather than a per-slot coin flip: it has the same per-item inclusion
/// probability, strictly lower variance, and — being a closed-form function of the
/// two `seen` counts — is order-independent, which a stream of coin flips is not.
///
/// The single leftover slot from flooring goes to the side with the larger
/// fractional part, breaking ties on the larger `seen`. Both rules are symmetric
/// in the operands.
///
/// Finally each count is clamped to what its side actually holds, with the excess
/// **shifted to the other side** rather than resampled in place. That shift is the
/// whole point: a side with `seen < capacity` holds only `seen` items and the
/// proportional split routinely over-asks it, and drawing the shortfall from the
/// same side again is what used to duplicate rows while dropping others. Because
/// `out_len ≤ lenₛ + lenₒ` always holds, the shift can never overflow the
/// receiving side.
fn split_slots(out_len: usize, seen: (u64, u64), len: (usize, usize)) -> (usize, usize) {
    let total = seen.0 + seen.1;
    if total == 0 || out_len == 0 {
        return (0, 0);
    }
    let (out, total) = (out_len as u128, total as u128);
    let (wa, wb) = (out * seen.0 as u128, out * seen.1 as u128);

    let mut ka = (wa / total) as usize;
    let mut kb = (wb / total) as usize;

    // Flooring two shares of one whole can leave at most one slot unassigned.
    if ka + kb < out_len {
        let (ra, rb) = (wa % total, wb % total);
        if ra > rb || (ra == rb && seen.0 >= seen.1) {
            ka += 1;
        } else {
            kb += 1;
        }
    }

    if ka > len.0 {
        kb += ka - len.0;
        ka = len.0;
    }
    if kb > len.1 {
        ka += kb - len.1;
        kb = len.1;
    }
    debug_assert!(ka <= len.0 && kb <= len.1 && ka + kb == out_len);
    (ka, kb)
}

/// Append `k` items chosen uniformly **without replacement** from `items`.
///
/// The choice is derived from a seed hashed out of the side's own state, so it is
/// a pure function of that side and never touches the receiver's `add`-stream PRNG
/// — that mutable coupling is what made `merge` depend on call order.
fn take_without_replacement<T: Clone>(out: &mut Vec<T>, items: &[T], k: usize, state_seed: u64) {
    if k == 0 || items.is_empty() {
        return;
    }
    if k >= items.len() {
        out.extend_from_slice(items);
        return;
    }
    let mut rng = XorShift64::new(mix64(state_seed));
    let mut pool: Vec<usize> = (0..items.len()).collect();
    for _ in 0..k {
        let i = rng.below(pool.len() as u64) as usize;
        out.push(items[pool.swap_remove(i)].clone());
    }
}

impl<T: Clone> Mergeable for ReservoirSample<T> {
    /// Combine `other` into `self`, yielding a uniform sample of the *union* of
    /// both streams. Capacities must match.
    ///
    /// **Scheme.** The merged reservoir has `min(self.seen + other.seen, capacity)`
    /// slots, split between the two sides in proportion to their `total_seen` (see
    /// [`split_slots`]), and each side then contributes that many of its own items
    /// drawn uniformly **without replacement**. Each side's reservoir is already a
    /// uniform sample of its stream, so an item of side `s` survives with
    /// probability `(lenₛ/seenₛ)·(kₛ/lenₛ) = kₛ/seenₛ = capacity/total` — uniform
    /// over the union, and identical for both sides.
    ///
    /// **No item is ever duplicated or dropped.** A merged slot is always a
    /// distinct row of one of the two source samples. In particular, when
    /// `total_seen ≤ capacity` neither side is over-asked, so the merge is exactly
    /// the concatenation of the two samples — the union survives intact.
    ///
    /// **Order-independence.** The result is a function of the two operand
    /// *states*: the split is closed-form and symmetric, and each side's draw is
    /// seeded from that side alone. So `a.merge(&b)` and `b.merge(&a)` yield the
    /// same sample (as a multiset — the slot order follows the receiver). The one
    /// residual asymmetry is a leftover slot when the two sides have *identical*
    /// `total_seen`, where it goes to the receiver.
    ///
    /// **Associativity is exact only while the union fits in capacity.** Past that
    /// point a bounded reservoir merge is lossy by construction — `(a·b)` discards
    /// rows before `c` is ever weighed — so `(a·b)·c` and `a·(b·c)` agree in
    /// *distribution* but not pointwise. No slot-allocation scheme can recover
    /// that; only carrying every partition's sample to a single n-way merge can,
    /// which is what [`crate::merge_all`] callers should prefer for large fan-ins.
    fn merge(&mut self, other: &ReservoirSample<T>) {
        assert_eq!(self.capacity, other.capacity, "capacity mismatch");

        let total = self.seen + other.seen;
        let out_len = (self.capacity as u64).min(total) as usize;
        let (k_self, k_other) = split_slots(
            out_len,
            (self.seen, other.seen),
            (self.items.len(), other.items.len()),
        );

        let mut merged: Vec<T> = Vec::with_capacity(out_len);
        take_without_replacement(&mut merged, &self.items, k_self, state_seed(self));
        take_without_replacement(&mut merged, &other.items, k_other, state_seed(other));

        self.items = merged;
        self.seen = total;
    }
}

/// A side's draw seed: its own observable state, and nothing else. Deliberately
/// excludes the live PRNG position so a merge is reproducible from a serialized
/// reservoir and independent of how the side was fed.
fn state_seed<T: Clone>(r: &ReservoirSample<T>) -> u64 {
    mix64(r.seen)
        ^ mix64(r.items.len() as u64).rotate_left(17)
        ^ mix64(r.capacity as u64).rotate_left(41)
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

    /// Regression (merge dropped and duplicated rows): 100 partitions × 5 rows
    /// into a capacity-1000 reservoir. The union is 500 rows — half of capacity —
    /// so the merge must be a pure concatenation. Before the fix only ~385
    /// distinct rows survived, the rest being duplicates of other rows.
    #[test]
    fn merge_all_small_union_is_exact_concatenation() {
        let parts = (0..100u64).map(|p| {
            let mut r = ReservoirSample::new(1_000);
            for i in 0..5u64 {
                r.add(p * 5 + i);
            }
            r
        });
        let merged = merge_all(parts).unwrap();
        assert_eq!(merged.total_seen(), 500);
        let mut got: Vec<u64> = merged.sample().to_vec();
        got.sort_unstable();
        let want: Vec<u64> = (0..500).collect();
        assert_eq!(got, want, "small union must be an exact concatenation");
    }

    /// Regression (two-way form of the same bug): 10 + 20 rows into capacity 100
    /// yielded 30 slots but only 26 distinct rows.
    #[test]
    fn merge_two_way_small_union_keeps_every_row() {
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
        let mut got: Vec<u64> = a.sample().to_vec();
        got.sort_unstable();
        let mut want: Vec<u64> = (0..10).chain(100..120).collect();
        want.sort_unstable();
        assert_eq!(got, want, "no row may be dropped or duplicated");
    }

    /// Every merged slot must be a *distinct* source row whenever the union fits
    /// in capacity — checked across a spread of lopsided shapes.
    #[test]
    fn merge_never_duplicates_when_union_fits() {
        for (na, nb) in [(1u64, 1u64), (1, 99), (50, 49), (3, 7), (0, 40), (40, 0)] {
            let cap = 100;
            let mut a = ReservoirSample::new(cap);
            let mut b = ReservoirSample::new(cap);
            for i in 0..na {
                a.add(i);
            }
            for i in 0..nb {
                b.add(1_000 + i);
            }
            a.merge(&b);
            let mut got: Vec<u64> = a.sample().to_vec();
            got.sort_unstable();
            let len = got.len();
            got.dedup();
            assert_eq!(len, got.len(), "duplicate rows for shape ({na}, {nb})");
            assert_eq!(len as u64, na + nb, "lost rows for shape ({na}, {nb})");
        }
    }

    /// Regression (merge was order-dependent): the merged sample must be a
    /// function of the two operand *states*, not of which side is `self` nor of
    /// how much randomness the receiver's `add` stream happened to consume.
    #[test]
    fn merge_is_commutative() {
        fn built(n: u64, offset: u64, cap: usize) -> ReservoirSample<u64> {
            let mut r = ReservoirSample::new(cap);
            for i in 0..n {
                r.add(offset + i);
            }
            r
        }
        for (na, nb, cap) in [
            (6_000u64, 4_000u64, 100usize),
            (10, 20, 100),
            (100_000, 7, 64),
            (5_000, 5_000, 128),
            (1, 9_999, 32),
        ] {
            let a = built(na, 0, cap);
            let b = built(nb, 1_000_000, cap);

            let mut ab = a.clone();
            ab.merge(&b);
            let mut ba = b.clone();
            ba.merge(&a);

            let mut lhs: Vec<u64> = ab.sample().to_vec();
            let mut rhs: Vec<u64> = ba.sample().to_vec();
            lhs.sort_unstable();
            rhs.sort_unstable();
            assert_eq!(lhs, rhs, "merge not commutative for ({na}, {nb}, {cap})");
            assert_eq!(ab.total_seen(), ba.total_seen());
        }
    }

    /// Associativity holds exactly whenever the union fits in capacity (the merge
    /// is a concatenation there). Beyond capacity a bounded reservoir merge is
    /// lossy at every step, so pointwise associativity is unattainable — see the
    /// note on [`ReservoirSample::merge`].
    #[test]
    fn merge_is_associative_when_union_fits() {
        fn built(n: u64, offset: u64) -> ReservoirSample<u64> {
            let mut r = ReservoirSample::new(1_000);
            for i in 0..n {
                r.add(offset + i);
            }
            r
        }
        let (a, b, c) = (built(30, 0), built(40, 100), built(50, 200));

        let mut left = a.clone();
        left.merge(&b);
        left.merge(&c);

        let mut bc = b.clone();
        bc.merge(&c);
        let mut right = a.clone();
        right.merge(&bc);

        let mut lhs: Vec<u64> = left.sample().to_vec();
        let mut rhs: Vec<u64> = right.sample().to_vec();
        lhs.sort_unstable();
        rhs.sort_unstable();
        assert_eq!(lhs, rhs);
        assert_eq!(lhs.len(), 120);
    }

    /// The full-reservoir case must stay unbiased: merging many full partitions of
    /// a uniform stream keeps the merged sample's mean near the true mean, and the
    /// per-partition contributions stay proportional to what each side saw.
    #[test]
    fn merge_full_reservoirs_stays_unbiased() {
        // Two sides with a 3:1 seen ratio should contribute slots in ~3:1.
        let cap = 400;
        let mut a = ReservoirSample::new(cap);
        let mut b = ReservoirSample::new(cap);
        for i in 0..30_000u64 {
            a.add(i);
        }
        for i in 1_000_000..1_010_000u64 {
            b.add(i);
        }
        a.merge(&b);
        assert_eq!(a.len(), cap);
        let from_b = a.sample().iter().filter(|&&v| v >= 1_000_000).count();
        let expected = cap / 4;
        assert!(
            (from_b as i64 - expected as i64).abs() <= 2,
            "side weighting off: {from_b} slots from b, expected ~{expected}"
        );

        // And the sample of the low side is still spread over its whole range.
        let lows: Vec<u64> = a.sample().iter().copied().filter(|&v| v < 30_000).collect();
        let mean = lows.iter().sum::<u64>() as f64 / lows.len() as f64;
        assert!(
            (mean - 14_999.5).abs() < 3_000.0,
            "low-side sample mean {mean} is biased"
        );
    }

    #[test]
    fn distinct_seeds_decorrelate_partitions() {
        // Same-length streams with the crate default seed retain the same stream
        // positions; a per-partition seed breaks that correlation.
        let mut same_a = ReservoirSample::new(8);
        let mut same_b = ReservoirSample::new(8);
        for i in 0..1_000u64 {
            same_a.add(i);
            same_b.add(i);
        }
        assert_eq!(same_a.sample(), same_b.sample());

        let mut seeded = ReservoirSample::with_seed(8, 42);
        for i in 0..1_000u64 {
            seeded.add(i);
        }
        assert_ne!(
            seeded.sample(),
            same_a.sample(),
            "a distinct seed must retain distinct stream positions"
        );
        assert_eq!(seeded.len(), 8);
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
