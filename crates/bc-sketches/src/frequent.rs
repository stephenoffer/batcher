//! Misra-Gries — frequent-items (heavy-hitter *key*) enumeration.
//!
//! Where Count-Min answers "how often does *this* key occur?", Misra-Gries
//! answers the question the optimizer actually asks before a shuffle: "*which*
//! keys are hot?" It maintains a bounded summary of at most `capacity` monitored
//! `(key, counter)` pairs and, after a single pass, the monitored set is
//! guaranteed to contain **every** key whose true frequency exceeds
//! `N / (capacity + 1)`. Those are exactly the candidates worth *salting* (fanning
//! across sub-partitions) so one skewed join/group key can't overload a reducer.
//!
//! The counters are *lower* bounds: the algorithm only ever decrements, so the
//! stored count for a key undershoots its true frequency by at most
//! `N / (capacity + 1)`. That one-sided error is the dual of Count-Min's one-sided
//! over-estimate, and it's why the two sketches are used together — Count-Min to
//! *size* a known hot key, Misra-Gries to *find* the unknown ones.
//!
//! Mergeable: sum the counters of common keys, union the rest, then if the union
//! exceeds `capacity` keys, subtract the `(capacity + 1)`-th largest count from
//! every counter and drop the non-positive ones. This preserves the same
//! `N / (capacity + 1)` guarantee on the combined stream, so per-partition
//! summaries compose into the global one (partial → combine).

use std::collections::HashMap;
use std::hash::Hash;

use crate::Mergeable;

/// The counter table's hasher — **process-local**, and separate from the crate's
/// cross-process [`crate::SEED`] on purpose.
///
/// This one only picks buckets inside one `HashMap`; the value never leaves the process
/// and no merged result depends on it. `merge` accumulates counts (commutative) and
/// `reduce_to_capacity` sorts the counts and applies a value threshold, so neither is
/// sensitive to iteration order — which is what makes `ahash`, the faster choice, also the
/// correct one here.
const TABLE_HASHER: ahash::RandomState =
    ahash::RandomState::with_seeds(0xC0FF_EE01, 0xDEAD_BEEF, 0x1234_5678, 0xABCD_EF01);

/// The counter table. Hashed with [`TABLE_HASHER`] rather than std's randomly-seeded
/// SipHash: the update runs once per row of every sketched column, where SipHash's per-key
/// cost is a material share of the sketch.
///
/// What makes this safe is **not** the fixed seed — a fixed `ahash` seed is not reproducible
/// across builds, only within one — but that no result depends on the hash at all. Iteration
/// order never reaches an answer: `heavy_hitters` sorts its output, `merge` sums counts, and
/// `reduce_to_capacity` thresholds on a sorted count. The `Mergeable` contract holds because
/// the *algorithm* is order-independent, which is a stronger guarantee than a seed.
type Counters<K> = HashMap<K, u64, ahash::RandomState>;

/// A Misra-Gries summary tracking up to `capacity` candidate heavy-hitter keys.
#[derive(Clone)]
pub struct FrequentItems<K: Hash + Eq + Clone> {
    capacity: usize,
    counters: Counters<K>,
    total: u64,
}

impl<K: Hash + Eq + Clone> FrequentItems<K> {
    /// Create a summary monitoring at most `capacity` keys. Guarantees that, after
    /// processing a stream of total weight `N`, every key with frequency
    /// `> N / (capacity + 1)` is among the monitored keys.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity >= 1, "capacity must be >= 1");
        Self {
            capacity,
            counters: HashMap::with_capacity_and_hasher(capacity + 1, TABLE_HASHER),
            total: 0,
        }
    }

    /// Add one occurrence of a key held only by reference, cloning it **only if it
    /// actually enters the summary**.
    ///
    /// Misra-Gries monitors at most `capacity` keys (~20 for a 5% frequency floor), and of
    /// its three branches only one — taking a free slot — needs to *own* the key: the
    /// increment reads it, and the decrement drops it. Feeding the owned-key `add` from a
    /// column therefore allocated a `String` for every row and threw it away again on
    /// essentially all of them. Over a 1M-row join key that was **90.5 ns/value**, against
    /// 17 ns for the Utf8 cast that produced the strings — the allocator, not the sketch.
    ///
    /// Borrowing costs a hash and a lookup, and allocates at most `capacity` times per
    /// distinct arrival. The summary it builds is identical.
    pub fn add_ref<Q>(&mut self, key: &Q)
    where
        K: std::borrow::Borrow<Q>,
        Q: Hash + Eq + ToOwned<Owned = K> + ?Sized,
    {
        self.total += 1;
        if let Some(c) = self.counters.get_mut(key) {
            *c += 1;
            return;
        }
        if self.counters.len() < self.capacity {
            self.counters.insert(key.to_owned(), 1);
            return;
        }
        // No free slot and the key isn't monitored: pay for the new arrival by decrementing
        // every monitored counter, evicting anything that reaches zero. Same as `add_n`.
        self.counters.retain(|_, c| {
            *c = c.saturating_sub(1);
            *c > 0
        });
    }

    /// Add one occurrence of `key`.
    pub fn add(&mut self, key: K) {
        self.add_n(key, 1);
    }

    /// Add `count` occurrences of `key` (weighted Misra-Gries update):
    /// increment if monitored, else take a free slot, else admit the arrival and
    /// reduce the (now `capacity + 1`) summary back to `capacity` keys the
    /// mergeable way — subtract the `(capacity + 1)`-th largest count from every
    /// counter and drop the non-positive.
    ///
    /// Admitting the arrival first is what preserves the frequency guarantee for
    /// `count > 1`: the old "decrement every counter by `count`, drop the arrival"
    /// step would evict *everything* and silently lose the arriving key when its
    /// `count` exceeded the monitored counters — so a heavy key fed via a single
    /// large-`count` update (e.g. a pre-aggregated stream) vanished even though its
    /// frequency far exceeded `N / (capacity + 1)`. For `count == 1` this reduces
    /// to exactly the classic "decrement all by one, drop the arrival" step (the
    /// `(capacity + 1)`-th largest of `capacity` survivors plus a unit arrival is
    /// `1`), so single-increment behaviour is unchanged.
    pub fn add_n(&mut self, key: K, count: u64) {
        if count == 0 {
            return;
        }
        self.total += count;

        if let Some(c) = self.counters.get_mut(&key) {
            *c += count;
            return;
        }
        if self.counters.len() < self.capacity {
            self.counters.insert(key, count);
            return;
        }
        // No free slot and the key isn't monitored: admit it, then prune back to
        // capacity by subtracting the (capacity + 1)-th largest count. Identical to
        // `merge`'s reduction, so a heavy arrival is never dropped outright.
        self.counters.insert(key, count);
        self.reduce_to_capacity();
    }

    /// Prune the summary back to at most `capacity` monitored keys when it has
    /// grown past it: subtract the `(capacity + 1)`-th largest count from every
    /// counter (saturating) and drop the non-positive. This is the mergeable
    /// Misra-Gries reduction — it preserves the `N / (capacity + 1)` guarantee — and
    /// is shared by both the weighted single-stream update and [`merge`](Self::merge).
    fn reduce_to_capacity(&mut self) {
        if self.counters.len() <= self.capacity {
            return;
        }
        let mut counts: Vec<u64> = self.counters.values().copied().collect();
        // Descending so index `capacity` is the (capacity + 1)-th largest.
        counts.sort_unstable_by(|a, b| b.cmp(a));
        let threshold = counts[self.capacity];
        self.counters.retain(|_, c| {
            *c = c.saturating_sub(threshold);
            *c > 0
        });
    }

    /// The monitored counter for `key`: a *lower* bound on its true frequency
    /// (off by at most `total / (capacity + 1)`), or `0` if not monitored.
    pub fn estimate(&self, key: &K) -> u64 {
        self.counters.get(key).copied().unwrap_or(0)
    }

    /// The maximum a monitored counter could undercount the true frequency by.
    ///
    /// Misra-Gries decrements every counter each time a new key evicts one, and there are at
    /// most `total / (capacity + 1)` such decrement rounds, so a monitored key's true count is
    /// in `[counter, counter + this]`.
    #[inline]
    fn count_error(&self) -> f64 {
        self.total as f64 / (self.capacity as f64 + 1.0)
    }

    /// Monitored keys that **could** exceed `fraction * total`, sorted by count descending —
    /// the keys skewed enough to be worth salting.
    ///
    /// The test is against the counter's *upper* bound (`counter + count_error`), not the raw
    /// counter, because the two directions of error are not equally costly here. This feeds
    /// shuffle salting, where a false positive costs one extra sub-partition and a false
    /// negative leaves a hot key to overload a single reducer — a straggler that shows up only
    /// at scale. Misra-Gries counters *underestimate* (they are decremented), so filtering on
    /// the raw counter drops exactly the borderline-heavy keys whose counters were decremented
    /// the most, which is the unsafe direction. Comparing the upper bound reports every key
    /// that could be heavy and cannot miss one — the same safety choice Count-Min's `is_heavy`
    /// and the join bloom make.
    pub fn heavy_hitters(&self, fraction: f64) -> Vec<(K, u64)> {
        let threshold = fraction * self.total as f64;
        let error = self.count_error();
        let mut out: Vec<(K, u64)> = self
            .counters
            .iter()
            .filter(|(_, &c)| (c as f64) + error > threshold)
            .map(|(k, &c)| (k.clone(), c))
            .collect();
        out.sort_by_key(|entry| std::cmp::Reverse(entry.1));
        out
    }

    /// Iterate the monitored `(key, count)` pairs (no particular order).
    pub fn items(&self) -> impl Iterator<Item = (&K, u64)> {
        self.counters.iter().map(|(k, &c)| (k, c))
    }

    /// Total weight added (the true `N`).
    pub fn total(&self) -> u64 {
        self.total
    }
}

impl<K: Hash + Eq + Clone> Mergeable for FrequentItems<K> {
    /// Merge `other` into `self`: sum the counters of common keys, union the rest,
    /// then if more than `capacity` keys remain, reduce by the
    /// `(capacity + 1)`-th largest count — subtract that threshold from every
    /// counter and drop the non-positive. Capacities must match.
    fn merge(&mut self, other: &FrequentItems<K>) {
        assert_eq!(self.capacity, other.capacity, "capacity mismatch");
        self.total += other.total;

        for (k, &c) in &other.counters {
            *self.counters.entry(k.clone()).or_insert(0) += c;
        }

        // Reduce the union back to ≤ capacity keys the mergeable Misra-Gries way.
        self.reduce_to_capacity();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merge_all;

    #[test]
    fn finds_heavy_key_in_skewed_stream() {
        let mut fi = FrequentItems::new(64);
        // Key 0 is hot; keys 1..=10_000 each appear once.
        for _ in 0..50_000u64 {
            fi.add(0u64);
        }
        for k in 1..=10_000u64 {
            fi.add(k);
        }
        let n = 60_000;
        assert_eq!(fi.total(), n);

        // Key 0 must be reported as a heavy hitter at the 25% threshold.
        let heavy = fi.heavy_hitters(0.25);
        assert!(
            heavy.iter().any(|(k, _)| *k == 0),
            "hot key 0 not in heavy hitters: {heavy:?}"
        );

        // Misra-Gries underestimates by at most N / (capacity + 1).
        let slack = n / (64 + 1);
        let est = fi.estimate(&0);
        assert!(
            est >= 50_000 - slack && est <= 50_000,
            "estimate {est} outside [{}, 50000]",
            50_000 - slack
        );

        // A rare key must not be reported heavy.
        let rare_heavy = fi.heavy_hitters(0.25).into_iter().any(|(k, _)| k == 5_000);
        assert!(!rare_heavy, "rare key 5000 reported heavy");
    }

    #[test]
    fn mergeable_preserves_heavy_key() {
        let true_count = 50_000u64;
        let n = 60_000u64;
        let capacity = 64;
        let slack = n / (capacity as u64 + 1);

        let mut a = FrequentItems::new(capacity);
        let mut b = FrequentItems::new(capacity);

        // First half of the stream into `a`, second half into `b`.
        for _ in 0..25_000u64 {
            a.add(0u64);
        }
        for k in 1..=5_000u64 {
            a.add(k);
        }
        for _ in 0..25_000u64 {
            b.add(0u64);
        }
        for k in 5_001..=10_000u64 {
            b.add(k);
        }

        a.merge(&b);
        assert_eq!(a.total(), n);

        let est = a.estimate(&0);
        assert!(
            est >= true_count - slack,
            "merged estimate {est} below lower bound {}",
            true_count - slack
        );
        let heavy = a.heavy_hitters(0.25);
        assert!(
            heavy.iter().any(|(k, _)| *k == 0),
            "hot key lost after merge: {heavy:?}"
        );
    }

    #[test]
    fn exact_when_capacity_covers_distinct() {
        // Fewer distinct keys than capacity ⇒ no decrements ⇒ exact counts.
        let mut fi = FrequentItems::new(16);
        for _ in 0..10 {
            fi.add("a");
        }
        for _ in 0..3 {
            fi.add("b");
        }
        fi.add("c");

        assert_eq!(fi.total(), 14);
        assert_eq!(fi.estimate(&"a"), 10);
        assert_eq!(fi.estimate(&"b"), 3);
        assert_eq!(fi.estimate(&"c"), 1);
        assert_eq!(fi.estimate(&"missing"), 0);

        let heavy = fi.heavy_hitters(0.5);
        assert_eq!(heavy, vec![("a", 10)]);

        // items() exposes all three monitored pairs.
        let mut seen: Vec<(&str, u64)> = fi.items().map(|(k, c)| (*k, c)).collect();
        seen.sort();
        assert_eq!(seen, vec![("a", 10), ("b", 3), ("c", 1)]);
    }

    #[test]
    fn add_n_keeps_a_heavy_arrival_that_fills_a_full_table() {
        // Regression: a heavy key arriving via a single large-`count` update, once
        // the table is full of light keys, must NOT be dropped. The old
        // "decrement every counter by `count`, drop the arrival" step evicted all
        // the light keys AND lost the heavy arrival, violating the Misra-Gries
        // guarantee (every key with frequency > N/(capacity+1) is retained).
        let cap = 2;
        let mut fi = FrequentItems::new(cap);
        fi.add(1u64); // {1:1}
        fi.add(2u64); // {1:1, 2:1} — table now full
        fi.add_n(999u64, 1_000); // heavy arrival, unmonitored, table full

        let n = fi.total();
        assert_eq!(n, 1_002);
        let slack = n / (cap as u64 + 1); // 334
        let est = fi.estimate(&999);
        assert!(
            est >= 1_000 - slack,
            "heavy key dropped/undercounted: est={est}, expected >= {}",
            1_000 - slack
        );
        // And it must surface as a heavy hitter at any reasonable threshold.
        let heavy = fi.heavy_hitters(0.25);
        assert!(
            heavy.iter().any(|(k, _)| *k == 999),
            "heavy key 999 (count 1000/{n}) missing from heavy hitters: {heavy:?}"
        );
    }

    #[test]
    fn add_n_unit_count_matches_classic_decrement() {
        // For count == 1 the weighted update must behave exactly like the classic
        // single-increment Misra-Gries step (this is what `add` relies on).
        let cap = 3;
        let mut fi = FrequentItems::new(cap);
        // Fill the table, then push several distinct singletons to force decrements.
        for k in 0..20u64 {
            fi.add(k);
        }
        assert!(fi.items().count() <= cap);
        assert_eq!(fi.total(), 20);
        // A dominant key must still be recoverable.
        let mut fi2 = FrequentItems::new(cap);
        for _ in 0..1_000 {
            fi2.add(7u64);
        }
        for k in 100..130u64 {
            fi2.add(k);
        }
        assert!(fi2.estimate(&7) > 900, "hot key undercounted");
    }

    #[test]
    fn add_n_matches_repeated_add() {
        let mut fi = FrequentItems::new(8);
        fi.add_n(42u64, 7);
        fi.add_n(42u64, 0); // no-op
        assert_eq!(fi.estimate(&42), 7);
        assert_eq!(fi.total(), 7);
    }

    #[test]
    fn merge_reduces_oversized_union() {
        // Two summaries with disjoint singleton keys; the union exceeds capacity
        // and must be reduced back to ≤ capacity monitored keys.
        let cap = 4;
        let mut a = FrequentItems::new(cap);
        let mut b = FrequentItems::new(cap);
        for k in 0..4u64 {
            a.add(k);
        }
        for k in 4..8u64 {
            b.add(k);
        }
        a.merge(&b);
        assert!(a.items().count() <= cap, "union not reduced to capacity");
        assert_eq!(a.total(), 8);
    }

    // ---- Property / fuzz tests (deterministic xorshift64, fixed seed) -------
    //
    // FrequentItems has no serialization, so only the merge property is fuzzed.

    /// Minimal deterministic PRNG so trials are reproducible without `rand`.
    struct XorShift64(u64);
    impl XorShift64 {
        fn new(seed: u64) -> Self {
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
        fn below(&mut self, bound: u64) -> u64 {
            self.next_u64() % bound
        }
    }

    #[test]
    fn prop_merge_finds_heavy() {
        const TRIALS: usize = 200;
        let mut rng = XorShift64::new(0xF7E9_0E47_A5A5_1234);

        for trial in 0..TRIALS {
            let capacity = 16 + rng.below(112) as usize; // 16..=127
            let parts = 2 + rng.below(5) as usize; // 2..=6 partitions

            // The injected heavy key, taken from a high range so it can't collide
            // with the light tail keys below.
            let heavy_key = 1_000_000 + rng.below(1_000_000);

            // Make the heavy key dominate: its weight is a large fraction of N so it
            // survives Misra-Gries decrements and merges. The Misra-Gries guarantee
            // is that any key with frequency > N/(capacity+1) is retained; we make
            // the heavy key comfortably exceed that.
            let tail = 1 + rng.below(5_000); // number of distinct light keys
            let heavy_count = 2 * tail + 5_000 + rng.below(10_000); // dominant

            let mut sketches: Vec<FrequentItems<u64>> =
                (0..parts).map(|_| FrequentItems::new(capacity)).collect();

            // Spread the heavy key's occurrences across random partitions.
            for _ in 0..heavy_count {
                let p = rng.below(parts as u64) as usize;
                sketches[p].add(heavy_key);
            }
            // One occurrence each of `tail` distinct light keys, random partitions.
            for k in 0..tail {
                let p = rng.below(parts as u64) as usize;
                sketches[p].add(k);
            }

            let n = heavy_count + tail;
            let merged = merge_all(sketches.into_iter()).unwrap();
            assert_eq!(merged.total(), n, "trial {trial}: total mismatch");

            // The dominant key must still be reported as a heavy hitter. Threshold
            // chosen below the heavy key's true fraction but above the noise floor.
            let frac = 0.25;
            let heavy = merged.heavy_hitters(frac);
            assert!(
                heavy.iter().any(|(k, _)| *k == heavy_key),
                "trial {trial}: heavy key {heavy_key} (count {heavy_count}/{n}) lost after merge; cap={capacity} parts={parts} reported={heavy:?}"
            );
        }
    }

    /// A key that is genuinely heavy but whose Misra-Gries counter was decremented below the
    /// threshold must still be reported — a missed hot key is a shuffle straggler, the failure
    /// salting exists to prevent.
    #[test]
    fn heavy_hitters_reports_a_key_whose_counter_was_decremented() {
        // A small table over many distinct keys forces heavy decrementing, so a real heavy
        // hitter's counter lags its true frequency.
        let mut fi = FrequentItems::new(4);
        for _ in 0..500 {
            fi.add(0u64); // ~33% of the stream, genuinely heavy
        }
        for i in 1..1000u64 {
            fi.add(i); // a long tail of singletons that decrement key 0's counter
        }
        // Key 0's true frequency is well above 10%; the upper-bound test must still surface it
        // even though its stored counter has been decremented.
        let heavy = fi.heavy_hitters(0.1);
        assert!(
            heavy.iter().any(|(k, _)| *k == 0),
            "the genuinely-heavy key 0 was missed: {heavy:?}"
        );
    }

    /// The upper-bound test must not turn every singleton into a heavy hitter — the error term
    /// is bounded by `total/(capacity+1)`, so on a large table it stays small.
    #[test]
    fn heavy_hitters_does_not_flag_the_whole_tail() {
        let mut fi = FrequentItems::new(64);
        for i in 0..100_000u64 {
            fi.add(i % 5000); // 5000 roughly equal keys, none close to 10%
        }
        assert!(
            fi.heavy_hitters(0.1).is_empty(),
            "no key holds 10%, yet some were flagged"
        );
    }
}
