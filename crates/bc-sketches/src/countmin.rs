//! Count-Min — frequency (heavy-hitter) estimation.
//!
//! Estimates how often a key occurs in `depth × width` counters, never
//! *under*-counting (collisions only ever inflate). The optimizer uses it to find
//! **skewed keys** before a shuffle: a key whose estimated frequency dwarfs the
//! average gets *salted* (fanned across sub-partitions) so one hot key can't
//! overload a single reducer. Mergeable (counters add element-wise), so per-node
//! frequency sketches combine into the global picture.

use std::hash::Hash;

use crate::{hash_one, Mergeable};

/// A Count-Min frequency sketch over hashable keys.
#[derive(Clone)]
pub struct CountMinSketch {
    width: usize,
    depth: usize,
    counts: Vec<u64>, // depth × width, row-major
    total: u64,
}

impl CountMinSketch {
    /// Create a sketch with `depth` rows of `width` counters. More width → less
    /// over-estimation; more depth → lower failure probability.
    pub fn new(width: usize, depth: usize) -> Self {
        assert!(width >= 1 && depth >= 1, "dimensions must be >= 1");
        Self {
            width,
            depth,
            counts: vec![0; width * depth],
            total: 0,
        }
    }

    /// Size the sketch for additive error ≤ `epsilon · N` with probability
    /// `1 − delta` (`width = ⌈e/ε⌉`, `depth = ⌈ln(1/δ)⌉`).
    pub fn with_error(epsilon: f64, delta: f64) -> Self {
        assert!(epsilon > 0.0 && (0.0..1.0).contains(&delta));
        let width = (std::f64::consts::E / epsilon).ceil() as usize;
        let depth = (1.0 / delta).ln().ceil().max(1.0) as usize;
        Self::new(width.max(1), depth)
    }

    /// Total weight added (the true `N`), used to judge "heavy" relative to average.
    pub fn total(&self) -> u64 {
        self.total
    }

    // The `i`-th row index for a key hash, via Kirsch–Mitzenmacher double hashing
    // (`h1 + i·h2`) — `depth` independent-enough functions from one 64-bit hash.
    fn index(&self, hash: u64, row: usize) -> usize {
        let h1 = hash as u32 as u64;
        let h2 = (hash >> 32) | 1; // odd → full period
        row * self.width + ((h1.wrapping_add((row as u64).wrapping_mul(h2))) as usize % self.width)
    }

    /// Add `count` occurrences of a pre-hashed key.
    pub fn add_hash(&mut self, hash: u64, count: u64) {
        self.total += count;
        for row in 0..self.depth {
            let idx = self.index(hash, row);
            self.counts[idx] += count;
        }
    }

    /// Add one occurrence of a hashable key.
    pub fn add<T: Hash + ?Sized>(&mut self, key: &T) {
        self.add_hash(hash_one(key), 1);
    }

    /// Estimate the frequency of a pre-hashed key (an upper bound on the truth).
    pub fn estimate_hash(&self, hash: u64) -> u64 {
        (0..self.depth)
            .map(|row| self.counts[self.index(hash, row)])
            .min()
            .unwrap_or(0)
    }

    /// Estimate the frequency of a hashable key.
    pub fn estimate<T: Hash + ?Sized>(&self, key: &T) -> u64 {
        self.estimate_hash(hash_one(key))
    }

    /// Whether a key's estimated frequency exceeds `fraction · N` — the test for
    /// "this key is skewed enough to salt".
    pub fn is_heavy<T: Hash + ?Sized>(&self, key: &T, fraction: f64) -> bool {
        self.total > 0 && (self.estimate(key) as f64) > fraction * self.total as f64
    }

    /// Serialize to a byte blob. Layout (all little-endian):
    /// `[width: u64][depth: u64][total: u64][counts: width·depth × u64]`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(24 + self.counts.len() * 8);
        out.extend_from_slice(&(self.width as u64).to_le_bytes());
        out.extend_from_slice(&(self.depth as u64).to_le_bytes());
        out.extend_from_slice(&self.total.to_le_bytes());
        for &c in &self.counts {
            out.extend_from_slice(&c.to_le_bytes());
        }
        out
    }

    /// Reconstruct from [`to_bytes`](Self::to_bytes). Returns `None` on malformed
    /// input (zero dimensions, or a length that doesn't match `width·depth`).
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let read_u64 = |off: usize| -> Option<u64> {
            let chunk: [u8; 8] = bytes.get(off..off + 8)?.try_into().ok()?;
            Some(u64::from_le_bytes(chunk))
        };
        let width = read_u64(0)? as usize;
        let depth = read_u64(8)? as usize;
        let total = read_u64(16)?;
        if width < 1 || depth < 1 {
            return None;
        }
        let n = width.checked_mul(depth)?;
        // Header (24 bytes) + n counters, each 8 bytes, and nothing more.
        if bytes.len() != 24 + n.checked_mul(8)? {
            return None;
        }
        let mut counts = Vec::with_capacity(n);
        for i in 0..n {
            counts.push(read_u64(24 + i * 8)?);
        }
        Some(Self {
            width,
            depth,
            counts,
            total,
        })
    }
}

impl Mergeable for CountMinSketch {
    /// Add counters element-wise. Dimensions must match.
    fn merge(&mut self, other: &CountMinSketch) {
        assert_eq!(
            (self.width, self.depth),
            (other.width, other.depth),
            "dimension mismatch"
        );
        for (a, b) in self.counts.iter_mut().zip(&other.counts) {
            *a += *b;
        }
        self.total += other.total;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_underestimates() {
        let mut cm = CountMinSketch::with_error(0.001, 0.01);
        for i in 0..100_000u64 {
            cm.add(&(i % 5000)); // each key appears 20 times
        }
        for k in 0..5000u64 {
            assert!(cm.estimate(&k) >= 20, "underestimated key {k}");
        }
        assert_eq!(cm.total(), 100_000);
    }

    #[test]
    fn estimate_close_for_heavy_hitters() {
        let mut cm = CountMinSketch::with_error(0.0005, 0.01);
        // One hot key plus a long tail.
        for _ in 0..50_000 {
            cm.add(&"hot");
        }
        for i in 0..50_000u64 {
            cm.add(&i);
        }
        let est = cm.estimate(&"hot");
        let err = (est - 50_000) as f64 / 50_000.0;
        assert!(err < 0.05, "hot key over-estimated by {err} (est {est})");
        assert!(cm.is_heavy(&"hot", 0.25));
        assert!(!cm.is_heavy(&7u64, 0.25));
    }

    #[test]
    fn bytes_roundtrip_preserves_estimates() {
        let mut cm = CountMinSketch::new(512, 4);
        for i in 0..20_000u64 {
            cm.add(&(i % 1000));
        }
        let bytes = cm.to_bytes();
        let back = CountMinSketch::from_bytes(&bytes).expect("valid blob");
        assert_eq!(back.total(), cm.total());
        for k in 0..1000u64 {
            assert_eq!(back.estimate(&k), cm.estimate(&k));
        }
    }

    #[test]
    fn from_bytes_rejects_malformed() {
        assert!(CountMinSketch::from_bytes(&[]).is_none());
        assert!(CountMinSketch::from_bytes(&[0; 23]).is_none()); // short header
                                                                 // Zero dimension.
        let mut bad = 0u64.to_le_bytes().to_vec();
        bad.extend_from_slice(&4u64.to_le_bytes());
        bad.extend_from_slice(&0u64.to_le_bytes());
        assert!(CountMinSketch::from_bytes(&bad).is_none());
        // Valid header but wrong number of counters.
        let mut wrong = 4u64.to_le_bytes().to_vec();
        wrong.extend_from_slice(&2u64.to_le_bytes()); // depth → expect 8 counters
        wrong.extend_from_slice(&0u64.to_le_bytes()); // total
        wrong.extend_from_slice(&[0u8; 8]); // only 1 counter present
        assert!(CountMinSketch::from_bytes(&wrong).is_none());
    }

    #[test]
    fn merge_adds_counts() {
        let mut a = CountMinSketch::new(2048, 4);
        let mut b = CountMinSketch::new(2048, 4);
        for _ in 0..1000 {
            a.add(&"x");
        }
        for _ in 0..1500 {
            b.add(&"x");
        }
        a.merge(&b);
        assert_eq!(a.total(), 2500);
        assert!(a.estimate(&"x") >= 2500);
    }

    // ---- Property / fuzz tests (deterministic xorshift64, fixed seed) -------

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
    fn prop_serialize_roundtrip() {
        const TRIALS: usize = 200;
        let mut rng = XorShift64::new(0xC0117_C0117_u64 ^ 0x9E37_79B9);

        for trial in 0..TRIALS {
            let width = 1 + rng.below(1_024) as usize; // 1..=1024
            let depth = 1 + rng.below(6) as usize; // 1..=6
            let n = rng.below(3_000); // up to ~3k adds
            let key_space = 1 + rng.below(500); // distinct key universe

            let mut cm = CountMinSketch::new(width, depth);
            let mut keys: Vec<u64> = Vec::new();
            for _ in 0..n {
                let key = rng.below(key_space);
                if keys.len() < 32 {
                    keys.push(key);
                }
                cm.add(&key);
            }

            let back = CountMinSketch::from_bytes(&cm.to_bytes()).expect("valid blob");
            assert_eq!(back.total(), cm.total(), "trial {trial}: total mismatch");
            // Estimates must be exactly equal after a faithful roundtrip.
            for k in 0..key_space {
                assert_eq!(
                    back.estimate(&k),
                    cm.estimate(&k),
                    "trial {trial}: width={width} depth={depth} key={k} estimate mismatch"
                );
            }
        }
    }

    #[test]
    fn prop_merge_adds() {
        const TRIALS: usize = 200;
        let mut rng = XorShift64::new(0xADD5_ADD5_ADD5_ADD5);

        for trial in 0..TRIALS {
            // Shared dimensions for both partitions (merge requires matching dims).
            let width = 256 + rng.below(1_024) as usize;
            let depth = 2 + rng.below(4) as usize;
            let key_space = 1 + rng.below(400);

            let mut a = CountMinSketch::new(width, depth);
            let mut b = CountMinSketch::new(width, depth);

            // Build the exact combined truth alongside the two sketches.
            let mut truth: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
            let na = rng.below(2_000);
            let nb = rng.below(2_000);
            for _ in 0..na {
                let k = rng.below(key_space);
                a.add(&k);
                *truth.entry(k).or_insert(0) += 1;
            }
            for _ in 0..nb {
                let k = rng.below(key_space);
                b.add(&k);
                *truth.entry(k).or_insert(0) += 1;
            }

            a.merge(&b);
            assert_eq!(a.total(), na + nb, "trial {trial}: total after merge");

            // Count-Min never under-counts: merged estimate ≥ true combined count.
            for (&k, &true_count) in &truth {
                let est = a.estimate(&k);
                assert!(
                    est >= true_count,
                    "trial {trial}: key={k} est={est} < true={true_count}"
                );
            }
        }
    }
}
