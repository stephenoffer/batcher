//! Bloom filter — approximate set membership for runtime join filters.
//!
//! A bloom filter answers "have I seen this key?" in a fixed bit array, with no
//! false negatives (a key that was added always tests present) and a tunable false
//! positive rate. The distributed join uses it for **sideways information passing**:
//! build a bloom over the small side's join keys, ship it to the large side's
//! mappers, and drop large-side rows whose key can't be in the small side *before*
//! they are shuffled across the network. Because there are no false negatives the
//! filter only ever removes provably-non-matching rows — the join result is
//! unchanged, only the bytes shuffled shrink.
//!
//! Mergeable (bit arrays of equal dimensions OR together), so per-partition blooms
//! built independently combine into the bloom of their union — the partial→combine
//! contract every sketch here shares.

use std::hash::Hash;

use crate::{hash_one, Mergeable};

/// An approximate-membership bloom filter over hashable keys.
#[derive(Clone, PartialEq, Eq)]
pub struct BloomFilter {
    bits: Vec<u64>,  // bit array, packed 64 bits per word
    num_bits: u64,   // logical bit count (>= 64; a multiple of 64)
    num_hashes: u32, // hash functions per key (>= 1)
}

impl BloomFilter {
    /// A bloom with `num_bits` bits (rounded up to a multiple of 64) and `num_hashes`
    /// hash functions. Prefer [`with_params`](Self::with_params) for a target error.
    pub fn new(num_bits: u64, num_hashes: u32) -> Self {
        let num_bits = num_bits.max(64).div_ceil(64) * 64;
        Self {
            bits: vec![0; (num_bits / 64) as usize],
            num_bits,
            num_hashes: num_hashes.max(1),
        }
    }

    /// Size a bloom for `expected_items` with false-positive rate `fp_rate`.
    ///
    /// Uses the standard optimum: `m = -n·ln p / (ln 2)²` bits and
    /// `k = (m/n)·ln 2` hashes. Clamps to sane minimums so a tiny/empty side still
    /// yields a usable (if generous) filter.
    pub fn with_params(expected_items: u64, fp_rate: f64) -> Self {
        let n = expected_items.max(1) as f64;
        let p = fp_rate.clamp(1e-6, 0.5);
        let ln2 = std::f64::consts::LN_2;
        let m = (-n * p.ln() / (ln2 * ln2)).ceil().max(64.0);
        let k = ((m / n) * ln2).round().max(1.0);
        Self::new(m as u64, k as u32)
    }

    /// Add a pre-hashed key.
    pub fn add_hash(&mut self, hash: u64) {
        for (word, bit) in Self::positions(hash, self.num_bits, self.num_hashes) {
            self.bits[word] |= 1u64 << bit;
        }
    }

    /// Add one hashable key.
    pub fn add<T: Hash + ?Sized>(&mut self, key: &T) {
        self.add_hash(hash_one(key));
    }

    /// Whether a pre-hashed key *may* be present: `false` is definitive (never
    /// added); `true` may be a false positive.
    pub fn contains_hash(&self, hash: u64) -> bool {
        Self::positions(hash, self.num_bits, self.num_hashes)
            .all(|(word, bit)| self.bits[word] & (1u64 << bit) != 0)
    }

    /// Whether a hashable key may be present (see [`contains_hash`](Self::contains_hash)).
    pub fn contains<T: Hash + ?Sized>(&self, key: &T) -> bool {
        self.contains_hash(hash_one(key))
    }

    pub fn num_bits(&self) -> u64 {
        self.num_bits
    }

    pub fn num_hashes(&self) -> u32 {
        self.num_hashes
    }

    // The `(word, bit)` positions a key maps to, via Kirsch–Mitzenmacher double
    // hashing (`h1 + i·h2`) — `num_hashes` independent-enough indices from one hash.
    // Takes the dimensions by value so it borrows nothing (callers mutate `bits`).
    fn positions(hash: u64, num_bits: u64, num_hashes: u32) -> impl Iterator<Item = (usize, u32)> {
        let h1 = hash as u32 as u64;
        let h2 = (hash >> 32) | 1; // odd → full period
        (0..num_hashes as u64).map(move |i| {
            let pos = h1.wrapping_add(i.wrapping_mul(h2)) % num_bits;
            ((pos / 64) as usize, (pos % 64) as u32)
        })
    }

    /// Serialize to bytes for shipping across the FFI / to distributed workers.
    /// Layout: `num_bits` (u64 LE), `num_hashes` (u32 LE), then the packed words.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(12 + self.bits.len() * 8);
        out.extend_from_slice(&self.num_bits.to_le_bytes());
        out.extend_from_slice(&self.num_hashes.to_le_bytes());
        for word in &self.bits {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out
    }

    /// Reconstruct a bloom from [`to_bytes`](Self::to_bytes); `None` if malformed.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 12 {
            return None;
        }
        let num_bits = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
        let num_hashes = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
        let words = &bytes[12..];
        if num_bits == 0 || num_bits % 64 != 0 || words.len() != (num_bits / 64) as usize * 8 {
            return None;
        }
        let bits = words
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Some(Self {
            bits,
            num_bits,
            num_hashes,
        })
    }
}

impl Mergeable for BloomFilter {
    /// OR the bit arrays — the union of two sets' blooms. Dimensions must match
    /// (same `num_bits`/`num_hashes`); a mismatch is a construction error.
    fn merge(&mut self, other: &Self) {
        assert_eq!(
            (self.num_bits, self.num_hashes),
            (other.num_bits, other.num_hashes),
            "cannot merge bloom filters with different dimensions"
        );
        for (a, b) in self.bits.iter_mut().zip(&other.bits) {
            *a |= *b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merge_all;

    #[test]
    fn no_false_negatives() {
        let mut bloom = BloomFilter::with_params(1_000, 0.01);
        for i in 0..1_000u64 {
            bloom.add(&i);
        }
        for i in 0..1_000u64 {
            assert!(bloom.contains(&i), "added key {i} must test present");
        }
    }

    #[test]
    fn false_positive_rate_near_target() {
        let mut bloom = BloomFilter::with_params(10_000, 0.01);
        for i in 0..10_000u64 {
            bloom.add(&i);
        }
        // Probe 10k keys never added; the false-positive rate should be near 1%.
        let fp = (10_000..20_000u64).filter(|k| bloom.contains(k)).count();
        assert!(fp < 300, "false positives {fp} far above the ~1% target");
    }

    #[test]
    fn merge_is_union() {
        let parts = (0..4).map(|p| {
            let mut b = BloomFilter::with_params(40_000, 0.01);
            for i in 0..40_000u64 {
                if i % 4 == p {
                    b.add(&i);
                }
            }
            b
        });
        let merged = merge_all(parts).unwrap();
        for i in 0..40_000u64 {
            assert!(merged.contains(&i), "merged bloom must contain every key");
        }
    }

    #[test]
    fn round_trips_through_bytes() {
        let mut bloom = BloomFilter::with_params(500, 0.02);
        for i in 0..500u64 {
            bloom.add(&(i * 7));
        }
        let restored = BloomFilter::from_bytes(&bloom.to_bytes()).expect("valid bytes");
        assert!(bloom == restored);
        for i in 0..500u64 {
            assert!(restored.contains(&(i * 7)));
        }
    }

    #[test]
    fn rejects_malformed_bytes() {
        assert!(BloomFilter::from_bytes(&[0, 1, 2]).is_none());
        let mut ok = BloomFilter::with_params(64, 0.01).to_bytes();
        ok.push(0xFF); // trailing junk → wrong word count
        assert!(BloomFilter::from_bytes(&ok).is_none());
    }
}
