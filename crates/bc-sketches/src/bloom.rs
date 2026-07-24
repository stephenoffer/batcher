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
    /// The textbook formulas — `m = -n·ln p / (ln 2)²` bits, `k = (m/n)·ln 2` hashes — are
    /// derived for a *real-valued* `k`, and a filter has to use an integer one. Rounding `k`
    /// after fixing `m` silently misses the target: at `n = 1000, p = 0.01` it gives
    /// `m = 9586, k = 7`, whose actual rate is `(1 - e^(-kn/m))^k = 1.01%` — over the rate
    /// that was asked for. A runtime join filter sized that way passes slightly more probe
    /// rows than the cost model was promised.
    ///
    /// So the integer `k` is chosen first, then `m` is solved for *that* `k`:
    ///
    /// ``p = (1 - e^(-kn/m))^k  ⟹  m = -k·n / ln(1 - p^(1/k))``
    ///
    /// Both integers bracketing the real optimum are evaluated and the one needing fewer bits
    /// wins, so the filter meets its target rate at the smallest size that can. Clamps to sane
    /// minimums so a tiny or empty build side still yields a usable (if generous) filter.
    pub fn with_params(expected_items: u64, fp_rate: f64) -> Self {
        let n = expected_items.max(1) as f64;
        let p = fp_rate.clamp(1e-6, 0.5);
        let ideal = -p.ln() / std::f64::consts::LN_2; // the real-valued optimum for k
        let (mut best_bits, mut best_k) = (f64::INFINITY, 1u32);
        for k in [ideal.floor().max(1.0) as u32, ideal.ceil().max(1.0) as u32] {
            let bits = Self::bits_for(n, p, k);
            if bits < best_bits {
                best_bits = bits;
                best_k = k;
            }
        }
        Self::new(best_bits.ceil().max(64.0) as u64, best_k)
    }

    /// Bits needed so that `k` hashes over `n` items achieve false-positive rate `p`.
    ///
    /// Inverts `p = (1 - e^(-kn/m))^k`. A `k` too small for the target (`p^(1/k)` reaching 1)
    /// cannot achieve it at any size, and is reported as infinitely expensive so the caller
    /// picks the other candidate.
    fn bits_for(n: f64, p: f64, k: u32) -> f64 {
        let root = p.powf(1.0 / k as f64);
        if root >= 1.0 {
            return f64::INFINITY;
        }
        -(k as f64) * n / (1.0 - root).ln()
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
    //
    // `h1` is the **whole** 64-bit hash, not its low half. Truncating it to 32 bits caps the
    // reachable index at `2^32` for the `i = 0` probe, so every bit above 512 MB of filter was
    // only ever reachable through the `i·h2` terms — the addressable space collapses exactly
    // when the build side is large enough to need it. The full width costs nothing and is
    // uniform over `num_bits` to within the negligible modulo bias of a 64-bit value.
    fn positions(hash: u64, num_bits: u64, num_hashes: u32) -> impl Iterator<Item = (usize, u32)> {
        let h1 = hash;
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
        // `num_hashes == 0` is the dangerous one: it makes `positions` empty, so
        // `contains_hash`'s `.all()` is vacuously true and the filter matches every
        // key — a silently unsound join filter rather than a decode error. `new()`
        // clamps it away, so only a corrupt or foreign blob can carry it here.
        if num_bits == 0
            || num_hashes == 0
            || num_bits % 64 != 0
            || words.len() != (num_bits / 64) as usize * 8
        {
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

    /// Regression: `from_bytes` validated `num_bits` but not `num_hashes`. A blob
    /// carrying `num_hashes == 0` deserialized into a filter whose `positions`
    /// iterator is empty, so `contains_hash`'s `.all()` was vacuously true — a
    /// join filter that matches *every* key. `new()` clamps to `max(1)`, so this
    /// shape can only arrive from a corrupt or foreign blob over the shuffle,
    /// which is exactly the input `from_bytes` exists to police.
    #[test]
    fn rejects_zero_num_hashes() {
        let mut bytes = BloomFilter::with_params(1_000, 0.01).to_bytes();
        bytes[8..12].copy_from_slice(&0u32.to_le_bytes());
        assert!(
            BloomFilter::from_bytes(&bytes).is_none(),
            "num_hashes == 0 must be rejected, not become a match-everything filter"
        );
    }

    /// The sizing must actually **meet** the false-positive rate it was asked for.
    ///
    /// Fixing `m` from the real-valued optimum and then rounding `k` overshoots: at
    /// `n = 1000, p = 1%` it produced `m = 9586, k = 7`, whose analytic rate is 1.008% — a
    /// filter that quietly passes more probe rows than the join cost model assumed. Solving
    /// for `m` given the integer `k` closes it. Checked analytically (the closed form) and
    /// empirically (a measured miss rate over keys never added).
    #[test]
    fn achieves_the_requested_false_positive_rate() {
        for (n, p) in [
            (1_000u64, 0.01),
            (10_000, 0.01),
            (1_000, 0.001),
            (50_000, 0.05),
        ] {
            let f = BloomFilter::with_params(n, p);
            let k = f.num_hashes() as f64;
            let analytic = (1.0 - (-k * n as f64 / f.num_bits() as f64).exp()).powf(k);
            assert!(
                analytic <= p * 1.000_001,
                "n={n} p={p}: analytic rate {analytic} exceeds the target"
            );
        }

        let mut f = BloomFilter::with_params(10_000, 0.01);
        for i in 0..10_000u64 {
            f.add(&i);
        }
        let probes = 200_000u64;
        let hits = (0..probes).filter(|i| f.contains(&(i + 1_000_000))).count();
        let measured = hits as f64 / probes as f64;
        assert!(measured < 0.013, "measured false-positive rate {measured}");
    }

    /// A filter larger than 512 MB must still address all of its bits.
    ///
    /// With `h1` truncated to 32 bits the first probe could only ever land in the first
    /// `2^32` bit positions, so the tail of a large filter was reachable by fewer hash
    /// functions than it was sized for — the false-positive rate degrades exactly where the
    /// build side is big enough to need the filter most.
    #[test]
    fn probes_reach_the_whole_address_space_of_a_large_filter() {
        let num_bits = (1u64 << 33) + 64; // > 2^32 bits
        let mut high = 0usize;
        for i in 0..20_000u64 {
            let (word, _) = BloomFilter::positions(crate::hash_one(&i), num_bits, 1)
                .next()
                .expect("one probe");
            if (word as u64) * 64 >= (1u64 << 32) {
                high += 1;
            }
        }
        assert!(
            high > 8_000,
            "only {high}/20000 first probes landed above 2^32 bits; expected about half"
        );
    }
}
