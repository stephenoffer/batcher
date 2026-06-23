//! DDSketch — relative-error quantile sketch (Masson, Rim, Lee).
//!
//! Unlike KLL's *rank* error, DDSketch gives a *relative-value* guarantee: every
//! quantile estimate `q̂` satisfies `|q̂ - q| / |q| ≤ α`. That bound holds no
//! matter how skewed or heavy-tailed the data is, which is exactly what the
//! optimizer wants when costing joins/aggregates over power-law distributions —
//! a p99 that's off by 1% of its *value* is far more useful than one off by 1%
//! of the *rank* when values span many orders of magnitude.
//!
//! Structure: a value `v > 0` maps to the logarithmic bucket
//! `i = ⌈log_γ(v)⌉` with `γ = (1+α)/(1-α)`; the bucket's representative value is
//! `2·γ^i / (γ+1)`, which lies within a factor `1±α` of every value in the
//! bucket. Buckets are counts in a `BTreeMap` (ordered iteration = quantile walk).
//! Zeros are counted separately (log is undefined at 0), negatives live in a
//! mirror map keyed by `|v|`. Min/max are tracked exactly so `q=0`/`q=1` are precise.

use crate::Mergeable;
use std::collections::BTreeMap;

/// A DDSketch over `f64` values with a relative-accuracy guarantee `α`.
#[derive(Clone)]
pub struct DDSketch {
    alpha: f64,
    gamma: f64,
    ln_gamma: f64,
    /// Bucket counts for strictly positive values, keyed by log-bucket index.
    positive: BTreeMap<i32, u64>,
    /// Bucket counts for negative values, keyed by the log-bucket index of `|v|`.
    negative: BTreeMap<i32, u64>,
    /// Count of exact zeros (log is undefined there).
    zeros: u64,
    n: u64,
    min: f64,
    max: f64,
}

/// Default relative accuracy (1%).
const DEFAULT_ALPHA: f64 = 0.01;

impl Default for DDSketch {
    fn default() -> Self {
        Self::new(DEFAULT_ALPHA)
    }
}

impl DDSketch {
    /// Create an empty sketch with relative accuracy `alpha ∈ (0, 1)` (e.g. 0.01
    /// for 1%). Smaller `alpha` → tighter buckets and more memory.
    pub fn new(alpha: f64) -> Self {
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "alpha must be in (0, 1), got {alpha}"
        );
        let gamma = (1.0 + alpha) / (1.0 - alpha);
        Self {
            alpha,
            gamma,
            ln_gamma: gamma.ln(),
            positive: BTreeMap::new(),
            negative: BTreeMap::new(),
            zeros: 0,
            n: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Number of values seen.
    pub fn count(&self) -> u64 {
        self.n
    }

    /// True if no (non-NaN) value has been added.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Exact minimum / maximum seen (`None` if empty).
    pub fn min(&self) -> Option<f64> {
        (self.n > 0).then_some(self.min)
    }
    pub fn max(&self) -> Option<f64> {
        (self.n > 0).then_some(self.max)
    }

    /// The configured relative accuracy `α`.
    pub fn relative_accuracy(&self) -> f64 {
        self.alpha
    }

    /// Log-bucket index for a strictly positive magnitude. `⌈log_γ(v)⌉`.
    #[inline]
    fn index(&self, v: f64) -> i32 {
        (v.ln() / self.ln_gamma).ceil() as i32
    }

    /// Representative value of bucket `i` (the bucket's geometric "centre"): it is
    /// within a factor `1±α` of every magnitude that maps to `i`.
    #[inline]
    fn value_of(&self, i: i32) -> f64 {
        2.0 * self.gamma.powi(i) / (self.gamma + 1.0)
    }

    /// Add one value. NaN/±inf are ignored (they have no finite bucket).
    pub fn add(&mut self, v: f64) {
        self.add_n(v, 1);
    }

    /// Add `count` copies of `v`. NaN/±inf are ignored.
    pub fn add_n(&mut self, v: f64, count: u64) {
        if count == 0 || !v.is_finite() {
            return;
        }
        if v > 0.0 {
            let i = self.index(v);
            *self.positive.entry(i).or_insert(0) += count;
        } else if v < 0.0 {
            let i = self.index(-v);
            *self.negative.entry(i).or_insert(0) += count;
        } else {
            self.zeros += count;
        }
        self.n += count;
        if v < self.min {
            self.min = v;
        }
        if v > self.max {
            self.max = v;
        }
    }

    /// Approximate value at quantile `q ∈ [0, 1]` (`None` if empty). `q=0`/`q=1`
    /// return the exact min/max; otherwise the result is within relative error
    /// `α` of the true quantile.
    pub fn quantile(&self, q: f64) -> Option<f64> {
        if self.n == 0 {
            return None;
        }
        let q = q.clamp(0.0, 1.0);
        if q <= 0.0 {
            return Some(self.min);
        }
        if q >= 1.0 {
            return Some(self.max);
        }
        // Rank we are walking toward. `floor` matches the convention that the
        // value at quantile q is the smallest x whose cumulative count exceeds
        // q·(n-1), but a 0-based target over n items is fine to within one item.
        let target = q * (self.n - 1) as f64;
        let mut cum = 0u64;

        // Negative buckets, walked from most-negative (largest |v|) to least.
        for (&i, &c) in self.negative.iter().rev() {
            cum += c;
            if cum as f64 > target {
                return Some(-self.value_of(i));
            }
        }
        // Then exact zeros.
        cum += self.zeros;
        if self.zeros > 0 && cum as f64 > target {
            return Some(0.0);
        }
        // Then positive buckets, ascending.
        for (&i, &c) in self.positive.iter() {
            cum += c;
            if cum as f64 > target {
                return Some(self.value_of(i));
            }
        }
        Some(self.max)
    }

    /// Convenience: the median.
    pub fn median(&self) -> Option<f64> {
        self.quantile(0.5)
    }

    /// Approximate fraction of values ≤ `x`, in `[0, 1]` — the selectivity of
    /// `col <= x`. Returns 0 for an empty sketch.
    pub fn rank(&self, x: f64) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        let mut below = 0u64;
        if x >= 0.0 {
            // x ≥ 0: every negative and every zero is ≤ x.
            below += self.negative.values().sum::<u64>();
            below += self.zeros;
            // Positive buckets whose representative value ≤ x.
            for (&i, &c) in self.positive.iter() {
                if self.value_of(i) <= x {
                    below += c;
                } else {
                    break;
                }
            }
        } else {
            // x < 0: only sufficiently large-magnitude negatives are ≤ x.
            // Negative value -value_of(i) ≤ x  ⇔  value_of(i) ≥ -x.
            for (&i, &c) in self.negative.iter().rev() {
                if -self.value_of(i) <= x {
                    below += c;
                } else {
                    break;
                }
            }
        }
        below as f64 / self.n as f64
    }

    /// Serialize to a byte blob. Layout (all little-endian):
    /// `[alpha: f64][zeros: u64][n: u64][min: f64][max: f64]`
    /// `[pos_len: u64]{[index: i32][count: u64]}×pos_len`
    /// `[neg_len: u64]{[index: i32][count: u64]}×neg_len`.
    /// `gamma`/`ln_gamma` are derived from `alpha` on load.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.alpha.to_le_bytes());
        out.extend_from_slice(&self.zeros.to_le_bytes());
        out.extend_from_slice(&self.n.to_le_bytes());
        out.extend_from_slice(&self.min.to_le_bytes());
        out.extend_from_slice(&self.max.to_le_bytes());
        for map in [&self.positive, &self.negative] {
            out.extend_from_slice(&(map.len() as u64).to_le_bytes());
            for (&i, &c) in map.iter() {
                out.extend_from_slice(&i.to_le_bytes());
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        out
    }

    /// Reconstruct from [`to_bytes`](Self::to_bytes). Returns `None` on truncated
    /// or otherwise malformed input.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut c = Cursor::new(bytes);
        let alpha = c.f64()?;
        if !(alpha > 0.0 && alpha < 1.0) {
            return None;
        }
        let zeros = c.u64()?;
        let n = c.u64()?;
        let min = c.f64()?;
        let max = c.f64()?;
        let mut maps: [BTreeMap<i32, u64>; 2] = [BTreeMap::new(), BTreeMap::new()];
        for map in maps.iter_mut() {
            let len = c.u64()? as usize;
            for _ in 0..len {
                let i = c.i32()?;
                let count = c.u64()?;
                // Duplicate keys would silently lose counts → reject.
                if map.insert(i, count).is_some() {
                    return None;
                }
            }
        }
        if !c.is_done() {
            return None; // trailing garbage → reject
        }
        let [positive, negative] = maps;
        // Consistency: bucket counts + zeros must equal n.
        let sum: u64 = positive.values().sum::<u64>() + negative.values().sum::<u64>() + zeros;
        if sum != n {
            return None;
        }
        let gamma = (1.0 + alpha) / (1.0 - alpha);
        Some(Self {
            alpha,
            gamma,
            ln_gamma: gamma.ln(),
            positive,
            negative,
            zeros,
            n,
            min,
            max,
        })
    }
}

/// Minimal little-endian read cursor that returns `None` instead of panicking on
/// short reads.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let end = self.pos.checked_add(N)?;
        let slice = self.bytes.get(self.pos..end)?;
        self.pos = end;
        slice.try_into().ok()
    }

    fn u64(&mut self) -> Option<u64> {
        self.take::<8>().map(u64::from_le_bytes)
    }

    fn i32(&mut self) -> Option<i32> {
        self.take::<4>().map(i32::from_le_bytes)
    }

    fn f64(&mut self) -> Option<f64> {
        self.take::<8>().map(f64::from_le_bytes)
    }

    fn is_done(&self) -> bool {
        self.pos == self.bytes.len()
    }
}

impl Mergeable for DDSketch {
    /// Add bucket counts element-wise. `alpha` (hence `gamma`) must match — merging
    /// sketches with different bucket geometries is meaningless.
    fn merge(&mut self, other: &DDSketch) {
        assert_eq!(
            self.alpha, other.alpha,
            "alpha mismatch: {} vs {}",
            self.alpha, other.alpha
        );
        if other.n == 0 {
            return;
        }
        for (&i, &c) in other.positive.iter() {
            *self.positive.entry(i).or_insert(0) += c;
        }
        for (&i, &c) in other.negative.iter() {
            *self.negative.entry(i).or_insert(0) += c;
        }
        self.zeros += other.zeros;
        self.n += other.n;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merge_all;

    // Deterministic shuffle so insertion order isn't sorted.
    fn shuffled(n: u64) -> Vec<f64> {
        let mut v: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut state = 0x1234_5678u64;
        for i in (1..v.len()).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let j = (state as usize) % (i + 1);
            v.swap(i, j);
        }
        v
    }

    #[test]
    fn quantiles_within_relative_error() {
        let n = 100_000u64;
        let alpha = 0.01;
        let mut s = DDSketch::new(alpha);
        // Use 1..=n so true values are strictly positive (relative error is
        // undefined at 0).
        for x in shuffled(n) {
            s.add(x + 1.0);
        }
        assert_eq!(s.count(), n);
        for &q in &[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] {
            let est = s.quantile(q).unwrap();
            // True value at quantile q over 1..=n.
            let truth = q * (n - 1) as f64 + 1.0;
            let rel = (est - truth).abs() / truth;
            assert!(
                rel <= alpha + 1e-9,
                "q={q} est={est} truth={truth} rel-error={rel} > alpha={alpha}"
            );
        }
    }

    #[test]
    fn handles_negatives_and_zeros() {
        let mut s = DDSketch::new(0.01);
        // Symmetric distribution -1000..=1000 including 0.
        for x in -1000i64..=1000 {
            s.add(x as f64);
        }
        assert_eq!(s.count(), 2001);
        // Median of a symmetric set centred at 0 is ≈ 0.
        let med = s.median().unwrap();
        assert!(med.abs() <= 1.0, "median {med} not near 0");
        // p10 ≈ -800, p90 ≈ +800 within relative error of the magnitude.
        let p90 = s.quantile(0.9).unwrap();
        assert!(p90 > 0.0, "p90 {p90} should be positive");
        let truth90 = 800.0;
        assert!(
            (p90 - truth90).abs() / truth90 <= 0.01 + 1e-9,
            "p90 {p90} vs {truth90}"
        );
        let p10 = s.quantile(0.1).unwrap();
        assert!(p10 < 0.0, "p10 {p10} should be negative");
        let truth10 = -800.0;
        assert!(
            (p10 - truth10).abs() / truth10.abs() <= 0.01 + 1e-9,
            "p10 {p10} vs {truth10}"
        );
    }

    #[test]
    fn min_max_exact() {
        let mut s = DDSketch::new(0.01);
        for x in shuffled(50_000) {
            s.add(x + 1.0);
        }
        assert_eq!(s.min(), Some(1.0));
        assert_eq!(s.max(), Some(50_000.0));
        assert_eq!(s.quantile(0.0), Some(1.0));
        assert_eq!(s.quantile(1.0), Some(50_000.0));
    }

    #[test]
    fn rank_is_selectivity() {
        let mut s = DDSketch::new(0.01);
        for x in shuffled(100_000) {
            s.add(x + 1.0);
        }
        // ~30% of values (1..=100000) are ≤ 30_000.
        let sel = s.rank(30_000.0);
        assert!((sel - 0.3).abs() < 0.02, "selectivity {sel}");
        // rank below everything → ~0, above everything → 1.
        assert!(s.rank(0.0) < 0.001, "rank(0)={}", s.rank(0.0));
        assert_eq!(s.rank(200_000.0), 1.0);
    }

    #[test]
    fn merge_matches_single() {
        let n = 100_000u64;
        let alpha = 0.01;
        let vals = shuffled(n);
        let mut whole = DDSketch::new(alpha);
        for &x in &vals {
            whole.add(x + 1.0);
        }
        let mut a = DDSketch::new(alpha);
        let mut b = DDSketch::new(alpha);
        for (i, &x) in vals.iter().enumerate() {
            if i % 2 == 0 {
                a.add(x + 1.0);
            } else {
                b.add(x + 1.0);
            }
        }
        a.merge(&b);
        assert_eq!(a.count(), n);
        // Element-wise bucket merge is exact: every quantile equals the single-pass.
        for &q in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0] {
            assert_eq!(
                a.quantile(q),
                whole.quantile(q),
                "q={q}: merge must equal single-pass exactly"
            );
        }
    }

    #[test]
    fn merge_all_partitions() {
        let n = 60_000u64;
        let alpha = 0.01;
        let vals = shuffled(n);
        let mut whole = DDSketch::new(alpha);
        for &x in &vals {
            whole.add(x + 1.0);
        }
        let parts = (0..4).map(|p| {
            let mut s = DDSketch::new(alpha);
            for (i, &x) in vals.iter().enumerate() {
                if i % 4 == p {
                    s.add(x + 1.0);
                }
            }
            s
        });
        let merged = merge_all(parts).unwrap();
        assert_eq!(merged.count(), n);
        for &q in &[0.25, 0.5, 0.75] {
            assert_eq!(merged.quantile(q), whole.quantile(q));
        }
    }

    #[test]
    #[should_panic(expected = "alpha mismatch")]
    fn merge_rejects_alpha_mismatch() {
        let mut a = DDSketch::new(0.01);
        let b = DDSketch::new(0.02);
        a.add(1.0);
        a.merge(&b);
    }

    #[test]
    fn bytes_roundtrip_preserves_quantiles() {
        let mut s = DDSketch::new(0.01);
        for x in shuffled(100_000) {
            s.add(x - 50_000.0); // span negatives, zero, positives
        }
        let bytes = s.to_bytes();
        let back = DDSketch::from_bytes(&bytes).expect("valid blob");
        assert_eq!(back.count(), s.count());
        assert_eq!(back.min(), s.min());
        assert_eq!(back.max(), s.max());
        for &q in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            assert_eq!(back.quantile(q), s.quantile(q));
        }
        assert_eq!(back.rank(0.0), s.rank(0.0));
    }

    #[test]
    fn empty_sketch() {
        let s = DDSketch::new(0.01);
        assert!(s.is_empty());
        assert_eq!(s.count(), 0);
        assert_eq!(s.quantile(0.5), None);
        assert_eq!(s.min(), None);
        assert_eq!(s.max(), None);
        assert_eq!(s.rank(0.0), 0.0);
    }

    #[test]
    fn empty_sketch_roundtrips() {
        let s = DDSketch::new(0.01);
        let back = DDSketch::from_bytes(&s.to_bytes()).expect("valid blob");
        assert!(back.is_empty());
        assert_eq!(back.quantile(0.5), None);
        assert_eq!(back.relative_accuracy(), 0.01);
    }

    #[test]
    fn from_bytes_rejects_malformed() {
        assert!(DDSketch::from_bytes(&[]).is_none());
        assert!(DDSketch::from_bytes(&[0; 7]).is_none()); // truncated
                                                          // Trailing garbage.
        let mut s = DDSketch::new(0.01);
        s.add(1.0);
        let mut bytes = s.to_bytes();
        bytes.push(0);
        assert!(DDSketch::from_bytes(&bytes).is_none());
        // Invalid alpha (>= 1).
        let mut bad = 2.0f64.to_le_bytes().to_vec();
        bad.extend_from_slice(&0u64.to_le_bytes()); // zeros
        bad.extend_from_slice(&0u64.to_le_bytes()); // n
        bad.extend_from_slice(&f64::INFINITY.to_le_bytes());
        bad.extend_from_slice(&f64::NEG_INFINITY.to_le_bytes());
        bad.extend_from_slice(&0u64.to_le_bytes()); // pos_len
        bad.extend_from_slice(&0u64.to_le_bytes()); // neg_len
        assert!(DDSketch::from_bytes(&bad).is_none());
    }

    #[test]
    fn add_n_and_nonfinite() {
        let mut s = DDSketch::new(0.01);
        s.add_n(5.0, 10);
        s.add(f64::NAN);
        s.add(f64::INFINITY);
        s.add_n(7.0, 0); // no-op
        assert_eq!(s.count(), 10);
        assert_eq!(s.quantile(0.5), Some(s.quantile(0.5).unwrap()));
    }

    // ---- Property / fuzz tests (deterministic xorshift64, fixed seed) -------

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
        fn finite(&mut self) -> f64 {
            let frac = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            frac * 2_000_000.0 - 1_000_000.0
        }
    }

    #[test]
    fn prop_serialize_roundtrip() {
        const TRIALS: usize = 200;
        let mut rng = XorShift64::new(0x11C1_7A55_9000_0001);
        for trial in 0..TRIALS {
            let alpha = 0.001 + (rng.below(100) as f64) / 1000.0; // (0.001, 0.1)
            let n = rng.below(4_000);
            let mut s = DDSketch::new(alpha);
            for _ in 0..n {
                s.add(rng.finite());
            }
            let back = DDSketch::from_bytes(&s.to_bytes()).expect("valid blob");
            for &q in &[0.0, 0.25, 0.5, 0.75, 1.0] {
                assert_eq!(
                    back.quantile(q),
                    s.quantile(q),
                    "trial {trial}: alpha={alpha} n={n} q={q} mismatch"
                );
            }
            assert_eq!(back.count(), s.count());
            assert_eq!(back.min(), s.min());
            assert_eq!(back.max(), s.max());
        }
    }

    #[test]
    fn prop_merge_matches_single() {
        const TRIALS: usize = 150;
        let mut rng = XorShift64::new(0x7E57_C0DE_2222_3333);
        for trial in 0..TRIALS {
            let alpha = 0.005 + (rng.below(50) as f64) / 1000.0;
            let n = 2_000 + rng.below(2_000);
            let vals: Vec<f64> = (0..n).map(|_| rng.finite()).collect();
            let mut whole = DDSketch::new(alpha);
            for &x in &vals {
                whole.add(x);
            }
            let parts = 2 + rng.below(4) as usize;
            let mut sketches: Vec<DDSketch> = (0..parts).map(|_| DDSketch::new(alpha)).collect();
            for &x in &vals {
                let p = rng.below(parts as u64) as usize;
                sketches[p].add(x);
            }
            let merged = merge_all(sketches.into_iter()).unwrap();
            assert_eq!(merged.count(), n);
            // Bucket merge is exact → quantiles equal the single-pass exactly.
            for &q in &[0.1, 0.25, 0.5, 0.75, 0.9] {
                assert_eq!(
                    merged.quantile(q),
                    whole.quantile(q),
                    "trial {trial}: alpha={alpha} n={n} q={q} merge != single"
                );
            }
        }
    }
}
