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

    /// Upper boundary of bucket `i`: bucket `i` holds magnitudes in
    /// `(γ^(i-1), γ^i]`, so `γ^i` is the largest magnitude it can contain.
    ///
    /// This — not [`value_of`](Self::value_of), the bucket's geometric *centre* —
    /// is what a "is the whole bucket ≤ x?" test must compare against. Testing the
    /// centre drops a bucket whenever `x` falls in its upper α-band, which is how
    /// `rank` used to lose the maximum's own bucket and return 0.99 for `rank(max)`.
    #[inline]
    fn upper_bound_of(&self, i: i32) -> f64 {
        self.gamma.powi(i)
    }

    /// Approximate fraction of values ≤ `x`, in `[0, 1]` — the selectivity of
    /// `col <= x`. Returns 0 for an empty sketch.
    ///
    /// Clamped at the tracked exact min/max (as `TDigest::rank` and
    /// `KllSketch::rank` are), so `rank(max) == 1.0` and `rank(< min) == 0.0`
    /// exactly. Without that clamp `selectivity_gt = 1 - rank` carries a non-zero
    /// floor no predicate can ever clear, which biases the optimizer's cost model.
    pub fn rank(&self, x: f64) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        if x >= self.max {
            return 1.0;
        }
        if x < self.min {
            return 0.0;
        }
        // `below` accumulates whole buckets strictly below `x`; `partial` accumulates the
        // fraction of the one bucket that straddles `x`. Interpolating that straddling bucket
        // — rather than counting 0 or all of it — turns a step CDF into a continuous one, so
        // two predicates a hair apart get selectivities a hair apart instead of jumping by a
        // whole bucket's mass. The interpolation is log-uniform within the bucket, matching
        // the sketch's own geometric spacing: a value's position is `log_γ(x/lower) ∈ [0, 1]`
        // across the bucket `(lower, upper]`.
        let mut below = 0u64;
        let mut partial = 0.0;
        if x >= 0.0 {
            // x ≥ 0: every negative and every zero is ≤ x.
            below += self.negative.values().sum::<u64>();
            below += self.zeros;
            for (&i, &c) in self.positive.iter() {
                let upper = self.upper_bound_of(i);
                if upper <= x {
                    below += c;
                } else {
                    partial = c as f64 * self.bucket_fraction_below(i, x);
                    break;
                }
            }
        } else {
            // x < 0: only sufficiently large-magnitude negatives are ≤ x. Bucket `i` of the
            // mirror map holds values in `[-γ^i, -γ^(i-1))`, so the whole bucket is ≤ x when
            // its *least* negative end is: -γ^(i-1) ≤ x.
            for (&i, &c) in self.negative.iter().rev() {
                if -self.upper_bound_of(i - 1) <= x {
                    below += c;
                } else if -self.upper_bound_of(i) < x {
                    // The straddling negative bucket: `[-γ^i, -γ^(i-1))`, ordered so more of
                    // it is ≤ x the closer x is to its most-negative end.
                    let hi = -self.upper_bound_of(i - 1);
                    let lo = -self.upper_bound_of(i);
                    partial = c as f64 * ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
                    break;
                } else {
                    break;
                }
            }
        }
        (below as f64 + partial) / self.n as f64
    }

    /// The fraction of a positive bucket `(γ^(i-1), γ^i]` whose values are ≤ `x`.
    ///
    /// Log-uniform within the bucket, so the interpolation follows the same geometric spacing
    /// the sketch uses to place values: `(ln x - ln lower) / (ln upper - ln lower)`.
    #[inline]
    fn bucket_fraction_below(&self, i: i32, x: f64) -> f64 {
        let lower = self.upper_bound_of(i - 1);
        if x <= lower {
            return 0.0;
        }
        let span = self.ln_gamma; // ln(upper) - ln(lower) == ln(γ)
        if span <= 0.0 {
            return 0.0;
        }
        ((x.ln() - lower.ln()) / span).clamp(0.0, 1.0)
    }

    /// Serialize to a byte blob. Layout (all little-endian):
    /// `[alpha: f64][zeros: u64][n: u64][min: f64][max: f64]`
    /// `[pos_len: u64]{[index: i32][count: u64]}×pos_len`
    /// `[neg_len: u64]{[index: i32][count: u64]}×neg_len`.
    /// `gamma`/`ln_gamma` are derived from `alpha` on load.
    pub fn to_bytes(&self) -> Vec<u8> {
        // Exact wire size: 5×8 header + per map (u64 len + entries×(i32+u64)) — one allocation.
        let entries = self.positive.len() + self.negative.len();
        let mut out = Vec::with_capacity(40 + 2 * 8 + entries * 12);
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

    /// Regression: `rank` compared `x` against each bucket's geometric *centre*
    /// rather than its upper boundary, so a bucket was dropped whenever `x` landed
    /// in its upper α-band — and unlike `TDigest`/`KllSketch`, `rank` had no
    /// min/max clamp. `rank(max)` came back 0.99 instead of 1.0, giving
    /// `selectivity_gt = 1 - rank` a non-zero floor it could never clear.
    #[test]
    fn rank_at_max_is_one() {
        let mut d = DDSketch::new(0.01);
        for i in 1..=100 {
            d.add(i as f64);
        }
        assert_eq!(d.rank(100.0), 1.0, "rank at the maximum must be exactly 1");
        assert_eq!(d.rank(1.0e9), 1.0);
        assert_eq!(d.rank(0.5), 0.0, "rank below the minimum must be 0");

        // Signed data: the same must hold on the negative side.
        let mut e = DDSketch::new(0.01);
        for i in -50..=50 {
            e.add(i as f64);
        }
        assert_eq!(e.rank(50.0), 1.0);
        assert_eq!(e.rank(-50.0 - 1e-9), 0.0);
        // The max clamp is the asymmetric one that matters: `1 - rank` is the
        // `>` selectivity, so a missing top bucket becomes a floor no predicate
        // can clear. At the minimum, under-counting by one bucket is inside α.
        assert!(e.rank(-49.0) < 0.05);
    }

    /// `rank` must be monotone and must honour DDSketch's *value* guarantee: the
    /// reported fraction is the true fraction ≤ some `x'` within one γ-bucket of
    /// `x` — it can only be wrong about values in `x`'s own bucket.
    #[test]
    fn rank_respects_relative_accuracy() {
        for alpha in [0.01, 0.05] {
            // Signed data spanning zero, plus a Pareto tail.
            let mut vals: Vec<f64> = (-500..=500).map(|i| i as f64 * 0.37).collect();
            for i in 1..=2_000u32 {
                vals.push(10.0 / (i as f64 / 2_000.0).powf(1.5)); // heavy tail
                vals.push(-3.0 / (i as f64 / 2_000.0).powf(1.2));
            }
            let mut d = DDSketch::new(alpha);
            for &v in &vals {
                d.add(v);
            }
            let n = vals.len() as f64;
            let frac_le = |t: f64| vals.iter().filter(|&&v| v <= t).count() as f64 / n;

            let mut probes: Vec<f64> = vec![-1e6, -1_000.0, -1.0, -0.001, 0.0, 0.001, 1.0, 1e6];
            probes.extend((0..40).map(|i| d.quantile(i as f64 / 40.0).unwrap()));

            let mut prev = 0.0;
            probes.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for &x in &probes {
                let r = d.rank(x);
                assert!((0.0..=1.0).contains(&r), "rank {r} out of range at {x}");
                assert!(r >= prev - 1e-12, "rank not monotone at {x}: {prev} -> {r}");
                prev = r;

                // Widen by one bucket (factor γ) in *value* space; the band must
                // contain the estimate.
                let gamma = (1.0 + alpha) / (1.0 - alpha);
                let (lo, hi) = if x >= 0.0 {
                    (x / gamma, x * gamma)
                } else {
                    (x * gamma, x / gamma)
                };
                assert!(
                    r >= frac_le(lo) - 1e-12 && r <= frac_le(hi) + 1e-12,
                    "alpha={alpha} x={x}: rank {r} outside [{}, {}]",
                    frac_le(lo),
                    frac_le(hi)
                );
            }
            assert_eq!(d.rank(d.max().unwrap()), 1.0);
        }
    }

    /// The α relative-error guarantee on *quantiles* is what the bucket-boundary
    /// change must not disturb.
    #[test]
    fn quantile_relative_error_within_alpha() {
        for alpha in [0.01, 0.05] {
            let mut vals: Vec<f64> = (1..=5_000).map(|i| i as f64 * 0.11).collect();
            for i in 1..=2_000u32 {
                vals.push(10.0 / (i as f64 / 2_000.0).powf(1.5));
            }
            let mut d = DDSketch::new(alpha);
            for &v in &vals {
                d.add(v);
            }
            let mut sorted = vals.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            for i in 1..100 {
                let q = i as f64 / 100.0;
                let k = (q * (sorted.len() - 1) as f64).round() as usize;
                let got = d.quantile(q).unwrap();
                // ±1 rank of slack for the 0-based index convention; the assertion
                // under test is the *relative value* error, which is the guarantee.
                let err = (k.saturating_sub(1)..=(k + 1).min(sorted.len() - 1))
                    .map(|j| (got - sorted[j]).abs() / sorted[j].abs())
                    .fold(f64::INFINITY, f64::min);
                assert!(err <= alpha + 1e-9, "alpha={alpha} q={q}: rel err {err}");
            }
        }
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

    /// `rank` must vary *continuously* with `x`, not jump by a whole bucket's mass at each
    /// bucket boundary — the selectivity of `col <= x` and `col <= x + ε` should differ by ε.
    #[test]
    fn rank_interpolates_within_a_bucket() {
        let mut d = DDSketch::new(0.02);
        for i in 1..=100_000u64 {
            d.add(i as f64);
        }
        // March across a decade in the middle of the range; adjacent ranks must move smoothly
        // and monotonically, with no single step carrying a whole bucket.
        let mut prev = d.rank(20_000.0);
        let mut x = 20_000.0;
        while x < 40_000.0 {
            x += 50.0;
            let r = d.rank(x);
            assert!(r >= prev - 1e-9, "rank went backwards at {x}");
            assert!(r - prev < 0.01, "rank jumped by {} at {x}", r - prev);
            prev = r;
        }
    }

    /// Interpolation must not cost the relative-accuracy guarantee: a known quantile's rank
    /// still comes back right.
    #[test]
    fn interpolated_rank_stays_accurate() {
        let mut d = DDSketch::new(0.01);
        for i in 0..100_000u64 {
            d.add(i as f64);
        }
        for (x, want) in [(10_000.0, 0.1), (50_000.0, 0.5), (90_000.0, 0.9)] {
            let got = d.rank(x);
            assert!((got - want).abs() < 0.02, "rank({x}) = {got}, want {want}");
        }
    }

    /// The interpolation must handle negative values too — the mirror map is ordered the
    /// opposite way, so getting the direction wrong would make `rank` non-monotone at 0.
    #[test]
    fn rank_is_monotone_across_zero() {
        let mut d = DDSketch::new(0.02);
        for i in -50_000..50_000i64 {
            d.add(i as f64);
        }
        let mut prev = 0.0;
        let mut x = -10_000.0;
        while x < 10_000.0 {
            let r = d.rank(x);
            assert!(r >= prev - 1e-9, "rank not monotone at {x}: {r} < {prev}");
            prev = r;
            x += 100.0;
        }
    }
}
