//! HyperLogLog++ — distinct-count (cardinality) estimation.
//!
//! Estimates the number of distinct values in a column using a fixed `6·m` bits
//! with ~`1.04/√m` relative error. Cheap distinct counts are what let the
//! optimizer choose join build sides and size aggregations without a second scan.
//! The cardinality estimator is Ertl's improved maximum-likelihood form
//! (`sigma`/`tau` corrections), which is continuous and essentially unbiased across
//! the whole range — so it needs neither a linear-counting handover threshold nor
//! HyperLogLog++'s empirical per-precision bias tables.

use std::hash::Hash;

use arrow::array::{Array, ArrayRef};
use arrow::row::{RowConverter, SortField};

use crate::{hash_one, Mergeable, SEED};

/// A HyperLogLog++ distinct-count sketch.
#[derive(Clone)]
pub struct HyperLogLog {
    precision: u8,
    registers: Vec<u8>,
}

impl HyperLogLog {
    /// Create an empty sketch with `2^precision` registers (precision 4..=18).
    pub fn new(precision: u8) -> Self {
        assert!((4..=18).contains(&precision), "precision must be in 4..=18");
        Self {
            precision,
            registers: vec![0; 1usize << precision],
        }
    }

    /// A sensible default (precision 14 → ~0.81% error, 16 KB).
    pub fn default_precision() -> Self {
        Self::new(14)
    }

    fn m(&self) -> usize {
        self.registers.len()
    }

    /// Add a pre-computed 64-bit hash.
    pub fn add_hash(&mut self, hash: u64) {
        let p = self.precision as u32;
        let idx = (hash >> (64 - p)) as usize;
        // Rank = position of the leftmost 1 in the remaining bits (+1).
        let w = hash << p;
        let rank = (w.leading_zeros() + 1).min(64 - p + 1) as u8;
        if rank > self.registers[idx] {
            self.registers[idx] = rank;
        }
    }

    /// Add one hashable value.
    pub fn add<T: Hash>(&mut self, value: &T) {
        self.add_hash(hash_one(value));
    }

    /// Add every non-null value of an Arrow array.
    ///
    /// Primitive numeric/temporal and string/binary columns — the overwhelming common
    /// case — are hashed *directly* from their native values, which is ~10–50× faster
    /// than encoding every value through Arrow's general `RowConverter`. Equal values
    /// hash equally within a column's (fixed) type, so the distinct estimate is
    /// unchanged; the row-format path remains the fallback for exotic types
    /// (nested/dictionary/etc.) so any orderable column is still counted.
    pub fn add_array(&mut self, array: &ArrayRef) {
        if self.add_array_fast(array) {
            return;
        }
        let converter = match RowConverter::new(vec![SortField::new(array.data_type().clone())]) {
            Ok(c) => c,
            Err(_) => return, // unorderable type: skip (estimate stays 0)
        };
        let rows = match converter.convert_columns(std::slice::from_ref(array)) {
            Ok(r) => r,
            Err(_) => return,
        };
        for i in 0..array.len() {
            if array.is_null(i) {
                continue;
            }
            self.add_hash(SEED.hash_one(rows.row(i)));
        }
    }

    /// Hash primitive / string / binary columns directly from native values. Returns
    /// `false` for a type it does not fast-path, so the caller uses the row-format
    /// fallback. The hash is `SEED.hash_one` of the native value; floats are routed
    /// through [`canon_float_bits`] so `-0.0`/`0.0` and every NaN bit pattern share
    /// one distinct identity — the same identity the exact distinct / GROUP BY path
    /// and the approx-aggregate HLL use (`raw to_bits()` counted `-0.0` and `0.0`, and
    /// distinct NaN payloads, as separate distinct values, over-counting the ndv).
    fn add_array_fast(&mut self, array: &ArrayRef) -> bool {
        use arrow::array::*;
        use arrow::datatypes::DataType as DT;

        macro_rules! prim {
            ($ty:ty, $hashval:expr) => {{
                let a = array.as_any().downcast_ref::<$ty>().expect("dtype matched");
                if a.null_count() == 0 {
                    for &v in a.values().iter() {
                        let h = $hashval(v);
                        self.add_hash(SEED.hash_one(&h));
                    }
                } else {
                    for i in 0..a.len() {
                        if a.is_valid(i) {
                            let h = $hashval(a.value(i));
                            self.add_hash(SEED.hash_one(&h));
                        }
                    }
                }
                true
            }};
        }

        match array.data_type() {
            DT::Int8 => prim!(Int8Array, |v: i8| v as i64),
            DT::Int16 => prim!(Int16Array, |v: i16| v as i64),
            DT::Int32 => prim!(Int32Array, |v: i32| v as i64),
            DT::Int64 => prim!(Int64Array, |v: i64| v),
            DT::UInt8 => prim!(UInt8Array, |v: u8| v as u64),
            DT::UInt16 => prim!(UInt16Array, |v: u16| v as u64),
            DT::UInt32 => prim!(UInt32Array, |v: u32| v as u64),
            DT::UInt64 => prim!(UInt64Array, |v: u64| v),
            DT::Float16 => prim!(Float16Array, |v| canon_float_bits(f64::from(v))),
            DT::Float32 => prim!(Float32Array, |v: f32| canon_float_bits(v as f64)),
            DT::Float64 => prim!(Float64Array, |v: f64| canon_float_bits(v)),
            DT::Date32 => prim!(Date32Array, |v: i32| v as i64),
            DT::Date64 => prim!(Date64Array, |v: i64| v),
            DT::Utf8 => {
                let a = array.as_any().downcast_ref::<StringArray>().expect("utf8");
                for i in 0..a.len() {
                    if a.is_valid(i) {
                        self.add_hash(SEED.hash_one(a.value(i)));
                    }
                }
                true
            }
            DT::LargeUtf8 => {
                let a = array
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .expect("lutf8");
                for i in 0..a.len() {
                    if a.is_valid(i) {
                        self.add_hash(SEED.hash_one(a.value(i)));
                    }
                }
                true
            }
            DT::Binary => {
                let a = array
                    .as_any()
                    .downcast_ref::<BinaryArray>()
                    .expect("binary");
                for i in 0..a.len() {
                    if a.is_valid(i) {
                        self.add_hash(SEED.hash_one(a.value(i)));
                    }
                }
                true
            }
            _ => false,
        }
    }

    /// Estimate the number of distinct values added.
    ///
    /// Uses Ertl's improved estimator rather than the textbook
    /// `α_m·m²/Σ2^-r` raw estimator with a linear-counting small-range switch. The
    /// difference is structural, not a tuning detail:
    ///
    /// * The classic pair is **two** estimators glued at a threshold, and the raw one is
    ///   biased high for `m < n < 5m`. Whatever the handover point, the join is a
    ///   discontinuity, and the sweep that picked the previous constant could only trade one
    ///   region's bias against another's — the best it achieved was 0.75% RMSE with a 0.38%
    ///   worst-case bias. HyperLogLog++ patches the same gap with ~15×200 empirically
    ///   measured per-precision bias constants.
    /// * Ertl's estimator is **one** continuous function of the register multiplicities,
    ///   derived by maximizing the Poisson likelihood of the observed registers. It is
    ///   essentially unbiased over the entire range — from an empty sketch to a saturated one
    ///   — with relative error ≈ `1.04/√m` throughout, and it needs no bias tables and no
    ///   threshold at all.
    ///
    /// The two correction terms are what make the whole range work: `σ` accounts for the
    /// registers still at zero (the information linear counting extracts, but in closed form
    /// and blended rather than switched), and `τ` accounts for registers saturated at the
    /// maximum rank (the large-range end). Between them the estimator degrades gracefully
    /// instead of stepping.
    pub fn estimate(&self) -> f64 {
        let m = self.m() as f64;
        let q = 64 - self.precision as u32; // registers hold ranks 0..=q+1
                                            // Register multiplicities: `counts[k]` = how many registers hold rank `k`. Ertl's
                                            // estimator is a function of this histogram alone, which is also a cheaper pass than
                                            // the float harmonic sum it replaces (integer increments, no FP per register).
        let mut counts = vec![0u32; q as usize + 2];
        for &r in &self.registers {
            counts[r as usize] += 1;
        }
        if counts[0] == self.registers.len() as u32 {
            return 0.0; // every register untouched: the sketch is empty
        }
        let mut z = m * tau((m - counts[q as usize + 1] as f64) / m);
        for k in (1..=q as usize).rev() {
            z = 0.5 * (z + counts[k] as f64);
        }
        z += m * sigma(counts[0] as f64 / m);
        if z <= 0.0 || !z.is_finite() {
            // Every register saturated at the maximum rank — unreachable with a 64-bit hash
            // below ~2^58 distinct values, but the estimate is unbounded there rather than
            // wrong, so report the largest finite count instead of an infinity.
            return f64::MAX;
        }
        ALPHA_INF * m * m / z
    }

    /// Serialize to a self-describing byte blob: `[precision: u8][registers…]`.
    ///
    /// The register count is implied by `precision` (`2^precision`), so no length
    /// prefix is needed. Suitable for storing as a metadata blob.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 + self.registers.len());
        out.push(self.precision);
        out.extend_from_slice(&self.registers);
        out
    }

    /// Reconstruct from [`to_bytes`](Self::to_bytes). Returns `None` on malformed
    /// input (bad precision, or a length that doesn't equal `1 + 2^precision`).
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let (&precision, registers) = bytes.split_first()?;
        if !(4..=18).contains(&precision) {
            return None;
        }
        let expected = 1usize << precision;
        if registers.len() != expected {
            return None;
        }
        Some(Self {
            precision,
            registers: registers.to_vec(),
        })
    }
}

impl Mergeable for HyperLogLog {
    /// Merge register-wise (element-wise max). Precision must match.
    fn merge(&mut self, other: &HyperLogLog) {
        assert_eq!(self.precision, other.precision, "precision mismatch");
        for (a, b) in self.registers.iter_mut().zip(&other.registers) {
            if *b > *a {
                *a = *b;
            }
        }
    }
}

/// Canonicalize a float's bits for distinct identity, so equal-by-value floats hash
/// identically: `-0.0`/`0.0` collapse to `+0.0` and every NaN bit pattern collapses to
/// one canonical NaN. This matches the exact distinct / GROUP BY float identity (and the
/// approx-aggregate HLL, fixed in B103); raw `to_bits()` would over-count the ndv because
/// `(-0.0).to_bits() != (0.0).to_bits()` and NaN payloads differ.
#[inline]
fn canon_float_bits(v: f64) -> u64 {
    if v.is_nan() {
        f64::NAN.to_bits()
    } else if v == 0.0 {
        0 // folds both `+0.0` and `-0.0`
    } else {
        v.to_bits()
    }
}

/// The HyperLogLog normalization constant in the limit, `α_∞ = 1/(2·ln 2)`.
///
/// The classic per-`m` constants (0.673, 0.697, 0.709, `0.7213/(1 + 1.079/m)`) correct the
/// *raw* estimator's finite-`m` bias. Ertl's estimator handles that bias through the `σ`/`τ`
/// terms instead, so the exact limiting constant is the right one at every precision.
const ALPHA_INF: f64 = 0.721_347_520_444_481_7;

/// `σ(x) = x + Σ_{k≥1} x^(2^k) · 2^(k-1)` — the zero-register correction.
///
/// This is the closed form of the information carried by registers that are still zero: with
/// `x` the fraction of empty registers, `σ` says how much cardinality mass is hiding below the
/// sketch's resolution. It is what makes a separate linear-counting branch unnecessary — the
/// correction is *blended in* rather than switched to, so there is no discontinuity to tune.
///
/// The series converges quadratically (`x^(2^k)` squares each step), so the loop runs a
/// handful of iterations before the sum stops changing in double precision. `σ(1) = ∞` is the
/// correct limit: an all-zero sketch carries no information about the cardinality, and the
/// caller short-circuits that case.
fn sigma(mut x: f64) -> f64 {
    if x == 1.0 {
        return f64::INFINITY;
    }
    let mut y = 1.0;
    let mut z = x;
    loop {
        x *= x;
        let previous = z;
        z += x * y;
        y += y;
        if z == previous {
            return z;
        }
    }
}

/// `τ(x)` — the saturated-register correction, the large-range counterpart of [`sigma`].
///
/// With `x` the fraction of registers *not* pinned at the maximum rank, `τ` accounts for the
/// cardinality the sketch can no longer resolve at the top of its range. It is the term that
/// replaces HyperLogLog++'s large-range correction, and with a 64-bit hash it is essentially
/// zero in practice — but including it is what lets one continuous formula cover the whole
/// range rather than three glued pieces.
///
/// Converges by repeated square roots (again quadratic), and `τ(0) = τ(1) = 0` are the exact
/// boundary values.
fn tau(mut x: f64) -> f64 {
    if x == 0.0 || x == 1.0 {
        return 0.0;
    }
    let mut y = 1.0;
    let mut z = 1.0 - x;
    loop {
        x = x.sqrt();
        let previous = z;
        y *= 0.5;
        z -= (1.0 - x).powi(2) * y;
        if z == previous {
            return z / 3.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merge_all;
    use arrow::array::Int64Array;
    use std::sync::Arc;

    /// Averaging independent trials cancels HLL's sampling variance (which is
    /// `1.04/sqrt(m)` per trial) and leaves only the estimator's systematic bias.
    fn mean_relative_error(precision: u8, n: u64, trials: u64) -> f64 {
        let mut total = 0.0;
        for t in 0..trials {
            let mut hll = HyperLogLog::new(precision);
            // A distinct, well-separated key space per trial.
            let salt = (t + 1).wrapping_mul(0xD6E8_FEB8_6659_FD93);
            for i in 0..n {
                hll.add(&(i ^ salt));
            }
            total += (hll.estimate() - n as f64) / n as f64;
        }
        total / trials as f64
    }

    #[test]
    fn unbiased_across_the_whole_transition_range() {
        // The range `0.5m < n < 5m` is where the classic raw estimator is worst and where a
        // linear-counting handover has to be glued in. Ertl's estimator is one continuous
        // function there, so it should show no bias anywhere across it — including at the
        // load factors (2.4–3.0) where the old two-estimator hybrid had its seam. Averaged
        // over 24 trials the sampling error is ~0.17%, so a 1% bound cannot pass by luck.
        let m = 1u64 << 14;
        for mult in [0.5_f64, 1.0, 2.0, 2.4, 2.5, 2.6, 3.0, 4.0, 5.0] {
            let n = (mult * m as f64) as u64;
            let bias = mean_relative_error(14, n, 24);
            assert!(
                bias.abs() < 0.01,
                "n/m = {mult}: mean relative error {:.3}% exceeds 1% — the estimator is biased",
                bias * 100.0
            );
        }
    }

    #[test]
    fn accurate_far_below_one_register_per_value() {
        // The sparse end, where almost every register is still zero and the estimate rests
        // entirely on the `sigma` correction.
        let m = 1u64 << 14;
        for n in [16u64, 128, m / 8] {
            let bias = mean_relative_error(14, n, 16);
            assert!(
                bias.abs() < 0.01,
                "n = {n}: sparse-range bias {:.3}%",
                bias * 100.0
            );
        }
    }

    #[test]
    fn empty_sketch_estimates_zero() {
        assert_eq!(HyperLogLog::new(12).estimate(), 0.0);
    }

    #[test]
    fn correction_terms_have_their_exact_boundary_values() {
        // The two closed forms anchor the estimator at the ends of its range; a wrong
        // boundary value would show up as a bias that no amount of averaging removes.
        assert_eq!(tau(0.0), 0.0);
        assert_eq!(tau(1.0), 0.0);
        assert_eq!(sigma(0.0), 0.0);
        assert!(sigma(1.0).is_infinite());
        // Both series are monotone in their argument over (0, 1).
        assert!(sigma(0.9) > sigma(0.5) && sigma(0.5) > sigma(0.1));
        assert!(tau(0.1) > tau(0.5) && tau(0.5) > tau(0.9));
    }

    #[test]
    fn relative_error_within_bounds() {
        // Precision 14 → m=16384, expected error ~0.81%; allow 3% slack.
        let mut hll = HyperLogLog::new(14);
        let n = 1_000_000u64;
        for i in 0..n {
            hll.add(&i);
        }
        let est = hll.estimate();
        let err = (est - n as f64).abs() / n as f64;
        assert!(err < 0.03, "relative error {err} too high (est {est})");
    }

    #[test]
    fn small_cardinality_is_accurate() {
        let mut hll = HyperLogLog::new(14);
        for i in 0..100u64 {
            hll.add(&i);
        }
        let est = hll.estimate();
        // The `sigma` correction should make a sketch this sparse near-exact.
        assert!((est - 100.0).abs() < 5.0, "small estimate {est}");
    }

    #[test]
    fn merge_equals_combined() {
        let mut a = HyperLogLog::new(12);
        let mut b = HyperLogLog::new(12);
        let mut both = HyperLogLog::new(12);
        for i in 0..50_000u64 {
            if i % 2 == 0 {
                a.add(&i);
            } else {
                b.add(&i);
            }
            both.add(&i);
        }
        a.merge(&b);
        // Merged sketch estimates the union (50k distinct), like the combined one.
        let rel = (a.estimate() - both.estimate()).abs() / both.estimate();
        assert!(rel < 0.02, "merge diverged: {rel}");
    }

    #[test]
    fn bytes_roundtrip_preserves_estimate() {
        let mut hll = HyperLogLog::new(12);
        for i in 0..40_000u64 {
            hll.add(&i);
        }
        let bytes = hll.to_bytes();
        let back = HyperLogLog::from_bytes(&bytes).expect("valid blob");
        assert_eq!(back.precision, hll.precision);
        assert_eq!(back.registers, hll.registers);
        assert!((back.estimate() - hll.estimate()).abs() < 1e-9);
    }

    #[test]
    fn from_bytes_rejects_malformed() {
        assert!(HyperLogLog::from_bytes(&[]).is_none());
        // Valid precision byte but wrong register count.
        assert!(HyperLogLog::from_bytes(&[12, 0, 0, 0]).is_none());
        // Out-of-range precision.
        assert!(HyperLogLog::from_bytes(&[3, 0]).is_none());
        assert!(HyperLogLog::from_bytes(&[19, 0]).is_none());
        // Precision 4 → exactly 16 registers required.
        assert!(HyperLogLog::from_bytes(&[4, 0]).is_none());
        let ok: Vec<u8> = std::iter::once(4u8)
            .chain(std::iter::repeat_n(0, 16))
            .collect();
        assert!(HyperLogLog::from_bytes(&ok).is_some());
    }

    #[test]
    fn merge_is_associative_for_estimates() {
        let build = |start: u64, end: u64| {
            let mut h = HyperLogLog::new(12);
            for i in start..end {
                h.add(&i);
            }
            h
        };
        let (a, b, cc) = (
            build(0, 20_000),
            build(15_000, 35_000),
            build(30_000, 50_000),
        );

        let mut left = a.clone();
        let mut bc = b.clone();
        bc.merge(&cc);
        left.merge(&bc);

        let mut right = a.clone();
        right.merge(&b);
        right.merge(&cc);

        assert!((left.estimate() - right.estimate()).abs() < 1e-9);
    }

    #[test]
    fn add_array_folds_signed_zero_and_nan_like_exact_distinct() {
        use arrow::array::Float64Array;
        // Distinct identity for floats: `-0.0`≡`0.0` and every NaN bit pattern is one
        // value — the same identity the exact distinct / GROUP BY path uses (and the
        // approx-aggregate HLL path, fixed in B103). So the distinct set of
        // {-0.0, 0.0, NaN, NaN', 1.5} is {0.0, NaN, 1.5} → 3, not 5.
        let arr: ArrayRef = Arc::new(Float64Array::from(vec![
            -0.0,
            0.0,
            f64::NAN,
            f64::from_bits(0x7ff8_0000_0000_0001), // a different NaN bit pattern
            1.5,
        ]));
        let mut hll = HyperLogLog::new(14);
        hll.add_array(&arr);
        // Small cardinality → HLL is exact (linear counting), so this must be exactly 3.
        let est = hll.estimate().round();
        assert_eq!(est, 3.0, "signed-zero/NaN not folded: estimate {est}");
    }

    #[test]
    fn add_array_folds_signed_zero_and_nan_for_float16() {
        use arrow::array::Float64Array;
        use arrow::compute::cast;
        use arrow::datatypes::DataType;
        // Float16 must fold `-0.0`≡`0.0` and every NaN to one distinct identity, the
        // same as Float32/Float64 and the exact distinct / GROUP BY path. Before the
        // fix Float16 fell through to the row-format fallback, which does not fold
        // signed zero, so {-0.0, 0.0, NaN, NaN', 1.5} over-counted to 4 instead of 3.
        let src: ArrayRef = Arc::new(Float64Array::from(vec![
            -0.0,
            0.0,
            f64::NAN,
            f64::from_bits(0x7ff8_0000_0000_0001),
            1.5,
        ]));
        let f16 = cast(&src, &DataType::Float16).expect("cast to f16");
        let mut hll = HyperLogLog::new(14);
        hll.add_array(&f16);
        let est = hll.estimate().round();
        assert_eq!(est, 3.0, "f16 signed-zero/NaN not folded: estimate {est}");
    }

    #[test]
    fn add_array_counts_distinct() {
        let arr: ArrayRef = Arc::new(Int64Array::from(
            (0..10_000).map(|i| i % 1000).collect::<Vec<_>>(),
        ));
        let mut hll = HyperLogLog::new(14);
        hll.add_array(&arr);
        let est = hll.estimate();
        assert!((est - 1000.0).abs() / 1000.0 < 0.05, "ndv estimate {est}");
    }

    // ---- Property / fuzz tests (deterministic xorshift64, fixed seed) -------

    /// Minimal deterministic PRNG so trials are reproducible without `rand`.
    struct XorShift64(u64);
    impl XorShift64 {
        fn new(seed: u64) -> Self {
            // Avoid the all-zero state (xorshift's fixed point).
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
        /// Uniform in `[0, bound)` (bound > 0).
        fn below(&mut self, bound: u64) -> u64 {
            self.next_u64() % bound
        }
    }

    #[test]
    fn prop_merge_associative() {
        const TRIALS: usize = 200;
        let mut rng = XorShift64::new(0x5151_A55E_551A_71E5);

        for trial in 0..TRIALS {
            // Random distinct-value count and partition count.
            let distinct = 1_000 + rng.below(4_000); // 1k..5k distinct integers
            let parts = 2 + rng.below(6) as usize; // 2..=7 partitions

            // Distinct universe: a random base offset so values differ per trial.
            let base = rng.next_u64() & 0x00FF_FFFF_FFFF_FFFF;

            // Assign each distinct value to a random partition (disjoint partitions).
            let mut hlls: Vec<HyperLogLog> = (0..parts).map(|_| HyperLogLog::new(14)).collect();
            for v in 0..distinct {
                let key = base.wrapping_add(v);
                let p = rng.below(parts as u64) as usize;
                hlls[p].add(&key);
            }

            let merged = merge_all(hlls.into_iter()).unwrap();
            let err = (merged.estimate() - distinct as f64).abs() / distinct as f64;
            assert!(
                err < 0.05,
                "trial {trial}: distinct={distinct} parts={parts} rel-error={err}"
            );
        }
    }

    #[test]
    fn prop_serialize_roundtrip() {
        const TRIALS: usize = 300;
        let mut rng = XorShift64::new(0xA5A5_F00D_1234_BEEF);

        for trial in 0..TRIALS {
            let precision = (4 + rng.below(15)) as u8; // 4..=18
            let n = rng.below(3_000); // up to ~3k adds (may include duplicates)

            let mut hll = HyperLogLog::new(precision);
            for _ in 0..n {
                let key = rng.next_u64();
                hll.add(&key);
            }

            let bytes = hll.to_bytes();
            let back = HyperLogLog::from_bytes(&bytes).expect("valid blob");
            // Roundtrip must be bit-identical → estimate is exactly equal.
            assert_eq!(
                back.estimate().to_bits(),
                hll.estimate().to_bits(),
                "trial {trial}: precision={precision} n={n} estimate mismatch after roundtrip"
            );
        }
    }
}
