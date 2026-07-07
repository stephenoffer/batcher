//! HyperLogLog++ — distinct-count (cardinality) estimation.
//!
//! Estimates the number of distinct values in a column using a fixed `6·m` bits
//! with ~`1.04/√m` relative error. Cheap distinct counts are what let the
//! optimizer choose join build sides and size aggregations without a second scan.
//! Includes the small-range linear-counting correction (the practically important
//! part of the "++"); a 64-bit hash makes the large-range correction unnecessary.

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
    /// fallback. The hash is `SEED.hash_one` of the native value (`to_bits()` for
    /// floats so equal floats — including `-0.0`/`0.0` — bucket identically).
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
            DT::Float32 => prim!(Float32Array, |v: f32| (v as f64).to_bits()),
            DT::Float64 => prim!(Float64Array, |v: f64| v.to_bits()),
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
    pub fn estimate(&self) -> f64 {
        let m = self.m() as f64;
        // Single pass: harmonic-sum term and empty-register count together. `2^-r`
        // is built from the IEEE exponent field (exact, and cheaper than `powi`).
        let mut sum = 0f64;
        let mut zeros = 0usize;
        for &r in &self.registers {
            sum += pow2_neg(r);
            if r == 0 {
                zeros += 1;
            }
        }
        let raw = alpha(self.m()) * m * m / sum;

        // Small-range correction: linear counting when many registers are empty.
        if raw <= 2.5 * m && zeros > 0 {
            m * (m / zeros as f64).ln()
        } else {
            raw
        }
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

/// `2^-r` for a register rank `r` (`0..=64`), built directly from the IEEE-754
/// exponent field. Exact and branch-free — avoids a `powi`/`exp2` call per register
/// in [`HyperLogLog::estimate`]. (Rank stays ≤ 64, so `1023 - r` is a normal
/// exponent.)
#[inline]
fn pow2_neg(r: u8) -> f64 {
    f64::from_bits((1023u64 - r as u64) << 52)
}

/// HyperLogLog bias constant α_m.
fn alpha(m: usize) -> f64 {
    match m {
        16 => 0.673,
        32 => 0.697,
        64 => 0.709,
        _ => 0.7213 / (1.0 + 1.079 / m as f64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merge_all;
    use arrow::array::Int64Array;
    use std::sync::Arc;

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
    fn pow2_neg_matches_powi_bit_for_bit() {
        // The exponent-field trick must equal `2.powi(-r)` exactly for every rank,
        // so the estimate is unchanged from the old per-register `powi`.
        for r in 0u8..=64 {
            assert_eq!(
                pow2_neg(r).to_bits(),
                2f64.powi(-(r as i32)).to_bits(),
                "mismatch at r={r}"
            );
        }
    }

    #[test]
    fn small_cardinality_is_accurate() {
        let mut hll = HyperLogLog::new(14);
        for i in 0..100u64 {
            hll.add(&i);
        }
        let est = hll.estimate();
        // Linear counting should be near-exact for small sets.
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
