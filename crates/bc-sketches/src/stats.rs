//! Per-column statistics derived from a single scan.
//!
//! Bundles the cheap, mergeable summaries the optimizer wants for one column:
//! row/null counts, a distinct-count estimate (HLL), and — for numeric columns —
//! a quantile sketch (KLL) that yields range selectivity, histogram boundaries,
//! and exact min/max. One `ColumnStats` per partition merges into the column's
//! global stats, mirroring the engine's partial→combine discipline.

use arrow::array::{Array, ArrayRef};

use crate::{HyperLogLog, KllSketch, Mergeable};

/// Cheap, mergeable statistics for one column, computed in a single pass.
#[derive(Clone)]
pub struct ColumnStats {
    pub count: usize,
    pub null_count: usize,
    /// Total in-memory Arrow buffer bytes across all values seen, for average
    /// per-row byte-width estimation (`avg_byte_width`). Additive across
    /// partitions, so it merges with the same partial→combine discipline as the
    /// rest of the sketch. This is the *measured* width the cost model uses for
    /// variable-width columns, where `bc_arrow::fixed_width` returns `None`.
    total_bytes: u64,
    distinct: HyperLogLog,
    /// Present only for numeric/temporal columns (quantiles need an ordered domain).
    quantiles: Option<KllSketch>,
}

impl ColumnStats {
    /// Compute stats for an Arrow array. A quantile sketch is built only when the
    /// column is numeric/temporal; otherwise `quantiles` stays `None`.
    pub fn from_array(array: &ArrayRef) -> Self {
        let mut distinct = HyperLogLog::default_precision();
        distinct.add_array(array);

        let mut kll = KllSketch::default();
        kll.add_array(array); // no-op for non-numeric types
        let quantiles = (!kll.is_empty()).then_some(kll);

        Self {
            count: array.len(),
            null_count: array.null_count(),
            total_bytes: array.get_array_memory_size() as u64,
            distinct,
            quantiles,
        }
    }

    /// Average in-memory bytes per row — the column's measured byte width
    /// (`total_bytes / count`), `0.0` when empty. For variable-width columns
    /// (`Utf8`/`Binary`/`List`/…), where `bc_arrow::fixed_width` cannot give a
    /// static width, this is what the cost model uses to size memory, IO, and
    /// broadcast eligibility in bytes rather than a uniform per-row guess.
    ///
    /// It is the *whole-buffer* footprint (values + offsets + validity) divided
    /// by rows, so it slightly over-counts fixed overhead on tiny columns — the
    /// intent is a memory-true estimate, not the logical payload size.
    pub fn avg_byte_width(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_bytes as f64 / self.count as f64
        }
    }

    /// Estimated number of distinct (non-null) values.
    pub fn distinct_estimate(&self) -> f64 {
        self.distinct.estimate()
    }

    /// Estimated selectivity of `col <= x` (fraction of rows kept), if numeric.
    pub fn rank(&self, x: f64) -> Option<f64> {
        self.quantiles.as_ref().map(|q| q.rank(x))
    }

    /// Approximate value at quantile `q ∈ [0, 1]`, if numeric.
    pub fn quantile(&self, q: f64) -> Option<f64> {
        self.quantiles.as_ref().and_then(|s| s.quantile(q))
    }

    /// Exact min / max for numeric columns.
    pub fn min(&self) -> Option<f64> {
        self.quantiles.as_ref().and_then(|s| s.min())
    }
    pub fn max(&self) -> Option<f64> {
        self.quantiles.as_ref().and_then(|s| s.max())
    }

    // ---- Selectivity helpers (cost-model building blocks) -----------------
    //
    // These wrap the distinct/quantile sketches in the shapes a predicate cost
    // model wants. Range selectivities need the KLL sketch and so return `None`
    // for non-numeric columns; equality and null fractions only need the
    // distinct/count fields and are always available. Every selectivity is
    // clamped to `[0.0, 1.0]`.

    /// Estimated fraction of rows kept by `col = <value>`, under a uniform
    /// distribution: `1 / distinct`. Independent of the literal (the uniform
    /// model assigns every distinct value the same mass). Distinct is guarded to
    /// be `>= 1` and the result is clamped to `(0, 1]`.
    pub fn selectivity_eq(&self) -> f64 {
        let distinct = self.distinct_estimate().max(1.0);
        (1.0 / distinct).clamp(f64::MIN_POSITIVE, 1.0)
    }

    /// Estimated fraction of rows kept by `col <= x` — exactly `rank(x)`.
    /// `None` for non-numeric columns.
    pub fn selectivity_le(&self, x: f64) -> Option<f64> {
        self.rank(x).map(|r| r.clamp(0.0, 1.0))
    }

    /// Estimated fraction of rows kept by `col < x`. `None` for non-numeric
    /// columns.
    ///
    /// Approximation: `rank(x)` is the fraction `<= x`; subtracting the
    /// equality mass (`selectivity_eq`, the per-value mass under uniformity)
    /// removes the rows equal to `x`, yielding `max(0, rank(x) - selectivity_eq)`.
    /// This is exact only when `x` is a value present in the column; for an `x`
    /// absent from the column it slightly under-counts, but it is the standard
    /// uniform-model estimate and keeps `lt <= le`.
    pub fn selectivity_lt(&self, x: f64) -> Option<f64> {
        self.rank(x)
            .map(|r| (r - self.selectivity_eq()).clamp(0.0, 1.0))
    }

    /// Estimated fraction of rows kept by `col > x` — `1 - rank(x)`. `None` for
    /// non-numeric columns.
    pub fn selectivity_gt(&self, x: f64) -> Option<f64> {
        self.rank(x).map(|r| (1.0 - r).clamp(0.0, 1.0))
    }

    /// Estimated fraction of rows kept by `col >= x` — `1 - selectivity_lt(x)`.
    /// `None` for non-numeric columns.
    pub fn selectivity_ge(&self, x: f64) -> Option<f64> {
        self.selectivity_lt(x).map(|s| (1.0 - s).clamp(0.0, 1.0))
    }

    /// Fraction of rows that are null: `null_count / count` (0.0 when empty).
    pub fn null_fraction(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            (self.null_count as f64 / self.count as f64).clamp(0.0, 1.0)
        }
    }

    /// Equi-depth histogram boundaries: `buckets + 1` values where boundary `i`
    /// is the `i / buckets` quantile, so each adjacent pair bounds a bucket
    /// holding ~`1 / buckets` of the rows. `None` for non-numeric columns or
    /// when `buckets == 0`.
    pub fn histogram_boundaries(&self, buckets: usize) -> Option<Vec<f64>> {
        if buckets == 0 {
            return None;
        }
        let sketch = self.quantiles.as_ref()?;
        // One sorted pass for all boundaries (vs. re-sorting the sketch per call).
        let qs: Vec<f64> = (0..=buckets).map(|i| i as f64 / buckets as f64).collect();
        sketch.quantiles(&qs).into_iter().collect()
    }

    /// Serialize to a byte blob composing the inner sketches. Layout (LE):
    /// `[count: u64][null_count: u64][total_bytes: u64][hll_len: u64][hll…][has_kll: u8]`
    /// and, when the flag is 1, `[kll_len: u64][kll…]`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.count as u64).to_le_bytes());
        out.extend_from_slice(&(self.null_count as u64).to_le_bytes());
        out.extend_from_slice(&self.total_bytes.to_le_bytes());

        let hll = self.distinct.to_bytes();
        out.extend_from_slice(&(hll.len() as u64).to_le_bytes());
        out.extend_from_slice(&hll);

        match &self.quantiles {
            Some(kll) => {
                out.push(1);
                let kll = kll.to_bytes();
                out.extend_from_slice(&(kll.len() as u64).to_le_bytes());
                out.extend_from_slice(&kll);
            }
            None => out.push(0),
        }
        out
    }

    /// Reconstruct from [`to_bytes`](Self::to_bytes). Returns `None` on truncated
    /// input, an unrecognized flag byte, or a malformed inner sketch.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut pos = 0usize;

        let mut read_u64 = || -> Option<u64> {
            let chunk: [u8; 8] = bytes.get(pos..pos + 8)?.try_into().ok()?;
            pos += 8;
            Some(u64::from_le_bytes(chunk))
        };
        let count = read_u64()? as usize;
        let null_count = read_u64()? as usize;
        let total_bytes = read_u64()?;
        let hll_len = read_u64()? as usize;

        let hll_bytes = bytes.get(pos..pos.checked_add(hll_len)?)?;
        pos += hll_len;
        let distinct = HyperLogLog::from_bytes(hll_bytes)?;

        let &flag = bytes.get(pos)?;
        pos += 1;
        let quantiles = match flag {
            0 => None,
            1 => {
                let kll_len = {
                    let chunk: [u8; 8] = bytes.get(pos..pos + 8)?.try_into().ok()?;
                    pos += 8;
                    u64::from_le_bytes(chunk) as usize
                };
                let kll_bytes = bytes.get(pos..pos.checked_add(kll_len)?)?;
                pos += kll_len;
                Some(KllSketch::from_bytes(kll_bytes)?)
            }
            _ => return None,
        };

        if pos != bytes.len() {
            return None; // trailing garbage → reject
        }

        Some(Self {
            count,
            null_count,
            total_bytes,
            distinct,
            quantiles,
        })
    }
}

impl Mergeable for ColumnStats {
    /// Merge stats from another partition's scan of the same column.
    fn merge(&mut self, other: &ColumnStats) {
        self.count += other.count;
        self.null_count += other.null_count;
        self.total_bytes += other.total_bytes;
        self.distinct.merge(&other.distinct);
        match (self.quantiles.as_mut(), other.quantiles.as_ref()) {
            (Some(a), Some(b)) => a.merge(b),
            (None, Some(b)) => self.quantiles = Some(b.clone()),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use std::sync::Arc;

    #[test]
    fn numeric_column_has_quantiles() {
        let arr: ArrayRef = Arc::new(Int64Array::from((0..10_000).collect::<Vec<_>>()));
        let stats = ColumnStats::from_array(&arr);
        assert_eq!(stats.count, 10_000);
        assert_eq!(stats.min(), Some(0.0));
        assert_eq!(stats.max(), Some(9_999.0));
        let median = stats.quantile(0.5).unwrap();
        assert!((median - 5_000.0).abs() / 10_000.0 < 0.02);
        // ~25% of rows are < 2500.
        assert!((stats.rank(2_500.0).unwrap() - 0.25).abs() < 0.02);
    }

    #[test]
    fn string_column_has_no_quantiles() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c", "a"]));
        let stats = ColumnStats::from_array(&arr);
        assert_eq!(stats.count, 4);
        assert!(stats.quantile(0.5).is_none());
        assert!(stats.rank(0.0).is_none());
        // Distinct still works on any type.
        assert!((stats.distinct_estimate() - 3.0).abs() < 1.0);
    }

    #[test]
    fn bytes_roundtrip_numeric_column() {
        let arr: ArrayRef = Arc::new(Int64Array::from((0..10_000).collect::<Vec<_>>()));
        let stats = ColumnStats::from_array(&arr);
        let back = ColumnStats::from_bytes(&stats.to_bytes()).expect("valid blob");
        assert_eq!(back.count, stats.count);
        assert_eq!(back.null_count, stats.null_count);
        assert!((back.distinct_estimate() - stats.distinct_estimate()).abs() < 1e-9);
        assert_eq!(back.min(), stats.min());
        assert_eq!(back.max(), stats.max());
        assert_eq!(back.quantile(0.5), stats.quantile(0.5));
        assert_eq!(back.rank(2_500.0), stats.rank(2_500.0));
    }

    #[test]
    fn bytes_roundtrip_string_column_has_no_quantiles() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c", "a"]));
        let stats = ColumnStats::from_array(&arr);
        let back = ColumnStats::from_bytes(&stats.to_bytes()).expect("valid blob");
        assert_eq!(back.count, 4);
        assert!(back.quantile(0.5).is_none());
        assert!(back.rank(0.0).is_none());
        assert!((back.distinct_estimate() - stats.distinct_estimate()).abs() < 1e-9);
    }

    #[test]
    fn from_bytes_rejects_malformed() {
        assert!(ColumnStats::from_bytes(&[]).is_none());
        assert!(ColumnStats::from_bytes(&[0; 10]).is_none()); // truncated
        let arr: ArrayRef = Arc::new(Int64Array::from((0..100).collect::<Vec<_>>()));
        let mut bytes = ColumnStats::from_array(&arr).to_bytes();
        bytes.push(0); // trailing garbage
        assert!(ColumnStats::from_bytes(&bytes).is_none());
    }

    #[test]
    fn merge_combines_quantiles_and_distinct() {
        let a_arr: ArrayRef = Arc::new(Int64Array::from((0..5_000).collect::<Vec<_>>()));
        let b_arr: ArrayRef = Arc::new(Int64Array::from((5_000..10_000).collect::<Vec<_>>()));
        let mut a = ColumnStats::from_array(&a_arr);
        a.merge(&ColumnStats::from_array(&b_arr));
        assert_eq!(a.count, 10_000);
        assert_eq!(a.min(), Some(0.0));
        assert_eq!(a.max(), Some(9_999.0));
        assert!((a.distinct_estimate() - 10_000.0).abs() / 10_000.0 < 0.05);
    }

    #[test]
    fn selectivity_numeric_column() {
        let arr: ArrayRef = Arc::new(Int64Array::from((0..10_000).collect::<Vec<_>>()));
        let stats = ColumnStats::from_array(&arr);

        // col <= 2500 keeps ~25% of rows; col > 2500 keeps the rest.
        assert!((stats.selectivity_le(2_500.0).unwrap() - 0.25).abs() < 0.02);
        assert!((stats.selectivity_gt(2_500.0).unwrap() - 0.75).abs() < 0.02);

        // col < 2500 ≈ rank - 1/distinct, essentially the same here.
        assert!((stats.selectivity_lt(2_500.0).unwrap() - 0.25).abs() < 0.02);
        assert!(stats.selectivity_lt(2_500.0).unwrap() <= stats.selectivity_le(2_500.0).unwrap());

        // col >= 2500 = 1 - lt, complements col < 2500.
        let ge = stats.selectivity_ge(2_500.0).unwrap();
        assert!((ge + stats.selectivity_lt(2_500.0).unwrap() - 1.0).abs() < 1e-9);

        // Equality under uniformity ≈ 1 / 10_000 distinct values.
        assert!((stats.selectivity_eq() - 1.0 / 10_000.0).abs() < 1e-4);

        // No nulls in this column.
        assert_eq!(stats.null_fraction(), 0.0);

        // 4 equi-depth buckets → 5 strictly increasing boundaries spanning the range.
        let bounds = stats.histogram_boundaries(4).unwrap();
        assert_eq!(bounds.len(), 5);
        for w in bounds.windows(2) {
            assert!(
                w[0] <= w[1],
                "boundaries must be non-decreasing: {bounds:?}"
            );
        }
        assert!(bounds.first().unwrap().abs() < 100.0);
        assert!((bounds.last().unwrap() - 9_999.0).abs() < 100.0);

        // buckets == 0 → None.
        assert!(stats.histogram_boundaries(0).is_none());
    }

    #[test]
    fn selectivity_string_column() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c", "a"]));
        let stats = ColumnStats::from_array(&arr);

        // Range selectivities need a numeric sketch → None.
        assert!(stats.selectivity_le(0.0).is_none());
        assert!(stats.selectivity_lt(0.0).is_none());
        assert!(stats.selectivity_gt(0.0).is_none());
        assert!(stats.selectivity_ge(0.0).is_none());
        assert!(stats.histogram_boundaries(4).is_none());

        // Equality and null fraction still work on any type.
        assert!(stats.selectivity_eq() > 0.0 && stats.selectivity_eq() <= 1.0);
        // ~3 distinct values → ~1/3 each.
        assert!((stats.selectivity_eq() - 1.0 / 3.0).abs() < 0.1);
        assert_eq!(stats.null_fraction(), 0.0);
    }

    #[test]
    fn avg_byte_width_wider_for_strings_than_ints() {
        let ints: ArrayRef = Arc::new(Int64Array::from((0..1_000).collect::<Vec<_>>()));
        let strings: ArrayRef = Arc::new(StringArray::from(
            (0..1_000).map(|i| format!("value-{i}")).collect::<Vec<_>>(),
        ));
        let iw = ColumnStats::from_array(&ints).avg_byte_width();
        let sw = ColumnStats::from_array(&strings).avg_byte_width();
        assert!(iw > 0.0 && sw > 0.0);
        // A column of ~7-char strings (plus offsets) is wider per row than i64.
        assert!(sw > iw, "string width {sw} should exceed int width {iw}");
    }

    #[test]
    fn avg_byte_width_empty_is_zero() {
        let arr: ArrayRef = Arc::new(Int64Array::from(Vec::<i64>::new()));
        assert_eq!(ColumnStats::from_array(&arr).avg_byte_width(), 0.0);
    }

    #[test]
    fn total_bytes_merges_additively() {
        let a_arr: ArrayRef = Arc::new(Int64Array::from((0..500).collect::<Vec<_>>()));
        let b_arr: ArrayRef = Arc::new(Int64Array::from((500..1_000).collect::<Vec<_>>()));
        let a = ColumnStats::from_array(&a_arr);
        let b = ColumnStats::from_array(&b_arr);
        let (wa, wb) = (a.avg_byte_width(), b.avg_byte_width());
        let mut merged = a;
        merged.merge(&b);
        assert_eq!(merged.count, 1_000);
        // Per-row width is preserved across the additive merge (same dtype both sides).
        let expected = (wa + wb) / 2.0;
        assert!((merged.avg_byte_width() - expected).abs() < 1e-6);
    }

    #[test]
    fn total_bytes_survives_roundtrip() {
        let arr: ArrayRef = Arc::new(StringArray::from(
            (0..1_000).map(|i| format!("v{i}")).collect::<Vec<_>>(),
        ));
        let stats = ColumnStats::from_array(&arr);
        let back = ColumnStats::from_bytes(&stats.to_bytes()).expect("valid blob");
        assert!((back.avg_byte_width() - stats.avg_byte_width()).abs() < 1e-9);
    }

    #[test]
    fn null_fraction_with_nulls() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(1),
            None,
            Some(3),
            None,
            Some(5),
        ]));
        let stats = ColumnStats::from_array(&arr);
        assert_eq!(stats.count, 5);
        assert_eq!(stats.null_count, 2);
        assert!((stats.null_fraction() - 2.0 / 5.0).abs() < 1e-9);
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
        /// A finite f64 in `[-1e6, 1e6)` (never NaN/inf).
        fn finite(&mut self) -> f64 {
            let frac = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            frac * 2_000_000.0 - 1_000_000.0
        }
    }

    #[test]
    fn prop_columnstats_roundtrip() {
        const TRIALS: usize = 200;
        let mut rng = XorShift64::new(0x57A75_57A75_u64 ^ 0xDEAD_BEEF);

        for trial in 0..TRIALS {
            let n = 1 + rng.below(3_000) as usize; // 1..=3000 rows

            // Alternate between Int64 and Float64 arrays across trials.
            let stats = if rng.next_u64() & 1 == 0 {
                let vals: Vec<i64> = (0..n).map(|_| rng.next_u64() as i64).collect();
                let arr: ArrayRef = Arc::new(Int64Array::from(vals));
                ColumnStats::from_array(&arr)
            } else {
                let vals: Vec<f64> = (0..n).map(|_| rng.finite()).collect();
                let arr: ArrayRef = Arc::new(Float64Array::from(vals));
                ColumnStats::from_array(&arr)
            };

            let back = ColumnStats::from_bytes(&stats.to_bytes()).expect("valid blob");

            // Distinct estimate must be bit-identical after roundtrip.
            assert_eq!(
                back.distinct_estimate().to_bits(),
                stats.distinct_estimate().to_bits(),
                "trial {trial}: n={n} distinct_estimate mismatch"
            );
            // Min / max are exact and must match.
            assert_eq!(back.min(), stats.min(), "trial {trial}: min mismatch");
            assert_eq!(back.max(), stats.max(), "trial {trial}: max mismatch");
            // Every quantile (numeric column) must be exactly equal.
            for &q in &[0.0, 0.25, 0.5, 0.75, 1.0] {
                assert_eq!(
                    back.quantile(q),
                    stats.quantile(q),
                    "trial {trial}: n={n} q={q} quantile mismatch"
                );
            }
            assert_eq!(back.count, stats.count);
            assert_eq!(back.null_count, stats.null_count);
        }
    }
}
