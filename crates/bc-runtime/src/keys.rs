//! The one canonical form for grouping/partitioning keys.
//!
//! Every path that turns a key column into a hash — the group assigner
//! (`agg::group::assign`), the radix combine (`agg::group::combine`), and the shuffle
//! (`shuffle`) — MUST agree on what makes two keys "the same". They are separate code
//! paths for performance reasons, but they answer one semantic question, so the answer
//! lives here once.
//!
//! Getting this wrong does not merely reorder rows: if the shuffle disagrees with the
//! assigner about key identity, two rows that are one group land on different reducers
//! and the query returns **two groups where the oracle returns one** — a silent
//! wrong-answer bug that only appears once the input is big enough to shuffle. Both
//! sides of that divergence have happened here (a float key split across `-0.0`/`0.0`;
//! null int keys scattered across every bucket), which is why the policy is centralized
//! rather than restated per call site.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array};
use arrow::datatypes::DataType;

/// A fixed hash for null keys so every null row lands in one partition — and therefore one
/// group. Grouping inside the partition still compares keys, so a non-null value that
/// collides with this hash is never conflated with null; only co-location depends on it.
pub(crate) const NULL_HASH: u64 = 0xa5a5_5a5a_dead_beef;

/// Canonical `u64` key bits for an `f64`.
///
/// Every NaN bit-pattern maps to one value and negative zero maps to positive zero, so
/// raw-bit hashing and equality agree with SQL `GROUP BY` semantics: all NaNs form one
/// group and the two zeros form one group (matching DuckDB). Every other value keeps its
/// exact bits, so distinct finite values stay distinct.
#[inline]
pub(crate) fn canon_f64(v: f64) -> u64 {
    if v.is_nan() {
        0x7ff8_0000_0000_0000 // one canonical quiet NaN
    } else if v == 0.0 {
        0 // +0.0 bits (folds -0.0 into 0.0)
    } else {
        v.to_bits()
    }
}

/// Total order over `f64` for `min`/`max`, matching the order `ORDER BY` sorts in.
///
/// Raw IEEE comparison is *not* a total order: every comparison with NaN is false, so a
/// `max()` written as `v > cur` silently **ignores** NaN and returns the largest non-NaN value.
/// That contradicts our own `ORDER BY`, which sorts NaN last (i.e. treats it as the greatest
/// value) — so `max(x)` disagreed with `SELECT x ORDER BY x DESC LIMIT 1` on the same column,
/// and with DuckDB. Here all NaNs compare equal and greater than every number, which is the
/// order the sort already uses.
///
/// `-0.0` and `0.0` compare `Equal` (as they do in `=`, `GROUP BY`, and [`canon_f64`]), so which
/// of the two an extreme returns is first-seen — the same rule every other engine applies.
#[inline]
pub(crate) fn float_total_cmp(a: f64, b: f64) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        // Neither is NaN, so `partial_cmp` is total here.
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

/// Rewrite float key columns into their canonical form, leaving every other column as-is.
///
/// The raw-hash paths call [`canon_f64`] per value, but the general shuffle path encodes
/// keys with arrow's `RowConverter`, whose float encoding is *not* canonical: it maps
/// `-0.0` and `0.0` to different bytes, so they would hash to different buckets and never
/// meet at a reducer. Canonicalizing the array up front means every downstream encoder —
/// `RowConverter` included — sees the same bits the assigner groups by, so no key-identity
/// policy has to be restated inside each encoder.
///
/// Returns `None` when there is nothing to rewrite (the common case: no float key), so the
/// caller keeps using the original arrays with no allocation.
pub(crate) fn canonicalize_float_keys(keys: &[ArrayRef]) -> Option<Vec<ArrayRef>> {
    if !keys.iter().any(|k| k.data_type() == &DataType::Float64) {
        return None;
    }
    Some(
        keys.iter()
            .map(|k| {
                let Some(f) = k.as_any().downcast_ref::<Float64Array>() else {
                    return Arc::clone(k);
                };
                // `canon_f64` returns hash bits; map back to the f64 the bits denote so the
                // column keeps its Float64 type (the encoder still sees a float array, just a
                // canonical one). `f64::from_bits` round-trips every value canon_f64 emits.
                let canon: Float64Array = f
                    .iter()
                    .map(|v| v.map(|x| f64::from_bits(canon_f64(x))))
                    .collect();
                Arc::new(canon) as ArrayRef
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canon_folds_both_zeros_and_all_nans() {
        assert_eq!(canon_f64(0.0), canon_f64(-0.0));
        assert_eq!(canon_f64(f64::NAN), canon_f64(-f64::NAN));
        assert_eq!(
            canon_f64(f64::from_bits(0x7ff8_0000_0000_0001)),
            canon_f64(f64::NAN)
        );
        // distinct finite values stay distinct
        assert_ne!(canon_f64(1.0), canon_f64(1.5));
        assert_ne!(canon_f64(1.0), canon_f64(-1.0));
    }

    #[test]
    fn canonicalize_rewrites_negative_zero_and_preserves_nulls() {
        let keys: Vec<ArrayRef> = vec![Arc::new(Float64Array::from(vec![
            Some(-0.0),
            Some(0.0),
            None,
            Some(2.5),
        ]))];
        let out = canonicalize_float_keys(&keys).expect("float key present");
        let f = out[0].as_any().downcast_ref::<Float64Array>().unwrap();
        // both zeros now carry the same bits, so any encoder buckets them together
        assert_eq!(f.value(0).to_bits(), f.value(1).to_bits());
        assert!(f.is_null(2));
        assert_eq!(f.value(3), 2.5);
    }

    #[test]
    fn canonicalize_skips_non_float_keys() {
        use arrow::array::Int64Array;
        let keys: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![1, 2, 3]))];
        assert!(canonicalize_float_keys(&keys).is_none());
    }
}
