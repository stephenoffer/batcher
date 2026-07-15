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

/// Canonical `f32` bits, mirroring [`canon_f64`] for a 32-bit float.
///
/// Top-level `Float32` keys are widened to `Float64` at the FFI boundary, but a `Float32`
/// *nested* inside a list/struct key is not, so a float leaf can still arrive as `f32` and
/// must fold `-0.0`/all-NaN the same way. Every NaN → one quiet NaN, `-0.0` → `0.0`.
#[inline]
fn canon_f32(v: f32) -> f32 {
    if v.is_nan() {
        f32::from_bits(0x7fc0_0000) // one canonical quiet NaN
    } else if v == 0.0 {
        0.0 // folds -0.0 into 0.0
    } else {
        v
    }
}

/// Whether a data type has a floating-point leaf that needs canonicalizing — a top-level
/// float, or a float nested inside a list/struct key. Dictionary and top-level narrow
/// floats are decoded/widened at the FFI boundary, so only these shapes reach the engine.
fn contains_float(dt: &DataType) -> bool {
    match dt {
        DataType::Float32 | DataType::Float64 => true,
        DataType::List(f) | DataType::LargeList(f) | DataType::FixedSizeList(f, _) => {
            contains_float(f.data_type())
        }
        DataType::Struct(fields) => fields.iter().any(|f| contains_float(f.data_type())),
        _ => false,
    }
}

/// Canonicalize the float leaves of one array, preserving its type, offsets, and nulls.
///
/// Returns `None` when the array holds no float leaf (nothing to rewrite), so a
/// non-float column is passed through with no allocation. Recurses through
/// `List`/`LargeList`/`FixedSizeList`/`Struct` so a float buried in a nested key is folded
/// exactly as a top-level one — the `RowConverter` these keys are later encoded with splits
/// `-0.0`/`0.0` (and every NaN) at *every* depth, not just the top.
fn canon_array(arr: &ArrayRef) -> Option<ArrayRef> {
    use arrow::array::{FixedSizeListArray, Float32Array, LargeListArray, ListArray, StructArray};
    match arr.data_type() {
        DataType::Float64 => {
            let f = arr.as_any().downcast_ref::<Float64Array>()?;
            // `canon_f64` returns hash bits; map back to the f64 they denote so the column
            // keeps its Float64 type. `f64::from_bits` round-trips every value it emits.
            let canon: Float64Array = f
                .iter()
                .map(|v| v.map(|x| f64::from_bits(canon_f64(x))))
                .collect();
            Some(Arc::new(canon) as ArrayRef)
        }
        DataType::Float32 => {
            let f = arr.as_any().downcast_ref::<Float32Array>()?;
            let canon: Float32Array = f.iter().map(|v| v.map(canon_f32)).collect();
            Some(Arc::new(canon) as ArrayRef)
        }
        DataType::List(field) => {
            let l = arr.as_any().downcast_ref::<ListArray>()?;
            let child = canon_array(l.values())?;
            Some(Arc::new(ListArray::new(
                field.clone(),
                l.offsets().clone(),
                child,
                l.nulls().cloned(),
            )) as ArrayRef)
        }
        DataType::LargeList(field) => {
            let l = arr.as_any().downcast_ref::<LargeListArray>()?;
            let child = canon_array(l.values())?;
            Some(Arc::new(LargeListArray::new(
                field.clone(),
                l.offsets().clone(),
                child,
                l.nulls().cloned(),
            )) as ArrayRef)
        }
        DataType::FixedSizeList(field, size) => {
            let l = arr.as_any().downcast_ref::<FixedSizeListArray>()?;
            let child = canon_array(l.values())?;
            Some(Arc::new(FixedSizeListArray::new(
                field.clone(),
                *size,
                child,
                l.nulls().cloned(),
            )) as ArrayRef)
        }
        DataType::Struct(fields) => {
            let s = arr.as_any().downcast_ref::<StructArray>()?;
            let mut changed = false;
            let cols: Vec<ArrayRef> = s
                .columns()
                .iter()
                .map(|c| match canon_array(c) {
                    Some(n) => {
                        changed = true;
                        n
                    }
                    None => Arc::clone(c),
                })
                .collect();
            // A struct with no float leaf changed nothing — pass it through untouched.
            changed.then(|| {
                Arc::new(StructArray::new(fields.clone(), cols, s.nulls().cloned())) as ArrayRef
            })
        }
        _ => None,
    }
}

/// Rewrite float key columns into their canonical form, leaving every other column as-is.
///
/// The raw-hash paths call [`canon_f64`] per value, but the general shuffle/join path
/// encodes keys with arrow's `RowConverter`, whose float encoding is *not* canonical: it
/// maps `-0.0` and `0.0` to different bytes (and each NaN bit-pattern to its own), so they
/// would hash to different buckets and never meet at a reducer. Canonicalizing the array up
/// front means every downstream encoder — `RowConverter` included — sees the same bits the
/// assigner groups by, so no key-identity policy has to be restated inside each encoder.
///
/// A float leaf **nested** inside a list or struct key is canonicalized too: the encoder
/// splits `-0.0`/`0.0` at every depth, so a `GROUP BY`/`JOIN`/`DISTINCT` on a `List<Float64>`
/// or `Struct{Float64}` column silently split one group/dropped one match without this.
///
/// Returns `None` when there is nothing to rewrite (the common case: no float key), so the
/// caller keeps using the original arrays with no allocation.
pub(crate) fn canonicalize_float_keys(keys: &[ArrayRef]) -> Option<Vec<ArrayRef>> {
    if !keys.iter().any(|k| contains_float(k.data_type())) {
        return None;
    }
    Some(
        keys.iter()
            .map(|k| canon_array(k).unwrap_or_else(|| Arc::clone(k)))
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

    /// A `RowConverter` encodes two `Rows` equal iff their canonical bytes match. This
    /// helper drives the actual encoder used by the join/shuffle general path so a test
    /// proves the *observable* key identity, not just the array bits.
    fn rows_equal_after_canon(a: ArrayRef, b: ArrayRef) -> bool {
        use arrow::row::{RowConverter, SortField};
        let dt = a.data_type().clone();
        let ca = canon_array(&a).unwrap_or(a);
        let cb = canon_array(&b).unwrap_or(b);
        let conv = RowConverter::new(vec![SortField::new(dt)]).unwrap();
        let ra = conv.convert_columns(&[ca]).unwrap();
        let rb = conv.convert_columns(&[cb]).unwrap();
        ra.row(0) == rb.row(0)
    }

    /// A float nested in a `List` key must fold `-0.0`/`0.0` (and NaN) — otherwise a join
    /// on a list-of-floats column silently drops the match. This is the bug the recursion
    /// fixes; without it `RowConverter` gives `List[-0.0]` and `List[0.0]` different bytes.
    #[test]
    fn canonicalize_folds_float_inside_list_key() {
        use arrow::array::{Float64Array, ListArray};
        use arrow::buffer::OffsetBuffer;
        use arrow::datatypes::Field;
        let field = Arc::new(Field::new("item", DataType::Float64, true));
        let mk = |v: f64| -> ArrayRef {
            Arc::new(ListArray::new(
                field.clone(),
                OffsetBuffer::new(vec![0, 1].into()),
                Arc::new(Float64Array::from(vec![v])),
                None,
            ))
        };
        assert!(
            rows_equal_after_canon(mk(-0.0), mk(0.0)),
            "List[-0.0] must equal List[0.0]"
        );
        assert!(
            rows_equal_after_canon(mk(f64::NAN), mk(f64::from_bits(0x7ff8_0000_0000_0001))),
            "List[NaN] patterns must unify"
        );
        // Distinct finite values inside a list must stay distinct.
        assert!(!rows_equal_after_canon(mk(1.0), mk(2.0)));
    }

    /// A float nested in a `Struct` key field must fold too (a `GROUP BY struct_col`).
    #[test]
    fn canonicalize_folds_float_inside_struct_key() {
        use arrow::array::{Float64Array, Int64Array, StructArray};
        use arrow::datatypes::{Field, Fields};
        let fields: Fields = vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Float64, false),
        ]
        .into();
        let mk = |f: f64| -> ArrayRef {
            Arc::new(StructArray::new(
                fields.clone(),
                vec![
                    Arc::new(Int64Array::from(vec![7])) as ArrayRef,
                    Arc::new(Float64Array::from(vec![f])) as ArrayRef,
                ],
                None,
            ))
        };
        assert!(
            rows_equal_after_canon(mk(-0.0), mk(0.0)),
            "struct with -0.0 field must equal struct with 0.0 field"
        );
        assert!(!rows_equal_after_canon(mk(1.0), mk(2.0)));
    }

    /// End-to-end: a hash join keyed on a `List<Float64>` must match `-0.0` with `0.0`.
    /// Before the recursion this returned zero rows (a silent dropped match).
    #[test]
    fn join_on_list_float_key_matches_signed_zero() {
        use crate::join::{hash_join_indices, JoinType};
        use arrow::array::{Float64Array, ListArray};
        use arrow::buffer::OffsetBuffer;
        use arrow::datatypes::Field;
        let field = Arc::new(Field::new("item", DataType::Float64, true));
        let mk = |v: f64| -> Vec<ArrayRef> {
            vec![Arc::new(ListArray::new(
                field.clone(),
                OffsetBuffer::new(vec![0, 1].into()),
                Arc::new(Float64Array::from(vec![v])),
                None,
            ))]
        };
        let idx = hash_join_indices(&mk(-0.0), &mk(0.0), JoinType::Inner).unwrap();
        assert_eq!(idx.left.len(), 1, "list<-0.0> must join list<0.0>");
    }

    /// A list of a non-float type is left untouched (no needless allocation / rewrite).
    #[test]
    fn canonicalize_skips_non_float_nested_keys() {
        use arrow::array::{Int64Array, ListArray};
        use arrow::buffer::OffsetBuffer;
        use arrow::datatypes::Field;
        let field = Arc::new(Field::new("item", DataType::Int64, true));
        let list: ArrayRef = Arc::new(ListArray::new(
            field,
            OffsetBuffer::new(vec![0, 1].into()),
            Arc::new(Int64Array::from(vec![1])),
            None,
        ));
        assert!(canonicalize_float_keys(&[list]).is_none());
    }
}
