//! `list.jaccard` over string element types.
//!
//! The shared numeric path in `list.rs` casts every element array to Float64 before
//! reducing, which is right for the vector distances but wrong for `jaccard`: it is a
//! positional *equality* rate, and Utf8 -> Float64 is a lossy safe cast that yields nulls
//! rather than an error. A string list therefore scored 0.0 agreement against an identical
//! copy of itself. This module is the native comparison that case needs.

use std::sync::Arc;

use arrow::array::ArrayRef;
use arrow::datatypes::DataType;

use crate::ExprError;

/// Whether a `List` column's elements are one of the string types `jaccard` compares natively.
pub(crate) fn is_string_list(list: &arrow::array::ListArray) -> bool {
    use arrow::array::Array;
    matches!(
        list.values().data_type(),
        DataType::Utf8 | DataType::LargeUtf8
    )
}

/// `jaccard` over two `List<Utf8>` columns: the fraction of positions holding equal strings.
///
/// Mirrors the numeric branch's contract exactly — a null row on either side is null, an
/// empty shared prefix is null (no positions to agree on), and comparison runs over
/// `min(len_a, len_b)` positions.
pub(crate) fn jaccard_utf8(
    la: &arrow::array::ListArray,
    ra: &arrow::array::ListArray,
) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, AsArray, Float64Builder};

    let (lv, rv) = (la.values(), ra.values());
    let left_str = lv.as_string_opt::<i32>();
    let right_str = rv.as_string_opt::<i32>();
    let (lo, ro) = (la.value_offsets(), ra.value_offsets());
    let mut b = Float64Builder::with_capacity(la.len());
    for i in 0..la.len() {
        if la.is_null(i) || ra.is_null(i) {
            b.append_null();
            continue;
        }
        let (ls, le) = (lo[i] as usize, lo[i + 1] as usize);
        let (rs, re) = (ro[i] as usize, ro[i + 1] as usize);
        let n = (le - ls).min(re - rs);
        if n == 0 {
            b.append_null();
            continue;
        }
        let mut agree = 0usize;
        for k in 0..n {
            let (li, ri) = (ls + k, rs + k);
            // A null element on either side is a disagreement, matching the numeric branch,
            // where a null cannot equal anything.
            let eq = match (left_str, right_str) {
                (Some(l), Some(r)) => {
                    !l.is_null(li) && !r.is_null(ri) && l.value(li) == r.value(ri)
                }
                _ => {
                    let (l, r) = (lv.as_string::<i64>(), rv.as_string::<i64>());
                    !l.is_null(li) && !r.is_null(ri) && l.value(li) == r.value(ri)
                }
            };
            agree += usize::from(eq);
        }
        b.append_value(agree as f64 / n as f64);
    }
    Ok(Arc::new(b.finish()))
}
