//! `list.jaccard` over string element types, and the non-zero-set Jaccard of two vectors.
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
            let eq = if let (Some(l), Some(r)) = (left_str, right_str) {
                !l.is_null(li) && !r.is_null(ri) && l.value(li) == r.value(ri)
            } else {
                let (l, r) = (lv.as_string::<i64>(), rv.as_string::<i64>());
                !l.is_null(li) && !r.is_null(ri) && l.value(li) == r.value(ri)
            };
            agree += usize::from(eq);
        }
        b.append_value(agree as f64 / n as f64);
    }
    Ok(Arc::new(b.finish()))
}

/// `jaccard(mode="nonzero")`: `|A∩B| / |A∪B|` over the two vectors' non-zero positions
/// (Daft `jaccard_similarity`).
///
/// A position is in the set when its element is non-null and not equal to zero, so NaN is
/// in and `-0.0` is out. The vectors must have the same length, as Daft's fixed-size
/// embeddings do; a null row on either side is null, and two vectors with no non-zero
/// position have no Jaccard ratio (0/0), so they are null too.
pub(crate) fn jaccard_nonzero(
    la: &arrow::array::ListArray,
    ra: &arrow::array::ListArray,
) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, AsArray, Float64Builder};
    use arrow::compute::cast;
    use arrow::datatypes::Float64Type;

    let lc = cast(la.values(), &DataType::Float64)?;
    let rc = cast(ra.values(), &DataType::Float64)?;
    let (lf, rf) = (
        lc.as_primitive::<Float64Type>(),
        rc.as_primitive::<Float64Type>(),
    );
    let (lo, ro) = (la.value_offsets(), ra.value_offsets());
    let nonzero = |a: &arrow::array::Float64Array, k: usize| a.is_valid(k) && a.value(k) != 0.0;
    let mut b = Float64Builder::with_capacity(la.len());
    for i in 0..la.len() {
        if la.is_null(i) || ra.is_null(i) {
            b.append_null();
            continue;
        }
        let (ls, le) = (lo[i] as usize, lo[i + 1] as usize);
        let (rs, re) = (ro[i] as usize, ro[i + 1] as usize);
        if le - ls != re - rs {
            return Err(ExprError::InvalidArgument {
                func: "list.jaccard(mode='nonzero')".into(),
                reason: format!(
                    "list dimensions must be equal, got left length {} and right length {}",
                    le - ls,
                    re - rs
                ),
            });
        }
        let (mut both, mut either) = (0usize, 0usize);
        for k in 0..(le - ls) {
            let (l, r) = (nonzero(lf, ls + k), nonzero(rf, rs + k));
            both += usize::from(l && r);
            either += usize::from(l || r);
        }
        if either == 0 {
            b.append_null();
        } else {
            b.append_value(both as f64 / either as f64);
        }
    }
    Ok(Arc::new(b.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{AsArray, Float64Builder, ListBuilder};
    use arrow::datatypes::Float64Type;

    fn vectors(rows: &[Option<Vec<Option<f64>>>]) -> ArrayRef {
        let mut b = ListBuilder::new(Float64Builder::new());
        for row in rows {
            match row {
                Some(vs) => {
                    for v in vs {
                        b.values().append_option(*v);
                    }
                    b.append(true);
                }
                None => b.append(false),
            }
        }
        Arc::new(b.finish())
    }

    /// Daft 0.7.25 `jaccard_similarity`: `[1, NaN, -0.0]` against `[0, 1, 1]` is 1/3 —
    /// NaN is a non-zero position and `-0.0` is not — and `[1, 2, 3]` against `[3, 1, 5]`
    /// is 1.0 where the positional agreement rate is 0.0.
    #[test]
    fn nonzero_sets_follow_daft() {
        let l = vectors(&[
            Some(vec![Some(1.0), Some(f64::NAN), Some(-0.0)]),
            Some(vec![Some(1.0), Some(2.0), Some(3.0)]),
            Some(vec![Some(0.0), None]),
            None,
        ]);
        let r = vectors(&[
            Some(vec![Some(0.0), Some(1.0), Some(1.0)]),
            Some(vec![Some(3.0), Some(1.0), Some(5.0)]),
            Some(vec![Some(0.0), Some(0.0)]),
            Some(vec![Some(1.0)]),
        ]);
        let out = jaccard_nonzero(l.as_list::<i32>(), r.as_list::<i32>()).unwrap();
        let got: Vec<Option<f64>> = out.as_primitive::<Float64Type>().iter().collect();
        assert_eq!(got, vec![Some(1.0 / 3.0), Some(1.0), None, None]);
        let short = vectors(&[Some(vec![Some(1.0)])]);
        assert!(jaccard_nonzero(l.slice(0, 1).as_list::<i32>(), short.as_list::<i32>()).is_err());
    }
}
