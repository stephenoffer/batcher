//! Input coercion and the numeric inner loop shared by the vector-distance kernels.
//!
//! Two concerns live here, both split out of `eval/list.rs` to keep that file inside
//! its size limit:
//!
//! 1. **`FixedSizeList` acceptance.** `.image.to_tensor()` and every embedding column
//!    written through `io/formats/ml/tensor.py` are `FixedSizeList` (that is what
//!    `arrow.fixed_shape_tensor` is built on). The vector kernels historically accepted
//!    only `List`, so the one Arrow type the project *designed* for embeddings was
//!    rejected by the very kernels meant to consume it. `as_var_list` normalizes both
//!    into a `List<i32>` view so the kernels have a single shape to reason about.
//! 2. **A `Float32` fast path.** Embeddings are almost universally `f32`. Casting the
//!    values child to `Float64` up front doubles the bytes the inner loop streams — at
//!    768 dims that is ~3 KB/row of avoidable memory traffic. Accumulating in `f64`
//!    from `f32` inputs is *bit-for-bit* identical to casting first, because `f32 → f64`
//!    is exact and every accumulation still happens at `f64` width.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, PrimitiveArray};
use arrow::compute::cast;
use arrow::datatypes::{ArrowPrimitiveType, DataType, Field};

use crate::ExprError;

/// Normalize a list-like array into a `List<i32>`.
///
/// This is the single place the three list encodings become one, so every `.list` method
/// means the same thing whichever one a column arrives in:
///
/// * **`List`** passes through untouched (a cheap `Arc` clone).
/// * **`FixedSizeList`** is cast, materializing an `n+1` offset buffer but sharing the
///   values buffer — the offsets are negligible next to the embedding payload. This is
///   how Arrow and Parquet store a vector column, and what DuckDB's `ARRAY` maps to.
/// * **`LargeList`** is cast down to 32-bit offsets. This is the encoding Arrow reaches
///   for once a list column passes `i32::MAX` offsets, and it is what an Arrow reader
///   hands back for a `large_list` Parquet column regardless of the actual size — so
///   rejecting it made the whole `.list` namespace unusable on data that is not
///   necessarily large at all.
///
/// The `LargeList` narrowing is safe *per morsel*, not in general: execution is batched
/// at 16,384 rows, so a single batch would need over 131,000 child elements per row on
/// average to exceed a 32-bit offset. Beyond that the Arrow cast **errors** rather than
/// truncating, so the failure is loud. Making the kernels generic over the offset type is
/// the fix that removes the bound entirely; it touches every kernel and is not this
/// change.
///
/// Anything else is a type error naming the function and the offending side, so the
/// message stays actionable.
pub(crate) fn as_var_list(arr: &ArrayRef, func: &str) -> Result<ArrayRef, ExprError> {
    match arr.data_type() {
        DataType::List(_) => Ok(Arc::clone(arr)),
        DataType::FixedSizeList(field, _) | DataType::LargeList(field) => {
            let target = DataType::List(Arc::new(Field::new(
                field.name(),
                field.data_type().clone(),
                field.is_nullable(),
            )));
            cast(arr, &target).map_err(ExprError::from)
        }
        other => Err(ExprError::ExpectedType {
            func: func.to_string(),
            want: "a List argument",
            got: other.to_string(),
        }),
    }
}

/// The four running sums every vector-distance kernel needs, plus the minhash
/// agreement count. One pass computes all of them so `dot`/`cosine`/`l2` share a
/// single traversal of the pair.
#[derive(Default)]
pub(crate) struct PairSums {
    pub dot: f64,
    pub lnorm: f64,
    pub rnorm: f64,
    pub dist2: f64,
    /// `Σ |xᵢ − yᵢ|` — the L1 / Manhattan distance.
    pub dist1: f64,
    /// Count of positions where the two values are equal (the minhash agreement count).
    pub agree: usize,
    /// Count of positions where the two values differ (the Hamming distance). This is
    /// `pairs − agree`, tracked directly so a null-dropped pair counts as neither.
    pub disagree: usize,
}

/// Accumulate the running sums over `n` paired elements starting at `ls`/`rs`.
///
/// Generic over the element type so the `Float32` and `Float64` children share one
/// implementation. Accumulators are always `f64` regardless of `T`, which is what makes
/// the `f32` path numerically identical to casting the child to `f64` first.
///
/// A null on either side drops that pair, matching the documented kernel semantics.
pub(crate) fn accumulate_pair<T>(
    lf: &PrimitiveArray<T>,
    rf: &PrimitiveArray<T>,
    ls: usize,
    rs: usize,
    n: usize,
) -> PairSums
where
    T: ArrowPrimitiveType,
    T::Native: Into<f64> + PartialEq,
{
    let mut s = PairSums::default();
    for k in 0..n {
        let (lk, rk) = (ls + k, rs + k);
        if !lf.is_valid(lk) || !rf.is_valid(rk) {
            continue;
        }
        let (lv, rv) = (lf.value(lk), rf.value(rk));
        // Exact for minhash signatures, whose values are bounded to 32 bits. Compared at
        // the native width before widening, so `f32` signatures agree exactly.
        let equal = lv == rv;
        s.agree += usize::from(equal);
        s.disagree += usize::from(!equal);
        let (x, y): (f64, f64) = (lv.into(), rv.into());
        s.dot += x * y;
        s.lnorm += x * x;
        s.rnorm += y * y;
        let diff = x - y;
        s.dist2 += diff * diff;
        s.dist1 += diff.abs();
    }
    s
}
/// Both list children under one element type, so a single `RowConverter` can encode them.
///
/// A `List<Null>` is what an all-empty (or all-null) list column infers to, and it turns up in
/// ordinary use: one side of a comparison whose rows all came back empty. Concatenating its
/// child with a `List<Utf8>` child fails outright, so an empty retrieval against a populated
/// one raised where it should have scored zero. Casting the `Null` child to the other side's
/// type is exact — it has no values to lose — and leaves the answer the same.
///
/// A genuine element-type mismatch (`List<Utf8>` against `List<Int64>`) still reaches the
/// caller's `concat`, which reports it as the error it is.
pub(crate) fn align_children(
    left: &arrow::array::ArrayRef,
    right: &arrow::array::ArrayRef,
) -> Result<(arrow::array::ArrayRef, arrow::array::ArrayRef), ExprError> {
    use arrow::compute::cast;
    use arrow::datatypes::DataType::Null;

    match (left.data_type(), right.data_type()) {
        (Null, other) if !matches!(other, Null) => Ok((cast(left, other)?, right.clone())),
        (other, Null) if !matches!(other, Null) => Ok((left.clone(), cast(right, other)?)),
        _ => Ok((left.clone(), right.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        ArrayRef, AsArray, FixedSizeListArray, Float32Array, Float64Array, ListArray,
    };
    use arrow::datatypes::{Float32Type, Float64Type};

    fn fsl(values: Vec<f32>, size: i32) -> ArrayRef {
        let child = Arc::new(Float32Array::from(values)) as ArrayRef;
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        Arc::new(FixedSizeListArray::new(field, size, child, None)) as ArrayRef
    }

    #[test]
    fn fixed_size_list_normalizes_to_list() {
        let out = as_var_list(&fsl(vec![1.0, 2.0, 3.0, 4.0], 2), "list.Dot").unwrap();
        assert!(matches!(out.data_type(), DataType::List(_)));
        let list = out.as_list::<i32>();
        assert_eq!(list.len(), 2);
        assert_eq!(list.value_offsets(), &[0, 2, 4]);
    }

    #[test]
    fn var_list_passes_through_unchanged() {
        let src: ArrayRef = Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>(vec![
            Some(vec![Some(1.0), Some(2.0)]),
        ]));
        let out = as_var_list(&src, "list.Dot").unwrap();
        assert_eq!(out.data_type(), src.data_type());
    }

    #[test]
    fn non_list_input_is_a_typed_error() {
        let src: ArrayRef = Arc::new(Float64Array::from(vec![1.0]));
        assert!(as_var_list(&src, "list.Dot").is_err());
    }

    /// The whole justification for the `f32` path: it must agree with the widened one
    /// exactly, not approximately.
    #[test]
    fn f32_accumulation_matches_f64_bit_for_bit() {
        let raw: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.2).collect();
        let other: Vec<f32> = (0..64).map(|i| (i as f32) * -0.07 + 1.5).collect();

        let a32 = Float32Array::from(raw.clone());
        let b32 = Float32Array::from(other.clone());
        let a64 = Float64Array::from(raw.iter().map(|&v| v as f64).collect::<Vec<_>>());
        let b64 = Float64Array::from(other.iter().map(|&v| v as f64).collect::<Vec<_>>());

        let s32 = accumulate_pair::<Float32Type>(&a32, &b32, 0, 0, 64);
        let s64 = accumulate_pair::<Float64Type>(&a64, &b64, 0, 0, 64);

        assert_eq!(s32.dot.to_bits(), s64.dot.to_bits());
        assert_eq!(s32.lnorm.to_bits(), s64.lnorm.to_bits());
        assert_eq!(s32.rnorm.to_bits(), s64.rnorm.to_bits());
        assert_eq!(s32.dist2.to_bits(), s64.dist2.to_bits());
        assert_eq!(s32.agree, s64.agree);
    }

    #[test]
    fn nulls_drop_the_pair_on_either_side() {
        let a = Float32Array::from(vec![Some(1.0), None, Some(3.0)]);
        let b = Float32Array::from(vec![Some(1.0), Some(2.0), None]);
        let s = accumulate_pair::<Float32Type>(&a, &b, 0, 0, 3);
        assert_eq!(s.dot, 1.0);
        assert_eq!(s.agree, 1);
    }
}
