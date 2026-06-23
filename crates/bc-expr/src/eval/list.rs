//! List/struct evaluation for `Expr::List`/`ListGet`/`ListContains`/`StructField`
//! (split out of `lib.rs`).

use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, RecordBatch};
use arrow::compute::{cast, is_null};
use arrow::datatypes::DataType;

use crate::eval::binary::eval_binary;
use crate::{BinaryOp, Expr, ExprError, ListBinaryFunc, ListFunc, Literal};

/// Evaluate an array literal `[e0, e1, …]`: each row becomes a `List` whose values
/// are the per-row element values. Elements are coerced to a common type (Utf8 if
/// any is string, else Float64 if any is floating, else the first element's type).
pub(crate) fn eval_array(elements: &[Expr], batch: &RecordBatch) -> Result<ArrayRef, ExprError> {
    use arrow::array::{new_empty_array, Array, ListArray};
    use arrow::buffer::OffsetBuffer;
    use arrow::compute::interleave;
    use arrow::datatypes::Field;

    let n_rows = batch.num_rows();
    let raw: Vec<ArrayRef> = elements
        .iter()
        .map(|e| e.eval(batch))
        .collect::<Result<_, _>>()?;
    let elem_ty = array_common_type(&raw);
    let cols: Vec<ArrayRef> = raw
        .iter()
        .map(|a| cast(a, &elem_ty))
        .collect::<Result<_, _>>()?;

    let n_elem = cols.len();
    let child: ArrayRef = if n_elem == 0 {
        new_empty_array(&elem_ty)
    } else {
        let refs: Vec<&dyn Array> = cols.iter().map(|a| a.as_ref()).collect();
        let mut idx = Vec::with_capacity(n_rows * n_elem);
        for row in 0..n_rows {
            for el in 0..n_elem {
                idx.push((el, row));
            }
        }
        interleave(&refs, &idx)?
    };
    let offsets = OffsetBuffer::from_lengths((0..n_rows).map(|_| n_elem));
    let field = Arc::new(Field::new("item", elem_ty, true));
    Ok(Arc::new(ListArray::new(field, offsets, child, None)))
}

/// `list_join(list, sep)` — concatenate each row's list elements (cast to Utf8,
/// nulls skipped, matching DuckDB `string_agg`) with `sep`. A null/empty list row
/// yields null. → Utf8.
pub(crate) fn eval_list_join(list: &ArrayRef, sep: &str) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, AsArray, StringArray};

    let lst = require_list(list, "list_join")?;
    let elems = cast(lst.values(), &DataType::Utf8)?;
    let elems = elems.as_string::<i32>();
    let offsets = lst.value_offsets();
    let out: StringArray = (0..lst.len())
        .map(|row| {
            if lst.is_null(row) {
                return None;
            }
            let (s, e) = (offsets[row] as usize, offsets[row + 1] as usize);
            let parts: Vec<&str> = (s..e)
                .filter(|&i| elems.is_valid(i))
                .map(|i| elems.value(i))
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join(sep))
            }
        })
        .collect();
    Ok(Arc::new(out))
}

/// The common element type for an array literal: Utf8 dominates, then Float64,
/// else the first element's type (Int64 for an empty literal).
fn array_common_type(arrays: &[ArrayRef]) -> DataType {
    let mut ty = arrays
        .first()
        .map_or(DataType::Int64, |a| a.data_type().clone());
    for a in arrays.iter().skip(1) {
        ty = match (&ty, a.data_type()) {
            (DataType::Utf8, _) | (_, DataType::Utf8) => DataType::Utf8,
            (DataType::Float64, _) | (_, DataType::Float64) => DataType::Float64,
            _ => ty,
        };
    }
    ty
}

/// Downcast to a `List` array or raise a clear type error.
pub(crate) fn require_list<'a>(
    arr: &'a ArrayRef,
    func: &str,
) -> Result<&'a arrow::array::GenericListArray<i32>, ExprError> {
    use arrow::array::AsArray;
    if !matches!(arr.data_type(), DataType::List(_)) {
        return Err(ExprError::ExpectedString {
            func: func.into(),
            got: arr.data_type().to_string(),
        });
    }
    Ok(arr.as_list::<i32>())
}

/// Rebuild a `List` column by choosing, for each row, which child indices (global)
/// to keep and in what order. List-level nulls are preserved. Type-preserving.
pub(crate) fn rebuild_list<F>(
    list: &arrow::array::GenericListArray<i32>,
    per_row: F,
) -> Result<ArrayRef, ExprError>
where
    F: Fn(usize, usize) -> Vec<u32>,
{
    use arrow::array::{Array, ListArray, UInt32Array};
    use arrow::buffer::OffsetBuffer;
    use arrow::compute::take;
    use arrow::datatypes::Field;

    let offsets = list.value_offsets();
    let child = list.values();
    let mut take_idx: Vec<u32> = Vec::new();
    let mut new_offsets: Vec<i32> = Vec::with_capacity(list.len() + 1);
    new_offsets.push(0);
    for i in 0..list.len() {
        if !list.is_null(i) {
            let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
            take_idx.extend(per_row(s, e));
        }
        new_offsets.push(take_idx.len() as i32);
    }
    let taken = take(child.as_ref(), &UInt32Array::from(take_idx), None)?;
    let field = Arc::new(Field::new("item", child.data_type().clone(), true));
    let nulls = list.nulls().cloned();
    let out = ListArray::try_new(field, OffsetBuffer::new(new_offsets.into()), taken, nulls)?;
    Ok(Arc::new(out))
}

/// `flatten`: concatenate each row's inner lists of a `List<List<T>>` into one
/// `List<T>`, in order. Null inner lists are skipped; a null outer row stays null.
fn eval_flatten(list: &arrow::array::GenericListArray<i32>) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, AsArray, ListArray, UInt32Array};
    use arrow::buffer::OffsetBuffer;
    use arrow::compute::take;
    use arrow::datatypes::Field;

    let inner = list.values();
    let DataType::List(item_field) = inner.data_type() else {
        return Err(ExprError::ExpectedString {
            func: "list.flatten".into(),
            got: format!(
                "List<{}> (flatten needs a list of lists)",
                inner.data_type()
            ),
        });
    };
    let item_field = Arc::new(Field::new("item", item_field.data_type().clone(), true));
    let inner_list = inner.as_list::<i32>();
    let grandchild = inner_list.values();
    let outer_off = list.value_offsets();
    let inner_off = inner_list.value_offsets();

    let mut take_idx: Vec<u32> = Vec::new();
    let mut new_offsets: Vec<i32> = Vec::with_capacity(list.len() + 1);
    new_offsets.push(0);
    for i in 0..list.len() {
        if !list.is_null(i) {
            let (s, e) = (outer_off[i] as usize, outer_off[i + 1] as usize);
            for j in s..e {
                if inner_list.is_null(j) {
                    continue;
                }
                let (is_, ie) = (inner_off[j] as usize, inner_off[j + 1] as usize);
                take_idx.extend((is_..ie).map(|k| k as u32));
            }
        }
        new_offsets.push(take_idx.len() as i32);
    }
    let taken = take(grandchild.as_ref(), &UInt32Array::from(take_idx), None)?;
    let nulls = list.nulls().cloned();
    let out = ListArray::try_new(
        item_field,
        OffsetBuffer::new(new_offsets.into()),
        taken,
        nulls,
    )?;
    Ok(Arc::new(out))
}

/// `list.contains(value)`: true where any element equals the literal (null lists
/// and non-matching rows are false / null per element nullness).
pub(crate) fn eval_list_contains(arr: &ArrayRef, value: &Literal) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, BooleanBuilder};

    let list = require_list(arr, "list.contains")?;
    let offsets = list.value_offsets();
    // Compare element-wise against a one-element literal of the child type by
    // casting the child to the literal's natural array and scanning per row.
    let target = value.to_array(1);
    let child = cast(list.values(), target.data_type())?;
    let eqs = eval_binary(BinaryOp::Eq, &child, &value.to_array(child.len()))?;
    let eq = eqs.as_any().downcast_ref::<BooleanArray>().expect("bool");

    let mut b = BooleanBuilder::with_capacity(list.len());
    for i in 0..list.len() {
        if list.is_null(i) {
            b.append_null();
            continue;
        }
        let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
        let found = (s..e).any(|k| eq.is_valid(k) && eq.value(k));
        b.append_value(found);
    }
    Ok(Arc::new(b.finish()))
}

/// Extract field `name` from a `Struct` column, propagating struct-level nulls.
pub(crate) fn eval_struct_field(arr: &ArrayRef, name: &str) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, AsArray};

    if !matches!(arr.data_type(), DataType::Struct(_)) {
        return Err(ExprError::ExpectedString {
            func: "struct.field".into(),
            got: arr.data_type().to_string(),
        });
    }
    let s = arr.as_struct();
    let child = s
        .column_by_name(name)
        .cloned()
        .ok_or_else(|| ExprError::UnknownColumn(name.to_string()))?;
    // A null struct row makes the extracted field null too.
    if s.null_count() > 0 {
        let mask = is_null(s)?;
        Ok(arrow::compute::nullif(&child, &mask)?)
    } else {
        Ok(child)
    }
}

/// `list[index]`: gather the indexed element of each row's list, preserving the
/// element type and producing null where out of range / null. A non-negative
/// `index` counts from the front (0-based); a negative `index` counts from the
/// back (`-1` is the last element), matching Polars/Python indexing.
pub(crate) fn eval_list_get(arr: &ArrayRef, index: i64) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, AsArray, UInt32Array};
    use arrow::compute::take;

    if !matches!(arr.data_type(), DataType::List(_)) {
        return Err(ExprError::ExpectedString {
            func: "list.get".into(),
            got: arr.data_type().to_string(),
        });
    }
    let list = arr.as_list::<i32>();
    let offsets = list.value_offsets();
    let take_idx: UInt32Array = (0..list.len())
        .map(|i| {
            if list.is_null(i) {
                return None;
            }
            let (start, end) = (offsets[i] as i64, offsets[i + 1] as i64);
            // Negative indices address from the end (`-1` → last element).
            let pos = if index < 0 {
                end + index
            } else {
                start + index
            };
            (pos >= start && pos < end).then_some(pos as u32)
        })
        .collect();
    Ok(take(list.values().as_ref(), &take_idx, None)?)
}

/// Pairwise reduction over two numeric `List` columns (`dot`/`cosine_similarity`/
/// `l2_distance`) → Float64. Elements are paired up to the shorter row length; a
/// null on either side (row or element) drops that pair. A null list row → null.
pub(crate) fn eval_list_binary(
    func: ListBinaryFunc,
    left: &ArrayRef,
    right: &ArrayRef,
) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, AsArray, Float64Builder};
    use arrow::datatypes::Float64Type;

    for (name, arr) in [("left", left), ("right", right)] {
        if !matches!(arr.data_type(), DataType::List(_)) {
            return Err(ExprError::ExpectedString {
                func: format!("list.{func:?} ({name})"),
                got: arr.data_type().to_string(),
            });
        }
    }
    let (la, ra) = (left.as_list::<i32>(), right.as_list::<i32>());
    let lc = cast(la.values(), &DataType::Float64)?;
    let rc = cast(ra.values(), &DataType::Float64)?;
    let (lf, rf) = (
        lc.as_primitive::<Float64Type>(),
        rc.as_primitive::<Float64Type>(),
    );
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
        let (mut dot, mut lnorm, mut rnorm, mut dist2) = (0f64, 0f64, 0f64, 0f64);
        for k in 0..n {
            let (lk, rk) = (ls + k, rs + k);
            if !lf.is_valid(lk) || !rf.is_valid(rk) {
                continue;
            }
            let (x, y) = (lf.value(lk), rf.value(rk));
            dot += x * y;
            lnorm += x * x;
            rnorm += y * y;
            dist2 += (x - y) * (x - y);
        }
        match func {
            ListBinaryFunc::Dot => b.append_value(dot),
            ListBinaryFunc::L2Distance => b.append_value(dist2.sqrt()),
            ListBinaryFunc::CosineSimilarity => {
                let denom = lnorm.sqrt() * rnorm.sqrt();
                if denom == 0.0 {
                    b.append_null(); // a zero-magnitude vector has no direction
                } else {
                    b.append_value(dot / denom);
                }
            }
        }
    }
    Ok(Arc::new(b.finish()))
}

/// Per-row scalar reduction over a `List` column.
pub(crate) fn eval_list(func: ListFunc, arr: &ArrayRef) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, AsArray, Float64Builder, Int64Builder};

    if !matches!(arr.data_type(), DataType::List(_)) {
        return Err(ExprError::ExpectedString {
            func: format!("{func:?}"),
            got: arr.data_type().to_string(),
        });
    }
    let list = arr.as_list::<i32>();
    let offsets = list.value_offsets();

    // List-returning ops rebuild a List with the same element type.
    if let ListFunc::Reverse = func {
        return rebuild_list(list, |s, e| (s..e).rev().map(|k| k as u32).collect());
    }
    if let ListFunc::Sort = func {
        use arrow::compute::sort_to_indices;
        let child = list.values();
        return rebuild_list(list, |s, e| {
            let slice = child.slice(s, e - s);
            match sort_to_indices(&slice, None, None) {
                Ok(local) => local.values().iter().map(|&l| s as u32 + l).collect(),
                Err(_) => (s..e).map(|k| k as u32).collect(),
            }
        });
    }
    if let ListFunc::Unique = func {
        // Distinct elements in first-occurrence order, dropping nulls. Elements are
        // deduped by their Float64-cast bits (this engine's lists are numeric).
        let child = cast(list.values(), &DataType::Float64)?;
        let f = child.as_primitive::<arrow::datatypes::Float64Type>();
        return rebuild_list(list, |s, e| {
            let mut seen = std::collections::HashSet::new();
            (s..e)
                .filter(|&k| f.is_valid(k) && seen.insert(f.value(k).to_bits()))
                .map(|k| k as u32)
                .collect()
        });
    }

    // `flatten`: List<List<T>> → List<T>, concatenating each row's inner lists.
    if let ListFunc::Flatten = func {
        return eval_flatten(list);
    }

    if let ListFunc::Len = func {
        let mut b = Int64Builder::with_capacity(list.len());
        for i in 0..list.len() {
            if list.is_null(i) {
                b.append_null();
            } else {
                b.append_value((offsets[i + 1] - offsets[i]) as i64);
            }
        }
        return Ok(Arc::new(b.finish()));
    }

    // Numeric reductions: view the child elements as Float64.
    let child = cast(list.values(), &DataType::Float64)?;
    let f = child.as_primitive::<arrow::datatypes::Float64Type>();

    if let ListFunc::NUnique = func {
        let mut b = Int64Builder::with_capacity(list.len());
        for i in 0..list.len() {
            if list.is_null(i) {
                b.append_null();
                continue;
            }
            let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
            let mut seen = std::collections::HashSet::new();
            for k in s..e {
                if f.is_valid(k) {
                    seen.insert(f.value(k).to_bits());
                }
            }
            b.append_value(seen.len() as i64);
        }
        return Ok(Arc::new(b.finish()));
    }

    // arg_min/arg_max: 0-based index (within the row) of the min/max non-null
    // element, first occurrence on ties; empty/all-null/null row → null.
    if matches!(func, ListFunc::ArgMin | ListFunc::ArgMax) {
        let want_min = matches!(func, ListFunc::ArgMin);
        let mut b = Int64Builder::with_capacity(list.len());
        for i in 0..list.len() {
            if list.is_null(i) {
                b.append_null();
                continue;
            }
            let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
            let mut best: Option<(f64, i64)> = None;
            for (local, k) in (s..e).enumerate() {
                if !f.is_valid(k) {
                    continue;
                }
                let v = f.value(k);
                let better = match best {
                    None => true,
                    Some((bv, _)) if want_min => v < bv,
                    Some((bv, _)) => v > bv,
                };
                if better {
                    best = Some((v, local as i64));
                }
            }
            match best {
                Some((_, idx)) => b.append_value(idx),
                None => b.append_null(),
            }
        }
        return Ok(Arc::new(b.finish()));
    }

    let mut b = Float64Builder::with_capacity(list.len());
    for i in 0..list.len() {
        if list.is_null(i) {
            b.append_null();
            continue;
        }
        let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
        let vals: Vec<f64> = (s..e)
            .filter(|&k| f.is_valid(k))
            .map(|k| f.value(k))
            .collect();
        if vals.is_empty() {
            b.append_null();
            continue;
        }
        // Median sorts the row's values and takes the middle (average of the two
        // middle for an even count).
        if let ListFunc::Median = func {
            let mut sorted = vals.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            let m = if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            };
            b.append_value(m);
            continue;
        }
        // Sample variance / std need ≥2 values; null otherwise.
        if matches!(func, ListFunc::Std | ListFunc::Var) {
            if vals.len() < 2 {
                b.append_null();
                continue;
            }
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let ss: f64 = vals.iter().map(|&x| (x - mean) * (x - mean)).sum();
            let variance = ss / (vals.len() as f64 - 1.0);
            b.append_value(match func {
                ListFunc::Var => variance,
                ListFunc::Std => variance.sqrt(),
                _ => unreachable!(),
            });
            continue;
        }
        let r = match func {
            ListFunc::Sum => vals.iter().sum(),
            ListFunc::Min => vals.iter().copied().fold(f64::INFINITY, f64::min),
            ListFunc::Max => vals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            ListFunc::Mean => vals.iter().sum::<f64>() / vals.len() as f64,
            ListFunc::Product => vals.iter().product(),
            ListFunc::L2Norm => vals.iter().map(|&x| x * x).sum::<f64>().sqrt(),
            _ => unreachable!("len/n_unique/sort/reverse/unique/std/var/flatten handled above"),
        };
        b.append_value(r);
    }
    Ok(Arc::new(b.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Builder, ListBuilder};

    fn lists(rows: &[Option<Vec<f64>>]) -> ArrayRef {
        let mut b = ListBuilder::new(Float64Builder::new());
        for row in rows {
            match row {
                Some(vs) => {
                    for v in vs {
                        b.values().append_value(*v);
                    }
                    b.append(true);
                }
                None => b.append(false),
            }
        }
        Arc::new(b.finish())
    }

    fn f64s(a: &ArrayRef) -> Vec<Option<f64>> {
        use arrow::array::{Array, AsArray};
        let x = a.as_primitive::<arrow::datatypes::Float64Type>();
        (0..x.len())
            .map(|i| (!x.is_null(i)).then(|| x.value(i)))
            .collect()
    }

    #[test]
    fn dot_cosine_l2_distance() {
        let a = lists(&[
            Some(vec![1.0, 0.0]),
            Some(vec![1.0, 2.0]),
            Some(vec![0.0, 0.0]),
        ]);
        let b = lists(&[
            Some(vec![0.0, 1.0]),
            Some(vec![2.0, 4.0]),
            Some(vec![1.0, 1.0]),
        ]);
        let dot = eval_list_binary(ListBinaryFunc::Dot, &a, &b).unwrap();
        assert_eq!(f64s(&dot), vec![Some(0.0), Some(10.0), Some(0.0)]);
        let cos = eval_list_binary(ListBinaryFunc::CosineSimilarity, &a, &b).unwrap();
        let c = f64s(&cos);
        assert_eq!(c[0], Some(0.0)); // orthogonal
        assert!((c[1].unwrap() - 1.0).abs() < 1e-9); // parallel
        assert_eq!(c[2], None); // zero-norm vector -> null
        let dist = eval_list_binary(ListBinaryFunc::L2Distance, &a, &b).unwrap();
        let d = f64s(&dist);
        assert!((d[1].unwrap() - 5f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn binary_null_row_propagates() {
        let a = lists(&[None, Some(vec![1.0])]);
        let b = lists(&[Some(vec![1.0]), None]);
        let dot = eval_list_binary(ListBinaryFunc::Dot, &a, &b).unwrap();
        assert_eq!(f64s(&dot), vec![None, None]);
    }
}
