//! List/struct evaluation for `Expr::List`/`ListGet`/`ListContains`/`StructField`
//! (split out of `lib.rs`).

use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, RecordBatch};
use arrow::compute::{cast, is_null};
use arrow::datatypes::DataType;

use bc_arrow::float_total_cmp;

use crate::eval::binary::eval_binary;
use crate::eval::list_ops::{accumulate_pair, as_var_list};
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
    let lst = lst.as_list::<i32>();
    let elems = cast(lst.values(), &DataType::Utf8)?;
    let elems = elems.as_string::<i32>();
    let offsets = lst.value_offsets();
    let out: StringArray = (0..lst.len())
        .map(|row| {
            if lst.is_null(row) {
                return None;
            }
            let (s, e) = (offsets[row] as usize, offsets[row + 1] as usize);
            // An *empty* list joins to "" (DuckDB), but a non-empty list whose elements are
            // all null stays null — so distinguish "no elements" from "no non-null elements".
            if e == s {
                return Some(String::new());
            }
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

/// A copy of a list's child in which float elements are canonicalized, for use as a
/// *comparison key* only — the values themselves are still taken from the original.
///
/// Arrow's `RowConverter` (and `f64::to_bits`) encode `-0.0` and `0.0` differently, and
/// every NaN payload differently again, so a naive dedup/set-op over floats splits values
/// that `=`, `GROUP BY` and the join keys all consider equal. Non-float children need no
/// rewrite and are returned as-is.
///
/// The `Float32` case is widened to `Float64` first so a list of one element type compares
/// against a list of the other; canonicalization itself is
/// [`bc_arrow::canon_float_array`], the engine's single definition of float identity.
pub(crate) fn float_canonical_key(child: &ArrayRef) -> Result<ArrayRef, ExprError> {
    if !matches!(child.data_type(), DataType::Float64 | DataType::Float32) {
        return Ok(child.clone());
    }
    Ok(bc_arrow::canon_float_array(&cast(
        child,
        &DataType::Float64,
    )?))
}

/// The type a list's child and a comparison literal must both be promoted to so that
/// neither side is *narrowed*.
///
/// Casting the child down to the literal's type truncates: comparing a `List<Float64>`
/// against `Int(2)` cast `2.5` to `2`, so `list.contains([2.5], 2)` reported *true*
/// (DuckDB: false). Promote instead — a float on either side wins, text wins over both.
pub(crate) fn compare_type(child: &DataType, literal: &DataType) -> DataType {
    let is_text = |t: &DataType| matches!(t, DataType::Utf8 | DataType::LargeUtf8);
    let is_float =
        |t: &DataType| matches!(t, DataType::Float16 | DataType::Float32 | DataType::Float64);
    if child == literal {
        child.clone()
    } else if is_text(child) || is_text(literal) {
        DataType::Utf8
    } else if child.is_numeric() && literal.is_numeric() {
        if is_float(child) || is_float(literal) {
            DataType::Float64
        } else {
            DataType::Int64
        }
    } else {
        // Anything else (bool, temporal, decimal, nested): cast the *literal* to the
        // child's type — never the other way, which is what truncated.
        child.clone()
    }
}

/// Cast a list child and a comparison literal to a common type and evaluate `child = lit`
/// element-wise. Shared by `list.contains` / `list.position` / `map.get`.
pub(crate) fn eq_against_literal(
    child: &ArrayRef,
    value: &Literal,
) -> Result<BooleanArray, ExprError> {
    let ty = compare_type(child.data_type(), value.to_array(1).data_type());
    let lhs = cast(child, &ty)?;
    let rhs = cast(&value.to_array(child.len()), &ty)?;
    // Float membership must use the engine's key identity, not raw comparison: `-0.0`
    // and `0.0` are one value under `GROUP BY`/join keys and DuckDB `list_contains`, and
    // `list.unique`/the set-ops already fold them (`float_canonical_key`). Without this,
    // `[-0.0].contains(0.0)` was false and `[0.0].position(-0.0)` was null. Canonicalizing
    // both sides (folds `-0.0`, unifies NaN) makes membership agree with dedup and DuckDB.
    let (lhs, rhs) = if matches!(ty, DataType::Float64 | DataType::Float32) {
        (float_canonical_key(&lhs)?, float_canonical_key(&rhs)?)
    } else {
        (lhs, rhs)
    };
    let eqs = eval_binary(BinaryOp::Eq, &lhs, &rhs)?;
    eqs.as_any()
        .downcast_ref::<BooleanArray>()
        .cloned()
        .ok_or_else(|| ExprError::ExpectedBoolean {
            op: "list element comparison".into(),
            got: eqs.data_type().to_string(),
        })
}

/// Per-element comparison keys (`arrow::row::Rows`) over a list's flat child, used to
/// dedup `list.unique` / `list.n_unique` for *any* element type. Float children are
/// canonicalized first so `-0.0`/`0.0` and every NaN share one key, matching the
/// engine's `GROUP BY`/join-key identity. Nulls encode too, but callers skip them.
pub(crate) fn element_identity(child: &ArrayRef) -> Result<arrow::row::Rows, ExprError> {
    use arrow::row::{RowConverter, SortField};
    let key = float_canonical_key(child)?;
    let converter = RowConverter::new(vec![SortField::new(key.data_type().clone())])?;
    Ok(converter.convert_columns(std::slice::from_ref(&key))?)
}

/// A `List` view of `arr`, accepting a `FixedSizeList` by widening it, or a clear
/// type error.
///
/// Embeddings arrive as `FixedSizeList` — that is how Arrow and Parquet store a vector
/// column, and what DuckDB's `ARRAY` type maps to. The vector kernels already coerced
/// (see `list_ops::coerce::as_var_list`), but the *indexing* half of the namespace did
/// not, so the same embedding column could be normalized and summed yet not subscripted,
/// sliced, or searched. Coercing in one place is what makes `.list` mean the same thing
/// for both list encodings.
///
/// Returns an owned array because widening allocates offsets; for an already-variable
/// `List` it is a refcount bump, and next to an embedding payload the offsets are noise.
pub(crate) fn require_list(arr: &ArrayRef, func: &str) -> Result<ArrayRef, ExprError> {
    crate::eval::list_ops::coerce::as_var_list(arr, func)
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
    // Each kept index references a child element at most once, so the gather list never
    // exceeds the child length — pre-size to it and skip the extend loop's reallocations.
    let mut take_idx: Vec<u32> = Vec::with_capacity(child.len());
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

/// `list.contains(value)`: true where any element equals the literal (null lists
/// and non-matching rows are false / null per element nullness).
pub(crate) fn eval_list_contains(arr: &ArrayRef, value: &Literal) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, BooleanBuilder};

    let list = require_list(arr, "list.contains")?;
    use arrow::array::AsArray;
    let list = list.as_list::<i32>();
    let offsets = list.value_offsets();
    // Compare element-wise against the literal, both promoted to a common type so the
    // child is never narrowed onto the literal (that truncated `[2.5].contains(2)`).
    let eq = eq_against_literal(list.values(), value)?;

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

/// `list.position(value)` — the 1-based index of the first element equal to the
/// literal `value`; null if absent or the list row is null (DuckDB `list_position`).
/// → Int64.
pub(crate) fn eval_list_position(arr: &ArrayRef, value: &Literal) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, Int64Builder};

    let list = require_list(arr, "list.position")?;
    use arrow::array::AsArray;
    let list = list.as_list::<i32>();
    let offsets = list.value_offsets();
    let eq = eq_against_literal(list.values(), value)?;

    let mut b = Int64Builder::with_capacity(list.len());
    for i in 0..list.len() {
        if list.is_null(i) {
            b.append_null();
            continue;
        }
        let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
        match (s..e).position(|k| eq.is_valid(k) && eq.value(k)) {
            Some(p) => b.append_value((p + 1) as i64),
            None => b.append_null(), // not found → null (DuckDB), not 0 (Spark)
        }
    }
    Ok(Arc::new(b.finish()))
}

/// Extract field `name` from a `Struct` column, propagating struct-level nulls.
/// Build a `Struct` column from named sub-expressions (one field per `NamedExpr`),
/// each field carrying that expression's per-row value. The read-side counterpart
/// is `eval_struct_field`.
pub(crate) fn eval_make_struct(
    fields: &[crate::NamedExpr],
    batch: &RecordBatch,
) -> Result<ArrayRef, ExprError> {
    use arrow::array::StructArray;
    use arrow::datatypes::{Field, Fields};

    let mut arrow_fields: Vec<Arc<Field>> = Vec::with_capacity(fields.len());
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(fields.len());
    for f in fields {
        let arr = f.value.eval(batch)?;
        arrow_fields.push(Arc::new(Field::new(&f.name, arr.data_type().clone(), true)));
        columns.push(arr);
    }
    Ok(Arc::new(StructArray::new(
        Fields::from(arrow_fields),
        columns,
        None,
    )))
}

pub(crate) fn eval_struct_field(arr: &ArrayRef, name: &str) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, AsArray};

    if !matches!(arr.data_type(), DataType::Struct(_)) {
        return Err(ExprError::ExpectedType {
            func: "struct.field".into(),
            want: "a Struct argument",
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

    let arr = require_list(arr, "list.get")?;
    let list = arr.as_list::<i32>();
    let offsets = list.value_offsets();
    let take_idx: UInt32Array = (0..list.len())
        .map(|i| {
            if list.is_null(i) {
                return None;
            }
            let (start, end) = (offsets[i] as i64, offsets[i + 1] as i64);
            // Negative indices address from the end (`-1` → last element). Saturating so a
            // huge/`i64::MIN` index can't overflow — it just lands out of range → null.
            let pos = if index < 0 {
                end.saturating_add(index)
            } else {
                start.saturating_add(index)
            };
            (pos >= start && pos < end).then_some(pos as u32)
        })
        .collect();
    Ok(take(list.values().as_ref(), &take_idx, None)?)
}

/// Pairwise reduction over two numeric `List` columns (`dot`/`cosine_similarity`/
/// `l2_distance`) → Float64. The vector-distance ops require the two lists in a row to
/// have equal length — a dimension mismatch is an error (matching DuckDB
/// `list_cosine_similarity`/`list_distance`/`list_dot_product`, which raise "list
/// dimensions must be equal"); pairing only up to `min(len_a, len_b)` silently truncated
/// a mismatched embedding to a bogus distance (`cosine_similarity([1,2],[1,2,3]) ≈ 1.0`)
/// that corrupts KNN. `jaccard` (a minhash agreement rate, not a DuckDB function) keeps
/// its documented "agreement over the shared prefix" behavior. A null on either side (row
/// or element) drops that pair; a null list row → null.
pub(crate) fn eval_list_binary(
    func: ListBinaryFunc,
    left: &ArrayRef,
    right: &ArrayRef,
) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, AsArray, Float64Builder};
    use arrow::datatypes::{Float32Type, Float64Type};

    // `FixedSizeList` is normalized to `List` rather than rejected: it is the physical
    // type behind `arrow.fixed_shape_tensor`, so every embedding column this project
    // produces arrives in that shape.
    let left = as_var_list(left, &format!("list.{func:?} (left)"))?;
    let right = as_var_list(right, &format!("list.{func:?} (right)"))?;
    let (la, ra) = (left.as_list::<i32>(), right.as_list::<i32>());

    // Embeddings are overwhelmingly `f32`; keeping them at native width halves the bytes
    // the inner loop streams. Accumulation is `f64` either way, so the two paths agree
    // bit-for-bit (`accumulate_pair`'s tests pin this).
    let both_f32 = matches!(la.values().data_type(), DataType::Float32)
        && matches!(ra.values().data_type(), DataType::Float32);
    let (lc, rc) = if both_f32 {
        (Arc::clone(la.values()), Arc::clone(ra.values()))
    } else {
        (
            cast(la.values(), &DataType::Float64)?,
            cast(ra.values(), &DataType::Float64)?,
        )
    };
    let (lo, ro) = (la.value_offsets(), ra.value_offsets());

    // The vector-distance ops are only defined on equal-length vectors; a mismatch is an
    // error, not a silent truncation to the shorter length (which returned a bogus ~1.0).
    let dims_must_match = matches!(
        func,
        ListBinaryFunc::Dot
            | ListBinaryFunc::L2Distance
            | ListBinaryFunc::L1Distance
            | ListBinaryFunc::Hamming
            | ListBinaryFunc::CosineSimilarity
    );

    let mut b = Float64Builder::with_capacity(la.len());
    for i in 0..la.len() {
        if la.is_null(i) || ra.is_null(i) {
            b.append_null();
            continue;
        }
        let (ls, le) = (lo[i] as usize, lo[i + 1] as usize);
        let (rs, re) = (ro[i] as usize, ro[i + 1] as usize);
        let (llen, rlen) = (le - ls, re - rs);
        if dims_must_match && llen != rlen {
            return Err(ExprError::InvalidArgument {
                func: format!("list.{func:?}"),
                reason: format!(
                    "list dimensions must be equal, got left length {llen} and right length {rlen}"
                ),
            });
        }
        let n = llen.min(rlen);
        let s = if both_f32 {
            accumulate_pair::<Float32Type>(
                lc.as_primitive::<Float32Type>(),
                rc.as_primitive::<Float32Type>(),
                ls,
                rs,
                n,
            )
        } else {
            accumulate_pair::<Float64Type>(
                lc.as_primitive::<Float64Type>(),
                rc.as_primitive::<Float64Type>(),
                ls,
                rs,
                n,
            )
        };
        match func {
            ListBinaryFunc::Dot => b.append_value(s.dot),
            ListBinaryFunc::L2Distance => b.append_value(s.dist2.sqrt()),
            ListBinaryFunc::L1Distance => b.append_value(s.dist1),
            ListBinaryFunc::Hamming => b.append_value(s.disagree as f64),
            ListBinaryFunc::Jaccard => {
                if n == 0 {
                    b.append_null(); // no positions to agree on
                } else {
                    b.append_value(s.agree as f64 / n as f64)
                }
            }
            ListBinaryFunc::CosineSimilarity => {
                let denom = s.lnorm.sqrt() * s.rnorm.sqrt();
                if denom == 0.0 {
                    b.append_null(); // a zero-magnitude vector has no direction
                } else {
                    b.append_value(s.dot / denom);
                }
            }
        }
    }
    Ok(Arc::new(b.finish()))
}

/// Per-row scalar reduction over a `List` column.
pub(crate) fn eval_list(func: ListFunc, arr: &ArrayRef) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, AsArray, Float64Array, Float64Builder, Int64Array};

    // Accept `FixedSizeList` here too, so `.list.l2_norm()`/`.normalize()`/`.mean()` work
    // on a tensor column rather than only on a variable-length list.
    let arr = as_var_list(arr, &format!("{func:?}"))?;
    let list = arr.as_list::<i32>();
    let offsets = list.value_offsets();

    // List-returning ops rebuild a List with the same element type.
    if let ListFunc::Reverse = func {
        return rebuild_list(list, |s, e| (s..e).rev().map(|k| k as u32).collect());
    }
    if let ListFunc::Sort = func {
        use arrow::compute::{sort_to_indices, SortOptions};
        let child = list.values();
        // DuckDB `list_sort` is ascending, NULLS LAST. arrow-rs `sort_to_indices` defaults
        // to NULLS FIRST, so pass explicit options — otherwise nulls sorted to the front and
        // disagreed with DuckDB. NaN must sort as the greatest value (before the trailing
        // nulls): arrow's raw-bit order does that for a *positive* NaN but ranks a *negative*
        // NaN below -inf, so sort the **canonical** key (every NaN → one quiet NaN, `-0.0` →
        // `0.0`) to match the engine's float identity, then gather the *original* elements.
        let child_key = bc_arrow::canon_float_array(child);
        let opts = SortOptions {
            descending: false,
            nulls_first: false,
        };
        return rebuild_list(list, |s, e| {
            let slice = child_key.slice(s, e - s);
            match sort_to_indices(&slice, Some(opts), None) {
                Ok(local) => local.values().iter().map(|&l| s as u32 + l).collect(),
                Err(_) => (s..e).map(|k| k as u32).collect(),
            }
        });
    }
    if let ListFunc::Unique = func {
        // Distinct elements in first-occurrence order, dropping nulls. Element identity is
        // type-general (any element type, not only numeric — casting a string list to
        // Float64 nulled every element and returned an empty list) and float-canonical, so
        // `-0.0`/`0.0` and every NaN collapse the way `GROUP BY` and the join keys do.
        let keys = element_identity(list.values())?;
        return rebuild_list(list, |s, e| {
            let mut seen = std::collections::HashSet::new();
            (s..e)
                .filter(|&k| !list.values().is_null(k) && seen.insert(keys.row(k).owned()))
                .map(|k| k as u32)
                .collect()
        });
    }

    // `flatten`: List<List<T>> → List<T>, concatenating each row's inner lists.
    if let ListFunc::Flatten = func {
        return crate::eval::list_ops::eval_flatten(list);
    }

    // Per-row list-returning numeric transforms — extracted to `list_ops::list_reduce`
    // to keep this file within its size budget.
    match func {
        ListFunc::Normalize => return crate::eval::list_ops::list_reduce::normalize(list),
        ListFunc::Softmax => return crate::eval::list_ops::list_reduce::softmax(list),
        ListFunc::ArgSort => return crate::eval::list_ops::list_reduce::arg_sort(list),
        ListFunc::CumSum => return crate::eval::list_ops::list_reduce::cum_sum(list),
        ListFunc::Diff => return crate::eval::list_ops::list_reduce::diff(list),
        _ => {}
    }

    if let ListFunc::Len = func {
        let n = (0..list.len())
            .map(|i| (!list.is_null(i)).then(|| (offsets[i + 1] - offsets[i]) as i64));
        return Ok(Arc::new(n.collect::<Int64Array>()));
    }

    if let ListFunc::NUnique = func {
        // Count distinct non-null elements, type-general and float-canonical (see `Unique`).
        let keys = element_identity(list.values())?;
        let child = list.values();
        let n = (0..list.len()).map(|i| {
            let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
            (!list.is_null(i)).then(|| {
                (s..e)
                    .filter(|&k| !child.is_null(k))
                    .map(|k| keys.row(k).owned())
                    .collect::<std::collections::HashSet<_>>()
                    .len() as i64
            })
        });
        return Ok(Arc::new(n.collect::<Int64Array>()));
    }

    // `min`/`max` over any non-float child (integers, decimals, strings, bools, dates,
    // …). DuckDB `list_min`/`list_max` are defined on every comparable type and return
    // the *exact* element. Casting the child to Float64 both nulled non-numeric elements
    // (`list.min(['apple'])` → null) and lost integer precision above 2^53
    // (`list.min([2^53+1, 2^53+2])` → 2^53, a value not even in the list). Only floats stay
    // on the numeric path below, whose NaN / total-order semantics are well-tested; every
    // other type gathers the min/max non-null element here, preserving its own type.
    let child_is_float = matches!(
        list.values().data_type(),
        DataType::Float16 | DataType::Float32 | DataType::Float64
    );
    if matches!(func, ListFunc::Min | ListFunc::Max) && !child_is_float {
        use arrow::array::UInt32Array;
        use arrow::compute::{sort_to_indices, take, SortOptions};
        let child = list.values();
        let want_min = matches!(func, ListFunc::Min);
        // Ascending with nulls last, so non-null values occupy the front in value order:
        // min is the first non-null, max the last non-null. Null elements are ignored.
        let opts = SortOptions {
            descending: false,
            nulls_first: false,
        };
        let take_idx: UInt32Array = (0..list.len())
            .map(|i| {
                if list.is_null(i) {
                    return None;
                }
                let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
                if e == s {
                    return None;
                }
                let slice = child.slice(s, e - s);
                let ord = sort_to_indices(&slice, Some(opts), None).ok()?;
                let mut valid = ord
                    .values()
                    .iter()
                    .map(|&l| s as u32 + l)
                    .filter(|&g| child.is_valid(g as usize));
                if want_min {
                    valid.next()
                } else {
                    valid.next_back()
                }
            })
            .collect();
        return Ok(take(child.as_ref(), &take_idx, None)?);
    }

    // Exact integer `sum`/`avg`. The Float64 view below rounds every element to 53
    // significant bits, so `list_sum([2^53+1, 2])` returned 2^53+2 where DuckDB returns the
    // true 2^53+3 — the same precision loss already fixed for `min`/`max` above. Integer
    // children accumulate in `i64` instead; `sum` returns Int64 (what the scalar `sum`
    // aggregate returns for an Int64 column) and `avg` divides that exact total as f64,
    // staying Float64. Overflow is an error, not a wrap, matching `bc-runtime`'s `sum_acc`
    // (`checked_add` → `SumOverflow`); the wrapping convention in `eval/binary.rs` exists
    // for JIT parity on scalar arithmetic and deliberately does not extend to reductions.
    // A null row, an empty row, and an all-null row all have no values, hence null.
    let int_child = matches!(
        list.values().data_type(),
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
    );
    if matches!(func, ListFunc::Sum | ListFunc::Mean) && int_child {
        let child = cast(list.values(), &DataType::Int64)?;
        let v = child.as_primitive::<arrow::datatypes::Int64Type>();
        let rows = (0..list.len())
            .map(|i| {
                let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
                let row = v.slice(s, e - s);
                let n = row.len() - row.null_count();
                let t = arrow::compute::sum_checked(&row)?;
                Ok((!list.is_null(i)).then_some(t).flatten().map(|t| (t, n)))
            })
            .collect::<Result<Vec<_>, ExprError>>()?;
        return Ok(if matches!(func, ListFunc::Sum) {
            Arc::new(
                rows.iter()
                    .map(|r| r.map(|(t, _)| t))
                    .collect::<Int64Array>(),
            )
        } else {
            let m = rows.iter().map(|r| r.map(|(t, n)| t as f64 / n as f64));
            Arc::new(m.collect::<Float64Array>())
        });
    }

    // Numeric reductions: view the child elements as Float64.
    let child = cast(list.values(), &DataType::Float64)?;
    let f = child.as_primitive::<arrow::datatypes::Float64Type>();

    // arg_min/arg_max: 0-based index (within the row) of the min/max non-null
    // element, first occurrence on ties; empty/all-null/null row → null.
    if matches!(func, ListFunc::ArgMin | ListFunc::ArgMax) {
        let want_min = matches!(func, ListFunc::ArgMin);
        let idx = (0..list.len()).map(|i| {
            let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
            let mut best: Option<(f64, i64)> = None;
            for (local, k) in (s..e).enumerate() {
                if list.is_null(i) || !f.is_valid(k) {
                    continue;
                }
                let v = f.value(k);
                // Order by the engine's total float order (NaN greatest), so `arg_max`
                // points at the same element `max` returns. Naive `v < bv`/`v > bv`
                // silently skipped NaN, so `arg_max([1.0, NaN, 2.0])` gave 2 (the `2.0`)
                // while `max` returned NaN — the two disagreed.
                let better = match best {
                    None => true,
                    Some((bv, _)) if want_min => float_total_cmp(v, bv).is_lt(),
                    Some((bv, _)) => float_total_cmp(v, bv).is_gt(),
                };
                if better {
                    best = Some((v, local as i64));
                }
            }
            best.map(|(_, idx)| idx)
        });
        return Ok(Arc::new(idx.collect::<Int64Array>()));
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
            // NaN sorts greatest (the engine's total float order), so the median matches
            // DuckDB; a bare `partial_cmp` leaves NaN unordered and misplaces the middle.
            sorted.sort_by(|a, b| float_total_cmp(*a, *b));
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
            // `float_total_cmp` treats every NaN (either sign) as the greatest value, the
            // order the engine's aggregate/`ORDER BY`/`greatest` paths and DuckDB all use.
            // `f64::min`/`f64::max` silently *drop* NaN (so `list.max([1.0, NaN])` wrongly
            // gave 1.0), and `f64::total_cmp` ranks `-NaN` as the *least* value (so
            // `list.max([1.0, -NaN, 2.0])` gave 2.0 where DuckDB gives NaN).
            ListFunc::Min => vals
                .iter()
                .copied()
                .reduce(|a, b| if float_total_cmp(b, a).is_lt() { b } else { a })
                .expect("non-empty"),
            ListFunc::Max => vals
                .iter()
                .copied()
                .reduce(|a, b| if float_total_cmp(b, a).is_gt() { b } else { a })
                .expect("non-empty"),
            ListFunc::Mean => vals.iter().sum::<f64>() / vals.len() as f64,
            ListFunc::Product => vals.iter().product(),
            ListFunc::L2Norm => vals.iter().map(|&x| x * x).sum::<f64>().sqrt(),
            ListFunc::L1Norm => vals.iter().map(|&x| x.abs()).sum(),
            // Max of the absolute values (the MaxAbs-scaling divisor). `vals` is non-empty
            // here (empty rows are nulled above), so folding from 0.0 is safe.
            ListFunc::MaxAbs => vals.iter().map(|&x| x.abs()).fold(0.0f64, f64::max),
            _ => unreachable!("len/n_unique/sort/reverse/unique/std/var/flatten handled above"),
        };
        b.append_value(r);
    }
    Ok(Arc::new(b.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Float64Builder, Int64Array, ListBuilder};

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
    fn vector_distance_ops_error_on_dimension_mismatch() {
        // DuckDB errors when the two lists differ in length; the old code paired only the
        // shared prefix, so `cosine_similarity([1,2],[1,2,3])` silently returned ~1.0.
        let a = lists(&[Some(vec![1.0, 2.0])]);
        let b = lists(&[Some(vec![1.0, 2.0, 3.0])]);
        for func in [
            ListBinaryFunc::Dot,
            ListBinaryFunc::L2Distance,
            ListBinaryFunc::CosineSimilarity,
        ] {
            let err = eval_list_binary(func, &a, &b).unwrap_err();
            assert!(
                matches!(err, ExprError::InvalidArgument { .. }),
                "expected a dimension-mismatch error for {func:?}, got {err:?}"
            );
        }
        // A null list row is still null (no length to compare), not an error.
        let an = lists(&[None]);
        let bn = lists(&[Some(vec![1.0, 2.0, 3.0])]);
        let out = eval_list_binary(ListBinaryFunc::Dot, &an, &bn).unwrap();
        assert_eq!(f64s(&out), vec![None]);
    }

    #[test]
    fn binary_null_row_propagates() {
        let a = lists(&[None, Some(vec![1.0])]);
        let b = lists(&[Some(vec![1.0]), None]);
        let dot = eval_list_binary(ListBinaryFunc::Dot, &a, &b).unwrap();
        assert_eq!(f64s(&dot), vec![None, None]);
    }

    #[test]
    fn normalize_to_unit_length() {
        use arrow::array::{Array, AsArray};
        let a = lists(&[
            Some(vec![3.0, 4.0]),      // norm 5 -> [0.6, 0.8]
            Some(vec![0.0, 0.0]),      // zero vector -> zeros
            None,                      // null row stays null
            Some(vec![1.0, 1.0, 1.0]), // unit -> each 1/sqrt(3)
        ]);
        let out = eval_list(ListFunc::Normalize, &a).unwrap();
        let list = out.as_list::<i32>();
        // Row 0: [0.6, 0.8], L2 norm == 1.
        let r0 = f64s(&list.value(0));
        assert!((r0[0].unwrap() - 0.6).abs() < 1e-9 && (r0[1].unwrap() - 0.8).abs() < 1e-9);
        // Row 1: zero vector → zeros (no NaN from div-by-zero).
        assert_eq!(f64s(&list.value(1)), vec![Some(0.0), Some(0.0)]);
        // Row 2: null preserved.
        assert!(list.is_null(2));
        // Row 3: unit length.
        let r3 = f64s(&list.value(3));
        let norm: f64 = r3.iter().map(|v| v.unwrap().powi(2)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn softmax_produces_a_probability_distribution() {
        use arrow::array::{Array, AsArray};
        let a = lists(&[
            Some(vec![0.0, 0.0]),      // equal logits → [0.5, 0.5]
            Some(vec![1.0, 2.0, 3.0]), // increasing → increasing probs, sum 1
            None,                      // null row stays null
        ]);
        let out = eval_list(ListFunc::Softmax, &a).unwrap();
        let list = out.as_list::<i32>();
        let r0 = f64s(&list.value(0));
        assert!((r0[0].unwrap() - 0.5).abs() < 1e-12 && (r0[1].unwrap() - 0.5).abs() < 1e-12);
        let r1 = f64s(&list.value(1));
        let sum: f64 = r1.iter().map(|v| v.unwrap()).sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!(r1[0].unwrap() < r1[1].unwrap() && r1[1].unwrap() < r1[2].unwrap());
        // Exact stable-softmax value for the last element of [1,2,3].
        let denom = (1.0f64 - 3.0).exp() + (2.0f64 - 3.0).exp() + 1.0;
        assert!((r1[2].unwrap() - 1.0 / denom).abs() < 1e-12);
        assert!(list.is_null(2));
    }

    #[test]
    fn l1_norm_sums_absolute_values() {
        use arrow::array::{Array, Float64Array};
        let a = lists(&[Some(vec![3.0, -4.0]), Some(vec![]), None]);
        let out = eval_list(ListFunc::L1Norm, &a).unwrap();
        let f = out.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(f.value(0), 7.0); // |3| + |-4|
        assert!(f.is_null(1)); // empty row → null (no elements to sum)
        assert!(f.is_null(2)); // null row → null
    }

    #[test]
    fn max_abs_is_the_largest_magnitude() {
        use arrow::array::{Array, Float64Array};
        let a = lists(&[Some(vec![1.0, -5.0, 3.0]), Some(vec![]), None]);
        let out = eval_list(ListFunc::MaxAbs, &a).unwrap();
        let f = out.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(f.value(0), 5.0); // max(|1|, |-5|, |3|)
        assert!(f.is_null(1) && f.is_null(2));
    }

    #[test]
    fn cum_sum_accumulates_and_preserves_nulls() {
        use arrow::array::{Array, AsArray};
        let a = lists(&[
            Some(vec![1.0, 2.0, 3.0]), // → [1, 3, 6]
            None,                      // null row stays null
            Some(vec![]),              // empty → empty
        ]);
        let out = eval_list(ListFunc::CumSum, &a).unwrap();
        let list = out.as_list::<i32>();
        assert_eq!(f64s(&list.value(0)), vec![Some(1.0), Some(3.0), Some(6.0)]);
        assert!(list.is_null(1));
        assert_eq!(f64s(&list.value(2)), Vec::<Option<f64>>::new());
    }

    #[test]
    fn diff_is_first_difference_with_leading_null() {
        use arrow::array::{Array, AsArray};
        let a = lists(&[
            Some(vec![1.0, 2.0, 4.0, 7.0]), // → [null, 1, 2, 3]
            None,                           // null row stays null
            Some(vec![5.0]),                // single element → [null]
            Some(vec![]),                   // empty → empty
        ]);
        let out = eval_list(ListFunc::Diff, &a).unwrap();
        let list = out.as_list::<i32>();
        assert_eq!(
            f64s(&list.value(0)),
            vec![None, Some(1.0), Some(2.0), Some(3.0)]
        );
        assert!(list.is_null(1));
        assert_eq!(f64s(&list.value(2)), vec![None]);
        assert_eq!(f64s(&list.value(3)), Vec::<Option<f64>>::new());
    }

    #[test]
    fn arg_sort_returns_ascending_indices() {
        use arrow::array::{Array, AsArray, Int64Array};
        let a = lists(&[
            Some(vec![0.3, 0.9, 0.1]), // ascending order of values → indices [2, 0, 1]
            Some(vec![5.0, 5.0, 1.0]), // ties keep original order → [2, 0, 1]
            None,                      // null row stays null
            Some(vec![]),              // empty → empty
        ]);
        let out = eval_list(ListFunc::ArgSort, &a).unwrap();
        let list = out.as_list::<i32>();
        let idx = |i: usize| {
            list.value(i)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        };
        assert_eq!(idx(0), vec![2, 0, 1]);
        assert_eq!(idx(1), vec![2, 0, 1]); // stable on ties
        assert!(list.is_null(2));
        assert_eq!(idx(3), Vec::<i64>::new());
    }

    /// Build a `List<Utf8>` for the type-general dedup tests.
    fn str_lists(rows: &[Option<Vec<&str>>]) -> ArrayRef {
        use arrow::array::{ListBuilder, StringBuilder};
        let mut b = ListBuilder::new(StringBuilder::new());
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

    fn strs(a: &ArrayRef) -> Vec<Option<String>> {
        use arrow::array::{Array, AsArray};
        let x = a.as_string::<i32>();
        (0..x.len())
            .map(|i| (!x.is_null(i)).then(|| x.value(i).to_string()))
            .collect()
    }

    #[test]
    fn contains_and_position_promote_rather_than_narrow_the_child() {
        // `[2.5].contains(2)` must be false: casting the child to the Int literal truncated
        // 2.5→2 and wrongly reported true.
        let a = lists(&[Some(vec![2.5])]);
        let c = eval_list_contains(&a, &Literal::Int(2)).unwrap();
        let cb = c.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(!cb.value(0));
        let p = eval_list_position(&a, &Literal::Int(2)).unwrap();
        assert!(p.as_any().downcast_ref::<Int64Array>().unwrap().is_null(0));
        // The genuinely-present float still matches.
        let c2 = eval_list_contains(&a, &Literal::Float(2.5)).unwrap();
        assert!(c2.as_any().downcast_ref::<BooleanArray>().unwrap().value(0));
    }

    #[test]
    fn contains_and_position_fold_negative_zero_and_nan() {
        // `-0.0` and `0.0` are one value (DuckDB `list_contains`, GROUP BY, join keys).
        // Raw comparison ranked `-0.0 < 0.0`, so `[-0.0].contains(0.0)` was false.
        let neg = lists(&[Some(vec![-0.0])]);
        let c = eval_list_contains(&neg, &Literal::Float(0.0)).unwrap();
        assert!(c.as_any().downcast_ref::<BooleanArray>().unwrap().value(0));
        let pos = lists(&[Some(vec![0.0])]);
        let p = eval_list_position(&pos, &Literal::Float(-0.0)).unwrap();
        assert_eq!(p.as_any().downcast_ref::<Int64Array>().unwrap().value(0), 1);
        // NaN matches NaN (DuckDB `list_contains([NaN], 'nan')` is true).
        let nanl = lists(&[Some(vec![1.0, f64::NAN])]);
        let cn = eval_list_contains(&nanl, &Literal::Float(f64::NAN)).unwrap();
        assert!(cn.as_any().downcast_ref::<BooleanArray>().unwrap().value(0));
    }

    #[test]
    fn sort_places_nulls_last_and_nan_greatest_like_duckdb() {
        use arrow::array::AsArray;
        let a = lists(&[Some(vec![3.0, f64::NAN, 1.0])]);
        // append a null element via a manual builder
        let mut b = ListBuilder::new(Float64Builder::new());
        b.values().append_value(3.0);
        b.values().append_value(f64::NAN);
        b.values().append_value(1.0);
        b.values().append_null();
        b.append(true);
        let with_null: ArrayRef = Arc::new(b.finish());
        let out = eval_list(ListFunc::Sort, &with_null).unwrap();
        let row = out.as_list::<i32>().value(0);
        let v = f64s(&row);
        // [1.0, 3.0, NaN, null] — numbers ascending, NaN then the trailing null.
        assert_eq!(v[0], Some(1.0));
        assert_eq!(v[1], Some(3.0));
        assert!(v[2].unwrap().is_nan());
        assert_eq!(v[3], None);
        let _ = a;
    }

    #[test]
    fn min_max_arg_treat_negative_nan_as_greatest() {
        // `-NaN` must rank as the greatest value (DuckDB), not the least (`f64::total_cmp`).
        let a = lists(&[Some(vec![1.0, -f64::NAN, 2.0])]);
        assert!(f64s(&eval_list(ListFunc::Max, &a).unwrap())[0]
            .unwrap()
            .is_nan());
        assert_eq!(f64s(&eval_list(ListFunc::Min, &a).unwrap())[0], Some(1.0));
        // arg_max points at the NaN (index 1), consistent with max.
        let am = eval_list(ListFunc::ArgMax, &a).unwrap();
        assert_eq!(
            am.as_any().downcast_ref::<Int64Array>().unwrap().value(0),
            1
        );
    }

    #[test]
    fn median_orders_nan_as_greatest() {
        let a = lists(&[Some(vec![1.0, f64::NAN, 2.0])]);
        // sorted [1, 2, NaN] → middle is 2.0 (DuckDB list_median).
        assert_eq!(
            f64s(&eval_list(ListFunc::Median, &a).unwrap())[0],
            Some(2.0)
        );
    }

    #[test]
    fn unique_and_n_unique_are_type_general_and_float_canonical() {
        use arrow::array::AsArray;
        // Strings used to be cast to Float64 → all null → empty; now they dedup properly.
        let s = str_lists(&[Some(vec!["a", "b", "a"])]);
        let u = eval_list(ListFunc::Unique, &s).unwrap();
        assert_eq!(
            strs(&u.as_list::<i32>().value(0)),
            vec![Some("a".into()), Some("b".into())]
        );
        let nu = eval_list(ListFunc::NUnique, &s).unwrap();
        assert_eq!(
            nu.as_any().downcast_ref::<Int64Array>().unwrap().value(0),
            2
        );

        // -0.0 and 0.0 are one value (DuckDB folds them, as does GROUP BY).
        let z = lists(&[Some(vec![0.0, -0.0])]);
        let uz = eval_list(ListFunc::Unique, &z).unwrap();
        assert_eq!(uz.as_list::<i32>().value(0).len(), 1);
        let nuz = eval_list(ListFunc::NUnique, &z).unwrap();
        assert_eq!(
            nuz.as_any().downcast_ref::<Int64Array>().unwrap().value(0),
            1
        );
    }

    #[test]
    fn min_max_are_type_general_over_strings() {
        use arrow::array::AsArray;
        // DuckDB `list_min`/`list_max` work on any comparable type. Casting the child to
        // Float64 nulled every string element, so min/max wrongly returned null.
        let s = str_lists(&[
            Some(vec!["banana", "apple", "cherry"]),
            Some(vec![]),
            None,
            Some(vec!["z"]),
        ]);
        let mn = eval_list(ListFunc::Min, &s).unwrap();
        assert_eq!(
            strs(&mn),
            vec![Some("apple".into()), None, None, Some("z".into())]
        );
        let mx = eval_list(ListFunc::Max, &s).unwrap();
        assert_eq!(
            strs(&mx),
            vec![Some("cherry".into()), None, None, Some("z".into())]
        );
        // Null elements are ignored, not treated as the min/max.
        let mut b = ListBuilder::new(arrow::array::StringBuilder::new());
        b.values().append_null();
        b.values().append_value("m");
        b.values().append_null();
        b.append(true);
        let with_null: ArrayRef = Arc::new(b.finish());
        let mn2 = eval_list(ListFunc::Min, &with_null).unwrap();
        assert_eq!(mn2.as_string::<i32>().value(0), "m");
        assert_eq!(
            eval_list(ListFunc::Max, &with_null)
                .unwrap()
                .as_string::<i32>()
                .value(0),
            "m"
        );
    }

    #[test]
    fn min_max_over_integers_are_exact_above_2_53() {
        use arrow::array::{Int64Array, Int64Builder, ListBuilder};
        // Routing i64 through f64 rounded `2^53+1` to `2^53`, so `min` returned a value not
        // even in the list. The exact element type must be preserved.
        let mut b = ListBuilder::new(Int64Builder::new());
        for v in [(1i64 << 53) + 1, (1i64 << 53) + 2, i64::MAX] {
            b.values().append_value(v);
        }
        b.append(true);
        let arr: ArrayRef = Arc::new(b.finish());
        let mn = eval_list(ListFunc::Min, &arr).unwrap();
        let mn = mn.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(mn.value(0), (1i64 << 53) + 1);
        let mx = eval_list(ListFunc::Max, &arr).unwrap();
        let mx = mx.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(mx.value(0), i64::MAX);
    }

    #[test]
    fn join_empty_list_is_empty_string_not_null() {
        use arrow::array::{Array, AsArray};
        // [], ["a", null], null  →  "", "a", null  (DuckDB array_to_string).
        let mut b = ListBuilder::new(arrow::array::StringBuilder::new());
        b.append(true); // empty list
        b.values().append_value("a");
        b.values().append_null();
        b.append(true);
        b.append(false); // null list
        let arr: ArrayRef = Arc::new(b.finish());
        let out = eval_list_join(&arr, "-").unwrap();
        let s = out.as_string::<i32>();
        assert_eq!(s.value(0), ""); // empty list → ""
        assert_eq!(s.value(1), "a");
        assert!(s.is_null(2));
    }

    #[test]
    fn list_get_saturates_on_extreme_index() {
        use arrow::array::Array;
        // i64::MIN as a negative index must not overflow `end + index`; it lands out of
        // range and yields null rather than panicking.
        let a = lists(&[Some(vec![10.0, 20.0])]);
        let out = eval_list_get(&a, i64::MIN).unwrap();
        assert!(out
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap()
            .is_null(0));
        let out2 = eval_list_get(&a, i64::MAX).unwrap();
        assert!(out2
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap()
            .is_null(0));
    }

    /// Build a `List<Int64>`; an inner `None` is a null *element*, an outer `None` a
    /// null *row*.
    fn int_lists(rows: &[Option<Vec<Option<i64>>>]) -> ArrayRef {
        use arrow::array::Int64Builder;
        let mut b = ListBuilder::new(Int64Builder::new());
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

    fn i64s(a: &ArrayRef) -> Vec<Option<i64>> {
        use arrow::array::AsArray;
        let x = a.as_primitive::<arrow::datatypes::Int64Type>();
        (0..x.len())
            .map(|i| (!x.is_null(i)).then(|| x.value(i)))
            .collect()
    }

    /// The bug this path exists for: routing an integer `list_sum` through Float64
    /// rounds every element to 53 significant bits, so `[2^53+1, 2]` summed to
    /// 2^53+2 instead of the true 2^53+3. DuckDB returns the exact integer.
    #[test]
    fn int_sum_is_exact_above_two_pow_53() {
        let big = (1i64 << 53) + 1;
        let a = int_lists(&[Some(vec![Some(big), Some(2)])]);
        let s = eval_list(ListFunc::Sum, &a).unwrap();
        assert_eq!(s.data_type(), &DataType::Int64, "sum of Int64 stays Int64");
        assert_eq!(i64s(&s), vec![Some(big + 2)]);
        // The float path would have produced this instead — pin the difference.
        assert_ne!(i64s(&s)[0].unwrap() as f64, (big as f64 + 2.0) - 1.0);
    }

    /// `list_avg` keeps a Float64 *result* but must accumulate the total exactly,
    /// or the division inherits the same lost bit.
    #[test]
    fn int_mean_accumulates_exactly_and_stays_float() {
        let big = (1i64 << 53) + 1;
        let a = int_lists(&[Some(vec![Some(big), Some(2)])]);
        let m = eval_list(ListFunc::Mean, &a).unwrap();
        assert_eq!(m.data_type(), &DataType::Float64);
        assert_eq!(f64s(&m), vec![Some((big + 2) as f64 / 2.0)]);
    }

    /// Null elements are skipped; a null row, an empty row and an all-null row each
    /// have no values to sum and so are null (DuckDB agrees on all three).
    #[test]
    fn int_sum_null_empty_and_all_null_rows() {
        let a = int_lists(&[
            Some(vec![Some(1), None, Some(2)]), // nulls skipped
            Some(vec![]),                       // empty row
            Some(vec![None, None]),             // all-null row
            None,                               // null row
            Some(vec![Some(7)]),                // single element
            Some(vec![Some(-5), Some(3)]),      // negatives
        ]);
        let s = eval_list(ListFunc::Sum, &a).unwrap();
        assert_eq!(i64s(&s), vec![Some(3), None, None, None, Some(7), Some(-2)]);
        let m = eval_list(ListFunc::Mean, &a).unwrap();
        assert_eq!(
            f64s(&m),
            vec![Some(1.5), None, None, None, Some(7.0), Some(-1.0)]
        );
    }

    /// i64 boundary: `i64::MAX` alone is representable and must come back exactly
    /// (the float path returns 9223372036854775808.0, which is not even an i64).
    /// Overflowing the accumulator errors rather than wrapping, matching the scalar
    /// `sum` aggregate (`bc-runtime`'s `checked_add` → `SumOverflow`).
    #[test]
    fn int_sum_i64_boundary_and_overflow_errors() {
        let exact = int_lists(&[Some(vec![Some(i64::MAX)]), Some(vec![Some(i64::MIN)])]);
        let s = eval_list(ListFunc::Sum, &exact).unwrap();
        assert_eq!(i64s(&s), vec![Some(i64::MAX), Some(i64::MIN)]);

        let over = int_lists(&[Some(vec![Some(i64::MAX), Some(1)])]);
        assert!(
            eval_list(ListFunc::Sum, &over).is_err(),
            "i64 overflow must error, not wrap"
        );
    }

    /// A narrow integer child (Int32) also takes the exact path — the FFI boundary
    /// normalizes narrow ints to Int64, but a list child can still arrive narrow.
    #[test]
    fn narrow_int_child_takes_the_int_path() {
        use arrow::array::{Int32Builder, ListBuilder};
        let mut b = ListBuilder::new(Int32Builder::new());
        for v in [1, 2, 3] {
            b.values().append_value(v);
        }
        b.append(true);
        let a: ArrayRef = Arc::new(b.finish());
        let s = eval_list(ListFunc::Sum, &a).unwrap();
        assert_eq!(s.data_type(), &DataType::Int64);
        assert_eq!(i64s(&s), vec![Some(6)]);
    }

    /// The float child path is untouched: still Float64 out, still NaN-aware, and
    /// `product` (which DuckDB also returns as a double) is unchanged for ints too.
    #[test]
    fn float_child_path_unchanged() {
        let a = lists(&[Some(vec![1.5, 2.5]), Some(vec![]), None]);
        let s = eval_list(ListFunc::Sum, &a).unwrap();
        assert_eq!(s.data_type(), &DataType::Float64);
        assert_eq!(f64s(&s), vec![Some(4.0), None, None]);
        let m = eval_list(ListFunc::Mean, &a).unwrap();
        assert_eq!(f64s(&m), vec![Some(2.0), None, None]);

        let ints = int_lists(&[Some(vec![Some(2), Some(3)])]);
        let p = eval_list(ListFunc::Product, &ints).unwrap();
        assert_eq!(p.data_type(), &DataType::Float64, "product stays double");
        assert_eq!(f64s(&p), vec![Some(6.0)]);
    }
}
