//! `StrFunc` evaluation with **per-row** parameters.
//!
//! `Expr::Str` fixes `pattern`/`replacement`/`start`/`length` at plan time, which is what
//! every kernel in this module is written against. SQL does not: `replace(s, old, new)`,
//! `substr(s, from_col, len_col)` and `s LIKE pattern_col` all take columns, and the
//! translator had to refuse each of them ("requires a constant string pattern") across
//! roughly thirty function spellings.
//!
//! Rather than write a second, row-wise implementation of each — thirty more chances to
//! disagree with the constant one — this groups the rows by their **distinct parameter
//! tuple** and calls the existing kernel once per group. The identity it rests on is that
//! every `StrFunc` is elementwise in its input and constant in its parameters, so
//! restricting the input to the rows that share one parameter tuple and applying the
//! kernel there gives exactly what a per-row evaluation would.
//!
//! Cost tracks the parameter column's *cardinality*, not the row count: a per-category
//! delimiter is one kernel call per category. All-distinct parameters degrade to one call
//! per row, which is the price of the shape rather than of this design.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{new_null_array, Array, ArrayRef, Int64Array, StringArray};
use arrow::compute::{cast, interleave, take};
use arrow::datatypes::DataType;

use super::eval_str;
use crate::{ExprError, StrFunc};

/// One row's parameter tuple, as the key the rows are grouped by.
type ParamKey = (Option<String>, Option<String>, Option<i64>, Option<i64>);

/// Evaluate `func` over `values` with the parameters taken per row from `params`.
///
/// Each element of `params` is the already-evaluated column for one slot, in the order
/// `(pattern, replacement, start, length)`; `None` means the slot is unset for this call.
/// A row whose *supplied* parameter is NULL yields NULL, as SQL requires of a function
/// whose argument is null.
pub(crate) fn eval_str_dynamic(
    func: StrFunc,
    values: &ArrayRef,
    pattern: Option<&ArrayRef>,
    replacement: Option<&ArrayRef>,
    start: Option<&ArrayRef>,
    length: Option<&ArrayRef>,
) -> Result<ArrayRef, ExprError> {
    let rows = values.len();
    let pattern = to_strings(pattern)?;
    let replacement = to_strings(replacement)?;
    let start = to_ints(start)?;
    let length = to_ints(length)?;

    // Group the row indices by their parameter tuple. A row with a null parameter is not
    // grouped at all — its answer is null regardless of the input.
    let mut groups: HashMap<ParamKey, Vec<u32>> = HashMap::new();
    let mut null_rows: Vec<usize> = Vec::new();
    for row in 0..rows {
        let (Some(p), Some(r), Some(st), Some(ln)) = (
            slot_str(&pattern, row),
            slot_str(&replacement, row),
            slot_int(&start, row),
            slot_int(&length, row),
        ) else {
            null_rows.push(row);
            continue;
        };
        groups.entry((p, r, st, ln)).or_default().push(row as u32);
    }

    // One kernel call per distinct parameter tuple, over just that group's rows.
    let mut parts: Vec<ArrayRef> = Vec::with_capacity(groups.len() + 1);
    // Slot 0 is reserved for the nulls; its type is only known after the first real
    // group runs, so it is filled in at the end.
    let mut placement: Vec<(usize, usize)> = vec![(0, 0); rows];
    for (key, idx) in groups {
        let (p, r, st, ln) = key;
        let taken = take(values, &arrow::array::UInt32Array::from(idx.clone()), None)?;
        let out = eval_str(func, &taken, p.as_deref(), r.as_deref(), st, ln)?;
        let slot = parts.len() + 1;
        for (pos, &row) in idx.iter().enumerate() {
            placement[row as usize] = (slot, pos);
        }
        parts.push(out);
    }
    let dtype = parts
        .first()
        .map(|a| a.data_type().clone())
        .unwrap_or(DataType::Utf8);
    // One null row is enough: every null-parameter output row points at the same slot.
    let nulls = new_null_array(&dtype, 1);
    for &row in &null_rows {
        placement[row] = (0, 0);
    }
    let mut arrays: Vec<&dyn Array> = Vec::with_capacity(parts.len() + 1);
    arrays.push(nulls.as_ref());
    for a in &parts {
        arrays.push(a.as_ref());
    }
    Ok(interleave(&arrays, &placement)?)
}

/// A parameter column as Utf8, or `None` when the slot is unset.
fn to_strings(arr: Option<&ArrayRef>) -> Result<Option<ArrayRef>, ExprError> {
    match arr {
        None => Ok(None),
        Some(a) if matches!(a.data_type(), DataType::Utf8) => Ok(Some(Arc::clone(a))),
        Some(a) => Ok(Some(cast(a, &DataType::Utf8)?)),
    }
}

/// A parameter column as Int64, or `None` when the slot is unset.
fn to_ints(arr: Option<&ArrayRef>) -> Result<Option<ArrayRef>, ExprError> {
    match arr {
        None => Ok(None),
        Some(a) if matches!(a.data_type(), DataType::Int64) => Ok(Some(Arc::clone(a))),
        Some(a) => Ok(Some(cast(a, &DataType::Int64)?)),
    }
}

/// `Some(None)` for an unset slot, `Some(Some(v))` for a value, `None` for a NULL — which
/// is what makes the whole row's answer null.
fn slot_str(arr: &Option<ArrayRef>, row: usize) -> Option<Option<String>> {
    let Some(a) = arr else { return Some(None) };
    if a.is_null(row) {
        return None;
    }
    let s = a.as_any().downcast_ref::<StringArray>()?;
    Some(Some(s.value(row).to_string()))
}

/// The integer counterpart of [`slot_str`].
fn slot_int(arr: &Option<ArrayRef>, row: usize) -> Option<Option<i64>> {
    let Some(a) = arr else { return Some(None) };
    if a.is_null(row) {
        return None;
    }
    let v = a.as_any().downcast_ref::<Int64Array>()?;
    Some(Some(v.value(row)))
}
