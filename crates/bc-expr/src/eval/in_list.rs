//! `x IN (lit, lit, …)` — hash-set membership.
//!
//! Replaces the O(N·k) `(x = l0) OR (x = l1) OR …` chain the SQL front end would
//! otherwise build with a single hash-set lookup per row (O(N) total). Null input →
//! null, matching the OR-of-equals Kleene semantics it folds from (a null never
//! equals any literal, and `NULL OR NULL = NULL`). This is also the kernel a runtime
//! join filter uses to prune a probe side by the build side's key set.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, Date32Array, Int64Array, StringArray};
use arrow::datatypes::DataType;
use arrow::error::ArrowError;

use crate::{ExprError, Literal};

/// Evaluate `array IN set` to a `BooleanArray` (null where `array` is null).
pub(crate) fn eval_in_list(array: &ArrayRef, set: &[Literal]) -> Result<ArrayRef, ExprError> {
    let out: BooleanArray = match array.data_type() {
        DataType::Int64 => {
            let a = array.as_any().downcast_ref::<Int64Array>().expect("int64");
            let members: HashSet<i64> = set.iter().filter_map(literal_i64).collect();
            membership(
                a.len(),
                |i| a.is_valid(i),
                |i| members.contains(&a.value(i)),
            )
        }
        DataType::Date32 => {
            let a = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .expect("date32");
            let members: HashSet<i32> = set.iter().filter_map(literal_date).collect();
            membership(
                a.len(),
                |i| a.is_valid(i),
                |i| members.contains(&a.value(i)),
            )
        }
        DataType::Utf8 => {
            let a = array.as_any().downcast_ref::<StringArray>().expect("utf8");
            let members: HashSet<&str> = set.iter().filter_map(literal_str).collect();
            membership(a.len(), |i| a.is_valid(i), |i| members.contains(a.value(i)))
        }
        other => {
            // `InList` is only emitted (by the fold rule) for these column types, so an
            // other dtype is a planner bug rather than user data.
            return Err(
                ArrowError::ComputeError(format!("in_list unsupported for {other:?}")).into(),
            );
        }
    };
    Ok(Arc::new(out))
}

/// One bool per row: `null` where invalid, else whether the value is a member.
/// `contains` is only called on valid rows, so it never reads a null slot.
fn membership(
    n: usize,
    valid: impl Fn(usize) -> bool,
    contains: impl Fn(usize) -> bool,
) -> BooleanArray {
    (0..n).map(|i| valid(i).then(|| contains(i))).collect()
}

fn literal_i64(lit: &Literal) -> Option<i64> {
    match lit {
        Literal::Int(v) => Some(*v),
        _ => None,
    }
}

fn literal_date(lit: &Literal) -> Option<i32> {
    match lit {
        Literal::Date(v) => Some(*v),
        _ => None,
    }
}

fn literal_str(lit: &Literal) -> Option<&str> {
    match lit {
        Literal::Str(v) => Some(v.as_str()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(arr: ArrayRef, set: &[Literal]) -> Vec<Option<bool>> {
        let out = eval_in_list(&arr, set).unwrap();
        let b = out.as_any().downcast_ref::<BooleanArray>().unwrap();
        (0..b.len())
            .map(|i| (!b.is_null(i)).then(|| b.value(i)))
            .collect()
    }

    #[test]
    fn int_membership_with_nulls() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), Some(2), None, Some(5)]));
        let set = [Literal::Int(1), Literal::Int(5)];
        // 1 ∈ set, 2 ∉, null → null, 5 ∈
        assert_eq!(
            run(arr, &set),
            vec![Some(true), Some(false), None, Some(true)]
        );
    }

    #[test]
    fn str_membership() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("13"), Some("99"), None]));
        let set = [Literal::Str("13".into()), Literal::Str("31".into())];
        assert_eq!(run(arr, &set), vec![Some(true), Some(false), None]);
    }

    #[test]
    fn empty_set_is_all_false_or_null() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None]));
        assert_eq!(run(arr, &[]), vec![Some(false), None]);
    }
}
