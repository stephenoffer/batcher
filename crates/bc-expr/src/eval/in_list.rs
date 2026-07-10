//! `x IN (lit, lit, …)` — hash-set membership.
//!
//! Replaces the O(N·k) `(x = l0) OR (x = l1) OR …` chain the SQL front end would
//! otherwise build with a single hash-set lookup per row (O(N) total). Null input →
//! null, matching the OR-of-equals Kleene semantics it folds from (a null never
//! equals any literal, and `NULL OR NULL = NULL`). This is also the kernel a runtime
//! join filter uses to prune a probe side by the build side's key set.

use std::collections::HashSet;
use std::hash::Hash;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, Date32Array, Int64Array, StringArray};
use arrow::datatypes::DataType;
use arrow::error::ArrowError;

use crate::{ExprError, Literal};

/// At/below this many members, a linear scan of the set beats hashing every input row.
///
/// The per-row cost of a hash set is a full hash (`std`'s SipHash for strings — tens of
/// cycles) plus a probe; a linear scan is `len` equality compares, each a length check
/// then a short `memcmp`/integer compare that rejects in one or two operations. For a
/// tiny set (the overwhelmingly common `IN ('MAIL', 'SHIP')` / `IN (1, 2, 3)` shape) the
/// scan drops the per-row hash entirely — the win where the input column is cache-resident
/// and the kernel is compute-bound. (When the column is large and streamed from RAM the
/// filter is memory-bandwidth-bound on materializing the values, so this trims
/// instructions without moving wall-clock — e.g. TPC-H Q12's 60M-row `l_shipmode` scan.)
/// Above the threshold the hash set's O(1) probe dominates, so the set is built and
/// probed as before.
const LINEAR_SCAN_MAX: usize = 8;

/// A membership test over a set of `T`: a linear scan when the set is tiny
/// ([`LINEAR_SCAN_MAX`]), a hash set otherwise. `contains` is identical either way (set
/// membership is method-independent), so the produced mask is bit-for-bit unchanged.
enum Members<T> {
    Linear(Vec<T>),
    Hashed(HashSet<T>),
}

impl<T: Hash + Eq> Members<T> {
    fn new(items: Vec<T>) -> Self {
        if items.len() <= LINEAR_SCAN_MAX {
            Members::Linear(items)
        } else {
            Members::Hashed(items.into_iter().collect())
        }
    }

    #[inline]
    fn contains(&self, value: &T) -> bool {
        match self {
            Members::Linear(items) => items.iter().any(|m| m == value),
            Members::Hashed(set) => set.contains(value),
        }
    }
}

/// Evaluate `array IN set` to a `BooleanArray` (null where `array` is null).
pub(crate) fn eval_in_list(array: &ArrayRef, set: &[Literal]) -> Result<ArrayRef, ExprError> {
    let out: BooleanArray = match array.data_type() {
        DataType::Int64 => {
            let a = array.as_any().downcast_ref::<Int64Array>().expect("int64");
            let members = Members::new(set.iter().filter_map(literal_i64).collect());
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
            let members = Members::new(set.iter().filter_map(literal_date).collect());
            membership(
                a.len(),
                |i| a.is_valid(i),
                |i| members.contains(&a.value(i)),
            )
        }
        DataType::Utf8 => {
            let a = array.as_any().downcast_ref::<StringArray>().expect("utf8");
            let members = Members::new(set.iter().filter_map(literal_str).collect());
            membership(
                a.len(),
                |i| a.is_valid(i),
                |i| members.contains(&a.value(i)),
            )
        }
        // A dictionary-encoded column: evaluate membership on the *dictionary values* (a
        // handful of distinct entries) once, then gather one bit per row through the keys.
        // This is the classic dict-accelerated `IN` — O(distinct + rows) rather than
        // O(rows) full-value probes — and is bit-identical to the decoded path: `take`
        // maps a null key to a null output (a null row) and a null dictionary value to its
        // null membership bit, exactly matching `NULL IN set = NULL`.
        DataType::Dictionary(_, _) => {
            let dict = array.as_any_dictionary();
            let member_over_values = eval_in_list(dict.values(), set)?;
            let out = arrow::compute::take(&member_over_values, dict.keys(), None)?;
            return Ok(out);
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

    #[test]
    fn large_set_uses_hashed_path_identically() {
        // A set past LINEAR_SCAN_MAX takes the HashSet branch; membership must match the
        // linear branch exactly. 12 members (> 8), probing values in and out of the set.
        let set: Vec<Literal> = (0..12).map(Literal::Int).collect();
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![Some(0), Some(11), Some(12), None]));
        assert_eq!(
            run(arr, &set),
            vec![Some(true), Some(true), Some(false), None]
        );
    }

    #[test]
    fn dictionary_in_list_equals_decoded() {
        use arrow::array::DictionaryArray;
        use arrow::datatypes::Int32Type;
        // A low-cardinality Utf8 dictionary column with a null row.
        let values = ["MAIL", "SHIP", "AIR", "RAIL"];
        let dict: DictionaryArray<Int32Type> =
            [Some("MAIL"), Some("AIR"), None, Some("SHIP"), Some("RAIL")]
                .into_iter()
                .collect();
        let _ = values;
        let dict_arr: ArrayRef = Arc::new(dict.clone());
        let decoded: ArrayRef = arrow::compute::cast(&dict_arr, &arrow::datatypes::DataType::Utf8)
            .expect("decode dict");
        let set = [Literal::Str("MAIL".into()), Literal::Str("SHIP".into())];
        // The dict-accelerated path must equal the decoded full-value path, bit for bit.
        assert_eq!(run(dict_arr, &set), run(decoded, &set));
        // And the expected values: MAIL∈, AIR∉, null→null, SHIP∈, RAIL∉.
        assert_eq!(
            run(Arc::new(dict), &set),
            vec![Some(true), Some(false), None, Some(true), Some(false)]
        );
    }

    #[test]
    fn small_and_large_string_sets_agree() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("MAIL"), Some("AIR"), None]));
        let small = [Literal::Str("MAIL".into()), Literal::Str("SHIP".into())];
        // Pad to > LINEAR_SCAN_MAX so the same values take the hashed path.
        let mut large = small.to_vec();
        large.extend((0..10).map(|i| Literal::Str(format!("X{i}"))));
        assert_eq!(
            run(arr.clone(), &small),
            vec![Some(true), Some(false), None]
        );
        assert_eq!(run(arr, &large), vec![Some(true), Some(false), None]);
    }
}
