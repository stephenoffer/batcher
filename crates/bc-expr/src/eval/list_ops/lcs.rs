//! `list.lcs_length` — the longest common subsequence length of two lists.
//!
//! The one overlap measure that reads *order*. `multiset_overlap` counts how many elements two
//! bags share and cannot tell `the cat sat` from `sat cat the`; an LCS counts the longest run
//! of elements appearing in both, in the same relative order, and scores the second far lower.
//! That is the whole difference between ROUGE-N and ROUGE-L, and it is why summarization is
//! scored with the latter: a summary that uses the right words in the wrong order is not a
//! summary.
//!
//! **This is the expensive one.** The DP is `O(n·m)` in the two rows' lengths, against `O(n+m)`
//! for every other list op here. On tokenized sentences (tens of elements) that is nothing; on
//! two thousand-token documents it is a million cell updates per row, and a corpus of those is
//! a real cost rather than a rounding error. Truncate long texts, or score at the sentence
//! level, rather than reaching for it on whole documents.
//!
//! Elements are compared through Arrow's row encoding, so token strings, token ids, and n-gram
//! strings all work through one path. A null row on either side yields null; a null element
//! matches nothing, so it can never extend a subsequence.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Builder, ListArray};
use arrow::compute::concat;
use arrow::row::{Row, RowConverter, SortField};

use crate::ExprError;

/// The LCS length of each row's two lists, as Float64.
///
/// Float64 rather than Int64 to match every other `ListBinaryFunc`, so the ratios built on it
/// never have to cast.
pub(crate) fn eval_lcs_length(la: &ListArray, ra: &ListArray) -> Result<ArrayRef, ExprError> {
    // One converter over both children, so an element from either side encodes to the same
    // bytes — the same element-identity `list_set` and `multiset_overlap` use.
    let (lv, rv) = (la.values(), ra.values());
    let combined = concat(&[lv.as_ref(), rv.as_ref()])?;
    let roffset = lv.len();
    let key = crate::eval::list::float_canonical_key(&combined)?;
    let converter = RowConverter::new(vec![SortField::new(key.data_type().clone())])?;
    let rows = converter.convert_columns(std::slice::from_ref(&key))?;

    let (lo, ro) = (la.value_offsets(), ra.value_offsets());
    let mut out = Float64Builder::with_capacity(la.len());
    // Two rolling rows rather than the full table: the DP only ever reads the previous row, so
    // the memory is O(min(n, m)) instead of O(n·m). On a long document that is the difference
    // between a few kilobytes and a few megabytes per row.
    let mut previous: Vec<u32> = Vec::new();
    let mut current: Vec<u32> = Vec::new();
    for i in 0..la.len() {
        if la.is_null(i) || ra.is_null(i) {
            out.append_null();
            continue;
        }
        let left: Vec<Row> = (lo[i] as usize..lo[i + 1] as usize)
            .filter(|&k| !lv.is_null(k))
            .map(|k| rows.row(k))
            .collect();
        let right: Vec<Row> = (ro[i] as usize..ro[i + 1] as usize)
            .filter(|&k| !rv.is_null(k))
            .map(|k| rows.row(roffset + k))
            .collect();
        if left.is_empty() || right.is_empty() {
            out.append_value(0.0);
            continue;
        }
        // Iterate the longer side outside so the rolling rows are sized by the shorter one.
        let (outer, inner) = if left.len() >= right.len() {
            (&left, &right)
        } else {
            (&right, &left)
        };
        previous.clear();
        previous.resize(inner.len() + 1, 0);
        for a in outer {
            current.clear();
            current.push(0);
            for (j, b) in inner.iter().enumerate() {
                let value = if a == b {
                    previous[j] + 1
                } else {
                    current[j].max(previous[j + 1])
                };
                current.push(value);
            }
            std::mem::swap(&mut previous, &mut current);
        }
        out.append_value(previous[inner.len()] as f64);
    }
    Ok(Arc::new(out.finish()))
}

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, AsArray};
    use arrow::array::{Int64Builder, ListBuilder, StringBuilder};
    use arrow::datatypes::Float64Type;

    use super::*;

    fn strings(rows: &[Option<Vec<Option<&str>>>]) -> ListArray {
        let mut b = ListBuilder::new(StringBuilder::new());
        for row in rows {
            match row {
                Some(values) => {
                    for v in values {
                        match v {
                            Some(s) => b.values().append_value(s),
                            None => b.values().append_null(),
                        }
                    }
                    b.append(true);
                }
                None => b.append(false),
            }
        }
        b.finish()
    }

    fn ints(rows: &[Vec<i64>]) -> ListArray {
        let mut b = ListBuilder::new(Int64Builder::new());
        for row in rows {
            for v in row {
                b.values().append_value(*v);
            }
            b.append(true);
        }
        b.finish()
    }

    fn values(out: &ArrayRef) -> Vec<Option<f64>> {
        let a = out.as_primitive::<Float64Type>();
        (0..a.len())
            .map(|i| (!a.is_null(i)).then(|| a.value(i)))
            .collect()
    }

    fn words(row: &[&str]) -> ListArray {
        strings(&[Some(row.iter().map(|s| Some(*s)).collect())])
    }

    #[test]
    fn an_identical_sequence_matches_completely() {
        let a = words(&["the", "cat", "sat"]);
        assert_eq!(values(&eval_lcs_length(&a, &a).unwrap()), vec![Some(3.0)]);
    }

    /// The property that separates this from a bag intersection.
    #[test]
    fn a_reordering_scores_below_the_original() {
        let ordered = words(&["the", "cat", "sat"]);
        let shuffled = words(&["sat", "cat", "the"]);
        let got = values(&eval_lcs_length(&ordered, &shuffled).unwrap());
        assert_eq!(got, vec![Some(1.0)]);
    }

    #[test]
    fn a_subsequence_need_not_be_contiguous() {
        let a = words(&["a", "x", "b", "y", "c"]);
        let b = words(&["a", "b", "c"]);
        assert_eq!(values(&eval_lcs_length(&a, &b).unwrap()), vec![Some(3.0)]);
    }

    #[test]
    fn disjoint_sequences_share_nothing() {
        let a = words(&["a", "b"]);
        let b = words(&["c", "d"]);
        assert_eq!(values(&eval_lcs_length(&a, &b).unwrap()), vec![Some(0.0)]);
    }

    #[test]
    fn the_result_is_symmetric() {
        let a = words(&["a", "b", "c", "d"]);
        let b = words(&["b", "d", "a"]);
        let forward = values(&eval_lcs_length(&a, &b).unwrap());
        let backward = values(&eval_lcs_length(&b, &a).unwrap());
        assert_eq!(forward, backward);
    }

    #[test]
    fn an_empty_row_shares_nothing_but_is_not_null() {
        let a = strings(&[Some(vec![])]);
        let b = words(&["a"]);
        assert_eq!(values(&eval_lcs_length(&a, &b).unwrap()), vec![Some(0.0)]);
    }

    #[test]
    fn a_null_row_on_either_side_is_null() {
        let a = strings(&[None, Some(vec![Some("a")])]);
        let b = strings(&[Some(vec![Some("a")]), None]);
        assert_eq!(values(&eval_lcs_length(&a, &b).unwrap()), vec![None, None]);
    }

    /// A null element cannot extend a subsequence, on either side.
    #[test]
    fn null_elements_are_skipped_rather_than_matched() {
        let a = strings(&[Some(vec![None, Some("a"), None])]);
        let b = strings(&[Some(vec![None, Some("a")])]);
        assert_eq!(values(&eval_lcs_length(&a, &b).unwrap()), vec![Some(1.0)]);
    }

    #[test]
    fn integer_elements_use_the_same_path() {
        let a = ints(&[vec![1, 2, 3, 4]]);
        let b = ints(&[vec![2, 4]]);
        assert_eq!(values(&eval_lcs_length(&a, &b).unwrap()), vec![Some(2.0)]);
    }

    #[test]
    fn the_length_never_exceeds_the_shorter_row() {
        let a = words(&["a", "b", "c", "d", "e"]);
        let b = words(&["a", "b"]);
        let got = values(&eval_lcs_length(&a, &b).unwrap())[0].unwrap();
        assert!(got <= 2.0);
    }

    /// Rows are independent — the rolling DP buffers are reused and must be reset.
    #[test]
    fn rows_do_not_leak_state_into_each_other() {
        let a = strings(&[
            Some(vec![Some("a"), Some("b"), Some("c")]),
            Some(vec![Some("z")]),
        ]);
        let b = strings(&[
            Some(vec![Some("a"), Some("b"), Some("c")]),
            Some(vec![Some("q")]),
        ]);
        assert_eq!(
            values(&eval_lcs_length(&a, &b).unwrap()),
            vec![Some(3.0), Some(0.0)]
        );
    }
}
