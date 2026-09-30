//! `list.lcs_length` — the longest common subsequence length of two lists.
//!
//! The one overlap measure that reads *order*. `multiset_overlap` counts how many elements two
//! bags share and cannot tell `the cat sat` from `sat cat the`; an LCS counts the longest run
//! of elements appearing in both, in the same relative order, and scores the second far lower.
//! That is the whole difference between ROUGE-N and ROUGE-L, and it is why summarization is
//! scored with the latter: a summary that uses the right words in the wrong order is not a
//! summary.
//!
//! **This is the expensive one.** An LCS is `O(n·m)` in the two rows' lengths, against `O(n+m)`
//! for every other list op here. The kernel is the bit-parallel form (Allison and Dix 1986, in
//! Hyyrö's 2004 formulation): the shorter row becomes one bit vector per distinct element, and
//! each element of the longer row advances all `m` DP cells at once with a handful of 64-bit
//! word operations. That divides the work by 64, so two thousand-token documents cost about
//! 64 thousand word updates per row rather than four million cell updates, with the same
//! answer. It is still quadratic: truncate very long texts, or score at the sentence level.
//! When the shorter row has so many distinct elements that its bit vectors would not fit a
//! bounded budget, the kernel falls back to the classic two-row DP rather than allocate them.
//!
//! Elements are compared through Arrow's row encoding, so token strings, token ids, and n-gram
//! strings all work through one path. A null row on either side yields null; a null element
//! matches nothing, so it can never extend a subsequence.

use std::collections::HashMap;
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
    // `List<Null>` — what an all-empty list column infers to — cannot be concatenated with a
    // `List<Utf8>` child, so align the element types before encoding them together.
    let (lv, rv) = super::align_children(la.values(), ra.values())?;
    let combined = concat(&[lv.as_ref(), rv.as_ref()])?;
    let roffset = lv.len();
    let key = crate::eval::list::float_canonical_key(&combined)?;
    let converter = RowConverter::new(vec![SortField::new(key.data_type().clone())])?;
    let rows = converter.convert_columns(std::slice::from_ref(&key))?;

    let (lo, ro) = (la.value_offsets(), ra.value_offsets());
    let mut out = Float64Builder::with_capacity(la.len());
    let mut scratch = Scratch::default();
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
        // The longer side is scanned, the shorter one is encoded, so the bit vectors (or the
        // rolling DP rows) are sized by the shorter one.
        let (outer, inner) = if left.len() >= right.len() {
            (&left, &right)
        } else {
            (&right, &left)
        };
        out.append_value(f64::from(lcs_length(outer, inner, &mut scratch)));
    }
    Ok(Arc::new(out.finish()))
}

/// The most 64-bit words the per-element match vectors of one row may occupy (32 MiB). Past
/// it, the classic DP's `O(m)` memory is the better trade.
const MATCH_WORD_BUDGET: usize = 1 << 22;

/// The shorter-row length from which the bit-parallel kernel beats the DP.
const BIT_PARALLEL_MIN: usize = 32;

/// Buffers reused across rows, so a batch allocates once rather than once per row.
#[derive(Default)]
struct Scratch {
    masks: Vec<u64>,
    vector: Vec<u64>,
    previous: Vec<u32>,
    current: Vec<u32>,
}

/// The LCS length of `outer` and `inner`, with `inner` the shorter side.
fn lcs_length(outer: &[Row<'_>], inner: &[Row<'_>], scratch: &mut Scratch) -> u32 {
    if inner.is_empty() {
        return 0;
    }
    // Below one word's worth of cells the hash map costs more than the DP it replaces:
    // measured in release at 4 to 16 elements the DP is 1.3x to 4x faster, and from 32 up
    // the bit-parallel kernel wins (3x at 64, 54x at 2,000).
    if inner.len() < BIT_PARALLEL_MIN {
        return dp_lcs_length(outer, inner, scratch);
    }
    let words = inner.len().div_ceil(64);
    let mut ids: HashMap<&[u8], usize> = HashMap::with_capacity(inner.len());
    for row in inner {
        let next = ids.len();
        ids.entry(row.as_ref()).or_insert(next);
    }
    if ids.len().saturating_mul(words) > MATCH_WORD_BUDGET {
        return dp_lcs_length(outer, inner, scratch);
    }
    // One bit vector per distinct inner element: bit j is set where `inner[j]` is it.
    scratch.masks.clear();
    scratch.masks.resize(ids.len() * words, 0);
    for (j, row) in inner.iter().enumerate() {
        let id = ids[row.as_ref()];
        scratch.masks[id * words + j / 64] |= 1u64 << (j % 64);
    }
    // V starts all ones; each outer element with match vector M updates
    // V <- (V + (V & M)) | (V & !M), the addition carrying across words. The LCS is the
    // number of zero bits V ends with among its low `m`.
    scratch.vector.clear();
    scratch.vector.resize(words, u64::MAX);
    for row in outer {
        let Some(&id) = ids.get(row.as_ref()) else {
            continue; // matches nothing: M = 0 leaves V unchanged
        };
        let mask = &scratch.masks[id * words..(id + 1) * words];
        let mut carry = 0u64;
        for (v, &m) in scratch.vector.iter_mut().zip(mask) {
            let u = *v & m;
            let (sum, first) = v.overflowing_add(u);
            let (sum, second) = sum.overflowing_add(carry);
            carry = u64::from(first || second);
            *v = sum | (*v & !m);
        }
    }
    let tail = inner.len() % 64;
    let mut zeros = 0u32;
    for (w, v) in scratch.vector.iter().enumerate() {
        let live = if w + 1 == words && tail != 0 {
            (1u64 << tail) - 1
        } else {
            u64::MAX
        };
        zeros += (!v & live).count_ones();
    }
    zeros
}

/// The classic two-row DP: `O(n·m)` cell updates, `O(m)` memory. The reference the
/// bit-parallel kernel is tested against, and its fallback when the match vectors are too big.
fn dp_lcs_length(outer: &[Row<'_>], inner: &[Row<'_>], scratch: &mut Scratch) -> u32 {
    let (previous, current) = (&mut scratch.previous, &mut scratch.current);
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
        std::mem::swap(previous, current);
    }
    previous[inner.len()]
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

    /// The bit-parallel kernel agrees with the classic DP everywhere, including across the
    /// 64-bit word boundaries its carries run through and on heavy repetition.
    #[test]
    fn the_bit_parallel_kernel_matches_the_dp_on_random_rows() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        let converter =
            RowConverter::new(vec![SortField::new(arrow::datatypes::DataType::Int64)]).unwrap();
        let mut scratch = Scratch::default();
        for case in 0..400 {
            // Both sides at least `BIT_PARALLEL_MIN`, so the bit-parallel kernel is what runs.
            let (n, m) = (32 + next(300) as usize, 32 + next(200) as usize);
            let alphabet = [2, 4, 16, 1000][case % 4];
            let values: Vec<i64> = (0..n + m).map(|_| next(alphabet) as i64).collect();
            let column: ArrayRef = Arc::new(arrow::array::Int64Array::from(values));
            let rows = converter.convert_columns(&[column]).unwrap();
            let outer: Vec<Row> = (0..n).map(|k| rows.row(k)).collect();
            let inner: Vec<Row> = (n..n + m).map(|k| rows.row(k)).collect();
            let (long, short) = if outer.len() >= inner.len() {
                (&outer, &inner)
            } else {
                (&inner, &outer)
            };
            let fast = lcs_length(long, short, &mut scratch);
            let slow = dp_lcs_length(long, short, &mut scratch);
            assert_eq!(fast, slow, "case {case}: n={n} m={m} alphabet={alphabet}");
        }
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
