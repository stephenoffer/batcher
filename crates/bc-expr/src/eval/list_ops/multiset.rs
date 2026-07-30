//! `list.multiset_overlap` — the clipped multiset intersection size of two lists.
//!
//! Per row, `Σ_v min(count_left(v), count_right(v))`: how many of the left list's
//! elements the right list can account for, **counting repeats**. That clipping is the
//! whole point and is what separates it from `set_intersection().len()`, which counts a
//! value once no matter how often it repeats.
//!
//! It exists because the standard generation metrics are defined on it. BLEU's modified
//! n-gram precision clips each candidate n-gram at the number of times it appears in the
//! reference, precisely so that a model emitting `the the the the` cannot score a perfect
//! unigram precision against a reference holding one `the`. A set intersection scores that
//! degenerate output 1.0; this counts it 1/4. ROUGE-N's numerator is the same quantity read
//! from the reference's side.
//!
//! Comparison is type-general: elements are compared through Arrow's order-preserving row
//! encoding, so `List<Utf8>` (n-grams, tokens), `List<Int64>` (token ids) and the rest all
//! work through one path. A null list row on either side yields null; a null *element* is
//! dropped, since a null never equals anything.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Builder, ListArray};
use arrow::compute::concat;
use arrow::row::{OwnedRow, RowConverter, SortField};

use crate::ExprError;

/// The clipped multiset intersection size of each row's two lists, as Float64.
///
/// Float64 rather than Int64 to match every other `ListBinaryFunc`, so the metric
/// expressions that divide by it never have to cast.
pub(crate) fn eval_multiset_overlap(la: &ListArray, ra: &ListArray) -> Result<ArrayRef, ExprError> {
    // Strings are the overwhelmingly common element here — every generation metric feeds this
    // n-gram text — and the general path below is expensive for them: it concatenates both
    // children (copying every string) and row-encodes the result before a single comparison.
    // Hashing `&str` directly skips both, and the borrow is safe because the arrays outlive
    // the call. Measured on 20k pairs of 40-token text, this is the difference between a
    // metric that scans a corpus and one that is only usable on a sample.
    if let (Some(left), Some(right)) = (utf8_child(la), utf8_child(ra)) {
        return Ok(utf8_overlap(la, left, ra, right));
    }
    // One converter over both children so an element from either side encodes to the same
    // bytes; the children are concatenated (left keeps its index, right `k` maps to
    // `left.len() + k`) exactly as `list_set` does.
    // `List<Null>` — what an all-empty list column infers to — cannot be concatenated with a
    // `List<Utf8>` child, so align the element types before encoding them together.
    let (lv, rv) = super::align_children(la.values(), ra.values())?;
    let combined = concat(&[lv.as_ref(), rv.as_ref()])?;
    let roffset = lv.len();
    // Compare through the float-canonical key so `-0.0`/`0.0` and every NaN collapse the
    // way `=` and `GROUP BY` do, matching `list_set`'s element identity.
    let key = crate::eval::list::float_canonical_key(&combined)?;
    let converter = RowConverter::new(vec![SortField::new(key.data_type().clone())])?;
    let rows = converter.convert_columns(std::slice::from_ref(&key))?;

    let (lo, ro) = (la.value_offsets(), ra.value_offsets());
    let mut out = Float64Builder::with_capacity(la.len());
    // Reused across rows so a batch of short lists does not allocate a map per row.
    let mut counts: HashMap<OwnedRow, i64> = HashMap::new();
    for i in 0..la.len() {
        if la.is_null(i) || ra.is_null(i) {
            out.append_null();
            continue;
        }
        counts.clear();
        // Count the *right* side first, then walk the left drawing down that budget: one
        // pass each, and the draw-down is what implements the clip.
        for k in ro[i] as usize..ro[i + 1] as usize {
            if rv.is_null(k) {
                continue;
            }
            *counts.entry(rows.row(roffset + k).owned()).or_insert(0) += 1;
        }
        let mut overlap = 0i64;
        for k in lo[i] as usize..lo[i + 1] as usize {
            if lv.is_null(k) {
                continue;
            }
            if let Some(remaining) = counts.get_mut(&rows.row(k).owned()) {
                if *remaining > 0 {
                    *remaining -= 1;
                    overlap += 1;
                }
            }
        }
        out.append_value(overlap as f64);
    }
    Ok(Arc::new(out.finish()))
}

/// The `Utf8` child of a list column, when that is what it holds.
fn utf8_child(list: &ListArray) -> Option<&arrow::array::StringArray> {
    use arrow::array::AsArray;
    list.values().as_string_opt::<i32>()
}

/// `eval_multiset_overlap` over two `List<Utf8>` columns, hashing the strings directly.
///
/// Identical contract to the general path: a null row on either side is null, an empty row is
/// a real zero, and a null element matches nothing.
fn utf8_overlap(
    la: &ListArray,
    left: &arrow::array::StringArray,
    ra: &ListArray,
    right: &arrow::array::StringArray,
) -> ArrayRef {
    let (lo, ro) = (la.value_offsets(), ra.value_offsets());
    let mut out = Float64Builder::with_capacity(la.len());
    let mut counts: HashMap<&str, i64> = HashMap::new();
    for i in 0..la.len() {
        if la.is_null(i) || ra.is_null(i) {
            out.append_null();
            continue;
        }
        counts.clear();
        for k in ro[i] as usize..ro[i + 1] as usize {
            if !right.is_null(k) {
                *counts.entry(right.value(k)).or_insert(0) += 1;
            }
        }
        let mut overlap = 0i64;
        for k in lo[i] as usize..lo[i + 1] as usize {
            if left.is_null(k) {
                continue;
            }
            if let Some(remaining) = counts.get_mut(left.value(k)) {
                if *remaining > 0 {
                    *remaining -= 1;
                    overlap += 1;
                }
            }
        }
        out.append_value(overlap as f64);
    }
    Arc::new(out.finish())
}

#[cfg(test)]
mod tests {
    use arrow::array::{Int64Builder, ListBuilder, StringBuilder};

    use super::*;

    fn str_lists(rows: &[Option<Vec<Option<&str>>>]) -> ListArray {
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

    fn int_lists(rows: &[Vec<i64>]) -> ListArray {
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
        use arrow::array::AsArray;
        use arrow::datatypes::Float64Type;
        let a = out.as_primitive::<Float64Type>();
        (0..a.len())
            .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
            .collect()
    }

    /// The clip is the whole point: four `the`s against one reference `the` count once.
    #[test]
    fn repeats_are_clipped_at_the_reference_count() {
        let left = str_lists(&[Some(vec![
            Some("the"),
            Some("the"),
            Some("the"),
            Some("the"),
        ])]);
        let right = str_lists(&[Some(vec![Some("the"), Some("cat")])]);
        let out = eval_multiset_overlap(&left, &right).unwrap();
        assert_eq!(values(&out), vec![Some(1.0)]);
    }

    /// Repeats present on *both* sides are counted as many times as both can pay for.
    #[test]
    fn repeats_on_both_sides_count_up_to_the_minimum() {
        let left = str_lists(&[Some(vec![Some("a"), Some("a"), Some("a")])]);
        let right = str_lists(&[Some(vec![Some("a"), Some("a")])]);
        let out = eval_multiset_overlap(&left, &right).unwrap();
        assert_eq!(values(&out), vec![Some(2.0)]);
    }

    #[test]
    fn disjoint_lists_overlap_in_nothing() {
        let left = str_lists(&[Some(vec![Some("a"), Some("b")])]);
        let right = str_lists(&[Some(vec![Some("c")])]);
        let out = eval_multiset_overlap(&left, &right).unwrap();
        assert_eq!(values(&out), vec![Some(0.0)]);
    }

    /// A null row on either side is null; an empty list is a real zero, not a null.
    #[test]
    fn null_rows_propagate_and_empty_rows_are_zero() {
        let left = str_lists(&[None, Some(vec![]), Some(vec![Some("a")])]);
        let right = str_lists(&[Some(vec![Some("a")]), Some(vec![Some("a")]), None]);
        let out = eval_multiset_overlap(&left, &right).unwrap();
        assert_eq!(values(&out), vec![None, Some(0.0), None]);
    }

    /// A null element cannot match anything, on either side.
    #[test]
    fn null_elements_never_match() {
        let left = str_lists(&[Some(vec![None, Some("a")])]);
        let right = str_lists(&[Some(vec![None, Some("a")])]);
        let out = eval_multiset_overlap(&left, &right).unwrap();
        assert_eq!(values(&out), vec![Some(1.0)]);
    }

    /// Type-general: integer token ids go through the same path as strings.
    #[test]
    fn integer_elements_use_the_same_path() {
        let left = int_lists(&[vec![1, 2, 2, 3]]);
        let right = int_lists(&[vec![2, 2, 4]]);
        let out = eval_multiset_overlap(&left, &right).unwrap();
        assert_eq!(values(&out), vec![Some(2.0)]);
    }

    /// The overlap is symmetric — `min` does not care which side is asked first.
    #[test]
    fn overlap_is_symmetric() {
        let left = str_lists(&[Some(vec![Some("a"), Some("a"), Some("b")])]);
        let right = str_lists(&[Some(vec![Some("a"), Some("c")])]);
        let forward = eval_multiset_overlap(&left, &right).unwrap();
        let backward = eval_multiset_overlap(&right, &left).unwrap();
        assert_eq!(values(&forward), values(&backward));
    }

    /// A list overlaps itself completely — the identity every ratio built on this relies on
    /// to score an exact reproduction 1.0.
    #[test]
    fn a_list_fully_overlaps_itself() {
        let rows = str_lists(&[Some(vec![Some("a"), Some("a"), Some("b"), Some("c")])]);
        let out = eval_multiset_overlap(&rows, &rows).unwrap();
        assert_eq!(values(&out), vec![Some(4.0)]);
    }

    /// `-0.0` and `0.0` are the same element, as they are for `=` and `GROUP BY`.
    #[test]
    fn negative_zero_matches_zero() {
        use arrow::array::Float64Builder as FB;
        let mut lb = ListBuilder::new(FB::new());
        lb.values().append_value(-0.0);
        lb.append(true);
        let left = lb.finish();
        let mut rb = ListBuilder::new(FB::new());
        rb.values().append_value(0.0);
        rb.append(true);
        let right = rb.finish();
        let out = eval_multiset_overlap(&left, &right).unwrap();
        assert_eq!(values(&out), vec![Some(1.0)]);
    }

    /// A batch is evaluated row-independently — the reused count map must not leak.
    #[test]
    fn rows_do_not_leak_counts_into_each_other() {
        let left = str_lists(&[
            Some(vec![Some("a"), Some("a")]),
            Some(vec![Some("b")]),
            Some(vec![Some("a")]),
        ]);
        let right = str_lists(&[
            Some(vec![Some("a")]),
            Some(vec![Some("c")]),
            Some(vec![Some("a"), Some("a")]),
        ]);
        let out = eval_multiset_overlap(&left, &right).unwrap();
        assert_eq!(values(&out), vec![Some(1.0), Some(0.0), Some(1.0)]);
    }
}
