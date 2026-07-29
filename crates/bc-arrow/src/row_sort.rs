//! A stable multi-column sort permutation over the Arrow row format.
//!
//! Sorting by several keys means comparing several columns per pair. Arrow does that with
//! `lexsort_to_indices`, which is a *comparator* sort: it monomorphizes a specialized
//! comparator for 2, 3 and 4 columns (`sort_fixed_column::<N>`) and falls back to a generic
//! one beyond that. The alternative every engine that competes on sort throughput uses is to
//! encode the key columns into one order-preserving byte string per row up front, so each
//! comparison becomes a `memcmp` — DuckDB calls it a sort key, Polars its row encoding, and
//! arrow-rs ships the encoder as `arrow::row::RowConverter` without using it for sorting.
//!
//! Encoding is not free, so which wins is a question about the crossover, not a matter of
//! principle. Measured on 1,048,576 rows with heavy ties and nulls by
//! `report_the_row_encoding_crossover`
//! (`cargo test --release -p bc-arrow --test row_sort_equivalence -- --ignored --nocapture`):
//!
//! | Keys | `lexsort_to_indices` | Row-encoded | Ratio |
//! |---|---|---|---|
//! | 1 | 300 ms | 397 ms | 0.76x |
//! | 2 | 444 ms | 460 ms | 0.97x |
//! | 3 | 797 ms | 533 ms | **1.49x** |
//! | 4 | 820 ms | 610 ms | **1.34x** |
//! | 5 | 942 ms | 627 ms | **1.50x** |
//!
//! Absolute times move a few percent run to run on a shared machine; the crossover does not.
//!
//! So the encode pass costs about what one or two columns of comparator dispatch cost, and
//! pays from three keys on. [`stable_lexsort_indices`] therefore declines below
//! [`MIN_KEYS_FOR_ROW_ENCODING`] rather than leaving that judgement to each caller: the
//! threshold belongs next to the measurement that sets it.
//!
//! ## The row-index column this removes
//!
//! `lexsort_to_indices` is **unstable**, and the engine needs a deterministic tie order: the
//! sequential oracle, the parallel sample-sort and the external merge sort each run a sort
//! over a differently-sized slice, and rows tied on every key must come out in the same
//! relative order in all three or `seq == par` fails. The existing fix appends an ascending
//! row-index *column* to the key list, making every row unique and so forcing a total order.
//!
//! That is a whole extra key for the sort to carry. The observation is that the index only
//! ever decides a comparison in which every real key compared equal, so it does not need to
//! be one of the comparands at all — it can live in the comparator:
//!
//! ```text
//! encode([k1..kn, row_index])            vs   encode([k1..kn])
//! compare(rows[a], rows[b])                    compare(rows[a], rows[b]).then(a.cmp(b))
//! ```
//!
//! Both are lexicographic over the same sequence, because the encoding is order-preserving
//! and the index is the last key either way. The permutation is therefore identical, which
//! is what the caller relies on and what
//! `stable_row_sort_matches_lexsort_with_index_tiebreak` pins against that very path.
//!
//! Keeping the index out of the comparands also lets this stay `sort_unstable_by`
//! (pattern-defeating quicksort, no allocation) instead of needing a stable merge sort: the
//! comparator is a total order, so an unstable algorithm is already deterministic.
//!
//! ## What this deliberately does not do
//!
//! There is no top-N form. Arrow's partial sort keeps a bounded region of size `limit`
//! (`partial_sort`), which is `O(n log limit)`; selecting the `limit`-th encoded row and
//! sorting the prefix measured 0.60x to 0.66x against it at every limit tried, because the
//! encode pass is `O(n)` in *row width* while the comparator sort it replaces barely touches
//! the tail. A bounded heap over encoded rows might close that, and until something measures
//! it the multi-key top-N keeps arrow's path.

use arrow::array::{Array, ArrayRef, UInt32Array};
use arrow::compute::SortOptions;
use arrow::row::{RowConverter, SortField};

/// Key count from which row encoding beats arrow's comparator sort.
///
/// Set from the table in the module docs: below three keys the encode pass costs more than
/// the comparator dispatch it removes. It is a measured crossover on one machine and one
/// data shape, not a law — re-run the equivalence test before moving it.
pub const MIN_KEYS_FOR_ROW_ENCODING: usize = 3;

/// The permutation that sorts `columns` lexicographically under `options`, with rows tied on
/// every column left in input order.
///
/// Returns `None` when this is not the right tool, in which case the caller keeps its
/// existing path. That is a routing decision and never an error, so there is always a
/// general comparison sort behind it. `None` means one of: fewer than
/// [`MIN_KEYS_FOR_ROW_ENCODING`] columns, a column-count or length mismatch, more rows than
/// a `UInt32Array` permutation can address, or a type the row encoder rejects
/// (`RowConverter::new` is fallible for some nested and extension types).
pub fn stable_lexsort_indices(
    columns: &[ArrayRef],
    options: &[SortOptions],
) -> Option<UInt32Array> {
    if columns.len() < MIN_KEYS_FOR_ROW_ENCODING || columns.len() != options.len() {
        return None;
    }
    let rows_len = columns[0].len();
    if columns.iter().any(|c| c.len() != rows_len) || rows_len > u32::MAX as usize {
        return None;
    }

    let fields: Vec<SortField> = columns
        .iter()
        .zip(options)
        .map(|(c, o)| SortField::new_with_options(c.data_type().clone(), *o))
        .collect();
    let converter = RowConverter::new(fields).ok()?;
    let rows = converter.convert_columns(columns).ok()?;

    let mut permutation: Vec<u32> = (0..rows_len as u32).collect();
    // The `.then(a.cmp(&b))` is the whole stability argument: it makes the comparator a total
    // order over distinct indices, so an unstable sort cannot place tied rows arbitrarily —
    // it must place them in ascending original position. See the module docs for why this is
    // identical to encoding the index as a trailing key column.
    permutation.sort_unstable_by(|&a, &b| {
        rows.row(a as usize)
            .cmp(&rows.row(b as usize))
            .then(a.cmp(&b))
    });
    Some(UInt32Array::from(permutation))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use arrow::compute::{lexsort_to_indices, SortColumn};

    fn opts(descending: bool, nulls_first: bool) -> SortOptions {
        SortOptions {
            descending,
            nulls_first,
        }
    }

    /// The permutation the engine produced before this existed: the keys plus an appended
    /// ascending row-index column, through arrow's unstable `lexsort_to_indices`.
    fn reference(columns: &[ArrayRef], options: &[SortOptions]) -> UInt32Array {
        let mut sort_columns: Vec<SortColumn> = columns
            .iter()
            .zip(options)
            .map(|(values, o)| SortColumn {
                values: Arc::clone(values),
                options: Some(*o),
            })
            .collect();
        let n = columns[0].len();
        sort_columns.push(SortColumn {
            values: Arc::new(UInt32Array::from_iter_values(0..n as u32)),
            options: Some(opts(false, false)),
        });
        lexsort_to_indices(&sort_columns, None).expect("reference lexsort")
    }

    /// Three columns with dense ties and nulls inside them, so the tie-break decides most
    /// comparisons and a stability bug would be visible rather than lucky.
    fn key_columns() -> Vec<ArrayRef> {
        let n = 400;
        let a: Int64Array = (0..n)
            .map(|i| if i % 11 == 0 { None } else { Some(i % 5) })
            .collect();
        let b: Int64Array = (0..n)
            .map(|i| if i % 7 == 0 { None } else { Some(i % 3) })
            .collect();
        let c: Int64Array = (0..n).map(|i| Some(i % 2)).collect();
        vec![
            Arc::new(a) as ArrayRef,
            Arc::new(b) as ArrayRef,
            Arc::new(c) as ArrayRef,
        ]
    }

    /// The permutation must equal what the path it replaces produces. This is the whole
    /// correctness claim: the sort gets cheaper and the answer does not move.
    #[test]
    fn stable_row_sort_matches_lexsort_with_index_tiebreak() {
        let columns = key_columns();
        for descending in [false, true] {
            for nulls_first in [false, true] {
                let options = vec![opts(descending, nulls_first); columns.len()];
                let expected = reference(&columns, &options);
                let got = stable_lexsort_indices(&columns, &options)
                    .expect("three int keys are row-encodable");
                assert_eq!(
                    got, expected,
                    "descending={descending} nulls_first={nulls_first}"
                );
            }
        }
    }

    /// Ties resolve to input order, which is what makes the sequential oracle, the parallel
    /// sample-sort and the external merge sort agree. Asserted under `descending` too,
    /// because flipping the key order must not flip the tie-break.
    #[test]
    fn ties_resolve_to_input_order() {
        for descending in [false, true] {
            let all_equal: Vec<ArrayRef> = (0..3)
                .map(|_| Arc::new(Int64Array::from(vec![7_i64; 64])) as ArrayRef)
                .collect();
            let got = stable_lexsort_indices(&all_equal, &[opts(descending, false); 3])
                .expect("sortable");
            assert_eq!(got.values(), &(0..64_u32).collect::<Vec<_>>());
        }
    }

    /// Mixed per-column directions, which a single `SortOptions` cannot express and where an
    /// encoding bug shows up as one column sorted the wrong way.
    #[test]
    fn per_column_directions_are_independent() {
        let a: ArrayRef = Arc::new(Int64Array::from(vec![1_i64, 1, 2, 2]));
        let b: ArrayRef = Arc::new(Int64Array::from(vec![10_i64, 20, 10, 20]));
        let c: ArrayRef = Arc::new(Int64Array::from(vec![0_i64; 4]));
        let columns = vec![a, b, c];
        let options = [opts(false, false), opts(true, false), opts(false, false)];
        // `a` ascending, `b` descending → (1,20), (1,10), (2,20), (2,10).
        let got = stable_lexsort_indices(&columns, &options).expect("sortable");
        assert_eq!(got.values(), &[1_u32, 0, 3, 2]);
        assert_eq!(got, reference(&columns, &options));
    }

    /// A string key mixed with numeric ones: `ORDER BY name, id, seq`, the shape this exists
    /// for. Checked against the reference rather than a hand-written expectation, so the
    /// null placement and collation come from arrow either way.
    #[test]
    fn string_and_numeric_keys_sort_together() {
        let s: ArrayRef = Arc::new(StringArray::from(vec![
            Some("b"),
            Some("a"),
            None,
            Some("a"),
        ]));
        let i: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), Some(2), Some(3), Some(1)]));
        let j: ArrayRef = Arc::new(Int64Array::from(vec![Some(0), Some(0), Some(0), Some(0)]));
        let columns = vec![s, i, j];
        for nulls_first in [false, true] {
            let options = vec![opts(false, nulls_first); 3];
            let got = stable_lexsort_indices(&columns, &options).expect("sortable");
            assert_eq!(
                got,
                reference(&columns, &options),
                "nulls_first={nulls_first}"
            );
        }
    }

    /// Floats reach the encoder as raw bits; the engine canonicalizes `-0.0`/NaN *before*
    /// this is called (`crate::float_ident`), so this only has to agree with the reference on
    /// whatever arrow does with them.
    #[test]
    fn float_keys_match_the_reference_path() {
        let f: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(1.5),
            None,
            Some(-2.0),
            Some(1.5),
            Some(0.0),
        ]));
        let g: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(1),
            Some(1),
            Some(1),
            None,
            Some(1),
        ]));
        let h: ArrayRef = Arc::new(Int64Array::from(vec![0_i64; 5]));
        let columns = vec![f, g, h];
        for nulls_first in [false, true] {
            let options = vec![opts(false, nulls_first); 3];
            let got = stable_lexsort_indices(&columns, &options).expect("sortable");
            assert_eq!(
                got,
                reference(&columns, &options),
                "nulls_first={nulls_first}"
            );
        }
    }

    /// Below the measured crossover this must decline, so a one- or two-key sort keeps
    /// arrow's comparator path. Without this the change would be a regression on the most
    /// common sort there is.
    #[test]
    fn declines_below_the_measured_crossover() {
        let a: ArrayRef = Arc::new(Int64Array::from(vec![3_i64, 1, 2]));
        assert_eq!(MIN_KEYS_FOR_ROW_ENCODING, 3);
        assert!(stable_lexsort_indices(&[Arc::clone(&a)], &[opts(false, false)]).is_none());
        assert!(stable_lexsort_indices(
            &[Arc::clone(&a), Arc::clone(&a)],
            &[opts(false, false); 2]
        )
        .is_none());
        assert!(stable_lexsort_indices(
            &[Arc::clone(&a), Arc::clone(&a), Arc::clone(&a)],
            &[opts(false, false); 3]
        )
        .is_some());
    }

    #[test]
    fn declines_a_mismatched_request() {
        let a: ArrayRef = Arc::new(Int64Array::from(vec![1_i64, 2]));
        let b: ArrayRef = Arc::new(Int64Array::from(vec![1_i64, 2, 3]));
        assert!(stable_lexsort_indices(&[], &[]).is_none());
        // Option count disagrees with column count.
        assert!(stable_lexsort_indices(
            &[Arc::clone(&a), Arc::clone(&a), Arc::clone(&a)],
            &[opts(false, false); 2]
        )
        .is_none());
        // Column lengths disagree.
        assert!(stable_lexsort_indices(
            &[Arc::clone(&a), Arc::clone(&a), b],
            &[opts(false, false); 3]
        )
        .is_none());
    }
}
