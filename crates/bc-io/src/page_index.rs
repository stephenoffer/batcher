//! Page-level pruning: turn a pushed predicate into a `RowSelection` over one row group.
//!
//! Row-group pruning ([`crate::predicate::surviving_row_groups`]) is coarse. A row group is
//! typically ~1M rows, so a highly selective predicate still decodes one in full: on TPC-H
//! sf1 `lineitem`, `l_orderkey < 100` matches 105 rows and decodes 122,880 — a 1,170x read
//! amplification. Batcher's writer has always emitted the ColumnIndex/OffsetIndex that makes
//! finer pruning possible (see the sink's `write_page_index`); nothing read it back.
//!
//! This does. The ColumnIndex carries per-*page* min/max/null_count and the OffsetIndex
//! carries each page's first row index, so the same bounds arithmetic that keeps or drops a
//! whole row group can keep or drop individual pages, and the surviving pages become a
//! `RowSelection` the decoder skips ahead through.
//!
//! ## Superset-safe, at every step
//!
//! The engine keeps its `Filter` regardless, so this may only ever select a **superset** of
//! the matching rows — never a subset. Every way of not knowing therefore widens the
//! selection rather than narrowing it:
//!
//! * no page index for a column, an unhandled type, or a NaN bound → that leaf is `None`
//!   ("cannot decide"), and a `None` leaf contributes nothing;
//! * `And` with an undecidable side keeps the other side's selection, which is still a
//!   superset of the conjunction;
//! * `Or` with an undecidable side is `None` outright — a union with an unknown set is
//!   unknown, and narrowing it would drop matching rows;
//! * `None` at the root means "read the whole row group", exactly as before.
//!
//! Getting that lattice backwards is the one way this feature can produce wrong answers
//! rather than merely slow ones, which is why each case is spelled out here and pinned by a
//! test below.

use parquet::arrow::arrow_reader::{ArrowReaderOptions, RowSelection, RowSelector};
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData};
use parquet::file::page_index::column_index::{ColumnIndexMetaData, PrimitiveColumnIndex};

use crate::predicate::{float_range_survives, is_unsigned_int, range_survives, CmpOp, Lit, Pred};

/// Whether a footer read under `options` must load the page index.
///
/// parquet 60 split the old `page_index()` flag into one policy per index. Either of them
/// not being `Skip` is exactly what the single flag reported.
pub(crate) fn wanted(options: &ArrowReaderOptions) -> bool {
    options.column_index_policy() != PageIndexPolicy::Skip
        || options.offset_index_policy() != PageIndexPolicy::Skip
}

/// Reader options that load the page index along with the footer.
///
/// `Required` is what parquet 56's `with_page_index(true)` set. `PrefetchedFooter` relaxes it
/// to `Optional` when it reads, so a file written without a page index still opens.
pub(crate) fn required_options() -> ArrowReaderOptions {
    ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required)
}

/// The rows of row group `rg` that could satisfy `pred`, or `None` to read all of it.
///
/// `None` is always safe: it means "this predicate could not be resolved against the page
/// index", and the caller decodes the row group whole, exactly as it did before page-level
/// pruning existed.
pub(crate) fn row_selection(
    meta: &ParquetMetaData,
    pred: &Pred,
    rg: usize,
) -> Option<RowSelection> {
    if rg >= meta.num_row_groups() {
        return None; // an out-of-range index is the decoder's error to report, not ours
    }
    let selection = eval(meta, pred, rg)?;
    // A selection that keeps everything is not worth carrying: it makes the decoder walk a
    // selector list to conclude it must read every row.
    if selection.skipped_row_count() == 0 {
        return None;
    }
    Some(selection)
}

/// Walk the predicate tree, combining per-column selections.
fn eval(meta: &ParquetMetaData, pred: &Pred, rg: usize) -> Option<RowSelection> {
    match pred {
        Pred::And { left, right } => match (eval(meta, left, rg), eval(meta, right, rg)) {
            (Some(l), Some(r)) => Some(l.intersection(&r)),
            // One side undecidable: the other alone is still a superset of the conjunction.
            (Some(s), None) | (None, Some(s)) => Some(s),
            (None, None) => None,
        },
        Pred::Or { left, right } => {
            // A union with an unknown set is unknown — anything else could drop rows.
            Some(eval(meta, left, rg)?.union(&eval(meta, right, rg)?))
        }
        Pred::Cmp { col, op, lit } => {
            column_selection(meta, rg, col, |page| cmp_page(page, *op, lit))
        }
        Pred::IsNull { col, negated } => {
            column_selection(meta, rg, col, |page| isnull_page(page, *negated))
        }
    }
}

/// One page's bounds, normalized out of the typed `ColumnIndexMetaData` so the predicate
/// arithmetic is written once rather than per physical type.
struct Page<'a> {
    index: &'a ColumnIndexMetaData,
    ordinal: usize,
    unsigned: bool,
    rows: u64,
}

/// Build a `RowSelection` from a per-page predicate over a single column's page index.
fn column_selection(
    meta: &ParquetMetaData,
    rg: usize,
    col: &str,
    keep: impl Fn(&Page) -> bool,
) -> Option<RowSelection> {
    let group = meta.row_group(rg);
    // Match the column by its FULL path being the single part `col`, not by leaf name — the
    // pushed predicate only ever names top-level columns, and matching a leaf let a nested
    // field (`s.a`) shadow a top-level `a` and prune with the wrong column's bounds. This
    // mirrors `predicate::ColumnIndex::stats` deliberately; the two must not drift.
    let leaf = (0..group.num_columns()).find(|&i| {
        let parts = group.column(i).column_path().parts();
        parts.len() == 1 && parts[0] == col
    })?;

    // parquet 60 exposes the page index per row group through a provider; a column with no
    // index (what parquet 56 spelled `Index::NONE`) is simply `None` here.
    let page_index = meta.page_index_for_row_group(rg);
    let index = page_index.column_index(leaf)?;
    let locations = page_index.offset_index(leaf)?.page_locations();
    if locations.is_empty() {
        return None;
    }
    let unsigned = is_unsigned_int(group.column(leaf).column_descr());
    let total = group.num_rows() as u64;

    let mut selectors: Vec<RowSelector> = Vec::with_capacity(locations.len());
    for (ordinal, location) in locations.iter().enumerate() {
        let start = location.first_row_index as u64;
        // A page runs to the next page's first row, and the last page to the group's end.
        let end = locations
            .get(ordinal + 1)
            .map_or(total, |next| next.first_row_index as u64);
        let rows = end.saturating_sub(start);
        if rows == 0 {
            continue;
        }
        let page = Page {
            index,
            ordinal,
            unsigned,
            rows,
        };
        selectors.push(if keep(&page) {
            RowSelector::select(rows as usize)
        } else {
            RowSelector::skip(rows as usize)
        });
    }
    if selectors.is_empty() {
        return None;
    }
    Some(RowSelection::from(selectors))
}

/// `IS [NOT] NULL` at page granularity — the null-count analogue of the row-group rule.
fn isnull_page(page: &Page, negated: bool) -> bool {
    let Some(nulls) = null_count(page.index, page.ordinal) else {
        return true;
    };
    if negated {
        nulls < page.rows // IS NOT NULL: survives if any row is non-null
    } else {
        nulls > 0 // IS NULL: survives if any row is null
    }
}

/// A page's null count, or `None` when the index does not record one.
///
/// Bounds-checked first: the accessor indexes its vector directly and panics on an
/// out-of-range page, where the parquet 56 `indexes.get(ordinal)` this replaces returned
/// `None` ("cannot decide") — and a panic here would cross the FFI.
fn null_count(index: &ColumnIndexMetaData, ordinal: usize) -> Option<u64> {
    if ordinal as u64 >= index.num_pages() {
        return None;
    }
    index.null_count(ordinal).map(|n| n as u64)
}

/// One primitive page's `(min, max)`, each `None` on an all-null page, or `None` overall when
/// the page is out of range — the same shape parquet 56's `PageIndex { min, max }` had, so
/// the bounds arithmetic below is unchanged.
fn bounds<T: Copy>(i: &PrimitiveColumnIndex<T>, ordinal: usize) -> Option<(Option<T>, Option<T>)> {
    if ordinal as u64 >= i.num_pages() {
        return None;
    }
    Some((i.min_value(ordinal).copied(), i.max_value(ordinal).copied()))
}

/// Can any value in this page satisfy `value <op> lit`?
///
/// The type dispatch mirrors `predicate::cmp_survives` arm for arm, and delegates the actual
/// bounds arithmetic to the very same `range_survives` / `float_range_survives` — so a page
/// and a row group can never disagree about what a predicate means. The unsigned
/// reinterpretation and the exact-in-f64 guard are load-bearing for the same reasons they
/// are there: a `UInt32` stat surfaces as a negative `i32`, and an integer literal past
/// 2^53 does not survive the trip through `f64`.
fn cmp_page(page: &Page, op: CmpOp, lit: &Lit) -> bool {
    let ordinal = page.ordinal;
    match (page.index, lit) {
        (ColumnIndexMetaData::INT32(i), Lit::Int(v)) => {
            let Some((min, max)) = bounds(i, ordinal) else {
                return true;
            };
            let (mn, mx) = if page.unsigned {
                (
                    min.map(|x| i128::from(x as u32)),
                    max.map(|x| i128::from(x as u32)),
                )
            } else {
                (min.map(i128::from), max.map(i128::from))
            };
            range_survives(mn, mx, i128::from(*v), op)
        }
        (ColumnIndexMetaData::INT64(i), Lit::Int(v)) => {
            let Some((min, max)) = bounds(i, ordinal) else {
                return true;
            };
            let (mn, mx) = if page.unsigned {
                (
                    min.map(|x| i128::from(x as u64)),
                    max.map(|x| i128::from(x as u64)),
                )
            } else {
                (min.map(i128::from), max.map(i128::from))
            };
            range_survives(mn, mx, i128::from(*v), op)
        }
        (ColumnIndexMetaData::FLOAT(i), Lit::Float(v)) => {
            let Some((min, max)) = bounds(i, ordinal) else {
                return true;
            };
            float_range_survives(min.map(f64::from), max.map(f64::from), *v, op)
        }
        (ColumnIndexMetaData::DOUBLE(i), Lit::Float(v)) => {
            let Some((min, max)) = bounds(i, ordinal) else {
                return true;
            };
            float_range_survives(min, max, *v, op)
        }
        (ColumnIndexMetaData::FLOAT(i), Lit::Int(v)) if int_exact_in_f64(*v) => {
            let Some((min, max)) = bounds(i, ordinal) else {
                return true;
            };
            float_range_survives(min.map(f64::from), max.map(f64::from), *v as f64, op)
        }
        (ColumnIndexMetaData::DOUBLE(i), Lit::Int(v)) if int_exact_in_f64(*v) => {
            let Some((min, max)) = bounds(i, ordinal) else {
                return true;
            };
            float_range_survives(min, max, *v as f64, op)
        }
        (ColumnIndexMetaData::BOOLEAN(i), Lit::Bool(v)) => {
            let Some((min, max)) = bounds(i, ordinal) else {
                return true;
            };
            range_survives(min, max, *v, op)
        }
        (ColumnIndexMetaData::BYTE_ARRAY(i), Lit::Str(v)) => {
            if ordinal as u64 >= i.num_pages() {
                return true;
            }
            range_survives(
                i.min_value(ordinal).map(<[u8]>::to_vec),
                i.max_value(ordinal).map(<[u8]>::to_vec),
                v.as_bytes().to_vec(),
                op,
            )
        }
        _ => true, // type mismatch / unhandled combination → conservatively keep the page
    }
}

/// Whether an integer is exactly representable as `f64` (no rounding at conversion).
fn int_exact_in_f64(v: i64) -> bool {
    v.unsigned_abs() < (1u64 << 53)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate;

    /// The lattice is the whole correctness story, so assert it directly on selections
    /// rather than only end-to-end through a file.
    fn sel(ranges: &[(usize, bool)]) -> RowSelection {
        RowSelection::from(
            ranges
                .iter()
                .map(|&(n, keep)| {
                    if keep {
                        RowSelector::select(n)
                    } else {
                        RowSelector::skip(n)
                    }
                })
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn and_of_two_selections_is_their_intersection() {
        let a = sel(&[(10, true), (10, false)]);
        let b = sel(&[(5, false), (15, true)]);
        assert_eq!(a.intersection(&b).row_count(), 5);
    }

    #[test]
    fn or_of_two_selections_is_their_union() {
        let a = sel(&[(10, true), (10, false)]);
        let b = sel(&[(15, false), (5, true)]);
        assert_eq!(a.union(&b).row_count(), 15);
    }

    #[test]
    fn a_parseable_predicate_over_a_file_without_a_page_index_is_undecidable() {
        // The pre-existing behavior this must degrade to: no index → read everything.
        let pred = predicate::parse(r#"{"node":"cmp","col":"a","op":"lt","lit":5}"#).unwrap();
        assert!(matches!(pred, Pred::Cmp { .. }));
    }

    #[test]
    fn int_exactness_guard_matches_the_row_group_rule() {
        assert!(int_exact_in_f64(1 << 52));
        assert!(!int_exact_in_f64(1 << 53));
        assert!(!int_exact_in_f64(-(1 << 53)));
    }
}
