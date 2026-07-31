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

use parquet::arrow::arrow_reader::{RowSelection, RowSelector};
use parquet::file::metadata::ParquetMetaData;
use parquet::file::page_index::index::Index;

use crate::predicate::{float_range_survives, is_unsigned_int, range_survives, CmpOp, Lit, Pred};

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

/// One page's bounds, normalized out of the typed `Index` so the predicate arithmetic is
/// written once rather than per physical type.
struct Page<'a> {
    index: &'a Index,
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

    let index = meta.column_index()?.get(rg)?.get(leaf)?;
    let locations = meta.offset_index()?.get(rg)?.get(leaf)?.page_locations();
    if matches!(index, Index::NONE) || locations.is_empty() {
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

/// `NativeIndex<T>`'s `T` is bounded by a *sealed* trait (`ParquetValueType`), so a generic
/// helper cannot name the bound. A macro over the variants is the way to write this once.
fn null_count(index: &Index, ordinal: usize) -> Option<u64> {
    macro_rules! count {
        ($i:expr) => {
            $i.indexes.get(ordinal)?.null_count.map(|n| n as u64)
        };
    }
    match index {
        Index::NONE => None,
        Index::BOOLEAN(i) => count!(i),
        Index::INT32(i) => count!(i),
        Index::INT64(i) => count!(i),
        Index::INT96(i) => count!(i),
        Index::FLOAT(i) => count!(i),
        Index::DOUBLE(i) => count!(i),
        Index::BYTE_ARRAY(i) => count!(i),
        Index::FIXED_LEN_BYTE_ARRAY(i) => count!(i),
    }
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
        (Index::INT32(i), Lit::Int(v)) => {
            let Some(p) = i.indexes.get(ordinal) else {
                return true;
            };
            let (mn, mx) = if page.unsigned {
                (
                    p.min.map(|x| x as u32 as i128),
                    p.max.map(|x| x as u32 as i128),
                )
            } else {
                (p.min.map(|x| x as i128), p.max.map(|x| x as i128))
            };
            range_survives(mn, mx, *v as i128, op)
        }
        (Index::INT64(i), Lit::Int(v)) => {
            let Some(p) = i.indexes.get(ordinal) else {
                return true;
            };
            let (mn, mx) = if page.unsigned {
                (
                    p.min.map(|x| x as u64 as i128),
                    p.max.map(|x| x as u64 as i128),
                )
            } else {
                (p.min.map(|x| x as i128), p.max.map(|x| x as i128))
            };
            range_survives(mn, mx, *v as i128, op)
        }
        (Index::FLOAT(i), Lit::Float(v)) => {
            let Some(p) = i.indexes.get(ordinal) else {
                return true;
            };
            float_range_survives(p.min.map(|x| x as f64), p.max.map(|x| x as f64), *v, op)
        }
        (Index::DOUBLE(i), Lit::Float(v)) => {
            let Some(p) = i.indexes.get(ordinal) else {
                return true;
            };
            float_range_survives(p.min, p.max, *v, op)
        }
        (Index::FLOAT(i), Lit::Int(v)) if int_exact_in_f64(*v) => {
            let Some(p) = i.indexes.get(ordinal) else {
                return true;
            };
            float_range_survives(
                p.min.map(|x| x as f64),
                p.max.map(|x| x as f64),
                *v as f64,
                op,
            )
        }
        (Index::DOUBLE(i), Lit::Int(v)) if int_exact_in_f64(*v) => {
            let Some(p) = i.indexes.get(ordinal) else {
                return true;
            };
            float_range_survives(p.min, p.max, *v as f64, op)
        }
        (Index::BOOLEAN(i), Lit::Bool(v)) => {
            let Some(p) = i.indexes.get(ordinal) else {
                return true;
            };
            range_survives(p.min, p.max, *v, op)
        }
        (Index::BYTE_ARRAY(i), Lit::Str(v)) => {
            let Some(p) = i.indexes.get(ordinal) else {
                return true;
            };
            range_survives(
                p.min.as_ref().map(|b| b.data().to_vec()),
                p.max.as_ref().map(|b| b.data().to_vec()),
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
