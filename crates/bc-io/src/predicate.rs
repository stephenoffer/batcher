//! Row-group pruning from a pushed predicate's zone maps (footer statistics).
//!
//! The Python control plane records the `Filter` sitting directly above a `Scan` as the
//! source's pushed predicate and translates its pushable subset to a compact JSON here
//! (`to_native_predicate`). This module parses that JSON and, for each candidate
//! row-group, asks "could any row in this group satisfy the predicate?" using the
//! group's per-column min/max/null-count statistics — skipping the whole group's
//! column-chunk GETs + decode when the answer is provably no.
//!
//! **Correctness contract:** pruning is *superset-safe*. The engine ALWAYS keeps the
//! `Filter` operator (source pushdown is best-effort), so returning extra rows is fine —
//! the one thing that must never happen is dropping a row-group that *could* match. Every
//! uncertain case (missing statistics, a type the evaluator does not handle exactly, a
//! literal that cannot be compared without lossy conversion) therefore **keeps** the
//! group. Comparisons are done only in exact arithmetic (`i64`/`f64` where lossless,
//! byte-lexicographic for strings, `bool`), never a lossy `i64 -> f64` at magnitudes that
//! could round across the literal.

use parquet::file::metadata::{ParquetMetaData, RowGroupMetaData};
use parquet::file::statistics::Statistics;
use serde::Deserialize;

/// One node of the pushed predicate's pushable subset.
#[derive(Deserialize, Clone)]
#[serde(tag = "node", rename_all = "snake_case")]
pub(crate) enum Pred {
    /// `col <op> lit`, already normalized so the column is on the left.
    Cmp { col: String, op: CmpOp, lit: Lit },
    /// Both sides must hold — a group survives only if it survives both.
    And { left: Box<Pred>, right: Box<Pred> },
    /// Either side may hold — a group survives if it survives either.
    Or { left: Box<Pred>, right: Box<Pred> },
    /// `col IS NULL` (`negated=false`) / `col IS NOT NULL` (`negated=true`).
    IsNull { col: String, negated: bool },
}

#[derive(Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub(crate) enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// A comparison literal. `untagged` so the JSON carries a bare `5` / `5.5` / `true` /
/// `"x"`; variant order matters (bool before int before float) so an integer never
/// deserializes as a float.
#[derive(Deserialize, Clone)]
#[serde(untagged)]
pub(crate) enum Lit {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

/// Parse the compact predicate JSON, or `None` if it is malformed (caller reads all
/// candidate row-groups — an unparseable predicate must never fail the scan).
pub(crate) fn parse(json: &str) -> Option<Pred> {
    serde_json::from_str(json).ok()
}

/// The subset of `candidates` whose statistics do not prove the predicate empty.
///
/// An out-of-range candidate index is *kept*, not indexed into: `meta.row_group(rg)`
/// panics on an out-of-bounds index, so probing one here would turn a bad split index
/// into a process-aborting panic on the FFI path — whereas the un-pruned decode path
/// reports the same index as a clean `Parquet error: row group N out of bounds`.
/// Predicate pushdown must never change *which* errors a read produces, only skip
/// provably-empty groups; so we defer the OOB index unchanged to the decoder.
pub(crate) fn surviving_row_groups(
    meta: &ParquetMetaData,
    pred: &Pred,
    candidates: &[usize],
) -> Vec<usize> {
    let n = meta.num_row_groups();
    candidates
        .iter()
        .copied()
        .filter(|&rg| rg >= n || rg_survives(meta.row_group(rg), pred))
        .collect()
}

fn rg_survives(rg: &RowGroupMetaData, pred: &Pred) -> bool {
    match pred {
        Pred::And { left, right } => rg_survives(rg, left) && rg_survives(rg, right),
        Pred::Or { left, right } => rg_survives(rg, left) || rg_survives(rg, right),
        Pred::IsNull { col, negated } => isnull_survives(rg, col, *negated),
        Pred::Cmp { col, op, lit } => cmp_survives(rg, col, *op, lit),
    }
}

/// The `Statistics` for the *top-level* column `col`, plus whether it is an *unsigned*
/// integer, if present as a flat leaf.
///
/// The pushed predicate only ever names a top-level column (`to_native_predicate` emits
/// bare column names), so the match must be against the column's *full* path being the
/// single part `col` — NOT its leaf name. Matching on the leaf alone let a nested field
/// (`s.a`) whose leaf collides with a top-level column (`a`) shadow it: the wrong
/// column's min/max was then used to prune, which silently dropped every matching row of
/// a file that happened to carry a like-named struct field. A predicate on a genuinely
/// nested column therefore finds no stats and (correctly, conservatively) keeps the group.
///
/// The unsigned flag is load-bearing: Parquet stores UINT_8/16/32/64 in a *signed*
/// physical `INT32`/`INT64`, with its min/max computed by *unsigned* order. A large
/// unsigned value (e.g. 3e9 in a `UInt32`) therefore surfaces as a negative `i32` stat, so
/// interpreting it as signed silently prunes away the rows that actually match. The caller
/// reinterprets the bits as unsigned when this is set.
fn col_stats<'a>(rg: &'a RowGroupMetaData, col: &str) -> Option<(&'a Statistics, bool)> {
    (0..rg.num_columns()).find_map(|i| {
        let cc = rg.column(i);
        let parts = cc.column_path().parts();
        if parts.len() == 1 && parts[0] == col {
            cc.statistics()
                .map(|s| (s, is_unsigned_int(cc.column_descr())))
        } else {
            None
        }
    })
}

/// Whether a column's logical/converted type is an *unsigned* integer, so its signed
/// physical min/max stats must be read back as unsigned before comparison.
pub(crate) fn is_unsigned_int(descr: &parquet::schema::types::ColumnDescriptor) -> bool {
    use parquet::basic::{ConvertedType, LogicalType};
    match descr.logical_type() {
        Some(LogicalType::Integer { is_signed, .. }) => !is_signed,
        // Pre-`LogicalType` files carry the same information in the deprecated converted type.
        _ => matches!(
            descr.converted_type(),
            ConvertedType::UINT_8
                | ConvertedType::UINT_16
                | ConvertedType::UINT_32
                | ConvertedType::UINT_64
        ),
    }
}

/// `IS [NOT] NULL` pruning: a group with a known null count can be skipped when it holds
/// no nulls (`IS NULL`) or only nulls (`IS NOT NULL`); an unknown count keeps the group.
fn isnull_survives(rg: &RowGroupMetaData, col: &str, negated: bool) -> bool {
    let Some((stats, _unsigned)) = col_stats(rg, col) else {
        return true;
    };
    let Some(nulls) = stats.null_count_opt() else {
        return true;
    };
    if negated {
        nulls < rg.num_rows() as u64 // IS NOT NULL: survives if any row is non-null
    } else {
        nulls > 0 // IS NULL: survives if any row is null
    }
}

fn cmp_survives(rg: &RowGroupMetaData, col: &str, op: CmpOp, lit: &Lit) -> bool {
    let Some((stats, unsigned)) = col_stats(rg, col) else {
        return true; // no stats for this column → cannot prune
    };
    match (stats, lit) {
        // Exact integer arithmetic for integer columns vs an integer literal. Comparisons
        // run in `i128` so an unsigned column's full range (up to `u64::MAX`) and a signed
        // `i64` literal both fit losslessly. Unsigned stats are reinterpreted from their
        // signed physical bits before widening (a large `UInt32` reads back as a negative
        // `i32`; taking it as signed would prune away the matching rows).
        (Statistics::Int32(s), Lit::Int(v)) => {
            let (mn, mx) = if unsigned {
                (i32_unsigned(s.min_opt()), i32_unsigned(s.max_opt()))
            } else {
                (
                    s.min_opt().map(|x| *x as i128),
                    s.max_opt().map(|x| *x as i128),
                )
            };
            range_survives(mn, mx, *v as i128, op)
        }
        (Statistics::Int64(s), Lit::Int(v)) => {
            let (mn, mx) = if unsigned {
                (i64_unsigned(s.min_opt()), i64_unsigned(s.max_opt()))
            } else {
                (
                    s.min_opt().map(|x| *x as i128),
                    s.max_opt().map(|x| *x as i128),
                )
            };
            range_survives(mn, mx, *v as i128, op)
        }
        // Float columns vs a float literal (the stored floats compare exactly).
        (Statistics::Float(s), Lit::Float(v)) => float_range_survives(
            s.min_opt().map(|x| *x as f64),
            s.max_opt().map(|x| *x as f64),
            *v,
            op,
        ),
        (Statistics::Double(s), Lit::Float(v)) => {
            float_range_survives(s.min_opt().copied(), s.max_opt().copied(), *v, op)
        }
        // Float column vs an integer literal: exact only when the int is representable
        // in f64 (|v| < 2^53); larger magnitudes keep the group (no lossy prune).
        (Statistics::Float(s), Lit::Int(v)) if int_exact_in_f64(*v) => float_range_survives(
            s.min_opt().map(|x| *x as f64),
            s.max_opt().map(|x| *x as f64),
            *v as f64,
            op,
        ),
        (Statistics::Double(s), Lit::Int(v)) if int_exact_in_f64(*v) => {
            float_range_survives(s.min_opt().copied(), s.max_opt().copied(), *v as f64, op)
        }
        (Statistics::Boolean(s), Lit::Bool(v)) => {
            range_survives(s.min_opt().copied(), s.max_opt().copied(), *v, op)
        }
        (Statistics::ByteArray(s), Lit::Str(v)) => range_survives(
            s.min_opt().map(|b| b.data().to_vec()),
            s.max_opt().map(|b| b.data().to_vec()),
            v.as_bytes().to_vec(),
            op,
        ),
        _ => true, // type mismatch / unhandled combination → conservatively keep
    }
}

/// Whether an integer is exactly representable as `f64` (no rounding at conversion).
fn int_exact_in_f64(v: i64) -> bool {
    v.unsigned_abs() < (1u64 << 53)
}

/// Reinterpret a signed `i32` physical stat as its unsigned value, widened to `i128`.
fn i32_unsigned(v: Option<&i32>) -> Option<i128> {
    v.map(|x| *x as u32 as i128)
}

/// Reinterpret a signed `i64` physical stat as its unsigned value, widened to `i128`.
fn i64_unsigned(v: Option<&i64>) -> Option<i128> {
    v.map(|x| *x as u64 as i128)
}

/// `range_survives` for float bounds, but a NaN bound *keeps* the group (never prunes).
///
/// Per the Parquet spec a writer must exclude NaN from float min/max, but writers have
/// violated this (parquet-mr < 1.10, PARQUET-1246, wrote NaN into double/float stats when
/// NaN was the first value in a page). A NaN bound compares false against every literal,
/// so feeding it to `range_survives` would prune the group for *any* ordering predicate
/// (`NaN > lit`, `NaN < lit` are both false) — silently dropping rows that actually match.
/// Since these readers ingest untrusted files from every writer, a NaN bound is treated as
/// "unknown" and the group is kept (superset-safe), matching how a missing stat is handled.
pub(crate) fn float_range_survives(
    min: Option<f64>,
    max: Option<f64>,
    lit: f64,
    op: CmpOp,
) -> bool {
    if min.is_some_and(f64::is_nan) || max.is_some_and(f64::is_nan) {
        return true;
    }
    range_survives(min, max, lit, op)
}

/// Can any value in `[min, max]` satisfy `value <op> lit`? Unknown bounds keep the group.
pub(crate) fn range_survives<T: PartialOrd>(
    min: Option<T>,
    max: Option<T>,
    lit: T,
    op: CmpOp,
) -> bool {
    match op {
        // `col == lit` is possible iff lit lies within [min, max].
        CmpOp::Eq => min.is_none_or(|mn| mn <= lit) && max.is_none_or(|mx| lit <= mx),
        // `col != lit` can only be pruned when every value equals lit (min == max == lit);
        // conservatively keep otherwise.
        CmpOp::Ne => !(min.is_some_and(|mn| mn == lit) && max.is_some_and(|mx| mx == lit)),
        // Some value `< lit` exists iff min < lit; `<= lit` iff min <= lit.
        CmpOp::Lt => min.is_none_or(|mn| mn < lit),
        CmpOp::Le => min.is_none_or(|mn| mn <= lit),
        // Some value `> lit` exists iff max > lit; `>= lit` iff max >= lit.
        CmpOp::Gt => max.is_none_or(|mx| mx > lit),
        CmpOp::Ge => max.is_none_or(|mx| mx >= lit),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_prunes_nothing_without_stats() {
        let p = parse(r#"{"node":"cmp","col":"a","op":"lt","lit":5}"#);
        assert!(matches!(p, Some(Pred::Cmp { .. })));
    }

    /// Every `node` tag the Python control plane emits must deserialize here.
    ///
    /// This is the wire contract with `python/batcher/io/predicate.py::to_native_predicate`,
    /// and it is the one contract whose breach is *silent*: `parse` returning `None` only
    /// costs pruning, never correctness, so a drifted tag produces right answers and zero
    /// row-groups skipped. It shipped that way — Python emitted `"null"` while this enum's
    /// `rename_all = "snake_case"` spells the variant `"is_null"`, so any predicate containing
    /// a null test silently pruned nothing. Pin all four tags, not just `cmp`.
    #[test]
    fn every_python_emitted_tag_parses() {
        assert!(matches!(
            parse(r#"{"node":"is_null","col":"a","negated":false}"#),
            Some(Pred::IsNull { negated: false, .. })
        ));
        assert!(matches!(
            parse(r#"{"node":"is_null","col":"a","negated":true}"#),
            Some(Pred::IsNull { negated: true, .. })
        ));
        assert!(matches!(
            parse(
                r#"{"node":"and","left":{"node":"cmp","col":"a","op":"gt","lit":5},
                       "right":{"node":"is_null","col":"b","negated":true}}"#
            ),
            Some(Pred::And { .. })
        ));
        assert!(matches!(
            parse(
                r#"{"node":"or","left":{"node":"cmp","col":"a","op":"le","lit":1.5},
                       "right":{"node":"cmp","col":"b","op":"eq","lit":"x"}}"#
            ),
            Some(Pred::Or { .. })
        ));
        // A conjunction is all-or-nothing: one unparseable arm drops the whole predicate,
        // which is why the drifted tag disabled pruning for the *entire* filter.
        assert!(parse(
            r#"{"node":"and","left":{"node":"cmp","col":"a","op":"gt","lit":5},
                          "right":{"node":"null","col":"b","negated":true}}"#
        )
        .is_none());
    }

    #[test]
    fn range_logic_matches_zone_map_semantics() {
        // Column values in [10, 20].
        let mn = Some(10i64);
        let mx = Some(20i64);
        // a < 5  → 10 < 5 false → prunable (no survive)
        assert!(!range_survives(mn, mx, 5, CmpOp::Lt));
        // a < 15 → 10 < 15 true → survive
        assert!(range_survives(mn, mx, 15, CmpOp::Lt));
        // a > 25 → 20 > 25 false → prune
        assert!(!range_survives(mn, mx, 25, CmpOp::Gt));
        // a == 12 → 12 in [10,20] → survive
        assert!(range_survives(mn, mx, 12, CmpOp::Eq));
        // a == 30 → out of range → prune
        assert!(!range_survives(mn, mx, 30, CmpOp::Eq));
        // a == 30 but unknown max → keep (cannot prune)
        assert!(range_survives(Some(10i64), None, 30, CmpOp::Eq));
        // a != 12 with a non-degenerate range → keep
        assert!(range_survives(mn, mx, 12, CmpOp::Ne));
        // a != 7 where the whole group is exactly 7 → prunable
        assert!(!range_survives(Some(7i64), Some(7i64), 7, CmpOp::Ne));
    }

    #[test]
    fn nan_float_bound_never_prunes() {
        // A writer that violated the spec and put NaN into float min/max (parquet-mr
        // < 1.10, PARQUET-1246) must NOT cause row-group pruning: NaN compares false
        // against every literal, so the raw `range_survives` would prune for any
        // ordering op and silently drop the group's real (non-NaN) matching rows.
        for op in [
            CmpOp::Eq,
            CmpOp::Ne,
            CmpOp::Lt,
            CmpOp::Le,
            CmpOp::Gt,
            CmpOp::Ge,
        ] {
            assert!(
                float_range_survives(Some(f64::NAN), Some(f64::NAN), 2.0, op),
                "NaN min/max must keep the group"
            );
            assert!(float_range_survives(Some(f64::NAN), Some(5.0), 2.0, op));
            assert!(float_range_survives(Some(1.0), Some(f64::NAN), 2.0, op));
        }
        // Proof the guard is load-bearing: the unguarded range logic *would* prune here.
        assert!(
            !range_survives(Some(f64::NAN), Some(f64::NAN), 2.0, CmpOp::Gt),
            "raw range_survives prunes on a NaN bound — the bug the guard prevents"
        );
        // A clean (non-NaN) float range still prunes exactly as before.
        assert!(!float_range_survives(Some(1.0), Some(2.0), 5.0, CmpOp::Gt));
        assert!(float_range_survives(Some(1.0), Some(2.0), 1.5, CmpOp::Gt));
    }

    #[test]
    fn f64_exactness_guard() {
        assert!(int_exact_in_f64(0));
        assert!(int_exact_in_f64((1i64 << 53) - 1));
        assert!(!int_exact_in_f64(1i64 << 53));
        assert!(!int_exact_in_f64(-(1i64 << 60)));
    }
}
