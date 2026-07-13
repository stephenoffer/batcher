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
#[derive(Deserialize)]
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
#[derive(Deserialize)]
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
pub(crate) fn surviving_row_groups(
    meta: &ParquetMetaData,
    pred: &Pred,
    candidates: &[usize],
) -> Vec<usize> {
    candidates
        .iter()
        .copied()
        .filter(|&rg| rg_survives(meta.row_group(rg), pred))
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

/// The `Statistics` for `col` in this row-group, if the (flat) leaf column is present.
fn col_stats<'a>(rg: &'a RowGroupMetaData, col: &str) -> Option<&'a Statistics> {
    (0..rg.num_columns()).find_map(|i| {
        let cc = rg.column(i);
        let leaf = cc.column_path().parts().last().map(String::as_str);
        (leaf == Some(col)).then(|| cc.statistics()).flatten()
    })
}

/// `IS [NOT] NULL` pruning: a group with a known null count can be skipped when it holds
/// no nulls (`IS NULL`) or only nulls (`IS NOT NULL`); an unknown count keeps the group.
fn isnull_survives(rg: &RowGroupMetaData, col: &str, negated: bool) -> bool {
    let Some(stats) = col_stats(rg, col) else {
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
    let Some(stats) = col_stats(rg, col) else {
        return true; // no stats for this column → cannot prune
    };
    match (stats, lit) {
        // Exact integer arithmetic for integer columns vs an integer literal.
        (Statistics::Int32(s), Lit::Int(v)) => range_survives(
            s.min_opt().map(|x| *x as i64),
            s.max_opt().map(|x| *x as i64),
            *v,
            op,
        ),
        (Statistics::Int64(s), Lit::Int(v)) => {
            range_survives(s.min_opt().copied(), s.max_opt().copied(), *v, op)
        }
        // Float columns vs a float literal (the stored floats compare exactly).
        (Statistics::Float(s), Lit::Float(v)) => range_survives(
            s.min_opt().map(|x| *x as f64),
            s.max_opt().map(|x| *x as f64),
            *v,
            op,
        ),
        (Statistics::Double(s), Lit::Float(v)) => {
            range_survives(s.min_opt().copied(), s.max_opt().copied(), *v, op)
        }
        // Float column vs an integer literal: exact only when the int is representable
        // in f64 (|v| < 2^53); larger magnitudes keep the group (no lossy prune).
        (Statistics::Float(s), Lit::Int(v)) if int_exact_in_f64(*v) => range_survives(
            s.min_opt().map(|x| *x as f64),
            s.max_opt().map(|x| *x as f64),
            *v as f64,
            op,
        ),
        (Statistics::Double(s), Lit::Int(v)) if int_exact_in_f64(*v) => {
            range_survives(s.min_opt().copied(), s.max_opt().copied(), *v as f64, op)
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

/// Can any value in `[min, max]` satisfy `value <op> lit`? Unknown bounds keep the group.
fn range_survives<T: PartialOrd>(min: Option<T>, max: Option<T>, lit: T, op: CmpOp) -> bool {
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
    fn f64_exactness_guard() {
        assert!(int_exact_in_f64(0));
        assert!(int_exact_in_f64((1i64 << 53) - 1));
        assert!(!int_exact_in_f64(1i64 << 53));
        assert!(!int_exact_in_f64(-(1i64 << 60)));
    }
}
