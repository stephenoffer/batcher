//! Bloom-filter pruning: skip a row group whose bloom proves an equality cannot match.
//!
//! Range-based pruning — footer statistics and the page index — is only as good as the
//! clustering of the data. On a **high-cardinality, unordered** column every page's
//! `[min, max]` spans nearly the whole domain, so an equality predicate prunes nothing:
//! measured on 2M rows of random keys in one row group, `k == <value>` kept 2,000,000 rows
//! after page pruning, a 1.00x reduction. That is exactly the shape a bloom filter answers,
//! and it is the common shape for join keys, user ids, device ids and hashes.
//!
//! Batcher's own writer does not emit blooms (the pyarrow version pinned here has no option
//! for it), but Spark, Databricks and DuckDB-written lakes routinely do — so this is about
//! reading the ecosystem's files well, not our own.
//!
//! ## Why this is safe
//!
//! A bloom filter has **no false negatives**: `check(v) == false` means `v` is definitely
//! absent from the row group. A false *positive* merely keeps a row group that holds no
//! match, which costs a decode and is then removed by the engine's `Filter`. So pruning on
//! a negative verdict cannot lose a row.
//!
//! ## The one way this could lose rows
//!
//! The bloom hashes the column's **physical** bytes. Checking an `i64` against an `INT32`
//! column's bloom hashes different bytes, so it can answer "absent" for a value that is
//! present — a wrong answer, silently. Every check here therefore requires the literal's
//! type to match the column's physical type *exactly*, and anything else declines to prune.
//! `Utf8`/`ByteArray` is matched the same way. This is the reason `eq_term` carries the
//! physical type through rather than checking on the literal alone.

use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::basic::Type as PhysicalType;
use parquet::file::metadata::ParquetMetaData;

use crate::predicate::{CmpOp, Lit, Pred};

/// One `col == lit` term that a bloom filter could decide, with its leaf column resolved.
struct EqTerm<'a> {
    column: usize,
    physical: PhysicalType,
    lit: &'a Lit,
}

/// Whether `pred` is provably unsatisfiable in row group `rg`, per the blooms.
///
/// Fetches at most one bloom per equality column, and only for predicates whose shape makes
/// a negative verdict conclusive. Any error, missing bloom, or type mismatch answers
/// `false` — "not proven absent" — so the row group is read exactly as it would have been.
pub(crate) async fn provably_absent<T>(
    builder: &mut ParquetRecordBatchStreamBuilder<T>,
    meta: &ParquetMetaData,
    pred: &Pred,
    rg: usize,
) -> bool
where
    T: AsyncFileReader + Send + 'static,
{
    // Nothing to fetch for a predicate with no decidable equality — the common case, and it
    // must not cost a round trip.
    if !has_eq(pred) {
        return false;
    }
    absent(builder, meta, pred, rg).await
}

/// A cheap syntactic pre-check so a predicate with no `Eq` never triggers I/O.
fn has_eq(pred: &Pred) -> bool {
    match pred {
        Pred::Cmp { op: CmpOp::Eq, .. } => true,
        Pred::And { left, right } | Pred::Or { left, right } => has_eq(left) || has_eq(right),
        _ => false,
    }
}

/// The pruning lattice, dual to `page_index`'s.
///
/// Here the value computed is "provably empty", so the combinators invert relative to
/// "survives": a conjunction is empty if **either** side is empty, a disjunction only if
/// **both** are. Anything not decidable is `false`, which keeps the row group.
async fn absent<T>(
    builder: &mut ParquetRecordBatchStreamBuilder<T>,
    meta: &ParquetMetaData,
    pred: &Pred,
    rg: usize,
) -> bool
where
    T: AsyncFileReader + Send + 'static,
{
    match pred {
        Pred::Cmp {
            col,
            op: CmpOp::Eq,
            lit,
        } => match eq_term(meta, rg, col, lit) {
            Some(term) => check_absent(builder, rg, &term).await,
            None => false,
        },
        Pred::And { left, right } => {
            // Either side proving emptiness proves the conjunction empty. Short-circuit so a
            // decided left side does not pay for the right side's fetch.
            Box::pin(absent(builder, meta, left, rg)).await
                || Box::pin(absent(builder, meta, right, rg)).await
        }
        Pred::Or { left, right } => {
            Box::pin(absent(builder, meta, left, rg)).await
                && Box::pin(absent(builder, meta, right, rg)).await
        }
        _ => false,
    }
}

/// Resolve `col` to a leaf index and physical type, if it is a top-level flat column.
///
/// Matches the FULL column path being the single part `col`, exactly as `predicate` and
/// `page_index` do — a nested field whose leaf name collides with a top-level column must
/// never be used to prune the top-level one.
fn eq_term<'a>(meta: &ParquetMetaData, rg: usize, col: &str, lit: &'a Lit) -> Option<EqTerm<'a>> {
    let group = meta.row_group(rg);
    (0..group.num_columns()).find_map(|i| {
        let descr = group.column(i).column_descr();
        let parts = descr.path().parts();
        (parts.len() == 1 && parts[0] == col).then(|| EqTerm {
            column: i,
            physical: descr.physical_type(),
            lit,
        })
    })
}

/// Fetch the column's bloom and ask it about the literal.
///
/// Returns `true` only on a definite "not present" from a bloom whose column's physical type
/// matches the literal exactly. Every other outcome — no bloom, a fetch error, a type
/// mismatch — returns `false` and the row group is read.
async fn check_absent<T>(
    builder: &mut ParquetRecordBatchStreamBuilder<T>,
    rg: usize,
    term: &EqTerm<'_>,
) -> bool
where
    T: AsyncFileReader + Send + 'static,
{
    let Ok(Some(sbbf)) = builder
        .get_row_group_column_bloom_filter(rg, term.column)
        .await
    else {
        return false;
    };
    // The physical type gate. Hashing an `i64` against an `INT32` bloom hashes different
    // bytes and can answer "absent" for a value that is present — the one way this feature
    // returns a wrong answer rather than merely doing extra work.
    match (term.physical, term.lit) {
        (PhysicalType::INT32, Lit::Int(v)) => match i32::try_from(*v) {
            // A literal outside i32 cannot be equal to any INT32 value, but proving that is
            // the range pruner's job; here it simply is not a valid probe.
            Ok(narrow) => !sbbf.check(&narrow),
            Err(_) => false,
        },
        (PhysicalType::INT64, Lit::Int(v)) => !sbbf.check(v),
        (PhysicalType::FLOAT, Lit::Float(v)) => !sbbf.check(&(*v as f32)),
        (PhysicalType::DOUBLE, Lit::Float(v)) => !sbbf.check(v),
        (PhysicalType::BYTE_ARRAY, Lit::Str(v)) => !sbbf.check(&v.as_str()),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate;

    #[test]
    fn a_predicate_without_equality_never_triggers_a_fetch() {
        // The guard that keeps this free for the overwhelmingly common predicate shapes.
        for json in [
            r#"{"node":"cmp","col":"a","op":"lt","lit":5}"#,
            r#"{"node":"is_null","col":"a","negated":false}"#,
            r#"{"node":"and","left":{"node":"cmp","col":"a","op":"gt","lit":1},"right":{"node":"cmp","col":"a","op":"lt","lit":9}}"#,
        ] {
            assert!(!has_eq(&predicate::parse(json).unwrap()), "{json}");
        }
    }

    #[test]
    fn an_equality_anywhere_in_the_tree_is_found() {
        for json in [
            r#"{"node":"cmp","col":"a","op":"eq","lit":5}"#,
            r#"{"node":"and","left":{"node":"cmp","col":"a","op":"lt","lit":9},"right":{"node":"cmp","col":"b","op":"eq","lit":3}}"#,
            r#"{"node":"or","left":{"node":"cmp","col":"a","op":"eq","lit":1},"right":{"node":"cmp","col":"b","op":"gt","lit":3}}"#,
        ] {
            assert!(has_eq(&predicate::parse(json).unwrap()), "{json}");
        }
    }
}
