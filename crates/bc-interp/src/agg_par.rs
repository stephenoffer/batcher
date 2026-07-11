//! The high-cardinality parallel aggregate: partition first, aggregate once.
//!
//! The default parallel aggregate is `partial → combine → finalize`: hash each morsel into
//! its own group table, then merge the tables. That is the right shape when grouping
//! *reduces* — `GROUP BY l_returnflag` turns 16,384 rows into 3, and the merge is trivial.
//!
//! It is the wrong shape when it does not. `GROUP BY l_orderkey` over TPC-H `lineitem`
//! yields ~4 rows per group, so a 16,384-row morsel's partial has ~4,096 groups and the
//! merge inherits nearly the whole relation. `GROUP BY l_orderkey, l_linenumber` reduces
//! nothing at all: every partial row survives, the combine concatenates 60 M rows of keys
//! and states, hashes them, bins them, and gathers them again. Measured at sf10: the whole
//! per-morsel hash build (~60 M inserts) is thrown away, and `combine` costs ~35 ns per
//! *partial row* — 2.25 s for a group-by that DuckDB answers in 429 ms.
//!
//! When grouping does not reduce, the pre-aggregation is pure overhead. Partition the input
//! morsels by group key instead, and aggregate each partition exactly once. Equal keys
//! co-locate, so the partitions are key-disjoint and each one's partial is already final:
//! `combine` degenerates to `combine([p]) ≡ p` and the union of the partitions is the
//! answer. One hash build over the relation instead of two, one gather instead of three.
//!
//! This is not a second aggregation semantics — it is `partition → partial → finalize`,
//! the exact composition `bc_interp::dist` runs across machines, executed across cores.
//!
//! **Choosing between them is a runtime decision, not an estimate.** The optimizer's `ndv`
//! for a group key is a sketch, and after a filter it is a guess about a distribution
//! nobody has measured. So the executor measures: it partials a sample of morsels — work
//! the reducing path needs anyway — and reads the reduction those partials actually
//! achieved. Below [`REDUCTION_CEILING`] the sample says grouping reduces, and its partials
//! are handed straight back to the standard path, unwasted.
//!
//! Memory is the caller's call, not this module's. The partition path holds the gathered
//! relation where the reducing path can spill its partials through grace partitioning, so
//! `par` admits [`partition_footprint`] against the memory pool first and keeps the bounded
//! shape when the pool says no. Under pressure, bounded beats fast.

use arrow::array::RecordBatch;
use bc_expr::Expr;
use bc_ir::{AggregateItem, ProjectionItem};
use bc_runtime::agg;
use rayon::prelude::*;

use crate::error::InterpError;
use crate::ops::{self, AggJit};

/// Partial rows kept per input row, above which pre-aggregation is judged not to pay.
///
/// The two paths have different shapes, so the crossover is measurable rather than
/// arguable. `combine` costs roughly 35 ns per *partial* row, so the reducing path grows
/// linearly in this ratio; the partition path gathers the relation once and is flat in it.
/// Aggregating a 60 M-row table on 96 cores, over a synthetic key of varying cardinality
/// (milliseconds, lower is better):
///
/// | rows kept per input row | 0.012 | 0.049 | 0.100 | 0.182 | 0.342 | 0.683 | 0.999 |
/// |-------------------------|------:|------:|------:|------:|------:|------:|------:|
/// | partial -> combine      |  39.5 |  70.7 | 125.7 | 197.6 | 350.9 | 745.6 |1351.6 |
/// | partition -> aggregate  | 274.8 | 225.7 | 198.2 | 187.6 | 185.3 | 189.7 | 370.2 |
///
/// They cross just under 0.18. Rounding up to 0.20 keeps the reducing path wherever it is
/// clearly better and concedes only near-ties, where the two are within a few percent. (An
/// earlier guess of 0.66 would have left `GROUP BY l_orderkey` — 0.25 rows kept, four
/// lineitems per order — on the reducing path at nearly twice the cost.)
const REDUCTION_CEILING: f64 = 0.20;

/// Fraction of the input the sample may cost, and the floor below which a fraction is too
/// small to estimate a reduction from. Each morsel is an independent 16 k-row observation,
/// so a handful already pins the ratio; the divisor is what keeps the *discarded* work
/// bounded when the sample says "partition".
const SAMPLE_DIVISOR: usize = 16;
const MIN_SAMPLE_MORSELS: usize = 4;

/// Morsels sampled to measure the reduction. The sample is exactly the work the reducing
/// path would do first anyway, so a sample that says "reducing" costs nothing — but a
/// sample that says "partition" is thrown away, so it must stay a small *fraction of the
/// input*, not a fixed count.
///
/// One morsel per core (the previous rule) is ~1.5 M rows at 96 cores: 3 % of a 60 M-row
/// relation, but **31 %** of a 5 M-row one, where it dominated the aggregate — a 5 M-row
/// five-aggregate group-by spent 24 of its 35 ms here, partial-aggregating a third of the
/// input only to discard it. Never sample more morsels than there are cores, either: the
/// sample runs in one parallel pass.
fn sample_size(threads: usize, morsels: usize) -> usize {
    let cap = threads.min(morsels);
    let want = (morsels / SAMPLE_DIVISOR).max(MIN_SAMPLE_MORSELS);
    want.min(cap).max(1)
}

/// What the executor should do with the aggregate's input, having measured it.
pub(crate) enum AggPlan {
    /// Grouping does not reduce. Partition on these keys and aggregate each partition once
    /// — *if* the caller can hold the partitioned relation in memory. It is the caller that
    /// owns admission, so it may still decline; [`partials`] then computes the usual path.
    Partition(Vec<String>),
    /// Grouping reduces (or partitioning was declined outright). Every morsel's partial,
    /// for the caller's usual `combine → finalize`.
    Partials(Vec<agg::Partial>),
}

/// Decide the aggregate's shape by measuring what its group-by actually reduces.
///
/// `may_partition` is the caller's veto — see [`partitionable`]. When it is `None` this is
/// exactly the per-morsel partial map the executor has always run, with no sampling.
pub(crate) fn decide(
    morsels: &[RecordBatch],
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    jit: &AggJit,
    may_partition: Option<&[String]>,
) -> Result<AggPlan, InterpError> {
    let Some(keys) = may_partition else {
        return Ok(AggPlan::Partials(partials(
            morsels, group_keys, aggregates, jit,
        )?));
    };

    // Sample: partial the first `n` morsels and read the reduction they achieved.
    let n = sample_size(rayon::current_num_threads(), morsels.len());
    let sampled = partials(&morsels[..n], group_keys, aggregates, jit)?;
    let rows_in: usize = morsels[..n].iter().map(|b| b.num_rows()).sum();
    let rows_out: usize = sampled
        .iter()
        .map(|p| p.group_columns.first().map_or(0, |c| c.len()))
        .sum();
    if rows_in > 0 && (rows_out as f64 / rows_in as f64) >= REDUCTION_CEILING {
        return Ok(AggPlan::Partition(keys.to_vec()));
    }

    // Reducing: the sample is the first slice of the work, so keep it and do the rest.
    let mut all = sampled;
    all.par_extend(partials(&morsels[n..], group_keys, aggregates, jit)?.into_par_iter());
    Ok(AggPlan::Partials(all))
}

/// One partial per morsel — the reducing path's first step, and the fallback when the
/// caller declines a [`AggPlan::Partition`] it cannot fit in memory.
pub(crate) fn partials(
    morsels: &[RecordBatch],
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    jit: &AggJit,
) -> Result<Vec<agg::Partial>, InterpError> {
    morsels
        .par_iter()
        .map(|b| ops::eval_partial_jit(b, group_keys, aggregates, jit))
        .collect()
}

/// Peak bytes the partition path holds: the gathered, partitioned relation (~1× the input)
/// **plus** the source morsels it was gathered from, which stay live until the gather
/// completes — so the true working set is ~2× the input, not 1×. The caller admits this
/// against the memory pool before committing (and records it as the operator's peak), so a
/// 1× estimate systematically under-admitted the non-reducing group-by this path exists for
/// and could OOM where the reducing/spilling path would have stayed bounded.
pub(crate) fn partition_footprint(input_bytes: u64) -> usize {
    input_bytes.saturating_mul(2) as usize
}

/// The group keys to partition on, or `None` when this aggregate must keep the reducing path.
///
/// Declined when the keys are computed rather than plain columns (nothing to route on),
/// when there is a single morsel (partitioning one hash table costs and saves nothing), or
/// when there are no group keys (a global aggregate has one group by definition). Whether
/// the *memory* is there is a separate question, and the caller's: see [`partition_footprint`].
pub(crate) fn partitionable(
    group_keys: &[ProjectionItem],
    morsels: &[RecordBatch],
) -> Option<Vec<String>> {
    if group_keys.is_empty() || morsels.len() < 2 {
        return None;
    }
    plain_key_columns(group_keys)
}

/// The group keys as plain input-column names, or `None` if any key is computed.
///
/// Partitioning routes rows by the *values* of columns present in the morsel. A computed
/// key (`GROUP BY x + 1`, `GROUP BY date_trunc(...)`) has no such column, so that plan
/// keeps the reducing path rather than growing a second expression-evaluation site here.
pub(crate) fn plain_key_columns(group_keys: &[ProjectionItem]) -> Option<Vec<String>> {
    group_keys
        .iter()
        .map(|k| match &k.expr {
            Expr::Col { name } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

/// Aggregate `morsels` by partitioning on `keys`, then grouping each partition once.
///
/// Each partition holds every row for the keys that hashed to it and no row for any other,
/// so its partial is final: `finalize` applies directly and the partitions' output batches
/// concatenate to the whole result. Group order is unspecified for a hash aggregate — the
/// standard path's order already depends on the worker count — so callers compare these as
/// multisets, exactly as they do today.
pub(crate) fn partitioned_aggregate(
    morsels: &[RecordBatch],
    keys: &[String],
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    jit: &AggJit,
    funcs: &[agg::AggFunc],
    partitions: usize,
) -> Result<Vec<RecordBatch>, InterpError> {
    let buckets = ops::partition_morsels(morsels, keys, partitions)?;
    buckets
        .par_iter()
        .filter(|b| b.num_rows() > 0)
        .map(|bucket| {
            let partial = ops::eval_partial_jit(bucket, group_keys, aggregates, jit)?;
            let agg_columns = agg::finalize(funcs, &partial)?;
            ops::build_agg_batch(group_keys, aggregates, &partial.group_columns, &agg_columns)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sample must stay a small fraction of the input, because the partition path
    /// discards it. One morsel per core made a 306-morsel (5 M-row) aggregate sample 96
    /// morsels — a third of the data.
    #[test]
    fn sample_is_a_bounded_fraction_of_the_input() {
        assert_eq!(sample_size(96, 306), 19); // 6% of the morsels, not 31%
        assert_eq!(sample_size(96, 1600), 96); // capped by cores, never above them
        assert_eq!(sample_size(96, 64), 4); // small input: the floor, not 64
        assert_eq!(sample_size(96, 2), 2); // fewer morsels than the floor
        assert_eq!(sample_size(1, 306), 1); // single core
        assert!((1..=306).contains(&sample_size(8, 306)));
    }

    /// Never sample more morsels than exist, nor zero of them.
    #[test]
    fn sample_size_is_in_range() {
        for threads in [1usize, 2, 8, 96] {
            for morsels in [1usize, 2, 3, 10, 100, 306, 10_000] {
                let n = sample_size(threads, morsels);
                assert!(
                    n >= 1 && n <= morsels,
                    "threads={threads} morsels={morsels} n={n}"
                );
                assert!(n <= threads.max(1), "sample exceeds cores");
            }
        }
    }
    use arrow::array::{ArrayRef, Int64Array};
    use bc_ir::AggFunc;
    use std::sync::Arc;

    fn morsel(keys: &[i64], vals: &[i64]) -> RecordBatch {
        RecordBatch::try_from_iter(vec![
            ("k", Arc::new(Int64Array::from(keys.to_vec())) as ArrayRef),
            ("v", Arc::new(Int64Array::from(vals.to_vec())) as ArrayRef),
        ])
        .unwrap()
    }

    fn group_keys() -> Vec<ProjectionItem> {
        vec![ProjectionItem {
            expr: Expr::Col { name: "k".into() },
            alias: "k".into(),
        }]
    }

    fn aggregates() -> Vec<AggregateItem> {
        vec![AggregateItem {
            func: AggFunc::Sum,
            input: Some(Expr::Col { name: "v".into() }),
            input2: None,
            alias: "s".into(),
            param: None,
        }]
    }

    /// Collect `(key, sum)` pairs from output batches, sorted — hash-group order is
    /// unspecified, so every assertion here compares multisets.
    fn pairs(batches: &[RecordBatch]) -> Vec<(i64, i64)> {
        let mut out = Vec::new();
        for b in batches {
            let k = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let s = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..b.num_rows() {
                out.push((k.value(i), s.value(i)));
            }
        }
        out.sort();
        out
    }

    fn run(morsels: &[RecordBatch], partitions: usize) -> Vec<RecordBatch> {
        let (gk, aggs) = (group_keys(), aggregates());
        let jit = ops::compile_agg(&gk, &aggs, &morsels[0]);
        let funcs = ops::agg_funcs(&aggs);
        partitioned_aggregate(morsels, &["k".into()], &gk, &aggs, &jit, &funcs, partitions).unwrap()
    }

    /// The invariant: partitioning then aggregating equals aggregating the whole relation.
    /// A key split across morsels must still land in exactly one partition and sum once.
    #[test]
    fn partitioned_aggregate_matches_the_whole_relation() {
        let morsels = [
            morsel(&[1, 2, 3, 1], &[10, 20, 30, 1]),
            morsel(&[2, 3, 1], &[2, 3, 100]),
            morsel(&[4], &[7]),
        ];
        assert_eq!(
            pairs(&run(&morsels, 8)),
            vec![(1, 111), (2, 22), (3, 33), (4, 7)]
        );
    }

    /// The partition count is a scheduling choice, never a semantic one.
    #[test]
    fn the_result_is_independent_of_the_partition_count() {
        let morsels = [
            morsel(&[1, 2, 1, 3], &[1, 2, 3, 4]),
            morsel(&[2, 1], &[5, 6]),
        ];
        let expect = vec![(1, 10), (2, 7), (3, 4)];
        for p in [1, 2, 3, 7, 16, 64] {
            assert_eq!(pairs(&run(&morsels, p)), expect, "partitions={p}");
        }
    }

    /// An all-distinct key is the case this path exists for: no reduction, every row a group.
    #[test]
    fn an_all_distinct_key_yields_one_group_per_row() {
        let morsels = [morsel(&[1, 2, 3], &[1, 1, 1]), morsel(&[4, 5], &[1, 1])];
        assert_eq!(
            pairs(&run(&morsels, 4)),
            vec![(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
        );
    }

    /// A single group is the opposite extreme, and must still be correct if it gets here.
    #[test]
    fn a_single_group_collapses_to_one_row() {
        let morsels = [morsel(&[7, 7], &[1, 2]), morsel(&[7], &[3])];
        assert_eq!(pairs(&run(&morsels, 8)), vec![(7, 6)]);
    }

    fn plan(morsels: &[RecordBatch], may_partition: bool) -> AggPlan {
        let (gk, aggs) = (group_keys(), aggregates());
        let jit = ops::compile_agg(&gk, &aggs, &morsels[0]);
        let keys = vec!["k".to_string()];
        decide(
            morsels,
            &gk,
            &aggs,
            &jit,
            may_partition.then_some(keys.as_slice()),
        )
        .unwrap()
    }

    /// A reducing group-by keeps the partial/combine path — and every morsel is partialled
    /// exactly once, the sampled ones included. Ten rows to a group is well under
    /// `REDUCTION_CEILING`; a two-row group would sit above it and rightly partition.
    #[test]
    fn a_reducing_group_by_keeps_the_partial_path_and_wastes_no_sample() {
        let ones = [1i64; 10];
        let morsels = [morsel(&ones, &ones), morsel(&ones, &ones)];
        match plan(&morsels, true) {
            AggPlan::Partials(p) => assert_eq!(p.len(), morsels.len()),
            AggPlan::Partition(_) => panic!("4 rows into 1 group must read as reducing"),
        }
    }

    /// An all-distinct key is routed to the partition path.
    #[test]
    fn an_all_distinct_group_by_partitions() {
        let morsels = [morsel(&[1, 2], &[1, 1]), morsel(&[3, 4], &[1, 1])];
        match plan(&morsels, true) {
            AggPlan::Partition(keys) => assert_eq!(keys, vec!["k".to_string()]),
            AggPlan::Partials(_) => panic!("an all-distinct key does not reduce"),
        }
    }

    /// The caller's veto is absolute: no partitioning, whatever the measurement says.
    #[test]
    fn a_vetoed_aggregate_never_partitions() {
        let morsels = [morsel(&[1, 2], &[1, 1]), morsel(&[3, 4], &[1, 1])];
        assert!(matches!(plan(&morsels, false), AggPlan::Partials(_)));
    }

    /// Both paths must agree — that is the whole contract. Same input, same answer.
    #[test]
    fn the_two_paths_agree_on_the_same_input() {
        let morsels = [
            morsel(&[1, 2, 3, 1], &[10, 20, 30, 1]),
            morsel(&[2, 3, 1], &[2, 3, 100]),
        ];
        let (gk, aggs) = (group_keys(), aggregates());
        let funcs = ops::agg_funcs(&aggs);

        let AggPlan::Partials(ps) = plan(&morsels, false) else {
            unreachable!()
        };
        let merged = agg::combine(&ps, &funcs).unwrap();
        let cols = agg::finalize(&funcs, &merged).unwrap();
        let combined = ops::build_agg_batch(&gk, &aggs, &merged.group_columns, &cols).unwrap();

        assert_eq!(
            pairs(std::slice::from_ref(&combined)),
            pairs(&run(&morsels, 8))
        );
    }

    /// A global aggregate and a lone morsel have nothing to partition.
    #[test]
    fn partitionable_declines_the_shapes_it_must() {
        let one = [morsel(&[1, 2], &[1, 1])];
        let two = [morsel(&[1], &[1]), morsel(&[2], &[1])];
        assert_eq!(partitionable(&group_keys(), &two), Some(vec!["k".into()]));
        assert_eq!(partitionable(&[], &two), None, "global aggregate");
        assert_eq!(partitionable(&group_keys(), &one), None, "single morsel");
    }

    /// A computed key has no column to route on, so the partition path must decline it.
    #[test]
    fn a_computed_group_key_is_not_partitionable() {
        assert_eq!(
            plain_key_columns(&group_keys()),
            Some(vec!["k".to_string()])
        );
        let computed = vec![ProjectionItem {
            expr: Expr::Binary {
                op: bc_expr::BinaryOp::Add,
                left: Box::new(Expr::Col { name: "k".into() }),
                right: Box::new(Expr::Lit {
                    value: bc_expr::Literal::Int(1),
                }),
            },
            alias: "k1".into(),
        }];
        assert_eq!(plain_key_columns(&computed), None);
    }
}
