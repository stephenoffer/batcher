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
pub(crate) fn sample_size(threads: usize, morsels: usize) -> usize {
    let cap = threads.min(morsels);
    let want = (morsels / SAMPLE_DIVISOR).max(MIN_SAMPLE_MORSELS);
    want.min(cap).max(1)
}

/// What the executor should do with the aggregate's input, having measured it.
pub(crate) enum AggPlan {
    /// Grouping does not reduce. Partition on these keys, `width` ways, and aggregate each
    /// partition once — *if* the caller can hold the partitioned relation in memory. It is
    /// the caller that owns admission, so it may still decline; [`partials`] then computes
    /// the usual path.
    Partition {
        keys: Vec<String>,
        width: usize,
        /// The sample's group-count estimate, carried so a caller that *declines* the
        /// partitioned shape can still size the reducing path's regroup from it.
        groups: usize,
    },
    /// Grouping reduces (or partitioning was declined outright). Every morsel's partial,
    /// for the caller's usual `combine → finalize`, and the group count the sample estimates
    /// the merge will produce — which is what sizes the regroup's width
    /// (`agg::combine_sized`). `0` when nothing was sampled, meaning "not measured".
    Partials {
        partials: Vec<agg::Partial>,
        groups: usize,
    },
}

/// Groups one partition may hold before its hash table stops being cache-resident — the
/// number [`radix_width`] divides the estimated group count by.
///
/// The partition path's cost splits in two: a *split* (hash + gather) that grows with the
/// partition count, and an *aggregate* whose speed is decided by whether a partition's hash
/// table stays in a core's private cache. One partition per core sizes that table by the
/// relation's *whole* cardinality, so past some group count every probe misses and the
/// aggregate step falls off a cliff. Splitting finer is what moves it back.
///
/// Where that cliff is, is the whole question. Measured single-process on a 16-core box, one
/// `SUM` over an `Int64` key, min-of-2, aggregate step only (milliseconds):
///
/// | groups | rows | 16 parts | 32 | 64 | 128 | 256 | 1024 |
/// |--------|------|---------:|---:|---:|----:|----:|-----:|
/// | 100 k  | 8 M  |   8.8 |   — |  8.4 |   — |  7.0 | 16.8 |
/// | 200 k  | 4 M  |  26.0 | 33.0| 29.1| 59.9| 46.3| 58.0 |
/// | 1 M    | 8 M  |  24.2 |   — | 28.8|   — | 21.5 | 17.2 |
/// | 1.7 M  | 4 M  | 125.4 | 72.0| 74.0| 62.0| 48.9| 56.1 |
/// | 3.1 M  | 4 M  | 141.0 | 81.0| 73.0| 52.3| 51.6| 60.2 |
/// | 3.5 M  | 8 M  |  91.9 |   — | 42.1|   — | 23.6 | 19.5 |
/// | 6.3 M  | 8 M  | 126.1 |   — | 41.6|   — | 29.0 | 21.7 |
///
/// Read it by *groups per partition*, which is what the cache sees. At 200 k groups over 16
/// partitions each holds 12.5 k and one-per-core is already the best row — splitting further
/// only pays for the wider split. At 1.7 M over 16 each holds 108 k, and splitting is worth
/// 2.6x. The turn is therefore somewhere between 12 k and 108 k groups per partition, and
/// 32,768 sits inside it: it leaves every cardinality up to ~500 k on a 16-core box at
/// exactly the width it had before, and widens only where the measurements are large,
/// monotone, and repeated across two sessions.
///
/// The conservatism is deliberate and was earned. An earlier value of 8,192 widened 200 k
/// groups from 16 partitions to 32 and showed up as a *regression* end to end. This box runs
/// several agent sessions at once — the rows above were taken at load averages between 10
/// and 30 on 16 cores — so a difference under ~30 % here is not resolvable, and a divisor
/// tuned to one is fitting noise. Only the cliff is big enough to be real; aim at that.
const GROUPS_PER_PARTITION: usize = 32_768;

/// Ceiling on the partition count. Past this the split's per-partition setup dominates at
/// every measured cardinality, and the output batches fall below a useful morsel size.
const MAX_PARTITIONS: usize = 2_048;

/// How many partitions to split `estimated_groups` across on `threads` cores.
///
/// Never below one per core (the pool must fill) and never above [`MAX_PARTITIONS`]. This is
/// a performance choice only: the partitions are key-disjoint at any width, so the relation
/// they union to is the same one.
pub(crate) fn radix_width(estimated_groups: usize, threads: usize) -> usize {
    estimated_groups
        .div_ceil(GROUPS_PER_PARTITION)
        .clamp(threads.max(1), MAX_PARTITIONS)
        .next_power_of_two()
        .min(MAX_PARTITIONS)
}

/// Estimate the whole relation's group count from what a sample of morsels grouped to.
///
/// A morsel of `m` rows drawn from a key domain of `d` distinct values yields
/// `m · (1 - e^(-m/d)) / (m/d)` distinct keys — the coupon-collector curve. The sample gives
/// the left-hand side (`sample_groups / sample_rows`, the average per-morsel reduction), so
/// inverting it recovers `d`, and the same curve then projects `d` forward to the group count
/// over all `total_rows`.
///
/// This is the one number the executor cannot get from the optimizer: `ndv` for a group key
/// is a sketch taken before any filter ran, and the partition path is reached precisely when
/// grouping is *not* reducing, which is where a stale sketch is least trustworthy. Measured
/// against uniform keys at 8 M rows it recovers the true group count to within 1 %
/// (3.46 M estimated against 3,458,758 actual; 6.35 M against 6,294,849).
///
/// It is only ever a width, so a skewed or clustered key that breaks the uniformity
/// assumption costs a partition count off by a small factor — which the flat region of the
/// table in [`GROUPS_PER_PARTITION`] absorbs.
///
/// The result is clamped into the range the answer provably lies in whatever the
/// distribution: at least the *average* per-morsel group count (some morsel saw at least
/// that many distinct keys, and the relation has at least as many as any one morsel), and at
/// most one group per row. Note the floor is the average and **not** `sample_groups`, which
/// sums the per-morsel counts and so counts a key once per morsel it appears in — bounding
/// below by that sum reported 242 k groups for a 100 k-group key, over-partitioning by 2.4x.
fn estimated_groups(
    sample_rows: usize,
    sample_groups: usize,
    sample_morsels: usize,
    total_rows: usize,
) -> usize {
    if sample_rows == 0 || sample_groups == 0 || sample_morsels == 0 {
        return 0;
    }
    let ratio = sample_groups as f64 / sample_rows as f64;
    // `(1 - e^-x) / x` falls monotonically from 1 (at x -> 0) toward 0, so bisection on it is
    // exact to the bit in a fixed number of steps. The bracket's low end pins the
    // every-row-distinct case, where the curve is flat and `d` is unbounded; the clamp below
    // turns that into "at most one group per row", which is the true bound.
    let (mut lo, mut hi) = (1e-9_f64, 64.0_f64);
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if (1.0 - (-mid).exp()) / mid > ratio {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let domain = (sample_rows as f64 / sample_morsels as f64) / lo;
    let seen = domain * (1.0 - (-(total_rows as f64) / domain).exp());
    if !seen.is_finite() {
        return total_rows;
    }
    let floor = (sample_groups / sample_morsels).max(1);
    (seen as usize).clamp(floor.min(total_rows), total_rows)
}

/// Read a sample's partials and say how wide to partition — or `None` to keep the reducing
/// path.
///
/// The **one** definition of the shape rule, so the plain aggregate and the fused one cannot
/// drift apart on it. `sample_rows`/`sample_morsels` describe the rows the sample covered,
/// `total_rows` the whole relation.
pub(crate) fn width_from_sample(
    sample: &[agg::Partial],
    sample_rows: usize,
    sample_morsels: usize,
    total_rows: usize,
) -> Option<usize> {
    let rows_out: usize = sample
        .iter()
        .map(|p| p.group_columns.first().map_or(0, |c| c.len()))
        .sum();
    if sample_rows == 0 || (rows_out as f64 / sample_rows as f64) < REDUCTION_CEILING {
        return None;
    }
    // The same sample that chose the shape also sizes it: how many groups the whole
    // relation holds decides how finely to split it. See [`GROUPS_PER_PARTITION`].
    let groups = estimated_groups(sample_rows, rows_out, sample_morsels, total_rows);
    Some(radix_width(groups, rayon::current_num_threads()))
}

/// The group count a sample says the whole relation will produce.
///
/// The *reducing* path's use of the same measurement [`width_from_sample`] makes for the
/// partition path: the merge cannot know its own output size, so the sample tells it, and
/// `agg::combine_sized` sizes the regroup's width from it. Unconditional — grouping that
/// reduces still needs a width — where `width_from_sample` answers only when it does not.
pub(crate) fn groups_from_sample(
    sample: &[agg::Partial],
    sample_rows: usize,
    sample_morsels: usize,
    total_rows: usize,
) -> usize {
    let rows_out: usize = sample
        .iter()
        .map(|p| p.group_columns.first().map_or(0, |c| c.len()))
        .sum();
    estimated_groups(sample_rows, rows_out, sample_morsels, total_rows)
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
        return Ok(AggPlan::Partials {
            partials: partials(morsels, group_keys, aggregates, jit)?,
            groups: 0,
        });
    };

    // Sample: partial the first `n` morsels and read the reduction they achieved.
    let threads = rayon::current_num_threads().max(1);
    let n = sample_size(threads, morsels.len());
    let sampled = partials(&morsels[..n], group_keys, aggregates, jit)?;
    let rows_in: usize = morsels[..n].iter().map(|b| b.num_rows()).sum();
    let total_rows: usize = morsels.iter().map(|b| b.num_rows()).sum();
    if let Some(width) = width_from_sample(&sampled, rows_in, n, total_rows) {
        let groups = groups_from_sample(&sampled, rows_in, n, total_rows);
        // The sample says a *morsel* does not reduce — but a morsel is 16,384 rows, and a
        // group count well under that reduces enormously over a whole worker's share. Both
        // readings are right and they choose different shapes, so the group count decides
        // between them. See [`chunked_partials`].
        if chunking_pays(threads, groups, total_rows) {
            return Ok(AggPlan::Partials {
                partials: chunked_partials(morsels, group_keys, aggregates, jit, threads)?,
                groups,
            });
        }
        return Ok(AggPlan::Partition {
            keys: keys.to_vec(),
            width,
            groups,
        });
    }

    // Reducing: the sample is the first slice of the work, so keep it and do the rest. The
    // sample's group estimate travels with them — the merge that follows cannot measure its
    // own output size, and this is the only place that has.
    let groups = groups_from_sample(&sampled, rows_in, n, total_rows);
    let mut all = sampled;
    all.par_extend(partials(&morsels[n..], group_keys, aggregates, jit)?.into_par_iter());
    Ok(AggPlan::Partials {
        partials: all,
        groups,
    })
}

/// Partial rows the merge may inherit, as a fraction of the input, before chunking is not
/// worth its concatenation.
///
/// Chunking replaces the partition path's hash-and-gather over the whole relation with one
/// contiguous copy per worker, and pays for it with a `combine` over `workers x groups`
/// partial rows. So the question is only ever how big that merge is relative to the relation
/// it saves gathering, and a quarter is the point past which the merge is the larger of the
/// two — at which point the gather it avoids is no longer the expensive half.
const CHUNK_MERGE_CEILING: f64 = 0.25;

/// Whether one partial per worker beats partitioning, for a group count this size.
fn chunking_pays(threads: usize, groups: usize, total_rows: usize) -> bool {
    if groups == 0 || total_rows == 0 || threads < 2 {
        return false;
    }
    (threads as f64) * (groups as f64) <= CHUNK_MERGE_CEILING * (total_rows as f64)
}

/// One partial per **worker** rather than per morsel: concatenate each worker's share and
/// hash it once, into one table.
///
/// The reducing path builds a hash table per 16,384-row morsel, and the sample rejects it
/// when a morsel's partial keeps too many of its rows. That test is right about the morsel
/// and blind to the *relation*: a `GROUP BY` producing 10,000 groups fills a morsel's table
/// almost completely — 0.61 rows kept per input row, three times the ceiling — while
/// reducing 10 M rows to 10 thousand. So the aggregate was routed to the partition path,
/// which hashes and **gathers the entire relation** to avoid a merge that a worker-sized
/// table makes small: 96 workers x 10,000 groups is 960 k partial rows against the 6.1 M a
/// per-morsel partial hands the same merge, and against 10 M rows gathered.
///
/// Measured on the H2O `groupby` suite at its 1e7-row tier, this is the 10,000-group band
/// specifically, and it is where the suite loses: `sum(v1) BY id1, id2` ran at **2.86x**
/// DuckDB on two string keys and 2.02x on two integer ones, while the same query at 100
/// groups (0.92x) and at 100,000 (1.44x / 0.82x) sits either side of it.
///
/// The partials are the same partials — `eval_partial_jit` over a contiguous slice of the
/// same rows, in the same order — so `combine`, `finalize`, the spill path and the
/// distributed reduce are untouched. This is invariant #7's `partial` computed over a bigger
/// unit, and nothing else.
pub(crate) fn chunked_partials(
    morsels: &[RecordBatch],
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    jit: &AggJit,
    chunks: usize,
) -> Result<Vec<agg::Partial>, InterpError> {
    let per = morsels.len().div_ceil(chunks.max(1)).max(1);
    let schema = morsels[0].schema();
    morsels
        .par_chunks(per)
        .map(|chunk| {
            // One morsel is already contiguous; concatenating it would copy it for nothing.
            match chunk {
                [only] => ops::eval_partial_jit(only, group_keys, aggregates, jit),
                many => {
                    let joined = arrow::compute::concat_batches(&schema, many)
                        .map_err(crate::error::InterpError::from)?;
                    ops::eval_partial_jit(&joined, group_keys, aggregates, jit)
                }
            }
        })
        .collect()
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
    partitioned_partials(morsels, keys, group_keys, aggregates, jit, partitions)?
        .par_iter()
        .map(|partial| {
            let agg_columns = agg::finalize(funcs, partial)?;
            ops::build_agg_batch(group_keys, aggregates, &partial.group_columns, &agg_columns)
        })
        .collect()
}

/// Partition `morsels` on `keys` and group each partition once, returning one partial per
/// non-empty partition.
///
/// The partitions are **key-disjoint**: a row's bucket is a function of its key value alone,
/// so every row of a group lands in the same one and each partial is already final. That is
/// what lets [`partitioned_aggregate`] finalize each independently, and what lets a caller
/// that needs them as *one* partial glue them with `agg::concat_disjoint` — a concat — rather
/// than `agg::combine`, which would re-hash the whole relation to rediscover that no key is
/// shared.
pub(crate) fn partitioned_partials(
    morsels: &[RecordBatch],
    keys: &[String],
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    jit: &AggJit,
    partitions: usize,
) -> Result<Vec<agg::Partial>, InterpError> {
    ops::partition_morsels(morsels, keys, partitions)?
        .par_iter()
        .filter(|b| b.num_rows() > 0)
        .map(|bucket| ops::eval_partial_jit(bucket, group_keys, aggregates, jit))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows a morsel of `m` rows draws from a domain of `d` distinct keys yields, per the
    /// coupon-collector curve [`estimated_groups`] inverts. The oracle for the tests below.
    fn distinct_in(rows: f64, domain: f64) -> f64 {
        domain * (1.0 - (-rows / domain).exp())
    }

    /// The estimator must recover the group count a uniform key really produces, because
    /// that count is what sizes the partitioning. Checked against the closed form at four
    /// cardinalities spanning "grouping barely reduces" to "every row is its own group".
    #[test]
    fn group_estimate_tracks_the_real_group_count() {
        let morsel = 16_384.0;
        let total = 8_000_000.0;
        for domain in [100_000.0, 1_000_000.0, 4_000_000.0, 16_000_000.0] {
            let sample_morsels = 16;
            let sample_rows = morsel * sample_morsels as f64;
            let sample_groups = distinct_in(morsel, domain) * sample_morsels as f64;
            let est = estimated_groups(
                sample_rows as usize,
                sample_groups as usize,
                sample_morsels,
                total as usize,
            ) as f64;
            let actual = distinct_in(total, domain);
            let err = (est - actual).abs() / actual;
            assert!(
                err < 0.05,
                "domain={domain}: estimated {est}, actual {actual}"
            );
        }
    }

    /// The estimate is a width input, so it must stay inside the bounds that hold for *any*
    /// key distribution — never fewer groups than the sample already saw, never more than
    /// one per row — however adversarial the sample is.
    #[test]
    fn group_estimate_stays_within_its_provable_bounds() {
        for (rows, groups, morsels, total) in [
            (16_384, 16_384, 1, 8_000_000), // every sampled row distinct
            (16_384, 1, 1, 8_000_000),      // one group
            (16_384, 8_000, 1, 16_384),     // the sample is the whole relation
            (0, 0, 0, 8_000_000),           // nothing sampled
            (1, 1, 1, 1),                   // a one-row relation
        ] {
            let est = estimated_groups(rows, groups, morsels, total);
            assert!(est <= total, "{est} groups over {total} rows");
            if rows > 0 && groups > 0 {
                // The floor is the *average* per-morsel count, not the sum: see the note in
                // `estimated_groups` on why the sum is not a lower bound.
                let seen_by_one_morsel = (groups / morsels).min(total);
                assert!(
                    est >= seen_by_one_morsel,
                    "{est} below the {seen_by_one_morsel} seen"
                );
            }
        }
    }

    /// The width must fill the pool at any cardinality and never run away: one partition per
    /// core is the floor even for a handful of groups, and the ceiling holds however many
    /// groups are estimated.
    #[test]
    fn radix_width_is_bounded_by_cores_and_the_ceiling() {
        for threads in [1usize, 4, 16, 96] {
            assert!(radix_width(1, threads) >= threads, "below one per core");
            assert!(
                radix_width(usize::MAX, threads) <= MAX_PARTITIONS,
                "past the ceiling"
            );
            for groups in [1usize, 10_000, 1_000_000, 100_000_000] {
                let w = radix_width(groups, threads);
                assert!(w.is_power_of_two(), "width {w} is not a power of two");
                assert!((threads..=MAX_PARTITIONS).contains(&w) || w >= threads);
            }
        }
    }

    /// The property the width exists for: a partition's share of the groups stays bounded as
    /// the aggregate grows, until the ceiling binds. That is what keeps each partition's hash
    /// table cache-resident instead of scaling with the relation.
    #[test]
    fn width_bounds_the_groups_per_partition() {
        let ceiling = MAX_PARTITIONS * GROUPS_PER_PARTITION;
        for groups in [100_000usize, 1_000_000, 4_000_000, ceiling] {
            let w = radix_width(groups, 16);
            assert!(
                groups / w <= GROUPS_PER_PARTITION,
                "{groups} groups over {w} partitions exceeds the target"
            );
        }
    }

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
            AggPlan::Partials { partials, .. } => assert_eq!(partials.len(), morsels.len()),
            AggPlan::Partition { .. } => panic!("4 rows into 1 group must read as reducing"),
        }
    }

    /// An all-distinct key is routed to the partition path.
    #[test]
    fn an_all_distinct_group_by_partitions() {
        let morsels = [morsel(&[1, 2], &[1, 1]), morsel(&[3, 4], &[1, 1])];
        match plan(&morsels, true) {
            AggPlan::Partition { keys, .. } => assert_eq!(keys, vec!["k".to_string()]),
            AggPlan::Partials { .. } => panic!("an all-distinct key does not reduce"),
        }
    }

    /// The caller's veto is absolute: no partitioning, whatever the measurement says.
    #[test]
    fn a_vetoed_aggregate_never_partitions() {
        let morsels = [morsel(&[1, 2], &[1, 1]), morsel(&[3, 4], &[1, 1])];
        assert!(matches!(plan(&morsels, false), AggPlan::Partials { .. }));
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

        let AggPlan::Partials { partials: ps, .. } = plan(&morsels, false) else {
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
