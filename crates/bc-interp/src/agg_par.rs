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
    // TEMPORARY: pin the width to measure how the split scales with it.
    if let Ok(w) = std::env::var("BATCHER_AGG_PARTS") {
        if let Ok(w) = w.parse::<usize>() {
            return w.max(1);
        }
    }
    let threads = threads.max(1);
    let want = estimated_groups
        .div_ceil(GROUPS_PER_PARTITION)
        .clamp(threads, MAX_PARTITIONS);
    // Round **up to a multiple of the worker count**, not up to a power of two.
    //
    // The split fans out one task per partition, so its critical path is
    // `ceil(parts / workers)` buckets however small the last round is — and going a single
    // partition over the pool starts a whole second round. That is a cliff, not a slope, and
    // the old rule walked straight off it: it rounded a wanted width of 61 (the pool this box
    // gives a 10 M-row aggregate) up to 64.
    //
    // Measured by alternating the arms round by round in one process, so no arm can inherit
    // the other's cache state — H2O `groupby` at its 1e7-row tier, median of nine
    // (milliseconds, `BATCHER_AGG_PARTS` pinning the width):
    //
    // | partitions | 48 | 56 | **61** | 62 | 64 | 72 |
    // |---|---:|---:|---:|---:|---:|---:|
    // | q3 `sum,avg BY id3` | 51.1 | 48.1 | **46.4** | 62.0 | 61.2 | 56.1 |
    // | q5 `three sums BY id6` | 33.2 | 33.3 | **32.8** | 40.6 | 39.7 | 39.6 |
    // | q7 `max-min BY id3` | 50.5 | 49.2 | **49.5** | 63.1 | 62.2 | 57.1 |
    //
    // Every width at or below the pool is flat; 62 is 20-25 % worse than 61. Nothing about
    // powers of two is involved — 64 is merely where the old rounding happened to land.
    // `bucket_of` has always handled a non-power-of-two count (Lemire multiply-shift over the
    // hash's high bits instead of a mask over its low ones), so the rounding bought nothing
    // that had to be paid for.
    want.div_ceil(threads)
        .saturating_mul(threads)
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
pub(crate) fn estimated_groups(
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

/// Disjointness above which the sample's morsels are taken to cover *different* keys.
///
/// A morsel that shares almost none of its keys with the others is evidence the key is spread
/// across the relation rather than drawn from a small domain, and the two readings of the same
/// `(rows, groups)` ratio differ by orders of magnitude. Set high because the correction it
/// gates is a large one: only a sample whose morsels genuinely barely overlap should take it.
const SPREAD_MIN: f64 = 0.9;

/// [`estimated_groups`], told **also** how many distinct keys the sample held in total.
///
/// The coupon-collector inversion behind `estimated_groups` reads one morsel's ratio as a
/// property of the key's *domain*, which is right only when a morsel's rows are drawn from that
/// domain uniformly. A **clustered** key breaks that assumption completely, and TPC-H's
/// `l_orderkey` is the canonical example: four rows per key laid out in key order, so a
/// 4,096-row morsel holds 1,024 distinct keys and the inversion concludes the whole relation
/// holds about **1,050** of them. It holds 15,000,000 — an under-read of four orders of
/// magnitude, and it is not a rounding problem but a modelling one: "4,096 rows kept 1,024" is
/// exactly what a 1,050-value domain looks like too.
///
/// What separates the two is whether the sampled morsels keep the *same* keys or different
/// ones. A small domain has every morsel holding nearly all of it, so the sample's union is far
/// below the sum of its morsels' counts; a clustered key has each morsel covering its own
/// stretch, so the union *is* the sum. When the morsels are that disjoint the domain cannot be
/// as small as the inversion says, and the count instead scales with the rows — so the sample's
/// own distinct-per-row rate, read forward to the whole relation, is the estimate.
///
/// It costs the decision it feeds, not a rounding error: at sf10 the under-read routed a
/// 15M-group aggregate to `chunked_partials`, whose merge then re-grouped ~59M partial rows.
/// Measured on TPC-H q21's decorrelation group-by — 60M rows to 15M groups, four aggregates —
/// **483 ms on the path the bad estimate chose against 286 ms on the partition path** the
/// corrected one chooses.
///
/// **It over-reads a uniformly-random key with a genuinely huge domain**, and that is the
/// accepted trade: 60M rows over 15M random keys estimates ~59M rather than 15M, because a
/// sample that has not begun to saturate cannot tell "clustered" from "enormous". Both answers
/// are on the same side of every decision this feeds — the partition path, and a radix width
/// that is merely wider than it needed to be.
/// **And floored at the union itself, which is a count rather than a model.** The sample's
/// rows are rows of the relation, so every key its merged partials hold is a key the relation
/// holds: `union_groups` is a measured lower bound and an estimate below it is wrong by
/// construction, whatever the curve says. `estimated_groups` cannot apply this bound — it is
/// handed the *sum* of the per-morsel counts, which counts a key once per morsel it appears in,
/// and its own doc records that bounding below by that sum over-partitioned a 100 k-group key
/// by 2.4x. The union is that same quantity with the double-counting removed, and it is only
/// available here.
///
/// It is not a rounding correction. A **skewed** key defeats both of the estimates above at
/// once: the common values put every morsel's keys in every other morsel's, so the disjointness
/// test above fails and the linear read is not taken, while the long tail makes each morsel's
/// own ratio look like a small domain. ClickBench `GROUP BY URL` over the 1 M-row mirror is
/// exactly that shape — 275,494 groups, estimated at ~10 k — and the under-read is what routes
/// it to `chunked_partials`, whose `concat_batches` is a full copy of the relation (11% of the
/// query) before a partial that does not reduce and a merge over ~700 k partial rows.
pub(crate) fn estimated_groups_spread(
    sample_rows: usize,
    per_morsel_groups: usize,
    union_groups: usize,
    sample_morsels: usize,
    total_rows: usize,
) -> usize {
    let saturating = estimated_groups(sample_rows, per_morsel_groups, sample_morsels, total_rows);
    if per_morsel_groups == 0 || union_groups == 0 || sample_rows == 0 {
        return saturating;
    }
    // Measured, not modelled: these keys were seen in rows of this relation.
    let observed = union_groups.min(total_rows);
    if (union_groups as f64) / (per_morsel_groups as f64) < SPREAD_MIN {
        return saturating.max(observed);
    }
    let linear = (total_rows as f64) * (union_groups as f64) / (sample_rows as f64);
    saturating
        .max(linear as usize)
        .max(observed)
        .min(total_rows)
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
        // How many distinct keys the sample held *in total*, which is what tells a clustered
        // key from a small domain — see [`estimated_groups_spread`]. It is one merge of the
        // sample's own partials, which are already in hand and small (the sample is a bounded
        // fraction of the input), and it is only asked for on the non-reducing branch, where
        // the answer decides between two shapes that differ by hundreds of milliseconds.
        let funcs = ops::agg_funcs(aggregates);
        let union = agg::combine(&sampled, &funcs)
            .map(|p| p.group_columns.first().map_or(0, |c| c.len()))
            .unwrap_or(0);
        let per_morsel: usize = sampled
            .iter()
            .map(|p| p.group_columns.first().map_or(0, |c| c.len()))
            .sum();
        let groups = estimated_groups_spread(rows_in, per_morsel, union, n, total_rows);
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
pub(crate) fn chunking_pays(threads: usize, groups: usize, total_rows: usize) -> bool {
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

/// Contiguous runs of morsels whose first key column's value ranges do not overlap.
///
/// This is the partition the relation already has, for free. `partitioned_aggregate` pays
/// [`ops::partition_morsels`] — a gather of every row of every column into hash buckets — to
/// obtain key-disjoint pieces. When the key arrives **ordered**, the pieces are already there:
/// morsel `i`'s largest key is below morsel `i + 1`'s smallest, so a cut between them separates
/// the key space exactly as a hash bucket does, and applying it copies nothing at all.
///
/// That is the shape of every `GROUP BY` on a clustered key — TPC-H's `l_orderkey`, a lakehouse
/// table's declared sort key, a time-ordered ingest — and it is where the gather hurts most,
/// because a key that barely reduces is precisely the one whose gather moves the whole relation.
/// Measured on 60M rows grouping to 15M on a sorted `Int64` key, `min`/`max` aggregates:
///
/// | phase | ms |
/// |---|---:|
/// | `partition_morsels` (the gather) | 150 |
/// | the aggregation itself | 70 |
///
/// **Separation is established, never assumed.** The bounds are read off the data with arrow's
/// `min`/`max`, so an input that merely *claims* an order — a lakehouse `sorted_by` nothing
/// enforces on write — cannot make this fire. Getting it wrong would not be slow, it would split
/// one group across two runs and emit it twice, which is why the test is on values rather than
/// on metadata.
///
/// Why the **first** key column alone decides: if run A's first-column maximum is strictly below
/// run B's first-column minimum, then no composite key can appear in both, whatever the later
/// columns hold. A tighter test would admit more inputs; this one is sufficient and needs one
/// column's bounds.
///
/// Declines a null-bearing key (`min`/`max` skip nulls, so a null key could sit in two runs and
/// become two groups) and anything but `Int64` — the analytical key shape after the FFI boundary
/// widens narrow integers, and the one whose ordering is unambiguous. Returns `None` when fewer
/// than [`MIN_DISJOINT_RUNS_PER_THREAD`] runs per worker can be cut, which is the low-cardinality
/// case: one group then spans many morsels, no cut is legal, and the existing paths are right.
pub(crate) fn key_disjoint_runs(
    morsels: &[RecordBatch],
    keys: &[String],
    workers: usize,
) -> Option<Vec<std::ops::Range<usize>>> {
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::{DataType, Int64Type};

    let key = keys.first()?;
    if morsels.len() < 2 {
        return None;
    }
    // One morsel's key bounds, or `None` for a shape this cannot reason about.
    let bounds_of = |b: &RecordBatch| -> Option<(i64, i64)> {
        let col = b.column_by_name(key)?;
        if col.data_type() != &DataType::Int64 || col.null_count() > 0 || col.is_empty() {
            return None;
        }
        let a = col.as_primitive::<Int64Type>();
        Some((
            arrow::compute::kernels::aggregate::min(a)?,
            arrow::compute::kernels::aggregate::max(a)?,
        ))
    };

    // Aim for two runs per worker and accept one, so the overshoot below has somewhere to go.
    let total: usize = morsels.iter().map(|b| b.num_rows()).sum();
    let target = total
        .div_ceil(workers.max(1).saturating_mul(MIN_DISJOINT_RUNS_PER_THREAD))
        .max(1);

    // **Estimate the cut rate before scanning.** Reading every morsel's bounds costs a pass
    // over the key column — 6 ms on TPC-H sf10's 60M-row `l_orderkey` — and the answer is
    // usually no. A relation whose morsels arrive interleaved (ten parquet files read in
    // parallel, say) has a legal cut at well under 1% of its boundaries, where an ordered one
    // has a cut wherever a group happens not to straddle a morsel edge — for a key with a few
    // rows per group, most of them. Those two rates are orders of magnitude apart, so a sample
    // of boundaries separates them without touching the rest of the relation.
    //
    // Sampling is sound here because it only decides whether to *look*: the cut points the runs
    // are actually built from are read off the data in the scan below, never estimated.
    let boundaries = morsels.len() - 1;
    // `min` then `max`, never `clamp`: with fewer morsels than the sample wants, `clamp`'s
    // bounds cross and it panics.
    let samples = workers.saturating_mul(2).max(8).min(boundaries).max(1);
    let mut hits = 0usize;
    for s in 0..samples {
        let i = s.saturating_mul(boundaries) / samples;
        if bounds_of(&morsels[i])?.1 < bounds_of(&morsels[i + 1])?.0 {
            hits += 1;
        }
    }
    // Extrapolate to the whole relation and require comfortably more cuts than runs wanted —
    // a cut only helps where it falls near a target boundary, so parity would not be enough.
    if hits.saturating_mul(boundaries) < workers.saturating_mul(2).saturating_mul(samples) {
        return None;
    }

    let bounds: Vec<(i64, i64)> = morsels.par_iter().map(bounds_of).collect::<Option<_>>()?;

    // A cut must be legal against **everything that follows**, not the next morsel alone. This
    // was `hi < bounds[i + 1].0` with `hi` reset per cut, which asks the weaker question; the
    // two agree on an ordered relation and not on a **concatenation of two**, where the left
    // side cuts freely as its keys ascend and the right restarts at the bottom, leaving every
    // left run overlapping the run holding the right. `concat_disjoint` then emits each shared
    // key twice — a `GROUP BY` returning one key on two rows. `INTERSECT`/`EXCEPT` lower to
    // `union → GROUP BY k` and build exactly that: over 6M sorted rows against 1.5M,
    // `INTERSECT` returned 4,906 where the answer is 1,472,588. Prefix max against suffix min
    // is the fix, still one O(morsels) pass, and the prefix max is **not** reset at a cut —
    // the guarantee is about all earlier runs, not the current one.
    let mut suffix_min = vec![i64::MAX; morsels.len() + 1];
    for i in (0..morsels.len()).rev() {
        suffix_min[i] = suffix_min[i + 1].min(bounds[i].0);
    }

    // Close a run at the first *legal* cut past its share of the rows.
    let mut runs: Vec<std::ops::Range<usize>> = Vec::new();
    let (mut start, mut rows, mut prefix_max) = (0usize, 0usize, i64::MIN);
    for i in 0..morsels.len() {
        rows += morsels[i].num_rows();
        prefix_max = prefix_max.max(bounds[i].1);
        let cuttable = i + 1 < morsels.len() && prefix_max < suffix_min[i + 1];
        if cuttable && rows >= target {
            runs.push(start..i + 1);
            (start, rows) = (i + 1, 0);
        }
    }
    runs.push(start..morsels.len());
    (runs.len() >= workers.max(1).max(2)).then_some(runs)
}

/// Runs *aimed for* per worker — the cut points are then accepted wherever the key allows one,
/// and the path is taken when at least one run per worker came out.
///
/// Aiming at one per worker and demanding one per worker cannot both be met: a run closes at the
/// first legal cut *past* its share of the rows, so it always overshoots slightly and the count
/// lands just under the worker count. Aiming at two and accepting one leaves that slack, and
/// gives rayon something to steal with when the cuts fall unevenly.
///
/// The one-per-worker floor is also what keeps this off the low-cardinality group-by: there a
/// group spans many morsels, almost no cut between them is legal, and the handful of runs that
/// come out would run the aggregate on a handful of cores.
const MIN_DISJOINT_RUNS_PER_THREAD: usize = 2;

/// Aggregate each key-disjoint run of morsels independently, and concatenate.
///
/// Each run holds every row for the keys inside it and no row for any other, so its partial is
/// already final — the identical argument [`partitioned_aggregate`] makes about its hash
/// buckets, reached without the gather. A run of one morsel is aggregated directly; a longer one
/// aggregates each morsel and merges with `combine`, which is cheap here because it is a merge
/// *within* a run (a few morsels' worth of groups, cache-resident) rather than across the whole
/// relation.
pub(crate) fn disjoint_run_aggregate(
    morsels: &[RecordBatch],
    runs: &[std::ops::Range<usize>],
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    jit: &AggJit,
    funcs: &[agg::AggFunc],
) -> Result<Vec<RecordBatch>, InterpError> {
    runs.par_iter()
        .map(|run| {
            let partial = match &morsels[run.clone()] {
                [] => return Ok(None),
                [only] => ops::eval_partial_jit(only, group_keys, aggregates, jit)?,
                many => {
                    let parts: Vec<agg::Partial> = many
                        .iter()
                        .map(|b| ops::eval_partial_jit(b, group_keys, aggregates, jit))
                        .collect::<Result<_, _>>()?;
                    agg::combine(&parts, funcs)?
                }
            };
            let agg_columns = agg::finalize(funcs, &partial)?;
            ops::build_agg_batch(group_keys, aggregates, &partial.group_columns, &agg_columns)
                .map(Some)
        })
        .filter_map(|r| r.transpose())
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
    let t0 = std::time::Instant::now();
    let buckets = ops::partition_morsels(morsels, keys, partitions)?;
    let split = t0.elapsed();
    let t1 = std::time::Instant::now();
    let out: Result<Vec<agg::Partial>, InterpError> = buckets
        .par_iter()
        .filter(|b| b.num_rows() > 0)
        .map(|bucket| ops::eval_partial_jit(bucket, group_keys, aggregates, jit))
        .collect();
    if std::env::var("BATCHER_DEBUG_AGGSPLIT").is_ok() {
        eprintln!(
            "AGGSPLIT split={:?} aggregate={:?} parts={}",
            split,
            t1.elapsed(),
            partitions
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A morsel of `rows` consecutive keys starting at `from`, each repeated `per` times.
    fn keyed(from: i64, rows: usize, per: i64) -> RecordBatch {
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        let k: Vec<i64> = (0..rows as i64).map(|i| from + i / per).collect();
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(k))]).expect("batch")
    }

    /// **A concatenation of two ordered relations is not an ordered relation**, and cutting it
    /// as though it were is how a `GROUP BY` returns the same key twice.
    ///
    /// This is the shape `INTERSECT`/`EXCEPT` produce — they lower to `union → GROUP BY k`, so
    /// the aggregate sees one side's morsels ascending and then the other side's ascending
    /// again from the bottom. Cutting on "is this run below the *next* morsel?" cuts freely
    /// through the first side and then cannot cut again, leaving every one of those runs
    /// overlapping the run that holds the second side. Over 6M rows against 1.5M that made
    /// `INTERSECT` return 4,906 rows where the answer is 1,472,588.
    ///
    /// The property asserted is the one `concat_disjoint` is promised, checked on the cut
    /// points rather than inferred: no key may appear in two runs.
    #[test]
    fn two_ordered_relations_concatenated_are_never_cut_into_overlapping_runs() {
        use arrow::array::AsArray;
        use arrow::datatypes::Int64Type;

        let per = 4i64;
        // Each "side" is ordered on its own; together they restart the key space once.
        let side: Vec<RecordBatch> = (0..200)
            .map(|m| keyed(m as i64 * 1_000 / per, 1_000, per))
            .collect();
        let morsels: Vec<RecordBatch> = side.iter().chain(side.iter()).cloned().collect();
        let keys = vec!["k".to_string()];

        let Some(runs) = key_disjoint_runs(&morsels, &keys, 8) else {
            return; // declining is a correct answer for this shape
        };
        let mut seen: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        for (r, run) in runs.iter().enumerate() {
            for m in run.clone() {
                for &k in morsels[m].column(0).as_primitive::<Int64Type>().values() {
                    if let Some(&prev) = seen.get(&k) {
                        assert_eq!(prev, r, "key {k} appears in runs {prev} and {r}");
                    } else {
                        seen.insert(k, r);
                    }
                }
            }
        }
    }

    /// The runs this cuts must be **key-disjoint**, because that is the entire licence for
    /// finalizing each one on its own: a key appearing in two runs is emitted as two groups,
    /// silently. So the property is checked directly on the cut points rather than inferred
    /// from the ordering they were derived from.
    #[test]
    fn every_cut_separates_the_key_space() {
        use arrow::array::AsArray;
        use arrow::datatypes::Int64Type;
        // 400 morsels of 1,000 rows, four rows per key: most morsel edges fall inside a group,
        // so the cuts have to be *found* rather than taken at every boundary.
        let per = 4i64;
        let morsels: Vec<RecordBatch> = (0..400)
            .map(|m| keyed(m as i64 * 1_000 / per, 1_000, per))
            .collect();
        let keys = vec!["k".to_string()];
        let runs = key_disjoint_runs(&morsels, &keys, 8).expect("an ordered key yields runs");
        assert!(
            runs.len() >= 8,
            "wanted at least one run per worker, got {}",
            runs.len()
        );
        // The runs tile the morsels exactly, and no key spans two of them.
        assert_eq!(runs[0].start, 0);
        assert_eq!(runs[runs.len() - 1].end, morsels.len());
        let mut last_max: Option<i64> = None;
        for run in &runs {
            let (mut lo, mut hi) = (i64::MAX, i64::MIN);
            for b in &morsels[run.clone()] {
                let a = b.column(0).as_primitive::<Int64Type>();
                for i in 0..a.len() {
                    lo = lo.min(a.value(i));
                    hi = hi.max(a.value(i));
                }
            }
            if let Some(prev) = last_max {
                assert!(
                    prev < lo,
                    "run starting at {lo} overlaps the previous run ending at {prev}"
                );
            }
            last_max = Some(hi);
        }
    }

    /// A key whose morsels interleave has no legal cut, and must be refused rather than cut
    /// somewhere that splits a group. This is the shape of a relation read from several files
    /// in parallel, which is what TPC-H's `lineitem` arrives as.
    #[test]
    fn an_interleaved_key_yields_no_runs() {
        // Every morsel spans the whole key range, so no boundary separates anything.
        let morsels: Vec<RecordBatch> = (0..400).map(|_| keyed(0, 1_000, 1)).collect();
        assert!(key_disjoint_runs(&morsels, &["k".to_string()], 8).is_none());
    }

    /// A null in the key is refused: `min`/`max` skip nulls, so a null-keyed row could sit in
    /// two runs and be emitted as two groups.
    #[test]
    fn a_null_key_is_refused() {
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, true)]));
        let morsels: Vec<RecordBatch> = (0..400)
            .map(|m| {
                let k: Vec<Option<i64>> = (0..1_000i64)
                    .map(|i| {
                        if i == 7 {
                            None
                        } else {
                            Some(m as i64 * 1_000 + i)
                        }
                    })
                    .collect();
                RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(k))])
                    .expect("batch")
            })
            .collect();
        assert!(key_disjoint_runs(&morsels, &["k".to_string()], 8).is_none());
    }

    /// Rows a morsel of `m` rows draws from a domain of `d` distinct keys yields, per the
    /// coupon-collector curve [`estimated_groups`] inverts. The oracle for the tests below.
    fn distinct_in(rows: f64, domain: f64) -> f64 {
        domain * (1.0 - (-rows / domain).exp())
    }

    /// A **clustered** key is the one shape the coupon-collector inversion cannot read, and
    /// getting it wrong is what routes a 15M-group aggregate to the merge-heavy path. Four
    /// rows per key laid out in key order: every morsel holds 1,024 distinct keys out of 4,096
    /// rows, exactly as a 1,050-value domain would — and the morsels share no key, which a
    /// small domain never does.
    #[test]
    fn a_clustered_key_is_not_read_as_a_tiny_domain() {
        let (morsels, per_morsel_rows, per_morsel_groups) = (61usize, 4096usize, 1024usize);
        let rows = morsels * per_morsel_rows;
        let summed = morsels * per_morsel_groups;
        let total = 59_986_052usize;
        // The bare inversion: four orders of magnitude low, and the reason this test exists.
        assert!(
            estimated_groups(rows, summed, morsels, total) < 2_000,
            "the unaided inversion is expected to under-read a clustered key"
        );
        // Disjoint morsels (union == sum) say the domain cannot be that small.
        let spread = estimated_groups_spread(rows, summed, summed, morsels, total);
        assert!(
            (14_000_000..=16_000_000).contains(&spread),
            "a clustered key must read as ~15M groups, got {spread}"
        );
    }

    /// …and a genuinely small domain must still read as small. Every morsel holds nearly the
    /// whole domain, so the sample's union is far below the sum of its morsels' counts and the
    /// correction stays out of the way.
    #[test]
    fn a_small_domain_is_still_read_as_small() {
        let (morsels, per_morsel_rows, domain) = (61usize, 4096usize, 1000usize);
        let rows = morsels * per_morsel_rows;
        let summed = morsels * domain; // every morsel sees all of it
        let spread = estimated_groups_spread(rows, summed, domain, morsels, 59_986_052);
        assert!(
            spread < 2_000,
            "a 1,000-value domain must not be inflated by the spread correction, got {spread}"
        );
        assert_eq!(spread, estimated_groups(rows, summed, morsels, 59_986_052));
    }

    /// A **skewed** key defeats both readings at once, and the union is what rescues it.
    ///
    /// Hot values put most of every morsel's keys in every other morsel's, so the disjointness
    /// test fails and the linear read is not taken; the long tail makes each morsel's own ratio
    /// look like a domain of ten thousand. ClickBench `GROUP BY URL` is this shape: 275,494
    /// groups in 1 M rows, read as ~10 k. The union of the sample's own partials is 30,000
    /// keys that were *counted*, so no estimate below it can be right.
    #[test]
    fn a_skewed_key_is_floored_at_the_keys_the_sample_actually_held() {
        let (morsels, per_morsel_rows) = (4usize, 16_384usize);
        let rows = morsels * per_morsel_rows;
        let summed = 36_000usize; // ~9,000 distinct per morsel
        let union = 30_000usize; // heavy overlap: 0.83 of the sum, under SPREAD_MIN
        let total = 1_000_000usize;
        assert!(
            (union as f64) / (summed as f64) < SPREAD_MIN,
            "the fixture must be on the branch the linear read does not reach"
        );
        let base = estimated_groups(rows, summed, morsels, total);
        assert!(
            base < union,
            "the unaided inversion is expected to fall below the counted union, got {base}"
        );
        assert_eq!(
            estimated_groups_spread(rows, summed, union, morsels, total),
            union,
            "an estimate below a measured lower bound is wrong however the curve reads"
        );
    }

    /// The floor is a *lower* bound and never becomes the answer on its own: where the model
    /// already reads higher than the union, the model wins.
    #[test]
    fn the_union_floor_does_not_lower_a_larger_estimate() {
        let (morsels, per_morsel_rows, per_morsel_groups) = (61usize, 4096usize, 1024usize);
        let rows = morsels * per_morsel_rows;
        let summed = morsels * per_morsel_groups;
        let total = 59_986_052usize;
        let spread = estimated_groups_spread(rows, summed, summed, morsels, total);
        assert!(
            spread > summed,
            "the clustered read is far above its own union and must stay there, got {spread}"
        );
    }

    /// The correction only ever raises the estimate, and never past one group per row — both
    /// properties the decisions downstream rely on.
    #[test]
    fn the_spread_correction_only_raises_and_stays_bounded() {
        let total = 1_000_000usize;
        for union in [1usize, 10, 1_000, 40_000, 60_000] {
            let summed = 61 * 1_000;
            let rows = 61 * 4_096;
            let base = estimated_groups(rows, summed, 61, total);
            let spread = estimated_groups_spread(rows, summed, union, 61, total);
            assert!(
                spread >= base,
                "the correction must never lower the estimate"
            );
            assert!(spread <= total, "never more groups than rows");
        }
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
                // Every worker takes the same number of buckets, so the split has no
                // straggler round — the property that replaced "round to a power of two".
                assert!(
                    w.is_multiple_of(threads) || w == MAX_PARTITIONS,
                    "width {w} is not a multiple of {threads} workers"
                );
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
            interpolation: None,
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
