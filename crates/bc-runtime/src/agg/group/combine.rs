//! Parallel hash-radix `combine` regroup for a high-cardinality aggregate.
//!
//! Hash-radix partitions the concatenated partials by key, so every row of a group lands
//! in one partition; each partition is then grouped *and* merged independently across
//! threads with **no cross-partition merge**, turning the otherwise-serial per-group
//! accumulate scan (which dominates a many-group combine) into a parallel one.

use arrow::array::{Array, ArrayRef, AsArray};
use arrow::compute::interleave;
use arrow::datatypes::{
    ArrowPrimitiveType, DataType, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type,
    UInt32Type, UInt64Type, UInt8Type,
};
use rayon::prelude::*;

use super::assign::assign_groups;
use super::hash::hash_partial_keys;
use crate::agg::{
    accumulate, merge_approx_distinct, merge_approx_quantile, merge_arg_extreme, merge_counted,
    merge_covar, merge_distinct, merge_median, merge_moments, merge_welford, AggFunc, Partial,
};
use crate::error::RuntimeError;

/// One merged radix partition: its group-key columns, and per aggregate its state columns.
type MergedPartition = (Vec<ArrayRef>, Vec<Vec<ArrayRef>>);

/// Copy the rows named by `(part_of[i], row_of[i])` out of `cols` into one flat array.
///
/// The whole-column move for a primitive: read the source value, write the output, no
/// builder. Validity is gathered the same way and only when some source actually has a null,
/// so a null-free column allocates no bitmap — the array `interleave` would have produced,
/// either way.
///
/// **Nullable is not a special case here, and used to be.** Gathering only the values meant
/// the caller had to refuse any column carrying a null and hand it to `interleave` instead —
/// and a null is not rare in this position: an outer join makes every column below it
/// nullable, so on a query built from them the fast path could not fire at all.
/// `interleave_primitive` was **5.1% of TPC-DS q78**, whose three CTEs are each a `LEFT JOIN`
/// feeding a three-key `GROUP BY`. A value under a null slot is unspecified but readable
/// (arrow allocates the whole values buffer), so reading it and then masking it is exactly
/// what arrow's own kernels do.
fn gather_primitive<T: ArrowPrimitiveType>(
    cols: &[&dyn Array],
    part_of: &[u32],
    row_of: &[u32],
) -> ArrayRef {
    use arrow::array::PrimitiveArray;
    use arrow::buffer::NullBuffer;
    use std::sync::Arc;

    let arrs: Vec<&PrimitiveArray<T>> = cols.iter().map(|c| c.as_primitive::<T>()).collect();
    let vals: Vec<&[T::Native]> = arrs.iter().map(|a| a.values().as_ref()).collect();
    let mut out: Vec<T::Native> = Vec::with_capacity(part_of.len());
    out.extend(
        part_of
            .iter()
            .zip(row_of)
            .map(|(&p, &r)| vals[p as usize][r as usize]),
    );
    let nulls = arrs.iter().any(|a| a.null_count() > 0).then(|| {
        NullBuffer::from_iter(
            part_of
                .iter()
                .zip(row_of)
                .map(|(&p, &r)| arrs[p as usize].is_valid(r as usize)),
        )
    });
    Arc::new(PrimitiveArray::<T>::new(out.into(), nulls))
}

/// Gather one output column, taking the flat typed path when every source permits it.
///
/// `interleave` is the general answer — it handles strings, nested types and nulls — but it
/// wants a materialized `&[(usize, usize)]`, **sixteen bytes of index per output row**, and
/// builds through `MutableArrayData`. On a high-cardinality combine that index array is the
/// dominant traffic: ClickBench q32 (`GROUP BY WatchID, ClientIP` over ~10M near-unique
/// groups) spent 27% of the query inside `interleave_primitive`, moving two key columns and
/// four state columns that way.
///
/// This is the same trade `ops::repartition` already makes on the shuffle side, applied to
/// the other place that gathers by row address. The `u32` planes are half the index bytes,
/// and a primitive column copies values (and, where it has any, validity) directly.
///
/// Anything else — a nested state, a temporal or decimal type this does not name, a source
/// set of mixed types — falls through to `interleave` unchanged, sharing one lazily built
/// pair vector. Strings take their own bulk path (`gather::gather_strings`).
fn gather(
    cols: &[&dyn Array],
    part_of: &[u32],
    row_of: &[u32],
    pairs: &mut Option<Vec<(usize, usize)>>,
) -> Result<ArrayRef, RuntimeError> {
    macro_rules! fast {
        ($($dt:pat => $ty:ty),* $(,)?) => {
            match cols.first().map(|c| c.data_type()) {
                $(Some($dt) => return Ok(gather_primitive::<$ty>(cols, part_of, row_of)),)*
                _ => {}
            }
        };
    }
    // Every source must be the *same* primitive type as the first, since the fast path
    // downcasts them all to one `T`. Partials of one aggregate share a schema in practice;
    // checking is cheap and turns a would-be panic into the interleave fallback.
    let uniform = cols
        .split_first()
        .is_some_and(|(h, t)| t.iter().all(|c| c.data_type() == h.data_type()));
    if uniform {
        fast! {
            DataType::Int8 => Int8Type, DataType::Int16 => Int16Type,
            DataType::Int32 => Int32Type, DataType::Int64 => Int64Type,
            DataType::UInt8 => UInt8Type, DataType::UInt16 => UInt16Type,
            DataType::UInt32 => UInt32Type, DataType::UInt64 => UInt64Type,
            DataType::Float32 => arrow::datatypes::Float32Type,
            DataType::Float64 => arrow::datatypes::Float64Type,
            DataType::Date32 => arrow::datatypes::Date32Type,
            DataType::Date64 => arrow::datatypes::Date64Type,
        }
    }
    // A string key is the *other* common high-cardinality group key, and arrow's `interleave`
    // costs it what it costs any variable-width column: `MutableArrayData::extend` per row,
    // through the sixteen-byte pair vector below. `gather_strings` reads the same two planes
    // the primitive path does and fills one output buffer across cores.
    if let Some(out) = crate::gather::gather_strings(cols, part_of, row_of) {
        return Ok(out);
    }
    let pairs = pairs.get_or_insert_with(|| {
        part_of
            .iter()
            .zip(row_of)
            .map(|(&p, &r)| (p as usize, r as usize))
            .collect()
    });
    interleave(cols, pairs).map_err(RuntimeError::from)
}

/// Parallel `combine` regroup via hash-radix partitioning. Returns the merged group-key
/// columns and, per aggregate, its merged state columns — identical to the serial
/// `assign_groups` + `merge_state` path (group *order* differs, which callers treat as
/// unspecified, like any hash aggregate).
///
/// `group_concat` are the concatenated partial group-key columns; `state_concats[a]` are
/// aggregate `a`'s concatenated partial-state columns; both have `total_rows` rows.
pub(crate) fn combine_radix(
    parts: &[Partial],
    funcs: &[AggFunc],
    total_rows: usize,
    partitions: usize,
) -> Result<(Vec<ArrayRef>, Vec<Vec<ArrayRef>>), RuntimeError> {
    let per = combine_radix_parts(parts, funcs, total_rows, partitions)?;
    // Concatenate partition outputs (key-disjoint → concat == merge). Fan the per-column
    // concats across cores — on a high-cardinality distinct/group-by these output columns
    // are millions of rows and the concat is a second full copy that otherwise runs serial.
    let group_columns: Vec<ArrayRef> = (0..parts[0].group_columns.len())
        .into_par_iter()
        .map(|k| concat_col(per.iter().map(|(g, _)| &g[k])))
        .collect::<Result<_, _>>()?;
    let states: Vec<Vec<ArrayRef>> = (0..funcs.len())
        .map(|a| {
            (0..per[0].1[a].len())
                .map(|c| concat_col(per.iter().map(|(_, s)| &s[a][c])))
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;
    Ok((group_columns, states))
}

/// [`combine_radix`] **without the final concat**: the merged partitions, each a
/// `(group_columns, per-aggregate state columns)` pair.
///
/// The partitions are key-disjoint by construction — that is the whole premise of the radix
/// regroup — so their union *is* the merged relation and a caller that can emit several
/// morsels never needs them glued together. Concatenating them costs a second full copy of
/// the grouped output, which on a high-cardinality string key is the single largest term in
/// the combine (ClickBench q33: 15 ms of a 42 ms combine, moving 43 MB to reproduce data the
/// caller immediately re-morselizes). Emitting the partitions also hands the next operator
/// `partitions` batches to work on instead of one, so a downstream sort or projection fans
/// back out across cores.
pub(crate) fn combine_radix_parts(
    parts: &[Partial],
    funcs: &[AggFunc],
    total_rows: usize,
    partitions: usize,
) -> Result<Vec<MergedPartition>, RuntimeError> {
    // Bin row indices by `hash(key) % partitions` so equal keys co-locate in one bucket.
    //
    // Hashed per partial and flattened in partial order rather than over a concatenation of
    // them, because that concatenation is a full copy of the key column and the merge never
    // reads it as one array — the gather below addresses rows as `(partial, row)` anyway.
    // At a high group count the copy is the merge's largest term, and its single multi-tens-of-
    // MB allocation pays for its own page faults on top of the bytes it moves.
    let hashes = hash_partial_keys(parts, total_rows)?;
    // Global row `i` lives in partial `owner[i]` at `i - starts[owner[i]]` — the map back from
    // the flattened numbering the bucketing uses to the arrays the gather reads.
    let mut starts: Vec<u32> = Vec::with_capacity(parts.len() + 1);
    let mut acc = 0u32;
    for p in parts {
        starts.push(acc);
        acc += p.group_columns.first().map_or(0, |c| c.len()) as u32;
    }
    starts.push(acc);
    // Parallel stable counting-sort into the per-bucket index lists: each row-range chunk
    // bins its rows into a flat **per-chunk CSR** (histogram → prefix-sum → one scatter
    // pass), then bucket `b`'s global list is the chunks' `b`-slices concatenated in chunk
    // order. Using a flat CSR per chunk — rather than a `Vec<Vec<u32>>` of `partitions`
    // growing vectors — is what keeps this from allocating `O(threads × partitions)` growing
    // buffers (thousands at a high core count, each reallocating on push): the storm that
    // made the combine *regress* past ~16 cores on a high-cardinality DISTINCT. A serial
    // single-pass bin of a 6 M-row concat was ~60 ms; this spreads it across cores and pays
    // two flat allocations per chunk instead. Per-bucket order is unspecified for a hash
    // aggregate, so any consistent order is fine (this yields ascending-within-chunk).
    let buckets: Vec<Vec<u32>> = {
        let nthreads = rayon::current_num_threads().max(1);
        let chunk = total_rows.div_ceil(nthreads).max(1);
        // Each chunk returns `(rows, offsets)`: `rows[offsets[b]..offsets[b + 1]]` are that
        // chunk's global row indices for bucket `b`, ascending.
        let per_chunk: Vec<(Vec<u32>, Vec<u32>)> = hashes
            .par_chunks(chunk)
            .enumerate()
            .map(|(ci, slice)| {
                let base = (ci * chunk) as u32;
                let mut offsets = vec![0u32; partitions + 1];
                for &h in slice {
                    offsets[(h % partitions as u64) as usize + 1] += 1;
                }
                for b in 0..partitions {
                    offsets[b + 1] += offsets[b];
                }
                let mut cursor = offsets[..partitions].to_vec();
                let mut rows = vec![0u32; slice.len()];
                for (j, &h) in slice.iter().enumerate() {
                    let b = (h % partitions as u64) as usize;
                    rows[cursor[b] as usize] = base + j as u32;
                    cursor[b] += 1;
                }
                (rows, offsets)
            })
            .collect();
        (0..partitions)
            .into_par_iter()
            .map(|p| {
                let total: usize = per_chunk
                    .iter()
                    .map(|(_, off)| (off[p + 1] - off[p]) as usize)
                    .sum();
                let mut out = Vec::with_capacity(total);
                for (rows, off) in &per_chunk {
                    out.extend_from_slice(&rows[off[p] as usize..off[p + 1] as usize]);
                }
                out
            })
            .collect()
    };

    // Each partition groups + merges independently — its keys appear in no other
    // partition, so its merged groups are final and a plain concat is the whole result.
    // The gather reads straight from the partials through `(partial, row)` pairs, so no
    // column is ever materialized in full.
    let n_keys = parts[0].group_columns.len();
    let per: Vec<MergedPartition> = buckets
        .par_iter()
        .map(|idx| -> Result<_, RuntimeError> {
            // The row addresses, split into two `u32` planes rather than one
            // `Vec<(usize, usize)>`. See `gather` for why: the pair form is 16 bytes of
            // index per output row, and this bucket may hold millions.
            let mut part_of: Vec<u32> = Vec::with_capacity(idx.len());
            let mut row_of: Vec<u32> = Vec::with_capacity(idx.len());
            for &g in idx {
                let p = starts.partition_point(|&s| s <= g) - 1;
                part_of.push(p as u32);
                row_of.push(g - starts[p]);
            }
            // Built only if some column declines the flat gather, and then shared by all
            // of them — a string key would otherwise rebuild it per column.
            let mut pairs: Option<Vec<(usize, usize)>> = None;
            let mut keys_p: Vec<ArrayRef> = Vec::with_capacity(n_keys);
            for k in 0..n_keys {
                let cols: Vec<&dyn Array> =
                    parts.iter().map(|p| p.group_columns[k].as_ref()).collect();
                keys_p.push(gather(&cols, &part_of, &row_of, &mut pairs)?);
            }
            let (local_ids, n_local, group_cols_p) = assign_groups(&keys_p, idx.len())?;
            let mut states_p = Vec::with_capacity(funcs.len());
            for (a, &func) in funcs.iter().enumerate() {
                let mut state_p: Vec<ArrayRef> = Vec::with_capacity(parts[0].states[a].len());
                for c in 0..parts[0].states[a].len() {
                    let cols: Vec<&dyn Array> =
                        parts.iter().map(|p| p.states[a][c].as_ref()).collect();
                    state_p.push(gather(&cols, &part_of, &row_of, &mut pairs)?);
                }
                states_p.push(merge_state(func, &state_p, &local_ids, n_local)?);
            }
            Ok((group_cols_p, states_p))
        })
        .collect::<Result<_, _>>()?;

    Ok(per)
}

/// Concatenate a sequence of arrays (the per-partition outputs) into one.
///
/// Through [`crate::gather::concat_columns`], not arrow's `concat` directly: a
/// high-cardinality group-by concatenates a string key column here twice (the partials in,
/// the partitions out) and arrow's per-row path made those two copies the dominant cost of
/// the whole combine.
fn concat_col<'a>(arrs: impl Iterator<Item = &'a ArrayRef>) -> Result<ArrayRef, RuntimeError> {
    let owned: Vec<&dyn Array> = arrs.map(|a| a.as_ref()).collect();
    crate::gather::concat_columns(&owned)
}
/// Merge already-partial state columns into one group via the function's
/// associative reducer (single-pass, reusing `accumulate`). Counts/sums merge by
/// summing the partial states; min/max by min/max; mean by summing both the
/// partial sums and the partial counts.
pub(crate) fn merge_state(
    func: AggFunc,
    state: &[ArrayRef],
    group_ids: &[u32],
    num_groups: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    Ok(match func {
        AggFunc::CountStar | AggFunc::Count | AggFunc::Sum => {
            accumulate(AggFunc::Sum, Some(&state[0]), group_ids, num_groups)?
        }
        // Distinct sets merge by unioning the per-group value lists (dedup again).
        AggFunc::CountDistinct => vec![merge_distinct(&state[0], group_ids, num_groups)?],
        AggFunc::Median
        | AggFunc::Quantile(_)
        | AggFunc::ListAgg
        // The contiguity statistics carry `Median`'s value list, so they merge by the same
        // concatenation. This arm *is* their mergeability.
        | AggFunc::NLength(_)
        | AggFunc::LCount(_)
        | AggFunc::AuN
        | AggFunc::Histogram
        | AggFunc::Entropy
        | AggFunc::Mad
        | AggFunc::QuantileDisc(_) => {
            vec![merge_median(&state[0], group_ids, num_groups)?]
        }
        // Counted states merge by summing the counts of equal values (see `agg::counted`);
        // addition is associative and commutative, which is this pair's mergeability.
        AggFunc::Mode | AggFunc::ApproxTopK(_) => merge_counted(state, group_ids, num_groups)?,
        // `any_value` merges with the same min reducer that built its partial.
        // Compensated states merge by compensated-adding the sums and summing the
        // compensations — the same fold the partial performs, so `combine([p]) == p`.
        AggFunc::KahanSum => merge_kahan(&state[0], &state[1], group_ids, num_groups),
        AggFunc::Min | AggFunc::AnyValue => {
            accumulate(AggFunc::Min, Some(&state[0]), group_ids, num_groups)?
        }
        AggFunc::Max => accumulate(AggFunc::Max, Some(&state[0]), group_ids, num_groups)?,
        // Boolean state re-folds via the same AND/OR reducer (associative).
        AggFunc::BoolAnd | AggFunc::BoolOr => {
            accumulate(func, Some(&state[0]), group_ids, num_groups)?
        }
        // Product / bitwise state re-folds via the same associative op.
        AggFunc::Product | AggFunc::BitAnd | AggFunc::BitOr | AggFunc::BitXor => {
            accumulate(func, Some(&state[0]), group_ids, num_groups)?
        }
        // Per-group HLL sketches union across partitions.
        AggFunc::ApproxCountDistinct => {
            vec![merge_approx_distinct(&state[0], group_ids, num_groups)?]
        }
        // Per-group KLL sketches merge across partitions.
        AggFunc::ApproxQuantile(_) => {
            vec![merge_approx_quantile(&state[0], group_ids, num_groups)?]
        }
        // 2-column (key, value) state: keep the extreme-key pair per group.
        AggFunc::ArgMin | AggFunc::ArgMax => merge_arg_extreme(
            state,
            group_ids,
            num_groups,
            matches!(func, AggFunc::ArgMax),
        )?,
        AggFunc::Mean => vec![
            accumulate(AggFunc::Sum, Some(&state[0]), group_ids, num_groups)?
                .into_iter()
                .next()
                .unwrap(),
            accumulate(AggFunc::Sum, Some(&state[1]), group_ids, num_groups)?
                .into_iter()
                .next()
                .unwrap(),
        ],
        // Welford (mean, M2, count) states merge with Chan's parallel formula — summing
        // them would be wrong (mean/M2 are not additive across partitions).
        AggFunc::Var | AggFunc::Stddev => {
            merge_welford(&state[0], &state[1], &state[2], group_ids, num_groups)
        }
        // Central-moment states merge with the parallel (Terriberry/Chan) higher-moment
        // formulas — summing them would be wrong (mean/M2/M3/M4 and the co-moment are not
        // additive across partitions), and the old sum-of-powers form catastrophically
        // cancelled at a large offset.
        AggFunc::Skewness | AggFunc::Kurtosis | AggFunc::KurtosisPop => {
            merge_moments(state, group_ids, num_groups)?
        }
        AggFunc::CovarPop | AggFunc::CovarSamp | AggFunc::Corr => {
            merge_covar(state, group_ids, num_groups)?
        }
    })
}

// --- serial group-id assignment (the per-morsel grouping core; the parallel
// hash-radix combine above reuses these fast-path key hashers) ------------------

/// Merge `(sum, compensation)` states: compensated-add the sums, and carry the
/// compensations along so nothing that was already corrected for is lost.
fn merge_kahan(
    sums: &ArrayRef,
    comps: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
) -> Vec<ArrayRef> {
    use std::sync::Arc;

    use arrow::array::{Array, AsArray, Float64Array};
    use arrow::datatypes::Float64Type;

    let (s, c) = (
        sums.as_primitive::<Float64Type>(),
        comps.as_primitive::<Float64Type>(),
    );
    let mut out_sum = vec![0f64; num_groups];
    let mut out_comp = vec![0f64; num_groups];
    let mut valid = vec![false; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if s.is_null(i) {
            continue;
        }
        let g = g as usize;
        crate::agg::accum::neumaier_add(&mut out_sum[g], &mut out_comp[g], s.value(i));
        out_comp[g] += c.value(i);
        valid[g] = true;
    }
    let mask = |vals: Vec<f64>| -> ArrayRef {
        Arc::new(Float64Array::from_iter(
            vals.into_iter()
                .zip(valid.iter())
                .map(|(v, ok)| ok.then_some(v)),
        ))
    };
    vec![mask(out_sum), mask(out_comp)]
}

/// Partial rows each radix partition needs before the parallel regroup earns its setup.
///
/// The crossover is a function of rows *per partition*, not of the total: the parallel path's
/// overhead is per-partition (a bucket list, a gather, a hash table, an output array), while
/// the serial path's is per-row and single-threaded. Measured on a 92-core box over
/// ClickBench `hits` (min-of-5 wall, whole query):
///
/// | partial rows | serial | parallel |
/// |--------------|-------:|---------:|
/// | 14 k (`GROUP BY RegionID`)   | 2.5 ms  | 5.2 ms |
/// | 23 k (`GROUP BY SearchPhrase`)| 7.0 ms | 5.5 ms |
/// | 35 k (`GROUP BY Title`)      | 16.0 ms | 12.4 ms |
/// | 182 k (`GROUP BY URL`, filtered) | 49.3 ms | 15.1 ms |
///
/// The turn is just under 23 k there — `92 × 256` — and the 182 k row is why this matters: a
/// fixed 200 k threshold left a string group-by *just* under it on the serial merge, paying
/// 38 ms of single-threaded `assign_groups` for work the parallel path does in 5.
const MIN_ROWS_PER_RADIX_PARTITION: usize = 256;

/// Radix partitions a `combine` fans out over, sized by the group count the merge is
/// expected to produce.
///
/// One partition per core is the floor — the independent group-and-merge tasks must fill the
/// pool — and it is also the right answer for most aggregates, because their group tables
/// already fit in cache. It is the wrong answer for the ones that do not: past a few hundred
/// thousand groups per partition every probe misses, and splitting finer is worth up to 1.35x
/// (see the table on `crate::agg::combine_sized`).
///
/// **A flat width cannot serve both ends, and trying was measured.** Oversubscribing 4x
/// unconditionally read as 1.2-1.6x on high-cardinality shapes and **0.85-0.90x** on low ones
/// in a whole-query A/B, consistently, over six paired rounds. So the width is a function of
/// the estimate the executor measured rather than of the machine alone; dividing by
/// `GROUPS_PER_RADIX_PARTITION` lands on the measured best or within a few percent at every
/// cardinality in that table, and — because the divisor is far above any small aggregate's
/// group count — leaves everything below ~500 k groups on a 16-core box at exactly the
/// one-per-core width it had before.
///
/// `estimated_groups == 0` means the caller did not measure; that keeps one per core.
///
/// The 512 ceiling caps per-partition setup on huge boxes; the floor of 2 keeps the path
/// meaningful on a single-core one.
pub(crate) fn radix_partitions(estimated_groups: usize) -> usize {
    let per_core = rayon::current_num_threads().clamp(2, 512);
    estimated_groups
        .div_ceil(GROUPS_PER_RADIX_PARTITION)
        .clamp(per_core, RADIX_PARTITIONS_MAX)
        .next_power_of_two()
        .min(RADIX_PARTITIONS_MAX)
}

/// Groups one radix partition may hold before its table stops being cache-resident. Shared in
/// spirit with `bc_interp::agg_par::GROUPS_PER_PARTITION` (the partition path's twin) and
/// calibrated the same way — the two paths gather differently, so the numbers are measured
/// separately rather than assumed equal.
const GROUPS_PER_RADIX_PARTITION: usize = 32_768;

/// Ceiling on the regroup width: every measured cardinality regresses past this.
const RADIX_PARTITIONS_MAX: usize = 512;

/// The machine-derived partial-row count above which `combine` regroups in parallel: enough
/// rows to keep one-per-core partitions busy. Resolved here, beside the width it is expressed
/// in, so the two cannot drift.
pub(crate) fn radix_parallel_default() -> usize {
    // Expressed in one-per-core partitions deliberately: the question the threshold asks is
    // "are there enough rows to keep the pool busy?", which is about the core count, not
    // about the cache-sized width the regroup then runs at. Tying it to `radix_partitions`
    // would let a wide, high-cardinality merge silently raise its own crossover.
    rayon::current_num_threads()
        .clamp(2, 512)
        .saturating_mul(MIN_ROWS_PER_RADIX_PARTITION)
}

/// [`combine`] for partials the caller **knows** hold disjoint key sets — a concatenation,
/// with no regroup.
///
/// `combine` cannot assume disjointness: any two partials may share a key, so it must hash
/// every row to find out. When the partials came from a hash *partitioning* of one relation
/// they provably cannot, and the merge is then exactly "put the rows together" — every group
/// already appears in exactly one input, so there is nothing to reduce. That turns an O(rows)
/// hash-and-gather into a `memcpy` per column.
///
/// # Correctness precondition
///
/// **The caller guarantees that no group key appears in two of `parts`.** This is not
/// checkable here at any sensible cost, and violating it does not error — it emits the same
/// key twice, so a `SUM` splits across two output rows. Only pass partials produced by
/// partitioning one relation on the *whole* group key (`agg_par::partitioned_partials`,
/// `combine_radix_parts`). When in doubt, call [`combine`]: it is slower and always right.
///
/// The relation returned is the one `combine` would return, in a different group order —
/// which is unspecified for a hash aggregate, as it already is across worker counts.
pub fn concat_disjoint(parts: &[Partial]) -> Result<Partial, RuntimeError> {
    assert!(
        !parts.is_empty(),
        "concat_disjoint requires at least one partial"
    );
    if parts.len() == 1 {
        let p = &parts[0];
        return Ok(Partial {
            group_columns: p.group_columns.clone(),
            states: p.states.clone(),
        });
    }
    let group_columns: Vec<ArrayRef> = (0..parts[0].group_columns.len())
        .into_par_iter()
        .map(|k| concat_col(parts.iter().map(|p| &p.group_columns[k])))
        .collect::<Result<_, _>>()?;
    let states: Vec<Vec<ArrayRef>> = (0..parts[0].states.len())
        .map(|a| {
            (0..parts[0].states[a].len())
                .map(|c| concat_col(parts.iter().map(|p| &p.states[a][c])))
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;
    Ok(Partial {
        group_columns,
        states,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array};

    use super::*;

    /// The typed gather must equal `interleave` element for element, **including validity**.
    ///
    /// Nulls are the half that was not covered: the fast path used to refuse a column with
    /// any, so the values loop had never been asked to carry a bitmap. It is also the case
    /// that matters, because an outer join makes every column below it nullable — which is
    /// most of what a TPC-DS query is built from.
    ///
    /// Checked against arrow rather than against expected values, so this cannot drift into
    /// pinning the shortcut's own idea of the answer.
    #[test]
    fn the_typed_gather_equals_interleave_with_and_without_nulls() {
        let n = 500;
        let a_int: ArrayRef = Arc::new(Int64Array::from(
            (0..n)
                .map(|i| (i % 7 != 0).then_some(i as i64))
                .collect::<Vec<_>>(),
        ));
        let b_int: ArrayRef = Arc::new(Int64Array::from(
            (0..n)
                .map(|i| (i % 3 != 0).then_some(-(i as i64)))
                .collect::<Vec<_>>(),
        ));
        let dense: ArrayRef = Arc::new(Int64Array::from((0..n as i64).collect::<Vec<_>>()));
        let floats: ArrayRef = Arc::new(Float64Array::from(
            (0..n)
                .map(|i| (i % 5 != 0).then_some(i as f64 / 4.0))
                .collect::<Vec<_>>(),
        ));

        let part_of: Vec<u32> = (0..n).map(|i| (i % 2) as u32).collect();
        let row_of: Vec<u32> = (0..n).map(|i| ((i * 131) % n) as u32).collect();
        let pairs: Vec<(usize, usize)> = part_of
            .iter()
            .zip(&row_of)
            .map(|(&p, &r)| (p as usize, r as usize))
            .collect();

        for sources in [
            [a_int.clone(), b_int.clone()],   // nulls on both sides
            [dense.clone(), a_int.clone()],   // nulls on one side only
            [dense.clone(), dense.clone()],   // no nulls at all
            [floats.clone(), floats.clone()], // a float with nulls
        ] {
            let cols: Vec<&dyn Array> = sources.iter().map(|c| c.as_ref()).collect();
            let want = interleave(&cols, &pairs).unwrap();
            let mut scratch = None;
            let got = gather(&cols, &part_of, &row_of, &mut scratch).unwrap();
            assert_eq!(want.as_ref(), got.as_ref(), "{:?}", cols[0].data_type());
            assert_eq!(want.null_count(), got.null_count());
        }
    }
}
