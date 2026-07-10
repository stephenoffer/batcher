//! Parallel hash-radix `combine` regroup for a high-cardinality aggregate.
//!
//! Hash-radix partitions the concatenated partials by key, so every row of a group lands
//! in one partition; each partition is then grouped *and* merged independently across
//! threads with **no cross-partition merge**, turning the otherwise-serial per-group
//! accumulate scan (which dominates a many-group combine) into a parallel one.

use arrow::array::{Array, ArrayRef, AsArray, UInt32Array};
use arrow::compute::{concat, take};
use arrow::datatypes::{
    ArrowPrimitiveType, BinaryType, DataType, Int16Type, Int32Type, Int64Type, Int8Type,
    LargeBinaryType, LargeUtf8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type, Utf8Type,
};
use arrow::row::{RowConverter, SortField};
use rayon::prelude::*;

use super::assign::assign_groups;
use super::{NULL_HASH, SEED};
use crate::agg::{
    accumulate, merge_approx_distinct, merge_approx_quantile, merge_arg_extreme, merge_distinct,
    merge_median, AggFunc,
};
use crate::error::RuntimeError;

/// Parallel `combine` regroup via hash-radix partitioning. Returns the merged group-key
/// columns and, per aggregate, its merged state columns — identical to the serial
/// `assign_groups` + `merge_state` path (group *order* differs, which callers treat as
/// unspecified, like any hash aggregate).
///
/// `group_concat` are the concatenated partial group-key columns; `state_concats[a]` are
/// aggregate `a`'s concatenated partial-state columns; both have `total_rows` rows.
pub(crate) fn combine_radix(
    group_concat: &[ArrayRef],
    state_concats: &[Vec<ArrayRef>],
    funcs: &[AggFunc],
    total_rows: usize,
    partitions: usize,
) -> Result<(Vec<ArrayRef>, Vec<Vec<ArrayRef>>), RuntimeError> {
    // Bin row indices by `hash(key) % partitions` so equal keys co-locate in one bucket.
    let hashes = hash_keys(group_concat, total_rows)?;
    // Parallel stable counting-sort into the per-bucket index lists: each row-range chunk
    // bins its rows independently, then bucket `b`'s global list is the chunks' `b`-lists
    // concatenated in chunk order. A serial single-pass bin of a 6 M-row concat was the
    // dominant cost of a high-cardinality combine (~60 ms); this spreads it across cores.
    // Per-bucket order is unspecified for a hash aggregate, so any consistent order is fine.
    let buckets: Vec<Vec<u32>> = {
        let nthreads = rayon::current_num_threads().max(1);
        let chunk = total_rows.div_ceil(nthreads).max(1);
        let per_chunk: Vec<Vec<Vec<u32>>> = hashes
            .par_chunks(chunk)
            .enumerate()
            .map(|(ci, slice)| {
                let base = (ci * chunk) as u32;
                let mut b: Vec<Vec<u32>> = vec![Vec::new(); partitions];
                for (j, &h) in slice.iter().enumerate() {
                    b[(h % partitions as u64) as usize].push(base + j as u32);
                }
                b
            })
            .collect();
        (0..partitions)
            .into_par_iter()
            .map(|p| {
                let total: usize = per_chunk.iter().map(|c| c[p].len()).sum();
                let mut out = Vec::with_capacity(total);
                for c in &per_chunk {
                    out.extend_from_slice(&c[p]);
                }
                out
            })
            .collect()
    };

    // Each partition groups + merges independently — its keys appear in no other
    // partition, so its merged groups are final and a plain concat is the whole result.
    let per: Vec<(Vec<ArrayRef>, Vec<Vec<ArrayRef>>)> = buckets
        .par_iter()
        .map(|idx| -> Result<_, RuntimeError> {
            let ti = UInt32Array::from(idx.clone());
            let keys_p: Vec<ArrayRef> = group_concat
                .iter()
                .map(|c| take(c.as_ref(), &ti, None))
                .collect::<Result<_, _>>()?;
            let (local_ids, n_local, group_cols_p) = assign_groups(&keys_p, idx.len())?;
            let mut states_p = Vec::with_capacity(funcs.len());
            for (a, &func) in funcs.iter().enumerate() {
                let state_p: Vec<ArrayRef> = state_concats[a]
                    .iter()
                    .map(|c| take(c.as_ref(), &ti, None))
                    .collect::<Result<_, _>>()?;
                states_p.push(merge_state(func, &state_p, &local_ids, n_local)?);
            }
            Ok((group_cols_p, states_p))
        })
        .collect::<Result<_, _>>()?;

    // Concatenate partition outputs (key-disjoint → concat == merge). Fan the per-column
    // concats across cores — on a high-cardinality distinct/group-by these output columns
    // are millions of rows and the concat is a second full copy that otherwise runs serial.
    let group_columns: Vec<ArrayRef> = (0..group_concat.len())
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

/// Per-row key hash for bucketing — a single primitive-int or byte key hashes its native
/// values directly (no row encoding); everything else goes through arrow's row encoding.
/// Nulls hash to a fixed sentinel so they co-locate (and thus form one group).
fn hash_keys(group_keys: &[ArrayRef], num_rows: usize) -> Result<Vec<u64>, RuntimeError> {
    if group_keys.len() == 1 {
        let arr = &group_keys[0];
        match arr.data_type() {
            DataType::Int8 => return Ok(hash_primitive::<Int8Type>(arr, num_rows)),
            DataType::Int16 => return Ok(hash_primitive::<Int16Type>(arr, num_rows)),
            DataType::Int32 => return Ok(hash_primitive::<Int32Type>(arr, num_rows)),
            DataType::Int64 => return Ok(hash_primitive::<Int64Type>(arr, num_rows)),
            DataType::UInt8 => return Ok(hash_primitive::<UInt8Type>(arr, num_rows)),
            DataType::UInt16 => return Ok(hash_primitive::<UInt16Type>(arr, num_rows)),
            DataType::UInt32 => return Ok(hash_primitive::<UInt32Type>(arr, num_rows)),
            DataType::UInt64 => return Ok(hash_primitive::<UInt64Type>(arr, num_rows)),
            DataType::Utf8 => return Ok(hash_bytes::<Utf8Type>(arr, num_rows)),
            DataType::LargeUtf8 => return Ok(hash_bytes::<LargeUtf8Type>(arr, num_rows)),
            DataType::Binary => return Ok(hash_bytes::<BinaryType>(arr, num_rows)),
            DataType::LargeBinary => return Ok(hash_bytes::<LargeBinaryType>(arr, num_rows)),
            _ => {}
        }
    }
    // Multi-column all-`Int64` (null-free) fast path: fold each column's raw `i64` into
    // one hasher per row, skipping the `RowConverter` encode the general path runs. This
    // is the composite-int-key regroup (e.g. DISTINCT `(l_orderkey, l_suppkey)`); narrow
    // ints normalize to `Int64` at the FFI boundary. Bucketing only needs equal keys to
    // hash equally, which this preserves — so the merged relation is unchanged.
    if group_keys.len() >= 2
        && group_keys
            .iter()
            .all(|a| a.data_type() == &DataType::Int64 && a.null_count() == 0)
    {
        use std::hash::{BuildHasher, Hasher};
        let cols: Vec<&arrow::array::Int64Array> = group_keys
            .iter()
            .map(|a| a.as_primitive::<Int64Type>())
            .collect();
        return Ok((0..num_rows)
            .into_par_iter()
            .map(|i| {
                let mut h = SEED.build_hasher();
                for c in &cols {
                    h.write_i64(c.value(i));
                }
                h.finish()
            })
            .collect());
    }
    let fields: Vec<SortField> = group_keys
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    let rows = converter.convert_columns(group_keys)?;
    Ok((0..num_rows)
        .into_par_iter()
        .map(|i| SEED.hash_one(rows.row(i)))
        .collect())
}

fn hash_primitive<T>(arr: &ArrayRef, num_rows: usize) -> Vec<u64>
where
    T: ArrowPrimitiveType,
    T::Native: std::hash::Hash + Sync,
{
    let a = arr.as_primitive::<T>();
    let nulls = a.nulls();
    let values = a.values();
    (0..num_rows)
        .into_par_iter()
        .map(|i| {
            if nulls.map(|n| n.is_null(i)).unwrap_or(false) {
                NULL_HASH
            } else {
                SEED.hash_one(values[i])
            }
        })
        .collect()
}

fn hash_bytes<T>(arr: &ArrayRef, num_rows: usize) -> Vec<u64>
where
    T: arrow::array::types::ByteArrayType,
    for<'a> &'a T::Native: std::hash::Hash,
{
    let a = arr.as_bytes::<T>();
    (0..num_rows)
        .into_par_iter()
        .map(|i| {
            if a.is_null(i) {
                NULL_HASH
            } else {
                SEED.hash_one(a.value(i))
            }
        })
        .collect()
}

/// Concatenate a sequence of arrays (the per-partition outputs) into one.
fn concat_col<'a>(arrs: impl Iterator<Item = &'a ArrayRef>) -> Result<ArrayRef, RuntimeError> {
    let owned: Vec<&dyn Array> = arrs.map(|a| a.as_ref()).collect();
    Ok(concat(&owned)?)
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
        | AggFunc::Mode
        | AggFunc::Histogram => {
            vec![merge_median(&state[0], group_ids, num_groups)?]
        }
        AggFunc::Min => accumulate(AggFunc::Min, Some(&state[0]), group_ids, num_groups)?,
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
        // (sum, sumsq, count) all merge by summing.
        AggFunc::Var | AggFunc::Stddev => (0..3)
            .map(|c| {
                accumulate(AggFunc::Sum, Some(&state[c]), group_ids, num_groups)
                    .map(|mut v| v.swap_remove(0))
            })
            .collect::<Result<Vec<_>, _>>()?,
        // The sum-of-powers state columns (5 for skew/kurt, 6 for covar/corr) all
        // merge by summing — the property that makes these mergeable.
        AggFunc::Skewness | AggFunc::Kurtosis => sum_each_column(state, group_ids, num_groups)?,
        AggFunc::CovarPop | AggFunc::CovarSamp | AggFunc::Corr => {
            sum_each_column(state, group_ids, num_groups)?
        }
    })
}

/// Merge each partial-state column by summing it across partitions (the shared
/// reducer for every sum-of-powers aggregate). Column 0 is an Int64 count; summing
/// it stays Int64, the Float64 moment columns stay Float64.
fn sum_each_column(
    state: &[ArrayRef],
    group_ids: &[u32],
    num_groups: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    (0..state.len())
        .map(|c| {
            accumulate(AggFunc::Sum, Some(&state[c]), group_ids, num_groups)
                .map(|mut v| v.swap_remove(0))
        })
        .collect()
}

// --- serial group-id assignment (the per-morsel grouping core; the parallel
// hash-radix combine above reuses these fast-path key hashers) ------------------
