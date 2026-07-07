//! Parallel hash-radix combine for the large-input `combine` regroup.
//!
//! Split out of `agg` along the parallel-grouping seam: the serial `assign_groups`
//! stays in the parent (the per-morsel hot path and the correctness reference),
//! while this is the high-cardinality `combine` fast path the executor reaches once the
//! concatenated partials cross the radix-parallel threshold.
//!
//! The win is **parallelizing the merge**, not just the grouping. Hash-radix partitions
//! the concatenated partials by key, so every row of a group lands in one partition;
//! each partition is then grouped *and* merged independently across threads with **no
//! cross-partition merge**, turning the otherwise-serial per-group accumulate scan
//! (which dominates a many-group combine) into a parallel one. A single primitive-int
//! or byte key hashes its native values directly — skipping the `RowConverter` encoding
//! the general multi-key path needs.

use arrow::array::{
    Array, ArrayRef, AsArray, GenericBinaryArray, GenericStringArray, Int64Array, UInt32Array,
};
use arrow::compute::{concat, take};
use arrow::datatypes::{
    ArrowPrimitiveType, BinaryType, DataType, Int16Type, Int32Type, Int64Type, Int8Type,
    LargeBinaryType, LargeUtf8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type, Utf8Type,
};
use arrow::row::{RowConverter, SortField};
use hashbrown::hash_table::Entry;
use hashbrown::HashTable;
use rayon::prelude::*;

use super::{
    accumulate, merge_approx_distinct, merge_approx_quantile, merge_arg_extreme, merge_distinct,
    merge_median, AggFunc,
};
use crate::error::RuntimeError;

// Same seed the serial `assign_groups` uses — bucketing is independent of the seed, but
// sharing it keeps the paths consistent when one is checked against the other.
const SEED: ahash::RandomState = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);

// A fixed hash for null keys so every null row lands in one partition (and thus one
// group). Grouping inside the partition still compares keys, so a non-null value that
// collides here is never conflated with null — only co-location depends on this value.
const NULL_HASH: u64 = 0xa5a5_5a5a_dead_beef;

/// Parallel `combine` regroup via hash-radix partitioning. Returns the merged group-key
/// columns and, per aggregate, its merged state columns — identical to the serial
/// `assign_groups` + `merge_state` path (group *order* differs, which callers treat as
/// unspecified, like any hash aggregate).
///
/// `group_concat` are the concatenated partial group-key columns; `state_concats[a]` are
/// aggregate `a`'s concatenated partial-state columns; both have `total_rows` rows.
pub(super) fn combine_radix(
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
pub(super) fn merge_state(
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

/// Assign each row a dense group id, returning the ids, the group count, and the
/// distinct group-key columns (in first-seen order).
pub(crate) fn assign_groups(
    group_keys: &[ArrayRef],
    num_rows: usize,
) -> Result<(Vec<u32>, usize, Vec<ArrayRef>), RuntimeError> {
    if group_keys.is_empty() {
        // Global aggregate: a single group over all rows.
        return Ok((vec![0; num_rows], 1, Vec::new()));
    }
    // Fast path: a single integer key column hashes its native values directly,
    // skipping the RowConverter encoding pass (a per-row allocation + copy) that the
    // general path needs for multi-column / variable-length / float keys. Integers are
    // exact under raw hashing — floats (NaN, ±0.0) and strings keep the RowConverter,
    // which imposes a correct total order. This is the common GROUP BY <int id> case.
    if group_keys.len() == 1 {
        use arrow::datatypes::DataType;
        let arr = &group_keys[0];
        match arr.data_type() {
            DataType::Int8 => return assign_groups_int::<Int8Type>(arr, num_rows),
            DataType::Int16 => return assign_groups_int::<Int16Type>(arr, num_rows),
            DataType::Int32 => return assign_groups_int::<Int32Type>(arr, num_rows),
            DataType::Int64 => return assign_groups_int::<Int64Type>(arr, num_rows),
            DataType::UInt8 => return assign_groups_int::<UInt8Type>(arr, num_rows),
            DataType::UInt16 => return assign_groups_int::<UInt16Type>(arr, num_rows),
            DataType::UInt32 => return assign_groups_int::<UInt32Type>(arr, num_rows),
            DataType::UInt64 => return assign_groups_int::<UInt64Type>(arr, num_rows),
            // String/binary keys hash their bytes directly, the same win as the integer
            // path — a `GROUP BY <status/category/region>` is exactly this shape.
            DataType::Utf8 => return assign_groups_bytes::<Utf8Type>(arr, num_rows),
            DataType::LargeUtf8 => return assign_groups_bytes::<LargeUtf8Type>(arr, num_rows),
            DataType::Binary => return assign_groups_bytes::<BinaryType>(arr, num_rows),
            DataType::LargeBinary => return assign_groups_bytes::<LargeBinaryType>(arr, num_rows),
            _ => {}
        }
    }
    // Multi-column all-`Int64` fast path: a composite integer key (e.g. DISTINCT
    // `(l_orderkey, l_suppkey)`, or GROUP BY on two id columns) hashes/compares the
    // raw values directly, skipping the `RowConverter` per-row encode + copy the
    // general path runs. Narrow ints normalize to `Int64` at the FFI boundary, so this
    // is the common composite-int-key shape. Gated to null-free columns; a nullable key
    // keeps the `RowConverter` oracle (which encodes nulls with correct SQL grouping).
    // The group ids and first-seen representative columns are identical to that oracle,
    // so this is a pure performance short-circuit.
    if group_keys.len() >= 2
        && group_keys
            .iter()
            .all(|a| a.data_type() == &arrow::datatypes::DataType::Int64 && a.null_count() == 0)
    {
        let cols: Vec<&Int64Array> = group_keys
            .iter()
            .map(|a| a.as_primitive::<Int64Type>())
            .collect();
        return assign_groups_int64_multi(&cols, group_keys, num_rows);
    }
    // Multi-column mixed Int64 / string / binary fast path: hash and compare the raw
    // column values directly, skipping the `RowConverter` per-row encode + allocation the
    // general path runs over every row — the dominant cost of a multi-string GROUP BY /
    // DISTINCT at scale (e.g. `GROUP BY l_returnflag, l_linestatus` over 60M rows, where
    // the single-key byte path already wins but the two-key RowConverter path does not).
    // Gated to null-free columns of these types; anything else (floats, temporals, nested,
    // or a nullable key) keeps the row-encoded oracle. Equal rows hash equally and the
    // element-wise equality check is exact, so the group ids and first-seen representative
    // columns are identical to that oracle — a pure performance short-circuit.
    if group_keys.len() >= 2 && group_keys.iter().all(is_raw_multikey_col) {
        return assign_groups_multi_raw(group_keys, num_rows);
    }
    let fields: Vec<SortField> = group_keys
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    let rows = converter.convert_columns(group_keys)?;

    // Group via a raw hash table keyed by *row index* — we store only the
    // first-seen row of each group and compare encoded rows directly, avoiding
    // the per-row owned-key allocation an `IndexMap<OwnedRow, _>` would incur.
    // Size for the worst case (all rows distinct): the table holds at most
    // `num_rows` entries, so pre-sizing avoids the rehash cascade a small initial
    // capacity forces on a high-cardinality group-by (the hot per-morsel path).
    let state = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);
    let mut table: HashTable<u32> = HashTable::with_capacity(num_rows.max(1));
    let mut reps: Vec<u32> = Vec::new(); // group_id -> first-seen row index
    let mut group_ids = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let row_i = rows.row(i);
        let hash = state.hash_one(row_i);
        let gid = match table.entry(
            hash,
            |&g| rows.row(reps[g as usize] as usize) == row_i,
            |&g| state.hash_one(rows.row(reps[g as usize] as usize)),
        ) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let gid = reps.len() as u32;
                reps.push(i as u32);
                e.insert(gid);
                gid
            }
        };
        group_ids.push(gid);
    }

    let num_groups = reps.len();
    let group_columns = converter.convert_rows(reps.iter().map(|&i| rows.row(i as usize)))?;
    Ok((group_ids, num_groups, group_columns))
}

/// Single-integer-key `assign_groups`: hash the native values directly (no row
/// encoding). Nulls form one group (SQL semantics); the output key column is the
/// representative rows `take`n from the input, so type and the null carry through.
fn assign_groups_int<T>(
    arr: &ArrayRef,
    num_rows: usize,
) -> Result<(Vec<u32>, usize, Vec<ArrayRef>), RuntimeError>
where
    T: ArrowPrimitiveType,
    T::Native: std::hash::Hash + Eq,
{
    let a = arr.as_primitive::<T>();
    let state = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);
    let mut table: HashTable<u32> = HashTable::with_capacity(num_rows.max(1));
    let mut reps: Vec<u32> = Vec::new(); // group_id -> first-seen row index
    let mut group_ids = Vec::with_capacity(num_rows);
    let mut null_gid: Option<u32> = None;

    for i in 0..num_rows {
        if a.is_null(i) {
            let gid = *null_gid.get_or_insert_with(|| {
                let g = reps.len() as u32;
                reps.push(i as u32);
                g
            });
            group_ids.push(gid);
            continue;
        }
        let v = a.value(i);
        let hash = state.hash_one(v);
        // The table holds only non-null groups, so a rep is always a valid value.
        let gid = match table.entry(
            hash,
            |&g| a.value(reps[g as usize] as usize) == v,
            |&g| state.hash_one(a.value(reps[g as usize] as usize)),
        ) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let gid = reps.len() as u32;
                reps.push(i as u32);
                e.insert(gid);
                gid
            }
        };
        group_ids.push(gid);
    }

    let num_groups = reps.len();
    let group_columns = vec![arrow::compute::take(arr, &UInt32Array::from(reps), None)?];
    Ok((group_ids, num_groups, group_columns))
}

/// Single string/binary-key `assign_groups`: hash each value's bytes directly (no row
/// encoding), the byte-keyed analog of [`assign_groups_int`]. Nulls form one group
/// (SQL semantics); the output key column is the representative rows `take`n from the
/// input, so the type and nulls carry through.
fn assign_groups_bytes<T>(
    arr: &ArrayRef,
    num_rows: usize,
) -> Result<(Vec<u32>, usize, Vec<ArrayRef>), RuntimeError>
where
    T: arrow::array::types::ByteArrayType,
    for<'a> &'a T::Native: std::hash::Hash + Eq,
{
    let a = arr.as_bytes::<T>();
    let state = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);
    let mut table: HashTable<u32> = HashTable::with_capacity(num_rows.max(1));
    let mut reps: Vec<u32> = Vec::new(); // group_id -> first-seen row index
    let mut group_ids = Vec::with_capacity(num_rows);
    let mut null_gid: Option<u32> = None;

    for i in 0..num_rows {
        if a.is_null(i) {
            let gid = *null_gid.get_or_insert_with(|| {
                let g = reps.len() as u32;
                reps.push(i as u32);
                g
            });
            group_ids.push(gid);
            continue;
        }
        let v = a.value(i);
        let hash = state.hash_one(v);
        // The table holds only non-null groups, so a rep is always a valid value.
        let gid = match table.entry(
            hash,
            |&g| a.value(reps[g as usize] as usize) == v,
            |&g| state.hash_one(a.value(reps[g as usize] as usize)),
        ) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let gid = reps.len() as u32;
                reps.push(i as u32);
                e.insert(gid);
                gid
            }
        };
        group_ids.push(gid);
    }

    let num_groups = reps.len();
    let group_columns = vec![arrow::compute::take(arr, &UInt32Array::from(reps), None)?];
    Ok((group_ids, num_groups, group_columns))
}

/// Multi-column all-`Int64` `assign_groups`: hash/compare the raw `i64` values of every
/// key column directly, skipping the `RowConverter` encode the general path runs per row.
/// `cols` are the (null-free) `Int64` key columns; `group_keys` the originals the
/// representative group columns are `take`n from (so type carries through). Callers gate
/// this on all-`Int64`, null-free keys — a nullable/other-typed key uses the row-encoded
/// oracle, which this reproduces exactly (same first-seen group ids and reps).
fn assign_groups_int64_multi(
    cols: &[&Int64Array],
    group_keys: &[ArrayRef],
    num_rows: usize,
) -> Result<(Vec<u32>, usize, Vec<ArrayRef>), RuntimeError> {
    use std::hash::{BuildHasher, Hasher};
    let state = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);
    // Hash a row's composite key by folding each column's raw value into one hasher —
    // no per-row allocation (unlike encoding a key tuple), the same values the equality
    // check compares.
    let hash_row = |i: usize| -> u64 {
        let mut h = state.build_hasher();
        for c in cols {
            h.write_i64(c.value(i));
        }
        h.finish()
    };
    let eq_rows = |a: usize, b: usize| -> bool { cols.iter().all(|c| c.value(a) == c.value(b)) };

    let mut table: HashTable<u32> = HashTable::with_capacity(num_rows.max(1));
    let mut reps: Vec<u32> = Vec::new(); // group_id -> first-seen row index
    let mut group_ids = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let hash = hash_row(i);
        let gid = match table.entry(
            hash,
            |&g| eq_rows(reps[g as usize] as usize, i),
            |&g| hash_row(reps[g as usize] as usize),
        ) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let gid = reps.len() as u32;
                reps.push(i as u32);
                e.insert(gid);
                gid
            }
        };
        group_ids.push(gid);
    }

    let num_groups = reps.len();
    let reps_arr = UInt32Array::from(reps);
    let group_columns = group_keys
        .iter()
        .map(|a| arrow::compute::take(a, &reps_arr, None))
        .collect::<Result<_, _>>()?;
    Ok((group_ids, num_groups, group_columns))
}

/// A key column the mixed raw multi-key grouper can hash/compare directly: a null-free
/// `Int64`, `Utf8`/`LargeUtf8`, or `Binary`/`LargeBinary` column.
fn is_raw_multikey_col(a: &ArrayRef) -> bool {
    use arrow::datatypes::DataType::{Binary, Int64, LargeBinary, LargeUtf8, Utf8};
    a.null_count() == 0
        && matches!(
            a.data_type(),
            Int64 | Utf8 | LargeUtf8 | Binary | LargeBinary
        )
}

/// One key column of the mixed raw multi-key grouper, borrowed as its concrete array so
/// the hot loop hashes/compares raw values (no `RowConverter`, no per-row allocation).
enum RawKeyCol<'a> {
    Int(&'a Int64Array),
    Str32(&'a GenericStringArray<i32>),
    Str64(&'a GenericStringArray<i64>),
    Bin32(&'a GenericBinaryArray<i32>),
    Bin64(&'a GenericBinaryArray<i64>),
}

impl RawKeyCol<'_> {
    fn hash_into<H: std::hash::Hasher>(&self, i: usize, h: &mut H) {
        use std::hash::Hash;
        match self {
            RawKeyCol::Int(a) => a.value(i).hash(h),
            RawKeyCol::Str32(a) => a.value(i).as_bytes().hash(h),
            RawKeyCol::Str64(a) => a.value(i).as_bytes().hash(h),
            RawKeyCol::Bin32(a) => a.value(i).hash(h),
            RawKeyCol::Bin64(a) => a.value(i).hash(h),
        }
    }
    fn eq_at(&self, x: usize, y: usize) -> bool {
        match self {
            RawKeyCol::Int(a) => a.value(x) == a.value(y),
            RawKeyCol::Str32(a) => a.value(x) == a.value(y),
            RawKeyCol::Str64(a) => a.value(x) == a.value(y),
            RawKeyCol::Bin32(a) => a.value(x) == a.value(y),
            RawKeyCol::Bin64(a) => a.value(x) == a.value(y),
        }
    }
}

/// Multi-column `assign_groups` for a mix of `Int64` / string / binary keys, hashing and
/// comparing the raw values directly (the caller gates on [`is_raw_multikey_col`]). Folds
/// every column's value into one hasher per row and compares element-wise on collision —
/// same first-seen group ids and representative columns as the `RowConverter` oracle, but
/// without its per-row encode + allocation (the scaling cost on a large multi-string key).
fn assign_groups_multi_raw(
    group_keys: &[ArrayRef],
    num_rows: usize,
) -> Result<(Vec<u32>, usize, Vec<ArrayRef>), RuntimeError> {
    use std::hash::{BuildHasher, Hasher};
    let cols: Vec<RawKeyCol> = group_keys
        .iter()
        .map(|a| {
            use arrow::datatypes::DataType::{Binary, Int64, LargeBinary, LargeUtf8, Utf8};
            match a.data_type() {
                Int64 => RawKeyCol::Int(a.as_primitive::<Int64Type>()),
                Utf8 => RawKeyCol::Str32(a.as_string::<i32>()),
                LargeUtf8 => RawKeyCol::Str64(a.as_string::<i64>()),
                Binary => RawKeyCol::Bin32(a.as_binary::<i32>()),
                LargeBinary => RawKeyCol::Bin64(a.as_binary::<i64>()),
                _ => unreachable!("caller gates on is_raw_multikey_col"),
            }
        })
        .collect();
    let state = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);
    let hash_row = |i: usize| -> u64 {
        let mut h = state.build_hasher();
        for c in &cols {
            c.hash_into(i, &mut h);
        }
        h.finish()
    };
    let eq_rows = |a: usize, b: usize| -> bool { cols.iter().all(|c| c.eq_at(a, b)) };

    let mut table: HashTable<u32> = HashTable::with_capacity(num_rows.max(1));
    let mut reps: Vec<u32> = Vec::new(); // group_id -> first-seen row index
    let mut group_ids = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let hash = hash_row(i);
        let gid = match table.entry(
            hash,
            |&g| eq_rows(reps[g as usize] as usize, i),
            |&g| hash_row(reps[g as usize] as usize),
        ) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let gid = reps.len() as u32;
                reps.push(i as u32);
                e.insert(gid);
                gid
            }
        };
        group_ids.push(gid);
    }

    let num_groups = reps.len();
    let reps_arr = UInt32Array::from(reps);
    let group_columns = group_keys
        .iter()
        .map(|a| arrow::compute::take(a, &reps_arr, None))
        .collect::<Result<_, _>>()?;
    Ok((group_ids, num_groups, group_columns))
}
