//! Assign each row of a batch a dense group id — the per-morsel hot path of every hash
//! aggregate, `DISTINCT`, and partitioned window.
//!
//! Typed fast paths avoid arrow's `RowConverter` (a per-row encode + allocation) wherever
//! the key can be hashed and compared in its native representation, and a **dense
//! direct-map** path drops the hash entirely when the key's value range is small.

use arrow::array::{
    Array, ArrayRef, AsArray, GenericBinaryArray, GenericStringArray, Int64Array, UInt32Array,
};
use arrow::datatypes::{
    ArrowNativeType, ArrowPrimitiveType, BinaryType, Int16Type, Int32Type, Int64Type, Int8Type,
    LargeBinaryType, LargeUtf8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type, Utf8Type,
};
use arrow::row::{RowConverter, SortField};
use hashbrown::hash_table::Entry;
use hashbrown::HashTable;

use crate::error::RuntimeError;

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
/// Rows-to-span ratio and absolute cap for the dense direct-map grouping path.
///
/// The map is `span` × `u32`, so the cap bounds it at 4 MiB; the ratio keeps the
/// zero-fill (and the map's cache footprint) proportional to the morsel it groups, so a
/// sparse key like `{0, 1<<20}` never buys a huge map to hold two groups.
const DENSE_SPAN_ROW_FACTOR: usize = 4;
const DENSE_SPAN_MAX: usize = 1 << 20;

/// `(min, span)` when `a`'s value range is dense enough to group by direct indexing,
/// else `None`. Costs one linear min/max pass, which is cheap next to the hash probe it
/// replaces (and is skipped entirely for a nullable key).
fn dense_span<T>(a: &arrow::array::PrimitiveArray<T>, num_rows: usize) -> Option<(isize, usize)>
where
    T: ArrowPrimitiveType,
    T::Native: PartialOrd,
{
    let mut lo = a.value(0);
    let mut hi = lo;
    for i in 1..num_rows {
        let v = a.value(i);
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    // `to_isize` refuses a `u64` past `isize::MAX`, and `checked_sub` a range that
    // overflows — both fall back to the hash path rather than wrap.
    let lo_i = lo.to_isize()?;
    let span = hi.to_isize()?.checked_sub(lo_i)?.checked_add(1)? as usize;
    let budget = num_rows
        .saturating_mul(DENSE_SPAN_ROW_FACTOR)
        .clamp(1024, DENSE_SPAN_MAX);
    (span <= budget).then_some((lo_i, span))
}

/// Single integer-key `assign_groups`.
///
/// Two paths, both assigning group ids in first-seen row order so they produce identical
/// `group_ids`, identical `reps`, and hence identical output:
///
/// - **Dense direct-map** when the key's value range is small (dictionary codes, dense
///   ids, low-cardinality enums — the common `GROUP BY <int id>` shape): index a
///   `Vec<u32>` by `key - min`. No hash, no key comparison, no indirection through
///   `reps` into the values array. Measured (single-threaded, 2 M rows, one `SUM`):
///   8.0 → 4.0 ns/row at 1 group, 11.2 → 5.9 at 1 k, 34.8 → 23.7 at 10 k.
/// - **Hash table** otherwise (sparse or huge-range keys), presized to the worst case so a
///   high-cardinality morsel never pays a rehash cascade.
fn assign_groups_int<T>(
    arr: &ArrayRef,
    num_rows: usize,
) -> Result<(Vec<u32>, usize, Vec<ArrayRef>), RuntimeError>
where
    T: ArrowPrimitiveType,
    T::Native: std::hash::Hash + Eq + PartialOrd,
{
    let a = arr.as_primitive::<T>();
    let mut reps: Vec<u32> = Vec::new(); // group_id -> first-seen row index
    let mut group_ids = Vec::with_capacity(num_rows);

    // A nullable key keeps the hash path: nulls form one SQL group, which the direct map
    // has no slot for.
    if a.null_count() == 0 && num_rows > 0 {
        if let Some((lo, span)) = dense_span::<T>(a, num_rows) {
            let mut map: Vec<u32> = vec![u32::MAX; span];
            for i in 0..num_rows {
                // `to_isize` succeeded for min and max above, so it succeeds for every
                // value between them.
                let slot = &mut map[(a.value(i).to_isize().unwrap_or(lo) - lo) as usize];
                if *slot == u32::MAX {
                    *slot = reps.len() as u32;
                    reps.push(i as u32);
                }
                group_ids.push(*slot);
            }
            let num_groups = reps.len();
            let group_columns = vec![arrow::compute::take(arr, &UInt32Array::from(reps), None)?];
            return Ok((group_ids, num_groups, group_columns));
        }
    }

    let state = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);
    let mut table: HashTable<u32> = HashTable::with_capacity(num_rows.max(1));
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
/// Per-column minima and mixed-radix strides when the *product* of the columns' value
/// ranges fits the dense-map budget, else `None`. All arithmetic goes through `i128` /
/// `checked_mul`, so an overflowing range falls back to the hash path rather than wrapping.
fn dense_multi_span(
    cols: &[&Int64Array],
    num_rows: usize,
) -> Option<(Vec<i64>, Vec<usize>, usize)> {
    let mut lows = Vec::with_capacity(cols.len());
    let mut spans: Vec<usize> = Vec::with_capacity(cols.len());
    for c in cols {
        let mut lo = c.value(0);
        let mut hi = lo;
        for i in 1..num_rows {
            let v = c.value(i);
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        let span = usize::try_from(hi as i128 - lo as i128 + 1).ok()?;
        lows.push(lo);
        spans.push(span);
    }
    let mut total: usize = 1;
    for &s in &spans {
        total = total.checked_mul(s)?;
    }
    let budget = num_rows
        .saturating_mul(DENSE_SPAN_ROW_FACTOR)
        .clamp(1024, DENSE_SPAN_MAX);
    if total > budget {
        return None;
    }
    // Row-major strides: the last column varies fastest.
    let mut strides = vec![1usize; cols.len()];
    for j in (0..cols.len().saturating_sub(1)).rev() {
        strides[j] = strides[j + 1] * spans[j + 1];
    }
    Some((lows, strides, total))
}

fn assign_groups_int64_multi(
    cols: &[&Int64Array],
    group_keys: &[ArrayRef],
    num_rows: usize,
) -> Result<(Vec<u32>, usize, Vec<ArrayRef>), RuntimeError> {
    use std::hash::{BuildHasher, Hasher};

    // Dense composite fast path: when every column's value range is small enough that the
    // *product* of the ranges fits the span budget, the composite key is a mixed-radix
    // index into a direct map — no hashing, no per-row equality walk over the columns.
    // `GROUP BY <flag>, <status>` and `GROUP BY <bucket>, <id>` are exactly this shape.
    // Same first-seen id order as the hash path below, so the output is identical.
    if num_rows > 0 {
        if let Some((lows, strides, span)) = dense_multi_span(cols, num_rows) {
            let mut map: Vec<u32> = vec![u32::MAX; span];
            let mut reps: Vec<u32> = Vec::new();
            let mut group_ids = Vec::with_capacity(num_rows);
            for i in 0..num_rows {
                let mut idx = 0usize;
                for (j, c) in cols.iter().enumerate() {
                    idx += ((c.value(i) as i128 - lows[j] as i128) as usize) * strides[j];
                }
                let slot = &mut map[idx];
                if *slot == u32::MAX {
                    *slot = reps.len() as u32;
                    reps.push(i as u32);
                }
                group_ids.push(*slot);
            }
            let num_groups = reps.len();
            let reps_arr = UInt32Array::from(reps);
            let group_columns = group_keys
                .iter()
                .map(|a| arrow::compute::take(a, &reps_arr, None))
                .collect::<Result<_, _>>()?;
            return Ok((group_ids, num_groups, group_columns));
        }
    }

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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int32Array, Int64Array, UInt64Array};

    use super::*;

    /// Naive first-seen grouping — the semantics both `assign_groups_int` paths must match.
    fn reference(vals: &[Option<i64>]) -> (Vec<u32>, usize, Vec<u32>) {
        let mut seen: Vec<Option<i64>> = Vec::new();
        let mut reps: Vec<u32> = Vec::new();
        let mut ids = Vec::with_capacity(vals.len());
        for (i, v) in vals.iter().enumerate() {
            match seen.iter().position(|s| s == v) {
                Some(g) => ids.push(g as u32),
                None => {
                    ids.push(seen.len() as u32);
                    seen.push(*v);
                    reps.push(i as u32);
                }
            }
        }
        (ids, seen.len(), reps)
    }

    fn check_i64(vals: Vec<Option<i64>>) {
        let arr: ArrayRef = Arc::new(Int64Array::from(vals.clone()));
        let (ids, n, cols) = assign_groups(&[arr.clone()], vals.len()).unwrap();
        let (want_ids, want_n, want_reps) = reference(&vals);
        assert_eq!(ids, want_ids, "group_ids for {vals:?}");
        assert_eq!(n, want_n, "num_groups for {vals:?}");
        let want_cols = arrow::compute::take(&arr, &UInt32Array::from(want_reps), None).unwrap();
        assert_eq!(&cols[0], &want_cols, "group columns for {vals:?}");
    }

    /// Keys 0..999: dense, so the direct-map path runs.
    #[test]
    fn dense_keys_match_reference() {
        let vals: Vec<Option<i64>> = (0..4000).map(|i| Some((i % 1000) as i64)).collect();
        check_i64(vals);
    }

    /// Negative keys exercise the `min` offset.
    #[test]
    fn dense_negative_keys_match_reference() {
        let vals: Vec<Option<i64>> = (0..2000).map(|i| Some((i % 51) as i64 - 25)).collect();
        check_i64(vals);
    }

    /// A single distinct value: span 1.
    #[test]
    fn dense_single_value_matches_reference() {
        check_i64(vec![Some(7); 256]);
    }

    /// A huge, sparse range exceeds the span budget and falls back to the hash path.
    #[test]
    fn sparse_keys_fall_back_and_match_reference() {
        let vals: Vec<Option<i64>> = (0..500).map(|i| Some(i as i64 * 1_000_003)).collect();
        check_i64(vals);
    }

    /// Two far-apart values: span is huge but only 2 groups — must NOT build a giant map.
    #[test]
    fn two_distant_values_fall_back_and_match_reference() {
        let vals: Vec<Option<i64>> = (0..1000)
            .map(|i| Some(if i % 2 == 0 { 0 } else { 1 << 40 }))
            .collect();
        check_i64(vals);
    }

    /// Nulls form one SQL group and keep the hash path.
    #[test]
    fn nullable_keys_match_reference() {
        let vals: Vec<Option<i64>> = (0..600)
            .map(|i| {
                if i % 7 == 0 {
                    None
                } else {
                    Some((i % 13) as i64)
                }
            })
            .collect();
        check_i64(vals);
    }

    /// `i64::MIN`/`i64::MAX` would overflow the span arithmetic; must fall back, not wrap.
    #[test]
    fn extreme_i64_range_falls_back() {
        check_i64(vec![
            Some(i64::MIN),
            Some(i64::MAX),
            Some(0),
            Some(i64::MIN),
        ]);
    }

    /// `u64` values beyond `isize::MAX` cannot map to an index; must fall back.
    #[test]
    fn huge_u64_keys_fall_back() {
        let vals = vec![u64::MAX, 1, u64::MAX, 2];
        let arr: ArrayRef = Arc::new(UInt64Array::from(vals));
        let (ids, n, _) = assign_groups(&[arr], 4).unwrap();
        assert_eq!(ids, vec![0, 1, 0, 2]);
        assert_eq!(n, 3);
    }

    /// Naive first-seen grouping over composite keys.
    fn reference_multi(rows: &[(i64, i64)]) -> (Vec<u32>, usize) {
        let mut seen: Vec<(i64, i64)> = Vec::new();
        let mut ids = Vec::with_capacity(rows.len());
        for r in rows {
            match seen.iter().position(|s| s == r) {
                Some(g) => ids.push(g as u32),
                None => {
                    ids.push(seen.len() as u32);
                    seen.push(*r);
                }
            }
        }
        (ids, seen.len())
    }

    fn check_multi(rows: Vec<(i64, i64)>) {
        let a: ArrayRef = Arc::new(Int64Array::from(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        ));
        let b: ArrayRef = Arc::new(Int64Array::from(
            rows.iter().map(|r| r.1).collect::<Vec<_>>(),
        ));
        let (ids, n, cols) = assign_groups(&[a, b], rows.len()).unwrap();
        let (want_ids, want_n) = reference_multi(&rows);
        assert_eq!(ids, want_ids);
        assert_eq!(n, want_n);
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].len(), want_n);
    }

    /// Small ranges in both columns: the dense composite map runs.
    #[test]
    fn dense_two_int_keys_match_reference() {
        let rows: Vec<(i64, i64)> = (0..3000)
            .map(|i| ((i % 7) as i64, (i % 11) as i64))
            .collect();
        check_multi(rows);
    }

    /// Negative values in both columns exercise the per-column `min` offsets.
    #[test]
    fn dense_two_int_keys_negative_match_reference() {
        let rows: Vec<(i64, i64)> = (0..2000)
            .map(|i| ((i % 5) as i64 - 2, (i % 9) as i64 - 4))
            .collect();
        check_multi(rows);
    }

    /// One wide column makes the product exceed the budget: fall back to the hash path.
    #[test]
    fn sparse_two_int_keys_fall_back_and_match_reference() {
        let rows: Vec<(i64, i64)> = (0..500)
            .map(|i| (i as i64 * 7919, (i % 3) as i64))
            .collect();
        check_multi(rows);
    }

    /// Extreme per-column ranges must not overflow the mixed-radix arithmetic.
    #[test]
    fn extreme_two_int_keys_fall_back() {
        check_multi(vec![(i64::MIN, 0), (i64::MAX, 1), (0, 0), (i64::MIN, 0)]);
    }

    /// A narrower int type takes the same dense path.
    #[test]
    fn dense_i32_keys_match_reference() {
        let vals: Vec<i32> = (0..1000).map(|i| (i % 37) as i32).collect();
        let arr: ArrayRef = Arc::new(Int32Array::from(vals.clone()));
        let (ids, n, _) = assign_groups(&[arr], vals.len()).unwrap();
        let as_i64: Vec<Option<i64>> = vals.iter().map(|&v| Some(v as i64)).collect();
        let (want_ids, want_n, _) = reference(&as_i64);
        assert_eq!(ids, want_ids);
        assert_eq!(n, want_n);
    }
}
