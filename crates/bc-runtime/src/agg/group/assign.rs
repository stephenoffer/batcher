//! Assign each row of a batch a dense group id — the per-morsel hot path of every hash
//! aggregate, `DISTINCT`, and partitioned window.
//!
//! Typed fast paths avoid arrow's `RowConverter` (a per-row encode + allocation) wherever
//! the key can be hashed and compared in its native representation, and a **dense
//! direct-map** path drops the hash entirely when the key's value range is small.

use arrow::array::{
    Array, ArrayRef, AsArray, GenericBinaryArray, GenericByteArray, GenericStringArray, Int64Array,
    UInt32Array,
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
            // A dictionary-encoded key groups on its integer *codes*, not its (repeated)
            // values — the whole point of the encoding. Without this it falls through to the
            // RowConverter path and is ~7x SLOWER than the same column decoded, so a dictionary
            // is a net loss; this makes it the fast path (canonical dicts; else it falls back
            // to decoding). Preserved through the FFI boundary by `normalize` for string values.
            DataType::Dictionary(_, _) => return assign_groups_dict(arr, num_rows),
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
        // All-single-byte fast path: when every key column is a null-free length-1 byte string
        // (the two hottest composite TPC-H group keys — `GROUP BY l_returnflag, l_linestatus`),
        // pack one byte per column into a `u64` with *no* length tags (all lengths are 1, so the
        // bytes alone are injective) and group on the integer. Tight packing keeps the value
        // range small, so `int_group_ids` takes the dense direct-map — no hashing, no probe. The
        // group ids and reps are identical to the packed/hash oracle (distinct byte tuples map to
        // distinct `u64`s). Up to 8 columns fit; anything wider or longer keeps the paths below.
        if group_keys.len() <= 8 {
            if let Some((ids, reps)) = bytes1_multi_group_ids(group_keys, num_rows) {
                let num_groups = reps.len();
                let reps_arr = UInt32Array::from(reps);
                let group_columns = group_keys
                    .iter()
                    .map(|a| arrow::compute::take(a, &reps_arr, None))
                    .collect::<Result<_, _>>()?;
                return Ok((ids, num_groups, group_columns));
            }
        }
        // Packed fixed-width fast path: when the whole composite key fits in 16 bytes (short
        // strings + `Int64`s — e.g. `GROUP BY l_returnflag, l_linestatus`, two 1-char keys),
        // pack each row into one `u128` and hash/compare *that* single value. It replaces the
        // per-row hasher object + per-column `hash_into`/`eq_at` (which re-reads the rep row's
        // columns through their offset buffers) with one register-width key — ~1.3x on the
        // measured 60M-row two-key group-by. Distinct composite keys map to distinct packed
        // values (a length tag per string column keeps a shorter value from aliasing a padded
        // longer one, and fixed per-column slots keep columns from bleeding into each other),
        // so the group ids and representative columns are identical to `assign_groups_multi_raw`
        // — a pure short-circuit. Anything wider keeps the general raw path.
        if let Some(layout) = packed_layout(group_keys) {
            return assign_groups_packed(group_keys, &layout, num_rows);
        }
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
pub(crate) const DENSE_SPAN_ROW_FACTOR: usize = 4;
pub(crate) const DENSE_SPAN_MAX: usize = 1 << 20;

/// `(min, span)` when `a`'s value range is dense enough to group by direct indexing,
/// else `None`. Costs one linear min/max pass, which is cheap next to the hash probe it
/// replaces (and is skipped entirely for a nullable key).
/// The dense-map budget for `n` rows: keeps the map's zero-fill and cache footprint
/// proportional to the input, and bounded at 4 MiB of `u32` slots.
pub(crate) fn dense_budget(n: usize) -> usize {
    n.saturating_mul(DENSE_SPAN_ROW_FACTOR)
        .clamp(1024, DENSE_SPAN_MAX)
}

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
    (span <= dense_budget(num_rows)).then_some((lo_i, span))
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
    let (group_ids, reps) = int_group_ids::<T>(arr.as_primitive::<T>(), num_rows);
    let num_groups = reps.len();
    let group_columns = vec![arrow::compute::take(arr, &UInt32Array::from(reps), None)?];
    Ok((group_ids, num_groups, group_columns))
}

/// Assign a dense group id to every row over an integer key, returning `(group_ids, reps)`.
///
/// `reps[g]` is the first-seen row index of group `g`, so the caller builds the output group
/// column with a single `take` — from the key array (the plain-int path) *or* from a
/// dictionary the key indexes (the dictionary path), which is the only thing the two differ
/// in. Two strategies, both first-seen order: a **dense direct-map** when the value range is
/// small (dictionary codes, low-cardinality ids — no hash, no comparison), else a hash table.
fn int_group_ids<T>(a: &arrow::array::PrimitiveArray<T>, num_rows: usize) -> (Vec<u32>, Vec<u32>)
where
    T: ArrowPrimitiveType,
    T::Native: std::hash::Hash + Eq + PartialOrd,
{
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
            return (group_ids, reps);
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
    (group_ids, reps)
}

/// Single dictionary-key `assign_groups`: group by the dictionary's integer **codes** rather
/// than decoding it to values and hashing those.
///
/// A dictionary column's `keys` are already a dense low-cardinality integer key, so routing
/// them through [`int_group_ids`] hits the same direct-map fast path a plain `GROUP BY <id>`
/// does — and skips the O(rows) decode the FFI boundary would otherwise pay to materialize
/// the values (~36 ms at 6 M rows). The group column is `take`n from the dictionary at each
/// group's first-seen row, then decoded once to the value type (tiny — one row per group).
///
/// Correct only when the dictionary is **canonical** (its values are distinct): otherwise two
/// codes could denote the same value and grouping by code would split one SQL group in two.
/// A non-canonical dictionary — or one with a null code or null value, whose SQL null-group
/// coalescing the code path does not model — falls back to decoding, which is exactly the
/// pre-existing behavior. Both the guard scan and the canonical-values check are over the
/// (small) value array, cheap next to the per-row decode they replace.
fn assign_groups_dict(
    arr: &ArrayRef,
    num_rows: usize,
) -> Result<(Vec<u32>, usize, Vec<ArrayRef>), RuntimeError> {
    let dict = arr.as_any_dictionary();
    let values = dict.values();
    let value_type = values.data_type().clone();
    // Decode-and-fall-back for the cases the code path cannot serve correctly.
    if dict.keys().null_count() != 0 || values.null_count() != 0 || !values_are_distinct(values)? {
        let decoded = arrow::compute::cast(arr, &value_type)?;
        return assign_groups(std::slice::from_ref(&decoded), num_rows);
    }
    // Group by the codes. `keys` is one of the signed/unsigned integer index types.
    let keys = dict.keys();
    macro_rules! by_keys {
        ($T:ty) => {{
            let (group_ids, reps) = int_group_ids::<$T>(keys.as_primitive::<$T>(), num_rows);
            (group_ids, reps)
        }};
    }
    use arrow::datatypes::DataType;
    let (group_ids, reps) = match keys.data_type() {
        DataType::Int8 => by_keys!(Int8Type),
        DataType::Int16 => by_keys!(Int16Type),
        DataType::Int32 => by_keys!(Int32Type),
        DataType::Int64 => by_keys!(Int64Type),
        DataType::UInt8 => by_keys!(UInt8Type),
        DataType::UInt16 => by_keys!(UInt16Type),
        DataType::UInt32 => by_keys!(UInt32Type),
        DataType::UInt64 => by_keys!(UInt64Type),
        _ => {
            // An exotic key type: decode and use the general path.
            let decoded = arrow::compute::cast(arr, &value_type)?;
            return assign_groups(std::slice::from_ref(&decoded), num_rows);
        }
    };
    let num_groups = reps.len();
    // The distinct group-key values, one per group (first-seen row), decoded to the plain
    // value type — small (num_groups rows), so this decode is not the O(rows) one avoided.
    let group_vals = arrow::compute::take(arr, &UInt32Array::from(reps), None)?;
    let group_columns = vec![arrow::compute::cast(&group_vals, &value_type)?];
    Ok((group_ids, num_groups, group_columns))
}

/// Whether every value in `values` is distinct — a canonical dictionary's invariant.
///
/// Computed by group-assigning the (small) value array itself: distinct iff the group count
/// equals the value count. `values` is the dictionary's value list, not the per-row column,
/// so this is cheap next to the per-row decode the dictionary path avoids.
fn values_are_distinct(values: &ArrayRef) -> Result<bool, RuntimeError> {
    let (_ids, n, _cols) = assign_groups(std::slice::from_ref(values), values.len())?;
    Ok(n == values.len())
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
    // Short-string fast path: when every key is null-free and ≤ 7 bytes (the flag / status /
    // short-code columns that dominate low-cardinality TPC-H group-bys — `l_returnflag`,
    // `l_linestatus`, `o_orderpriority`'s prefix), pack `(len, bytes)` into one `u64` and group
    // on that integer. This routes a low-cardinality key straight to `int_group_ids`' dense
    // direct-map — no per-row `ahash` over a byte slice, no hash-table probe. Distinct
    // `(len, bytes)` pairs map to distinct `u64`s (length in the high byte, bytes little-endian
    // in the low 56 bits), so the group ids and first-seen reps are identical to the hash path
    // below — a pure performance short-circuit. The representative group column is still
    // `take`n from the original byte array, so its type and values carry through unchanged.
    if a.null_count() == 0 {
        // Single-byte keys (`l_returnflag`, `l_linestatus` — the two hottest TPC-H group keys)
        // are the extreme case: the values buffer is exactly one contiguous byte per row, so we
        // read it straight as a `&[u8]` and dense-map on the byte value (256 slots). No offset
        // indirection, no `u64` scratch array, no hashing — one pass over `num_rows` bytes.
        if let Some((ids, reps)) = byte1_group_ids::<T>(a, num_rows) {
            let num_groups = reps.len();
            let group_columns = vec![arrow::compute::take(arr, &UInt32Array::from(reps), None)?];
            return Ok((ids, num_groups, group_columns));
        }
        // Short strings (≤ 7 bytes): pack `(len, bytes)` into a `u64` and group on the integer,
        // routing a low-cardinality key to `int_group_ids`' dense direct-map instead of hashing
        // byte slices. Distinct `(len, bytes)` → distinct `u64`, so the groups are identical.
        if let Some(packed) = pack_short_bytes::<T>(a, num_rows) {
            let keys = arrow::array::UInt64Array::from(packed);
            let (group_ids, reps) = int_group_ids::<UInt64Type>(&keys, num_rows);
            let num_groups = reps.len();
            let group_columns = vec![arrow::compute::take(arr, &UInt32Array::from(reps), None)?];
            return Ok((group_ids, num_groups, group_columns));
        }
    }
    let state = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);
    let mut table: HashTable<u32> = HashTable::with_capacity(num_rows.max(1));
    let mut reps: Vec<u32> = Vec::new(); // group_id -> first-seen row index
                                         // Each group's representative bytes, held beside its id. Equality then compares the
                                         // probe value against this slice directly, instead of re-fetching the rep row through
                                         // the array's offsets buffer (`a.value(reps[g])`) — a dependent load into a cold,
                                         // randomly-addressed buffer on every hash-chain step. The slices borrow the array's
                                         // values buffer (stable for the array's lifetime), so this copies nothing. ~25% faster
                                         // on the low-cardinality GROUP BYs that dominate TPC-H (l_returnflag, l_linestatus,
                                         // o_orderpriority), where the same few chains are walked millions of times.
    let mut rep_bytes: Vec<&[u8]> = Vec::new();
    let mut group_ids = Vec::with_capacity(num_rows);
    let mut null_gid: Option<u32> = None;

    for i in 0..num_rows {
        if a.is_null(i) {
            let gid = *null_gid.get_or_insert_with(|| {
                let g = reps.len() as u32;
                reps.push(i as u32);
                rep_bytes.push(&[]); // never compared (the null group skips the table)
                g
            });
            group_ids.push(gid);
            continue;
        }
        let v: &[u8] = a.value(i).as_ref();
        let hash = state.hash_one(v);
        let gid = match table.entry(
            hash,
            |&g| rep_bytes[g as usize] == v,
            |&g| state.hash_one(rep_bytes[g as usize]),
        ) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let gid = reps.len() as u32;
                reps.push(i as u32);
                rep_bytes.push(v);
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

/// Group a null-free byte array whose every value is exactly one byte, or `None` if any value
/// is not length-1 (the caller then tries the wider short-string / hash paths).
///
/// Length-1 means the values buffer is `num_rows` contiguous bytes starting at the array's first
/// offset, so row `i`'s key is simply `bytes[i]`. A 256-slot direct map turns grouping into a
/// single branch-light pass with no hashing, no probing, and no scratch allocation — the fastest
/// possible path for the flag/status columns that dominate low-cardinality TPC-H group-bys.
/// First-seen order and representatives match the hash oracle exactly (each new byte value takes
/// the next group id at its first-seen row).
fn byte1_group_ids<T>(a: &GenericByteArray<T>, num_rows: usize) -> Option<(Vec<u32>, Vec<u32>)>
where
    T: arrow::array::types::ByteArrayType,
{
    if num_rows == 0 {
        return None;
    }
    let offsets = a.value_offsets();
    let first = offsets[0];
    // Every value length-1 ⇔ the span is exactly `num_rows` bytes. `value_offsets` is monotone,
    // so this single subtraction rules out any longer (or empty) value without scanning them.
    let last = offsets[num_rows];
    if last.as_usize().wrapping_sub(first.as_usize()) != num_rows {
        return None;
    }
    let base = first.as_usize();
    let bytes = &a.value_data()[base..base + num_rows];
    let mut slot = [u32::MAX; 256];
    let mut reps: Vec<u32> = Vec::new();
    let mut group_ids = Vec::with_capacity(num_rows);
    for (i, &b) in bytes.iter().enumerate() {
        let s = &mut slot[b as usize];
        if *s == u32::MAX {
            *s = reps.len() as u32;
            reps.push(i as u32);
        }
        group_ids.push(*s);
    }
    Some((group_ids, reps))
}

/// Group a composite key whose every column is a null-free length-1 byte string, or `None` if
/// any column is not that shape. Each row's key is `byte(col0) | byte(col1)<<8 | ...` — one byte
/// per column, no length tags (all lengths are 1). Injective across distinct byte tuples, so the
/// groups match the packed/hash oracle; and the tight packing keeps the value range small, so the
/// downstream `int_group_ids` takes its dense direct-map. Caller gates on ≤ 8 columns.
fn bytes1_multi_group_ids(cols: &[ArrayRef], num_rows: usize) -> Option<(Vec<u32>, Vec<u32>)> {
    use arrow::datatypes::DataType::{Binary, LargeUtf8, Utf8};
    // Each column's contiguous length-1 byte slice (base offset applied), or bail.
    let mut byte_cols: Vec<&[u8]> = Vec::with_capacity(cols.len());
    for a in cols {
        if a.null_count() != 0 {
            return None;
        }
        let (data, offsets_len1): (&[u8], bool) = match a.data_type() {
            Utf8 => len1_bytes(a.as_string::<i32>(), num_rows)?,
            LargeUtf8 => len1_bytes(a.as_string::<i64>(), num_rows)?,
            Binary => len1_bytes(a.as_binary::<i32>(), num_rows)?,
            _ => return None,
        };
        debug_assert!(offsets_len1);
        byte_cols.push(data);
    }
    // Two single-byte keys (the `GROUP BY l_returnflag, l_linestatus` shape — TPC-H Q1) are the
    // dominant case, so serve them with a **single pass**: the two bytes form a 16-bit index
    // `b0 | b1<<8` into a 64 k-slot direct map, exactly as the one-byte path dense-maps 256 slots.
    // This replaces the general path's three passes (build a `u64` key per row, scan it for its
    // min/max span, then map) with one. The 256 KiB map is zeroed per morsel, but that cost is
    // trivial next to the passes it removes and fully parallel across morsels. Distinct byte pairs
    // map to distinct indices, so the groups and first-seen reps are identical to the oracle.
    if let [c0, c1] = byte_cols[..] {
        let mut slot = vec![u32::MAX; 1 << 16];
        let mut reps: Vec<u32> = Vec::new();
        let mut group_ids = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let idx = (c0[i] as usize) | ((c1[i] as usize) << 8);
            let s = &mut slot[idx];
            if *s == u32::MAX {
                *s = reps.len() as u32;
                reps.push(i as u32);
            }
            group_ids.push(*s);
        }
        return Some((group_ids, reps));
    }
    // Three or more single-byte columns: pack one byte per column into a `u64` (injective, since
    // every length is 1) and route to the integer dense-map. A 3+-byte index would need a map too
    // large to zero per morsel, so the packed `int_group_ids` (which dense-spans the actual value
    // range) is the right shape here.
    let mut keys: Vec<u64> = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let mut k = 0u64;
        for (j, bytes) in byte_cols.iter().enumerate() {
            k |= (bytes[i] as u64) << (8 * j);
        }
        keys.push(k);
    }
    let arr = arrow::array::UInt64Array::from(keys);
    Some(int_group_ids::<UInt64Type>(&arr, num_rows))
}

/// The contiguous length-1 values slice of a byte array (offset base applied), or `None` if not
/// every value is exactly one byte. `(slice, true)` on success; the bool documents the invariant.
fn len1_bytes<T>(a: &GenericByteArray<T>, num_rows: usize) -> Option<(&[u8], bool)>
where
    T: arrow::array::types::ByteArrayType,
{
    if num_rows == 0 {
        return None;
    }
    let offsets = a.value_offsets();
    let base = offsets[0].as_usize();
    if offsets[num_rows].as_usize().wrapping_sub(base) != num_rows {
        return None;
    }
    Some((&a.value_data()[base..base + num_rows], true))
}

/// Pack each null-free byte-string ≤ 7 bytes into a `u64` group key, or `None` if any value
/// exceeds 7 bytes (in which case the caller keeps the byte-slice hash path).
///
/// The key is `(len << 56) | little_endian(bytes)`: the length occupies the high byte and the
/// ≤ 7 payload bytes the low 56 bits, so two values collide iff they have the same length and
/// the same bytes — i.e. iff the strings are equal. That injectivity is what lets the integer
/// grouping produce the exact same groups as hashing the slices directly. One linear pass over
/// the offsets bails out the moment a value is too long, so a long-string column pays only a
/// cheap scan before falling back.
fn pack_short_bytes<T>(a: &GenericByteArray<T>, num_rows: usize) -> Option<Vec<u64>>
where
    T: arrow::array::types::ByteArrayType,
{
    let mut out = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let v: &[u8] = a.value(i).as_ref();
        let len = v.len();
        if len > 7 {
            return None;
        }
        let mut key = (len as u64) << 56;
        for (j, &b) in v.iter().enumerate() {
            key |= (b as u64) << (8 * j);
        }
        out.push(key);
    }
    Some(out)
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

/// Total packed width of the whole composite key, or `None` if it does not fit in 16 bytes
/// (so the general raw path handles it). `Int64` occupies 8 bytes; a string/binary column
/// occupies `1 + max_value_len` (a length tag plus its longest value). The per-column max
/// length is one cheap pass over the offsets — far less than the grouping it accelerates.
fn packed_layout(cols: &[ArrayRef]) -> Option<Vec<usize>> {
    use arrow::datatypes::DataType::{Binary, Int64, LargeBinary, LargeUtf8, Utf8};
    let mut widths = Vec::with_capacity(cols.len());
    let mut total = 0usize;
    for a in cols {
        let w = match a.data_type() {
            Int64 => 8,
            Utf8 => {
                1 + bytes_max_len(
                    a.as_string::<i32>().value_data(),
                    a.as_string::<i32>().value_offsets(),
                )
            }
            LargeUtf8 => {
                1 + bytes_max_len(
                    a.as_string::<i64>().value_data(),
                    a.as_string::<i64>().value_offsets(),
                )
            }
            Binary => {
                1 + bytes_max_len(
                    a.as_binary::<i32>().value_data(),
                    a.as_binary::<i32>().value_offsets(),
                )
            }
            LargeBinary => {
                1 + bytes_max_len(
                    a.as_binary::<i64>().value_data(),
                    a.as_binary::<i64>().value_offsets(),
                )
            }
            _ => return None,
        };
        total += w;
        if total > 16 {
            return None;
        }
        widths.push(w);
    }
    Some(widths)
}

/// The longest single value in an offset-encoded byte array (max adjacent offset gap).
fn bytes_max_len<O: arrow::array::OffsetSizeTrait>(_data: &[u8], offsets: &[O]) -> usize {
    offsets
        .windows(2)
        .map(|w| (w[1] - w[0]).as_usize())
        .max()
        .unwrap_or(0)
}

/// Multi-column `assign_groups` for a composite key that packs into 16 bytes. Each row is
/// packed into a `u128` with the fixed per-column `widths` from [`packed_layout`]; grouping
/// then runs on that one value. Result-identical to [`assign_groups_multi_raw`].
fn assign_groups_packed(
    group_keys: &[ArrayRef],
    widths: &[usize],
    num_rows: usize,
) -> Result<(Vec<u32>, usize, Vec<ArrayRef>), RuntimeError> {
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
                _ => unreachable!("packed_layout gates the types"),
            }
        })
        .collect();
    // Slot offset of each column within the 16-byte key.
    let mut offs = Vec::with_capacity(widths.len());
    let mut acc = 0usize;
    for &w in widths {
        offs.push(acc);
        acc += w;
    }
    let pack = |i: usize| -> u128 {
        let mut buf = [0u8; 16];
        for (c, &o) in cols.iter().zip(&offs) {
            match c {
                RawKeyCol::Int(a) => buf[o..o + 8].copy_from_slice(&a.value(i).to_le_bytes()),
                RawKeyCol::Str32(a) => write_bytes(&mut buf, o, a.value(i).as_bytes()),
                RawKeyCol::Str64(a) => write_bytes(&mut buf, o, a.value(i).as_bytes()),
                RawKeyCol::Bin32(a) => write_bytes(&mut buf, o, a.value(i)),
                RawKeyCol::Bin64(a) => write_bytes(&mut buf, o, a.value(i)),
            }
        }
        u128::from_le_bytes(buf)
    };

    let state = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);
    let mut table: HashTable<u32> = HashTable::with_capacity(num_rows.max(1));
    let mut reps: Vec<u32> = Vec::new();
    let mut keys: Vec<u128> = Vec::new();
    let mut group_ids = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let pk = pack(i);
        let hash = state.hash_one(pk);
        let gid = match table.entry(
            hash,
            |&g| keys[g as usize] == pk,
            |&g| state.hash_one(keys[g as usize]),
        ) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let gid = reps.len() as u32;
                reps.push(i as u32);
                keys.push(pk);
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

/// Write a string/binary value into its slot: a length tag then the bytes. The `packed_layout`
/// gate guarantees `1 + v.len()` fits the slot, so the length never exceeds 255.
fn write_bytes(buf: &mut [u8; 16], off: usize, v: &[u8]) {
    buf[off] = v.len() as u8;
    buf[off + 1..off + 1 + v.len()].copy_from_slice(v);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int32Array, Int64Array, StringArray, UInt64Array};

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

    /// The packed short-key path must be identical to the general raw multi-key path.
    #[test]
    fn packed_multikey_matches_raw_path() {
        // Two short string keys (the `l_returnflag, l_linestatus` shape → packs) and a case
        // that would alias without the per-column length tag ("a","bc" vs "ab","c").
        let rf: ArrayRef = Arc::new(StringArray::from(vec![
            "A", "AB", "A", "N", "R", "A", "N", "AB",
        ]));
        let ls: ArrayRef = Arc::new(StringArray::from(vec![
            "BC", "C", "BC", "O", "O", "F", "O", "C",
        ]));
        let keys = vec![rf, ls];
        assert!(packed_layout(&keys).is_some(), "short keys must pack");
        let (pids, pn, pcols) = assign_groups(&keys, 8).unwrap();
        let (rids, rn, rcols) = assign_groups_multi_raw(&keys, 8).unwrap();
        assert_eq!(pids, rids, "packed group_ids diverge from raw");
        assert_eq!(pn, rn, "packed num_groups diverge from raw");
        assert_eq!(pcols, rcols, "packed rep columns diverge from raw");
    }

    /// A key wider than 16 bytes falls back to the raw path (still correct), not the packer.
    #[test]
    fn wide_multikey_falls_back_and_is_correct() {
        let long: ArrayRef = Arc::new(StringArray::from(vec![
            "this-is-a-very-long-category-value-past-sixteen-bytes",
            "another-long-one",
            "this-is-a-very-long-category-value-past-sixteen-bytes",
        ]));
        let k2: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 2, 1]));
        let keys = vec![long, k2];
        assert!(packed_layout(&keys).is_none(), "wide key must not pack");
        let (ids, n, _cols) = assign_groups(&keys, 3).unwrap();
        assert_eq!(ids, vec![0, 1, 0]);
        assert_eq!(n, 2);
    }

    /// A dictionary key must group identically to the same column decoded to plain values —
    /// including a NON-canonical dictionary (a repeated value + an unused entry) and nulls.
    #[test]
    fn dictionary_key_matches_decoded() {
        use arrow::array::{DictionaryArray, Int32Array};
        use arrow::datatypes::Int32Type;
        // Dictionary ["A","B","A","C"]: entry 0 and 2 both decode to "A" (non-canonical),
        // entry 3 ("C") is unused. Codes reference A(0), B(1), A-via-2, B, null, C-unused? no —
        // use codes that exercise the duplicate: rows decode to A,B,A,B,null,A.
        let values = StringArray::from(vec!["A", "B", "A", "C"]);
        let keys = Int32Array::from(vec![Some(0), Some(1), Some(2), Some(1), None, Some(0)]);
        let dict = DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).unwrap();
        let dict_arr: ArrayRef = Arc::new(dict);
        let plain: ArrayRef =
            arrow::compute::cast(&dict_arr, &arrow::datatypes::DataType::Utf8).unwrap();

        let (dids, dn, dcols) = assign_groups(&[dict_arr], 6).unwrap();
        let (pids, pn, pcols) = assign_groups(&[plain], 6).unwrap();
        // Codes 0 and 2 both mean "A", so rows 0,2,5 share a group despite different codes.
        assert_eq!(dids, pids, "dict group_ids diverge from decoded");
        assert_eq!(dn, pn, "dict num_groups diverge from decoded");
        assert_eq!(dcols, pcols, "dict rep columns diverge from decoded");
    }

    /// A mixed int + short-string composite key packs and stays correct.
    #[test]
    fn packed_mixed_int_and_str() {
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![10i64, 20, 10, 10]));
        let strs: ArrayRef = Arc::new(StringArray::from(vec!["x", "y", "x", "z"]));
        let keys = vec![ints, strs];
        assert!(packed_layout(&keys).is_some());
        let (pids, pn, _) = assign_groups(&keys, 4).unwrap();
        let (rids, rn, _) = assign_groups_multi_raw(&keys, 4).unwrap();
        assert_eq!(pids, rids);
        assert_eq!(pn, rn);
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

    /// Naive first-seen grouping over strings — the oracle the short-string u64 packing must
    /// reproduce exactly (same ids, same distinct count, same first-seen representatives).
    fn reference_str(vals: &[&str]) -> (Vec<u32>, usize, Vec<String>) {
        let mut seen: Vec<&str> = Vec::new();
        let mut ids = Vec::with_capacity(vals.len());
        for v in vals {
            match seen.iter().position(|s| s == v) {
                Some(g) => ids.push(g as u32),
                None => {
                    ids.push(seen.len() as u32);
                    seen.push(v);
                }
            }
        }
        (
            ids,
            seen.len(),
            seen.iter().map(|s| s.to_string()).collect(),
        )
    }

    fn check_str(vals: Vec<&str>) {
        let arr: ArrayRef = Arc::new(StringArray::from(vals.clone()));
        let (ids, n, cols) = assign_groups(&[arr], vals.len()).unwrap();
        let (want_ids, want_n, want_reps) = reference_str(&vals);
        assert_eq!(ids, want_ids, "group_ids for {vals:?}");
        assert_eq!(n, want_n, "group count for {vals:?}");
        let got = cols[0].as_any().downcast_ref::<StringArray>().unwrap();
        let got_reps: Vec<String> = (0..got.len()).map(|i| got.value(i).to_string()).collect();
        assert_eq!(got_reps, want_reps, "reps for {vals:?}");
    }

    /// Low-cardinality single-char keys (the `l_returnflag` / `l_linestatus` shape) take the
    /// short-string fast path and match the byte-slice oracle exactly.
    #[test]
    fn short_string_flags_match_reference() {
        check_str(vec!["A", "N", "R", "A", "R", "N", "A", "A", "R"]);
        check_str(vec!["O", "F", "O", "O", "F"]);
    }

    /// Mixed short lengths (≤ 7 bytes) still pack injectively: `"a"` and `"aa"` and `""` are
    /// distinct groups even though their bytes overlap, because the length is in the key.
    #[test]
    fn short_strings_mixed_lengths_stay_distinct() {
        check_str(vec!["", "a", "aa", "a", "aaa", "", "aa", "b", "ab", "ba"]);
    }

    /// A key longer than 7 bytes forces the fallback hash path — which must still be correct.
    #[test]
    fn long_strings_fall_back_and_match_reference() {
        check_str(vec![
            "1-URGENT", "2-HIGH", "1-URGENT", "5-LOW", "2-HIGH", "3-MEDIUM",
        ]);
    }

    /// The packing preserves order-of-first-appearance across a boundary of short and long
    /// values so the two paths never disagree on which representative row is first.
    #[test]
    fn boundary_length_seven_and_eight() {
        check_str(vec!["abcdefg", "abcdefgh", "abcdefg", "abcdefgh"]);
    }

    /// Two single-byte string keys (the `GROUP BY l_returnflag, l_linestatus` shape) take the
    /// tight all-single-byte pack and match a naive two-column first-seen oracle exactly.
    #[test]
    fn two_single_byte_keys_match_reference() {
        let flags = vec!["A", "N", "R", "A", "R", "N", "A", "A", "R", "N"];
        let stat = vec!["O", "F", "O", "F", "O", "O", "O", "F", "F", "F"];
        let a: ArrayRef = Arc::new(StringArray::from(flags.clone()));
        let b: ArrayRef = Arc::new(StringArray::from(stat.clone()));
        let (ids, n, cols) = assign_groups(&[a, b], flags.len()).unwrap();
        // Oracle: first-seen grouping over the (flag, status) pairs.
        let mut seen: Vec<(&str, &str)> = Vec::new();
        let mut want_ids = Vec::new();
        for (f, s) in flags.iter().zip(&stat) {
            match seen.iter().position(|p| p == &(*f, *s)) {
                Some(g) => want_ids.push(g as u32),
                None => {
                    want_ids.push(seen.len() as u32);
                    seen.push((f, s));
                }
            }
        }
        assert_eq!(ids, want_ids);
        assert_eq!(n, seen.len());
        let gf = cols[0].as_any().downcast_ref::<StringArray>().unwrap();
        let gs = cols[1].as_any().downcast_ref::<StringArray>().unwrap();
        let got: Vec<(String, String)> = (0..gf.len())
            .map(|i| (gf.value(i).to_string(), gs.value(i).to_string()))
            .collect();
        let want: Vec<(String, String)> = seen
            .iter()
            .map(|(f, s)| (f.to_string(), s.to_string()))
            .collect();
        assert_eq!(got, want);
    }

    /// A single-byte key mixed with a longer key falls back to the packed/raw path (the tight
    /// all-single-byte pack declines), and the result is still correct.
    #[test]
    fn single_byte_plus_long_key_falls_back() {
        let a: ArrayRef = Arc::new(StringArray::from(vec!["A", "N", "A", "N"]));
        let b: ArrayRef = Arc::new(StringArray::from(vec!["xx", "yy", "xx", "zz"]));
        let (ids, n, _) = assign_groups(&[a, b], 4).unwrap();
        assert_eq!(ids, vec![0, 1, 0, 2]);
        assert_eq!(n, 3);
    }
}
