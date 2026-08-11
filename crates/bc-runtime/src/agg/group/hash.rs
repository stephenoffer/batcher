//! Hashing a set of group-key columns to the `u64` the radix combine buckets on.
//!
//! Split out of `combine.rs`, which orchestrates the regroup; this is only the key →
//! hash half of it. The one property everything here exists to preserve is that **equal
//! keys hash equally across every partial**, because the bucketing is what co-locates a
//! group's rows in a single partition. Two things break that silently rather than loudly,
//! and both are decided once for the whole relation rather than per column or per partial:
//! the float canonicalization (`canon_f64`, so `-0.0` and `0.0` cannot land apart) and the
//! null gate (the multi-column fast paths require null-free inputs).
//!
//! `crate::keys` remains the canonical statement of key *identity*; this derives its hashes
//! from that rather than restating the rules.

use arrow::array::{Array, ArrayRef, AsArray};
use arrow::datatypes::{
    ArrowPrimitiveType, BinaryType, DataType, Int16Type, Int32Type, Int64Type, Int8Type,
    LargeBinaryType, LargeUtf8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type, Utf8Type,
};
use arrow::row::{RowConverter, SortField};
use rayon::prelude::*;

use super::{NULL_HASH, SEED};
use crate::agg::Partial;
use crate::error::RuntimeError;
use crate::keys::canon_f64;

/// [`hash_keys_gated`] over the partials' key columns, flattened in partial order.
///
/// Equal keys must hash equally for the bucketing to co-locate them. Hashing each partial
/// separately — rather than a concatenation of them, which would be a full copy of a column
/// the merge never reads as one array — is only sound if the *encoding* is fixed for the whole
/// relation, and one of `hash_keys_gated`'s gates is not a property of a row at all: the
/// multi-column fast paths require every key column to be **null-free**, which is a property
/// of the partial.
///
/// So a partial that happens to hold no null hashed its keys with the raw fold while a partial
/// holding one anywhere in any key column hashed the same key through arrow's row encoder. The
/// two disagree completely, the same key landed in two radix buckets, and buckets merge by
/// plain `concat` on the key-disjoint premise — so nothing ever reconciled them. The result was
/// **duplicate groups with identical keys**, for rows that were not themselves null: a
/// composite `GROUP BY` or `DISTINCT` over a key with a single NULL anywhere silently returned
/// too many rows. Measured on TPC-DS q98's grouping at sf1: 2,581 rows for 2,521 groups.
///
/// `null_free` is therefore decided once, across every partial, and imposed on all of them.
/// This is the same class of defect — and the same fix — as the float canonicalization
/// [`hash_keys_gated`] hoisted for; the null gate was the one left stated per encoder.
pub(super) fn hash_partial_keys(
    parts: &[Partial],
    total_rows: usize,
) -> Result<Vec<u64>, RuntimeError> {
    let null_free = parts
        .iter()
        .all(|p| p.group_columns.iter().all(|c| c.null_count() == 0));
    let per: Vec<Vec<u64>> = parts
        .par_iter()
        .map(|p| {
            let rows = p.group_columns.first().map_or(0, |c| c.len());
            hash_keys_gated(&p.group_columns, rows, null_free)
        })
        .collect::<Result<_, _>>()?;
    let mut out = Vec::with_capacity(total_rows);
    for h in per {
        out.extend_from_slice(&h);
    }
    Ok(out)
}

/// Per-row key hash for bucketing — a single primitive-int or byte key hashes its native
/// values directly (no row encoding); everything else goes through arrow's row encoding.
/// Nulls hash to a fixed sentinel so they co-locate (and thus form one group).
///
/// `null_free` says whether the multi-column raw-fold fast paths may be used, and it must
/// describe the **whole relation being bucketed**, not the columns passed here — otherwise two
/// slices of one relation encode the same key differently and never meet in a radix bucket.
/// It is a parameter rather than something read off `group_keys` for exactly that reason; see
/// [`hash_partial_keys`], which is the only caller and decides it once. Passing `false` when
/// the columns happen to be null-free is always safe (it only costs the row encoder); passing
/// `true` when any of them is not is a correctness bug — the raw fold reads a null slot's
/// arbitrary bytes.
fn hash_keys_gated(
    group_keys: &[ArrayRef],
    num_rows: usize,
    null_free: bool,
) -> Result<Vec<u64>, RuntimeError> {
    // Canonicalize float keys ONCE, up front — the same shape `bucket_of_rows` uses — so
    // every path below (typed fast path, mixed fold, or `RowConverter` fallback) buckets on
    // the bits `assign_groups` grouped by. Stating the policy per-encoder is what let the
    // `RowConverter` fallback drift: arrow's row format is deliberately non-canonical for
    // floats, so a group whose representative is `-0.0` in one partial and `0.0` in another
    // (legal — `assign_groups` takes reps from the original column) hashed into different
    // radix buckets, and buckets merge by plain `concat` on the "key-disjoint" assumption,
    // so the two were never reconciled: two output groups where the oracle returns one.
    // Differing NaN payloads split the same way. It reached the fallback for any composite
    // key mixing a float with a non-`is_hashable_mixed` type, any composite key with a
    // nullable column, and any float nested in a `List`/`Struct` — and only above
    // `RADIX_PARALLEL_THRESHOLD`, so no small test could see it.
    let canon = crate::keys::canonicalize_float_keys(group_keys);
    let group_keys: &[ArrayRef] = canon.as_deref().unwrap_or(group_keys);
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
            // Float bucketing MUST use the same canonical bits `assign` groups by, or a `-0.0`
            // and a `0.0` (one group) would land in different radix partitions and never merge.
            DataType::Float64 => return Ok(hash_f64_canon(arr, num_rows)),
            _ => {}
        }
    }
    // Multi-column all-`Int64` (null-free) fast path: fold each column's raw `i64` into
    // one hasher per row, skipping the `RowConverter` encode the general path runs. This
    // is the composite-int-key regroup (e.g. DISTINCT `(l_orderkey, l_suppkey)`); narrow
    // ints normalize to `Int64` at the FFI boundary. Bucketing only needs equal keys to
    // hash equally, which this preserves — so the merged relation is unchanged.
    if null_free
        && group_keys.len() >= 2
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
    // Multi-column MIXED Int64 / string / binary (null-free) fast path: fold each column's raw
    // value into one hasher per row, in parallel, skipping the `RowConverter` — whose
    // `convert_columns` is a serial per-row byte encode. That encode is the entire cost of a
    // `COUNT(DISTINCT id) GROUP BY flag` combine, which regroups tens of millions of
    // `(flag, id)` partial rows (measured: the DISTINCT ran at ~12% CPU / ~1s, all in this
    // encode). Equal null-free rows fold the same bytes in the same order, so they bucket
    // identically. Nullable keys keep the `RowConverter` (it co-locates nulls into one group).
    if null_free
        && group_keys.len() >= 2
        && group_keys
            .iter()
            .all(|a| a.null_count() == 0 && is_hashable_mixed(a.data_type()))
    {
        return Ok(hash_mixed(group_keys, num_rows));
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

/// Types the null-free mixed-key fast hash handles directly (no `RowConverter`).
fn is_hashable_mixed(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Int64
            | DataType::Float64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary
    )
}

/// One key column, downcast once, feeding its per-row raw value to a hasher.
enum MixedCol<'a> {
    Int(&'a [i64]),
    Float(&'a [f64]),
    Str32(&'a arrow::array::GenericStringArray<i32>),
    Str64(&'a arrow::array::GenericStringArray<i64>),
    Bin32(&'a arrow::array::GenericBinaryArray<i32>),
    Bin64(&'a arrow::array::GenericBinaryArray<i64>),
}

impl MixedCol<'_> {
    #[inline]
    fn write<H: std::hash::Hasher>(&self, h: &mut H, i: usize) {
        match self {
            MixedCol::Int(v) => h.write_i64(v[i]),
            MixedCol::Float(v) => h.write_u64(canon_f64(v[i])),
            MixedCol::Str32(a) => h.write(a.value(i).as_bytes()),
            MixedCol::Str64(a) => h.write(a.value(i).as_bytes()),
            MixedCol::Bin32(a) => h.write(a.value(i)),
            MixedCol::Bin64(a) => h.write(a.value(i)),
        }
    }
}

/// Per-row hash of a null-free mixed Int64/string/binary composite key, in parallel — the
/// `RowConverter`-free bucketing hash for the high-cardinality DISTINCT / many-group combine.
/// Caller has checked every column is null-free and [`is_hashable_mixed`].
fn hash_mixed(group_keys: &[ArrayRef], num_rows: usize) -> Vec<u64> {
    use std::hash::{BuildHasher, Hasher};
    let cols: Vec<MixedCol> = group_keys
        .iter()
        .map(|k| match k.data_type() {
            DataType::Int64 => MixedCol::Int(k.as_primitive::<Int64Type>().values()),
            DataType::Float64 => {
                MixedCol::Float(k.as_primitive::<arrow::datatypes::Float64Type>().values())
            }
            DataType::Utf8 => MixedCol::Str32(k.as_string::<i32>()),
            DataType::LargeUtf8 => MixedCol::Str64(k.as_string::<i64>()),
            DataType::Binary => MixedCol::Bin32(k.as_binary::<i32>()),
            DataType::LargeBinary => MixedCol::Bin64(k.as_binary::<i64>()),
            _ => unreachable!("caller gated on is_hashable_mixed"),
        })
        .collect();
    (0..num_rows)
        .into_par_iter()
        .map(|i| {
            let mut h = SEED.build_hasher();
            for c in &cols {
                c.write(&mut h, i);
            }
            h.finish()
        })
        .collect()
}

/// Per-row hash of a single `Float64` key over its canonical bits (nulls → `NULL_HASH`), so a
/// float key buckets exactly as `assign` groups it. See [`canon_f64`].
fn hash_f64_canon(arr: &ArrayRef, num_rows: usize) -> Vec<u64> {
    let a = arr.as_primitive::<arrow::datatypes::Float64Type>();
    let nulls = a.nulls();
    let values = a.values();
    (0..num_rows)
        .into_par_iter()
        .map(|i| {
            if nulls.map(|n| n.is_null(i)).unwrap_or(false) {
                NULL_HASH
            } else {
                SEED.hash_one(canon_f64(values[i]))
            }
        })
        .collect()
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
