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
use arrow::datatypes::ArrowNativeType;
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

/// Most range buckets [`range_buckets`] will cut: bounds the per-chunk offset arrays the
/// counting sort allocates, the way `RADIX_PARTITIONS_MAX` bounds the hash width.
const RANGE_PARTITIONS_MAX: usize = 1 << 14;

/// Bucket ids by **value range** instead of hash, for a single null-free integer key dense
/// enough to direct-map: `(bucket per row, bucket count)`, or `None` to hash as before.
///
/// The radix combine only needs equal keys in one bucket, and a contiguous slice of the value
/// range gives that as surely as a hash does. What it adds is that each bucket's keys then span
/// `span / buckets` values, so `assign_groups` takes its dense direct map inside every bucket
/// (`assign::int_group_ids`) instead of a hash table holding a hash-scattered `1 / buckets` of
/// the groups. That table is what a near-unique integer key paid for: one partition per core
/// over TPC-H Q13's 15M `o_custkey` groups at sf100 is ~1.9M groups a table, each probe a
/// cache miss, and `int_group_ids` was a quarter of the query.
///
/// Taken only when the whole key range is dense against the rows (`DENSE_SPAN_ROW_FACTOR`),
/// so a uniform key fills its buckets; the bucket count is raised until a bucket's span fits
/// the dense map, so a bucket a skewed key overfills only falls back to hashing, inside that
/// bucket. Never fewer buckets than `partitions`, so the parallelism the caller sized is kept.
pub(super) fn range_buckets(
    parts: &[Partial],
    total_rows: usize,
    partitions: usize,
) -> Option<(Vec<u64>, usize)> {
    let first = parts.first()?.group_columns.first()?;
    if parts.iter().any(|p| p.group_columns.len() != 1) {
        return None;
    }
    macro_rules! by_type {
        ($($t:ty),*) => {
            match first.data_type() {
                $(d if d == &<$t>::DATA_TYPE => range_buckets_typed::<$t>(parts, total_rows, partitions),)*
                _ => None,
            }
        };
    }
    by_type!(Int64Type, Int32Type, Int16Type, Int8Type, UInt32Type, UInt16Type, UInt8Type)
}

fn range_buckets_typed<T>(
    parts: &[Partial],
    total_rows: usize,
    partitions: usize,
) -> Option<(Vec<u64>, usize)>
where
    T: ArrowPrimitiveType,
    T::Native: PartialOrd,
{
    let arrays: Vec<&arrow::array::PrimitiveArray<T>> = parts
        .iter()
        .map(|p| p.group_columns[0].as_primitive_opt::<T>())
        .collect::<Option<_>>()?;
    if arrays
        .iter()
        .any(|a| a.null_count() > 0 || a.data_type() != &T::DATA_TYPE)
    {
        return None;
    }
    let (lo, hi) = arrays
        .par_iter()
        .filter(|a| !a.is_empty())
        .map(|a| {
            let (mut lo, mut hi) = (a.value(0), a.value(0));
            for &v in a.values().iter() {
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
            }
            (lo, hi)
        })
        .reduce_with(|x, y| {
            let lo = if y.0 < x.0 { y.0 } else { x.0 };
            let hi = if y.1 > x.1 { y.1 } else { x.1 };
            (lo, hi)
        })?;
    // `to_isize` and the checked arithmetic refuse a range that overflows, as `dense_span` does.
    let lo_i = lo.to_isize()?;
    let span = hi.to_isize()?.checked_sub(lo_i)?.checked_add(1)? as usize;
    if span > total_rows.saturating_mul(super::assign::DENSE_SPAN_ROW_FACTOR) {
        return None;
    }
    let threads = rayon::current_num_threads().max(1);
    let buckets = span
        .div_ceil(super::assign::DENSE_SPAN_MAX)
        .max(partitions)
        .div_ceil(threads)
        .saturating_mul(threads)
        .min(RANGE_PARTITIONS_MAX);
    let width = span.div_ceil(buckets) as u64;
    let mut out = vec![0u64; total_rows];
    let mut rest = out.as_mut_slice();
    let mut slices = Vec::with_capacity(arrays.len());
    for a in &arrays {
        let (head, tail) = rest.split_at_mut(a.len());
        slices.push(head);
        rest = tail;
    }
    // The same wrapping offset `int_group_ids` indexes its map with: every value lies in
    // `[lo, hi]`, so `v - lo` is in `[0, span)` modulo 2^64 for signed and unsigned keys alike.
    slices
        .into_par_iter()
        .zip(arrays.par_iter())
        .for_each(|(dst, a)| {
            for (d, v) in dst.iter_mut().zip(a.values().iter()) {
                *d = (v.as_usize().wrapping_sub(lo_i as usize) as u64) / width;
            }
        });
    Some((out, buckets))
}

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
    // The same fold for a key that holds NULLs, with a presence tag before each column's value
    // so a NULL hashes as itself rather than as whatever bytes sit in its slot. Sound because
    // `null_free` is decided once for the whole relation: every partial of a nullable key takes
    // this path, never the tagless one above, so a key hashes identically wherever it occurs.
    // Before it, one NULL anywhere sent the whole combine through the serial row encoder, which
    // was TPC-DS q47's hottest kernel (its `i_brand`/`i_category` keys carry NULLs).
    if !null_free
        && group_keys.len() >= 2
        && group_keys.iter().all(|a| is_hashable_mixed(a.data_type()))
    {
        return Ok(hash_mixed_nullable(group_keys, num_rows));
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

/// [`hash_mixed`] for columns that may hold NULLs: each column contributes a presence byte and,
/// when present, its value, so every NULL of a column hashes alike and apart from every value.
fn hash_mixed_nullable(group_keys: &[ArrayRef], num_rows: usize) -> Vec<u64> {
    use std::hash::{BuildHasher, Hasher};
    let cols: Vec<(MixedCol, Option<&arrow::buffer::NullBuffer>)> = group_keys
        .iter()
        .map(|k| {
            let col = match k.data_type() {
                DataType::Int64 => MixedCol::Int(k.as_primitive::<Int64Type>().values()),
                DataType::Float64 => {
                    MixedCol::Float(k.as_primitive::<arrow::datatypes::Float64Type>().values())
                }
                DataType::Utf8 => MixedCol::Str32(k.as_string::<i32>()),
                DataType::LargeUtf8 => MixedCol::Str64(k.as_string::<i64>()),
                DataType::Binary => MixedCol::Bin32(k.as_binary::<i32>()),
                DataType::LargeBinary => MixedCol::Bin64(k.as_binary::<i64>()),
                _ => unreachable!("caller gated on is_hashable_mixed"),
            };
            (col, k.nulls())
        })
        .collect();
    (0..num_rows)
        .into_par_iter()
        .map(|i| {
            let mut h = SEED.build_hasher();
            for (c, nulls) in &cols {
                if nulls.is_some_and(|n| n.is_null(i)) {
                    h.write_u8(0);
                } else {
                    h.write_u8(1);
                    c.write(&mut h, i);
                }
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
            if nulls.is_some_and(|n| n.is_null(i)) {
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
            if nulls.is_some_and(|n| n.is_null(i)) {
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array};

    use super::*;
    use crate::agg::{combine_partitioned, finalize, partial, AggCall, AggFunc};

    const FUNCS: [AggFunc; 4] = [AggFunc::Sum, AggFunc::CountStar, AggFunc::Min, AggFunc::Max];

    fn calls(v: &ArrayRef) -> Vec<AggCall> {
        vec![
            AggCall::new(AggFunc::Sum, Some(v.clone())),
            AggCall::new(AggFunc::CountStar, None),
            AggCall::new(AggFunc::Min, Some(v.clone())),
            AggCall::new(AggFunc::Max, Some(v.clone())),
        ]
    }

    fn cell(a: &ArrayRef, i: usize) -> String {
        if let Some(x) = a.as_any().downcast_ref::<Int64Array>() {
            return x.value(i).to_string();
        }
        let x = a.as_any().downcast_ref::<Float64Array>().unwrap();
        format!("{:.6}", x.value(i))
    }

    fn rows(keys: &ArrayRef, aggs: &[ArrayRef]) -> BTreeMap<i64, Vec<String>> {
        let keys = keys.as_primitive::<Int64Type>();
        (0..keys.len())
            .map(|i| (keys.value(i), aggs.iter().map(|a| cell(a, i)).collect()))
            .collect()
    }

    fn partials_of(keys: &ArrayRef, vals: &ArrayRef, chunk: usize) -> Vec<Partial> {
        (0..keys.len() / chunk)
            .map(|c| {
                let (k, v) = (keys.slice(c * chunk, chunk), vals.slice(c * chunk, chunk));
                partial(std::slice::from_ref(&k), &calls(&v), chunk).unwrap()
            })
            .collect()
    }

    /// Range bucketing must give the relation one whole-input aggregate gives, split into
    /// key-disjoint partitions. Keys are dense, negative as well as positive, and scattered so
    /// that every group spans many partials; the whole-input reference groups once and never
    /// reaches the combine, so it cannot share a defect with the path under test.
    #[test]
    fn range_bucketed_combine_is_the_whole_input_aggregate() {
        let n = 40_000usize;
        let keys: ArrayRef = Arc::new(Int64Array::from(
            (0..n as i64)
                .map(|i| (i * 7_919) % 5_000 - 2_500)
                .collect::<Vec<_>>(),
        ));
        let vals: ArrayRef = Arc::new(Int64Array::from(
            (0..n as i64).map(|i| i % 13 - 6).collect::<Vec<_>>(),
        ));
        let partials = partials_of(&keys, &vals, 500);
        let total: usize = partials.iter().map(|p| p.group_columns[0].len()).sum();
        let (buckets, count) = range_buckets(&partials, total, 2).expect("a dense key ranges");
        assert!(count >= 2 && buckets.iter().all(|&b| (b as usize) < count));

        let whole = partial(std::slice::from_ref(&keys), &calls(&vals), n).unwrap();
        let want = rows(&whole.group_columns[0], &finalize(&FUNCS, &whole).unwrap());
        let parts = combine_partitioned(&partials, &FUNCS, 1).unwrap();
        assert!(parts.len() > 1, "the partitioned path did not engage");
        let mut got = BTreeMap::new();
        for p in &parts {
            for (k, row) in rows(&p.group_columns[0], &finalize(&FUNCS, p).unwrap()) {
                assert!(
                    got.insert(k, row).is_none(),
                    "group {k} is in two partitions"
                );
            }
        }
        assert_eq!(got, want);
    }

    /// A null key, or a range too sparse to direct-map, keeps the hash bucketing.
    #[test]
    fn a_null_or_sparse_key_is_not_range_bucketed() {
        let vals: ArrayRef = Arc::new(Int64Array::from(vec![1i64; 1_000]));
        let nullable: ArrayRef = Arc::new(Int64Array::from(
            (0..1_000i64)
                .map(|i| (i % 10 != 0).then_some(i))
                .collect::<Vec<_>>(),
        ));
        let sparse: ArrayRef = Arc::new(Int64Array::from(
            (0..1_000i64).map(|i| i * 1_000_003).collect::<Vec<_>>(),
        ));
        for keys in [nullable, sparse] {
            let partials = partials_of(&keys, &vals, 100);
            assert!(range_buckets(&partials, 1_000, 2).is_none());
        }
    }
}
