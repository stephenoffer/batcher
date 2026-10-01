//! Window ordering for a key tuple narrow enough to pack into one `u64`.
//!
//! [`super::ordered_partitions_by_global_sort`] orders `(partition keys, order keys, row)` and
//! has two spellings: a packed sort for exactly one numeric partition key and one numeric order
//! key, and a `RowConverter` sort for everything else. The second is what a window with a
//! *composite* `ORDER BY` reaches — `PARTITION BY supplier ORDER BY order, line` — and it is
//! the slow one for a reason that has nothing to do with the encode: its inline sort prefix is
//! the row's leading eight bytes, which on that shape are the *partition key*. Every row of a
//! partition shares them, so almost every comparison falls through to a random-access
//! comparison of two encoded rows. Measured on TPC-H `lineitem` (6M rows, 10,000 supplier
//! partitions), that window cost ~530 ms where DuckDB takes ~400 ms.
//!
//! The keys of that shape are integers whose *live* ranges are far narrower than their types:
//! 10,000 suppliers, 6M orders and seven line numbers need 14 + 23 + 3 bits. So this packs the
//! whole tuple into one order-preserving `u64` — a dense partition id in the high bits, then
//! each order key's offset from its measured minimum — and sorts `(packed, row)` pairs, where
//! every comparison is a register compare.
//!
//! ## Why the order is the same one
//!
//! Each order key goes through [`crate::keys::u64_order_keys`], the engine's order-preserving
//! integer for a primitive column (complemented for `DESC`); subtracting the column's minimum
//! preserves that order, and nulls take the slot below the minimum (`NULLS FIRST`) or above the
//! maximum (`NULLS LAST`). Concatenating fixed-width fields compares field by field, most
//! significant first, so the packed word orders exactly as the key tuple does, and the trailing
//! row index breaks every remaining tie the way the general path's comparator does.
//!
//! The partition component only has to keep a partition's rows contiguous — nothing downstream
//! reads the order *between* partitions, since every function scatters back to the row it came
//! from (`assign_partitions` already hands them over in first-seen order). It is the measured
//! range of an integer partition key when every key has one, and otherwise the dense group id
//! the shared grouper assigns, which admits every `GROUP BY` key type (strings included) and
//! agrees with it on what a partition is, nulls included.
//!
//! ## Where it declines
//!
//! An empty `ORDER BY`, an order key with no order-preserving `u64` (strings, decimals,
//! booleans, nested types), or a tuple wider than 64 bits. A decline returns `None` before any
//! grouping work, and the caller runs the general path exactly as before.

use arrow::array::{Array, ArrayRef};
use arrow::compute::SortOptions;
use rayon::prelude::*;

use crate::error::RuntimeError;

/// Rows below which the elementwise passes here stay on the calling thread.
const PARALLEL_MIN_ROWS: usize = 1 << 15;

/// The bit width that holds every value in `0 .. card`, i.e. `ceil(log2(card))`.
fn bits_for(card: u128) -> u32 {
    if card <= 1 {
        0
    } else {
        128 - (card - 1).leading_zeros()
    }
}

/// One key column as offsets from its minimum, with the bit width those offsets need.
///
/// The offsets are computed where they are packed ([`RangeKey::at`]) rather than written back
/// over `ranks`, which would be one more pass over every key for nothing.
struct RangeKey {
    /// The engine's order-preserving `u64` per row (complemented for `DESC`); null slots unread.
    ranks: Vec<u64>,
    nulls: Option<arrow::buffer::NullBuffer>,
    lo: u64,
    /// `1` when nulls take slot `0` (`NULLS FIRST`), so live offsets start above it.
    live_base: u64,
    null_slot: u64,
    bits: u32,
}

impl RangeKey {
    /// Row `i`'s field value, in `0 .. 2^bits`.
    #[inline]
    fn at(&self, i: usize) -> u64 {
        match &self.nulls {
            Some(nb) if nb.is_null(i) => self.null_slot,
            _ => self.ranks[i].wrapping_sub(self.lo) + self.live_base,
        }
    }
}

/// `arr` as order-preserving offsets in `0 .. 2^bits`, or `None` when it has no `u64` order.
///
/// Nulls take one extra slot at the end their placement asks for.
fn range_key(arr: &ArrayRef, opts: SortOptions) -> Option<RangeKey> {
    let ranks = crate::keys::u64_order_keys(arr, opts.descending)?;
    let nulls = arr.nulls().filter(|nb| nb.null_count() > 0).cloned();
    let live = |i: usize| nulls.as_ref().is_none_or(|nb| nb.is_valid(i));
    let fold = |acc: (u64, u64), (i, &v): (usize, &u64)| {
        if live(i) {
            (acc.0.min(v), acc.1.max(v))
        } else {
            acc
        }
    };
    let (lo, hi) = if ranks.len() >= PARALLEL_MIN_ROWS {
        ranks
            .par_iter()
            .enumerate()
            .fold(|| (u64::MAX, 0u64), fold)
            .reduce(|| (u64::MAX, 0u64), |a, b| (a.0.min(b.0), a.1.max(b.1)))
    } else {
        ranks.iter().enumerate().fold((u64::MAX, 0u64), fold)
    };
    let has_nulls = nulls.is_some();
    // An all-null column has no live minimum: every row takes the one null slot.
    let span: u128 = if lo > hi { 0 } else { u128::from(hi - lo) + 1 };
    let card = span + u128::from(has_nulls);
    let bits = bits_for(card);
    if bits > 64 {
        return None;
    }
    Some(RangeKey {
        ranks,
        live_base: u64::from(has_nulls && opts.nulls_first),
        null_slot: if opts.nulls_first { 0 } else { span as u64 },
        nulls,
        lo,
        bits,
    })
}

/// `(packed word, row)` per row: `base` (a dense partition id, when there is one) above
/// `fields`, most significant first. The caller has checked that every width fits in 64 bits.
fn pack(fields: &[RangeKey], num_rows: usize, base: Option<&[u32]>) -> Vec<(u64, u32)> {
    let total: u32 = fields.iter().map(|f| f.bits).sum();
    // A zero-width field holds only zeros, and a shift by the full word width would overflow
    // rather than produce them, so every shift saturates to zero instead.
    let shl = |v: u64, by: u32| v.checked_shl(by).unwrap_or(0);
    let word = |i: usize| {
        let mut w = base.map_or(0, |g| shl(u64::from(g[i]), total));
        let mut shift = total;
        for f in fields {
            shift -= f.bits;
            w |= shl(f.at(i), shift);
        }
        (w, i as u32)
    };
    if num_rows >= PARALLEL_MIN_ROWS {
        (0..num_rows).into_par_iter().map(word).collect()
    } else {
        (0..num_rows).map(word).collect()
    }
}

/// The per-partition ordered row lists for `(partition_keys, order_keys)`, or `None` when the
/// key tuple does not pack into one `u64`. See the module docs for why the order is identical
/// to [`super::ordered_partitions_by_global_sort`]'s general path.
pub(super) fn ordered_partitions_range_packed(
    partition_keys: &[ArrayRef],
    order_keys: &[(ArrayRef, SortOptions)],
    num_rows: usize,
) -> Result<Option<Vec<Vec<usize>>>, RuntimeError> {
    if order_keys.is_empty() || num_rows == 0 || num_rows > u32::MAX as usize {
        return Ok(None);
    }
    let mut order_fields = Vec::with_capacity(order_keys.len());
    let mut order_bits = 0u32;
    for (arr, opts) in order_keys {
        let Some(f) = range_key(arr, *opts) else {
            return Ok(None);
        };
        order_bits += f.bits;
        if order_bits > 64 {
            return Ok(None);
        }
        order_fields.push(f);
    }

    // The partition component: integer keys by their measured ranges when they all have one
    // (no hash table at all), otherwise the shared grouper's dense ids. Asking the grouper is
    // only worth it when a dense id is sure to fit beside the order keys, which `num_rows`
    // bounds without looking at the data.
    let mut part_fields: Vec<RangeKey> = Vec::new();
    let mut groups: Option<Vec<u32>> = None;
    let mut part_bits = 0u32;
    let default_opts = SortOptions::default();
    let all_ranged = partition_keys
        .iter()
        .all(|k| match range_key(k, default_opts) {
            Some(f) if part_bits + f.bits + order_bits <= 64 => {
                part_bits += f.bits;
                part_fields.push(f);
                true
            }
            _ => false,
        });
    if !all_ranged {
        part_fields.clear();
        let id_bits = bits_for(num_rows as u128);
        if id_bits + order_bits > 64 {
            return Ok(None);
        }
        let (ids, n, _) = crate::agg::assign_groups(partition_keys, num_rows)?;
        part_bits = bits_for(n as u128);
        groups = Some(ids);
    }

    let mut fields = part_fields;
    fields.extend(order_fields);
    let mut keyed = pack(&fields, num_rows, groups.as_deref());
    drop(fields);
    drop(groups);
    // `(word, row)` is unique per row, so the unstable sort is a deterministic total order.
    keyed.par_sort_unstable();

    if part_bits == 0 {
        return Ok(Some(vec![keyed.iter().map(|&(_, i)| i as usize).collect()]));
    }
    // `order_bits` may be 64 only when `part_bits` is 0, handled above.
    let part_of = |w: u64| w.checked_shr(order_bits).unwrap_or(0);
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut start = 0usize;
    for pos in 1..=keyed.len() {
        if pos == keyed.len() || part_of(keyed[pos].0) != part_of(keyed[start].0) {
            out.push(keyed[start..pos].iter().map(|&(_, i)| i as usize).collect());
            start = pos;
        }
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Date32Array, Float64Array, Int32Array, Int64Array, StringArray};
    use arrow::row::{RowConverter, SortField};

    use super::*;

    /// The oracle: partitions by encoded partition key, each ordered by a stable sort on the
    /// encoded order keys (ties to input order). Shares no code with the packed path.
    fn oracle(parts: &[ArrayRef], order: &[(ArrayRef, SortOptions)], n: usize) -> Vec<Vec<usize>> {
        let pconv = RowConverter::new(
            parts
                .iter()
                .map(|a| SortField::new(a.data_type().clone()))
                .collect(),
        )
        .unwrap();
        let prows = pconv.convert_columns(parts).unwrap();
        let oconv = RowConverter::new(
            order
                .iter()
                .map(|(a, o)| SortField::new_with_options(a.data_type().clone(), *o))
                .collect(),
        )
        .unwrap();
        let ocols: Vec<ArrayRef> = order.iter().map(|(a, _)| a.clone()).collect();
        let orows = oconv.convert_columns(&ocols).unwrap();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut seen: Vec<usize> = Vec::new();
        for i in 0..n {
            match seen
                .iter()
                .position(|&r| parts.is_empty() || prows.row(r) == prows.row(i))
            {
                Some(g) => groups[g].push(i),
                None => {
                    seen.push(i);
                    groups.push(vec![i]);
                }
            }
        }
        for g in &mut groups {
            g.sort_by(|&a, &b| orows.row(a).cmp(&orows.row(b)));
        }
        groups
    }

    /// Partitions carry no order between them, so both sides are listed by their first row;
    /// the order *within* every partition is compared exactly.
    fn canonical(mut v: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        v.sort_by_key(|p| p.iter().copied().min());
        v
    }

    fn check(parts: &[ArrayRef], order: &[(ArrayRef, SortOptions)], n: usize) {
        let got = ordered_partitions_range_packed(parts, order, n)
            .unwrap()
            .expect("packable");
        let want = oracle(parts, order, n);
        let (got, want) = (canonical(got), canonical(want));
        assert_eq!(got, want);
    }

    fn opts(descending: bool, nulls_first: bool) -> SortOptions {
        SortOptions {
            descending,
            nulls_first,
        }
    }

    #[test]
    fn composite_order_matches_the_row_encoded_sort_in_every_direction() {
        let n = 3_000usize;
        let part: ArrayRef = Arc::new(Int64Array::from_iter_values(
            (0..n as i64).map(|i| (i * 7919) % 37 - 18),
        ));
        let o1: ArrayRef = Arc::new(Int64Array::from_iter(
            (0..n as i64).map(|i| (i % 11 != 3).then_some((i * 31) % 13 - 6)),
        ));
        let o2: ArrayRef = Arc::new(Int32Array::from_iter(
            (0..n as i32).map(|i| (i % 17 != 0).then_some((i * 13) % 5)),
        ));
        for d1 in [false, true] {
            for n1 in [false, true] {
                for d2 in [false, true] {
                    for n2 in [false, true] {
                        let order = [(o1.clone(), opts(d1, n1)), (o2.clone(), opts(d2, n2))];
                        check(std::slice::from_ref(&part), &order, n);
                        check(&[], &order, n);
                    }
                }
            }
        }
    }

    #[test]
    fn floats_and_dates_and_string_partitions_match() {
        let n = 2_000usize;
        // A float's order spans the whole word once both signs appear, so it packs only on its
        // own; the date beside it is what a string partition shares the word with.
        let vals = [
            -1.5,
            0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            2.25,
            0.0,
        ];
        let f: ArrayRef = Arc::new(Float64Array::from_iter(
            (0..n).map(|i| (i % 9 != 4).then_some(vals[i % vals.len()])),
        ));
        let d: ArrayRef = Arc::new(Date32Array::from_iter(
            (0..n as i32).map(|i| (i % 29 != 1).then_some(18_000 + (i * 7) % 400)),
        ));
        let s: ArrayRef = Arc::new(StringArray::from_iter(
            (0..n).map(|i| (i % 13 != 0).then(|| format!("store-{}", i % 23))),
        ));
        for desc in [false, true] {
            for nf in [false, true] {
                check(&[], &[(f.clone(), opts(desc, nf))], n);
                let by_date = [(d.clone(), opts(desc, nf))];
                check(std::slice::from_ref(&s), &by_date, n);
                check(&[s.clone(), d.clone()], &by_date, n);
                // The float cannot share the word with a partition id: it declines.
                let by_float = [(f.clone(), opts(desc, nf))];
                assert!(
                    ordered_partitions_range_packed(std::slice::from_ref(&s), &by_float, n)
                        .unwrap()
                        .is_none()
                );
            }
        }
    }

    #[test]
    fn extremes_one_row_all_null_and_constant_keys() {
        let big: ArrayRef = Arc::new(Int64Array::from(vec![i64::MIN, i64::MAX, 0, i64::MIN]));
        let p: ArrayRef = Arc::new(Int64Array::from(vec![1, 1, 1, 1]));
        let alt: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 1, 2]));
        // A full-width i64 range needs all 64 bits for itself: it packs alone, beside a
        // zero-width constant, and declines with any field that needs a bit.
        check(&[], &[(big.clone(), opts(false, false))], 4);
        check(&[], &[(big.clone(), opts(true, true))], 4);
        check(
            &[],
            &[
                (p.clone(), opts(false, false)),
                (big.clone(), opts(true, false)),
            ],
            4,
        );
        let two = [(big.clone(), opts(false, false)), (alt, opts(false, false))];
        assert!(ordered_partitions_range_packed(&[], &two, 4)
            .unwrap()
            .is_none());
        // A constant partition key needs zero bits and is still one partition.
        check(std::slice::from_ref(&p), &[(big, opts(false, true))], 4);

        let one: ArrayRef = Arc::new(Int64Array::from(vec![Some(5)]));
        check(
            std::slice::from_ref(&one),
            &[(one.clone(), opts(true, true))],
            1,
        );

        let nulls: ArrayRef = Arc::new(Int64Array::from(vec![None::<i64>; 5]));
        let ord: ArrayRef = Arc::new(Int64Array::from(vec![3, 1, 2, 1, 3]));
        check(
            std::slice::from_ref(&nulls),
            &[(nulls.clone(), opts(false, false))],
            5,
        );
        check(std::slice::from_ref(&nulls), &[(ord, opts(true, false))], 5);
    }

    #[test]
    fn non_packable_order_keys_decline() {
        let s: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let i: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
        assert!(ordered_partitions_range_packed(
            std::slice::from_ref(&i),
            &[(s, opts(false, false))],
            2
        )
        .unwrap()
        .is_none());
        assert!(ordered_partitions_range_packed(&[i], &[], 2)
            .unwrap()
            .is_none());
    }
}
