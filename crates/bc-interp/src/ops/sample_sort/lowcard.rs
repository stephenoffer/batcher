//! Rank-routing for a **single low-cardinality string sort key**.
//!
//! `ORDER BY <a column with seven values>` is the shape the sample-sort serves worst, and it
//! is not rare — a shipmode, a status, a region, a country code. That design rests on
//! quantile boundaries separating the rows into balanced ranges, and a column with fewer
//! distinct values than there are cores cannot supply them. Everything the sample-sort then
//! does to recover is work this module removes rather than performs:
//!
//! * routing binary-searches ~64 boundaries per row, which on seven distinct values is six
//!   `memcmp`s through the offset buffer to rediscover one of seven answers;
//! * `split_constant_ranges` has to *prove* each oversized bucket constant before it may cut
//!   it up, and the proof reads every row of the range — a full random-access pass over the
//!   column, on the exact shape where every bucket is oversized.
//!
//! Ranking asks the question once per **distinct value** instead: hash each row to a dense id
//! against a map that fits in L1 at this cardinality, order the handful of distinct values
//! among themselves, and hand the caller a per-row bucket that *is* the key's rank. The
//! caller's existing counting-sort scatter then does the rest, and every bucket it produces
//! is constant by construction — so the proof pass is not merely faster, it is not needed.
//!
//! This is the whole-relation, parallel half of the argument [`crate::ops::str_sort::rank_sort_live`]
//! already makes for one range serially.
//!
//! ## Why the routing is exact
//!
//! A bucket is one distinct key value (or the nulls), so equal keys never straddle a bucket
//! and buckets are emitted in final key order — the two properties the sample-sort's
//! concatenation-without-merge relies on. `descending` is baked into the rank, so the caller
//! must **not** reverse; `nulls_first` places the null bucket at either end, and inverts
//! independently of `descending`, exactly as arrow's `SortOptions` specifies. Ties resolve to
//! input order because `bucket_indices` builds each bucket in ascending row order.
//!
//! ## Where it declines
//!
//! Above [`MAX_RANK_DISTINCT`] the bucket count stops being a sensible partitioning, so the
//! caller keeps the sample-sort — which is the right algorithm again there. The pre-check is a strided sample, for the reason `str_sort` gives:
//! the first rows of a partitioned scan are not a sample of the column. A sample that
//! under-estimates costs nothing, because the chunk pass abandons the path the moment the
//! true distinct count passes the cap.

use arrow::array::{Array, ArrayRef, GenericStringArray, OffsetSizeTrait};
use arrow::datatypes::DataType;
use rayon::prelude::*;

/// Distinct values above which ranking stops paying and the caller keeps the sample-sort.
///
/// The bound is the caller's scatter, not this pass: `bucket_indices` allocates one vector
/// per bucket per chunk, so `d` buckets on a 96-core box is `96 x d` short vectors before a
/// row moves. 256 keeps that at ~25,000 and covers the shapes this exists for — a shipmode,
/// a status, a region, a country code are all well inside it.
///
/// It is also where the problem itself stops. The sample-sort's difficulty is that fewer
/// distinct values than ranges leaves it nothing to cut on; past four times the 64 ranges it
/// asks for, its boundaries separate the relation properly and its per-range comparison sort
/// is the right algorithm again.
const MAX_RANK_DISTINCT: usize = 256;

/// Rows sampled to estimate the distinct count before committing to the ranked path.
const SAMPLE_ROWS: usize = 4096;

/// Per-row bucket id ordering `key` under `descending`/`nulls_first`, plus the bucket count —
/// or `None` when `key` is not a string column or holds too many distinct values for ranking
/// to beat comparing.
///
/// Bucket `b` holds exactly the rows sharing one key value (or all the nulls), and the
/// buckets are numbered in final sorted order, so the caller may concatenate them as they
/// come with no merge and no reverse.
pub(crate) fn rank_part_of(
    key: &ArrayRef,
    descending: bool,
    nulls_first: bool,
) -> Option<(Vec<u32>, usize)> {
    match key.data_type() {
        DataType::Utf8 => ranks_of(
            key.as_any().downcast_ref::<GenericStringArray<i32>>()?,
            descending,
            nulls_first,
        ),
        DataType::LargeUtf8 => ranks_of(
            key.as_any().downcast_ref::<GenericStringArray<i64>>()?,
            descending,
            nulls_first,
        ),
        _ => None,
    }
}

/// Whether a strided sample of `arr` holds few enough distinct values for ranking to pay.
fn sample_is_low_cardinality<O: OffsetSizeTrait>(arr: &GenericStringArray<O>) -> bool {
    let n = arr.len();
    let step = (n / SAMPLE_ROWS).max(1);
    let mut seen: ahash::AHashSet<&str> = ahash::AHashSet::new();
    for i in (0..n).step_by(step) {
        if arr.is_null(i) {
            continue;
        }
        seen.insert(arr.value(i));
        if seen.len() > MAX_RANK_DISTINCT {
            return false;
        }
    }
    true
}

fn ranks_of<O: OffsetSizeTrait>(
    arr: &GenericStringArray<O>,
    descending: bool,
    nulls_first: bool,
) -> Option<(Vec<u32>, usize)> {
    if !sample_is_low_cardinality(arr) {
        return None;
    }
    let n = arr.len();
    let chunk = n.div_ceil(rayon::current_num_threads().max(1)).max(1);

    // Pass 1, parallel: each chunk maps its rows to *chunk-local* dense ids, so no two
    // threads share a map. `u32::MAX` is the reserved code for a null, which no local id can
    // reach — the cap keeps them in the low hundreds.
    let coded: Option<Vec<(Vec<&str>, Vec<u32>)>> = (0..n)
        .into_par_iter()
        .step_by(chunk)
        .map(|start| {
            let end = (start + chunk).min(n);
            let mut ids: ahash::AHashMap<&str, u32> = ahash::AHashMap::new();
            let mut distinct: Vec<&str> = Vec::new();
            let mut codes: Vec<u32> = Vec::with_capacity(end - start);
            for i in start..end {
                if arr.is_null(i) {
                    codes.push(u32::MAX);
                    continue;
                }
                let value = arr.value(i);
                let id = match ids.get(value) {
                    Some(&id) => id,
                    None => {
                        if distinct.len() >= MAX_RANK_DISTINCT {
                            return None; // too many distinct values for ranking to pay
                        }
                        let id = distinct.len() as u32;
                        distinct.push(value);
                        ids.insert(value, id);
                        id
                    }
                };
                codes.push(id);
            }
            Some((distinct, codes))
        })
        .collect();
    let coded = coded?;

    // Merge the chunks' value sets and order them. These are the only byte comparisons this
    // path performs, and there are `d log d` of them for a `d` of 256 at worst.
    let mut all: Vec<&str> = coded.iter().flat_map(|(d, _)| d.iter().copied()).collect();
    all.sort_unstable();
    all.dedup();
    let d = all.len();
    if d > MAX_RANK_DISTINCT {
        return None;
    }
    // The null bucket sits at whichever end `nulls_first` names, and stays there whatever
    // `descending` does to the value ranks.
    let null_rank = if nulls_first { 0u32 } else { d as u32 };
    let value_base = u32::from(nulls_first);

    // Per-chunk `local id -> global bucket`, so pass 2 hashes nothing.
    let remap: Vec<Vec<u32>> = coded
        .iter()
        .map(|(distinct, _)| {
            distinct
                .iter()
                .map(|v| {
                    let pos = all.partition_point(|x| x < v) as u32;
                    value_base + if descending { d as u32 - 1 - pos } else { pos }
                })
                .collect()
        })
        .collect();

    // Pass 2, parallel: translate each chunk's local codes into global buckets, straight into
    // the caller's per-row routing vector.
    let mut part_of = vec![0u32; n];
    part_of
        .par_chunks_mut(chunk)
        .zip(coded.par_iter())
        .zip(remap.par_iter())
        .for_each(|((dst, (_, codes)), map)| {
            for (slot, &c) in dst.iter_mut().zip(codes.iter()) {
                *slot = if c == u32::MAX {
                    null_rank
                } else {
                    map[c as usize]
                };
            }
        });
    Some((part_of, d + 1))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::StringArray;
    use arrow::compute::SortOptions;

    use super::*;
    use crate::ops::str_sort::stable_sort_indices_str;

    /// A key column of `n` rows cycling through `values`.
    fn key(values: &[Option<&str>], n: usize) -> ArrayRef {
        Arc::new(StringArray::from(
            (0..n).map(|i| values[i % values.len()]).collect::<Vec<_>>(),
        ))
    }

    /// Concatenating the buckets in order must reproduce the serial stable permutation —
    /// including ties, which is the property the whole module rests on.
    fn assert_matches_oracle(values: &[Option<&str>], n: usize, opts: SortOptions) {
        let arr = key(values, n);
        let (part_of, buckets) = rank_part_of(&arr, opts.descending, opts.nulls_first)
            .expect("a low-cardinality key should rank");
        let got: Vec<u32> = bc_runtime::shuffle::bucket_indices(&part_of, buckets)
            .into_iter()
            .flatten()
            .collect();
        let oracle = stable_sort_indices_str(&arr, opts).unwrap();
        assert_eq!(got, oracle.values().to_vec(), "opts {opts:?}");
    }

    const SEVEN: [Option<&str>; 7] = [
        Some("AIR"),
        Some("FOB"),
        Some("MAIL"),
        Some("RAIL"),
        Some("REG AIR"),
        Some("SHIP"),
        Some("TRUCK"),
    ];

    #[test]
    fn ranked_routing_matches_the_serial_oracle_in_every_direction() {
        for descending in [false, true] {
            for nulls_first in [false, true] {
                assert_matches_oracle(
                    &SEVEN,
                    200_000,
                    SortOptions {
                        descending,
                        nulls_first,
                    },
                );
            }
        }
    }

    #[test]
    fn nulls_are_placed_and_tie_broken_like_the_oracle() {
        let values = [Some("b"), None, Some("a"), None, Some("c")];
        for descending in [false, true] {
            for nulls_first in [false, true] {
                assert_matches_oracle(
                    &values,
                    200_000,
                    SortOptions {
                        descending,
                        nulls_first,
                    },
                );
            }
        }
    }

    #[test]
    fn a_single_distinct_value_is_the_identity() {
        assert_matches_oracle(
            &[Some("only")],
            200_000,
            SortOptions {
                descending: false,
                nulls_first: false,
            },
        );
    }

    #[test]
    fn an_all_null_key_is_the_identity() {
        assert_matches_oracle(
            &[None],
            200_000,
            SortOptions {
                descending: true,
                nulls_first: true,
            },
        );
    }

    #[test]
    fn a_high_cardinality_key_declines() {
        let owned: Vec<String> = (0..2048).map(|i| format!("v{i:06}")).collect();
        let values: Vec<Option<&str>> = owned.iter().map(|s| Some(s.as_str())).collect();
        assert!(rank_part_of(&key(&values, 200_000), false, false).is_none());
    }

    #[test]
    fn a_non_string_key_declines() {
        let numeric: ArrayRef = Arc::new(arrow::array::Int64Array::from_iter_values(0..1000));
        assert!(rank_part_of(&numeric, false, false).is_none());
    }
}
