//! A comparison sort for records ordered by a `u64` key, cut first into buckets by the key's
//! live high bits.
//!
//! Two sorts here end in a plain `sort_unstable` over a few hundred thousand fixed-width
//! records: a float key's `(rank, position)` pairs ([`super::pair_sort_indices`]) and the
//! composite key's position-carrying words ([`super::packed::packed_multi_sort_indices`]).
//! Both exist because a scatter-heavy LSD radix loses to a sequential comparison sort there,
//! and both are then the larger part of their operator: `perf` puts **43%** of a 6M-row
//! `ORDER BY <float>` and **24%** of `ORDER BY <date>, <int>` inside that one `sort_unstable`
//! on a 16-core box, where the sample-sort hands each worker a ~400K-row range.
//!
//! A comparison sort of `n` records costs `n log n` comparisons; most of them separate records
//! whose keys differ in their *high* bits, which one counting pass can settle for all of them at
//! once. So this makes one most-significant-digit pass — the bucket is the key's offset from the
//! range's minimum, shifted down to a few thousand buckets of ~32 records — and finishes each
//! bucket with the same `sort_unstable`, now over a few dozen records. The counting pass reads
//! the records sequentially and the scatter writes each one once, so the pass costs about what
//! two levels of the comparison sort's recursion cost and replaces ten or more.
//!
//! **The result is the one `sort_unstable` produces.** The bucket is a monotone function of the
//! key and the caller's `Ord` orders by the key first, so a record in a lower bucket is smaller
//! than every record in a higher one; within a bucket the records are sorted by their own `Ord`.
//! Both callers' records carry a unique row position, so their `Ord` is a total order with no
//! equal elements and the sorted sequence is unique — there is no tie order for this to change.
//!
//! A skewed key — one outlier stretching the range so most records share a bucket — costs the
//! pass and recurses into the crowded bucket, whose own range is narrower by the bucket width;
//! a bucket of equal keys is handed straight to the comparison sort, which finishes an already
//! ordered run in one linear scan.

/// Records below which the counting pass costs more than the comparison levels it removes.
const MSD_MIN_ROWS: usize = 1 << 12;

/// Records per bucket the pass aims for: small enough that each finishing sort is an insertion
/// sort or a shallow quicksort, large enough that the counters stay cache-resident.
const TARGET_BUCKET_ROWS: usize = 32;

/// Upper bound on the bucket count, `2^16` counters of 8 bytes: 512 KiB, inside L2.
const MAX_BUCKET_BITS: u32 = 16;

/// Sort `items` ascending by their `Ord`, where `key` is order-preserving for it: `key(a) <
/// key(b)` must imply `a < b`. The output is exactly `items.sort_unstable()`'s; see the module
/// docs for why.
pub(super) fn sort_by_high_bits<T, K>(items: &mut [T], key: K)
where
    T: Copy + Ord,
    K: Fn(&T) -> u64 + Copy,
{
    let n = items.len();
    if n < MSD_MIN_ROWS {
        items.sort_unstable();
        return;
    }
    let (lo, hi) = items.iter().fold((u64::MAX, 0u64), |(lo, hi), t| {
        let k = key(t);
        (lo.min(k), hi.max(k))
    });
    if lo == hi {
        items.sort_unstable();
        return;
    }
    let span_bits = u64::BITS - (hi - lo).leading_zeros();
    let wanted = (n / TARGET_BUCKET_ROWS)
        .max(1)
        .ilog2()
        .clamp(4, MAX_BUCKET_BITS);
    let bucket_bits = wanted.min(span_bits);
    let shift = span_bits - bucket_bits;
    let bucket = |t: &T| ((key(t) - lo) >> shift) as usize;

    // `start[b]` is bucket `b`'s first slot; `start[b + 1] - start[b]` its size.
    let buckets = 1usize << bucket_bits;
    let mut start = vec![0usize; buckets + 1];
    for t in items.iter() {
        start[bucket(t) + 1] += 1;
    }
    for b in 0..buckets {
        start[b + 1] += start[b];
    }
    let scratch: Vec<T> = items.to_vec();
    let mut next = start.clone();
    for t in &scratch {
        let b = bucket(t);
        items[next[b]] = *t;
        next[b] += 1;
    }
    drop(scratch);
    for b in 0..buckets {
        let part = &mut items[start[b]..start[b + 1]];
        if part.len() > 1 {
            sort_by_high_bits(part, key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic xorshift, so the inputs are reproducible without a dependency.
    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    fn check_pairs(keys: Vec<u64>) {
        let mut got: Vec<(u64, u32)> = keys
            .iter()
            .enumerate()
            .map(|(i, &k)| (k, i as u32))
            .collect();
        let mut want = got.clone();
        want.sort_unstable();
        sort_by_high_bits(&mut got, |p| p.0);
        assert_eq!(got, want);
    }

    #[test]
    fn random_keys_sort_exactly_like_sort_unstable() {
        for (n, seed) in [(0usize, 1u64), (1, 2), (4_095, 3), (4_096, 4), (100_000, 5)] {
            let mut r = rng(seed);
            check_pairs((0..n).map(|_| r()).collect());
        }
    }

    #[test]
    fn duplicates_skew_and_narrow_ranges_sort_exactly() {
        let mut r = rng(9);
        // Heavy duplication: ties are broken by the position half of the record.
        check_pairs((0..50_000).map(|_| r() % 7).collect());
        // One outlier stretching the range, so nearly everything shares a bucket and recurses.
        let mut skewed: Vec<u64> = (0..50_000).map(|_| 1_000 + r() % 1_000).collect();
        skewed[17] = u64::MAX;
        skewed[18] = 0;
        check_pairs(skewed);
        // All equal, and already ordered.
        check_pairs(vec![42; 10_000]);
        check_pairs((0..10_000).collect());
        check_pairs((0..10_000).rev().collect());
    }

    #[test]
    fn whole_words_sort_exactly() {
        let mut r = rng(11);
        let mut got: Vec<u64> = (0..60_000u64).map(|i| ((r() % 5_000) << 32) | i).collect();
        let mut want = got.clone();
        want.sort_unstable();
        sort_by_high_bits(&mut got, |w| *w);
        assert_eq!(got, want);
    }
}
