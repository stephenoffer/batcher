//! Exact key-range membership bitmap — the probe pre-filter for a mid-range `Int64` build key.
//!
//! A hash probe for a key the build side does not hold still pays the whole lookup: two
//! multiplies to hash it, a SIMD tag compare, and the branch mispredictions of a table walk
//! that ends in "absent". On a selective join that is nearly every row. TPC-H q17 probes all
//! 6M `lineitem` rows against the 204 parts of one brand and container, twice, and measured
//! single-threaded **76% of the query's CPU sat in that lookup at ~11 ns a row** — against a
//! table small enough to live in L1. The table was never the cost; asking it was.
//!
//! [`super::dense::DenseHeads`] would answer that question with no hashing, but it spends 32
//! bits per slot of key range and so refuses any build whose keys spread wider than a few
//! slots per row — which is exactly the *filtered* dimension (204 parts spread over 200,000
//! keys) a selective join builds on. A bitmap spends **one** bit per slot, so the same range
//! costs 25 KB, and a probe is a bounds check plus one bit test with no hash at all.
//!
//! It is **exact**, not a bloom: a bit is set for precisely the non-null build keys, so a
//! clear bit proves the key absent and a set bit proves it present. It only ever skips a
//! lookup whose answer was already `None`, so it can never change which rows pair.

use super::{IndexBuf, JoinKeys, JoinTable, JoinType, Prefilter};

/// Probe rows per selection block: small enough that the selection vector stays in L1.
const BLOCK: usize = 1024;

/// The widest key range given a bitmap: 2^23 slots, a 1 MiB bitmap. Past L2 a bit test is a
/// cache miss of its own, and the hash lookup it saves is not much more.
const MAX_SPAN: usize = 1 << 23;

/// A bitmap over the key range `lo..lo + span`, one bit per possible key.
pub(super) struct KeyBits {
    lo: i64,
    span: u64,
    words: Vec<u64>,
}

impl KeyBits {
    /// A bitmap over the non-null `keys`, or `None` when their range is wider than
    /// [`MAX_SPAN`] (or there are no non-null keys) and the hash table stands alone.
    ///
    /// `lo`/`span` are the non-null keys' bounds, already computed by the caller for the
    /// dense-map admission decision, so the keys are not scanned for them twice.
    pub(super) fn build(keys: &[i64], null: &[bool], lo: i64, span: usize) -> Option<Self> {
        if span == 0 || span > MAX_SPAN {
            return None;
        }
        let mut words = vec![0u64; span.div_ceil(64)];
        for (&k, &is_null) in keys.iter().zip(null) {
            if !is_null {
                // The bounds cover every non-null key, so `k - lo` lies in `0..span`.
                let slot = (k - lo) as usize;
                words[slot >> 6] |= 1u64 << (slot & 63);
            }
        }
        Some(Self {
            lo,
            span: span as u64,
            words,
        })
    }

    /// Whether `key` is a build key. Exact in both directions.
    #[inline(always)]
    pub(super) fn contains(&self, key: i64) -> bool {
        // Wrapping arithmetic then an unsigned compare folds `key < lo` and
        // `key >= lo + span` into one branch: a key below `lo` wraps to a huge slot.
        let slot = key.wrapping_sub(self.lo) as u64;
        slot < self.span && (self.words[(slot >> 6) as usize] >> (slot & 63)) & 1 == 1
    }

    /// [`Self::contains`] as a 0/1 with no data-dependent branch, for a selection loop that
    /// counts rather than tests: a selective probe's hits are rare and unpredictable, so a
    /// branch on each is a misprediction on each.
    #[inline(always)]
    fn contains_bit(&self, key: i64) -> usize {
        let slot = key.wrapping_sub(self.lo) as u64;
        let in_range = slot < self.span;
        // Out of range reads word 0, whose bit is then masked off by `in_range`.
        let at = if in_range { slot } else { 0 };
        ((self.words[(at >> 6) as usize] >> (at & 63)) & in_range as u64) as usize
    }

    /// Heap bytes held, for [`super::JoinTable::heap_bytes`].
    pub(super) fn heap_bytes(&self) -> usize {
        self.words.capacity() * std::mem::size_of::<u64>()
    }
}

impl JoinTable {
    /// [`JoinTable::probe_range`] for an Inner or Semi join whose pre-filter is the bitmap, one
    /// block at a time: a branchless pass selects the rows that match, then only those are
    /// visited. Because the bitmap is exact, a Semi join needs **no** lookup at all — a set bit
    /// on a non-null key *is* a match. Rows are emitted in ascending order within the range, the
    /// order the per-row loop emits them. Returns how many rows the bitmap rejected.
    ///
    /// Anti is deliberately not done this way: it emits the rows that *miss*, which on a
    /// selective join is nearly all of them, and there the per-row loop's always-taken branch
    /// beat a selection vector that copies every row (measured 47.7 ms against 51.3 ms on TPC-H
    /// `lineitem` anti a 204-key build).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn probe_range_bits<K: JoinKeys>(
        &self,
        keys: &K,
        range: std::ops::Range<usize>,
        left_null: Option<&[bool]>,
        join_type: JoinType,
        (bits, key): (&KeyBits, &[i64]),
        left_out: &mut IndexBuf,
        right_out: &mut IndexBuf,
    ) -> u64 {
        debug_assert!(matches!(join_type, JoinType::Inner | JoinType::Semi));
        let mut sel = [0u32; BLOCK];
        let mut matched = 0usize;
        let mut start = range.start;
        while start < range.end {
            let end = (start + BLOCK).min(range.end);
            let mut n = 0;
            for i in start..end {
                // A match: the bit is set and the key is not null.
                sel[n] = i as u32;
                n += bits.contains_bit(key[i]) & usize::from(!left_null.is_some_and(|m| m[i]));
            }
            matched += n;
            for &i in &sel[..n] {
                if join_type == JoinType::Semi {
                    left_out.push(i);
                    right_out.push_null();
                    continue;
                }
                // Selected ⇒ present and non-null, so the lookup finds a head.
                let mut rejected = 0;
                let mut r = self.head_for(keys, i as usize, false, Prefilter::None, &mut rejected);
                while let Some(row) = r {
                    left_out.push(i);
                    right_out.push(row);
                    r = (!self.unique)
                        .then(|| self.next[row as usize])
                        .filter(|&nxt| nxt != u32::MAX);
                }
            }
            start = end;
        }
        (range.len() - matched) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn membership_is_exact_including_bounds_and_nulls() {
        let keys = [5i64, 9, 70, 5, 1_000];
        let null = [false, false, false, false, true];
        // Bounds of the non-null keys only: the null row's 1_000 is outside them.
        let bits = KeyBits::build(&keys, &null, 5, 66).expect("small span");
        for k in -3..200 {
            assert_eq!(bits.contains(k), [5, 9, 70].contains(&k), "key {k}");
        }
        assert!(!bits.contains(i64::MIN));
        assert!(!bits.contains(i64::MAX));
    }

    #[test]
    fn refuses_a_span_past_the_cap() {
        assert!(KeyBits::build(&[0, 1], &[false, false], 0, MAX_SPAN + 1).is_none());
        assert!(KeyBits::build(&[0, 1], &[false, false], 0, MAX_SPAN).is_some());
    }

    /// End to end through the public join entry: a build key spread ~20 slots per row is
    /// refused by the dense map and admitted here, so the bitmap is the path under test. Every
    /// join type, duplicates on both sides, nulls on both sides, probe keys below, inside and
    /// above the range, and a probe long enough to finish the pre-filter's trial — held to a
    /// by-key oracle rather than to another fast path.
    #[test]
    fn joins_through_the_bitmap_match_an_oracle() {
        use crate::join::{hash_join_indices, JoinType};
        use arrow::array::{Array, ArrayRef, Int64Array};
        use std::sync::Arc;

        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let right: Vec<Option<i64>> = (0..3_000)
            .map(|_| (next() % 17 != 0).then(|| (next() % 60_000) as i64 + 1_000))
            .collect();
        // Mostly misses (a selective join), some hits, some outside the build range.
        let left: Vec<Option<i64>> = (0..200_000)
            .map(|i| {
                (next() % 23 != 0).then(|| match i % 10 {
                    0 => right[(next() % 3_000) as usize].unwrap_or(5),
                    1 => -((next() % 1_000) as i64),
                    _ => (next() % 70_000) as i64,
                })
            })
            .collect();
        let bounds = super::super::dense::key_bounds(
            &right.iter().map(|k| k.unwrap_or(0)).collect::<Vec<_>>(),
            right.len(),
            &right.iter().map(Option::is_none).collect::<Vec<_>>(),
        )
        .expect("non-null keys");
        assert!(
            super::super::dense::DenseHeads::build(
                &right.iter().map(|k| k.unwrap_or(0)).collect::<Vec<_>>(),
                right.len(),
                &right.iter().map(Option::is_none).collect::<Vec<_>>(),
            )
            .is_none()
                && bounds.span <= MAX_SPAN,
            "fixture must be refused by the dense map and admitted by the bitmap"
        );
        let l: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(left.clone()))];
        let r: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(right.clone()))];
        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            let got = hash_join_indices(&l, &r, jt).expect("join");
            let opt = |a: &arrow::array::UInt32Array, i: usize| a.is_valid(i).then(|| a.value(i));
            let mut got: Vec<_> = (0..got.left.len())
                .map(|i| (opt(&got.left, i), opt(&got.right, i)))
                .collect();
            got.sort_unstable();
            let mut by_key = std::collections::HashMap::<i64, Vec<u32>>::new();
            for (ri, rk) in right.iter().enumerate() {
                if let Some(k) = rk {
                    by_key.entry(*k).or_default().push(ri as u32);
                }
            }
            let mut want = Vec::new();
            let mut right_hit = vec![false; right.len()];
            for (li, &lk) in left.iter().enumerate() {
                let matched: &[u32] = lk.and_then(|k| by_key.get(&k)).map_or(&[], Vec::as_slice);
                for &ri in matched {
                    right_hit[ri as usize] = true;
                }
                let li = Some(li as u32);
                match jt {
                    JoinType::Semi if !matched.is_empty() => want.push((li, None)),
                    JoinType::Anti if matched.is_empty() => want.push((li, None)),
                    JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full => {
                        want.extend(matched.iter().map(|&ri| (li, Some(ri))));
                        if matched.is_empty() && matches!(jt, JoinType::Left | JoinType::Full) {
                            want.push((li, None));
                        }
                    }
                    _ => {}
                }
            }
            if matches!(jt, JoinType::Right | JoinType::Full) {
                want.extend(
                    (0..right.len())
                        .filter(|&ri| !right_hit[ri])
                        .map(|ri| (None, Some(ri as u32))),
                );
            }
            want.sort_unstable();
            assert_eq!(got, want, "{jt:?} disagrees with the by-key oracle");
        }
    }

    #[test]
    fn extreme_keys_do_not_overflow() {
        let keys = [i64::MAX - 1, i64::MAX];
        let bits = KeyBits::build(&keys, &[false, false], i64::MAX - 1, 2).expect("span 2");
        assert!(bits.contains(i64::MAX) && bits.contains(i64::MAX - 1));
        assert!(!bits.contains(i64::MIN) && !bits.contains(0));
    }
}
