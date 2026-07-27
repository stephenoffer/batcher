//! Dense direct-map join heads — a perfect hash for a small-range integer build key.
//!
//! A hash join's build side is its dominant cost on the shapes that matter: measured on the
//! canonical `lineitem ⋈ orders` at 6M ⋈ 1.5M, building the shared table took ~29.5 ms
//! against ~6.7 ms for the entire 6M-row probe. Nearly all of that build is work a *dense*
//! key does not need — hashing every build row, then hashing every probe row again, and
//! chasing a chain through a table far larger than L2.
//!
//! A surrogate primary key is dense by construction. `orders.o_orderkey`,
//! `part.p_partkey`, `customer.c_custkey` — every star-schema dimension — is a contiguous
//! (or near-contiguous) run of integers, and that is precisely the join the analytical
//! workload is made of. For those, `map[key - lo]` *is* the hash table: one indexed load,
//! no hashing on either side, and no collision chain to walk.
//!
//! This is the join-side counterpart of the dense direct-map `agg::group::assign` already
//! uses for group keys, and it is gated the same way — on the key's observed value range.
//!
//! **Identical results by construction.** The map stores the same chain heads the hash table
//! stores, and [`super::JoinTable`] threads the same `next` chain through both, so a key with
//! duplicates walks the same chain in the same order. Null build rows are skipped here
//! exactly as they are skipped by the hash build (NULL never matches). The map is a lookup
//! structure only — it changes how a head is *found*, never which rows pair.

/// The dense map's slot for "no build row carries this key".
const EMPTY: u32 = u32::MAX;

/// A direct-indexed map from an integer key to its chain head.
pub(super) struct DenseHeads {
    /// The key that slot 0 stands for; a key `k` lives at `k - lo`.
    lo: i64,
    /// `map[k - lo]` is the head build row for key `k`, or [`EMPTY`].
    map: Vec<u32>,
}

/// What a dense build produces — exactly the three things [`super::JoinTable`] needs to
/// assemble state indistinguishable from the hash path's.
pub(super) struct DenseBuild {
    /// The lookup structure replacing the sharded hash tables.
    pub(super) heads: DenseHeads,
    /// `(row, next)` links for repeated keys; empty when the build key is unique.
    pub(super) links: Vec<(u32, u32)>,
    /// Whether no build key repeats, so every chain has length exactly 1.
    pub(super) unique: bool,
}

/// The widest value range worth direct-mapping, as a multiple of the build row count.
///
/// Chosen by measurement, not by the memory arithmetic, because a real surrogate key is
/// **not** contiguous. TPC-H generates `orders.o_orderkey` sparsely — 1.5M orders spread
/// across a 6M key range — so a budget tight enough to guarantee no extra memory (the map is
/// `span * 4` bytes against the chained table's ~`rows * 9`, so 2x) would refuse the single
/// most common shape this exists to serve. Sweeping span/rows on 6M ⋈ 1.5M join+aggregate:
///
/// | span / rows | ms |
/// |---|---:|
/// | 1x (contiguous) | 44.8 |
/// | **4x (TPC-H `o_orderkey`)** | **49.8** |
/// | 8x | 82.9 |
/// | hash table | 86.8 |
///
/// The benefit is intact at 4x and gone by 8x, where the map no longer stays cache-resident.
/// So the budget sits at 4 — it covers the real key and stops where the win does.
///
/// The cost is memory: at the limit the map is ~16 bytes per build row against the hash
/// table's ~9. That is bounded, not open-ended — this path only ever sees a build side under
/// [`super::RADIX_MIN_BUILD_ROWS_BROADCAST`], capping the map at ~32 MB — and it is spent on
/// the *small* side of a broadcast join.
const MAX_SPAN_PER_ROW: usize = 4;

/// The smallest span always allowed, so a tiny build is not refused for having a few gaps.
const MIN_SPAN: usize = 1024;

impl DenseHeads {
    /// Build a dense map over the non-null build keys, or `None` when their range is too
    /// wide to index (the caller then keeps the hash table).
    ///
    /// `null` marks build rows whose key is null; they are skipped, never inserted, and so
    /// can never be found by a probe — the same NULL semantics the hash build enforces.
    /// Returns the map alongside `unique` (no key repeated) and the `next` chain links, so
    /// the caller assembles exactly the state the hash path produces.
    pub(super) fn build(keys: &[i64], rows: usize, null: &[bool]) -> Option<DenseBuild> {
        let (lo, span) = span_of(keys, rows, null)?;
        let mut map = vec![EMPTY; span];
        // Chain links for repeated keys, collected rather than written through a
        // pre-allocated `next`: a unique build key (every dimension table) then costs no
        // allocation at all, matching `build_sharded`'s reason for doing the same.
        let mut chain: Vec<(u32, u32)> = Vec::new();
        for i in 0..rows {
            if null[i] {
                continue;
            }
            // `span_of` proved every non-null key lies in `lo..lo + span`, so the index is
            // in bounds for exactly the rows this loop visits.
            let slot = &mut map[(keys[i] - lo) as usize];
            if *slot != EMPTY {
                // Prepend, exactly as the hash build does, so the chain order — and with it
                // the emitted row order — is identical.
                chain.push((i as u32, *slot));
            }
            *slot = i as u32;
        }
        let unique = chain.is_empty();
        Some(DenseBuild {
            heads: Self { lo, map },
            links: chain,
            unique,
        })
    }

    /// The chain head for probe key `k`, or `None` when no build row carries it.
    #[inline]
    pub(super) fn head(&self, k: i64) -> Option<u32> {
        // A probe key outside the build's range simply has no build row — the same answer
        // the hash table gives, reached without a lookup.
        let idx = k.checked_sub(self.lo)?;
        let slot = *self.map.get(usize::try_from(idx).ok()?)?;
        (slot != EMPTY).then_some(slot)
    }
}

/// `(lo, span)` for the non-null keys when the range is worth direct-mapping, else `None`.
///
/// One linear min/max pass, which is cheap next to the per-row hashing it replaces. An
/// all-null (or empty) build has no range and falls back.
fn span_of(keys: &[i64], rows: usize, null: &[bool]) -> Option<(i64, usize)> {
    let mut lo = i64::MAX;
    let mut hi = i64::MIN;
    let mut seen = 0usize;
    for i in 0..rows {
        if null[i] {
            continue;
        }
        let k = keys[i];
        lo = lo.min(k);
        hi = hi.max(k);
        seen += 1;
    }
    if seen == 0 {
        return None;
    }
    // `hi - lo` can overflow i64 for extreme keys; a checked span refuses those outright.
    let span = usize::try_from(hi.checked_sub(lo)?.checked_add(1)?).ok()?;
    let budget = seen.saturating_mul(MAX_SPAN_PER_ROW).max(MIN_SPAN);
    (span <= budget).then_some((lo, span))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A contiguous unique key — the dimension-table shape — maps every row to itself and
    /// reports `unique`, so the probe can skip the chain load entirely.
    #[test]
    fn contiguous_unique_keys_map_directly() {
        let keys: Vec<i64> = (0..1000).collect();
        let null = vec![false; keys.len()];
        let b = DenseHeads::build(&keys, keys.len(), &null).unwrap();
        let (heads, chain, unique) = (b.heads, b.links, b.unique);
        assert!(unique, "no key repeats");
        assert!(chain.is_empty(), "a unique build needs no chain links");
        for (row, &k) in keys.iter().enumerate() {
            assert_eq!(heads.head(k), Some(row as u32));
        }
        assert_eq!(heads.head(-1), None, "below the range");
        assert_eq!(heads.head(1000), None, "above the range");
    }

    /// Repeated keys chain in prepend order, and the head is the LAST row seen — the same
    /// order the hash build produces, which is what keeps the emitted rows identical.
    #[test]
    fn repeated_keys_chain_in_prepend_order() {
        let keys = [5i64, 7, 5, 7, 5];
        let null = vec![false; keys.len()];
        let b = DenseHeads::build(&keys, keys.len(), &null).unwrap();
        let (heads, chain, unique) = (b.heads, b.links, b.unique);
        assert!(!unique);
        assert_eq!(heads.head(5), Some(4), "last row seen is the head");
        assert_eq!(heads.head(7), Some(3));
        // Row 2 links to row 0, row 4 links to row 2; row 3 links to row 1.
        assert_eq!(chain, vec![(2, 0), (3, 1), (4, 2)]);
    }

    /// Null build rows are never inserted, so a probe can never match one (NULL != NULL).
    #[test]
    fn null_build_rows_are_never_found() {
        let keys = [1i64, 2, 3];
        let null = [false, true, false];
        let heads = DenseHeads::build(&keys, 3, &null).unwrap().heads;
        assert_eq!(heads.head(1), Some(0));
        assert_eq!(
            heads.head(2),
            None,
            "the null row is absent despite its slot value"
        );
        assert_eq!(heads.head(3), Some(2));
    }

    /// A sparse key range is refused, so the caller keeps the hash table rather than
    /// allocating a map far larger than the structure it would replace.
    #[test]
    fn a_sparse_range_is_refused() {
        let keys = [0i64, 1_000_000_000];
        let null = [false, false];
        assert!(DenseHeads::build(&keys, 2, &null).is_none());
        // ...but a small build is allowed its `MIN_SPAN` of slack.
        let tight = [0i64, 500];
        assert!(DenseHeads::build(&tight, 2, &[false, false]).is_some());
    }

    /// Keys at the extremes cannot be made to overflow the span arithmetic.
    #[test]
    fn extreme_keys_do_not_overflow() {
        let keys = [i64::MIN, i64::MAX];
        assert!(DenseHeads::build(&keys, 2, &[false, false]).is_none());
    }

    /// An all-null (or empty) build has no range at all and falls back rather than panicking.
    #[test]
    fn an_all_null_build_falls_back() {
        assert!(DenseHeads::build(&[1, 2], 2, &[true, true]).is_none());
        assert!(DenseHeads::build(&[], 0, &[]).is_none());
    }
}
