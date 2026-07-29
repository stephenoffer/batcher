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
//! The fill runs across cores past [`PARALLEL_BUILD_MIN_ROWS`], for the reason
//! `join::build` gives for sharding the hash build: the build is the join's sequential
//! prefix, and a dense build is selected precisely for the *large* surrogate-key joins.
//!
//! **Identical results by construction.** The map stores the same chain heads the hash table
//! stores, and [`super::JoinTable`] threads the same `next` chain through both, so a key with
//! duplicates walks the same chain in the same order. Null build rows are skipped here
//! exactly as they are skipped by the hash build (NULL never matches). The map is a lookup
//! structure only — it changes how a head is *found*, never which rows pair.

use rayon::prelude::*;

use super::radix;

/// The dense map's slot value for "no build row carries this key".
///
/// Zero, not `u32::MAX`, so a slot holds `row + 1` rather than `row`. That one-line change of
/// encoding is what lets the map be **allocated already empty**: `vec![0u32; span]` lowers to
/// `alloc_zeroed`, so the pages come from the OS zeroed and are faulted in lazily by the
/// threads that write them, where `vec![u32::MAX; span]` is a single-threaded memset of the
/// whole map before any work starts. At the sizes this path actually sees — TPC-H sf10's
/// `orders.o_orderkey` spans 60,000,000 slots, a **240 MB** map — that memset alone was tens
/// of milliseconds of pure Amdahl on the critical path of nearly every large join.
const EMPTY: u32 = 0;

/// Build rows below which the serial fill wins: the partition pass and the rayon fan-out cost
/// more than they save on a map small enough to be built in a few hundred microseconds.
const PARALLEL_BUILD_MIN_ROWS: usize = 1 << 16;

/// Cap on the number of disjoint map ranges built concurrently. Past this the ranges get too
/// small to amortize their own bookkeeping, mirroring `build::MAX_SHARDS`.
const MAX_PARTS: usize = 64;

/// A direct-indexed map from an integer key to its chain head.
pub(super) struct DenseHeads {
    /// The key that slot 0 stands for; a key `k` lives at `k - lo`.
    lo: i64,
    /// `map[k - lo]` is the head build row **plus one** for key `k`, or [`EMPTY`].
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
/// The benefit was intact at 4x and gone by 8x, so the budget sits at 4 — it covers the real
/// key and stops where the win does. Note the sweep predates the parallel fill, so part of
/// what it charged a wider span for was the serial build rather than the probe; widening it
/// on that basis was tried and reverted anyway (see the note below the constants).
///
/// The cost is memory: at the limit the map is ~16 bytes per build row against the hash
/// table's ~9. It is proportional to the build, never open-ended.
///
/// It is **not** the ~32 MB this once claimed. That figure came from the build side being
/// bounded by [`super::RADIX_MIN_BUILD_ROWS_BROADCAST`], and
/// [`super::BroadcastProbe::over_any_build`] — the fused join→aggregate path — deliberately
/// removed that ceiling. TPC-H sf10's `orders.o_orderkey` then arrives here as 15,000,000
/// rows spanning 60,000,000 slots: a 240 MB map, and the build side of nearly every large
/// TPC-H join. The map is still the right structure at that size (no hashing on either side,
/// and both `o_orderkey` and the `l_orderkey` probing it are near-sorted, so the accesses
/// stay sequential) — but it is only affordable because the fill runs across cores and the
/// map is allocated already-zeroed. Serial, it was ~200 ms of Amdahl per join.
const MAX_SPAN_PER_ROW: usize = 4;

/// The smallest span always allowed, so a tiny build is not refused for having a few gaps.
const MIN_SPAN: usize = 1024;

// A second, *absolute* admission rule was tried here and reverted, which is worth recording
// so it is not re-attempted blind. The idea: `MAX_SPAN_PER_ROW` asks "is the key dense
// relative to the build", which mis-reads a **filtered** build — `orders` restricted to one
// year of seven keeps 2.3M of 15M rows but still spans the whole 60M key range, so the ratio
// reads 26x and refuses. Admitting it whenever the map fit 256 MiB did what it promised in
// isolation: `lineitem ⋈ orders(1994)` at sf10 went from 6.03 s of CPU to 2.20 s.
//
// It also took **TPC-H q7 from ~166 ms to ~285 ms**, reproducibly, and the suite total did
// not move. A 240 MB map for a 2.3M-row build is only cheap while the probe walks it in key
// order; where it does not, it is a quarter-gigabyte of cache pressure standing in for a
// 20 MB hash table. One measured win against one measured 1.7x regression is not evidence for
// a wider rule, so the ratio stands alone until something distinguishes the two cases — the
// probe side's key ordering is the obvious candidate, and nothing measures it today.

impl DenseHeads {
    /// Build a dense map over the non-null build keys, or `None` when their range is too
    /// wide to index (the caller then keeps the hash table).
    ///
    /// `null` marks build rows whose key is null; they are skipped, never inserted, and so
    /// can never be found by a probe — the same NULL semantics the hash build enforces.
    /// Returns the map alongside `unique` (no key repeated) and the `next` chain links, so
    /// the caller assembles exactly the state the hash path produces.
    pub(super) fn build(keys: &[i64], rows: usize, null: &[bool]) -> Option<DenseBuild> {
        // A slot holds `row + 1`, so the last representable build row is `u32::MAX - 2`. Every
        // caller is already u32-indexed far below that; refusing here keeps the encoding total
        // rather than relying on that.
        if rows >= u32::MAX as usize - 1 {
            return None;
        }
        let (lo, span) = span_of(keys, rows, null)?;
        // `fill_parallel` carries each row's slot as a `u32` through the partition pass, so a
        // span past `u32::MAX` would truncate it and send rows to the wrong map range — a
        // silently wrong join, not a crash. Refuse it here rather than in one of the two fills,
        // so the serial and parallel paths cannot disagree about which builds are admissible.
        // (Reachable only past a billion build rows, where the map would exceed 16 GB anyway.)
        if span > u32::MAX as usize {
            return None;
        }
        let (map, links) = if rows >= PARALLEL_BUILD_MIN_ROWS && rayon::current_num_threads() > 1 {
            Self::fill_parallel(keys, null, lo, span)
        } else {
            Self::fill_serial(keys, null, lo, span)
        };
        let unique = links.is_empty();
        Some(DenseBuild {
            heads: Self { lo, map },
            links,
            unique,
        })
    }

    /// The single-threaded fill: walk rows in order, prepending each into its slot.
    ///
    /// This is the definition the parallel fill must reproduce, and the oracle the tests below
    /// compare against — a key's chain is its build rows in descending order, headed by the
    /// last row seen.
    fn fill_serial(
        keys: &[i64],
        null: &[bool],
        lo: i64,
        span: usize,
    ) -> (Vec<u32>, Vec<(u32, u32)>) {
        let rows = null.len();
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
                chain.push((i as u32, *slot - 1));
            }
            *slot = i as u32 + 1;
        }
        (map, chain)
    }

    /// The same fill, spread across cores — **bit-identical** to [`Self::fill_serial`].
    ///
    /// The map is cut into `parts` contiguous slot ranges and each range is filled by one
    /// worker, so no two workers ever touch the same slot and no synchronization is needed.
    /// [`radix::partition_side`] hands each range its build rows in ascending `abs_row` order,
    /// which is precisely the order the serial loop visits them in — so every key's chain comes
    /// out in the same descending-row order, headed by the same row. Only the clock differs.
    ///
    /// This is the join-build counterpart of `build::build_sharded`, and it exists for the same
    /// reason: the fill was the join's sequential prefix. A dense build is chosen exactly for
    /// the *large* surrogate-key joins (`lineitem ⋈ orders` at TPC-H sf10 builds 15,000,000
    /// rows into a 60,000,000-slot map), so leaving it serial capped the whole join at one core
    /// while the probe beside it already scaled across ninety-six.
    fn fill_parallel(
        keys: &[i64],
        null: &[bool],
        lo: i64,
        span: usize,
    ) -> (Vec<u32>, Vec<(u32, u32)>) {
        let parts = rayon::current_num_threads()
            .min(MAX_PARTS)
            .next_power_of_two()
            .clamp(2, MAX_PARTS);
        // Every slot lies in `0..span`, so `slot / range_len` is always a valid part index.
        let range_len = span.div_ceil(parts).max(1);
        let slot_of = |i: usize| (keys[i] - lo) as u32;
        let buckets =
            radix::partition_side(slot_of, null, parts, |slot| (*slot as usize) / range_len);

        // `alloc_zeroed`: the map arrives empty from the OS (see `EMPTY`), so the only writes
        // are the build rows themselves, made by the worker that owns their range.
        let mut map = vec![EMPTY; span];
        let chains: Vec<Vec<(u32, u32)>> = map
            .par_chunks_mut(range_len)
            .enumerate()
            .zip(buckets.par_iter())
            .map(|((p, slice), rows_here)| {
                let base = p * range_len;
                let mut chain: Vec<(u32, u32)> = Vec::new();
                for &(slot, abs) in rows_here {
                    let cell = &mut slice[slot as usize - base];
                    if *cell != EMPTY {
                        chain.push((abs, *cell - 1));
                    }
                    *cell = abs + 1;
                }
                chain
            })
            .collect();
        // The link *vector's* order is immaterial — `build::stitch_chain` writes each
        // `next[row]`, and a row appears in exactly one range — so a flat concatenation is
        // enough. What must be preserved is each key's chain, and disjoint ranges plus
        // ascending rows within a range already guarantee it.
        (map, chains.concat())
    }

    /// The chain head for probe key `k`, or `None` when no build row carries it.
    #[inline]
    pub(super) fn head(&self, k: i64) -> Option<u32> {
        // A probe key outside the build's range simply has no build row — the same answer
        // the hash table gives, reached without a lookup.
        let idx = k.checked_sub(self.lo)?;
        let slot = *self.map.get(usize::try_from(idx).ok()?)?;
        // `then`, not `then_some`: the latter evaluates its argument eagerly, and `slot - 1`
        // underflows on the empty slot this is testing for.
        (slot != EMPTY).then(|| slot - 1)
    }
}

/// `(lo, span)` for the non-null keys when the range is worth direct-mapping, else `None`.
///
/// One linear min/max pass, which is cheap next to the per-row hashing it replaces — but
/// "cheap per row" over a 15,000,000-row build is still milliseconds on one core, ahead of a
/// fill that now uses all of them, so it reduces across cores too. `(min, max, count)` is a
/// commutative monoid, so the split is invisible in the result.
fn span_of(keys: &[i64], rows: usize, null: &[bool]) -> Option<(i64, usize)> {
    const CHUNK: usize = 1 << 16;
    let bounds = |r: std::ops::Range<usize>| {
        let mut lo = i64::MAX;
        let mut hi = i64::MIN;
        let mut seen = 0usize;
        for i in r {
            if null[i] {
                continue;
            }
            let k = keys[i];
            lo = lo.min(k);
            hi = hi.max(k);
            seen += 1;
        }
        (lo, hi, seen)
    };
    let (lo, hi, seen) = if rows >= PARALLEL_BUILD_MIN_ROWS && rayon::current_num_threads() > 1 {
        (0..rows)
            .into_par_iter()
            .step_by(CHUNK)
            .map(|s| bounds(s..(s + CHUNK).min(rows)))
            .reduce(
                || (i64::MAX, i64::MIN, 0usize),
                |a, b| (a.0.min(b.0), a.1.max(b.1), a.2 + b.2),
            )
    } else {
        bounds(0..rows)
    };
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

    /// The claim the parallel fill rests on: it is invisible in the result.
    ///
    /// Duplicates are the sharp edge — they are what the chain exists for, and a fill that
    /// visited rows out of order would reorder or re-head them. Every key here repeats across
    /// the whole row range and the keys are laid out so slots land in *different* map ranges,
    /// so a range-ownership mistake shows up as a differing head or chain.
    #[test]
    fn the_parallel_fill_reproduces_the_serial_fill_exactly() {
        for (rows, keyspan) in [
            (PARALLEL_BUILD_MIN_ROWS + 1234, 4096usize),
            (300_000, 90_000),
        ] {
            let keys: Vec<i64> = (0..rows).map(|i| (i % keyspan) as i64).collect();
            let null: Vec<bool> = (0..rows).map(|i| i % 97 == 0).collect();
            let (lo, span) = span_of(&keys, rows, &null).expect("a range");

            let (smap, mut slinks) = DenseHeads::fill_serial(&keys, &null, lo, span);
            let (pmap, mut plinks) = DenseHeads::fill_parallel(&keys, &null, lo, span);

            assert_eq!(smap, pmap, "every slot must hold the same chain head");
            // The link vector's order is immaterial (see `fill_parallel`); the set is not.
            slinks.sort_unstable();
            plinks.sort_unstable();
            assert_eq!(slinks, plinks, "every chain link must be identical");
        }
    }

    /// A build big enough to fill in parallel still answers every probe the serial build
    /// answers — the end-to-end statement of the test above, through the public entry point.
    #[test]
    fn a_parallel_build_answers_every_key() {
        let rows = PARALLEL_BUILD_MIN_ROWS * 2;
        let keys: Vec<i64> = (0..rows as i64).collect();
        let null = vec![false; rows];
        let built = DenseHeads::build(&keys, rows, &null).expect("dense");
        assert!(built.unique, "distinct keys leave no chain");
        for (row, &k) in keys.iter().enumerate() {
            assert_eq!(built.heads.head(k), Some(row as u32));
        }
        assert_eq!(built.heads.head(-1), None);
        assert_eq!(built.heads.head(rows as i64), None);
    }

    /// Row 0 is the sharp edge of the `row + 1` slot encoding: with `EMPTY == 0` a map that
    /// stored the row directly would report the first build row as "no such key".
    #[test]
    fn build_row_zero_is_found_not_read_as_empty() {
        let heads = DenseHeads::build(&[7i64, 9], 2, &[false, false])
            .expect("dense")
            .heads;
        assert_eq!(heads.head(7), Some(0), "row 0 must be a real head");
        assert_eq!(heads.head(8), None, "the gap is still empty");
        assert_eq!(heads.head(9), Some(1));
    }

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

    /// The memory rule must not let a *tiny* build claim a huge map: 64x bounds the waste,
    /// so a thousand rows scattered over a billion keys still falls back to the hash table.
    #[test]
    fn a_tiny_build_cannot_claim_a_huge_map() {
        let keys: Vec<i64> = (0..1_000i64).map(|i| i * 1_000_000).collect();
        let null = vec![false; keys.len()];
        assert!(
            span_of(&keys, keys.len(), &null).is_none(),
            "1,000 rows may not summon a 4 GB map"
        );
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
