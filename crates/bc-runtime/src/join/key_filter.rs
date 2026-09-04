//! The build side's key set, digested into a filter the probe side applies *before* the join.
//!
//! An equi-join only ever emits a probe row whose key equals some build-side key. So once the
//! build side exists — and in every executor here it exists before a single probe row is read —
//! its key set is a **superset filter** on the probe side: a probe row whose key is not in it
//! produces nothing, and removing it cannot change the result. This is the sideways-information
//! passing DuckDB and Spark AQE rely on, and the distributed join already does over the network
//! (`bc_sketches::BloomFilter`'s own docs describe that use).
//!
//! What this module adds is the *single-node* form, and the reason it is worth having is where
//! the filter gets applied. [`super::use_probe_bloom_with`] already pre-filters probe keys, but
//! only inside the join, after the probe side has been scanned, filtered and projected — so it
//! saves a hash lookup and nothing else. A `KeyFilter` is cheap to carry, so the executor can
//! sink it down the probe pipeline to the *scan*, where dropping a row also drops every
//! predicate, projection and copy that row would have cost on the way up. TPC-H q21 is the
//! shape: 411 of 10,000 suppliers survive `n_name = 'SAUDI ARABIA'`, and the 6M-row `lineitem`
//! probe underneath it is reduced ~24x before its date predicate is ever evaluated.
//!
//! ## Soundness
//!
//! The only unsafe answer is a false *negative* — dropping a row that did have a match. Every
//! representation here is therefore chosen to have none:
//!
//!   - the `[lo, hi]` guard rejects only keys strictly outside the build side's own extremes;
//!   - the membership test is the *literal* key set, not a sketch.
//!
//! Anything the digest declines to represent returns `None` from [`KeyFilter::build`] — "filter
//! nothing" — rather than guessing. The one thing this module deliberately does **not** do is
//! approximate: see [`MAX_DISTINCT_KEYS`] for the measurement that ruled a bloom out.
//!
//! ## Two exact representations, chosen by the key's *span*
//!
//! A surrogate join key is dense by construction — `p_partkey`, `o_orderkey`, `s_suppkey` are
//! runs of integers — so its membership set is better held as a **bitmap over `[lo, hi]`** than
//! as a hash set: smaller, built with no hashing, and probed with one load and a shift instead
//! of a hash and a chain walk. [`KeySet`] holds whichever is smaller, and both are exact, so
//! which one a build side gets is invisible in the result.
//!
//! That choice is not a micro-optimization — it is what lets the digest serve the shape it was
//! written for. [`MAX_DISTINCT_KEYS`] bounds the *hash set*, and it has to, because a hash set
//! of a million keys is a megabyte of random-access probe target. A bitmap has no such problem,
//! so a dense key is admitted on its span alone. The cliff that removed was measurable and
//! sharp: TPC-H sf10 `lineitem ⋈ part` with the build side narrowed to N distinct part keys ran
//! at 36.6 ms for N = 62,500 and **74.7 ms** for N = 66,666, the two sides of a cap that had
//! nothing to do with the data. `p_name LIKE '%green%'` — TPC-H q9's own filter — keeps 108,782
//! parts and landed on the wrong side of it.
//!
//! Null keys need no special case beyond dropping them: `NULL = NULL` is NULL, not TRUE, so a
//! null-keyed probe row never matches. It is therefore correct to mark it `false` — but only
//! for a join whose probe side is *reducible* at all. That is the caller's decision (an anti or
//! outer join must keep its unmatched probe rows), and [`KeyFilter`] deliberately does not know
//! the join type: it answers "can this key match?", nothing more.

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray};
use arrow::buffer::BooleanBuffer;
use arrow::compute::kernels::aggregate::{max as arrow_max, min as arrow_min};
use arrow::datatypes::{DataType, Int64Type};
use hashbrown::HashSet;

/// Distinct keys past which the build side is **abandoned**, not approximated.
///
/// This is the load-bearing constant, and it was learned the expensive way. The first version
/// fell back to a bloom past this point instead of giving up, on the reasoning that a bloom is
/// cheap per key. It is — but the *build* is not free, and neither is the probe it then adds to
/// every row of the other side. TPC-H q4 is `orders SEMI lineitem`, whose build is the 3.8M-row
/// side: digesting it cost ~26M random bit-writes, and the filter it produced then passed
/// almost every `orders` row, because 1.5M distinct order keys against a 1.5M-row probe is not
/// a filter at all. q4 went from ~36 ms to ~272 ms — a 7.5x blow-up, far outside any
/// benchmark noise, and the reason the cap is a hard refusal rather than a fallback.
///
/// The economics are the whole point of the optimization: it wins when a **small, selective**
/// build side meets a large probe side. A build side with more distinct keys than this is not
/// that shape, so the honest answer is no filter. Bounding *distinct* keys rather than rows
/// also bounds the build cost itself — [`KeyFilter::build`] stops the moment the cap is passed,
/// so a high-cardinality key column is abandoned after ~65k rows rather than scanned in full.
///
/// 65,536 `i64`s is a ~1 MB hash table: L2-resident on any modern core, so the per-probe-row
/// lookup stays a cache hit rather than becoming the cost it was meant to save.
const MAX_DISTINCT_KEYS: usize = 1 << 16;

/// Rows past which the build side is not scanned at all.
///
/// Only reachable by a key column with few distinct values (anything else trips
/// [`MAX_DISTINCT_KEYS`] first and exits early), so this is a bound on the pathological case:
/// a huge build side over a handful of keys. Scanning 4M `i64`s is a few milliseconds against
/// the hash table already built beside it, and the filter such a side yields is extremely
/// selective — worth the scan.
const MAX_BUILD_ROWS: usize = 1 << 22;

/// Bits of key span a dense bitmap may cover per non-null build row.
///
/// The bitmap costs `span / 8` bytes; the hash set it replaces costs ~10.3 bytes per key
/// (hashbrown holds an `i64` plus a control byte at a 7/8 load factor). So the bitmap is the
/// *smaller* structure whenever `span < 82 * rows`, and 64 is that crossover rounded down to a
/// shift. Sizing the choice by which representation is smaller — rather than by a tuned
/// constant — is what keeps this honest at every scale: a key so sparse that the bitmap would be
/// the bigger object is exactly the key the hash set should hold.
const DENSE_SPAN_PER_ROW: u128 = 64;

/// The smallest span always allowed a bitmap, so a handful of build rows a few thousand apart
/// are not pushed onto the hash set to save a few hundred bytes.
const MIN_DENSE_SPAN: u128 = 1 << 13;

/// Absolute ceiling on a bitmap's span — 256 Mi keys, a 32 MiB map.
///
/// [`DENSE_SPAN_PER_ROW`] alone is a *relative* bound, and a relative bound on a 4M-row build
/// side (the [`MAX_BUILD_ROWS`] limit) would permit a 32 GiB allocation. This is the guard that
/// makes the memory a stated number rather than a consequence.
const MAX_DENSE_SPAN: u128 = 1 << 28;

/// How a [`KeyFilter`] holds its build keys. Both arms are **exact** — no false positives and
/// no false negatives — so the arm a build side lands on changes speed and memory, never rows.
enum KeySet {
    /// A bitmap over `[lo, hi]`: key `k` is present iff bit `k - lo` is set. The representation
    /// a surrogate key gets, and the one that made the digest worth extending past
    /// [`MAX_DISTINCT_KEYS`] — see the module note.
    Dense(Vec<u64>),
    /// The literal key set, for a span too sparse to bitmap. Bounded by [`MAX_DISTINCT_KEYS`].
    Sparse(HashSet<i64, ahash::RandomState>),
}

impl KeySet {
    /// Whether `offset` (already known to be within `[0, span)`) is a member.
    #[inline]
    fn contains(&self, key: i64, offset: usize) -> bool {
        match self {
            // `offset < span` is guaranteed by the caller's `[lo, hi]` guard, and the bitmap
            // covers `span` bits, so the word index is in bounds.
            KeySet::Dense(bits) => (bits[offset >> 6] >> (offset & 63)) & 1 == 1,
            KeySet::Sparse(set) => set.contains(&key),
        }
    }
}

/// The build side's key set, as a membership test over `Int64` probe keys.
///
/// Restricted to a single `Int64` key column — the analytical join shape once the FFI boundary
/// has widened narrow integers, and the same shape [`super::stream::BroadcastProbe`] fast-paths.
/// A composite or string key returns `None` from [`KeyFilter::build`] rather than growing a
/// second encoding here; the row-encoded form is what the join's own hash table is for.
pub struct KeyFilter {
    /// The build keys' extremes. Two predictable compares reject an out-of-range key with no
    /// hash at all — and on a clustered fact table (an ordered surrogate key, a date) that is
    /// most of the rows, whole morsels at a time.
    lo: i64,
    hi: i64,
    /// The build side's keys, exactly. Not a sketch: an approximate membership test would let
    /// through rows the exact set rejects, and at these sizes the exact set is both smaller and
    /// faster than the bloom that would approximate it (see [`MAX_DISTINCT_KEYS`]).
    keys: KeySet,
    /// Distinct non-null build keys held. Counted during the digest, because neither
    /// representation can answer it afterwards in constant time.
    distinct: usize,
}

impl KeyFilter {
    /// Digest a build side's single `Int64` key column, or `None` if it is not worth digesting.
    ///
    /// `None` for a non-`Int64` key, a build side past [`MAX_BUILD_ROWS`], an empty or all-null
    /// key column (the join yields nothing, and the caller's own empty-side handling says so
    /// more clearly than a filter that rejects everything), or — the case that matters — a key
    /// column with more than [`MAX_DISTINCT_KEYS`] distinct values, which is not the selective
    /// shape this optimization pays for.
    pub fn build(keys: &ArrayRef) -> Option<Self> {
        if keys.data_type() != &DataType::Int64 || keys.len() > MAX_BUILD_ROWS {
            return None;
        }
        let a = keys.as_primitive::<Int64Type>();
        // The extremes come from arrow's own null-skipping SIMD reduction rather than the
        // digest loop, because they decide *which* digest to run — and because a build side
        // that turns out to be sparse should not have paid for a hash-set insert per row on
        // the way to finding that out. `None` here is an empty or all-null key column.
        let (lo, hi) = match (arrow_min(a), arrow_max(a)) {
            (Some(lo), Some(hi)) => (lo, hi),
            _ => return None,
        };
        // `i128` because `hi - lo` overflows `i64` on the extremes, and a span that wide is
        // refused rather than wrapped into a small one.
        let span = (i128::from(hi) - i128::from(lo) + 1) as u128;
        let rows = a.len() - a.null_count();
        if span <= dense_span_budget(rows) {
            return Some(Self::dense(a, lo, hi, span));
        }
        Self::sparse(a, lo, hi)
    }

    /// The bitmap digest: one bit per key in `[lo, hi]`, set from the build side in one pass.
    fn dense(a: &arrow::array::Int64Array, lo: i64, hi: i64, span: u128) -> Self {
        let mut bits = vec![0u64; span.div_ceil(64) as usize];
        let mut distinct = 0usize;
        let mut set = |v: i64| {
            // In range by construction: `lo <= v <= hi` for every non-null build key.
            let offset = (i128::from(v) - i128::from(lo)) as usize;
            let word = &mut bits[offset >> 6];
            let mask = 1u64 << (offset & 63);
            distinct += usize::from(*word & mask == 0);
            *word |= mask;
        };
        match a.nulls() {
            // The null-free case reads the values buffer straight through — no per-row validity
            // branch, which is the shape a foreign-key column actually has.
            None => a.values().iter().copied().for_each(&mut set),
            Some(nulls) => {
                for i in nulls.valid_indices() {
                    set(a.value(i));
                }
            }
        }
        Self {
            lo,
            hi,
            keys: KeySet::Dense(bits),
            distinct,
        }
    }

    /// The hash-set digest, for a span too sparse to bitmap. Bounded by [`MAX_DISTINCT_KEYS`].
    fn sparse(a: &arrow::array::Int64Array, lo: i64, hi: i64) -> Option<Self> {
        let mut set: HashSet<i64, ahash::RandomState> = HashSet::default();
        for i in 0..a.len() {
            if a.is_null(i) {
                continue; // a null key matches nothing; it is not part of the set
            }
            set.insert(a.value(i));
            if set.len() > MAX_DISTINCT_KEYS {
                // Give up here rather than at the end: this is what keeps the digest's cost
                // proportional to how useful it can be, instead of to the build side's size.
                return None;
            }
        }
        let distinct = set.len();
        Some(Self {
            lo,
            hi,
            keys: KeySet::Sparse(set),
            distinct,
        })
    }

    /// Whether a probe key matches a build key.
    ///
    /// Exact in both directions — there are no false positives to reason about, only the
    /// `[lo, hi]` guard short-circuiting a lookup whose answer would have been `false` anyway.
    #[inline]
    fn may_match(&self, key: i64) -> bool {
        // The `[lo, hi]` guard runs first for both representations: it rejects an out-of-range
        // key with two predictable compares, and it is what makes the bitmap offset in bounds.
        if key < self.lo || key > self.hi {
            return false;
        }
        let offset = (i128::from(key) - i128::from(self.lo)) as usize;
        self.keys.contains(key, offset)
    }

    /// A mask over `probe`: `true` where the key may match, `false` where it provably cannot.
    ///
    /// `None` when `probe` is not `Int64` — the caller then applies no filter, which is always
    /// correct. Null probe keys mask to `false` (`NULL = NULL` is not TRUE, so they match
    /// nothing); see the module note on which join types may act on that.
    ///
    /// The mask has **no nulls of its own** — it is a decision about every row, not a
    /// three-valued predicate — so `filter_record_batch` keeps exactly the `true` rows.
    pub fn mask(&self, probe: &ArrayRef) -> Option<BooleanArray> {
        if probe.data_type() != &DataType::Int64 {
            return None;
        }
        let a = probe.as_primitive::<Int64Type>();
        // `value(i)` at a null slot reads the values buffer in bounds and its answer is ANDed
        // away below, so the loop stays branchless on validity — the same trade
        // `bc_expr::eval::in_list` makes.
        let values = BooleanBuffer::collect_bool(a.len(), |i| self.may_match(a.value(i)));
        let values = match a.nulls() {
            None => values,
            Some(nulls) => &values & nulls.inner(),
        };
        Some(BooleanArray::new(values, None))
    }

    /// Distinct build keys held — the ceiling on how many probe keys can survive the filter.
    #[must_use]
    pub fn distinct_keys(&self) -> usize {
        self.distinct
    }

    /// The build keys' `[min, max]`.
    #[must_use]
    pub fn bounds(&self) -> (i64, i64) {
        (self.lo, self.hi)
    }
}

/// The widest span a bitmap may cover for a build side of `rows` non-null keys.
///
/// Both bounds are load-bearing and they answer different questions.
/// [`DENSE_SPAN_PER_ROW`] asks "is the bitmap the smaller of the two exact representations?",
/// which is what makes the choice self-justifying rather than tuned. [`MAX_DENSE_SPAN`] asks
/// "is it an amount of memory worth naming?", which the relative bound alone cannot: at
/// [`MAX_BUILD_ROWS`] the ratio would admit 32 GiB. [`MIN_DENSE_SPAN`] keeps a small, slightly
/// gappy build side on the cheaper structure instead of refusing it over a few hundred bytes.
fn dense_span_budget(rows: usize) -> u128 {
    ((rows as u128).saturating_mul(DENSE_SPAN_PER_ROW)).clamp(MIN_DENSE_SPAN, MAX_DENSE_SPAN)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::Int64Array;

    use super::*;

    fn filter_of(keys: Vec<Option<i64>>) -> Option<KeyFilter> {
        let a: ArrayRef = Arc::new(Int64Array::from(keys));
        KeyFilter::build(&a)
    }

    fn mask_of(f: &KeyFilter, probe: Vec<Option<i64>>) -> Vec<bool> {
        let a: ArrayRef = Arc::new(Int64Array::from(probe));
        let m = f.mask(&a).unwrap();
        (0..m.len()).map(|i| m.value(i)).collect()
    }

    /// The property the whole module rests on: a key that IS in the build side always passes.
    /// A false negative here would silently delete join output rows.
    #[test]
    fn never_rejects_a_key_that_is_present() {
        for n in [1usize, 7, 100, 4_096, MAX_DISTINCT_KEYS] {
            // Spread the keys out so the range guard is not doing the work by accident.
            let keys: Vec<Option<i64>> = (0..n).map(|i| Some((i as i64) * 7 - 3)).collect();
            let f = filter_of(keys.clone()).expect("digestible");
            let probe: Vec<Option<i64>> = keys.clone();
            assert!(
                mask_of(&f, probe).iter().all(|&b| b),
                "n={n}: a present key was rejected"
            );
        }
    }

    /// Out-of-range keys are rejected by the bounds guard alone, for both representations.
    #[test]
    fn rejects_keys_outside_the_build_range() {
        for n in [10usize, 40_000] {
            let keys: Vec<Option<i64>> = (0..n).map(|i| Some(i as i64)).collect();
            let f = filter_of(keys).expect("digestible");
            assert_eq!(f.bounds(), (0, n as i64 - 1));
            assert_eq!(
                mask_of(
                    &f,
                    vec![Some(-1), Some(n as i64), Some(i64::MIN), Some(i64::MAX)]
                ),
                vec![false, false, false, false]
            );
        }
    }

    /// The exact form has no false positives at all, so an in-range non-member is dropped.
    #[test]
    fn exact_form_drops_an_in_range_non_member() {
        let f = filter_of(vec![Some(0), Some(10), Some(20)]).expect("digestible");
        assert_eq!(f.distinct_keys(), 3);
        assert_eq!(
            mask_of(&f, vec![Some(0), Some(5), Some(10), Some(15), Some(20)]),
            vec![true, false, true, false, true]
        );
    }

    /// A null probe key matches nothing (`NULL = NULL` is not TRUE), and a null build key is
    /// not part of the set — so it must not widen the range either.
    #[test]
    fn nulls_match_nothing_and_do_not_widen_the_range() {
        let f = filter_of(vec![Some(5), None, Some(9)]).expect("digestible");
        assert_eq!(f.bounds(), (5, 9));
        assert_eq!(
            mask_of(&f, vec![Some(5), None, Some(9), Some(7)]),
            vec![true, false, true, false]
        );
    }

    /// An all-null or empty build side is not digestible: the caller's own empty-side path owns
    /// what a join against nothing yields.
    #[test]
    fn undigestible_sides_return_none() {
        assert!(filter_of(vec![]).is_none());
        assert!(filter_of(vec![None, None]).is_none());
        let f32s: ArrayRef = Arc::new(arrow::array::Float64Array::from(vec![1.0, 2.0]));
        assert!(KeyFilter::build(&f32s).is_none());
    }

    /// A non-`Int64` probe column is not filtered rather than being wrongly compared.
    #[test]
    fn non_int_probe_is_not_masked() {
        let f = filter_of(vec![Some(1), Some(2)]).expect("digestible");
        let probe: ArrayRef = Arc::new(arrow::array::Float64Array::from(vec![1.0]));
        assert!(f.mask(&probe).is_none());
    }

    /// Past the distinct-key cap a **sparse** build side is abandoned rather than approximated
    /// — this is the guard that stopped TPC-H q4 paying 236 ms for a filter that removed
    /// nothing. The keys are spread far enough apart that the bitmap cannot take them, which is
    /// what makes this a test of the cap rather than of the span budget.
    #[test]
    fn a_high_cardinality_sparse_build_side_is_abandoned() {
        let n = MAX_DISTINCT_KEYS + 1;
        let stride = (DENSE_SPAN_PER_ROW as i64) * 4;
        let keys: Vec<Option<i64>> = (0..n).map(|i| Some(i as i64 * stride)).collect();
        assert!(filter_of(keys).is_none());
    }

    /// The same cardinality, *densely* packed, is digested — the cliff this module's second
    /// representation exists to remove. A contiguous surrogate key is the shape a real join
    /// has, and it used to be refused for the sole reason that it had more than
    /// [`MAX_DISTINCT_KEYS`] of them.
    #[test]
    fn a_high_cardinality_dense_build_side_is_digested() {
        let n = MAX_DISTINCT_KEYS * 4 + 1;
        let keys: Vec<Option<i64>> = (0..n).map(|i| Some(i as i64)).collect();
        let f = filter_of(keys).expect("a dense key must digest past the sparse cap");
        assert_eq!(f.distinct_keys(), n);
        assert_eq!(
            mask_of(
                &f,
                vec![Some(0), Some(n as i64 - 1), Some(n as i64), Some(-1)]
            ),
            vec![true, true, false, false]
        );
    }

    /// Both representations answer identically over the same key set, including the in-range
    /// gaps that are the only place an approximate filter would differ. This is the property
    /// that makes the choice of representation invisible in the result.
    #[test]
    fn the_two_representations_agree_exactly() {
        // Same keys, two spans: one inside the bitmap budget, one far outside it.
        let members: Vec<i64> = (0..2_000).map(|i| i * 3).collect();
        let dense = filter_of(members.iter().map(|&k| Some(k)).collect()).expect("dense");
        let sparse_keys: Vec<i64> = members
            .iter()
            .map(|&k| k * (DENSE_SPAN_PER_ROW as i64) * 8)
            .collect();
        let sparse = filter_of(sparse_keys.iter().map(|&k| Some(k)).collect()).expect("sparse");
        assert!(matches!(dense.keys, KeySet::Dense(_)));
        assert!(matches!(sparse.keys, KeySet::Sparse(_)));
        assert_eq!(dense.distinct_keys(), sparse.distinct_keys());
        // Probe every key and every gap between them, on each filter's own scale.
        let probe_d: Vec<Option<i64>> = (0..6_000).map(Some).collect();
        let probe_s: Vec<Option<i64>> = probe_d
            .iter()
            .map(|k| k.map(|k| k * (DENSE_SPAN_PER_ROW as i64) * 8))
            .collect();
        assert_eq!(mask_of(&dense, probe_d), mask_of(&sparse, probe_s));
    }

    /// A span wide enough to blow the absolute memory ceiling is refused a bitmap even though
    /// the per-row ratio would allow it, and falls back to the exact set.
    #[test]
    fn the_absolute_span_ceiling_bounds_the_bitmap() {
        let rows = 1_000_000usize;
        assert!(dense_span_budget(rows) <= MAX_DENSE_SPAN);
        // Two keys astride the whole i64 range must not wrap into a small span.
        let f = filter_of(vec![Some(i64::MIN), Some(i64::MAX)]).expect("digestible");
        assert!(matches!(f.keys, KeySet::Sparse(_)));
        assert_eq!(
            mask_of(&f, vec![Some(i64::MIN), Some(0), Some(i64::MAX)]),
            vec![true, false, true]
        );
    }

    /// A build side that is *large* but low-cardinality is still digested: it is exactly the
    /// shape the filter is most valuable on.
    #[test]
    fn a_large_low_cardinality_build_side_is_digested() {
        let keys: Vec<Option<i64>> = (0..500_000).map(|i| Some(i64::from(i % 8))).collect();
        let f = filter_of(keys).expect("low-cardinality sides must digest");
        assert_eq!(f.distinct_keys(), 8);
        assert_eq!(
            mask_of(&f, vec![Some(3), Some(9), Some(-1)]),
            vec![true, false, false]
        );
    }
}
