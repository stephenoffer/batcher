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
//! Null keys need no special case beyond dropping them: `NULL = NULL` is NULL, not TRUE, so a
//! null-keyed probe row never matches. It is therefore correct to mark it `false` — but only
//! for a join whose probe side is *reducible* at all. That is the caller's decision (an anti or
//! outer join must keep its unmatched probe rows), and [`KeyFilter`] deliberately does not know
//! the join type: it answers "can this key match?", nothing more.

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray};
use arrow::buffer::BooleanBuffer;
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
    keys: HashSet<i64, ahash::RandomState>,
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
        let mut set: HashSet<i64, ahash::RandomState> = HashSet::default();
        let (mut lo, mut hi) = (i64::MAX, i64::MIN);
        for i in 0..a.len() {
            if a.is_null(i) {
                continue; // a null key matches nothing; it is not part of the set
            }
            let v = a.value(i);
            lo = lo.min(v);
            hi = hi.max(v);
            set.insert(v);
            if set.len() > MAX_DISTINCT_KEYS {
                // Give up here rather than at the end: this is what keeps the digest's cost
                // proportional to how useful it can be, instead of to the build side's size.
                return None;
            }
        }
        if lo > hi {
            return None; // no non-null keys at all
        }
        Some(Self { lo, hi, keys: set })
    }

    /// Whether a probe key matches a build key.
    ///
    /// Exact in both directions — there are no false positives to reason about, only the
    /// `[lo, hi]` guard short-circuiting a lookup whose answer would have been `false` anyway.
    #[inline]
    fn may_match(&self, key: i64) -> bool {
        key >= self.lo && key <= self.hi && self.keys.contains(&key)
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
    pub fn distinct_keys(&self) -> usize {
        self.keys.len()
    }

    /// The build keys' `[min, max]`.
    pub fn bounds(&self) -> (i64, i64) {
        (self.lo, self.hi)
    }
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

    /// Past the distinct-key cap the build side is abandoned rather than approximated — this
    /// is the guard that stopped TPC-H q4 paying 236 ms for a filter that removed nothing.
    #[test]
    fn a_high_cardinality_build_side_is_abandoned() {
        let n = MAX_DISTINCT_KEYS + 1;
        let keys: Vec<Option<i64>> = (0..n).map(|i| Some(i as i64)).collect();
        assert!(filter_of(keys).is_none());
    }

    /// A build side that is *large* but low-cardinality is still digested: it is exactly the
    /// shape the filter is most valuable on.
    #[test]
    fn a_large_low_cardinality_build_side_is_digested() {
        let keys: Vec<Option<i64>> = (0..500_000).map(|i| Some((i % 8) as i64)).collect();
        let f = filter_of(keys).expect("low-cardinality sides must digest");
        assert_eq!(f.distinct_keys(), 8);
        assert_eq!(
            mask_of(&f, vec![Some(3), Some(9), Some(-1)]),
            vec![true, false, false]
        );
    }
}
