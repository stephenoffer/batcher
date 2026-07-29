//! A shared, monotonically tightening bound on a top-N's cut-off, so a morsel that cannot
//! reach the answer is never examined.
//!
//! `ORDER BY x LIMIT k` over a large input spends nearly all of its time selecting each
//! morsel's local best `k` rows, and for a small `k` almost every morsel contributes nothing.
//! DataFusion attacks this by turning its top-K heap's boundary row into a real predicate and
//! republishing it to the scan through a `DynamicFilterPhysicalExpr`
//! (`datafusion/physical-plan/src/topk/mod.rs::update_filter`), which lets a Parquet reader
//! prune whole row groups on it.
//!
//! This is the cheap half of that idea, applied where Batcher already has the data in hand: a
//! morsel's *key range* is one SIMD min/max pass, and if the whole range is worse than a value
//! that `k` rows already beat, the morsel can be dropped without selecting anything. It needs
//! no plan rewrite, no scan plumbing and no new operator — just a number shared between
//! workers.
//!
//! ## Why the bound is sound
//!
//! [`TopNBound`] holds a value `v` for which **at least `k` rows are known to be at least as
//! good**. That is a weaker claim than "the k-th best is `v`", and deliberately: the max of the
//! first key over any `k` candidates satisfies it, and unlike the true k-th best it can be read
//! off an unordered candidate set in one pass.
//!
//! Given such a `v`, a row whose first key is *strictly* worse than `v` is worse than each of
//! those `k` rows, because the first key dominates the comparison. So it cannot be in the
//! global top-`k` under any tie-break, and neither can a whole morsel whose entire key range is
//! strictly worse. Two details make that argument airtight:
//!
//! * **Strictly.** Ties are never dropped. Batcher resolves a tie on the key by original
//!   `(morsel, row)` position, so a row *equal* to `v` may well belong in the answer and a
//!   `>=` test would silently lose it. Every comparison here is strict.
//! * **No nulls on either side.** A null orders before or after every value depending on
//!   `nulls_first`, so it is not comparable to `v` by magnitude. Both publishing and skipping
//!   require the relevant key column to have no nulls, which costs nothing to check and
//!   removes the whole question.
//!
//! Because the bound only ever *tightens* ([`TopNBound::publish`] keeps the better value with
//! a `fetch_min`/`fetch_max`), a worker reading a stale one simply skips less. That is why this
//! needs no lock and why racing workers cannot make it unsound: a looser bound is always safe,
//! and the bound is never loosened.
//!
//! ## When it pays, and why it switches itself off
//!
//! The bound is a morsel's *local* k-th best, which is a much weaker value than the global
//! cut-off. That distinction decides everything about when this helps, and measuring it is what
//! put the self-disable in:
//!
//! * **Clustered keys** — time-ordered data, a key-partitioned scan, anything already roughly
//!   sorted — put the good rows in a few morsels, so the rest have their entire range on the
//!   wrong side of the bound and are dropped. This is the case the mechanism exists for, and it
//!   is a common one.
//! * **Uniformly random keys** exclude *nothing*. A morsel's minimum sits at a far smaller
//!   quantile than its own hundredth-smallest, so no morsel's range is ever wholly worse than a
//!   bound derived that way. Measured on 256 random morsels: **0 of 256 skipped**, for a range
//!   check costing 2.5% of the selection it hoped to avoid.
//!
//! A 2.5% tax on the shapes it cannot help is not worth paying for a whole query, so
//! [`TopNBound::excludes_range`] watches itself: after [`PROBE_MORSELS`] checks that have
//! excluded nothing it turns off for good, capping the loss at a bounded, tiny constant while
//! leaving the clustered-data win intact. This is the same self-disabling shape
//! `bc_interp::stream::runtime_filter`'s `Gauge` uses on a runtime join filter, for the same
//! reason: a probe that is cheap per morsel is still not free across a whole query.
//!
//! ## Scope
//!
//! Integer-like keys only: the signed integers and the temporal types, which is what `ORDER BY
//! <id|date|timestamp|count> LIMIT k` uses. Floats are excluded because an order-preserving
//! `f64`-to-`i64` mapping is easy to get subtly wrong around `-0.0` and NaN, and Batcher's
//! float identity is already a delicate contract (`bc_arrow::float_ident`); the win does not
//! justify putting a second encoding of it here.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};

use arrow::array::{Array, ArrayRef};
use arrow::datatypes::DataType;

/// Range checks to give the bound before concluding it cannot help this data and switching off.
///
/// Sized so the wasted work is a rounding error: a check is ~2.5% of the selection it replaces,
/// so 32 fruitless ones cost under one morsel's selection in total. Large enough that a
/// clustered input, whose first morsels legitimately exclude nothing while the bound is still
/// being established, is not written off before it has had a chance.
pub const PROBE_MORSELS: u64 = 32;

/// A bound on a top-N's cut-off, shared across the workers selecting its morsels.
///
/// See the module docs for the soundness argument, and for why this switches itself off on data
/// it cannot help. `descending` must match the top-N's first sort key, because it decides which
/// end of the range is "better".
#[derive(Debug)]
pub struct TopNBound {
    /// The bound value, meaningful only while `established` is true.
    value: AtomicI64,
    established: AtomicBool,
    /// Range checks made against an established bound, and how many excluded a morsel. Once the
    /// first reaches [`PROBE_MORSELS`] with the second still zero, the check stops paying for
    /// itself and `off` latches.
    checks: AtomicU64,
    exclusions: AtomicU64,
    off: AtomicBool,
    descending: bool,
}

impl TopNBound {
    /// A bound for a top-N whose first key sorts in the given direction. Starts unestablished,
    /// so nothing is skipped until some worker has seen `k` candidates.
    pub fn new(descending: bool) -> Self {
        Self {
            value: AtomicI64::new(if descending { i64::MIN } else { i64::MAX }),
            established: AtomicBool::new(false),
            checks: AtomicU64::new(0),
            exclusions: AtomicU64::new(0),
            off: AtomicBool::new(false),
            descending,
        }
    }

    /// Record that at least `k` rows are known to be at least as good as `value` on the first
    /// key. Keeps the tightest such claim seen.
    pub fn publish(&self, value: i64) {
        if self.descending {
            self.value.fetch_max(value, Ordering::Relaxed);
        } else {
            self.value.fetch_min(value, Ordering::Relaxed);
        }
        self.established.store(true, Ordering::Release);
    }

    /// The current bound, or `None` while no worker has established one.
    pub fn get(&self) -> Option<i64> {
        // `Acquire` pairs with `publish`'s `Release`, so a `true` here means the value store
        // that preceded it is visible. A stale `false` costs a missed skip, nothing more.
        self.established
            .load(Ordering::Acquire)
            .then(|| self.value.load(Ordering::Relaxed))
    }

    /// Whether every row of a morsel whose first-key range is `[min, max]` is strictly worse
    /// than the bound, and so cannot appear in the answer.
    ///
    /// `false` whenever the bound is not yet established or the range touches it, so a caller
    /// that gets `false` must do the full selection.
    pub fn excludes_range(&self, min: i64, max: i64) -> bool {
        if self.off.load(Ordering::Relaxed) {
            return false;
        }
        let Some(bound) = self.get() else {
            // No bound yet, so this is not evidence the bound is useless — do not count it.
            return false;
        };
        let excluded = if self.descending {
            // Descending: better is larger.
            max < bound
        } else {
            // Ascending: better is smaller, so the morsel is hopeless when even its smallest key
            // is worse than a value `k` rows already beat.
            min > bound
        };
        if excluded {
            self.exclusions.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        // Give up once enough checks against a real bound have excluded nothing. See the module
        // docs: on uniformly random keys that is every check, and the probe is not free.
        if self.checks.fetch_add(1, Ordering::Relaxed) + 1 >= PROBE_MORSELS
            && self.exclusions.load(Ordering::Relaxed) == 0
        {
            self.off.store(true, Ordering::Relaxed);
        }
        false
    }

    /// Whether the check has switched itself off for this operator. Observability only; the skip
    /// is invisible in a result either way.
    pub fn is_off(&self) -> bool {
        self.off.load(Ordering::Relaxed)
    }

    /// The bound value to publish from a candidate set's first-key column, or `None` when the
    /// set cannot support one (the wrong type, or any null — see the module docs).
    ///
    /// Ascending takes the candidates' **max**: every candidate is then at least as good as it.
    /// Descending takes the **min**, for the same reason.
    pub fn candidate_bound(&self, first_key: &ArrayRef) -> Option<i64> {
        let (min, max) = i64_key_range(first_key)?;
        Some(if self.descending { min } else { max })
    }
}

/// The `[min, max]` of an integer-like arrow array widened to `i64`, or `None` for a type this
/// does not handle, an empty array, or an array containing a null.
///
/// The null rejection is load-bearing rather than conservative: the caller's soundness argument
/// compares by magnitude, and a null has no magnitude. See the module docs.
pub fn i64_key_range(array: &ArrayRef) -> Option<(i64, i64)> {
    use arrow::array::{
        Date32Array, Date64Array, Int16Array, Int32Array, Int64Array, Int8Array,
        TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
        TimestampSecondArray,
    };
    use arrow::compute::kernels::aggregate::{max, min};
    use arrow::datatypes::TimeUnit;

    if array.is_empty() || array.null_count() > 0 {
        return None;
    }
    // One arm per concrete primitive, because arrow's `min`/`max` are generic over the typed
    // array rather than `dyn Array`. Widening to `i64` is lossless for every type listed and
    // order-preserving, which is all the caller needs.
    macro_rules! range {
        ($ty:ty) => {{
            let a = array.as_any().downcast_ref::<$ty>()?;
            Some((i64::from(min(a)?), i64::from(max(a)?)))
        }};
    }
    match array.data_type() {
        DataType::Int8 => range!(Int8Array),
        DataType::Int16 => range!(Int16Array),
        DataType::Int32 => range!(Int32Array),
        DataType::Date32 => range!(Date32Array),
        DataType::Int64 => {
            let a = array.as_any().downcast_ref::<Int64Array>()?;
            Some((min(a)?, max(a)?))
        }
        DataType::Date64 => {
            let a = array.as_any().downcast_ref::<Date64Array>()?;
            Some((min(a)?, max(a)?))
        }
        DataType::Timestamp(unit, _) => match unit {
            TimeUnit::Second => {
                let a = array.as_any().downcast_ref::<TimestampSecondArray>()?;
                Some((min(a)?, max(a)?))
            }
            TimeUnit::Millisecond => {
                let a = array.as_any().downcast_ref::<TimestampMillisecondArray>()?;
                Some((min(a)?, max(a)?))
            }
            TimeUnit::Microsecond => {
                let a = array.as_any().downcast_ref::<TimestampMicrosecondArray>()?;
                Some((min(a)?, max(a)?))
            }
            TimeUnit::Nanosecond => {
                let a = array.as_any().downcast_ref::<TimestampNanosecondArray>()?;
                Some((min(a)?, max(a)?))
            }
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use arrow::array::{Date32Array, Float64Array, Int32Array, Int64Array, StringArray};

    fn ints(v: Vec<Option<i64>>) -> ArrayRef {
        Arc::new(Int64Array::from(v))
    }

    #[test]
    fn nothing_is_skipped_before_a_bound_exists() {
        let b = TopNBound::new(false);
        assert!(b.get().is_none());
        assert!(!b.excludes_range(i64::MAX, i64::MAX));
    }

    #[test]
    fn ascending_excludes_only_strictly_worse_ranges() {
        let b = TopNBound::new(false);
        b.publish(100);
        assert_eq!(b.get(), Some(100));
        // Every key worse than the bound: hopeless.
        assert!(b.excludes_range(101, 200));
        // Touches the bound: a tie could still win on position, so it must be examined.
        assert!(!b.excludes_range(100, 200));
        // Straddles or beats it.
        assert!(!b.excludes_range(1, 200));
        assert!(!b.excludes_range(-5, 0));
    }

    #[test]
    fn descending_excludes_only_strictly_worse_ranges() {
        let b = TopNBound::new(true);
        b.publish(100);
        assert!(b.excludes_range(0, 99));
        assert!(!b.excludes_range(0, 100));
        assert!(!b.excludes_range(0, 500));
    }

    /// The bound may only tighten. A later, worse claim must not loosen it, or a morsel that
    /// should have been examined would be skipped.
    #[test]
    fn the_bound_only_tightens() {
        let asc = TopNBound::new(false);
        asc.publish(100);
        asc.publish(500);
        assert_eq!(asc.get(), Some(100));
        asc.publish(20);
        assert_eq!(asc.get(), Some(20));

        let desc = TopNBound::new(true);
        desc.publish(100);
        desc.publish(20);
        assert_eq!(desc.get(), Some(100));
        desc.publish(500);
        assert_eq!(desc.get(), Some(500));
    }

    /// Ascending publishes the candidates' max and descending their min, because that is the
    /// value all `k` candidates are known to be at least as good as.
    #[test]
    fn candidate_bound_takes_the_worst_of_the_candidates() {
        let candidates = ints(vec![Some(3), Some(9), Some(1)]);
        assert_eq!(TopNBound::new(false).candidate_bound(&candidates), Some(9));
        assert_eq!(TopNBound::new(true).candidate_bound(&candidates), Some(1));
    }

    /// A null in the candidate set means fewer than `k` rows are comparable by magnitude, so
    /// no bound may be published from it.
    #[test]
    fn a_null_candidate_publishes_no_bound() {
        let candidates = ints(vec![Some(3), None, Some(1)]);
        assert!(TopNBound::new(false).candidate_bound(&candidates).is_none());
        assert!(TopNBound::new(true).candidate_bound(&candidates).is_none());
    }

    /// The operator's actual sequence, so the pieces are shown to compose and not just to work
    /// individually: morsel 0 yields `k` candidates and publishes a bound, and the morsels after
    /// it are then excluded on their range alone.
    ///
    /// Without this, the equivalence tests above the operator would pass whether or not the skip
    /// ever engaged — a skip that silently never fires is the failure mode a correctness test
    /// cannot see.
    #[test]
    fn a_published_bound_excludes_the_later_morsels() {
        for descending in [false, true] {
            let bound = TopNBound::new(descending);
            // Morsel 0 holds the best rows. Its `k = 3` candidates are 0, 1, 2 ascending, or
            // 200, 199, 198 descending.
            let candidates = if descending {
                ints(vec![Some(200), Some(199), Some(198)])
            } else {
                ints(vec![Some(0), Some(1), Some(2)])
            };
            assert!(
                !bound.excludes_range(50, 60),
                "nothing skips before a bound"
            );
            let v = bound
                .candidate_bound(&candidates)
                .expect("three non-null candidates bound the cut-off");
            bound.publish(v);

            // A later morsel entirely on the wrong side is skipped. `[50, 60]` is strictly
            // worse than both bounds: above 2 ascending, below 198 descending.
            assert!(
                bound.excludes_range(50, 60),
                "descending={descending}: a range strictly worse than the bound must be skipped"
            );
            // ...one that reaches the bound is not, because a tie is decided by position.
            let touching = if descending { (50, v) } else { (v, 60) };
            assert!(
                !bound.excludes_range(touching.0, touching.1),
                "descending={descending}: a range touching the bound must still be examined"
            );
            // ...and one that beats it certainly is not.
            let better = if descending {
                (v + 1, v + 10)
            } else {
                (v - 10, v - 1)
            };
            assert!(!bound.excludes_range(better.0, better.1));
        }
    }

    /// On data the bound cannot help, the check must stop paying for itself. Without this the
    /// probe is a permanent 2.5% tax on every uniformly-distributed top-N — measured at 0 of 256
    /// morsels excluded, so the tax buys literally nothing there.
    #[test]
    fn a_useless_check_switches_itself_off() {
        let b = TopNBound::new(false);
        b.publish(100);
        for i in 0..PROBE_MORSELS {
            assert!(!b.excludes_range(0, 200), "check {i} excludes nothing");
        }
        assert!(
            b.is_off(),
            "after {PROBE_MORSELS} fruitless checks it must give up"
        );
        // And having given up it stays off, even for a range it would have excluded.
        assert!(!b.excludes_range(101, 200));
    }

    /// A check that *does* exclude keeps the mechanism alive indefinitely, so a clustered input
    /// is never written off after a fixed number of morsels.
    #[test]
    fn a_productive_check_never_switches_off() {
        let b = TopNBound::new(false);
        b.publish(100);
        for _ in 0..(PROBE_MORSELS * 4) {
            // Alternate a productive check with an unproductive one.
            assert!(b.excludes_range(101, 200));
            assert!(!b.excludes_range(0, 200));
        }
        assert!(
            !b.is_off(),
            "a mechanism that is working must not be disabled"
        );
    }

    /// Checks made before a bound exists are not evidence about the data, so they must not count
    /// toward the give-up budget — otherwise a top-N whose first morsels arrive before any
    /// publish would disable itself having never had a bound to test.
    #[test]
    fn checks_before_a_bound_do_not_count_toward_giving_up() {
        let b = TopNBound::new(false);
        for _ in 0..(PROBE_MORSELS * 2) {
            assert!(!b.excludes_range(0, 200));
        }
        assert!(!b.is_off());
        b.publish(100);
        assert!(b.excludes_range(101, 200), "the bound still works");
    }

    #[test]
    fn key_range_widens_the_integer_and_temporal_types() {
        assert_eq!(i64_key_range(&ints(vec![Some(-7), Some(4)])), Some((-7, 4)));
        let i32s: ArrayRef = Arc::new(Int32Array::from(vec![5_i32, -2]));
        assert_eq!(i64_key_range(&i32s), Some((-2, 5)));
        let dates: ArrayRef = Arc::new(Date32Array::from(vec![9_131_i32, 0]));
        assert_eq!(i64_key_range(&dates), Some((0, 9_131)));
    }

    /// A type this does not handle must return `None` so the caller does the full selection,
    /// never a wrong skip. Floats are excluded on purpose — see the module docs.
    #[test]
    fn key_range_declines_unsupported_types_and_nulls() {
        let floats: ArrayRef = Arc::new(Float64Array::from(vec![1.0_f64, 2.0]));
        assert!(i64_key_range(&floats).is_none());
        let strings: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        assert!(i64_key_range(&strings).is_none());
        assert!(i64_key_range(&ints(vec![Some(1), None])).is_none());
        assert!(i64_key_range(&ints(vec![])).is_none());
    }
}
