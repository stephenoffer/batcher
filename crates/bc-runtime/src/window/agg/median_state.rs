//! The running median's two-heap, kept apart from the aggregates that use it.
//!
//! It is a data structure rather than an aggregate arm: a max-heap of the lower half and a
//! min-heap of the upper, rebalanced after each insert so the median sits at one or both
//! tops. `O(log n)` per row and `O(1)` to read, which is the cost that lets an order
//! statistic sit beside the `O(1)` folds in the parent module instead of being refused.
//!
//! What it deliberately does not have is `remove`. That is the whole reason an explicit
//! frame still declines: a sliding median would have to evict the leaving row, and the
//! heaps cannot do it in better than linear time.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// `f64` under a total order, so the running median's heaps can hold NaN without
/// `BinaryHeap` needing `Ord` it cannot get from `f64`. The order is the engine's own
/// (`keys::float_total_cmp`), which is what makes this median rank NaN where the
/// `GROUP BY` one does.
#[derive(Clone, Copy, PartialEq)]
struct TotalF64(f64);

impl Eq for TotalF64 {}

impl Ord for TotalF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        crate::keys::float_total_cmp(self.0, other.0)
    }
}

impl PartialOrd for TotalF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A median that admits values one at a time, in `O(log n)` each.
///
/// The lower half sits in a max-heap and the upper half in a min-heap, kept within one
/// element of each other, so the median is always at one or both tops. This is the
/// structure the module docstring asks an order statistic to bring: no sort per row, and
/// no rebuild.
#[derive(Default)]
pub(super) struct RunningMedian {
    lower: BinaryHeap<TotalF64>,
    upper: BinaryHeap<Reverse<TotalF64>>,
}

impl RunningMedian {
    pub(super) fn push(&mut self, v: f64) {
        let v = TotalF64(v);
        match self.lower.peek() {
            Some(top) if v > *top => self.upper.push(Reverse(v)),
            _ => self.lower.push(v),
        }
        // Restore |lower| == |upper| or |lower| == |upper| + 1, so the odd-count median is
        // always `lower`'s top and the even-count one is the mean of the two tops.
        if self.lower.len() > self.upper.len() + 1 {
            let m = self.lower.pop().expect("lower is non-empty");
            self.upper.push(Reverse(m));
        } else if self.upper.len() > self.lower.len() {
            let Reverse(m) = self.upper.pop().expect("upper is non-empty");
            self.lower.push(m);
        }
    }

    pub(super) fn median(&self) -> Option<f64> {
        let lo = self.lower.peek()?.0;
        if self.lower.len() > self.upper.len() {
            return Some(lo);
        }
        let hi = self
            .upper
            .peek()
            .expect("balanced heaps are both non-empty")
            .0
             .0;
        Some((lo + hi) / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two-heap must agree with the quickselect the `GROUP BY` median uses, at *every*
    /// prefix length — both parities, and with the values arriving in an adversarial order
    /// so the rebalance is exercised in both directions.
    #[test]
    fn the_running_median_agrees_with_quickselect_at_every_prefix() {
        let feed = [5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 0.0, -3.0];
        let mut state = RunningMedian::default();
        let mut seen: Vec<f64> = Vec::new();
        assert_eq!(state.median(), None, "no values is null, not zero");
        for v in feed {
            state.push(v);
            seen.push(v);
            let expected = crate::agg::median::quickselect_median(&mut seen.clone());
            assert_eq!(
                state.median(),
                Some(expected),
                "prefix of {} disagreed",
                seen.len()
            );
        }
    }

    /// Ascending and descending feeds are the two orders that keep one heap starved, so a
    /// rebalance bug shows up here and nowhere else.
    #[test]
    fn a_monotone_feed_keeps_the_heaps_balanced() {
        for descending in [false, true] {
            let mut state = RunningMedian::default();
            let mut seen: Vec<f64> = Vec::new();
            for k in 0..21i32 {
                let v = if descending { -k } else { k } as f64;
                state.push(v);
                seen.push(v);
                let expected = crate::agg::median::quickselect_median(&mut seen.clone());
                assert_eq!(state.median(), Some(expected));
            }
        }
    }

    /// NaN must land where the `GROUP BY` median puts it, which is what the total order in
    /// `TotalF64` is for — a `PartialOrd` heap would have ordered it arbitrarily.
    #[test]
    fn nan_ranks_the_same_way_the_group_by_median_ranks_it() {
        let feed = [1.0, f64::NAN, 3.0];
        let mut state = RunningMedian::default();
        for v in feed {
            state.push(v);
        }
        let expected = crate::agg::median::quickselect_median(&mut feed.to_vec());
        assert_eq!(state.median().map(f64::to_bits), Some(expected.to_bits()));
    }
}
