//! Which rows a window frame covers — the geometry, with no arithmetic over them.
//!
//! A frame answers one question per row: the half-open position range `[a, b)` of its
//! ordered partition that the aggregate runs over. Three unit systems answer it, and they
//! are genuinely different relations rather than three spellings of one:
//!
//! * **`ROWS`** counts physical rows, so `2 PRECEDING` is always two rows back.
//! * **`GROUPS`** counts peer groups (rows sharing an ORDER BY value), so a tie counts once.
//! * **`RANGE`** counts in the ORDER BY key's own **values**. `RANGE BETWEEN 300000000
//!   PRECEDING AND CURRENT ROW` over a microsecond timestamp is "the last five minutes",
//!   a window whose row count varies with how densely the series was sampled. Peer-shaped
//!   `RANGE` bounds (`CURRENT ROW`, `UNBOUNDED`) resolve through peer groups; a numeric one
//!   binary-searches the key's values ([`RangeSearch`]).
//!
//! The bound types mirror `bc_ir::FrameBound`/`FrameUnits` (bc-runtime does not depend on
//! bc-ir — the interpreter maps the IR enums to these, exactly as it does for
//! [`crate::window::WindowFn`]).
//!
//! **Both edges are non-decreasing in the row position** under every unit, which is the
//! property the kernels in [`super`] depend on: it is what makes the frame a FIFO queue and
//! the whole pass O(n) instead of O(n * frame width).

use arrow::array::{Array, ArrayRef};
use arrow::row::Rows;

use crate::error::RuntimeError;

/// One edge of a frame (mirror of `bc_ir::FrameBound`). The offsets are
/// non-negative counts relative to the current row — physical rows for `ROWS`,
/// peer groups for `GROUPS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameBound {
    UnboundedPreceding,
    Preceding(u64),
    CurrentRow,
    Following(u64),
    UnboundedFollowing,
}

/// Frame unit (mirror of `bc_ir::FrameUnits`): how the bound offsets are counted.
///
/// `Rows` counts physical rows. `Groups` counts peer groups (rows with an equal ORDER BY
/// value). `Range` counts in the ORDER BY key's *own values*: `RANGE BETWEEN 300000000
/// PRECEDING AND CURRENT ROW` over a microsecond timestamp is "the last five minutes", a
/// window whose row count varies with how densely the series was sampled. A `Range` bound
/// that is `CURRENT ROW` or `UNBOUNDED` resolves through peer groups exactly as `Groups`
/// does; a numeric one resolves by searching the key's values ([`RangeOrder`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameUnit {
    Rows,
    Range,
    Groups,
}

impl Frame {
    /// Whether either bound is a numeric `RANGE` offset, which needs the order key's
    /// *values* rather than only its peer structure.
    pub(crate) fn is_value_range(self) -> bool {
        self.unit == FrameUnit::Range
            && [self.start, self.end]
                .iter()
                .any(|b| matches!(b, FrameBound::Preceding(_) | FrameBound::Following(_)))
    }
}

/// The single ORDER BY key of a value-based `RANGE` frame, read as numbers.
///
/// SQL allows exactly one ORDER BY column with a numeric `RANGE` bound, for the reason this
/// type makes concrete: the bound is arithmetic on the key, and there is no arithmetic on a
/// tuple. Temporal keys are normalized to microseconds by [`crate::measure`], so an interval
/// bound means the same thing against a second- or nanosecond-resolution column.
pub struct RangeOrder {
    keys: crate::measure::NumericKeys,
    /// Per input row: whether the order key is non-null. Nulls sort into one contiguous
    /// block at one end of each partition, and a row whose key is null frames only its own
    /// null peers — there is no distance from a null to anything.
    valid: Vec<bool>,
    descending: bool,
}

impl RangeOrder {
    /// The elapsed distance between two rows' order keys, in the key's own units
    /// (microseconds for a temporal key), or `None` when either key is null.
    ///
    /// This is what lets a recurrence decay by *time* rather than by row position, which is
    /// the difference between smoothing an irregular sensor correctly and treating an hour's
    /// silence as one step. Unsigned, because a decay depends on how far apart two readings
    /// are and not on which way the partition is ordered.
    pub(crate) fn gap(&self, a: usize, b: usize) -> Option<f64> {
        if !self.valid[a] || !self.valid[b] {
            return None;
        }
        self.keys.distance(a, &self.keys, b)
    }

    /// Read the order key column, or `None` when it has no distance (a string, a struct),
    /// in which case the caller declines the frame rather than approximating it.
    pub fn read(key: &ArrayRef, descending: bool) -> Result<Option<Self>, RuntimeError> {
        Ok(
            crate::measure::NumericKeys::read(key)?.map(|keys| RangeOrder {
                keys,
                valid: (0..key.len()).map(|i| key.is_valid(i)).collect(),
                descending,
            }),
        )
    }
}

/// The per-partition search state a value-based `RANGE` bound binary-searches over.
struct RangeSearch<'a> {
    order: &'a RangeOrder,
    part: &'a [usize],
    /// The contiguous `[lo, hi)` position range whose keys are non-null. A sort places every
    /// null at one end, so the non-null keys are contiguous *and* monotone, which is what
    /// makes the search valid.
    lo: usize,
    hi: usize,
}

impl RangeSearch<'_> {
    fn new<'a>(order: &'a RangeOrder, part: &'a [usize]) -> RangeSearch<'a> {
        let lo = part
            .iter()
            .position(|&r| order.valid[r])
            .unwrap_or(part.len());
        let hi = part
            .iter()
            .rposition(|&r| order.valid[r])
            .map_or(lo, |p| p + 1);
        RangeSearch {
            order,
            part,
            lo,
            hi,
        }
    }

    /// The frame's first position for a bound `delta` away in value space.
    ///
    /// Ascending keys want the first row at or above the target; descending keys the first
    /// row at or below it. Both are `partition_point` over a predicate that is true for a
    /// prefix, and both are non-decreasing in `pos` because the key is monotone — which is
    /// what keeps the one-pass sliding aggregate valid.
    fn start(&self, pos: usize, delta: i128) -> usize {
        let row = self.part[pos];
        let slice = &self.part[self.lo..self.hi];
        let desc = self.order.descending;
        self.lo
            + slice.partition_point(|&r| {
                let c = self.order.keys.shifted_cmp(row, delta, r);
                if desc {
                    c.is_gt()
                } else {
                    c.is_lt()
                }
            })
    }

    /// The frame's one-past-last position for a bound `delta` away in value space.
    fn end(&self, pos: usize, delta: i128) -> usize {
        let row = self.part[pos];
        let slice = &self.part[self.lo..self.hi];
        let desc = self.order.descending;
        self.lo
            + slice.partition_point(|&r| {
                let c = self.order.keys.shifted_cmp(row, delta, r);
                if desc {
                    c.is_ge()
                } else {
                    c.is_le()
                }
            })
    }
}

/// Everything a frame needs beyond the row position: peer groups for the peer-counted
/// bounds, and the key values for the value-counted ones. A frame may mix the two
/// (`RANGE BETWEEN 300000000 PRECEDING AND CURRENT ROW` does), so both can be present.
pub(crate) struct FrameCtx<'a> {
    peers: Option<PeerGroups>,
    range: Option<RangeSearch<'a>>,
}

/// An explicit frame: the inclusive `[start, end]` range each output row aggregates
/// over, counted in `unit`s.
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    pub unit: FrameUnit,
    pub start: FrameBound,
    pub end: FrameBound,
}

/// The peer-group structure of an ordered partition: which group each position
/// belongs to, and each group's `[start, end)` position range. Peers are adjacent
/// ordered rows with an equal ORDER BY value. Used to resolve `RANGE`/`GROUPS` frame
/// bounds to a contiguous position range (peers are contiguous once sorted).
pub(crate) struct PeerGroups {
    group_of: Vec<usize>,    // group index per ordered position
    group_start: Vec<usize>, // first position of each group
    group_end: Vec<usize>,   // one-past-last position of each group
}

impl PeerGroups {
    pub(crate) fn new(part: &[usize], rows: &Rows) -> Self {
        let len = part.len();
        let mut group_of = Vec::with_capacity(len);
        let mut group_start = Vec::new();
        let mut group_end = Vec::new();
        let mut g = 0usize;
        // Carry the previous position's encoded row rather than re-reading it. `part` holds
        // row indices into a sort permutation, so `rows.row(part[pos])` is a random access
        // into the encoded buffer — and the old form did two of them per position to compare
        // neighbours, when one of the two was already read on the previous iteration.
        let mut prev = None;
        for (pos, &p) in part.iter().enumerate() {
            let cur = rows.row(p);
            if pos == 0 {
                group_start.push(0);
            } else if prev != Some(cur) {
                group_end.push(pos);
                group_start.push(pos);
                g += 1;
            }
            prev = Some(cur);
            group_of.push(g);
        }
        if len > 0 {
            group_end.push(len);
        }
        PeerGroups {
            group_of,
            group_start,
            group_end,
        }
    }

    fn num(&self) -> usize {
        self.group_start.len()
    }
}

/// Resolve a frame to the half-open `[a, b)` position range within an ordered
/// partition of length `len` for the row at `pos`. `ROWS` counts physical rows;
/// `RANGE`/`GROUPS` count peer groups via `peers`. Both `a` and `b` are
/// non-decreasing in `pos`, which is what lets the aggregate slide in one pass.
pub(crate) fn frame_bounds(
    frame: Frame,
    pos: usize,
    len: usize,
    ctx: Option<&FrameCtx<'_>>,
) -> (usize, usize) {
    // A value-based RANGE bound is arithmetic on the key, so it is resolved by searching
    // the key's values; the peer path below counts *groups* and cannot express it.
    if let Some(search) = ctx.and_then(|c| c.range.as_ref()) {
        if frame.is_value_range() {
            return value_range_bounds(frame, pos, len, search, ctx.and_then(|c| c.peers.as_ref()));
        }
    }
    match frame.unit {
        FrameUnit::Rows => frame_half_open(frame, pos, len),
        FrameUnit::Range | FrameUnit::Groups => {
            let pg = ctx
                .and_then(|c| c.peers.as_ref())
                .expect("RANGE/GROUPS frame requires peer groups");
            let g = pg.group_of[pos] as i64;
            let ng = pg.num() as i64;
            // Offsets saturate into `i64` (they are `u64` in the IR) so a huge peer-group
            // offset clamps to the partition edge instead of wrapping negative / overflowing.
            let lo = match frame.start {
                FrameBound::UnboundedPreceding => 0,
                FrameBound::Preceding(k) => {
                    pg.group_start[g.saturating_sub(sat_i64(k)).max(0) as usize]
                }
                FrameBound::CurrentRow => pg.group_start[g as usize],
                FrameBound::Following(k) => {
                    let gi = g.saturating_add(sat_i64(k));
                    if gi >= ng {
                        len
                    } else {
                        pg.group_start[gi as usize]
                    }
                }
                FrameBound::UnboundedFollowing => len,
            };
            let hi = match frame.end {
                FrameBound::UnboundedPreceding => 0,
                FrameBound::Preceding(k) => {
                    let gi = g.saturating_sub(sat_i64(k));
                    if gi < 0 {
                        0
                    } else {
                        pg.group_end[gi as usize]
                    }
                }
                FrameBound::CurrentRow => pg.group_end[g as usize],
                FrameBound::Following(k) => {
                    pg.group_end[g.saturating_add(sat_i64(k)).min(ng - 1) as usize]
                }
                FrameBound::UnboundedFollowing => len,
            };
            (lo.min(len), hi.min(len))
        }
    }
}

/// Resolve a value-based `RANGE` frame to the half-open `[a, b)` position range.
///
/// Each numeric bound is a *value* offset from the current row's key: `n PRECEDING` covers
/// keys down to `key - n` and `n FOLLOWING` up to `key + n`, whichever rows those turn out
/// to be. `CURRENT ROW` and `UNBOUNDED` in the same frame still mean what they mean, so they
/// resolve through the peer groups beside the search.
///
/// A row whose own key is null has no distance to anything, so its frame is exactly its null
/// peer group — Postgres' rule, and the only one under which the frame is well defined.
fn value_range_bounds(
    frame: Frame,
    pos: usize,
    len: usize,
    search: &RangeSearch<'_>,
    peers: Option<&PeerGroups>,
) -> (usize, usize) {
    let pg = peers.expect("a RANGE frame always carries peer groups");
    if !search.order.valid[search.part[pos]] {
        let g = pg.group_of[pos];
        return (pg.group_start[g].min(len), pg.group_end[g].min(len));
    }
    // `PRECEDING` means *earlier in the ordering*, which is a smaller value ascending and a
    // **larger** one descending — so the sort direction flips the offset's sign in value
    // space, it does not merely mirror which end of the partition the search lands on.
    // Reading `PRECEDING` as "down the value axis" regardless produced an empty frame for
    // every descending row, which looks like a null column rather than like a bug.
    let sign: i128 = if search.order.descending { -1 } else { 1 };
    let delta = |b: FrameBound| -> i128 {
        match b {
            FrameBound::Preceding(k) => -sign * sat_i64(k) as i128,
            FrameBound::Following(k) => sign * sat_i64(k) as i128,
            _ => 0,
        }
    };
    let g = pg.group_of[pos];
    let lo = match frame.start {
        FrameBound::UnboundedPreceding => 0,
        FrameBound::CurrentRow => pg.group_start[g],
        b => search.start(pos, delta(b)),
    };
    let hi = match frame.end {
        FrameBound::UnboundedFollowing => len,
        FrameBound::CurrentRow => pg.group_end[g],
        b => search.end(pos, delta(b)),
    };
    (lo.min(len), hi.min(len))
}

/// Saturate a `u64` frame offset into an `i64`. The IR carries offsets as `u64`, so a
/// bound like `10_000_000_000_000_000_000 PRECEDING` (valid, `> i64::MAX`) must NOT be
/// truncated by a raw `as i64` (which wraps negative and silently flips the bound's
/// direction). Capping at `i64::MAX` keeps a huge offset a huge offset — it clamps to the
/// partition edge below, as intended — and, combined with saturating adds, removes the
/// `pos + k + 1` overflow panic that `ROWS BETWEEN CURRENT ROW AND <i64::MAX> FOLLOWING`
/// triggered (DuckDB accepts that frame).
fn sat_i64(k: u64) -> i64 {
    k.min(i64::MAX as u64) as i64
}

/// Resolve a `ROWS` frame to the half-open `[a, b)` row range within an ordered
/// partition of length `len` for the row at `pos`. Both `a` and `b` are
/// non-decreasing in `pos` (each is `pos + const`, clamped to `[0, len]`), which is
/// what lets the aggregate slide in one pass. An empty frame yields `a >= b`.
pub(super) fn frame_half_open(frame: Frame, pos: usize, len: usize) -> (usize, usize) {
    let (pos, n) = (pos as i64, len as i64);
    // `pos + 1` cannot overflow (`pos < len <= isize::MAX`); the offset adds saturate so a
    // near-`i64::MAX` offset clamps to the partition edge instead of overflowing.
    let lo = match frame.start {
        FrameBound::UnboundedPreceding => 0,
        FrameBound::Preceding(k) => pos.saturating_sub(sat_i64(k)),
        FrameBound::CurrentRow => pos,
        FrameBound::Following(k) => pos.saturating_add(sat_i64(k)),
        FrameBound::UnboundedFollowing => n, // start past the last row → empty
    };
    let hi_excl = match frame.end {
        FrameBound::UnboundedPreceding => 0, // end before the first row → empty
        FrameBound::Preceding(k) => (pos + 1).saturating_sub(sat_i64(k)),
        FrameBound::CurrentRow => pos + 1,
        FrameBound::Following(k) => (pos + 1).saturating_add(sat_i64(k)),
        FrameBound::UnboundedFollowing => n,
    };
    (lo.clamp(0, n) as usize, hi_excl.clamp(0, n) as usize)
}

/// Build whatever a partition's frame needs beyond the row position: peer groups for a
/// `RANGE`/`GROUPS` frame, plus a value search when the `RANGE` bounds are numeric.
/// `None` for a `ROWS` frame, which needs neither.
pub(crate) fn frame_ctx<'a>(
    frame: Frame,
    part: &'a [usize],
    order_rows: Option<&Rows>,
    range_order: Option<&'a RangeOrder>,
) -> Option<FrameCtx<'a>> {
    match frame.unit {
        FrameUnit::Rows => None,
        FrameUnit::Range | FrameUnit::Groups => Some(FrameCtx {
            peers: Some(PeerGroups::new(
                part,
                order_rows.expect("range/groups needs order"),
            )),
            range: range_order
                .filter(|_| frame.is_value_range())
                .map(|o| RangeSearch::new(o, part)),
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::Int64Array;
    use arrow::row::{RowConverter, SortField};

    use super::*;

    /// Build the pieces a value-based `RANGE` bound needs: the key read as numbers, the
    /// peer groups its `CURRENT ROW` bounds resolve through, and the whole-partition index
    /// list. `keys` are given in *ordered* position order, as the window kernel produces.
    fn ctx(keys: Vec<Option<i64>>, descending: bool) -> (RangeOrder, PeerGroups, Vec<usize>) {
        let arr: ArrayRef = Arc::new(Int64Array::from(keys.clone()));
        let order = RangeOrder::read(&arr, descending)
            .unwrap()
            .expect("int is measurable");
        let conv = RowConverter::new(vec![SortField::new(arr.data_type().clone())]).unwrap();
        let rows = conv.convert_columns(std::slice::from_ref(&arr)).unwrap();
        let part: Vec<usize> = (0..keys.len()).collect();
        let peers = PeerGroups::new(&part, &rows);
        (order, peers, part)
    }

    fn bounds(
        keys: Vec<Option<i64>>,
        descending: bool,
        start: FrameBound,
        end: FrameBound,
    ) -> Vec<(usize, usize)> {
        let (order, peers, part) = ctx(keys.clone(), descending);
        let search = RangeSearch::new(&order, &part);
        let frame = Frame {
            unit: FrameUnit::Range,
            start,
            end,
        };
        (0..keys.len())
            .map(|pos| value_range_bounds(frame, pos, keys.len(), &search, Some(&peers)))
            .collect()
    }

    /// The window is a span of *values*, so an isolated key gets a frame of one row while a
    /// dense run gets many — the whole point of a `RANGE` frame over a `ROWS` one.
    #[test]
    fn a_trailing_value_window_widens_and_narrows_with_the_data() {
        let keys = vec![Some(1), Some(2), Some(3), Some(100), Some(101)];
        let got = bounds(
            keys,
            false,
            FrameBound::Preceding(2),
            FrameBound::CurrentRow,
        );
        // Values within 2 below each key: {1}, {1,2}, {1,2,3}, {100}, {100,101}.
        assert_eq!(got, vec![(0, 1), (0, 2), (0, 3), (3, 4), (3, 5)]);
    }

    /// Every row tied on the order key shares one window, because the window is defined by
    /// the value and they have the same value.
    #[test]
    fn tied_keys_share_one_window() {
        let keys = vec![Some(5), Some(5), Some(5), Some(9)];
        let got = bounds(
            keys,
            false,
            FrameBound::Preceding(1),
            FrameBound::Following(1),
        );
        assert_eq!(got[0], got[1]);
        assert_eq!(got[1], got[2]);
        assert_eq!(got[0], (0, 3), "the three fives, and nothing else");
    }

    /// Under a descending order `PRECEDING` means a *larger* value. Reading it as "down the
    /// value axis" regardless yields an empty frame for every row, which looks like a null
    /// column rather than like a bug.
    #[test]
    fn descending_order_measures_preceding_upward() {
        let keys = vec![Some(10), Some(9), Some(8), Some(1)];
        let got = bounds(keys, true, FrameBound::Preceding(2), FrameBound::CurrentRow);
        // For key 8 (position 2) the window is values in [8, 10] -> positions 0..3.
        assert_eq!(got, vec![(0, 1), (0, 2), (0, 3), (3, 4)]);
    }

    /// A null order key has no distance to anything, so its frame is exactly its own null
    /// peer group — and a non-null row never reaches across into the nulls.
    #[test]
    fn a_null_key_frames_only_its_own_peers() {
        let keys = vec![None, None, Some(1), Some(2)];
        let got = bounds(
            keys,
            false,
            FrameBound::Preceding(10),
            FrameBound::Following(10),
        );
        assert_eq!(got[0], (0, 2), "the null peer group");
        assert_eq!(got[1], (0, 2));
        assert_eq!(got[2], (2, 4), "a real key never reaches into the nulls");
        assert_eq!(got[3], (2, 4));
    }

    /// A frame whose whole span sits below the smallest key is empty, not clamped to the
    /// first row — the difference between "no data in the last five minutes" and a wrong
    /// number.
    #[test]
    fn a_window_with_no_rows_in_it_is_empty() {
        let keys = vec![Some(0), Some(100)];
        let got = bounds(
            keys,
            false,
            FrameBound::Preceding(10),
            FrameBound::Preceding(5),
        );
        assert!(got[0].0 >= got[0].1, "nothing 5..10 below 0");
        assert!(got[1].0 >= got[1].1, "nothing 5..10 below 100");
    }

    /// A huge offset must not wrap: the shift is done in `i128`, so `i64::MIN - i64::MAX`
    /// stays hugely negative instead of flipping positive and inverting the bound.
    ///
    /// Note what the right answer *is* here, because it is not "everything": a value offset
    /// of `i64::MAX` from `0` still cannot reach `i64::MIN`, so that row is genuinely
    /// outside the window. A row offset would have clamped to the partition edge; a value
    /// offset answers a question about values, and the keys here span the whole domain.
    #[test]
    fn a_huge_value_offset_does_not_wrap() {
        let keys = vec![Some(i64::MIN), Some(0), Some(i64::MAX)];
        let got = bounds(
            keys,
            false,
            FrameBound::Preceding(u64::MAX),
            FrameBound::Following(u64::MAX),
        );
        // From i64::MIN: everything at or below `MIN + MAX == -1`, so only itself.
        assert_eq!(got[0], (0, 1));
        // From 0 and from i64::MAX: everything at or above `-MAX`, which excludes MIN.
        assert_eq!(got[1], (1, 3));
        assert_eq!(got[2], (1, 3));
        // Every frame is well formed — a wrap would have produced start > end.
        for (a, b) in got {
            assert!(a <= b, "a huge offset inverted the frame: ({a}, {b})");
        }
    }

    /// Both edges must be non-decreasing in the row position, whatever the bounds — the
    /// property every sliding kernel above depends on for its single pass.
    #[test]
    fn both_edges_are_non_decreasing_in_position() {
        let keys: Vec<Option<i64>> = (0..40)
            .map(|i| if i % 7 == 3 { None } else { Some((i * 3) % 29) })
            .collect();
        let mut sorted: Vec<Option<i64>> = keys.clone();
        sorted.sort();
        let offsets = [
            FrameBound::UnboundedPreceding,
            FrameBound::Preceding(5),
            FrameBound::CurrentRow,
            FrameBound::Following(5),
            FrameBound::UnboundedFollowing,
        ];
        for descending in [false, true] {
            let ordered: Vec<Option<i64>> = match descending {
                false => sorted.clone(),
                true => sorted.iter().rev().copied().collect(),
            };
            for &start in &offsets {
                for &end in &offsets {
                    let got = bounds(ordered.clone(), descending, start, end);
                    for w in got.windows(2) {
                        assert!(
                            w[0].0 <= w[1].0 && w[0].1 <= w[1].1,
                            "desc={descending} {start:?}..{end:?} went backwards: {w:?}"
                        );
                    }
                }
            }
        }
    }
}
