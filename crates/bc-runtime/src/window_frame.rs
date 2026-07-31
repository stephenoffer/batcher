//! Explicit `ROWS` window frames — sliding-window aggregates.
//!
//! The default window frame (`RANGE UNBOUNDED PRECEDING TO CURRENT ROW`, with
//! peer-tie semantics) is handled by `window::running_aggregate`. This module adds
//! *explicit* `ROWS BETWEEN <start> AND <end>` frames: for each row, aggregate the
//! physical rows in `[start, end]` of its ordered partition. The result is the
//! same relation a SQL engine produces for `ROWS` frames.
//!
//! Bounds are mirror types of `bc_ir::FrameBound` (bc-runtime does not depend on
//! bc-ir — the interpreter maps the IR enum to these, exactly as it does for
//! [`crate::window::WindowFn`]). Only the aggregate functions
//! (`sum`/`avg`/`min`/`max`/`count`) take a frame.
//!
//! Both frame edges are non-decreasing in the row position (each is `pos + const`,
//! clamped), so the frame only ever slides right — the frame is a FIFO queue. The
//! kernel exploits this to run in **one pass**: `count`/`sum` over integers keep a
//! running accumulator (drop the leaving row, add the entering one — O(n)); `sum`/`avg`
//! over floats use a [`FifoSum`] (a two-stack sliding aggregate that never subtracts,
//! because subtracting floats is catastrophically unstable); `min`/`max` keep a
//! monotonic deque (O(n) amortized). No frame is rescanned.

use std::collections::VecDeque;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Float64Array, Int64Array, StringArray, UInt32Array,
};
use arrow::compute::take;
use arrow::datatypes::{DataType, Float64Type, Int64Type};
use arrow::row::Rows;

use crate::error::RuntimeError;
use crate::window::WindowFn;

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
/// `Rows` counts physical rows; `Range`/`Groups` count peer groups (rows with an
/// equal ORDER BY value). `Range` is only reached for peer bounds (CURRENT ROW /
/// UNBOUNDED) — a numeric `RANGE` offset falls back upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameUnit {
    Rows,
    Range,
    Groups,
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

/// Sliding-window sum over `f64` that only ever **adds**, never subtracts — a FIFO of
/// values kept as two cumulative-sum stacks (the classic "queue from two stacks").
///
/// The naive O(1) slide (`sum += entering; sum -= leaving`) is catastrophically
/// unstable on floats: over `[1e16, 1, 1]` a trailing 2-row sum computes
/// `1e16 + 1 - 1e16 == 0`, where the true window sum `1 + 1` is `2` (this is exactly
/// the divergence from DuckDB that motivated the structure). Here the reported sum is
/// always the cumulative sum of precisely the values currently in the window, so it
/// equals a fresh re-add — at O(1) amortized cost, no subtraction of a stale value.
#[derive(Default)]
struct FifoSum {
    /// `(value, cumulative sum of this entry and all below it)` — the push side.
    back: Vec<(f64, f64)>,
    /// The pop side; filled by draining `back` in reverse when it empties.
    front: Vec<(f64, f64)>,
}

impl FifoSum {
    /// Append a value at the back of the window.
    fn push(&mut self, v: f64) {
        let s = self.back.last().map_or(0.0, |&(_, s)| s) + v;
        self.back.push((v, s));
    }

    /// Remove the oldest value from the front of the window.
    fn pop(&mut self) {
        if self.front.is_empty() {
            // Reverse `back` into `front`, rebuilding cumulative sums bottom-up so the
            // oldest value ends up on top of `front` (popped first — FIFO order).
            let mut s = 0.0;
            while let Some((v, _)) = self.back.pop() {
                s += v;
                self.front.push((v, s));
            }
        }
        self.front.pop();
    }

    /// The exact sum of every value currently in the window.
    fn sum(&self) -> f64 {
        self.back.last().map_or(0.0, |&(_, s)| s) + self.front.last().map_or(0.0, |&(_, s)| s)
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
    peers: Option<&PeerGroups>,
) -> (usize, usize) {
    match frame.unit {
        FrameUnit::Rows => frame_half_open(frame, pos, len),
        FrameUnit::Range | FrameUnit::Groups => {
            let pg = peers.expect("RANGE/GROUPS frame requires peer groups");
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
fn frame_half_open(frame: Frame, pos: usize, len: usize) -> (usize, usize) {
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

/// Compute an explicit-frame aggregate, scattered to original row order. `ROWS`
/// frames ignore `order_rows`; `RANGE`/`GROUPS` frames require it (peer groups).
pub fn framed_aggregate(
    func: WindowFn,
    ordered: &[Vec<usize>],
    values: &ArrayRef,
    frame: Frame,
    order_rows: Option<&Rows>,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    if frame.unit != FrameUnit::Rows && order_rows.is_none() {
        return Err(RuntimeError::WindowRequiresOrder {
            func: func.name().to_string(),
        });
    }
    match func {
        WindowFn::Count => Ok(framed_count(ordered, values, frame, order_rows, num_rows)),
        WindowFn::Sum | WindowFn::Avg | WindowFn::Min | WindowFn::Max => match values.data_type() {
            DataType::Int64 => framed_i64(func, ordered, values, frame, order_rows, num_rows),
            DataType::Float64 => Ok(framed_f64(
                func, ordered, values, frame, order_rows, num_rows,
            )),
            // String MIN/MAX over an explicit frame — the running/whole-partition paths
            // already support Utf8 min/max, so the framed path must too (DuckDB answers
            // `min(s) OVER (… ROWS …)`; erroring here aborted the query).
            DataType::Utf8 if matches!(func, WindowFn::Min | WindowFn::Max) => Ok(
                framed_str_minmax(func, ordered, values, frame, order_rows, num_rows),
            ),
            // Boolean framed MIN (AND) / MAX (OR), `false < true` — consistent with the
            // running and whole-partition boolean paths and DuckDB.
            DataType::Boolean if matches!(func, WindowFn::Min | WindowFn::Max) => Ok(
                framed_bool_minmax(func, ordered, values, frame, order_rows, num_rows),
            ),
            other => Err(RuntimeError::UnsupportedWindow {
                func: func.name().to_string(),
                dtype: other.to_string(),
            }),
        },
        // The folds `window_agg` owns get a frame through the same two-stack slide, so
        // this arm is now only reached by an aggregate with no sliding form at all.
        other if crate::window_agg::is_extended_aggregate(other) => {
            crate::window_agg::framed(other, ordered, values, frame, order_rows, num_rows)
        }
        other => Err(RuntimeError::UnsupportedWindow {
            func: other.name().to_string(),
            dtype: "explicit frame".to_string(),
        }),
    }
}

/// Positional value functions (`first_value`/`last_value`/`nth_value`) over an
/// explicit frame: each output row selects its frame's first / last / nth row's
/// value (type-generic via `take`), or `null` when the frame is empty (or the
/// `nth` row is past the frame end).
///
/// This is what makes SQL's default value-function frame —
/// `RANGE UNBOUNDED PRECEDING TO CURRENT ROW` — a *running* value: `last_value`
/// becomes the current peer group's value and `nth_value` is null until the frame
/// grows to the `nth` row, rather than the whole-partition value the frameless
/// [`crate::window`] path computes. `lag`/`lead`/the fills never carry a frame, so
/// they are rejected here.
pub fn framed_value(
    func: WindowFn,
    ordered: &[Vec<usize>],
    values: &ArrayRef,
    nth: i64,
    frame: Frame,
    order_rows: Option<&Rows>,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    if frame.unit != FrameUnit::Rows && order_rows.is_none() {
        return Err(RuntimeError::WindowRequiresOrder {
            func: func.name().to_string(),
        });
    }
    // Per output row, the ordered-partition position whose value it takes (`None` →
    // null). Scatter back to original row order via a single `take`.
    let mut src: Vec<Option<u32>> = vec![None; num_rows];
    for part in ordered {
        let len = part.len();
        let peers = peer_groups(frame, part, order_rows);
        for pos in 0..len {
            let (a, b) = frame_bounds(frame, pos, len, peers.as_ref());
            let take_pos: Option<usize> = if a >= b {
                None // empty frame → null
            } else {
                match func {
                    WindowFn::FirstValue => Some(a),
                    WindowFn::LastValue => Some(b - 1),
                    // `nth_value`: the `nth`-th row (1-based) counting from the frame
                    // start; null if the frame holds fewer than `nth` rows.
                    // Compare the 1-based offset against the frame *width* rather than
                    // forming `a + (nth - 1)`: `nth` arrives unvalidated from the IR, and
                    // that sum overflows i64 for a large `nth`, wrapping to a negative that
                    // passes an `idx < b` bound check and then indexes out of bounds.
                    WindowFn::NthValue => {
                        (nth >= 1 && nth - 1 < (b - a) as i64).then(|| a + (nth - 1) as usize)
                    }
                    other => {
                        return Err(RuntimeError::UnsupportedWindow {
                            func: other.name().to_string(),
                            dtype: "explicit frame".to_string(),
                        })
                    }
                }
            };
            src[part[pos]] = take_pos.map(|p| part[p] as u32);
        }
    }
    Ok(take(values.as_ref(), &UInt32Array::from(src), None)?)
}

/// Build the peer-group structure for a partition when the frame needs it (RANGE/
/// GROUPS); `None` for ROWS frames.
pub(crate) fn peer_groups(
    frame: Frame,
    part: &[usize],
    order_rows: Option<&Rows>,
) -> Option<PeerGroups> {
    match frame.unit {
        FrameUnit::Rows => None,
        FrameUnit::Range | FrameUnit::Groups => Some(PeerGroups::new(
            part,
            order_rows.expect("range/groups needs order"),
        )),
    }
}

/// `count` over the frame: number of non-null values (0 for an empty frame),
/// slid in one pass — add the entering row, subtract the leaving one.
fn framed_count(
    ordered: &[Vec<usize>],
    values: &ArrayRef,
    frame: Frame,
    order_rows: Option<&Rows>,
    num_rows: usize,
) -> ArrayRef {
    let mut out = vec![0i64; num_rows];
    for part in ordered {
        let len = part.len();
        let peers = peer_groups(frame, part, order_rows);
        let (mut cur_a, mut cur_b, mut cnt) = (0usize, 0usize, 0i64);
        for pos in 0..len {
            let (a, b) = frame_bounds(frame, pos, len, peers.as_ref());
            while cur_b < b {
                if values.is_valid(part[cur_b]) {
                    cnt += 1;
                }
                cur_b += 1;
            }
            while cur_a < a {
                // Only undo rows that were actually added (guards empty frames where
                // `a` overtakes `b`).
                if cur_a < cur_b && values.is_valid(part[cur_a]) {
                    cnt -= 1;
                }
                cur_a += 1;
            }
            cur_b = cur_b.max(cur_a);
            out[part[pos]] = cnt;
        }
    }
    Arc::new(Int64Array::from(out))
}

/// Integer-input frame aggregate. `sum`/`min`/`max` stay `Int64`; `avg` is
/// `Float64`. An all-null or empty frame yields null (`avg` too).
fn framed_i64(
    func: WindowFn,
    ordered: &[Vec<usize>],
    values: &ArrayRef,
    frame: Frame,
    order_rows: Option<&Rows>,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    let arr = values.as_primitive::<Int64Type>();
    let mut out_i = vec![None::<i64>; num_rows];
    let mut out_f = vec![None::<f64>; num_rows];
    let is_min = func == WindowFn::Min;
    let need_extreme = matches!(func, WindowFn::Min | WindowFn::Max);
    // Only SUM/AVG accumulate the running window sum; a MIN/MAX/COUNT query must not,
    // or a window whose values sum past i64::MAX would spuriously error (`checked_add`
    // → SumOverflow) on a query where the sum is never read.
    let need_sum = matches!(func, WindowFn::Sum | WindowFn::Avg);
    for part in ordered {
        let len = part.len();
        let peers = peer_groups(frame, part, order_rows);
        let (mut cur_a, mut cur_b) = (0usize, 0usize);
        let (mut sum, mut cnt) = (0i64, 0i64);
        // Monotonic deque of partition positions holding the running min/max front.
        let mut dq: VecDeque<usize> = VecDeque::new();
        for pos in 0..len {
            let (a, b) = frame_bounds(frame, pos, len, peers.as_ref());
            // Remove the *leaving* rows before adding the *entering* ones. Both bounds are
            // non-decreasing, so either order yields the same frame — but adding first
            // makes the accumulator transiently hold the union of the old and new frames,
            // a superset of both. For i64 SUM that union can overflow when neither frame
            // does (`ROWS CURRENT ROW` over `[i64::MAX, 1]`), and `checked_add` would abort
            // a perfectly valid query with `SumOverflow`. Removing first keeps `sum` exactly
            // over `[cur_a, cur_b)` at every point.
            while cur_a < a {
                if cur_a < cur_b && arr.is_valid(part[cur_a]) {
                    if need_sum {
                        sum -= arr.value(part[cur_a]);
                    }
                    cnt -= 1;
                }
                cur_a += 1;
            }
            cur_b = cur_b.max(cur_a);
            if need_extreme {
                while let Some(&front) = dq.front() {
                    if front < cur_a {
                        dq.pop_front();
                    } else {
                        break;
                    }
                }
            }
            while cur_b < b {
                let row = part[cur_b];
                if arr.is_valid(row) {
                    let v = arr.value(row);
                    // `checked_add` so an i64 framed SUM that overflows errors instead of
                    // wrapping (matching the non-framed running/whole-partition SUM paths).
                    if need_sum {
                        sum = sum.checked_add(v).ok_or(RuntimeError::SumOverflow)?;
                    }
                    cnt += 1;
                    if need_extreme {
                        while let Some(&back) = dq.back() {
                            let bv = arr.value(part[back]);
                            if (is_min && bv >= v) || (!is_min && bv <= v) {
                                dq.pop_back();
                            } else {
                                break;
                            }
                        }
                        dq.push_back(cur_b);
                    }
                }
                cur_b += 1;
            }
            if cnt == 0 {
                continue; // empty / all-null frame → null
            }
            match func {
                WindowFn::Sum => out_i[part[pos]] = Some(sum),
                WindowFn::Avg => out_f[part[pos]] = Some(sum as f64 / cnt as f64),
                WindowFn::Min | WindowFn::Max => {
                    out_i[part[pos]] = dq.front().map(|&f| arr.value(part[f]));
                }
                _ => unreachable!("framed_i64 on non-aggregate"),
            }
        }
    }
    Ok(if func == WindowFn::Avg {
        Arc::new(Float64Array::from(out_f))
    } else {
        Arc::new(Int64Array::from(out_i))
    })
}

/// Float-input frame aggregate (`sum`/`avg`/`min`/`max`, all `Float64`).
fn framed_f64(
    func: WindowFn,
    ordered: &[Vec<usize>],
    values: &ArrayRef,
    frame: Frame,
    order_rows: Option<&Rows>,
    num_rows: usize,
) -> ArrayRef {
    let arr = values.as_primitive::<Float64Type>();
    let mut out = vec![None::<f64>; num_rows];
    let is_min = func == WindowFn::Min;
    let need_extreme = matches!(func, WindowFn::Min | WindowFn::Max);
    let need_sum = matches!(func, WindowFn::Sum | WindowFn::Avg);
    for part in ordered {
        let len = part.len();
        let peers = peer_groups(frame, part, order_rows);
        let (mut cur_a, mut cur_b) = (0usize, 0usize);
        // `sum` is a two-stack FIFO (adds only, never subtracts) so a sliding SUM/AVG
        // over large-magnitude floats stays exact; `cnt` is an exact integer counter.
        let mut sum = FifoSum::default();
        let mut cnt = 0i64;
        let mut dq: VecDeque<usize> = VecDeque::new();
        for pos in 0..len {
            let (a, b) = frame_bounds(frame, pos, len, peers.as_ref());
            while cur_b < b {
                let row = part[cur_b];
                // One FIFO entry per physical position (nulls contribute 0.0 to the sum
                // but are not counted), so the FIFO length tracks `cur_b - cur_a` exactly
                // and pops stay aligned with the sliding `[cur_a, cur_b)` window.
                if need_sum {
                    sum.push(if arr.is_valid(row) {
                        arr.value(row)
                    } else {
                        0.0
                    });
                }
                if arr.is_valid(row) {
                    let v = arr.value(row);
                    cnt += 1;
                    if need_extreme {
                        // Total-order comparison so NaN sorts greatest, matching aggregate
                        // MIN/MAX and DuckDB; raw `>=`/`<=` are all-false against NaN, which
                        // corrupted the monotonic deque (NaN neither popped nor was popped).
                        while let Some(&back) = dq.back() {
                            let bv = arr.value(part[back]);
                            let drop = if is_min {
                                crate::keys::float_total_cmp(bv, v).is_ge()
                            } else {
                                crate::keys::float_total_cmp(bv, v).is_le()
                            };
                            if drop {
                                dq.pop_back();
                            } else {
                                break;
                            }
                        }
                        dq.push_back(cur_b);
                    }
                }
                cur_b += 1;
            }
            while cur_a < a {
                if cur_a < cur_b {
                    if need_sum {
                        sum.pop();
                    }
                    if arr.is_valid(part[cur_a]) {
                        cnt -= 1;
                    }
                }
                cur_a += 1;
            }
            cur_b = cur_b.max(cur_a);
            if need_extreme {
                while let Some(&front) = dq.front() {
                    if front < cur_a {
                        dq.pop_front();
                    } else {
                        break;
                    }
                }
            }
            if cnt == 0 {
                continue;
            }
            out[part[pos]] = Some(match func {
                WindowFn::Sum => sum.sum(),
                WindowFn::Avg => sum.sum() / cnt as f64,
                WindowFn::Min | WindowFn::Max => dq.front().map_or(0.0, |&f| arr.value(part[f])),
                _ => unreachable!("framed_f64 on non-aggregate"),
            });
        }
    }
    Arc::new(Float64Array::from(out))
}

/// String-input frame aggregate (`min`/`max` only). Same monotonic-deque slide as the
/// numeric `min`/`max`, comparing UTF-8 byte order (`<`/`>`, matching the running and
/// whole-partition string paths and DuckDB's binary collation). An empty / all-null
/// frame yields null.
fn framed_str_minmax(
    func: WindowFn,
    ordered: &[Vec<usize>],
    values: &ArrayRef,
    frame: Frame,
    order_rows: Option<&Rows>,
    num_rows: usize,
) -> ArrayRef {
    let arr = values.as_any().downcast_ref::<StringArray>().expect("utf8");
    let is_min = func == WindowFn::Min;
    let mut out: Vec<Option<String>> = vec![None; num_rows];
    for part in ordered {
        let len = part.len();
        let peers = peer_groups(frame, part, order_rows);
        let (mut cur_a, mut cur_b) = (0usize, 0usize);
        // Monotonic deque of partition positions holding the running min/max front.
        let mut dq: VecDeque<usize> = VecDeque::new();
        for pos in 0..len {
            let (a, b) = frame_bounds(frame, pos, len, peers.as_ref());
            while cur_b < b {
                let row = part[cur_b];
                if arr.is_valid(row) {
                    let v = arr.value(row);
                    while let Some(&back) = dq.back() {
                        let bv = arr.value(part[back]);
                        if (is_min && bv >= v) || (!is_min && bv <= v) {
                            dq.pop_back();
                        } else {
                            break;
                        }
                    }
                    dq.push_back(cur_b);
                }
                cur_b += 1;
            }
            while cur_a < a {
                cur_a += 1;
            }
            cur_b = cur_b.max(cur_a);
            while let Some(&front) = dq.front() {
                if front < cur_a {
                    dq.pop_front();
                } else {
                    break;
                }
            }
            out[part[pos]] = dq.front().map(|&f| arr.value(part[f]).to_string());
        }
    }
    Arc::new(StringArray::from(out))
}

/// Boolean-input frame aggregate (`min`/`max` only), ordering `false < true` (min = AND,
/// max = OR). Same monotonic-deque slide as the string/numeric min/max; an empty /
/// all-null frame yields null.
fn framed_bool_minmax(
    func: WindowFn,
    ordered: &[Vec<usize>],
    values: &ArrayRef,
    frame: Frame,
    order_rows: Option<&Rows>,
    num_rows: usize,
) -> ArrayRef {
    let arr = values.as_boolean();
    let is_min = func == WindowFn::Min;
    let mut out: Vec<Option<bool>> = vec![None; num_rows];
    for part in ordered {
        let len = part.len();
        let peers = peer_groups(frame, part, order_rows);
        let (mut cur_a, mut cur_b) = (0usize, 0usize);
        let mut dq: VecDeque<usize> = VecDeque::new();
        for pos in 0..len {
            let (a, b) = frame_bounds(frame, pos, len, peers.as_ref());
            while cur_b < b {
                let row = part[cur_b];
                if arr.is_valid(row) {
                    let v = arr.value(row);
                    while let Some(&back) = dq.back() {
                        let bv = arr.value(part[back]);
                        if (is_min && bv >= v) || (!is_min && bv <= v) {
                            dq.pop_back();
                        } else {
                            break;
                        }
                    }
                    dq.push_back(cur_b);
                }
                cur_b += 1;
            }
            while cur_a < a {
                cur_a += 1;
            }
            cur_b = cur_b.max(cur_a);
            while let Some(&front) = dq.front() {
                if front < cur_a {
                    dq.pop_front();
                } else {
                    break;
                }
            }
            out[part[pos]] = dq.front().map(|&f| arr.value(part[f]));
        }
    }
    Arc::new(BooleanArray::from(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_half_open_clamps_and_empties() {
        let f = Frame {
            unit: FrameUnit::Rows,
            start: FrameBound::Preceding(1),
            end: FrameBound::Following(1),
        };
        assert_eq!(frame_half_open(f, 0, 5), (0, 2)); // clamped at left edge
        assert_eq!(frame_half_open(f, 4, 5), (3, 5)); // clamped at right edge
        assert_eq!(frame_half_open(f, 2, 5), (1, 4));

        // 1 FOLLOWING .. 2 FOLLOWING at the last row → empty (a >= b).
        let ff = Frame {
            unit: FrameUnit::Rows,
            start: FrameBound::Following(1),
            end: FrameBound::Following(2),
        };
        let (a, b) = frame_half_open(ff, 4, 5);
        assert!(a >= b);
        let (a0, b0) = frame_half_open(f, 0, 0); // empty partition
        assert!(a0 >= b0);
    }

    /// A frame offset near/above `i64::MAX` (valid in the `u64` IR) must saturate to the
    /// partition edge, not wrap negative (`k as i64`) or overflow (`pos + k + 1`). Before
    /// the fix, `CURRENT ROW .. i64::MAX FOLLOWING` panicked with "attempt to add with
    /// overflow", and `<huge> PRECEDING .. CURRENT ROW` flipped to an empty frame (all
    /// null). DuckDB accepts the `i64::MAX` frame and returns the suffix/prefix aggregate.
    #[test]
    fn huge_frame_offsets_saturate_not_overflow() {
        let len = 5usize;
        // CURRENT ROW .. i64::MAX FOLLOWING → to the partition end.
        let f_end = Frame {
            unit: FrameUnit::Rows,
            start: FrameBound::CurrentRow,
            end: FrameBound::Following(i64::MAX as u64),
        };
        assert_eq!(frame_half_open(f_end, 0, len), (0, 5));
        assert_eq!(frame_half_open(f_end, 3, len), (3, 5));

        // A huge (> i64::MAX) PRECEDING start → the partition start (whole prefix).
        let f_start = Frame {
            unit: FrameUnit::Rows,
            start: FrameBound::Preceding(u64::MAX),
            end: FrameBound::CurrentRow,
        };
        assert_eq!(frame_half_open(f_start, 0, len), (0, 1));
        assert_eq!(frame_half_open(f_start, 4, len), (0, 5));

        // A huge PRECEDING *end* still yields an empty frame (ends before the start).
        let f_pe = Frame {
            unit: FrameUnit::Rows,
            start: FrameBound::UnboundedPreceding,
            end: FrameBound::Preceding(u64::MAX),
        };
        let (a, b) = frame_half_open(f_pe, 3, len);
        assert!(a >= b, "huge PRECEDING end must be empty, got ({a},{b})");

        // End to end through the aggregate kernel: suffix SUM must not panic and equals
        // the whole-suffix total.
        let values: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50]));
        let ordered = vec![vec![0usize, 1, 2, 3, 4]];
        let s = framed_aggregate(WindowFn::Sum, &ordered, &values, f_end, None, len).unwrap();
        assert_eq!(
            s.as_primitive::<Int64Type>().values(),
            &[150, 140, 120, 90, 50]
        );
    }

    #[test]
    fn rows_frame_sliding_sum_and_avg() {
        // One partition [0,1,2,3,4] over values [10,20,30,40,50], ROWS 1 PRECEDING
        // .. CURRENT ROW → trailing pair sums: 10,30,50,70,90.
        let values: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50]));
        let ordered = vec![vec![0usize, 1, 2, 3, 4]];
        let frame = Frame {
            unit: FrameUnit::Rows,
            start: FrameBound::Preceding(1),
            end: FrameBound::CurrentRow,
        };
        let s = framed_aggregate(WindowFn::Sum, &ordered, &values, frame, None, 5).unwrap();
        let s = s.as_primitive::<Int64Type>();
        assert_eq!(s.values(), &[10, 30, 50, 70, 90]);

        // avg over the same trailing pair: 10, 15, 25, 35, 45.
        let a = framed_aggregate(WindowFn::Avg, &ordered, &values, frame, None, 5).unwrap();
        let a = a.as_primitive::<Float64Type>();
        assert_eq!(a.values(), &[10.0, 15.0, 25.0, 35.0, 45.0]);
    }

    #[test]
    fn rows_frame_centered_min_max_count() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![5, 1, 9, 3, 7]));
        let ordered = vec![vec![0usize, 1, 2, 3, 4]];
        let frame = Frame {
            unit: FrameUnit::Rows,
            start: FrameBound::Preceding(1),
            end: FrameBound::Following(1),
        };
        let mn = framed_aggregate(WindowFn::Min, &ordered, &values, frame, None, 5).unwrap();
        assert_eq!(mn.as_primitive::<Int64Type>().values(), &[1, 1, 1, 3, 3]);
        let mx = framed_aggregate(WindowFn::Max, &ordered, &values, frame, None, 5).unwrap();
        assert_eq!(mx.as_primitive::<Int64Type>().values(), &[5, 9, 9, 9, 7]);
        let c = framed_aggregate(WindowFn::Count, &ordered, &values, frame, None, 5).unwrap();
        assert_eq!(c.as_primitive::<Int64Type>().values(), &[2, 3, 3, 3, 2]);
    }

    /// String MIN/MAX over an explicit ROWS frame must slide like the numeric path
    /// (and match a naive per-row recompute), not error `UnsupportedWindow`.
    #[test]
    fn rows_frame_string_min_max() {
        use arrow::array::StringArray;
        let values: ArrayRef = Arc::new(StringArray::from(vec![
            Some("d"),
            None,
            Some("a"),
            Some("c"),
            Some("b"),
        ]));
        let ordered = vec![vec![0usize, 1, 2, 3, 4]];
        let frame = Frame {
            unit: FrameUnit::Rows,
            start: FrameBound::Preceding(1),
            end: FrameBound::Following(1),
        };
        let raw = ["d", "", "a", "c", "b"];
        let valid = [true, false, true, true, true];
        let mn = framed_aggregate(WindowFn::Min, &ordered, &values, frame, None, 5).unwrap();
        let mx = framed_aggregate(WindowFn::Max, &ordered, &values, frame, None, 5).unwrap();
        let mn = mn.as_any().downcast_ref::<StringArray>().unwrap();
        let mx = mx.as_any().downcast_ref::<StringArray>().unwrap();
        for pos in 0..5usize {
            let (a, b) = frame_half_open(frame, pos, 5);
            let win: Vec<&str> = (a..b).filter(|&j| valid[j]).map(|j| raw[j]).collect();
            let want_mn = win.iter().min().copied();
            let want_mx = win.iter().max().copied();
            assert_eq!(
                mn.is_valid(pos).then(|| mn.value(pos)),
                want_mn,
                "min pos {pos}"
            );
            assert_eq!(
                mx.is_valid(pos).then(|| mx.value(pos)),
                want_mx,
                "max pos {pos}"
            );
        }
    }

    /// Boolean MIN (AND) / MAX (OR) over an explicit ROWS frame must slide (not error),
    /// ordering `false < true`, matching a naive recompute.
    #[test]
    fn rows_frame_bool_min_max() {
        use arrow::array::BooleanArray;
        let raw = [Some(true), Some(false), None, Some(true), Some(false)];
        let values: ArrayRef = Arc::new(BooleanArray::from(raw.to_vec()));
        let ordered = vec![vec![0usize, 1, 2, 3, 4]];
        let frame = Frame {
            unit: FrameUnit::Rows,
            start: FrameBound::Preceding(1),
            end: FrameBound::Following(1),
        };
        let mn = framed_aggregate(WindowFn::Min, &ordered, &values, frame, None, 5).unwrap();
        let mx = framed_aggregate(WindowFn::Max, &ordered, &values, frame, None, 5).unwrap();
        let mn = mn.as_any().downcast_ref::<BooleanArray>().unwrap();
        let mx = mx.as_any().downcast_ref::<BooleanArray>().unwrap();
        for pos in 0..5usize {
            let (a, b) = frame_half_open(frame, pos, 5);
            let win: Vec<bool> = (a..b).filter_map(|j| raw[j]).collect();
            assert_eq!(
                mn.is_valid(pos).then(|| mn.value(pos)),
                win.iter().copied().min(),
                "min pos {pos}"
            );
            assert_eq!(
                mx.is_valid(pos).then(|| mx.value(pos)),
                win.iter().copied().max(),
                "max pos {pos}"
            );
        }
    }

    /// GROUPS frames aggregate by peer group (ties in the ORDER BY key), not by
    /// physical row count.
    #[test]
    fn groups_frame_aggregates_by_peer_group() {
        use arrow::row::{RowConverter, SortField};
        // Order key [10,10,20,20,30] → peer groups {0,1}, {2,3}, {4}.
        let keys: ArrayRef = Arc::new(Int64Array::from(vec![10, 10, 20, 20, 30]));
        let conv = RowConverter::new(vec![SortField::new(keys.data_type().clone())]).unwrap();
        let rows = conv.convert_columns(std::slice::from_ref(&keys)).unwrap();
        let values: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5]));
        let ordered = vec![vec![0usize, 1, 2, 3, 4]];

        // GROUPS CURRENT ROW .. CURRENT ROW → sum within the current peer group.
        let f = Frame {
            unit: FrameUnit::Groups,
            start: FrameBound::CurrentRow,
            end: FrameBound::CurrentRow,
        };
        let s = framed_aggregate(WindowFn::Sum, &ordered, &values, f, Some(&rows), 5).unwrap();
        assert_eq!(s.as_primitive::<Int64Type>().values(), &[3, 3, 7, 7, 5]);

        // GROUPS 1 PRECEDING .. CURRENT ROW → current group plus the one before.
        let f2 = Frame {
            unit: FrameUnit::Groups,
            start: FrameBound::Preceding(1),
            end: FrameBound::CurrentRow,
        };
        let s2 = framed_aggregate(WindowFn::Sum, &ordered, &values, f2, Some(&rows), 5).unwrap();
        assert_eq!(s2.as_primitive::<Int64Type>().values(), &[3, 3, 10, 10, 12]);

        // RANGE UNBOUNDED PRECEDING .. CURRENT ROW → cumulative through current peers.
        let f3 = Frame {
            unit: FrameUnit::Range,
            start: FrameBound::UnboundedPreceding,
            end: FrameBound::CurrentRow,
        };
        let s3 = framed_aggregate(WindowFn::Sum, &ordered, &values, f3, Some(&rows), 5).unwrap();
        assert_eq!(s3.as_primitive::<Int64Type>().values(), &[3, 3, 10, 10, 15]);
    }

    /// The default value-function frame (`RANGE UNBOUNDED PRECEDING TO CURRENT ROW`)
    /// makes `last_value` running (the current peer group's value) and `nth_value`
    /// null-until-the-nth-row — matching DuckDB / standard SQL, not the whole-partition
    /// value the frameless path gives. Order key `[10,10,20,20,30]` has peer groups
    /// {0,1},{2,3},{4}, so `last_value` = [2,2,4,4,5] and `nth_value(v,2)` = [null,2,2,2,4].
    #[test]
    fn default_range_frame_running_last_and_nth_value() {
        use arrow::row::{RowConverter, SortField};
        let keys: ArrayRef = Arc::new(Int64Array::from(vec![10, 10, 20, 20, 30]));
        let conv = RowConverter::new(vec![SortField::new(keys.data_type().clone())]).unwrap();
        let rows = conv.convert_columns(std::slice::from_ref(&keys)).unwrap();
        let values: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5]));
        let ordered = vec![vec![0usize, 1, 2, 3, 4]];
        let frame = Frame {
            unit: FrameUnit::Range,
            start: FrameBound::UnboundedPreceding,
            end: FrameBound::CurrentRow,
        };

        let lv = framed_value(
            WindowFn::LastValue,
            &ordered,
            &values,
            1,
            frame,
            Some(&rows),
            5,
        )
        .unwrap();
        assert_eq!(lv.as_primitive::<Int64Type>().values(), &[2, 2, 4, 4, 5]);

        let fv = framed_value(
            WindowFn::FirstValue,
            &ordered,
            &values,
            1,
            frame,
            Some(&rows),
            5,
        )
        .unwrap();
        assert_eq!(fv.as_primitive::<Int64Type>().values(), &[1, 1, 1, 1, 1]);

        let nv = framed_value(
            WindowFn::NthValue,
            &ordered,
            &values,
            2,
            frame,
            Some(&rows),
            5,
        )
        .unwrap();
        let nv = nv.as_primitive::<Int64Type>();
        let got: Vec<Option<i64>> = (0..5)
            .map(|i| nv.is_valid(i).then(|| nv.value(i)))
            .collect();
        // Frames [0,2),[0,2),[0,4),[0,4),[0,5) all hold the 2nd row (index 1 = value 2),
        // even at the first output row, because its peer includes row 1.
        assert_eq!(got, vec![Some(2), Some(2), Some(2), Some(2), Some(2)]);

        // nth_value past the frame end is null: nth_value(v, 3) is null until the frame
        // holds 3 rows (from the second peer group onward: index 2 = value 3).
        let nv3 = framed_value(
            WindowFn::NthValue,
            &ordered,
            &values,
            3,
            frame,
            Some(&rows),
            5,
        )
        .unwrap();
        let nv3 = nv3.as_primitive::<Int64Type>();
        let got3: Vec<Option<i64>> = (0..5)
            .map(|i| nv3.is_valid(i).then(|| nv3.value(i)))
            .collect();
        assert_eq!(got3, vec![None, None, Some(3), Some(3), Some(3)]);
    }

    /// A ROWS frame on a value function selects the frame's first/last/nth physical row.
    #[test]
    fn rows_frame_value_functions() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50]));
        let ordered = vec![vec![0usize, 1, 2, 3, 4]];
        // ROWS BETWEEN 1 PRECEDING AND CURRENT ROW.
        let frame = Frame {
            unit: FrameUnit::Rows,
            start: FrameBound::Preceding(1),
            end: FrameBound::CurrentRow,
        };
        let lv = framed_value(WindowFn::LastValue, &ordered, &values, 1, frame, None, 5).unwrap();
        assert_eq!(
            lv.as_primitive::<Int64Type>().values(),
            &[10, 20, 30, 40, 50]
        );
        let fv = framed_value(WindowFn::FirstValue, &ordered, &values, 1, frame, None, 5).unwrap();
        // frame start = max(pos-1, 0): [10,10,20,30,40].
        assert_eq!(
            fv.as_primitive::<Int64Type>().values(),
            &[10, 10, 20, 30, 40]
        );
    }

    /// A huge `nth` must yield null, not overflow the frame-relative index.
    ///
    /// Regression: the index was computed as `a as i64 + (nth - 1)`, which wraps
    /// negative for a large `nth` once the frame start `a` is past 0. The wrapped value
    /// then passed the `idx < b as i64` bound check and was cast back with `as usize`,
    /// indexing the partition far out of bounds — a panic in release, an arithmetic
    /// overflow abort in debug. `nth` reaches here unvalidated from the IR offset (the
    /// Python builder checks only `n >= 1`). Comparing the offset against the frame
    /// *width* instead never forms the large sum.
    #[test]
    fn nth_value_with_huge_n_is_null_not_a_panic() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50]));
        let ordered = vec![vec![0usize, 1, 2, 3, 4]];
        // ROWS BETWEEN 2 PRECEDING AND CURRENT ROW — `a` reaches 2, so `a + (nth-1)`
        // overflows for `nth` near `i64::MAX`.
        let frame = Frame {
            unit: FrameUnit::Rows,
            start: FrameBound::Preceding(2),
            end: FrameBound::CurrentRow,
        };
        for nth in [i64::MAX, i64::MAX - 1, 1 << 62] {
            let got =
                framed_value(WindowFn::NthValue, &ordered, &values, nth, frame, None, 5).unwrap();
            assert_eq!(got.null_count(), 5, "nth_value({nth}) must be all null");
        }
    }

    /// A sliding float SUM over large-magnitude values must stay exact. The naive
    /// add-then-subtract slide computed `1e16 + 1 - 1e16 == 0` for a trailing 2-row
    /// window over `[1e16, 1, 1, 1]`, where DuckDB (and a fresh re-add) gives 2.0. The
    /// FIFO-of-two-stacks sum never subtracts, so it recovers the exact window sum.
    #[test]
    fn sliding_float_sum_is_exact_over_large_magnitudes() {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![1e16, 1.0, 1.0, 1.0]));
        let ordered = vec![vec![0usize, 1, 2, 3]];
        let frame = Frame {
            unit: FrameUnit::Rows,
            start: FrameBound::Preceding(1),
            end: FrameBound::CurrentRow,
        };
        let s = framed_aggregate(WindowFn::Sum, &ordered, &values, frame, None, 4).unwrap();
        assert_eq!(
            s.as_primitive::<Float64Type>().values(),
            &[1e16, 1e16, 2.0, 2.0]
        );
        // AVG shares the same accumulator, so it must be exact too: (1+1)/2 = 1.0.
        let a = framed_aggregate(WindowFn::Avg, &ordered, &values, frame, None, 4).unwrap();
        assert_eq!(a.as_primitive::<Float64Type>().values()[2], 1.0);
    }

    /// An i64 framed SUM that overflows must error (like the non-framed running SUM),
    /// not wrap silently / panic in debug. A trailing 2-row window whose two entries
    /// sum past i64::MAX triggers it.
    #[test]
    fn framed_i64_sum_overflow_errors() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![i64::MAX, 1]));
        let ordered = vec![vec![0usize, 1]];
        let frame = Frame {
            unit: FrameUnit::Rows,
            start: FrameBound::Preceding(1),
            end: FrameBound::CurrentRow,
        };
        let r = framed_aggregate(WindowFn::Sum, &ordered, &values, frame, None, 2);
        assert!(matches!(r, Err(RuntimeError::SumOverflow)));
    }

    /// A framed `MIN`/`MAX` over i64 must NOT error just because the window's values
    /// happen to sum past `i64::MAX` — the sum is irrelevant to the extremes. Before the
    /// fix, `framed_i64` accumulated the running window sum unconditionally (with
    /// `checked_add` → `SumOverflow`), so `min(x) OVER (… ROWS …)` aborted the query on
    /// large-magnitude input where DuckDB returns the minimum.
    #[test]
    fn framed_i64_minmax_does_not_spuriously_overflow_on_sum() {
        let big = 1i64 << 62; // three of these sum to 3·2^62 >> i64::MAX
        let values: ArrayRef = Arc::new(Int64Array::from(vec![big, big, big]));
        let ordered = vec![vec![0usize, 1, 2]];
        let frame = Frame {
            unit: FrameUnit::Rows,
            start: FrameBound::Preceding(2),
            end: FrameBound::CurrentRow,
        };
        let mn = framed_aggregate(WindowFn::Min, &ordered, &values, frame, None, 3).unwrap();
        assert_eq!(mn.as_primitive::<Int64Type>().values(), &[big, big, big]);
        let mx = framed_aggregate(WindowFn::Max, &ordered, &values, frame, None, 3).unwrap();
        assert_eq!(mx.as_primitive::<Int64Type>().values(), &[big, big, big]);
    }

    /// A sliding i64 SUM/AVG must not overflow on a frame whose own values fit.
    ///
    /// Regression: the slide added the *entering* rows before removing the *leaving*
    /// ones, so between the two loops the accumulator held the union of the old and new
    /// frames — a superset of either. With `ROWS BETWEEN CURRENT ROW AND CURRENT ROW`
    /// over `[i64::MAX, 1]`, every individual frame is a single value that fits, but the
    /// transient union `{i64::MAX, 1}` does not, and `checked_add` aborted the whole
    /// query with `SumOverflow`. Both bounds are non-decreasing, so removing first is
    /// equivalent and keeps the frame exact at every point.
    #[test]
    fn framed_i64_sum_does_not_overflow_on_a_frame_that_fits() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![i64::MAX, 1]));
        let ordered = vec![vec![0usize, 1]];
        let frame = Frame {
            unit: FrameUnit::Rows,
            start: FrameBound::CurrentRow,
            end: FrameBound::CurrentRow,
        };
        let s = framed_aggregate(WindowFn::Sum, &ordered, &values, frame, None, 2).unwrap();
        assert_eq!(s.as_primitive::<Int64Type>().values(), &[i64::MAX, 1]);
        let a = framed_aggregate(WindowFn::Avg, &ordered, &values, frame, None, 2).unwrap();
        assert_eq!(
            a.as_primitive::<Float64Type>().values(),
            &[i64::MAX as f64, 1.0]
        );
    }

    /// The O(n) sliding kernel must match a naive O(n·w) recompute for every frame
    /// shape — including nulls, empty frames, and multiple partitions.
    #[test]
    fn sliding_matches_naive_oracle() {
        // Two partitions over 9 rows; some nulls. `ordered` lists row indices in
        // each partition's sort order (deliberately not identity, to exercise the
        // scatter back to original order).
        let raw = vec![
            Some(5),
            None,
            Some(3),
            Some(8),
            Some(1),
            None,
            Some(7),
            Some(2),
            Some(4),
        ];
        let values: ArrayRef = Arc::new(Int64Array::from(raw.clone()));
        let ordered = vec![vec![0usize, 2, 4, 6, 8], vec![1usize, 3, 5, 7]];
        let n = raw.len();

        let bounds = [
            FrameBound::UnboundedPreceding,
            FrameBound::Preceding(2),
            FrameBound::Preceding(1),
            FrameBound::CurrentRow,
            FrameBound::Following(1),
            FrameBound::Following(2),
            FrameBound::UnboundedFollowing,
        ];
        for &start in &bounds {
            for &end in &bounds {
                let frame = Frame {
                    unit: FrameUnit::Rows,
                    start,
                    end,
                };
                for func in [
                    WindowFn::Sum,
                    WindowFn::Avg,
                    WindowFn::Min,
                    WindowFn::Max,
                    WindowFn::Count,
                ] {
                    let got = framed_aggregate(func, &ordered, &values, frame, None, n).unwrap();
                    let want = naive(func, &ordered, &raw, frame, n);
                    assert_eq!(fmt(&got), want, "func={func:?} start={start:?} end={end:?}");
                }
            }
        }
    }

    /// The float SUM/AVG sliding kernel (FIFO of two stacks) must match a naive
    /// per-row recompute across every frame shape, including nulls and empty frames —
    /// the same cross-check as `sliding_matches_naive_oracle` but on the float path.
    #[test]
    fn sliding_float_matches_naive_oracle() {
        let raw: Vec<Option<f64>> = vec![
            Some(5.0),
            None,
            Some(3.0),
            Some(8.0),
            Some(1.0),
            None,
            Some(7.0),
            Some(2.0),
            Some(4.0),
        ];
        let values: ArrayRef = Arc::new(Float64Array::from(raw.clone()));
        let ordered = vec![vec![0usize, 2, 4, 6, 8], vec![1usize, 3, 5, 7]];
        let n = raw.len();
        let bounds = [
            FrameBound::UnboundedPreceding,
            FrameBound::Preceding(2),
            FrameBound::Preceding(1),
            FrameBound::CurrentRow,
            FrameBound::Following(1),
            FrameBound::Following(2),
            FrameBound::UnboundedFollowing,
        ];
        for &start in &bounds {
            for &end in &bounds {
                let frame = Frame {
                    unit: FrameUnit::Rows,
                    start,
                    end,
                };
                for func in [WindowFn::Sum, WindowFn::Avg] {
                    let got = framed_aggregate(func, &ordered, &values, frame, None, n).unwrap();
                    let want = naive_f64(func, &ordered, &raw, frame, n);
                    assert_eq!(fmt(&got), want, "func={func:?} start={start:?} end={end:?}");
                }
            }
        }
    }

    // Naive float reference: recompute each row's frame directly.
    fn naive_f64(
        func: WindowFn,
        ordered: &[Vec<usize>],
        raw: &[Option<f64>],
        frame: Frame,
        n: usize,
    ) -> Vec<Option<f64>> {
        let mut out = vec![None; n];
        for part in ordered {
            let len = part.len();
            for pos in 0..len {
                let (a, b) = frame_half_open(frame, pos, len);
                let vals: Vec<f64> = (a..b).filter_map(|j| raw[part[j]]).collect();
                out[part[pos]] = if vals.is_empty() {
                    None
                } else {
                    let s: f64 = vals.iter().sum();
                    Some(if func == WindowFn::Avg {
                        s / vals.len() as f64
                    } else {
                        s
                    })
                };
            }
        }
        out
    }

    // Naive reference: recompute each row's frame directly. Returns each output as
    // an `Option<f64>` so int/float/count compare uniformly.
    fn naive(
        func: WindowFn,
        ordered: &[Vec<usize>],
        raw: &[Option<i64>],
        frame: Frame,
        n: usize,
    ) -> Vec<Option<f64>> {
        let mut out = vec![None; n];
        for part in ordered {
            let len = part.len();
            for pos in 0..len {
                let (a, b) = frame_half_open(frame, pos, len);
                let vals: Vec<i64> = (a..b).filter_map(|j| raw[part[j]]).collect();
                let v = match func {
                    WindowFn::Count => Some(vals.len() as f64),
                    _ if vals.is_empty() => None,
                    WindowFn::Sum => Some(vals.iter().sum::<i64>() as f64),
                    WindowFn::Avg => Some(vals.iter().sum::<i64>() as f64 / vals.len() as f64),
                    WindowFn::Min => Some(*vals.iter().min().unwrap() as f64),
                    WindowFn::Max => Some(*vals.iter().max().unwrap() as f64),
                    _ => None,
                };
                out[part[pos]] = v;
            }
        }
        out
    }

    /// Independent GROUPS-frame oracle: a row is in `pos`'s frame iff its peer-group
    /// index lies in `[g+start_off, g+end_off]` (clamped to `[0, G-1]`), computed by
    /// group membership rather than the position-range formula the kernel uses — so a
    /// bug in `frame_bounds`' RANGE/GROUPS resolution is caught, not mirrored.
    fn naive_groups(
        func: WindowFn,
        part: &[usize],
        group_of: &[usize],
        num_groups: usize,
        raw: &[Option<i64>],
        start: FrameBound,
        end: FrameBound,
    ) -> Vec<Option<f64>> {
        let g_lo = |g: i64| -> i64 {
            match start {
                FrameBound::UnboundedPreceding => i64::MIN,
                FrameBound::Preceding(k) => g - k as i64,
                FrameBound::CurrentRow => g,
                FrameBound::Following(k) => g + k as i64,
                FrameBound::UnboundedFollowing => i64::MAX,
            }
        };
        let g_hi = |g: i64| -> i64 {
            match end {
                FrameBound::UnboundedPreceding => i64::MIN,
                FrameBound::Preceding(k) => g - k as i64,
                FrameBound::CurrentRow => g,
                FrameBound::Following(k) => g + k as i64,
                FrameBound::UnboundedFollowing => i64::MAX,
            }
        };
        let _ = num_groups;
        let mut out = vec![None; raw.len()];
        for (pos, &row) in part.iter().enumerate() {
            let g = group_of[pos] as i64;
            // Unclamped bounds: a group index (always in `[0, G-1]`) qualifies iff it
            // falls in `[g_lo, g_hi]`. A `g_hi < 0` (or `g_lo > G-1`) therefore excludes
            // every group — an empty frame — rather than clamping into a spurious group.
            let (lo, hi) = (g_lo(g), g_hi(g));
            let vals: Vec<i64> = (0..part.len())
                .filter(|&j| {
                    let gj = group_of[j] as i64;
                    lo <= gj && gj <= hi
                })
                .filter_map(|j| raw[part[j]])
                .collect();
            out[row] = match func {
                WindowFn::Count => Some(vals.len() as f64),
                _ if vals.is_empty() => None,
                WindowFn::Sum => Some(vals.iter().sum::<i64>() as f64),
                WindowFn::Avg => Some(vals.iter().sum::<i64>() as f64 / vals.len() as f64),
                WindowFn::Min => Some(*vals.iter().min().unwrap() as f64),
                WindowFn::Max => Some(*vals.iter().max().unwrap() as f64),
                _ => None,
            };
        }
        out
    }

    /// The GROUPS sliding kernel (sum/avg/min/max/count) must match the independent
    /// group-membership oracle across every frame shape — including Following bounds,
    /// nulls, and multi-group partitions the existing single-frame tests never exercise.
    #[test]
    fn groups_frame_matches_naive_oracle() {
        use arrow::row::{RowConverter, SortField};
        // Order key with ties → peer groups {0,1},{2},{3,4,5},{6}. G = 4.
        let key_vals = [10i64, 10, 20, 30, 30, 30, 40];
        let keys: ArrayRef = Arc::new(Int64Array::from(key_vals.to_vec()));
        let conv = RowConverter::new(vec![SortField::new(keys.data_type().clone())]).unwrap();
        let rows = conv.convert_columns(std::slice::from_ref(&keys)).unwrap();
        let raw: Vec<Option<i64>> = vec![Some(1), None, Some(3), Some(4), None, Some(6), Some(7)];
        let values: ArrayRef = Arc::new(Int64Array::from(raw.clone()));
        let part: Vec<usize> = (0..7).collect();
        let ordered = vec![part.clone()];
        let n = raw.len();

        // Group index per position (peers are contiguous once sorted).
        let mut group_of = vec![0usize; n];
        let mut g = 0usize;
        for pos in 1..n {
            if key_vals[pos] != key_vals[pos - 1] {
                g += 1;
            }
            group_of[pos] = g;
        }
        let num_groups = g + 1;

        let bounds = [
            FrameBound::UnboundedPreceding,
            FrameBound::Preceding(2),
            FrameBound::Preceding(1),
            FrameBound::CurrentRow,
            FrameBound::Following(1),
            FrameBound::Following(2),
            FrameBound::UnboundedFollowing,
        ];
        // Offset ordering of a bound, for the valid-frame constraint `start <= end`
        // (a SQL frame requires the start bound to be at or before the end bound).
        let rank = |b: FrameBound| -> i64 {
            match b {
                FrameBound::UnboundedPreceding => i64::MIN,
                FrameBound::Preceding(k) => -(k as i64),
                FrameBound::CurrentRow => 0,
                FrameBound::Following(k) => k as i64,
                FrameBound::UnboundedFollowing => i64::MAX,
            }
        };
        for &start in &bounds {
            for &end in &bounds {
                // A valid SQL frame: start at/before end, end is not UNBOUNDED PRECEDING,
                // start is not UNBOUNDED FOLLOWING.
                if rank(start) > rank(end)
                    || end == FrameBound::UnboundedPreceding
                    || start == FrameBound::UnboundedFollowing
                {
                    continue;
                }
                let frame = Frame {
                    unit: FrameUnit::Groups,
                    start,
                    end,
                };
                for func in [
                    WindowFn::Sum,
                    WindowFn::Avg,
                    WindowFn::Min,
                    WindowFn::Max,
                    WindowFn::Count,
                ] {
                    let got =
                        framed_aggregate(func, &ordered, &values, frame, Some(&rows), n).unwrap();
                    let want = naive_groups(func, &part, &group_of, num_groups, &raw, start, end);
                    assert_eq!(fmt(&got), want, "func={func:?} start={start:?} end={end:?}");
                }
            }
        }
    }

    fn fmt(arr: &ArrayRef) -> Vec<Option<f64>> {
        match arr.data_type() {
            DataType::Int64 => {
                let a = arr.as_primitive::<Int64Type>();
                (0..a.len())
                    .map(|i| a.is_valid(i).then(|| a.value(i) as f64))
                    .collect()
            }
            DataType::Float64 => {
                let a = arr.as_primitive::<Float64Type>();
                (0..a.len())
                    .map(|i| a.is_valid(i).then(|| a.value(i)))
                    .collect()
            }
            _ => unreachable!(),
        }
    }
}
