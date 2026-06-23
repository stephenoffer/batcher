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
//! clamped), so the frame only ever slides right. The kernel exploits this to run
//! in **one pass**: `sum`/`avg`/`count` keep a running accumulator (add the entering
//! row, subtract the leaving one — O(n)); `min`/`max` keep a monotonic deque
//! (O(n) amortized). No frame is rescanned.

use std::collections::VecDeque;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Float64Array, Int64Array};
use arrow::datatypes::{DataType, Float64Type, Int64Type};

use crate::error::RuntimeError;
use crate::window::WindowFn;

/// One edge of a `ROWS` frame (mirror of `bc_ir::FrameBound`). The offsets are
/// non-negative row counts relative to the current row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameBound {
    UnboundedPreceding,
    Preceding(u64),
    CurrentRow,
    Following(u64),
    UnboundedFollowing,
}

/// An explicit `ROWS` frame: the inclusive `[start, end]` row range each output
/// row aggregates over.
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    pub start: FrameBound,
    pub end: FrameBound,
}

/// Resolve a frame to the half-open `[a, b)` row range within an ordered partition
/// of length `len` for the row at `pos`. Both `a` and `b` are non-decreasing in
/// `pos` (each is `pos + const`, clamped to `[0, len]`), which is what lets the
/// aggregate slide in one pass. An empty frame yields `a >= b`.
fn frame_half_open(frame: Frame, pos: usize, len: usize) -> (usize, usize) {
    let (pos, n) = (pos as i64, len as i64);
    let lo = match frame.start {
        FrameBound::UnboundedPreceding => 0,
        FrameBound::Preceding(k) => pos - k as i64,
        FrameBound::CurrentRow => pos,
        FrameBound::Following(k) => pos + k as i64,
        FrameBound::UnboundedFollowing => n, // start past the last row → empty
    };
    let hi_excl = match frame.end {
        FrameBound::UnboundedPreceding => 0, // end before the first row → empty
        FrameBound::Preceding(k) => pos - k as i64 + 1,
        FrameBound::CurrentRow => pos + 1,
        FrameBound::Following(k) => pos + k as i64 + 1,
        FrameBound::UnboundedFollowing => n,
    };
    (lo.clamp(0, n) as usize, hi_excl.clamp(0, n) as usize)
}

/// Compute an explicit-`ROWS`-frame aggregate, scattered to original row order.
pub fn framed_aggregate(
    func: WindowFn,
    ordered: &[Vec<usize>],
    values: &ArrayRef,
    frame: Frame,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    match func {
        WindowFn::Count => Ok(framed_count(ordered, values, frame, num_rows)),
        WindowFn::Sum | WindowFn::Avg | WindowFn::Min | WindowFn::Max => match values.data_type() {
            DataType::Int64 => Ok(framed_i64(func, ordered, values, frame, num_rows)),
            DataType::Float64 => Ok(framed_f64(func, ordered, values, frame, num_rows)),
            other => Err(RuntimeError::UnsupportedWindow {
                func: func.name().to_string(),
                dtype: other.to_string(),
            }),
        },
        other => Err(RuntimeError::UnsupportedWindow {
            func: other.name().to_string(),
            dtype: "explicit frame".to_string(),
        }),
    }
}

/// `count` over the frame: number of non-null values (0 for an empty frame),
/// slid in one pass — add the entering row, subtract the leaving one.
fn framed_count(
    ordered: &[Vec<usize>],
    values: &ArrayRef,
    frame: Frame,
    num_rows: usize,
) -> ArrayRef {
    let mut out = vec![0i64; num_rows];
    for part in ordered {
        let len = part.len();
        let (mut cur_a, mut cur_b, mut cnt) = (0usize, 0usize, 0i64);
        for pos in 0..len {
            let (a, b) = frame_half_open(frame, pos, len);
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
    num_rows: usize,
) -> ArrayRef {
    let arr = values.as_primitive::<Int64Type>();
    let mut out_i = vec![None::<i64>; num_rows];
    let mut out_f = vec![None::<f64>; num_rows];
    let is_min = func == WindowFn::Min;
    let need_extreme = matches!(func, WindowFn::Min | WindowFn::Max);
    for part in ordered {
        let len = part.len();
        let (mut cur_a, mut cur_b) = (0usize, 0usize);
        let (mut sum, mut cnt) = (0i64, 0i64);
        // Monotonic deque of partition positions holding the running min/max front.
        let mut dq: VecDeque<usize> = VecDeque::new();
        for pos in 0..len {
            let (a, b) = frame_half_open(frame, pos, len);
            while cur_b < b {
                let row = part[cur_b];
                if arr.is_valid(row) {
                    let v = arr.value(row);
                    sum += v;
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
            while cur_a < a {
                if cur_a < cur_b && arr.is_valid(part[cur_a]) {
                    sum -= arr.value(part[cur_a]);
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
    if func == WindowFn::Avg {
        Arc::new(Float64Array::from(out_f))
    } else {
        Arc::new(Int64Array::from(out_i))
    }
}

/// Float-input frame aggregate (`sum`/`avg`/`min`/`max`, all `Float64`).
fn framed_f64(
    func: WindowFn,
    ordered: &[Vec<usize>],
    values: &ArrayRef,
    frame: Frame,
    num_rows: usize,
) -> ArrayRef {
    let arr = values.as_primitive::<Float64Type>();
    let mut out = vec![None::<f64>; num_rows];
    let is_min = func == WindowFn::Min;
    let need_extreme = matches!(func, WindowFn::Min | WindowFn::Max);
    for part in ordered {
        let len = part.len();
        let (mut cur_a, mut cur_b) = (0usize, 0usize);
        let (mut sum, mut cnt) = (0f64, 0i64);
        let mut dq: VecDeque<usize> = VecDeque::new();
        for pos in 0..len {
            let (a, b) = frame_half_open(frame, pos, len);
            while cur_b < b {
                let row = part[cur_b];
                if arr.is_valid(row) {
                    let v = arr.value(row);
                    sum += v;
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
            while cur_a < a {
                if cur_a < cur_b && arr.is_valid(part[cur_a]) {
                    sum -= arr.value(part[cur_a]);
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
            if cnt == 0 {
                continue;
            }
            out[part[pos]] = Some(match func {
                WindowFn::Sum => sum,
                WindowFn::Avg => sum / cnt as f64,
                WindowFn::Min | WindowFn::Max => dq.front().map_or(sum, |&f| arr.value(part[f])),
                _ => unreachable!("framed_f64 on non-aggregate"),
            });
        }
    }
    Arc::new(Float64Array::from(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_half_open_clamps_and_empties() {
        let f = Frame {
            start: FrameBound::Preceding(1),
            end: FrameBound::Following(1),
        };
        assert_eq!(frame_half_open(f, 0, 5), (0, 2)); // clamped at left edge
        assert_eq!(frame_half_open(f, 4, 5), (3, 5)); // clamped at right edge
        assert_eq!(frame_half_open(f, 2, 5), (1, 4));

        // 1 FOLLOWING .. 2 FOLLOWING at the last row → empty (a >= b).
        let ff = Frame {
            start: FrameBound::Following(1),
            end: FrameBound::Following(2),
        };
        let (a, b) = frame_half_open(ff, 4, 5);
        assert!(a >= b);
        let (a0, b0) = frame_half_open(f, 0, 0); // empty partition
        assert!(a0 >= b0);
    }

    #[test]
    fn rows_frame_sliding_sum_and_avg() {
        // One partition [0,1,2,3,4] over values [10,20,30,40,50], ROWS 1 PRECEDING
        // .. CURRENT ROW → trailing pair sums: 10,30,50,70,90.
        let values: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50]));
        let ordered = vec![vec![0usize, 1, 2, 3, 4]];
        let frame = Frame {
            start: FrameBound::Preceding(1),
            end: FrameBound::CurrentRow,
        };
        let s = framed_aggregate(WindowFn::Sum, &ordered, &values, frame, 5).unwrap();
        let s = s.as_primitive::<Int64Type>();
        assert_eq!(s.values(), &[10, 30, 50, 70, 90]);

        // avg over the same trailing pair: 10, 15, 25, 35, 45.
        let a = framed_aggregate(WindowFn::Avg, &ordered, &values, frame, 5).unwrap();
        let a = a.as_primitive::<Float64Type>();
        assert_eq!(a.values(), &[10.0, 15.0, 25.0, 35.0, 45.0]);
    }

    #[test]
    fn rows_frame_centered_min_max_count() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![5, 1, 9, 3, 7]));
        let ordered = vec![vec![0usize, 1, 2, 3, 4]];
        let frame = Frame {
            start: FrameBound::Preceding(1),
            end: FrameBound::Following(1),
        };
        let mn = framed_aggregate(WindowFn::Min, &ordered, &values, frame, 5).unwrap();
        assert_eq!(mn.as_primitive::<Int64Type>().values(), &[1, 1, 1, 3, 3]);
        let mx = framed_aggregate(WindowFn::Max, &ordered, &values, frame, 5).unwrap();
        assert_eq!(mx.as_primitive::<Int64Type>().values(), &[5, 9, 9, 9, 7]);
        let c = framed_aggregate(WindowFn::Count, &ordered, &values, frame, 5).unwrap();
        assert_eq!(c.as_primitive::<Int64Type>().values(), &[2, 3, 3, 3, 2]);
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
                let frame = Frame { start, end };
                for func in [
                    WindowFn::Sum,
                    WindowFn::Avg,
                    WindowFn::Min,
                    WindowFn::Max,
                    WindowFn::Count,
                ] {
                    let got = framed_aggregate(func, &ordered, &values, frame, n).unwrap();
                    let want = naive(func, &ordered, &raw, frame, n);
                    assert_eq!(fmt(&got), want, "func={func:?} start={start:?} end={end:?}");
                }
            }
        }
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
