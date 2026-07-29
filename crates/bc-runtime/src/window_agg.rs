//! The window aggregates beyond `sum`/`avg`/`min`/`max`/`count`.
//!
//! DuckDB, Spark and Polars all let *any* aggregate be a window function; this engine
//! had five. The nine added here are exactly the ones whose running form is **O(1) per
//! row**, which is what makes them safe to compute the same way the existing five are:
//!
//! * the moment aggregates `var` and `stddev`, from a running Welford `(n, mean, M2)` —
//!   the same recurrence `agg/var.rs` keeps, so a window and a `GROUP BY` over the same
//!   rows agree by construction rather than by coincidence;
//! * the folds `product`, `bool_and`, `bool_or`, `bit_and`, `bit_or`, `bit_xor`, each a
//!   running application of an associative operator;
//! * `count_distinct`, from a running hash set.
//!
//! The *population* forms `var_pop`/`stddev_pop` are absent for a different reason than
//! the order statistics: the engine's aggregate vocabulary has no tag for them (the
//! DataFrame spellings are composites over `var`/`stddev`), so a `WindowFn` variant would
//! have nothing able to construct it.
//!
//! Order statistics (`median`, `quantile`, `mode`) are deliberately absent: their running
//! form needs a sorted structure, so adding them here would put an `O(n log n)` — or
//! worse — kernel behind the same call shape as an `O(n)` one, with nothing at the call
//! site to say so. They stay unsupported until they get a structure that earns them.
//!
//! Each family appears twice, and that is not duplication: the **whole-partition** form
//! reduces by dense group id in one linear pass with no ordering at all, while the
//! **running** form accumulates along the ordered partition and shares each peer group's
//! value at its boundary. They compute different relations from the same operator.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{Float64Type, Int64Type};
use arrow::row::Rows;

use crate::error::RuntimeError;
use crate::window::WindowFn;

/// Running **Welford** `(n, mean, M2)` state for the moment aggregates.
///
/// This is the same recurrence `agg/var.rs` keeps, and for the same reason. The obvious
/// state is `(n, Σx, Σx²)` with `variance = (Σx² − n·mean²)/(n−1)`, which subtracts two
/// nearly equal large numbers and cancels catastrophically when the mean dwarfs the
/// spread: over `[1e9+1, 1e9+2, 1e9+3]` it returns exactly `0` where the answer is `1`.
///
/// The first version of this module used sum-of-powers, and its doc comment claimed the
/// `GROUP BY` aggregate did too — it does not, and has not since that exact case was
/// fixed there. So the window and the group aggregate disagreed by a factor of infinity
/// on a column of large near-equal values, while the test fixture (values 1 through 8)
/// could not see it. Welford makes the two agree because it *is* the same recurrence, not
/// because they were checked against each other on small numbers.
#[derive(Default, Clone, Copy)]
struct Moments {
    n: i64,
    mean: f64,
    m2: f64,
}

impl Moments {
    /// One online Welford update: shift the mean by the new value's share of the
    /// deviation, then accumulate the centred product. No sum of squares is ever formed,
    /// so there is nothing to cancel.
    #[inline]
    fn push(&mut self, v: f64) {
        self.n += 1;
        let delta = v - self.mean;
        self.mean += delta / self.n as f64;
        self.m2 += delta * (v - self.mean);
    }

    /// The variance under `ddof` (1 = sample), or `None` when there are too few values
    /// for it to be defined. `ddof` is a parameter rather than a constant because the
    /// population form is the same state with a different divisor, and will want it when
    /// the aggregate vocabulary grows a tag for it.
    #[inline]
    fn variance(&self, ddof: i64) -> Option<f64> {
        if self.n <= ddof {
            return None;
        }
        // M2 is a sum of squares of *deviations*, so it is non-negative by construction;
        // the clamp guards only against a -0.0 sneaking into `sqrt`.
        Some((self.m2.max(0.0)) / (self.n - ddof) as f64)
    }

    #[inline]
    fn finish(&self, func: WindowFn) -> Option<f64> {
        match func {
            WindowFn::Var => self.variance(1),
            WindowFn::Stddev => self.variance(1).map(f64::sqrt),
            _ => None,
        }
    }
}

/// Whether `func` is one of the aggregates this module computes.
pub(crate) fn is_extended_aggregate(func: WindowFn) -> bool {
    matches!(
        func,
        WindowFn::Var
            | WindowFn::Stddev
            | WindowFn::Product
            | WindowFn::BoolAnd
            | WindowFn::BoolOr
            | WindowFn::BitAnd
            | WindowFn::BitOr
            | WindowFn::BitXor
            | WindowFn::CountDistinct
    )
}

/// Whole-partition form: one value per partition, broadcast to each of its rows.
///
/// Takes dense group ids rather than per-partition index lists for the same reason
/// `window_partition_agg` does — one linear pass over the rows, then a gather.
pub(crate) fn broadcast(
    func: WindowFn,
    group_ids: &[u32],
    num_groups: usize,
    values: &ArrayRef,
) -> Result<ArrayRef, RuntimeError> {
    match func {
        WindowFn::Var | WindowFn::Stddev => {
            let f = numeric(values, func)?;
            let mut state = vec![Moments::default(); num_groups];
            for (i, &g) in group_ids.iter().enumerate() {
                if f.is_valid(i) {
                    state[g as usize].push(f.value(i));
                }
            }
            let out: Float64Array = group_ids
                .iter()
                .map(|&g| state[g as usize].finish(func))
                .collect();
            Ok(Arc::new(out))
        }
        WindowFn::Product => {
            let f = numeric(values, func)?;
            // `None` marks a group with no non-null value, which is NULL rather than the
            // empty product 1 — the rule every other aggregate here follows.
            let mut state: Vec<Option<f64>> = vec![None; num_groups];
            for (i, &g) in group_ids.iter().enumerate() {
                if f.is_valid(i) {
                    let slot = &mut state[g as usize];
                    *slot = Some(slot.unwrap_or(1.0) * f.value(i));
                }
            }
            let out: Float64Array = group_ids.iter().map(|&g| state[g as usize]).collect();
            Ok(Arc::new(out))
        }
        WindowFn::BoolAnd | WindowFn::BoolOr => {
            let b = boolean(values, func)?;
            let is_and = func == WindowFn::BoolAnd;
            let mut state: Vec<Option<bool>> = vec![None; num_groups];
            for (i, &g) in group_ids.iter().enumerate() {
                if b.is_valid(i) {
                    let slot = &mut state[g as usize];
                    *slot = Some(match *slot {
                        None => b.value(i),
                        Some(a) if is_and => a && b.value(i),
                        Some(a) => a || b.value(i),
                    });
                }
            }
            let out: BooleanArray = group_ids.iter().map(|&g| state[g as usize]).collect();
            Ok(Arc::new(out))
        }
        WindowFn::BitAnd | WindowFn::BitOr | WindowFn::BitXor => {
            let a = integer(values, func)?;
            let mut state: Vec<Option<i64>> = vec![None; num_groups];
            for (i, &g) in group_ids.iter().enumerate() {
                if a.is_valid(i) {
                    let slot = &mut state[g as usize];
                    *slot = Some(bit_fold(func, *slot, a.value(i)));
                }
            }
            let out: Int64Array = group_ids.iter().map(|&g| state[g as usize]).collect();
            Ok(Arc::new(out))
        }
        WindowFn::CountDistinct => {
            let mut state: Vec<HashSet<Key>> = (0..num_groups).map(|_| HashSet::new()).collect();
            for (i, &g) in group_ids.iter().enumerate() {
                if let Some(k) = key_at(values, i)? {
                    state[g as usize].insert(k);
                }
            }
            let out: Int64Array = group_ids
                .iter()
                .map(|&g| state[g as usize].len() as i64)
                .collect();
            Ok(Arc::new(out))
        }
        other => Err(unsupported(other, values)),
    }
}

/// Running form over the ordered partitions: each row sees the accumulation of every row
/// up to and including its peer group, matching SQL's default `RANGE` frame.
pub(crate) fn running(
    func: WindowFn,
    ordered: &[Vec<usize>],
    order_rows: &Rows,
    values: &ArrayRef,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    match func {
        WindowFn::Var | WindowFn::Stddev => {
            let f = numeric(values, func)?;
            let mut out: Vec<Option<f64>> = vec![None; num_rows];
            for part in ordered {
                let mut state = Moments::default();
                let mut group_start = 0usize;
                for pos in 0..part.len() {
                    if f.is_valid(part[pos]) {
                        state.push(f.value(part[pos]));
                    }
                    if peer_end(part, order_rows, pos) {
                        let v = state.finish(func);
                        for j in group_start..=pos {
                            out[part[j]] = v;
                        }
                        group_start = pos + 1;
                    }
                }
            }
            Ok(Arc::new(Float64Array::from(out)))
        }
        WindowFn::Product => {
            let f = numeric(values, func)?;
            let mut out: Vec<Option<f64>> = vec![None; num_rows];
            for part in ordered {
                let (mut acc, mut group_start): (Option<f64>, usize) = (None, 0);
                for pos in 0..part.len() {
                    if f.is_valid(part[pos]) {
                        acc = Some(acc.unwrap_or(1.0) * f.value(part[pos]));
                    }
                    if peer_end(part, order_rows, pos) {
                        for j in group_start..=pos {
                            out[part[j]] = acc;
                        }
                        group_start = pos + 1;
                    }
                }
            }
            Ok(Arc::new(Float64Array::from(out)))
        }
        WindowFn::BoolAnd | WindowFn::BoolOr => {
            let b = boolean(values, func)?;
            let is_and = func == WindowFn::BoolAnd;
            let mut out: Vec<Option<bool>> = vec![None; num_rows];
            for part in ordered {
                let (mut acc, mut group_start): (Option<bool>, usize) = (None, 0);
                for pos in 0..part.len() {
                    if b.is_valid(part[pos]) {
                        let v = b.value(part[pos]);
                        acc = Some(match acc {
                            None => v,
                            Some(a) if is_and => a && v,
                            Some(a) => a || v,
                        });
                    }
                    if peer_end(part, order_rows, pos) {
                        for j in group_start..=pos {
                            out[part[j]] = acc;
                        }
                        group_start = pos + 1;
                    }
                }
            }
            Ok(Arc::new(BooleanArray::from(out)))
        }
        WindowFn::BitAnd | WindowFn::BitOr | WindowFn::BitXor => {
            let a = integer(values, func)?;
            let mut out: Vec<Option<i64>> = vec![None; num_rows];
            for part in ordered {
                let (mut acc, mut group_start): (Option<i64>, usize) = (None, 0);
                for pos in 0..part.len() {
                    if a.is_valid(part[pos]) {
                        acc = Some(bit_fold(func, acc, a.value(part[pos])));
                    }
                    if peer_end(part, order_rows, pos) {
                        for j in group_start..=pos {
                            out[part[j]] = acc;
                        }
                        group_start = pos + 1;
                    }
                }
            }
            Ok(Arc::new(Int64Array::from(out)))
        }
        WindowFn::CountDistinct => {
            let mut out = vec![0i64; num_rows];
            for part in ordered {
                let mut seen: HashSet<Key> = HashSet::new();
                let mut group_start = 0usize;
                for pos in 0..part.len() {
                    if let Some(k) = key_at(values, part[pos])? {
                        seen.insert(k);
                    }
                    if peer_end(part, order_rows, pos) {
                        let n = seen.len() as i64;
                        for j in group_start..=pos {
                            out[part[j]] = n;
                        }
                        group_start = pos + 1;
                    }
                }
            }
            Ok(Arc::new(Int64Array::from(out)))
        }
        other => Err(unsupported(other, values)),
    }
}

/// One step of the bitwise fold.
#[inline]
fn bit_fold(func: WindowFn, acc: Option<i64>, v: i64) -> i64 {
    match (func, acc) {
        (_, None) => v,
        (WindowFn::BitAnd, Some(a)) => a & v,
        (WindowFn::BitOr, Some(a)) => a | v,
        (WindowFn::BitXor, Some(a)) => a ^ v,
        (_, Some(a)) => a,
    }
}

/// The distinct-count key. Floats are keyed by their bit pattern *after* canonicalizing
/// `-0.0` and every NaN payload, so `count_distinct` agrees with the grouping key
/// identity in `keys.rs` rather than inventing a second one.
#[derive(Clone, PartialEq, Eq, Hash)]
enum Key {
    Int(i64),
    Float(u64),
    Bool(bool),
    Str(String),
}

fn key_at(values: &ArrayRef, i: usize) -> Result<Option<Key>, RuntimeError> {
    use arrow::datatypes::DataType;
    if !values.is_valid(i) {
        return Ok(None);
    }
    Ok(Some(match values.data_type() {
        DataType::Int64 => Key::Int(values.as_primitive::<Int64Type>().value(i)),
        DataType::Float64 => {
            let v = values.as_primitive::<Float64Type>().value(i);
            Key::Float(bc_arrow::float_ident::canon_f64_bits(v))
        }
        DataType::Boolean => Key::Bool(values.as_boolean().value(i)),
        DataType::Utf8 => Key::Str(
            values
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("utf8")
                .value(i)
                .to_string(),
        ),
        other => {
            return Err(RuntimeError::UnsupportedWindow {
                func: "count_distinct".to_string(),
                dtype: other.to_string(),
            })
        }
    }))
}

/// True when `pos` is the last row of its peer group (the next row differs on the order
/// keys, or it is the partition's last row).
#[inline]
fn peer_end(part: &[usize], order_rows: &Rows, pos: usize) -> bool {
    pos + 1 == part.len() || order_rows.row(part[pos]) != order_rows.row(part[pos + 1])
}

fn numeric(values: &ArrayRef, func: WindowFn) -> Result<Float64Array, RuntimeError> {
    use arrow::datatypes::DataType;
    let f =
        arrow::compute::cast(values, &DataType::Float64).map_err(|_| unsupported(func, values))?;
    Ok(f.as_primitive::<Float64Type>().clone())
}

fn integer(values: &ArrayRef, func: WindowFn) -> Result<Int64Array, RuntimeError> {
    use arrow::datatypes::DataType;
    let a =
        arrow::compute::cast(values, &DataType::Int64).map_err(|_| unsupported(func, values))?;
    Ok(a.as_primitive::<Int64Type>().clone())
}

fn boolean(values: &ArrayRef, func: WindowFn) -> Result<BooleanArray, RuntimeError> {
    use arrow::datatypes::DataType;
    if values.data_type() != &DataType::Boolean {
        return Err(unsupported(func, values));
    }
    Ok(values.as_boolean().clone())
}

fn unsupported(func: WindowFn, values: &ArrayRef) -> RuntimeError {
    RuntimeError::UnsupportedWindow {
        func: func.name().to_string(),
        dtype: values.data_type().to_string(),
    }
}

// --- explicit frames, via one generic two-stack slide ------------------------

/// A sliding window over an **associative, commutative** fold, kept as two cumulative
/// stacks — the "queue from two stacks" trick.
///
/// This is the generalization of `window_frame::FifoSum`, which is the same structure
/// specialized to `+`. Nothing about it was ever specific to addition: it exists because
/// the naive O(1) slide (apply the entering value, *un-apply* the leaving one) needs an
/// **inverse**, and most folds do not have one. `product` cannot divide back out a zero;
/// `bit_and` and `bool_and` cannot un-AND at all. The two-stack form never un-applies —
/// the reported value is always the fold of exactly the elements currently in the window
/// — so it serves every fold at O(1) amortized cost, which is why generalizing it gives
/// all six a framed form at once instead of six hand-written kernels.
///
/// Commutativity is required because `value()` combines the two stacks' accumulators
/// without regard to which side is older. Every fold routed here (`*`, `&`, `|`, `^`,
/// `AND`, `OR`) is commutative; a non-commutative one would need the halves ordered.
struct SlidingFold<T, F> {
    /// `(value, fold of this entry and everything below it)` — the push side.
    back: Vec<(T, T)>,
    /// The pop side; filled by draining `back` in reverse when it empties.
    front: Vec<(T, T)>,
    combine: F,
}

impl<T: Clone, F: Fn(&T, &T) -> T> SlidingFold<T, F> {
    fn new(combine: F) -> Self {
        Self {
            back: Vec::new(),
            front: Vec::new(),
            combine,
        }
    }

    fn push(&mut self, v: T) {
        let acc = match self.back.last() {
            Some((_, a)) => (self.combine)(a, &v),
            None => v.clone(),
        };
        self.back.push((v, acc));
    }

    fn pop(&mut self) {
        if self.front.is_empty() {
            // Reverse `back` into `front`, rebuilding accumulators bottom-up so the
            // oldest entry ends on top of `front` and is popped first (FIFO order).
            while let Some((v, _)) = self.back.pop() {
                let acc = match self.front.last() {
                    Some((_, a)) => (self.combine)(a, &v),
                    None => v.clone(),
                };
                self.front.push((v, acc));
            }
        }
        self.front.pop();
    }

    /// The fold of every value currently in the window, or `None` when it is empty.
    fn value(&self) -> Option<T> {
        match (self.back.last(), self.front.last()) {
            (Some((_, a)), Some((_, b))) => Some((self.combine)(a, b)),
            (Some((_, a)), None) => Some(a.clone()),
            (None, Some((_, b))) => Some(b.clone()),
            (None, None) => None,
        }
    }
}

/// Explicit-frame form of the folds: slide a [`SlidingFold`] across each partition,
/// pushing entering rows and popping leaving ones.
///
/// Null rows are simply not pushed, so an all-null frame reports `None` — the same rule
/// the running and whole-partition forms follow, and the reason the fold's identity
/// element is never needed (an empty frame is null, not `1`/`true`/`0`).
pub(crate) fn framed(
    func: WindowFn,
    ordered: &[Vec<usize>],
    values: &ArrayRef,
    frame: crate::window_frame::Frame,
    order_rows: Option<&Rows>,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    match func {
        WindowFn::Product => {
            let f = numeric(values, func)?;
            let out = slide(
                ordered,
                frame,
                order_rows,
                num_rows,
                |row| f.is_valid(row).then(|| f.value(row)),
                |a, b| a * b,
            );
            Ok(Arc::new(Float64Array::from(out)))
        }
        WindowFn::BoolAnd | WindowFn::BoolOr => {
            let b = boolean(values, func)?;
            let is_and = func == WindowFn::BoolAnd;
            let out = slide(
                ordered,
                frame,
                order_rows,
                num_rows,
                |row| b.is_valid(row).then(|| b.value(row)),
                move |x, y| if is_and { *x && *y } else { *x || *y },
            );
            Ok(Arc::new(BooleanArray::from(out)))
        }
        WindowFn::BitAnd | WindowFn::BitOr | WindowFn::BitXor => {
            let a = integer(values, func)?;
            let out = slide(
                ordered,
                frame,
                order_rows,
                num_rows,
                |row| a.is_valid(row).then(|| a.value(row)),
                move |x, y| bit_fold(func, Some(*x), *y),
            );
            Ok(Arc::new(Int64Array::from(out)))
        }
        // `var`/`stddev` keep a Welford state, whose combine is Chan's parallel formula
        // rather than a plain operator, and `count_distinct` needs a multiset rather
        // than a fold. Neither is served by this slide, so both keep refusing a frame
        // rather than being given a wrong one.
        other => Err(RuntimeError::UnsupportedWindow {
            func: other.name().to_string(),
            dtype: "explicit frame".to_string(),
        }),
    }
}

/// The shared slide: walk each partition's frame bounds once, maintaining the fold.
///
/// Both frame edges are non-decreasing in the row position, which is what lets the window
/// be a FIFO and the whole pass be O(n) amortized rather than O(n · frame width).
fn slide<T: Clone, G: Fn(usize) -> Option<T>, F: Fn(&T, &T) -> T + Copy>(
    ordered: &[Vec<usize>],
    frame: crate::window_frame::Frame,
    order_rows: Option<&Rows>,
    num_rows: usize,
    get: G,
    combine: F,
) -> Vec<Option<T>> {
    let mut out: Vec<Option<T>> = vec![None; num_rows];
    for part in ordered {
        let len = part.len();
        let peers = crate::window_frame::peer_groups(frame, part, order_rows);
        let (mut cur_a, mut cur_b) = (0usize, 0usize);
        let mut fold = SlidingFold::new(combine);
        // One FIFO entry per *physical* position, so the queue length tracks
        // `cur_b - cur_a` exactly and pops stay aligned with the sliding window. A null
        // row occupies a slot holding `None`, which the fold skips.
        let mut slots: Vec<bool> = Vec::new();
        for pos in 0..len {
            let (a, b) = crate::window_frame::frame_bounds(frame, pos, len, peers.as_ref());
            while cur_b < b {
                match get(part[cur_b]) {
                    Some(v) => {
                        fold.push(v);
                        slots.push(true);
                    }
                    None => slots.push(false),
                }
                cur_b += 1;
            }
            while cur_a < a {
                if cur_a < cur_b && slots[cur_a] {
                    fold.pop();
                }
                cur_a += 1;
            }
            cur_b = cur_b.max(cur_a);
            out[part[pos]] = fold.value();
        }
    }
    out
}
