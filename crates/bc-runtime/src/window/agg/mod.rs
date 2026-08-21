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
//! `median` is the one order statistic admitted, and only because it arrives with the
//! structure the rest are still waiting for. Whole-partition it is a single quickselect —
//! literally `agg/median.rs`'s kernel, so a window and a `GROUP BY` median over the same
//! rows agree by construction. Along an ordered partition it is a **two-heap**: a max-heap
//! of the lower half and a min-heap of the upper, rebalanced after each insert, which is
//! `O(log n)` per row and `O(1)` to read. That is the same order as the folds above, not a
//! sort hidden behind their call shape.
//!
//! What that structure cannot do is *delete*, so an **explicit frame** still declines: a
//! sliding median would have to rebuild, and `O(n·k)` behind an `O(n)` call shape is
//! exactly what this module refuses. `quantile` and `mode` stay out for the original
//! reason — no structure yet earns them.
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
use crate::window::agg::median_state::RunningMedian;
use crate::window::frame::RangeOrder;
use crate::window::WindowFn;

mod median_state;

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

    /// The state for a single value — the identity for [`Moments::merge`]'s unit.
    #[inline]
    fn of(v: f64) -> Self {
        Self {
            n: 1,
            mean: v,
            m2: 0.0,
        }
    }

    /// Two states combined, by Chan's parallel formula — the *mergeable* counterpart of
    /// [`Moments::push`], and the same one `bc-runtime`'s distributed variance uses.
    ///
    /// It is associative and commutative in exact arithmetic, which is what lets a
    /// sliding frame keep the state in the two-stack FIFO ([`SlidingFold`]) instead of
    /// re-accumulating each frame from scratch: the window's state is the combine of the
    /// two stacks' folds, so nothing is ever *subtracted* — the operation Welford has no
    /// inverse for, and the reason this pair used to refuse a frame outright.
    #[inline]
    fn merge(a: &Self, b: &Self) -> Self {
        if a.n == 0 {
            return *b;
        }
        if b.n == 0 {
            return *a;
        }
        let (na, nb) = (a.n as f64, b.n as f64);
        let n = na + nb;
        let delta = b.mean - a.mean;
        Self {
            n: a.n + b.n,
            mean: a.mean + delta * nb / n,
            m2: a.m2 + b.m2 + delta * delta * na * nb / n,
        }
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
            | WindowFn::Median
    )
}

/// Whole-partition form: one value per partition, broadcast to each of its rows.
///
/// Takes dense group ids rather than per-partition index lists for the same reason
/// `partition_agg` does — one linear pass over the rows, then a gather.
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
            let reader = KeyReader::new(values)?;
            let mut state: Vec<HashSet<Key>> = (0..num_groups).map(|_| HashSet::new()).collect();
            for (i, &g) in group_ids.iter().enumerate() {
                if let Some(k) = reader.key(values, i)? {
                    state[g as usize].insert(k);
                }
            }
            let out: Int64Array = group_ids
                .iter()
                .map(|&g| state[g as usize].len() as i64)
                .collect();
            Ok(Arc::new(out))
        }
        WindowFn::Median => {
            let f = numeric(values, func)?;
            // One value list per partition, then the *same* quickselect `GROUP BY` median
            // runs — sharing the kernel is what keeps the two spellings from drifting.
            let mut groups: Vec<Vec<f64>> = vec![Vec::new(); num_groups];
            for (i, &g) in group_ids.iter().enumerate() {
                if f.is_valid(i) {
                    groups[g as usize].push(f.value(i));
                }
            }
            let per_group: Vec<Option<f64>> = groups
                .iter_mut()
                .map(|v| (!v.is_empty()).then(|| crate::agg::median::quickselect_median(v)))
                .collect();
            let out: Float64Array = group_ids.iter().map(|&g| per_group[g as usize]).collect();
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
        WindowFn::Median => {
            let f = numeric(values, func)?;
            let mut out: Vec<Option<f64>> = vec![None; num_rows];
            for part in ordered {
                let mut state = RunningMedian::default();
                let mut group_start = 0usize;
                for pos in 0..part.len() {
                    if f.is_valid(part[pos]) {
                        state.push(f.value(part[pos]));
                    }
                    if peer_end(part, order_rows, pos) {
                        let v = state.median();
                        for j in group_start..=pos {
                            out[part[j]] = v;
                        }
                        group_start = pos + 1;
                    }
                }
            }
            Ok(Arc::new(Float64Array::from(out)))
        }
        WindowFn::CountDistinct => {
            let reader = KeyReader::new(values)?;
            let mut out = vec![0i64; num_rows];
            for part in ordered {
                let mut seen: HashSet<Key> = HashSet::new();
                let mut group_start = 0usize;
                for pos in 0..part.len() {
                    if let Some(k) = reader.key(values, part[pos])? {
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
    /// A row-encoded value, for every type the four cases above do not name.
    Encoded(Box<[u8]>),
}

/// The distinct-key reader for one value column.
///
/// `COUNT(DISTINCT x)` only needs to tell two values apart, so it is the one aggregate here
/// with no reason to care about `x`'s type — yet it was the one that rejected the most types.
/// `COUNT(DISTINCT order_date) OVER (…)` failed on every temporal and decimal column while
/// `COUNT(DISTINCT order_date) … GROUP BY` answered, because [`key_at`] could only name
/// `Int64`/`Float64`/`Boolean`/`Utf8`.
///
/// Anything else is encoded once per column with Arrow's `RowConverter` — the same mechanism
/// [`crate::window::coerce::select_extreme`] uses for `MIN`/`MAX`, so "the same value" means
/// here what it means there and in the group assigner. The four named types keep their
/// direct keys: they are the common case, and encoding them would allocate per row where
/// they currently do not.
enum KeyReader {
    /// One of the four types [`key_at`] reads directly.
    Direct,
    /// Row-encoded bytes, built once for the whole column.
    Encoded(arrow::row::Rows),
    /// A `Null`-typed column: every row is null, so no key is ever produced. `RowConverter`
    /// has no sort field for `Null`, and building one would be pointless work regardless.
    AllNull,
}

impl KeyReader {
    fn new(values: &ArrayRef) -> Result<Self, RuntimeError> {
        use arrow::datatypes::DataType;
        match values.data_type() {
            DataType::Int64 | DataType::Float64 | DataType::Boolean | DataType::Utf8 => {
                Ok(Self::Direct)
            }
            DataType::Null => Ok(Self::AllNull),
            other => {
                let conv =
                    arrow::row::RowConverter::new(vec![arrow::row::SortField::new(other.clone())])?;
                let rows = conv.convert_columns(std::slice::from_ref(values))?;
                Ok(Self::Encoded(rows))
            }
        }
    }

    fn key(&self, values: &ArrayRef, i: usize) -> Result<Option<Key>, RuntimeError> {
        match self {
            Self::Direct => key_at(values, i),
            Self::AllNull => Ok(None),
            Self::Encoded(rows) => {
                if !values.is_valid(i) {
                    return Ok(None);
                }
                Ok(Some(Key::Encoded(rows.row(i).as_ref().into())))
            }
        }
    }
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
/// This is the generalization of `frame::FifoSum`, which is the same structure
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
    frame: crate::window::frame::Frame,
    order_rows: Option<&Rows>,
    range_order: Option<&RangeOrder>,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    match func {
        WindowFn::Product => {
            let f = numeric(values, func)?;
            let out = slide(
                ordered,
                frame,
                order_rows,
                range_order,
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
                range_order,
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
                range_order,
                num_rows,
                |row| a.is_valid(row).then(|| a.value(row)),
                move |x, y| bit_fold(func, Some(*x), *y),
            );
            Ok(Arc::new(Int64Array::from(out)))
        }
        // Welford has no inverse, so a frame cannot be maintained by subtracting the
        // leaving row — which is why this pair used to refuse one. It does not need to:
        // `slide` is a *fold* over any associative combine, and `Moments::merge` is one
        // (Chan's parallel formula). The state is therefore carried in the same two-stack
        // FIFO every other framed aggregate uses, at the same O(n) amortized cost.
        WindowFn::Var | WindowFn::Stddev => {
            let f = numeric(values, func)?;
            let folded = slide(
                ordered,
                frame,
                order_rows,
                range_order,
                num_rows,
                |row| f.is_valid(row).then(|| Moments::of(f.value(row))),
                Moments::merge,
            );
            let out: Vec<Option<f64>> = folded
                .into_iter()
                .map(|m| m.and_then(|m| m.finish(func)))
                .collect();
            Ok(Arc::new(Float64Array::from(out)))
        }
        // `count_distinct` needs a multiset rather than a fold, and `median` needs an order
        // statistic, which `slide`'s associative combine cannot express — merging two
        // sorted halves is associative but costs O(k) a step, so it would be O(n·k) behind
        // an O(n) call shape. Both keep refusing a frame rather than being given a wrong
        // one; `median` still answers the frameless and running forms above.
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
    frame: crate::window::frame::Frame,
    order_rows: Option<&Rows>,
    range_order: Option<&RangeOrder>,
    num_rows: usize,
    get: G,
    combine: F,
) -> Vec<Option<T>> {
    let mut out: Vec<Option<T>> = vec![None; num_rows];
    for part in ordered {
        let len = part.len();
        let ctx = crate::window::frame::frame_ctx(frame, part, order_rows, range_order);
        let (mut cur_a, mut cur_b) = (0usize, 0usize);
        let mut fold = SlidingFold::new(combine);
        // One FIFO entry per *physical* position, so the queue length tracks
        // `cur_b - cur_a` exactly and pops stay aligned with the sliding window. A null
        // row occupies a slot holding `None`, which the fold skips.
        let mut slots: Vec<bool> = Vec::new();
        for pos in 0..len {
            let (a, b) = crate::window::frame::frame_bounds(frame, pos, len, ctx.as_ref());
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Date32Array, Decimal128Array, Int64Array, NullArray};

    /// `COUNT(DISTINCT d) OVER (PARTITION BY k)` over a `Date32` column counts distinct dates.
    ///
    /// The four directly-keyed types never reached the encoder, so this is the case that
    /// used to raise `UnsupportedWindow` while the same aggregate under a `GROUP BY`
    /// answered — see [`KeyReader`].
    #[test]
    fn count_distinct_counts_dates() {
        // Two partitions: rows 0..3 are group 0 (dates 100, 100, 200 -> 2 distinct),
        // rows 3..5 are group 1 (date 300, NULL -> 1 distinct).
        let values: ArrayRef = Arc::new(Date32Array::from(vec![
            Some(100),
            Some(100),
            Some(200),
            Some(300),
            None,
        ]));
        let out = broadcast(WindowFn::CountDistinct, &[0, 0, 0, 1, 1], 2, &values).unwrap();
        let out = out.as_primitive::<Int64Type>();
        assert_eq!(out.values(), &[2, 2, 2, 1, 1]);
    }

    /// The same, for a `Decimal128` column — the other type family that raised.
    #[test]
    fn count_distinct_counts_decimals() {
        let values: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(100), Some(250), Some(100)])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        );
        let out = broadcast(WindowFn::CountDistinct, &[0, 0, 0], 1, &values).unwrap();
        assert_eq!(out.as_primitive::<Int64Type>().values(), &[2, 2, 2]);
    }

    /// An all-`Null` column has no distinct values, and must say `0` rather than raise.
    /// `RowConverter` has no sort field for `Null`, which is why it gets its own arm.
    #[test]
    fn count_distinct_over_an_all_null_column_is_zero() {
        let values: ArrayRef = Arc::new(NullArray::new(3));
        let out = broadcast(WindowFn::CountDistinct, &[0, 0, 1], 2, &values).unwrap();
        assert_eq!(out.as_primitive::<Int64Type>().values(), &[0, 0, 0]);
    }

    /// The directly-keyed types still take the direct path and still agree.
    #[test]
    fn count_distinct_still_counts_int64_directly() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), Some(1), Some(2), None]));
        let out = broadcast(WindowFn::CountDistinct, &[0, 0, 0, 0], 1, &values).unwrap();
        assert_eq!(out.as_primitive::<Int64Type>().values(), &[2, 2, 2, 2]);
    }

    /// The whole-partition form must be the `GROUP BY` answer, per group, with nulls skipped
    /// and an all-null group null rather than zero.
    #[test]
    fn the_broadcast_median_skips_nulls_and_leaves_an_empty_group_null() {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(1.0),
            Some(3.0),
            None,
            Some(8.0),
            None,
        ]));
        let out = broadcast(WindowFn::Median, &[0, 0, 0, 0, 1], 2, &values).unwrap();
        let out = out.as_primitive::<Float64Type>();
        // Group 0 is [1, 3, 8] → 3; group 1 has only a null → null.
        assert_eq!(out.value(0), 3.0);
        assert_eq!(out.value(3), 3.0);
        assert!(out.is_null(4));
    }
}
