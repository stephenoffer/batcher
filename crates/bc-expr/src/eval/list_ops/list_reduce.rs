//! Per-row numeric transforms over a `List` row for `eval/list.rs` (`normalize`, `softmax`,
//! `log_softmax`, `arg_sort`, `cum_sum`, `diff`, `entropy`). Most map a `List`/tensor row to a
//! new list of the same length; `entropy` reduces the row to one number. All cast the child to
//! `Float64` first. Split out of `list.rs` to keep that file inside its size budget; the null
//! contract (null row → null; null element preserved) is uniform.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, Float64Builder, GenericListArray, Int64Builder, ListBuilder,
};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type};

use crate::ExprError;

/// The `Float64` view of the list's child plus its offsets — the shared setup for every op.
fn f64_child(list: &GenericListArray<i32>) -> Result<(ArrayRef, Vec<usize>), ExprError> {
    let child = cast(list.values(), &DataType::Float64)?;
    let offsets = list.value_offsets().iter().map(|&o| o as usize).collect();
    Ok((child, offsets))
}

/// L2-normalize each row to unit length: `xᵢ / sqrt(Σ xⱼ²)`. Zero vector → zeros.
pub(crate) fn normalize(list: &GenericListArray<i32>) -> Result<ArrayRef, ExprError> {
    let (child, off) = f64_child(list)?;
    let f = child.as_primitive::<Float64Type>();
    let mut b = ListBuilder::new(Float64Builder::new());
    for i in 0..list.len() {
        if list.is_null(i) {
            b.append_null();
            continue;
        }
        let (s, e) = (off[i], off[i + 1]);
        let norm = (s..e)
            .filter(|&k| f.is_valid(k))
            .map(|k| f.value(k) * f.value(k))
            .sum::<f64>()
            .sqrt();
        let vb = b.values();
        for k in s..e {
            if f.is_valid(k) {
                vb.append_value(if norm > 0.0 { f.value(k) / norm } else { 0.0 });
            } else {
                vb.append_null();
            }
        }
        b.append(true);
    }
    Ok(Arc::new(b.finish()))
}

/// Numerically-stable softmax per row: `exp(xᵢ − max) / Σ exp(xⱼ − max)`.
pub(crate) fn softmax(list: &GenericListArray<i32>) -> Result<ArrayRef, ExprError> {
    let (child, off) = f64_child(list)?;
    let f = child.as_primitive::<Float64Type>();
    let mut b = ListBuilder::new(Float64Builder::new());
    for i in 0..list.len() {
        if list.is_null(i) {
            b.append_null();
            continue;
        }
        let (s, e) = (off[i], off[i + 1]);
        let max = (s..e)
            .filter(|&k| f.is_valid(k))
            .map(|k| f.value(k))
            .fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = (s..e)
            .filter(|&k| f.is_valid(k))
            .map(|k| (f.value(k) - max).exp())
            .sum();
        let vb = b.values();
        for k in s..e {
            if f.is_valid(k) {
                vb.append_value(if sum > 0.0 {
                    (f.value(k) - max).exp() / sum
                } else {
                    0.0
                });
            } else {
                vb.append_null();
            }
        }
        b.append(true);
    }
    Ok(Arc::new(b.finish()))
}

/// The 0-based indices that sort each row ascending (stable; nulls placed last).
pub(crate) fn arg_sort(list: &GenericListArray<i32>) -> Result<ArrayRef, ExprError> {
    let (child, off) = f64_child(list)?;
    let f = child.as_primitive::<Float64Type>();
    let mut b = ListBuilder::new(Int64Builder::new());
    for i in 0..list.len() {
        if list.is_null(i) {
            b.append_null();
            continue;
        }
        let (s, e) = (off[i], off[i + 1]);
        let mut valid: Vec<usize> = (s..e).filter(|&k| f.is_valid(k)).collect();
        let nulls: Vec<usize> = (s..e).filter(|&k| !f.is_valid(k)).collect();
        valid.sort_by(|&a, &c| f.value(a).total_cmp(&f.value(c)));
        let vb = b.values();
        for k in valid.into_iter().chain(nulls) {
            vb.append_value((k - s) as i64);
        }
        b.append(true);
    }
    Ok(Arc::new(b.finish()))
}

/// Cumulative sum per row (element `i` = `Σ_{j≤i} xⱼ`); a null element stays null and the
/// running total continues.
pub(crate) fn cum_sum(list: &GenericListArray<i32>) -> Result<ArrayRef, ExprError> {
    let (child, off) = f64_child(list)?;
    let f = child.as_primitive::<Float64Type>();
    let mut b = ListBuilder::new(Float64Builder::new());
    for i in 0..list.len() {
        if list.is_null(i) {
            b.append_null();
            continue;
        }
        let (s, e) = (off[i], off[i + 1]);
        let mut running = 0f64;
        let vb = b.values();
        for k in s..e {
            if f.is_valid(k) {
                running += f.value(k);
                vb.append_value(running);
            } else {
                vb.append_null();
            }
        }
        b.append(true);
    }
    Ok(Arc::new(b.finish()))
}

/// First difference per row: element `i` is `xᵢ − xᵢ₋₁`, element 0 null. A null at
/// either neighbor makes that difference null (Polars `list.diff`). Same length out.
pub(crate) fn diff(list: &GenericListArray<i32>) -> Result<ArrayRef, ExprError> {
    let (child, off) = f64_child(list)?;
    let f = child.as_primitive::<Float64Type>();
    let mut b = ListBuilder::new(Float64Builder::new());
    for i in 0..list.len() {
        if list.is_null(i) {
            b.append_null();
            continue;
        }
        let (s, e) = (off[i], off[i + 1]);
        let vb = b.values();
        for k in s..e {
            if k == s || !f.is_valid(k) || !f.is_valid(k - 1) {
                vb.append_null();
            } else {
                vb.append_value(f.value(k) - f.value(k - 1));
            }
        }
        b.append(true);
    }
    Ok(Arc::new(b.finish()))
}

/// `log_softmax` — `xᵢ − max − ln Σ exp(xⱼ − max)`, the log-domain sibling of `softmax`.
///
/// Computed in the log domain rather than as `softmax(...).ln()`: a probability that
/// underflows to 0 in the linear form gives `-inf` there, while here it stays a large
/// negative finite number, which is the whole reason a training or scoring pipeline reads
/// log-probabilities instead of probabilities.
pub(crate) fn log_softmax(list: &GenericListArray<i32>) -> Result<ArrayRef, ExprError> {
    let (child, off) = f64_child(list)?;
    let f = child.as_primitive::<Float64Type>();
    let mut b = ListBuilder::new(Float64Builder::new());
    for i in 0..list.len() {
        if list.is_null(i) {
            b.append_null();
            continue;
        }
        let (s, e) = (off[i], off[i + 1]);
        let max = (s..e)
            .filter(|&k| f.is_valid(k))
            .map(|k| f.value(k))
            .fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = (s..e)
            .filter(|&k| f.is_valid(k))
            .map(|k| (f.value(k) - max).exp())
            .sum();
        let log_sum = sum.ln();
        let vb = b.values();
        for k in s..e {
            if f.is_valid(k) {
                vb.append_value(f.value(k) - max - log_sum);
            } else {
                vb.append_null();
            }
        }
        b.append(true);
    }
    Ok(Arc::new(b.finish()))
}

/// Shannon entropy in **nats** of each row read as a distribution: `−Σ pᵢ ln pᵢ`.
///
/// The row is normalized by its own sum first, so it works on a probability vector, a raw
/// count vector, or an unnormalized weight vector alike. A zero or negative total has no
/// distribution to measure and yields null; a non-positive element is skipped, since
/// `p ln p` is undefined there and treating it as zero is the convention.
///
/// This is the per-row uncertainty of a classifier's output or an attention distribution:
/// 0 when all the mass is on one outcome, `ln n` when it is spread evenly over `n`.
pub(crate) fn entropy(list: &GenericListArray<i32>) -> Result<ArrayRef, ExprError> {
    let (child, off) = f64_child(list)?;
    let f = child.as_primitive::<Float64Type>();
    let mut b = Float64Builder::with_capacity(list.len());
    for i in 0..list.len() {
        if list.is_null(i) {
            b.append_null();
            continue;
        }
        let (s, e) = (off[i], off[i + 1]);
        let total: f64 = (s..e)
            .filter(|&k| f.is_valid(k) && f.value(k) > 0.0)
            .map(|k| f.value(k))
            .sum();
        // NaN totals fall here too: a row that cannot be normalized has no distribution.
        if total.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            b.append_null();
            continue;
        }
        let h: f64 = (s..e)
            .filter(|&k| f.is_valid(k) && f.value(k) > 0.0)
            .map(|k| {
                let p = f.value(k) / total;
                -p * p.ln()
            })
            .sum();
        // `-p ln p` is `-0.0` when p is 1, so a perfectly certain row would otherwise report
        // `-0.0`. Adding zero folds it to `0.0`, which is what a reader expects to see and
        // what `== 0.0` comparisons downstream already assume.
        b.append_value(h + 0.0);
    }
    Ok(Arc::new(b.finish()))
}
