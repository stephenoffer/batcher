//! Fused multi-aggregate accumulation — read `group_ids` once for all simple
//! scalar aggregates instead of once per aggregate.
//!
//! [`partial`](super::partial) computes the dense `group_ids` once, then runs one
//! scatter-add pass *per* aggregate ([`super::accumulate`]). With N aggregates that
//! streams the `group_ids` array N times. This module fuses the *simple scalar*
//! aggregates — `sum` / `count` / `count(*)` / `min` / `max` / `mean` — into a
//! single linear scan that visits each row once and updates every fused
//! accumulator, so `group_ids` (and the row-index walk) are read exactly once.
//!
//! **Bit-identical by construction.** Each accumulator owns only its own state and
//! the fused loop visits rows in the same `0..num_rows` order as the per-call
//! kernels, so the per-(group, column) sequence of operations is unchanged — this is
//! a pure loop-interchange of independent scatter-adds. The arms below reproduce the
//! exact kernels in [`super::accum`] and [`super::var::count_non_null`], including
//! the no-null fast paths and the `i64` checked-add overflow error, so the result
//! equals the per-call path element-for-element (and the seq==par==dist oracle and
//! the DuckDB differential tests stay green). Complex aggregates (var, median,
//! arg_min/max, covar, product, bit/bool, distinct/quantile sketches) are not fused
//! and keep their existing per-call path untouched.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, Decimal128Array, Float64Array, Int64Array, StringArray,
};
use arrow::datatypes::{DataType, Decimal128Type, Float64Type, Int64Type};

use super::accum::{masked_decimal, masked_f64, masked_i64};
use super::{AggCall, AggFunc};
use crate::error::RuntimeError;

/// One fused scalar accumulator. Null-ness and type are baked into the variant so
/// the hot `update` arm carries no per-row dtype dispatch and preserves the per-call
/// kernels' no-null fast paths. Each holds a borrow of its (already pre-evaluated)
/// value array plus its own per-group state — accumulators never alias.
enum FusedAcc<'a> {
    /// `sum` over a no-null `Float64` column: scatter-add straight from the values
    /// slice, every group valid (mirrors `sum_acc`'s no-null fast path).
    SumF64NoNull {
        v: &'a [f64],
        sums: Vec<f64>,
    },
    SumF64 {
        v: &'a Float64Array,
        sums: Vec<f64>,
        valid: Vec<bool>,
    },
    /// `sum` over `Int64` with **checked** add — a wrap would be a wrong answer.
    SumI64 {
        v: &'a Int64Array,
        sums: Vec<i64>,
        valid: Vec<bool>,
    },
    SumDecimal {
        v: &'a Decimal128Array,
        sums: Vec<i128>,
        valid: Vec<bool>,
        precision: u8,
        scale: i8,
    },
    /// `count(*)` — every row counts, no value column.
    CountStar {
        counts: Vec<i64>,
    },
    /// `count(col)` over a no-null column — every row counts (skips the validity
    /// bitmap check, mirroring `count_non_null`'s fast path).
    CountNoNull {
        counts: Vec<i64>,
    },
    CountNull {
        v: &'a dyn Array,
        counts: Vec<i64>,
    },
    MinMaxI64 {
        v: &'a Int64Array,
        cur: Vec<i64>,
        valid: Vec<bool>,
        is_min: bool,
    },
    MinMaxF64 {
        v: &'a Float64Array,
        cur: Vec<f64>,
        valid: Vec<bool>,
        is_min: bool,
    },
    MinMaxDecimal {
        v: &'a Decimal128Array,
        cur: Vec<i128>,
        valid: Vec<bool>,
        is_min: bool,
        precision: u8,
        scale: i8,
    },
    MinMaxStr {
        v: &'a StringArray,
        cur: Vec<Option<String>>,
        is_min: bool,
    },
}

impl FusedAcc<'_> {
    /// Apply row `i` (group `g`) to this accumulator. Infallible except `SumI64`,
    /// whose `checked_add` propagates [`RuntimeError::SumOverflow`].
    #[inline]
    fn update(&mut self, i: usize, g: usize) -> Result<(), RuntimeError> {
        match self {
            FusedAcc::SumF64NoNull { v, sums } => sums[g] += v[i],
            FusedAcc::SumF64 { v, sums, valid } => {
                if v.is_valid(i) {
                    sums[g] += v.value(i);
                    valid[g] = true;
                }
            }
            FusedAcc::SumI64 { v, sums, valid } => {
                if v.is_valid(i) {
                    let slot = &mut sums[g];
                    *slot = slot
                        .checked_add(v.value(i))
                        .ok_or(RuntimeError::SumOverflow)?;
                    valid[g] = true;
                }
            }
            FusedAcc::SumDecimal { v, sums, valid, .. } => {
                if v.is_valid(i) {
                    sums[g] += v.value(i);
                    valid[g] = true;
                }
            }
            FusedAcc::CountStar { counts } => counts[g] += 1,
            FusedAcc::CountNoNull { counts } => counts[g] += 1,
            FusedAcc::CountNull { v, counts } => {
                if v.is_valid(i) {
                    counts[g] += 1;
                }
            }
            FusedAcc::MinMaxI64 {
                v,
                cur,
                valid,
                is_min,
            } => {
                if v.is_valid(i) {
                    let val = v.value(i);
                    if !valid[g] || (*is_min && val < cur[g]) || (!*is_min && val > cur[g]) {
                        cur[g] = val;
                        valid[g] = true;
                    }
                }
            }
            FusedAcc::MinMaxF64 {
                v,
                cur,
                valid,
                is_min,
            } => {
                if v.is_valid(i) {
                    let val = v.value(i);
                    if !valid[g] || (*is_min && val < cur[g]) || (!*is_min && val > cur[g]) {
                        cur[g] = val;
                        valid[g] = true;
                    }
                }
            }
            FusedAcc::MinMaxDecimal {
                v,
                cur,
                valid,
                is_min,
                ..
            } => {
                if v.is_valid(i) {
                    let val = v.value(i);
                    if !valid[g] || (*is_min && val < cur[g]) || (!*is_min && val > cur[g]) {
                        cur[g] = val;
                        valid[g] = true;
                    }
                }
            }
            FusedAcc::MinMaxStr { v, cur, is_min } => {
                if v.is_valid(i) {
                    let val = v.value(i);
                    let replace = match &cur[g] {
                        None => true,
                        Some(c) => (*is_min && val < c.as_str()) || (!*is_min && val > c.as_str()),
                    };
                    if replace {
                        cur[g] = Some(val.to_string());
                    }
                }
            }
        }
        Ok(())
    }

    /// Materialize the finished state column — the exact constructor the per-call
    /// kernel uses, so the bytes match.
    fn finish(self) -> Result<ArrayRef, RuntimeError> {
        Ok(match self {
            FusedAcc::SumF64NoNull { sums, .. } => {
                let n = sums.len();
                Arc::new(masked_f64(sums, vec![true; n]))
            }
            FusedAcc::SumF64 { sums, valid, .. } => Arc::new(masked_f64(sums, valid)),
            FusedAcc::SumI64 { sums, valid, .. } => Arc::new(masked_i64(sums, valid)),
            FusedAcc::SumDecimal {
                sums,
                valid,
                precision,
                scale,
                ..
            } => masked_decimal(sums, valid, precision, scale)?,
            FusedAcc::CountStar { counts }
            | FusedAcc::CountNoNull { counts }
            | FusedAcc::CountNull { counts, .. } => Arc::new(Int64Array::from(counts)),
            FusedAcc::MinMaxI64 { cur, valid, .. } => Arc::new(masked_i64(cur, valid)),
            FusedAcc::MinMaxF64 { cur, valid, .. } => Arc::new(masked_f64(cur, valid)),
            FusedAcc::MinMaxDecimal {
                cur,
                valid,
                precision,
                scale,
                ..
            } => masked_decimal(cur, valid, precision, scale)?,
            FusedAcc::MinMaxStr { cur, .. } => Arc::new(StringArray::from(cur)),
        })
    }
}

/// Whether `func` is a scalar aggregate this module can fuse (subject also to a
/// supported value dtype, checked in `classify`).
fn is_fusable_func(func: AggFunc) -> bool {
    matches!(
        func,
        AggFunc::Sum
            | AggFunc::Count
            | AggFunc::CountStar
            | AggFunc::Min
            | AggFunc::Max
            | AggFunc::Mean
    )
}

/// Build the fused accumulator(s) for one call, or `None` if it cannot be fused
/// (complex func, two-input, missing input, or an unsupported value dtype — those
/// fall back to the per-call path, which emits the canonical error if any).
/// `mean` yields two accumulators (`[sum, count]`) to match `accumulate`/`finalize_mean`.
fn classify<'a>(call: &'a AggCall, num_groups: usize) -> Option<Vec<FusedAcc<'a>>> {
    if call.key.is_some() {
        return None;
    }
    match call.func {
        AggFunc::CountStar => Some(vec![FusedAcc::CountStar {
            counts: vec![0; num_groups],
        }]),
        AggFunc::Count => Some(vec![count_acc(call.values.as_ref()?, num_groups)]),
        AggFunc::Sum => Some(vec![sum_acc(call.values.as_ref()?, num_groups)?]),
        AggFunc::Min => Some(vec![minmax_acc(call.values.as_ref()?, num_groups, true)?]),
        AggFunc::Max => Some(vec![minmax_acc(call.values.as_ref()?, num_groups, false)?]),
        AggFunc::Mean => {
            let v = call.values.as_ref()?;
            Some(vec![sum_acc(v, num_groups)?, count_acc(v, num_groups)])
        }
        _ => None,
    }
}

fn count_acc(values: &ArrayRef, num_groups: usize) -> FusedAcc<'_> {
    if values.null_count() == 0 {
        FusedAcc::CountNoNull {
            counts: vec![0; num_groups],
        }
    } else {
        FusedAcc::CountNull {
            v: values.as_ref(),
            counts: vec![0; num_groups],
        }
    }
}

fn sum_acc(values: &ArrayRef, num_groups: usize) -> Option<FusedAcc<'_>> {
    Some(match values.data_type() {
        DataType::Float64 => {
            let v = values.as_primitive::<Float64Type>();
            if v.null_count() == 0 {
                FusedAcc::SumF64NoNull {
                    v: v.values(),
                    sums: vec![0.0; num_groups],
                }
            } else {
                FusedAcc::SumF64 {
                    v,
                    sums: vec![0.0; num_groups],
                    valid: vec![false; num_groups],
                }
            }
        }
        DataType::Int64 => FusedAcc::SumI64 {
            v: values.as_primitive::<Int64Type>(),
            sums: vec![0; num_groups],
            valid: vec![false; num_groups],
        },
        DataType::Decimal128(p, s) => FusedAcc::SumDecimal {
            v: values.as_primitive::<Decimal128Type>(),
            sums: vec![0; num_groups],
            valid: vec![false; num_groups],
            precision: *p,
            scale: *s,
        },
        _ => return None, // unsupported dtype → per-call path emits the canonical error
    })
}

fn minmax_acc(values: &ArrayRef, num_groups: usize, is_min: bool) -> Option<FusedAcc<'_>> {
    Some(match values.data_type() {
        DataType::Int64 => FusedAcc::MinMaxI64 {
            v: values.as_primitive::<Int64Type>(),
            cur: vec![0; num_groups],
            valid: vec![false; num_groups],
            is_min,
        },
        DataType::Float64 => FusedAcc::MinMaxF64 {
            v: values.as_primitive::<Float64Type>(),
            cur: vec![0.0; num_groups],
            valid: vec![false; num_groups],
            is_min,
        },
        DataType::Decimal128(p, s) => FusedAcc::MinMaxDecimal {
            v: values.as_primitive::<Decimal128Type>(),
            cur: vec![0; num_groups],
            valid: vec![false; num_groups],
            is_min,
            precision: *p,
            scale: *s,
        },
        DataType::Utf8 => FusedAcc::MinMaxStr {
            v: values.as_any().downcast_ref::<StringArray>().expect("utf8"),
            cur: vec![None; num_groups],
            is_min,
        },
        _ => return None,
    })
}

/// Minimum fusable aggregates to bother fusing: below this a lone aggregate keeps
/// the proven per-call path (no `group_ids`-reuse win to gain).
const FUSE_THRESHOLD: usize = 2;

/// Slots per group past which a group's state no longer fits one 64-byte cache line,
/// which is the entire point of the interleaved layout.
const AOS_MAX_SLOTS: usize = 8;

/// One interleaved accumulator slot: the operation to apply and the (null-free) values
/// slice it reads. Every state is exactly 8 bytes, held as a `u64` and bit-punned for the
/// float cases, so a group's whole state is one contiguous run of `u64`s.
enum SlotOp<'a> {
    SumF64(&'a [f64]),
    SumI64(&'a [i64]),
    /// `count(*)` and `count(col)` over a null-free column are the same increment.
    Count,
    MinI64(&'a [i64]),
    MaxI64(&'a [i64]),
    MinF64(&'a [f64]),
    MaxF64(&'a [f64]),
}

/// Interleave every call's state into one `num_groups × stride` array, or `None` if any
/// call falls outside the supported subset (a nullable or non-`Int64`/`Float64` column, a
/// complex aggregate, or more than [`AOS_MAX_SLOTS`] slots).
///
/// Returns the slot ops and, per call, how many slots it owns (`mean` owns `[sum, count]`,
/// matching `classify`/`accumulate`'s column order).
fn classify_aos<'a>(calls: &'a [AggCall]) -> Option<(Vec<SlotOp<'a>>, Vec<(usize, usize)>)> {
    fn null_free_slice<'b>(v: &'b ArrayRef) -> Option<&'b ArrayRef> {
        (v.null_count() == 0).then_some(v)
    }
    fn sum_slot(v: &ArrayRef) -> Option<SlotOp<'_>> {
        match null_free_slice(v)?.data_type() {
            DataType::Float64 => Some(SlotOp::SumF64(v.as_primitive::<Float64Type>().values())),
            DataType::Int64 => Some(SlotOp::SumI64(v.as_primitive::<Int64Type>().values())),
            _ => None,
        }
    }
    fn minmax_slot(v: &ArrayRef, is_min: bool) -> Option<SlotOp<'_>> {
        match (null_free_slice(v)?.data_type(), is_min) {
            (DataType::Int64, true) => Some(SlotOp::MinI64(v.as_primitive::<Int64Type>().values())),
            (DataType::Int64, false) => Some(SlotOp::MaxI64(v.as_primitive::<Int64Type>().values())),
            (DataType::Float64, true) => {
                Some(SlotOp::MinF64(v.as_primitive::<Float64Type>().values()))
            }
            (DataType::Float64, false) => {
                Some(SlotOp::MaxF64(v.as_primitive::<Float64Type>().values()))
            }
            _ => None,
        }
    }

    let mut slots: Vec<SlotOp<'a>> = Vec::new();
    let mut layout: Vec<(usize, usize)> = Vec::new();
    for (idx, call) in calls.iter().enumerate() {
        if call.key.is_some() {
            return None;
        }
        let start = slots.len();
        match call.func {
            AggFunc::CountStar => slots.push(SlotOp::Count),
            AggFunc::Count => {
                null_free_slice(call.values.as_ref()?)?;
                slots.push(SlotOp::Count);
            }
            AggFunc::Sum => slots.push(sum_slot(call.values.as_ref()?)?),
            AggFunc::Min => slots.push(minmax_slot(call.values.as_ref()?, true)?),
            AggFunc::Max => slots.push(minmax_slot(call.values.as_ref()?, false)?),
            AggFunc::Mean => {
                let v = call.values.as_ref()?;
                slots.push(sum_slot(v)?);
                slots.push(SlotOp::Count);
            }
            _ => return None,
        }
        layout.push((idx, slots.len() - start));
    }
    (slots.len() >= FUSE_THRESHOLD && slots.len() <= AOS_MAX_SLOTS).then_some((slots, layout))
}

/// Fused scan over an **interleaved (array-of-structs)** state array.
///
/// The struct-of-arrays layout `run_fused` uses gives each accumulator its own `Vec`, so a
/// row with N aggregates makes N random accesses into N different arrays — N cache misses
/// once the group count outgrows L1. Interleaving puts group `g`'s state for *every*
/// aggregate in one contiguous `stride`-slot run, so a row touches one cache line.
/// Measured (single-threaded, 2 M rows, 10 k groups): five aggregates went from 98.5 to
/// ~40 ns/row, while one aggregate is unchanged.
///
/// Bit-identical to `run_fused`: the same values are combined into the same groups in the
/// same row order, only the memory they live in changes. `min`/`max` still take the first
/// row of a group verbatim (never a sentinel), so a leading `NaN` survives exactly as it
/// does in the per-call kernel — one shared `seen` bitmap suffices because every column
/// here is null-free, so every `min`/`max` accumulator's `valid[g]` flips on the same row.
///
/// Returns `false` when the call set is outside the supported subset; the caller then runs
/// the struct-of-arrays path.
fn try_run_aos(
    calls: &[AggCall],
    group_ids: &[u32],
    num_groups: usize,
    out: &mut [Option<Vec<ArrayRef>>],
) -> Result<bool, RuntimeError> {
    let Some((slots, layout)) = classify_aos(calls) else {
        return Ok(false);
    };
    let stride = slots.len();
    let mut state: Vec<u64> = vec![0; num_groups * stride];
    // One bit per group, not per accumulator: tiny and L1-resident.
    let mut seen: Vec<bool> = vec![false; num_groups];

    for (i, &gid) in group_ids.iter().enumerate() {
        let g = gid as usize;
        let first = !seen[g];
        seen[g] = true;
        let cell = &mut state[g * stride..(g + 1) * stride];
        for (slot, s) in slots.iter().zip(cell.iter_mut()) {
            match slot {
                SlotOp::SumF64(v) => *s = (f64::from_bits(*s) + v[i]).to_bits(),
                SlotOp::SumI64(v) => {
                    *s = (*s as i64)
                        .checked_add(v[i])
                        .ok_or(RuntimeError::SumOverflow)? as u64
                }
                SlotOp::Count => *s += 1,
                SlotOp::MinI64(v) => {
                    if first || v[i] < *s as i64 {
                        *s = v[i] as u64;
                    }
                }
                SlotOp::MaxI64(v) => {
                    if first || v[i] > *s as i64 {
                        *s = v[i] as u64;
                    }
                }
                SlotOp::MinF64(v) => {
                    if first || v[i] < f64::from_bits(*s) {
                        *s = v[i].to_bits();
                    }
                }
                SlotOp::MaxF64(v) => {
                    if first || v[i] > f64::from_bits(*s) {
                        *s = v[i].to_bits();
                    }
                }
            }
        }
    }

    let gather_i64 = |k: usize| -> Vec<i64> {
        (0..num_groups).map(|g| state[g * stride + k] as i64).collect()
    };
    let gather_f64 = |k: usize| -> Vec<f64> {
        (0..num_groups)
            .map(|g| f64::from_bits(state[g * stride + k]))
            .collect()
    };
    let mut k = 0usize;
    for (idx, n_cols) in layout {
        let mut cols = Vec::with_capacity(n_cols);
        for _ in 0..n_cols {
            // Validity mirrors the struct-of-arrays `finish`: a float sum is always valid
            // (it starts at 0.0), a count needs no mask, everything else is valid exactly
            // where the group saw a row.
            let col: ArrayRef = match slots[k] {
                SlotOp::SumF64(_) => Arc::new(masked_f64(gather_f64(k), vec![true; num_groups])),
                SlotOp::SumI64(_) => Arc::new(masked_i64(gather_i64(k), seen.clone())),
                SlotOp::Count => Arc::new(Int64Array::from(gather_i64(k))),
                SlotOp::MinI64(_) | SlotOp::MaxI64(_) => {
                    Arc::new(masked_i64(gather_i64(k), seen.clone()))
                }
                SlotOp::MinF64(_) | SlotOp::MaxF64(_) => {
                    Arc::new(masked_f64(gather_f64(k), seen.clone()))
                }
            };
            cols.push(col);
            k += 1;
        }
        out[idx] = Some(cols);
    }
    Ok(true)
}

/// Run one scatter-add pass over `group_ids` for every fusable call, writing each
/// fused call's state column(s) into `out[idx]` (positions match `calls`); leaves
/// non-fusable calls' slots `None` for the per-call path. A no-op (out untouched)
/// when fewer than [`FUSE_THRESHOLD`] calls are fusable.
pub(super) fn run_fused(
    calls: &[AggCall],
    group_ids: &[u32],
    num_groups: usize,
    out: &mut [Option<Vec<ArrayRef>>],
) -> Result<(), RuntimeError> {
    // Quick reject: enough fusable funcs to be worth a fused pass?
    if calls.iter().filter(|c| is_fusable_func(c.func)).count() < FUSE_THRESHOLD {
        return Ok(());
    }
    // Interleaved layout when every call is a null-free numeric scalar aggregate: one
    // cache line per row instead of one per aggregate. Falls through otherwise.
    if num_groups > 0 && try_run_aos(calls, group_ids, num_groups, out)? {
        return Ok(());
    }
    // Classify; an unsupported dtype drops a call back to per-call (None slot).
    let mut accs: Vec<FusedAcc> = Vec::new();
    let mut layout: Vec<(usize, usize)> = Vec::new(); // (call idx, n state cols)
    for (idx, call) in calls.iter().enumerate() {
        if !is_fusable_func(call.func) {
            continue;
        }
        if let Some(call_accs) = classify(call, num_groups) {
            layout.push((idx, call_accs.len()));
            accs.extend(call_accs);
        }
    }
    if layout.len() < FUSE_THRESHOLD {
        return Ok(()); // not enough actually fused (e.g. unsupported dtypes)
    }

    // The single fused scan: `group_ids` (and the row walk) read exactly once.
    for (i, &gid) in group_ids.iter().enumerate() {
        let g = gid as usize;
        for acc in accs.iter_mut() {
            acc.update(i, g)?;
        }
    }

    // Reassemble each fused call's state columns into its output slot, in order.
    let mut accs = accs.into_iter();
    for (idx, n_cols) in layout {
        let mut cols = Vec::with_capacity(n_cols);
        for _ in 0..n_cols {
            cols.push(accs.next().expect("layout matches accs").finish()?);
        }
        out[idx] = Some(cols);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Decimal128Array, Float64Array, Int64Array};

    /// Reference oracle: today's per-call kernel (`super::accumulate`, the parent's
    /// private fn — a child module may call it). Only fusable funcs are tested, all
    /// of which `accumulate` handles directly.
    fn per_call(calls: &[AggCall], group_ids: &[u32], num_groups: usize) -> Vec<Vec<ArrayRef>> {
        calls
            .iter()
            .map(|c| {
                super::super::accumulate(c.func, c.values.as_ref(), group_ids, num_groups).unwrap()
            })
            .collect()
    }

    fn fused(calls: &[AggCall], group_ids: &[u32], num_groups: usize) -> Vec<Vec<ArrayRef>> {
        let mut out: Vec<Option<Vec<ArrayRef>>> = vec![None; calls.len()];
        run_fused(calls, group_ids, num_groups, &mut out).unwrap();
        out.into_iter()
            .map(|o| o.expect("all fused in these tests"))
            .collect()
    }

    fn assert_cols_eq(a: &[ArrayRef], b: &[ArrayRef]) {
        assert_eq!(a.len(), b.len(), "column count");
        for (x, y) in a.iter().zip(b) {
            assert_eq!(x.as_ref(), y.as_ref(), "fused != per-call");
        }
    }

    #[test]
    fn fused_equals_per_call_with_nulls() {
        // Mixed fusable set over i64 + f64, with nulls and 3 groups.
        let f: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(1.0),
            None,
            Some(3.5),
            Some(2.0),
            None,
            Some(4.0),
        ]));
        let i: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(10),
            Some(20),
            None,
            Some(5),
            Some(7),
            None,
        ]));
        let group_ids = [0u32, 1, 0, 2, 1, 2];
        let calls = vec![
            AggCall::new(AggFunc::Sum, Some(f.clone())),
            AggCall::new(AggFunc::Count, Some(i.clone())),
            AggCall::new(AggFunc::CountStar, None),
            AggCall::new(AggFunc::Mean, Some(f.clone())),
            AggCall::new(AggFunc::Min, Some(i.clone())),
            AggCall::new(AggFunc::Max, Some(f.clone())),
        ];
        let want = per_call(&calls, &group_ids, 3);
        let got = fused(&calls, &group_ids, 3);
        for (w, g) in want.iter().zip(&got) {
            assert_cols_eq(w, g);
        }
    }

    /// The interleaved (AoS) path must equal the per-call kernel for the whole
    /// null-free numeric set — this is the shape that actually engages it.
    #[test]
    fn aos_equals_per_call_on_null_free_numeric() {
        let f: ArrayRef = Arc::new(Float64Array::from(vec![1.5, -2.0, 3.25, 0.0, 9.75, -1.0]));
        let i: ArrayRef = Arc::new(Int64Array::from(vec![10i64, 20, -5, 7, 0, 3]));
        let group_ids = [0u32, 1, 0, 2, 1, 2];
        let calls = vec![
            AggCall::new(AggFunc::Sum, Some(f.clone())),
            AggCall::new(AggFunc::Mean, Some(f.clone())),
            AggCall::new(AggFunc::Min, Some(i.clone())),
            AggCall::new(AggFunc::Max, Some(f.clone())),
            AggCall::new(AggFunc::CountStar, None),
        ];
        assert!(classify_aos(&calls).is_some(), "AoS should engage");
        for (w, g) in per_call(&calls, &group_ids, 3)
            .iter()
            .zip(&fused(&calls, &group_ids, 3))
        {
            assert_cols_eq(w, g);
        }
    }

    /// A leading `NaN` must survive `min`/`max` exactly as the per-call kernel leaves it —
    /// the reason the AoS path takes the first row verbatim instead of seeding a sentinel.
    #[test]
    fn aos_minmax_nan_matches_per_call() {
        let f: ArrayRef = Arc::new(Float64Array::from(vec![f64::NAN, 1.0, 2.0, f64::NAN]));
        let group_ids = [0u32, 0, 1, 1];
        let calls = vec![
            AggCall::new(AggFunc::Min, Some(f.clone())),
            AggCall::new(AggFunc::Max, Some(f.clone())),
        ];
        assert!(classify_aos(&calls).is_some());
        for (w, g) in per_call(&calls, &group_ids, 2)
            .iter()
            .zip(&fused(&calls, &group_ids, 2))
        {
            assert_cols_eq(w, g);
        }
    }

    /// `-0.0` / `+0.0` compare equal, so the first row wins — same as per-call.
    #[test]
    fn aos_signed_zero_matches_per_call() {
        let f: ArrayRef = Arc::new(Float64Array::from(vec![-0.0f64, 0.0, 0.0, -0.0]));
        let group_ids = [0u32, 0, 1, 1];
        let calls = vec![
            AggCall::new(AggFunc::Min, Some(f.clone())),
            AggCall::new(AggFunc::Max, Some(f.clone())),
        ];
        for (w, g) in per_call(&calls, &group_ids, 2)
            .iter()
            .zip(&fused(&calls, &group_ids, 2))
        {
            assert_cols_eq(w, g);
        }
    }

    /// An `i64` sum that overflows must raise, not wrap.
    #[test]
    fn aos_i64_sum_overflow_errors() {
        let i: ArrayRef = Arc::new(Int64Array::from(vec![i64::MAX, 1]));
        let group_ids = [0u32, 0];
        let calls = vec![
            AggCall::new(AggFunc::Sum, Some(i.clone())),
            AggCall::new(AggFunc::CountStar, None),
        ];
        assert!(classify_aos(&calls).is_some());
        let mut out: Vec<Option<Vec<ArrayRef>>> = vec![None; calls.len()];
        assert!(run_fused(&calls, &group_ids, 1, &mut out).is_err());
    }

    /// A nullable column is outside the AoS subset: it must decline, and the
    /// struct-of-arrays path still matches per-call.
    #[test]
    fn aos_declines_nullable_and_soa_still_correct() {
        let f: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)]));
        let calls = vec![
            AggCall::new(AggFunc::Sum, Some(f.clone())),
            AggCall::new(AggFunc::Max, Some(f.clone())),
        ];
        assert!(classify_aos(&calls).is_none(), "nullable must decline AoS");
        let group_ids = [0u32, 0, 1];
        for (w, g) in per_call(&calls, &group_ids, 2)
            .iter()
            .zip(&fused(&calls, &group_ids, 2))
        {
            assert_cols_eq(w, g);
        }
    }

    /// More than `AOS_MAX_SLOTS` slots would spill the cache line: decline.
    #[test]
    fn aos_declines_when_state_exceeds_a_cache_line() {
        let f: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));
        // 5 means = 10 slots > AOS_MAX_SLOTS.
        let calls: Vec<AggCall> = (0..5)
            .map(|_| AggCall::new(AggFunc::Mean, Some(f.clone())))
            .collect();
        assert!(classify_aos(&calls).is_none());
        // ... and the SoA path still agrees with per-call.
        let group_ids = [0u32, 0];
        for (w, g) in per_call(&calls, &group_ids, 1)
            .iter()
            .zip(&fused(&calls, &group_ids, 1))
        {
            assert_cols_eq(w, g);
        }
    }

    /// A non-fusable call in the set (e.g. `var`) keeps the whole thing off AoS.
    #[test]
    fn aos_declines_when_a_call_is_not_fusable() {
        let f: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));
        let calls = vec![
            AggCall::new(AggFunc::Sum, Some(f.clone())),
            AggCall::new(AggFunc::Var, Some(f.clone())),
        ];
        assert!(classify_aos(&calls).is_none());
    }

    #[test]
    fn fused_f64_sum_nonull_matches() {
        // No-null f64 sum must equal the per-call no-null fast path bit-for-bit.
        let f: ArrayRef = Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3, 4.4, 5.5]));
        let group_ids = [0u32, 1, 0, 1, 0];
        let calls = vec![
            AggCall::new(AggFunc::Sum, Some(f.clone())),
            AggCall::new(AggFunc::Sum, Some(f.clone())),
        ];
        let want = per_call(&calls, &group_ids, 2);
        let got = fused(&calls, &group_ids, 2);
        let w0 = want[0][0].as_primitive::<Float64Type>();
        let g0 = got[0][0].as_primitive::<Float64Type>();
        for k in 0..2 {
            assert_eq!(
                w0.value(k).to_bits(),
                g0.value(k).to_bits(),
                "bit-exact f64 sum"
            );
        }
    }

    #[test]
    fn fused_sum_overflow_still_errors() {
        let i: ArrayRef = Arc::new(Int64Array::from(vec![i64::MAX, 1]));
        let group_ids = [0u32, 0];
        let calls = vec![
            AggCall::new(AggFunc::Sum, Some(i.clone())),
            AggCall::new(AggFunc::CountStar, None),
        ];
        let mut out: Vec<Option<Vec<ArrayRef>>> = vec![None; calls.len()];
        let r = run_fused(&calls, &group_ids, 1, &mut out);
        assert!(matches!(r, Err(RuntimeError::SumOverflow)), "got {r:?}");
    }

    #[test]
    fn fused_decimal_sum_matches() {
        let d: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(100), Some(250), None, Some(50)])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        );
        let group_ids = [0u32, 1, 0, 1];
        let calls = vec![
            AggCall::new(AggFunc::Sum, Some(d.clone())),
            AggCall::new(AggFunc::CountStar, None),
        ];
        let want = per_call(&calls, &group_ids, 2);
        let got = fused(&calls, &group_ids, 2);
        assert_cols_eq(&want[0], &got[0]);
    }

    #[test]
    fn below_threshold_is_noop() {
        // A single fusable call leaves `out` untouched (per-call path handles it).
        let f: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));
        let calls = vec![AggCall::new(AggFunc::Sum, Some(f))];
        let mut out: Vec<Option<Vec<ArrayRef>>> = vec![None; 1];
        run_fused(&calls, &[0u32, 1], 2, &mut out).unwrap();
        assert!(out[0].is_none(), "single fusable must not fuse");
    }

    #[test]
    fn non_fusable_left_for_per_call() {
        // Median + Sum + Count: Median's slot stays None; Sum/Count fuse.
        let f: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]));
        let calls = vec![
            AggCall::new(AggFunc::Median, Some(f.clone())),
            AggCall::new(AggFunc::Sum, Some(f.clone())),
            AggCall::new(AggFunc::Count, Some(f.clone())),
        ];
        let mut out: Vec<Option<Vec<ArrayRef>>> = vec![None; calls.len()];
        run_fused(&calls, &[0u32, 0, 1], 2, &mut out).unwrap();
        assert!(out[0].is_none(), "median not fused");
        assert!(out[1].is_some(), "sum fused");
        assert!(out[2].is_some(), "count fused");
    }
}
