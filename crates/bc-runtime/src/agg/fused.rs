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
    /// `sum` over a no-null `Int64` column: scatter-add straight from the values slice with
    /// no per-row validity branch and no per-row `valid` write, since every group is non-empty
    /// and therefore all-valid. The exact counterpart of `SumF64NoNull`, and of the no-null arm
    /// `accum::sum_acc` has had all along — this module's docstring claims to reproduce
    /// "the no-null fast paths" and, for `Int64` alone, did not. Integer `sum` is the most
    /// common aggregate in every suite the engine is measured on, so fusing two of them made
    /// each *slower per row* than not fusing at all.
    SumI64NoNull {
        v: &'a [i64],
        sums: Vec<i64>,
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

/// Run one accumulator arm's body over a block of rows, resolving the group id per row.
///
/// The whole point of the arms below: the `match` that selects an arm happens **once per
/// block**, not once per row, so each arm's body compiles to a tight monomorphic loop over
/// `start..end` with the concrete array type in hand. Writing the loop inside every arm by
/// hand would be the same code twelve times, and the twelfth would eventually differ from the
/// first.
macro_rules! block_loop {
    ($ids:expr, $start:expr, $end:expr, |$i:ident, $g:ident| $body:block) => {
        for $i in $start..$end {
            let $g = $ids[$i] as usize;
            $body
        }
    };
}

impl FusedAcc<'_> {
    /// Apply rows `start..end` (whose group ids are `ids[start..end]`) to this accumulator.
    /// Infallible except `SumI64`/`SumDecimal`, whose `checked_add` propagates
    /// [`RuntimeError::SumOverflow`].
    ///
    /// Takes a *block* rather than a row because the enum dispatch is otherwise the operator.
    /// The driver used to call a per-row `update`, so a three-aggregate group-by over 10M rows
    /// paid **30M** `match`es on top of the arithmetic — measured at ~1.5 ns per row per
    /// aggregate against DuckDB's ~0.29, which was most of a low-cardinality group-by's cost
    /// (2.49x DuckDB on a single `sum` into 100 groups, 3.55x on three).
    ///
    /// **Bit-identical to the per-row form**, and for the same reason the module docstring
    /// gives: accumulators never alias, and each still sees every row exactly once in
    /// increasing `i`. Only the interleaving *between* accumulators changes, which no
    /// accumulator can observe.
    #[inline]
    fn update_block(&mut self, ids: &[u32], start: usize, end: usize) -> Result<(), RuntimeError> {
        match self {
            // The two no-null sums walk *zipped slices* rather than indexing by `i`. Both
            // slices are bounds-checked once, at the slicing, instead of twice per row, which
            // is what lets the loop body reduce to a load, an indexed add and a store. Every
            // other arm has to index, because it consults a validity bitmap by position.
            FusedAcc::SumF64NoNull { v, sums } => {
                for (&gid, &x) in ids[start..end].iter().zip(&v[start..end]) {
                    sums[gid as usize] += x;
                }
            }
            FusedAcc::SumI64NoNull { v, sums } => {
                for (&gid, &x) in ids[start..end].iter().zip(&v[start..end]) {
                    let slot = &mut sums[gid as usize];
                    // Checked throughout, exactly as `accum::sum_acc` is: a silent i64 wrap
                    // would be a wrong answer. It costs a not-taken branch, not a fast path.
                    *slot = slot.checked_add(x).ok_or(RuntimeError::SumOverflow)?;
                }
            }
            FusedAcc::SumF64 { v, sums, valid } => {
                block_loop!(ids, start, end, |i, g| {
                    if v.is_valid(i) {
                        sums[g] += v.value(i);
                        valid[g] = true;
                    }
                })
            }
            FusedAcc::SumI64 { v, sums, valid } => {
                block_loop!(ids, start, end, |i, g| {
                    if v.is_valid(i) {
                        let slot = &mut sums[g];
                        *slot = slot
                            .checked_add(v.value(i))
                            .ok_or(RuntimeError::SumOverflow)?;
                        valid[g] = true;
                    }
                })
            }
            FusedAcc::SumDecimal { v, sums, valid, .. } => {
                block_loop!(ids, start, end, |i, g| {
                    if v.is_valid(i) {
                        // checked_add: a decimal SUM past i128 range errors, not wraps (as the
                        // i64 SumInt arm above does).
                        sums[g] = sums[g]
                            .checked_add(v.value(i))
                            .ok_or(RuntimeError::SumOverflow)?;
                        valid[g] = true;
                    }
                })
            }
            FusedAcc::CountStar { counts } | FusedAcc::CountNoNull { counts } => {
                block_loop!(ids, start, end, |i, g| {
                    let _ = i;
                    counts[g] += 1;
                })
            }
            FusedAcc::CountNull { v, counts } => {
                block_loop!(ids, start, end, |i, g| {
                    if v.is_valid(i) {
                        counts[g] += 1;
                    }
                })
            }
            FusedAcc::MinMaxI64 {
                v,
                cur,
                valid,
                is_min,
            } => {
                block_loop!(ids, start, end, |i, g| {
                    if v.is_valid(i) {
                        let val = v.value(i);
                        if !valid[g] || (*is_min && val < cur[g]) || (!*is_min && val > cur[g]) {
                            cur[g] = val;
                            valid[g] = true;
                        }
                    }
                })
            }
            FusedAcc::MinMaxF64 {
                v,
                cur,
                valid,
                is_min,
            } => {
                block_loop!(ids, start, end, |i, g| {
                    if v.is_valid(i) {
                        let val = v.value(i);
                        // Same total order the per-call `minmax_acc` uses (`crate::keys`), not
                        // raw IEEE `<`/`>` — otherwise NaN never wins here and the fused path
                        // disagrees with the per-call path, which
                        // `fused_minmax_nan_matches_per_call` pins.
                        let ord = crate::keys::float_total_cmp(val, cur[g]);
                        let wins = if *is_min {
                            ord == std::cmp::Ordering::Less
                        } else {
                            ord == std::cmp::Ordering::Greater
                        };
                        if !valid[g] || wins {
                            cur[g] = val;
                            valid[g] = true;
                        }
                    }
                })
            }
            FusedAcc::MinMaxDecimal {
                v,
                cur,
                valid,
                is_min,
                ..
            } => {
                block_loop!(ids, start, end, |i, g| {
                    if v.is_valid(i) {
                        let val = v.value(i);
                        if !valid[g] || (*is_min && val < cur[g]) || (!*is_min && val > cur[g]) {
                            cur[g] = val;
                            valid[g] = true;
                        }
                    }
                })
            }
            FusedAcc::MinMaxStr { v, cur, is_min } => {
                block_loop!(ids, start, end, |i, g| {
                    if v.is_valid(i) {
                        let val = v.value(i);
                        let replace = match &cur[g] {
                            None => true,
                            Some(c) => {
                                (*is_min && val < c.as_str()) || (!*is_min && val > c.as_str())
                            }
                        };
                        if replace {
                            cur[g] = Some(val.to_string());
                        }
                    }
                })
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
            FusedAcc::SumI64NoNull { sums, .. } => {
                let n = sums.len();
                Arc::new(masked_i64(sums, vec![true; n]))
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
        DataType::Int64 => {
            let v = values.as_primitive::<Int64Type>();
            if v.null_count() == 0 {
                FusedAcc::SumI64NoNull {
                    v: v.values(),
                    sums: vec![0; num_groups],
                }
            } else {
                FusedAcc::SumI64 {
                    v,
                    sums: vec![0; num_groups],
                    valid: vec![false; num_groups],
                }
            }
        }
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

/// Rows per block in the fused scan. 8,192 `u32` group ids is 32 KiB — an L1-sized working
/// set, so every accumulator in a block sweeps the *same* ids out of cache. Big enough that
/// the per-block enum dispatch is amortized to nothing (one `match` per 8,192 rows), small
/// enough that the ids do not fall out of L1 before the last accumulator reads them. Half a
/// morsel (`bc_arrow` uses 16,384 rows), so a morsel is exactly two blocks and a short final
/// block is the common case rather than a special one.
const FUSE_BLOCK_ROWS: usize = 8_192;

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
    // NB: an interleaved (array-of-structs) state array — group `g`'s state for every
    // aggregate in one contiguous cache line, instead of one `Vec` per accumulator — was
    // implemented and A/B'd inside one binary. It does NOT pay off: `partial` runs per
    // *morsel*, so `num_groups` never exceeds the 16,384-row morsel and all the
    // accumulator arrays stay L2-resident whatever the layout. Measured at 2 M rows, five
    // aggregates: +2% at 10k groups, **-5%** at 1.5M groups. The per-aggregate cost is the
    // scatter's dependent load-modify-store, not a cache miss per accumulator.
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

    // The fused scan, blocked: each accumulator sweeps one cache-resident block of
    // `group_ids` with its own tight monomorphic loop, then the next accumulator sweeps the
    // same block. That keeps both properties rather than trading one for the other — the
    // enum dispatch is paid per *block* instead of per row (see `update_block`), and
    // `group_ids` still stays hot rather than being streamed from DRAM once per aggregate,
    // which is what a plain accumulator-outer loop would do.
    for start in (0..group_ids.len()).step_by(FUSE_BLOCK_ROWS) {
        let end = (start + FUSE_BLOCK_ROWS).min(group_ids.len());
        for acc in accs.iter_mut() {
            acc.update_block(group_ids, start, end)?;
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

    /// The whole null-free numeric set must equal the per-call kernel — the common
    /// `sum, mean, min, max, count` shape.
    #[test]
    fn fused_equals_per_call_on_null_free_numeric() {
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
        for (w, g) in per_call(&calls, &group_ids, 3)
            .iter()
            .zip(&fused(&calls, &group_ids, 3))
        {
            assert_cols_eq(w, g);
        }
    }

    /// A leading `NaN` must survive `min`/`max` exactly as the per-call kernel leaves it:
    /// the first row of a group is taken verbatim, never compared against a sentinel.
    #[test]
    fn fused_minmax_nan_matches_per_call() {
        let f: ArrayRef = Arc::new(Float64Array::from(vec![f64::NAN, 1.0, 2.0, f64::NAN]));
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

    /// `-0.0` and `+0.0` compare equal, so the first row of the group wins — same as per-call.
    #[test]
    fn fused_signed_zero_matches_per_call() {
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

    /// The `Int64` counterpart of `fused_f64_sum_nonull_matches`, and the test whose absence
    /// let the two paths drift: `accum::sum_acc` has had a no-null integer fast path all along
    /// and the fused arm did not, so fusing two integer sums was *slower per row* than not
    /// fusing. Nothing failed — the answers agreed — which is exactly why only a test that
    /// exercises this shape keeps the arm from being dropped again.
    #[test]
    fn fused_i64_sum_nonull_matches_per_call() {
        let i: ArrayRef = Arc::new(Int64Array::from(vec![10i64, -20, 5, 7, 0, -3]));
        let group_ids = [0u32, 1, 0, 2, 1, 2];
        let calls = vec![
            AggCall::new(AggFunc::Sum, Some(i.clone())),
            AggCall::new(AggFunc::Sum, Some(i.clone())),
            AggCall::new(AggFunc::CountStar, None),
        ];
        let want = per_call(&calls, &group_ids, 3);
        let got = fused(&calls, &group_ids, 3);
        for (w, g) in want.iter().zip(&got) {
            assert_cols_eq(w, g);
        }
        // And the validity the no-null arm asserts rather than accumulates: every group is
        // non-empty, so every sum is non-null.
        let sums = got[0][0].as_primitive::<Int64Type>();
        assert_eq!(sums.null_count(), 0, "a no-null input yields no null sums");
    }

    /// A nullable `Int64` column must still take the validity-tracking arm, so a group whose
    /// every row is null comes back null rather than as a zero the no-null arm would assert.
    #[test]
    fn fused_i64_sum_with_nulls_keeps_group_validity() {
        let i: ArrayRef = Arc::new(Int64Array::from(vec![Some(4i64), None, Some(6), None]));
        let group_ids = [0u32, 1, 0, 1];
        let calls = vec![
            AggCall::new(AggFunc::Sum, Some(i.clone())),
            AggCall::new(AggFunc::CountStar, None),
        ];
        let want = per_call(&calls, &group_ids, 2);
        let got = fused(&calls, &group_ids, 2);
        assert_cols_eq(&want[0], &got[0]);
        let sums = got[0][0].as_primitive::<Int64Type>();
        assert_eq!(sums.value(0), 10);
        assert!(sums.is_null(1), "an all-null group sums to null, not 0");
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
