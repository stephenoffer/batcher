//! Per-type accumulator helpers for `sum`/`min`/`max` and the masked-array and
//! concat utilities they share.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BinaryArray, BooleanArray, Decimal128Array, Float64Array, Int64Array,
    LargeBinaryArray, LargeStringArray, StringArray, UInt32Array,
};
use arrow::compute::take;
use arrow::datatypes::{DataType, Decimal128Type, Float64Type, Int64Type};

use super::{accumulate, arg_extreme_state, covar_state, AggCall, AggFunc, Partial};
use crate::error::RuntimeError;

/// Produce the partial-state columns for one aggregate call. The two-input functions
/// (`arg_min`/`arg_max` carry an ordering key; `covar`/`corr` carry a second value) go to
/// their dedicated builders; every single-input function goes to [`accumulate`].
pub(super) fn accumulate_call(
    call: &AggCall,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    match call.func {
        AggFunc::ArgMin | AggFunc::ArgMax => arg_extreme_state(
            require(call.values.as_ref(), call.func)?,
            require(call.key.as_ref(), call.func)?,
            group_ids,
            num_groups,
            matches!(call.func, AggFunc::ArgMax),
        ),
        AggFunc::CovarPop | AggFunc::CovarSamp | AggFunc::Corr => covar_state(
            require(call.values.as_ref(), call.func)?,
            require(call.key.as_ref(), call.func)?,
            group_ids,
            num_groups,
        ),
        _ => accumulate(call.func, call.values.as_ref(), group_ids, num_groups),
    }
}

/// Global (no-`GROUP BY`) partial aggregation: every row is the single group 0. The scalar
/// reductions (`sum`/`min`/`max`/`count`/`mean`) hit their `num_groups == 1` kernels; the
/// rest scatter into the one group through a shared zero-id buffer. Group columns are empty
/// (a global aggregate has no key), matching the grouped `partial`'s `Partial` shape.
///
/// **The zero-id buffer is built only if some call actually reads it**, and that is the whole
/// performance story of a keyless aggregate. It is one `u32` per row — 64 KiB at a full morsel —
/// and a `SUM` over 6 M rows folds 366 morsels, so unconditionally allocating and zeroing it
/// moved **23 MB the query has no use for**, against the 48 MB the column itself is worth. The
/// cost lands twice: single-threaded it was 8.6 ms against `pyarrow`'s 5.1 ms for the same
/// reduction, and in the pool it competes for the very memory bandwidth the sum is bound by, so
/// the parallel path saturated at 2.5x. The kernels that never read the buffer are exactly those
/// with a whole-column reduction ([`global_reduces_whole_column`]), and they now get an empty
/// slice — the convention [`count_non_null`] and `var::var_state` already documented but nothing
/// supplied.
pub(crate) fn global_partial(calls: &[AggCall], num_rows: usize) -> Result<Partial, RuntimeError> {
    let mut zeros: Vec<u32> = Vec::new();
    let mut states = Vec::with_capacity(calls.len());
    for call in calls {
        // `COUNT(*)` reads no values at all, so its whole-column answer is the row count — which
        // the scatter loop can only recover by counting a buffer of zeros it was handed.
        if matches!(call.func, AggFunc::CountStar) {
            states.push(vec![
                Arc::new(Int64Array::from(vec![num_rows as i64])) as ArrayRef
            ]);
            continue;
        }
        let ids: &[u32] = if global_reduces_whole_column(call) {
            &[]
        } else {
            if zeros.len() < num_rows {
                zeros = vec![0u32; num_rows];
            }
            &zeros[..num_rows]
        };
        states.push(accumulate_call(call, ids, 1)?);
    }
    Ok(Partial {
        group_columns: Vec::new(),
        states,
    })
}

/// Whether `call`'s kernel answers a single group from the whole column, never reading the
/// per-row group ids.
///
/// This is the *complement* of the `num_groups == 1` short-circuits in [`sum_acc`],
/// [`minmax_acc`] and [`count_non_null`], and it has to stay their complement exactly: claiming a
/// pair that still scatters would hand it an empty slice and silently return a zero-row state.
/// `global_partial_agrees_with_the_scatter_path` holds the two in step over every aggregate, so a
/// short-circuit added or removed without updating this shows up as a failing test rather than a
/// wrong answer.
fn global_reduces_whole_column(call: &AggCall) -> bool {
    let Some(dt) = call.values.as_ref().map(|v| v.data_type()) else {
        return false;
    };
    match call.func {
        // `count_non_null` subtracts the null count; no type has a scatter-only path.
        AggFunc::Count => true,
        // Arrow's SIMD reduction, for the two types `sum_acc` routes to it. Decimal keeps the
        // exact `i128` scatter.
        AggFunc::Sum => matches!(dt, DataType::Int64 | DataType::Float64),
        // A `Mean` over `Float64` reduces with arrow's SIMD kernel; over `Int64` it reduces
        // into the exact 128-bit accumulator `sum_acc`'s `mean_int` arm supplies. Both are
        // whole-column. A decimal `Mean` was widened to `Float64` by `widen_mean_inputs`
        // before this runs, so it arrives here as the float case.
        AggFunc::Mean => matches!(dt, DataType::Float64 | DataType::Int64),
        // Int64 and Decimal128 only: a float `MIN`/`MAX` must keep the scatter, whose comparator
        // ranks NaN the way the engine's float identity says and arrow's kernel does not.
        AggFunc::Min | AggFunc::Max => matches!(dt, DataType::Int64 | DataType::Decimal128(_, _)),
        _ => false,
    }
}

pub(crate) fn sum_acc(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
    func: AggFunc,
) -> Result<ArrayRef, RuntimeError> {
    // An integer `Mean` sums into a 128-bit accumulator rather than the `i64` one `Sum`
    // keeps, because an `i64` running sum overflows on ordinary columns (IDs, nanosecond
    // timestamps, cents) where the *mean* is perfectly ordinary. It reads the `Int64` column
    // directly: the state is wider than the input, but nothing has to materialize a wider
    // copy of the input to say so. See `super::MEAN_INT_ACCUMULATOR`.
    if func == AggFunc::Mean && matches!(values.data_type(), DataType::Int64) {
        return mean_sum_i128(values.as_primitive::<Int64Type>(), group_ids, num_groups);
    }
    // Global-sum fast path: when there is a single group every row maps to it, so the
    // group sum equals the whole-column sum — arrow's SIMD reduction kernels beat the
    // scalar scatter loop below (the dominant cost of a global `SUM`/`COUNT`-derived
    // aggregate, and of every distributed `combine` that folds a handful of partials).
    // Null-only / empty input yields a null sum (SQL semantics), matching `masked_*`.
    if num_groups == 1 {
        match values.data_type() {
            DataType::Int64 => {
                let s = arrow::compute::sum_checked(values.as_primitive::<Int64Type>())
                    .map_err(|_| RuntimeError::SumOverflow)?;
                return Ok(Arc::new(masked_i64(
                    vec![s.unwrap_or(0)],
                    vec![s.is_some()],
                )));
            }
            DataType::Float64 => {
                let s = arrow::compute::sum(values.as_primitive::<Float64Type>());
                return Ok(Arc::new(masked_f64(
                    vec![s.unwrap_or(0.0)],
                    vec![s.is_some()],
                )));
            }
            // Decimal keeps the scatter loop below (exact i128 accumulation, scale-aware).
            _ => {}
        }
    }
    match values.data_type() {
        DataType::Int64 => {
            let arr = values.as_primitive::<Int64Type>();
            let mut sums = vec![0i64; num_groups];
            // Checked add throughout: a silent i64 wrap would be a wrong answer. (DuckDB
            // promotes BIGINT sums to 128-bit; we error rather than corrupt until that
            // wider-output promotion lands.)
            if arr.null_count() == 0 {
                // No-null fast path: gather straight from the values slice, skipping the
                // per-row validity branch and the per-row `valid` write (every group is
                // non-empty and all-valid) — mirrors the Float64 path below.
                for (&g, &v) in group_ids.iter().zip(arr.values()) {
                    let slot = &mut sums[g as usize];
                    // `checked_add` with the error built **only on overflow**. `ok_or` takes its argument by
                    // value, so it constructs a `RuntimeError` on every row and then drops it — and that enum is
                    // as wide as its widest variant (several carry `String`s), so the per-row cost is a wide
                    // stack write plus drop glue rather than a branch. It measured **11.4% of a 100-group
                    // `SUM` over 10M rows**, under `core::ptr::drop_glue::<RuntimeError>`, which is where it
                    // hides: the symbol names the error type, not the aggregate.
                    match slot.checked_add(v) {
                        Some(n) => *slot = n,
                        None => return Err(RuntimeError::SumOverflow),
                    }
                }
                return Ok(Arc::new(masked_i64(sums, vec![true; num_groups])));
            }
            let mut valid = vec![false; num_groups];
            for (i, &g) in group_ids.iter().enumerate() {
                if arr.is_valid(i) {
                    let slot = &mut sums[g as usize];
                    match slot.checked_add(arr.value(i)) {
                        Some(n) => *slot = n,
                        None => return Err(RuntimeError::SumOverflow),
                    }
                    valid[g as usize] = true;
                }
            }
            Ok(Arc::new(masked_i64(sums, valid)))
        }
        DataType::Float64 => {
            let arr = values.as_primitive::<Float64Type>();
            let mut sums = vec![0f64; num_groups];
            if arr.null_count() == 0 {
                // No-null fast path: gather straight from the values slice, skipping
                // both the per-row validity branch *and* the per-row `valid` write —
                // every group is non-empty (it exists because a row mapped to it) and
                // has only non-null values, so all groups are valid. Removing the 6M
                // redundant bool writes (per aggregate) is the dominant SUM/AVG path.
                for (&g, &v) in group_ids.iter().zip(arr.values()) {
                    sums[g as usize] += v;
                }
                return Ok(Arc::new(masked_f64(sums, vec![true; num_groups])));
            }
            let mut valid = vec![false; num_groups];
            for (i, &g) in group_ids.iter().enumerate() {
                if arr.is_valid(i) {
                    sums[g as usize] += arr.value(i);
                    valid[g as usize] = true;
                }
            }
            Ok(Arc::new(masked_f64(sums, valid)))
        }
        // Decimal sums accumulate in i128 (scale preserved). `checked_add` so a sum past
        // i128's range errors instead of silently wrapping to a negative value, mirroring
        // the i64 SUM path (DuckDB raises on decimal overflow).
        DataType::Decimal128(p, s) => {
            let arr = values.as_primitive::<Decimal128Type>();
            let mut sums = vec![0i128; num_groups];
            // **No no-null arm here, and that is a measurement rather than an omission.** The
            // `Int64` and `Float64` arms above each keep one, so the symmetry is tempting; it
            // was built (both here and as a `SumDecimalNoNull` in `fused`) and it moved
            // nothing. On 10 M rows into 100 groups, four fused decimal sums ran 12.7 / 13.0 /
            // 15.2 ms across three runs of the *same* binary, against 14.7 with the arm
            // removed — the effect is smaller than the box's own spread — and TPC-H q1, which
            // is this exact shape over `l_extendedprice`, read 16.8 ms either way. A decimal
            // column is 16 bytes a row, so this loop is bound by the load, not by the validity
            // branch the arm deletes. See `benchmarks/BENCHMARK_RESULTS.md`.
            let mut valid = vec![false; num_groups];
            for (i, &g) in group_ids.iter().enumerate() {
                if arr.is_valid(i) {
                    let slot = &mut sums[g as usize];
                    *slot = slot
                        .checked_add(arr.value(i))
                        .ok_or(RuntimeError::SumOverflow)?;
                    valid[g as usize] = true;
                }
            }
            Ok(masked_decimal(sums, valid, *p, *s)?)
        }
        other => Err(RuntimeError::UnsupportedAggregate {
            func: func.name().to_string(),
            dtype: other.to_string(),
        }),
    }
}

/// Sum an `Int64` column into the exact 128-bit accumulator an integer `Mean` finalizes from.
///
/// **This never overflows, and therefore never checks.** The widest sum of `n` `i64` values is
/// `n · 2^63`, and `i128` holds that for every `n` a machine can address (`n < 2^64`). That is
/// what lets the no-null arm be a bare `+=` where the `Decimal128` scatter this replaced needed
/// a `checked_add` per row — and why an integer `AVG` can read its `Int64` input directly
/// instead of being handed a widened copy of it.
///
/// The state it returns is byte-identical to what casting the column to
/// [`super::MEAN_INT_ACCUMULATOR`] and running the decimal scatter produced, so `combine`,
/// `finalize_mean`, spill and the distributed reduce are all unchanged.
fn mean_sum_i128(
    arr: &Int64Array,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<ArrayRef, RuntimeError> {
    let DataType::Decimal128(precision, scale) = super::MEAN_INT_ACCUMULATOR else {
        unreachable!("MEAN_INT_ACCUMULATOR is a Decimal128 by construction")
    };
    // Whole-column reduction: with one group every row maps to it, so the group sum is the
    // column sum and the group ids are never read. This arm is the one
    // `global_reduces_whole_column` claims for `Mean`/`Int64`, and must stay its complement.
    if num_groups == 1 {
        let mut total: i128 = 0;
        if arr.null_count() == 0 {
            for &v in arr.values() {
                total += v as i128;
            }
        } else {
            for i in 0..arr.len() {
                if arr.is_valid(i) {
                    total += arr.value(i) as i128;
                }
            }
        }
        // Null-only / empty input yields a null sum (SQL semantics), as `masked_*` does.
        let any = arr.len() > arr.null_count();
        return masked_decimal(vec![total], vec![any], precision, scale);
    }
    let mut sums = vec![0i128; num_groups];
    if arr.null_count() == 0 {
        // No-null fast path: no per-row validity branch and no per-row `valid` write — every
        // group is non-empty (it exists because a row mapped to it) and all its values are
        // non-null, so every group is valid.
        for (&g, &v) in group_ids.iter().zip(arr.values()) {
            sums[g as usize] += v as i128;
        }
        return masked_decimal(sums, vec![true; num_groups], precision, scale);
    }
    let mut valid = vec![false; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if arr.is_valid(i) {
            sums[g as usize] += arr.value(i) as i128;
            valid[g as usize] = true;
        }
    }
    masked_decimal(sums, valid, precision, scale)
}

/// Build a masked `Decimal128Array` with the given precision/scale.
pub(crate) fn masked_decimal(
    values: Vec<i128>,
    valid: Vec<bool>,
    precision: u8,
    scale: i8,
) -> Result<ArrayRef, RuntimeError> {
    let arr: Decimal128Array = values
        .into_iter()
        .zip(valid)
        .map(|(v, ok)| ok.then_some(v))
        .collect();
    let arr = arr
        .with_precision_and_scale(precision, scale)
        .map_err(|e| RuntimeError::UnsupportedAggregate {
            func: "decimal".to_string(),
            dtype: e.to_string(),
        })?;
    Ok(Arc::new(arr))
}

pub(crate) fn minmax_acc(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
    is_min: bool,
    func: AggFunc,
) -> Result<ArrayRef, RuntimeError> {
    // Global min/max fast path, mirroring `sum_acc` above: with a single group every row maps
    // to it, so the group extreme *is* the whole-column extreme and arrow's SIMD reduction
    // beats the scalar scatter loop. Null-only / empty input yields null, as SQL requires and
    // as `masked_i64` encodes.
    //
    // Measured on 20M `Int64` rows behind a filter (an unfiltered `MIN`/`MAX` never reaches
    // here — it is answered from the source's exact column statistics): **~24-28 ms → ~4.3 ms,
    // roughly 6x**. The scatter loop it replaces carries a per-row bounds-checked index and a
    // per-row validity branch that together defeat vectorization; arrow's reduction has neither.
    //
    // **Int64 only, deliberately — do not extend this to Float64.** Arrow's `min`/`max` compare
    // with raw IEEE `<`/`>`, which is false against NaN, so a NaN can never win and `max()`
    // silently skips it. That is precisely the bug the Float64 arm below documents and fixes
    // with `crate::keys::float_total_cmp`; routing floats through arrow here would reintroduce
    // it and disagree with both our own ORDER BY (NaN sorts last, i.e. greatest) and DuckDB.
    // Integers have no such value, so they are safe.
    if num_groups == 1 {
        match values.data_type() {
            DataType::Int64 => {
                let arr = values.as_primitive::<Int64Type>();
                let e = if is_min {
                    arrow::compute::min(arr)
                } else {
                    arrow::compute::max(arr)
                };
                return Ok(Arc::new(masked_i64(
                    vec![e.unwrap_or(0)],
                    vec![e.is_some()],
                )));
            }
            // Decimal128 joins on the same argument: within one column precision and scale are
            // fixed, so ordering the raw `i128` payloads *is* ordering the decimal values —
            // which is exactly what the scatter arm below does, one row at a time.
            DataType::Decimal128(p, s) => {
                let arr = values.as_primitive::<Decimal128Type>();
                let e = if is_min {
                    arrow::compute::min(arr)
                } else {
                    arrow::compute::max(arr)
                };
                return masked_decimal(vec![e.unwrap_or(0)], vec![e.is_some()], *p, *s);
            }
            _ => {}
        }
    }
    match values.data_type() {
        DataType::Int64 => {
            let arr = values.as_primitive::<Int64Type>();
            let mut cur = vec![0i64; num_groups];
            let mut valid = vec![false; num_groups];
            for (i, &g) in group_ids.iter().enumerate() {
                if arr.is_valid(i) {
                    let (g, v) = (g as usize, arr.value(i));
                    if !valid[g] || (is_min && v < cur[g]) || (!is_min && v > cur[g]) {
                        cur[g] = v;
                        valid[g] = true;
                    }
                }
            }
            Ok(Arc::new(masked_i64(cur, valid)))
        }
        DataType::Float64 => {
            let arr = values.as_primitive::<Float64Type>();
            let mut cur = vec![0f64; num_groups];
            let mut valid = vec![false; num_groups];
            for (i, &g) in group_ids.iter().enumerate() {
                if arr.is_valid(i) {
                    let (g, v) = (g as usize, arr.value(i));
                    // NOT `v < cur` / `v > cur`: raw IEEE comparison is false against NaN, so a
                    // NaN could never win and `max()` silently skipped it — disagreeing with our
                    // own ORDER BY (which sorts NaN last, i.e. greatest) and with DuckDB. The
                    // total order lives in `crate::keys` with the rest of the float semantics.
                    let ord = crate::keys::float_total_cmp(v, cur[g]);
                    let wins = if is_min {
                        ord == std::cmp::Ordering::Less
                    } else {
                        ord == std::cmp::Ordering::Greater
                    };
                    if !valid[g] || wins {
                        cur[g] = v;
                        valid[g] = true;
                    }
                }
            }
            Ok(Arc::new(masked_f64(cur, valid)))
        }
        DataType::Decimal128(p, s) => {
            let arr = values.as_primitive::<Decimal128Type>();
            let mut cur = vec![0i128; num_groups];
            let mut valid = vec![false; num_groups];
            for (i, &g) in group_ids.iter().enumerate() {
                if arr.is_valid(i) {
                    let (g, v) = (g as usize, arr.value(i));
                    if !valid[g] || (is_min && v < cur[g]) || (!is_min && v > cur[g]) {
                        cur[g] = v;
                        valid[g] = true;
                    }
                }
            }
            masked_decimal(cur, valid, *p, *s)
        }
        // Byte-ordered min/max over string and binary columns. Rust's `<`/`>` on `&str` and
        // `&[u8]` is lexicographic by byte, which is exactly DuckDB's default (binary) collation,
        // so `min`/`max` agree on ordering, NULL-skipping, empty strings, and unicode. The four
        // arms cover the 32- and 64-bit offset variants of both `Utf8` and `Binary`; without the
        // `LargeUtf8`/`Binary`/`LargeBinary` arms these raised "not supported for column type",
        // while DuckDB happily computes them.
        DataType::Utf8 => byte_minmax(
            values.as_string::<i32>(),
            group_ids,
            num_groups,
            is_min,
            |cur| {
                Arc::new(
                    cur.into_iter()
                        .map(bytes_to_string)
                        .collect::<StringArray>(),
                )
            },
        ),
        DataType::LargeUtf8 => byte_minmax(
            values.as_string::<i64>(),
            group_ids,
            num_groups,
            is_min,
            |cur| {
                Arc::new(
                    cur.into_iter()
                        .map(bytes_to_string)
                        .collect::<LargeStringArray>(),
                )
            },
        ),
        DataType::Binary => byte_minmax(
            values.as_binary::<i32>(),
            group_ids,
            num_groups,
            is_min,
            |cur| Arc::new(BinaryArray::from_iter(cur)),
        ),
        DataType::LargeBinary => byte_minmax(
            values.as_binary::<i64>(),
            group_ids,
            num_groups,
            is_min,
            |cur| Arc::new(LargeBinaryArray::from_iter(cur)),
        ),
        // SQL orders booleans `false < true`, so a group's minimum is the AND of its values
        // and its maximum is the OR — the very folds `bool_and`/`bool_or` already perform
        // (which is how DuckDB defines them too). Delegating rather than restating the fold
        // keeps one implementation of the boolean reduction.
        //
        // This arm is not a new capability so much as the closing of a gap that had already
        // become a *wrong answer*: a boolean column's footer records an exact min/max, so
        // `min(flag)` over a Parquet scan was answered `false` from metadata while the same
        // query over the same rows in memory raised "aggregate min is not supported for
        // column type Boolean". A metadata shortcut that can answer what the engine cannot
        // is not a shortcut; it is a second, disagreeing implementation.
        DataType::Boolean => bool_acc(values, group_ids, num_groups, is_min, func),
        // Temporal min/max. Date/Time/Timestamp/Duration are stored as integers whose
        // natural (chronological) order IS their integer order — a later instant has a larger
        // underlying value in the same unit, and a tz-aware Timestamp's i64 is the UTC instant,
        // so comparison is correct regardless of the display zone. We compare on the cast-to-i64
        // representation but `take` the *winner from the original typed array*, so the result
        // keeps the exact unit and timezone (a cast of the reduced integer back could not).
        //
        // Without this arm `min`/`max` over a Date/Timestamp column raised "not supported for
        // column type Date32" while the Parquet footer answered the same query from metadata —
        // the very metadata-vs-engine disagreement the Boolean arm above was added to close.
        DataType::Date32
        | DataType::Date64
        | DataType::Time32(_)
        | DataType::Time64(_)
        | DataType::Timestamp(_, _)
        | DataType::Duration(_) => temporal_minmax(values, group_ids, num_groups, is_min),
        other => Err(RuntimeError::UnsupportedAggregate {
            func: func.name().to_string(),
            dtype: other.to_string(),
        }),
    }
}

/// Per-group min/max over a temporal column via its underlying `i64` order, returning the
/// winning rows `take`n from the original array (so unit/timezone are preserved exactly).
/// Null-skipping; an empty/all-null group yields null. Associative — the partial state is
/// itself a temporal column, so merging re-enters this same reducer.
fn temporal_minmax(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
    is_min: bool,
) -> Result<ArrayRef, RuntimeError> {
    let ints = arrow::compute::cast(values, &DataType::Int64)?;
    let arr = ints.as_primitive::<Int64Type>();
    let mut cur = vec![0i64; num_groups];
    let mut best = vec![0u32; num_groups];
    let mut valid = vec![false; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if arr.is_valid(i) {
            let (g, v) = (g as usize, arr.value(i));
            if !valid[g] || (is_min && v < cur[g]) || (!is_min && v > cur[g]) {
                cur[g] = v;
                best[g] = i as u32;
                valid[g] = true;
            }
        }
    }
    // A null index makes `take` emit null, so empty/all-null groups become null.
    let idx: UInt32Array = best
        .into_iter()
        .zip(valid)
        .map(|(i, ok)| ok.then_some(i))
        .collect();
    Ok(take(values.as_ref(), &idx, None)?)
}

/// Per-group byte-lexicographic min/max over a string/binary column. Stores each group's
/// winning bytes and lets `build` re-wrap them as the concrete output array (string or
/// binary, 32- or 64-bit offsets). Null-skipping; an all-null group stays `None` (→ null).
/// The comparison is `&[u8]` order — identical to DuckDB's default binary collation, and to
/// `&str` order for valid UTF-8.
fn byte_minmax<T, F>(
    arr: &arrow::array::GenericByteArray<T>,
    group_ids: &[u32],
    num_groups: usize,
    is_min: bool,
    build: F,
) -> Result<ArrayRef, RuntimeError>
where
    T: arrow::array::types::ByteArrayType,
    T::Native: AsRef<[u8]>,
    F: FnOnce(Vec<Option<Vec<u8>>>) -> ArrayRef,
{
    let mut cur: Vec<Option<Vec<u8>>> = vec![None; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if arr.is_valid(i) {
            let g = g as usize;
            let v: &[u8] = arr.value(i).as_ref();
            // Overwrite the held bytes in place rather than allocating a fresh `Vec` for each
            // new extreme. On input that arrives already ordered — a sorted file, a column
            // read back from a sorted lakehouse table — every row is a new extreme, so the
            // old form allocated once per *row* instead of once per group.
            match &mut cur[g] {
                slot @ None => *slot = Some(v.to_vec()),
                Some(c) => {
                    if (is_min && v < c.as_slice()) || (!is_min && v > c.as_slice()) {
                        c.clear();
                        c.extend_from_slice(v);
                    }
                }
            }
        }
    }
    Ok(build(cur))
}

/// Reconstruct a `String` from bytes that came out of a `Utf8`/`LargeUtf8` array (so they are
/// valid UTF-8 by construction — the `expect` guards a genuine invariant, never user data).
fn bytes_to_string(b: Option<Vec<u8>>) -> Option<String> {
    b.map(|b| String::from_utf8(b).expect("bytes from a Utf8 array are valid UTF-8"))
}

/// Boolean reduction per group: `bool_and` (logical AND of non-null values) or
/// `bool_or` (logical OR). Nulls are ignored; a group with no non-null value
/// yields null. Associative and idempotent over a single partial, so the same
/// function merges already-partial boolean state — AND/OR commute and associate.
pub(crate) fn bool_acc(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
    is_and: bool,
    func: AggFunc,
) -> Result<ArrayRef, RuntimeError> {
    let arr = values
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| RuntimeError::UnsupportedAggregate {
            func: func.name().to_string(),
            dtype: values.data_type().to_string(),
        })?;
    let mut cur = vec![false; num_groups];
    let mut valid = vec![false; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if arr.is_valid(i) {
            let (g, v) = (g as usize, arr.value(i));
            if !valid[g] {
                cur[g] = v;
                valid[g] = true;
            } else if is_and {
                cur[g] = cur[g] && v;
            } else {
                cur[g] = cur[g] || v;
            }
        }
    }
    let out: BooleanArray = cur
        .into_iter()
        .zip(valid)
        .map(|(v, ok)| ok.then_some(v))
        .collect();
    Ok(Arc::new(out))
}

/// Fold each group's non-null Int64 values with a bitwise op (`bit_and`/`bit_or`/
/// `bit_xor`). Null-skipping; an all-null group yields null. The op is associative
/// and commutative, so the same fold merges already-partial state across partitions.
pub(crate) fn bitfold_acc(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
    func: AggFunc,
) -> Result<ArrayRef, RuntimeError> {
    let arr = values
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| RuntimeError::UnsupportedAggregate {
            func: func.name().to_string(),
            dtype: values.data_type().to_string(),
        })?;
    let mut cur = vec![0i64; num_groups];
    let mut valid = vec![false; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if arr.is_valid(i) {
            let (g, v) = (g as usize, arr.value(i));
            if !valid[g] {
                cur[g] = v;
                valid[g] = true;
            } else {
                cur[g] = match func {
                    AggFunc::BitAnd => cur[g] & v,
                    AggFunc::BitOr => cur[g] | v,
                    AggFunc::BitXor => cur[g] ^ v,
                    _ => unreachable!("bitfold_acc on non-bitwise func"),
                };
            }
        }
    }
    Ok(Arc::new(masked_i64(cur, valid)))
}

/// Product of each group's non-null values as Float64 (DuckDB `product` returns
/// DOUBLE; f64 avoids the silent integer overflow a wrapping i64 product would hit).
/// Null-skipping; an all-null group yields null. Associative — the same fold merges
/// the (already-Float64) partial state.
pub(crate) fn product_acc(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<ArrayRef, RuntimeError> {
    let f = arrow::compute::cast(values, &DataType::Float64)?;
    let arr = f.as_primitive::<Float64Type>();
    let mut cur = vec![1f64; num_groups];
    let mut valid = vec![false; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if arr.is_valid(i) {
            let g = g as usize;
            cur[g] *= arr.value(i);
            valid[g] = true;
        }
    }
    Ok(Arc::new(masked_f64(cur, valid)))
}

/// `kahan_sum`/`fsum` — **compensated** summation, as a 2-column state `(sum, c)`.
///
/// A plain float sum loses the low bits of every addend whose magnitude is far below the
/// running total, so summing a long column of small values against a large one drifts:
/// the classic `1e16 + 1.0 + 1.0 …` loses every 1. Neumaier's variant tracks that lost
/// part in `c` and adds it back at the end, which is exact for the cases that matter and
/// never worse than the naive sum.
///
/// Neumaier rather than plain Kahan because it is also correct when the *addend* is the
/// larger of the two, which is the case a running total meets on its first rows.
///
/// Mergeable: two `(sum, c)` states combine by compensated-adding the sums and adding the
/// compensations, which is associative and commutative up to the same rounding the
/// single-node path performs — so a partitioned run and a single-node one agree.
pub(crate) fn kahan_acc(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    let f = arrow::compute::cast(values, &DataType::Float64)?;
    let arr = f.as_primitive::<Float64Type>();
    let mut sums = vec![0f64; num_groups];
    let mut comps = vec![0f64; num_groups];
    let mut valid = vec![false; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if arr.is_valid(i) {
            let g = g as usize;
            neumaier_add(&mut sums[g], &mut comps[g], arr.value(i));
            valid[g] = true;
        }
    }
    Ok(vec![
        Arc::new(masked_f64(sums, valid.clone())),
        Arc::new(masked_f64(comps, valid)),
    ])
}

/// Add `value` into the running `(sum, compensation)` pair (Neumaier).
pub(crate) fn neumaier_add(sum: &mut f64, compensation: &mut f64, value: f64) {
    let t = *sum + value;
    // The lost low bits live in whichever operand was smaller in magnitude.
    *compensation += if sum.abs() >= value.abs() {
        (*sum - t) + value
    } else {
        (value - t) + *sum
    };
    *sum = t;
}

pub(crate) fn masked_i64(vals: Vec<i64>, valid: Vec<bool>) -> Int64Array {
    Int64Array::from_iter(vals.into_iter().zip(valid).map(|(v, ok)| ok.then_some(v)))
}

pub(crate) fn masked_f64(vals: Vec<f64>, valid: Vec<bool>) -> Float64Array {
    Float64Array::from_iter(vals.into_iter().zip(valid).map(|(v, ok)| ok.then_some(v)))
}

/// Concatenate the partials' columns for a `combine`.
///
/// Through [`crate::gather::concat_columns`], not arrow's `concat` directly — on a
/// high-cardinality string key this is the copy that dominates the whole combine.
pub(crate) fn concat_col<'a>(
    it: impl Iterator<Item = &'a ArrayRef>,
) -> Result<ArrayRef, RuntimeError> {
    let cols: Vec<&dyn Array> = it.map(|a| a.as_ref()).collect();
    crate::gather::concat_columns(&cols)
}

pub(crate) fn require(values: Option<&ArrayRef>, func: AggFunc) -> Result<&ArrayRef, RuntimeError> {
    values.ok_or_else(|| RuntimeError::MissingAggregateInput {
        func: func.name().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int64_sum_overflow_errors_instead_of_wrapping() {
        // i64::MAX + 1 in one group must error, not silently wrap to i64::MIN.
        let values: ArrayRef = Arc::new(Int64Array::from(vec![i64::MAX, 1]));
        let group_ids = [0u32, 0];
        let r = sum_acc(&values, &group_ids, 1, AggFunc::Sum);
        assert!(matches!(r, Err(RuntimeError::SumOverflow)), "got {r:?}");
    }

    #[test]
    fn int64_sum_in_range_is_unaffected() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30]));
        let group_ids = [0u32, 0, 0];
        let out = sum_acc(&values, &group_ids, 1, AggFunc::Sum).unwrap();
        assert_eq!(out.as_primitive::<Int64Type>().value(0), 60);
    }

    /// A null value still suppresses nothing but itself, and a group whose every value is
    /// null is null rather than zero — the property the fast path must not reach for.
    #[test]
    fn decimal_sum_with_nulls_keeps_sql_semantics() {
        let values: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(500i128), None, None])
                .with_precision_and_scale(7, 2)
                .unwrap(),
        );
        let out = sum_acc(&values, &[0u32, 0, 1], 2, AggFunc::Sum).unwrap();
        let out = out.as_primitive::<Decimal128Type>();
        assert_eq!(out.value(0), 500);
        assert!(out.is_null(1), "an all-null group sums to NULL, not 0");
    }

    /// `SUM` over no rows is NULL. A keyless aggregate over an empty input reaches this with
    /// `num_groups == 1` and an empty id slice, and the no-null fast path — where every group
    /// is valid because a row made it — must not claim that case.
    #[test]
    fn decimal_sum_over_no_rows_is_null() {
        let values: ArrayRef = Arc::new(
            Decimal128Array::from(Vec::<i128>::new())
                .with_precision_and_scale(7, 2)
                .unwrap(),
        );
        let out = sum_acc(&values, &[], 1, AggFunc::Sum).unwrap();
        assert!(out.as_primitive::<Decimal128Type>().is_null(0));
    }

    #[test]
    fn product_skips_nulls_and_uses_f64() {
        // [2, 3, null, 4] in one group → 24.0; the null is skipped, no overflow.
        let values: ArrayRef = Arc::new(Int64Array::from(vec![Some(2), Some(3), None, Some(4)]));
        let group_ids = [0u32, 0, 0, 0];
        let out = product_acc(&values, &group_ids, 1).unwrap();
        assert_eq!(out.as_primitive::<Float64Type>().value(0), 24.0);
    }

    #[test]
    fn minmax_over_binary_and_large_strings() {
        use arrow::array::{BinaryArray, LargeBinaryArray, LargeStringArray};
        // Two groups over each byte type; nulls are skipped, ordering is bytewise.
        let g = [0u32, 0, 1, 0];

        // Binary: g0 = {b"foo", b"bar", null}; g1 = {b"zzz"}.
        let bin: ArrayRef = Arc::new(BinaryArray::from_opt_vec(vec![
            Some(b"foo"),
            Some(b"bar"),
            Some(b"zzz"),
            None,
        ]));
        let min = minmax_acc(&bin, &g, 2, true, AggFunc::Min).unwrap();
        let max = minmax_acc(&bin, &g, 2, false, AggFunc::Max).unwrap();
        let min = min.as_any().downcast_ref::<BinaryArray>().unwrap();
        let max = max.as_any().downcast_ref::<BinaryArray>().unwrap();
        assert_eq!(min.value(0), b"bar");
        assert_eq!(max.value(0), b"foo");
        assert_eq!(min.value(1), b"zzz");

        // LargeUtf8 min/max.
        let ls: ArrayRef = Arc::new(LargeStringArray::from(vec![
            Some("delta"),
            Some("alpha"),
            Some("omega"),
            None,
        ]));
        let lmin = minmax_acc(&ls, &g, 2, true, AggFunc::Min).unwrap();
        let lmin = lmin.as_any().downcast_ref::<LargeStringArray>().unwrap();
        assert_eq!(lmin.value(0), "alpha");

        // LargeBinary max.
        let lb: ArrayRef = Arc::new(LargeBinaryArray::from_opt_vec(vec![
            Some(b"aa"),
            Some(b"ab"),
            Some(b"zz"),
            None,
        ]));
        let lbmax = minmax_acc(&lb, &g, 2, false, AggFunc::Max).unwrap();
        let lbmax = lbmax.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
        assert_eq!(lbmax.value(0), b"ab");
    }

    #[test]
    fn temporal_minmax_over_date_and_timestamp() {
        use arrow::array::{Date32Array, TimestampMicrosecondArray};
        // Date32 (days since epoch), two groups, a null skipped.
        let g = [0u32, 0, 1, 0];
        let dates: ArrayRef = Arc::new(Date32Array::from(vec![
            Some(19_723), // 2024-01-03
            Some(19_721), // 2024-01-01
            Some(19_725), // 2024-01-05 (group 1)
            None,
        ]));
        let min = minmax_acc(&dates, &g, 2, true, AggFunc::Min).unwrap();
        let max = minmax_acc(&dates, &g, 2, false, AggFunc::Max).unwrap();
        // Type preserved (not cast to Int64).
        assert_eq!(min.data_type(), &DataType::Date32);
        let min = min.as_any().downcast_ref::<Date32Array>().unwrap();
        let max = max.as_any().downcast_ref::<Date32Array>().unwrap();
        assert_eq!(min.value(0), 19_721);
        assert_eq!(max.value(0), 19_723);
        assert_eq!(min.value(1), 19_725);

        // Timestamp with a timezone: the underlying i64 (UTC instant) drives the order, and the
        // timezone is carried through by `take`ing the original array.
        let tz_ty = DataType::Timestamp(
            arrow::datatypes::TimeUnit::Microsecond,
            Some("+05:00".into()),
        );
        let ts: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![30i64, 10, 99, 20])
                .with_timezone_opt(Some("+05:00")),
        );
        let tmin = minmax_acc(&ts, &[0u32, 0, 0, 0], 1, true, AggFunc::Min).unwrap();
        assert_eq!(tmin.data_type(), &tz_ty);
        let tmin = tmin
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        assert_eq!(tmin.value(0), 10);
    }

    #[test]
    fn global_int_minmax_fastpath_matches_the_scatter_loop() {
        // The `num_groups == 1` arm routes Int64 min/max through arrow's SIMD reduction instead
        // of the scatter loop. It must agree with that loop on every edge the loop handles, so
        // each case is checked against the two-group form of the same data, which cannot take
        // the fast path.
        let cases: Vec<Vec<Option<i64>>> = vec![
            vec![Some(5), Some(-3), Some(9), Some(0)], // plain
            vec![Some(5), None, Some(-3), None],       // interleaved nulls
            vec![None, None],                          // all null -> null
            vec![],                                    // empty -> null
            vec![Some(7)],                             // single row
            vec![Some(i64::MIN), Some(i64::MAX)],      // range ends
            vec![Some(4), Some(4), Some(4)],           // all equal
        ];
        for values in cases {
            let arr: ArrayRef = Arc::new(Int64Array::from(values.clone()));
            let n = arr.len();
            for is_min in [true, false] {
                let func = if is_min { AggFunc::Min } else { AggFunc::Max };
                // Fast path: one group.
                let fast = minmax_acc(&arr, &vec![0u32; n], 1, is_min, func).unwrap();
                // Reference: two groups, all rows in group 0, so group 0 holds the same answer
                // while `num_groups != 1` keeps it on the scalar loop.
                let slow = minmax_acc(&arr, &vec![0u32; n], 2, is_min, func).unwrap();
                let f = fast.as_any().downcast_ref::<Int64Array>().unwrap();
                let s = slow.as_any().downcast_ref::<Int64Array>().unwrap();
                assert_eq!(
                    f.is_null(0),
                    s.is_null(0),
                    "validity differs for {values:?} is_min={is_min}"
                );
                if !f.is_null(0) {
                    assert_eq!(
                        f.value(0),
                        s.value(0),
                        "value differs for {values:?} is_min={is_min}"
                    );
                }
            }

            // Decimal128 takes the same fast path and must agree with its own scatter arm,
            // including that precision and scale survive it.
            let dec: ArrayRef = Arc::new(
                values
                    .iter()
                    .map(|v| v.map(|x| x as i128))
                    .collect::<Decimal128Array>()
                    .with_precision_and_scale(20, 3)
                    .unwrap(),
            );
            for is_min in [true, false] {
                let func = if is_min { AggFunc::Min } else { AggFunc::Max };
                let fast = minmax_acc(&dec, &vec![0u32; n], 1, is_min, func).unwrap();
                let slow = minmax_acc(&dec, &vec![0u32; n], 2, is_min, func).unwrap();
                assert_eq!(fast.data_type(), &DataType::Decimal128(20, 3));
                let f = fast.as_any().downcast_ref::<Decimal128Array>().unwrap();
                let s = slow.as_any().downcast_ref::<Decimal128Array>().unwrap();
                assert_eq!(f.is_null(0), s.is_null(0), "decimal validity {values:?}");
                if !f.is_null(0) {
                    assert_eq!(f.value(0), s.value(0), "decimal value {values:?}");
                }
            }
        }
    }

    #[test]
    fn temporal_minmax_is_mergeable_across_partitions() {
        use crate::agg::{combine, finalize, group_aggregate, AggCall};
        use arrow::array::Date32Array;
        // The mergeable-algebra invariant: combine_finalize(partition(partial(x))) == single-node.
        let keys = |k: Vec<&str>| -> ArrayRef { Arc::new(StringArray::from(k)) };
        let dates = |d: Vec<Option<i32>>| -> ArrayRef { Arc::new(Date32Array::from(d)) };
        let funcs = [AggFunc::Min, AggFunc::Max];
        let call = |v: &ArrayRef| {
            vec![
                AggCall::new(AggFunc::Min, Some(v.clone())),
                AggCall::new(AggFunc::Max, Some(v.clone())),
            ]
        };

        // Whole input, single node.
        let k_all = keys(vec!["a", "a", "b", "a", "b"]);
        let v_all = dates(vec![Some(5), None, Some(9), Some(2), Some(7)]);
        let whole = group_aggregate(std::slice::from_ref(&k_all), &call(&v_all), 5).unwrap();

        // Split into two partitions, partial each, combine, finalize.
        let (k1, v1) = (
            keys(vec!["a", "a", "b"]),
            dates(vec![Some(5), None, Some(9)]),
        );
        let (k2, v2) = (keys(vec!["a", "b"]), dates(vec![Some(2), Some(7)]));
        let p1 = crate::agg::partial(std::slice::from_ref(&k1), &call(&v1), 3).unwrap();
        let p2 = crate::agg::partial(std::slice::from_ref(&k2), &call(&v2), 2).unwrap();
        let merged = combine(&[p1, p2], &funcs).unwrap();
        let dist_cols = finalize(&funcs, &merged).unwrap();

        // Align both by group key and compare min/max per group.
        let by_key =
            |gk: &ArrayRef, aggs: &[ArrayRef]| -> std::collections::HashMap<String, (i32, i32)> {
                let k = gk.as_string::<i32>();
                let mn = aggs[0].as_any().downcast_ref::<Date32Array>().unwrap();
                let mx = aggs[1].as_any().downcast_ref::<Date32Array>().unwrap();
                (0..k.len())
                    .map(|i| (k.value(i).to_string(), (mn.value(i), mx.value(i))))
                    .collect()
            };
        let single = by_key(&whole.group_columns[0], &whole.agg_columns);
        let dist = by_key(&merged.group_columns[0], &dist_cols);
        assert_eq!(
            single, dist,
            "temporal min/max must be partition-independent"
        );
        assert_eq!(single["a"], (2, 5));
        assert_eq!(single["b"], (7, 9));
    }

    #[test]
    fn bitfold_and_or_xor() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![6, 3, 5]));
        let g = [0u32, 0, 0];
        let and = bitfold_acc(&values, &g, 1, AggFunc::BitAnd).unwrap();
        let or = bitfold_acc(&values, &g, 1, AggFunc::BitOr).unwrap();
        let xor = bitfold_acc(&values, &g, 1, AggFunc::BitXor).unwrap();
        assert_eq!(and.as_primitive::<Int64Type>().value(0), 6 & 3 & 5);
        assert_eq!(or.as_primitive::<Int64Type>().value(0), 6 | 3 | 5);
        assert_eq!(xor.as_primitive::<Int64Type>().value(0), 6 ^ 3 ^ 5);
    }
}

#[cfg(test)]
mod global_partial_tests {
    use std::sync::Arc;

    use arrow::array::{BooleanArray, Float64Array, Int64Array, StringArray};

    use super::*;

    /// [`global_partial`] hands an empty group-id slice to every kernel
    /// [`global_reduces_whole_column`] claims, and a real buffer of zeros to the rest. This holds
    /// the claim to its consequence, aggregate by aggregate: the two must produce the identical
    /// partial state.
    ///
    /// The failure it exists to catch is silent and total. A kernel wrongly claimed here iterates
    /// an empty slice and returns a state for **zero rows** — not a wrong number, an absent one —
    /// and every keyless query using that aggregate answers null. So the coverage is the whole
    /// `AggFunc` surface rather than the handful currently claimed, because the risk is in what
    /// someone claims *next*.
    #[test]
    fn global_partial_agrees_with_the_scatter_path() {
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(3i64),
            None,
            Some(-7),
            Some(3),
            Some(9),
        ]));
        // Signed zero and NaN are in here deliberately: `MIN`/`MAX` over floats must keep the
        // scatter comparator, and this is what says so if it ever stops.
        let floats: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(1.5f64),
            None,
            Some(-0.0),
            Some(f64::NAN),
            Some(2.0),
        ]));
        let strings: ArrayRef = Arc::new(StringArray::from(vec![
            Some("b"),
            None,
            Some("a"),
            Some("b"),
            Some("c"),
        ]));
        let bools: ArrayRef = Arc::new(BooleanArray::from(vec![
            Some(true),
            None,
            Some(false),
            Some(true),
            Some(true),
        ]));
        let n = ints.len();

        let single = [
            AggFunc::CountStar,
            AggFunc::Count,
            AggFunc::CountDistinct,
            AggFunc::Sum,
            AggFunc::Min,
            AggFunc::Max,
            AggFunc::Mean,
            AggFunc::Var,
            AggFunc::Stddev,
            AggFunc::Median,
            AggFunc::Quantile(250),
            AggFunc::QuantileDisc(250),
            AggFunc::ApproxQuantile(500),
            AggFunc::ApproxCountDistinct,
            AggFunc::ListAgg,
            AggFunc::Mode,
            // The contiguity statistics ride `Median`'s state, so they must agree with the
            // scatter path exactly as it does — this list is what says so if they stop.
            AggFunc::NLength(500),
            AggFunc::LCount(500),
            AggFunc::AuN,
            AggFunc::Histogram,
            AggFunc::AnyValue,
            AggFunc::Entropy,
            AggFunc::Mad,
            AggFunc::Product,
            AggFunc::KahanSum,
            AggFunc::ApproxTopK(2),
            AggFunc::Skewness,
            AggFunc::Kurtosis,
            AggFunc::KurtosisPop,
            AggFunc::BitAnd,
            AggFunc::BitOr,
            AggFunc::BitXor,
        ];
        let mut cases: Vec<AggCall> = Vec::new();
        for func in single {
            for values in [&ints, &floats, &strings, &bools] {
                cases.push(AggCall::new(func, Some(Arc::clone(values))));
            }
        }
        for func in [AggFunc::BoolAnd, AggFunc::BoolOr] {
            cases.push(AggCall::new(func, Some(Arc::clone(&bools))));
        }
        for func in [
            AggFunc::ArgMin,
            AggFunc::ArgMax,
            AggFunc::CovarPop,
            AggFunc::CovarSamp,
            AggFunc::Corr,
        ] {
            cases.push(AggCall::with_key(
                func,
                Some(Arc::clone(&floats)),
                Some(Arc::clone(&ints)),
            ));
        }

        let zeros = vec![0u32; n];
        for call in &cases {
            let lazy = global_partial(std::slice::from_ref(call), n);
            let eager = accumulate_call(call, &zeros, 1);
            match (lazy, eager) {
                (Ok(lazy), Ok(eager)) => {
                    let lazy = &lazy.states[0];
                    assert_eq!(
                        lazy.len(),
                        eager.len(),
                        "{:?} over {:?}: state arity",
                        call.func,
                        call.values.as_ref().map(|v| v.data_type())
                    );
                    for (got, want) in lazy.iter().zip(&eager) {
                        assert_eq!(
                            got.as_ref(),
                            want.as_ref(),
                            "{:?} over {:?}",
                            call.func,
                            call.values.as_ref().map(|v| v.data_type())
                        );
                    }
                }
                // A function that rejects a type must reject it the same way on both paths;
                // what must never happen is one succeeding where the other errors.
                (Err(_), Err(_)) => {}
                (lazy, eager) => panic!(
                    "{:?} over {:?}: one path succeeded and the other did not ({}, {})",
                    call.func,
                    call.values.as_ref().map(|v| v.data_type()),
                    lazy.is_ok(),
                    eager.is_ok()
                ),
            }
        }
    }
}
