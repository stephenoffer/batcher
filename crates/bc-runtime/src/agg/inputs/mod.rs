//! Bringing an aggregate call's inputs to a type the accumulator kernels read.
//!
//! Every kernel in `accum`/`fused` downcasts to a concrete array, so a dictionary column,
//! an all-null `Null`-typed column and a narrow `Int32`/`Float32` all reach them as types
//! they reject outright. Each rewrite here is a *pre*-pass over the call list, applied
//! once before the global and grouped paths split, so both — and every fused, combine and
//! distributed step downstream — carry one input type.
//!
//! All three return `None` when nothing needed changing, so the common path allocates
//! nothing.

use std::sync::Arc;

use arrow::array::ArrayRef;
use arrow::datatypes::DataType;

use crate::agg::{AggCall, AggFunc};
use crate::error::RuntimeError;

/// Decode any dictionary-encoded value/ordering-key input to its plain value type,
/// returning a fresh call list only when some input was a dictionary (else `None`, so the
/// common non-dictionary path allocates nothing). Keeps the typed accumulator kernels
/// oblivious to dictionary encoding, mirroring the scalar `decode_dict` at the `Col` leaf.
pub(super) fn decode_dict_call_inputs(
    calls: &[AggCall],
) -> Result<Option<Vec<AggCall>>, RuntimeError> {
    let is_dict = |a: &Option<ArrayRef>| {
        matches!(
            a.as_ref().map(|x| x.data_type()),
            Some(DataType::Dictionary(..))
        )
    };
    if !calls.iter().any(|c| is_dict(&c.values) || is_dict(&c.key)) {
        return Ok(None);
    }
    let decode = |a: &Option<ArrayRef>| -> Result<Option<ArrayRef>, RuntimeError> {
        match a {
            Some(arr) => match arr.data_type() {
                DataType::Dictionary(_, v) => Ok(Some(arrow::compute::cast(arr, v)?)),
                _ => Ok(Some(arr.clone())),
            },
            None => Ok(None),
        }
    };
    let out = calls
        .iter()
        .map(|c| {
            Ok(AggCall {
                func: c.func,
                values: decode(&c.values)?,
                key: decode(&c.key)?,
            })
        })
        .collect::<Result<Vec<_>, RuntimeError>>()?;
    Ok(Some(out))
}

/// Whether `func`'s accumulator works in `f64` and so takes a DECIMAL input widened.
///
/// `SUM`, `MIN`/`MAX` and the bitwise family are deliberately absent: the first two are
/// exact or type-preserving on a decimal and widening them would *lose* what makes them
/// worth having, and the last cannot take one at all. Everything here already returns a
/// DOUBLE, so the widening costs nothing a decimal input could have kept — and without it
/// `STDDEV(price)` over a DECIMAL column, which DuckDB answers, was refused outright.
fn widens_decimal(func: AggFunc) -> bool {
    matches!(
        func,
        AggFunc::Mean
            | AggFunc::Var
            | AggFunc::Stddev
            | AggFunc::Median
            | AggFunc::Quantile(_)
            | AggFunc::QuantileDisc(_)
            | AggFunc::ApproxQuantile(_)
            | AggFunc::Mad
            | AggFunc::Product
            | AggFunc::Skewness
            | AggFunc::Kurtosis
            | AggFunc::KurtosisPop
            | AggFunc::KahanSum
            | AggFunc::Corr
            | AggFunc::CovarPop
            | AggFunc::CovarSamp
    )
}

/// Widen every `f64`-accumulating call's Decimal128/Decimal256 input to Float64. Returns a fresh call
/// list only when some widening happened (else `None`, so the common path allocates nothing).
///
/// **An integer `Mean` sums exactly, and is deliberately *not* widened here.** Accumulating
/// in Float64 is only lossless while every partial sum stays under 2^53, and an `i64` column
/// routinely is not: `AVG` over `[-1, 2^62, -2^62, 2, -3, 0]` returned `-1/6` (or `0`,
/// depending on the order the groups were reduced in) where the true mean is `-2/6`. The
/// values were exactly representable — the *running sum* was not, so each large addend
/// swallowed the small ones. That is a wrong answer, not a rounding difference, and it is
/// silent: a column of IDs, nanosecond timestamps or cents hits it at ordinary magnitudes.
/// DuckDB sums into a HUGEINT for exactly this reason. Batcher does too, but in the
/// accumulator rather than in the input — see [`super::MEAN_INT_ACCUMULATOR`]. `finalize_mean`
/// divides the exact sum by the count in f64, so only the final division rounds.
///
/// Decimal *is* widened to Float64: its sums are already exact in `sum_acc`, but they carry
/// the *input's* declared precision, which a sum can exceed — and `avg` returns DOUBLE in
/// DuckDB either way, so the promotion costs nothing a decimal AVG could have kept.
pub(super) fn widen_mean_inputs(calls: &[AggCall]) -> Result<Option<Vec<AggCall>>, RuntimeError> {
    // The accumulator each widened type takes, or `None` for a type that stays as it is.
    let target = |c: &AggCall| -> Option<DataType> {
        if !widens_decimal(c.func) {
            return None;
        }
        match c.values.as_ref()?.data_type() {
            DataType::Decimal128(_, _) | DataType::Decimal256(_, _) => Some(DataType::Float64),
            _ => None,
        }
    };
    if !calls.iter().any(|c| target(c).is_some()) {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(calls.len());
    for c in calls {
        let values = match target(c) {
            Some(dt) => Some(arrow::compute::cast(c.values.as_ref().unwrap(), &dt)?),
            None => c.values.clone(),
        };
        out.push(AggCall::with_key(c.func, values, c.key.clone()));
    }
    Ok(Some(out))
}

/// Rewrite every call's value/ordering input whose type `target` names a replacement for,
/// returning a fresh call list only when something changed (else `None`, so the common
/// path allocates nothing).
///
/// The one place a per-type input rewrite is expressed, so the rules below cannot drift on
/// how they walk the call list.
pub(super) fn map_call_inputs(
    calls: &[AggCall],
    target: impl Fn(&DataType) -> Option<DataType>,
) -> Result<Option<Vec<AggCall>>, RuntimeError> {
    let hit = |a: &Option<ArrayRef>| a.as_ref().is_some_and(|x| target(x.data_type()).is_some());
    if !calls.iter().any(|c| hit(&c.values) || hit(&c.key)) {
        return Ok(None);
    }
    let convert = |a: &Option<ArrayRef>| -> Result<Option<ArrayRef>, RuntimeError> {
        match a {
            Some(arr) => match target(arr.data_type()) {
                Some(dt) => Ok(Some(arrow::compute::cast(arr, &dt)?)),
                None => Ok(Some(Arc::clone(arr))),
            },
            None => Ok(None),
        }
    };
    let out = calls
        .iter()
        .map(|c| {
            Ok(AggCall::with_key(
                c.func,
                convert(&c.values)?,
                convert(&c.key)?,
            ))
        })
        .collect::<Result<Vec<_>, RuntimeError>>()?;
    Ok(Some(out))
}

/// Bring every call input to a width the typed accumulators read.
///
/// Two rules, one pass. **Narrow numerics** widen exactly as `bc_py::normalize_to` does at
/// the FFI boundary — but that only sees *scanned* columns, so a width introduced inside
/// the query (`SUM(CAST(x AS INTEGER))`, an Int32 from a format that carries one) reached
/// the accumulators as `Int32`/`Float32`, which they reject outright: *aggregate sum is
/// not supported for column type Int32*. The window path already widens the same way
/// (`window::coerce::widen_target`). **An all-null column** carries Arrow's `Null` type,
/// which every typed kernel also rejects — so `SUM`/`MIN`/`MAX` over one *errored* where
/// DuckDB returns NULL, and `COUNT` counted every row instead of 0; an all-null `Int64`
/// flows through correctly and the result type is immaterial because the value is null.
///
/// `MIN`/`MAX` widen too, returning BIGINT where the input was INTEGER — the same
/// promotion the FFI boundary applies to a scanned narrow column, so the two agree.
pub(super) fn normalize_call_inputs(
    calls: &[AggCall],
) -> Result<Option<Vec<AggCall>>, RuntimeError> {
    map_call_inputs(calls, |dt| match dt {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::Null => Some(DataType::Int64),
        DataType::Float16 | DataType::Float32 => Some(DataType::Float64),
        _ => None,
    })
}
