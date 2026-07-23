//! The `Expr::eval` dispatch — split out of `lib.rs` so the wire-contract enum
//! definitions stay there and the (large) per-variant dispatch lives here. This is
//! an inherent `impl Expr`, so `Expr::eval` is available crate-wide regardless of
//! module. Behavior is unchanged — moved verbatim.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow::compute::kernels::boolean;
use arrow::compute::kernels::zip::zip;
use arrow::compute::{is_not_null, is_null};

use crate::eval::binary::{
    as_bool, coerce_numeric, eval_binary, try_dict_compare, try_scalar_binary,
};
use crate::eval::cast::cast_expr;
use crate::eval::date::{
    eval_date, eval_date_offset, eval_date_trunc, eval_strftime, eval_strptime,
    eval_window_buckets, eval_window_start, parse_dtype,
};
use crate::eval::generate::eval_sequence;
use crate::eval::in_list::eval_in_list;
use crate::eval::list::{
    eval_array, eval_list, eval_list_binary, eval_list_contains, eval_list_get, eval_list_join,
    eval_list_position, eval_make_struct, eval_struct_field, rebuild_list, require_list,
};
use crate::eval::list_ops::{eval_list_filter, eval_list_set, eval_list_transform, eval_list_zip};
use crate::eval::map::eval_map;
use crate::eval::math::{
    eval_coalesce, eval_extreme, eval_is_inf, eval_is_nan, eval_math, eval_math2,
};
use crate::eval::media::{eval_audio, eval_image, eval_video};
use crate::eval::str::{eval_str, try_dict_str};
use crate::eval::timezone::eval_convert_timezone;
use crate::{BinaryOp, Expr, ExprError};

/// Decode a dictionary-encoded array to its value type; identity for any other array.
///
/// The scalar `eval` path decodes at the `Col` leaf so every downstream kernel sees a
/// plain array and stays oblivious to dictionary encoding; only the dict-native ops read a
/// column directly. `DictionaryArray` is a standard Arrow type, so this keeps the
/// Arrow-only columnar contract while letting dictionary-encoded inputs (common in Parquet)
/// flow through the engine without a blanket decode at the FFI boundary.
pub(crate) fn decode_dict(arr: ArrayRef) -> Result<ArrayRef, ExprError> {
    if let arrow::datatypes::DataType::Dictionary(_, value) = arr.data_type() {
        Ok(arrow::compute::cast(&arr, value)?)
    } else {
        Ok(arr)
    }
}

impl Expr {
    /// Evaluate the expression against `batch`, returning a full-length column.
    pub fn eval(&self, batch: &RecordBatch) -> Result<ArrayRef, ExprError> {
        match self {
            Expr::Col { name } => {
                let arr = batch
                    .column_by_name(name)
                    .cloned()
                    .ok_or_else(|| ExprError::UnknownColumn(name.clone()))?;
                // Decode a dictionary column to its value type at the leaf, so every
                // downstream scalar kernel sees a plain array and never has to special-case
                // dictionary encoding. Identity (a cheap `Arc` clone) for any non-dictionary
                // column — the overwhelming common case — so existing data is bit-unchanged.
                // The dict-native ops (currently `InList`) bypass this and read the column
                // directly to keep the dictionary.
                decode_dict(arr)
            }
            Expr::Lit { value } => Ok(value.to_array(batch.num_rows())),
            Expr::Not { input } => {
                let arr = input.eval(batch)?;
                let b = as_bool(&arr, "not")?;
                Ok(Arc::new(boolean::not(b)?))
            }
            Expr::Binary { op, left, right } => {
                // Fast path: comparing a *dictionary* column to a literal compares the
                // distinct values and gathers through the keys, never decoding the column.
                // Checked first, because `try_scalar_binary` would `eval` the column — which
                // decodes the dictionary at the leaf — only to then reject a non-numeric type
                // and fall through, having already paid for the decode it was trying to avoid.
                if let Some(out) = try_dict_compare(*op, left, right, batch)? {
                    return Ok(out);
                }
                // Fast path: a numeric literal operand broadcasts as a scalar instead
                // of materializing a full N-length array (bit-identical result).
                if let Some(out) = try_scalar_binary(*op, left, right, batch)? {
                    return Ok(out);
                }
                let l = left.eval(batch)?;
                let r = right.eval(batch)?;
                eval_binary(*op, &l, &r)
            }
            Expr::Cast {
                input,
                dtype,
                try_cast,
            } => {
                let arr = input.eval(batch)?;
                cast_expr(&arr, &parse_dtype(dtype)?, *try_cast)
            }
            Expr::IsNull { input } => Ok(Arc::new(is_null(&input.eval(batch)?)?)),
            Expr::IsNotNull { input } => Ok(Arc::new(is_not_null(&input.eval(batch)?)?)),
            Expr::IsNan { input } => eval_is_nan(&input.eval(batch)?),
            Expr::IsInf { input } => eval_is_inf(&input.eval(batch)?),
            Expr::Case {
                branches,
                otherwise,
            } => {
                // Fold from the default upward: later branches are overridden by
                // earlier ones (first matching WHEN wins).
                let mut acc = otherwise.eval(batch)?;
                for branch in branches.iter().rev() {
                    let mask_arr = branch.when.eval(batch)?;
                    let mask = as_bool(&mask_arr, "case")?;
                    // SQL CASE semantics: a WHEN that evaluates to NULL is *not*
                    // taken (it falls through to ELSE), matching DuckDB. `zip` would
                    // otherwise let a null mask pick the THEN branch, so collapse a
                    // null mask element to false (true only where value AND valid).
                    let mask = match mask.nulls() {
                        Some(n) => BooleanArray::new(mask.values() & n.inner(), None),
                        None => mask.clone(),
                    };
                    let then = branch.then.eval(batch)?;
                    // `zip` requires matching branch types; coerce Int64/Float64
                    // (and decimal) to a common numeric type the way COALESCE and
                    // the binary ops do, so a `when(...).then(0).otherwise(x)` over a
                    // float column (or `clip`/`fill_nan`) doesn't error on a mixed
                    // int/float literal.
                    let (then, acc_c) = coerce_numeric(&then, &acc)?;
                    acc = zip(&mask, &then.as_ref(), &acc_c.as_ref())?;
                }
                Ok(acc)
            }
            Expr::Str {
                func,
                input,
                pattern,
                replacement,
                start,
                length,
            } => {
                // Fast path: a *dictionary* column applies the function to its distinct
                // values and gathers through the keys, never decoding the column. Checked
                // before `eval`, which decodes at the leaf — the whole point is to avoid
                // materializing the decoded column at all.
                if let Some(out) = try_dict_str(
                    *func,
                    input,
                    batch,
                    pattern.as_deref(),
                    replacement.as_deref(),
                    *start,
                    *length,
                )? {
                    return Ok(out);
                }
                let arr = input.eval(batch)?;
                eval_str(
                    *func,
                    &arr,
                    pattern.as_deref(),
                    replacement.as_deref(),
                    *start,
                    *length,
                )
            }
            Expr::Date { func, input } => {
                let arr = input.eval(batch)?;
                eval_date(*func, &arr)
            }
            Expr::Image {
                func,
                input,
                width,
                height,
                mean,
                std,
                channels_first,
            } => {
                let arr = input.eval(batch)?;
                eval_image(
                    *func,
                    &arr,
                    *width,
                    *height,
                    mean.as_deref(),
                    std.as_deref(),
                    *channels_first,
                )
            }
            Expr::Audio {
                func,
                input,
                rate,
                n_fft,
                hop_length,
                n_mels,
                n_mfcc,
            } => {
                let arr = input.eval(batch)?;
                eval_audio(*func, &arr, *rate, *n_fft, *hop_length, *n_mels, *n_mfcc)
            }
            Expr::Video { func, input } => {
                let arr = input.eval(batch)?;
                eval_video(*func, &arr)
            }
            Expr::Coalesce { inputs } => eval_coalesce(inputs, batch),
            Expr::InList { input, set } => {
                // Read a `Col` input directly (keeping any dictionary encoding) so the
                // dict-accelerated membership path fires; any other input is evaluated
                // normally (already decoded at its `Col` leaves).
                let arr = match input.as_ref() {
                    Expr::Col { name } => batch
                        .column_by_name(name)
                        .cloned()
                        .ok_or_else(|| ExprError::UnknownColumn(name.clone()))?,
                    _ => input.eval(batch)?,
                };
                eval_in_list(&arr, set)
            }
            Expr::Array { elements } => eval_array(elements, batch),
            Expr::Hash { inputs, seed } => {
                let args: Vec<_> = inputs
                    .iter()
                    .map(|e| e.eval(batch))
                    .collect::<Result<_, _>>()?;
                crate::eval::hash::eval_hash(&args, *seed, batch.num_rows())
            }
            Expr::Sequence { start, stop, step } => {
                let (s, e, d) = (start.eval(batch)?, stop.eval(batch)?, step.eval(batch)?);
                eval_sequence(&s, &e, &d)
            }
            Expr::ListSet { op, left, right } => {
                let (l, r) = (left.eval(batch)?, right.eval(batch)?);
                eval_list_set(*op, &l, &r)
            }
            Expr::ListZip { op, left, right } => {
                let (l, r) = (left.eval(batch)?, right.eval(batch)?);
                eval_list_zip(*op, &l, &r)
            }
            Expr::ListTransform { input, func } => eval_list_transform(&input.eval(batch)?, func),
            Expr::ListFilter { input, pred } => eval_list_filter(&input.eval(batch)?, pred),
            Expr::MakeStruct { fields } => eval_make_struct(fields, batch),
            Expr::ListJoin { input, separator } => eval_list_join(&input.eval(batch)?, separator),
            Expr::Math { func, input } => {
                let arr = input.eval(batch)?;
                eval_math(*func, &arr)
            }
            Expr::List { func, input } => {
                let arr = input.eval(batch)?;
                eval_list(*func, &arr)
            }
            Expr::NullIf { left, right } => {
                let l = left.eval(batch)?;
                let r = right.eval(batch)?;
                let eq = eval_binary(BinaryOp::Eq, &l, &r)?;
                let mask = as_bool(&eq, "nullif")?;
                Ok(arrow::compute::nullif(&l, mask)?)
            }
            Expr::Greatest { inputs } => eval_extreme(inputs, batch, true),
            Expr::Least { inputs } => eval_extreme(inputs, batch, false),
            Expr::Math2 { func, left, right } => {
                let l = left.eval(batch)?;
                let r = right.eval(batch)?;
                eval_math2(*func, &l, &r)
            }
            Expr::ListGet { input, index } => {
                let arr = input.eval(batch)?;
                eval_list_get(&arr, *index)
            }
            Expr::ListSimhash {
                input,
                num_bits,
                seed,
            } => {
                let arr = input.eval(batch)?;
                crate::eval::list_ops::eval_list_simhash(&arr, *num_bits, *seed)
            }
            Expr::StructField { input, field } => {
                let arr = input.eval(batch)?;
                eval_struct_field(&arr, field)
            }
            Expr::ListContains { input, value } => {
                let arr = input.eval(batch)?;
                eval_list_contains(&arr, value)
            }
            Expr::ListPosition { input, value } => {
                let arr = input.eval(batch)?;
                eval_list_position(&arr, value)
            }
            Expr::Map { func, input, key } => {
                let arr = input.eval(batch)?;
                eval_map(*func, &arr, key.as_ref())
            }
            Expr::ListBinary { func, left, right } => {
                let l = left.eval(batch)?;
                let r = right.eval(batch)?;
                eval_list_binary(*func, &l, &r)
            }
            Expr::DateTrunc { input, unit } => {
                let arr = input.eval(batch)?;
                eval_date_trunc(&arr, unit)
            }
            Expr::Strftime { input, format } => {
                let arr = input.eval(batch)?;
                eval_strftime(&arr, format)
            }
            Expr::ConvertTimezone {
                input,
                from_tz,
                to_tz,
            } => {
                let arr = input.eval(batch)?;
                eval_convert_timezone(&arr, from_tz, to_tz)
            }
            Expr::Strptime { input, format } => {
                let arr = input.eval(batch)?;
                eval_strptime(&arr, format)
            }
            Expr::DateOffset {
                input,
                months,
                days,
                micros,
            } => {
                let arr = input.eval(batch)?;
                eval_date_offset(&arr, *months, *days, *micros)
            }
            Expr::WindowStart {
                input,
                width_micros,
                origin_micros,
            } => {
                let arr = input.eval(batch)?;
                eval_window_start(&arr, *width_micros, *origin_micros)
            }
            Expr::WindowBuckets {
                input,
                width_micros,
                slide_micros,
            } => {
                let arr = input.eval(batch)?;
                eval_window_buckets(&arr, *width_micros, *slide_micros)
            }
            Expr::ListSlice {
                input,
                offset,
                length,
            } => {
                let arr = input.eval(batch)?;
                let list = require_list(&arr, "list.slice")?;
                rebuild_list(list, |s, e| {
                    // Saturating throughout: a huge `offset`/`length` (up to i64::MAX)
                    // otherwise overflows the `+` before the `.min(e)` clamp — panicking
                    // in debug and wrapping to a giant `usize` (capacity overflow) in
                    // release. `list.slice(3, i64::MAX)` must clamp to the list end.
                    let begin = (s as i64).saturating_add((*offset).max(0)).min(e as i64) as usize;
                    let end = match length {
                        Some(l) => {
                            (begin as i64).saturating_add((*l).max(0)).min(e as i64) as usize
                        }
                        None => e,
                    };
                    (begin..end).map(|k| k as u32).collect()
                })
            }
        }
    }
}

#[cfg(test)]
mod dict_tests {
    use super::*;
    use crate::{Literal, StrFunc};
    use arrow::array::DictionaryArray;
    use arrow::datatypes::{DataType, Field, Int32Type, Schema};

    /// A batch with one `Dictionary<Int32, Utf8>` column `s`, plus the same column decoded
    /// to plain `Utf8` under name `s` — for asserting every expression agrees on both.
    fn dict_and_plain() -> (RecordBatch, RecordBatch) {
        let dict: DictionaryArray<Int32Type> =
            [Some("MAIL"), Some("AIR"), None, Some("SHIP"), Some("MAIL")]
                .into_iter()
                .collect();
        let dict_arr: ArrayRef = Arc::new(dict);
        let plain: ArrayRef = arrow::compute::cast(&dict_arr, &DataType::Utf8).unwrap();
        let mk = |a: ArrayRef| {
            let schema = Schema::new(vec![Field::new("s", a.data_type().clone(), true)]);
            RecordBatch::try_new(Arc::new(schema), vec![a]).unwrap()
        };
        (mk(dict_arr), mk(plain))
    }

    fn col() -> Expr {
        Expr::Col { name: "s".into() }
    }

    /// Every scalar expression must produce the same result over a dictionary column as
    /// over its decoded form — the decode-at-`Col` safety net plus the dict-native `InList`.
    #[test]
    fn scalar_exprs_agree_dict_vs_decoded() {
        let (d, p) = dict_and_plain();
        let exprs = [
            // comparison against a string literal (decoded at the Col leaf)
            Expr::Binary {
                op: BinaryOp::Eq,
                left: Box::new(col()),
                right: Box::new(Expr::Lit {
                    value: Literal::Str("MAIL".into()),
                }),
            },
            // a string function (decoded at the Col leaf)
            Expr::Str {
                func: StrFunc::Upper,
                input: Box::new(col()),
                pattern: None,
                replacement: None,
                start: None,
                length: None,
            },
            // IN membership (dict-native fast path over the dictionary column)
            Expr::InList {
                input: Box::new(col()),
                set: vec![Literal::Str("MAIL".into()), Literal::Str("SHIP".into())],
            },
        ];
        for e in &exprs {
            let od = e.eval(&d).expect("eval over dict");
            let op = e.eval(&p).expect("eval over plain");
            assert_eq!(od.as_ref(), op.as_ref(), "mismatch for {e:?}");
        }
    }
}
