//! The `Expr::eval` dispatch — split out of `lib.rs` so the wire-contract enum
//! definitions stay there and the (large) per-variant dispatch lives here. This is
//! an inherent `impl Expr`, so `Expr::eval` is available crate-wide regardless of
//! module. Behavior is unchanged — moved verbatim.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow::compute::kernels::boolean;
use arrow::compute::kernels::zip::zip;
use arrow::compute::{is_not_null, is_null};

use crate::eval::binary::{as_bool, coerce_numeric, eval_binary, try_scalar_binary};
use crate::eval::cast::cast_expr;
use crate::eval::date::{
    eval_date, eval_date_offset, eval_date_trunc, eval_strftime, eval_strptime,
    eval_window_buckets, eval_window_start, parse_dtype,
};
use crate::eval::generate::eval_sequence;
use crate::eval::list::{
    eval_array, eval_list, eval_list_binary, eval_list_contains, eval_list_get, eval_list_join,
    eval_list_position, eval_make_struct, eval_struct_field, rebuild_list, require_list,
};
use crate::eval::list_ops::{eval_list_filter, eval_list_set, eval_list_transform};
use crate::eval::map::eval_map;
use crate::eval::math::{
    eval_coalesce, eval_extreme, eval_is_inf, eval_is_nan, eval_math, eval_math2,
};
use crate::eval::media::{eval_audio, eval_image, eval_video};
use crate::eval::str::eval_str;
use crate::eval::timezone::eval_convert_timezone;
use crate::{BinaryOp, Expr, ExprError};

impl Expr {
    /// Evaluate the expression against `batch`, returning a full-length column.
    pub fn eval(&self, batch: &RecordBatch) -> Result<ArrayRef, ExprError> {
        match self {
            Expr::Col { name } => batch
                .column_by_name(name)
                .cloned()
                .ok_or_else(|| ExprError::UnknownColumn(name.clone())),
            Expr::Lit { value } => Ok(value.to_array(batch.num_rows())),
            Expr::Not { input } => {
                let arr = input.eval(batch)?;
                let b = as_bool(&arr, "not")?;
                Ok(Arc::new(boolean::not(b)?))
            }
            Expr::Binary { op, left, right } => {
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
            } => {
                let arr = input.eval(batch)?;
                eval_image(*func, &arr, *width, *height)
            }
            Expr::Audio { func, input } => {
                let arr = input.eval(batch)?;
                eval_audio(*func, &arr)
            }
            Expr::Video { func, input } => {
                let arr = input.eval(batch)?;
                eval_video(*func, &arr)
            }
            Expr::Coalesce { inputs } => eval_coalesce(inputs, batch),
            Expr::Array { elements } => eval_array(elements, batch),
            Expr::Sequence { start, stop, step } => {
                let (s, e, d) = (start.eval(batch)?, stop.eval(batch)?, step.eval(batch)?);
                eval_sequence(&s, &e, &d)
            }
            Expr::ListSet { op, left, right } => {
                let (l, r) = (left.eval(batch)?, right.eval(batch)?);
                eval_list_set(*op, &l, &r)
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
                    let begin = (s as i64 + (*offset).max(0)).min(e as i64) as usize;
                    let end = match length {
                        Some(l) => (begin as i64 + (*l).max(0)).min(e as i64) as usize,
                        None => e,
                    };
                    (begin..end).map(|k| k as u32).collect()
                })
            }
        }
    }
}
