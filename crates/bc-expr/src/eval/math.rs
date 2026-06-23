//! Numeric evaluation for `Expr::Math`/`Math2`/`Coalesce`/`Greatest`/`Least`
//! (split out of `lib.rs`).

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch};
use arrow::compute::kernels::cmp;
use arrow::compute::kernels::zip::zip;
use arrow::compute::{cast, is_not_null};
use arrow::datatypes::DataType;

use crate::eval::binary::coerce_numeric;
use crate::{Expr, ExprError, Math2Func, MathFunc};

/// `is_nan(x)`: true where a value is IEEE NaN (a float-only notion, distinct from
/// null). Nulls propagate (null → null). NaN is impossible for non-float numerics,
/// so casting them to Float64 yields all-false, which is correct. This is a
/// first-class op rather than the `x != x` lowering, because the engine's `!=`
/// operator uses *total ordering* (where `NaN == NaN`), so `x != x` would never
/// flag a NaN. The Tier-1 JIT does not compile `IsNan` and falls back here.
pub(crate) fn eval_is_nan(array: &ArrayRef) -> Result<ArrayRef, ExprError> {
    let f = cast(array, &DataType::Float64)?;
    let a = f
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("cast to f64");
    let out: BooleanArray = if a.null_count() == 0 {
        a.values().iter().map(|&x| Some(x.is_nan())).collect()
    } else {
        (0..a.len())
            .map(|i| (!a.is_null(i)).then(|| a.value(i).is_nan()))
            .collect()
    };
    Ok(Arc::new(out))
}

/// Two-argument math: align both sides to Float64, apply element-wise (nulls
/// propagate). `round`'s second argument is the (per-row) decimal-place count.
pub(crate) fn eval_math2(
    func: Math2Func,
    l: &ArrayRef,
    r: &ArrayRef,
) -> Result<ArrayRef, ExprError> {
    let lf = cast(l, &DataType::Float64)?;
    let rf = cast(r, &DataType::Float64)?;
    let a = lf.as_any().downcast_ref::<Float64Array>().expect("f64");
    let b = rf.as_any().downcast_ref::<Float64Array>().expect("f64");
    let out: Float64Array = if a.null_count() == 0 && b.null_count() == 0 {
        // No-null fast path: walk both raw slices, no per-element validity branch.
        a.values()
            .iter()
            .zip(b.values())
            .map(|(&x, &y)| apply_binary(func, x, y))
            .collect()
    } else {
        (0..a.len())
            .map(|i| {
                (!a.is_null(i) && !b.is_null(i)).then(|| apply_binary(func, a.value(i), b.value(i)))
            })
            .collect()
    };
    Ok(Arc::new(out))
}

/// One two-argument math op on scalar `f64`s (shared by both null paths).
#[inline]
fn apply_binary(func: Math2Func, x: f64, y: f64) -> f64 {
    match func {
        Math2Func::Pow => x.powf(y),
        Math2Func::Atan2 => x.atan2(y),
        Math2Func::Round => {
            let f = 10f64.powi(y as i32);
            (x * f).round() / f
        }
    }
}

/// `GREATEST`/`LEAST`: fold the inputs element-wise, ignoring nulls (a null
/// argument never wins; the result is null only where every argument is null).
pub(crate) fn eval_extreme(
    inputs: &[Expr],
    batch: &RecordBatch,
    greatest: bool,
) -> Result<ArrayRef, ExprError> {
    if inputs.is_empty() {
        return Err(ExprError::MissingArgument {
            func: if greatest { "greatest" } else { "least" }.into(),
            arg: "inputs",
        });
    }
    let mut acc = inputs[0].eval(batch)?;
    for next in &inputs[1..] {
        let b = next.eval(batch)?;
        let cmp = if greatest {
            cmp::gt_eq(&acc, &b)?
        } else {
            cmp::lt_eq(&acc, &b)?
        };
        // Where both are non-null, pick the winner; null elsewhere. Then coalesce
        // with each side so a lone non-null still survives.
        let both = zip(&cmp, &acc.as_ref(), &b.as_ref())?;
        acc = coalesce_arrays(&[both, acc, b])?;
    }
    Ok(acc)
}

/// First non-null per row across the given arrays (array-level COALESCE).
fn coalesce_arrays(arrs: &[ArrayRef]) -> Result<ArrayRef, ExprError> {
    let mut acc = arrs[arrs.len() - 1].clone();
    for a in arrs[..arrs.len() - 1].iter().rev() {
        let mask = is_not_null(a)?;
        acc = zip(&mask, &a.as_ref(), &acc.as_ref())?;
    }
    Ok(acc)
}

/// COALESCE: first non-null among the inputs, per row. Folds from the last input
/// upward so earlier inputs win.
pub(crate) fn eval_coalesce(inputs: &[Expr], batch: &RecordBatch) -> Result<ArrayRef, ExprError> {
    if inputs.is_empty() {
        return Err(ExprError::MissingArgument {
            func: "coalesce".to_string(),
            arg: "inputs",
        });
    }
    let mut acc = inputs[inputs.len() - 1].eval(batch)?;
    for expr in inputs[..inputs.len() - 1].iter().rev() {
        let v = expr.eval(batch)?;
        let present = is_not_null(&v)?;
        // Promote mixed numeric inputs to a common type (e.g. coalesce(int,float)
        // → float) so `zip` sees matching types, matching SQL coercion.
        let (v, acc_c) = coerce_numeric(&v, &acc)?;
        acc = zip(&present, &v.as_ref(), &acc_c.as_ref())?;
    }
    Ok(acc)
}

/// Unary math. `abs` keeps the input numeric type; `round`/`floor`/`ceil`/`sqrt`
/// yield Float64 (integer inputs are promoted).
pub(crate) fn eval_math(func: MathFunc, arr: &ArrayRef) -> Result<ArrayRef, ExprError> {
    use MathFunc::*;
    match (func, arr.data_type()) {
        (Abs, DataType::Int64) => {
            let a = arr.as_any().downcast_ref::<Int64Array>().unwrap();
            let out: Int64Array = if a.null_count() == 0 {
                a.values().iter().map(|&v| v.abs()).collect()
            } else {
                a.iter().map(|o| o.map(|v| v.abs())).collect()
            };
            Ok(Arc::new(out))
        }
        (_, DataType::Int64) => {
            // Promote integers to Float64 and apply the float function.
            let f = cast(arr, &DataType::Float64)?;
            eval_math(func, &f)
        }
        (_, DataType::Float64) => {
            let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
            // No-null fast path: map the raw slice (no per-element validity branch,
            // so the simple ops auto-vectorize); otherwise propagate nulls.
            let out: Float64Array = if a.null_count() == 0 {
                a.values().iter().map(|&v| apply_unary(func, v)).collect()
            } else {
                a.iter().map(|o| o.map(|v| apply_unary(func, v))).collect()
            };
            Ok(Arc::new(out))
        }
        (_, other) => Err(ExprError::ExpectedString {
            func: format!("{func:?}"),
            got: other.to_string(),
        }),
    }
}

/// One unary math op on a scalar `f64` (shared by both null paths of `eval_math`).
#[inline]
fn apply_unary(func: MathFunc, v: f64) -> f64 {
    use MathFunc::*;
    match func {
        Abs => v.abs(),
        Round => v.round(),
        Floor => v.floor(),
        Ceil => v.ceil(),
        Sqrt => v.sqrt(),
        Ln => v.ln(),
        Log10 => v.log10(),
        Log2 => v.log2(),
        Exp => v.exp(),
        Sin => v.sin(),
        Cos => v.cos(),
        Tan => v.tan(),
        Sign => {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        Trunc => v.trunc(),
        Cbrt => v.cbrt(),
        Asin => v.asin(),
        Acos => v.acos(),
        Atan => v.atan(),
        Sinh => v.sinh(),
        Cosh => v.cosh(),
        Tanh => v.tanh(),
        Degrees => v.to_degrees(),
        Radians => v.to_radians(),
        Cot => 1.0 / v.tan(),
    }
}
