//! `cast` evaluation with DuckDB float→int rounding semantics.
//!
//! Split out of `lib.rs` for file size; the `Expr::Cast` variant and its wire tags
//! stay in `lib.rs`. Behavior is unchanged.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array};
use arrow::compute::{cast_with_options, CastOptions};

use crate::ExprError;

/// Cast `arr` to `target` with DuckDB float→int semantics. Arrow's float→int cast
/// truncates toward zero; DuckDB rounds half-to-even (`cast(2.5)` = 2, `cast(3.5)`
/// = 4), so float inputs are rounded to an integral value before the cast. All
/// other casts defer to the arrow kernel unchanged. (The JIT never compiles
/// float→int, so this interpreter-only behavior keeps tier parity intact.)
///
/// `try_cast` selects arrow's *safe* cast (a value that cannot be converted
/// becomes NULL — DuckDB `TRY_CAST`); the strict default (`false`) errors on an
/// invalid value (DuckDB `CAST`).
pub(crate) fn cast_expr(
    arr: &ArrayRef,
    target: &arrow::datatypes::DataType,
    try_cast: bool,
) -> Result<ArrayRef, ExprError> {
    use arrow::datatypes::DataType::{
        Float16, Float32, Float64, Int16, Int32, Int64, Int8, UInt16, UInt32, UInt64, UInt8,
    };
    let opts = CastOptions {
        safe: try_cast,
        ..Default::default()
    };
    let int_target = matches!(
        target,
        Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64
    );
    let float_src = matches!(arr.data_type(), Float16 | Float32 | Float64);
    if int_target && float_src {
        // Round half-to-even first (DuckDB), then cast the now-integral floats.
        let f = cast_with_options(arr, &Float64, &opts)?;
        let f = f
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("cast to Float64 yields Float64Array");
        let rounded: Float64Array = f.iter().map(|o| o.map(f64::round_ties_even)).collect();
        let rounded: ArrayRef = Arc::new(rounded);
        return Ok(cast_with_options(&rounded, target, &opts)?);
    }
    Ok(cast_with_options(arr, target, &opts)?)
}
