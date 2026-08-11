//! Evaluation of the `Expr::Spatial` variant — the array-level half of rigid-body
//! support.
//!
//! This module is the Arrow half: evaluate the arguments, bring them to `Float64`,
//! walk the rows. `apply` is the other half — one row of numbers in, one number out,
//! with no Arrow type in sight.
//!
//! `bc-spatial` knows rotations and poses and knows nothing about Arrow; this module is
//! the seam, the same way `eval/geo` is the seam for `bc-geo`. It evaluates each
//! argument to a column, brings them all to `Float64`, walks the rows, and builds one
//! `Float64` output.
//!
//! # Why the arguments are cast once rather than read polymorphically
//!
//! Every argument here is a number, and a point cloud read from a `.bin` sweep or a
//! Parquet file is very often `Float32` — half the bytes, and no worse than `Float64`
//! for a lidar return measured to the centimetre. Casting each argument column once
//! with arrow's kernel costs one vectorized pass and lets the row loop read from a
//! plain `Float64Array` with no per-row type dispatch. A per-row match, which is what
//! `eval/geo` does because its arguments are heterogeneous, would run ten times per
//! point on the hot function in this family.
//!
//! # Null semantics
//!
//! A null in any argument makes the result null, which is the ordinary rule for scalar
//! arithmetic and needs no exception here. A *zero* quaternion is also null: it names no
//! rotation, so there is no answer to give, and nulling it keeps a single corrupt pose
//! in a hundred-million-row log from ending the query. `WHERE quat_norm(...) = 0` finds
//! those rows.
//!
//! Nothing in this family raises on a row's *value*. The only error it can produce is a
//! query-level one — an argument that is not a number at all.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, Float64Builder, RecordBatch};
use arrow::compute::cast;
use arrow::datatypes::DataType;

use crate::{Expr, ExprError, SpatialFunc};

mod apply;

use apply::apply;

/// Evaluate a rigid-body function over a batch.
pub fn eval_spatial(
    func: SpatialFunc,
    args: &[Expr],
    batch: &RecordBatch,
) -> Result<ArrayRef, ExprError> {
    if args.len() != func.arity() {
        return Err(ExprError::InvalidArgument {
            func: fn_name(func),
            reason: format!("expects {} argument(s), got {}", func.arity(), args.len()),
        });
    }
    let cols: Vec<ArrayRef> = args
        .iter()
        .map(|a| a.eval(batch).and_then(|c| to_f64(&c, func)))
        .collect::<Result<_, _>>()?;
    let floats: Vec<&Float64Array> = cols
        .iter()
        .map(|c| {
            c.as_any()
                .downcast_ref::<Float64Array>()
                .expect("cast to Float64 above")
        })
        .collect();

    let rows = batch.num_rows();
    let mut out = Float64Builder::with_capacity(rows);
    // A row's arguments, reused across rows so the loop allocates nothing. Ten is the
    // widest arity in the family (a pose and a point).
    let mut a = [0.0f64; 10];
    let n = floats.len();
    // Nullability is a property of the *column*, so ask once rather than once per value.
    // The hot shape here is a pose and a point with no nulls anywhere, and that shape was
    // otherwise paying ten bitmap probes per point across a sweep of a hundred thousand.
    let nullable: Vec<bool> = floats.iter().map(|c| c.null_count() > 0).collect();
    let any_nulls = nullable.iter().any(|n| *n);
    let raw: Vec<&[f64]> = floats.iter().map(|c| c.values().as_ref()).collect();

    'row: for i in 0..rows {
        if any_nulls {
            for (slot, (col, may_be_null)) in a[..n].iter_mut().zip(floats.iter().zip(&nullable)) {
                if *may_be_null && col.is_null(i) {
                    out.append_null();
                    continue 'row;
                }
                *slot = col.value(i);
            }
        } else {
            // No column can be null, so read straight out of the value buffers.
            for (slot, values) in a[..n].iter_mut().zip(&raw) {
                *slot = values[i];
            }
        }
        match apply(func, &a[..n]) {
            Some(v) => out.append_value(v),
            None => out.append_null(),
        }
    }
    Ok(Arc::new(out.finish()) as ArrayRef)
}

/// The name a user typed, for error messages — `SpatialFunc`'s `Debug` is the Rust
/// variant (`QuatRotateX`) and the user wrote `quat_rotate_x`.
fn fn_name(func: SpatialFunc) -> String {
    let debug = format!("{func:?}");
    let mut out = String::with_capacity(debug.len() + 6);
    for (i, c) in debug.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            out.push('_');
        }
        out.extend(c.to_lowercase());
    }
    // `Se3TransformX` debug-prints with the digit glued to `Se`, so the split above
    // yields `se3_transform_x` already — but `QuatFromRotmatX` needs no special case
    // either. Nothing further to do; the loop is the whole rule.
    out
}

/// Bring one argument column to `Float64`.
///
/// Rejects anything arrow will not cast losslessly-enough to a float, which is the only
/// error this family can raise. Boolean is refused explicitly: arrow will happily cast
/// it to 0.0/1.0, and a boolean reaching a coordinate slot is a query mistake worth
/// naming rather than silently treating as an origin-or-unit-metre offset.
fn to_f64(col: &ArrayRef, func: SpatialFunc) -> Result<ArrayRef, ExprError> {
    if col.data_type() == &DataType::Float64 {
        return Ok(Arc::clone(col));
    }
    let ok = matches!(
        col.data_type(),
        DataType::Float16
            | DataType::Float32
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
            | DataType::Null
    );
    if !ok {
        return Err(ExprError::ExpectedType {
            func: fn_name(func),
            want: "a numeric argument",
            got: col.data_type().to_string(),
        });
    }
    cast(col, &DataType::Float64).map_err(|e| ExprError::InvalidArgument {
        func: fn_name(func),
        reason: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BooleanArray, Float32Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use bc_spatial::Quat;

    fn batch(cols: Vec<(&str, ArrayRef)>) -> RecordBatch {
        let fields: Vec<Field> = cols
            .iter()
            .map(|(n, a)| Field::new(*n, a.data_type().clone(), true))
            .collect();
        let arrays: Vec<ArrayRef> = cols.into_iter().map(|(_, a)| a).collect();
        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
    }

    fn lit(v: f64) -> Expr {
        Expr::Lit {
            value: crate::Literal::Float(v),
        }
    }

    fn col(name: &str) -> Expr {
        Expr::Col { name: name.into() }
    }

    fn run(func: SpatialFunc, args: Vec<Expr>, b: &RecordBatch) -> ArrayRef {
        eval_spatial(func, &args, b).unwrap()
    }

    fn floats(a: &ArrayRef) -> &Float64Array {
        a.as_any().downcast_ref::<Float64Array>().unwrap()
    }

    /// A one-row batch, so a pure-literal expression has somewhere to be evaluated.
    fn one_row() -> RecordBatch {
        batch(vec![(
            "k",
            Arc::new(Float64Array::from(vec![0.0])) as ArrayRef,
        )])
    }

    fn quat_lits(q: Quat) -> Vec<Expr> {
        vec![lit(q.x), lit(q.y), lit(q.z), lit(q.w)]
    }

    fn axis_angle(ax: f64, ay: f64, az: f64, angle: f64) -> Quat {
        let (s, c) = (angle * 0.5).sin_cos();
        Quat::new(ax * s, ay * s, az * s, c)
    }

    #[test]
    fn rotate_reads_its_arguments_in_the_documented_order() {
        // A quarter turn about Z sends the X basis vector to Y.
        let q = axis_angle(0.0, 0.0, 1.0, std::f64::consts::FRAC_PI_2);
        let b = one_row();
        let mut args = quat_lits(q);
        args.extend([lit(1.0), lit(0.0), lit(0.0)]);
        let x = run(SpatialFunc::QuatRotateX, args.clone(), &b);
        let y = run(SpatialFunc::QuatRotateY, args, &b);
        assert!(floats(&x).value(0).abs() < 1e-12);
        assert!((floats(&y).value(0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn se3_transform_reads_translation_before_rotation() {
        // Translation (10, 0, 0), a quarter turn about Z, point (1, 0, 0).
        // Rotate-then-translate lands at (10, 1, 0); the other order at (0, 11, 0).
        let q = axis_angle(0.0, 0.0, 1.0, std::f64::consts::FRAC_PI_2);
        let b = one_row();
        let mut args = vec![lit(10.0), lit(0.0), lit(0.0)];
        args.extend(quat_lits(q));
        args.extend([lit(1.0), lit(0.0), lit(0.0)]);
        let x = run(SpatialFunc::Se3TransformX, args.clone(), &b);
        let y = run(SpatialFunc::Se3TransformY, args, &b);
        assert!((floats(&x).value(0) - 10.0).abs() < 1e-12);
        assert!((floats(&y).value(0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_float32_point_cloud_needs_no_cast_from_the_caller() {
        // The reason arguments are cast rather than matched: a lidar sweep is Float32.
        let q = axis_angle(0.0, 0.0, 1.0, std::f64::consts::FRAC_PI_2);
        let b = batch(vec![
            (
                "px",
                Arc::new(Float32Array::from(vec![1.0f32, 2.0])) as ArrayRef,
            ),
            (
                "py",
                Arc::new(Float32Array::from(vec![0.0f32, 0.0])) as ArrayRef,
            ),
            (
                "pz",
                Arc::new(Float32Array::from(vec![0.0f32, 0.0])) as ArrayRef,
            ),
        ]);
        let mut args = quat_lits(q);
        args.extend([col("px"), col("py"), col("pz")]);
        let y = run(SpatialFunc::QuatRotateY, args, &b);
        assert!((floats(&y).value(0) - 1.0).abs() < 1e-6);
        assert!((floats(&y).value(1) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn a_null_in_any_argument_nulls_the_row() {
        let q = axis_angle(0.0, 0.0, 1.0, 0.3);
        let b = batch(vec![(
            "px",
            Arc::new(Float64Array::from(vec![Some(1.0), None])) as ArrayRef,
        )]);
        let mut args = quat_lits(q);
        args.extend([col("px"), lit(0.0), lit(0.0)]);
        let out = run(SpatialFunc::QuatRotateX, args, &b);
        assert!(!out.is_null(0));
        assert!(out.is_null(1));
    }

    #[test]
    fn a_zero_quaternion_nulls_the_row_rather_than_failing_the_query() {
        let b = one_row();
        let args = vec![
            lit(0.0),
            lit(0.0),
            lit(0.0),
            lit(0.0),
            lit(1.0),
            lit(2.0),
            lit(3.0),
        ];
        let out = run(SpatialFunc::QuatRotateX, args, &b);
        assert!(out.is_null(0));
    }

    #[test]
    fn quat_norm_reports_the_zero_rather_than_nulling_it() {
        // The one function that must answer for a zero quaternion, because it is how a
        // user finds the rows the others nulled.
        let b = one_row();
        let args = vec![lit(0.0), lit(0.0), lit(0.0), lit(0.0)];
        let out = run(SpatialFunc::QuatNorm, args, &b);
        assert_eq!(floats(&out).value(0), 0.0);
    }

    #[test]
    fn wrong_argument_count_is_a_query_error() {
        let b = one_row();
        let err = eval_spatial(SpatialFunc::QuatRotateX, &[lit(1.0)], &b).unwrap_err();
        assert!(format!("{err}").contains("quat_rotate_x"), "{err}");
    }

    #[test]
    fn a_non_numeric_argument_is_a_query_error() {
        let b = batch(vec![(
            "s",
            Arc::new(StringArray::from(vec!["a"])) as ArrayRef,
        )]);
        let args = vec![col("s"), lit(0.0), lit(0.0), lit(1.0)];
        assert!(eval_spatial(SpatialFunc::QuatNorm, &args, &b).is_err());
    }

    #[test]
    fn a_boolean_argument_is_refused_rather_than_read_as_zero_or_one() {
        let b = batch(vec![(
            "flag",
            Arc::new(BooleanArray::from(vec![true])) as ArrayRef,
        )]);
        let args = vec![col("flag"), lit(0.0), lit(0.0), lit(1.0)];
        assert!(eval_spatial(SpatialFunc::QuatNorm, &args, &b).is_err());
    }

    #[test]
    fn every_function_in_the_vocabulary_evaluates_at_its_declared_arity() {
        // The table `arity` returns and the slots `apply` reads must agree for every
        // variant. A mismatch is an index panic on the first row, which no differential
        // test that happens not to name that function would ever reach.
        use SpatialFunc::*;
        let all = [
            QuatNorm,
            QuatNormalizeX,
            QuatNormalizeY,
            QuatNormalizeZ,
            QuatNormalizeW,
            QuatInverseX,
            QuatInverseY,
            QuatInverseZ,
            QuatInverseW,
            QuatAngle,
            QuatToRoll,
            QuatToPitch,
            QuatToYaw,
            QuatFromEulerX,
            QuatFromEulerY,
            QuatFromEulerZ,
            QuatFromEulerW,
            QuatFromRotmatX,
            QuatFromRotmatY,
            QuatFromRotmatZ,
            QuatFromRotmatW,
            QuatMultiplyX,
            QuatMultiplyY,
            QuatMultiplyZ,
            QuatMultiplyW,
            QuatAngularDistance,
            QuatSlerpX,
            QuatSlerpY,
            QuatSlerpZ,
            QuatSlerpW,
            QuatRotateX,
            QuatRotateY,
            QuatRotateZ,
            QuatInverseRotateX,
            QuatInverseRotateY,
            QuatInverseRotateZ,
            Se3TransformX,
            Se3TransformY,
            Se3TransformZ,
            Se3InverseTransformX,
            Se3InverseTransformY,
            Se3InverseTransformZ,
        ];
        assert_eq!(all.len(), 42, "a variant was added without a case here");
        let b = one_row();
        for func in all {
            // 0.5 everywhere: a valid non-zero quaternion, a valid point, a valid `t`,
            // and a matrix that `from_rotation_matrix` accepts.
            let args: Vec<Expr> = (0..func.arity()).map(|_| lit(0.5)).collect();
            let out =
                eval_spatial(func, &args, &b).unwrap_or_else(|e| panic!("{}: {e}", fn_name(func)));
            assert_eq!(out.len(), 1, "{}", fn_name(func));
            assert!(!out.is_null(0), "{} nulled a valid row", fn_name(func));
        }
    }

    #[test]
    fn function_names_render_as_the_user_wrote_them() {
        assert_eq!(fn_name(SpatialFunc::QuatRotateX), "quat_rotate_x");
        assert_eq!(fn_name(SpatialFunc::Se3TransformZ), "se3_transform_z");
        assert_eq!(
            fn_name(SpatialFunc::QuatAngularDistance),
            "quat_angular_distance"
        );
    }

    #[test]
    fn a_sliced_batch_reads_from_the_right_offset() {
        // The no-null fast path reads the value buffer directly instead of calling
        // `value(i)`. A sliced array carries an offset, and a buffer read that ignored it
        // would return a neighbouring row's coordinate — the right *shape* of answer,
        // silently wrong, on exactly the batches a morsel-driven engine produces.
        let full = batch(vec![(
            "px",
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])) as ArrayRef,
        )]);
        let sliced = full.slice(2, 3);
        let args = vec![
            lit(0.0),
            lit(0.0),
            lit(0.0),
            lit(1.0),
            col("px"),
            lit(0.0),
            lit(0.0),
        ];
        // The identity rotation, so each output is its input coordinate.
        let out = run(SpatialFunc::QuatRotateX, args, &sliced);
        let got: Vec<f64> = (0..out.len()).map(|i| floats(&out).value(i)).collect();
        assert_eq!(got, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn a_sliced_batch_with_nulls_reads_from_the_right_offset() {
        let full = batch(vec![(
            "px",
            Arc::new(Float64Array::from(vec![
                Some(1.0),
                None,
                Some(3.0),
                None,
                Some(5.0),
            ])) as ArrayRef,
        )]);
        let sliced = full.slice(2, 3);
        let args = vec![
            lit(0.0),
            lit(0.0),
            lit(0.0),
            lit(1.0),
            col("px"),
            lit(0.0),
            lit(0.0),
        ];
        let out = run(SpatialFunc::QuatRotateX, args, &sliced);
        assert_eq!(floats(&out).value(0), 3.0);
        assert!(out.is_null(1));
        assert_eq!(floats(&out).value(2), 5.0);
    }

    #[test]
    fn an_empty_batch_yields_an_empty_column() {
        let b = batch(vec![(
            "px",
            Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef,
        )]);
        let args = vec![lit(0.0), lit(0.0), lit(0.0), lit(1.0)];
        let out = run(SpatialFunc::QuatNorm, args, &b);
        assert_eq!(out.len(), 0);
    }
}
