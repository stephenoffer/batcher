//! `COALESCE`: the first argument with a value supplies the row's value.
//!
//! See the [module docs](super) for why the arguments are evaluated selectively and what
//! that changes about errors.

use arrow::array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow::buffer::BooleanBuffer;
use arrow::compute::is_not_null;
use arrow::compute::kernels::boolean;
use arrow::compute::kernels::zip::zip;

use super::eval_over;
use crate::eval::coerce::coerce_numeric;
use crate::{Expr, ExprError};

/// Evaluate `COALESCE`, computing each argument only over the rows that still need a
/// value — which may be none of them, and often is.
pub(crate) fn eval_coalesce(inputs: &[Expr], batch: &RecordBatch) -> Result<ArrayRef, ExprError> {
    let Some((first, rest)) = inputs.split_first() else {
        return Err(ExprError::MissingArgument {
            func: "coalesce".to_string(),
            arg: "inputs",
        });
    };
    let n = batch.num_rows();
    let mut acc = first.eval(batch)?;
    for expr in rest {
        // Nulls in `acc` are the rows still looking for a value, and an argument is
        // evaluated only over those — down to *none* of them, where `eval_over` answers
        // from an empty gather without doing any per-row work. That is the whole saving on
        // a mostly-non-null leading argument, and the reason a later argument cannot raise
        // on rows that never needed it.
        //
        // The loop deliberately does **not** stop early when nothing is left to fill.
        // `COALESCE`'s result *type* is the promotion of every argument's type, whether or
        // not a row ever reaches one: `coalesce(int_with_no_nulls, float)` is DOUBLE in
        // DuckDB and was DOUBLE here, and a version of this that broke out of the loop
        // returned BIGINT — the right values under a wrong column type, which
        // `assert_same` is int/float tolerant by design and could never have caught.
        let needed = still_null(&acc)
            .unwrap_or_else(|| BooleanArray::new(BooleanBuffer::new_unset(n), None));
        let value = eval_over(expr, batch, &needed, n)?;
        // Promote mixed numeric inputs to a common type (e.g. coalesce(int, float) →
        // float) so `zip` sees matching types, matching SQL coercion.
        let (value, acc_c) = coerce_numeric(&value, &acc)?;
        // Take this argument only where the row both still needs a value and this
        // argument has one. The `needed` half is load-bearing and not implied by the
        // other: an argument evaluated at full width (the heuristic declined to gather)
        // is non-null on rows that were already filled, and taking it there overwrote
        // them — `coalesce(f, i, 0)` returned `i` on every row `f` had a value for.
        let present = boolean::and(&needed, &is_not_null(&value)?)?;
        acc = zip(&present, &value.as_ref(), &acc_c.as_ref())?;
    }
    Ok(acc)
}

/// The rows of `arr` that are still null, or `None` when none are.
///
/// **`logical_nulls`, not `nulls`.** An all-null column reads as Arrow's `Null` type, whose
/// nullity is implied by the type rather than carried in a validity buffer — `nulls()`
/// answers `None` for it, meaning "no null buffer", which is indistinguishable here from
/// "no nulls". Reading it that way marked *no* row as needing a fallback, so
/// `coalesce(v, k)` over a `Null`-typed `v` returned all nulls instead of falling through
/// to `k`. The fold this replaced could not have the bug because it tested each argument
/// with `is_not_null`, which is logical-null aware; the selective rewrite reintroduced it
/// by asking the accumulator for its buffer instead. `logical_nulls` is also what keeps
/// this right for dictionary- and run-end-encoded inputs, whose nulls live in the values
/// array rather than the top-level one.
fn still_null(arr: &ArrayRef) -> Option<BooleanArray> {
    let nulls = arr.logical_nulls()?;
    if nulls.null_count() == 0 {
        return None;
    }
    Some(BooleanArray::new(!nulls.inner(), None))
}

#[cfg(test)]
mod tests {
    use arrow::array::{Float64Array, Int64Array, NullArray};
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::Literal;

    fn col(name: &str) -> Expr {
        Expr::Col { name: name.into() }
    }

    /// `i` = 1..=5 then null; `f` = 1.5, null, 3.5, 4.5, null, 6.5.
    fn sample() -> RecordBatch {
        let i = Int64Array::from(vec![Some(1), Some(2), Some(3), Some(4), Some(5), None]);
        let f = Float64Array::from(vec![Some(1.5), None, Some(3.5), Some(4.5), None, Some(6.5)]);
        let schema = Schema::new(vec![
            Field::new("i", DataType::Int64, true),
            Field::new("f", DataType::Float64, true),
        ]);
        RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![std::sync::Arc::new(i), std::sync::Arc::new(f)],
        )
        .expect("sample batch")
    }

    /// The result type is the promotion of *every* argument's type, even when no row
    /// reaches the argument that widens it.
    ///
    /// `coalesce(i, f)` over an `i` with no nulls needs nothing from `f`, and an
    /// implementation that stopped there returned BIGINT where DuckDB returns DOUBLE.
    /// The values are identical either way, and `assert_same` tolerates int/float, so the
    /// column type is the only thing that can catch it — asserted here rather than in the
    /// differential suite for exactly that reason.
    #[test]
    fn the_result_type_promotes_over_arguments_no_row_reaches() {
        let batch = sample();
        let full = Expr::Binary {
            op: crate::BinaryOp::Add,
            left: Box::new(col("i")),
            right: Box::new(Expr::Lit {
                value: Literal::Int(0),
            }),
        };
        // `i + 0` is null only on the last row, so give the fallback nothing to do at all
        // by coalescing a literal first.
        let out = eval_coalesce(
            &[
                Expr::Lit {
                    value: Literal::Int(1),
                },
                full,
                col("f"),
            ],
            &batch,
        )
        .expect("coalesce");
        assert_eq!(out.data_type(), &DataType::Float64);
    }

    /// A later `COALESCE` argument fills only the rows still missing a value.
    ///
    /// This is the one the rewrite got wrong first: the argument was evaluated at full
    /// width (the gather heuristic declined a two-row selection), so it was non-null on
    /// rows the first argument had already answered, and taking it there replaced them.
    /// `coalesce(f, i, 0)` came back as `i` everywhere instead of `f` where `f` had a
    /// value. Nothing about the *lazy* half of the change is visible here, which is why
    /// it needs its own test rather than riding on the error-laziness one.
    #[test]
    fn a_later_coalesce_argument_fills_only_the_missing_rows() {
        let batch = sample();
        let out = eval_coalesce(
            &[
                col("f"),
                col("i"),
                Expr::Lit {
                    value: Literal::Int(0),
                },
            ],
            &batch,
        )
        .expect("coalesce");
        let a = out
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("float result");
        let got: Vec<Option<f64>> = (0..a.len())
            .map(|i| (!a.is_null(i)).then(|| a.value(i)))
            .collect();
        assert_eq!(
            got,
            vec![
                Some(1.5),
                Some(2.0),
                Some(3.5),
                Some(4.5),
                Some(5.0),
                Some(6.5)
            ]
        );
    }

    /// An all-`Null`-typed argument falls through to the next one.
    ///
    /// A column every value of which is missing reads as Arrow's `Null` type, which carries
    /// no validity buffer -- so `nulls()` answers `None` for it, exactly as it does for a
    /// column with no nulls at all. Reading nullity that way made `still_null` report that no
    /// row wanted a fallback, and `coalesce(v, k)` returned all nulls instead of `k`. The
    /// values are what catch this: the *type* is right either way, because `coerce_numeric`
    /// casts the `Null` accumulator to the other argument's type regardless.
    #[test]
    fn an_all_null_typed_argument_falls_through_to_the_next() {
        let k = Int64Array::from(vec![Some(1), Some(2)]);
        let schema = Schema::new(vec![
            Field::new("v", DataType::Null, true),
            Field::new("k", DataType::Int64, true),
        ]);
        let batch = RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![
                std::sync::Arc::new(NullArray::new(2)),
                std::sync::Arc::new(k),
            ],
        )
        .expect("null-typed batch");

        let out = eval_coalesce(&[col("v"), col("k")], &batch).expect("coalesce");
        let a = out
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int result");
        let got: Vec<Option<i64>> = (0..a.len())
            .map(|i| (!a.is_null(i)).then(|| a.value(i)))
            .collect();
        assert_eq!(got, vec![Some(1), Some(2)]);
    }
}
