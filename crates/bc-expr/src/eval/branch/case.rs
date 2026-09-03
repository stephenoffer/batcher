//! `CASE`: the first branch whose condition holds supplies the row's value.
//!
//! See the [module docs](super) for why the branches are evaluated selectively and what
//! that changes about errors.

use arrow::array::{ArrayRef, BooleanArray, RecordBatch};
use arrow::buffer::BooleanBuffer;
use arrow::compute::kernels::zip::zip;

use super::eval_over;
use crate::eval::coerce::{as_bool, coerce_numeric};
use crate::subset::truthy;
use crate::{CaseBranch, Expr, ExprError};

/// Evaluate a `CASE` expression, computing each branch only over the rows it selects.
pub(crate) fn eval_case(
    branches: &[CaseBranch],
    otherwise: &Expr,
    batch: &RecordBatch,
) -> Result<ArrayRef, ExprError> {
    let n = batch.num_rows();

    // Pass one: the conditions, folded into **disjoint** selections. `unclaimed` is the
    // rows no earlier branch has taken, so intersecting it with each condition is what
    // turns "first matching WHEN wins" into a partition of the rows — and the partition
    // is what makes pass two's combination order-independent.
    //
    // Every mask here is null-free (`truthy` folds a null condition into false, which is
    // the SQL rule: a `WHEN` that evaluates to NULL is not taken). That is why the fold
    // runs on the bit buffers directly rather than through the boolean kernels: with no
    // validity to reconcile there is nothing for them to do, and `unclaimed ^ selection`
    // removes exactly the newly claimed rows because a selection is always a subset of
    // what was unclaimed.
    let mut selections: Vec<BooleanArray> = Vec::with_capacity(branches.len());
    let mut unclaimed = BooleanBuffer::new_set(n);
    for branch in branches {
        // The condition is itself evaluated only over the rows still unclaimed, on the
        // same terms as a body: an expensive `WHEN` in a ladder — several `LIKE`s, a
        // regex per arm — is exactly as wasteful over rows an earlier arm already took.
        // The first arm always sees the whole batch (nothing is claimed yet), so the
        // common single-condition `CASE` pays nothing for this.
        let still = BooleanArray::new(unclaimed.clone(), None);
        let evaluated = eval_over(&branch.when, batch, &still, n)?;
        let mask = truthy(as_bool(&evaluated, "case")?);
        let selection = mask.values() & &unclaimed;
        unclaimed = &unclaimed ^ &selection;
        selections.push(BooleanArray::new(selection, None));
    }

    // Pass two: the bodies, each over its own rows. The `otherwise` arm takes whatever
    // no branch claimed.
    let mut acc = eval_over(otherwise, batch, &BooleanArray::new(unclaimed, None), n)?;
    for (branch, selection) in branches.iter().zip(&selections).rev() {
        let value = eval_over(&branch.then, batch, selection, n)?;
        // `zip` requires matching branch types; coerce Int64/Float64 (and decimal) to a
        // common numeric type the way COALESCE and the binary ops do, so a
        // `when(...).then(0).otherwise(x)` over a float column (or `clip`/`fill_nan`)
        // doesn't error on a mixed int/float literal.
        let (value, acc_c) = coerce_numeric(&value, &acc)?;
        acc = zip(selection, &value.as_ref(), &acc_c.as_ref())?;
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{BinaryOp, Literal};

    fn col(name: &str) -> Expr {
        Expr::Col { name: name.into() }
    }

    fn lit_int(v: i64) -> Expr {
        Expr::Lit {
            value: Literal::Int(v),
        }
    }

    fn cmp(op: BinaryOp, name: &str, v: i64) -> Expr {
        Expr::Binary {
            op,
            left: Box::new(col(name)),
            right: Box::new(lit_int(v)),
        }
    }

    /// `i` = 1..=5 then null; `f` = 1.5, null, 3.5, 4.5, null, 6.5; `s` = non-numeric text.
    fn sample() -> RecordBatch {
        let i = Int64Array::from(vec![Some(1), Some(2), Some(3), Some(4), Some(5), None]);
        let f = Float64Array::from(vec![Some(1.5), None, Some(3.5), Some(4.5), None, Some(6.5)]);
        let s = StringArray::from(vec!["a", "b", "c", "d", "e", "f"]);
        let schema = Schema::new(vec![
            Field::new("i", DataType::Int64, true),
            Field::new("f", DataType::Float64, true),
            Field::new("s", DataType::Utf8, true),
        ]);
        RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![
                std::sync::Arc::new(i),
                std::sync::Arc::new(f),
                std::sync::Arc::new(s),
            ],
        )
        .expect("sample batch")
    }

    fn floats(arr: &ArrayRef) -> Vec<Option<f64>> {
        let a = arr
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("float result");
        (0..a.len())
            .map(|i| (!a.is_null(i)).then(|| a.value(i)))
            .collect()
    }

    /// A branch no row selects is not evaluated, so a body that would fail on the rows it
    /// does not claim cannot fail the query. SQL says so and DuckDB agrees; this engine
    /// used to raise.
    #[test]
    fn an_unselected_case_branch_does_not_raise() {
        let batch = sample();
        let branches = vec![CaseBranch {
            when: cmp(BinaryOp::Lt, "i", 0),
            then: Expr::Cast {
                input: Box::new(col("s")),
                dtype: "int64".into(),
                try_cast: false,
            },
        }];
        let out = eval_case(&branches, &lit_int(7), &batch).expect("case");
        let a = out
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int result");
        assert_eq!(
            (0..a.len()).map(|i| a.value(i)).collect::<Vec<_>>(),
            vec![7; 6]
        );
    }

    /// A `WHEN` is not evaluated on rows an earlier arm already took, so a condition that
    /// would fail on those rows cannot fail the query. SQL evaluates a `CASE` ladder in
    /// order and stops at the first match; DuckDB does too.
    #[test]
    fn a_condition_after_a_matching_arm_does_not_raise() {
        let batch = sample();
        let numeric = Expr::Binary {
            op: BinaryOp::Gt,
            left: Box::new(Expr::Cast {
                input: Box::new(col("s")),
                dtype: "int64".into(),
                try_cast: false,
            }),
            right: Box::new(lit_int(0)),
        };
        let branches = vec![
            // `i IS NULL OR i IS NOT NULL` is true on every row, so nothing is left for
            // the second condition to be asked about.
            CaseBranch {
                when: Expr::Binary {
                    op: BinaryOp::Or,
                    left: Box::new(Expr::IsNull {
                        input: Box::new(col("i")),
                    }),
                    right: Box::new(Expr::IsNotNull {
                        input: Box::new(col("i")),
                    }),
                },
                then: lit_int(1),
            },
            CaseBranch {
                when: numeric,
                then: lit_int(2),
            },
        ];
        let out = eval_case(&branches, &lit_int(3), &batch).expect("case");
        let a = out
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int result");
        assert_eq!(
            (0..a.len()).map(|i| a.value(i)).collect::<Vec<_>>(),
            vec![1; 6]
        );
    }

    /// The other half: a branch a row *does* select still raises.
    #[test]
    fn a_selected_case_branch_still_raises() {
        let batch = sample();
        let branches = vec![CaseBranch {
            when: cmp(BinaryOp::Gt, "i", 0),
            then: Expr::Cast {
                input: Box::new(col("s")),
                dtype: "int64".into(),
                try_cast: false,
            },
        }];
        assert!(eval_case(&branches, &lit_int(7), &batch).is_err());
    }

    /// Overlapping conditions still resolve first-match-wins after the rewrite turned the
    /// masks into a disjoint partition.
    #[test]
    fn the_first_matching_branch_wins() {
        let batch = sample();
        let branches = vec![
            CaseBranch {
                when: cmp(BinaryOp::Ge, "i", 3),
                then: Expr::Lit {
                    value: Literal::Float(30.0),
                },
            },
            CaseBranch {
                when: cmp(BinaryOp::Ge, "i", 1),
                then: Expr::Lit {
                    value: Literal::Float(10.0),
                },
            },
        ];
        let out = eval_case(
            &branches,
            &Expr::Lit {
                value: Literal::Float(-1.0),
            },
            &batch,
        )
        .expect("case");
        assert_eq!(
            floats(&out),
            vec![
                Some(10.0),
                Some(10.0),
                Some(30.0),
                Some(30.0),
                Some(30.0),
                // `i` is null on the last row, so neither condition is true.
                Some(-1.0)
            ]
        );
    }

    /// A `WHEN` that evaluates to NULL is not taken — it falls through to `ELSE` rather
    /// than letting a null mask pick the `THEN`.
    #[test]
    fn a_null_condition_falls_through() {
        let batch = sample();
        let branches = vec![CaseBranch {
            when: cmp(BinaryOp::Gt, "i", 4),
            then: Expr::Lit {
                value: Literal::Float(1.0),
            },
        }];
        let out = eval_case(
            &branches,
            &Expr::Lit {
                value: Literal::Float(0.0),
            },
            &batch,
        )
        .expect("case");
        assert_eq!(
            floats(&out),
            vec![
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(1.0),
                Some(0.0)
            ]
        );
    }
}
