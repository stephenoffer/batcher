//! A `CASE` whose arms are all literals, built in one pass instead of one array per arm.
//!
//! Bucketing a column into named bands — `CASE WHEN x < a THEN 'lo' WHEN x < b THEN 'mid' ELSE
//! 'hi' END`, grouped — is the shape a dashboard query is made of, and the general algorithm in
//! [`super::case`] is the wrong one for it. That algorithm materializes each arm's value over
//! the **whole batch** and then folds the arms together with `zip`, so a four-arm `CASE` over six
//! million rows builds eight full-length arrays to produce one. With `Utf8` arms that measured
//! 21.2 ms against DuckDB's 10.2 for `op-expr-case`, while the identical `CASE` returning
//! integers won at 6.8 ms — the difference is not the branching, it is the copying, and a string
//! costs the most to copy.
//!
//! When every arm is a literal the answer needs no arm arrays at all. The selections are already
//! a *partition* of the rows (`case` folds them that way so the arms combine order-independently),
//! so one pass over them assigns each row an arm index, and a second builds the column directly:
//! offsets from the arms' lengths for a string, a value write for a number. Two passes over the
//! rows, one allocation, no `zip`.
//!
//! **This computes exactly what the general path computes**, which the tests hold it to by
//! running both over the same batch. It declines — returning `None`, so the caller proceeds
//! unchanged — for anything it does not cover: a non-literal arm, arms of mixed families
//! (a numeric coercion the general path handles through `coerce_numeric`), or more arms than an
//! arm index can hold.

use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
use arrow::buffer::{OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, TimeUnit};

use crate::{CaseBranch, Expr, Literal};

/// Arms past this many leave the fast path; the index is one byte a row.
const MAX_ARMS: usize = 255;

/// The literal an arm yields, or `None` when the arm is not a literal.
fn arm_literal(expr: &Expr) -> Option<&Literal> {
    match expr {
        Expr::Lit { value } => Some(value),
        _ => None,
    }
}

/// One arm index per row: the arm that claimed it, or `arms - 1` (the `otherwise`) for a row no
/// branch claimed.
///
/// The selections partition the rows, so a row is written at most once and the order the arms
/// are applied in cannot change the result.
fn arm_of_each_row(selections: &[BooleanArray], n: usize, otherwise: u8) -> Vec<u8> {
    let mut arms = vec![otherwise; n];
    for (a, selection) in selections.iter().enumerate() {
        for row in selection.values().set_indices() {
            arms[row] = a as u8;
        }
    }
    arms
}

/// Build a `Utf8` column whose row `i` is `values[arms[i]]`.
///
/// The offsets come from the arms' own lengths, so the data buffer is sized exactly and each row
/// is one `extend_from_slice` of a string that is already in cache — there is no per-row
/// allocation and no re-copy of the accumulated column, which is what the `zip` fold pays.
fn build_utf8(arms: &[u8], values: &[&str]) -> ArrayRef {
    let lens: Vec<i32> = values.iter().map(|s| s.len() as i32).collect();
    let mut offsets: Vec<i32> = Vec::with_capacity(arms.len() + 1);
    let mut total: i32 = 0;
    offsets.push(0);
    for &a in arms {
        total += lens[a as usize];
        offsets.push(total);
    }
    let mut data: Vec<u8> = Vec::with_capacity(total as usize);
    for &a in arms {
        data.extend_from_slice(values[a as usize].as_bytes());
    }
    Arc::new(StringArray::new(
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        data.into(),
        None,
    ))
}

/// Build a primitive column whose row `i` is `values[arms[i]]`, as `dtype`.
fn build_primitive(arms: &[u8], values: &[i64], dtype: &DataType) -> ArrayRef {
    let raw: Vec<i64> = arms.iter().map(|&a| values[a as usize]).collect();
    let ints = Int64Array::new(ScalarBuffer::from(raw), None);
    if matches!(dtype, DataType::Int64) {
        return Arc::new(ints);
    }
    // Timestamp/Date are the same integers under a different type, so the cast is metadata only.
    arrow::compute::cast(&(Arc::new(ints) as ArrayRef), dtype).expect("int -> temporal is total")
}

/// Build a `Float64` column whose row `i` is `values[arms[i]]`.
fn build_float(arms: &[u8], values: &[f64]) -> ArrayRef {
    let raw: Vec<f64> = arms.iter().map(|&a| values[a as usize]).collect();
    Arc::new(Float64Array::new(ScalarBuffer::from(raw), None))
}

/// Build a `Boolean` column whose row `i` is `values[arms[i]]`.
fn build_bool(arms: &[u8], values: &[bool]) -> ArrayRef {
    Arc::new(BooleanArray::from_iter(
        arms.iter().map(|&a| Some(values[a as usize])),
    ))
}

/// The whole `CASE` when every arm is a literal of one family, or `None` to use the general path.
///
/// `selections[i]` are the rows branch `i` claimed and `unclaimed` the rows left to `otherwise`;
/// together they partition `0..n`, which is what `super::case::eval_case` establishes before
/// calling this.
pub(crate) fn eval_literal_case(
    branches: &[CaseBranch],
    otherwise: &Expr,
    selections: &[BooleanArray],
    n: usize,
) -> Option<ArrayRef> {
    if branches.len() >= MAX_ARMS {
        return None;
    }
    let mut literals: Vec<&Literal> = Vec::with_capacity(branches.len() + 1);
    for branch in branches {
        literals.push(arm_literal(&branch.then)?);
    }
    literals.push(arm_literal(otherwise)?);
    let arms = arm_of_each_row(selections, n, (literals.len() - 1) as u8);

    // One family per `CASE`, matching what the general path would produce. A mixed int/float
    // ladder is left to `coerce_numeric`, and a mixed string/number one is an error there —
    // either way this must not invent an answer of its own.
    if literals.iter().all(|l| matches!(l, Literal::Str(_))) {
        let values: Vec<&str> = literals
            .iter()
            .map(|l| match l {
                Literal::Str(s) => s.as_str(),
                _ => unreachable!("every arm matched Str above"),
            })
            .collect();
        return Some(build_utf8(&arms, &values));
    }
    if literals.iter().all(|l| matches!(l, Literal::Bool(_))) {
        let values: Vec<bool> = literals
            .iter()
            .map(|l| match l {
                Literal::Bool(v) => *v,
                _ => unreachable!("every arm matched Bool above"),
            })
            .collect();
        return Some(build_bool(&arms, &values));
    }
    if literals.iter().all(|l| matches!(l, Literal::Float(_))) {
        let values: Vec<f64> = literals
            .iter()
            .map(|l| match l {
                Literal::Float(v) => *v,
                _ => unreachable!("every arm matched Float above"),
            })
            .collect();
        return Some(build_float(&arms, &values));
    }
    let dtype = match literals[0] {
        Literal::Int(_) => DataType::Int64,
        Literal::Timestamp(_) => DataType::Timestamp(TimeUnit::Microsecond, None),
        Literal::Date(_) => DataType::Date32,
        _ => return None,
    };
    let want = std::mem::discriminant(literals[0]);
    if !literals.iter().all(|l| std::mem::discriminant(*l) == want) {
        return None;
    }
    let values: Vec<i64> = literals
        .iter()
        .map(|l| match l {
            Literal::Int(v) | Literal::Timestamp(v) => *v,
            Literal::Date(v) => i64::from(*v),
            _ => unreachable!("the discriminant check above admits only these"),
        })
        .collect();
    Some(build_primitive(&arms, &values, &dtype))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array, RecordBatch, StringArray};
    use arrow::datatypes::{Field, Schema};

    use super::*;
    use crate::{BinaryOp, CaseBranch, Literal};

    /// `x` walks 0..n so each arm of the ladders below claims a known, non-empty run, and the
    /// last rows fall through to `otherwise`.
    fn batch(n: usize) -> RecordBatch {
        let x = Int64Array::from_iter_values((0..n as i64).map(|i| i % 40));
        let schema = Schema::new(vec![Field::new("x", DataType::Int64, true)]);
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(x)]).expect("batch")
    }

    fn lt(v: i64) -> Expr {
        Expr::Binary {
            op: BinaryOp::Lt,
            left: Box::new(Expr::Col { name: "x".into() }),
            right: Box::new(Expr::Lit {
                value: Literal::Int(v),
            }),
        }
    }

    fn ladder(values: &[Literal]) -> (Vec<CaseBranch>, Expr) {
        let branches = values[..values.len() - 1]
            .iter()
            .enumerate()
            .map(|(i, v)| CaseBranch {
                when: lt((i as i64 + 1) * 10),
                then: Expr::Lit { value: v.clone() },
            })
            .collect();
        (
            branches,
            Expr::Lit {
                value: values[values.len() - 1].clone(),
            },
        )
    }

    /// The whole correctness argument: the one-pass build and the general `zip` fold are the
    /// same column. Run over every literal family, at a size that crosses several arms.
    #[test]
    fn the_literal_path_and_the_general_fold_agree() {
        let b = batch(1_000);
        let families: Vec<Vec<Literal>> = vec![
            vec![
                Literal::Str("a".into()),
                Literal::Str("bb".into()),
                Literal::Str(String::new()),
                Literal::Str("dddd".into()),
            ],
            vec![
                Literal::Int(1),
                Literal::Int(-2),
                Literal::Int(3),
                Literal::Int(4),
            ],
            vec![
                Literal::Float(1.5),
                Literal::Float(-0.0),
                Literal::Float(f64::INFINITY),
                Literal::Float(2.5),
            ],
            vec![
                Literal::Bool(true),
                Literal::Bool(false),
                Literal::Bool(true),
                Literal::Bool(false),
            ],
            vec![
                Literal::Date(0),
                Literal::Date(-1),
                Literal::Date(19_000),
                Literal::Date(1),
            ],
            vec![
                Literal::Timestamp(0),
                Literal::Timestamp(-1),
                Literal::Timestamp(1_700_000_000_000_000),
                Literal::Timestamp(5),
            ],
        ];
        for values in families {
            let (branches, otherwise) = ladder(&values);
            let expr = Expr::Case {
                branches: branches.clone(),
                otherwise: Box::new(otherwise.clone()),
            };
            let fast = expr.eval(&b).expect("fast path");
            // The general fold, reached by making one arm a non-literal that computes the same
            // value — `x - x + v` is `v` for every row, and is not an `Expr::Lit`.
            let general_branches: Vec<CaseBranch> = branches
                .iter()
                .enumerate()
                .map(|(i, br)| {
                    if i != 0 {
                        return br.clone();
                    }
                    CaseBranch {
                        when: br.when.clone(),
                        then: Expr::Case {
                            branches: vec![CaseBranch {
                                when: Expr::Lit {
                                    value: Literal::Bool(true),
                                },
                                then: br.then.clone(),
                            }],
                            otherwise: Box::new(br.then.clone()),
                        },
                    }
                })
                .collect();
            let general = Expr::Case {
                branches: general_branches,
                otherwise: Box::new(otherwise),
            }
            .eval(&b)
            .expect("general path");
            assert_eq!(fast.as_ref(), general.as_ref(), "{values:?}");
        }
    }

    /// Declines, each for its own reason, and the caller then runs the general path.
    #[test]
    fn a_shape_it_does_not_cover_declines() {
        let b = batch(16);
        // A non-literal arm.
        let mixed = Expr::Case {
            branches: vec![CaseBranch {
                when: lt(10),
                then: Expr::Col { name: "x".into() },
            }],
            otherwise: Box::new(Expr::Lit {
                value: Literal::Int(0),
            }),
        };
        assert!(mixed.eval(&b).is_ok(), "the general path still answers it");
        // Mixed families stay with `coerce_numeric`.
        let (branches, otherwise) = ladder(&[Literal::Int(1), Literal::Float(2.0)]);
        let selections = vec![BooleanArray::from(vec![true; 16])];
        assert!(eval_literal_case(&branches, &otherwise, &selections, 16).is_none());
    }

    /// An empty batch has no rows to assign and must still produce the right *type*, which is
    /// what every downstream breaker reads off a zero-row result.
    #[test]
    fn an_empty_batch_keeps_the_column_type() {
        let b = batch(0);
        let expr = Expr::Case {
            branches: vec![CaseBranch {
                when: lt(10),
                then: Expr::Lit {
                    value: Literal::Str("a".into()),
                },
            }],
            otherwise: Box::new(Expr::Lit {
                value: Literal::Str("b".into()),
            }),
        };
        let out = expr.eval(&b).expect("empty case");
        assert_eq!(out.len(), 0);
        assert_eq!(out.data_type(), &DataType::Utf8);
    }

    /// Strings of different lengths, including empty ones, must land at the right offsets.
    #[test]
    fn string_arms_of_different_lengths_land_at_the_right_offsets() {
        let b = batch(100);
        let expr = Expr::Case {
            branches: vec![
                CaseBranch {
                    when: lt(10),
                    then: Expr::Lit {
                        value: Literal::Str(String::new()),
                    },
                },
                CaseBranch {
                    when: lt(20),
                    then: Expr::Lit {
                        value: Literal::Str("xyz".into()),
                    },
                },
            ],
            otherwise: Box::new(Expr::Lit {
                value: Literal::Str("q".into()),
            }),
        };
        let out = expr.eval(&b).expect("case");
        let s = out.as_any().downcast_ref::<StringArray>().expect("utf8");
        for i in 0..100 {
            let x = i % 40;
            let want = if x < 10 {
                ""
            } else if x < 20 {
                "xyz"
            } else {
                "q"
            };
            assert_eq!(s.value(i), want, "row {i}");
        }
    }

    #[test]
    fn float_arms_keep_their_exact_bits() {
        let b = batch(8);
        let expr = Expr::Case {
            branches: vec![CaseBranch {
                when: lt(4),
                then: Expr::Lit {
                    value: Literal::Float(-0.0),
                },
            }],
            otherwise: Box::new(Expr::Lit {
                value: Literal::Float(0.0),
            }),
        };
        let out = expr.eval(&b).expect("case");
        let f = out.as_any().downcast_ref::<Float64Array>().expect("f64");
        assert!(f.value(0).is_sign_negative() && f.value(7).is_sign_positive());
    }
}
