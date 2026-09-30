//! A `CASE` whose branches are nested (list, struct), or a comparison of nested values,
//! never compiles: the JIT's subset is scalar, so it must decline and leave the row to the
//! interpreter, which is the oracle. Pinned because the control plane lowers `CASE ... END`
//! without an `ELSE` to a typed NULL (`nullif(v, v)`) over those same branch values, and
//! because nested `=`/`<` now evaluate in the interpreter rather than raising.

use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, Int64Array, ListArray, RecordBatch, StructArray};
use arrow::datatypes::{DataType, Field, Int64Type, Schema};
use bc_expr::{BinaryOp, CaseBranch, Expr};

fn col(name: &str) -> Expr {
    Expr::Col { name: name.into() }
}

fn batch_with(value: ArrayRef) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("c", DataType::Boolean, false),
        Field::new("v", value.data_type().clone(), true),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(BooleanArray::from(vec![true, false])), value],
    )
    .expect("batch")
}

fn assert_declines_but_interpreter_runs(value: ArrayRef) {
    let batch = batch_with(value);
    for otherwise in [
        col("v"),
        Expr::NullIf {
            left: Box::new(col("v")),
            right: Box::new(col("v")),
        },
    ] {
        let case = Expr::Case {
            branches: vec![CaseBranch {
                when: col("c"),
                then: col("v"),
            }],
            otherwise: Box::new(otherwise),
        };
        assert!(
            bc_codegen::compile_expr(&case, &batch).is_err(),
            "a nested CASE must fall back to the interpreter"
        );
        let out = case.eval(&batch).expect("the interpreter evaluates it");
        assert_eq!(out.data_type(), batch.column(1).data_type());
    }
    for op in [
        BinaryOp::Eq,
        BinaryOp::Ne,
        BinaryOp::Lt,
        BinaryOp::Le,
        BinaryOp::Gt,
        BinaryOp::Ge,
    ] {
        let cmp = Expr::Binary {
            op,
            left: Box::new(col("v")),
            right: Box::new(col("v")),
        };
        assert!(
            bc_codegen::compile_expr(&cmp, &batch).is_err(),
            "a nested {op:?} must fall back to the interpreter"
        );
        let out = cmp.eval(&batch).expect("the interpreter evaluates it");
        assert_eq!(out.data_type(), &DataType::Boolean);
    }
}

#[test]
fn a_list_case_falls_back() {
    let v: ArrayRef = Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
        Some(vec![Some(1), Some(2)]),
        None,
    ]));
    assert_declines_but_interpreter_runs(v);
}

#[test]
fn a_struct_case_falls_back() {
    let x: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None]));
    let v: ArrayRef = Arc::new(StructArray::from(vec![(
        Arc::new(Field::new("x", DataType::Int64, true)),
        x,
    )]));
    assert_declines_but_interpreter_runs(v);
}
