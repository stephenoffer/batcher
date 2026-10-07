//! `<integer column> / <integer literal>` and `% <integer literal>` without a per-row check.
//!
//! The array path ([`super::binary::eval_binary`]) has to be ready for any divisor column, so
//! it materializes the literal as an N-row array, scans it for zeros (which it nulls rather
//! than letting the CPU trap), and runs arrow's *checked* kernel, which tests every row for
//! the one overflowing pair `i64::MIN / -1`. Against a literal all three are decided once: a
//! divisor that is neither `0` nor `-1` can neither trap nor overflow on any row, so the
//! quotient or remainder is a plain loop. A power-of-two divisor -- the `% 2` of every
//! even/odd split -- needs no division instruction at all.
//!
//! `x % 2 = 0` over six million rows was 13% of a whole full-outer-join query's CPU, more
//! than its hash table build.
//!
//! **Bit-identical to the array path**, which is what lets it sit in the oracle: both
//! truncate toward zero (Rust's, arrow's and DuckDB's integer `/` and `%`), and a null row
//! stays null. The divisors the array path treats specially (`0`, `-1`) are declined here and
//! keep it.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Array};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;

use crate::eval::binary::eval_binary;
use crate::{BinaryOp, Expr, ExprError, Literal};

/// `left / lit` or `left % lit` for an integer literal on the right, or `None` for any other
/// shape. A left operand that turns out not to be `Int64` is answered by the array path from
/// the already-evaluated column, so nothing is evaluated twice.
pub(crate) fn try_int_div_mod_literal(
    op: BinaryOp,
    left: &Expr,
    right: &Expr,
    batch: &RecordBatch,
) -> Result<Option<ArrayRef>, ExprError> {
    let is_div = match op {
        BinaryOp::Div => true,
        BinaryOp::Mod => false,
        _ => return Ok(None),
    };
    let Expr::Lit {
        value: lit @ Literal::Int(d),
    } = right
    else {
        return Ok(None);
    };
    let d = *d;
    if d == 0 || d == -1 {
        return Ok(None);
    }
    let l = left.eval(batch)?;
    if l.data_type() != &DataType::Int64 {
        return eval_binary(op, &l, &lit.to_array(l.len())).map(Some);
    }
    let a = l
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("checked Int64 above");
    Ok(Some(Arc::new(div_mod_i64(is_div, a, d))))
}

/// The quotient (`is_div`) or remainder of every row by `d`, truncating toward zero. `d` is
/// neither `0` nor `-1`, so no row can trap or overflow; values under null slots are computed
/// and masked like any other.
fn div_mod_i64(is_div: bool, a: &Int64Array, d: i64) -> Int64Array {
    let v = a.values();
    let out: Vec<i64> = if d > 0 && d & (d - 1) == 0 {
        // A power of two: a shift and a mask, corrected toward zero for a negative row. The
        // correction is `(x >> 63) & m`, all ones in `m`'s bits exactly when `x` is negative.
        let (k, m) = (d.trailing_zeros(), d - 1);
        if is_div {
            v.iter().map(|&x| (x + ((x >> 63) & m)) >> k).collect()
        } else {
            v.iter()
                .map(|&x| {
                    let r = x & m;
                    // A negative `x` with a nonzero low part has remainder `r - d`.
                    r - ((x >> 63) & -i64::from(r != 0) & d)
                })
                .collect()
        }
    } else if is_div {
        v.iter().map(|&x| x / d).collect()
    } else {
        v.iter().map(|&x| x % d).collect()
    };
    Int64Array::new(out.into(), a.nulls().cloned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Field, Schema};

    fn batch(values: Vec<Option<i64>>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values))]).unwrap()
    }

    /// Every divisor shape against the array path, row for row, over the values where
    /// truncation, sign and overflow go wrong.
    #[test]
    fn matches_the_array_path_on_every_divisor_shape() {
        let mut values: Vec<Option<i64>> = vec![
            Some(i64::MIN),
            Some(i64::MIN + 1),
            Some(i64::MAX),
            Some(i64::MAX - 1),
            Some(0),
            None,
            Some(1),
            Some(-1),
        ];
        values.extend((-70..70).map(Some));
        values.extend((0..64).flat_map(|k| {
            let p = 1i64 << k;
            [Some(p), Some(p.wrapping_neg()), Some(p.wrapping_sub(1))]
        }));
        let b = batch(values);
        let col = Expr::Col { name: "x".into() };
        let divisors = [
            1,
            2,
            3,
            4,
            7,
            8,
            1 << 20,
            1 << 62,
            -2,
            -3,
            -4,
            i64::MAX,
            i64::MIN,
            i64::MIN + 1,
        ];
        for d in divisors {
            for op in [BinaryOp::Div, BinaryOp::Mod] {
                let lit = Expr::Lit {
                    value: Literal::Int(d),
                };
                let fast = try_int_div_mod_literal(op, &col, &lit, &b)
                    .unwrap()
                    .expect("served");
                let slow =
                    eval_binary(op, b.column(0), &Literal::Int(d).to_array(b.num_rows())).unwrap();
                assert_eq!(fast.as_ref(), slow.as_ref(), "{op:?} {d}");
            }
        }
    }

    /// The divisors the array path gives special meaning to are left to it.
    #[test]
    fn declines_zero_and_minus_one_and_other_shapes() {
        let b = batch(vec![Some(i64::MIN), Some(4)]);
        let col = Expr::Col { name: "x".into() };
        for d in [0, -1] {
            let lit = Expr::Lit {
                value: Literal::Int(d),
            };
            assert!(try_int_div_mod_literal(BinaryOp::Mod, &col, &lit, &b)
                .unwrap()
                .is_none());
        }
        let two = Expr::Lit {
            value: Literal::Int(2),
        };
        assert!(try_int_div_mod_literal(BinaryOp::Add, &col, &two, &b)
            .unwrap()
            .is_none());
        assert!(try_int_div_mod_literal(BinaryOp::Mod, &two, &col, &b)
            .unwrap()
            .is_none());
    }
}
