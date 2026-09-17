//! `<string column> <cmp> <string literal>` from an 8-byte big-endian prefix.
//!
//! Arrow's generic kernel compares each row to the literal with a slice `cmp`, which is an
//! out-of-line `memcmp` call per row. A `LIKE 'the%'` that Kyber has made sargable becomes
//! `s >= 'the' AND s < 'thf'` — two of those per row — and over TPC-H's 6M `l_comment` values
//! that measured **79 ms of single-threaded CPU, 65% of it in `memcmp` and the kernel around
//! it**, for a question the first three bytes answer.
//!
//! Reading a string's first eight bytes as a big-endian `u64`, zero-padded past its length,
//! orders it exactly as a byte-wise comparison orders those eight bytes. When the two prefixes
//! differ, that is the whole answer, and on a selective range it is nearly every row. When they
//! are equal, one more fact decides it: if either string is eight bytes or shorter, the shorter
//! one is the longer one's prefix followed by zero bytes, so the *lengths* order them. Only two
//! strings both longer than eight bytes, agreeing on all eight, fall back to comparing slices.
//! Byte-wise order is exactly Arrow's order for `Utf8`, so the result is identical, nulls
//! included: a null row stays null.
//!
//! That alone did not move the query: the kernel then spent its time *waiting on memory*, since
//! the column is 160 MB of bytes and the two comparisons walked it twice. [`try_string_range`]
//! answers `s >= 'the' AND s < 'thf'` from one walk, loading each row's prefix once for both
//! bounds.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, GenericStringArray, OffsetSizeTrait};
use arrow::buffer::BooleanBuffer;
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;

use crate::{BinaryOp, Expr, ExprError, Literal};

/// The comparison, or `None` when this kernel does not serve the pair: the types are not the
/// same `Utf8`/`LargeUtf8`, the literal is null, or `op` is not a comparison. `lit_on_right`
/// says which side of `op` the literal is on.
pub(crate) fn string_scalar_cmp(
    op: BinaryOp,
    arr: &ArrayRef,
    lit_arr: &ArrayRef,
    lit_on_right: bool,
) -> Option<ArrayRef> {
    if arr.data_type() != lit_arr.data_type() || lit_arr.is_null(0) {
        return None;
    }
    let op = row_op(op, lit_on_right)?;
    let bits = match arr.data_type() {
        DataType::Utf8 => compare::<i32>(arr.as_string(), lit_arr.as_string::<i32>().value(0), op),
        DataType::LargeUtf8 => {
            compare::<i64>(arr.as_string(), lit_arr.as_string::<i64>().value(0), op)
        }
        _ => return None,
    };
    Some(Arc::new(BooleanArray::new(bits, arr.nulls().cloned())))
}

/// `left AND right` in one pass over the column, when both are a comparison of the **same**
/// string column with a string literal, one bounding it below (`>`/`>=`) and one above
/// (`<`/`<=`) — in either order and with each literal on either side. `None` for every other
/// shape, which the caller then evaluates as it always has.
///
/// Identical to the two-kernel evaluation, nulls included: where the column is null both
/// comparisons are null and so is their Kleene `AND`; everywhere else both are known booleans
/// and so is their conjunction.
pub(crate) fn try_string_range(
    left: &Expr,
    right: &Expr,
    batch: &RecordBatch,
) -> Result<Option<ArrayRef>, ExprError> {
    let (Some((col_l, op_l, lit_l)), Some((col_r, op_r, lit_r))) =
        (bound_of(left), bound_of(right))
    else {
        return Ok(None);
    };
    if col_l != col_r {
        return Ok(None);
    }
    let ((lo_op, lo), (hi_op, hi)) = match (is_lower(op_l), is_lower(op_r)) {
        (Some(true), Some(false)) => ((op_l, lit_l), (op_r, lit_r)),
        (Some(false), Some(true)) => ((op_r, lit_r), (op_l, lit_l)),
        _ => return Ok(None),
    };
    // Read the column the way `Expr::Col` does (a dictionary decodes at the leaf), but only
    // once the shape has matched, so a declined `AND` has paid nothing.
    let arr = Expr::Col {
        name: col_l.to_owned(),
    }
    .eval(batch)?;
    let bits = match arr.data_type() {
        DataType::Utf8 => range::<i32>(arr.as_string(), (lo_op, lo), (hi_op, hi)),
        DataType::LargeUtf8 => range::<i64>(arr.as_string(), (lo_op, lo), (hi_op, hi)),
        _ => return Ok(None),
    };
    Ok(Some(Arc::new(BooleanArray::new(
        bits,
        arr.nulls().cloned(),
    ))))
}

/// `(column, op, literal)` for `col OP 'lit'` or `'lit' OP col`, with `op` read as `col OP lit`.
fn bound_of(expr: &Expr) -> Option<(&str, BinaryOp, &str)> {
    let Expr::Binary { op, left, right } = expr else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (
            Expr::Col { name },
            Expr::Lit {
                value: Literal::Str(s),
            },
        ) => Some((name, row_op(*op, true)?, s)),
        (
            Expr::Lit {
                value: Literal::Str(s),
            },
            Expr::Col { name },
        ) => Some((name, row_op(*op, false)?, s)),
        _ => None,
    }
}

/// `Some(true)` for a lower bound, `Some(false)` for an upper one, `None` otherwise.
fn is_lower(op: BinaryOp) -> Option<bool> {
    match op {
        BinaryOp::Gt | BinaryOp::Ge => Some(true),
        BinaryOp::Lt | BinaryOp::Le => Some(false),
        _ => None,
    }
}

/// The comparison read as `row OP literal`: a literal on the left mirrors the operator.
fn row_op(op: BinaryOp, lit_on_right: bool) -> Option<BinaryOp> {
    use BinaryOp::{Eq, Ge, Gt, Le, Lt, Ne};
    Some(match (op, lit_on_right) {
        (Eq | Ne, _) | (Lt | Le | Gt | Ge, true) => op,
        (Lt, false) => Gt,
        (Le, false) => Ge,
        (Gt, false) => Lt,
        (Ge, false) => Le,
        _ => return None,
    })
}

/// A literal prepared for comparison: its bytes and its eight-byte prefix.
struct Needle<'a> {
    bytes: &'a [u8],
    prefix: u64,
}

impl<'a> Needle<'a> {
    fn new(lit: &'a str) -> Self {
        let bytes = lit.as_bytes();
        Self {
            bytes,
            prefix: prefix(bytes, 0, bytes.len()),
        }
    }

    /// This literal's [`short_key`] when it is at most eight bytes, else `None`.
    fn short_key(&self) -> Option<u128> {
        (self.bytes.len() <= 8).then(|| short_key(self.bytes, 0, self.bytes.len()))
    }

    /// The row `values[start..end]`, whose prefix is `row_prefix`, ordered against this literal.
    #[inline(always)]
    fn order(&self, values: &[u8], start: usize, end: usize, row_prefix: u64) -> Ordering {
        match row_prefix.cmp(&self.prefix) {
            Ordering::Equal if (end - start).min(self.bytes.len()) <= 8 => {
                (end - start).cmp(&self.bytes.len())
            }
            Ordering::Equal => values[start..end].cmp(self.bytes),
            decided => decided,
        }
    }
}

/// Whether `ord` satisfies `op`, for the six comparisons.
#[inline(always)]
fn holds(op: BinaryOp, ord: Ordering) -> bool {
    match op {
        BinaryOp::Eq => ord.is_eq(),
        BinaryOp::Ne => ord.is_ne(),
        BinaryOp::Lt => ord.is_lt(),
        BinaryOp::Le => ord.is_le(),
        BinaryOp::Gt => ord.is_gt(),
        _ => ord.is_ge(),
    }
}

fn compare<O: OffsetSizeTrait>(
    strings: &GenericStringArray<O>,
    lit: &str,
    op: BinaryOp,
) -> BooleanBuffer {
    let needle = Needle::new(lit);
    if let Some(key) = needle.short_key() {
        // Short literal: every row orders by one integer compare, with no branch on the data.
        return match op {
            BinaryOp::Eq => fill_short(strings, |row| row == key),
            BinaryOp::Ne => fill_short(strings, |row| row != key),
            BinaryOp::Lt => fill_short(strings, |row| row < key),
            BinaryOp::Le => fill_short(strings, |row| row <= key),
            BinaryOp::Gt => fill_short(strings, |row| row > key),
            _ => fill_short(strings, |row| row >= key),
        };
    }
    let offsets = strings.value_offsets();
    let values = strings.values().as_slice();
    let ord = |i: usize| {
        let (start, end) = (offsets[i].as_usize(), offsets[i + 1].as_usize());
        needle.order(values, start, end, prefix(values, start, end - start))
    };
    let n = strings.len();
    // One closure per operator, so the operator is not a branch inside the row loop.
    match op {
        BinaryOp::Eq => BooleanBuffer::collect_bool(n, |i| ord(i).is_eq()),
        BinaryOp::Ne => BooleanBuffer::collect_bool(n, |i| ord(i).is_ne()),
        BinaryOp::Lt => BooleanBuffer::collect_bool(n, |i| ord(i).is_lt()),
        BinaryOp::Le => BooleanBuffer::collect_bool(n, |i| ord(i).is_le()),
        BinaryOp::Gt => BooleanBuffer::collect_bool(n, |i| ord(i).is_gt()),
        _ => BooleanBuffer::collect_bool(n, |i| ord(i).is_ge()),
    }
}

fn range<O: OffsetSizeTrait>(
    strings: &GenericStringArray<O>,
    (lo_op, lo): (BinaryOp, &str),
    (hi_op, hi): (BinaryOp, &str),
) -> BooleanBuffer {
    let (lo, hi) = (Needle::new(lo), Needle::new(hi));
    if let (Some(lo_key), Some(hi_key)) = (lo.short_key(), hi.short_key()) {
        let strict_lo = matches!(lo_op, BinaryOp::Gt);
        let strict_hi = matches!(hi_op, BinaryOp::Lt);
        return match (strict_lo, strict_hi) {
            (false, true) => fill_short(strings, |row| (row >= lo_key) & (row < hi_key)),
            (false, false) => fill_short(strings, |row| (row >= lo_key) & (row <= hi_key)),
            (true, true) => fill_short(strings, |row| (row > lo_key) & (row < hi_key)),
            (true, false) => fill_short(strings, |row| (row > lo_key) & (row <= hi_key)),
        };
    }
    let offsets = strings.value_offsets();
    let values = strings.values().as_slice();
    let n = strings.len();
    // Filled a word at a time rather than through `collect_bool`, whose per-row closure did
    // not inline: a full call per row, in a loop whose whole budget is a few nanoseconds.
    let mut words = vec![0u64; n.div_ceil(64)];
    for (w, word) in words.iter_mut().enumerate() {
        let base = w * 64;
        let mut bits = 0u64;
        for (bit, pair) in offsets[base..=(base + 64).min(n)].windows(2).enumerate() {
            let (start, end) = (pair[0].as_usize(), pair[1].as_usize());
            let row = prefix(values, start, end - start);
            // A prefix equal to neither bound decides both from the `u64` compare alone, which
            // is nearly every row; only a tie reads past the prefix.
            let inside = if row != lo.prefix && row != hi.prefix {
                holds(lo_op, row.cmp(&lo.prefix)) & holds(hi_op, row.cmp(&hi.prefix))
            } else {
                holds(lo_op, lo.order(values, start, end, row))
                    & holds(hi_op, hi.order(values, start, end, row))
            };
            bits |= u64::from(inside) << bit;
        }
        *word = bits;
    }
    BooleanBuffer::new(words.into(), 0, n)
}

/// One bit per row, `test` applied to each row's [`short_key`], filled a word at a time.
#[inline(always)]
fn fill_short<O: OffsetSizeTrait>(
    strings: &GenericStringArray<O>,
    test: impl Fn(u128) -> bool,
) -> BooleanBuffer {
    let offsets = strings.value_offsets();
    let values = strings.values().as_slice();
    let n = strings.len();
    let mut words = vec![0u64; n.div_ceil(64)];
    for (w, word) in words.iter_mut().enumerate() {
        let base = w * 64;
        let mut bits = 0u64;
        for (bit, pair) in offsets[base..=(base + 64).min(n)].windows(2).enumerate() {
            let (start, end) = (pair[0].as_usize(), pair[1].as_usize());
            bits |= u64::from(test(short_key(values, start, end - start))) << bit;
        }
        *word = bits;
    }
    BooleanBuffer::new(words.into(), 0, n)
}

/// A row's order against any literal of at most eight bytes, as one integer: its prefix, then
/// its length clamped to nine.
///
/// For such a literal, a row with a different prefix is ordered by the prefix, and a row with
/// the same prefix is ordered by length — see the module note — and every row longer than
/// eight bytes is longer than the literal, so lengths past eight need not be told apart.
#[inline(always)]
fn short_key(values: &[u8], start: usize, len: usize) -> u128 {
    (u128::from(prefix(values, start, len)) << 8) | len.min(9) as u128
}

/// The first eight bytes of `bytes[start..start + len]` as a big-endian `u64`, zero past `len`.
#[inline(always)]
fn prefix(bytes: &[u8], start: usize, len: usize) -> u64 {
    let word = match bytes.get(start..start + 8) {
        Some(eight) => u64::from_be_bytes(eight.try_into().expect("eight bytes")),
        // The buffer ends within eight bytes of this string: copy what there is.
        None => {
            let mut buf = [0u8; 8];
            let tail = &bytes[start..start + len.min(8)];
            buf[..tail.len()].copy_from_slice(tail);
            return u64::from_be_bytes(buf);
        }
    };
    // Keep the top `len` bytes. A shift of 64 is not defined, so a zero length clears it all.
    word & u64::MAX
        .checked_shl(8 * (8 - len.min(8)) as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Datum;
    use arrow::array::{LargeStringArray, StringArray};
    use arrow::compute::kernels::cmp;

    const OPS: [BinaryOp; 6] = [
        BinaryOp::Eq,
        BinaryOp::Ne,
        BinaryOp::Lt,
        BinaryOp::Le,
        BinaryOp::Gt,
        BinaryOp::Ge,
    ];

    fn arrow_oracle(op: BinaryOp, l: &dyn Datum, r: &dyn Datum) -> BooleanArray {
        match op {
            BinaryOp::Eq => cmp::eq(l, r),
            BinaryOp::Ne => cmp::neq(l, r),
            BinaryOp::Lt => cmp::lt(l, r),
            BinaryOp::Le => cmp::lt_eq(l, r),
            BinaryOp::Gt => cmp::gt(l, r),
            _ => cmp::gt_eq(l, r),
        }
        .expect("comparable")
    }

    /// Strings chosen to straddle every branch: empty, shorter/equal/longer than eight bytes,
    /// embedded and trailing NUL bytes (which zero padding must not confuse with an end), a
    /// shared eight-byte prefix with a later difference, multi-byte UTF-8, and nulls — held to
    /// Arrow's own kernel, both operand orders, sliced and unsliced, both offset widths.
    #[test]
    fn agrees_with_arrow_on_every_operator_and_edge() {
        let rows: Vec<Option<&str>> = vec![
            Some(""),
            Some("t"),
            Some("th"),
            Some("the"),
            Some("the\0"),
            Some("the\0\0\0\0\0\0"),
            Some("thf"),
            Some("thea"),
            Some("abcdefgh"),
            Some("abcdefgh\0"),
            Some("abcdefghi"),
            Some("abcdefgha"),
            Some("abcdefg"),
            Some("abcdefgi"),
            Some("é"),
            Some("\u{ffff}zz"),
            None,
            Some("zzzzzzzzzzzzzzzzzzzzz"),
        ];
        let literals = [
            "",
            "t",
            "the",
            "the\0",
            "thf",
            "abcdefgh",
            "abcdefgh\0",
            "abcdefghi",
            "abcdefg",
            "é",
            "zzzzzzzzzzzzzzzzzzzza",
        ];
        let small = StringArray::from(rows.clone());
        let large = LargeStringArray::from(rows);
        for (arr, sliced) in [
            (
                Arc::new(small.clone()) as ArrayRef,
                Arc::new(small.slice(3, 12)) as ArrayRef,
            ),
            (
                Arc::new(large.clone()) as ArrayRef,
                Arc::new(large.slice(3, 12)) as ArrayRef,
            ),
        ] {
            for lit in literals {
                let lit_arr: ArrayRef = match arr.data_type() {
                    DataType::Utf8 => Arc::new(StringArray::from(vec![lit])),
                    _ => Arc::new(LargeStringArray::from(vec![lit])),
                };
                let scalar = arrow::array::Scalar::new(lit_arr.clone());
                for column in [&arr, &sliced] {
                    for op in OPS {
                        for lit_on_right in [true, false] {
                            let got = string_scalar_cmp(op, column, &lit_arr, lit_on_right)
                                .expect("served");
                            let c: &dyn Array = column.as_ref();
                            let want = if lit_on_right {
                                arrow_oracle(op, &c, &scalar)
                            } else {
                                arrow_oracle(op, &scalar, &c)
                            };
                            assert_eq!(
                                got.as_boolean(),
                                &want,
                                "{op:?} lit={lit:?} lit_on_right={lit_on_right}"
                            );
                        }
                    }
                }
            }
        }
    }

    fn bound(col_left: bool, op: BinaryOp, lit: &str) -> Expr {
        let col = Box::new(Expr::Col { name: "s".into() });
        let lit = Box::new(Expr::Lit {
            value: Literal::Str(lit.into()),
        });
        let (left, right) = if col_left { (col, lit) } else { (lit, col) };
        Expr::Binary { op, left, right }
    }

    /// Every lower/upper operator pair, each literal on either side, both `AND` orders, over
    /// the edge strings above plus a dictionary-encoded copy — held to Arrow's two kernels
    /// joined by Kleene `AND`.
    #[test]
    fn a_fused_range_equals_the_two_comparisons_anded() {
        use arrow::array::DictionaryArray;
        use arrow::compute::kernels::boolean::and_kleene;
        use arrow::datatypes::{Field, Int32Type, Schema};

        let rows: Vec<Option<&str>> = vec![
            Some(""),
            Some("t"),
            Some("the"),
            Some("the\0"),
            Some("thea"),
            Some("thf"),
            Some("thez"),
            Some("abcdefghij"),
            Some("abcdefghiz"),
            None,
            Some("zz"),
            Some("thf\0"),
        ];
        let plain: ArrayRef = Arc::new(StringArray::from(rows.clone()));
        let dict: ArrayRef = Arc::new(rows.into_iter().collect::<DictionaryArray<Int32Type>>());
        let ranges = [
            ("the", "thf"),
            ("", "t"),
            ("abcdefghi", "abcdefghj"),
            ("thf", "the"),
        ];
        for column in [plain, dict] {
            let schema = Schema::new(vec![Field::new("s", column.data_type().clone(), true)]);
            let batch = RecordBatch::try_new(Arc::new(schema), vec![column]).unwrap();
            for (lo, hi) in ranges {
                for lo_op in [BinaryOp::Gt, BinaryOp::Ge] {
                    for hi_op in [BinaryOp::Lt, BinaryOp::Le] {
                        for (lo_left, hi_left, swap) in [
                            (true, true, false),
                            (false, true, true),
                            (true, false, false),
                            (false, false, true),
                        ] {
                            // A literal on the left carries the mirrored operator, so
                            // `'the' <= s` is still the lower bound.
                            let mirror = |op, col_left| {
                                if col_left {
                                    op
                                } else {
                                    row_op(op, false).unwrap()
                                }
                            };
                            let a = bound(lo_left, mirror(lo_op, lo_left), lo);
                            let b = bound(hi_left, mirror(hi_op, hi_left), hi);
                            let (l, r) = if swap { (&b, &a) } else { (&a, &b) };
                            let got = try_string_range(l, r, &batch).unwrap().expect("served");
                            let decoded = Expr::Col { name: "s".into() }.eval(&batch).unwrap();
                            let c: &dyn Array = decoded.as_ref();
                            let lo_s = arrow::array::Scalar::new(StringArray::from(vec![lo]));
                            let hi_s = arrow::array::Scalar::new(StringArray::from(vec![hi]));
                            let want = and_kleene(
                                &arrow_oracle(lo_op, &c, &lo_s),
                                &arrow_oracle(hi_op, &c, &hi_s),
                            )
                            .unwrap();
                            assert_eq!(
                                got.as_boolean(),
                                &want,
                                "{lo_op:?} {lo:?} {hi_op:?} {hi:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_range_declines_two_columns_two_lower_bounds_and_a_non_string() {
        use arrow::datatypes::{Field, Schema};
        let schema = Schema::new(vec![
            Field::new("s", DataType::Utf8, true),
            Field::new("t", DataType::Utf8, true),
        ]);
        let col: ArrayRef = Arc::new(StringArray::from(vec!["a"]));
        let batch = RecordBatch::try_new(Arc::new(schema), vec![col.clone(), col]).unwrap();
        let other = Expr::Binary {
            op: BinaryOp::Lt,
            left: Box::new(Expr::Col { name: "t".into() }),
            right: Box::new(Expr::Lit {
                value: Literal::Str("b".into()),
            }),
        };
        let lower = bound(true, BinaryOp::Ge, "a");
        assert!(try_string_range(&lower, &other, &batch).unwrap().is_none());
        assert!(
            try_string_range(&lower, &bound(true, BinaryOp::Gt, "a"), &batch)
                .unwrap()
                .is_none()
        );
        let int_bound = Expr::Binary {
            op: BinaryOp::Lt,
            left: Box::new(Expr::Col { name: "s".into() }),
            right: Box::new(Expr::Lit {
                value: Literal::Int(3),
            }),
        };
        assert!(try_string_range(&lower, &int_bound, &batch)
            .unwrap()
            .is_none());
    }

    #[test]
    fn declines_what_it_cannot_serve() {
        let s: ArrayRef = Arc::new(StringArray::from(vec!["a"]));
        let l: ArrayRef = Arc::new(LargeStringArray::from(vec!["a"]));
        let null: ArrayRef = Arc::new(StringArray::from(vec![None::<&str>]));
        assert!(string_scalar_cmp(BinaryOp::Lt, &s, &l, true).is_none());
        assert!(string_scalar_cmp(BinaryOp::Lt, &s, &null, true).is_none());
        assert!(string_scalar_cmp(BinaryOp::Add, &s, &s, true).is_none());
    }

    #[test]
    fn a_string_ending_at_the_buffer_edge_reads_no_further() {
        // The last string's eight-byte window runs past the values buffer.
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["abcdefghijklmnop", "ab"]));
        let lit: ArrayRef = Arc::new(StringArray::from(vec!["ab"]));
        let got = string_scalar_cmp(BinaryOp::Eq, &arr, &lit, true).expect("served");
        assert_eq!(got.as_boolean(), &BooleanArray::from(vec![false, true]));
    }
}
