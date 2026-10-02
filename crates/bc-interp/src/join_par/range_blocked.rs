//! A range join whose right side does not fit: block-nested over chunks of both sides.
//!
//! The streaming range join (`par.rs`, via [`super::probe_stream`]) holds the right side
//! whole and streams the left past it, because a left row's matches depend on the whole
//! right side. That dependency runs in both directions, though: a pair `(l, r)` matches or
//! not on its two rows alone, so the match set of the whole join is the union of the match
//! sets of every (left chunk, right chunk) pair. That is what lets the right side be cut too.
//!
//! What does not decompose per pair is the *unmatched* rows an outer, semi or anti join
//! emits: a left row is unmatched only if no right chunk matched it, and a right row only if
//! no left chunk did. So each chunk pair runs as an inner join (a semi join where only the
//! verdict is needed), a mark is kept per row of the side whose verdict is pending, and each
//! side's remainder is emitted exactly once:
//!
//! * a right row's verdict is complete when its chunk has met every left chunk, so the
//!   `Right`/`Full` remainder is emitted per right chunk, from one `bool` per row of the chunk
//!   that is resident anyway;
//! * a left row's verdict is complete only after the last right chunk, so `Left`/`Full`/
//!   `Semi`/`Anti` keep one `bool` per left row — a byte per row of a side the executor
//!   already holds — and emit at the end.
//!
//! Peak working memory is one right chunk plus one left chunk and those marks, whatever the
//! right side's size. The price is a pass over the left side per right chunk, which is why
//! this runs only when the right side does not fit and the single-chunk sweep is used
//! otherwise. The result is the same multiset of rows the in-memory join returns.

use arrow::array::{RecordBatch, UInt32Array};
use bc_ir::{JoinOutputCol, JoinType, RangeCondition};
use bc_runtime::join::JoinIndices;

use super::probe_stream::chunk_slice_by_bytes;
use crate::error::InterpError;
use crate::ops;

/// Join `left` against `right` on `conditions` with working memory bounded by
/// `budget_bytes`, split evenly between one left and one right chunk.
pub(crate) fn range_join_blocked(
    left: &[RecordBatch],
    right: &[RecordBatch],
    conditions: &[RangeCondition],
    join_type: JoinType,
    output: &[JoinOutputCol],
    budget_bytes: usize,
) -> Result<Vec<RecordBatch>, InterpError> {
    let chunk = (budget_bytes / 2).max(1);
    let left_schema = left.first().ok_or(InterpError::EmptyJoinInput)?.schema();
    let right_schema = right.first().ok_or(InterpError::EmptyJoinInput)?.schema();
    let left_rows: usize = left.iter().map(RecordBatch::num_rows).sum();
    let tracks_left = matches!(
        join_type,
        JoinType::Left | JoinType::Full | JoinType::Semi | JoinType::Anti
    );
    let tracks_right = matches!(join_type, JoinType::Right | JoinType::Full);
    let emits_pairs = !matches!(join_type, JoinType::Semi | JoinType::Anti);
    // Only the verdict is needed for semi/anti, and a semi pass returns each matched left row
    // once rather than once per matching right row.
    let pair_type = if emits_pairs {
        JoinType::Inner
    } else {
        JoinType::Semi
    };
    let mut left_seen = vec![false; if tracks_left { left_rows } else { 0 }];
    let mut out = Vec::new();

    for rc in chunk_slice_by_bytes(right, chunk) {
        let rc = rc?;
        let mut right_seen = vec![false; if tracks_right { rc.num_rows() } else { 0 }];
        let mut offset = 0usize;
        for lc in chunk_slice_by_bytes(left, chunk) {
            let lc = lc?;
            let idx = ops::range_join_indices(&lc, &rc, conditions, pair_type)?;
            if tracks_left {
                for l in idx.left.iter().flatten() {
                    left_seen[offset + l as usize] = true;
                }
            }
            if tracks_right {
                for r in idx.right.iter().flatten() {
                    right_seen[r as usize] = true;
                }
            }
            if emits_pairs {
                push_rows(&mut out, ops::gather_join_output(&lc, &rc, &idx, output)?);
            }
            offset += lc.num_rows();
        }
        if tracks_right {
            // Right rows no left chunk matched, null-extended through the kernel itself.
            let leftovers = ops::take_batch(&rc, &unmarked(&right_seen, false))?;
            if leftovers.num_rows() > 0 {
                let empty_left = RecordBatch::new_empty(left_schema.clone());
                let idx =
                    ops::range_join_indices(&empty_left, &leftovers, conditions, JoinType::Right)?;
                push_rows(
                    &mut out,
                    ops::gather_join_output(&empty_left, &leftovers, &idx, output)?,
                );
            }
        }
    }

    if tracks_left {
        emit_left_verdicts(
            left,
            &left_seen,
            join_type,
            conditions,
            output,
            &RecordBatch::new_empty(right_schema),
            &mut out,
        )?;
    }
    Ok(out)
}

/// Emit the left rows whose verdict needed every right chunk: the unmatched ones for
/// `Left`/`Full`/`Anti` (null-extended by the kernel against an empty right side), the
/// matched ones for `Semi`.
fn emit_left_verdicts(
    left: &[RecordBatch],
    seen: &[bool],
    join_type: JoinType,
    conditions: &[RangeCondition],
    output: &[JoinOutputCol],
    empty_right: &RecordBatch,
    out: &mut Vec<RecordBatch>,
) -> Result<(), InterpError> {
    let mut offset = 0usize;
    for batch in left {
        let n = batch.num_rows();
        let marks = &seen[offset..offset + n];
        offset += n;
        let keep_matched = matches!(join_type, JoinType::Semi);
        let rows = ops::take_batch(batch, &unmarked(marks, keep_matched))?;
        if rows.num_rows() == 0 {
            continue;
        }
        let batch = if keep_matched {
            // A semi join emits left columns only, so the right index is never read.
            let idx = JoinIndices {
                left: UInt32Array::from_iter_values(0..rows.num_rows() as u32),
                right: UInt32Array::from(vec![None::<u32>; rows.num_rows()]),
            };
            ops::gather_join_output(&rows, empty_right, &idx, output)?
        } else {
            let idx = ops::range_join_indices(&rows, empty_right, conditions, join_type)?;
            ops::gather_join_output(&rows, empty_right, &idx, output)?
        };
        push_rows(out, batch);
    }
    Ok(())
}

/// Positions whose mark equals `want`.
fn unmarked(marks: &[bool], want: bool) -> UInt32Array {
    marks
        .iter()
        .enumerate()
        .filter(|(_, &m)| m == want)
        .map(|(i, _)| i as u32)
        .collect()
}

fn push_rows(out: &mut Vec<RecordBatch>, batch: RecordBatch) {
    if batch.num_rows() > 0 {
        out.push(batch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, ArrayRef, AsArray, Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use bc_ir::{JoinSide, RangeOp};
    use std::sync::Arc;

    fn side(prefix: &str, vals: &[Option<f64>], batch_rows: usize) -> Vec<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new(format!("{prefix}k"), DataType::Float64, true),
            Field::new(format!("{prefix}id"), DataType::Int64, false),
        ]));
        vals.chunks(batch_rows)
            .enumerate()
            .map(|(c, ks)| {
                let k: ArrayRef = Arc::new(Float64Array::from(ks.to_vec()));
                let id: ArrayRef = Arc::new(Int64Array::from_iter_values(
                    (0..ks.len()).map(|i| (c * batch_rows + i) as i64),
                ));
                RecordBatch::try_new(schema.clone(), vec![k, id]).unwrap()
            })
            .collect()
    }

    fn output(join_type: JoinType) -> Vec<JoinOutputCol> {
        let col = |side: JoinSide, name: &str| JoinOutputCol {
            side,
            name: name.to_string(),
            alias: name.to_string(),
        };
        let mut cols = vec![col(JoinSide::Left, "lk"), col(JoinSide::Left, "lid")];
        if !matches!(join_type, JoinType::Semi | JoinType::Anti) {
            cols.push(col(JoinSide::Right, "rk"));
            cols.push(col(JoinSide::Right, "rid"));
        }
        cols
    }

    fn rows(batches: &[RecordBatch]) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for b in batches {
            for i in 0..b.num_rows() {
                let cells: Vec<String> = b
                    .columns()
                    .iter()
                    .map(|c| match c.data_type() {
                        _ if c.is_null(i) => "∅".to_string(),
                        DataType::Float64 => format!(
                            "{:x}",
                            c.as_primitive::<arrow::datatypes::Float64Type>()
                                .value(i)
                                .to_bits()
                        ),
                        _ => arrow::util::display::array_value_to_string(c, i).unwrap(),
                    })
                    .collect();
                out.push(cells.join("|"));
            }
        }
        out.sort();
        out
    }

    fn keys(n: usize, salt: usize) -> Vec<Option<f64>> {
        (0..n)
            .map(|i| match (i * 7 + salt) % 19 {
                0 => None,
                1 => Some(-0.0),
                2 => Some(0.0),
                3 => Some(f64::NAN),
                k => Some(((i * 3 + salt) % 23) as f64 - (k as f64) / 4.0),
            })
            .collect()
    }

    /// Every join flavor, one and two conditions, at budgets from one row per chunk up to a
    /// single chunk, equals the in-memory kernel over both whole relations.
    #[test]
    fn blocked_range_join_equals_the_whole_relation_join() {
        let left = side("l", &keys(300, 1), 37);
        let right = side("r", &keys(220, 5), 29);
        let lw = ops::materialize(&left).unwrap();
        let rw = ops::materialize(&right).unwrap();
        let cond = |op| RangeCondition {
            left_key: "lk".into(),
            right_key: "rk".into(),
            op,
        };
        let shapes = [
            vec![cond(RangeOp::Le)],
            vec![cond(RangeOp::Gt)],
            vec![cond(RangeOp::Ge), cond(RangeOp::Lt)],
        ];
        for conditions in &shapes {
            for jt in [
                JoinType::Inner,
                JoinType::Left,
                JoinType::Right,
                JoinType::Full,
                JoinType::Semi,
                JoinType::Anti,
            ] {
                let output = output(jt);
                let expected = ops::range_join_batches(&lw, &rw, conditions, jt, &output).unwrap();
                for budget in [1, 900, 4_000, usize::MAX / 2] {
                    let got =
                        range_join_blocked(&left, &right, conditions, jt, &output, budget).unwrap();
                    assert_eq!(
                        rows(&got),
                        rows(std::slice::from_ref(&expected)),
                        "{jt:?} {conditions:?} at budget {budget}"
                    );
                }
            }
        }
    }
}
