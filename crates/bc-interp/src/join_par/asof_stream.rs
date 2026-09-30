//! A keyless ASOF join that does not fit: a merge over two out-of-core sorted streams.
//!
//! With `by` keys an ASOF join grace-partitions like an equi join, because the nearest match
//! never crosses a `by` group. Without them every left row may match any right row, so there
//! is no key to partition on, and the join used to refuse an input over the envelope.
//!
//! What it needs instead is order. Sort both sides out of core by `(on, input position)` —
//! the position is the tie-break the kernel's stable sort already applies to equal right
//! keys — and walk the left side in chunks. A left row at key `t` can only ever be matched to
//! the last right row at or before `t` or the first at or after it (strictly before or after
//! when exact matches are excluded), so of all the right rows whose keys fall inside a
//! chunk's range, only the *first and last row of each segment* matter, where the segments
//! are the runs of right keys lying between, or equal to, the chunk's distinct left keys.
//! Every other right row is one the kernel could never pick, and is dropped as the right
//! stream passes.
//!
//! Each chunk is then joined by the unchanged in-memory kernel against that thinned slice,
//! plus the few rows carried from the previous chunk (the last right row below its highest
//! key and the first and last rows equal to it) and the first right row past its range. The
//! kernel sees every candidate it would have chosen over the whole right side, and nothing
//! it would choose instead: any extra row lies after the true candidate in key order only if
//! its key is past the target. So the chosen right row — and with it `tolerance`, `nearest`
//! and `allow_exact_matches` — is identical, and the result is the in-memory join's multiset
//! of rows.
//!
//! Peak working memory is one left chunk plus a slice of at most four right rows per distinct
//! left key in it, whatever either side's size.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::row::{OwnedRow, RowConverter, Rows, SortField};
use bc_ir::{AsofDirection, JoinOutputCol, SortKey};
use bc_runtime::agg::spill::SpillCodec;

use crate::error::InterpError;
use crate::ops;

/// The hidden column carrying each row's input position, the final sort key.
const SEQ_COL: &str = "__bc_asof_stream_seq";

/// The smallest left chunk worth a kernel call.
pub(crate) const MIN_CHUNK_BYTES: usize = 1 << 20;

/// The out-of-core parameters a streamed ASOF join runs under.
pub(crate) struct AsofSpill<'a> {
    pub chunk_bytes: usize,
    pub dir: &'a Path,
    pub fanin: usize,
    pub run_target_bytes: u64,
    pub codec: SpillCodec,
    pub cancel: Option<&'a bc_resource::CancelToken>,
}

/// The join itself: everything [`crate::ops::asof_join_batches`] takes besides the inputs.
pub(crate) struct AsofSpec<'a> {
    pub left_on: &'a str,
    pub right_on: &'a str,
    pub direction: AsofDirection,
    pub tolerance: Option<f64>,
    pub allow_exact_matches: bool,
    pub output: &'a [JoinOutputCol],
}

/// Whether [`asof_streaming`] computes this join exactly: both sides carry rows, and their
/// `on` columns share one flat type (the kernel encodes both with one converter, and a
/// nested key would be canonicalized differently by the out-of-core sort).
pub(crate) fn streamable(left: &[RecordBatch], right: &[RecordBatch], spec: &AsofSpec<'_>) -> bool {
    let on_type = |batches: &[RecordBatch], name: &str| -> Option<DataType> {
        let b = batches.iter().find(|b| b.num_rows() > 0)?;
        if b.schema().column_with_name(SEQ_COL).is_some() {
            return None;
        }
        Some(b.column_by_name(name)?.data_type().clone())
    };
    match (on_type(left, spec.left_on), on_type(right, spec.right_on)) {
        (Some(l), Some(r)) => {
            l == r && !l.is_nested() && !matches!(l, DataType::Dictionary(..) | DataType::Null)
        }
        _ => false,
    }
}

/// Join `left` to `right` as a keyless ASOF join with bounded working memory. Caller
/// guarantees [`streamable`] returned `true`. Returns the output batches and bytes spilled.
pub(crate) fn asof_streaming(
    left: Vec<RecordBatch>,
    right: Vec<RecordBatch>,
    spec: &AsofSpec<'_>,
    spill: &AsofSpill<'_>,
) -> Result<(Vec<RecordBatch>, u64), InterpError> {
    let left_schema = left
        .iter()
        .find(|b| b.num_rows() > 0)
        .ok_or(InterpError::EmptyJoinInput)?
        .schema();
    let right_schema = right
        .iter()
        .find(|b| b.num_rows() > 0)
        .ok_or(InterpError::EmptyJoinInput)?
        .schema();
    let (mut lstore, lspill) = sort_by_on(left, &left_schema, spec.left_on, "left", spill)?;
    let (mut rstore, rspill) = sort_by_on(right, &right_schema, spec.right_on, "right", spill)?;
    let lsrc = lstore
        .open_reader(0)?
        .into_iter()
        .flatten()
        .map(|b| b.map_err(InterpError::from));
    let rsrc = rstore
        .open_reader(0)?
        .into_iter()
        .flatten()
        .map(|b| b.map_err(InterpError::from));
    let mut ls = Stream::new(lsrc);
    let mut rs = Stream::new(rsrc);
    let mut merge = Merge {
        spec,
        right_schema: right_schema.clone(),
        conv: None,
        carry: Vec::new(),
    };
    let mut out = Vec::new();
    loop {
        let chunk = ls.take_bytes(spill.chunk_bytes)?;
        if chunk.is_empty() {
            break;
        }
        let chunk = strip(&ops::materialize(&chunk)?, &left_schema)?;
        let batch = merge.join_chunk(&chunk, &mut rs)?;
        if batch.num_rows() > 0 {
            out.push(batch);
        }
    }
    // The right stream may hold rows past the left side's last key; they cannot match, but
    // they must be read to make the short-read check honest.
    while rs.pull()? {
        rs.buf.clear();
    }
    lstore.verify_rows(0, ls.read)?;
    rstore.verify_rows(0, rs.read)?;
    Ok((out, lspill + rspill))
}

/// Sort one side out of core by `(on, input position)`, nulls last.
fn sort_by_on(
    parts: Vec<RecordBatch>,
    schema: &Arc<Schema>,
    on: &str,
    side: &str,
    spill: &AsofSpill<'_>,
) -> Result<(bc_runtime::agg::spill::DiskSpillStore, u64), InterpError> {
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(Field::new(SEQ_COL, DataType::UInt64, false));
    let tagged = Arc::new(Schema::new(fields));
    let mut next = 0u64;
    let mut with_seq = Vec::with_capacity(parts.len());
    for b in parts {
        if b.num_rows() == 0 {
            continue;
        }
        let n = b.num_rows() as u64;
        let mut cols = b.columns().to_vec();
        cols.push(Arc::new(UInt64Array::from_iter_values(next..next + n)));
        next += n;
        with_seq.push(RecordBatch::try_new(tagged.clone(), cols)?);
    }
    let key = |name: &str| SortKey {
        expr: bc_expr::Expr::Col {
            name: name.to_string(),
        },
        descending: false,
        nulls_first: false,
    };
    ops::external_sort_to_final_store(
        with_seq,
        &[key(on), key(SEQ_COL)],
        &spill.dir.join(format!("asof-{side}")),
        spill.fanin,
        spill.run_target_bytes,
        spill.codec,
        spill.cancel,
    )?
    .ok_or(InterpError::EmptyJoinInput)
}

/// A sorted batch stream consumed row-exactly.
struct Stream<I> {
    src: I,
    buf: VecDeque<RecordBatch>,
    done: bool,
    read: u64,
}

impl<I: Iterator<Item = Result<RecordBatch, InterpError>>> Stream<I> {
    fn new(src: I) -> Self {
        Self {
            src,
            buf: VecDeque::new(),
            done: false,
            read: 0,
        }
    }

    fn pull(&mut self) -> Result<bool, InterpError> {
        while !self.done {
            match self.src.next() {
                Some(b) => {
                    let b = b?;
                    self.read += b.num_rows() as u64;
                    if b.num_rows() > 0 {
                        self.buf.push_back(b);
                        return Ok(true);
                    }
                }
                None => self.done = true,
            }
        }
        Ok(false)
    }

    fn front(&mut self) -> Result<Option<RecordBatch>, InterpError> {
        if self.buf.is_empty() {
            self.pull()?;
        }
        Ok(self.buf.front().cloned())
    }

    /// Consume `n` rows of the front batch.
    fn advance(&mut self, n: usize) {
        if let Some(front) = self.buf.pop_front() {
            if n < front.num_rows() {
                self.buf.push_front(front.slice(n, front.num_rows() - n));
            }
        }
    }

    /// Consume rows until about `bytes` have been taken (at least one row).
    fn take_bytes(&mut self, bytes: usize) -> Result<Vec<RecordBatch>, InterpError> {
        let mut out = Vec::new();
        let mut taken = 0usize;
        while taken < bytes {
            let Some(front) = self.front()? else {
                break;
            };
            let rows = front.num_rows();
            let per_row = (ops::sliced_batch_bytes(&front) / rows).max(1);
            let want = (bytes - taken).div_ceil(per_row).clamp(1, rows);
            taken += want * per_row;
            out.push(front.slice(0, want));
            self.advance(want);
        }
        Ok(out)
    }
}

/// The merge state: the key encoder and the right rows carried between chunks.
struct Merge<'a> {
    spec: &'a AsofSpec<'a>,
    right_schema: Arc<Schema>,
    conv: Option<RowConverter>,
    /// Right rows at or below the previous chunk's highest key that a later left key can
    /// still pick, in key order.
    carry: Vec<RecordBatch>,
}

impl Merge<'_> {
    /// Encode the canonical `on` key of `batch`, so `-0.0`/`0.0` and every NaN compare equal.
    fn encode(&mut self, batch: &RecordBatch, on: &str) -> Result<Rows, InterpError> {
        let col = batch
            .column_by_name(on)
            .ok_or_else(|| InterpError::UnknownJoinColumn(on.to_string()))?;
        let canon = bc_arrow::canon_float_array(col);
        if self.conv.is_none() {
            self.conv = Some(RowConverter::new(vec![SortField::new(
                canon.data_type().clone(),
            )])?);
        }
        Ok(self
            .conv
            .as_ref()
            .expect("built above")
            .convert_columns(&[canon])?)
    }

    /// Join one sorted left chunk, consuming the right stream up to the chunk's highest key.
    fn join_chunk<I: Iterator<Item = Result<RecordBatch, InterpError>>>(
        &mut self,
        chunk: &RecordBatch,
        rs: &mut Stream<I>,
    ) -> Result<RecordBatch, InterpError> {
        let lkeys = self.encode(chunk, self.spec.left_on)?;
        let on = chunk
            .column_by_name(self.spec.left_on)
            .expect("encoded above")
            .clone();
        // The chunk's distinct non-null keys, ascending (nulls sort last and never match).
        let mut targets: Vec<OwnedRow> = Vec::new();
        for i in 0..chunk.num_rows() {
            if on.is_null(i) {
                break;
            }
            let k = lkeys.row(i);
            if targets.last().is_none_or(|t| t.row() != k) {
                targets.push(k.owned());
            }
        }
        let mut slice: Vec<RecordBatch> = std::mem::take(&mut self.carry);
        if let Some(hi) = targets.last().cloned() {
            let kept = self.consume_through(&targets, &hi, rs)?;
            slice.extend(kept);
            let carry = self.carry_for(&slice, &hi)?;
            // The first right row past the chunk's range: the forward candidate of its top.
            if let Some(front) = rs.front()? {
                let col = front
                    .column_by_name(self.spec.right_on)
                    .ok_or_else(|| InterpError::UnknownJoinColumn(self.spec.right_on.into()))?;
                if !col.is_null(0) {
                    slice.push(strip(&front.slice(0, 1), &self.right_schema)?);
                }
            }
            self.carry = carry;
        } else {
            // Every left row in the chunk has a null key: nothing can match, nothing moves.
            self.carry = slice.clone();
        }
        let right = if slice.is_empty() {
            RecordBatch::new_empty(self.right_schema.clone())
        } else {
            arrow::compute::concat_batches(&self.right_schema, &slice)?
        };
        ops::asof_join_batches(
            chunk,
            &right,
            self.spec.left_on,
            self.spec.right_on,
            &[],
            &[],
            self.spec.direction,
            self.spec.tolerance,
            self.spec.allow_exact_matches,
            self.spec.output,
        )
    }

    /// Consume every right row with a key at or below `hi`, keeping the first and last row
    /// of each segment the chunk's `targets` cut the keys into.
    fn consume_through<I: Iterator<Item = Result<RecordBatch, InterpError>>>(
        &mut self,
        targets: &[OwnedRow],
        hi: &OwnedRow,
        rs: &mut Stream<I>,
    ) -> Result<Vec<RecordBatch>, InterpError> {
        // (segment, row) of every kept candidate across the consumed batches, gathered per
        // batch and thinned once more at the end, since a segment can span batches.
        let mut kept: Vec<(usize, RecordBatch)> = Vec::new();
        let mut j = 0usize;
        while let Some(front) = rs.front()? {
            let keys = self.encode(&front, self.spec.right_on)?;
            let col = front
                .column_by_name(self.spec.right_on)
                .expect("encoded above")
                .clone();
            let n = front.num_rows();
            // Rows at or below `hi`: a prefix, since the stream is sorted and nulls are last.
            let end = (0..n)
                .find(|&i| col.is_null(i) || keys.row(i) > hi.row())
                .unwrap_or(n);
            let mut seg_first: Vec<(usize, usize)> = Vec::new();
            for i in 0..end {
                let k = keys.row(i);
                while j < targets.len() && targets[j].row() < k {
                    j += 1;
                }
                let seg = if j < targets.len() && targets[j].row() == k {
                    2 * j + 1
                } else {
                    2 * j
                };
                match seg_first.last_mut() {
                    Some((s, _)) if *s == seg => {}
                    _ => seg_first.push((seg, i)),
                }
            }
            // First and last row of each segment within this batch.
            let mut picks: Vec<(usize, u32)> = Vec::new();
            for (w, &(seg, start)) in seg_first.iter().enumerate() {
                let stop = seg_first.get(w + 1).map_or(end, |&(_, s)| s);
                picks.push((seg, start as u32));
                if stop - 1 > start {
                    picks.push((seg, (stop - 1) as u32));
                }
            }
            if !picks.is_empty() {
                let idx: UInt32Array = picks.iter().map(|&(_, r)| r).collect();
                let rows = strip(&ops::take_batch(&front, &idx)?, &self.right_schema)?;
                for (p, &(seg, _)) in picks.iter().enumerate() {
                    kept.push((seg, rows.slice(p, 1)));
                }
            }
            rs.advance(end);
            if end < n {
                break;
            }
        }
        // Thin across batch boundaries: keep the first and last of each segment.
        let mut out: Vec<RecordBatch> = Vec::new();
        for w in 0..kept.len() {
            let seg = kept[w].0;
            let first = w == 0 || kept[w - 1].0 != seg;
            let last = w + 1 == kept.len() || kept[w + 1].0 != seg;
            if first || last {
                out.push(kept[w].1.clone());
            }
        }
        Ok(out)
    }

    /// The rows of `slice` a later chunk (whose keys are all at or above `hi`) can still
    /// pick: the last row below `hi`, and the first and last rows equal to it.
    fn carry_for(
        &mut self,
        slice: &[RecordBatch],
        hi: &OwnedRow,
    ) -> Result<Vec<RecordBatch>, InterpError> {
        let mut below: Option<RecordBatch> = None;
        let mut equal: Vec<RecordBatch> = Vec::new();
        for row in slice {
            let k = self.encode(row, self.spec.right_on)?;
            if k.row(0) < hi.row() {
                below = Some(row.clone());
            } else if k.row(0) == hi.row() {
                equal.push(row.clone());
            }
        }
        let mut carry: Vec<RecordBatch> = below.into_iter().collect();
        if let Some(first) = equal.first() {
            carry.push(first.clone());
        }
        if equal.len() > 1 {
            carry.push(equal.last().expect("non-empty").clone());
        }
        Ok(carry)
    }
}

/// `batch` without the hidden position column.
fn strip(batch: &RecordBatch, schema: &Arc<Schema>) -> Result<RecordBatch, InterpError> {
    let cols: Vec<ArrayRef> = batch.columns()[..schema.fields().len()].to_vec();
    Ok(RecordBatch::try_new(schema.clone(), cols)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, AsArray, Float64Array, Int64Array};
    use bc_ir::JoinSide;

    fn side(prefix: &str, keys: &[Option<f64>], batch_rows: usize) -> Vec<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new(format!("{prefix}t"), DataType::Float64, true),
            Field::new(format!("{prefix}id"), DataType::Int64, false),
        ]));
        keys.chunks(batch_rows)
            .enumerate()
            .map(|(c, ks)| {
                let t: ArrayRef = Arc::new(Float64Array::from(ks.to_vec()));
                let id: ArrayRef = Arc::new(Int64Array::from_iter_values(
                    (0..ks.len()).map(|i| (c * batch_rows + i) as i64),
                ));
                RecordBatch::try_new(schema.clone(), vec![t, id]).unwrap()
            })
            .collect()
    }

    fn keys(n: usize, salt: usize, spread: usize) -> Vec<Option<f64>> {
        (0..n)
            .map(|i| match (i * 7 + salt) % 17 {
                0 => None,
                1 => Some(-0.0),
                2 => Some(0.0),
                3 => Some(f64::NAN),
                _ => Some(((i * 13 + salt) % spread) as f64 / 2.0),
            })
            .collect()
    }

    fn output() -> Vec<JoinOutputCol> {
        let col = |side, name: &str| JoinOutputCol {
            side,
            name: name.to_string(),
            alias: name.to_string(),
        };
        vec![
            col(JoinSide::Left, "lt"),
            col(JoinSide::Left, "lid"),
            col(JoinSide::Right, "rt"),
            col(JoinSide::Right, "rid"),
        ]
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

    /// Every direction, with and without exact matches and a tolerance, over right sides
    /// denser and sparser than the left and full of ties, `-0.0`/`0.0`, NaN and nulls, at
    /// chunk sizes from one row up: equal to the kernel over both whole relations, including
    /// which of several tied right rows is chosen.
    #[test]
    fn streamed_asof_equals_the_whole_relation_join() {
        let output = output();
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        for (nl, nr, spread) in [(400, 900, 40), (700, 120, 400), (300, 300, 9)] {
            let left = side("l", &keys(nl, 1, spread), 61);
            let right = side("r", &keys(nr, 4, spread), 53);
            let lw = ops::materialize(&left).unwrap();
            let rw = ops::materialize(&right).unwrap();
            for direction in [
                AsofDirection::Backward,
                AsofDirection::Forward,
                AsofDirection::Nearest,
            ] {
                for allow_exact_matches in [true, false] {
                    for tolerance in [None, Some(3.0)] {
                        let spec = AsofSpec {
                            left_on: "lt",
                            right_on: "rt",
                            direction,
                            tolerance,
                            allow_exact_matches,
                            output: &output,
                        };
                        let expected = ops::asof_join_batches(
                            &lw,
                            &rw,
                            "lt",
                            "rt",
                            &[],
                            &[],
                            direction,
                            tolerance,
                            allow_exact_matches,
                            &output,
                        )
                        .unwrap();
                        assert!(streamable(&left, &right, &spec));
                        for chunk_bytes in [1, 500, 1 << 20] {
                            let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let dir = std::env::temp_dir()
                                .join(format!("bc_asofstream_{}_{call}", std::process::id()));
                            let spill = AsofSpill {
                                chunk_bytes,
                                dir: &dir,
                                fanin: 3,
                                run_target_bytes: 2048,
                                codec: SpillCodec::None,
                                cancel: None,
                            };
                            let (got, spilled) =
                                asof_streaming(left.clone(), right.clone(), &spec, &spill).unwrap();
                            assert!(spilled > 0);
                            assert_eq!(
                                rows(&got),
                                rows(std::slice::from_ref(&expected)),
                                "{direction:?} exact={allow_exact_matches} tol={tolerance:?} \
                                 chunk={chunk_bytes} nl={nl} nr={nr}"
                            );
                            let _ = std::fs::remove_dir_all(&dir);
                        }
                    }
                }
            }
        }
    }

    /// Mismatched or nested key types are declined, not approximated.
    #[test]
    fn unusual_key_types_are_declined() {
        let output = output();
        let spec = AsofSpec {
            left_on: "lt",
            right_on: "rt",
            direction: AsofDirection::Backward,
            tolerance: None,
            allow_exact_matches: true,
            output: &output,
        };
        let left = side("l", &keys(5, 1, 4), 5);
        let schema = Arc::new(Schema::new(vec![Field::new("rt", DataType::Int64, true)]));
        let ints = vec![RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![1i64, 2])) as ArrayRef],
        )
        .unwrap()];
        assert!(!streamable(&left, &ints, &spec));
        assert!(!streamable(&left, &[], &spec));
    }
}
