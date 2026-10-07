//! Bounded-memory window execution for a partition that does not fit the envelope.
//!
//! The grace window (`window_spill.rs`) bounds memory by splitting the input on its
//! `PARTITION BY` keys, which cannot help the two shapes where one partition is the whole
//! problem: a window with no `PARTITION BY` at all (the relation is one partition), and a
//! partitioned window whose hot key keeps its bucket over budget at every re-split depth.
//! Both used to end the same way: a typed `MemoryBudgetExceeded`, or the in-memory kernel run
//! over a bucket it had no room for.
//!
//! This module computes the same answer from one ordered stream instead. The input is sorted
//! out of core by `(partition keys, order keys, input position)` — the exact total order the
//! kernel itself ranks by, because the kernel breaks ties on the original row index — and the
//! sorted stream is then cut into chunks the in-memory kernel runs over one at a time. Each
//! chunk's results are *corrected* by a small carried state, so no partition is ever resident
//! whole:
//!
//! * `row_number`, `rank` and `dense_rank` add the rows (or peer groups) the partition had
//!   before the chunk. A peer group that straddles a cut keeps its previous rank.
//! * a running `count` (the SQL default frame, peers included) adds the count reached at the
//!   previous chunk's end. Its cuts are placed on peer boundaries, because every row of a peer
//!   group shares the count at the group's end.
//! * a frameless `first_value` takes the partition's first value, carried as one row.
//! * `lag`/`lead` run over the chunk plus the rows either side that their offsets can reach,
//!   and the context rows are sliced away again.
//!
//! Every correction is integer arithmetic or a copied value, so the result is **bit-identical**
//! to the in-memory kernel, not merely close. Anything else — a float running sum, whose
//! re-association would move the low bits, a frame, `percent_rank`/`ntile` (which need the
//! partition's size before its first row), a fused `rank <= k` — is declined by
//! [`streamable`], and the caller keeps its existing behaviour.
//!
//! The output is the same relation the kernel produces, in sorted order rather than input
//! order; a window's output is an unordered relation, which is how the grace path already
//! returns it.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Int64Array, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Int64Type, Schema};
use arrow::row::{OwnedRow, RowConverter, Rows, SortField};
use bc_ir::{SortKey, WindowFn, WindowFunc};
use bc_runtime::agg::spill::SpillCodec;

use crate::error::InterpError;
use crate::ops;

/// The hidden column carrying each row's input position, the final sort key.
const SEQ_COL: &str = "__bc_window_stream_seq";

/// The largest `lag`/`lead` distance streamed. The context held either side of a chunk is
/// this many rows, so an offset past it would reintroduce the unbounded state this avoids.
pub(crate) const MAX_CONTEXT_ROWS: i64 = 1 << 16;

/// The smallest chunk worth a kernel call. Below it the per-call setup dominates, and a
/// pathological envelope (one byte) would otherwise run the kernel once per row.
pub(crate) const MIN_CHUNK_BYTES: usize = 1 << 20;

/// Whether [`window_streaming`] computes this window exactly.
///
/// Checked before any input is consumed, so a `false` leaves the caller free to take its
/// existing path. The key types must be flat: the out-of-core sort canonicalizes a float key
/// at the top level only, while the kernel folds a float nested inside a list or struct, so a
/// nested key could put two rows the kernel calls peers into different chunks.
pub(crate) fn streamable(
    parts: &[RecordBatch],
    partition_keys: &[bc_expr::Expr],
    order_keys: &[SortKey],
    functions: &[WindowFunc],
    rank_limit: Option<usize>,
) -> Result<bool, InterpError> {
    if rank_limit.is_some() || order_keys.is_empty() || functions.is_empty() {
        return Ok(false);
    }
    if !functions.iter().all(function_streams) {
        return Ok(false);
    }
    let Some(sample) = parts.iter().find(|b| b.num_rows() > 0) else {
        return Ok(false);
    };
    if sample.schema().column_with_name(SEQ_COL).is_some() {
        return Ok(false);
    }
    for e in partition_keys
        .iter()
        .chain(order_keys.iter().map(|k| &k.expr))
    {
        let dt = e.eval(sample)?.data_type().clone();
        if dt.is_nested() || matches!(dt, DataType::Dictionary(..)) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether one function has an exact carried-state correction (see the module docs).
fn function_streams(f: &WindowFunc) -> bool {
    if f.frame.is_some() || f.alpha.is_some() || f.half_life.is_some() {
        return false;
    }
    match f.func {
        WindowFn::RowNumber | WindowFn::Rank | WindowFn::DenseRank => true,
        WindowFn::Count => f.input.is_some(),
        WindowFn::FirstValue => f.input.is_some() && !f.ignore_nulls,
        WindowFn::Lag | WindowFn::Lead => {
            f.input.is_some()
                && !f.ignore_nulls
                && f.offset
                    .checked_abs()
                    .is_some_and(|o| o <= MAX_CONTEXT_ROWS)
        }
        _ => false,
    }
}

/// The out-of-core parameters a streamed window runs under.
pub(crate) struct StreamSpill<'a> {
    /// Target bytes per kernel chunk.
    pub chunk_bytes: usize,
    /// Directory the sorted runs are written under.
    pub dir: &'a Path,
    /// Merge fan-in of the external sort.
    pub fanin: usize,
    /// Pass-0 run size of the external sort.
    pub run_target_bytes: u64,
    /// Spill codec.
    pub codec: SpillCodec,
    /// Cancellation, checked by the external sort's merge passes.
    pub cancel: Option<&'a bc_resource::CancelToken>,
}

/// Compute a window over `parts` with bounded working memory. Returns the output batches
/// and the bytes spilled. Caller guarantees [`streamable`] returned `true`.
pub(crate) fn window_streaming(
    parts: Vec<RecordBatch>,
    partition_keys: &[bc_expr::Expr],
    order_keys: &[SortKey],
    functions: &[WindowFunc],
    spill: &StreamSpill<'_>,
) -> Result<(Vec<RecordBatch>, u64), InterpError> {
    let Some(first) = parts.iter().find(|b| b.num_rows() > 0) else {
        return Ok((Vec::new(), 0));
    };
    let input_schema = first.schema();
    let with_seq = append_seq(parts, &input_schema)?;
    let mut keys: Vec<SortKey> = partition_keys
        .iter()
        .map(|e| SortKey {
            expr: e.clone(),
            descending: false,
            nulls_first: false,
        })
        .collect();
    keys.extend(order_keys.iter().cloned());
    keys.push(SortKey {
        expr: bc_expr::Expr::Col {
            name: SEQ_COL.to_string(),
        },
        descending: false,
        nulls_first: false,
    });
    let Some((mut store, spill_bytes)) = ops::external_sort_to_final_store(
        with_seq,
        &keys,
        &spill.dir.join("window-stream"),
        spill.fanin,
        spill.run_target_bytes,
        spill.codec,
        spill.cancel,
    )?
    else {
        return Ok((Vec::new(), 0));
    };
    let reader = store.open_reader(0)?;
    let src = reader
        .into_iter()
        .flatten()
        .map(|b| b.map_err(InterpError::from));
    let mut walk = Walk::new(partition_keys, order_keys, functions, &input_schema);
    let mut sorted = Sorted::new(src);
    let mut out = Vec::new();
    while let Some(batch) = walk.next_chunk(&mut sorted, spill.chunk_bytes)? {
        out.push(batch);
    }
    store.verify_rows(0, sorted.read)?;
    Ok((out, spill_bytes))
}

/// Tag every row with its position in `parts`, the tie-break the kernel applies.
fn append_seq(
    parts: Vec<RecordBatch>,
    schema: &Arc<Schema>,
) -> Result<Vec<RecordBatch>, InterpError> {
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(Field::new(SEQ_COL, DataType::UInt64, false));
    let tagged = Arc::new(Schema::new(fields));
    let mut next = 0u64;
    let mut out = Vec::with_capacity(parts.len());
    for b in parts {
        if b.num_rows() == 0 {
            continue;
        }
        let n = b.num_rows() as u64;
        let seq: ArrayRef = Arc::new(UInt64Array::from_iter_values(next..next + n));
        next += n;
        let mut cols = b.columns().to_vec();
        cols.push(seq);
        out.push(RecordBatch::try_new(tagged.clone(), cols)?);
    }
    Ok(out)
}

/// A sorted stream of batches that can be consumed a row at a time and peeked ahead.
struct Sorted<I> {
    src: I,
    buf: VecDeque<RecordBatch>,
    buffered: usize,
    done: bool,
    /// Rows read off the stream, for the short-read check.
    read: u64,
}

impl<I: Iterator<Item = Result<RecordBatch, InterpError>>> Sorted<I> {
    fn new(src: I) -> Self {
        Self {
            src,
            buf: VecDeque::new(),
            buffered: 0,
            done: false,
            read: 0,
        }
    }

    /// Buffer one more non-empty batch; `false` once the stream is exhausted.
    fn pull(&mut self) -> Result<bool, InterpError> {
        while !self.done {
            match self.src.next() {
                Some(b) => {
                    let b = b?;
                    self.read += b.num_rows() as u64;
                    if b.num_rows() > 0 {
                        self.buffered += b.num_rows();
                        self.buf.push_back(b);
                        return Ok(true);
                    }
                }
                None => self.done = true,
            }
        }
        Ok(false)
    }

    /// Buffer at least `n` rows, or everything that is left.
    fn fill(&mut self, n: usize) -> Result<(), InterpError> {
        while self.buffered < n && self.pull()? {}
        Ok(())
    }

    /// The first buffered batch, pulling one if the buffer is empty.
    fn front(&mut self) -> Result<Option<&RecordBatch>, InterpError> {
        if self.buf.is_empty() {
            self.pull()?;
        }
        Ok(self.buf.front())
    }

    /// Remove and return up to `n` buffered rows.
    fn take(&mut self, n: usize) -> Vec<RecordBatch> {
        let mut out = Vec::new();
        let mut need = n;
        while need > 0 {
            let Some(front) = self.buf.pop_front() else {
                break;
            };
            let rows = front.num_rows();
            if rows <= need {
                need -= rows;
                self.buffered -= rows;
                out.push(front);
            } else {
                out.push(front.slice(0, need));
                self.buf.push_front(front.slice(need, rows - need));
                self.buffered -= need;
                need = 0;
            }
        }
        out
    }

    /// Up to the first `n` buffered rows, without consuming them.
    fn peek(&self, n: usize) -> Vec<RecordBatch> {
        let mut out = Vec::new();
        let mut need = n;
        for b in &self.buf {
            if need == 0 {
                break;
            }
            let k = b.num_rows().min(need);
            out.push(b.slice(0, k));
            need -= k;
        }
        out
    }
}

/// What the rows before the current chunk leave behind for it.
struct Carry {
    /// Partition key of the previous chunk's last row (`None` with no `PARTITION BY`).
    part: Option<OwnedRow>,
    /// Partition + order key of that row: its peer group.
    peer: OwnedRow,
    /// Rows of that row's partition seen so far.
    rows: i64,
    /// Each carried function's corrected value at that row, as a one-row array.
    last: Vec<ArrayRef>,
}

/// The state of one streamed window: the functions split by how they stream, the key
/// encoders, and what the previous chunk carried forward.
struct Walk<'a> {
    partition_keys: &'a [bc_expr::Expr],
    order_keys: &'a [SortKey],
    functions: &'a [WindowFunc],
    /// Functions computed per chunk and corrected by the carry, and their positions.
    carried: Vec<WindowFunc>,
    carried_at: Vec<usize>,
    /// `lag`/`lead`, computed over the chunk plus context, and their positions.
    shifted: Vec<WindowFunc>,
    shifted_at: Vec<usize>,
    /// Rows of context each shifted call needs before and after a chunk.
    back: usize,
    ahead: usize,
    /// Whether chunks must end on a peer boundary (a running `count`).
    align: bool,
    input_schema: Arc<Schema>,
    part_conv: Option<RowConverter>,
    peer_conv: Option<RowConverter>,
    carry: Option<Carry>,
    /// The last `back` input rows, for the next chunk's `lag`.
    tail: Vec<RecordBatch>,
}

impl<'a> Walk<'a> {
    fn new(
        partition_keys: &'a [bc_expr::Expr],
        order_keys: &'a [SortKey],
        functions: &'a [WindowFunc],
        input_schema: &Arc<Schema>,
    ) -> Self {
        let (mut carried, mut carried_at, mut shifted, mut shifted_at) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut back, mut ahead) = (0usize, 0usize);
        for (i, f) in functions.iter().enumerate() {
            match f.func {
                WindowFn::Lag | WindowFn::Lead => {
                    // A negative offset flips direction: `lag(v, -n)` is `lead(v, n)`.
                    let reach = f.offset.unsigned_abs() as usize;
                    let backward = (f.func == WindowFn::Lag) == (f.offset >= 0);
                    if backward {
                        back = back.max(reach);
                    } else {
                        ahead = ahead.max(reach);
                    }
                    shifted.push(f.clone());
                    shifted_at.push(i);
                }
                _ => {
                    carried.push(f.clone());
                    carried_at.push(i);
                }
            }
        }
        Self {
            partition_keys,
            order_keys,
            functions,
            align: functions.iter().any(|f| f.func == WindowFn::Count),
            carried,
            carried_at,
            shifted,
            shifted_at,
            back,
            ahead,
            input_schema: input_schema.clone(),
            part_conv: None,
            peer_conv: None,
            carry: None,
            tail: Vec::new(),
        }
    }

    /// Encode the partition keys (`None` with no `PARTITION BY`) and the peer keys of `batch`.
    ///
    /// Floats are canonicalized first, so `-0.0`/`0.0` and every NaN compare equal, the
    /// identity the kernel groups and ranks by.
    fn encode(&mut self, batch: &RecordBatch) -> Result<(Option<Rows>, Rows), InterpError> {
        let eval = |e: &bc_expr::Expr| -> Result<ArrayRef, InterpError> {
            Ok(bc_arrow::canon_float_array(&e.eval(batch)?))
        };
        let part: Vec<ArrayRef> = self
            .partition_keys
            .iter()
            .map(eval)
            .collect::<Result<_, _>>()?;
        let mut peer = part.clone();
        for k in self.order_keys {
            peer.push(eval(&k.expr)?);
        }
        let converter = |cols: &[ArrayRef]| {
            RowConverter::new(
                cols.iter()
                    .map(|c| SortField::new(c.data_type().clone()))
                    .collect(),
            )
        };
        if self.peer_conv.is_none() {
            self.peer_conv = Some(converter(&peer)?);
            if !part.is_empty() {
                self.part_conv = Some(converter(&part)?);
            }
        }
        let part_rows = match &self.part_conv {
            Some(c) => Some(c.convert_columns(&part)?),
            None => None,
        };
        let peer_conv = self.peer_conv.as_ref().expect("built above");
        Ok((part_rows, peer_conv.convert_columns(&peer)?))
    }

    /// Consume the next chunk off `sorted` and return its window output, or `None` at the end.
    fn next_chunk<I: Iterator<Item = Result<RecordBatch, InterpError>>>(
        &mut self,
        sorted: &mut Sorted<I>,
        chunk_bytes: usize,
    ) -> Result<Option<RecordBatch>, InterpError> {
        let mut taken: Vec<RecordBatch> = Vec::new();
        let mut bytes = 0usize;
        while bytes < chunk_bytes {
            let Some(front) = sorted.front()? else {
                break;
            };
            let rows = front.num_rows();
            let per_row = (ops::sliced_batch_bytes(front) / rows).max(1);
            let want = (chunk_bytes - bytes).div_ceil(per_row).clamp(1, rows);
            bytes += want * per_row;
            taken.extend(sorted.take(want));
        }
        if taken.is_empty() {
            return Ok(None);
        }
        if self.align {
            self.extend_to_peer_boundary(sorted, &mut taken)?;
        }
        let tagged = ops::materialize(&taken)?;
        drop(taken);
        let core = strip_seq(&tagged, &self.input_schema)?;
        drop(tagged);
        let n = core.num_rows();
        let n_in = self.input_schema.fields().len();

        let mut columns: Vec<Option<ArrayRef>> = vec![None; self.functions.len()];
        if !self.carried.is_empty() {
            let out = ops::window_batch(
                &core,
                self.partition_keys,
                self.order_keys,
                &self.carried,
                None,
            )?;
            let (part_rows, peer_rows) = self.encode(&core)?;
            let corrected = self.correct(&out.columns()[n_in..], part_rows.as_ref(), &peer_rows)?;
            self.update_carry(&corrected, part_rows.as_ref(), &peer_rows, n);
            for (col, &at) in corrected.into_iter().zip(&self.carried_at) {
                columns[at] = Some(col);
            }
        }
        if !self.shifted.is_empty() {
            sorted.fill(self.ahead)?;
            let ahead: Vec<RecordBatch> = sorted
                .peek(self.ahead)
                .iter()
                .map(|b| strip_seq(b, &self.input_schema))
                .collect::<Result<_, _>>()?;
            let before: usize = self.tail.iter().map(RecordBatch::num_rows).sum();
            let mut ext_parts = self.tail.clone();
            ext_parts.push(core.clone());
            ext_parts.extend(ahead);
            let ext = ops::materialize(&ext_parts)?;
            let out = ops::window_batch(
                &ext,
                self.partition_keys,
                self.order_keys,
                &self.shifted,
                None,
            )?;
            for (j, &at) in self.shifted_at.iter().enumerate() {
                columns[at] = Some(out.column(n_in + j).slice(before, n));
            }
        }
        if self.back > 0 {
            self.tail = last_rows(&self.tail, &core, self.back);
        }
        let mut fields: Vec<Field> = self
            .input_schema
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        let mut cols = core.columns().to_vec();
        for (f, col) in self.functions.iter().zip(columns) {
            let col = col.expect("every function is carried or shifted");
            fields.push(Field::new(&f.alias, col.data_type().clone(), true));
            cols.push(col);
        }
        Ok(Some(RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            cols,
        )?))
    }

    /// Grow `taken` until its last row ends a peer group, so no group straddles a cut.
    fn extend_to_peer_boundary<I: Iterator<Item = Result<RecordBatch, InterpError>>>(
        &mut self,
        sorted: &mut Sorted<I>,
        taken: &mut Vec<RecordBatch>,
    ) -> Result<(), InterpError> {
        let last = taken.last().expect("non-empty chunk");
        let (_, rows) = self.encode(last)?;
        let boundary = rows.row(last.num_rows() - 1).owned();
        loop {
            let Some(front) = sorted.front()? else {
                return Ok(());
            };
            let front = front.clone();
            let (_, rows) = self.encode(&front)?;
            match (0..front.num_rows()).find(|&i| rows.row(i) != boundary.row()) {
                Some(i) => {
                    taken.extend(sorted.take(i));
                    return Ok(());
                }
                None => taken.extend(sorted.take(front.num_rows())),
            }
        }
    }

    /// Apply the carried state to the per-chunk results of the carried functions.
    fn correct(
        &self,
        cols: &[ArrayRef],
        part_rows: Option<&Rows>,
        peer_rows: &Rows,
    ) -> Result<Vec<ArrayRef>, InterpError> {
        let n = peer_rows.num_rows();
        let Some(carry) = &self.carry else {
            return Ok(cols.to_vec());
        };
        let continues = match (part_rows, &carry.part) {
            (Some(p), Some(prev)) => p.row(0) == prev.row(),
            (None, None) => true,
            _ => false,
        };
        if !continues {
            return Ok(cols.to_vec());
        }
        // The first partition's extent, and its first peer group's.
        let e = part_rows.map_or(n, |p| (1..n).find(|&i| p.row(i) != p.row(0)).unwrap_or(n));
        let g = (1..n)
            .find(|&i| peer_rows.row(i) != peer_rows.row(0))
            .unwrap_or(n);
        let same_peer = peer_rows.row(0) == carry.peer.row();
        let mut out = Vec::with_capacity(cols.len());
        for (j, (f, col)) in self.carried.iter().zip(cols).enumerate() {
            let prev = &carry.last[j];
            out.push(match f.func {
                WindowFn::RowNumber => shift_i64(col, e, |_, v| v + carry.rows),
                WindowFn::Rank => {
                    let at = prev.as_primitive::<Int64Type>().value(0);
                    shift_i64(col, e, |i, v| {
                        if same_peer && i < g {
                            at
                        } else {
                            v + carry.rows
                        }
                    })
                }
                WindowFn::DenseRank => {
                    let at = prev.as_primitive::<Int64Type>().value(0);
                    shift_i64(col, e, |_, v| v + at - i64::from(same_peer))
                }
                WindowFn::Count => {
                    let at = prev.as_primitive::<Int64Type>().value(0);
                    shift_i64(col, e, |_, v| v + at)
                }
                WindowFn::FirstValue => {
                    let idx: Vec<(usize, usize)> = (0..n)
                        .map(|i| if i < e { (0, 0) } else { (1, i) })
                        .collect();
                    arrow::compute::interleave(&[prev.as_ref(), col.as_ref()], &idx)?
                }
                other => unreachable!("{other:?} is declined by `streamable`"),
            });
        }
        Ok(out)
    }

    /// Record what the chunk's last row leaves for the next chunk.
    fn update_carry(
        &mut self,
        corrected: &[ArrayRef],
        part_rows: Option<&Rows>,
        peer_rows: &Rows,
        n: usize,
    ) {
        let last_part = part_rows.map(|p| p.row(n - 1).owned());
        // Where the last partition starts; 0 when the whole chunk is one partition.
        let start = part_rows.map_or(0, |p| {
            (0..n - 1)
                .rev()
                .find(|&i| p.row(i) != p.row(n - 1))
                .map_or(0, |i| i + 1)
        });
        let continued = self
            .carry
            .as_ref()
            .is_some_and(|c| match (&c.part, part_rows) {
                (Some(prev), Some(p)) => p.row(0) == prev.row(),
                (None, None) => true,
                _ => false,
            });
        let rows = if start == 0 && continued {
            self.carry.as_ref().map_or(0, |c| c.rows) + n as i64
        } else {
            (n - start) as i64
        };
        self.carry = Some(Carry {
            part: last_part,
            peer: peer_rows.row(n - 1).owned(),
            rows,
            last: corrected.iter().map(|c| c.slice(n - 1, 1)).collect(),
        });
    }
}

/// `col` with `f(i, value)` applied to its first `e` rows. The carried functions all
/// return non-null `Int64`.
fn shift_i64(col: &ArrayRef, e: usize, f: impl Fn(usize, i64) -> i64) -> ArrayRef {
    let a = col.as_primitive::<Int64Type>();
    let vals: Int64Array = (0..a.len())
        .map(|i| {
            (!a.is_null(i)).then(|| {
                let v = a.value(i);
                if i < e {
                    f(i, v)
                } else {
                    v
                }
            })
        })
        .collect();
    Arc::new(vals)
}

/// `batch` without the hidden position column.
fn strip_seq(batch: &RecordBatch, input_schema: &Arc<Schema>) -> Result<RecordBatch, InterpError> {
    let n = input_schema.fields().len();
    let cols = batch.columns()[..n].to_vec();
    Ok(RecordBatch::try_new(input_schema.clone(), cols)?)
}

/// The last `k` rows of `tail ++ [core]`.
fn last_rows(tail: &[RecordBatch], core: &RecordBatch, k: usize) -> Vec<RecordBatch> {
    let mut out: Vec<RecordBatch> = Vec::new();
    let mut need = k;
    for b in std::iter::once(core).chain(tail.iter().rev()) {
        if need == 0 {
            break;
        }
        let rows = b.num_rows();
        let take = rows.min(need);
        out.push(b.slice(rows - take, take));
        need -= take;
    }
    out.reverse();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, StringArray};
    use bc_ir::WindowFunc;

    fn func(f: WindowFn, input: Option<&str>, offset: i64, alias: &str) -> WindowFunc {
        WindowFunc {
            func: f,
            input: input.map(|c| bc_expr::Expr::Col {
                name: c.to_string(),
            }),
            offset,
            frame: None,
            alpha: None,
            half_life: None,
            ignore_nulls: false,
            opts: bc_ir::WindowOpts::default(),
            alias: alias.to_string(),
        }
    }

    fn col(name: &str) -> bc_expr::Expr {
        bc_expr::Expr::Col {
            name: name.to_string(),
        }
    }

    /// `n` rows of `(p, t, v, s)`: a skewed partition key, a heavily tied float order key
    /// (with `-0.0`, `0.0`, NaN and nulls), a nullable value and a string.
    fn input(n: usize, batch_rows: usize) -> Vec<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("p", DataType::Int64, true),
            Field::new("t", DataType::Float64, true),
            Field::new("v", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
        ]));
        let mut out = Vec::new();
        let mut start = 0;
        while start < n {
            let end = (start + batch_rows).min(n);
            let p: Int64Array = (start..end)
                .map(|i| match i % 11 {
                    0 => None,
                    1..=7 => Some(0),
                    k => Some(k as i64),
                })
                .collect();
            let t: Float64Array = (start..end)
                .map(|i| match i % 13 {
                    0 => None,
                    1 => Some(-0.0),
                    2 => Some(0.0),
                    3 => Some(f64::NAN),
                    k => Some(((i * 7 + k) % 17) as f64),
                })
                .collect();
            let v: Int64Array = (start..end)
                .map(|i| (i % 5 != 0).then_some(i as i64))
                .collect();
            let s: StringArray = (start..end).map(|i| Some(format!("s{}", i % 3))).collect();
            out.push(
                RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(p), Arc::new(t), Arc::new(v), Arc::new(s)],
                )
                .unwrap(),
            );
            start = end;
        }
        out
    }

    /// Every output row as a string, sorted: the relation as a multiset, with floats by bits.
    fn rows(batches: &[RecordBatch]) -> Vec<String> {
        let mut out = Vec::new();
        for b in batches {
            for i in 0..b.num_rows() {
                let cells: Vec<String> = b
                    .columns()
                    .iter()
                    .map(|c| {
                        if c.is_null(i) {
                            return "∅".to_string();
                        }
                        match c.data_type() {
                            DataType::Float64 => {
                                format!(
                                    "{:x}",
                                    c.as_primitive::<arrow::datatypes::Float64Type>()
                                        .value(i)
                                        .to_bits()
                                )
                            }
                            _ => arrow::util::display::array_value_to_string(c, i).unwrap(),
                        }
                    })
                    .collect();
                out.push(cells.join("|"));
            }
        }
        out.sort();
        out
    }

    fn check(pk: &[bc_expr::Expr], ok: &[SortKey], funcs: &[WindowFunc], chunk_bytes: usize) {
        let parts = input(3_000, 97);
        assert!(streamable(&parts, pk, ok, funcs, None).unwrap());
        let whole = ops::materialize(&parts).unwrap();
        let expected = ops::window_batch(&whole, pk, ok, funcs, None).unwrap();
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("bc_winstream_{}_{call}", std::process::id()));
        let spill = StreamSpill {
            chunk_bytes,
            dir: &dir,
            fanin: 3,
            run_target_bytes: 4096,
            codec: SpillCodec::None,
            cancel: None,
        };
        let (got, spilled) = window_streaming(parts, pk, ok, funcs, &spill).unwrap();
        assert!(
            spilled > 0,
            "the input must actually go through the out-of-core sort"
        );
        assert!(got.len() >= 2, "the stream must be cut into several chunks");
        assert_eq!(
            got[0].schema(),
            expected.schema(),
            "schema must match the kernel's"
        );
        assert_eq!(rows(&got), rows(&[expected]));
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn all_funcs() -> Vec<WindowFunc> {
        vec![
            func(WindowFn::RowNumber, None, 1, "rn"),
            func(WindowFn::Rank, None, 1, "rk"),
            func(WindowFn::DenseRank, None, 1, "dr"),
            func(WindowFn::Count, Some("v"), 1, "cnt"),
            func(WindowFn::FirstValue, Some("v"), 1, "fv"),
            func(WindowFn::Lag, Some("v"), 3, "lag3"),
            func(WindowFn::Lead, Some("s"), 2, "lead2"),
            func(WindowFn::Lag, Some("v"), -5, "lagm5"),
        ]
    }

    /// The global window — no `PARTITION BY`, the shape that could not spill at all —
    /// equals the in-memory kernel bit for bit, at chunk sizes from a few rows up.
    #[test]
    fn global_window_streams_identically() {
        let ok = [SortKey {
            expr: col("t"),
            descending: false,
            nulls_first: false,
        }];
        for chunk in [1, 300, 5_000, 40_000] {
            check(&[], &ok, &all_funcs(), chunk);
        }
    }

    /// Descending, nulls first, and a second order key.
    #[test]
    fn descending_multi_key_window_streams_identically() {
        let ok = [
            SortKey {
                expr: col("t"),
                descending: true,
                nulls_first: true,
            },
            SortKey {
                expr: col("s"),
                descending: false,
                nulls_first: false,
            },
        ];
        check(&[], &ok, &all_funcs(), 700);
    }

    /// A partitioned window with one hot partition (7 of every 11 rows) and a null key:
    /// partitions continue across chunk cuts and several start inside one chunk.
    #[test]
    fn partitioned_window_streams_identically() {
        let ok = [SortKey {
            expr: col("t"),
            descending: false,
            nulls_first: false,
        }];
        for chunk in [1, 450, 9_000] {
            check(&[col("p")], &ok, &all_funcs(), chunk);
        }
        // Without `count`, cuts land inside peer groups, which is what the rank carry is for.
        let ranks = vec![
            func(WindowFn::Rank, None, 1, "rk"),
            func(WindowFn::DenseRank, None, 1, "dr"),
            func(WindowFn::RowNumber, None, 1, "rn"),
        ];
        check(&[col("p")], &ok, &ranks, 1);
        check(&[], &ok, &ranks, 1);
    }

    /// End to end through the parallel executor: a global window whose pool cannot admit it
    /// used to raise `MemoryBudgetExceeded`; it now streams, reports a spill, and equals the
    /// sequential oracle. A hot partition past every grace re-split takes the same route.
    #[test]
    fn executor_streams_a_window_that_used_to_refuse() {
        use crate::par::{execute_parallel_with_metrics, ExecOptions, SpillOptions};
        use bc_ir::RelOp;
        use bc_resource::MemoryPool;
        let ok = vec![SortKey {
            expr: col("t"),
            descending: false,
            nulls_first: false,
        }];
        for pk in [vec![], vec![col("p")]] {
            let plan = RelOp::Window {
                input: Box::new(RelOp::Scan { source_id: 0 }),
                partition_keys: pk.clone(),
                order_keys: ok.clone(),
                functions: all_funcs(),
                rank_limit: None,
            };
            let oracle = crate::execute(&plan, &[input(3_000, 97)]).unwrap();
            let pool = MemoryPool::new(8);
            let _held = pool.try_reserve(8).unwrap(); // pool full: every breaker must spill
            let dir = std::env::temp_dir().join(format!(
                "bc_winstream_exec_{}_{}",
                std::process::id(),
                pk.len()
            ));
            let opts = ExecOptions {
                agg_spill: Some(SpillOptions {
                    memory_budget_bytes: 64,
                    dir: dir.clone(),
                    codec: SpillCodec::None,
                }),
                pool: Some(Arc::clone(&pool)),
                ..ExecOptions::default()
            };
            let (out, metrics) =
                execute_parallel_with_metrics(&plan, &[input(3_000, 97)], &opts).unwrap();
            assert!(
                metrics.ops.iter().any(|m| m.kind == "window" && m.spilled),
                "the window must report that it went out of core"
            );
            assert_eq!(rows(&out), rows(&oracle));
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// The shapes whose streamed answer would not be exact are declined, not approximated.
    #[test]
    fn inexact_shapes_are_declined() {
        let parts = input(10, 10);
        let ok = [SortKey {
            expr: col("t"),
            descending: false,
            nulls_first: false,
        }];
        let sum = [func(WindowFn::Sum, Some("v"), 1, "s")];
        assert!(!streamable(&parts, &[], &ok, &sum, None).unwrap());
        let rn = [func(WindowFn::RowNumber, None, 1, "rn")];
        assert!(!streamable(&parts, &[], &ok, &rn, Some(3)).unwrap());
        assert!(!streamable(&parts, &[], &[], &rn, None).unwrap());
        let far = [func(WindowFn::Lag, Some("v"), MAX_CONTEXT_ROWS + 1, "l")];
        assert!(!streamable(&parts, &[], &ok, &far, None).unwrap());
        let mut ignoring = func(WindowFn::FirstValue, Some("v"), 1, "f");
        ignoring.ignore_nulls = true;
        assert!(!streamable(&parts, &[], &ok, &[ignoring], None).unwrap());
        assert!(streamable(&parts, &[], &ok, &rn, None).unwrap());
    }
}
