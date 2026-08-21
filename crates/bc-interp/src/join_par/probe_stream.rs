//! Streaming a join's probe side past a build side that is already resident.
//!
//! Every bounded-memory join reaches the same shape once it has decided which side to hold:
//! the build side is a table, and the probe side is a stream that passes it. Holding the
//! probe side too — which is what "materialize both, then join" does — doubles the memory a
//! join needs and bounds it by the side whose size the operator never had to know.
//!
//! That mattered most exactly where spilling was supposed to help. A grace join sizes its
//! bucket count from the *build* side, so a probe bucket can be orders of magnitude over the
//! envelope while every build bucket fits; and re-partitioning cannot fix it, because a probe
//! bucket is over budget precisely when it holds a hot key, and a hot key re-hashes to one
//! sub-bucket however it is salted. The fix is not a better partitioner. It is not holding
//! the probe side.
//!
//! ## What makes it exact
//!
//! A probe chunk's matches are a function of that chunk and the build side alone — for the
//! grace join because equal keys co-partition, for the range join because a left row's
//! matches depend on the whole right side and on nothing else about the left. So `Inner`,
//! `Left`, `Semi` and `Anti` are exact per chunk: each emits only probe-driven rows, and no
//! later chunk can overturn a verdict already reached.
//!
//! `Right` and `Full` also emit build rows that matched *nothing*, which is a property of the
//! whole probe side rather than of any one chunk. Those two run their probe-driven half per
//! chunk and carry one mark per build row across the chunks, emitting the unmatched remainder
//! once at the end. The marks are a `bool` per row of the side that is resident anyway.
//!
//! ## Why it is parameterized by an index function
//!
//! The two callers differ in exactly one thing — how a (chunk, build) pair becomes index
//! pairs, a hash probe in one case and a sorted range sweep in the other. Everything after
//! that is identical, including the part that is easy to get subtly wrong: which flavor each
//! chunk runs, which side the marks are kept on, and how the remainder is null-extended.
//! Passing the index function in keeps that one implementation rather than two.

use arrow::array::{RecordBatch, UInt32Array};
use arrow::datatypes::SchemaRef;
use bc_ir::{JoinOutputCol, JoinType};
use bc_runtime::join::JoinIndices;

use crate::error::InterpError;
use crate::ops;

/// How a (probe chunk, build) pair becomes index pairs under a given join flavor.
pub(crate) type IndicesOf<'a> =
    &'a dyn Fn(&RecordBatch, &RecordBatch, JoinType) -> Result<JoinIndices, InterpError>;

/// The parts of a probe-streaming join that do not change from chunk to chunk.
pub(crate) struct ProbeStream<'a> {
    /// The flavor of the join as a whole. Individual chunks may run a different one — see
    /// [`ProbeStream::per_chunk`].
    pub join_type: JoinType,
    /// The output projection, in output order.
    pub output: &'a [JoinOutputCol],
    /// The probe side's schema, so the unmatched-build remainder can be null-extended
    /// against an empty relation of the right shape.
    pub probe_schema: SchemaRef,
}

impl ProbeStream<'_> {
    /// The flavor each probe chunk runs on its own.
    ///
    /// `Right`'s output is its matches plus a remainder that only the whole probe side can
    /// determine, so per chunk it is an inner join. `Full` additionally keeps its
    /// unmatched-*probe* half, which **is** exact per chunk, so per chunk it is a left join.
    fn per_chunk(&self) -> JoinType {
        match self.join_type {
            JoinType::Right => JoinType::Inner,
            JoinType::Full => JoinType::Left,
            t => t,
        }
    }

    /// Whether this flavor emits build rows that matched nothing.
    fn tracks_build(&self) -> bool {
        matches!(self.join_type, JoinType::Right | JoinType::Full)
    }

    /// Join every chunk against the resident `build`, appending results to `out`.
    ///
    /// Peak memory is `build` plus one chunk, whatever the probe side's size or key
    /// distribution. Returns the probe rows consumed, which a caller reading its chunks off
    /// disk needs in order to make the short-read check a streaming reader cannot make for
    /// itself.
    pub(crate) fn run(
        &self,
        chunks: impl Iterator<Item = Result<RecordBatch, InterpError>>,
        build: &RecordBatch,
        indices_of: IndicesOf<'_>,
        out: &mut Vec<RecordBatch>,
    ) -> Result<u64, InterpError> {
        let mut marks = self.tracks_build().then(|| vec![false; build.num_rows()]);
        let per_chunk = self.per_chunk();
        let mut probe_rows = 0u64;

        for chunk in chunks {
            let chunk = chunk?;
            probe_rows += chunk.num_rows() as u64;
            if chunk.num_rows() == 0 {
                continue;
            }
            let idx = indices_of(&chunk, build, per_chunk)?;
            if let Some(seen) = &mut marks {
                for r in idx.right.iter().flatten() {
                    seen[r as usize] = true;
                }
            }
            push_rows(
                out,
                ops::gather_join_output(&chunk, build, &idx, self.output)?,
            );
        }

        if let Some(seen) = marks {
            self.emit_unmatched_build(build, &seen, indices_of, out)?;
        }
        Ok(probe_rows)
    }

    /// Emit the `Right`/`Full` remainder: build rows no chunk ever matched, null-extended.
    ///
    /// Gathering the unmatched rows and right-joining them against an *empty* probe relation
    /// produces the null extension through the same index function and the same output
    /// assembler as every other row — rather than restating here which output columns come
    /// from which side, which is knowledge that should live in one place.
    fn emit_unmatched_build(
        &self,
        build: &RecordBatch,
        seen: &[bool],
        indices_of: IndicesOf<'_>,
        out: &mut Vec<RecordBatch>,
    ) -> Result<(), InterpError> {
        let unmatched: UInt32Array = seen
            .iter()
            .enumerate()
            .filter(|(_, &s)| !s)
            .map(|(r, _)| r as u32)
            .collect();
        if unmatched.is_empty() {
            return Ok(());
        }
        let leftovers = ops::take_batch(build, &unmatched)?;
        let empty_probe = RecordBatch::new_empty(self.probe_schema.clone());
        let idx = indices_of(&empty_probe, &leftovers, JoinType::Right)?;
        push_rows(
            out,
            ops::gather_join_output(&empty_probe, &leftovers, &idx, self.output)?,
        );
        Ok(())
    }
}

/// Group a stream of batches into chunks of at most `budget` bytes, concatenating each.
///
/// The unit a probe side streams in, and it is a real choice rather than a formality. The
/// build side is re-prepared once per chunk — a hash table rebuilt, or for a range join a
/// re-sort — so chunking by the *arriving batch* would pay that cost once per morsel.
///
/// It is worse than that where it matters most. A grace join's probe side reaches this
/// already sharded `p` ways by key, so a 16,384-row morsel arrives as up to 256 shards of a
/// few dozen rows each, and a table built per shard would be thousands of builds for one
/// bucket. Sizing chunks by the *memory envelope* instead makes the build count
/// `probe_bytes / budget`, which is **one** whenever the probe side fits — so a join that
/// never needed the bound behaves exactly as it did — and grows only for a probe side that
/// genuinely exceeds it, which is the case that used to be an OOM rather than a slower join.
///
/// A single batch already larger than the budget is passed through rather than split: it is
/// one allocation, and splitting it would copy it to make it smaller. A chunk of one batch is
/// passed through uncopied for the same reason.
pub(crate) fn chunk_by_bytes(
    batches: impl IntoIterator<Item = Result<RecordBatch, InterpError>>,
    budget: usize,
) -> impl Iterator<Item = Result<RecordBatch, InterpError>> {
    let budget = budget.max(1);
    let mut input = batches.into_iter();
    let mut done = false;
    std::iter::from_fn(move || {
        if done {
            return None;
        }
        let mut pending: Vec<RecordBatch> = Vec::new();
        let mut bytes = 0usize;
        while bytes < budget {
            match input.next() {
                Some(Ok(b)) => {
                    bytes += ops::sliced_batch_bytes(&b);
                    pending.push(b);
                }
                Some(Err(e)) => {
                    done = true;
                    return Some(Err(e));
                }
                None => {
                    done = true;
                    break;
                }
            }
        }
        (!pending.is_empty()).then(|| ops::materialize(&pending))
    })
}

/// [`chunk_by_bytes`] over an in-memory relation.
pub(crate) fn chunk_slice_by_bytes(
    batches: &[RecordBatch],
    budget: usize,
) -> impl Iterator<Item = Result<RecordBatch, InterpError>> + '_ {
    chunk_by_bytes(batches.iter().cloned().map(Ok), budget)
}

/// Push a join result, dropping it if it carries no rows.
///
/// A streamed probe emits one batch per chunk rather than one per operator, so an empty one
/// is ordinary — a bucket with no probe rows, an inner join that matched nothing — where it
/// used to be the exception. Carrying them costs every downstream operator a per-batch
/// dispatch over nothing. Callers guarantee the join still emits a schema-carrying batch when
/// the whole relation is empty.
fn push_rows(out: &mut Vec<RecordBatch>, batch: RecordBatch) {
    if batch.num_rows() > 0 {
        out.push(batch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Int64Array};
    use std::sync::Arc;

    fn ints(n: usize) -> RecordBatch {
        let v: Vec<i64> = (0..n as i64).collect();
        RecordBatch::try_from_iter(vec![("v", Arc::new(Int64Array::from(v)) as ArrayRef)]).unwrap()
    }

    fn chunk_rows(batches: &[RecordBatch], budget: usize) -> Vec<usize> {
        chunk_slice_by_bytes(batches, budget)
            .map(|c| c.unwrap().num_rows())
            .collect()
    }

    /// The no-regression property: an input inside the envelope is **one** chunk, so an
    /// operator that never needed to spill does exactly the single pass it always did.
    #[test]
    fn an_input_that_fits_is_a_single_chunk() {
        let batches = [ints(4), ints(4), ints(4)];
        // 12 rows x 8 bytes = 96; any budget at or above that is one chunk.
        assert_eq!(chunk_rows(&batches, 96), vec![12]);
        assert_eq!(chunk_rows(&batches, usize::MAX), vec![12]);
    }

    /// Past the envelope it splits, and the split conserves every row in input order.
    #[test]
    fn an_oversized_input_splits_without_losing_rows() {
        let batches = [ints(4), ints(4), ints(4), ints(4)];
        // 32 bytes per batch; a 64-byte budget takes two batches per chunk.
        assert_eq!(chunk_rows(&batches, 64), vec![8, 8]);
        // A budget under one batch still yields whole batches — a batch is one allocation
        // already, and splitting it would copy it to make it smaller.
        assert_eq!(chunk_rows(&batches, 1), vec![4, 4, 4, 4]);
        // Nothing in, nothing out.
        assert!(chunk_rows(&[], 64).is_empty());
    }
}
