//! [`Rebatcher`] — re-chunk a stream of batches to a target row count.
//!
//! Upstream morsels arrive at whatever size the pipeline produced them; an
//! expensive opaque operator (GPU inference especially) wants a *specific*, larger
//! batch size. The rebatcher coalesces small inputs and splits large ones so the
//! operator always sees ~`target_rows`-row batches, never losing or reordering a
//! row. The target can be retuned live by a [`BatchSizeController`](crate::BatchSizeController).

use arrow::compute::concat_batches;
use bc_arrow::{RecordBatch, SchemaRef};

use crate::{Result, UdfError};

/// Buffers incoming batches and emits batches of (up to) a target row count.
///
/// Emission happens lazily on [`push`](Rebatcher::push) (each push returns any
/// batches that just became ready) and on [`flush`](Rebatcher::flush) (the trailing
/// partial batch). Rows are never dropped, duplicated, or reordered.
pub struct Rebatcher {
    target_rows: usize,
    buffer: Vec<RecordBatch>,
    buffered_rows: usize,
    schema: Option<SchemaRef>,
}

impl Rebatcher {
    /// Create a rebatcher targeting `target_rows` (clamped to at least 1).
    pub fn new(target_rows: usize) -> Self {
        Self {
            target_rows: target_rows.max(1),
            buffer: Vec::new(),
            buffered_rows: 0,
            schema: None,
        }
    }

    /// The current target batch size in rows.
    pub fn target_rows(&self) -> usize {
        self.target_rows
    }

    /// Retune the target (clamped to at least 1). The new size takes effect on the
    /// next [`push`](Rebatcher::push)/[`flush`](Rebatcher::flush).
    pub fn set_target(&mut self, target_rows: usize) {
        self.target_rows = target_rows.max(1);
    }

    /// Rows currently buffered (less than `target_rows`, except transiently inside a
    /// `push` before emission).
    pub fn buffered_rows(&self) -> usize {
        self.buffered_rows
    }

    /// Push a batch; return any full `target_rows`-sized batches now ready.
    ///
    /// Empty (0-row) batches are ignored. A batch whose schema differs from earlier
    /// pushes is rejected with [`UdfError::SchemaMismatch`].
    pub fn push(&mut self, batch: RecordBatch) -> Result<Vec<RecordBatch>> {
        if batch.num_rows() == 0 {
            return Ok(Vec::new());
        }
        match &self.schema {
            None => self.schema = Some(batch.schema()),
            Some(s) if s != &batch.schema() => {
                return Err(UdfError::SchemaMismatch(format!(
                    "rebatcher schema {:?} but pushed {:?}",
                    s.fields(),
                    batch.schema().fields()
                )));
            }
            Some(_) => {}
        }
        self.buffered_rows += batch.num_rows();
        self.buffer.push(batch);
        self.emit_ready()
    }

    /// Emit the remaining buffered rows as one final (possibly small) batch, or
    /// `None` if nothing is buffered.
    pub fn flush(&mut self) -> Result<Option<RecordBatch>> {
        if self.buffered_rows == 0 {
            return Ok(None);
        }
        let Some(schema) = self.schema.clone() else {
            return Ok(None);
        };
        let combined = concat_batches(&schema, &self.buffer)?;
        self.buffer.clear();
        self.buffered_rows = 0;
        Ok(Some(combined))
    }

    /// While at least `target_rows` are buffered, concatenate and slice off full
    /// target-sized batches, keeping the remainder buffered.
    fn emit_ready(&mut self) -> Result<Vec<RecordBatch>> {
        if self.buffered_rows < self.target_rows {
            return Ok(Vec::new());
        }
        let Some(schema) = self.schema.clone() else {
            return Ok(Vec::new());
        };
        let combined = concat_batches(&schema, &self.buffer)?;
        self.buffer.clear();

        let total = combined.num_rows();
        let mut out = Vec::with_capacity(total / self.target_rows);
        let mut offset = 0;
        while total - offset >= self.target_rows {
            out.push(combined.slice(offset, self.target_rows));
            offset += self.target_rows;
        }
        let remainder = total - offset;
        if remainder > 0 {
            self.buffer.push(combined.slice(offset, remainder));
        }
        self.buffered_rows = remainder;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]))
    }

    fn batch(vals: Vec<i64>) -> RecordBatch {
        RecordBatch::try_new(schema(), vec![Arc::new(Int64Array::from(vals))]).unwrap()
    }

    fn values(b: &RecordBatch) -> Vec<i64> {
        b.column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    #[test]
    fn coalesces_small_batches() {
        let mut rb = Rebatcher::new(4);
        // five 2-row batches → 10 rows → two full 4-row batches, 2 buffered.
        let mut out = Vec::new();
        for i in 0..5 {
            out.extend(rb.push(batch(vec![2 * i, 2 * i + 1])).unwrap());
        }
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|b| b.num_rows() == 4));
        assert_eq!(rb.buffered_rows(), 2);
        let tail = rb.flush().unwrap().unwrap();
        assert_eq!(tail.num_rows(), 2);
        assert!(rb.flush().unwrap().is_none());
    }

    #[test]
    fn splits_large_batch() {
        let mut rb = Rebatcher::new(3);
        let out = rb.push(batch((0..10).collect())).unwrap();
        assert_eq!(out.len(), 3); // 3 + 3 + 3, remainder 1
        assert!(out.iter().all(|b| b.num_rows() == 3));
        assert_eq!(rb.buffered_rows(), 1);
        assert_eq!(values(&rb.flush().unwrap().unwrap()), vec![9]);
    }

    #[test]
    fn preserves_all_rows_in_order() {
        // Mixed input sizes whose concatenation is 0..100; output must reconstruct it.
        let mut rb = Rebatcher::new(7);
        let chunks = [0..10, 10..11, 11..50, 50..100];
        let mut got: Vec<i64> = Vec::new();
        for c in chunks {
            for b in rb.push(batch(c.collect())).unwrap() {
                got.extend(values(&b));
            }
        }
        if let Some(b) = rb.flush().unwrap() {
            got.extend(values(&b));
        }
        assert_eq!(got, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn rejects_schema_mismatch() {
        let other = Arc::new(Schema::new(vec![Field::new("w", DataType::Int64, false)]));
        let other_batch =
            RecordBatch::try_new(other, vec![Arc::new(Int64Array::from(vec![1]))]).unwrap();
        let mut rb = Rebatcher::new(4);
        rb.push(batch(vec![1, 2])).unwrap();
        assert!(matches!(
            rb.push(other_batch).unwrap_err(),
            UdfError::SchemaMismatch(_)
        ));
    }

    #[test]
    fn empty_inputs_and_flush_are_noops() {
        let mut rb = Rebatcher::new(4);
        assert!(rb.push(batch(vec![])).unwrap().is_empty());
        assert!(rb.flush().unwrap().is_none());
        assert_eq!(rb.buffered_rows(), 0);
    }

    #[test]
    fn target_zero_clamps_to_one() {
        let mut rb = Rebatcher::new(0);
        assert_eq!(rb.target_rows(), 1);
        let out = rb.push(batch(vec![1, 2])).unwrap();
        assert_eq!(out.len(), 2); // every row is its own batch
    }
}
