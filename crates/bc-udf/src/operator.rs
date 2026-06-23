//! The [`OpaqueOperator`] trait and its concrete [`FnOperator`] seam.
//!
//! An opaque operator is any per-batch transform the optimizer cannot see inside:
//! a Python `map_batches`, an embedding/inference call, a GPU kernel. The engine
//! treats it as a pipeline breaker and only needs four things — its output schema,
//! how to run one batch, whether it carries state, and its preferred batch size.

use bc_arrow::{RecordBatch, SchemaRef};

use crate::{Result, UdfError};

/// A per-batch transform the engine runs without inspecting its internals.
///
/// Implementors do the heavy work (model forward pass, Python call) in
/// [`execute_batch`](OpaqueOperator::execute_batch). The boundary is Arrow
/// `RecordBatch` in, `RecordBatch` out — zero bespoke formats — so one trait serves
/// native Rust ops, Python UDFs (wrapped by `bc-py`), and GPU inference alike.
pub trait OpaqueOperator: Send {
    /// The output schema produced for a given input schema. Called once at planning
    /// time; an operator may add, drop, or retype columns.
    fn schema_out(&self, input: &SchemaRef) -> Result<SchemaRef>;

    /// Transform one batch. This is where the expensive, opaque work happens; the
    /// engine schedules it as a breaker and releases the GIL around it (in `bc-py`).
    fn execute_batch(&self, batch: &RecordBatch) -> Result<RecordBatch>;

    /// Whether the operator carries cross-batch state (e.g. a model with a running
    /// cache). Stateful operators cannot be freely parallelized across workers.
    fn is_stateful(&self) -> bool {
        false
    }

    /// The batch size (rows) the operator runs most efficiently at — GPU operators
    /// want large batches. `None` lets the engine pick. Used to seed the
    /// [`Rebatcher`](crate::Rebatcher) target.
    fn preferred_batch_rows(&self) -> Option<usize> {
        None
    }
}

/// The concrete operator the rest of the engine builds on: an [`OpaqueOperator`]
/// whose body is a Rust closure plus a fixed output schema.
///
/// This is the seam `bc-py` wires Python into — it constructs a `FnOperator` whose
/// closure invokes a Python `map_batches` callback over the zero-copy Arrow
/// boundary (GIL released around the surrounding Rust work). Native callers can use
/// it directly for in-process transforms.
pub struct FnOperator {
    out_schema: SchemaRef,
    #[allow(clippy::type_complexity)]
    body: Box<dyn Fn(&RecordBatch) -> Result<RecordBatch> + Send + Sync>,
    stateful: bool,
    preferred_rows: Option<usize>,
}

impl FnOperator {
    /// Build an operator that produces `out_schema` by running `body` on each batch.
    pub fn new<F>(out_schema: SchemaRef, body: F) -> Self
    where
        F: Fn(&RecordBatch) -> Result<RecordBatch> + Send + Sync + 'static,
    {
        Self {
            out_schema,
            body: Box::new(body),
            stateful: false,
            preferred_rows: None,
        }
    }

    /// Mark the operator stateful (cannot be parallelized across workers).
    pub fn with_stateful(mut self, stateful: bool) -> Self {
        self.stateful = stateful;
        self
    }

    /// Declare a preferred batch size in rows (e.g. a GPU operator).
    pub fn with_preferred_rows(mut self, rows: usize) -> Self {
        self.preferred_rows = Some(rows);
        self
    }
}

impl OpaqueOperator for FnOperator {
    fn schema_out(&self, _input: &SchemaRef) -> Result<SchemaRef> {
        Ok(self.out_schema.clone())
    }

    fn execute_batch(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let out = (self.body)(batch)?;
        // The operator promised `out_schema`; enforce it so a buggy UDF surfaces a
        // clear error here rather than corrupting a downstream pipeline.
        if out.schema() != self.out_schema {
            return Err(UdfError::SchemaMismatch(format!(
                "operator declared {:?} but produced {:?}",
                self.out_schema.fields(),
                out.schema().fields()
            )));
        }
        Ok(out)
    }

    fn is_stateful(&self) -> bool {
        self.stateful
    }

    fn preferred_batch_rows(&self) -> Option<usize> {
        self.preferred_rows
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    fn schema(name: &str) -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, false)]))
    }

    fn batch(name: &str, vals: &[i64]) -> RecordBatch {
        RecordBatch::try_new(
            schema(name),
            vec![Arc::new(Int64Array::from(vals.to_vec()))],
        )
        .unwrap()
    }

    #[test]
    fn fn_operator_applies_body() {
        // double every value, same schema
        let op = FnOperator::new(schema("x"), |b| {
            let col = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let doubled: Int64Array = col.iter().map(|v| v.map(|x| x * 2)).collect();
            Ok(RecordBatch::try_new(b.schema(), vec![Arc::new(doubled)])?)
        });
        let out = op.execute_batch(&batch("x", &[1, 2, 3])).unwrap();
        let got = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(got.values(), &[2, 4, 6]);
        assert_eq!(op.schema_out(&schema("x")).unwrap(), schema("x"));
    }

    #[test]
    fn fn_operator_rejects_wrong_output_schema() {
        // declares column "x" but produces column "y" → SchemaMismatch
        let op = FnOperator::new(schema("x"), |_b| Ok(batch("y", &[0])));
        let err = op.execute_batch(&batch("x", &[1])).unwrap_err();
        assert!(matches!(err, UdfError::SchemaMismatch(_)));
    }

    #[test]
    fn fn_operator_propagates_body_error() {
        let op = FnOperator::new(schema("x"), |_b| {
            Err(UdfError::Operator("model oom".into()))
        });
        assert!(matches!(
            op.execute_batch(&batch("x", &[1])).unwrap_err(),
            UdfError::Operator(_)
        ));
    }

    #[test]
    fn flags_reflect_setters() {
        let op = FnOperator::new(schema("x"), |b| Ok(b.clone()))
            .with_stateful(true)
            .with_preferred_rows(4096);
        assert!(op.is_stateful());
        assert_eq!(op.preferred_batch_rows(), Some(4096));

        let plain = FnOperator::new(schema("x"), |b| Ok(b.clone()));
        assert!(!plain.is_stateful());
        assert_eq!(plain.preferred_batch_rows(), None);
    }
}
